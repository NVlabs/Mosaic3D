from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import functional as F
from torch_scatter import scatter, segment_csr

import src.utils.dist_utils as dist_utils
from src.models.losses.loss_base import LossBase
from src.utils.caption_utils import get_caption_batch, get_unique_caption_batch


class CaptionLossBase(LossBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, pred_feats, batch_dict: Dict) -> Tensor:
        return pred_feats, batch_dict

    def extract_text_features(
        self,
        captions: List[str],
        clip_encoder: nn.Module,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
    ):
        if embeddings is not None:
            text_features = torch.stack([item for sublist in embeddings for item in sublist], 0)
            device = text_features.device
            _, labels_per_caption, labels_per_segment = np.unique(
                text_features.cpu().numpy(), return_index=True, return_inverse=True, axis=0
            )
            text_features = text_features[labels_per_caption]
            labels_per_segment = torch.from_numpy(labels_per_segment).to(device)
            labels_per_caption = torch.from_numpy(labels_per_caption).to(device)
        else:
            # extract text features
            with torch.cuda.amp.autocast(enabled=True) and torch.inference_mode():
                text_features, labels_per_segment, labels_per_caption = get_unique_caption_batch(
                    captions, clip_encoder
                )
            text_features = (
                text_features.clone() if isinstance(text_features, torch.Tensor) else text_features
            )
        return text_features, labels_per_segment, labels_per_caption


class CaptionAlignmentLoss(CaptionLossBase):
    def __init__(
        self,
        normalize: bool = True,
        reduction: Literal["mean", "weighted_sum"] = "weighted_sum",
        **kwargs,
    ):
        super().__init__()
        self.normalize = normalize
        self.reduction = reduction

    def loss(
        self,
        point_features: Float[Tensor, "M D"],  # noqa: F722
        point_indices: Int[Tensor, "L"],  # noqa: F821
        caption_offsets: Int[Tensor, "B + 1"],  # noqa: F821
        num_points_per_caption: Int[Tensor, "B"],  # noqa: F821
        clip_encoder: nn.Module,
        captions: Optional[List[List[str]]] = None,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
    ) -> Tensor:
        # extract text features
        text_features, *_ = self.extract_text_features(captions, clip_encoder, embeddings)

        if self.normalize:
            point_features = nn.functional.normalize(point_features, dim=-1)
        rep_point_features = point_features[point_indices]
        segment_features = segment_csr(
            rep_point_features,
            caption_offsets.to(rep_point_features.device),
            reduce="sum",
        )

        segment_features = nn.functional.normalize(segment_features, dim=-1)
        text_features = torch.cat(text_features, 0)

        # Compute the cosine similarity
        loss = 1 - torch.einsum("ij,ij->i", segment_features, text_features)
        num_points_per_caption = num_points_per_caption.to(loss.device)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "weighted_sum":
            loss = (loss * num_points_per_caption).sum() / num_points_per_caption.sum()
        return loss


class DenseCaptionAlignmentLoss(CaptionLossBase):
    def __init__(
        self,
        normalize: bool = True,
        is_entity: bool = False,
        interpolate: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.normalize = normalize
        self.is_entity = is_entity
        self.interpolate = interpolate

    def extract_text_features(
        self,
        captions: List[str],
        clip_encoder: nn.Module,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
    ):
        if embeddings is not None:
            text_features = torch.stack([item for sublist in embeddings for item in sublist], 0)
        else:
            with torch.cuda.amp.autocast(enabled=True) and torch.inference_mode():
                text_features = get_caption_batch(
                    captions, clip_encoder, is_entity=self.is_entity, interpolate=self.interpolate
                )
            text_features = (
                text_features.clone() if isinstance(text_features, torch.Tensor) else text_features
            )
        return text_features

    def loss(
        self,
        point_features: Float[Tensor, "M D"],  # noqa: F722
        point_indices: Int[Tensor, "L"],  # noqa: F821
        caption_offests: Int[Tensor, "B + 1"],  # noqa: F821
        num_points_per_caption: Int[Tensor, "B"],  # noqa: F821
        clip_encoder: nn.Module,
        captions: Optional[List[List[str]]] = None,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
    ) -> Tensor:
        device, dtype = point_features.device, point_features.dtype

        # extract text features
        text_features, *_ = self.extract_text_features(captions, clip_encoder, embeddings)

        # Scatter and reduce caption embeddings
        flat_caption_embeddings = torch.cat(text_features, dim=0).to(dtype=dtype, device=device)
        caption_indices = torch.arange(len(flat_caption_embeddings)).repeat_interleave(
            num_points_per_caption
        )
        rep_caption_embeddings = flat_caption_embeddings[caption_indices]

        scattered_caption_embeddings = torch.zeros_like(point_features)
        scattered_caption_embeddings = scatter(
            rep_caption_embeddings,
            point_indices,
            dim=0,
            out=scattered_caption_embeddings,
            reduce="mean",
        )

        # Find which indices are not in corr_idx from 0 to len(pred_feats)
        mask = torch.zeros(len(point_features), dtype=torch.bool, device=device)
        mask[point_indices.unique()] = True

        # Use this mask to index into pred_feats and scattered_caption_embeddings
        pred_feats_masked = point_features[mask]
        if self.normalize:
            pred_feats_masked = nn.functional.normalize(pred_feats_masked, dim=-1)
        scattered_caption_embeddings_masked = scattered_caption_embeddings[mask]

        # Compute the cosine similarity (feats and embeddings are already normalized)
        loss = 1 - torch.einsum("ij,ij->i", pred_feats_masked, scattered_caption_embeddings_masked)

        return loss.mean()


class CaptionLoss(CaptionLossBase):
    def __init__(
        self,
        normalize: bool = True,
        use_logit_scale: Optional[bool] = False,
        reduction: Literal["mean", "weighted_sum"] = "weighted_sum",
        **kwargs,
    ):
        super().__init__()
        self.normalize = normalize
        self.loss_func = nn.NLLLoss(reduction="none")
        assert reduction in ["mean", "weighted_sum"]
        self.reduction = reduction
        self.use_logit_scale = use_logit_scale
        if use_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)

        self.kwargs = kwargs

    def loss(
        self,
        point_features: Float[Tensor, "M 512"],  # noqa: F722
        point_indices: Int[Tensor, "L"],  # noqa: F821
        caption_offsets: Int[Tensor, "B + 1"],  # noqa: F821
        num_points_per_caption: Int[Tensor, "B"],  # noqa: F821
        clip_encoder: nn.Module,
        captions: Optional[List[List[str]]] = None,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
        **kwargs,
    ) -> Tensor:
        device = point_features.device
        # extract text features
        text_features, labels_per_segment, labels_per_caption = self.extract_text_features(
            captions, clip_encoder, embeddings
        )

        # normalize point features
        if self.normalize:
            point_features = nn.functional.normalize(point_features, dim=-1)

        # Logit
        logits = point_features @ text_features.T.to(device)
        if self.use_logit_scale:
            logits = self.logit_scale.exp() * logits
        scores = F.log_softmax(logits, dim=-1)

        rep_scores = scores[point_indices]
        reduced_scores = segment_csr(rep_scores, caption_offsets.to(device), reduce="mean")

        # Compute the loss
        loss = self.loss_func(reduced_scores, labels_per_segment.to(device))

        # Compute the cosine similarity
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "weighted_sum":
            num_points_per_caption = num_points_per_caption.to(loss.device)
            return (loss * num_points_per_caption).sum() / (num_points_per_caption.sum())
        else:
            raise ValueError(f"Unknown reduce type: {self.reduce}")


class CaptionCLIPLoss(CaptionLossBase):
    def __init__(
        self,
        normalize: bool = True,
        reduction: Literal["mean", "weighted_sum"] = "weighted_sum",
        init_logit_scale: Optional[float] = 1 / 0.07,
        all_gather: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.reduction = reduction
        assert reduction in ["mean", "weighted_sum"]
        self.all_gather = all_gather

        if init_logit_scale is not None:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(init_logit_scale), requires_grad=True
            )
        else:
            self.logit_scale = torch.ones([]) * np.log(init_logit_scale)

    def loss(
        self,
        point_features: Float[Tensor, "M 512"],  # noqa: F722
        point_indices: Int[Tensor, "L"],  # noqa: F821
        caption_offsets: Int[Tensor, "B + 1"],  # noqa: F821
        clip_encoder: nn.Module,
        captions: Optional[List[List[str]]] = None,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
        **kwargs,
    ) -> Tensor:
        device = point_features.device
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # extract text features
        text_features, labels_per_segment, labels_per_caption = self.extract_text_features(
            captions, clip_encoder, embeddings
        )

        if world_size > 1 and self.all_gather:
            all_captions = [None for _ in range(world_size)]
            dist.all_gather_object(all_captions, captions)
            all_captions: List[List[str]] = [item for sublist in all_captions for item in sublist]
            (
                all_text_features,
                all_labels_per_segment,
                all_labels_per_caption,
            ) = self.extract_text_features(all_captions, clip_encoder)

        # aggregate point features
        rep_point_features = point_features[point_indices]
        segment_features = segment_csr(
            rep_point_features, caption_offsets.to(device), reduce="mean"
        )
        if self.normalize:
            segment_features = nn.functional.normalize(segment_features, dim=-1)

        if world_size > 1 and self.all_gather:
            all_segment_features = dist_utils.all_gather_different_shapes(segment_features)
            all_segment_features[dist.get_rank()] = segment_features  # this is for gradient
            all_segment_features = torch.cat(all_segment_features, 0)

        # logits
        if world_size > 1 and self.all_gather:
            logits_per_segment = self.logit_scale.exp() * (
                all_segment_features @ all_text_features.T
            )
            logits_per_caption = logits_per_segment.T
        else:
            logits_per_segment = self.logit_scale.exp() * (segment_features @ text_features.T)
            logits_per_caption = self.logit_scale.exp() * (text_features @ segment_features.T)

        target_per_segment = (
            all_labels_per_segment if world_size > 1 and self.all_gather else labels_per_segment
        )
        target_per_caption = (
            all_labels_per_caption if world_size > 1 and self.all_gather else labels_per_caption
        )
        total_loss = (
            torch.nn.functional.cross_entropy(logits_per_segment, target_per_segment.to(device))
            + torch.nn.functional.cross_entropy(logits_per_caption, target_per_caption.to(device))
        ) / 2

        return total_loss


class CaptionSigLIPLoss(CaptionCLIPLoss):
    def __init__(
        self,
        normalize: bool = True,
        reduction: Literal["mean", "weighted_sum"] = "weighted_sum",
        init_logit_scale: Optional[float] = 10,
        init_logit_bias: Optional[float] = -10,
        bidir: bool = False,
        all_gather: bool = True,
        pooling_first: bool = True,
        **kwargs,
    ):
        super().__init__(normalize, reduction, init_logit_scale, **kwargs)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

        self.bidir = bidir
        self.all_gather = all_gather
        self.pooling_first = pooling_first

    def get_ground_truth(
        self,
        shape,
        device,
        labels_per_segment: Optional[Int[Tensor, "B"]] = None,  # noqa: F821
        negative_only: bool = False,
    ) -> torch.Tensor:
        labels = torch.zeros(shape, device=device)
        if not negative_only:
            assert labels_per_segment is not None
            labels[range(shape[0]), labels_per_segment] = 1
        return labels

    def get_logits(self, features, text_features):
        logits = self.logit_scale.exp() * features @ text_features.T
        if self.logit_bias is not None:
            logits += self.logit_bias
        return logits

    def _loss(
        self,
        point_features,
        text_features,
        point_indices,
        caption_offsets,
        labels_per_segment: Optional[Int[Tensor, "B"]] = None,  # noqa: F821
        negative_only: bool = False,
    ):
        device = point_features.device

        # feature pooling first -> compute logits
        if self.pooling_first:
            rep_point_features = point_features[point_indices]
            segment_features = segment_csr(
                rep_point_features, caption_offsets.to(device), reduce="mean"
            )
            segment_features = nn.functional.normalize(segment_features, dim=-1)
            logits = self.get_logits(segment_features, text_features)
            labels = self.get_ground_truth(logits.shape, device, labels_per_segment, negative_only)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        # compute logits first -> probability pooling
        else:
            point_features = nn.functional.normalize(point_features, dim=-1)
            logits = self.get_logits(point_features, text_features)
            rep_logits = logits[point_indices]
            reduced_logits = segment_csr(rep_logits, caption_offsets.to(device), reduce="mean")
            labels = self.get_ground_truth(
                reduced_logits.shape, device, labels_per_segment, negative_only
            )
            loss = F.binary_cross_entropy_with_logits(reduced_logits, labels)
        return loss

    def loss(
        self,
        point_features: Float[Tensor, "M 512"],  # noqa: F722
        point_indices: Int[Tensor, "L"],  # noqa: F821
        caption_offsets: Int[Tensor, "B + 1"],  # noqa: F821
        num_points_per_caption: Int[Tensor, "B"],  # noqa: F821
        clip_encoder: nn.Module,
        captions: Optional[List[List[str]]] = None,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
        **kwargs,
    ) -> Tensor:
        device = point_features.device
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # extract text features
        text_features, labels_per_segment, _ = self.extract_text_features(
            captions, clip_encoder, embeddings
        )

        # loss
        loss = self._loss(
            point_features, text_features, point_indices, caption_offsets, labels_per_segment
        )

        if world_size > 1 and self.all_gather:
            # get max num captions
            all_shapes = dist_utils.all_gather_tensor_shapes(text_features)
            all_num_captions = all_shapes[:, 0]
            max_num_captions = all_num_captions.max()
            num_captions = text_features.shape[0]

            # pad text features
            text_features_padded = torch.zeros(
                max_num_captions, text_features.shape[1], device=device
            )
            text_features_padded[:num_captions] = text_features

            # exchange text features
            rank = dist.get_rank()
            right_rank = (rank + 1) % world_size
            left_rank = (rank - 1 + world_size) % world_size
            if self.bidir:
                text_features_to_right, text_features_to_left = text_features_padded
                num_captions_to_right = num_captions_to_left = all_num_captions[rank]
                num_bidir, remainder = divmod(world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = dist_utils.neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    num_captions_rev = dist_utils.neighbour_exchange_bidir(
                        left_rank,
                        right_rank,
                        num_captions_to_left,
                        num_captions_to_right,
                    )
                    for f, n in zip(text_features_recv, num_captions_rev):
                        loss += self._loss(
                            point_features,
                            f[:n],
                            point_indices,
                            caption_offsets,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv
                    num_captions_to_left, num_captions_to_right = num_captions_rev

                if remainder:
                    text_features_recv = dist_utils.neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right
                    )

                    loss += self._loss(
                        point_features,
                        text_features_recv,
                        point_indices,
                        caption_offsets,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features_padded
                num_captions_to_right = all_num_captions[rank]
                for i in range(world_size - 1):
                    text_features_from_left = dist_utils.neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right
                    )
                    num_captions_from_left = dist_utils.neighbour_exchange(
                        left_rank, right_rank, num_captions_to_right
                    )
                    _loss = self._loss(
                        point_features,
                        text_features_from_left[:num_captions_from_left],
                        point_indices,
                        caption_offsets,
                        negative_only=True,
                    )
                    loss += _loss
                    text_features_to_right = text_features_from_left
                    num_captions_to_right = num_captions_from_left

        return loss
