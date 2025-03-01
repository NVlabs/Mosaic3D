from typing import List, Optional

from jaxtyping import Float, Int
import torch
from torch import Tensor
import torch.nn as nn
import torch.distributed as dist

import src.utils.dist_utils as dist_utils
from src.models.losses.caption_loss import (
    CaptionAlignmentLoss,
    CaptionLoss,
    CaptionCLIPLoss,
    CaptionSigLIPLoss,
)


class MaskCaptionAlignmentLoss(CaptionAlignmentLoss):
    def loss(
        self,
        mask_features: Float[Tensor, "M C"],  # noqa: F722
        captions: List[List[str]],
        clip_encoder: nn.Module,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
    ) -> Tensor:
        # extract text features
        text_features, *_ = self.extract_text_features(captions, clip_encoder, embeddings)

        # normalize mask features
        if self.normalize:
            mask_features = nn.functional.normalize(mask_features, dim=-1)

        # compute the cosine similarity
        loss = 1 - mask_features @ text_features.T

        return loss.mean()


class MaskCaptionLoss(CaptionLoss):
    def loss(
        self,
        mask_features: Float[Tensor, "M C"],  # noqa: F722
        captions: List[List[str]],
        clip_encoder: nn.Module,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
    ) -> Tensor:
        device = mask_features[0].device

        # extract text features
        text_features, labels_per_mask, _ = self.extract_text_features(
            captions, clip_encoder, embeddings
        )

        # normalize mask features
        if self.normalize:
            mask_features = nn.functional.normalize(mask_features, dim=-1)

        # logit
        logits = mask_features @ text_features.T.to(device)
        if self.use_logit_scale:
            logits = self.logit_scale.exp() * logits
        scores = nn.functional.log_softmax(logits, dim=-1)

        # Compute the loss
        loss = self.loss_func(scores, labels_per_mask.to(device))

        return loss.mean()


class MaskCaptionCLIPLoss(CaptionCLIPLoss):
    def loss(
        self,
        mask_features: Float[Tensor, "M C"],  # noqa: F722
        captions: List[List[str]],
        clip_encoder: nn.Module,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
    ) -> Tensor:
        device = mask_features[0].device
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # extract text features
        text_features, labels_per_mask, labels_per_caption = self.extract_text_features(
            captions, clip_encoder, embeddings
        )

        if world_size > 1 and self.all_gather:
            all_captions = [None for _ in range(world_size)]
            dist.all_gather_object(all_captions, captions)
            all_captions: List[List[str]] = [item for sublist in all_captions for item in sublist]
            (
                all_text_features,
                all_labels_per_mask,
                all_labels_per_caption,
            ) = self.extract_text_features(all_captions, clip_encoder)

        # normalize mask features
        if self.normalize:
            mask_features = nn.functional.normalize(mask_features, dim=-1)

        if world_size > 1 and self.all_gather:
            all_mask_features = dist_utils.all_gather_different_shapes(mask_features)
            all_mask_features[dist.get_rank()] = mask_features  # this is for gradient
            all_mask_features = torch.cat(all_mask_features, 0)

        # logits
        if world_size > 1 and self.all_gather:
            logits_per_mask = self.logit_scale.exp() * (all_mask_features @ all_text_features.T)
            logits_per_caption = logits_per_mask.T
        else:
            logits_per_mask = self.logit_scale.exp() * (mask_features @ text_features.T)
            logits_per_caption = self.logit_scale.exp() * (text_features @ mask_features.T)

        target_per_mask = (
            all_labels_per_mask if world_size > 1 and self.all_gather else labels_per_mask
        )
        target_per_caption = (
            all_labels_per_caption if world_size > 1 and self.all_gather else labels_per_caption
        )
        total_loss = (
            torch.nn.functional.cross_entropy(logits_per_mask, target_per_mask.to(device))
            + torch.nn.functional.cross_entropy(logits_per_caption, target_per_caption.to(device))
        ) / 2

        return total_loss


class MaskCaptionSigLIPLoss(CaptionSigLIPLoss):
    def _loss(
        self,
        mask_features: Float[Tensor, "M C"],  # noqa: F722
        text_features: Float[Tensor, "N C"],  # noqa: F722
        labels_per_mask: Optional[Int[Tensor, "M"]] = None,  # noqa: F722, F821
        negative_only: bool = False,
    ) -> Tensor:
        device = mask_features[0].device

        mask_features = nn.functional.normalize(mask_features, dim=-1)
        logits = self.get_logits(mask_features, text_features)
        labels = self.get_ground_truth(logits.shape, device, labels_per_mask, negative_only)
        probs = nn.functional.sigmoid(logits)
        loss = nn.functional.binary_cross_entropy(probs, labels)
        return loss

    def loss(
        self,
        mask_features: Float[Tensor, "M C"],  # noqa: F722
        captions: List[List[str]],
        clip_encoder: nn.Module,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
    ) -> Tensor:
        device = mask_features[0].device
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # extract text features
        text_features, labels_per_mask, _ = self.extract_text_features(
            captions, clip_encoder, embeddings
        )

        # loss
        loss = self._loss(mask_features, text_features, labels_per_mask)

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
                        left_rank, right_rank, text_features_to_left, text_features_to_right
                    )
                    num_captions_rev = dist_utils.neighbour_exchange_bidir(
                        left_rank, right_rank, num_captions_to_left, num_captions_to_right
                    )
                    for f, n in zip(text_features_recv, num_captions_rev):
                        loss += self._loss(mask_features, f[:n], negative_only=True)
                    text_features_to_left, text_features_to_right = text_features_recv
                    num_captions_to_left, num_captions_to_right = num_captions_rev

                if remainder:
                    text_features_recv = dist_utils.neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right
                    )
                    loss += self._loss(mask_features, text_features_recv, negative_only=True)
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
                    loss += self._loss(
                        mask_features,
                        text_features_from_left[:num_captions_from_left],
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
                    num_captions_to_right = num_captions_from_left

        return loss
