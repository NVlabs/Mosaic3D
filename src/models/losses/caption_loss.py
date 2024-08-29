from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import functional as F
from torch_scatter import segment_csr

from src.models.losses.loss_base import LossBase


# TODO(cchoy): move this to data preprocessing
# Convert List[List[Tensor]] to concatenated Tensor, and offsets
def convert_list_list_tensor_to_tensor(
    batched_list_of_point_indices: List[List[Int[Tensor, "N"]]],  # noqa: F722, F821
    batch_offsets: Optional[Int[Tensor, "B + 1"]] = None,  # noqa: F722, F821
    valid_mask: Optional[Bool[Tensor, "L"]] = None,  # noqa: F722, F821
) -> Tuple[Int[Tensor, "L"], Int[Tensor, "M + 1"], Int[Tensor, "M"]]:  # noqa: F722, F821
    """Convert List[List[Tensor]] to concatenated indices, offsets, and counts."""
    # Get the counts of inner lists
    num_points_per_cap = [
        len(tensor) for sublist in batched_list_of_point_indices for tensor in sublist
    ]

    # Concatenate inner lists first and generate offsets for the inner lists
    batched_flat_point_indices = [
        torch.cat(sublist, dim=0) for sublist in batched_list_of_point_indices
    ]

    # Add batch offset if provided
    if batch_offsets is not None:
        if isinstance(batch_offsets, torch.Tensor):
            batch_offsets = batch_offsets.tolist()
        batched_flat_point_indices = [
            l + batch_offsets[i] for i, l in enumerate(batched_flat_point_indices)
        ]
    else:
        assert (
            len(batched_flat_point_indices) == 1
        ), "batch_offset must be provided if len(list_tensor) > 1"

    # Concatenate all lists and generate offsets for the outer lists
    point_indices = torch.cat(batched_flat_point_indices, 0)
    offsets = np.cumsum(num_points_per_cap)
    offsets = torch.tensor([0] + offsets.tolist())
    counts = torch.tensor(num_points_per_cap)

    # Apply valid mask
    if valid_mask is not None:
        valid_point_indices = valid_mask[point_indices]
        point_indices = point_indices[valid_point_indices]

        # Use cumsum to efficiently compute valid counts
        cumulative_valid = torch.cumsum(valid_point_indices, dim=0)

        # Compute new offsets directly from cumulative_valid
        new_offsets = torch.zeros_like(offsets)
        new_offsets[1:] = cumulative_valid[offsets[1:] - 1]

        # Compute valid counts
        valid_counts = new_offsets[1:] - new_offsets[:-1]

        offsets = new_offsets
        counts = valid_counts

        # Convert point_indices to compacted indices
        compacted_indices = torch.cumsum(valid_mask, dim=0)
        point_indices = compacted_indices[point_indices] - 1

    return point_indices, offsets, counts


class CaptionAlignmentLoss(LossBase):
    def __init__(
        self,
        loss_reduction: Literal["mean", "weighted"] = "weighted",
        normalize_input: bool = True,
    ):
        super().__init__()
        self.loss_reduction = loss_reduction
        self.normalize_input = normalize_input
        self.sim_func = nn.CosineSimilarity(dim=-1)

    def forward(self, pred_feats, batch_dict: Dict) -> Tensor:
        return pred_feats, batch_dict

    def loss(
        self,
        pred_feats: Float[Tensor, "M 512"],  # noqa: F821, F722
        caption_embeddings: List[Float[Tensor, "N 512"]],  # noqa: F821, F722
        batched_list_of_point_indices: List[Int[Tensor, "N"]],  # noqa: F821, F722
        input_batch_offsets: Int[Tensor, "B + 1"],  # noqa: F821, F722
        valid_mask: Optional[Bool[Tensor, "L"]] = None,  # noqa: F821
    ) -> Tensor:
        """Compute the caption loss.

        Args:
            x: The input tensor
            caption_embeddings: Batched caption embeddings. e.g. [all_caption_embeddings_for_batch0, all_caption_embeddings_for_batch1, ...]
            batched_list_of_point_indices: The list of point indices. e.g. [[point_indices_for_batch0_obj0, point_indices_for_batch0_obj1, ...], [point_indices_for_batch1_obj0, point_indices_for_batch1_obj1, ...], ...]
            batch_offsets: The batch offsets for the input tensor. e.g. [0, num_points_in_batch0, num_points_in_batch0 + num_points_in_batch1, ...]
            mappings: The mappings.
            binary_labels: The binary labels.
        """

        # assert the number of captions is the same as the number of caption indices
        assert all(
            [
                len(caption) == len(idx)
                for caption, idx in zip(caption_embeddings, batched_list_of_point_indices)
            ]
        )
        batch_size = len(batched_list_of_point_indices)
        assert len(caption_embeddings) == batch_size
        assert len(input_batch_offsets) == batch_size + 1

        # Convert data to CSR format
        corr_idx, offsets, counts = convert_list_list_tensor_to_tensor(
            batched_list_of_point_indices,
            batch_offsets=input_batch_offsets,
            valid_mask=valid_mask,
        )

        # Reduce features
        if self.normalize_input:
            pred_feats = nn.functional.normalize(pred_feats, dim=-1)
        rep_pred_feats = pred_feats[corr_idx]
        reduced_pred_feats = segment_csr(
            rep_pred_feats,
            offsets.to(rep_pred_feats.device),
            reduce="sum",
        )

        reduced_pred_feats = nn.functional.normalize(reduced_pred_feats, dim=-1)
        flat_caption_embeddings = torch.cat(caption_embeddings, 0)

        # Compute the cosine similarity
        loss = 1 - self.sim_func(reduced_pred_feats, flat_caption_embeddings)
        counts = counts.to(loss.device)
        if self.loss_reduction == "mean":
            loss = loss.mean()
        elif self.loss_reduction == "weighted":
            loss = (loss * counts).sum() / counts.sum()
        return loss


class CaptionLoss(LossBase):
    def __init__(
        self,
        normalize_input: bool = True,
        ignore_index: int = -100,
        use_logit_scale: Optional[bool] = False,
        loss_reduction: Literal["mean", "weighted_sum"] = "weighted_sum",
        **kwargs,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        self.loss_func = nn.NLLLoss(ignore_index=ignore_index, reduction="none")
        assert loss_reduction in ["mean", "weighted_sum"]
        self.loss_reduction = loss_reduction
        self.use_logit_scale = use_logit_scale
        if use_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)

        self.kwargs = kwargs

    def forward(self, pred_feats, batch_dict: Dict) -> Tensor:
        return pred_feats, batch_dict

    def loss(
        self,
        pred_feats: Float[Tensor, "M 512"],  # noqa: F722
        unique_caption_embeds: Float[Tensor, "N 512"],  # noqa: F722
        caption_targets: Int[Tensor, "M"],  # noqa: F821
        batched_list_of_point_indices: List[Int[Tensor, "N"]],  # noqa: F821
        input_batch_offsets: Int[Tensor, "B + 1"],  # noqa: F821
        valid_mask: Optional[Bool[Tensor, "L"]] = None,  # noqa: F821
    ) -> Tensor:
        """Compute the caption loss.

        Args:
            x: The input tensor
            caption_embeddings: Batched caption embeddings. e.g. [all_caption_embeddings_for_batch0, all_caption_embeddings_for_batch1, ...]
            batched_list_of_point_indices: The list of point indices. e.g. [[point_indices_for_batch0_obj0, point_indices_for_batch0_obj1, ...], [point_indices_for_batch1_obj0, point_indices_for_batch1_obj1, ...], ...]
            batch_offsets: The batch offsets for the input tensor. e.g. [0, num_points_in_batch0, num_points_in_batch0 + num_points_in_batch1, ...]
            mappings: The mappings.
            binary_labels: The binary labels.
        """
        batch_size = len(batched_list_of_point_indices)
        assert len(input_batch_offsets) == batch_size + 1
        # assert len(unique_caption_embeds) == len(np.unique(caption_targets))

        # Convert data to CSR format
        corr_idx, offsets, counts = convert_list_list_tensor_to_tensor(
            batched_list_of_point_indices,
            batch_offsets=input_batch_offsets,
            valid_mask=valid_mask,
        )

        # Reduce features
        if self.normalize_input:
            pred_feats = nn.functional.normalize(pred_feats, dim=-1)

        # Logit
        device = pred_feats.device
        caption_logits = pred_feats @ unique_caption_embeds.T.to(device)
        if self.use_logit_scale:
            caption_logits = self.logit_scale.exp() * caption_logits
        caption_scores = F.log_softmax(caption_logits, dim=-1)

        rep_caption_scores = caption_scores[corr_idx]
        reduced_caption_scores = segment_csr(
            rep_caption_scores,
            offsets.to(device),
            reduce="mean",
        )

        # Compute the loss
        loss = self.loss_func(reduced_caption_scores, caption_targets.to(caption_scores.device))

        # Compute the cosine similarity
        if self.loss_reduction == "mean":
            return loss.mean()
        elif self.loss_reduction == "weighted_sum":
            counts = counts.to(loss.device)
            return (loss * counts).sum() / (counts.sum())
        else:
            raise ValueError(f"Unknown reduce type: {self.reduce}")
