from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Int
from torch import Tensor
from torch_scatter import segment_csr
from warp.convnet.geometry.point_collection import PointCollection

from src.models.losses.loss_base import LossBase


# TODO(cchoy): move this to data preprocessing
# Convert List[List[Tensor]] to concatenated Tensor, and offsets
def convert_list_list_tensor_to_tensor(
    list_list_tensor: List[List[Int[Tensor, "N"]]],  # noqa: F722, F821
    batch_offset: Optional[Int[Tensor, "B + 1"]] = None,  # noqa: F722, F821
    remove_empty_list: bool = True,
    origin_idx: Int[Tensor, "B"] = None,  # noqa: F722, F821
    pc_count: Int[Tensor, "B"] = None,  # noqa: F722, F821
) -> Tuple[Int[Tensor, "L"], Int[Tensor, "M + 1"], Int[Tensor, "M"]]:  # noqa: F722, F821
    device = batch_offset.device
    # Concatenate inner lists first and generate offsets for the inner lists
    batched_origin_idx = origin_idx.split(batch_offset.diff().tolist())

    list_tensor = []
    counts = []
    for b in range(batch_offset.shape[0] - 1):
        origin_to_point_map = torch.ones(pc_count[b], device=device, dtype=torch.long) * -1
        origin_to_point_map[batched_origin_idx[b]] = torch.arange(
            batched_origin_idx[b].shape[0], device=device, dtype=torch.long
        )
        mapped_tensors = [origin_to_point_map[tensor] for tensor in list_list_tensor[b]]
        mapped_tensors = [tensor[tensor != -1] for tensor in mapped_tensors]
        num_valid = [(tensor != -1).sum().item() for tensor in mapped_tensors]
        counts.append(num_valid)
        list_tensor.append(torch.cat(mapped_tensors, 0))

    # Add batch offset if provided
    if batch_offset is not None:
        list_tensor = [l + batch_offset[i] for i, l in enumerate(list_tensor)]
    else:
        assert len(list_tensor) == 1, "batch_offset must be provided if len(list_tensor) > 1"

    # Concatenate all lists and generate offsets for the outer lists
    tensor = torch.cat(list_tensor, 0)
    counts_flat = [c for sublist in counts for c in sublist]
    cumsum = np.cumsum(counts_flat)
    non_empty = np.array(counts_flat) > 0
    counts_flat = torch.tensor(counts_flat, device=device, dtype=torch.float32)

    if remove_empty_list:
        offsets = torch.tensor([0] + cumsum[non_empty].tolist())
        counts_flat = counts_flat[non_empty]
    else:
        offsets = torch.tensor([0] + cumsum.tolist())

    return tensor, offsets, non_empty, counts_flat


class CaptionLoss(LossBase):
    def __init__(
        self,
        normalize_input: bool = True,
        novel_grad_only: bool = False,
        ignore_label: int = -100,
        reduce: Literal["mean", "weighted_sum"] = "weighted_sum",
        learnable_logit: bool = False,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        self.novel_grad_only = novel_grad_only
        self.reduce = reduce
        self.loss_func = nn.NLLLoss(ignore_index=ignore_label, reduction="none")

        self.logit_scale = 1.0
        if learnable_logit:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)

    def forward(self, pc: PointCollection, batch_dict: Dict) -> Tensor:
        return pc, batch_dict

    def loss(self, x: Tensor | PointCollection, batch_dict):
        if isinstance(x, PointCollection):
            adapter_feats = x.feature_tensor
        else:
            adapter_feats = x

        if self.normalize_input:
            adapter_feats = nn.functional.normalize(adapter_feats, dim=-1)

        # Extract caption information
        c2p_map = batch_dict["c2p_map"]
        caption_embed = batch_dict["caption_embed"]
        caption_idx = batch_dict["caption_idx"]
        origin_idx = batch_dict["origin_idx"]

        # Compute caption scores
        logit_scale = self.logit_scale
        if isinstance(logit_scale, nn.Parameter):
            logit_scale = logit_scale.exp()
        caption_logits = adapter_feats @ caption_embed.float().T * logit_scale
        caption_scores = nn.LogSoftmax(dim=-1)(caption_logits)

        if self.novel_grad_only:
            binary_labels = batch_dict["binary"]
            grad_mask = binary_labels == 1
            new_caption_scores = caption_scores.clone()
            new_caption_scores[grad_mask] = caption_scores[grad_mask].detach()

            # Use the new tensor for further operations
            caption_scores = new_caption_scores

        # Convert data to CSR format
        offset = batch_dict["offset"]
        zero_offset = torch.cat((torch.zeros(1, dtype=offset.dtype, device=offset.device), offset))
        corr_idx, offsets, non_empty, counts = convert_list_list_tensor_to_tensor(
            c2p_map,
            zero_offset,
            remove_empty_list=True,
            origin_idx=origin_idx,
            pc_count=batch_dict["pc_count"],
        )

        # Reduce caption scores
        rep_caption_scores = caption_scores[corr_idx]
        reduced_caption_scores = segment_csr(
            rep_caption_scores,
            offsets.to(caption_scores.device),
            reduce="mean",
        )

        loss = self.loss_func(reduced_caption_scores, caption_idx[non_empty])

        if self.reduce == "mean":
            return loss.mean()
        elif self.reduce == "weighted_sum":
            return ((loss * counts) / (counts.sum())).sum()
        else:
            raise ValueError(f"Unknown reduce type: {self.reduce}")
