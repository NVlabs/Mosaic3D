from typing import Dict, List, Optional, Tuple

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
) -> Tuple[Int[Tensor, "L"], Int[Tensor, "M + 1"], Int[Tensor, "M"]]:  # noqa: F722, F821
    # Concatenate inner lists first and generate offsets for the inner lists
    counts = [[len(l) for l in list_tensor] for list_tensor in list_list_tensor]  # noqa: E741
    list_tensor = [torch.cat(l, 0) for l in list_list_tensor]  # noqa: E741
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

    if remove_empty_list:
        offsets = torch.tensor([0] + cumsum[non_empty].tolist())
    else:
        offsets = torch.tensor([0] + cumsum.tolist())

    return tensor, offsets, non_empty


class CaptionLoss(LossBase):
    def __init__(
        self,
        normalize_input: bool = True,
        ignore_label: int = -100,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        self.loss_func = nn.NLLLoss(ignore_index=ignore_label, reduction="none")

    def forward(self, pc: PointCollection, batch_dict: Dict) -> Tensor:
        return pc, batch_dict

    def loss(self, pc: PointCollection, batch_dict):
        caption_infos = batch_dict["caption_infos"]
        adapter_feats = pc.feature_tensor
        if self.normalize_input:
            adapter_feats = nn.functional.normalize(adapter_feats, dim=-1)

        caption_type = "caption_view"
        caption_info = caption_infos[caption_type]

        # Extract caption information
        frame_corr_idx = caption_info["select_image_corr"]
        caption_embed = caption_info["caption_embed"]
        caption_idx = caption_info["caption_idx"]

        # Compute caption scores
        caption_logits = adapter_feats @ caption_embed.float().T
        caption_scores = nn.LogSoftmax(dim=-1)(caption_logits)

        # Convert data to CSR format
        corr_idx, offsets, non_empty = convert_list_list_tensor_to_tensor(
            frame_corr_idx, pc.offsets, remove_empty_list=True
        )

        # Reduce caption scores
        rep_caption_scores = caption_scores[corr_idx]
        reduced_caption_scores = segment_csr(
            rep_caption_scores,
            offsets.to(caption_scores.device),
            reduce="mean",
        )

        loss = self.loss_func(reduced_caption_scores, caption_idx[non_empty])
        return loss.mean()
