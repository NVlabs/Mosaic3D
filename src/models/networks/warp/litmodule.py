from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from overrides import override
from warp.convnet.geometry.ops.voxel_ops import voxel_downsample_mapping
from warp.convnet.geometry.point_collection import PointCollection

from src.models.lightning_modules.language_module import DenseLanguageLitModule
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class WarpLitModule(DenseLanguageLitModule):
    @override
    def match_labels(self, batch: Dict[str, Any], pred_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Match the output of the network to the labels in the batch."""
        pc = pred_dict["pcs"][0]
        assert isinstance(pc, PointCollection)
        return batch

    @override
    def _output_to_dict(self, output: Any, batch: Any) -> Dict[str, Any]:
        # binary, segment, and captions(c2pmap, caption_idx, origin_idx, offset)
        # Find mappings
        pc = output["pcs"][0]

        orig_map, pc_map = voxel_downsample_mapping(
            pc.coordinate_tensor,
            pc.offsets,
            batch[
                "coord"
            ],  # cchoy: Make sure the data_dict_to_input in network uses the same key (coord, not origin_coord)
            batch["offset"],
            voxel_size=pc.voxel_size,
        )

        # clip_feat
        output["clip_feat"] = output["clip_feat"][orig_map]
        output["binary_scores"] = output["binary_scores"][orig_map]

        if not self.training:
            logits = self.clip_alignment_loss.predict(output["clip_feat"], return_logit=True)
            output["logits"] = logits
        return output
