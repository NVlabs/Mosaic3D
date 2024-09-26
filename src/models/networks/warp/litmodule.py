from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from overrides import override

from src.models.lightning_modules.language_module import DenseLanguageLitModule
from src.utils import RankedLogger
from warpconvnet.geometry.ops.voxel_ops import voxel_downsample_mapping
from warpconvnet.geometry.point_collection import PointCollection

log = RankedLogger(__name__, rank_zero_only=True)


class WarpLitModule(DenseLanguageLitModule):
    @override
    def _output_to_dict(self, output: Any, batch: Any) -> Dict[str, Any]:
        # clear cache memory
        torch.cuda.empty_cache()

        # binary, segment, and captions(c2pmap, caption_idx, origin_idx, offset)
        # Find mappings
        st = output["st"]
        voxel_size = st.stride[0] * st.voxel_size

        up_map, down_map, _ = voxel_downsample_mapping(
            torch.floor(
                batch["coord"] / voxel_size
            ),  # cchoy: Make sure the data_dict_to_input in network uses the same key (coord, not origin_coord)
            batch["offset"],
            st.coordinates,
            st.offsets,
            find_nearest_for_invalid=True,
        )

        # clip_feat
        output["clip_feat"] = output["clip_feat"][down_map]
        if not self.training:
            logits = self.clip_alignment_loss.predict(output["clip_feat"], return_logit=True)
            output["logits"] = logits

        # Save the mappings
        output["mapping_indices"] = (up_map, down_map)
        return output
