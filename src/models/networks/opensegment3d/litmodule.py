from typing import Any, Dict

import torch
from overrides import override

from src.models.lightning_modules.mask_language_module import MaskLanguageLitModule


class OpenSegment3DLitModule(MaskLanguageLitModule):
    @override
    def _output_to_dict(self, output: Any, batch: Any) -> Dict[str, Any]:
        assert isinstance(output, Dict)
        output: Dict = output

        backbone_point = output["backbone_point"]
        pred_voxel_masks = torch.vstack(output["mask"])
        pred_point_masks = pred_voxel_masks[backbone_point.v2p_map]

        offset = backbone_point.offset
        pred_masks = [pred_point_masks[offset[i] : offset[i + 1]] for i in range(len(offset) - 1)]

        output["mask"] = pred_masks
        return output
