from typing import Any, Dict, Tuple

import torch

from src.models.components.structure import Point
from src.models.lightning_modules.language_module import DenseLanguageLitModule


class RegionPLCLitModule(DenseLanguageLitModule):
    def _output_to_dict(self, point: Any) -> Dict[str, Any]:
        clip_feat = point.sparse_conv_feat.features[point.v2p_map]
        out_dict = dict(point=point, clip_feat=clip_feat)
        # Check if binary scores are present
        if hasattr(point, "binary_scores"):
            out_dict["binary_scores"] = point.binary_scores

        if not self.training:
            logits = self.clip_alignment_loss.predict(clip_feat, return_logit=True)
            logits = logits[point.v2p_map]
            out_dict["logits"] = logits
        return out_dict
