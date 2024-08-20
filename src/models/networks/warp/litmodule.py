from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from warp.convnet.geometry.point_collection import PointCollection

from src.models.lightning_modules.language_module import DenseLanguageLitModule
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class WarpLitModule(DenseLanguageLitModule):
    def _output_to_dict(self, out_dict: Dict[str, Any]) -> Dict[str, Any]:
        clip_feat = out_dict["clip_feat"]
        if not self.training:
            logits = self.clip_alignment_loss.predict(clip_feat, return_logit=True)
            out_dict["logits"] = logits
        return out_dict
