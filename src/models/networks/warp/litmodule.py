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
        # clip_feat
        if not self.training:
            logits = self.clip_alignment_loss.predict(output["clip_feat"], return_logit=True)
            output["logits"] = logits

        return output
