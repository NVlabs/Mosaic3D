from typing import Any, Dict, Tuple

import torch
from overrides import override

from src.models.lightning_modules.language_module import DenseLanguageLitModule
from src.models.networks.ptv3.model import Point


class PTv3LitModule(DenseLanguageLitModule):
    @override
    def _output_to_dict(self, output: Any, batch: Any) -> Dict[str, Any]:
        assert isinstance(output, Point)
        output: Point = output
        clip_feat = output.feat
        out_dict = dict(point=output, clip_feat=clip_feat)
        return out_dict
