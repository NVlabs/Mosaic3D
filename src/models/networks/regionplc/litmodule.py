from typing import Any, Dict, Tuple

from overrides import override

from src.models.components.structure import Point
from src.models.lightning_modules.language_module import DenseLanguageLitModule
from src.models.lightning_modules.masked_language_module import MaskedDenseLanguageLitModule


class RegionPLCLitModule(DenseLanguageLitModule):
    @override
    def _output_to_dict(self, output: Any, batch: Any) -> Dict[str, Any]:
        assert isinstance(output, Point)
        output: Point = output
        clip_feat = output.sparse_conv_feat.features[output.v2p_map]
        out_dict = dict(point=output, clip_feat=clip_feat)
        # Check if binary scores are present
        if hasattr(output, "binary_scores"):
            out_dict["binary_scores"] = output.binary_scores
        return out_dict


class MaskedRegionPLCLitModule(MaskedDenseLanguageLitModule):
    @override
    def _output_to_dict(self, output: Any, batch: Any) -> Dict[str, Any]:
        assert isinstance(output, Point)
        output: Point = output
        clip_feat = output.sparse_conv_feat.features[output.v2p_map]
        out_dict = dict(point=output, clip_feat=clip_feat)
        # Check if binary scores are present
        if hasattr(output, "binary_scores"):
            out_dict["binary_scores"] = output.binary_scores
        return out_dict
