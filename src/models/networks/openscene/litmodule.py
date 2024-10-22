from typing import Any, Dict

from overrides import override

from src.models.lightning_modules.language_module import DenseLanguageLitModule


class OpenSceneLitModule(DenseLanguageLitModule):
    @override
    def _output_to_dict(self, output: Any, batch: Any) -> Dict[str, Any]:
        assert isinstance(output, dict)
        output: dict = output
        clip_feat = output["clip_feat"]
        out_dict = dict(clip_feat=clip_feat)
        return out_dict
