from typing import Any, Dict

from overrides import override

from src.models.lightning_modules.language_module import DenseLanguageLitModule
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class WarpLitModule(DenseLanguageLitModule):
    @override
    def _output_to_dict(self, output: Any, batch: Any) -> Dict[str, Any]:
        return output
