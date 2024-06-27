from .dynamic_mean_vfe import DynamicMeanVFE
from .dynamic_mean_vfe_norange import DynamicMeanVFENoRange

# from .indoor_vfe import IndoorVFE
from .mean_vfe import MeanVFE
from .vfe_template import VFETemplate

__all__ = {
    "VFETemplate": VFETemplate,
    "MeanVFE": MeanVFE,
    "DynMeanVFE": DynamicMeanVFE,
    # "IndoorVFE": IndoorVFE,
    "DynamicMeanVFENoRange": DynamicMeanVFENoRange,
}
