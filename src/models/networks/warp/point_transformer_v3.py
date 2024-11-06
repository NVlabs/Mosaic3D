from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from omegaconf import DictConfig
from overrides import override
from torch import Tensor


from src.models.networks.network_base import NetworkBaseDict
from src.models.networks.warp.adapter import Adapter
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.models.backbones.point_transformer_v3 import PointTransformerV3
from warpconvnet.nn.pools import PointToSparseWrapper


class PointTransformerV3ToCLIP(NetworkBaseDict):
    """Convert a point cloud to a language aligned feature vector using PointTransformerV3."""

    def __init__(
        self,
        backbone_cfg: Optional[DictConfig] = None,
        adapter_cfg: Optional[DictConfig] = None,
        voxel_size: float = 0.025,
        **kwargs,
    ):
        super().__init__()
        self.backbone_3d = None
        self.voxel_size = voxel_size
        self.setup(backbone_cfg, adapter_cfg, **kwargs)

    def setup(self, backbone_cfg: DictConfig, adapter_cfg: DictConfig, **kwargs):
        self.backbone_3d = PointToSparseWrapper(
            nn.Sequential(PointTransformerV3(**backbone_cfg), Adapter(**adapter_cfg)),
            voxel_size=self.voxel_size,
            concat_unpooled_pc=False,
            reduction=kwargs.get("reduction", "mean"),
            unique_method=kwargs.get("unique_method", "torch"),
        )

    def data_dict_to_input(self, data_dict: Dict):
        # Convert the dict to point collection
        return PointCollection(
            batched_features=data_dict["feat"],
            batched_coordinates=data_dict["coord"],
            offsets=data_dict["offset"],
        )

    @override
    def forward(self, data_dict: Dict) -> Dict[str, Tensor]:
        pc = self.data_dict_to_input(data_dict)
        out_pc = self.backbone_3d(pc)
        out_dict = dict(clip_feat=out_pc.features, pc=out_pc)
        return out_dict
