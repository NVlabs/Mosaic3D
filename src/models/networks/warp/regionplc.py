from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from omegaconf import DictConfig
from overrides import override
from torch import Tensor

from src.models.networks.network_base import NetworkBaseDict
from src.utils import RankedLogger
from warpconvnet.nn.sequential import Sequential
from warpconvnet.geometry.base_geometry import SpatialFeatures
from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.models.internal.backbones.regionplc import SparseConvUNet
from warpconvnet.nn.pools import PointToSparseWrapper

log = RankedLogger(__name__, rank_zero_only=True)


class Normalize(nn.Module):
    def __init__(self, p: int = 2, dim: int = -1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, p=self.p, dim=self.dim)


class ToFeatureAdapter(nn.Module):
    def __init__(
        self,
        in_channel: int,
        text_channel: int,
        last_norm: bool = False,
        normalize_output: bool = False,
    ):
        super().__init__()
        self.last_norm = last_norm
        self.normalize_output = normalize_output

        # vision adapter
        self.adapter = Sequential(
            nn.Linear(in_channel, text_channel),
            nn.Identity() if not last_norm else nn.LayerNorm(text_channel),
            Normalize(p=2, dim=-1) if normalize_output else nn.Identity(),
        )

    def forward(self, sf: SpatialFeatures) -> SpatialFeatures:
        return self.adapter(sf)


class RegionPLCToCLIP(NetworkBaseDict):
    """Convert a point cloud to a language aligned feature vector using MinkUNet18."""

    def __init__(
        self,
        backbone_cfg: Optional[DictConfig] = None,
        adapter_cfg: Optional[DictConfig] = None,
        voxel_size: float = 0.025,
        **kwargs,
    ):
        super().__init__()
        self.backbone_3d = None
        self.adapter = None
        self.voxel_size = voxel_size
        self.setup(backbone_cfg, adapter_cfg, **kwargs)

    def setup(self, backbone_cfg: DictConfig, adapter_cfg: DictConfig, **kwargs):
        self.backbone_3d = PointToSparseWrapper(
            nn.Sequential(SparseConvUNet(**backbone_cfg), ToFeatureAdapter(**adapter_cfg)),
            voxel_size=self.voxel_size,
            concat_unpooled_pc=False,
        )

    @override
    def data_dict_to_input(self, data_dict: Dict) -> Dict:
        pc = PointCollection(
            batched_features=data_dict["feat"],
            batched_coordinates=data_dict["coord"],
            offsets=data_dict["offset"],
        )
        return data_dict, pc

    @override
    def forward(self, data_dict: Dict) -> Dict[str, Tensor]:
        data_dict, pc = self.data_dict_to_input(data_dict)
        out_pc = self.backbone_3d(pc)
        out_feats = out_pc.features
        out_dict = dict(clip_feat=out_feats, pc=out_pc)
        return out_dict
