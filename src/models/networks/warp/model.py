from typing import Dict, List, Optional, Tuple, Literal, Union

import importlib
from jaxtyping import Float
from omegaconf import DictConfig
from overrides import override
import hydra

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.networks.network_base import NetworkBaseDict
from src.utils import RankedLogger
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.models.backbones.mink_unet import MinkUNetBase
from warpconvnet.models.backbones.point_transformer_v3 import PointTransformerV3
from warpconvnet.nn.modules.sparse_pool import PointToSparseWrapper

log = RankedLogger(__name__, rank_zero_only=True)


class WarpModel(NetworkBaseDict):
    """Convert a point cloud to a language aligned feature vector using various backbones."""

    def __init__(
        self,
        backbone: nn.Module,
        voxel_size: float = 0.025,
        **kwargs,
    ):
        super().__init__()
        self.backbone = self.create_backbone(backbone, voxel_size, **kwargs)

    def create_backbone(
        self,
        backbone: nn.Module,
        voxel_size: float = 0.025,
        **kwargs,
    ) -> nn.Module:
        return PointToSparseWrapper(
            backbone,
            voxel_size=voxel_size,
            concat_unpooled_pc=kwargs.get("concat_unpooled_pc", False),
        )

    @override
    def data_dict_to_input(self, data_dict: Dict) -> Dict:
        # Convert the dict to point collection
        data_dict["pc"] = Points(
            batched_features=data_dict["feat"],
            batched_coordinates=data_dict["coord"],
            offsets=data_dict["offset"],
        ).to(self.device)
        return data_dict

    @override
    def forward(self, data_dict: Dict) -> Dict[str, Tensor]:
        data_dict = self.data_dict_to_input(data_dict)
        out_pc = self.backbone(data_dict["pc"])
        out_dict = dict(clip_feat=out_pc.features, pc=data_dict["pc"])
        return out_dict
