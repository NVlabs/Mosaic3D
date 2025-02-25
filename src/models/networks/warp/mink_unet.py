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
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.models.backbones.mink_unet import MinkUNetBase
from warpconvnet.nn.modules.sparse_pool import PointToSparseWrapper

log = RankedLogger(__name__, rank_zero_only=True)


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
        self.adapter = nn.Sequential(
            nn.Linear(in_channel, text_channel),
            nn.Identity() if not last_norm else nn.LayerNorm(text_channel),
        )

    def forward(self, x: Voxels) -> Float[Tensor, "N C"]:  # noqa: F722, F821
        feats = self.adapter(x.features)
        if self.normalize_output:
            feats = F.normalize(feats, p=2, dim=-1)
        return feats


class MinkUNetToCLIP(NetworkBaseDict):
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
            MinkUNetBase(**backbone_cfg),
            voxel_size=self.voxel_size,
            concat_unpooled_pc=False,
        )
        self.adapter = None
        if adapter_cfg is not None:
            self.adapter = ToFeatureAdapter(**adapter_cfg)

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
        out_pc = self.backbone_3d(data_dict["pc"])
        if self.adapter is not None:
            clip_feat = self.adapter(out_pc)
            out_dict = dict(clip_feat=clip_feat, pc=data_dict["pc"])
        else:
            out_dict = dict(clip_feat=out_pc.features, pc=data_dict["pc"])
        return out_dict
