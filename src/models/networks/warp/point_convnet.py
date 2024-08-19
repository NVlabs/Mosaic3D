from typing import Dict, List, Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor
from warp.convnet.geometry.point_collection import PointCollection
from warp.convnet.models.point_conv_unet import PointConvUNet
from warp.convnet.nn.mlp import MLPBlock

from src.models.networks.network_base import NetworkBase
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ToFeatureAdapter(nn.Module):
    def __init__(
        self,
        in_channel: int,
        text_channel: int,
        last_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.text_channel = text_channel
        self.last_norm = last_norm
        self.eps = eps

        # vision adapter
        self.adapter = nn.Sequential(
            MLPBlock(in_channel, hidden_channels=text_channel, out_channels=text_channel),
            nn.Linear(text_channel, text_channel),
            nn.Identity() if not last_norm else nn.LayerNorm(text_channel),
        )

    def forward(self, x: PointCollection) -> Float[Tensor, "N C"]:  # noqa: F722
        feats = self.adapter(x.batched_features.batched_tensor)
        return feats


class PointConvUNetToCLIP(NetworkBase):
    """Convert a point cloud to a language aligned featuer vector."""

    def __init__(
        self,
        backbone_cfg: Optional[DictConfig] = None,
        adapter_cfg: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.backbone_3d = PointConvUNet(**backbone_cfg)
        self.adapter = ToFeatureAdapter(**adapter_cfg)

    def data_dict_to_input(self, batch_dict: Dict) -> Dict:
        offsets = batch_dict["offsets"].cpu().long()
        # Convert the dict to point collection
        pc = PointCollection(batch_dict["coord"], batch_dict["feats"], offsets=offsets).to(
            self.device
        )
        return pc

    def forward(self, batch_dict: Dict):
        pc = self.data_dict_to_input(batch_dict)
        # PointConvUNet returns a list of output from each layer from the last layer to the first layer
        # [Last layer, ..., First layer]
        out_pcs = self.backbone_3d(pc)
        adapter_feats = self.adapter(out_pcs[0])
        return adapter_feats
