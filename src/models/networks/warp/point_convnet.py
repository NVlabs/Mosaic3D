from typing import Dict, List, Optional, Tuple

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
        binary_channel: int = 0,
        last_norm: bool = True,
    ):
        super().__init__()
        channels = text_channel + binary_channel
        self.text_channel = text_channel
        self.use_binary = binary_channel > 0

        # vision adapter
        self.adapter = nn.Sequential(
            MLPBlock(in_channel, hidden_channels=text_channel, out_channels=channels),
            nn.Linear(channels, channels),
            nn.Identity() if not last_norm else nn.LayerNorm(channels),
        )

    def forward(
        self, x: PointCollection
    ) -> Tuple[Float[Tensor, "N C"], Float[Tensor, "N"]]:  # noqa: F722, F821
        feats = self.adapter(x.batched_features.batched_tensor)
        if self.use_binary:
            # split the features into text and binary features
            text_feats = feats[:, : self.text_channel]
            binary_feats = feats[:, self.text_channel :]
            return text_feats, binary_feats
        return (feats, None)


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
        offsets = batch_dict["offset"].cpu().long()
        # Convert the dict to point collection
        pc = PointCollection(batch_dict["coord"], batch_dict["feat"], offsets=offsets).to(
            self.device
        )
        return pc

    def forward(self, batch_dict: Dict) -> Dict[str, Tensor]:
        pc = self.data_dict_to_input(batch_dict)
        # PointConvUNet returns a list of output from each layer from the last layer to the first layer
        # [Last layer, ..., First layer]
        out_pcs = self.backbone_3d(pc)
        clip_feat, binary_scores = self.adapter(out_pcs[0])
        out_dict = dict(clip_feat=clip_feat, pcs=out_pcs)
        if binary_scores is not None:
            out_dict["binary_scores"] = binary_scores
        return out_dict
