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
from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.models.backbones.point_conv_unet import PointConvEncoderDecoder, PointConvUNet
from warpconvnet.nn.mlp import ResidualMLPBlock

log = RankedLogger(__name__, rank_zero_only=True)


class ToFeatureAdapter(nn.Module):
    def __init__(
        self,
        in_channel: int,
        text_channel: int,
        binary_channel: int = 0,
        last_norm: bool = False,
        normalize_output: bool = False,
    ):
        super().__init__()
        channels = text_channel + binary_channel
        self.text_channel = text_channel
        self.use_binary = binary_channel > 0
        self.last_norm = last_norm
        self.normalize_output = normalize_output

        # vision adapter
        self.adapter = nn.Sequential(
            ResidualMLPBlock(in_channel, hidden_channels=text_channel, out_channels=channels),
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
        else:
            text_feats = feats
            binary_feats = None

        if self.normalize_output:
            text_feats = F.normalize(text_feats, p=2, dim=-1)

        return text_feats, binary_feats


class PointConvUNetToCLIP(NetworkBaseDict):
    """Convert a point cloud to a language aligned featuer vector."""

    def __init__(
        self,
        backbone_cfg: Optional[DictConfig] = None,
        adapter_cfg: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.backbone_3d = None
        self.adapter = None
        self.setup(backbone_cfg, adapter_cfg, **kwargs)

    def setup(self, backbone_cfg: DictConfig, adapter_cfg: DictConfig, **kwargs):
        self.backbone_3d = PointConvUNet(**backbone_cfg)
        self.adapter = ToFeatureAdapter(**adapter_cfg)

    @override
    def data_dict_to_input(self, data_dict: Dict) -> Dict:
        offsets = data_dict["offset"].cpu().long()
        # Convert the dict to point collection
        pc = PointCollection(data_dict["coord"], data_dict["feat"], offsets=offsets).to(
            self.device
        )
        return pc

    @override
    def forward(self, data_dict: Dict) -> Dict[str, Tensor]:
        pc = self.data_dict_to_input(data_dict)
        # PointConvUNet returns a list of output from each layer from the last layer to the first layer
        # [Last layer, ..., First layer]
        out_pcs = self.backbone_3d(pc)
        clip_feat, binary_scores = self.adapter(out_pcs[0])
        out_dict = dict(clip_feat=clip_feat, pcs=out_pcs)
        if binary_scores is not None:
            out_dict["binary_scores"] = binary_scores
        return out_dict


class PointConvEncoderDecoderToCLIP(NetworkBaseDict):
    def __init__(
        self,
        backbone_cfg: Optional[DictConfig] = None,
        adapter_cfg: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.backbone_3d = None
        self.adapter = None
        self.setup(backbone_cfg, adapter_cfg, **kwargs)

    def setup(self, backbone_cfg: DictConfig, adapter_cfg: DictConfig, **kwargs):
        self.backbone_3d = PointConvEncoderDecoder(**backbone_cfg)
        self.adapter = ToFeatureAdapter(**adapter_cfg)

    @override
    def data_dict_to_input(self, data_dict: Dict) -> Dict:
        offsets = data_dict["offset"].cpu().long()
        # Convert the dict to point collection
        pc = PointCollection(data_dict["coord"], data_dict["feat"], offsets=offsets).to(
            self.device
        )
        return pc

    @override
    def forward(self, data_dict: Dict) -> Dict[str, Tensor]:
        pc = self.data_dict_to_input(data_dict)
        # PointConvUNet returns a list of output from each layer from the last layer to the first layer
        # [Last layer, ..., First layer]
        out_pc, dec_pcs, enc_pcs = self.backbone_3d(pc)
        clip_feat, binary_scores = self.adapter(out_pc)
        out_dict = dict(clip_feat=clip_feat, pcs=[out_pc, *dec_pcs, *enc_pcs])
        if binary_scores is not None:
            out_dict["binary_scores"] = binary_scores
        return out_dict
