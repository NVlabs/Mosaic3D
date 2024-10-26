from typing import Dict, List, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from omegaconf import DictConfig
from overrides import override
from torch import Tensor
import functools
import numpy as np

from src.models.networks.network_base import NetworkBaseDict
from src.utils import RankedLogger
from src.models.components.structure import mean_pooling

from warpconvnet.nn.sequential import Sequential
from warpconvnet.geometry.base_geometry import SpatialFeatures
from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warpconvnet.models.internal.backbones.regionplc import SparseConvUNet
from warpconvnet.nn.pools import PointToSparseWrapper
from warpconvnet.utils.batch_index import (
    batch_index_from_offset,
    offsets_from_batch_index,
)

log = RankedLogger(__name__, rank_zero_only=True)


def to_warp_sparse_tensor(data_dict: Dict, hash_method: Literal["fnv", "ravel"] = "ravel"):
    # RegionPLC specific data processing.
    # Only used for matching the input format of RegionPLC.
    # TODO(cchoy): Use more optimized warpconvnet functions
    assert {"feat", "coord"}.issubset(data_dict.keys())
    if "grid_coord" not in data_dict.keys():
        # if you don't want to operate GridSampling in data augmentation,
        # please add the following augmentation into your pipeline:
        # dict(type="Copy", keys_dict={"grid_size": 0.01}),
        # (adjust `grid_size` to what your want)
        assert {"grid_size", "coord"}.issubset(data_dict.keys())
        grid_coord = (data_dict["coord"] / data_dict["grid_size"]).int()
        data_dict["grid_coord"] = grid_coord - grid_coord.min(0)[0]

    batch_indices = batch_index_from_offset(data_dict["offset"])
    batched_coords = torch.cat(
        [batch_indices.view(-1, 1).int(), data_dict["grid_coord"].int()], dim=1
    ).contiguous()

    voxel_coords, voxel_feats, v2p_map = mean_pooling(
        batched_coords.int(),
        data_dict["feat"],
        return_inverse=True,
        hash_method=hash_method,
    )
    data_dict["v2p_map"] = v2p_map
    offsets = offsets_from_batch_index(voxel_coords[:, 0])

    sparse_conv_feat = SpatiallySparseTensor(
        batched_features=voxel_feats,
        batched_coordinates=voxel_coords[:, 1:],
        offsets=offsets,
    )

    return sparse_conv_feat, data_dict


class MLP(nn.Sequential):
    def __init__(self, channels, norm_fn=None, num_layers=2, last_norm=False, last_bias=True):
        assert len(channels) >= 2
        modules = []
        for i in range(num_layers - 1):
            modules.append(nn.Linear(channels[i], channels[i + 1]))
            if norm_fn:
                modules.append(norm_fn(channels[i + 1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(channels[-2], channels[-1], bias=last_bias))
        if last_norm:
            modules.append(norm_fn(channels[-1]))
            modules.append(nn.ReLU())

        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        if isinstance(self[-1], nn.Linear):
            nn.init.normal_(self[-1].weight, 0, 0.01)
            nn.init.constant_(self[-1].bias, 0)


class Adapter(nn.Module):
    def __init__(
        self,
        in_channel: int,
        text_channel: int,
        num_layers: int,
        last_norm: bool = True,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.text_channel = text_channel
        self.last_norm = last_norm

        # vision adapter
        adapter_channel_list = [in_channel, text_channel]
        if num_layers == 2:
            multiplier = int(np.log2(text_channel / in_channel))
            adapter_channel_list = [in_channel, in_channel * multiplier, text_channel]

        self.adapter = MLP(
            adapter_channel_list,
            norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
            num_layers=num_layers,
            last_norm=last_norm,
        )

    def forward(self, sf: SpatialFeatures) -> SpatialFeatures:
        feats = self.adapter(sf.features)
        sf = sf.replace(batched_features=feats)
        return sf


class RegionPLCToCLIP(NetworkBaseDict):
    """Convert a point cloud to a language aligned feature vector using MinkUNet18."""

    def __init__(
        self,
        backbone_cfg: Optional[DictConfig] = None,
        adapter_cfg: Optional[DictConfig] = None,
        voxel_size: float = 0.025,
        point_to_voxel_method: Literal["wrapper", "mean"] = "mean",
        **kwargs,
    ):
        super().__init__()
        self.backbone = None
        self.adapter = None
        self.voxel_size = voxel_size
        self.point_to_voxel_method = point_to_voxel_method
        self.setup(backbone_cfg, adapter_cfg, point_to_voxel_method, **kwargs)

    def setup(
        self,
        backbone_cfg: DictConfig,
        adapter_cfg: DictConfig,
        point_to_voxel_method: Literal["wrapper", "mean"] = "mean",
        **kwargs,
    ):
        backbone = SparseConvUNet(**backbone_cfg)
        if point_to_voxel_method == "wrapper":
            self.backbone = PointToSparseWrapper(
                backbone,
                voxel_size=self.voxel_size,
                concat_unpooled_pc=False,
            )
        else:
            self.backbone = backbone
        self.adapter = Adapter(**adapter_cfg)

    def data_dict_to_input(self, data_dict: Dict):
        if self.point_to_voxel_method == "mean":
            st, data_dict = to_warp_sparse_tensor(data_dict)
            return st, data_dict
        else:
            pc = PointCollection(
                batched_features=data_dict["feat"],
                batched_coordinates=data_dict["coord"],
                offsets=data_dict["offset"],
            )
            return pc, data_dict

    @override
    def forward(self, data_dict: Dict) -> Dict[str, Tensor]:
        sf, data_dict = self.data_dict_to_input(data_dict)
        out_pc = self.backbone(sf)
        out_pc = self.adapter(out_pc)
        if self.point_to_voxel_method == "mean":
            out_feats = out_pc.features
            out_feats = out_feats[data_dict["v2p_map"]]
        else:
            out_feats = out_pc.features
        out_dict = dict(clip_feat=out_feats, pc=out_pc)
        return out_dict
