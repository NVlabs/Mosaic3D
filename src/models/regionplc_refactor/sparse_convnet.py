import functools
from typing import Dict, List, Optional

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn

from src.models.components.structure import Point
from src.models.regionplc_refactor.backbone import SparseConvUNet
from src.models.regionplc_refactor.blocks import (
    MLP,
    ResidualBlock,
    UBlockDecoder,
    VGGBlock,
)


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
            last_norm_fn=last_norm,
        )

    def forward(self, x: Point) -> Point:  # noqa: F722
        feats = self.adapter(x.sparse_conv_feat.features)
        x.sparse_conv_feat = x.sparse_conv_feat.replace_feature(feats)
        return x


class BinaryHead(nn.Module):
    def __init__(
        self,
        ignore_label,
        in_channel,
        block_reps,
        block_residual,
        binary_thresh,
        num_blocks,
        hook_feature_list: List[str] = [],
        num_filters: Optional[int] = None,
        custom_sp1x1: bool = False,
        detach: bool = True,
        loss_weight: float = 1.0,
        voxel_loss: bool = False,
    ):
        super().__init__()
        self.binary_feat_input = []
        self.binary_thresh = binary_thresh
        self.num_blocks = num_blocks
        self.in_channel = in_channel
        self.ignore_label = ignore_label
        self.num_filters = num_filters
        self.hook_feature_list = hook_feature_list
        self.loss_weight = loss_weight
        self.voxel_loss = voxel_loss

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        if block_residual:
            block = functools.partial(ResidualBlock, custom_sp1x1=custom_sp1x1)
        else:
            block = VGGBlock

        if self.num_filters is not None:
            block_channels = self.num_filters
        else:
            assert self.num_blocks is not None
            block_channels = np.arange(1, 1 + self.num_blocks) * in_channel

        self.binary_encoder = UBlockDecoder(
            block_channels,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            detach=detach,
        )

        self.binary_classifier = spconv.SparseSequential(
            norm_fn(in_channel), nn.ReLU(), nn.Linear(in_channel, 1)
        )
        self.binary_loss_func = nn.BCEWithLogitsLoss()

        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, point: Point):
        binary_scores = self.binary_encoder(self.binary_feat_input)
        binary_scores = self.binary_classifier(binary_scores).features

        if not (self.training and self.voxel_loss):
            binary_scores = binary_scores[point.v2p_map.long()]

        binary_preds = (torch.sigmoid(binary_scores) > self.binary_thresh).long()
        # reset hooked features
        self.binary_feat_input = []

        point.binary_scores = binary_scores
        point.binary_preds = binary_preds

        return point

    def register_hook_for_binary_head(self, backbone):
        def get_features():
            def hook(model, input, output):
                self.binary_feat_input.append(output)

            return hook

        for module_name in self.hook_feature_list:
            eval("backbone." + module_name).register_forward_hook(get_features())


class SparseConvUNetTextSeg(nn.Module):
    def __init__(
        self,
        backbone_cfg,
        adapter_cfg,
        binary_head_cfg: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__()

        self.backbone = SparseConvUNet(**backbone_cfg)
        self.adapter = Adapter(**adapter_cfg)

        if binary_head_cfg is not None and binary_head_cfg:
            self.binary_head = BinaryHead(**binary_head_cfg)
            self.binary_head.register_hook_for_binary_head(self.backbone)

    def forward(self, batch_dict: Dict):
        point = Point(batch_dict)
        point.sparsify(pad=128)

        out_point = self.backbone(point)
        out_point = self.adapter(out_point)

        if hasattr(self, "binary_head") and self.binary_head is not None:
            out_point = self.binary_head(out_point)
        return out_point
