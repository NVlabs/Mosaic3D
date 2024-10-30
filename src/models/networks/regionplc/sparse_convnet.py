import functools
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn

from src.models.components.structure import Point, PointModule, PointSequential
from src.models.networks.ppt.pdnorm import PDNorm
from src.utils import RankedLogger

log = RankedLogger(__file__, rank_zero_only=True)


class Custom1x1Subm3d(spconv.SparseConv3d):
    """# current 1x1 conv in Spconv2.x has a bug.

    It will be removed after the bug is fixed
    """

    def forward(self, input):
        features = torch.mm(
            input.features, self.weight.view(self.out_channels, self.in_channels).T
        )
        if self.bias is not None:
            features += self.bias
        out_tensor = spconv.SparseConvTensor(
            features, input.indices, input.spatial_shape, input.batch_size
        )
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class ResidualBlock(PointModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None, custom_sp1x1=False):
        super().__init__()

        if custom_sp1x1:
            spconv_1x1 = Custom1x1Subm3d
        else:
            spconv_1x1 = spconv.SubMConv3d

        if in_channels == out_channels:
            self.i_branch = PointSequential(nn.Identity())
        else:
            self.i_branch = PointSequential(
                spconv_1x1(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = PointSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
        )

    def forward(self, point: Point):
        identity = spconv.SparseConvTensor(
            point.sparse_conv_feat.features,
            point.sparse_conv_feat.indices,
            point.sparse_conv_feat.spatial_shape,
            point.sparse_conv_feat.batch_size,
        )
        output = self.conv_branch(point)
        out_feats = output.sparse_conv_feat.features + self.i_branch(identity).features
        output.sparse_conv_feat = output.sparse_conv_feat.replace_feature(out_feats)
        output.feat = output.sparse_conv_feat.features
        return output


class VGGBlock(PointModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()
        self.conv_layers = PointSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(PointModule):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):
        super().__init__()
        self.nPlanes = nPlanes

        blocks = {
            f"block{i}": block(
                nPlanes[0],
                nPlanes[0],
                norm_fn,
                indice_key=f"subm{indice_key_id}",
            )
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = PointSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = PointSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key=f"spconv{indice_key_id}",
                ),
            )

            self.u = UBlock(
                nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1
            )

            self.deconv = PointSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key=f"spconv{indice_key_id}",
                ),
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail[f"block{i}"] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key=f"subm{indice_key_id}",
                )
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = PointSequential(blocks_tail)

    def forward(self, point: Point):
        output_point = self.blocks(point)
        identity = spconv.SparseConvTensor(
            output_point.sparse_conv_feat.features,
            output_point.sparse_conv_feat.indices,
            output_point.sparse_conv_feat.spatial_shape,
            output_point.sparse_conv_feat.batch_size,
        )

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output_point)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat(
                (identity.features, output_decoder.sparse_conv_feat.features), dim=1
            )
            sparse_conv_feat = output_point.sparse_conv_feat
            sparse_conv_feat = sparse_conv_feat.replace_feature(out_feats)
            output_point.sparse_conv_feat = sparse_conv_feat
            output_point.feat = output_point.sparse_conv_feat.features
            output_point = self.blocks_tail(output_point)
        return output_point


class UBlockDecoder(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1, detach=True):
        super().__init__()

        self.nPlanes = nPlanes
        self.detach = detach
        if len(nPlanes) > 1:
            self.u = UBlockDecoder(
                nPlanes[1:],
                norm_fn,
                block_reps,
                block,
                indice_key_id=indice_key_id + 1,
                detach=detach,
            )

            self.deconv = PointSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key=f"spconv{indice_key_id}",
                ),
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail[f"block{i}"] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key=f"subm{indice_key_id}",
                )
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = PointSequential(blocks_tail)

    def forward(self, point_list: List[Point]):
        output_point = point_list[0]
        if self.detach:
            identity = spconv.SparseConvTensor(
                output_point.sparse_conv_feat.features.detach(),
                output_point.sparse_conv_feat.indices,
                output_point.sparse_conv_feat.spatial_shape,
                output_point.sparse_conv_feat.batch_size,
            )
        else:
            identity = spconv.SparseConvTensor(
                output_point.sparse_conv_feat.features,
                output_point.sparse_conv_feat.indices,
                output_point.sparse_conv_feat.spatial_shape,
                output_point.sparse_conv_feat.batch_size,
            )

        if len(self.nPlanes) > 1:
            output_decoder = self.u(point_list[1:])
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat(
                (identity.features, output_decoder.sparse_conv_feat.features), dim=1
            )
            sparse_conv_feat = output_point.sparse_conv_feat
            sparse_conv_feat = sparse_conv_feat.replace_feature(out_feats)
            output_point.sparse_conv_feat = sparse_conv_feat
            output_point.feat = output_point.sparse_conv_feat.features
            output_point = self.blocks_tail(output_point)

        return output_point


class MLP(PointSequential):
    def __init__(self, channels, norm_fn=None, num_layers=2, last_norm_fn=False, last_bias=True):
        assert len(channels) >= 2
        modules = []
        for i in range(num_layers - 1):
            modules.append(nn.Linear(channels[i], channels[i + 1]))
            if norm_fn:
                modules.append(norm_fn(channels[i + 1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(channels[-2], channels[-1], bias=last_bias))
        if last_norm_fn:
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


class SparseConvUNet(PointModule):
    def __init__(
        self,
        in_channel: int,
        mid_channel: int,
        block_reps: int,
        block_residual: bool,
        custom_sp1x1: bool = False,
        num_blocks: Optional[int] = None,
        num_filters: Optional[int] = None,
        pdnorm_bn: bool = False,
        pdnorm_decouple: bool = False,
        pdnorm_adaptive: bool = False,
        pdnorm_affine: bool = True,
        pdnorm_conditions: Optional[List[str]] = None,
        zero_init: bool = False,
    ):
        super().__init__()
        if pdnorm_bn:
            norm_fn = functools.partial(
                PDNorm,
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
                norm_layer=functools.partial(
                    nn.BatchNorm1d, eps=1e-4, momentum=0.1, affine=pdnorm_affine
                ),
            )
        else:
            norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.block_reps = block_reps
        self.block_residual = block_residual
        self.custom_sp1x1: custom_sp1x1
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.zero_init = zero_init

        if self.block_residual:
            block = functools.partial(ResidualBlock, custom_sp1x1=custom_sp1x1)
        else:
            block = VGGBlock

        self.input_conv = PointSequential(
            spconv.SubMConv3d(
                self.in_channel,
                self.mid_channel,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="subm1",
            )
        )

        if self.num_filters is not None:
            block_channels = self.num_filters
        else:
            assert self.num_blocks is not None
            block_channels = [self.mid_channel * (i + 1) for i in range(self.num_blocks)]

        self.unet = UBlock(block_channels, norm_fn, self.block_reps, block, indice_key_id=1)
        self.output_layer = PointSequential(norm_fn(self.mid_channel), nn.ReLU())

        # init parameters
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.BatchNorm1d):
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, PDNorm):
            if self.zero_init:
                nn.init.constant_(m.modulation[-1].weight, 0.0)
                nn.init.constant_(m.modulation[-1].bias, 0.0)

    def forward(self, point: Point):
        output = self.input_conv(point)
        output = self.unet(output)
        output = self.output_layer(output)
        return point


class Adapter(PointModule):
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
        output = self.adapter(x)
        return output


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

        self.binary_classifier = PointSequential(
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
