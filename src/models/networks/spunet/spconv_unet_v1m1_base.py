"""SparseUNet Driven by SpConv (recommend)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from collections import OrderedDict
from functools import partial
from typing import List, Literal

import spconv.pytorch as spconv
import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from src.models.utils.misc import offset2batch
from src.models.utils.structure import Custom1x1Subm3d, Point


class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels,
        embed_channels,
        stride=1,
        norm_fn=None,
        indice_key=None,
        bias=False,
    ):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                Custom1x1Subm3d(in_channels, embed_channels, kernel_size=1, bias=False),
                norm_fn(embed_channels),
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


class Bottleneck(spconv.SparseModule):
    expansion = 4

    def __init__(
        self,
        in_channels,
        embed_channels,
        stride=1,
        norm_fn=None,
        indice_key=None,
        bias=False,
    ):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels * self.expansion:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                Custom1x1Subm3d(
                    in_channels,
                    embed_channels * self.expansion,
                    kernel_size=1,
                    bias=False,
                ),
                norm_fn(embed_channels * self.expansion),
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
            algo=spconv.ConvAlgo.Native,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
            algo=spconv.ConvAlgo.Native,
        )
        self.bn2 = norm_fn(embed_channels)
        self.conv3 = spconv.SubMConv3d(
            embed_channels,
            embed_channels * self.expansion,
            kernel_size=1,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
            algo=spconv.ConvAlgo.Native,
        )
        self.bn3 = norm_fn(embed_channels * self.expansion)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv3(out)
        out = out.replace_feature(self.bn3(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


class SpUNetBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        channels: List[int] = [32, 64, 128, 256, 256, 128, 96, 96],
        layers: List[int] = [2, 3, 4, 6, 2, 2, 2, 2],
        out_fpn: bool = False,
        hash_method: Literal["fnv", "ravel"] = "fnv",
        pooling_method: Literal["mean", "random"] = "mean",
        **kwargs,
    ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.out_fpn = out_fpn
        self.hash_method = hash_method
        self.pooling_method = pooling_method

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels,
                        channels[s],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    norm_fn(channels[s]),
                    nn.ReLU(),
                )
            )
            self.enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            # (f"block{i}", block(enc_channels, channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                            # if i == 0 else
                            (
                                f"block{i}",
                                block(
                                    channels[s],
                                    channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}",
                                ),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )
            # decode num_stages
            self.up.append(
                spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        channels[len(channels) - s - 2],
                        dec_channels,
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    norm_fn(dec_channels),
                    nn.ReLU(),
                )
            )
            self.dec.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            (
                                (
                                    f"block{i}",
                                    block(
                                        dec_channels + enc_channels,
                                        dec_channels,
                                        norm_fn=norm_fn,
                                        indice_key=f"subm{s}",
                                    ),
                                )
                                if i == 0
                                else (
                                    f"block{i}",
                                    block(
                                        dec_channels,
                                        dec_channels,
                                        norm_fn=norm_fn,
                                        indice_key=f"subm{s}",
                                    ),
                                )
                            )
                            for i in range(layers[len(channels) - s - 1])
                        ]
                    )
                )
            )

            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        self.final = Custom1x1Subm3d(channels[-1], out_channels, kernel_size=1, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_dict):
        if "grid_coord" in input_dict:
            grid_coord = input_dict["grid_coord"]
            feat = input_dict["feat"]
            offset = input_dict["offset"]
            batch = offset2batch(offset)
            sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 128).tolist()
            x = spconv.SparseConvTensor(
                features=feat,
                indices=torch.cat(
                    [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
                ).contiguous(),
                spatial_shape=sparse_shape,
                batch_size=batch[-1].tolist() + 1,
            )
        else:
            point = Point(input_dict)
            point.sparsify(
                pad=128,
                hash_method=self.hash_method,
                pooling_method=self.pooling_method,
            )
            x = point.sparse_conv_feat

        feature_maps = []
        x = self.conv_input(x)
        skips = [x]

        # enc forward
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)

        x = skips.pop(-1)
        feature_maps.append(x)

        # dec forward
        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            skip = skips.pop(-1)
            x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
            x = self.dec[s](x)

            if s != 0:
                feature_maps.append(x)

        x = self.final(x)
        feature_maps.append(x)

        if "grid_coord" in input_dict:
            return (x, feature_maps) if self.out_fpn else x
        else:
            point.sparse_conv_feat = x
            return (point, feature_maps) if self.out_fpn else point


class SpUNetBottleneck(SpUNetBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        **kwargs,
    ):
        super().__init__(in_channels, out_channels)
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = Bottleneck

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels * block.expansion if s > 0 else enc_channels,
                        channels[s] * block.expansion if s > 0 else channels[s],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                        algo=spconv.ConvAlgo.Native,
                    ),
                    norm_fn(channels[s] * block.expansion if s > 0 else channels[s]),
                    nn.ReLU(),
                )
            )
            self.enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            # (f"block{i}", block(enc_channels, channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                            # if i == 0 else
                            (
                                f"block{i}",
                                block(
                                    (
                                        channels[s] * block.expansion
                                        if (s > 0 or i > 0)
                                        else channels[s]
                                    ),
                                    channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}",
                                ),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )
            # decode num_stages
            self.up.append(
                spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        channels[len(channels) - s - 2] * block.expansion,
                        dec_channels,
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                        algo=spconv.ConvAlgo.Native,
                    ),
                    norm_fn(dec_channels),
                    nn.ReLU(),
                )
            )
            self.dec.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            (
                                (
                                    f"block{i}",
                                    block(
                                        (
                                            dec_channels + enc_channels * block.expansion
                                            if s > 0
                                            else dec_channels + enc_channels
                                        ),
                                        dec_channels,
                                        norm_fn=norm_fn,
                                        indice_key=f"subm{s}",
                                    ),
                                )
                                if i == 0
                                else (
                                    f"block{i}",
                                    block(
                                        dec_channels * block.expansion,
                                        dec_channels,
                                        norm_fn=norm_fn,
                                        indice_key=f"subm{s}",
                                    ),
                                )
                            )
                            for i in range(layers[len(channels) - s - 1])
                        ]
                    )
                )
            )

            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        self.final = Custom1x1Subm3d(
            channels[-1] * block.expansion, out_channels, kernel_size=1, bias=False
        )
        self.apply(self._init_weights)


if __name__ == "__main__":
    model_kargs = {
        "in_channels": 3,
        "out_channels": 512,
        "channels": (
            32,
            64,
            128,
            256,
            128,
            128,
            96,
            96,
        ),  # MinkUNet18A (OpenScene default)
        "layers": (2, 3, 4, 6, 2, 2, 2, 2),  # MinkUNet18A (OpenScene default),
    }

    # model = SpUNetBase(**model_kargs)
    model = SpUNetBottleneck(**model_kargs)
    print(model)
