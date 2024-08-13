import functools
from typing import Optional

import spconv.pytorch as spconv
import torch.nn as nn

from src.models.regionplc_refactor.modules import ResidualBlock, UBlock, VGGBlock
from src.utils import RankedLogger

log = RankedLogger(__file__, rank_zero_only=True)


class SparseUNet(nn.Module):
    def __init__(
        self,
        in_channel: int,
        mid_channel: int,
        block_reps: int,
        block_residual: bool,
        custom_sp1x1: bool = False,
        num_blocks: Optional[int] = None,
        num_filters: Optional[int] = None,
    ):
        super().__init__()
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.block_reps = block_reps
        self.block_residual = block_residual
        self.custom_sp1x1: custom_sp1x1
        self.num_blocks = num_blocks
        self.num_filters = num_filters

        if self.block_residual:
            block = functools.partial(ResidualBlock, custom_sp1x1=custom_sp1x1)
        else:
            block = VGGBlock

        self.input_conv = spconv.SparseSequential(
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
        self.output_layer = spconv.SparseSequential(norm_fn(self.mid_channel), nn.ReLU())

        # init parameters
        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, sparse_tensor: spconv.SparseConvTensor):
        output = self.input_conv(sparse_tensor)
        output = self.unet(output)
        output = self.output_layer(output)

        return output
