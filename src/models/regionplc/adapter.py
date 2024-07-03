import functools

import numpy as np
import torch
import torch.nn as nn

from src.models.regionplc.utils import basic_block_1d


class VLAdapter(nn.Module):
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
        self.num_layers = num_layers
        self.last_norm = last_norm

        # vision adapter
        self.adapter = self.build_vl_adapter(self.num_layers, self.in_channel, self.last_norm)

    def build_vl_adapter(self, num_adapter_layers, in_channel, last_norm):
        if num_adapter_layers < 1:
            return None

        if num_adapter_layers == 1:
            mid_channel_list = [in_channel, self.text_channel]
        elif num_adapter_layers == 2:
            multiplier = int(np.log2(self.text_channel / in_channel))
            mid_channel_list = [in_channel, in_channel * multiplier, self.text_channel]
        else:
            raise NotImplementedError

        adapter = basic_block_1d.MLP(
            mid_channel_list,
            norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
            num_layers=num_adapter_layers,
            last_norm_fn=last_norm,
        )
        return adapter

    def forward(self, batch_dict):
        backbone3d_feats = batch_dict["backbone_3d_feats"]

        # forward adapter
        if hasattr(self, "adapter") and self.adapter is not None:
            adapter_feats = self.adapter(backbone3d_feats)
        else:
            adapter_feats = backbone3d_feats
        torch.cuda.empty_cache()

        batch_dict["adapter_feats"] = adapter_feats
        return batch_dict
