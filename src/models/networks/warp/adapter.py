import functools
import numpy as np
import torch
import torch.nn as nn

from warpconvnet.geometry.base.geometry import Geometry


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

    def forward(self, sf: Geometry) -> Geometry:
        feats = self.adapter(sf.features)
        sf = sf.replace(batched_features=feats)
        return sf
