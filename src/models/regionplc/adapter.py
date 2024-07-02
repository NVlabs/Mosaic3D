import functools

import numpy as np
import torch
import torch.nn as nn

from src.models.regionplc.utils import basic_block_1d


class VLAdapter(nn.Module):
    def __init__(self, model_cfg, in_channel):
        super().__init__()
        self.model_cfg = model_cfg
        self.text_channel = model_cfg.TEXT_DIM

        # vision adapter
        adapter_last_norm = self.model_cfg.get("LAST_NORM", True)
        self.adapter = self.build_vl_adapter(
            self.model_cfg.NUM_ADAPTER_LAYERS, in_channel, adapter_last_norm
        )

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
