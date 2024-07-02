import functools

import numpy as np
import torch.nn as nn

from src.models.regionplc.utils import basic_block_1d


class KDHeadTemplate(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.in_feature_name = model_cfg.IN_FEAT_NAME
        self.feature_norm = model_cfg.FEAT_NORM

        self.criterion = nn.CosineSimilarity()
        self.forward_ret_dict = {}

    def forward(self, batch_dict):
        self.forward_ret_dict = {}
        feature = batch_dict[self.in_feature_name]
        out_feature = feature[batch_dict["v2p_map"].long()]

        if self.feature_norm:
            out_feature = out_feature / out_feature.norm(dim=-1, keepdim=True)

        if self.training:
            self.forward_ret_dict["output"] = out_feature
            self.forward_ret_dict["kd_labels"] = batch_dict["kd_labels"]
            self.forward_ret_dict["kd_labels_mask"] = batch_dict["kd_labels_mask"]
        return batch_dict

    def get_loss(self):
        pred = self.forward_ret_dict["output"]
        target = self.forward_ret_dict["kd_labels"]
        mask = self.forward_ret_dict["kd_labels_mask"]
        if target.shape[0] == mask.shape[0]:
            target = target[mask]
        else:
            assert target.shape[0] == mask.sum().item()
            pred = pred[mask]
        kd_loss = 1 - self.criterion(pred, target).mean()

        tb_dict = {"loss_kd": kd_loss.item()}
        return kd_loss, tb_dict


class BasicAdaptLayer(nn.Module):
    def __init__(self, block_cfg):
        super().__init__()
        self.block_cfg = block_cfg

        self.build_adaptation_layer(block_cfg)
        self.init_weights(weight_init="xavier")

    def init_weights(self, weight_init="xavier"):
        if weight_init == "kaiming":
            init_func = nn.init.kaiming_normal_
        elif weight_init == "xavier":
            init_func = nn.init.xavier_normal_
        elif weight_init == "normal":
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == "normal":
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def build_adaptation_layer(self, block_cfg):
        in_channel = block_cfg.in_channel
        out_channel = block_cfg.num_filters[0]
        num_adapter_layers = block_cfg.num_layers

        if num_adapter_layers == 1:
            mid_channel_list = [in_channel, out_channel]
        elif num_adapter_layers == 2:
            multiplier = int(np.log2(out_channel / in_channel))
            mid_channel_list = [in_channel, in_channel * multiplier, out_channel]
        else:
            raise NotImplementedError

        self.adapt_layer = basic_block_1d.MLP(
            mid_channel_list,
            norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
            num_layers=num_adapter_layers,
            last_norm_fn=False,
        )

    def forward(self, in_feature):
        out_feature = self.adapt_layer(in_feature)
        return out_feature
