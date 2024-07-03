import functools
from typing import List, Optional

import torch
import torch.nn as nn

from src.models.regionplc.utils.spconv_utils import spconv
from src.models.regionplc.utils.unet_blocks import (
    ResidualBlock,
    UBlockDecoder,
    VGGBlock,
)


class BinaryHead(nn.Module):
    def __init__(
        self,
        ignore_label,
        in_channel,
        block_reps,
        block_residual,
        binary_thresh,
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
            # assert self.num_blocks is not None
            block_channels = [
                in_channel,
                2 * in_channel,
                3 * in_channel,
                4 * in_channel,
                5 * in_channel,
                6 * in_channel,
                7 * in_channel,
            ]

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
        self.forward_ret_dict = {}
        self.binary_loss_func = nn.BCEWithLogitsLoss()

        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, batch_dict):
        self.forward_ret_dict = {}
        binary_scores = self.binary_encoder(self.binary_feat_input)
        binary_scores = self.binary_classifier(binary_scores).features

        if self.training and self.voxel_loss:
            pass
        else:
            binary_scores = binary_scores[batch_dict["v2p_map"].long()]

        binary_preds = (torch.sigmoid(binary_scores) > self.binary_thresh).long()

        self.binary_feat_input = []
        self.forward_ret_dict["binary_scores"] = binary_scores
        self.forward_ret_dict["binary_preds"] = binary_preds
        if self.training:
            self.forward_ret_dict["binary_labels"] = batch_dict["binary_labels"]

        batch_dict["binary_ret_dict"] = self.forward_ret_dict
        return batch_dict

    def register_hook_for_binary_head(self, backbone):
        def get_features():
            def hook(model, input, output):
                self.binary_feat_input.append(output)

            return hook

        for module_name in self.hook_feature_list:
            eval("backbone." + module_name).register_forward_hook(get_features())

    def get_loss(self):
        binary_scores = self.forward_ret_dict["binary_scores"]
        binary_labels = self.forward_ret_dict["binary_labels"]

        # filter unannotated categories
        mask = binary_labels != self.ignore_label
        binary_loss = self.binary_loss_func(
            binary_scores[mask], binary_labels[mask].reshape(-1, 1)
        )
        binary_loss = binary_loss * self.loss_weight

        tb_dict = {"binary_loss": binary_loss.item()}
        return binary_loss, tb_dict
