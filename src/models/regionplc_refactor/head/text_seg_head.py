from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

# from pcseg.config import cfg


class TextSegHead(nn.Module):
    def __init__(self, model_cfg, in_channel, ignore_label, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.in_channel = in_channel
        self.ignore_label = ignore_label
        self.text_channel = self.model_cfg.TEXT_EMBED.CHANNEL
        self.num_class = self.model_cfg.TEXT_EMBED.NUM_CLASS
        self.feat_norm = model_cfg.get("FEAT_NORM", False)
        self.loss_weight = self.model_cfg.get("LOSS_WEIGHT", 1.0)

        # create cls head
        self.cls_head = nn.Linear(self.text_channel, self.num_class, bias=False)

        if model_cfg.get("LOGIT_SCALE", None):
            self.logit_scale = (
                nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                if model_cfg.LOGIT_SCALE.learnable
                else model_cfg.LOGIT_SCALE.value
            )
        else:
            self.logit_scale = 1.0

        # fix text cls head
        for param in self.cls_head.parameters():
            param.requires_grad = False

        # open vocab
        self.valid_class_idx = [i for i in range(len(self.model_cfg.CLASS_NAMES))]
        if hasattr(self.model_cfg, "base_class_idx"):
            self.base_class_idx = self.model_cfg.base_class_idx
            self.novel_class_idx = self.model_cfg.novel_class_idx
        if hasattr(self.model_cfg, "ignore_class_idx"):
            self.ignore_class_idx = self.model_cfg.ignore_class_idx
            for i in self.ignore_class_idx:
                self.valid_class_idx.remove(i)

        self.seg_loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_label).cuda()
        self.forward_ret_dict = {}

    def set_cls_head_with_text_embed(self, text_embed):
        self.cls_head.load_state_dict(OrderedDict({"weight": text_embed.float()}))

    def forward(self, batch_dict):
        self.forward_ret_dict = {}

        adapter_feats = batch_dict["adapter_feats"]
        if self.feat_norm:
            adapter_feats = nn.functional.normalize(adapter_feats, dim=-1)

        if isinstance(self.logit_scale, nn.Parameter):
            logit_scale = self.logit_scale.exp()
        else:
            logit_scale = self.logit_scale

        semantic_scores = self.cls_head(adapter_feats) * logit_scale
        semantic_scores = semantic_scores[batch_dict["v2p_map"]]

        if self.training:
            if hasattr(self, "base_class_idx"):
                semantic_scores = semantic_scores[..., self.base_class_idx]
            else:
                semantic_scores = semantic_scores[..., self.valid_class_idx]
        else:
            new_semantic_scores = semantic_scores.detach().clone()
            new_semantic_scores[:] = -1e6
            new_semantic_scores[..., self.valid_class_idx] = semantic_scores[
                ..., self.valid_class_idx
            ]
            semantic_scores = new_semantic_scores

        # get semantic prediction results consider the binary calibrate
        if (not self.training) and batch_dict.get("binary_ret_dict"):
            binary_preds, semantic_preds = self.correct_seg_pred_with_binary_pred(
                batch_dict, semantic_scores
            )
        else:
            semantic_preds = semantic_scores.max(1)[1]
            if batch_dict.get("binary_ret_dict", None):
                binary_preds = batch_dict["binary_ret_dict"]["binary_preds"]
            else:
                binary_preds = None

        # for 2D fusion
        if not self.training and "adapter_feats_mask" in batch_dict:
            semantic_preds[~batch_dict["adapter_feats_mask"].bool().cuda()] = self.ignore_label

        # for captions
        batch_dict["seg_scores"] = semantic_scores
        batch_dict["seg_preds"] = semantic_preds

        self.forward_ret_dict["seg_scores"] = semantic_scores
        self.forward_ret_dict["seg_preds"] = semantic_preds
        self.forward_ret_dict["binary_preds"] = binary_preds

        # save gt label to forward_ret_dict
        self.forward_ret_dict["seg_labels"] = batch_dict["labels"]

        return batch_dict

    def get_loss(self):
        semantic_scores = self.forward_ret_dict["seg_scores"]
        semantic_labels = self.forward_ret_dict["seg_labels"]
        seg_loss = self.seg_loss_func(semantic_scores, semantic_labels) * self.loss_weight

        tb_dict = {"loss_seg": seg_loss.item()}
        return seg_loss, tb_dict

    def correct_seg_pred_with_binary_pred(self, batch_dict, semantic_scores):
        binary_preds = batch_dict["binary_ret_dict"]["binary_preds"]
        binary_scores = batch_dict["binary_ret_dict"]["binary_scores"]

        base_semantic_scores = semantic_scores[..., self.base_class_idx].softmax(dim=-1)
        novel_semantic_scores = semantic_scores[..., self.novel_class_idx].softmax(dim=-1)
        semantic_scores = semantic_scores.clone()
        semantic_scores[:] = 0.0
        semantic_scores[..., self.base_class_idx] = base_semantic_scores
        semantic_scores[..., self.novel_class_idx] = novel_semantic_scores
        sigmoid_binary_scores = torch.sigmoid(binary_scores)
        sigmoid_binary_scores = sigmoid_binary_scores.repeat(1, semantic_scores.shape[-1])
        sigmoid_binary_scores[..., self.novel_class_idx] = (
            1 - sigmoid_binary_scores[..., self.novel_class_idx]
        )

        semantic_scores = semantic_scores * sigmoid_binary_scores
        semantic_scores /= semantic_scores.sum(-1, keepdim=True)
        semantic_preds = semantic_scores.max(1)[1]
        return binary_preds, semantic_preds
