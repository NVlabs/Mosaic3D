import functools
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from torch_scatter import scatter

from src.models.components.structure import Point
from src.models.regionplc.utils.fp16 import force_fp32
from src.models.regionplc_refactor.modules import ResidualBlock, UBlockDecoder, VGGBlock
from src.utils import RankedLogger

log = RankedLogger(__file__, rank_zero_only=True)


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

    def get_loss(self, binary_scores, binary_labels):
        mask = binary_labels != self.ignore_label
        binary_loss = self.binary_loss_func(
            binary_scores[mask],
            binary_labels[mask].reshape(-1, 1),
        )
        return binary_loss * self.loss_weight


class TextSegHead(nn.Module):
    def __init__(
        self,
        in_channel: int,
        text_embed_path: str,
        ignore_label: int = -100,
        feat_norm: bool = False,
        loss_weight: float = 1.0,
        logit_scale: Optional[float] = 1.0,
        logit_learnable: Optional[bool] = False,
        base_class_idx: Optional[List[int]] = [],
        ignore_class_idx: Optional[List[int]] = [],
        novel_class_idx: Optional[List[int]] = [],
        eval_only: bool = False,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.ignore_label = ignore_label
        self.text_embed_path = text_embed_path
        self.feat_norm = feat_norm
        self.loss_weight = loss_weight
        if logit_learnable:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.logit_scale = logit_scale if logit_scale is not None else 1.0
        self.eval_only = eval_only

        # load pre-computed text embeddings
        text_embeddings = torch.load(self.text_embed_path, map_location="cpu").detach()
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        log.info(f"=> loaded text embeddings from {self.text_embed_path}")
        num_classes, text_channel = text_embeddings.shape

        # create cls head
        self.cls_head = nn.Linear(text_channel, num_classes, bias=False)

        # load text embeddings as cls head weight
        self.cls_head.load_state_dict(OrderedDict({"weight": text_embeddings.float()}))

        # fix text cls head
        for param in self.cls_head.parameters():
            param.requires_grad = False

        # open vocab
        self.valid_class_idx = [i for i in range(num_classes)]
        if base_class_idx is not None and len(base_class_idx) > 0:
            self.base_class_idx = base_class_idx
            self.novel_class_idx = novel_class_idx
        if ignore_class_idx is not None and len(ignore_class_idx) > 0:
            self.ignore_class_idx = ignore_class_idx
            for i in self.ignore_class_idx:
                self.valid_class_idx.remove(i)

        self.seg_loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_label).cuda()
        self.forward_ret_dict = {}

    def forward(self, point: Point):
        sparse_tensor = point.sparse_conv_feat
        if self.feat_norm:
            features = nn.functional.normalize(sparse_tensor.features, dim=-1)
            sparse_tensor = sparse_tensor.replace_feature(features)

        if isinstance(self.logit_scale, nn.Parameter):
            logit_scale = self.logit_scale.exp()
        else:
            logit_scale = self.logit_scale

        semantic_scores = self.cls_head(sparse_tensor.features) * logit_scale
        semantic_scores = semantic_scores[point.v2p_map]

        if self.training:
            if hasattr(self, "base_class_idx"):
                semantic_scores = semantic_scores[..., self.base_class_idx]
            else:
                semantic_scores = semantic_scores[..., self.valid_class_idx]
        else:
            new_semantic_scores = semantic_scores.detach().clone()
            new_semantic_scores[:] = (
                -1e6
                if not semantic_scores.dtype == torch.float16
                else torch.finfo(semantic_scores.dtype).min
            )
            new_semantic_scores[..., self.valid_class_idx] = semantic_scores[
                ..., self.valid_class_idx
            ]
            semantic_scores = new_semantic_scores

        # get semantic prediction results consider the binary calibrate
        if (not self.training) and {"binary_preds", "binary_scores"}.issubset(point.keys()):
            semantic_preds = self.correct_seg_pred_with_binary_pred(
                point.binary_scores, semantic_scores
            )
        else:
            semantic_preds = semantic_scores.max(1)[1]

        point.seg_scores = semantic_scores
        point.seg_preds = semantic_preds
        return point

    def get_loss(self, seg_scores, seg_labels):
        seg_loss = self.seg_loss_func(seg_scores, seg_labels) * self.loss_weight
        return seg_loss

    def correct_seg_pred_with_binary_pred(self, binary_scores, semantic_scores):
        base_semantic_scores = semantic_scores[..., self.base_class_idx].softmax(dim=-1)
        novel_semantic_scores = semantic_scores[..., self.novel_class_idx].softmax(dim=-1)
        semantic_scores = semantic_scores.clone().float()
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
        return semantic_preds


class CaptionHead(nn.Module):
    def __init__(
        self,
        ignore_label: int,
        feat_norm: bool,
        loss_weight: float = 0.5,
        logit_scale: float = 100.0,
        logit_learnable: bool = True,
        novel_grad_only: bool = False,
    ):
        super().__init__()
        self.feat_norm = feat_norm

        if logit_learnable:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.logit_scale = logit_scale if logit_scale is not None else 1.0

        self.novel_grad_only = novel_grad_only
        self.loss_weight = loss_weight
        self.loss_func = nn.NLLLoss(ignore_index=ignore_label, reduction="none")

    def forward(self, point: Point):
        if not self.training:
            return point

        sparse_tensor = point.sparse_conv_feat
        adapter_feats = sparse_tensor.features[point.v2p_map]

        if isinstance(self.logit_scale, nn.Parameter):
            logit_scale = self.logit_scale.exp()
        else:
            logit_scale = self.logit_scale

        caption_embed = point.caption_embed
        if caption_embed.shape[0] == 0:
            caption_ret_dict = {"zero_loss": adapter_feats.sum() * 0.0}
        else:
            caption_scores = self.get_caption_scores(adapter_feats, caption_embed, logit_scale)
            caption_ret_dict = self.forward_given_type_caption(point, caption_scores, logit_scale)

        point.update(caption_ret_dict)
        return point

    def get_caption_scores(self, adapter_feats, normed_caption_embed, caption_logit_scale):
        if self.feat_norm:
            adapter_feats = nn.functional.normalize(adapter_feats, dim=-1)
        caption_logits = adapter_feats @ normed_caption_embed.float().T * caption_logit_scale
        caption_scores = nn.LogSoftmax(dim=-1)(caption_logits)
        return caption_scores

    def forward_given_type_caption(self, point, caption_scores, logit_scale):
        ret_dict = {}
        pooled_scores, real_n_points, if_has_pts = self._forward_given_type_caption_cuda(
            point, caption_scores
        )
        # if some scene don't have suitable image, len(pooled_features) == 0
        if len(pooled_scores) > 0:
            pooled_scores = torch.cat(pooled_scores, 0)
            real_n_points = torch.cat(real_n_points, 0)
            exist_caption_idx = torch.cat(if_has_pts, 0)

            caption_labels = self.prepare_caption_labels(point.caption_idx, exist_caption_idx)
            ret_dict = {
                "caption_output": pooled_scores,
                "caption_labels": caption_labels,
                "caption_n_points": real_n_points,
            }
        ret_dict["zero_loss"] = 0.0 * caption_scores.sum() * logit_scale
        return ret_dict

    @force_fp32(apply_to=("caption_scores"))
    def _forward_given_type_caption_cuda(self, point, caption_scores):
        c2p_map = point.c2p_map
        batch = point.batch
        pc_count = point.pc_count
        origin_idx = point.origin_idx

        num_points = batch.bincount()
        device = caption_scores.device

        pooled_scores = []
        if_has_pts = []
        real_n_points = []
        for b in range(len(c2p_map)):
            caption_to_point_mapping = c2p_map[b]
            N = len(caption_to_point_mapping)
            D = caption_scores.shape[-1]

            if N == 0:
                continue

            point_to_origin_mapping = torch.full(
                (pc_count[b],), -1, dtype=torch.long, device=device
            )
            point_to_origin_mapping[origin_idx[batch == b]] = torch.arange(
                num_points[b], dtype=torch.long, device=device
            )

            caption_idx_bincount = torch.tensor(
                [0] + [len(idx) for idx in caption_to_point_mapping],
                dtype=torch.long,
                device=device,
            )
            caption_idx = (
                torch.arange(
                    len(caption_idx_bincount), device=device, dtype=torch.long
                ).repeat_interleave(caption_idx_bincount)
                - 1
            )

            if isinstance(caption_to_point_mapping[0], np.ndarray):
                caption_to_point_mapping = np.concatenate(caption_to_point_mapping, 0)
                caption_to_point_mapping = (
                    torch.from_numpy(caption_to_point_mapping).long().to(device)
                )
            else:
                caption_to_point_mapping = torch.cat(caption_to_point_mapping, 0).long().to(device)

            binary_labels = point.binary
            if self.novel_grad_only and binary_labels is not None:
                if binary_labels.shape[0] != batch.shape[0]:
                    binary_labels = binary_labels[point.v2p_map]
                else:
                    binary_labels = binary_labels
                base_mask = binary_labels[batch == b] == 1
            else:
                base_mask = torch.zeros((batch == b).sum().item())
            assert caption_scores.dtype == torch.float32

            cur_score = caption_scores[batch == b]
            if (base_mask > 0).any():
                cur_score[base_mask] = cur_score[base_mask].detach()
            invalid = (point_to_origin_mapping == -1).float().unsqueeze(-1) * (-1)

            score_and_count = torch.cat(
                [
                    cur_score[point_to_origin_mapping[caption_to_point_mapping]],
                    invalid[caption_to_point_mapping],
                ],
                dim=-1,
            )

            output_score_and_count = torch.zeros((N, D + 1)).to(caption_scores)
            # apply sum pooling
            scatter(
                score_and_count,
                caption_idx,
                dim=0,
                out=output_score_and_count,
                reduce="sum",
            )
            output_score, output_count = output_score_and_count.split([D, 1], dim=-1)
            _real_n_points = caption_idx_bincount[1:] + output_count.squeeze()

            denom = torch.zeros_like(_real_n_points)
            denom[_real_n_points > 0] = 1 / _real_n_points[_real_n_points > 0]

            output_score = output_score * denom.unsqueeze(-1)

            _if_has_pts = _real_n_points > 0
            pooled_scores.append(output_score[_if_has_pts])
            real_n_points.append(_real_n_points[_if_has_pts])
            if_has_pts.append(_if_has_pts)

        return pooled_scores, real_n_points, if_has_pts

    def prepare_caption_labels(self, caption_idx, exist_caption_idx=None):
        if exist_caption_idx is not None:
            caption_labels = caption_idx[exist_caption_idx]
        else:
            caption_labels = caption_idx

        return caption_labels

    def get_loss(self, caption_output, caption_labels, caption_n_points):
        caption_loss = self.loss_func(caption_output, caption_labels) * self.loss_weight
        if len(caption_loss.shape) > 0:
            assert len(caption_loss) == len(caption_n_points)
            caption_loss = ((caption_loss * caption_n_points) / (caption_n_points.sum())).sum()

        return caption_loss
