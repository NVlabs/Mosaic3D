import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter

from src.models.regionplc.ops.pool_by_idx.pool_by_idx_utils import avg_pool_by_idx
from src.models.regionplc.utils.fp16 import force_fp32


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

        self.loss_func = nn.NLLLoss(ignore_index=ignore_label, reduction="none")
        self.loss_weight = loss_weight
        self.novel_grad_only = novel_grad_only

        self.forward_ret_dict = {}

    def forward(self, batch_dict):
        self.forward_ret_dict = {}
        if not self.training:
            return batch_dict

        caption_infos = batch_dict["caption_infos"]
        v2p_map = batch_dict["v2p_map"]
        adapter_feats = batch_dict["adapter_feats"][v2p_map]

        if isinstance(self.logit_scale, nn.Parameter):
            logit_scale = self.logit_scale.exp()
        else:
            logit_scale = self.logit_scale

        caption_ret_dict = {}

        caption_type = "caption_view"
        caption_embed = caption_infos[caption_type]["caption_embed"]
        if caption_embed.shape[0] == 0:
            caption_ret_dict[caption_type] = {"zero_loss": adapter_feats.sum() * 0.0}
        else:
            caption_scores = self.get_caption_scores(adapter_feats, caption_embed, logit_scale)
            caption_ret_dict[caption_type] = self.forward_given_type_caption(
                batch_dict,
                caption_infos[caption_type],
                caption_scores,
                logit_scale,
            )
        self.forward_ret_dict = caption_ret_dict
        return batch_dict

    def get_caption_scores(self, adapter_feats, normed_caption_embed, caption_logit_scale):
        if self.feat_norm:
            adapter_feats = nn.functional.normalize(adapter_feats, dim=-1)
        caption_logits = adapter_feats @ normed_caption_embed.float().T * caption_logit_scale
        caption_scores = nn.LogSoftmax(dim=-1)(caption_logits)
        return caption_scores

    def forward_given_type_caption(self, batch_dict, caption_info, caption_scores, logit_scale):
        ret_dict = {}
        pooled_scores, real_n_points, if_has_pts = self._forward_given_type_caption_cuda2(
            batch_dict,
            caption_info,
            caption_scores,
        )
        # if some scene don't have suitable image, len(pooled_features) == 0
        if len(pooled_scores) > 0:
            pooled_scores = torch.cat(pooled_scores, 0)
            real_n_points = torch.cat(real_n_points, 0)
            exist_caption_idx = torch.cat(if_has_pts, 0)

            caption_idx = caption_info["caption_idx"]
            caption_labels = self.prepare_caption_labels(caption_idx, exist_caption_idx)
            ret_dict = {
                "caption_output": pooled_scores,
                "caption_labels": caption_labels,
                "caption_n_points": real_n_points,
            }
        ret_dict["zero_loss"] = 0.0 * caption_scores.sum() * logit_scale
        return ret_dict

    @force_fp32(apply_to=("caption_scores"))
    def _forward_given_type_caption_cuda(self, batch_dict, caption_info, caption_scores):
        frame_corr_idx = caption_info["select_image_corr"]
        batch_idx = batch_dict["batch_idxs"]
        origin_idx = batch_dict["origin_idx"]  # (N, )
        origin_num_points = batch_dict["pc_count"]
        num_points = batch_idx.bincount()
        device = caption_scores.device

        pooled_scores = []
        if_has_pts = []
        real_n_points = []
        for b in range(len(frame_corr_idx)):
            caption_to_point_mapping = frame_corr_idx[b]
            N = len(caption_to_point_mapping)
            D = caption_scores.shape[-1]

            if N == 0:
                continue

            point_to_origin_mapping = torch.full(
                (origin_num_points[b],), -1, dtype=torch.long, device=device
            )
            point_to_origin_mapping[origin_idx[batch_idx == b]] = torch.arange(
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

            if self.novel_grad_only and "binary_labels" in batch_dict:
                if batch_dict["binary_labels"].shape[0] != batch_idx.shape[0]:
                    binary_labels = batch_dict["binary_labels"][batch_dict["v2p_map"]]
                else:
                    binary_labels = batch_dict["binary_labels"]
                base_mask = binary_labels[batch_idx == b] == 1
            else:
                base_mask = torch.zeros((batch_idx == b).sum().item())
            assert caption_scores.dtype == torch.float32

            cur_score = caption_scores[batch_idx == b]
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

    @force_fp32(apply_to=("to_pool_obj"))
    def _forward_given_type_caption_cuda2(self, batch_dict, caption_info, to_pool_obj):
        frame_corr_idx = caption_info["select_image_corr"]
        batch_idx = batch_dict["batch_idxs"]

        origin_idx = batch_dict["origin_idx"]  # (N, )
        pooled_objs = []
        if_has_pts = []
        real_n_points = []
        for b in range(len(frame_corr_idx)):
            cur_n_points = (batch_idx == b).sum().item()
            origin_to_cur_idx = torch.ones((batch_dict["pc_count"][b],)).long().cuda() * (-1)
            origin_to_cur_idx[origin_idx[batch_idx == b]] = torch.arange(cur_n_points).cuda()

            caption2point_idx = frame_corr_idx[b]
            caption_idx_offset = torch.LongTensor([0] + [len(idx) for idx in caption2point_idx])
            caption_idx_offset = torch.cumsum(caption_idx_offset, 0).cuda()
            if len(caption_idx_offset) - 1 == 0:
                continue
            if isinstance(caption2point_idx[0], np.ndarray):
                caption2point_idx = np.concatenate(caption2point_idx, 0)
                caption2point_idx = torch.from_numpy(caption2point_idx).long().cuda()
            else:
                caption2point_idx = torch.cat(caption2point_idx, 0).long().cuda()

            n_cap_per_point = None
            if self.novel_grad_only and "binary_labels" in batch_dict:
                if batch_dict["binary_labels"].shape[0] != batch_idx.shape[0]:
                    binary_labels = batch_dict["binary_labels"][batch_dict["v2p_map"]]
                else:
                    binary_labels = batch_dict["binary_labels"]
                base_mask = binary_labels[batch_idx == b] == 1
            else:
                base_mask = torch.zeros((batch_idx == b).sum().item())
            assert to_pool_obj.dtype == torch.float32
            _pooled_objs, _real_n_points = avg_pool_by_idx(
                to_pool_obj[batch_idx == b],
                origin_to_cur_idx,
                caption2point_idx,
                caption_idx_offset,
                n_cap_per_point,
                base_mask.bool(),
            )
            _if_has_pts = _real_n_points > 0
            pooled_objs.append(_pooled_objs[_if_has_pts])
            real_n_points.append(_real_n_points[_if_has_pts])
            if_has_pts.append(_if_has_pts)

        return pooled_objs, real_n_points, if_has_pts

    def prepare_caption_labels(self, caption_idx, exist_caption_idx=None):
        if exist_caption_idx is not None:
            caption_labels = caption_idx[exist_caption_idx]
        else:
            caption_labels = caption_idx

        return caption_labels

    def get_loss(self):
        caption_loss = 0
        tb_dict = {}
        for caption_type in self.forward_ret_dict:
            if "caption_output" in self.forward_ret_dict[caption_type]:
                caption_output = self.forward_ret_dict[caption_type]["caption_output"]
                caption_labels = self.forward_ret_dict[caption_type]["caption_labels"]
                cur_caption_loss_weight = self.loss_weight
                cur_caption_loss = (
                    self.loss_func(caption_output, caption_labels) * cur_caption_loss_weight
                )
                if len(cur_caption_loss.shape) > 0:
                    caption_n_points = self.forward_ret_dict[caption_type]["caption_n_points"]
                    assert len(cur_caption_loss) == len(caption_n_points)
                    cur_caption_loss = (
                        (cur_caption_loss * caption_n_points) / (caption_n_points.sum())
                    ).sum()
                tb_dict[caption_type] = cur_caption_loss.item()
            else:
                tb_dict[caption_type] = 0.0
                # if some GPUs don't have loss, some GPUs have loss for backward, the process will stuck
                cur_caption_loss = self.forward_ret_dict[caption_type]["zero_loss"]

            caption_loss += cur_caption_loss

        return caption_loss, tb_dict
