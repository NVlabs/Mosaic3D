import numpy as np
import torch
import torch.nn as nn

from src.models.regionplc.ops.pool_by_idx.pool_by_idx_utils import avg_pool_by_idx
from src.models.regionplc.utils.fp16 import force_fp32


class CaptionHead(nn.Module):
    def __init__(self, model_cfg, ignore_label):
        super().__init__()
        self.model_cfg = model_cfg
        self.feat_norm = model_cfg.FEAT_NORM

        if model_cfg.LOGIT_SCALE.learnable:
            self.caption_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.caption_logit_scale = model_cfg.LOGIT_SCALE.value

        self.caption_loss_func = nn.NLLLoss(ignore_index=ignore_label, reduction="none")
        self.caption_loss_weight = model_cfg.LOSS_WEIGHT

        self.forward_ret_dict = {}

    def forward(self, batch_dict):
        self.forward_ret_dict = {}
        if not self.training:
            return batch_dict

        caption_infos = batch_dict["caption_infos"]
        v2p_map = batch_dict["v2p_map"]
        adapter_feats = batch_dict["adapter_feats"][v2p_map]

        if isinstance(self.caption_logit_scale, nn.Parameter):
            caption_logit_scale = self.caption_logit_scale.exp()
        else:
            caption_logit_scale = self.caption_logit_scale

        caption_ret_dict = {}

        caption_type = "caption_view"
        caption_embed = caption_infos[caption_type]["caption_embed"]
        if caption_embed.shape[0] == 0:
            caption_ret_dict[caption_type] = {"zero_loss": adapter_feats.sum() * 0.0}
        else:
            to_pool_obj = self.get_to_pool_obj(adapter_feats, caption_embed, caption_logit_scale)
            caption_ret_dict[caption_type] = self.forward_given_type_caption(
                batch_dict,
                caption_infos[caption_type],
                to_pool_obj,
                caption_logit_scale,
            )
        self.forward_ret_dict = caption_ret_dict
        return batch_dict

    def get_to_pool_obj(self, adapter_feats, normed_caption_embed, caption_logit_scale):
        if self.feat_norm:
            adapter_feats = nn.functional.normalize(adapter_feats, dim=-1)
        caption_scores = adapter_feats @ normed_caption_embed.float().T * caption_logit_scale
        to_pool_obj = nn.LogSoftmax(dim=-1)(caption_scores)
        return to_pool_obj

    def forward_given_type_caption(self, batch_dict, caption_info, to_pool_obj, logit_scale):
        ret_dict = {}
        pooled_objs, real_n_points, if_has_pts = self._forward_given_type_caption_cuda(
            batch_dict,
            caption_info,
            to_pool_obj,
        )
        # if some scene don't have suitable image, len(pooled_features) == 0
        if len(pooled_objs) > 0:
            pooled_objs = torch.cat(pooled_objs, 0)
            real_n_points = torch.cat(real_n_points, 0)
            exist_caption_idx = torch.cat(if_has_pts, 0)

            caption_idx = caption_info["caption_idx"]
            caption_labels = self.prepare_caption_labels(caption_idx, exist_caption_idx)
            ret_dict = {
                "caption_output": pooled_objs,
                "caption_labels": caption_labels,
                "caption_n_points": real_n_points,
            }
        ret_dict["zero_loss"] = 0.0 * to_pool_obj.sum() * logit_scale
        return ret_dict

    @force_fp32(apply_to=("to_pool_obj"))
    def _forward_given_type_caption_cuda(self, batch_dict, caption_info, to_pool_obj):
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
            if self.model_cfg.get("NOVEL_GRAD_ONLY", False) and "binary_labels" in batch_dict:
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

    def prepare_caption_loss_logit_and_labels(
        self,
        pooled_features,
        caption_embed,
        caption_logit_scale,
        caption_idx,
        exist_caption_idx=None,
    ):
        if self.feat_norm:
            pooled_features = nn.functional.normalize(pooled_features, dim=-1)
        normed_caption_embed = nn.functional.normalize(caption_embed, dim=-1)

        loss_type = self.model_cfg.get("LOSS_FUNC", "CrossEntropy")
        if loss_type == "CrossEntropy":
            caption_logit = pooled_features @ normed_caption_embed.float().T * caption_logit_scale

            if exist_caption_idx is not None:
                caption_labels = caption_idx[exist_caption_idx]
            else:
                caption_labels = caption_idx
        elif loss_type == "BYOL":
            caption_logit = pooled_features
            caption_labels = normed_caption_embed
            if exist_caption_idx is not None:
                caption_labels = caption_labels[exist_caption_idx]
        else:
            raise NotImplementedError

        return caption_logit, caption_labels

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
                cur_caption_loss_weight = self.caption_loss_weight[
                    caption_type.split("_")[-1].upper()
                ]
                cur_caption_loss = (
                    self.caption_loss_func(caption_output, caption_labels)
                    * cur_caption_loss_weight
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
