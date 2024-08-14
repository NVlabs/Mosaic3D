from typing import Dict, List, Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor
from warp.convnet.geometry.base_geometry import BatchedFeatures
from warp.convnet.geometry.point_collection import (
    BatchedContinuousCoordinates,
    PointCollection,
)
from warp.convnet.models.point_conv_unet import PointConvUNet

from src.models.losses.caption_loss import CaptionLoss
from src.models.losses.clip_alignment_loss import ClipAlignmentHead
from src.models.warp.mlp import MLPBlock
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class Adapter(nn.Module):
    def __init__(
        self,
        in_channel: int,
        text_channel: int,
        last_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.text_channel = text_channel
        self.last_norm = last_norm
        self.eps = eps

        # vision adapter
        self.adapter = nn.Sequential(
            MLPBlock(in_channel, hidden_channels=text_channel, out_channels=text_channel),
            nn.Linear(text_channel, text_channel),
            nn.Identity() if not last_norm else nn.LayerNorm(text_channel),
        )

    def forward(self, x: PointCollection) -> Float[Tensor, "N C"]:  # noqa: F722
        feats = self.adapter(x.batched_features.batched_tensor)
        return feats


class PointConvUNetTextSeg(nn.Module):
    def __init__(
        self,
        module_names: List[str],
        backbone_cfg: Optional[DictConfig] = None,
        adapter_cfg: Optional[DictConfig] = None,
        task_head_cfg: Optional[DictConfig] = None,
        caption_head_cfg: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.module_list = []
        self.module_names = module_names

        self.backbone_3d = PointConvUNet(**backbone_cfg)
        self.adapter = Adapter(**adapter_cfg)

        self.task_head = None
        if "task_head" in self.module_names and task_head_cfg:
            self.task_head = ClipAlignmentHead(**task_head_cfg)
            self.module_list.append(self.task_head)

        self.caption_head = None
        if "caption_head" in self.module_names and caption_head_cfg is not None:
            self.caption_head = CaptionLoss(**caption_head_cfg)
            self.module_list.append(self.caption_head)

    def forward(self, batch_dict: Dict):
        offsets = batch_dict["offsets"].cpu().long()
        pc = PointCollection(batch_dict["points_xyz"], batch_dict["feats"], offsets=offsets)
        # Convert the dict to point collection
        out_pcs = self.backbone_3d(pc)
        adapter_feats = self.adapter(out_pcs[0])
        return adapter_feats

        # if self.training:
        #     loss, tb_dict, disp_dict = self.get_training_loss()
        #     ret_dict["loss"] = loss
        #     return ret_dict, tb_dict, disp_dict
        # else:
        #     if hasattr(self, "inst_head") and self.inst_head is not None:
        #         ret_dict.update(self.inst_head.forward_ret_dict)
        #     return ret_dict

    def get_training_loss(self):
        disp_dict = {}
        tb_dict = {}

        # for segmentation loss
        if self.task_head is not None and not self.task_head.eval_only:
            seg_loss, tb_dict_seg = self.task_head.get_loss()
            tb_dict.update(tb_dict_seg)
        else:
            seg_loss = 0

        # for caption loss
        if self.caption_head is not None:
            caption_loss, tb_dict_caption = self.caption_head.get_loss()
            tb_dict.update(tb_dict_caption)
        else:
            caption_loss = 0

        loss = seg_loss + caption_loss
        tb_dict["loss"] = loss.item()
        disp_dict.update(tb_dict)

        return loss, tb_dict, disp_dict
