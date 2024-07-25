from typing import Dict, List

import torch.nn as nn
from warp.convnet.models.point_conv_unet import PointConvUNet

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PointConvUNetTextSeg(nn.Module):
    def __init__(
        self,
        module_names: List[str],
        backbone_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.module_list = []
        self.module_names = module_names

        self.backbone_3d = PointConvUNet(**backbone_cfg)
        self.module_list.append(self.backbone_3d)

    def forward(self, batch_dict: Dict):
        batch_dict = self.backbone_3d(batch_dict)

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        ret_dict = self.task_head.forward_ret_dict
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict["loss"] = loss
            return ret_dict, tb_dict, disp_dict
        else:
            if hasattr(self, "inst_head") and self.inst_head is not None:
                ret_dict.update(self.inst_head.forward_ret_dict)
            return ret_dict

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
