import torch.nn as nn

from src.models.regionplc.adapter import VLAdapter
from src.models.regionplc.backbone import SparseUNetIndoor
from src.models.regionplc.head.binary_head import BinaryHead
from src.models.regionplc.head.caption_head import CaptionHead
from src.models.regionplc.head.kd_head import KDHeadTemplate
from src.models.regionplc.head.text_seg_head import TextSegHead
from src.models.regionplc.vfe import IndoorVFEv2
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SparseUNetTextSeg(nn.Module):
    def __init__(
        self,
        vfe_cfg=None,
        backbone_cfg=None,
        adapter_cfg=None,
        binary_head_cfg=None,
        kd_head_cfg=None,
        task_head_cfg=None,
        caption_head_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.module_list = []

        if vfe_cfg:
            self.vfe = IndoorVFEv2(**vfe_cfg)
            self.module_list.append(self.vfe)

        if backbone_cfg:
            self.backbone_3d = SparseUNetIndoor(**backbone_cfg)
            self.module_list.append(self.backbone_3d)

        if adapter_cfg:
            self.adapter = VLAdapter(**adapter_cfg)
            self.module_list.append(self.adapter)

        if binary_head_cfg:
            self.binary_head = BinaryHead(**binary_head_cfg)
            self.module_list.append(self.binary_head)

        self.kd_head = None
        if kd_head_cfg:
            self.kd_head = KDHeadTemplate(**kd_head_cfg)
            self.module_list.append(self.kd_head)

        if task_head_cfg:
            self.task_head = TextSegHead(**task_head_cfg)
            self.module_list.append(self.task_head)

        self.inst_head = None

        if caption_head_cfg is not None:
            self.caption_head = CaptionHead(**caption_head_cfg)
            self.module_list.append(self.caption_head)

        if hasattr(self, "binary_head") and self.binary_head is not None:
            log.info("=> registering binary head hook")
            self.binary_head.register_hook_for_binary_head(self.backbone_3d)

    def forward(self, batch_dict):
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
        if self.task_head is not None:
            seg_loss, tb_dict_seg = self.task_head.get_loss()
            tb_dict.update(tb_dict_seg)
        else:
            seg_loss = 0

        # for binary loss
        if self.binary_head is not None:
            binary_loss, tb_dict_binary = self.binary_head.get_loss()
            tb_dict.update(tb_dict_binary)
        else:
            binary_loss = 0

        # for caption loss
        if self.caption_head is not None:
            caption_loss, tb_dict_caption = self.caption_head.get_loss()
            tb_dict.update(tb_dict_caption)
        else:
            caption_loss = 0

        # for inst loss
        if self.inst_head is not None:
            inst_loss, tb_dict_inst = self.inst_head.get_loss()
            tb_dict.update(tb_dict_inst)
        else:
            inst_loss = 0

        # for distillation loss
        if self.kd_head is not None:
            kd_loss, tb_dict_kd = self.kd_head.get_loss()
            tb_dict.update(tb_dict_kd)
        else:
            kd_loss = 0

        loss = seg_loss + binary_loss + caption_loss + inst_loss + kd_loss
        tb_dict["loss"] = loss.item()
        disp_dict.update(tb_dict)

        return loss, tb_dict, disp_dict
