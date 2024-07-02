import os

import torch
import torch.nn as nn

from src.models.regionplc_refactor.adapter import VLAdapter
from src.models.regionplc_refactor.backbone import SparseUNetIndoor
from src.models.regionplc_refactor.utils.spconv_utils import find_all_spconv_keys
from src.models.regionplc_refactor.vfe import IndoorVFE

from . import head


class SparseUNetTextSeg(nn.Module):
    def __init__(self, model_cfg, *args, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.module_topology = [
            "vfe",
            "backbone_3d",
            "adapter",
            "binary_head",
            "kd_head",
            "task_head",
            "inst_head",
            "caption_head",
        ]
        self.module_list = self.build_networks()
        if model_cfg.get("BINARY_HEAD", None):
            self.binary_head.register_hook_for_binary_head(self.backbone_3d)

    @property
    def mode(self):
        return "TRAIN" if self.training else "TEST"

    def build_networks(self):
        model_info_dict = {
            "module_list": [],
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, "build_%s" % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict["module_list"]

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get("VFE", None) is None:
            return None, model_info_dict

        vfe_module = IndoorVFE(
            model_cfg=self.model_cfg.VFE,
            num_point_features=self.model_cfg.get("NUM_POINT_FEATURES", None),
            voxel_size=self.model_cfg.get("VOXEL_SIZE", None),
            voxel_mode=self.model_cfg.get("voxel_mode", None),
        )
        model_info_dict["module_list"].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get("BACKBONE_3D", None) is None:
            return None, model_info_dict

        backbone3d_module = SparseUNetIndoor(model_cfg=self.model_cfg.BACKBONE_3D)
        model_info_dict["module_list"].append(backbone3d_module)
        return backbone3d_module, model_info_dict

    def build_task_head(self, model_info_dict):
        if self.model_cfg.get("TASK_HEAD", None) is None:
            return None, model_info_dict
        if "IN_CHANNEL" in self.model_cfg.TASK_HEAD:
            in_channel = self.model_cfg.TASK_HEAD.IN_CHANNEL
        else:
            in_channel = self.model_cfg.BACKBONE_3D.MID_CHANNEL
        task_head_module = head.__all__[self.model_cfg.TASK_HEAD.NAME](
            model_cfg=self.model_cfg.TASK_HEAD,
            in_channel=in_channel,
            ignore_label=self.model_cfg.ignore_label,
            num_class=len(self.model_cfg.CLASS_NAMES),
        )
        model_info_dict["module_list"].append(task_head_module)
        return task_head_module, model_info_dict

    def build_inst_head(self, model_info_dict):
        if self.model_cfg.get("INST_HEAD", None) is None:
            return None, model_info_dict

        if "IN_CHANNEL" in self.model_cfg.TASK_HEAD:
            in_channel = self.model_cfg.TASK_HEAD.IN_CHANNEL
        else:
            in_channel = self.model_cfg.BACKBONE_3D.MID_CHANNEL

        if hasattr(self.model_cfg, "base_inst_class_idx"):
            base_inst_class_idx = self.model_cfg.base_inst_class_idx
            novel_inst_class_idx = self.model_cfg.novel_inst_class_idx
        else:
            base_inst_class_idx = novel_inst_class_idx = None

        inst_head_module = head.__all__[self.model_cfg.INST_HEAD.NAME](
            model_cfg=self.model_cfg.INST_HEAD,
            in_channel=in_channel,
            inst_class_idx=self.model_cfg.inst_class_idx,
            sem2ins_classes=self.model_cfg.sem2ins_classes,
            valid_class_idx=self.model_cfg.valid_class_idx,
            label_shift=self.model_cfg.inst_label_shift,
            ignore_label=self.model_cfg.ignore_label,
            base_inst_class_idx=base_inst_class_idx,
            novel_inst_class_idx=novel_inst_class_idx,
        )
        model_info_dict["module_list"].append(inst_head_module)
        return inst_head_module, model_info_dict

    def build_adapter(self, model_info_dict):
        if self.model_cfg.get("ADAPTER", None) is None:
            return None, model_info_dict

        adapter_module = VLAdapter(
            model_cfg=self.model_cfg.ADAPTER,
            in_channel=self.model_cfg.BACKBONE_3D.MID_CHANNEL,
        )
        model_info_dict["module_list"].append(adapter_module)
        return adapter_module, model_info_dict

    def build_binary_head(self, model_info_dict):
        if self.model_cfg.get("BINARY_HEAD", None) is None:
            return None, model_info_dict

        binary_head_module = head.__all__[self.model_cfg.BINARY_HEAD.NAME](
            model_cfg=self.model_cfg.BINARY_HEAD,
            ignore_label=self.model_cfg.ignore_label,
            in_channel=self.model_cfg.BACKBONE_3D.MID_CHANNEL,
            block_reps=self.model_cfg.BACKBONE_3D.BLOCK_REPS,
            block_residual=self.model_cfg.BACKBONE_3D.BLOCK_RESIDUAL,
        )
        model_info_dict["module_list"].append(binary_head_module)
        return binary_head_module, model_info_dict

    def build_caption_head(self, model_info_dict):
        if self.model_cfg.get("CAPTION_HEAD", None) is None:
            return None, model_info_dict

        caption_head_module = head.__all__[self.model_cfg.CAPTION_HEAD.NAME](
            model_cfg=self.model_cfg.CAPTION_HEAD,
            ignore_label=self.model_cfg.ignore_label,
        )
        model_info_dict["module_list"].append(caption_head_module)
        return caption_head_module, model_info_dict

    def build_kd_head(self, model_info_dict):
        if self.model_cfg.get("KD_HEAD", None) is None:
            return None, model_info_dict

        kd_head_module = head.__all__[self.model_cfg.KD_HEAD.NAME](
            model_cfg=self.model_cfg.KD_HEAD
        )
        model_info_dict["module_list"].append(kd_head_module)
        return kd_head_module, model_info_dict

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

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(
                    -1, -2
                )  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, "currently only spconv 3D is supported"
                    val_implicit = val.permute(
                        4, 0, 1, 2, 3
                    )  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, epoch_id=None, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info(
            "==> Loading parameters from checkpoint %s to %s"
            % (filename, "CPU" if to_cpu else "GPU")
        )
        loc_type = torch.device("cpu") if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        try:
            model_state_disk = checkpoint["model_state"]
        except Exception as e:
            model_state_disk = checkpoint["state_dict"]

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info("==> Checkpoint trained from version: %s" % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info(f"Not updated weight {key}: {str(state_dict[key].shape)}")

        logger.info("==> Done (loaded %d/%d)" % (len(update_model_state), len(state_dict)))

        if epoch_id and epoch_id == "no_number" and "epoch" in checkpoint:
            epoch_id = checkpoint["epoch"]
        return epoch_id

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info(
            "==> Loading parameters from checkpoint %s to %s"
            % (filename, "CPU" if to_cpu else "GPU")
        )
        loc_type = torch.device("cpu") if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get("epoch", -1)
        it = checkpoint.get("it", 0.0)

        self._load_state_dict(checkpoint["model_state"], strict=True)

        if optimizer is not None:
            if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
                logger.info(
                    "==> Loading optimizer parameters from checkpoint %s to %s"
                    % (filename, "CPU" if to_cpu else "GPU")
                )
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            else:
                assert filename[-4] == ".", filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = f"{src_file}_optim.{ext}"
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt["optimizer_state"])

        if "version" in checkpoint:
            print("==> Checkpoint trained from version: %s" % checkpoint["version"])
        logger.info("==> Done")

        return it, epoch
