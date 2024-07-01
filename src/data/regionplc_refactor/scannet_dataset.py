import os
from pathlib import Path

import numpy as np
import torch

from src.utils import RankedLogger

from .indoor_dataset import IndoorDataset

log = RankedLogger(__name__, rank_zero_only=False)


class ScanNetDataset(IndoorDataset):
    def __init__(self, dataset_cfg, class_names, training, root_path):
        super().__init__(dataset_cfg, class_names, training, root_path)

        self.data_suffix = dataset_cfg.DATA_SPLIT.data_suffix
        self.split_file = dataset_cfg.DATA_SPLIT[self.mode]
        with open(
            os.path.join(dataset_cfg.DATA_SPLIT["root"], f"scannetv2_{self.split_file}.txt"),
        ) as fin:
            data_list = sorted(fin.readlines())
        self.data_list = [
            str(
                Path(self.root_path)
                / self.split_file.split("_")[0]
                / f"{d.strip()}{self.data_suffix}"
            )
            for d in data_list
        ]

        if hasattr(self, "caption_cfg") and self.caption_cfg.get(
            "CAPTION_CORR_PATH_IN_ONE_FILE", True
        ):
            (
                self.scene_image_corr_infos,
                self.scene_image_corr_entity_infos,
            ) = self.include_point_caption_idx()

        log.info(
            "Totally {} samples in {} set.".format(
                len(self.data_list) * (self.repeat if self.training else 1), self.mode
            )
        )

    def __len__(self):
        length = len(self.data_list) * (self.repeat if self.training else 1)
        return length

    def load_data(self, index):
        fn = self.data_list[index]
        if self.split_file.find("test") < 0:
            xyz, rgb, label, inst_label, *others = torch.load(fn)
        else:
            xyz, rgb = torch.load(fn)
            label = np.full(xyz.shape[0], self.ignore_label)
            inst_label = np.full(xyz.shape[0], self.ignore_label)

        # base / novel label
        if hasattr(self, "base_class_mapper"):
            binary_label = self.binary_class_mapper[label.astype(np.int64)].astype(np.float32)
        else:
            binary_label = np.ones_like(label)
        if self.class_mode == "base":
            label = self.base_class_mapper[label.astype(np.int64)]
        elif self.class_mode == "all" and hasattr(self, "ignore_class_idx"):
            label = self.valid_class_mapper[label.astype(np.int64)]
        inst_label[label == self.ignore_label] = self.ignore_label

        return xyz, rgb, label, inst_label, binary_label

    def get_kd_data_v2(self, scene_name):
        random_idx = np.random.randint(0, 5)
        kd_feat_path = os.path.join(self.kd_label_dir, f"{scene_name}_{random_idx}.pt")
        kd_data = torch.load(kd_feat_path)
        kd_feats = kd_data["feat"].numpy()[..., 0]
        kd_feats = kd_feats / (np.linalg.norm(kd_feats, axis=-1, keepdims=True) + 1e-6)
        kd_mask_full = kd_data["mask_full"].numpy()
        kd_feats_all = np.zeros((kd_mask_full.shape[0], kd_feats.shape[1])).astype(np.float16)
        kd_feats_all[kd_mask_full.nonzero()[0]] = kd_feats
        kd_feat_mask = np.zeros(kd_mask_full.shape, dtype=bool)
        kd_feat_mask[kd_mask_full.nonzero()[0][kd_data["mask"]]] = True
        return kd_feats_all.astype(np.float16), kd_feat_mask

    def __getitem__(self, item):
        index = item % len(self.data_list)
        xyz, rgb, label, inst_label, binary_label, *others = self.load_data(index)

        # === caption ===
        scene_name = self.data_list[index].split("/")[-1].split(".")[0]

        # get captioning data
        caption_data = None
        if hasattr(self, "caption_cfg") and self.training:
            if self.caption_cfg.get("CAPTION_CORR_PATH_IN_ONE_FILE", True):
                (
                    image_corr_dict,
                    image_name_dict,
                ) = self.get_caption_image_corr_and_name_from_memory(scene_name, index)
            else:
                image_corr_dict, image_name_dict = self.get_caption_image_corr_and_name_from_file(
                    scene_name
                )

            if self.training:
                caption_data = self.select_caption_and_idx_all(
                    scene_name, image_name_dict, image_corr_dict
                )

        if not self.rgb_norm:
            rgb = (rgb + 1) * 127.5

        data_dict = {
            "points_xyz": xyz,
            "rgb": rgb,
            "labels": label,
            "inst_label": inst_label,
            "binary_labels": binary_label,
            "origin_idx": np.arange(xyz.shape[0]).astype(np.int64),
            "pc_count": xyz.shape[0],
            "caption_data": caption_data,
            "ids": index,
            "scene_name": scene_name,
        }

        # get kd data
        # KD label will only carrys on 3D zero-shot
        if self.load_kd_label:
            kd_labels, kd_mask = self.get_kd_data_v2(scene_name)
            if self.training:
                data_dict["kd_labels"] = kd_labels
                data_dict["kd_labels_mask"] = kd_mask
            else:
                data_dict["adapter_feats"] = kd_labels
                data_dict["adapter_feats_mask"] = kd_mask

        if self.training:
            # perform augmentations
            data_dict = self.augmentor.forward(data_dict)
            if not data_dict["valid"]:
                return ScanNetDataset.__getitem__(self, np.random.randint(self.__len__()))
        else:
            xyz_voxel_scale = xyz * self.voxel_scale
            xyz_voxel_scale -= xyz_voxel_scale.min(0)
            data_dict["points_xyz_voxel_scale"] = xyz_voxel_scale
            data_dict["points"] = xyz

        # prepare features for voxelization
        if self.dataset_cfg.DATA_PROCESSOR.rgb_as_feat:
            data_dict["feats"] = data_dict["rgb"]

        if self.dataset_cfg.DATA_PROCESSOR.xyz_as_feat:
            if "feats" in data_dict:
                data_dict["feats"] = np.concatenate(
                    (data_dict["feats"], data_dict["points_xyz"]), axis=1
                )
            else:
                data_dict["feats"] = data_dict["points_xyz"]

        return data_dict


class ScanNetInstDataset(ScanNetDataset):
    def __init__(self, dataset_cfg, class_names, training, root_path, logger=None, split=None):
        ScanNetDataset.__init__(
            self,
            dataset_cfg,
            class_names,
            training,
            root_path,
            logger=logger,
            split=split,
        )
        self.inst_class_idx = dataset_cfg.inst_class_idx
        self.inst_label_shift = dataset_cfg.inst_label_shift
        if "base_class_idx" in dataset_cfg:
            # instance seg, stuff first
            self.base_inst_class_idx = (
                np.array(self.base_class_idx)[dataset_cfg.inst_label_shift :]
                - self.inst_label_shift
            )
            self.novel_inst_class_idx = np.array(self.novel_class_idx) - self.inst_label_shift
        self.sem2ins_classes = dataset_cfg.sem2ins_classes
        self.NYU_ID = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)
        # get instance infos
        # info = self.get_instance_info(xyz_mid, inst_label.astype(np.int32), label)
        label, inst_label, binary_label = (
            data_dict["labels"],
            data_dict["inst_label"],
            data_dict["binary_labels"],
        )
        points = data_dict["points_xyz"]
        if self.training:
            inst_label[binary_label == 0] = self.ignore_label
        inst_label = self.get_valid_inst_label(inst_label, label != self.ignore_label)
        if self.training and inst_label.max() < 0:
            return ScanNetInstDataset.__getitem__(self, np.random.randint(self.__len__()))
        info = self.get_inst_info(points, inst_label.astype(np.int32), label)

        data_dict["inst_label"] = inst_label
        data_dict.update(info)
        return data_dict

    def get_inst_info(self, xyz, instance_label, semantic_label):
        ret = super().get_inst_info(xyz, instance_label, semantic_label)
        ret["inst_cls"] = [x - self.inst_label_shift if x != -100 else x for x in ret["inst_cls"]]
        return ret
