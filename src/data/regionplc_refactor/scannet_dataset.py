import copy
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted

from src.data.regionplc_refactor.augmentor.data_augmentor import DataAugmentor
from src.data.regionplc_refactor.base_dataset import DatasetTemplate
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class ScanNetDataset(DatasetTemplate):
    def __init__(self, data_dir, split, dataset_cfg, class_names):
        super().__init__(data_dir, split, dataset_cfg, class_names)

        self.repeat = dataset_cfg.DATA_PROCESSOR.repeat
        self.voxel_scale = dataset_cfg.DATA_PROCESSOR.voxel_scale
        self.max_npoint = dataset_cfg.DATA_PROCESSOR.max_npoint
        self.full_scale = dataset_cfg.DATA_PROCESSOR.full_scale
        self.point_range = dataset_cfg.DATA_PROCESSOR.point_range
        self.voxel_mode = dataset_cfg.DATA_PROCESSOR.voxel_mode
        self.rgb_norm = dataset_cfg.DATA_PROCESSOR.rgb_norm
        self.cache = dataset_cfg.DATA_PROCESSOR.cache
        self.downsampling_scale = dataset_cfg.DATA_PROCESSOR.get("downsampling_scale", 1)

        self.augmentor = DataAugmentor(
            self.dataset_cfg,
            **{
                "ignore_label": self.ignore_label,
                "voxel_scale": self.voxel_scale,
                "voxel_down": dataset_cfg.DATA_PROCESSOR.get("voxel_down", 1),
                "full_scale": self.full_scale,
                "max_npoint": self.max_npoint,
            },
        )

        self.voxel_size = [
            1.0 / self.voxel_scale,
            1.0 / self.voxel_scale,
            1.0 / self.voxel_scale,
        ]

        num_point_features = 0
        if dataset_cfg.DATA_PROCESSOR.xyz_as_feat:
            num_point_features += 3

        if dataset_cfg.DATA_PROCESSOR.rgb_as_feat:
            num_point_features += 3

        with open(f"src/data/metadata/split_files/scannetv2_{self.split}.txt") as f:
            data_list = natsorted([line.strip() for line in f.readlines()])
        self.data_list = [
            str(Path(self.data_dir) / self.split / f"{data}.pth") for data in data_list
        ]

        if hasattr(self, "caption_cfg") and self.caption_cfg.get(
            "CAPTION_CORR_PATH_IN_ONE_FILE", True
        ):
            (
                self.scene_image_corr_infos,
                self.scene_image_corr_entity_infos,
            ) = self.include_point_caption_idx()

        log.info(f"Totally {self.__len__()} samples in {self.split} set.")

    def __len__(self):
        length = len(self.data_list) * (self.repeat if self.training else 1)
        return length

    def get_caption_image_corr_and_name_from_memory(self, scene_name, index):
        image_name_dict = {}
        image_corr_dict = {}

        if self.caption_cfg.get("SCENE", None) and self.caption_cfg.SCENE.ENABLED:
            image_name_dict["scene"] = None
            image_corr_dict["scene"] = None

        if hasattr(self, "scene_image_corr_infos") and self.scene_image_corr_infos is not None:
            if isinstance(self.scene_image_corr_infos, dict):
                # assert scene_name in self.scene_image_corr_infos
                info = copy.deepcopy(self.scene_image_corr_infos.get(scene_name, {}))
            else:
                cur_caption_idx = copy.deepcopy(self.scene_image_corr_infos[index])
                assert scene_name == cur_caption_idx["scene_name"]
                info = cur_caption_idx["infos"]
            if len(info) > 0:
                image_name_view, image_corr_view = zip(*info.items())
            else:
                image_name_view, image_corr_view = [], []
            image_name_dict["view"] = image_name_view
            image_corr_dict["view"] = image_corr_view

        if (
            hasattr(self, "scene_image_corr_entity_infos")
            and self.scene_image_corr_entity_infos is not None
        ):
            if isinstance(self.scene_image_corr_entity_infos, dict):
                # assert scene_name in self.scene_image_corr_entity_infos
                info = copy.deepcopy(self.scene_image_corr_entity_infos.get(scene_name, {}))
            else:
                cur_caption_idx = copy.deepcopy(self.scene_image_corr_entity_infos[index])
                assert scene_name == cur_caption_idx["scene_name"]
                info = cur_caption_idx["infos"]
            if len(info) > 0:
                image_name_entity, image_corr_entity = zip(*info.items())
            else:
                image_name_entity, image_corr_entity = [], []
            image_name_dict["entity"] = image_name_entity
            image_corr_dict["entity"] = image_corr_entity

        return image_corr_dict, image_name_dict

    def get_caption_image_corr_and_name_from_file(self, scene_name):
        image_name_dict = {}
        image_corr_dict = {}

        if self.caption_cfg.get("SCENE", None) and self.caption_cfg.SCENE.ENABLED:
            image_name_dict["scene"] = None
            image_corr_dict["scene"] = None

        if self.caption_cfg.get("VIEW", None) and self.caption_cfg.VIEW.ENABLED:
            path = self.data_dir / self.caption_cfg.VIEW.IMAGE_CORR_PATH / (scene_name + ".pickle")
            if os.path.exists(path):
                info = pickle.load(open(path, "rb"))
            else:
                info = {}
            if len(info) > 0:
                image_name_view, image_corr_view = zip(*info.items())
            else:
                image_name_view = image_corr_view = []
            image_name_dict["view"] = image_name_view
            image_corr_dict["view"] = image_corr_view

        if self.caption_cfg.get("ENTITY", None) and self.caption_cfg.ENTITY.ENABLED:
            path = (
                self.data_dir / self.caption_cfg.ENTITY.IMAGE_CORR_PATH / (scene_name + ".pickle")
            )
            if os.path.exists(path):
                info = pickle.load(open(path, "rb"))
            else:
                info = {}
            if len(info) > 0:
                image_name_entity, image_corr_entity = zip(*info.items())
            else:
                image_name_entity = image_corr_entity = []
            image_name_dict["entity"] = image_name_entity
            image_corr_dict["entity"] = image_corr_entity

        return image_corr_dict, image_name_dict

    def load_data(self, index):
        fn = self.data_list[index]
        if self.split != "test":
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

    def include_point_caption_idx(self):
        if self.need_view_caption and self.caption_cfg.VIEW.get("IMAGE_CORR_PATH", None):
            corr_path = self.caption_cfg.VIEW.IMAGE_CORR_PATH
            corr_path = self.data_dir / corr_path
            point_caption_idx = pickle.load(open(corr_path, "rb"))
        else:
            point_caption_idx = None

        entity_point_caption_idx = None
        return point_caption_idx, entity_point_caption_idx
