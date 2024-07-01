import copy
import os
import pickle

import numpy as np

from .augmentor.data_augmentor import DataAugmentor
from .base_dataset import DatasetTemplate


class IndoorDataset(DatasetTemplate):
    def __init__(self, data_dir: str, split: str, dataset_cfg=None, class_names=None):
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
            }
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

    # instance seg
    def get_valid_inst_label(self, inst_label, valid_mask=None):
        if valid_mask is not None:
            inst_label[~valid_mask] = self.ignore_label
        label_set = np.unique(inst_label[inst_label >= 0])
        if len(label_set) > 0:
            remapper = np.full((int(label_set.max()) + 1,), self.ignore_label)
            remapper[label_set.astype(np.int64)] = np.arange(len(label_set))
            inst_label[inst_label >= 0] = remapper[inst_label[inst_label >= 0].astype(np.int64)]
        return inst_label

    def get_inst_info(self, xyz, inst_label, semantic_label):
        inst_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * self.ignore_label
        # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        inst_pointnum = []  # (nInst), int
        inst_cls = []
        inst_num = int(inst_label.max()) + 1
        for i_ in range(inst_num):
            inst_idx_i = np.where(inst_label == i_)

            # inst_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            inst_info_i = inst_info[inst_idx_i]
            inst_info_i[:, 0:3] = mean_xyz_i
            inst_info_i[:, 3:6] = min_xyz_i
            inst_info_i[:, 6:9] = max_xyz_i
            inst_info[inst_idx_i] = inst_info_i

            # inst_pointnum
            inst_pointnum.append(inst_idx_i[0].size)

            # inst cls
            cls_idx = inst_idx_i[0][0]
            inst_cls.append(semantic_label[cls_idx])
        pt_offset_label = inst_info[:, 0:3] - xyz
        return {
            "inst_num": inst_num,
            "inst_pointnum": inst_pointnum,
            "inst_cls": inst_cls,
            "pt_offset_label": pt_offset_label,
        }

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
