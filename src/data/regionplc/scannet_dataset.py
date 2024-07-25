import copy
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset

from src.data.regionplc.augmentor.data_augmentor import DataAugmentor
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class ScanNetDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        class_names: List[str],
        # processor
        repeat: int,
        point_range: int,
        voxel_scale: int,
        max_npoint: int,
        full_scale: List[int],
        rgb_norm: bool,
        xyz_as_feat: bool,
        rgb_as_feat: bool,
        min_spatial_shape: int,
        # labels
        ignore_label: int,
        base_class_idx: List[int],
        novel_class_idx: List[int],
        ignore_class_idx: List[int],
        # augmemtation
        aug_cfg: Dict,
        # caption
        caption_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.class_names = class_names
        # processor
        self.repeat = repeat
        self.voxel_scale = voxel_scale
        self.max_npoint = max_npoint
        self.full_scale = full_scale
        self.point_range = point_range
        self.rgb_norm = rgb_norm
        self.xyz_as_feat = xyz_as_feat
        self.rgb_as_feat = rgb_as_feat
        self.min_spatial_shape = min_spatial_shape
        # labels
        self.ignore_label = ignore_label
        self.base_class_idx = base_class_idx
        self.novel_class_idx = novel_class_idx
        self.ignore_class_idx = ignore_class_idx
        # augmentation
        self.aug_cfg = aug_cfg
        # caption
        self.caption_cfg = caption_cfg

        # setup class mappers
        self.n_classes = len(self.class_names)
        if self.base_class_idx is not None and len(self.base_class_idx) > 0:
            self.base_class_mapper = self.build_class_mapper(
                self.base_class_idx, self.ignore_label
            )
            self.binary_class_mapper = self.build_binary_class_mapper(
                self.base_class_idx,
                self.novel_class_idx,
                self.ignore_class_idx,
                self.ignore_label,
            )

        self.valid_class_idx = np.arange(self.n_classes).tolist()
        if self.ignore_class_idx is not None and len(self.ignore_class_idx) > 0:
            for c in self.ignore_class_idx:
                self.valid_class_idx.remove(c)
        self.valid_class_mapper = self.build_class_mapper(
            self.valid_class_idx, self.ignore_label, squeeze_label=self.training
        )

        self.class_mode = (
            "base" if (self.training and hasattr(self, "base_class_mapper")) else "all"
        )

        # load captions
        if self.training and self.caption_cfg is not None:
            self.caption_keys = self.caption_cfg.KEY
            self.caption = self.get_caption_items(self.caption_cfg)
            self.scene_image_corr_infos = self.include_point_caption_idx()
            self.scene_image_corr_entity_infos = None

        # data augmentation
        self.augmentor = DataAugmentor(
            aug_cfg,
            **{
                "ignore_label": self.ignore_label,
                "voxel_scale": self.voxel_scale,
                "voxel_down": 1,
                "full_scale": self.full_scale,
                "max_npoint": self.max_npoint,
            },
        )

        # read files
        with open(f"src/data/metadata/split_files/scannetv2_{self.split}.txt") as f:
            data_list = natsorted([line.strip() for line in f.readlines()])
        self.data_list = [
            str(Path(self.data_dir) / self.split / f"{data}.pth") for data in data_list
        ]

        log.info(f"Totally {self.__len__()} samples in {self.split} set.")

    def __len__(self):
        length = len(self.data_list) * (self.repeat if self.training else 1)
        return length

    @property
    def training(self):
        return self.split == "train"

    @staticmethod
    def build_class_mapper(class_idx, ignore_idx, squeeze_label=True):
        remapper = np.ones(256, dtype=np.int64) * ignore_idx
        for i, x in enumerate(class_idx):
            if squeeze_label:
                remapper[x] = i
            else:
                remapper[x] = x
        return remapper

    @staticmethod
    def build_binary_class_mapper(base_class_idx, novel_class_idx, ignore_class_idx, ignore_idx):
        remapper = np.ones(256, dtype=np.int64) * ignore_idx  # base: 1, novel: 0
        for _, x in enumerate(base_class_idx):
            remapper[x] = 1
        for _, x in enumerate(novel_class_idx):
            remapper[x] = 0
        # ignored categories are mapped to novel
        for _, x in enumerate(ignore_class_idx):
            remapper[x] = 0
        return remapper

    def get_caption_image_corr_and_name_from_memory(self, scene_name, index):
        image_name_dict = {}
        image_corr_dict = {}

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

        return image_corr_dict, image_name_dict

    def get_caption_image_corr_and_name_from_file(self, scene_name):
        image_name_dict = {}
        image_corr_dict = {}

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

        return image_corr_dict, image_name_dict

    def load_data(self, index):
        fn = self.data_list[index]
        if self.split != "test":
            xyz, rgb, label, inst_label, *others = torch.load(fn, weights_only=False)
        else:
            xyz, rgb = torch.load(fn, weights_only=False)
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

    def include_point_caption_idx(self):
        if self.caption_cfg.VIEW.get("IMAGE_CORR_PATH", None):
            corr_path = self.caption_cfg.VIEW.IMAGE_CORR_PATH
            corr_path = self.data_dir / corr_path
            point_caption_idx = pickle.load(open(corr_path, "rb"))
        else:
            point_caption_idx = None

        return point_caption_idx

    def get_caption_items(self, caption_cfg):
        caption_items = {}
        for key in caption_cfg:
            if key in self.caption_keys and caption_cfg[key].ENABLED:
                caption_path = os.path.join(self.data_dir, caption_cfg[key].CAPTION_PATH)
                caption_items[key.lower()] = copy.deepcopy(json.load(open(caption_path)))
        return caption_items

    def select_caption_and_idx_all(self, scene_name, image_name_dict, image_corr_dict):
        if not hasattr(self, "caption_cfg"):
            return None

        ret = {}
        for key in self.caption_cfg:
            if key in self.caption_keys and self.caption_cfg[key].ENABLED:
                key_lower = key.lower()
                ret[key_lower] = self.select_caption_and_idx(
                    self.caption[key_lower],
                    self.caption_cfg[key],
                    scene_name,
                    image_name_dict[key_lower],
                    image_corr_dict[key_lower],
                )
        return ret

    @staticmethod
    def select_caption_and_idx(caption, caption_cfg, scene_name, image_names, image_corr_indices):
        if image_corr_indices is None:
            select_captions = [caption[scene_name]]
            select_image_corr = [None]
        else:
            assert len(caption[scene_name]) == len(image_names)
            select_image_names, select_image_corr = ScanNetDataset.select_images(
                caption_cfg, image_names, image_corr_indices
            )  # list (B, K), (B, K, N)
            # (B*K)
            select_captions = [caption[scene_name][n] for n in select_image_names]
        return {"idx": select_image_corr, "caption": select_captions}

    @staticmethod
    def select_images(caption_cfg, image_name, image_corr):
        if caption_cfg.RATIO == 1.0:
            return image_name, image_corr

        if image_name is None or len(image_name) == 0:  # lack 2d data
            selected_idx = None
        else:
            ratio = caption_cfg.RATIO
            selected_idx = np.random.choice(
                len(image_name), max(1, int(len(image_name) * ratio)), replace=False
            )

        if selected_idx is not None:
            selected_image_name = np.array(image_name)[selected_idx].tolist()
            selected_image_corr = np.array(image_corr, dtype=object)[selected_idx].tolist()
        else:
            selected_image_name = []
            selected_image_corr = []

        return selected_image_name, selected_image_corr

    def __getitem__(self, item):
        index = item % len(self.data_list)
        xyz, rgb, label, inst_label, binary_label, *_ = self.load_data(index)
        scene_name = self.data_list[index].split("/")[-1].split(".")[0]

        # get captioning data
        caption_data = None
        if hasattr(self, "caption_cfg") and self.caption_cfg is not None:
            (
                image_corr_dict,
                image_name_dict,
            ) = self.get_caption_image_corr_and_name_from_memory(scene_name, index)

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
        if self.rgb_as_feat:
            data_dict["feats"] = data_dict["rgb"]

        if self.xyz_as_feat:
            if "feats" in data_dict:
                data_dict["feats"] = np.concatenate(
                    (data_dict["feats"], data_dict["points_xyz"]), axis=1
                )
            else:
                data_dict["feats"] = data_dict["points_xyz"]

        return data_dict
