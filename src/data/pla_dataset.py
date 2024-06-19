import copy
import glob
import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.utils.data as data
from natsort import natsorted

from src.data.data_augmentor import DataAugmentor
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PLADataset(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        class_names: List[int],
        caption_cfg: Dict,
        # data augmentor
        aug_cfg: Dict,
        voxel_scale: float,
        full_scale,
        voxel_down,
        max_npoints,
        # data processor
        voxel_size: float,
        #
        base_class_idx: Optional[List[int]] = None,
        ignore_class_idx: Optional[List[int]] = None,
        novel_class_idx: Optional[List[int]] = None,
        ignore_label: int = -100,
    ):
        self.data_dir = data_dir
        self.split = split
        self.caption_cfg = caption_cfg

        self.data_paths = natsorted(glob.glob(os.path.join(self.data_dir, split, "*.pth")))

        # class mapper
        self.valid_class_idx = np.arange(len(class_names)).tolist()
        if base_class_idx is not None:
            self.base_class_mapper = self.build_class_mapper(base_class_idx, ignore_label)
            self.binary_class_mapper = self.build_binary_class_mapper(
                base_class_idx, novel_class_idx, ignore_class_idx, ignore_label
            )
        if ignore_class_idx is not None:
            for c in ignore_class_idx:
                self.valid_class_idx.remove(c)
        self.valid_class_mapper = self.build_class_mapper(
            self.valid_class_idx, ignore_label, squeeze_label=self.split == "train"
        )
        self.ignore_label = ignore_label

        # load caption
        captions = {}
        for key in caption_cfg:
            if caption_cfg[key].enabled:
                caption_path = os.path.join(data_dir, caption_cfg[key].caption_path)
                captions[key.lower()] = copy.deepcopy(json.load(open(caption_path)))
        self.captions = captions

        self.scene_image_corr_infos = pickle.load(
            open(os.path.join(self.data_dir, caption_cfg.view.image_corr_path), "rb")
        )

        # data processing
        self.augmentor = DataAugmentor(
            aug_cfg,
            **{
                "ignore_label": ignore_label,
                "voxel_scale": voxel_scale,
                "voxel_down": voxel_down,
                "full_scale": full_scale,
                "max_npoint": max_npoints,
            },
        )
        self.voxel_size = voxel_size

    @staticmethod
    def build_class_mapper(class_idx, ignore_label, squeeze_label=True):
        remapper = np.ones(256, dtype=np.int64) * ignore_label
        for i, x in enumerate(class_idx):
            if squeeze_label:
                remapper[x] = i
            else:
                remapper[x] = x
        return remapper

    @staticmethod
    def build_binary_class_mapper(base_class_idx, novel_class_idx, ignore_class_idx, ignore_label):
        remapper = np.ones(256, dtype=np.int64) * ignore_label  # base: 1, novel: 0
        for _, x in enumerate(base_class_idx):
            remapper[x] = 1
        for _, x in enumerate(novel_class_idx):
            remapper[x] = 0
        # ignored categories are mapped to novel
        for _, x in enumerate(ignore_class_idx):
            remapper[x] = 0
        return remapper

    def __len__(self):
        return self.data_paths

    def load_data(self, filepath):
        xyz, rgb, label, inst_label = torch.load(filepath)

        # class mapping
        semantic_label = self.valid_class_mapper[label.astype(np.int64)]
        inst_label[label == self.ignore_label] = self.ignore_label
        binary_label = np.ones_like(label)
        if hasattr(self, "base_class_mapper"):
            binary_label = self.binary_class_mapper[label.astype(np.int64)].astype(np.float32)
        return xyz, rgb, semantic_label, inst_label, binary_label

    def load_caption(self, scene_name, index):
        image_name_dict, image_corr_dict = {}, {}
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
        return image_name_dict, image_corr_dict

    def select_caption_and_idx_all(self, scene_name, image_name_dict, image_corr_dict):
        if not hasattr(self, "caption_cfg"):
            return None

        ret = {}
        for key in self.caption_cfg:
            if self.caption_cfg[key].enabled:
                key_lower = key.lower()
                ret[key_lower] = self.select_caption_and_idx(
                    self.captions[key_lower],
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
            assert len(caption[scene_name]) == len(
                image_names
            ), f"len caption: {len(caption[scene_name])}, len images: {len(image_names)}"
            select_image_names, select_image_corr = PLADataset.select_images(
                caption_cfg, image_names, image_corr_indices
            )  # list (B, K), (B, K, N)
            # (B*K)
            select_captions = [caption[scene_name][n] for n in select_image_names]
        return {"idx": select_image_corr, "caption": select_captions}

    @staticmethod
    def select_images(caption_cfg, image_name, image_corr):
        """
        TODO: put this part into dataset
        Select part of images for training
        """

        if caption_cfg.get("SAMPLE", 1) > 1:
            random_start = np.random.randint(caption_cfg.SAMPLE)
            image_name = (np.array(image_name)[random_start :: caption_cfg.SAMPLE]).tolist()
            image_corr = (
                np.array(image_corr, dtype=object)[random_start :: caption_cfg.SAMPLE]
            ).tolist()
        if caption_cfg.SELECT == "ratio" and caption_cfg.RATIO == 1.0:
            return image_name, image_corr

        if image_name is None or len(image_name) == 0:  # lack 2d data
            selected_idx = None
        elif caption_cfg.SELECT == "fixed":
            # view-level caotion: random select fixed number
            num = int(caption_cfg.NUM)
            selected_idx = np.random.choice(
                len(image_name), min(num, len(image_name)), replace=False
            )
        elif caption_cfg.SELECT == "ratio":
            ratio = caption_cfg.RATIO
            selected_idx = np.random.choice(
                len(image_name), max(1, int(len(image_name) * ratio)), replace=False
            )
        elif caption_cfg.SELECT == "hybrid":
            num = min(int(caption_cfg.NUM), int(len(image_name) * caption_cfg.RATIO))
            selected_idx = np.random.choice(
                len(image_name), min(max(1, num), len(image_name)), replace=False
            )
        elif caption_cfg.SELECT == "ratio_list":
            ratio_list = caption_cfg.RATIO_LIST
            ratio = caption_cfg.RATIO
            image_idx_sources = []
            if len(ratio_list) == 2:
                image_idx_sources.append(
                    [i for i in range(len(image_name)) if not image_name[i].startswith("app")]
                )
                image_idx_sources.append(
                    [i for i in range(len(image_name)) if image_name[i].startswith("app")]
                )
            elif len(ratio_list) > 2:
                image_idx_sources.append(
                    [i for i in range(len(image_name)) if not image_name[i].startswith("app")]
                )
                for jj in range(len(ratio_list) - 1):
                    image_idx_jj = [
                        i for i in range(len(image_name)) if image_name[i].startswith(f"app_{jj}")
                    ]
                    image_idx_sources.append(image_idx_jj)
            assert len(image_idx_sources) == len(ratio_list)

            sample_ratios = []
            for image_idx_source, desired_ratio in zip(image_idx_sources, ratio_list):
                sample_ratios.append(desired_ratio / (len(image_idx_source) + 1e-6))
            sample_ratios = np.array(sample_ratios) / (max(sample_ratios) + 1e-6) * ratio

            selected_idx = []
            for ii in range(len(sample_ratios)):
                if len(image_idx_sources[ii]) == 0:
                    continue
                selected_idx_ii = np.random.choice(
                    image_idx_sources[ii],
                    max(1, int(len(image_idx_sources[ii]) * sample_ratios[ii])),
                    replace=False,
                )
                selected_idx.append(selected_idx_ii)
            selected_idx = np.concatenate(selected_idx, axis=0)
        else:
            raise NotImplementedError

        if selected_idx is not None:
            selected_image_name = np.array(image_name)[selected_idx].tolist()
            selected_image_corr = np.array(image_corr, dtype=object)[selected_idx].tolist()
        else:
            selected_image_name = []
            selected_image_corr = []

        return selected_image_name, selected_image_corr

    def __getitem__(self, idx):
        filepath = self.data_paths[idx]
        scene_name = filepath.split("/")[-1][: -len(".pth")]

        xyz, rgb, semantic_label, inst_label, binary_label = self.load_data(filepath)
        if self.split == "train":
            image_name_dict, image_corr_dict = self.load_caption(scene_name, idx)
            caption = self.select_caption_and_idx_all(scene_name, image_name_dict, image_corr_dict)

        metadata = {"idx": idx, "scene_name": scene_name, "filepath": filepath}

        data_dict = {
            "points_xyz": xyz,
            "rgb": rgb,
            "labels": semantic_label,
            "inst_label": inst_label,
            "binary_labels": binary_label,
            "caption_data": caption,
            "origin_idx": np.arange(xyz.shape[0]).astype(np.int64),
            "metadata": metadata,
        }

        if self.split == "train":
            data_dict = self.augmentor.forward(data_dict)
        else:
            xyz_voxel_scale = xyz * self.voxel_scale
            xyz_voxel_scale -= xyz_voxel_scale.min(0)
            data_dict["points_xyz_voxel_scale"] = xyz_voxel_scale
            data_dict["points"] = xyz

        data_dict["feats"] = np.concatenate((data_dict["rgb"], data_dict["points_xyz"]), axis=1)
        # data_dict = self.data_processor.forward(data_dict)
        return data_dict


if __name__ == "__main__":
    from easydict import EasyDict as edict

    args = dict(
        data_dir="/home/junhal/datasets/pla",
        split="train",
        class_names=[
            "wall",
            "floor",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "bookshelf",
            "picture",
            "counter",
            "desk",
            "curtain",
            "refrigerator",
            "showercurtain",
            "toilet",
            "sink",
            "bathtub",
            "otherfurniture",
        ],
        caption_cfg={
            "scene": {"enabled": False, "caption_path": "text_embed/caption.json"},
            "view": {
                "enabled": True,
                "caption_path": "text_embed/caption_detic-template_and_kosmos_125k_iou0.2.json",
                "image_corr_path": "scannet_caption_idx_detic-template_and_kosmos_125k_iou0.2.pkl",
                "SELECT": "ratio",
                "NUM": 1,
                "RATIO": 1.0,
                "SAMPLE": 1,
                "GATHER_CAPTION": False,
            },
            "entity": {
                "enabled": False,
                "caption_path": "text_embed/caption_2d_intersect_v3.json",
                "image_corr_path": "scannetv2_matching_idx_intersect_v3",
                "SELECT": "ratio",
                "NUM": 1,
                "RATIO": 1.0,
            },
        },
        aug_cfg={
            "AUG_LIST": [],
            "scene_aug": {
                "scaling_scene": {"enabled": False, "p": 1.0, "value": [0.9, 1.1]},
                "rotation": {"p": 1.0, "value": [0.0, 0.0, 1.0]},
                "jitter": True,
                "color_jitter": True,
                "flip": {"p": 0.5},
                "random_jitter": {
                    "enabled": False,
                    "value": 0.01,
                    "accord_to_size": False,
                    "p": 1.0,
                },
            },
            "elastic": {
                "enabled": True,
                "value": [[6, 40], [20, 160]],
                "apply_to_feat": False,
                "p": 1.0,
            },
            "crop": {"step": 32},
            "shuffle": True,
        },
        voxel_scale=50,
        full_scale=[128, 512],
        voxel_down=1,
        max_npoints=250000,
        voxel_size=0.02,
        base_class_idx=[0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18],
        ignore_class_idx=[19],
        novel_class_idx=[5, 9, 12, 16],
        ignore_label=-100,
    )

    args_dict = edict(args)

    dataset = PLADataset(**args_dict)
    it = iter(dataset)

    batch = next(it)
