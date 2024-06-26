import copy
import glob
import json
import os
import pickle
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import torch.utils.data as data
from natsort import natsorted
from omegaconf import OmegaConf

from src.data.metadata.scannet import VALID_CLASS_IDS_20, VALID_CLASS_IDS_200
from src.data.transform import TRANSFORMS, Compose
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class RegionPLCDataset(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        caption_cfg: Dict,
        transform_cfg: None,
        preset: str = Literal["scannet", "scannet200"],
        base_class_idx: Optional[List[int]] = None,
        ignore_class_idx: Optional[List[int]] = None,
        novel_class_idx: Optional[List[int]] = None,
        ignore_label: int = -100,
    ):
        self.data_dir = data_dir
        self.split = split
        self.caption_cfg = caption_cfg

        # data paths
        self.data_paths = natsorted(glob.glob(os.path.join(self.data_dir, split, "*.pth")))

        # transforms
        self.transform_cfg = transform_cfg
        self.transform = Compose(OmegaConf.to_container(transform_cfg))

        # class mapper
        class_names = VALID_CLASS_IDS_200 if preset == "scannet200" else VALID_CLASS_IDS_20
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

        # load image correspondences
        self.scene_image_corr_infos = pickle.load(
            open(os.path.join(self.data_dir, caption_cfg.view.image_corr_path), "rb")
        )

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

        # normalize rgb to [0,255]
        rgb = np.clip((rgb + 1.0) * 127.5, 0, 255)

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
            select_image_names, select_image_corr = RegionPLCDataset.select_images(
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

        if caption_cfg.get("sample", 1) > 1:
            random_start = np.random.randint(caption_cfg.sample)
            image_name = (np.array(image_name)[random_start :: caption_cfg.sample]).tolist()
            image_corr = (
                np.array(image_corr, dtype=object)[random_start :: caption_cfg.sample]
            ).tolist()
        if caption_cfg.select == "ratio" and caption_cfg.ratio == 1.0:
            return image_name, image_corr

        if image_name is None or len(image_name) == 0:  # lack 2d data
            selected_idx = None
        elif caption_cfg.select == "fixed":
            # view-level caotion: random select fixed number
            num = int(caption_cfg.num)
            selected_idx = np.random.choice(
                len(image_name), min(num, len(image_name)), replace=False
            )
        elif caption_cfg.select == "ratio":
            ratio = caption_cfg.ratio
            selected_idx = np.random.choice(
                len(image_name), max(1, int(len(image_name) * ratio)), replace=False
            )
        elif caption_cfg.select == "hybrid":
            num = min(int(caption_cfg.num), int(len(image_name) * caption_cfg.ratio))
            selected_idx = np.random.choice(
                len(image_name), min(max(1, num), len(image_name)), replace=False
            )
        elif caption_cfg.select == "ratio_list":
            ratio_list = caption_cfg.ratio_list
            ratio = caption_cfg.ratio
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
        metadata = {"idx": idx, "scene_name": scene_name, "filepath": filepath}

        data_dict = {
            "coord": xyz,
            "color": rgb,
            "segment": semantic_label,
            "instance": inst_label,
            "binary": binary_label,
            "metadata": metadata,
        }
        if self.split == "train":
            image_name_dict, image_corr_dict = self.load_caption(scene_name, idx)
            caption = self.select_caption_and_idx_all(scene_name, image_name_dict, image_corr_dict)
            data_dict["caption"] = caption

        data_dict = self.transform(data_dict)
        return data_dict


if __name__ == "__main__":
    from easydict import EasyDict as edict

    conf = OmegaConf.load("./configs/data/regionplc.yaml")

    dataset_args = conf.train_dataset
    dataset_args.pop("_target_", None)
    dataset = RegionPLCDataset(**edict(OmegaConf.to_object(dataset_args)))
    it = iter(dataset)

    batch = next(it)
