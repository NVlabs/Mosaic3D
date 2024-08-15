import glob
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from natsort import natsorted
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.data.metadata.scannet import (
    CLASS_LABELS_20,
    CLASS_LABELS_200,
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)
from src.data.transform import Compose
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class ScanNetDataset(Dataset):
    CLASS_LABELS = CLASS_LABELS_20
    CLASS_IDS = VALID_CLASS_IDS_20

    def __init__(
        self,
        data_dir: str,
        split: str,
        transforms: None,
        caption_dir: Optional[str] = None,
        caption_subset: Optional[str] = None,
        object_sample_ratio: Optional[float] = None,
        base_class_idx: Optional[List[int]] = None,
        novel_class_idx: Optional[List[int]] = None,
        ignore_class_idx: Optional[List[int]] = None,
        ignore_label: int = -100,
        repeat: int = 1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.object_sample_ratio = object_sample_ratio
        self.class_names = self.CLASS_LABELS
        self.repeat = repeat

        # set caption dir for train dataset
        if self.split == "train":
            self.caption_dir = Path(caption_dir) / caption_subset
            assert self.caption_dir.exists(), f"{self.caption_dir} not exist."

        # read scene names for split
        with open(f"src/data/metadata/split_files/scannetv2_{self.split}.txt") as f:
            self.scene_names = natsorted([line.strip() for line in f.readlines()])

        # class label mappers
        self.base_class_idx = base_class_idx
        self.novel_class_idx = novel_class_idx
        self.ignore_class_idx = ignore_class_idx
        self.valid_class_idx = np.arange(len(self.CLASS_LABELS)).tolist()
        if base_class_idx is not None:
            self.base_class_mapper = self.build_class_mapper(base_class_idx, ignore_label)
            self.binary_class_mapper = self.build_binary_class_mapper(
                base_class_idx, novel_class_idx, ignore_class_idx, ignore_label
            )
        if ignore_class_idx is not None:
            for c in ignore_class_idx:
                self.valid_class_idx.remove(c)
        self.valid_class_mapper = self.build_class_mapper(self.valid_class_idx, ignore_label)
        self.ignore_label = ignore_label

        # data transform
        self.transforms = Compose(OmegaConf.to_container(transforms))

        log.info(f"Loaded {self.__len__()} samples in {self.split} set.")

        # backward compatibility for regionplc
        self.caption_cfg = dict(GATHER_CAPTION=False)

    @property
    def training(self):
        return self.split == "train"

    @staticmethod
    def build_class_mapper(class_idx, ignore_label, squeeze_label=False):
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
        length = len(self.scene_names) * (self.repeat if self.split == "train" else 1)
        return length

    def load_caption(self, scene_name):
        filepath = os.path.join(self.caption_dir, f"{scene_name}.npz")
        data = np.load(filepath)

        object_ids = data["object_ids"]
        captions = data["captions"]
        point_indices_flatten = data["point_indices"]
        num_points = data["num_points"]

        cumsum = np.cumsum(num_points)[:-1]
        point_indices = np.split(point_indices_flatten, cumsum)

        if self.object_sample_ratio < 1.0:
            sel = np.random.choice(
                np.arange(len(object_ids)),
                max(1, int(len(object_ids) * self.object_sample_ratio)),
                replace=False,
            )
            object_ids = object_ids[sel]
            captions = captions[sel]
            num_points = num_points[sel]
            point_indices = [point_indices[i] for i in sel]

        point_indices = [torch.from_numpy(indices).int() for indices in point_indices]
        captions = list(captions)
        return point_indices, captions

    def __getitem__(self, idx_original):
        idx = idx_original % len(self.scene_names)
        scene_name = self.scene_names[idx]
        scene_dir = self.data_dir / self.split / scene_name

        # load pcd data
        coord = np.load(scene_dir / "coord.npy")
        color = np.load(scene_dir / "color.npy")
        segment = np.load(scene_dir / "segment20.npy")
        instance = np.load(scene_dir / "instance.npy")

        # class mapping
        # base / novel label
        if hasattr(self, "base_class_mapper"):
            binary_label = self.binary_class_mapper[segment.astype(np.int64)].astype(np.float32)
        else:
            binary_label = np.ones_like(segment)

        if self.training:
            segment = self.base_class_mapper[segment.astype(np.int64)]
        elif not self.training and hasattr(self, "ignore_class_idx"):
            segment = self.valid_class_mapper[segment.astype(np.int64)]
        instance[segment == self.ignore_label] = self.ignore_label

        data_dict = dict(
            coord=coord,
            color=color,
            segment=segment,
            instance=instance,
            binary=binary_label,
            origin_idx=np.arange(coord.shape[0]).astype(np.int64),
        )

        # load captions
        if self.split == "train":
            caption_idx, caption_text = self.load_caption(scene_name)
            data_dict["caption_data"] = {"idx": caption_idx, "caption": caption_text}

        data_dict = self.transforms(data_dict)
        return data_dict


class ScanNet200Dataset(ScanNetDataset):
    CLASS_LABELS = CLASS_LABELS_200
    CLASS_IDS = VALID_CLASS_IDS_200

    def __init__(
        self,
        data_dir: str,
        caption_dir: str,
        split: str,
        transforms: None,
        base_class_idx: Optional[List[int]],
        novel_class_idx: Optional[List[int]],
        ignore_class_idx: Optional[List[int]],
        ignore_label: int = -100,
    ):
        super().__init__(
            data_dir,
            caption_dir,
            split,
            transforms,
            base_class_idx,
            novel_class_idx,
            ignore_class_idx,
            ignore_label,
        )
