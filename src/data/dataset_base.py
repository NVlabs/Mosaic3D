from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from natsort import natsorted
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.data.transform import Compose
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class DatasetBase(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        split: str,
        transforms: None,
        caption_dir: Optional[str] = None,
        caption_subset: Optional[Union[str, List[str]]] = None,
        segment_dir: Optional[str] = None,
        segment_subset: Optional[Union[str, List[str]]] = None,
        object_sample_ratio: Optional[float] = None,
        base_class_idx: Optional[List[int]] = None,
        novel_class_idx: Optional[List[int]] = None,
        ignore_class_idx: Optional[List[int]] = None,
        ignore_label: int = -100,
        repeat: int = 1,
        log_postfix: Optional[str] = None,
        mask_dir: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        assert self.data_dir.exists(), f"{self.data_dir} not exist."

        self.split = split
        self.object_sample_ratio = object_sample_ratio
        self.base_class_idx = base_class_idx
        self.novel_class_idx = novel_class_idx
        self.ignore_class_idx = ignore_class_idx
        self.ignore_label = ignore_label
        self.repeat = repeat
        self.log_postfix = log_postfix
        self.mask_dir = mask_dir

        # read scene names for split
        self.dataset_name = dataset_name
        split_file_path = (
            Path(__file__).parent
            / "metadata"
            / "split_files"
            / f"{self.dataset_name}_{self.split}.txt"
        )
        with open(split_file_path) as f:
            self.scene_names = natsorted([line.strip() for line in f.readlines()])

        # set caption dir for train dataset
        if self.split == "train":
            caption_subset = (
                [caption_subset] if isinstance(caption_subset, str) else caption_subset
            )
            self.caption_dir = [Path(caption_dir) / subset for subset in caption_subset]
            for subset_dir in self.caption_dir:
                assert subset_dir.exists(), f"{subset_dir} not exist."

        # set segment dir for train dataset
        if self.split == "train" and segment_dir and segment_subset:
            segment_subset = (
                [segment_subset] if isinstance(segment_subset, str) else segment_subset
            )
            self.segment_dir = [Path(segment_dir) / subset for subset in segment_subset]
            assert len(self.segment_dir) == len(
                self.caption_dir
            ), "segment_dir and caption_dir must have the same length"
            for subset_dir in self.segment_dir:
                assert subset_dir.exists(), f"{subset_dir} not exist."

        # class label mappers
        self.valid_class_idx = np.arange(len(self.CLASS_LABELS)).tolist()
        if base_class_idx is not None and len(base_class_idx):
            self.base_class_mapper = self.build_class_mapper(base_class_idx, ignore_label)
            self.binary_class_mapper = self.build_binary_class_mapper(
                base_class_idx, novel_class_idx, ignore_class_idx, ignore_label
            )
        if ignore_class_idx is not None:
            for c in ignore_class_idx:
                self.valid_class_idx.remove(c)
        self.valid_class_mapper = self.build_class_mapper(self.valid_class_idx, ignore_label)
        self.subset_mapper = self.build_subset_mapper()
        # foreground & background class indices
        self.fg_class_idx = [
            i
            for i, c in enumerate(self.CLASS_LABELS)
            if c not in ("wall", "floor", "ceiling") and "other" not in c
        ]
        self.bg_class_idx = list(set(range(len(self.CLASS_LABELS))) - set(self.fg_class_idx))
        self.instance_ignore_class_idx = [
            i for i, c in enumerate(self.CLASS_LABELS) if c in ("wall", "floor") or "other" in c
        ]

        # data transform
        transforms_cfg = OmegaConf.to_container(transforms)
        if mask_dir is not None:
            for transform_cfg in transforms_cfg:
                if transform_cfg["type"] == "Collect":
                    transform_cfg["keys"].append("masks_binary")
        self.transforms = Compose(transforms_cfg)

        log.info(f"Loaded {self.__len__()} samples in {self.split} set.")

    @property
    def use_base_class_mapper(self):
        return self.split == "train" and hasattr(self, "base_class_mapper")

    @staticmethod
    def build_class_mapper(class_idx, ignore_label, squeeze_label=False):
        num_classes = max(256, len(class_idx))
        remapper = np.ones(num_classes, dtype=np.int64) * ignore_label
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

    def build_subset_mapper(self):
        return None

    def __len__(self):
        length = len(self.scene_names) * (self.repeat if self.split == "train" else 1)
        return length

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abstractmethod
    def load_point_cloud(self, scene_name: str):
        raise NotImplementedError

    @abstractmethod
    def load_caption(self, scene_name: str):
        raise NotImplementedError

    def augment_captions(self, xyz, point_indices, captions):
        new_point_indices = []
        augmented_captions = []
        for idx, caption in zip(point_indices, captions):
            if len(idx) > 1:
                coords = xyz[idx]
                centroid = coords.mean(axis=0)
                centroid_str = f"({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
                caption = f"{caption}, located at {centroid_str})"
                augmented_captions.append(caption)
                new_point_indices.append(idx)
        return new_point_indices, augmented_captions
