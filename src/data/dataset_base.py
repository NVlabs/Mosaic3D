from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Optional

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
        assert self.data_dir.exists(), f"{self.data_dir} not exist."

        self.split = split
        self.object_sample_ratio = object_sample_ratio
        self.base_class_idx = base_class_idx
        self.novel_class_idx = novel_class_idx
        self.ignore_class_idx = ignore_class_idx
        self.ignore_label = ignore_label
        self.repeat = repeat

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
            self.caption_dir = Path(caption_dir) / caption_subset
            assert self.caption_dir.exists(), f"{self.caption_dir} not exist."

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
            for i in self.valid_class_idx
            if self.CLASS_LABELS[i] not in ("wall", "floor", "ceiling", "other", "otherfurniture")
        ]
        self.bg_class_idx = list(set(range(len(self.CLASS_LABELS))) - set(self.fg_class_idx))

        # data transform
        transforms_cfg = OmegaConf.to_container(transforms)
        self.transforms = Compose(transforms_cfg)

        log.info(f"Loaded {self.__len__()} samples in {self.split} set.")

    @property
    def use_base_class_mapper(self):
        return self.split == "train" and hasattr(self, "base_class_mapper")

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
