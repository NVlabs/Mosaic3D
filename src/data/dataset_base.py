from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from natsort import natsorted
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.data.utils.transform import Compose
from src.utils import RankedLogger
from src.utils.io import unpack_list_of_np_arrays

log = RankedLogger(__name__, rank_zero_only=False)


class BaseDataset(Dataset):
    """Base dataset class with core functionality for 3D datasets."""

    BACKGROUND_CLASSES = ("wall", "floor", "ceiling")
    CLASS_LABELS = None

    def __init__(
        self,
        data_dir: str,
        split: str,
        repeat: int = 1,
        ignore_label: int = -100,
        transforms: Optional[List[Dict]] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset_name = self.data_dir.stem
        assert self.data_dir.exists(), f"{self.data_dir} not exist."
        self.split = split
        self.repeat = repeat
        self.ignore_label = ignore_label

        # setup class mappers
        self.valid_class_idx = [
            i for i, c in enumerate(self.CLASS_LABELS) if not c.startswith("other")
        ]
        self.valid_class_mapper = self.build_class_mapper(self.valid_class_idx, self.ignore_label)
        self.fg_class_idx = [
            i
            for i, c in enumerate(self.CLASS_LABELS)
            if c not in self.BACKGROUND_CLASSES and "other" not in c
        ]
        self.bg_class_idx = list(set(range(len(self.CLASS_LABELS))) - set(self.fg_class_idx))
        self.instance_ignore_class_idx = [
            i for i, c in enumerate(self.CLASS_LABELS) if c in ("wall", "floor") or "other" in c
        ]
        self.subset_mapper = self.build_subset_mapper()

        # read split file
        split_file_path = (
            Path(__file__).parent
            / "metadata"
            / "split_files"
            / f"{self.dataset_name}_{self.split}.txt"
        )
        with open(split_file_path) as f:
            self.scene_names = natsorted(
                [line.strip() for line in f.readlines() if not line.startswith("#")]
            )

        # setup transforms
        self.transforms = lambda x: x
        if transforms is not None:
            transforms_cfg = OmegaConf.to_container(transforms)
            if hasattr(self, "mask_dir") and self.mask_dir is not None:
                for transform_cfg in transforms_cfg:
                    if transform_cfg["type"] == "Collect":
                        transform_cfg["keys"].append("masks_binary")
            self.transforms = Compose(transforms_cfg)

        log.info(
            f"Loaded dataset: {self.dataset_name} | "
            f"Split: {self.split} | "
            f"Number of samples: {len(self.scene_names)}"
        )

    def __len__(self):
        n = len(self.scene_names)
        if self.split == "train":
            n *= self.repeat
        return n

    @property
    def is_train(self):
        return self.split == "train"

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

    def build_subset_mapper(self):
        return None

    @abstractmethod
    def load_point_cloud(self, scene_name: str):
        """Load point cloud data for a given scene."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        """Get item by index."""
        raise NotImplementedError


class AnnotatedDataset(BaseDataset):
    """Dataset with annotation/caption capabilities."""

    CLASS_LABELS = None
    SEGMENT_FILE = None
    INSTANCE_FILE = None
    LOG_POSTFIX = None

    def __init__(
        self,
        data_dir: str,
        split: str,
        repeat: int = 1,
        ignore_label: int = -100,
        transforms: Optional[List[Dict]] = None,
        num_masks: Optional[int] = None,
        anno_sources: Optional[List[str]] = None,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            repeat=repeat,
            ignore_label=ignore_label,
            transforms=transforms,
        )
        self.num_masks = num_masks
        self.anno_sources = anno_sources or ["gsam2", "seem"]
        self.log_postfix = self.LOG_POSTFIX

    def load_point_cloud(self, scene_name: str):
        """Load point cloud data for a given scene."""
        scene_dir = self.data_dir / scene_name
        coord = np.load(scene_dir / "coord.npy").astype(np.float32)
        color = np.load(scene_dir / "color.npy")
        origin_idx = np.arange(coord.shape[0]).astype(np.int64)

        return_dict = dict(
            coord=coord,
            color=color,
            origin_idx=origin_idx,
        )

        if not self.is_train and self.SEGMENT_FILE is not None:
            segment_file = scene_dir / self.SEGMENT_FILE
            assert segment_file.exists(), f"{segment_file} not exist."
            segment_raw = np.load(segment_file)
            segment = self.valid_class_mapper[segment_raw.astype(np.int64)]
            return_dict["segment"] = segment

        if not self.is_train and self.INSTANCE_FILE is not None:
            instance_file = scene_dir / self.INSTANCE_FILE
            assert instance_file.exists(), f"{instance_file} not exist."
            assert "segment" in return_dict, "segment is required for instance"
            instance = np.load(instance_file)
            instance[return_dict["segment"] == self.ignore_label] = self.ignore_label
            return_dict["instance"] = instance

        return return_dict

    def load_caption(self, scene_name: str):
        """Load caption data for a given scene."""
        scene_dir = self.data_dir / scene_name
        anno_source = np.random.choice(self.anno_sources)
        caption_file = scene_dir / f"captions.{anno_source}.npz"
        point_indices_file = scene_dir / f"point_indices.{anno_source}.npz"
        assert caption_file.exists(), f"{caption_file} not exist."
        assert point_indices_file.exists(), f"{point_indices_file} not exist."
        captions = unpack_list_of_np_arrays(caption_file)
        point_indices = unpack_list_of_np_arrays(point_indices_file)
        captions = [item for sublist in captions for item in sublist]
        point_indices = [
            torch.from_numpy(item).int() for sublist in point_indices for item in sublist
        ]

        if self.num_masks is not None and self.num_masks < len(point_indices):
            sel = np.random.choice(len(point_indices), self.num_masks, replace=False)
            point_indices = [point_indices[i] for i in sel]
            captions = [captions[i] for i in sel]

        return dict(idx=point_indices, caption=captions)

    def __getitem__(self, idx_original):
        idx = idx_original % len(self.scene_names)
        scene_name = self.scene_names[idx]

        # load point cloud data
        data_dict = dict(scene_name=scene_name)
        point_cloud_data = self.load_point_cloud(scene_name)
        data_dict.update(point_cloud_data)

        if self.is_train:
            data_dict["caption_data"] = self.load_caption(scene_name)

        data_dict = self.transforms(data_dict)
        return data_dict
