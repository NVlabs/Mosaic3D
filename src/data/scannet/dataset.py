import os
from typing import Dict, List, Optional

import numpy as np
import torch

from src.data.dataset_base import AnnotatedDataset
from src.data.metadata.scannet import (
    CLASS_LABELS_20,
    CLASS_LABELS_200,
    COMMON_CLASSES_200,
    HEAD_CLASSES_200,
    TAIL_CLASSES_200,
)
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class ScanNetDataset(AnnotatedDataset):
    CLASS_LABELS = CLASS_LABELS_20
    SEGMENT_FILE = "segment20.npy"
    INSTANCE_FILE = "instance.npy"
    LOG_POSTFIX = "scannet20"

    def __init__(
        self,
        data_dir: str,
        split: str,
        ignore_label: int = -100,
        repeat: int = 1,
        transforms: Optional[List[Dict]] = None,
        num_masks: Optional[int] = None,
        mask_dir: Optional[str] = None,
    ):
        self.mask_dir = mask_dir
        super().__init__(
            data_dir=data_dir,
            split=split,
            repeat=repeat,
            ignore_label=ignore_label,
            transforms=transforms,
            num_masks=num_masks,
        )

    def __getitem__(self, idx_original):
        idx = idx_original % len(self.scene_names)
        scene_name = self.scene_names[idx]

        # load point cloud data
        data_dict = dict(scene_name=scene_name)
        point_cloud_data = self.load_point_cloud(scene_name)
        data_dict.update(point_cloud_data)

        if self.is_train:
            data_dict["caption_data"] = self.load_caption(scene_name)

        if not self.is_train and self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, f"{self.scene_names[idx_original]}.npz")
            mask_data = np.load(mask_path)
            masks_binary = mask_data["masks_binary"]
            data_dict["masks_binary"] = masks_binary

        data_dict = self.transforms(data_dict)
        return data_dict


class ScanNet200Dataset(ScanNetDataset):
    CLASS_LABELS = CLASS_LABELS_200
    SEGMENT_FILE = "segment200.npy"
    INSTANCE_FILE = "instance.npy"
    LOG_POSTFIX = "scannet200"

    def build_subset_mapper(self):
        mapper = {}
        mapper["subset_names"] = ["head", "common", "tail"]
        for name in self.CLASS_LABELS:
            if name in HEAD_CLASSES_200:
                mapper[name] = "head"
            elif name in COMMON_CLASSES_200:
                mapper[name] = "common"
            elif name in TAIL_CLASSES_200:
                mapper[name] = "tail"
            else:
                raise ValueError(f"Unknown class name: {name}")
        return mapper


if __name__ == "__main__":
    dataset = ScanNet200Dataset(
        data_dir="/datasets/mosaic3d/data/scannet",
        split="val",
        repeat=1,
        ignore_label=-100,
        transforms=None,
    )

    for i in range(5):
        rand_idx = np.random.randint(0, len(dataset))
        sample = dataset[i]

        for k in sample.keys():
            if isinstance(sample[k], (torch.Tensor, np.ndarray)):
                print(f"{k}: {sample[k]}")
