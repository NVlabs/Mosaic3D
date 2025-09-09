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
from src.utils.io import unpack_list_of_np_arrays

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
        anno_sources: Optional[List[str]] = None,
    ):
        self.mask_dir = mask_dir
        super().__init__(
            data_dir=data_dir,
            split=split,
            repeat=repeat,
            ignore_label=ignore_label,
            transforms=transforms,
            num_masks=num_masks,
            anno_sources=anno_sources,
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


class ScanNet200DatasetGathered(ScanNet200Dataset):
    """ScanNet200 dataset with caption merging (Alg. 1) from our paper."""

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

        num_captions_per_object = [len(c) for c in captions]
        # randomly select one caption per object
        idx_select_caption = np.cumsum(
            np.insert(num_captions_per_object, 0, 0)[0:-1]
        ) + np.random.randint(0, num_captions_per_object, len(num_captions_per_object))

        # flatten the list of list
        point_indices = [torch.from_numpy(indices).int() for indices in point_indices]
        captions = [item for sublist in captions for item in sublist]
        captions = [captions[i] for i in idx_select_caption]

        if self.num_masks is not None and self.num_masks < len(point_indices):
            sel = np.random.choice(len(point_indices), self.num_masks, replace=False)
            point_indices = [point_indices[i] for i in sel]
            captions = [captions[i] for i in sel]

        return dict(idx=point_indices, caption=captions)


if __name__ == "__main__":
    dataset = ScanNet200DatasetGathered(
        data_dir="/datasets/mosaic3d/data/scannet",
        split="train",
        repeat=1,
        ignore_label=-100,
        transforms=None,
        anno_sources=["segment3d-gathered"],
    )

    for i in range(5):
        rand_idx = np.random.randint(0, len(dataset))
        sample = dataset[i]

        for k in sample.keys():
            if isinstance(sample[k], (torch.Tensor, np.ndarray)):
                print(f"{k}: {sample[k]}")
