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


class ScanNet200DatasetMerged(ScanNet200Dataset):
    """ScanNet200 dataset with caption merging (Alg. 1) from our paper."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        ignore_label: int = -100,
        repeat: int = 1,
        transforms: Optional[List[Dict]] = None,
        num_masks: Optional[int] = None,
        mask_dir: Optional[str] = None,
        num_merged_captions: Optional[int] = None,
        min_score: float = 0.0,  # 0.0 means no filtering
    ):
        self.num_merged_captions = num_merged_captions
        self.min_score = min_score
        super().__init__(
            data_dir=data_dir,
            split=split,
            repeat=repeat,
            ignore_label=ignore_label,
            transforms=transforms,
            num_masks=num_masks,
            mask_dir=mask_dir,
        )

    def load_caption(self, scene_name: str):
        """Load caption data for a given scene."""
        scene_dir = self.data_dir / scene_name
        point_indices_file = scene_dir / "point_indices.segment3d.npz"
        assert point_indices_file.exists(), f"{point_indices_file} not exist."

        with np.load(point_indices_file) as data:
            packed = data["packed"]
            lengths = data["lengths"]
            scores = data["scores"]
            point_indices = [np.array(arr) for arr in np.split(packed, np.cumsum(lengths)[:-1])]

            assert len(point_indices) == len(
                scores
            ), f"len(point_indices) ({len(point_indices)}) != len(scores) ({len(scores)})"

        captions_list = [[] for _ in range(len(point_indices))]
        for anno_source in self.anno_sources:
            caption_file = scene_dir / f"captions.{anno_source}.npz"
            mapping_file = scene_dir / f"segment3d-mapping.{anno_source}.npz"
            assert caption_file.exists(), f"{caption_file} not exist."
            assert mapping_file.exists(), f"{mapping_file} not exist."
            captions = unpack_list_of_np_arrays(caption_file)
            mapping = unpack_list_of_np_arrays(mapping_file)
            mapping = [item for sublist in mapping for item in sublist]
            captions = [item for sublist in captions for item in sublist]
            assert len(mapping) == len(
                captions
            ), f"len(mapping) ({len(mapping)}) != len(captions) ({len(captions)})"

            for mask_idx, caption in zip(mapping, captions):
                if mask_idx == -1:
                    continue
                captions_list[mask_idx].append(caption)

        # Filter out masks without captions
        filtered_indices = []
        filtered_captions = []
        for idx, caps, score in zip(point_indices, captions_list, scores):
            if len(caps) == 0:
                continue

            if self.min_score > 0.0 and score < self.min_score:
                continue

            if self.num_merged_captions is not None and len(caps) > self.num_merged_captions:
                caps = np.random.choice(caps, self.num_merged_captions, replace=False)

            filtered_indices.append(idx)
            filtered_captions.append(", ".join(caps))

        return dict(idx=filtered_indices, caption=filtered_captions)


if __name__ == "__main__":
    dataset = ScanNet200DatasetMerged(
        data_dir="/datasets/mosaic3d/data/scannet",
        split="train",
        repeat=1,
        ignore_label=-100,
        transforms=None,
        num_merged_captions=4,
        min_score=0.5,
    )

    for i in range(5):
        rand_idx = np.random.randint(0, len(dataset))
        sample = dataset[i]

        for k in sample.keys():
            if isinstance(sample[k], (torch.Tensor, np.ndarray)):
                print(f"{k}: {sample[k]}")
