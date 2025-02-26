from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from src.data.dataset_base import DatasetBase
from src.data.metadata.matterport3d import (
    CLASS_LABELS_21,
    CLASS_LABELS_40,
    CLASS_LABELS_80,
    CLASS_LABELS_160,
)
from src.utils import RankedLogger
from src.utils.io import unpack_list_of_np_arrays

log = RankedLogger(__name__, rank_zero_only=False)


class Matterport3DDataset(DatasetBase):
    CLASS_LABELS = CLASS_LABELS_21

    def __init__(
        self,
        data_dir: str,
        split: str,
        transforms: None,
        segment_dir: Optional[str] = None,
        segment_subset: Optional[Union[str, List[str]]] = None,
        caption_dir: Optional[str] = None,
        caption_subset: Optional[Union[str, List[str]]] = None,
        object_num_max: Optional[int] = None,
        object_sample_ratio: Optional[float] = None,
        base_class_idx: Optional[List[int]] = None,
        novel_class_idx: Optional[List[int]] = None,
        ignore_class_idx: Optional[List[int]] = None,
        ignore_label: int = -100,
        repeat: int = 1,
        mask_dir: Optional[str] = None,
        log_postfix: Optional[str] = None,
    ):
        super().__init__(
            dataset_name="matterport3d",
            data_dir=data_dir,
            split=split,
            transforms=transforms,
            caption_dir=caption_dir,
            caption_subset=caption_subset,
            segment_dir=segment_dir,
            segment_subset=segment_subset,
            object_num_max=object_num_max,
            object_sample_ratio=object_sample_ratio,
            base_class_idx=base_class_idx,
            novel_class_idx=novel_class_idx,
            ignore_class_idx=ignore_class_idx,
            ignore_label=ignore_label,
            repeat=repeat,
            log_postfix=log_postfix,
            mask_dir=mask_dir,
        )

    def load_point_cloud(self, scene_name: str):
        filepath = self.data_dir / self.split / f"{scene_name}.pth"
        coord, color, segment = torch.load(filepath)
        color = np.clip((color + 1.0) * 127.5, 0, 255).astype(int)
        return coord, color, segment

    def load_caption(self, scene_name):
        all_point_indices = []
        all_captions = []

        for caption_dir, segment_dir in zip(self.caption_dir, self.segment_dir):
            indices_path = segment_dir / scene_name / "point_indices.npz"
            caption_path = caption_dir / scene_name / "captions.npz"
            if not indices_path.exists() or not caption_path.exists():
                continue

            point_indices = unpack_list_of_np_arrays(indices_path)
            captions = unpack_list_of_np_arrays(caption_path)

            # flatten the list of list
            point_indices = [item for sublist in point_indices for item in sublist]
            point_indices = [torch.from_numpy(indices).int() for indices in point_indices]
            captions = [item for sublist in captions for item in sublist]

            all_point_indices.extend(point_indices)
            all_captions.extend(captions)

        return all_point_indices, all_captions

    def __getitem__(self, idx_original):
        idx = idx_original % len(self.scene_names)
        scene_name = self.scene_names[idx]

        # load pcd data
        coord, color, segment = self.load_point_cloud(scene_name)

        # class mapping
        # base / novel label
        if hasattr(self, "base_class_mapper"):
            binary_label = self.binary_class_mapper[segment.astype(np.int64)].astype(np.float32)
        else:
            binary_label = np.ones_like(segment)

        if self.use_base_class_mapper:
            segment = self.base_class_mapper[segment.astype(np.int64)]
        else:
            segment = self.valid_class_mapper[segment.astype(np.int64)]

        data_dict = dict(
            coord=coord,
            color=color,
            segment=segment,
            binary=binary_label,
            origin_idx=np.arange(coord.shape[0]).astype(np.int64),
        )

        # load captions
        if self.split == "train":
            point_indices, captions = self.load_caption_and_sample(scene_name)
            data_dict["caption_data"] = {"idx": point_indices, "caption": captions}

        data_dict = self.transforms(data_dict)
        return data_dict


class Matterport3D40Dataset(Matterport3DDataset):
    CLASS_LABELS = CLASS_LABELS_40


class Matterport3D80Dataset(Matterport3DDataset):
    CLASS_LABELS = CLASS_LABELS_80


class Matterport3D160Dataset(Matterport3DDataset):
    CLASS_LABELS = CLASS_LABELS_160
