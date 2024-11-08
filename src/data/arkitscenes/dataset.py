from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from src.data.dataset_base import DatasetBase
from src.utils import RankedLogger
from src.utils.io import unpack_list_of_np_arrays

log = RankedLogger(__name__, rank_zero_only=False)


class ARKitScenesDataset(DatasetBase):
    CLASS_LABELS = []  # there is no GT semantic labels

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
        ignore_label: int = -100,
        repeat: int = 1,
        log_postfix: Optional[str] = None,
    ):
        super().__init__(
            dataset_name="arkitscenes",
            data_dir=data_dir,
            split=split,
            transforms=transforms,
            caption_dir=caption_dir,
            caption_subset=caption_subset,
            segment_dir=segment_dir,
            segment_subset=segment_subset,
            object_num_max=object_num_max,
            object_sample_ratio=object_sample_ratio,
            ignore_label=ignore_label,
            repeat=repeat,
            log_postfix=log_postfix,
        )

    def load_point_cloud(self, scene_name: str):
        split = "Training" if self.split == "train" else "Validation"
        scene_dir = self.data_dir / split / scene_name
        coord = np.load(scene_dir / "coord.npy")
        color = np.load(scene_dir / "color.npy")
        coord = coord.astype(np.float32)
        return coord, color

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
        coord, color = self.load_point_cloud(scene_name)
        segment = np.ones_like(coord[:, 0]) * self.ignore_label  # all are ignored
        binary_label = np.ones_like(segment)
        origin_idx = np.arange(coord.shape[0]).astype(np.int64)

        data_dict = dict(
            coord=coord,
            color=color,
            segment=segment,
            binary=binary_label,
            origin_idx=origin_idx,
        )

        # load captions
        if self.split == "train":
            point_indices, captions = self.load_caption_and_sample(scene_name)
            data_dict["caption_data"] = {"idx": point_indices, "caption": captions}

        data_dict = self.transforms(data_dict)
        return data_dict


if __name__ == "__main__":
    import os

    from natsort import natsorted

    datadir = "/datasets/arkitscenes/3dod"
    segment_dir = "/datasets/openvocab-3d-captions/segment.sam2.arkitscenes"
    caption_dir = "/datasets/openvocab-3d-captions/caption.osprey.arkitscenes"

    all_train = os.listdir(os.path.join(datadir, "Training"))
    all_val = os.listdir(os.path.join(datadir, "Validation"))

    segments = os.listdir(segment_dir)
    captions = os.listdir(caption_dir)

    missing_captions = list(set(segments) - set(captions))
    print(len(missing_captions))

    missing_train = list(set(all_train) - set(segments))
    print(len(missing_train))

    missing_val = list(set(all_val) - set(segments))
    print(len(missing_val))
    with open("src/data/metadata/split_files/arkitscenes_train.txt", "w") as f:
        for scene in natsorted(all_train):
            f.write(f"{scene}\n")

    with open("src/data/metadata/split_files/arkitscenes_val.txt", "w") as f:
        for scene in natsorted(all_val):
            f.write(f"{scene}\n")
