from typing import List, Optional, Union

import numpy as np
import torch

from src.data.dataset_base import DatasetBase
from src.data.metadata.scannetpp import CLASS_LABELS
from src.utils import RankedLogger
from src.utils.io import unpack_list_of_np_arrays

log = RankedLogger(__name__, rank_zero_only=False)


class ScanNetPPDataset(DatasetBase):
    CLASS_LABELS = CLASS_LABELS

    def __init__(
        self,
        data_dir: str,
        split: str,
        transforms: None,
        segment_dir: Optional[str] = None,
        segment_subset: Optional[Union[str, List[str]]] = None,
        caption_dir: Optional[str] = None,
        caption_subset: Optional[Union[str, List[str]]] = None,
        object_sample_ratio: Optional[float] = None,
        ignore_label: int = -100,
        repeat: int = 1,
        log_postfix: Optional[str] = None,
    ):
        super().__init__(
            dataset_name="scannetpp",
            data_dir=data_dir,
            split=split,
            transforms=transforms,
            caption_dir=caption_dir,
            caption_subset=caption_subset,
            segment_dir=segment_dir,
            segment_subset=segment_subset,
            object_sample_ratio=object_sample_ratio,
            ignore_label=ignore_label,
            repeat=repeat,
            log_postfix=log_postfix,
        )

    def load_point_cloud(self, scene_name: str):
        scene_dir = self.data_dir / scene_name
        coord = np.load(scene_dir / "coord.npy")
        color = np.load(scene_dir / "color.npy")
        coord = coord.astype(np.float32)
        return coord, color

    def load_caption(self, scene_name):
        all_point_indices = []
        all_captions = []

        for caption_dir, segment_dir in zip(self.caption_dir, self.segment_dir):
            indices_path = segment_dir / scene_name / "point_indices.npz"
            point_indices = unpack_list_of_np_arrays(indices_path)

            caption_path = caption_dir / scene_name / "captions.npz"
            captions = unpack_list_of_np_arrays(caption_path)

            # flatten the list of list
            point_indices = [item for sublist in point_indices for item in sublist]
            captions = [item for sublist in captions for item in sublist]

            if self.object_sample_ratio < 1.0:
                sel = np.random.choice(
                    np.arange(len(captions)),
                    max(1, int(len(captions) * self.object_sample_ratio)),
                    replace=False,
                )
                captions = [captions[i] for i in sel]
                point_indices = [point_indices[i] for i in sel]

            point_indices = [torch.from_numpy(indices).int() for indices in point_indices]

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
            try:
                point_indices, captions = self.load_caption(scene_name)
            except Exception as e:
                log.error(f"Error loading caption for scene {scene_name}: {e}", stacklevel=2)
                point_indices, captions = [], []
            data_dict["caption_data"] = {"idx": point_indices, "caption": captions}

        data_dict = self.transforms(data_dict)
        return data_dict
