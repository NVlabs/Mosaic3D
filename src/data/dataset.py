from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from natsort import natsorted
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.data.transform import Compose
from src.utils import RankedLogger
from src.utils.io import unpack_list_of_np_arrays

log = RankedLogger(__name__, rank_zero_only=False)


class AnnotatedDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        split: str,
        max_num_masks: int,
        ignore_label: int = -100,
        repeat: int = 1,
        transforms: Optional[List[Dict]] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        assert self.data_dir.exists(), f"{self.data_dir} not exist."

        self.dataset_name = dataset_name
        self.split = split
        self.max_num_masks = max_num_masks
        self.ignore_label = ignore_label
        self.repeat = repeat
        self.anno_sources = ["gsam2", "seem"]

        # read split
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

        self.transforms = lambda x: x
        if transforms is not None:
            transforms_cfg = OmegaConf.to_container(transforms)
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

    def __getitem__(self, idx_original):
        idx = idx_original % len(self.scene_names)
        scene_name = self.scene_names[idx]
        scene_dir = self.data_dir / scene_name

        # load point cloud
        coord = np.load(scene_dir / "geometry" / "coord.npy")
        color = np.load(scene_dir / "geometry" / "color.npy")

        data_dict = dict(
            coord=coord.astype(np.float32),
            color=color,
            origin_idx=np.arange(coord.shape[0]).astype(np.int64),
        )

        # load caption
        anno_source = np.random.choice(self.anno_sources)
        captions = unpack_list_of_np_arrays(scene_dir / anno_source / "captions.npz")
        point_indices = unpack_list_of_np_arrays(
            scene_dir / anno_source / "point_indices.npz"
        )
        captions = [item for sublist in captions for item in sublist]
        point_indices = [
            torch.from_numpy(item).int()
            for sublist in point_indices
            for item in sublist
        ]

        if self.max_num_masks is not None and self.max_num_masks < len(point_indices):
            sel = np.random.choice(
                len(point_indices), self.max_num_masks, replace=False
            )
            point_indices = [point_indices[i] for i in sel]
            captions = [captions[i] for i in sel]

        data_dict["caption_data"] = {"idx": point_indices, "caption": captions}
        data_dict = self.transforms(data_dict)
        return data_dict
