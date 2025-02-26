import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.transform import Compose
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class EmbodiedScanDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        transforms: Optional[OmegaConf] = None,
        repeat: int = 1,
        ignore_label: int = -100,
        object_num_max: Optional[int] = None,
        # for compatibility
        object_sample_ratio: Optional[float] = None,
        base_class_idx: Optional[List[int]] = None,
        novel_class_idx: Optional[List[int]] = None,
        ignore_class_idx: Optional[List[int]] = None,
    ):
        super().__init__()

        self.dataset_name = "embodiedscan"
        self.data_dir = Path(data_dir)
        assert self.data_dir.exists(), f"{self.data_dir} not exist."

        self.split = split
        self.repeat = repeat
        self.ignore_label = ignore_label
        self.object_num_max = object_num_max

        # load annotation
        self.mp3d_mapping = self._load_mp3d_mapping()
        self.anno = self.load_anno(split)
        self.scene_names = list(self.anno.keys())

        # data transform
        if transforms is not None:
            transforms_cfg = OmegaConf.to_container(transforms)
            self.transforms = Compose(transforms_cfg)
        else:
            self.transforms = lambda x: x

    def _load_mp3d_mapping(self):
        with open("src/data/embodiedscan/meta_data/mp3d_mapping.json") as f:
            mp3d_mapping = json.load(f)
        return mp3d_mapping

    def load_anno(self, split: str):
        anno = {}
        with open(self.data_dir / f"embodiedscan_{split}_vg_all_filtered.json") as f:
            anno_raw = json.load(f)

        for data in anno_raw:
            scan_id = data["scan_id"]
            if scan_id.startswith("matterport3d"):
                scene, room = scan_id.split("/")[1:]
                scene_mapped = self.mp3d_mapping[scene]
                scan_id = f"{scene_mapped}_{room}"
            else:
                scan_id = scan_id.split("/")[-1]

            target_id = data["target_id"]
            caption = data["text"]

            if scan_id in anno:
                anno[scan_id].append((caption, target_id))
            else:
                anno[scan_id] = [(caption, target_id)]

        return anno

    def load_point_cloud(self, scene_name: str):
        filepath = self.data_dir / "process_pcd" / f"{scene_name}.pth"
        coord, color, segment, instance = torch.load(filepath)  # color in (0, 1)
        color = (color * 255).astype(np.uint8)
        return coord, color, segment, instance

    def load_caption_and_sample(self, scene_name: str):
        captions, inst_ids = list(zip(*self.anno[scene_name]))
        argsort = np.argsort(inst_ids)
        inst_ids = np.array(inst_ids)[argsort]
        captions = np.array(captions)[argsort]
        _, inverse, count = np.unique(inst_ids, return_inverse=True, return_counts=True)
        idx_select = (
            np.cumsum(np.insert(count, 0, 0)[0:-1])
            + np.random.randint(0, count.max(), count.size) % count
        )
        inst_ids = inst_ids[idx_select]
        captions = list(captions[idx_select])

        if self.object_num_max is not None and self.object_num_max < len(captions):
            sel = np.random.choice(len(captions), self.object_num_max, replace=False)
            inst_ids = [inst_ids[i] for i in sel]
            captions = [captions[i] for i in sel]

        return inst_ids, captions

    def __len__(self):
        return len(self.scene_names) * self.repeat

    def __getitem__(self, idx_original: int):
        idx = idx_original % len(self.scene_names)
        scene_name = self.scene_names[idx]

        coord, color, segment, inst_labels = self.load_point_cloud(scene_name)
        binary_label = np.ones_like(segment)
        origin_idx = np.arange(coord.shape[0]).astype(np.int64)

        data_dict = dict(
            coord=coord,
            color=color,
            segment=segment,
            binary=binary_label,
            origin_idx=origin_idx,
            scene_name=scene_name,
        )

        if self.split == "train":
            inst_ids, captions = self.load_caption_and_sample(scene_name)
            point_indices = [
                torch.from_numpy(np.where(inst_labels == inst_id)[0]).int() for inst_id in inst_ids
            ]
            data_dict["caption_data"] = {"idx": point_indices, "caption": captions}

        data_dict = self.transforms(data_dict)
        return data_dict


def filter_erroneous_annotations(split: str):
    data_dir = Path("/datasets/mmscan_data/embodiedscan_split/embodiedscan-v1")

    # load annotation
    with open(
        f"/datasets/mmscan_data/embodiedscan_split/embodiedscan-v1/embodiedscan_{split}_vg_all.json"
    ) as f:
        anno_raw = json.load(f)

    # mp3d mapping
    with open("src/data/embodiedscan/meta_data/mp3d_mapping.json") as f:
        mp3d_mapping = json.load(f)

    anno = []
    for data in tqdm(anno_raw):
        scan_id = data["scan_id"]
        if scan_id.startswith("matterport3d"):
            scene, room = scan_id.split("/")[1:]
            scene_mapped = mp3d_mapping[scene]
            scan_id = f"{scene_mapped}_{room}"
        else:
            scan_id = scan_id.split("/")[-1]

        target_id = data["target_id"]
        instance = torch.load(data_dir / "process_pcd" / f"{scan_id}.pth")[-1]

        # if target_id is not in instance, then it is erroneous
        if target_id not in instance:
            continue

        anno.append(data)

    with open(
        f"/datasets/mmscan_data/embodiedscan_split/embodiedscan-v1/embodiedscan_{split}_vg_all_filtered.json",
        "w",
    ) as f:
        json.dump(anno, f)


if __name__ == "__main__":
    data_dir = "/datasets/mmscan_data/embodiedscan_split/embodiedscan-v1"
    dataset = EmbodiedScanDataset(data_dir, "train", None)

    dataset_size = len(dataset)
    print(f"dataset size: {dataset_size}")

    for i in tqdm(range(dataset_size)):
        sample = dataset[i]

    # filter_erroneous_annotations("train")
