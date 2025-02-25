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


class MMScanDataset(Dataset):
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

        self.dataset_name = "mmscan"
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

        # visual grounding
        with open(self.data_dir / "MMScan_samples" / "MMScan_VG_filtered.json") as f:
            anno_vg = json.load(f)[split]

        def convert_scan_id(scan_id: str):
            if scan_id.startswith("matterport3d"):
                scene, room = scan_id.split("/")[1:]
                scene_mapped = self.mp3d_mapping[scene]
                scan_id = f"{scene_mapped}_{room}"
            else:
                scan_id = scan_id.split("/")[-1]
            return scan_id

        for data in anno_vg:
            scan_id = convert_scan_id(data["scan_id"])
            target_id = data["target_id"]
            caption = data["text"]

            # Create caption-target pairs for all targets
            caption_pairs = [(caption, target) for target in target_id]

            if scan_id in anno:
                anno[scan_id].extend(caption_pairs)
            else:
                anno[scan_id] = caption_pairs

        # object captioning
        with open(self.data_dir / "MMScan_samples" / "MMScan_Caption_object_filtered.json") as f:
            anno_oc = json.load(f)[split]

        for data in anno_oc:
            scan_id = convert_scan_id(data["scan_id"])
            target_id = data["object_id"]
            caption = data["caption"]

            if scan_id in anno:
                anno[scan_id].append((caption, target_id))
            else:
                anno[scan_id] = [(caption, target_id)]

        return anno

    def load_point_cloud(self, scene_name: str):
        filepath = (
            self.data_dir.parent
            / "embodiedscan_split"
            / "embodiedscan-v1"
            / "process_pcd"
            / f"{scene_name}.pth"
        )
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


if __name__ == "__main__":
    data_dir = "/datasets/mmscan_data/MMScan-beta-release"
    dataset = MMScanDataset(data_dir, "train", None)

    dataset_size = len(dataset)
    print(f"dataset size: {dataset_size}")

    for i in tqdm(range(dataset_size)):
        sample = dataset[i]
