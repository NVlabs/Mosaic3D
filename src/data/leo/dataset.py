import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from natsort import natsorted
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.data.transform import Compose
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class LeoDataset(Dataset):
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

        self.dataset_name = "leo"
        self.data_dir = Path(data_dir)
        assert self.data_dir.exists(), f"{self.data_dir} not exist."

        self.split = split
        self.repeat = repeat
        self.ignore_label = ignore_label
        self.object_num_max = object_num_max

        # load annotation
        self.anno_scannet = self._load_scannet_anno(split)
        self.anno_3rscan = self._load_3rscan_anno(split)

        scene_names = list(self.anno_scannet.keys()) + list(self.anno_3rscan.keys())
        self.scene_names = scene_names

        # data transform
        if transforms is not None:
            transforms_cfg = OmegaConf.to_container(transforms)
            self.transforms = Compose(transforms_cfg)
        else:
            self.transforms = lambda x: x

    def _load_scannet_anno(self, split: str):
        anno = {}
        # load scannet nr3d
        with open(
            self.data_dir
            / "annotations"
            / "alignment"
            / "obj_scene_caption"
            / f"scannet_referit3d_nr3d_{split}.json",
        ) as f:
            anno_nr3d = json.load(f)

        for data in anno_nr3d:
            scan_id = data["scan_id"]
            target_id = data["target_id"]
            caption = data["utterance"]
            if scan_id in anno:
                anno[scan_id].append((caption, target_id))
            else:
                anno[scan_id] = [(caption, target_id)]

        # load scannet sr3d
        with open(
            self.data_dir
            / "annotations"
            / "alignment"
            / "obj_scene_caption"
            / f"scannet_referit3d_sr3d+_{split}.json",
        ) as f:
            anno_sr3d = json.load(f)

        for data in anno_sr3d:
            scan_id = data["scan_id"]
            target_id = data["target_id"]
            caption = data["utterance"]
            if scan_id in anno:
                anno[scan_id].append((caption, target_id))
            else:
                anno[scan_id] = [(caption, target_id)]

        # load scanrefer
        with open(
            self.data_dir / "annotations" / "instruction" / "scan2cap" / f"scanrefer_train.json",
        ) as f:
            anno_scanrefer = json.load(f)

        for data in anno_scanrefer:
            scan_id = data["scan_id"]
            target_id = data["target_id"]
            caption = data["utterance"]
            if scan_id in anno:
                anno[scan_id].append((caption, target_id))
            else:
                anno[scan_id] = [(caption, target_id)]

        # load scanqa
        with open(
            self.data_dir / "annotations" / "instruction" / "scanqa" / f"ScanQA_v1.0_{split}.json",
        ) as f:
            anno_scanqa = json.load(f)

        for data in anno_scanqa:
            scan_id = data["scene_id"]
            target_id = data["object_ids"][0]
            caption = data["answers"][0]
            if scan_id in anno:
                anno[scan_id].append((caption, target_id))
            else:
                anno[scan_id] = [(caption, target_id)]

        return anno

    def _load_3rscan_anno(self, split: str):
        anno = {}

        # load 3rscan scanscribe
        with open(
            self.data_dir
            / "annotations"
            / "alignment"
            / "obj_scene_caption"
            / "3rscan_scanscribe.json",
        ) as f:
            anno_scanscribe = json.load(f)

        for data in anno_scanscribe:
            scan_id = data["scan_id"]
            target_id = data["object_id"]
            caption = data["sentence"]
            if scan_id in anno:
                anno[scan_id].append((caption, target_id))
            else:
                anno[scan_id] = [(caption, target_id)]

        # load 3rscan prompted
        with open(
            self.data_dir
            / "annotations"
            / "alignment"
            / "obj_scene_caption"
            / "3rscan_prompted.json",
        ) as f:
            anno_prompted = json.load(f)

        for scan_id, prompts in anno_prompted.items():
            for k, responses in prompts.items():
                target_id = int(k.split("-")[-1])
                if scan_id not in anno:
                    anno[scan_id] = []
                anno[scan_id].extend([(d["response"], target_id) for d in responses])

        return anno

    def _load_scannet_pcd(self, scene_name: str):
        coord, color, _, inst_labels = torch.load(
            self.data_dir / "pcd_with_global_alignment" / f"{scene_name}.pth"
        )
        return coord, color, inst_labels

    def _load_3rscan_pcd(self, scene_name: str):
        coord, color, inst_labels = torch.load(
            self.data_dir / "3RScan-base" / "3RScan-ours-align" / scene_name / "pcds.pth"
        )
        return coord, color, inst_labels

    def load_caption_and_sample(self, scene_name: str):
        is_scannet = "scene" in scene_name
        if is_scannet:
            captions, inst_ids = list(zip(*self.anno_scannet[scene_name]))
        else:
            captions, inst_ids = list(zip(*self.anno_3rscan[scene_name]))

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

        is_scannet = "scene" in scene_name
        if is_scannet:
            coord, color, inst_labels = self._load_scannet_pcd(scene_name)
        else:
            coord, color, inst_labels = self._load_3rscan_pcd(scene_name)

        segment = np.ones_like(coord[:, 0]).astype(np.int64) * self.ignore_label  # all are ignored
        binary_label = np.ones_like(segment)
        origin_idx = np.arange(coord.shape[0]).astype(np.int64)

        data_dict = dict(
            coord=coord,
            color=color,
            segment=segment,
            binary=binary_label,
            origin_idx=origin_idx,
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
    import os

    data_dir = "/datasets/leo"
    dataset = LeoDataset(data_dir, "train", None)

    sample = dataset[0]
    import pdb

    pdb.set_trace()
    print(sample)
