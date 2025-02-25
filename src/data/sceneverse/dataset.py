import json
import os
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


class SceneverseDataset(Dataset):
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

        self.dataset_name = "sceneverse"
        self.data_dir = Path(data_dir)
        assert self.data_dir.exists(), f"{self.data_dir} not exist."

        self.split = split
        self.repeat = repeat
        self.ignore_label = ignore_label
        self.object_num_max = object_num_max

        self.subsets = [
            "3RScan",
            "ARKitScenes",
            "HM3D",
            "MultiScan",
            "ProcThor",
            "ScanNet",
            "Structured3D",
        ]

        # load annotation
        self.anno_all = {}
        for subset in self.subsets:
            anno = self._load_anno(subset)
            self.anno_all[subset] = anno

        # scene names
        scene_names_all = []
        for subset in self.subsets:
            anno = self.anno_all[subset]
            if subset == "ScanNet":
                split_file = (
                    self.data_dir / subset / "annotations" / "splits" / f"scannetv2_{split}.txt"
                )
            else:
                split_file = (
                    self.data_dir / subset / "annotations" / "splits" / f"{split}_split.txt"
                )
            with open(split_file, "r") as f:
                scene_names = [line.strip() for line in f]

            # filter scene names not in anno
            scene_names = [
                (subset, scene_name) for scene_name in scene_names if scene_name in anno
            ]
            scene_names_all.extend(scene_names)
        self.scene_names = scene_names_all

        # data transform
        if transforms is not None:
            transforms_cfg = OmegaConf.to_container(transforms)
            self.transforms = Compose(transforms_cfg)
        else:
            self.transforms = lambda x: x

    def _load_anno(self, subset: str):
        anno_files = [
            "nr3d.jsonl",
            "scanrefer.jsonl",
            "sr3d+.jsonl",
            "object_cap_train.json",
            "ssg_obj_caption_gpt.json",
            "ssg_obj_caption_template.json",
            "ssg_ref_chain_gpt.json",
            "ssg_ref_chain_template.json",
            "ssg_ref_rel2_gpt.json",
            "ssg_ref_rel2_template.json",
            "ssg_ref_relm_gpt.json",
            "ssg_ref_relm_template.json",
            "ssg_ref_star_gpt.json",
            "ssg_ref_star_template.json",
        ]
        anno_dir = self.data_dir / subset / "annotations"
        if subset == "ScanNet":
            anno_dir = anno_dir / "refer"

        # filter anno_files
        anno_files = [anno_file for anno_file in anno_files if (anno_dir / anno_file).exists()]

        anno = {}
        for i, anno_file in enumerate(anno_files):
            anno_path = anno_dir / anno_file
            log.info(f"[{subset}|{i}/{len(anno_files)}] Loading annotation from {anno_path}")

            if anno_file.endswith(".json"):
                with open(anno_path, "r") as f:
                    data = json.load(f)
            else:
                with open(anno_path, "r") as f:
                    data = [json.loads(line.strip()) for line in f]

            for item in data:
                scan_id = item["scan_id"].replace(f"{subset}_", "")
                target_id = int(item["target_id"])
                caption = (
                    item["utterance"]
                    if "utterance" in item
                    else item["conversations"][-1]["value"]
                )
                if scan_id in anno:
                    anno[scan_id].append((caption, target_id))
                else:
                    anno[scan_id] = [(caption, target_id)]
        return anno

    def load_point_cloud(self, subset: str, scene_name: str):
        pcd_path = (
            self.data_dir
            / subset
            / "scan_data"
            / "pcd_with_global_alignment"
            / f"{scene_name}.pth"
        )
        coord, color, *_, inst_labels = torch.load(pcd_path, map_location="cpu")
        return coord, color.astype(np.uint8), inst_labels.astype(int)

    def load_caption_and_sample(self, subset: str, scene_name: str):
        captions, inst_ids = list(zip(*self.anno_all[subset][scene_name]))

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
        subset, scene_name = self.scene_names[idx]
        coord, color, inst_labels = self.load_point_cloud(subset, scene_name)

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
            inst_ids, captions = self.load_caption_and_sample(subset, scene_name)
            point_indices = [
                torch.from_numpy(np.where(inst_labels == inst_id)[0]).int() for inst_id in inst_ids
            ]
            data_dict["caption_data"] = {"idx": point_indices, "caption": captions}

        data_dict = self.transforms(data_dict)
        return data_dict


if __name__ == "__main__":
    import os

    data_dir = "/datasets/SceneVerse"
    dataset = SceneverseDataset(data_dir, "train", None)

    sample = dataset[0]
    print(sample)
