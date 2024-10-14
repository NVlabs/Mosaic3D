import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.models.components.evaluator import InstanceSegmentationEvaluator
from src.data.metadata.scannet import (
    CLASS_LABELS_200,
    HEAD_CLASSES_200,
    COMMON_CLASSES_200,
    TAIL_CLASSES_200,
)


def test_mask3d(mask_dir: str = "/datasets/scannet_masks/mask3d_scannet200") -> None:
    scannet_dir = "/datasets/scannet_hf/val"

    with open("src/data/metadata/split_files/scannetv2_val.txt") as f:
        scene_names = [line.strip() for line in f.readlines()]

    mapper = {}
    mapper["subset_names"] = ["head", "common", "tail"]
    for name in CLASS_LABELS_200:
        if name in HEAD_CLASSES_200:
            mapper[name] = "head"
        elif name in COMMON_CLASSES_200:
            mapper[name] = "common"
        elif name in TAIL_CLASSES_200:
            mapper[name] = "tail"
        else:
            raise ValueError(f"Unknown class name: {name}")

    evaluator = InstanceSegmentationEvaluator(
        class_names=CLASS_LABELS_200,
        segment_ignore_index=[-1, 0, 2],
        instance_ignore_index=-1,
        subset_mapper=mapper,
    )

    for scene_name in tqdm(scene_names):
        mask_path = Path(mask_dir) / f"{scene_name}.npz"
        assert mask_path.exists(), f"{mask_path} not exist."
        mask_data = np.load(mask_path)

        pred = dict(
            pred_classes=mask_data["classes"],
            pred_scores=mask_data["scores"],
            pred_masks=mask_data["masks_binary"],
        )
        target = dict(
            segment=np.load(Path(scannet_dir) / scene_name / "segment200.npy"),
            instance=np.load(Path(scannet_dir) / scene_name / "instance.npy"),
        )

        evaluator.update(pred, target)

    metrics = evaluator.compute()

    keys = ["map", "map50", "map25", "map_head", "map_common", "map_tail"]
    for key in keys:
        print(f"{key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    test_mask3d()
