from pathlib import Path
from typing import List
import argparse

from natsort import natsorted
import torch
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src.models.components.evaluator import InstanceSegmentationEvaluator
from src.utils.io import unpack_list_of_np_arrays

console = Console()


def main(
    dataset: str,
    mask_dirs: List[str],
    split: str,
) -> None:
    gt_dir = Path(f"/datasets/{dataset.replace('v2', '')}_hf") / split.replace("-tiny", "")
    mask_dirs = [Path(mask_dir) for mask_dir in mask_dirs]

    # val split
    with open(f"src/data/metadata/split_files/{dataset}_{split}.txt") as f:
        val_scenes = natsorted(f.read().splitlines())

    # evaluator
    evaluator = InstanceSegmentationEvaluator(
        class_names=["object"],
        segment_ignore_index=[-1],
        instance_ignore_index=-1,
    )

    console.print(f"Evaluating {mask_dirs} on {dataset} ({split})...")
    for scene in tqdm(val_scenes):
        # GT masks
        gt_segment = torch.from_numpy(
            np.load(gt_dir / scene / "segment200.npy" if dataset == "scannetv2" else "segment.npy")
        )
        gt_segment[gt_segment != -1] = 0  # binarize
        gt_instance = torch.from_numpy(np.load(gt_dir / scene / "instance.npy"))
        num_points = len(gt_segment)

        # Point indices
        point_indices_all = []
        for mask_dir in mask_dirs:
            point_indices = unpack_list_of_np_arrays(mask_dir / scene / "point_indices.npz")
            if isinstance(point_indices[0], list):
                point_indices = [item for sublist in point_indices for item in sublist]
            point_indices_all.extend(point_indices)

        point_indices_all = [torch.from_numpy(indices).int() for indices in point_indices_all]

        # Point indices to masks
        num_masks = len(point_indices_all)
        pred_classes = torch.zeros(num_masks, dtype=torch.long)
        pred_scores = torch.ones(num_masks)
        pred_masks = torch.zeros((num_masks, num_points))
        for i, indices in enumerate(point_indices_all):
            pred_masks[i, indices] = 1

        evaluator.update(pred_classes, pred_scores, pred_masks, gt_segment, gt_instance)

    results = evaluator.compute()

    # Create a table to display results
    table = Table(title="Mask Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    metrics = {
        "AP": results["classes"]["object"]["ap"],
        "AP50": results["classes"]["object"]["ap50"],
        "AP25": results["classes"]["object"]["ap25"],
    }

    for metric_name, value in metrics.items():
        table.add_row(metric_name, f"{value*100:.1f}")

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="scannetv2")
    parser.add_argument(
        "--mask_dirs", action="append", default=[], help="Paths to mask directories"
    )
    parser.add_argument("--split", type=str, default="val")

    args = parser.parse_args()

    # Set default mask_dirs if none provided
    if not args.mask_dirs:
        args.mask_dirs = ["/datasets/scannet_masks/mask3d_scannet200_openins3d_converted"]

    main(
        dataset=args.dataset,
        mask_dirs=args.mask_dirs,
        split=args.split,
    )
