from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from natsort import natsorted
from tqdm import tqdm

from src.utils.io import save_result_to_file, unpack_list_of_np_arrays

SEGMENT3D_MASKS_DIR = Path("/home/chrockey/datasets/scannet_masks/segment3d")
MOSAIC3D_DIR = Path("/home/chrockey/datasets/mosaic3d/data/scannet")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, choices=["gsam2", "seem"])
    args = parser.parse_args()

    # train split
    split_file_path = (
        Path(__file__).parent.parent
        / "src"
        / "data"
        / "metadata"
        / "split_files"
        / "scannet_train.txt"
    )
    with open(split_file_path) as f:
        scene_names = natsorted(
            [line.strip() for line in f.readlines() if not line.startswith("#")]
        )

    for scene_name in tqdm(scene_names):
        mapping_file = MOSAIC3D_DIR / scene_name / f"segment3d-mapping.{args.source}.npz"
        # if mapping_file.exists():
        #     print(f"Skipping {scene_name} because it already exists")
        #     continue

        print(f"Processing {scene_name}...")
        segment3d_mask_file = SEGMENT3D_MASKS_DIR / scene_name / "point_indices.npz"
        assert segment3d_mask_file.exists(), f"Mask file {segment3d_mask_file} not found"

        segment3d_mask_data = np.load(segment3d_mask_file)
        segment3d_mask_point_indices = segment3d_mask_data["packed"]
        segment3d_mask_lengths = segment3d_mask_data["lengths"]
        segment3d_mask_scores = (
            segment3d_mask_data["scores"] if "scores" in segment3d_mask_data else None
        )

        # Convert point indices to masks
        # First, load the point cloud to get the total number of points
        coord_file = MOSAIC3D_DIR / scene_name / "coord.npy"
        coord = np.load(coord_file)
        num_points = coord.shape[0]

        # Create binary masks from point indices
        segment3d_masks = np.zeros((len(segment3d_mask_lengths), num_points), dtype=bool)
        start_idx = 0
        for i, length in enumerate(segment3d_mask_lengths):
            indices = segment3d_mask_point_indices[start_idx : start_idx + length]
            segment3d_masks[i, indices] = 1

        # Load point indices
        point_indices_file = MOSAIC3D_DIR / scene_name / f"point_indices.{args.source}.npz"
        point_indices_all = unpack_list_of_np_arrays(point_indices_file)

        # Loop over all point indices from the segment predictions
        num_objects_per_scene = [len(point_indices) for point_indices in point_indices_all]
        num_objects = sum(num_objects_per_scene)
        source_masks = np.zeros((num_objects, num_points), dtype=bool)
        point_indices_flatten = [item for sublist in point_indices_all for item in sublist]
        for i, point_indices in enumerate(point_indices_flatten):
            source_masks[i, point_indices] = 1

        # Convert numpy arrays to PyTorch tensors and move to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        source_masks_tensor = torch.from_numpy(source_masks).to(device)
        segment3d_masks_tensor = torch.from_numpy(segment3d_masks).to(device)

        # Compute IoU (Intersection over Union) using matrix multiplication on GPU
        # Convert boolean tensors to float for matrix multiplication
        source_masks_float = source_masks_tensor.float()
        segment3d_masks_float = segment3d_masks_tensor.float()

        # Calculate intersection
        intersection = torch.matmul(source_masks_float, segment3d_masks_float.T)

        # Calculate areas for union computation
        source_areas = source_masks_float.sum(dim=1, keepdim=True)
        segment3d_areas = segment3d_masks_float.sum(dim=1, keepdim=True)

        # Calculate union: area1 + area2 - intersection
        union = source_areas + segment3d_areas.T - intersection

        # Calculate IoU with epsilon to avoid division by zero
        iou = intersection / (union + 1e-8)

        # Find argmax and max values on GPU
        argmax = torch.argmax(iou, dim=1)
        max_iou = iou[torch.arange(iou.shape[0], device=device), argmax]

        # Create captions_to_masks on GPU
        # Now using IoU threshold instead of simple overlap count
        captions_to_masks = torch.where(max_iou > 0, argmax, torch.tensor(-1, device=device)).int()

        # Move results back to CPU and convert to numpy
        captions_to_masks = captions_to_masks.cpu().numpy()

        # Split the results using numpy
        captions_to_masks_all = np.split(captions_to_masks, np.cumsum(num_objects_per_scene)[:-1])

        # Free up CUDA memory
        torch.cuda.empty_cache()

        save_result_to_file(
            f"segment3d-mapping.{args.source}", MOSAIC3D_DIR / scene_name, captions_to_masks_all
        )
