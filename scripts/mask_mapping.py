import os
from pathlib import Path

import numpy as np
from natsort import natsorted
import torch
from src.utils.io import save_result_to_file, unpack_list_of_np_arrays

DATADIR = {
    "scannet200": {
        "pcd": "/datasets/scannet_hf",
        "gt_mask": "/datasets/scannet_masks/oracle_scannet200_full",
        "segment3d_mask": "/datasets/scannet_masks/segment3d",
    },
    "scannetpp": {
        "pcd": "/datasets/scannetpp_hf",
        "segment3d_mask": "/datasets/scannetpp_masks/segment3d",
    },
    "arkitscenes": {
        "pcd": "/datasets/arkitscenes/3dod",
        "segment3d_mask": "/datasets/arkitscenes_masks/segment3d",
    },
}

if __name__ == "__main__":
    import argparse

    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="scannet200",
        choices=["scannet200", "scannetpp", "arkitscenes"],
    )
    parser.add_argument("--mask_type", type=str, required=True)
    parser.add_argument("--segment_dir", type=str, required=True)
    parser.add_argument("--caption_dir", type=str, required=True)
    args = parser.parse_args()

    pcd_dir = Path(DATADIR[args.dataset]["pcd"])

    if args.mask_type not in DATADIR[args.dataset]:
        raise ValueError(f"Mask type {args.mask_type} is not supported for dataset {args.dataset}")

    mask_dir = Path(DATADIR[args.dataset][args.mask_type])
    segment_dir = Path(args.segment_dir)
    caption_dir = Path(args.caption_dir)

    scenes = natsorted(os.listdir(caption_dir))

    for scene in tqdm.tqdm(scenes):
        caption_to_mask_file = caption_dir / scene / f"caption_to_{args.mask_type}.npz"
        # if caption_to_mask_file.exists():
        #     print(f"Skipping {scene} because it already exists")
        #     continue

        print(f"Processing {scene}...")
        point_indices_all = unpack_list_of_np_arrays(segment_dir / scene / "point_indices.npz")

        # Load mask data (now in point indices format)
        mask_file = mask_dir / scene / "point_indices.npz"
        mask_data = np.load(mask_file)

        # Check if the mask data is in the new format (with packed and lengths)
        if "packed" in mask_data and "lengths" in mask_data:
            # New format with point indices
            mask_point_indices = mask_data["packed"]
            mask_lengths = mask_data["lengths"]
            scores = mask_data["scores"] if "scores" in mask_data else None

            # Convert point indices to masks
            # First, load the point cloud to get the total number of points
            coord_file = pcd_dir / "train" / scene / "coord.npy"
            if not coord_file.exists():
                coord_file = pcd_dir / "Training" / scene / "coord.npy"
            if not coord_file.exists():
                coord_file = pcd_dir / "val" / scene / "coord.npy"
            if not coord_file.exists():
                coord_file = pcd_dir / "Validation" / scene / "coord.npy"
            if not coord_file.exists():
                raise ValueError(f"Coord file {coord_file} does not exist")
            coord = np.load(coord_file)
            num_points = coord.shape[0]

            # Create masks from point indices
            masks = np.zeros((len(mask_lengths), num_points), dtype=bool)
            start_idx = 0
            for i, length in enumerate(mask_lengths):
                indices = mask_point_indices[start_idx : start_idx + length]
                masks[i, indices] = 1
                start_idx += length
        else:
            # Old format with binary masks
            masks = mask_data["masks_binary"]
            scores = mask_data["scores"] if "scores" in mask_data else None

        # loop over all point indices from the segment predictions
        num_objects_per_image = [len(point_indices) for point_indices in point_indices_all]
        num_objects = sum(num_objects_per_image)
        pred_masks = np.zeros((num_objects, masks.shape[1]), dtype=bool)
        point_indices_flatten = [item for sublist in point_indices_all for item in sublist]
        for i, point_indices in enumerate(point_indices_flatten):
            pred_masks[i, point_indices] = 1

        # Convert numpy arrays to PyTorch tensors and move to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred_masks_tensor = torch.from_numpy(pred_masks).to(device)
        masks_tensor = torch.from_numpy(masks).to(device)

        # Compute overlaps using matrix multiplication on GPU
        # Convert boolean tensors to float for matrix multiplication
        pred_masks_float = pred_masks_tensor.float()
        masks_float = masks_tensor.float()
        overlaps = torch.matmul(pred_masks_float, masks_float.T)

        # Find argmax and max values on GPU
        argmax = torch.argmax(overlaps, dim=1)
        max_overlaps = overlaps[torch.arange(overlaps.shape[0], device=device), argmax]

        # Create captions_to_masks_new on GPU
        captions_to_masks_new = torch.where(
            max_overlaps > 0, argmax, torch.tensor(-1, device=device)
        ).int()

        # Move results back to CPU and convert to numpy
        captions_to_masks_new = captions_to_masks_new.cpu().numpy()

        # Split the results using numpy
        captions_to_masks_all = np.split(
            captions_to_masks_new, np.cumsum(num_objects_per_image)[:-1]
        )

        # Free up CUDA memory
        torch.cuda.empty_cache()

        save_result_to_file(
            f"caption_to_{args.mask_type}", caption_dir / scene, captions_to_masks_all
        )
