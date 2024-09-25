from typing import List, Tuple

import hydra
import numpy as np
import pytest
import torch
import yaml
import argparse
from jaxtyping import Int
from lightning import LightningDataModule
from omegaconf import OmegaConf
from torch import Tensor

import warp as wp
from warpconvnet.geometry.ops.voxel_ops import voxel_downsample_random_indices
from warpconvnet.geometry.point_collection import PointCollection

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_histogram(num_points, bins=10):
    hist, bin_edges = np.histogram(num_points, bins=bins)
    print(f"Number of points in each point cloud: {len(num_points)}")
    for i in range(len(hist)):
        bin_range = f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}"
        print(f"{bin_range}: {hist[i]}")


@pytest.mark.parametrize(
    "config_path, voxel_size",
    [
        ("configs/data/regionplc_base15.yaml", 0.025),
    ],
)
def test_loading(config_path: str, voxel_size: str) -> None:
    with open(config_path) as f:
        omega_config_dict = yaml.safe_load(f.read())
    cfg = OmegaConf.create(omega_config_dict)
    cfg.val_dataset = cfg.train_dataset
    cfg.num_workers = 0
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    # collect all the num points
    num_points = []
    for i, batch_dict in enumerate(loader):
        assert isinstance(batch_dict, dict)

        offset = batch_dict["offset"]

        # random voxel downsample
        perm, down_offsets = voxel_downsample_random_indices(
            batched_points=batch_dict["coord"].to(DEVICE),
            offsets=offset,
            voxel_size=voxel_size,
        )

        # get the number of points in each point cloud
        num_points.extend(down_offsets[1:] - down_offsets[:-1])

        # Check caption_data
        caption_dict = batch_dict["caption_data"]
        batched_point_indices_of_captions = caption_dict["idx"]
        batched_captions: List[str] = caption_dict["caption"]

        for point_indices_of_captions, caption in zip(
            batched_point_indices_of_captions, batched_captions
        ):
            assert len(point_indices_of_captions) == len(caption)
            # assert all arrays are non-empty
            assert all(len(point_indices) > 0 for point_indices in point_indices_of_captions)

        if i == 4:
            break

    # count the frequency of each number of points
    print_histogram(num_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data loading with configurable path.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/data/ours_video_openvocab.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.025, help="Voxel size for downsampling"
    )
    args = parser.parse_args()

    wp.init()
    test_loading(config_path=args.config_path, voxel_size=args.voxel_size)
