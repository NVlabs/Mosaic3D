from typing import List, Tuple

import hydra
import numpy as np
import pytest
import torch
import warp as wp
import yaml
from jaxtyping import Int
from lightning import LightningDataModule
from omegaconf import OmegaConf
from torch import Tensor
from warp.convnet.geometry.ops.voxel_ops import voxel_downsample_random_indices
from warp.convnet.geometry.point_collection import PointCollection

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_histogram(num_points, bins=10):
    hist, bin_edges = np.histogram(num_points, bins=bins)
    print(f"Number of points in each point cloud: {len(num_points)}")
    for i in range(len(hist)):
        bin_range = f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}"
        print(f"{bin_range}: {hist[i]}")


@pytest.mark.parametrize(
    "voxel_size",
    [
        0.025,
    ],
)
def test_loading(voxel_size: str) -> None:
    with open("configs/data/regionplc_base15.yaml") as f:
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
    wp.init()
    test_loading(voxel_size=0.025)
