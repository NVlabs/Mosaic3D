import unittest

import torch
import torch.nn as nn

import warp as wp
from warpconvnet.geometry.point_collection import PointCollection
from src.models.components.structure import mean_pooling
from warpconvnet.nn.functional.point_pool import point_pool
from warpconvnet.nn.pools import PointToSparseWrapper
from warpconvnet.utils.batch_index import batch_index_from_offset
from warpconvnet.utils.unique import unique_inverse


class TestMeanPooling(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = torch.device("cuda:0")
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features).to(self.device)
        self.voxel_size = 0.01

    def test_mean_pooling(self):
        # convert the coordinatest to voxel coordinates
        voxel_coords = self.pc.coordinates // self.voxel_size
        batch_indices = batch_index_from_offset(
            self.pc.offsets, device=str(self.device)
        )
        batched_coords = torch.cat(
            [batch_indices.view(-1, 1).int(), voxel_coords.int()], dim=1
        ).contiguous()
        features = self.pc.features.clone()

        voxel_coords, voxel_feats, v2p_map = mean_pooling(
            batched_coords, features, return_inverse=True, hash_method="ravel"
        )

        # Test point_pool
        st, to_unique = point_pool(
            self.pc,
            reduction="mean",
            downsample_voxel_size=self.voxel_size,
            return_type="sparse",
            return_to_unique=True,
            unique_method="ravel",
        )

        self.assertTrue(torch.allclose(st.batch_indexed_coordinates, voxel_coords))
        self.assertTrue(torch.allclose(st.features, voxel_feats))
        self.assertTrue(torch.allclose(to_unique.to_orig_indices, v2p_map))

    def test_point_to_sparse_wrapper(self):
        # convert the coordinatest to voxel coordinates
        voxel_coords = self.pc.coordinates // self.voxel_size
        batch_indices = batch_index_from_offset(
            self.pc.offsets, device=str(self.device)
        )
        batched_coords = torch.cat(
            [batch_indices.view(-1, 1).int(), voxel_coords.int()], dim=1
        ).contiguous()
        features = self.pc.features.clone()

        voxel_coords, voxel_feats, v2p_map = mean_pooling(
            batched_coords, features, return_inverse=True, hash_method="ravel"
        )

        wrapper = PointToSparseWrapper(
            nn.Identity(),
            voxel_size=self.voxel_size,
            reduction="mean",
            unique_method="ravel",
            concat_unpooled_pc=False,
        )
        unpooled_pc = wrapper(self.pc)
        unpooled_voxel_feats = voxel_feats[v2p_map]

        self.assertTrue(torch.allclose(unpooled_pc.features, unpooled_voxel_feats))


if __name__ == "__main__":
    unittest.main()
