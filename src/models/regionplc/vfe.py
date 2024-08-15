import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter

from src.models.components.misc import offset2batch


def fnv_hash_vec(arr):
    """FNV64-1A."""
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


class IndoorVFE(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, batch):
        coords = batch["points_xyz_voxel_scale"]

        coords_np = coords
        device = "cpu"
        if isinstance(coords, torch.Tensor):
            coords_np = coords.cpu().numpy()
            device = coords.device

        key = fnv_hash_vec(coords_np)
        idx_sort = np.argsort(key)
        idx_sort_rev = np.argsort(idx_sort)
        key_sort = key[idx_sort]
        _, index, inverse, count = np.unique(
            key_sort, return_index=True, return_inverse=True, return_counts=True
        )
        voxel_ids = (
            offset2batch(torch.from_numpy(np.cumsum(np.insert(count, 0, 0))).to(device)) - 1
        )
        voxel_coords = coords[idx_sort[index]]
        feats = batch["feats"]
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats).to(device)
        voxel_feats = scatter(feats[idx_sort], voxel_ids, dim=0, reduce="mean")

        v2p_map_new = np.zeros_like(inverse)
        v2p_map_new[idx_sort] = inverse
        v2p_map_new = torch.from_numpy(v2p_map_new).to(voxel_feats.device).long()
        v2p_map = torch.from_numpy(inverse[idx_sort_rev]).to(voxel_feats.device).long()

        v2p_map_new = np.zeros_like(inverse)
        v2p_map_new[idx_sort] = inverse
        v2p_map_new = torch.from_numpy(v2p_map_new).to(voxel_feats.device).long()
        v2p_map = torch.from_numpy(inverse[idx_sort_rev]).to(voxel_feats.device).long()

        batch.update(
            {
                "voxel_features": voxel_feats,
                "voxel_coords": voxel_coords,
                "v2p_map": v2p_map,
            }
        )
        return batch
