import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter

from src.models.components.misc import offset2batch

try:
    from src.models.external_libs.softgroup_ops import softgroup_ops as sg_ops
except Exception:
    pass


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
    def __init__(self, model_cfg={}, voxel_mode=4, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.use_xyz = model_cfg.get("USE_XYZ", False)
        self.voxel_mode = voxel_mode

    def forward(self, batch):
        batch_size = batch["batch_size"]
        voxel_coords, v2p_map, p2v_map = sg_ops.voxelization_idx(
            batch["points_xyz_voxel_scale"].cpu(), batch_size, self.voxel_mode
        )
        voxel_coords, v2p_map, p2v_map = (
            voxel_coords.cuda(),
            v2p_map.cuda(),
            p2v_map.cuda(),
        )

        feats = batch["feats"]  # (N, C), float32, cuda

        voxel_feats = sg_ops.voxelization(feats, p2v_map, self.voxel_mode)

        batch.update(
            {
                "voxel_features": voxel_feats,
                "v2p_map": v2p_map.long(),
                "voxel_coords": voxel_coords,
            }
        )

        return batch


class IndoorVFEv2(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, batch):
        coords = batch["points_xyz_voxel_scale"]

        key = fnv_hash_vec(coords.cpu().numpy())
        idx_sort = np.argsort(key)
        idx_sort_rev = np.argsort(idx_sort)
        key_sort = key[idx_sort]
        _, index, inverse, count = np.unique(
            key_sort, return_index=True, return_inverse=True, return_counts=True
        )
        voxel_ids = (
            offset2batch(
                torch.from_numpy(np.cumsum(np.insert(count, 0, 0))).to(
                    batch["points_xyz_voxel_scale"].device
                )
            )
            - 1
        )
        voxel_coords = coords[idx_sort[index]]
        voxel_feats = scatter(batch["feats"][idx_sort], voxel_ids, dim=0, reduce="mean")

        batch.update(
            {
                "voxel_features": voxel_feats,
                "voxel_coords": voxel_coords,
                "v2p_map": torch.from_numpy(inverse[idx_sort_rev]).to(voxel_feats.device).long(),
            }
        )
        return batch
