import numpy as np
import torch
from lightning import seed_everything
from torch_scatter import scatter

from src.models.components.misc import offset2batch
from src.models.external_libs.softgroup_ops.ops import functions as sg_ops
from src.models.regionplc.vfe import fnv_hash_vec
from tests.helpers.timer import Meter


def ops_voxelize(coords, feats, batch_size):
    device = coords.device
    voxel_coords, v2p_map, p2v_map = sg_ops.voxelization_idx(coords.cpu(), batch_size, 4)
    voxel_coords, v2p_map, p2v_map = (
        voxel_coords.to(device),
        v2p_map.to(device),
        p2v_map.to(device),
    )

    voxel_feats = sg_ops.voxelization(feats, p2v_map, 4)
    return voxel_coords, voxel_feats, v2p_map, p2v_map


def torchscatter_voxelize(coords, feats, batch_size):
    # batch_idx, coords = coords.split([1, 3], dim=-1)

    key = fnv_hash_vec(coords.cpu().numpy())
    idx_sort = np.argsort(key)
    idx_sort_rev = np.argsort(idx_sort)
    key_sort = key[idx_sort]
    _, index, inverse, count = np.unique(
        key_sort, return_index=True, return_inverse=True, return_counts=True
    )

    sel = offset2batch(torch.from_numpy(np.cumsum(np.insert(count, 0, 0))).to(feats.device)) - 1

    voxel_coords = coords[idx_sort[index]]
    voxel_feats = scatter(feats[idx_sort], sel, dim=0, reduce="mean")
    return voxel_coords, voxel_feats, inverse[idx_sort_rev], idx_sort[index]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mul", type=int, default=1)
    args = parser.parse_args()

    seed_everything(0)

    N, C = 1000, 3
    M = 16

    N *= args.mul

    coords = torch.rand(N * 6, C) / 0.01
    coords -= coords.min(0).values

    coords = torch.floor(coords).long()

    offset = torch.cumsum(torch.IntTensor([0] + [1 * N, 2 * N, 3 * N]), 0)
    batch_idx = offset2batch(offset) - 1
    batched_coords = torch.cat([batch_idx.unsqueeze(-1), coords], dim=-1)

    feat = torch.rand(N * 6, M)

    batched_coords = batched_coords.cuda()
    feat = feat.cuda()

    timer_ops, timer_ts = Meter("ops"), Meter("torchscatter")
    ret0 = ops_voxelize(batched_coords, feat, batch_idx.max().item() + 1)
    ret1 = torchscatter_voxelize(batched_coords, feat, batch_idx.max().item() + 1)

    key0 = fnv_hash_vec(ret0[0].cpu().numpy())
    idx_sort0 = np.argsort(key0)
    key1 = fnv_hash_vec(ret1[0].cpu().numpy())
    idx_sort1 = np.argsort(key1)

    test_feat = torch.allclose(ret0[0], ret1[0])
    if test_feat:
        print("Feat all close")

    test_count = torch.allclose(ret0[1], ret1[1])
    if test_count:
        print("Counts all close")

    min_trial = 10

    print(f"N: {N}, M: {M}")

    for i in range(min_trial):
        with timer_ops:
            ret0 = ops_voxelize(batched_coords, feat, batch_idx.max().item() + 1)
            torch.cuda.synchronize()

    print(f"OPS: {timer_ops.min_time:.3f}, Mem: {timer_ops.max_memory_usage:.3f}")

    for i in range(min_trial):
        with timer_ts:
            ret1 = torchscatter_voxelize(batched_coords, feat, batch_idx.max().item() + 1)
            torch.cuda.synchronize()
    print(f"TS: {timer_ts.min_time:.3f}, Mem: {timer_ts.max_memory_usage:.3f}")
