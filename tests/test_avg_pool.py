import torch
from lightning import seed_everything
from torch_scatter import scatter

from src.models.components.misc import offset2batch
from src.models.regionplc.ops.pool_by_idx.pool_by_idx_utils import avg_pool_by_idx
from tests.helpers.timer import Meter


def ops_avg_pool(feat, point_idx, caption_idx, caption_offset, mask):
    pool_feat, n_feat = avg_pool_by_idx(
        feat,
        point_idx,
        torch.cat(caption_idx, dim=0).long(),
        caption_offset,
        None,
        mask,
    )
    return pool_feat, n_feat


def torchscatter_avg_pool(feat, point_idx, caption_idx, caption_offset, mask):
    batch_idx = offset2batch(caption_offset) - 1
    feat_mapped = feat[point_idx[torch.cat(caption_idx, dim=0)]]
    # pool_feat = scatter(feat_mapped, batch_idx, dim=0, reduce="mean")
    print(feat.shape, feat_mapped.shape, batch_idx.shape)
    n_feat = scatter(torch.ones_like(feat)[:, :1], batch_idx, dim=0, reduce="sum")
    # n_feat = torch.zeros(3).to(feat).long()
    pool_feat = torch.rand((3, 16)).to(feat)
    return pool_feat, n_feat


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mul", type=int, default=1)
    args = parser.parse_args()

    seed_everything(0)

    N, C = 1000, 16
    N0 = 900
    M = 1800

    N *= args.mul
    M *= args.mul
    N0 *= args.mul

    feat = torch.rand(N, C).cuda()
    point_idx = torch.randint(0, N, (N0,)).cuda()
    caption_idx = torch.randint(0, N0, (M,)).cuda().chunk(3, dim=0)
    caption_offset = torch.LongTensor([0] + [len(idx) for idx in caption_idx])
    caption_offset = torch.cumsum(caption_offset, 0).cuda()
    mask = torch.zeros(N).bool().cuda()

    timer_ops, timer_ts = Meter("ops"), Meter("torchscatter")
    ret0 = ops_avg_pool(feat, point_idx, caption_idx, caption_offset, mask)
    ret1 = torchscatter_avg_pool(feat, point_idx, caption_idx, caption_offset, mask)

    test_feat = torch.allclose(ret0[0], ret1[0])
    if test_feat:
        print("Feat all close")

    test_count = torch.allclose(ret0[1], ret1[1])
    if test_count:
        print("Counts all close")

    min_trial = 10

    print(f"N: {N}, M: {M}, N0: {N0}")

    for i in range(min_trial):
        with timer_ops:
            ret0 = ops_avg_pool(feat, point_idx, caption_idx, caption_offset, mask)
            torch.cuda.synchronize()

    print(f"OPS: {timer_ops.min_time:.3f}, Mem: {timer_ops.max_memory_usage:.3f}")

    for i in range(min_trial):
        with timer_ts:
            ret1 = torchscatter_avg_pool(feat, point_idx, caption_idx, caption_offset, mask)
            torch.cuda.synchronize()

    print(f"TS: {timer_ts.min_time:.3f}, Mem: {timer_ts.max_memory_usage:.3f}")
