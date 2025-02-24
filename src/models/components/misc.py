from typing import Tuple

import torch
import numpy as np
from jaxtyping import Int


@torch.inference_mode()
def offset2bincount(offset):
    prepend = None
    if offset[0] != 0:
        prepend = torch.tensor([0], device=offset.device, dtype=torch.long)
    return torch.diff(offset, prepend=prepend)


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(
        bincount
    )


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# reference: warpconvnet/utils/ravel.py
def ravel_multi_index(
    multi_index: Int[torch.Tensor, "* D"],  # noqa: F821
    spatial_shape: Tuple[int, ...],  # noqa: F821
) -> Int[torch.Tensor, "*"]:  # noqa: F722
    """
    Converts a tuple of index arrays into an array of flat indices.

    Args:
        multi_index: A tensor of coordinate vectors, (*, D).
        dims: The source shape.
    """
    # assert multi index is integer dtype
    assert multi_index.dtype in [torch.int16, torch.int32, torch.int64]
    assert multi_index.shape[-1] == len(spatial_shape)
    # Convert dims to a list of tuples
    if isinstance(spatial_shape, torch.Tensor):
        spatial_shape = spatial_shape.cpu().tolist()
    strides = torch.tensor(
        [np.prod(spatial_shape[i + 1 :]) for i in range(len(spatial_shape))], dtype=torch.int64
    ).to(multi_index.device)
    return (multi_index * strides).sum(dim=-1)


def ravel_multi_index_auto_shape(
    x: Int[torch.Tensor, "* D"],  # noqa: F821
    dim: int = 0,
) -> Int[torch.Tensor, "*"]:  # noqa: F722
    min_coords = x.min(dim=dim).values
    shifted_x = x - min_coords
    shape = shifted_x.max(dim=dim).values + 1
    raveled_x = ravel_multi_index(shifted_x, shape)
    return raveled_x
