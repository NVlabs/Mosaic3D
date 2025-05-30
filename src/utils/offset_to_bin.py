from jaxtyping import Int
from typing import Optional

import math

import torch
from torch import Tensor

from src.cuda.load import CupyKernel

bsearch_arange_kernel = CupyKernel("find_first_gt_bsearch.cu", "find_first_gt_bsearch_arange")
bsearch_kernel = CupyKernel("find_first_gt_bsearch.cu", "find_first_gt_bsearch")


@torch.no_grad()
def offset_to_bin(
    offsets: Int[Tensor, "M + 1"],  # noqa: F821
    indices: Optional[Int[Tensor, "N"]] = None,  # noqa: F821
    threads: int = 256,
) -> Int[Tensor, "M"]:  # noqa: F821
    """
    Convert the offsets to target indices.
    e.g. [0, 3, 5, 6] -> [0, 0, 0, 1, 1, 2]
    Assume that the first element of caption_offsets is 0.
    """
    M = offsets.shape[0]
    shared_mem_bytes = M * offsets.element_size()  # M * sizeof(int)

    if indices is None:
        N = offsets[-1].item()
        torch_out = torch.empty(N, dtype=torch.int32, device=offsets.device)

        # Launch parameters
        blocks = math.ceil(N / threads)

        # Launch binary-search kernel
        bsearch_arange_kernel(
            (blocks,),
            (threads,),
            (offsets.int().data_ptr(), M, N, torch_out.data_ptr()),
            shared_mem=shared_mem_bytes,
        )
    else:
        N = indices.shape[0]
        torch_out = torch.empty(N, dtype=torch.int32, device=indices.device)

        # Launch parameters
        blocks = math.ceil(N / threads)

        bsearch_kernel(
            (blocks,),
            (threads,),
            (offsets.int().data_ptr(), M, indices.int().data_ptr(), N, torch_out.data_ptr()),
            shared_mem=shared_mem_bytes,
        )
    return torch_out
