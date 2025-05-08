import cupy
import torch
import math
import numpy as np

from src.cuda.load import load_kernel

bsearch_kernel = load_kernel("find_first_gt_bsearch.cu", "find_first_gt_bsearch")

# 2) Prepare data on GPU (as torch.IntTensor)
M = 1024
N = 1 << 20
torch_srcM = torch.randint(0, 10000, (M,), dtype=torch.int32, device="cuda").sort()[0].contiguous()
torch_srcN = torch.randint(0, 10000, (N,), dtype=torch.int32, device="cuda")
torch_out = torch.empty(N, dtype=torch.int32, device="cuda")

# Launch parameters
threads = 256
blocks = math.ceil(N / threads)
shared_mem_bytes = M * torch_srcM.element_size()

# Launch binary-search kernel
bsearch_kernel(
    (blocks,),
    (threads,),
    (torch_srcM.data_ptr(), M, torch_srcN.data_ptr(), N, torch_out.data_ptr()),
    shared_mem=shared_mem_bytes,
)

# Print the first 100 results
print(torch_out[:100])


# Use numpy to verify the first 100 results
np_srcM = torch_srcM.cpu().numpy()
np_srcN = torch_srcN.cpu().numpy()
np_out = np.empty(100, dtype=np.int32)

# Use numpy to verify the first 100 results
np_out = np.empty(100, dtype=np.int32)

for i in range(100):
    np_out[i] = np.searchsorted(np_srcM, np_srcN[i])
    print(f"out: {np_out[i]}, M: {np_srcM[np_out[i] - 1]}-{np_srcM[np_out[i]]}, N: {np_srcN[i]}")

print(np_out)
