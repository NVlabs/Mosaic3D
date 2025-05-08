extern "C" __global__ void find_first_gt_bsearch(const int *__restrict__ srcM,
                                                 int M,
                                                 const int *__restrict__ srcN,
                                                 int N,
                                                 int *__restrict__ out)
{
    extern __shared__ int smem[]; // dynamic shared memory
    int tid = threadIdx.x;

    // Load sorted M elements into shared memory
    for (int i = tid; i < M; i += blockDim.x)
    {
        smem[i] = srcM[i];
    }
    __syncthreads(); // ensure smem[] is ready

    // Compute global thread index over N
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < N)
    {
        int key = srcN[idx];
        int left = 0, right = M; // search in [0, M)
        while (left < right)
        { // binary search loop
            int mid = (left + right) >> 1;
            if (key < smem[mid])
            {
                right = mid; // first greater must be left of or at mid
            }
            else
            {
                left = mid + 1; // skip mid, search right half
            }
        }
        // left == first index where smem[left] > key, or left==M if none
        out[idx] = (left < M) ? left : -1;
    }
}

// Find the first index where the arange index is greater than the sorted array M
extern "C" __global__ void find_first_gt_bsearch_arange(const int *__restrict__ srcM,
                                                        int M,
                                                        int N,
                                                        int *__restrict__ out)
{
    extern __shared__ int smem[]; // dynamic shared memory
    int tid = threadIdx.x;

    // Load sorted M elements into shared memory
    for (int i = tid; i < M; i += blockDim.x)
    {
        smem[i] = srcM[i];
    }
    __syncthreads(); // ensure smem[] is ready

    // Compute global thread index over N
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < N)
    {
        int key = idx;
        int left = 0, right = M; // search in [0, M)
        while (left < right)
        { // binary search loop
            int mid = (left + right) >> 1;
            if (key < smem[mid])
            {
                right = mid; // first greater must be left of or at mid
            }
            else
            {
                left = mid + 1; // skip mid, search right half
            }
        }
        // left == first index where smem[left] > key, or left==M if none
        out[idx] = (left < M) ? left : -1;
    }
}
