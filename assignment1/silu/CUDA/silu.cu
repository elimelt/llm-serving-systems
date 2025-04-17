#include <cuda_runtime.h>
#include <iostream>

#define BLOCKSIZE 256

// __device__ void silu_kernel(...);


__global__ void silu_kernel(float *x, float *o, int n) {
    int block_start = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    int index = block_start + thread_id;

    if (index < n) {
        o[index] = x[index] / (1 + expf(-x[index]));
    }
}

void silu(float *input, float *output, int n) {
    dim3 num_block((n + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 num_threads(BLOCKSIZE);
    silu_kernel<<<num_block, num_threads>>>(input, output, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}
