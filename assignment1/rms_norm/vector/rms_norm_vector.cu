#include "rms_norm_vector.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define GRID_SIZE 32

__device__ float atomicAddFloat(float* address, float val) {
    return atomicAdd(address, val);
}

__global__ void rms_norm_vector_kernel(float *input, float *weight, float *output, int cols, float epsilon) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum_squared = 0.0f;
    
    if (idx < cols) {
        float val = input[idx];
        sum_squared = val * val;
    }
    
    sdata[tid] = sum_squared;
    __syncthreads();
    
    for (int stride = BLOCK_SIZE / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    sum_squared = sdata[tid];
    
    if (tid < WARP_SIZE) {
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            sum_squared += __shfl_down_sync(0xffffffff, sum_squared, offset);
        }
    }
    
    __shared__ float block_sum;
    
    if (tid == 0) {
        block_sum = 0.0f;
        atomicAddFloat(&block_sum, sum_squared);
    }
    __syncthreads();
    
    float rms = sqrtf(block_sum / cols + epsilon);
    
    if (idx < cols) {
        output[idx] = (input[idx] / rms) * weight[idx];
    }
}

void rms_norm_vector(float *input, float *weight, float *output, int cols, float epsilon) {
    float *d_input, *d_weight, *d_output;
    
    cudaMalloc((void**)&d_input, cols * sizeof(float));
    cudaMalloc((void**)&d_weight, cols * sizeof(float));
    cudaMalloc((void**)&d_output, cols * sizeof(float));
    
    cudaMemcpy(d_input, input, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, cols * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((cols + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    rms_norm_vector_kernel<<<gridDim, blockDim>>>(d_input, d_weight, d_output, cols, epsilon);
    
    cudaMemcpy(output, d_output, cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}