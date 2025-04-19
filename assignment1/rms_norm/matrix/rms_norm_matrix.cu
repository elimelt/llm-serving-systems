#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define THREADS_PER_BLOCK 256

__global__ void rms_norm_kernel(const float *input, const float *weight, float *output,
                                int rows, int cols, float epsilon)
{
    // Each block works on one row.
    int row = blockIdx.x;

    // Shared memory for partial sums.
    __shared__ float sdata[THREADS_PER_BLOCK];
    sdata[threadIdx.x] = 0;

    // Each thread computes partial sum of squares (coalesced)
    // Note: ROWS * COLS reads from global memory
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
    {
        sdata[threadIdx.x] += input[row * cols + col] * input[row * cols + col];
    }
    __syncthreads();

    // Reduction: Sum all partial sums to get the full sum-of-squares.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // At this point, sdata[0] is the total sum of squares for the row.
    // Compute the RMS value.
    float rms = sqrtf(sdata[0] / cols + epsilon);

    // Normalize the row (coalesced)
    // Note: ROWS * COLS writes to global memory
    // Note: ROWS * COLS + COLS reads from global memory
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
    {
        output[row * cols + col] = input[row * cols + col] / rms * weight[col];
    }
}

void rms_norm_matrix(float *input, float *weight, float *output, int rows, int cols, float epsilon)
{
    // Launch kernel
    dim3 grid(rows);               // num blocks; each block is responsible for a row
    dim3 block(THREADS_PER_BLOCK); // num threads / block, num threads working on a single row
    rms_norm_kernel<<<grid, block>>>(input, weight, output, rows, cols, epsilon);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}
