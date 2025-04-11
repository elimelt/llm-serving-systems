#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define THREADS_PER_BLOCK 256

__global__ void rms_norm_kernel(const float *input, const float *weight, float *output,
                                int rows, int cols, float epsilon)
{
    // Each block works on one row.
    int row = blockIdx.x;
    int num_els = (cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int start_col = threadIdx.x * num_els;

    // Shared memory for partial sums.
    __shared__ float sdata[THREADS_PER_BLOCK];
    sdata[threadIdx.x] = 0;

    // Each thread computes a partial sum of squares for its assigned elements.
    for (int col = start_col; col < start_col + num_els && col < cols; col++) {
        sdata[threadIdx.x] += input[row * cols + col] * input[row * cols + col];
    }
    __syncthreads();
    
    // Reduction: Sum all partial sums to get the full sum-of-squares.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // At this point, sdata[0] is the total sum of squares for the row.
    // Compute the RMS value.
    float rms = sqrtf(sdata[0] / cols + epsilon);
    
    // Normalize the row
    for (int col = start_col; col < start_col + num_els && col < cols; col++) {
        output[row * cols + col] = input[row * cols + col] / rms * weight[row * cols + col];
    }
    
}

void rms_norm_matrix(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    // Allocate device memory
    size_t matrixSize = rows * cols * sizeof(float);
    float *d_input, *d_weight, *d_output;
    cudaMalloc((void**)&d_input, matrixSize);
    cudaMalloc((void**)&d_weight, matrixSize);
    cudaMalloc((void**)&d_output, matrixSize);
    
    // Copy data from host to device
    cudaMemcpy(d_input, input, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, matrixSize, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 grid(rows);
    dim3 block(THREADS_PER_BLOCK);
    rms_norm_kernel<<<grid, block>>>(d_input, d_weight, d_output, rows, cols, epsilon);
    
    // Wait for kernel completion and check for errors.
    cudaDeviceSynchronize();
    
    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Copy the result back to host
    cudaMemcpy(output, d_output, matrixSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}
