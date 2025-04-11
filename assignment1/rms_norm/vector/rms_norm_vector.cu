#include "rms_norm_vector.h"
#include <cuda_runtime.h>
#include <iostream>

#define ELEMENT_PER_BLOCK 256

__global__ void rms_norm_vector_kernel(float *input, float *weight, float *output, int cols, float epsilon) {
    __shared__ float sdata[ELEMENT_PER_BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < cols) ? input[i] * input[i] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // apply RMS normalization
    float rms = sqrtf(sdata[0] / cols + epsilon);
    output[i] = (i < cols) ? input[i] * weight[i] / rms : 0;
}

void rms_norm_vector(float *input, float *weight, float *output, int cols, float epsilon) {
    // Allocate device memory
    size_t vectorSize = cols * sizeof(float);
    float *d_input, *d_weight, *d_output;
    cudaMalloc((void**)&d_input, vectorSize);
    cudaMalloc((void**)&d_weight, vectorSize);
    cudaMalloc((void**)&d_output, vectorSize);
    
    // Copy data from host to device
    cudaMemcpy(d_input, input, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, vectorSize, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 grid((cols + ELEMENT_PER_BLOCK - 1) / ELEMENT_PER_BLOCK);
    dim3 block(ELEMENT_PER_BLOCK);
    rms_norm_vector_kernel<<<grid, block>>>(d_input, d_weight, d_output, cols, epsilon);
    
    // Wait for kernel completion and check for errors.
    cudaDeviceSynchronize();
    
    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Copy the result back to host
    cudaMemcpy(output, d_output, vectorSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}