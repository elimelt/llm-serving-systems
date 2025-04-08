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

// input and output are allocated on host device. need to be
// copied
void silu(float *input, float *output, int n) {
    // Allocate memory on the device
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 num_block((n + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 num_threads(BLOCKSIZE);
    silu_kernel<<<num_block, num_threads>>>(d_input, d_output, n);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Copy result back to host
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
