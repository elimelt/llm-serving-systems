#include <cuda_runtime.h>
#include "silu.h"
#include <iostream>

#define SIZE (8192 * 8192)
#define ITERS 10

int main() {
    // Allocate and initialize host memory
    float* h_input = new float[SIZE];
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory and copy input data
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, SIZE * sizeof(float));
    cudaMalloc((void**)&d_output, SIZE * sizeof(float));
    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel and measure time across multiple iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalTime = 0;
    
    for (int iter = 0; iter < ITERS; iter++) {
        cudaEventRecord(start);
        silu(d_input, d_output, SIZE);
        cudaEventRecord(stop);


        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;
        cudaDeviceSynchronize();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Average time per iteration: %f ms\n", totalTime / ITERS);

    // Free device/host memory
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    return 0;
}