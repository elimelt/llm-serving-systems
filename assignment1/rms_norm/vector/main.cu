#include<cuda_runtime.h>
#include "rms_norm_vector.h"
#include <stdio.h>

#define SIZE 1024 * 1024
#define ITERS 10

int main() {
    float* h_input = new float[SIZE];
    float* h_weight = new float[SIZE];

    for (int i = 0; i < SIZE; i++) {
        h_input[i] = static_cast<float>(i + 1);
        h_weight[i] = 1.0f;
    }

    size_t matrixSize = SIZE * sizeof(float);
    float *d_input, *d_weight, *d_output;
    cudaMalloc((void**)&d_input, matrixSize);
    cudaMalloc((void**)&d_weight, matrixSize);
    cudaMalloc((void**)&d_output, matrixSize);

    cudaMemcpy(d_input, h_input, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, matrixSize, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalTime = 0;

    for (int iter = 0; iter < ITERS; iter++) {
        cudaEventRecord(start);
        rms_norm_vector(d_input, d_weight, d_output, SIZE, 0.000001f);
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

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_weight;
    return 0;
}