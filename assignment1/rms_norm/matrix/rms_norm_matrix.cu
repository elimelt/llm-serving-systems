#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// __global__ void rms_norm_matrix_kernel(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row < rows) {
//         float sum = 0.0f;
//         for (int col = 0; col < cols; col++) {
//             float val = input[row + col * rows];
//             sum += val * val;
//         }
//         float rms = sqrt(sum / cols + epsilon);
//         for (int col = 0; col < cols; col++) {
//             int idx = row + col * rows;
//             output[idx] = (input[idx] / rms) * weight[idx];
//         }
//     }
// }

#define ELEMENTS_PER_BLOCK 256

__global__ void rms_norm_matrix_kernel(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    __shared__ float sdata[ELEMENTS_PER_BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < rows * cols) ? input[i] * input[i] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 2; s >>= 1) {
        // printf("s: %d, tid: %d, sdata[tid]: %f\n", s, tid, sdata[tid]);
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // get rms values:
    if (tid < rows) {
        sdata[tid] = sqrt((sdata[tid] / cols) + epsilon);
        printf("tid: %d, sdata[tid]: %f\n", tid, sdata[tid]);
    }
    __syncthreads();

    // Normalize the input and apply weight
    if (i < rows * cols) {
        int row = i % rows;
        float rms = sdata[row];
        output[i] = (input[i] / rms) * weight[i];
        printf("i: %d, row: %d, rms: %f, input[i]: %f, weight[i]: %f, output[i]: %f\n", i, row, rms, input[i], weight[i], output[i]);
    }
    __syncthreads();
}

void rms_norm_matrix(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc((void**)&d_input, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_weight, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_output, rows * cols * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockSize(ELEMENTS_PER_BLOCK);
    dim3 numBlocks((rows * cols + blockSize.x - 1) / blockSize.x);
    rms_norm_matrix_kernel<<<numBlocks, blockSize>>>(d_input, d_weight, d_output, rows, cols, epsilon);

    // Check for errors in kernel launch
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy the result back to host
    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}
