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

// __global__ void rms_norm_matrix_kernel(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
//     __shared__ float sdata[ELEMENTS_PER_BLOCK];
//     int tid = threadIdx.x;
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     sdata[tid] = (i < rows * cols) ? input[i] * input[i] : 0;
//     __syncthreads();

//     // Perform reduction in shared memory
//     for (unsigned int s = blockDim.x / 2; s > 2; s >>= 1) {
//         // printf("s: %d, tid: %d, sdata[tid]: %f\n", s, tid, sdata[tid]);
//         if (tid < s) {
//             sdata[tid] += sdata[tid + s];
//         }
//         __syncthreads();
//     }
    
//     // get rms values:
//     if (tid < rows) {
//         sdata[tid] = sqrt((sdata[tid] / cols) + epsilon);
//         printf("tid: %d, sdata[tid]: %f\n", tid, sdata[tid]);
//     }
//     __syncthreads();

//     // Normalize the input and apply weight
//     if (i < rows * cols) {
//         int row = i % rows;
//         float rms = sdata[row];
//         output[i] = (input[i] / rms) * weight[i];
//         printf("i: %d, row: %d, rms: %f, input[i]: %f, weight[i]: %f, output[i]: %f\n", i, row, rms, input[i], weight[i], output[i]);
//     }
//     __syncthreads();
// }

__global__ void rms_norm_kernel(const float *input, const float *weight, float *output,
                                int rows, int cols, float epsilon)
{
    // Each block works on one row.
    int row = blockIdx.x;
    
    // Declare shared memory for reduction (allocated at launch)
    extern __shared__ float sdata[];
    
    // Each thread computes a partial sum over its subset of the row.
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        int idx = row + col * rows;  // column-major index
        float val = input[idx];
        sum += val * val;
    }
    sdata[threadIdx.x] = sum;
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
    
    // Normalize the row: each thread processes multiple columns (strided).
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        int idx = row + col * rows;
        output[idx] = (input[idx] / rms) * weight[idx];
    }
}

void rms_norm_matrix(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
//     // Allocate device memory
//     float *d_input, *d_weight, *d_output;
//     cudaMalloc((void**)&d_input, rows * cols * sizeof(float));
//     cudaMalloc((void**)&d_weight, rows * cols * sizeof(float));
//     cudaMalloc((void**)&d_output, rows * cols * sizeof(float));

//     // Copy data from host to device
//     cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_weight, weight, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    
//     // Launch kernel
//     dim3 blockSize(ELEMENTS_PER_BLOCK);
//     dim3 numBlocks((rows * cols + blockSize.x - 1) / blockSize.x);
//     rms_norm_matrix_kernel<<<numBlocks, blockSize>>>(d_input, d_weight, d_output, rows, cols, epsilon);

//     // Check for errors in kernel launch
//     cudaDeviceSynchronize();
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
//     }

//     // Copy the result back to host
//     cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_input);
//     cudaFree(d_weight);
//     cudaFree(d_output);
    size_t matrixSize = rows * cols * sizeof(float);
    float *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr;
    
    cudaMalloc((void**)&d_input, matrixSize);
    cudaMalloc((void**)&d_weight, matrixSize);
    cudaMalloc((void**)&d_output, matrixSize);
    
    cudaMemcpy(d_input, input, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, matrixSize, cudaMemcpyHostToDevice);
    
    // Launch configuration:
    // Grid: one block per row.
    // Block: use a fixed number of threads (e.g. 256). Each thread will loop over columns.
    dim3 grid(rows);
    dim3 block(ELEMENTS_PER_BLOCK);
    
    // Allocate shared memory per block: one float per thread.
    size_t sharedMemSize = ELEMENTS_PER_BLOCK * sizeof(float);
    
    rms_norm_kernel<<<grid, block, sharedMemSize>>>(d_input, d_weight, d_output, rows, cols, epsilon);
    
    // Wait for kernel completion and check for errors.
    cudaDeviceSynchronize();
    
    cudaMemcpy(output, d_output, matrixSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}
