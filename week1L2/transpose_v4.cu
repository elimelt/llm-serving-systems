#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 128
#define PADDED_TILE_SIZE (TILE_SIZE + 1)

__global__ void transposeKernel(float* input, float* output, int num_rows, int num_cols) {
    extern __shared__ float tile[];
    int row_start = blockIdx.y * TILE_SIZE;
    int col_start = blockIdx.x * TILE_SIZE + threadIdx.x;

    // load data into shared memory
    for (int i = 0; i < TILE_SIZE; i += 1) {
        if (row_start + i >= num_rows || col_start >= num_cols) {
            tile[i * PADDED_TILE_SIZE + threadIdx.x] = 0.0f;
        } else {
            tile[i * PADDED_TILE_SIZE + threadIdx.x] = input[(row_start + i) * num_cols + col_start];
        }
    }
    __syncthreads();

    int output_row_start = blockIdx.x * TILE_SIZE;
    int output_col_start = blockIdx.y * TILE_SIZE + threadIdx.x;

    int num_output_rows = num_cols;
    int num_output_cols = num_rows;

    // write transposed data to output
    for (int i = 0; i < TILE_SIZE; i += 1) {
        if (output_row_start + i >= num_output_rows || output_col_start >= num_output_cols) {
            return;
        }
        output[(output_row_start + i) * num_output_cols + output_col_start] = tile[threadIdx.x * PADDED_TILE_SIZE + i];
    }
}

int main(){
    int num_rows = 8192;
    int num_cols = 8192;

    float *h_input, *h_output;
    float *d_input, *d_output;

    size_t size = num_rows * num_cols * sizeof(float);
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Initialize input matrix
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            h_input[i * num_cols + j] = static_cast<float>(i * num_cols + j);
        }
    }

    // Copy input matrix to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    cudaFuncSetAttribute(transposeKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(float) * TILE_SIZE * PADDED_TILE_SIZE);

    // Launch kernel
    dim3 gridDim((num_cols + TILE_SIZE - 1) / TILE_SIZE, (num_rows + TILE_SIZE - 1) / TILE_SIZE);
    dim3 blockDim(TILE_SIZE);
    transposeKernel<<<gridDim, blockDim, sizeof(float) * TILE_SIZE * PADDED_TILE_SIZE>>>(d_input, d_output, num_rows, num_cols);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy output matrix back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    // Verify the result
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            if (h_output[j * num_rows + i] != h_input[i * num_cols + j]) {
                printf("Mismatch at (%d, %d): %f != %f\n", i, j, h_output[j * num_rows + i], h_input[i * num_cols + j]);
                break;
            }
        }
    }
    printf("Transpose completed successfully.\n");
    
    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}