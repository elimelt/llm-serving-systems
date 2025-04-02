#include <cuda_runtime.h>
#include <stdio.h>

__global__ void transposeKernel(float* input, float* output, int num_rows, int num_cols) {
    int row = blockIdx.x;
    int col_start = threadIdx.x * num_cols / blockDim.x;
    int col_end = (threadIdx.x + 1) * num_cols / blockDim.x;

    for (int col = col_start; col < col_end; col++) {
        if (col < num_cols) {
            output[col * num_rows + row] = input[row * num_cols + col];
        }
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

    // Launch kernel
    dim3 gridDim(num_rows);
    dim3 blockDim(128);
    transposeKernel<<<gridDim, blockDim>>>(d_input, d_output, num_rows, num_cols);
    cudaDeviceSynchronize();

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