#include <cuda_runtime.h>
#include <stdio.h>

__global__ void transpose(float *A, float *B, int num_row, int num_column) {
    int row = blockIdx.x;
    int col_start = threadIdx.x;
    for (int i = col_start; i < num_column; i += blockDim.x) {
        B[i * num_row + row] = A[row * num_column + i];
    }
}

int main(){
    int num_row = 8192;
    int num_column = 8192;
    float * host_A, * host_B;
    float * device_A, * device_B;

    host_A = (float *)malloc(num_row * num_column * sizeof(float));
    host_B = (float *)malloc(num_row * num_column * sizeof(float));
    cudaMalloc((void**)&device_A, num_row * num_column * sizeof(float));
    cudaMalloc((void**)&device_B, num_row * num_column * sizeof(float));

    // Initialize host_A with some values
    for (int i = 0; i < num_row * num_column; i++) {
        host_A[i] = static_cast<float>(i);
    }
    // Copy host_A to device_A
    cudaMemcpy(device_A, host_A, num_row * num_column * sizeof(float), cudaMemcpyHostToDevice);
    // Launch kernel to transpose matrix
    dim3 grid(num_row);
    dim3 block(128);
    transpose<<<grid, block>>>(device_A, device_B, num_row, num_column);
    // Copy transposed matrix back to host_B
    cudaMemcpy(host_B, device_B, num_row * num_column * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_A);
    cudaFree(device_B);

    // check the result
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_column; j++) {
            if (host_B[i * num_column + j] != host_A[j * num_row + i]) {
                printf("Error at (%d, %d): %f != %f\n", i, j, host_B[i * num_column + j], host_A[j * num_row + i]);
                return -1;
            }
        }
    }
    printf("Transpose successful!\n");
}