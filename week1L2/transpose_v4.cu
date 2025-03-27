#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_ROW 128
#define TILE_COLUMN 128
#define TILE_COLUMN_PAD 129

__global__ void transpose(float *A, float *B, int num_row, int num_column) {
    extern __shared__ float tile[];
    int row_start = blockIdx.x * TILE_ROW + threadIdx.y;
    int col_start = blockIdx.y * TILE_COLUMN + threadIdx.x;
    for (int i = 0; i < TILE_ROW; i += 1) {
        tile[i* TILE_COLUMN_PAD + threadIdx.x] = A[(row_start + i) * num_column + col_start];
    }
    __syncthreads();
    row_start = blockIdx.y * TILE_COLUMN + threadIdx.y;
    col_start = blockIdx.x * TILE_ROW + threadIdx.x;

    for (int i = 0; i < TILE_COLUMN; i += 1) {
        B[(row_start + i) * num_row + col_start ] = tile [threadIdx.x * TILE_COLUMN_PAD + i];
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
    // config shared memory
    cudaFuncSetAttribute(
        transpose, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        98304  // 96 KB
    );
    dim3 grid((num_row + TILE_ROW - 1) / TILE_ROW, (num_column + TILE_COLUMN - 1) / TILE_COLUMN);
    dim3 block(TILE_COLUMN);
    transpose<<<grid, block, 98304>>>(device_A, device_B, num_row, num_column);
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