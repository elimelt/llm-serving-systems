#include <cuda_runtime.h>
#include <iostream>

#define BLOCKSIZE 256


__global__ void add2(int *a, int *b, int *c, size_t num) {
    int4* a4 = (int4*)a;
    int4* b4 = (int4*)b;
    int4* c4 = (int4*)c;

    int block_start = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    int index = block_start + thread_id;
    if (index < num / 4) {
        int4 a4_val = a4[index];
        int4 b4_val = b4[index];
        int4 c4_val;
        c4_val.x = a4_val.x + b4_val.x;
        c4_val.y = a4_val.y + b4_val.y;
        c4_val.z = a4_val.z + b4_val.z;
        c4_val.w = a4_val.w + b4_val.w;
        c4[index] = c4_val;
    }
}


int main() {
    size_t num = 1000000000;

    int * host_a = new int[num];
    int * host_b = new int[num];
    int * host_c = new int[num];

    // Initialize host arrays
    for (int i = 0; i < num; i++) {
        host_a[i] = i;
        host_b[i] = i;
    }
    

    // Allocate memory on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, num * sizeof(int));
    cudaMalloc((void**)&d_b, num * sizeof(int));
    cudaMalloc((void**)&d_c, num * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_a, host_a, num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, host_b, num * sizeof(int), cudaMemcpyHostToDevice);

    dim3 num_block((num / 4 + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 num_threads(BLOCKSIZE);
    add2<<<num_block, num_threads>>>(d_a, d_b, d_c, num);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    

    // Copy result back to host
    cudaMemcpy(host_c, d_c, num * sizeof(int), cudaMemcpyDeviceToHost);


    for (int i = 0; i < num; i++) {
        if (host_c[i] != host_a[i] + host_b[i]) {
            std::cerr << "Error at index " << i << ": " << host_c[i] << std::endl;
            break;
        }
    }

    std::cout << "Result: " << host_c[0] << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] host_a;
    delete[] host_b;
    delete[] host_c;

    return 0;
}