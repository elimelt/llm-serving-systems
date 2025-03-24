#include <cuda_runtime.h>
#include <iostream>

#define BLOCKSIZE 256


// Kernel function to add two vectors
__global__ void add(int *a, int *b, int *c, int num) {
    int block_start = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    int index = block_start + thread_id;
    if (index < num) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int num = 10000000;

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

    dim3 num_block((num + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 num_threads(BLOCKSIZE);
    add<<<num_block, num_threads>>>(d_a, d_b, d_c, num);

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