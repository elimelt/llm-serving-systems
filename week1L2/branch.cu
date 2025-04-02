#include <cuda_runtime.h>

__global__ void kernel() {
    int tid = threadIdx.x;
    int a;
    if (tid < 32) {
        a = 1;
    } else if (tid < 64) {
        a = 2;
    } else if (tid < 96) {
        a = 3;
    } else {
        a = 4;
    }
}

__global__ void kernel2() {
    int tid = threadIdx.x;
    int a;
    if (tid % 2 == 0) {
        a = 1;
    } else {
        a = 2;
    }
}

int main() {
    return 0;
}
