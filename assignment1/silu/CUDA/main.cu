#include <cuda_runtime.h>
#include "silu.h"
#include <iostream>

// #define SIZE (8192 * 8192)
#define SIZE (8192 * 8192 * 4) // 4 bytes per float

int main() {
    float* input = new float[SIZE];
    float* output = new float[SIZE];
    // Check the result

    for (int i = 0; i < SIZE; i++) {
        input[i] = static_cast<float>(i);
    }

    silu(input, output, SIZE);

    delete[] input;
    delete[] output;
    return 0;
}