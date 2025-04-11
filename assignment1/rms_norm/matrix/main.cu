#include<cuda_runtime.h>
#include "rms_norm_matrix.h"
#include <stdio.h>

#define ROWS 8192
#define COLS 8192
#define SIZE (ROWS * COLS)

int main() {
    float* input = new float[SIZE];
    float* weight = new float[SIZE];
    float* output = new float[SIZE];
    // Check the result

    for (int i = 0; i < SIZE; i++) {
        input[i] = static_cast<float>(i + 1);
        weight[i] = 1.0f;
    }

    rms_norm_matrix(input, weight, output, ROWS, COLS, 0.000001f);

    // for (int i = 0; i < SIZE; i++) {
    //     printf("output[%d] = %f\n", i, output[i]);
    // }

    delete[] input;
    delete[] weight;
    delete[] output;
    return 0;
}