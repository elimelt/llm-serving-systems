#include "copy_first_column.h"
#include <cuda_runtime.h>

void copy_first_column(float *h_A, float *d_A, int rows, int cols) {
    
    // access the first column of h_A and copy it to first cols element of h_A
    for (int i = 0; i < rows; i++) {
        h_A[i] = h_A[i * cols];
    }
    // copy the first cols element of h_A to d_A assuming everything is already allocated
    cudaMemcpy(d_A, h_A, rows * sizeof(float), cudaMemcpyHostToDevice);
}