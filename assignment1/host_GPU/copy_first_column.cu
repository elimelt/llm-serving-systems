#include "copy_first_column.h"
#include <cuda_runtime.h>
#include <stdio.h>

// h_A and d_A are already allocated by the caller.
// h_A is assumed to be in row-major order with 'cols' elements per row.
// d_A is assumed to be pre-allocated on the device and can hold 'rows' floats.
void copy_first_column(float *h_A, float *d_A, int rows, int cols) {
    // Set up CUDA events for timing the cudaMemcpy.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start timing just before the contiguous memcpy.
    cudaEventRecord(start, 0);
    
    // Pack the first column into contiguous memory.
    // This loop should be very fast given the loop count (rows = 8192).
    for (int i = 1; i < rows; ++i) {
        // Each row is stored contiguously in h_A with a stride of 'cols'
        h_A[i] = h_A[i * cols];
    }
    
    // Now copy the contiguous column from host to device.
    cudaMemcpy(d_A, h_A, rows * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken to copy first column: %f ms\n", milliseconds);
    
    // Clean up CUDA events and temporary pinned buffer.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
