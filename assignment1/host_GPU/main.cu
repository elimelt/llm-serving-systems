#include "copy_first_column.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define ROWS 8192 
#define COLS 65536
#define SIZE (ROWS * COLS)
#define NITERATIONS 10000
int main() {
    float total_time = 0.0f;
    float *output, *input;

    // Allocate host memory for input
    input = new float[SIZE];
    if (input == nullptr) {
        std::cerr << "Error allocating pinned host memory" << std::endl;
        return 1;
    }
     
    // Allocate device memory for output
    cudaError_t err = cudaMalloc((void**)&output, ROWS * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory: " << cudaGetErrorString(err) << std::endl;
        delete[] input;
        return 1;
    }
    // Initialize input data
    for (int i = 0; i < SIZE; i++) {
        input[i] = static_cast<float>((i + 1));
    }

    for (int j = 0; j < NITERATIONS; j++) {
   
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Warmup run
        copy_first_column(input, output, ROWS, COLS);
        
        // Record start event
        cudaEventRecord(start);
        
        // Execute the copy operation
        copy_first_column(input, output, ROWS, COLS);
        
        // Record stop event
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate and print execution time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        // std::cout << "Execution time: " << milliseconds * 1000.0f << " microseconds" << std::endl;
        total_time += milliseconds;
        // Verify the results
        // float *h_output = new float[ROWS];
        // err = cudaMemcpy(h_output, output, ROWS * sizeof(float), cudaMemcpyDeviceToHost);
        // if (err != cudaSuccess) {
        //     std::cerr << "Error copying from device to host: " << cudaGetErrorString(err) << std::endl;
        // } else {
        //     // Verify first few elements
        //     for (int i = 0; i < std::min(10, ROWS); i++) {
        //         float expected = input[i * COLS];
        //         if (h_output[i] != expected) {
        //             std::cerr << "Verification failed at index " << i 
        //                     << ": expected " << expected 
        //                     << ", got " << h_output[i] << std::endl;
        //             break;
        //         }
        //     }
        // }

        // Cleanup
        // delete[] h_output;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    delete[] input;
    cudaFree(output);
    std::cout << "Average execution time: " << total_time / NITERATIONS * 1000.0f << " microseconds" << std::endl;
    return 0;
}