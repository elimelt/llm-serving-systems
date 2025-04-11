#include "copy_first_column.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

#define ROWS 8192 
#define COLS 65536
#define SIZE (ROWS * COLS)

int main() {
  float *output, *input;
  cudaMallocHost((void**)&input, SIZE * sizeof(float));
  cudaMalloc((void**)&output, ROWS * sizeof(float));

  for (int i = 0; i < SIZE; i++) {
    input[i] = static_cast<float>(i + 1);
  }
  copy_first_column(input, output, ROWS, COLS);

  // // copy output from device to host
  // float *h_output = new float[ROWS];
  // cudaMemcpy(h_output, output, ROWS * sizeof(float), cudaMemcpyDeviceToHost);
  // // Print the output
  // std::cout << "Output: ";
  // for (int i = 0; i < ROWS; i++) {
  //   std::cout << h_output[i] << " ";
  // }
  // std::cout << std::endl;
  // // Free the host memory
  // delete[] h_output;

  cudaFreeHost(input);
  cudaFree(output);

  return 0;
}