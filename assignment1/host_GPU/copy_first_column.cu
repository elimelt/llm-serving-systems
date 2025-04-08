#include "copy_first_column.h"
#include <cuda_runtime.h>
#include <stdlib.h>

void copy_first_column(float *h_A, float *d_A, int rows, int cols)
{
  cudaMalloc((void **)&d_A, rows * cols * sizeof(float));
  cudaMemcpy(d_A, h_A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

  float *h_first_column = (float *)malloc(rows * sizeof(float));

  for (int i = 0; i < rows; i++)
  {
    cudaMemcpy(&h_first_column[i], &d_A[i * cols], sizeof(float), cudaMemcpyDeviceToHost);
  }

  cudaFree(d_A);

  free(h_first_column);
}