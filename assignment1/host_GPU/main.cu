#include "copy_first_column.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

inline float calculate_max_bandwidth(cudaDeviceProp prop)
{
  return static_cast<float>(prop.memoryClockRate) * 1000.0f * (static_cast<float>(prop.memoryBusWidth) / 8.0f) * 2.0f / 1.0e9f;
}

int main()
{
  std::vector<int> sizes = {1024, 4096, 16384, 65536};
  std::vector<int> widths = {256, 1024, 4096};

  std::cout << "Rows\tCols\tTime(ms)\tBW(GB/s)" << std::endl;

  double peak_bw = 0.0;

  for (auto rows : sizes)
  {
    for (auto cols : widths)
    {
      float *h_A = (float *)malloc(rows * cols * sizeof(float));
      float *d_A = nullptr;

      for (int i = 0; i < rows; i++)
      {
        for (int j = 0; j < cols; j++)
        {
          h_A[i * cols + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
      }

      auto start = std::chrono::high_resolution_clock::now();

      copy_first_column(h_A, d_A, rows, cols);

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> duration = end - start;

      double time_ms = duration.count();
      double bytes = static_cast<double>((rows * cols * sizeof(float)) + (rows * sizeof(float)));
      double bw = (bytes / (time_ms * 1.0e-3)) / 1.0e9;

      std::cout << rows << "\t" << cols << "\t" << time_ms << "\t" << bw << std::endl;

      if (bw > peak_bw)
      {
        peak_bw = bw;
      }

      free(h_A);
    }
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  double max_bw = calculate_max_bandwidth(prop);

  std::cout << "Device: " << prop.name << std::endl;
  std::cout << "Peak Bandwidth: " << peak_bw << " GB/s" << std::endl;
  std::cout << "Theoretical Bandwidth: " << max_bw << " GB/s" << std::endl;
  std::cout << "Efficiency: " << (peak_bw / max_bw) * 100.0 << "%" << std::endl;

  return 0;
}