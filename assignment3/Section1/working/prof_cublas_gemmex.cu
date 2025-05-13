#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>

double calc_flops(int N, int M, int K, float time_ms)
{
  float time_s = time_ms / 1000.0f;                                                                        // Convert milliseconds to seconds
  // For matrix multiplication C = A×B where A is [M,N], B is [N,K], and C is [M,K]
  // We perform 2*M*K*N operations (M*K output elements, each with N multiply-adds)
  return 2.0 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N) / (time_s * 1e12); // Convert to TFLOPS
}

float bench(int M = 1024, int N = 1024, int K = 1024, int n_iters = 100)
{
  // Initialize cuBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Allocate memory for matrices on host
  float *host_A = new float[M * N]; // Matrix A is M x N
  float *host_B = new float[N * K]; // Matrix B is N x K
  float *host_C = new float[M * K]; // Result C is M x K

  // Initialize host matrices
  for (int i = 0; i < M * N; i++)
  {
    host_A[i] = static_cast<float>(i % 100) / 100.0f;
  }
  for (int i = 0; i < N * K; i++)
  {
    host_B[i] = static_cast<float>(i % 100) / 100.0f;
  }

  // Allocate memory on the device
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, M * N * sizeof(float));
  cudaMalloc((void **)&d_B, N * K * sizeof(float));
  cudaMalloc((void **)&d_C, M * K * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_A, host_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, host_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

  int warm_up_count = 100;
  int profile_count = n_iters;
  size_t L2_size = 50 * 1024 * 1024;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Warm-up runs
  for (int i = 0; i < warm_up_count; ++i)
  {
    cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        M, K, N, // For C[M,K] = A[M,N] * B[N,K], dims are (M, K, N)
        &alpha,
        d_A,
        CUDA_R_32F,
        M, // lda = M for column-major (leading dimension of A)
        d_B,
        CUDA_R_32F,
        N, // ldb = N for column-major (leading dimension of B)
        &beta,
        d_C,
        CUDA_R_32F,
        M, // ldc = M for column-major (leading dimension of C)
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
  }

  std::cout << "Warm-up completed." << std::endl;

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  int *clear_l2_buffer;
  cudaMalloc(&clear_l2_buffer, L2_size);

  float total_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < profile_count; ++i)
  {
    cudaMemset(clear_l2_buffer, 0, L2_size); // Clear L2 cache
    cudaEventRecord(start);

    // Call cuBLAS GEMM
    cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        M, K, N, // For C[M,K] = A[M,N] * B[N,K], dims are (M, K, N)
        &alpha,
        d_A,
        CUDA_R_16F,
        M, // lda = M for column-major (leading dimension of A)
        d_B,
        CUDA_R_16F,
        N, // ldb = N for column-major (leading dimension of B)
        &beta,
        d_C,
        CUDA_R_32F,
        M, // ldc = M for column-major (leading dimension of C)
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;
  }

  std::cout << "Benchmark completed." << std::endl;

  float average_time = total_ms / profile_count;
  std::cout << "Matrix size: " << N << "x" << M << "x" << K
            << ", Average time: " << average_time << " ms" << std::endl;

  // Free the L2 buffer
  cudaFree(clear_l2_buffer);
  // Free CUDA events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] host_A;
  delete[] host_B;
  delete[] host_C;

  // Destroy cuBLAS handle
  cublasDestroy(handle);

  return average_time;
}

int main()
{
  // Define matrix sizes to benchmark
  std::vector<std::tuple<int, int, int>> matrix_sizes = {};
  auto Ms = std::vector<int>{128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048};
  auto NKs = std::vector<std::pair<int, int>>{
      {512, 512},
      {4096, 4096},
      {14336, 4096},
      {4096, 1024},
      {1024, 4096}};

  for (auto M : Ms)
    for (auto NK : NKs)
      matrix_sizes.push_back(std::make_tuple(M, NK.first, NK.second));  // Store as (M, N, K)

  // M: 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048
  // N, K: (512, 512)(4096, 4096)(14336, 4096)(4096, 1024)(1024, 4096)

  auto res = std::vector<float>();
  auto tflops = std::vector<double>();
  for (auto &size : matrix_sizes)
  {
    int M = std::get<0>(size);
    int N = std::get<1>(size);
    int K = std::get<2>(size);
    res.push_back(bench(M, N, K, 1000));  // Pass as M, N, K to match our desired dimensions
    double flops = calc_flops(M, N, K, res.back());  // Pass M, N, K in correct order
    tflops.push_back(flops);
  }

  // Create cublas_perf.csv
  std::ofstream file("cublas_perf.csv");
  if (file.is_open())
  {
    file << "M,N,K,library,tflops\n";  // Updated column names for clarity
    auto size_it = matrix_sizes.begin();
    for (size_t i = 0; i < res.size(); ++i, ++size_it)
    {
      file << std::get<0>(*size_it) << ","
           << std::get<1>(*size_it) << ","
           << std::get<2>(*size_it) << ","
           << "cublas" << ","
           << tflops[i] << "\n";
    }
    file.close();
  }
  else
  {
    std::cerr << "Unable to open file" << std::endl;
  }

  return 0;
}