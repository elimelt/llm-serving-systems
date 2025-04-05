// compile: nvcc path/to/rms.cu --output-file ./rms
// run: ./rms

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>

#define BLOCKSIZE 256
#define ELEMENTS_PER_THREAD 16 
#define BLOCK_PER_SM 8

inline void check_cuda_error(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

__global__ void rms_kernel(const int* __restrict__ input, double* __restrict__ blockSums, size_t n) {
    __shared__ double shared_mem[BLOCKSIZE];
    
    unsigned int tid = threadIdx.x,
                 block_start = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD,
                 stride = blockDim.x * gridDim.x;
    
    double sum = 0.0;
    
    for (auto base = block_start + tid; base < n; base += stride * ELEMENTS_PER_THREAD)
        for (auto j = 0; j < ELEMENTS_PER_THREAD; j++) {
            auto idx = base + j * blockDim.x;
            if (idx < n) {
                double val = input[idx];
                sum += val * val;
            }
        }
    
    // load partial sum into shared mem
    shared_mem[tid] = sum;
    __syncthreads();
    
    // reduce in block by halving threads each iter
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) shared_mem[tid] += shared_mem[tid + s];
        __syncthreads();
    }
    
    // base
    if (tid < 32) {
        volatile double* raw = shared_mem;
        raw[tid] += raw[tid + 32];
        raw[tid] += raw[tid + 16];
        raw[tid] += raw[tid + 8];
        raw[tid] += raw[tid + 4];
        raw[tid] += raw[tid + 2];
        raw[tid] += raw[tid + 1];
    }
    
    // write final result
    if (tid == 0) blockSums[blockIdx.x] = shared_mem[0];
}

double rms_on_cpu(const int* input, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += static_cast<double>(input[i]) * input[i];
    }
    return std::sqrt(sum / n);
}

void bench(size_t num, cudaDeviceProp deviceProp) {
  std::cout << std::endl 
            << "=== Testing with array size: " << num << " ===" 
            << std::endl;
  
  // test data filled with 0, 1, 2, ...
  std::vector<int> host_v(num);
  std::iota(host_v.begin(), host_v.end(), 0);
  
  double expected_rms = rms_on_cpu(host_v.data(), num);
  
  // allocate device memory
  int *d_v;                     // device vector (input)
  check_cuda_error(cudaMalloc(
    (void**)&d_v,               // ptr
    num * sizeof(int)           // size
  ));

  check_cuda_error(cudaMemcpy(
    d_v,                        // ptr     
    host_v.data(),              // src
    num * sizeof(int),          // size
    cudaMemcpyHostToDevice      // direction
  ));
  
  int numSMs = deviceProp.multiProcessorCount;
  int numBlocks = numSMs * BLOCK_PER_SM;
  
  double *d_blockSums;          // device array for block sums (output)
  check_cuda_error(cudaMalloc(
    (void**)&d_blockSums,       // ptr
    numBlocks * sizeof(double)  // size
  ));

  check_cuda_error(cudaMemset(
    d_blockSums,                // ptr
    0,                          // val
    numBlocks * sizeof(double)  // size
  ));
  
  // timing events
  cudaEvent_t start, stop;
  check_cuda_error(cudaEventCreate(&start));
  check_cuda_error(cudaEventCreate(&stop));
  
  // warmup
  rms_kernel<<<numBlocks, BLOCKSIZE>>>(d_v, d_blockSums, num);
  check_cuda_error(cudaGetLastError());
  check_cuda_error(cudaDeviceSynchronize());
  
  // actual benchmark
  const int num_iterations = 20;
  float total_milliseconds = 0.0f;
  for (int iter = 0; iter < num_iterations; iter++) {
      check_cuda_error(cudaEventRecord(start));
      rms_kernel<<<numBlocks, BLOCKSIZE>>>(d_v, d_blockSums, num);
      check_cuda_error(cudaEventRecord(stop));
      check_cuda_error(cudaEventSynchronize(stop));
      
      float milliseconds = 0;
      check_cuda_error(cudaEventElapsedTime(&milliseconds, start, stop));
      total_milliseconds += milliseconds;
      
      check_cuda_error(cudaGetLastError());
  }
  
  // copy result to host
  std::vector<double> host_blockSums(numBlocks);

  check_cuda_error(cudaMemcpy(
    host_blockSums.data(), 
    d_blockSums, 
    numBlocks * sizeof(double), 
    cudaMemcpyDeviceToHost
  ));
  
  // expected val
  double finalSum = 0.0;
  for (int i = 0; i < numBlocks; i++) finalSum += host_blockSums[i];
  
  double cuda_rms = std::sqrt(finalSum / num);
  
  const double eps = 1e-6;
  bool results_match = std::abs(cuda_rms - expected_rms) < eps * expected_rms;
  
  float avg_time_ms = total_milliseconds / num_iterations,
        bytes_processed = num * sizeof(int),
        bandwidth_GB_per_s = bytes_processed / (avg_time_ms * 1e-3) / 1e9;
  
  auto test = results_match ? "PASS" : "FAIL";

  std::cout << "Expected RMS: "     << expected_rms                   << std::endl;
  std::cout << "RMS result:   "     << cuda_rms                       << std::endl;
  std::cout << "test:         "     << test                           << std::endl;
  std::cout << "performance:  "                                       << std::endl;
  std::cout << "  - num blocks:   " << numBlocks                      << std::endl;
  std::cout << "  - avg time:     " << avg_time_ms        << " ms"    << std::endl;
  std::cout << "  - bandwidth:    " << bandwidth_GB_per_s << " GB/s"  << std::endl;
  
  // cleanup
  check_cuda_error(cudaEventDestroy(  start       ));
  check_cuda_error(cudaEventDestroy(  stop        ));
  check_cuda_error(cudaFree(          d_v         ));
  check_cuda_error(cudaFree(          d_blockSums ));
}

int main() {
    // if this break, try exporting CUDA_VISIBLE_DEVICES=<n>
    // e.g. export CUDA_VISIBLE_DEVICES=0
    // or export CUDA_VISIBLE_DEVICES=0,1,2,3 (you hog)
    int device = 0;
    cudaDeviceProp deviceProp;
    check_cuda_error(cudaGetDeviceProperties(&deviceProp, device));
    std::cout << "Running on: " << deviceProp.name << std::endl;
    
    for (size_t num : {10000000, 20000000, 50000000}) 
      bench(num, deviceProp);
    
    return 0;
}