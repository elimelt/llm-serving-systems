#include <cuda_runtime.h>
#include "rms_norm_vector.h"
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

#define MIN_SIZE 1024
#define MAX_SIZE 268435456
#define ITERATIONS 10
#define WARMUP_ITERATIONS 3
#define EPSILON 1e-5f

int main() {
    std::vector<int> sizes;
    
    for (int size = MIN_SIZE; size <= MAX_SIZE; size *= 4) {
        sizes.push_back(size);
    }
    
    std::cout << "Vector Size\tTime (ms)\tBandwidth (GB/s)" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    
    float max_bandwidth = 0.0f;
    int best_size = 0;
    
    for (int size : sizes) {
        float *input = (float*)malloc(size * sizeof(float));
        float *weight = (float*)malloc(size * sizeof(float));
        float *output = (float*)malloc(size * sizeof(float));
        
        for (int i = 0; i < size; i++) {
            input[i] = (float)(rand()) / RAND_MAX * 2.0f - 1.0f;
            weight[i] = (float)(rand()) / RAND_MAX * 2.0f - 1.0f;
        }
        
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            rms_norm_vector(input, weight, output, size, EPSILON);
        }
        
        cudaDeviceSynchronize();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < ITERATIONS; i++) {
            rms_norm_vector(input, weight, output, size, EPSILON);
        }
        
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> duration = end - start;
        double time_ms = duration.count() / ITERATIONS;
        
        double bytes_transferred = size * sizeof(float) * 3;
        double bandwidth = (bytes_transferred / (time_ms * 1.0e-3)) / 1.0e9;
        
        std::cout << std::setw(10) << size << "\t"
                  << std::fixed << std::setprecision(3) << std::setw(9) << time_ms << "\t"
                  << std::fixed << std::setprecision(2) << std::setw(9) << bandwidth << std::endl;
        
        if (bandwidth > max_bandwidth) {
            max_bandwidth = bandwidth;
            best_size = size;
        }
        
        free(input);
        free(weight);
        free(output);
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    double theoretical_bandwidth = prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) * 2.0 / 1.0e9;
    
    std::cout << "\nDevice: " << prop.name << std::endl;
    std::cout << "Memory Clock: " << prop.memoryClockRate << " kHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Theoretical Bandwidth: " << std::fixed << std::setprecision(2) << theoretical_bandwidth << " GB/s" << std::endl;
    std::cout << "Peak Measured Bandwidth: " << std::fixed << std::setprecision(2) << max_bandwidth << " GB/s" << std::endl;
    std::cout << "Efficiency: " << std::fixed << std::setprecision(2) << (max_bandwidth / theoretical_bandwidth * 100.0) << "%" << std::endl;
    std::cout << "Best Size: " << best_size << " elements" << std::endl;
    
    return 0;
}