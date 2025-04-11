#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>

int main() {
    std::ofstream outfile("transfer_speed.csv");
    outfile << "Size_Bytes,Pageable_HtoD_MBps,Pageable_DtoH_MBps,Pinned_HtoD_MBps,Pinned_DtoH_MBps\n";

    const int iterations = 100;

    // Loop over data sizes: 2^0 to 2^10 bytes
    for (int exp = 0; exp <= 10; ++exp) {
        size_t size = 1 << exp;  // Data size in bytes

        // Allocate pageable host memory (using new)
        char* h_pageable = new char[size];
        for (size_t i = 0; i < size; ++i) {
            h_pageable[i] = static_cast<char>(i % 256);
        }

        // Allocate pinned host memory (using cudaMallocHost)
        char* h_pinned = nullptr;
        cudaMallocHost((void**)&h_pinned, size);
        for (size_t i = 0; i < size; ++i) {
            h_pinned[i] = static_cast<char>(i % 256);
        }

        // Allocate device memory
        char* d_data = nullptr;
        cudaMalloc((void**)&d_data, size);

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float t;
        double total_pageable_htod = 0.0, total_pageable_dtoh = 0.0;
        double total_pinned_htod = 0.0, total_pinned_dtoh = 0.0;

        // Run 10 iterations per transfer size
        for (int iter = 0; iter < iterations; ++iter) {
            // Pageable Host-to-Device
            cudaEventRecord(start);
            cudaMemcpy(d_data, h_pageable, size, cudaMemcpyHostToDevice);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&t, start, stop);
            total_pageable_htod += t;

            // Pageable Device-to-Host
            cudaEventRecord(start);
            cudaMemcpy(h_pageable, d_data, size, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&t, start, stop);
            total_pageable_dtoh += t;

            // Pinned Host-to-Device
            cudaEventRecord(start);
            cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&t, start, stop);
            total_pinned_htod += t;

            // Pinned Device-to-Host
            cudaEventRecord(start);
            cudaMemcpy(h_pinned, d_data, size, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&t, start, stop);
            total_pinned_dtoh += t;
        }

        // Average times (in milliseconds)
        double avg_pageable_htod = total_pageable_htod / iterations;
        double avg_pageable_dtoh = total_pageable_dtoh / iterations;
        double avg_pinned_htod   = total_pinned_htod / iterations;
        double avg_pinned_dtoh   = total_pinned_dtoh / iterations;

        // Calculate bandwidth in MB/s
        double pageable_htod_bandwidth = (size / (avg_pageable_htod * 1e-3)) / (1024.0 * 1024.0);
        double pageable_dtoh_bandwidth = (size / (avg_pageable_dtoh * 1e-3)) / (1024.0 * 1024.0);
        double pinned_htod_bandwidth   = (size / (avg_pinned_htod * 1e-3))   / (1024.0 * 1024.0);
        double pinned_dtoh_bandwidth   = (size / (avg_pinned_dtoh * 1e-3))   / (1024.0 * 1024.0);

        // Output the averaged results
        outfile << size << "," 
                << std::fixed << std::setprecision(2) << pageable_htod_bandwidth << ","
                << std::fixed << std::setprecision(2) << pageable_dtoh_bandwidth << ","
                << std::fixed << std::setprecision(2) << pinned_htod_bandwidth << ","
                << std::fixed << std::setprecision(2) << pinned_dtoh_bandwidth << "\n";

        // Clean up for this data size
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_data);
        delete[] h_pageable;
        cudaFreeHost(h_pinned);
    }

    outfile.close();
    std::cout << "Measurement complete. Results written to transfer_speed.csv" << std::endl;
    return 0;
}
