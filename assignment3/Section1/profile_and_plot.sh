#!/bin/bash

# Simple profiling and plotting script for one GEMM shape

# Example shape (edit as needed)
B=1
M=128
N=1024
K=4096

CSV_FILE="gemm_perf.csv"
echo "batch_size,N,K,library,tflops" > $CSV_FILE

# Run CUBLAS profiling (edit the binary and arguments as needed)
# This should output a TFLOPS value
# CUBLAS_TFLOPS=$(./assignment3/Section1/prof_example_cublas $B $M $N $K | grep "TFLOPS" | awk '{print $NF}')
# echo "$B,$N,$K,cublas,$CUBLAS_TFLOPS" >> $CSV_FILE

# Run CUTLASS profiling (edit the binary and arguments as needed)
CUTLASS_PROFILER=/home/group2/cutlass/build/tools/profiler/cutlass_profiler
CUTLASS_TFLOPS=$($CUTLASS_PROFILER --m=$M --n=$N --k=$K --split_k_slices=1 --split_k_mode=serial --profiling-iterations=100 2>/dev/null | grep "GFLOPs" | awk '{print $NF/1000}' | sort -nr | head -1)
echo "$B,$N,$K,cutlass,$CUTLASS_TFLOPS" >> $CSV_FILE

# # Plot
# python3 assignment3/Section1/plot_gemm.py

# echo "Done! Check gemm_perf.csv and the generated plots." 