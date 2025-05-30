#!/bin/bash

CUTLASS_PROFILER="/home/group2/cutlass/build/tools/profiler/cutlass_profiler"
mkdir -p results

# (N,K) = (512, 512)
$CUTLASS_PROFILER --operation=Gemm \
  --kernels=*_tensorop*align8* \
  --m=128:2048:128 \
  --n=512 --k=512 \
  --A=f16:column --B=f16:column --C=f16:column --D=f16:column --accum=f32 \
  --num_groups=1 \
  --split_k_slices=1,2,4,8 --split_k_mode=serial \
  --output=./results/gemm_512_512.csv
 
# (N,K) = (4096, 4096) 
$CUTLASS_PROFILER --operation=Gemm \
  --kernels=*_tensorop*align8* \
  --m=128:2048:128 \
  --n=4096 --k=4096 \
  --A=f16:column --B=f16:column --C=f16:column --D=f16:column --accum=f32 \
  --num_groups=1 \
  --split_k_slices=1,2,4,8 --split_k_mode=serial \
  --output=./results/gemm_4096_4096.csv

# (N,K) = (14336, 4096)
$CUTLASS_PROFILER --operation=Gemm \
  --kernels=*_tensorop*align8* \
  --m=128:2048:128 \
  --n=14336 --k=4096 \
  --A=f16:column --B=f16:column --C=f16:column --D=f16:column --accum=f32 \
  --num_groups=1 \
  --split_k_slices=1,2,4,8 --split_k_mode=serial \
  --output=./results/gemm_14336_4096.csv


# (N,K) = (4096, 1024)
$CUTLASS_PROFILER --operation=Gemm \
  --kernels=*_tensorop*align8* \
  --m=128:2048:128 \
  --n=4096 --k=1024 \
  --A=f16:column --B=f16:column --C=f16:column --D=f16:column --accum=f32 \
  --num_groups=1 \
  --split_k_slices=1,2,4,8 --split_k_mode=serial \
  --output=./results/gemm_4096_1024.csv


# (N,K) = (1024, 4096)
$CUTLASS_PROFILER --operation=Gemm \
  --kernels=*_tensorop*align8* \
  --m=128:2048:128 \
  --n=1024 --k=4096 \
  --A=f16:column --B=f16:column --C=f16:column --D=f16:column --accum=f32 \
  --num_groups=1 \
  --split_k_slices=1,2,4,8 --split_k_mode=serial \
  --output=./results/gemm_1024_4096.csv

