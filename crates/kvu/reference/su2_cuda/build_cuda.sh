#!/usr/bin/env bash
# Compilation sur le gamer (dans ~/ym4_calcul/su2_cuda/). nvcc 13.3, Blackwell sm_120.
set -eu
cd "$(dirname "$0")"
mkdir -p bin logs
NVCC=/usr/local/cuda-13.3/bin/nvcc
$NVCC --version > logs/build_nvcc_version.txt 2>&1
nice -n 5 taskset -c 12-19 $NVCC -arch=sm_120 -O3 -std=c++17 -o bin/su2_cuda su2_cuda.cu > logs/build.log 2>&1
sha256sum su2_cuda.cu hip2cuda.h bin/su2_cuda > bin/SHA256SUMS
echo "build OK"
