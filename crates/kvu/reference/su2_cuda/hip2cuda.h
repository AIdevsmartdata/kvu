// hip2cuda.h — correspondance mécanique HIP -> CUDA pour le portage de su2_hip.cpp (chantier
// vortex_tests_20260927/gpu_port). Aucune sémantique changée : chaque appel HIP a son homologue
// CUDA direct ; hipLaunchKernelGGL(K, grille, bloc, shm, flux, args...) -> K<<<...>>>(args...).
#pragma once
#include <cuda_runtime.h>
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipMalloc cudaMalloc
#define hipMemset cudaMemset
#define hipMemcpy cudaMemcpy
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipFree cudaFree
#define hipEvent_t cudaEvent_t
#define hipEventCreate cudaEventCreate
#define hipEventDestroy cudaEventDestroy
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEventElapsedTime cudaEventElapsedTime
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipGetLastError cudaGetLastError
#define hipGetDeviceCount cudaGetDeviceCount
#define hipDeviceProp_t cudaDeviceProp
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipLaunchKernelGGL(K, G, B, S, ST, ...) K<<<(G), (B), (S), (ST)>>>(__VA_ARGS__)
