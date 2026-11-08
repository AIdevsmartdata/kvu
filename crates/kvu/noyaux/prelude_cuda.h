// prelude_cuda.h — prélude CUDA (NVRTC), inséré en tête de chaque source assemblée pour le backend CUDA.
// Propriétaire : T2. Aucun #include : NVRTC fournit lui-même __global__, __shared__, atomicAdd et
// sqrt/log/cos en double et en float. La macro KV_CUDA est le pendant de KV_HIP.
// Les noyaux restent `extern "C"` (recherche par nom avec cuModuleGetFunction).
#define KV_CUDA 1
