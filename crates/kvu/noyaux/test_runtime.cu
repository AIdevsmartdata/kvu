// test_runtime.cu — noyaux d'essai du runtime unifié (gates RT-1 / RT-2). Propriétaire : T2.
//
// Source UNIQUE pour HIP et CUDA : le prélude du backend fournit les mots-clés. Compilés avec
// `OptionsCompil::reference_k1()` (HIP -ffp-contract=off, CUDA --fmad=false), ces noyaux donnent au bit
// les mêmes résultats que le même calcul écrit en Rust sur l'hôte (aucune contraction a*x+y en FMA).
//
// Les deux réductions sont des COPIES des formes de su2_sweep.hip (arbre fixe de 256 fils) :
// l'ordre d'addition ne dépend ni de l'ordonnancement ni de l'architecture, donc le résultat est
// reproductible à l'octet et se rejoue exactement sur le CPU.

// x[i] = v + i·h pour i < n (remplissage de contrôle, exact tant que i < 2^53).
extern "C" __global__ void k_remplir(double* __restrict__ x, long n, double v, double h) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = v + (double)i * h;
}

// y[i] = a·x[i] + y[i] en FP64. Sans contraction, c'est exactement `a * x + y` écrit sur l'hôte.
extern "C" __global__ void k_saxpy_f64(long n, double a, const double* __restrict__ x, double* __restrict__ y) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

// Réduction déterministe par bloc (256 fils) : out[bloc] = somme des 256 éléments du bloc, en arbre fixe.
extern "C" __global__ void k_reduce(const double* __restrict__ in, long n, double* out) {
  __shared__ double sh[256];
  long i = (long)blockIdx.x * 256 + threadIdx.x;
  sh[threadIdx.x] = (i < n) ? in[i] : 0.0;
  __syncthreads();
  for (int w = 128; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = sh[0];
}

// Somme finale déterministe (1 bloc de 256) : chaque fil cumule en foulée de 256, puis arbre fixe.
extern "C" __global__ void k_final(const double* __restrict__ in, long n, double scale, double* out) {
  __shared__ double sh[256];
  double a = 0;
  for (long i = threadIdx.x; i < n; i += 256) a += in[i];
  sh[threadIdx.x] = a;
  __syncthreads();
  for (int w = 128; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = sh[0] * scale;
}

// Somme entière exacte par atomicAdd 64 bits : l'addition d'entiers est associative, le résultat est
// indépendant de l'ordre des fils (contrairement au flottant).
extern "C" __global__ void k_somme_u64(const unsigned long long* __restrict__ x, long n, unsigned long long* somme) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(somme, x[i]);
}
