// kevinotron/src/hip/su2_sweep.hip — lot K1 (PLAN_V3, 26/09/2026)
//
// Port HIP (hiprtc, compilation a l'execution) du generateur SU(2) autonome
//   /chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926/src/su2_hip.cpp
//   sha256 ed1e0fce255cb15d640ab603b3af0b6c50075bbb65252ce5d7e351abd3061e30 (lecture seule).
//
// REGLE DU PORT : le code device (Philox4x32-10, u01, algebre quaternionique, geometrie,
// agrafe, sample_x0, bain thermique, overrelaxation, plaquette, reductions, Polyakov) est
// recopie CARACTERE POUR CARACTERE depuis su2_hip.cpp (memes expressions, meme ordre
// d'evaluation, memes appels log/cos/sin/exp/sqrt/fmax). Seuls changements structurels :
//   (1) les noyaux sont `extern "C"` (lookup par nom, invariant A6 de wrap.rs) ;
//   (2) la struct Geo passee par valeur est remplacee par l'entier L, Geo etant reconstruite
//       sur le device par mkgeo(L) avec EXACTEMENT les formules entieres de ctx_init ;
//   (3) M_PI est defini ici si hiprtc ne le fournit pas (meme litteral que glibc) ;
//   (4) k_heatbath recoit un pointeur de diagnostic facultatif `diag_ntry` (nul en
//       production) : nombre d'essais KP/Creutz par lien, pour compter les decisions
//       divergentes CPU/HIP (gate 3) ; aucune operation flottante n'en depend ;
//   (5) mutant de controle negatif (gate 6), compile SEULEMENT si K1_MUTANT_METRO est
//       defini (jamais en production) : tirage de x0 par UN pas de Metropolis naif
//       (proposition uniforme sur [-1,1]) au lieu de Kennedy-Pendleton/Creutz.
// Politique de compilation : -O3 -ffp-contract=off (piege P3), --offload-arch explicite (A2).
// L'identite a l'octet est jugee contre su2_hip.cpp compile par hipcc avec
// -O3 -ffp-contract=off (gate 1, PLAN_V3 §K1).

#include <hip/hip_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef unsigned int u32;
typedef unsigned long long u64;

// ------------------------------------------------------------------------------------------
// Philox4x32-10 (Salmon et al. 2011, Random123)
// ------------------------------------------------------------------------------------------
struct U4 { u32 v[4]; };

__device__ inline U4 philox4x32_10(U4 c, u32 k0, u32 k1) {
  const u32 M0 = 0xD2511F53u, M1 = 0xCD9E8D57u, W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
  for (int r = 0; r < 10; ++r) {
    u64 p0 = (u64)M0 * (u64)c.v[0];
    u64 p1 = (u64)M1 * (u64)c.v[2];
    u32 hi0 = (u32)(p0 >> 32), lo0 = (u32)p0;
    u32 hi1 = (u32)(p1 >> 32), lo1 = (u32)p1;
    U4 n;
    n.v[0] = hi1 ^ c.v[1] ^ k0;
    n.v[1] = lo1;
    n.v[2] = hi0 ^ c.v[3] ^ k1;
    n.v[3] = lo0;
    c = n;
    k0 += W0; k1 += W1;
  }
  return c;
}

__device__ inline double u01(u32 a, u32 b) {
  u64 m = ((u64)(a >> 6) << 26) | (u64)(b >> 6);          // 26 + 26 = 52 bits
  return ((double)m + 0.5) * (1.0 / 4503599627370496.0);
}

enum : u32 { TAG_INIT = 0x494E4954u, TAG_HB = 0x48424154u };

// ------------------------------------------------------------------------------------------
// Algebre quaternionique (x=a0, y=a1, z=a2, w=a3)
// ------------------------------------------------------------------------------------------
struct Q { double x, y, z, w; };

__device__ inline Q qmul(Q a, Q b) {
  Q c;
  c.x = a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w;
  c.y = a.x * b.y + b.x * a.y - (a.z * b.w - a.w * b.z);
  c.z = a.x * b.z + b.x * a.z - (a.w * b.y - a.y * b.w);
  c.w = a.x * b.w + b.x * a.w - (a.y * b.z - a.z * b.y);
  return c;
}
__device__ inline Q qdag(Q a) { Q c = {a.x, -a.y, -a.z, -a.w}; return c; }
__device__ inline Q qadd(Q a, Q b) { Q c = {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; return c; }
__device__ inline double qnorm(Q a) { return sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w); }
__device__ inline Q qscale(Q a, double s) { Q c = {a.x * s, a.y * s, a.z * s, a.w * s}; return c; }

// ------------------------------------------------------------------------------------------
// Geometrie
// ------------------------------------------------------------------------------------------
struct Geo { int L; long V; long st[4]; };

// Reconstruction device de Geo : memes formules entieres que ctx_init de su2_hip.cpp.
__device__ inline Geo mkgeo(int L) {
  Geo g;
  g.L = L; g.V = (long)L * L * L * L;
  g.st[3] = 1; g.st[2] = L; g.st[1] = (long)L * L; g.st[0] = (long)L * L * L;
  return g;
}

__device__ inline long nb_up(const Geo& g, long s, int d, int xd) {
  return (xd == g.L - 1) ? s - (long)(g.L - 1) * g.st[d] : s + g.st[d];
}
__device__ inline long nb_dn(const Geo& g, long s, int d, int xd) {
  return (xd == 0) ? s + (long)(g.L - 1) * g.st[d] : s - g.st[d];
}
__device__ inline void coords(const Geo& g, long s, int x[4]) {
  x[3] = (int)(s % g.L); long t = s / g.L;
  x[2] = (int)(t % g.L); t /= g.L;
  x[1] = (int)(t % g.L); x[0] = (int)(t / g.L);
}
__device__ inline Q ld(const Q* __restrict__ U, const Geo& g, int mu, long s) { return U[(long)mu * g.V + s]; }

__device__ Q staple(const Q* __restrict__ U, const Geo& g, int mu, long s, const int x[4]) {
  Q A = {0, 0, 0, 0};
  long smu = nb_up(g, s, mu, x[mu]);
  for (int nu = 0; nu < 4; ++nu) {
    if (nu == mu) continue;
    long snu = nb_up(g, s, nu, x[nu]);
    long sdn = nb_dn(g, s, nu, x[nu]);
    long smu_dn = nb_dn(g, smu, nu, x[nu]);
    Q t = qmul(qmul(ld(U, g, nu, smu), qdag(ld(U, g, mu, snu))), qdag(ld(U, g, nu, s)));
    Q b = qmul(qmul(qdag(ld(U, g, nu, smu_dn)), qdag(ld(U, g, mu, sdn))), ld(U, g, nu, sdn));
    A = qadd(A, qadd(t, b));
  }
  return A;
}

__device__ inline long site_of(const Geo& g, long t, int parity, int x[4]) {
  long s = 2 * t;
  coords(g, s, x);
  int p = (x[0] + x[1] + x[2] + x[3]) & 1;
  if (p != parity) { s += 1; x[3] += 1; }
  return s;
}

// ------------------------------------------------------------------------------------------
// Noyaux
// ------------------------------------------------------------------------------------------
extern "C" __global__ void k_init(Q* U, int Lg, int hot, u32 k0, u32 k1) {
  Geo g = mkgeo(Lg);
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 4 * g.V) return;
  if (!hot) { Q e = {1, 0, 0, 0}; U[i] = e; return; }
  U4 c1 = {{(u32)i, 0xFFFFFFFFu, 0u, TAG_INIT}}, c2 = {{(u32)i, 0xFFFFFFFFu, 1u, TAG_INIT}};
  U4 r1 = philox4x32_10(c1, k0, k1), r2 = philox4x32_10(c2, k0, k1);
  double u1 = u01(r1.v[0], r1.v[1]), u2 = u01(r1.v[2], r1.v[3]);
  double u3 = u01(r2.v[0], r2.v[1]), u4 = u01(r2.v[2], r2.v[3]);
  double ra = sqrt(-2.0 * log(u1)), rb = sqrt(-2.0 * log(u3));
  Q q = {ra * cos(2 * M_PI * u2), ra * sin(2 * M_PI * u2), rb * cos(2 * M_PI * u4), rb * sin(2 * M_PI * u4)};
  U[i] = qscale(q, 1.0 / qnorm(q));   // Haar sur S^3
}

__device__ double sample_x0(double alpha, u32 li, u32 sweep_id, u32& draw, u32 k0, u32 k1, u32& ntry, bool& ok) {
  const u32 CAP = 1000000u;
  double x0 = 1.0; ok = false; ntry = 0;
  if (alpha >= 2.0) {
    while (ntry < CAP) {
      U4 r1 = philox4x32_10(U4{{li, sweep_id, draw++, TAG_HB}}, k0, k1);
      U4 r2 = philox4x32_10(U4{{li, sweep_id, draw++, TAG_HB}}, k0, k1);
      double a1 = u01(r1.v[0], r1.v[1]), a2 = u01(r1.v[2], r1.v[3]);
      double a3 = u01(r2.v[0], r2.v[1]), a4 = u01(r2.v[2], r2.v[3]);
      double c = cos(2 * M_PI * a2);
      double lam2 = -(log(a1) + c * c * log(a3)) / (2.0 * alpha);
      ++ntry;
      if (a4 * a4 <= 1.0 - lam2) { x0 = 1.0 - 2.0 * lam2; ok = true; break; }
    }
  } else {
    double e2 = exp(-2.0 * alpha);
    while (ntry < CAP) {
      U4 r1 = philox4x32_10(U4{{li, sweep_id, draw++, TAG_HB}}, k0, k1);
      double a1 = u01(r1.v[0], r1.v[1]), a2 = u01(r1.v[2], r1.v[3]);
      double y = 1.0 + log(e2 + a1 * (1.0 - e2)) / alpha;
      ++ntry;
      if (a2 * a2 <= 1.0 - y * y) { x0 = y; ok = true; break; }
    }
  }
  return x0;
}

#ifdef K1_MUTANT_METRO
// MUTANT (controle negatif, jamais en production) : UN pas de Metropolis naif sur x0 depuis
// x_cur, proposition uniforme sur [-1,1], rapport sqrt(1-y^2) exp(alpha y) / idem(x_cur).
// Chaine de Markov encore valide (bilan detaille), mais le tirage n'est PAS exact.
__device__ double sample_x0_mut(double alpha, double x_cur, u32 li, u32 sweep_id, u32& draw, u32 k0, u32 k1,
                                u32& ntry, bool& ok) {
  U4 r1 = philox4x32_10(U4{{li, sweep_id, draw++, TAG_HB}}, k0, k1);
  double y = 2.0 * u01(r1.v[0], r1.v[1]) - 1.0;
  double pc = sqrt(fmax(0.0, 1.0 - x_cur * x_cur));
  double py = sqrt(fmax(0.0, 1.0 - y * y)) * exp(alpha * (y - x_cur));
  ntry = 1; ok = true;
  return (u01(r1.v[2], r1.v[3]) * pc <= py) ? y : x_cur;
}
#endif

extern "C" __global__ void k_test_x0(double alpha, long n, u32 k0, u32 k1, double* out, unsigned long long* n_cap) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u32 draw = 0, ntry = 0; bool ok = false;
#ifdef K1_MUTANT_METRO
  double x0 = sample_x0_mut(alpha, 1.0, (u32)i, 0x7E57u, draw, k0, k1, ntry, ok);
#else
  double x0 = sample_x0(alpha, (u32)i, 0x7E57u, draw, k0, k1, ntry, ok);
#endif
  if (!ok) atomicAdd(n_cap, 1ull);
  out[i] = x0;
}

extern "C" __global__ void k_heatbath(Q* U, int Lg, int mu, int parity, double beta, u32 k0, u32 k1, u32 sweep_id,
                                      unsigned long long* n_cap, unsigned long long* n_tries, u32* diag_ntry) {
  Geo g = mkgeo(Lg);
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= g.V / 2) return;
  int x[4];
  long s = site_of(g, t, parity, x);
  long li = (long)mu * g.V + s;
  Q A = staple(U, g, mu, s, x);
  double k = qnorm(A);
  u32 draw = 0;
  Q W;
  if (!(k > 1e-300)) {   // agrafe nulle : mesure de Haar (jamais observe en pratique)
    U4 r1 = philox4x32_10(U4{{(u32)li, sweep_id, draw++, TAG_HB}}, k0, k1);
    U4 r2 = philox4x32_10(U4{{(u32)li, sweep_id, draw++, TAG_HB}}, k0, k1);
    double u1 = u01(r1.v[0], r1.v[1]), u2 = u01(r1.v[2], r1.v[3]);
    double u3 = u01(r2.v[0], r2.v[1]), u4 = u01(r2.v[2], r2.v[3]);
    double ra = sqrt(-2.0 * log(u1)), rb = sqrt(-2.0 * log(u3));
    Q q = {ra * cos(2 * M_PI * u2), ra * sin(2 * M_PI * u2), rb * cos(2 * M_PI * u4), rb * sin(2 * M_PI * u4)};
    U[li] = qscale(q, 1.0 / qnorm(q));
    if (diag_ntry) diag_ntry[li] = 0xFFFFFFFFu;
    return;
  }
  Q Vv = qscale(A, 1.0 / k);
  double alpha = beta * k;          // densite de x0 : sqrt(1-x0^2) exp(alpha x0)
  u32 ntry = 0; bool ok = false;
#ifdef K1_MUTANT_METRO
  double x0 = sample_x0_mut(alpha, qmul(U[li], Vv).x, (u32)li, sweep_id, draw, k0, k1, ntry, ok);
#else
  double x0 = sample_x0(alpha, (u32)li, sweep_id, draw, k0, k1, ntry, ok);
#endif
  atomicAdd(n_tries, (unsigned long long)ntry);
  if (diag_ntry) diag_ntry[li] = ok ? ntry : 0u;
  if (!ok) { atomicAdd(n_cap, 1ull); return; }   // compte ; lien inchange (attendu : 0)
  U4 rd = philox4x32_10(U4{{(u32)li, sweep_id, draw++, TAG_HB}}, k0, k1);
  double ct = 2.0 * u01(rd.v[0], rd.v[1]) - 1.0;
  double ph = 2 * M_PI * u01(rd.v[2], rd.v[3]);
  double st = sqrt(fmax(0.0, 1.0 - ct * ct));
  double rr = sqrt(fmax(0.0, 1.0 - x0 * x0));
  W.x = x0; W.y = rr * st * cos(ph); W.z = rr * st * sin(ph); W.w = rr * ct;
  Q Un = qmul(W, qdag(Vv));          // X = U V suit exp(alpha x0) dHaar  =>  U = X V^+
  U[li] = qscale(Un, 1.0 / qnorm(Un));
}

extern "C" __global__ void k_overrelax(Q* U, int Lg, int mu, int parity) {
  Geo g = mkgeo(Lg);
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= g.V / 2) return;
  int x[4];
  long s = site_of(g, t, parity, x);
  long li = (long)mu * g.V + s;
  Q A = staple(U, g, mu, s, x);
  double k = qnorm(A);
  if (!(k > 1e-300)) return;
  Q Vd = qdag(qscale(A, 1.0 / k));
  Q Un = qmul(qmul(Vd, qdag(U[li])), Vd);   // U' = V^+ U^+ V^+  (ReTr(U'A) = ReTr(UA))
  U[li] = qscale(Un, 1.0 / qnorm(Un));
}

// somme par site des 6 plaquettes 1/2 ReTr U_p
extern "C" __global__ void k_plaq_site(const Q* __restrict__ U, int Lg, double* out) {
  Geo g = mkgeo(Lg);
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= g.V) return;
  int x[4]; coords(g, s, x);
  double acc = 0;
  for (int mu = 0; mu < 4; ++mu) {
    long smu = nb_up(g, s, mu, x[mu]);
    for (int nu = mu + 1; nu < 4; ++nu) {
      long snu = nb_up(g, s, nu, x[nu]);
      Q p = qmul(qmul(ld(U, g, mu, s), ld(U, g, nu, smu)), qmul(qdag(ld(U, g, mu, snu)), qdag(ld(U, g, nu, s))));
      acc += p.x;
    }
  }
  out[s] = acc;
}

// reduction deterministe par bloc (256 threads)
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

// Somme finale deterministe (1 bloc de 256).
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

// Boucles de Polyakov 1/2 Tr prod_{k} U_mu(x + k e_mu), pour chaque mu et chaque site transverse.
extern "C" __global__ void k_polyakov(const Q* __restrict__ U, int Lg, double* out) {
  Geo g = mkgeo(Lg);
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long L3 = g.V / g.L;
  if (i >= 4 * L3) return;
  int mu = (int)(i / L3);
  long j = i % L3;
  int x[4]; x[mu] = 0;
  int dirs[3], nd = 0;
  for (int d = 0; d < 4; ++d) if (d != mu) dirs[nd++] = d;
  x[dirs[2]] = (int)(j % g.L); long t = j / g.L;
  x[dirs[1]] = (int)(t % g.L); x[dirs[0]] = (int)(t / g.L);
  long s = ((long)(x[0] * g.L + x[1]) * g.L + x[2]) * g.L + x[3];
  Q P = {1, 0, 0, 0};
  for (int k = 0; k < g.L; ++k) { P = qmul(P, ld(U, g, mu, s)); s += (k == g.L - 1) ? 0 : g.st[mu]; }
  out[i] = P.x;
}
