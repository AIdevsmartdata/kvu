// su2_cuda.cu — PORTAGE MÉCANIQUE de su2_hip.cpp (sha256 ed1e0fce…1e30) vers CUDA (sm_120).
// Généré par port.sh ; diff complet : port.diff. Algorithme, RNG et formats de sortie inchangés.
// su2_hip.cpp — Générateur d'ensembles SU(2) Wilson 4D sur GPU AMD (HIP, gfx1151).
//
// Chantier wilson_gpu_grandL_20260926. Écrit parce que kevinotron n'a AUCUNE voie SU(2) sur GPU
// (vérifié : --hip-sweep « group 'su2' has no HIP sweep driver » ; binaire v8 sans HIP).
//
// Action : S = beta * sum_p (1 - 1/2 ReTr U_p)   (identique à su2_pairs d'Astra).
// Représentation : U = a0 + i (a1 s1 + a2 s2 + a3 s3), quaternion unitaire (a0,a1,a2,a3), FP64.
//   Matrice : [[a0 + i a3, a2 + i a1], [-a2 + i a1, a0 - i a3]]  (même convention que
//   wilson_r.haar_field).
// Réseau : L^4 périodique, site s = ((x0*L + x1)*L + x2)*L + x3 ; lien U_mu(x) de x vers x+e_mu
//   (e_mu incrémente la coordonnée x_mu) ; direction temporelle = 3 (réseau isotrope).
// Sortie NPY : complex128 C-order (mu, x0, x1, x2, x3, i, j) — même disposition que
//   io::dump_config_npy de kevinotron / su2_pairs.
// Pas de Monte-Carlo : 1 sweep de bain thermique (Kennedy–Pendleton si alpha=beta*k >= 2,
//   Creutz sinon ; les deux exacts) + N_OR sweeps d'overrelaxation U -> V^+ U^+ V^+,
//   mise à jour en damier (4 directions x 2 parités ; les liens U_mu(x) de même parité de x
//   n'interagissent pas).
// RNG : Philox4x32-10 à compteur (clé = graine 64 bits, compteur = (lien, sweep, tirage, étiquette)).
//   Aucun état partagé : la chaîne est bit-reproductible quel que soit l'ordonnancement GPU.
// Réductions (plaquette, Polyakov) : arbre fixe par bloc + somme hôte séquentielle => déterministes.
//
// Usage :
//   su2_hip --L 16 --beta 2.5 --seed 123 --start hot|cold --therm 3000 --nmeas 2000 --skip 1
//           --nor 4 --dump-every 100 --outdir DIR [--dump-initial] [--source-hash H]
//   su2_hip --selftest            (tests unitaires hôte : Philox KAT, produit quaternion)
//   su2_hip --bench --L 16 --beta 2.5 --nsweeps 200   (chronométrage seul)

#include "hip2cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <sys/stat.h>

#define HIPCHK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  fprintf(stderr, "ERREUR CUDA %s:%d : %s (%s)\n", __FILE__, __LINE__, hipGetErrorString(e_), #x); \
  exit(3); } } while (0)

typedef uint32_t u32;
typedef uint64_t u64;

// ------------------------------------------------------------------------------------------
// Philox4x32-10 (Salmon et al. 2011, Random123)
// ------------------------------------------------------------------------------------------
struct U4 { u32 v[4]; };

__host__ __device__ inline U4 philox4x32_10(U4 c, u32 k0, u32 k1) {
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

// Deux u32 -> double uniforme dans ]0,1[ (52 bits, décalé d'un demi-pas : jamais 0 ni 1,
// (m+0.5)*2^-52 exactement représentable ; max = 1 - 2^-53 < 1).
__host__ __device__ inline double u01(u32 a, u32 b) {
  u64 m = ((u64)(a >> 6) << 26) | (u64)(b >> 6);          // 26 + 26 = 52 bits
  return ((double)m + 0.5) * (1.0 / 4503599627370496.0);
}

enum : u32 { TAG_INIT = 0x494E4954u, TAG_HB = 0x48424154u };

// ------------------------------------------------------------------------------------------
// Algèbre quaternionique (x=a0, y=a1, z=a2, w=a3)
// ------------------------------------------------------------------------------------------
struct Q { double x, y, z, w; };

__host__ __device__ inline Q qmul(Q a, Q b) {
  // (a0 + i a.s)(b0 + i b.s) = (a0 b0 - a.b) + i (a0 b + b0 a - a x b).s
  Q c;
  c.x = a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w;
  c.y = a.x * b.y + b.x * a.y - (a.z * b.w - a.w * b.z);
  c.z = a.x * b.z + b.x * a.z - (a.w * b.y - a.y * b.w);
  c.w = a.x * b.w + b.x * a.w - (a.y * b.z - a.z * b.y);
  return c;
}
__host__ __device__ inline Q qdag(Q a) { Q c = {a.x, -a.y, -a.z, -a.w}; return c; }
__host__ __device__ inline Q qadd(Q a, Q b) { Q c = {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; return c; }
__host__ __device__ inline double qnorm(Q a) { return sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w); }
__host__ __device__ inline Q qscale(Q a, double s) { Q c = {a.x * s, a.y * s, a.z * s, a.w * s}; return c; }

// ------------------------------------------------------------------------------------------
// Géométrie
// ------------------------------------------------------------------------------------------
struct Geo { int L; long V; long st[4]; };

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

// Somme des 6 agrafes A telle que la partie locale de l'action soit -(beta/2) ReTr(U_mu(x) A).
__device__ Q staple(const Q* __restrict__ U, const Geo& g, int mu, long s, const int x[4]) {
  Q A = {0, 0, 0, 0};
  long smu = nb_up(g, s, mu, x[mu]);
  for (int nu = 0; nu < 4; ++nu) {
    if (nu == mu) continue;
    long snu = nb_up(g, s, nu, x[nu]);
    long sdn = nb_dn(g, s, nu, x[nu]);
    // coordonnée x_nu de x+mu est x[nu] (mu != nu)
    long smu_dn = nb_dn(g, smu, nu, x[nu]);
    // haut : U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+
    Q t = qmul(qmul(ld(U, g, nu, smu), qdag(ld(U, g, mu, snu))), qdag(ld(U, g, nu, s)));
    // bas : U_nu(x+mu-nu)^+ U_mu(x-nu)^+ U_nu(x-nu)
    Q b = qmul(qmul(qdag(ld(U, g, nu, smu_dn)), qdag(ld(U, g, mu, sdn))), ld(U, g, nu, sdn));
    A = qadd(A, qadd(t, b));
  }
  return A;
}

// indice t dans [0, V/2) -> site de parité donnée (L pair)
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
__global__ void k_init(Q* U, Geo g, int hot, u32 k0, u32 k1) {
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

// Tirage exact de x0 selon sqrt(1-x0^2) exp(alpha x0) sur [-1,1] :
// Kennedy–Pendleton si alpha >= 2, Creutz sinon. draw = compteur de tirages Philox (incrémenté).
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

// Test unitaire : n tirages indépendants de x0 à alpha fixé (même fonction que le bain thermique).
__global__ void k_test_x0(double alpha, long n, u32 k0, u32 k1, double* out, unsigned long long* n_cap) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u32 draw = 0, ntry = 0; bool ok = false;
  double x0 = sample_x0(alpha, (u32)i, 0x7E57u, draw, k0, k1, ntry, ok);
  if (!ok) atomicAdd(n_cap, 1ull);
  out[i] = x0;
}

__global__ void k_heatbath(Q* U, Geo g, int mu, int parity, double beta, u32 k0, u32 k1, u32 sweep_id,
                           unsigned long long* n_cap, unsigned long long* n_tries) {
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= g.V / 2) return;
  int x[4];
  long s = site_of(g, t, parity, x);
  long li = (long)mu * g.V + s;
  Q A = staple(U, g, mu, s, x);
  double k = qnorm(A);
  u32 draw = 0;
  Q W;
  if (!(k > 1e-300)) {   // agrafe nulle : mesure de Haar (jamais observé en pratique)
    U4 r1 = philox4x32_10(U4{{(u32)li, sweep_id, draw++, TAG_HB}}, k0, k1);
    U4 r2 = philox4x32_10(U4{{(u32)li, sweep_id, draw++, TAG_HB}}, k0, k1);
    double u1 = u01(r1.v[0], r1.v[1]), u2 = u01(r1.v[2], r1.v[3]);
    double u3 = u01(r2.v[0], r2.v[1]), u4 = u01(r2.v[2], r2.v[3]);
    double ra = sqrt(-2.0 * log(u1)), rb = sqrt(-2.0 * log(u3));
    Q q = {ra * cos(2 * M_PI * u2), ra * sin(2 * M_PI * u2), rb * cos(2 * M_PI * u4), rb * sin(2 * M_PI * u4)};
    U[li] = qscale(q, 1.0 / qnorm(q));
    return;
  }
  Q Vv = qscale(A, 1.0 / k);
  double alpha = beta * k;          // densité de x0 : sqrt(1-x0^2) exp(alpha x0)
  u32 ntry = 0; bool ok = false;
  double x0 = sample_x0(alpha, (u32)li, sweep_id, draw, k0, k1, ntry, ok);
  atomicAdd(n_tries, (unsigned long long)ntry);
  if (!ok) { atomicAdd(n_cap, 1ull); return; }   // compté ; lien inchangé (attendu : 0)
  U4 rd = philox4x32_10(U4{{(u32)li, sweep_id, draw++, TAG_HB}}, k0, k1);
  double ct = 2.0 * u01(rd.v[0], rd.v[1]) - 1.0;
  double ph = 2 * M_PI * u01(rd.v[2], rd.v[3]);
  double st = sqrt(fmax(0.0, 1.0 - ct * ct));
  double rr = sqrt(fmax(0.0, 1.0 - x0 * x0));
  W.x = x0; W.y = rr * st * cos(ph); W.z = rr * st * sin(ph); W.w = rr * ct;
  Q Un = qmul(W, qdag(Vv));          // X = U V suit exp(alpha x0) dHaar  =>  U = X V^+
  U[li] = qscale(Un, 1.0 / qnorm(Un));
}

__global__ void k_overrelax(Q* U, Geo g, int mu, int parity) {
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
__global__ void k_plaq_site(const Q* __restrict__ U, Geo g, double* out) {
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

// réduction déterministe par bloc (256 threads)
__global__ void k_reduce(const double* __restrict__ in, long n, double* out) {
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

// Somme finale déterministe (1 bloc de 256) : chaque thread somme in[t], in[t+256], ... dans
// l'ordre, puis arbre fixe ; écrit sum*scale dans *out. Utilisé pour plaquette et Polyakov.
__global__ void k_final(const double* __restrict__ in, long n, double scale, double* out) {
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
__global__ void k_polyakov(const Q* __restrict__ U, Geo g, double* out) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long L3 = g.V / g.L;
  if (i >= 4 * L3) return;
  int mu = (int)(i / L3);
  long j = i % L3;
  // coordonnées transverses : les 3 directions != mu, dans l'ordre croissant
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

// ------------------------------------------------------------------------------------------
// Hôte
// ------------------------------------------------------------------------------------------
struct Args {
  int L = 8; double beta = 2.5; u64 seed = 1; bool hot = true;
  long therm = 1000, nmeas = 100, skip = 1; int nor = 4; long dump_every = 0;
  std::string outdir = "."; bool dump_initial = false; std::string source_hash = "non_fourni";
  bool selftest = false, bench = false; long nsweeps = 100; double test_alpha = -1; long test_n = 0;
};

static void usage() { fprintf(stderr, "voir en-tête de su2_hip.cpp\n"); exit(2); }

static Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto nxt = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
    if (k == "--L") a.L = atoi(nxt().c_str());
    else if (k == "--beta") a.beta = atof(nxt().c_str());
    else if (k == "--seed") a.seed = strtoull(nxt().c_str(), nullptr, 10);
    else if (k == "--start") { std::string v = nxt(); if (v == "hot") a.hot = true; else if (v == "cold") a.hot = false; else usage(); }
    else if (k == "--therm") a.therm = atol(nxt().c_str());
    else if (k == "--nmeas") a.nmeas = atol(nxt().c_str());
    else if (k == "--skip") a.skip = atol(nxt().c_str());
    else if (k == "--nor") a.nor = atoi(nxt().c_str());
    else if (k == "--dump-every") a.dump_every = atol(nxt().c_str());
    else if (k == "--outdir") a.outdir = nxt();
    else if (k == "--dump-initial") a.dump_initial = true;
    else if (k == "--source-hash") a.source_hash = nxt();
    else if (k == "--selftest") a.selftest = true;
    else if (k == "--bench") a.bench = true;
    else if (k == "--nsweeps") a.nsweeps = atol(nxt().c_str());
    else if (k == "--test-hb") { a.test_alpha = atof(nxt().c_str()); a.test_n = atol(nxt().c_str()); }
    else { fprintf(stderr, "argument inconnu : %s\n", k.c_str()); usage(); }
  }
  if (a.L < 2 || (a.L % 2) != 0) { fprintf(stderr, "L doit être pair >= 2\n"); exit(2); }
  if (!(a.beta >= 0) || !std::isfinite(a.beta)) { fprintf(stderr, "beta invalide\n"); exit(2); }
  if (a.skip < 1 || a.nor < 0) { fprintf(stderr, "skip >= 1, nor >= 0\n"); exit(2); }
  return a;
}

struct Ctx {
  Geo g; Q* dU = nullptr; double* dsite = nullptr; double* dpart = nullptr; double* dpol = nullptr;
  unsigned long long* dcap = nullptr; unsigned long long* dtries = nullptr;
  u32 k0, k1; u32 sweep_id = 0;
  std::vector<double> hpart, hpol;
  double t_hb = 0, t_or = 0; long n_hb = 0, n_or = 0;   // secondes GPU (événements) et nb de sweeps
  bool timed = true;                                      // false : pas de synchro par sweep (production)
  double* dhist = nullptr; long hcap = 0;                 // historique device : 5 doubles par pas (P, P0..P3)
  hipEvent_t e0, e1;
};

static void ctx_init(Ctx& c, const Args& a) {
  c.g.L = a.L; c.g.V = (long)a.L * a.L * a.L * a.L;
  c.g.st[3] = 1; c.g.st[2] = a.L; c.g.st[1] = (long)a.L * a.L; c.g.st[0] = (long)a.L * a.L * a.L;
  c.k0 = (u32)(a.seed & 0xFFFFFFFFu); c.k1 = (u32)(a.seed >> 32);
  HIPCHK(hipMalloc(&c.dU, sizeof(Q) * 4 * c.g.V));
  HIPCHK(hipMalloc(&c.dsite, sizeof(double) * c.g.V));
  long nb = (c.g.V + 255) / 256;
  HIPCHK(hipMalloc(&c.dpart, sizeof(double) * nb));
  c.hpart.resize(nb);
  long L3 = c.g.V / c.g.L;
  HIPCHK(hipMalloc(&c.dpol, sizeof(double) * 4 * L3));
  c.hpol.resize(4 * L3);
  HIPCHK(hipMalloc(&c.dcap, sizeof(unsigned long long)));
  HIPCHK(hipMalloc(&c.dtries, sizeof(unsigned long long)));
  HIPCHK(hipMemset(c.dcap, 0, sizeof(unsigned long long)));
  HIPCHK(hipMemset(c.dtries, 0, sizeof(unsigned long long)));
  HIPCHK(hipEventCreate(&c.e0)); HIPCHK(hipEventCreate(&c.e1));
  long n = 4 * c.g.V;
  hipLaunchKernelGGL(k_init, dim3((n + 255) / 256), dim3(256), 0, 0, c.dU, c.g, a.hot ? 1 : 0, c.k0, c.k1);
  HIPCHK(hipGetLastError()); HIPCHK(hipDeviceSynchronize());
}

static void sweep_hb(Ctx& c, double beta) {
  long nt = c.g.V / 2; dim3 gr((nt + 255) / 256), bl(256);
  if (c.timed) HIPCHK(hipEventRecord(c.e0));
  for (int mu = 0; mu < 4; ++mu)
    for (int p = 0; p < 2; ++p)
      hipLaunchKernelGGL(k_heatbath, gr, bl, 0, 0, c.dU, c.g, mu, p, beta, c.k0, c.k1, c.sweep_id, c.dcap, c.dtries);
  HIPCHK(hipGetLastError());
  if (c.timed) {
    HIPCHK(hipEventRecord(c.e1)); HIPCHK(hipEventSynchronize(c.e1));
    float ms; HIPCHK(hipEventElapsedTime(&ms, c.e0, c.e1)); c.t_hb += ms * 1e-3;
  }
  c.n_hb++;
  c.sweep_id++;
}

static void sweep_or(Ctx& c) {
  long nt = c.g.V / 2; dim3 gr((nt + 255) / 256), bl(256);
  if (c.timed) HIPCHK(hipEventRecord(c.e0));
  for (int mu = 0; mu < 4; ++mu)
    for (int p = 0; p < 2; ++p)
      hipLaunchKernelGGL(k_overrelax, gr, bl, 0, 0, c.dU, c.g, mu, p);
  HIPCHK(hipGetLastError());
  if (c.timed) {
    HIPCHK(hipEventRecord(c.e1)); HIPCHK(hipEventSynchronize(c.e1));
    float ms; HIPCHK(hipEventElapsedTime(&ms, c.e0, c.e1)); c.t_or += ms * 1e-3;
  }
  c.n_or++;
}

static void ctx_free(Ctx& c) {
  HIPCHK(hipDeviceSynchronize());
  HIPCHK(hipEventDestroy(c.e0)); HIPCHK(hipEventDestroy(c.e1));
  HIPCHK(hipFree(c.dU)); HIPCHK(hipFree(c.dsite)); HIPCHK(hipFree(c.dpart)); HIPCHK(hipFree(c.dpol));
  HIPCHK(hipFree(c.dcap)); HIPCHK(hipFree(c.dtries));
  if (c.dhist) HIPCHK(hipFree(c.dhist));
}

static void step(Ctx& c, const Args& a) { sweep_hb(c, a.beta); for (int i = 0; i < a.nor; ++i) sweep_or(c); }

// Mesure asynchrone : plaquette et 4 Polyakov écrits dans dhist[5*idx .. 5*idx+4], sans synchro.
static void measure_async(Ctx& c, long idx) {
  if (idx >= c.hcap) { fprintf(stderr, "ERREUR historique plein\n"); exit(7); }
  long V = c.g.V, nb = (V + 255) / 256, L3 = V / c.g.L;
  hipLaunchKernelGGL(k_plaq_site, dim3(nb), dim3(256), 0, 0, c.dU, c.g, c.dsite);
  hipLaunchKernelGGL(k_reduce, dim3(nb), dim3(256), 0, 0, c.dsite, V, c.dpart);
  hipLaunchKernelGGL(k_final, dim3(1), dim3(256), 0, 0, c.dpart, nb, 1.0 / (6.0 * (double)V), c.dhist + 5 * idx);
  hipLaunchKernelGGL(k_polyakov, dim3((4 * L3 + 255) / 256), dim3(256), 0, 0, c.dU, c.g, c.dpol);
  for (int mu = 0; mu < 4; ++mu)
    hipLaunchKernelGGL(k_final, dim3(1), dim3(256), 0, 0, c.dpol + mu * L3, L3, 1.0 / (double)L3, c.dhist + 5 * idx + 1 + mu);
  HIPCHK(hipGetLastError());
}

static double plaquette(Ctx& c) {
  long V = c.g.V;
  hipLaunchKernelGGL(k_plaq_site, dim3((V + 255) / 256), dim3(256), 0, 0, c.dU, c.g, c.dsite);
  long nb = (V + 255) / 256;
  hipLaunchKernelGGL(k_reduce, dim3(nb), dim3(256), 0, 0, c.dsite, V, c.dpart);
  HIPCHK(hipGetLastError());
  HIPCHK(hipMemcpy(c.hpart.data(), c.dpart, sizeof(double) * nb, hipMemcpyDeviceToHost));
  double s = 0; for (long i = 0; i < nb; ++i) s += c.hpart[i];
  return s / (6.0 * (double)V);
}

static void polyakov(Ctx& c, double P[4]) {
  long L3 = c.g.V / c.g.L, n = 4 * L3;
  hipLaunchKernelGGL(k_polyakov, dim3((n + 255) / 256), dim3(256), 0, 0, c.dU, c.g, c.dpol);
  HIPCHK(hipGetLastError());
  HIPCHK(hipMemcpy(c.hpol.data(), c.dpol, sizeof(double) * n, hipMemcpyDeviceToHost));
  for (int mu = 0; mu < 4; ++mu) { double s = 0; for (long j = 0; j < L3; ++j) s += c.hpol[mu * L3 + j]; P[mu] = s / (double)L3; }
}

static bool dump_npy(Ctx& c, const std::string& path) {
  long n = 4 * c.g.V;
  std::vector<Q> h(n);
  HIPCHK(hipMemcpy(h.data(), c.dU, sizeof(Q) * n, hipMemcpyDeviceToHost));
  int L = c.g.L;
  char descr[256];
  snprintf(descr, sizeof descr, "{'descr': '<c16', 'fortran_order': False, 'shape': (4, %d, %d, %d, %d, 2, 2), }", L, L, L, L);
  std::string hd = descr;
  size_t tot = 10 + hd.size() + 1;
  size_t pad = (64 - tot % 64) % 64;
  hd += std::string(pad, ' '); hd += "\n";
  std::string tmp = path + ".part";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) { fprintf(stderr, "ERREUR ouverture %s\n", tmp.c_str()); return false; }
  const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  fwrite(magic, 1, 8, f);
  uint16_t hl = (uint16_t)hd.size(); fwrite(&hl, 2, 1, f);   // hôte little-endian (x86_64)
  fwrite(hd.data(), 1, hd.size(), f);
  std::vector<double> buf; buf.reserve(8 * 65536);
  size_t nw = 0;
  for (long i = 0; i < n; ++i) {
    const Q& q = h[i];
    double m[8] = {q.x, q.w, q.z, q.y, -q.z, q.y, q.x, -q.w};   // [[a0+i a3, a2+i a1], [-a2+i a1, a0-i a3]]
    buf.insert(buf.end(), m, m + 8);
    if (buf.size() >= 8 * 65536) { nw += fwrite(buf.data(), 8, buf.size(), f); buf.clear(); }
  }
  if (!buf.empty()) nw += fwrite(buf.data(), 8, buf.size(), f);
  bool ok = (nw == (size_t)(8 * n)) && (fclose(f) == 0);
  if (!ok) { fprintf(stderr, "ERREUR écriture %s\n", tmp.c_str()); return false; }
  if (rename(tmp.c_str(), path.c_str()) != 0) { fprintf(stderr, "ERREUR rename %s\n", path.c_str()); return false; }
  return true;
}

static int selftest() {
  int bad = 0;
  // KAT Random123 philox4x32_10
  struct { u32 c[4], k[2], r[4]; } kat[3] = {
    {{0, 0, 0, 0}, {0, 0}, {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}},
    {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xffffffffu}, {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}},
    {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u}, {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}}};
  for (int t = 0; t < 3; ++t) {
    U4 c = {{kat[t].c[0], kat[t].c[1], kat[t].c[2], kat[t].c[3]}};
    U4 r = philox4x32_10(c, kat[t].k[0], kat[t].k[1]);
    bool ok = true; for (int j = 0; j < 4; ++j) ok &= (r.v[j] == kat[t].r[j]);
    printf("SELFTEST philox KAT %d : %s (%08x %08x %08x %08x)\n", t, ok ? "PASS" : "FAIL", r.v[0], r.v[1], r.v[2], r.v[3]);
    bad += !ok;
  }
  // bornes u01
  double lo = u01(0, 0), hi = u01(0xffffffffu, 0xffffffffu);
  bool okb = lo > 0 && hi < 1;
  printf("SELFTEST u01 bornes : %s (%.3e, 1-%.3e)\n", okb ? "PASS" : "FAIL", lo, 1 - hi);
  bad += !okb;
  return bad;
}

static void write_manifest(const Args& a, const Ctx& c, const std::string& path, const char* status) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) { fprintf(stderr, "ERREUR manifest\n"); exit(4); }
  fprintf(f, "{\n\"schema\":1,\"engine\":\"su2_cuda\",\"source_hash\":\"%s\",\"backend\":\"CUDA sm_120 FP64 (portage de su2_hip)\","
             "\"rng\":\"Philox4x32-10 compteur (lien,sweep,tirage,etiquette), cle=graine\",\"l\":%d,\"beta\":%.17g,"
             "\"seed\":%llu,\"start\":\"%s\",\"therm_steps\":%ld,\"measurements\":%ld,\"skip_steps\":%ld,\"or_per_step\":%d,"
             "\"dump_every\":%ld,\"action\":\"beta sum(1-ReTr(U_p)/2)\","
             "\"step\":\"1 sweep bain thermique (Kennedy-Pendleton alpha>=2, Creutz alpha<2) + or_per_step sweeps OR, damier 4 mu x 2 parites\","
             "\"layout\":\"complex128 C-order (mu,x0,x1,x2,x3,i,j), temporal=3\","
             "\"index\":\"0 = etat initial ; 1..N apres skip_steps pas ; dump au meme indice\",\"statut\":\"%s\"}\n",
          a.source_hash.c_str(), a.L, a.beta, (unsigned long long)a.seed, a.hot ? "Hot" : "Cold", a.therm, a.nmeas, a.skip,
          a.nor, a.dump_every, status);
  fclose(f);
}

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  if (a.selftest) { int b = selftest(); printf("SELFTEST %s\n", b ? "ECHEC" : "OK"); return b ? 1 : 0; }
  int ndev = 0; HIPCHK(hipGetDeviceCount(&ndev));
  hipDeviceProp_t pr; HIPCHK(hipGetDeviceProperties(&pr, 0));
  fprintf(stderr, "# su2_cuda : %d GPU, %s sm_%d%d\n", ndev, pr.name, pr.major, pr.minor);
  if (a.test_n > 0) {
    // moments de x0 : E[x0], E[x0^2] (sommes hôte séquentielles)
    double* d; unsigned long long* dc; HIPCHK(hipMalloc(&d, sizeof(double) * a.test_n));
    HIPCHK(hipMalloc(&dc, sizeof(unsigned long long))); HIPCHK(hipMemset(dc, 0, sizeof(unsigned long long)));
    hipLaunchKernelGGL(k_test_x0, dim3((a.test_n + 255) / 256), dim3(256), 0, 0, a.test_alpha, a.test_n,
                       (u32)(a.seed & 0xFFFFFFFFu), (u32)(a.seed >> 32), d, dc);
    HIPCHK(hipGetLastError()); HIPCHK(hipDeviceSynchronize());
    std::vector<double> h(a.test_n); HIPCHK(hipMemcpy(h.data(), d, sizeof(double) * a.test_n, hipMemcpyDeviceToHost));
    unsigned long long cap; HIPCHK(hipMemcpy(&cap, dc, sizeof cap, hipMemcpyDeviceToHost));
    double m1 = 0, m2 = 0, mn = 1, mx = -1;
    for (long i = 0; i < a.test_n; ++i) { m1 += h[i]; m2 += h[i] * h[i]; mn = fmin(mn, h[i]); mx = fmax(mx, h[i]); }
    m1 /= a.test_n; m2 /= a.test_n;
    printf("TESTHB alpha=%.17g n=%ld m1=%.17g m2=%.17g min=%.17g max=%.17g plafond=%llu\n", a.test_alpha, a.test_n, m1, m2, mn, mx, cap);
    HIPCHK(hipFree(d)); HIPCHK(hipFree(dc));
    return cap ? 6 : 0;
  }
  Ctx c; ctx_init(c, a);
  if (a.bench) {
    for (int i = 0; i < 5; ++i) step(c, a);   // chauffe
    c.t_hb = c.t_or = 0; c.n_hb = c.n_or = 0;
    auto w0 = std::chrono::steady_clock::now();
    for (long i = 0; i < a.nsweeps; ++i) sweep_hb(c, a.beta);
    for (long i = 0; i < a.nsweeps; ++i) sweep_or(c);
    double P = plaquette(c);
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
    double nl = 4.0 * c.g.V;
    printf("BENCH L=%d beta=%g V=%ld hb_sweeps=%ld hb_s_par_sweep=%.6e or_sweeps=%ld or_s_par_sweep=%.6e "
           "liens_par_s_hb=%.4e liens_par_s_or=%.4e wall=%.3f P=%.10f\n",
           a.L, a.beta, c.g.V, c.n_hb, c.t_hb / c.n_hb, c.n_or, c.t_or / c.n_or, nl * c.n_hb / c.t_hb, nl * c.n_or / c.t_or, wall, P);
    ctx_free(c);
    return 0;
  }
  mkdir(a.outdir.c_str(), 0755);
  write_manifest(a, c, a.outdir + "/manifest.json", "EN_COURS");
  c.timed = false;
  c.hcap = std::max(a.therm, a.nmeas) + 1;
  HIPCHK(hipMalloc(&c.dhist, sizeof(double) * 5 * c.hcap));
  std::vector<double> hh(5 * c.hcap);
  auto w0 = std::chrono::steady_clock::now();
  FILE* ft = fopen((a.outdir + "/therm.csv").c_str(), "w");
  FILE* fm = fopen((a.outdir + "/measurements.csv").c_str(), "w");
  if (!ft || !fm) { fprintf(stderr, "ERREUR csv\n"); return 4; }
  fprintf(ft, "step,plaquette,P0,P1,P2,P3\n");
  fprintf(fm, "measurement_index,total_steps,phase,plaquette,P0,P1,P2,P3\n");
  // rapatrie hist[i0..i1) et écrit les lignes
  auto flush = [&](FILE* f, long i0_, long i1_, bool therm) {
    if (i1_ <= i0_) return;
    HIPCHK(hipMemcpy(hh.data() + 5 * i0_, c.dhist + 5 * i0_, sizeof(double) * 5 * (i1_ - i0_), hipMemcpyDeviceToHost));
    for (long i = i0_; i < i1_; ++i) {
      const double* r = hh.data() + 5 * i;
      if (therm) fprintf(f, "%ld,%.17g,%.17g,%.17g,%.17g,%.17g\n", i, r[0], r[1], r[2], r[3], r[4]);
      else fprintf(f, "%ld,%ld,%s,%.17g,%.17g,%.17g,%.17g,%.17g\n", i, a.therm + i * a.skip, i == 0 ? "initial" : "sample",
                   r[0], r[1], r[2], r[3], r[4]);
    }
    fflush(f);
  };
  const long FL = 200;
  // --- thermalisation (indice 0 = état initial) ---
  measure_async(c, 0);
  if (a.dump_initial) {
    char nm[64]; snprintf(nm, sizeof nm, "/cfg_%06d.npy", 0);
    if (!dump_npy(c, a.outdir + nm)) return 5;
  }
  long done = 0;
  for (long t = 1; t <= a.therm; ++t) {
    step(c, a);
    measure_async(c, t);
    if (t % FL == 0) { flush(ft, done, t + 1, true); done = t + 1; }
  }
  flush(ft, done, a.therm + 1, true);
  fclose(ft);
  double w_therm = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
  // --- production : indice 0 = fin de thermalisation (copie de la dernière mesure) ---
  measure_async(c, 0);
  done = 0;
  int ndump = 0;
  for (long m = 1; m <= a.nmeas; ++m) {
    for (long k = 0; k < a.skip; ++k) step(c, a);
    measure_async(c, m);
    if (a.dump_every > 0 && m % a.dump_every == 0) {
      char nm[64]; snprintf(nm, sizeof nm, "/cfg_%06ld.npy", m);
      if (!dump_npy(c, a.outdir + nm)) return 5;
      ++ndump;
    }
    if (m % FL == 0) { flush(fm, done, m + 1, false); done = m + 1; }
  }
  flush(fm, done, a.nmeas + 1, false);
  fclose(fm);
  unsigned long long cap = 0, tries = 0;
  HIPCHK(hipMemcpy(&cap, c.dcap, sizeof cap, hipMemcpyDeviceToHost));
  HIPCHK(hipMemcpy(&tries, c.dtries, sizeof tries, hipMemcpyDeviceToHost));
  double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
  double nl = 4.0 * c.g.V;
  double hb_updates = nl * (double)c.n_hb;
  printf("PERF L=%d beta=%g hb_sweeps=%ld or_sweeps=%ld essais_par_lien_hb=%.5f plafond_atteint=%llu dumps=%d "
         "wall_therm=%.2f wall=%.2f pas_par_s=%.2f\n",
         a.L, a.beta, c.n_hb, c.n_or, (double)tries / fmax(hb_updates, 1.0), cap, ndump, w_therm, wall,
         (double)(a.therm + a.nmeas * a.skip) / fmax(wall, 1e-9));
  ctx_free(c);
  if (cap != 0) { fprintf(stderr, "ERREUR : plafond d'essais atteint %llu fois\n", cap); return 6; }
  write_manifest(a, c, a.outdir + "/manifest.json", "COMPLETE");
  FILE* fc = fopen((a.outdir + "/COMPLETE").c_str(), "w"); if (fc) fclose(fc);
  return 0;
}
