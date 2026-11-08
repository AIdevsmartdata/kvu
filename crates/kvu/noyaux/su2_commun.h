// su2_commun.h — noyaux SU(2) du kevinotron unifié : socle COMMUN (propriétaire : T3).
//
// CONTRAT DE BIT-IDENTITÉ. Tout ce qui suit la ligne « DEBUT DE LA RECOPIE » est la recopie CARACTÈRE
// POUR CARACTÈRE de src/hip/su2_sweep.hip (lot K1, sha256 da154635…, lignes 27 à 130) :
//   M_PI, Philox4x32-10, u01, énumération TAG_INIT/TAG_HB, struct Q et son algèbre, Geo/mkgeo,
//   nb_up/nb_dn/coords/ld, staple, site_of.
// Seul l'#include <hip/hip_runtime.h> n'est pas repris : il est déplacé dans le prélude du backend
// (prelude_hip.h / prelude_cuda.h), l'assemblage tenant lieu d'include (voir noyaux.rs).
// NE RIEN MODIFIER SOUS CETTE LIGNE : su2_ref64.cu, su2_bord.cu (T3), su2_f32.cu (T4), cond.cu (T5) et
// obs_*.cu (T7) en dépendent, et le test su2_noyaux_sources.rs compare ce bloc à su2_sweep.hip.
// Les étiquettes TAG_BORD et TAG_JAUGE (su2_bord.cu) sont déclarées dans su2_bord.cu, pas ici.
// ---------------------------------------------------------------------------------------------------
// DEBUT DE LA RECOPIE (src/hip/su2_sweep.hip, ligne 27)
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

// FIN DE LA RECOPIE (src/hip/su2_sweep.hip, ligne 130)
