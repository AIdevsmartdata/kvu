// su2_lien.cu : estimateur « à lien intégré » (multi-hit exact de Parisi–Petronzio–Rapuano, 1983) des sommes de plaquettes.
// Compilé seulement avec l'option --lien-integre de `su2 thermo mesure` (module à part : les modules sans l'option sont inchangés).
// Pour chaque lien l : A_l = somme PONDÉRÉE (coefficients de plan, β de chaîne dehors) de ses six agrafes ; κ_l = β·|A_l| ;
// V_l = A_l/|A_l| ; E[U_l | agrafes] = [I₂(κ)/I₁(κ)]·V_l†. Pour un ensemble E de plaquettes : S_E = ¼ Σ_l ½Tr(U_l B_l^E), B_l^E = somme NON
// pondérée des agrafes de l dont la plaquette est dans E. Estimateur : ¼ Σ_l [I₂/I₁](κ_l)·½Tr(V_l† B_l^E) = ¼ Σ_l r(κ_l)·<A_l,B_l^E>/|A_l|
// (produit scalaire R⁴ des quaternions). Même espérance que S_E. Sorties par site (trois tableaux, noms hérités) : [0] = Y0 = ½ Σ_{liens temporels} (S_t par les seuls liens temporels : chaque plaquette
// temporelle a deux liens temporels) ; [1] = Y_s = ½ Σ_{liens spatiaux} (S_t par les seuls liens spatiaux) ; [2] = Z_s = ¼ Σ_{liens spatiaux}
// (S_s : toute plaquette spatiale a quatre liens spatiaux, aucun lien temporel). Y0 et Y_s sont deux estimateurs sans biais de S_t ;
// l'estimateur « tous les liens » vaut (Y0 + Y_s)/2. S_tot : Y0 + Z_s et Y_s + Z_s (sommes cohérentes), assemblées côté hôte.
// Mêmes parcours et mêmes coefficients que g_agrafe (su2_geo.cu) : coef[p·V + coin de la plaquette].
// Les calculs sont faits en double (lecture float promue en F32), sommes réduites par k_reduce/k_final (accumulateurs F64).
#ifdef KVU_LI_F32
typedef QF LI_U;
typedef float LI_C;
__device__ inline Q li_ld(const QF* __restrict__ U, const GeoG& g, int mu, long s) { return qdbl(g_ldf(U, g, mu, s)); }
#else
typedef Q LI_U;
typedef double LI_C;
__device__ inline Q li_ld(const Q* __restrict__ U, const GeoG& g, int mu, long s) { return g_ld(U, g, mu, s); }
#endif

// I₂(κ)/I₁(κ) : fraction continue descendante r ← 1/(2n/κ + r), n = N … 2 ; N = 12 + 1.5κ + 6√κ (erreur relative < 3e-16 pour κ ≤ 300,
// vérifiée contre mpmath) ; κ → 0 : κ/4.
#ifdef KVU_LI_F32
// F32 : même fraction continue en FLOAT avec N = 6 + 0.7k + 2.5√k (erreur relative ~1e-7, vérifiée contre mpmath en float32 ; cible 1e-6).
__device__ inline double li_ratio(double kd) {
  if (!(kd > 1e-300)) return 0.0;
  if (kd < 1e-8) return 0.25 * kd;
  float k = (float)kd;
  int N = 6 + (int)(0.7f * k + 2.5f * sqrtf(k));
  float r = 0.0f;
  for (int n = N; n >= 2; --n) r = 1.0f / (2.0f * (float)n / k + r);
  return (double)r;
}
#else
__device__ inline double li_ratio(double k) {
  if (!(k > 1e-300)) return 0.0;
  if (k < 1e-8) return 0.25 * k;
  int N = 12 + (int)(1.5 * k + 6.0 * sqrt(k));
  double r = 0.0;
  for (int n = N; n >= 2; --n) r = 1.0 / (2.0 * n / k + r);
  return r;
}
#endif

__device__ inline double li_dot(Q a, Q b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

extern "C" __global__ void k_plaq_site_li(const LI_U* __restrict__ U, const long long* geo, const LI_C* __restrict__ coef,
                                          const double* __restrict__ betas, double* out_tot, double* out_t, double* out_s) {
  GeoG g = g_lire(geo);
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= g.V) return;
  const double beta = betas[blockIdx.y];
  U += (long)blockIdx.y * 4 * g.V;
  out_tot += (long)blockIdx.y * g.V;
  out_t += (long)blockIdx.y * g.V;
  out_s += (long)blockIdx.y * g.V;
  int x[4]; g_coords(g, s, x);
  double y0t = 0, yst = 0, yss = 0;
  for (int mu = 0; mu < 4; ++mu) {
    Q A = {0, 0, 0, 0}, Bt = {0, 0, 0, 0}, Bs = {0, 0, 0, 0};
    long smu = g_up(g, s, mu, x[mu]);
    for (int nu = 0; nu < 4; ++nu) {
      if (nu == mu) continue;
      long snu = g_up(g, s, nu, x[nu]);
      long sdn = g_dn(g, s, nu, x[nu]);
      long smu_dn = g_dn(g, smu, nu, x[nu]);
      Q t = qmul(qmul(li_ld(U, g, nu, smu), qdag(li_ld(U, g, mu, snu))), qdag(li_ld(U, g, nu, s)));
      Q b = qmul(qmul(qdag(li_ld(U, g, nu, smu_dn)), qdag(li_ld(U, g, mu, sdn))), li_ld(U, g, nu, sdn));
      if (coef) {
        int p = g_plan(mu, nu);
        A = qadd(A, qadd(qscale(t, (double)coef[(long)p * g.V + s]), qscale(b, (double)coef[(long)p * g.V + sdn])));
      } else {
        A = qadd(A, qadd(t, b));
      }
      if (mu == 0 || nu == 0) { Bt = qadd(Bt, qadd(t, b)); } else { Bs = qadd(Bs, qadd(t, b)); }
    }
    double an = qnorm(A);
    if (!(an > 1e-300)) continue;
    double c = li_ratio(beta * an) / an;
    double dt = li_dot(A, Bt), ds = li_dot(A, Bs);
    if (mu == 0) { y0t += c * dt; } else { yst += c * dt; yss += c * ds; }
  }
  out_tot[s] = 0.5 * y0t;    // Y0 : S_t par les seuls liens TEMPORELS (= ½ Σ_{l temporel} |a_l| R(β_t |a_l|) ; exactement 0 à β_t = 0)
  out_t[s] = 0.5 * yst;      // Y_s : S_t par les seuls liens SPATIAUX
  out_s[s] = 0.25 * yss;     // S_s par les liens spatiaux (seuls liens où il existe des plaquettes spatiales)
}
