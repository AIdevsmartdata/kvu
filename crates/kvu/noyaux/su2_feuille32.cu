// su2_feuille32.cu : chemin RAPIDE rectangulaire de la campagne « une boîte » : noyaux F32 (bain, surrelaxation) sur la géométrie
// régulière `GeoG` de su2_geo.cu, mesures promues en F64 (produits et réductions en double, arbre fixe : k_reduce / k_final).
// Module « su2_boite_une32 » = su2_commun.h + su2_f32.cu + su2_ref64.cu + su2_geo.cu + su2_feuille.cu + ce fichier.
// Même algorithme que k_heatbath_f32 / k_overrelax_f32 (su2_f32.cu : tirage de x0 `sample_x0f`, réunitarisation à chaque lien écrit,
// fonctions transcendantes IEEE), seuls changent la géométrie (côtés pairs quelconques, tampon `geo`), le coefficient par plaquette
// (torsion, poids réel de feuille, lu en double puis converti en float) et β = 0 (lien de Haar).
// Champ : QF (float4) par lien, disposition (c·4V + li), comme le chemin F32 cubique.

__device__ inline QF g_ldf(const QF* __restrict__ U, const GeoG& g, int mu, long s) { return U[(long)mu * g.V + s]; }

// ---- Chemin chaud (accel/gpu-1119) : MÊMES opérations flottantes, dans le MÊME ordre, que la version antérieure (sortie identique
// au bit) ; seuls changent l'arithmétique d'indices (32 bits au lieu de 64 : divisions entières de g_coords), l'indexation
// constante de x[] (registres au lieu de la mémoire locale), le coefficient de plaquette lu en float (conversion double→float
// faite une fois à l'initialisation, au lieu de (float)coef[...] à chaque lecture) et le compteur d'essais agrégé par warp
// (somme entière : l'ordre n'importe pas).
#ifndef KVU_COUTURE
struct GeoI { int l[4]; int V; int st[4]; };

__device__ inline GeoI h_lire(const long long* __restrict__ geo) {
  GeoI g;
  for (int d = 0; d < 4; ++d) { g.l[d] = (int)geo[d]; g.st[d] = (int)geo[5 + d]; }
  g.V = (int)geo[4];
  return g;
}
__device__ inline int h_up(const GeoI& g, int s, int d, int xd) {
  return (xd == g.l[d] - 1) ? s - (g.l[d] - 1) * g.st[d] : s + g.st[d];
}
__device__ inline int h_dn(const GeoI& g, int s, int d, int xd) {
  return (xd == 0) ? s + (g.l[d] - 1) * g.st[d] : s - g.st[d];
}
#else
// Couture décalée (voir su2_geo.cu) : voisins recalculés depuis le site seul, `xd` ignoré.
struct GeoI { int l[4]; int V; int st[4]; int cax; int cv[4]; };

__device__ inline GeoI h_lire(const long long* __restrict__ geo) {
  GeoI g;
  for (int d = 0; d < 4; ++d) { g.l[d] = (int)geo[d]; g.st[d] = (int)geo[5 + d]; }
  g.V = (int)geo[4];
  const long long* c = geo + 14 + geo[13];
  g.cax = (int)c[0];
  for (int d = 0; d < 4; ++d) g.cv[d] = (int)c[1 + d];
  return g;
}
__device__ inline int h_decale(const GeoI& g, int b, int signe) {
  int r = 0;
  for (int d = 0; d < 4; ++d) {
    int xd = (b / g.st[d]) % g.l[d];
    int y = signe >= 0 ? (xd + g.cv[d]) % g.l[d] : (xd + g.l[d] - g.cv[d]) % g.l[d];
    r += y * g.st[d];
  }
  return r;
}
__device__ inline int h_up(const GeoI& g, int s, int d, int xd) {
  int x = (s / g.st[d]) % g.l[d];
  if (x == g.l[d] - 1) {
    int b = s - (g.l[d] - 1) * g.st[d];
    return d == g.cax ? h_decale(g, b, 1) : b;
  }
  return s + g.st[d];
}
__device__ inline int h_dn(const GeoI& g, int s, int d, int xd) {
  int x = (s / g.st[d]) % g.l[d];
  if (x == 0) {
    int b = s + (g.l[d] - 1) * g.st[d];
    return d == g.cax ? h_decale(g, b, -1) : b;
  }
  return s - g.st[d];
}
#endif
__device__ inline int h_site_of(const GeoI& g, int t, int parity, int x[4]) {
  int s = 2 * t;
  unsigned u = (unsigned)s;
  x[3] = (int)(u % (unsigned)g.l[3]); u /= (unsigned)g.l[3];
  x[2] = (int)(u % (unsigned)g.l[2]); u /= (unsigned)g.l[2];
  x[1] = (int)(u % (unsigned)g.l[1]); x[0] = (int)(u / (unsigned)g.l[1]);
  int p = (x[0] + x[1] + x[2] + x[3]) & 1;
  if (p != parity) { s += 1; x[3] += 1; }
  return s;
}
// Voir su2_geo.cu (g_site_sel) : liste de sites par classe si geo[9] ≠ 0 (côtés impairs), sinon damier historique.
__device__ inline int h_site_sel(const GeoI& g, const long long* __restrict__ geo, int t, int cls, int x[4]) {
#ifdef KVU_LISTE_CLASSES
  if (geo[9]) {
    int s = (int)geo[14 + geo[10 + cls] + t];
    unsigned u = (unsigned)s;
    x[3] = (int)(u % (unsigned)g.l[3]); u /= (unsigned)g.l[3];
    x[2] = (int)(u % (unsigned)g.l[2]); u /= (unsigned)g.l[2];
    x[1] = (int)(u % (unsigned)g.l[1]); x[0] = (int)(u / (unsigned)g.l[1]);
    return s;
  }
#endif
  return h_site_of(g, t, cls, x);
}
__device__ inline int h_xmu(const int x[4], int mu) { return mu == 0 ? x[0] : (mu == 1 ? x[1] : (mu == 2 ? x[2] : x[3])); }
__device__ inline QF h_ldf(const QF* __restrict__ U, const GeoI& g, int mu, int s) { return U[mu * g.V + s]; }

__device__ QF h_agrafe_f(const QF* __restrict__ U, const GeoI& g, int mu, int s, const int x[4], const float* __restrict__ coef) {
  QF A = {0.0f, 0.0f, 0.0f, 0.0f};
  int smu = h_up(g, s, mu, h_xmu(x, mu));
#pragma unroll
  for (int nu = 0; nu < 4; ++nu) {
    if (nu == mu) continue;
    int snu = h_up(g, s, nu, x[nu]);
    int sdn = h_dn(g, s, nu, x[nu]);
    int smu_dn = h_dn(g, smu, nu, x[nu]);
    QF t = qfmul(qfmul(h_ldf(U, g, nu, smu), qfdag(h_ldf(U, g, mu, snu))), qfdag(h_ldf(U, g, nu, s)));
    QF b = qfmul(qfmul(qfdag(h_ldf(U, g, nu, smu_dn)), qfdag(h_ldf(U, g, mu, sdn))), h_ldf(U, g, nu, sdn));
    if (coef) {
      int p = g_plan(mu, nu);
      float zt = coef[p * g.V + s], zb = coef[p * g.V + sdn];
      A = qfadd(A, qfadd(qfscale(t, zt), qfscale(b, zb)));
    } else {
      A = qfadd(A, qfadd(t, b));
    }
  }
  return A;
}

extern "C" __global__ void k_init_f32g(QF* U, const long long* geo, const int* hot, const u32* k0s, const u32* k1s) {
  GeoG g = g_lire(geo);
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 4 * g.V) return;
  const int c = blockIdx.y;
  U += (long)c * 4 * g.V;
  if (!hot[c]) { QF e = {1.0f, 0.0f, 0.0f, 0.0f}; U[i] = e; return; }
  U[i] = haarf(philox4x32_10(U4{{(unsigned int)i, 0xFFFFFFFFu, 0u, KV32_TAG_INIT}}, k0s[c], k1s[c]));
}

extern "C" __global__ void __launch_bounds__(256) k_heatbath_f32g(QF* U, const long long* geo, int mu, int parity, const double* betas,
                                           const u32* k0s, const u32* k1s, u32 sweep_id, unsigned long long* n_cap,
                                           unsigned long long* n_tries, const float* __restrict__ coef,
                                           const unsigned char* __restrict__ masque) {
  GeoI g = h_lire(geo);
  const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const bool act = t < (int)g_nsel(geo, parity, g.V);               // tous les fils du warp atteignent la réduction de n_tries
  const int c = blockIdx.y;
  U += (long)c * 4 * g.V;
  const u32 k0 = k0s[c], k1 = k1s[c];
  const float beta = (float)betas[c];
  u32 ntry = 0; bool ok = false;
  if (act) {
    int x[4];
    int s = h_site_sel(g, geo, t, parity, x);
    int li = mu * g.V + s;
    if (masque && masque[li]) goto fin_masque;   // lien gelé (deux niveaux) : ni bain ni compteur
    {
    QF A = h_agrafe_f(U, g, mu, s, x, coef);
    float k = qfnorm(A);
    u32 draw = 0;
    if (!(k > 1e-30f) || beta == 0.0f) {
      U[li] = haarf(philox4x32_10(U4{{(unsigned int)li, sweep_id, draw++, KV32_TAG_HB}}, k0, k1));
    } else {
      QF Vv = qfscale(A, 1.0f / k);
      float alpha = beta * k;
      float x0 = sample_x0f(alpha, (unsigned int)li, sweep_id, draw, k0, k1, ntry, ok);
      if (!ok) {
        atomicAdd(n_cap + c, 1ull);           // compté ; lien inchangé (attendu : 0)
      } else {
        U4 rd = philox4x32_10(U4{{(unsigned int)li, sweep_id, draw++, KV32_TAG_HB}}, k0, k1);
        float ct = 2.0f * u01f(rd.v[0]) - 1.0f;
        float ph = KV32_2PI * u01f(rd.v[1]);
        float st = sqrtf(fmaxf(0.0f, 1.0f - ct * ct));
        float rr = sqrtf(fmaxf(0.0f, 1.0f - x0 * x0));
        QF W = {x0, rr * st * cosf(ph), rr * st * sinf(ph), rr * ct};
        U[li] = qfunit(qfmul(W, qfdag(Vv)));
      }
    }
    }
    fin_masque:;
  }
#if defined(KV_CUDA) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  unsigned int w = __reduce_add_sync(0xffffffffu, ntry);
  if ((threadIdx.x & 31) == 0 && w) atomicAdd(n_tries + c, (unsigned long long)w);
#else
  if (ntry) atomicAdd(n_tries + c, (unsigned long long)ntry);
#endif
}

extern "C" __global__ void __launch_bounds__(256) k_overrelax_f32g(QF* U, const long long* geo, int mu, int parity, const float* __restrict__ coef,
                                                                    const unsigned char* __restrict__ masque) {
  GeoI g = h_lire(geo);
  const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= (int)g_nsel(geo, parity, g.V)) return;
  U += (long)blockIdx.y * 4 * g.V;
  int x[4];
  int s = h_site_sel(g, geo, t, parity, x);
  int li = mu * g.V + s;
  if (masque && masque[li]) return;
  QF A = h_agrafe_f(U, g, mu, s, x, coef);
  float k = qfnorm(A);
  if (!(k > 1e-30f)) return;
  QF Vd = qfdag(qfscale(A, 1.0f / k));
  U[li] = qfunit(qfmul(qfmul(Vd, qfdag(U[li])), Vd));
}

// mesures en double (lecture float promue) : S_tot(s) et S_feuille(s) par site
extern "C" __global__ void k_plaq_site_feuille_f32g(const QF* __restrict__ U, const long long* geo,
                                                    const unsigned char* __restrict__ feuille, double* out_tot, double* out_fe) {
  GeoG g = g_lire(geo);
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= g.V) return;
  U += (long)blockIdx.y * 4 * g.V;
  out_tot += (long)blockIdx.y * g.V;
  out_fe += (long)blockIdx.y * g.V;
  int x[4]; g_coords(g, s, x);
  double tot = 0, fe = 0;
  for (int mu = 0; mu < 4; ++mu) {
    long smu = g_up(g, s, mu, x[mu]);
    for (int nu = mu + 1; nu < 4; ++nu) {
      long snu = g_up(g, s, nu, x[nu]);
      Q p = qmul(qmul(qdbl(g_ldf(U, g, mu, s)), qdbl(g_ldf(U, g, nu, smu))),
                 qmul(qdag(qdbl(g_ldf(U, g, mu, snu))), qdag(qdbl(g_ldf(U, g, nu, s)))));
      tot += p.x;
      if (feuille && feuille[(long)g_plan(mu, nu) * g.V + s]) fe += p.x;
    }
  }
  out_tot[s] = tot;
  out_fe[s] = fe;
}

// OPTION NON IDENTIQUE AU BIT (désactivée par défaut, KVU_MESURE_F32=1) : produits de plaquettes en FLOAT (même parcours que
// k_plaq_site_feuille_f32g), accumulation par site en double. Erreur de produit ~ 1e-7 relatif par plaquette, sans biais systématique
// (la réunitarisation des liens garde |U| = 1 à l'arrondi float près).
extern "C" __global__ void k_plaq_site_feuille_f32gs(const QF* __restrict__ U, const long long* geo,
                                                     const unsigned char* __restrict__ feuille, double* out_tot, double* out_fe) {
  GeoG g = g_lire(geo);
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= g.V) return;
  U += (long)blockIdx.y * 4 * g.V;
  out_tot += (long)blockIdx.y * g.V;
  out_fe += (long)blockIdx.y * g.V;
  int x[4]; g_coords(g, s, x);
  double tot = 0, fe = 0;
  for (int mu = 0; mu < 4; ++mu) {
    long smu = g_up(g, s, mu, x[mu]);
    for (int nu = mu + 1; nu < 4; ++nu) {
      long snu = g_up(g, s, nu, x[nu]);
      QF p = qfmul(qfmul(g_ldf(U, g, mu, s), g_ldf(U, g, nu, smu)),
                   qfmul(qfdag(g_ldf(U, g, mu, snu)), qfdag(g_ldf(U, g, nu, s))));
      tot += (double)p.x;
      if (feuille && feuille[(long)g_plan(mu, nu) * g.V + s]) fe += (double)p.x;
    }
  }
  out_tot[s] = tot;
  out_fe[s] = fe;
}
