// su2_f32.cu — chemin FP32 rapide du générateur SU(2) Wilson 4D (tâche T4).
//
// ASSEMBLAGE : prélude du backend, puis su2_commun.h (T3), puis CE fichier. On y réutilise de
// su2_commun.h : Philox4x32-10 (`U4`, `philox4x32_10`), la géométrie entière (`Geo`, `mkgeo`,
// `nb_up`, `nb_dn`, `coords`, `site_of`) et, pour les MESURES seulement, l'algèbre FP64
// (`Q`, `qmul`, `qdag`). Aucune fonction FP64 dans les chemins chauds (bain thermique,
// overrelaxation) : tout y est en `float`.
//
// RÈGLES DU CHEMIN RAPIDE (ARCHITECTURE_UNIFIEE.md §2, §4.4, §6.2) :
//  - stockage `float4` (QF), une chaîne c d'un lot commence à l'indice (c·4V) de lien, composante
//    k de l'indice local li = mu·V + s à l'adresse (c·4V + li)·4 + k (même disposition que le FP64) ;
//  - Philox INCHANGÉ : clé (graine basse, graine haute) de la chaîne, compteur
//    (li LOCAL, sweep, tirage, étiquette). Chaque mot de 32 bits donne un uniforme float
//    indépendant (`u01f`), donc un appel Philox (4 mots) nourrit un essai entier de Kennedy–Pendleton
//    là où le FP64 en consomme deux (une paire de mots par uniforme) ;
//  - `logf`, `cosf`, `sinf`, `expf`, `sqrtf` en précision IEEE de la bibliothèque : les intrinsèques
//    rapides `__logf`, `__expf`, `__sinf`, `__cosf`, `__fdividef` sont INTERDITES (elles feraient
//    dériver la loi du bain thermique) ;
//  - chaque lien écrit (bain thermique ET overrelaxation) est RÉUNITARISÉ (q/|q| en float) : pas de
//    dérive de la norme sur 10⁴ pas et plus (gate LOI-0) ;
//  - mesures (plaquette, Polyakov, tranches) : lecture float promue en double, produits et réductions
//    en double, à arbre fixe (mêmes arbres que k_reduce/k_final de su2_hip, copiés ici car il faut
//    des versions par chaîne : `blockIdx.y` = chaîne) ;
//  - contrôle négatif : si K1_MUTANT_METRO est défini, le tirage de x0 est remplacé par UN pas de
//    Metropolis naïf (jamais en production ; gate LOI-4 b).
//
// LOTS : `blockIdx.y` = chaîne ; β, clé Philox et départ sont lus dans des tableaux par chaîne.
// La chaîne c d'un lot reçoit donc EXACTEMENT les mêmes compteurs que si elle tournait seule.

#define KV32_2PI 6.28318530717958647692f

// Étiquettes du compteur Philox (mêmes valeurs que su2_hip.cpp : "INIT" et "HBAT").
enum : unsigned int { KV32_TAG_INIT = 0x494E4954u, KV32_TAG_HB = 0x48424154u };

// ------------------------------------------------------------------------------------------
// Uniforme float strictement dans ]0,1[
// ------------------------------------------------------------------------------------------
// 23 bits, milieu de cellule : (m + 0,5)·2⁻²³ avec m ∈ [0, 2²³−1]. m + 0,5 tient sur 24 bits, donc
// est EXACTEMENT représentable ; le résultat est dans [2⁻²⁴, 1 − 2⁻²⁴] : jamais 0 ni 1.
// (La forme à 24 bits ((a >> 8) + 0,5f)·2⁻²⁴ n'est pas exacte : pour m ≥ 2²³ le pas des float vaut 1,
// m + 0,5 est arrondi au pair le plus proche, et le plus grand m donne exactement 1,0f.)
__device__ inline float u01f(unsigned int a) {
  return ((float)(a >> 9) + 0.5f) * (1.0f / 8388608.0f);
}

// ------------------------------------------------------------------------------------------
// Algèbre quaternionique float (x=a0, y=a1, z=a2, w=a3), mêmes expressions que le FP64
// ------------------------------------------------------------------------------------------
struct __attribute__((aligned(16))) QF { float x, y, z, w; };

__device__ inline QF qfmul(QF a, QF b) {
  QF c;
  c.x = a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w;
  c.y = a.x * b.y + b.x * a.y - (a.z * b.w - a.w * b.z);
  c.z = a.x * b.z + b.x * a.z - (a.w * b.y - a.y * b.w);
  c.w = a.x * b.w + b.x * a.w - (a.y * b.z - a.z * b.y);
  return c;
}
__device__ inline QF qfdag(QF a) { QF c = {a.x, -a.y, -a.z, -a.w}; return c; }
__device__ inline QF qfadd(QF a, QF b) { QF c = {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; return c; }
__device__ inline float qfnorm(QF a) { return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w); }
__device__ inline QF qfscale(QF a, float s) { QF c = {a.x * s, a.y * s, a.z * s, a.w * s}; return c; }
// Réunitarisation : q / |q| (division et racine IEEE).
__device__ inline QF qfunit(QF a) { return qfscale(a, 1.0f / qfnorm(a)); }
// Lecture float promue en double (mesures).
__device__ inline Q qdbl(QF a) { Q c = {(double)a.x, (double)a.y, (double)a.z, (double)a.w}; return c; }

__device__ inline QF ldf(const QF* __restrict__ U, const Geo& g, int mu, long s) { return U[(long)mu * g.V + s]; }

// Somme des 6 agrafes (même parcours que `staple` de su2_commun.h).
__device__ QF staplef(const QF* __restrict__ U, const Geo& g, int mu, long s, const int x[4]) {
  QF A = {0.0f, 0.0f, 0.0f, 0.0f};
  long smu = nb_up(g, s, mu, x[mu]);
  for (int nu = 0; nu < 4; ++nu) {
    if (nu == mu) continue;
    long snu = nb_up(g, s, nu, x[nu]);
    long sdn = nb_dn(g, s, nu, x[nu]);
    long smu_dn = nb_dn(g, smu, nu, x[nu]);
    QF t = qfmul(qfmul(ldf(U, g, nu, smu), qfdag(ldf(U, g, mu, snu))), qfdag(ldf(U, g, nu, s)));
    QF b = qfmul(qfmul(qfdag(ldf(U, g, nu, smu_dn)), qfdag(ldf(U, g, mu, sdn))), ldf(U, g, nu, sdn));
    A = qfadd(A, qfadd(t, b));
  }
  return A;
}

// Quaternion de Haar sur S³ à partir d'un bloc Philox (4 mots) : Box–Muller sur deux paires.
__device__ inline QF haarf(U4 r) {
  float u1 = u01f(r.v[0]), u2 = u01f(r.v[1]), u3 = u01f(r.v[2]), u4 = u01f(r.v[3]);
  float ra = sqrtf(-2.0f * logf(u1)), rb = sqrtf(-2.0f * logf(u3));
  QF q = {ra * cosf(KV32_2PI * u2), ra * sinf(KV32_2PI * u2), rb * cosf(KV32_2PI * u4), rb * sinf(KV32_2PI * u4)};
  return qfunit(q);
}

// ------------------------------------------------------------------------------------------
// Tirage exact de x0 selon sqrt(1-x0²) exp(alpha x0) sur [-1,1]
// ------------------------------------------------------------------------------------------
// alpha ≥ 2 : Kennedy–Pendleton ; alpha < 2 : Creutz (proposition exponentielle puis rejet) ;
// alpha < 1e-6 : proposition uniforme (la forme stable ci-dessous y est exacte, mais on évite 1/alpha).
// `draw` compte les appels Philox (un par essai) ; `ntry` les essais ; `ok` faux si le plafond de 10⁶
// essais est atteint (compté, jamais observé).
__device__ float sample_x0f(float alpha, unsigned int li, unsigned int sweep_id, unsigned int& draw, unsigned int k0,
                            unsigned int k1, unsigned int& ntry, bool& ok) {
  const unsigned int CAP = 1000000u;
  float x0 = 1.0f; ok = false; ntry = 0;
  if (alpha >= 2.0f) {
    const float deux_alpha = 2.0f * alpha;
    while (ntry < CAP) {
      U4 r = philox4x32_10(U4{{li, sweep_id, draw++, KV32_TAG_HB}}, k0, k1);
      float a1 = u01f(r.v[0]), a2 = u01f(r.v[1]), a3 = u01f(r.v[2]), a4 = u01f(r.v[3]);
      float c = cosf(KV32_2PI * a2);
      float lam2 = -(logf(a1) + c * c * logf(a3)) / deux_alpha;
      ++ntry;
      if (a4 * a4 <= 1.0f - lam2) { x0 = 1.0f - 2.0f * lam2; ok = true; break; }
    }
  } else if (alpha < 1e-6f) {
    while (ntry < CAP) {
      U4 r = philox4x32_10(U4{{li, sweep_id, draw++, KV32_TAG_HB}}, k0, k1);
      float y = 2.0f * u01f(r.v[0]) - 1.0f, a2 = u01f(r.v[1]);
      ++ntry;
      if (a2 * a2 <= 1.0f - y * y) { x0 = y; ok = true; break; }
    }
  } else {
    const float um = -expm1f(-2.0f * alpha);        // 1 - exp(-2 alpha), sans annulation
    while (ntry < CAP) {
      U4 r = philox4x32_10(U4{{li, sweep_id, draw++, KV32_TAG_HB}}, k0, k1);
      float a1 = u01f(r.v[0]), a2 = u01f(r.v[1]);
      float y = 1.0f + log1pf(-a1 * um) / alpha;    // forme stable de 1 + log(e2 + a1 (1 - e2)) / alpha
      ++ntry;
      if (a2 * a2 <= 1.0f - y * y) { x0 = y; ok = true; break; }
    }
  }
  return x0;
}

#ifdef K1_MUTANT_METRO
// MUTANT (contrôle négatif, jamais en production) : UN pas de Metropolis naïf sur x0 depuis x_cur,
// proposition uniforme sur [-1,1]. La chaîne reste valide (bilan détaillé) mais le tirage n'est PAS exact.
__device__ float sample_x0f_mut(float alpha, float x_cur, unsigned int li, unsigned int sweep_id, unsigned int& draw,
                                unsigned int k0, unsigned int k1, unsigned int& ntry, bool& ok) {
  U4 r = philox4x32_10(U4{{li, sweep_id, draw++, KV32_TAG_HB}}, k0, k1);
  float y = 2.0f * u01f(r.v[0]) - 1.0f;
  float pc = sqrtf(fmaxf(0.0f, 1.0f - x_cur * x_cur));
  float py = sqrtf(fmaxf(0.0f, 1.0f - y * y)) * expf(alpha * (y - x_cur));
  ntry = 1; ok = true;
  return (u01f(r.v[1]) * pc <= py) ? y : x_cur;
}
#endif

// Test unitaire du tirage : n tirages indépendants de x0 à alpha fixé (même fonction que le bain thermique).
extern "C" __global__ void k_test_x0_f32(float alpha, long n, unsigned int k0, unsigned int k1, float* out,
                                         unsigned long long* n_cap) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned int draw = 0, ntry = 0; bool ok = false;
#ifdef K1_MUTANT_METRO
  float x0 = sample_x0f_mut(alpha, 1.0f, (unsigned int)i, 0x7E57u, draw, k0, k1, ntry, ok);
#else
  float x0 = sample_x0f(alpha, (unsigned int)i, 0x7E57u, draw, k0, k1, ntry, ok);
#endif
  if (!ok) atomicAdd(n_cap, 1ull);
  out[i] = x0;
}

// ------------------------------------------------------------------------------------------
// Noyaux de mise à jour (grille : x = indices de sites, y = chaîne du lot)
// ------------------------------------------------------------------------------------------

// Départ : froid (identité) ou chaud (Haar). Compteur (li, 0xFFFFFFFF, 0, INIT), clé de la chaîne.
// `departs[c]` = 1 : chaud. `cles[2c], cles[2c+1]` = clé Philox.
extern "C" __global__ void k_init_f32(QF* U, int Lg, const unsigned int* departs, const unsigned int* cles) {
  Geo g = mkgeo(Lg);
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 4 * g.V) return;
  const int c = blockIdx.y;
  U += (long)c * 4 * g.V;
  if (!departs[c]) { QF e = {1.0f, 0.0f, 0.0f, 0.0f}; U[i] = e; return; }
  U4 r = philox4x32_10(U4{{(unsigned int)i, 0xFFFFFFFFu, 0u, KV32_TAG_INIT}}, cles[2 * c], cles[2 * c + 1]);
  U[i] = haarf(r);
}

// Bain thermique sur les sites de parité `parity`, direction `mu`. `gel` : les liens spatiaux de
// l'hyperplan x0 = 0 sont gelés (R2-3, comme su2_bord.cu). Compteurs par chaîne : n_cap[c] (plafond
// atteint, attendu 0) et n_tries[c] (essais cumulés).
extern "C" __global__ void k_heatbath_f32(QF* U, int Lg, int mu, int parity, const float* betas,
                                          const unsigned int* cles, unsigned int sweep_id, int gel,
                                          unsigned long long* n_cap, unsigned long long* n_tries) {
  Geo g = mkgeo(Lg);
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= g.V / 2) return;
  const int c = blockIdx.y;
  U += (long)c * 4 * g.V;
  const unsigned int k0 = cles[2 * c], k1 = cles[2 * c + 1];
  int x[4];
  long s = site_of(g, t, parity, x);
  if (gel && mu != 0 && x[0] == 0) return;     // lien de bord gelé
  long li = (long)mu * g.V + s;
  QF A = staplef(U, g, mu, s, x);
  float k = qfnorm(A);
  unsigned int draw = 0;
  if (!(k > 1e-30f)) {   // agrafe nulle : mesure de Haar (jamais observé en pratique)
    U[li] = haarf(philox4x32_10(U4{{(unsigned int)li, sweep_id, draw++, KV32_TAG_HB}}, k0, k1));
    return;
  }
  QF Vv = qfscale(A, 1.0f / k);
  float alpha = betas[c] * k;       // densité de x0 : sqrt(1-x0²) exp(alpha x0)
  unsigned int ntry = 0; bool ok = false;
#ifdef K1_MUTANT_METRO
  float x0 = sample_x0f_mut(alpha, qfmul(U[li], Vv).x, (unsigned int)li, sweep_id, draw, k0, k1, ntry, ok);
#else
  float x0 = sample_x0f(alpha, (unsigned int)li, sweep_id, draw, k0, k1, ntry, ok);
#endif
  atomicAdd(n_tries + c, (unsigned long long)ntry);
  if (!ok) { atomicAdd(n_cap + c, 1ull); return; }   // compté ; lien inchangé (attendu : 0)
  U4 rd = philox4x32_10(U4{{(unsigned int)li, sweep_id, draw++, KV32_TAG_HB}}, k0, k1);
  float ct = 2.0f * u01f(rd.v[0]) - 1.0f;
  float ph = KV32_2PI * u01f(rd.v[1]);
  float st = sqrtf(fmaxf(0.0f, 1.0f - ct * ct));
  float rr = sqrtf(fmaxf(0.0f, 1.0f - x0 * x0));
  QF W = {x0, rr * st * cosf(ph), rr * st * sinf(ph), rr * ct};
  U[li] = qfunit(qfmul(W, qfdag(Vv)));          // X = U V suit exp(alpha x0) dHaar  =>  U = X V⁺
}

// Overrelaxation U → V⁺ U⁺ V⁺ (aucun compteur Philox consommé).
extern "C" __global__ void k_overrelax_f32(QF* U, int Lg, int mu, int parity, int gel) {
  Geo g = mkgeo(Lg);
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= g.V / 2) return;
  const int c = blockIdx.y;
  U += (long)c * 4 * g.V;
  int x[4];
  long s = site_of(g, t, parity, x);
  if (gel && mu != 0 && x[0] == 0) return;
  long li = (long)mu * g.V + s;
  QF A = staplef(U, g, mu, s, x);
  float k = qfnorm(A);
  if (!(k > 1e-30f)) return;
  QF Vd = qfdag(qfscale(A, 1.0f / k));
  U[li] = qfunit(qfmul(qfmul(Vd, qfdag(U[li])), Vd));   // ReTr(U'A) = ReTr(UA)
}

// ------------------------------------------------------------------------------------------
// Mesures : lecture float promue en double, produits et réductions en double, arbre fixe
// ------------------------------------------------------------------------------------------

// Somme par site des 6 plaquettes 1/2 ReTr U_p (en double). out : V valeurs par chaîne.
extern "C" __global__ void k_plaq_site_f32(const QF* __restrict__ U, int Lg, double* out) {
  Geo g = mkgeo(Lg);
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= g.V) return;
  const int c = blockIdx.y;
  U += (long)c * 4 * g.V;
  out += (long)c * g.V;
  int x[4]; coords(g, s, x);
  double acc = 0;
  for (int mu = 0; mu < 4; ++mu) {
    long smu = nb_up(g, s, mu, x[mu]);
    for (int nu = mu + 1; nu < 4; ++nu) {
      long snu = nb_up(g, s, nu, x[nu]);
      Q p = qmul(qmul(qdbl(ldf(U, g, mu, s)), qdbl(ldf(U, g, nu, smu))),
                 qmul(qdag(qdbl(ldf(U, g, mu, snu))), qdag(qdbl(ldf(U, g, nu, s)))));
      acc += p.x;
    }
  }
  out[s] = acc;
}

// Réduction par bloc (256 fils, arbre fixe). Chaîne c : lit in + c·istride, écrit out[c·ostride + bloc].
extern "C" __global__ void k_reduce_f32(const double* __restrict__ in, long n, long istride, double* out,
                                        long ostride) {
  __shared__ double sh[256];
  const int c = blockIdx.y;
  in += (long)c * istride;
  long i = (long)blockIdx.x * 256 + threadIdx.x;
  sh[threadIdx.x] = (i < n) ? in[i] : 0.0;
  __syncthreads();
  for (int w = 128; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[(long)c * ostride + blockIdx.x] = sh[0];
}

// Somme finale déterministe (1 bloc de 256 par chaîne) : chaque fil somme in[t], in[t+256], … dans l'ordre,
// puis arbre fixe ; écrit sum·scale dans out[c·ostride].
extern "C" __global__ void k_final_f32(const double* __restrict__ in, long n, long istride, double scale,
                                       double* out, long ostride) {
  __shared__ double sh[256];
  const int c = blockIdx.y;
  in += (long)c * istride;
  double a = 0;
  for (long i = threadIdx.x; i < n; i += 256) a += in[i];
  sh[threadIdx.x] = a;
  __syncthreads();
  for (int w = 128; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[(long)c * ostride] = sh[0] * scale;
}

// Boucles de Polyakov 1/2 Tr prod_k U_mu(x + k e_mu), pour chaque mu et chaque site transverse (en double).
// out : 4·L³ valeurs par chaîne.
extern "C" __global__ void k_polyakov_f32(const QF* __restrict__ U, int Lg, double* out) {
  Geo g = mkgeo(Lg);
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long L3 = g.V / g.L;
  if (i >= 4 * L3) return;
  const int c = blockIdx.y;
  U += (long)c * 4 * g.V;
  out += (long)c * 4 * L3;
  int mu = (int)(i / L3);
  long j = i % L3;
  int x[4]; x[mu] = 0;
  int dirs[3], nd = 0;
  for (int d = 0; d < 4; ++d) if (d != mu) dirs[nd++] = d;
  x[dirs[2]] = (int)(j % g.L); long t = j / g.L;
  x[dirs[1]] = (int)(t % g.L); x[dirs[0]] = (int)(t / g.L);
  long s = ((long)(x[0] * g.L + x[1]) * g.L + x[2]) * g.L + x[3];
  Q P = {1, 0, 0, 0};
  for (int k = 0; k < g.L; ++k) { P = qmul(P, qdbl(ldf(U, g, mu, s))); s += (k == g.L - 1) ? 0 : g.st[mu]; }
  out[i] = P.x;
}

// Par site : somme des 3 plaquettes spatiales (plans 12, 13, 23) et des 3 temporelles (01, 02, 03), 1/2 ReTr.
extern "C" __global__ void k_plaq_st_f32(const QF* __restrict__ U, int Lg, double* outS, double* outT) {
  Geo g = mkgeo(Lg);
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= g.V) return;
  const int c = blockIdx.y;
  U += (long)c * 4 * g.V;
  outS += (long)c * g.V; outT += (long)c * g.V;
  int x[4]; coords(g, s, x);
  double aS = 0, aT = 0;
  for (int mu = 0; mu < 4; ++mu) {
    long smu = nb_up(g, s, mu, x[mu]);
    for (int nu = mu + 1; nu < 4; ++nu) {
      long snu = nb_up(g, s, nu, x[nu]);
      Q p = qmul(qmul(qdbl(ldf(U, g, mu, s)), qdbl(ldf(U, g, nu, smu))),
                 qmul(qdag(qdbl(ldf(U, g, mu, snu))), qdag(qdbl(ldf(U, g, nu, s)))));
      if (mu == 0) aT += p.x; else aS += p.x;
    }
  }
  outS[s] = aS; outT[s] = aT;
}

// Un bloc par tranche t (sites contigus [t L³, (t+1) L³)) et par chaîne ; somme déterministe ;
// out[c·ostride + t] = somme · scale.
extern "C" __global__ void k_tranche_f32(const double* __restrict__ in, int Lg, long istride, double scale,
                                         double* out, long ostride) {
  __shared__ double sh[256];
  Geo g = mkgeo(Lg);
  const int c = blockIdx.y;
  long L3 = g.V / g.L; long base = (long)c * istride + (long)blockIdx.x * L3;
  double a = 0;
  for (long i = threadIdx.x; i < L3; i += 256) a += in[base + i];
  sh[threadIdx.x] = a;
  __syncthreads();
  for (int w = 128; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[(long)c * ostride + blockIdx.x] = sh[0] * scale;
}
