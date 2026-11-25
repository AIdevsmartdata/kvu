// su2_geo.cu — chemin C1 du générateur SU(2) sur GPU (kvu v0.2, B10) : réseau régulier à 4 côtés pairs, coefficients par
// plaquette (torsion de 't Hooft × β par plan) et masque de liens gelés, pilotés par les `Tables` du socle `geom`.
//
// Assemblage (module « su2_geo64 », src/su2/geo_gpu.rs) : prélude du backend, puis su2_commun.h, su2_ref64.cu, ce fichier.
// su2_ref64.cu est inclus pour RÉUTILISER sans copie sample_x0, k_reduce et k_final (mêmes arbres de réduction). Le texte
// assemblé du chemin C0 (su2_commun.h + su2_ref64.cu + su2_bord.cu) n'est PAS modifié : ce fichier est embarqué à part
// (include_str! dans geo_gpu.rs), hors du registre gelé crate::noyaux::TOUS, comme su2_boite.cu.
//
// GÉOMÉTRIE : tampon `geo` de 9 entiers 64 bits copiés des Tables (aucune formule de pas sur GPU) :
//   geo[0..4] = ℓ0..ℓ3, geo[4] = V, geo[5..9] = pas0..pas3 ; site s = ((x0ℓ1+x1)ℓ2+x2)ℓ3+x3, lien li = mu·V + s.
//   Voisins avec repli périodique et énumération d'une parité : vérifiés exhaustivement contre Tables (k_geo_test, porte G-GEO).
//   Côtés TOUS PAIRS (l'énumération s = 2t (+1) de g_site_of exige ℓ3 pair ; le damier exige tous les côtés pairs).
// COEFFICIENTS : coef[p·V + s] (p = indice du plan {mu,nu} dans l'ordre 01 02 03 12 13 23, s = coin de la plaquette) ;
//   pointeur nul ⇒ 1 partout (et EXACTEMENT les expressions de staple / k_plaq_site de C0).
//   Agrafe : A = Σ_{ν≠μ} (z_t·t + z_b·b), z_t au coin s, z_b au coin s − ν̂ (= torsion::agrafe_tordue et tube::agrafe).
// MASQUE : un octet par lien (indice local li), partagé par les chaînes du lot ; non nul ⇒ lien gelé ; nul ⇒ aucun.
// PHILOX : compteurs identiques au chemin C0 et au CPU : (li, sweep_id, draw, TAG_HB) ; départ (li, 0xFFFFFFFF, 0|1, TAG_INIT).
// LOTS : blockIdx.y = chaîne c ; champ de la chaîne c en U + c·4V ; betas[], k0s[], k1s[], hot[], n_cap[], n_tries[] par chaîne.
// MUTANTS (contrôles négatifs des portes P3/P4, JAMAIS en production) : KVU_GEO_MUTANT = 1 : z lu dans le mauvais plan (5 − p) ;
//   2 : pile décalée d'un site (z lu au coin + μ̂) ; 3 : z_b oublié (plaquette b sans coefficient) ; 4 : z ignoré dans la plaquette
//   mesurée (k_plaq_site_geo) ; 5 : z_b lu au coin s au lieu de s − ν̂ (contrat de Tables, relecture c04-geom).
//
// NOMS extern "C" (utilisés par src/su2/geo_gpu.rs) :
//   k_geo_test       (const long long* geo, u32* voisins, u32* sites)          grille ceil(V/256), bloc 256
//                    voisins[8s + i] : haut μ = i (i < 4), bas μ = i − 4 ; sites[p·(V/2) + t] = g_site_of(t, p)
//   k_init_geo       (Q* U, const long long* geo, const int* hot, const u32* k0s, const u32* k1s)   grille (ceil(4V/256), nch)
//   k_heatbath_geo   (Q* U, const long long* geo, int mu, int parity, const double* betas, const u32* k0s, const u32* k1s,
//                     u32 sweep_id, unsigned long long* n_cap, unsigned long long* n_tries, u32* diag_ntry,
//                     const double* coef, const unsigned char* masque)                              grille (ceil(V/2/256), nch)
//   k_overrelax_geo  (Q* U, const long long* geo, int mu, int parity, const double* coef, const unsigned char* masque)
//                                                                                                     grille (ceil(V/2/256), nch)
//   k_plaq_site_geo  (const Q* U, const long long* geo, const double* coef, double* out)               grille (ceil(V/256), nch)
//                    out : V valeurs par chaîne (somme des 6 plaquettes z·½ReTr U_p du site) ; puis k_reduce / k_final (su2_ref64.cu)
//   k_polyakov_geo   (const Q* U, const long long* geo, int mu, double* out, long out_pas)           grille (ceil((V/ℓμ)/256), nch)
//                    out[c·out_pas + j] = ½Tr de la boucle de Polyakov brute dans la direction mu, site de base j (mêmes ordre et
//                    produit que k_polyakov : directions restantes d0 < d1 < d2, x[d2] le plus rapide)

#ifndef KVU_GEO_MUTANT
#define KVU_GEO_MUTANT 0
#endif

#ifndef KVU_COUTURE
struct GeoG { long l[4]; long V; long st[4]; };

__device__ inline GeoG g_lire(const long long* __restrict__ geo) {
  GeoG g;
  for (int d = 0; d < 4; ++d) { g.l[d] = (long)geo[d]; g.st[d] = (long)geo[5 + d]; }
  g.V = (long)geo[4];
  return g;
}

__device__ inline long g_up(const GeoG& g, long s, int d, int xd) {
  return (xd == g.l[d] - 1) ? s - (g.l[d] - 1) * g.st[d] : s + g.st[d];
}
__device__ inline long g_dn(const GeoG& g, long s, int d, int xd) {
  return (xd == 0) ? s + (g.l[d] - 1) * g.st[d] : s - g.st[d];
}
#else
// COUTURE DÉCALÉE (option de compilation KVU_COUTURE, définie par les modules « une boîte » à couture seulement) : le tour de l'axe
// `cax` ramène le champ décalé de `cv` (réduit modulo les côtés ; geo[14 + geo[13]] = axe, puis les 4 composantes). Les voisins sont
// recalculés depuis le site SEUL (l'argument `xd` est ignoré : après un tour de l'axe, la coordonnée d'un voisin diffère de celle du
// site de départ) ; mêmes formules que `GeoA::voisin_couture` (geom/reseau.rs).
struct GeoG { long l[4]; long V; long st[4]; long cax; long cv[4]; };

__device__ inline GeoG g_lire(const long long* __restrict__ geo) {
  GeoG g;
  for (int d = 0; d < 4; ++d) { g.l[d] = (long)geo[d]; g.st[d] = (long)geo[5 + d]; }
  g.V = (long)geo[4];
  const long long* c = geo + 14 + geo[13];
  g.cax = (long)c[0];
  for (int d = 0; d < 4; ++d) g.cv[d] = (long)c[1 + d];
  return g;
}
__device__ inline long g_decale(const GeoG& g, long b, int signe) {
  long r = 0;
  for (int d = 0; d < 4; ++d) {
    long xd = (b / g.st[d]) % g.l[d];
    long y = signe >= 0 ? (xd + g.cv[d]) % g.l[d] : (xd + g.l[d] - g.cv[d]) % g.l[d];
    r += y * g.st[d];
  }
  return r;
}
__device__ inline long g_up(const GeoG& g, long s, int d, int xd) {
  long x = (s / g.st[d]) % g.l[d];
  if (x == g.l[d] - 1) {
    long b = s - (g.l[d] - 1) * g.st[d];
    return d == g.cax ? g_decale(g, b, 1) : b;
  }
  return s + g.st[d];
}
__device__ inline long g_dn(const GeoG& g, long s, int d, int xd) {
  long x = (s / g.st[d]) % g.l[d];
  if (x == 0) {
    long b = s + (g.l[d] - 1) * g.st[d];
    return d == g.cax ? g_decale(g, b, -1) : b;
  }
  return s - g.st[d];
}
#endif
__device__ inline void g_coords(const GeoG& g, long s, int x[4]) {
  x[3] = (int)(s % g.l[3]); long t = s / g.l[3];
  x[2] = (int)(t % g.l[2]); t /= g.l[2];
  x[1] = (int)(t % g.l[1]); x[0] = (int)(t / g.l[1]);
}
__device__ inline long g_site_of(const GeoG& g, long t, int parity, int x[4]) {
  long s = 2 * t;
  g_coords(g, s, x);
  int p = (x[0] + x[1] + x[2] + x[3]) & 1;
  if (p != parity) { s += 1; x[3] += 1; }
  return s;
}

// COLORIAGE A TROIS CLASSES (côtés impairs ; option de compilation KVU_LISTE_CLASSES, définie par les modules « une boîte » seulement).
// Disposition du tampon geo : [0..9) inchangé ; geo[9] = nombre de classes (0 : damier de parité, chemin historique ; 3 : liste) ;
// geo[10..14) décalages des classes dans la liste ; geo[14 ..) sites triés par classe puis par indice. Un lien (μ, s) ne dépend que de
// la classe de s : la clé Philox (indice de lien) est INCHANGÉE, donc le résultat ne dépend pas de l'ordre dans la classe.
#ifdef KVU_LISTE_CLASSES
__device__ inline long g_nsel(const long long* __restrict__ geo, int cls, long V) {
  return geo[9] ? (long)(geo[11 + cls] - geo[10 + cls]) : V / 2;
}
__device__ inline long g_site_sel(const GeoG& g, const long long* __restrict__ geo, long t, int cls, int x[4]) {
  if (geo[9]) { long s = (long)geo[14 + geo[10 + cls] + t]; g_coords(g, s, x); return s; }
  return g_site_of(g, t, cls, x);
}
#else
__device__ inline long g_nsel(const long long* __restrict__ geo, int cls, long V) { return V / 2; }
__device__ inline long g_site_sel(const GeoG& g, const long long* __restrict__ geo, long t, int cls, int x[4]) {
  return g_site_of(g, t, cls, x);
}
#endif
// Indice du plan {mu, nu} (mu ≠ nu) dans l'ordre 01 02 03 12 13 23 (= geom::indice_plan).
__device__ inline int g_plan(int mu, int nu) {
  int a = mu < nu ? mu : nu, b = mu < nu ? nu : mu;
  return a == 0 ? b - 1 : (a == 1 ? b + 1 : 5);
}
__device__ inline Q g_ld(const Q* __restrict__ U, const GeoG& g, int mu, long s) { return U[(long)mu * g.V + s]; }

// Agrafe du lien (s, mu), coefficients par plaquette : mêmes produits et même ordre que staple (su2_commun.h).
__device__ Q g_agrafe(const Q* __restrict__ U, const GeoG& g, int mu, long s, const int x[4],
                      const double* __restrict__ coef) {
  Q A = {0, 0, 0, 0};
  long smu = g_up(g, s, mu, x[mu]);
  for (int nu = 0; nu < 4; ++nu) {
    if (nu == mu) continue;
    long snu = g_up(g, s, nu, x[nu]);
    long sdn = g_dn(g, s, nu, x[nu]);
    long smu_dn = g_dn(g, smu, nu, x[nu]);
    Q t = qmul(qmul(g_ld(U, g, nu, smu), qdag(g_ld(U, g, mu, snu))), qdag(g_ld(U, g, nu, s)));
    Q b = qmul(qmul(qdag(g_ld(U, g, nu, smu_dn)), qdag(g_ld(U, g, mu, sdn))), g_ld(U, g, nu, sdn));
    if (coef) {
      int p = g_plan(mu, nu);
      long it = s, ib = sdn;
#if KVU_GEO_MUTANT == 1
      p = 5 - p;
#endif
#if KVU_GEO_MUTANT == 2
      it = smu; ib = smu_dn;
#endif
#if KVU_GEO_MUTANT == 5
      ib = s;
#endif
      double zt = coef[(long)p * g.V + it], zb = coef[(long)p * g.V + ib];
#if KVU_GEO_MUTANT == 3
      zb = 1.0;
#endif
      A = qadd(A, qadd(qscale(t, zt), qscale(b, zb)));
    } else {
      A = qadd(A, qadd(t, b));
    }
  }
  return A;
}

extern "C" __global__ void k_geo_test(const long long* geo, u32* voisins, u32* sites) {
  GeoG g = g_lire(geo);
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= g.V) return;
  int x[4]; g_coords(g, s, x);
  for (int mu = 0; mu < 4; ++mu) {
    voisins[8 * s + mu] = (u32)g_up(g, s, mu, x[mu]);
    voisins[8 * s + 4 + mu] = (u32)g_dn(g, s, mu, x[mu]);
  }
  long h = g.V / 2;
  int y[4];
  sites[s] = (u32)g_site_of(g, s % h, (int)(s / h), y);
}

extern "C" __global__ void k_init_geo(Q* U, const long long* geo, const int* hot, const u32* k0s, const u32* k1s) {
  GeoG g = g_lire(geo);
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 4 * g.V) return;
  const int c = blockIdx.y;
  U += (long)c * 4 * g.V;
  const u32 k0 = k0s[c], k1 = k1s[c];
  if (!hot[c]) { Q e = {1, 0, 0, 0}; U[i] = e; return; }
  U4 c1 = {{(u32)i, 0xFFFFFFFFu, 0u, TAG_INIT}}, c2 = {{(u32)i, 0xFFFFFFFFu, 1u, TAG_INIT}};
  U4 r1 = philox4x32_10(c1, k0, k1), r2 = philox4x32_10(c2, k0, k1);
  double u1 = u01(r1.v[0], r1.v[1]), u2 = u01(r1.v[2], r1.v[3]);
  double u3 = u01(r2.v[0], r2.v[1]), u4 = u01(r2.v[2], r2.v[3]);
  double ra = sqrt(-2.0 * log(u1)), rb = sqrt(-2.0 * log(u3));
  Q q = {ra * cos(2 * M_PI * u2), ra * sin(2 * M_PI * u2), rb * cos(2 * M_PI * u4), rb * sin(2 * M_PI * u4)};
  U[i] = qscale(q, 1.0 / qnorm(q));   // Haar sur S^3
}

extern "C" __global__ void k_heatbath_geo(Q* U, const long long* geo, int mu, int parity, const double* betas,
                                          const u32* k0s, const u32* k1s, u32 sweep_id, unsigned long long* n_cap,
                                          unsigned long long* n_tries, u32* diag_ntry, const double* __restrict__ coef,
                                          const unsigned char* __restrict__ masque) {
  GeoG g = g_lire(geo);
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= g_nsel(geo, parity, g.V)) return;
  const int c = blockIdx.y;                  // chaîne du lot
  U += (long)c * 4 * g.V;
  if (diag_ntry) diag_ntry += (long)c * 4 * g.V;
  const double beta = betas[c];
  const u32 k0 = k0s[c], k1 = k1s[c];
  int x[4];
  long s = g_site_sel(g, geo, t, parity, x);
  long li = (long)mu * g.V + s;              // indice de lien LOCAL (compteur Philox identique à la chaîne seule)
  if (masque && masque[li]) return;          // lien gelé par le masque
  Q A = g_agrafe(U, g, mu, s, x, coef);
  double k = qnorm(A);
  u32 draw = 0;
  Q W;
  if (!(k > 1e-300)) {   // agrafe nulle : mesure de Haar
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
  double alpha = beta * k;          // densité de x0 : sqrt(1-x0^2) exp(alpha x0)
  u32 ntry = 0; bool ok = false;
  double x0 = sample_x0(alpha, (u32)li, sweep_id, draw, k0, k1, ntry, ok);
  atomicAdd(n_tries + c, (unsigned long long)ntry);
  if (diag_ntry) diag_ntry[li] = ok ? ntry : 0u;
  if (!ok) { atomicAdd(n_cap + c, 1ull); return; }   // compté ; lien inchangé (attendu : 0)
  U4 rd = philox4x32_10(U4{{(u32)li, sweep_id, draw++, TAG_HB}}, k0, k1);
  double ct = 2.0 * u01(rd.v[0], rd.v[1]) - 1.0;
  double ph = 2 * M_PI * u01(rd.v[2], rd.v[3]);
  double st = sqrt(fmax(0.0, 1.0 - ct * ct));
  double rr = sqrt(fmax(0.0, 1.0 - x0 * x0));
  W.x = x0; W.y = rr * st * cos(ph); W.z = rr * st * sin(ph); W.w = rr * ct;
  Q Un = qmul(W, qdag(Vv));          // X = U V suit exp(alpha x0) dHaar  =>  U = X V^+
  U[li] = qscale(Un, 1.0 / qnorm(Un));
}

extern "C" __global__ void k_overrelax_geo(Q* U, const long long* geo, int mu, int parity,
                                           const double* __restrict__ coef, const unsigned char* __restrict__ masque) {
  GeoG g = g_lire(geo);
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= g_nsel(geo, parity, g.V)) return;
  U += (long)blockIdx.y * 4 * g.V;           // champ de la chaîne
  int x[4];
  long s = g_site_sel(g, geo, t, parity, x);
  long li = (long)mu * g.V + s;
  if (masque && masque[li]) return;          // lien gelé par le masque
  Q A = g_agrafe(U, g, mu, s, x, coef);
  double k = qnorm(A);
  if (!(k > 1e-300)) return;
  Q Vd = qdag(qscale(A, 1.0 / k));
  Q Un = qmul(qmul(Vd, qdag(U[li])), Vd);   // U' = V^+ U^+ V^+  (ReTr(U'A) = ReTr(UA))
  U[li] = qscale(Un, 1.0 / qnorm(Un));
}

// somme par site des 6 plaquettes z·½ReTr U_p (coin au site) ; out : V valeurs par chaîne
extern "C" __global__ void k_plaq_site_geo(const Q* __restrict__ U, const long long* geo, const double* __restrict__ coef,
                                           double* out) {
  GeoG g = g_lire(geo);
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= g.V) return;
  U += (long)blockIdx.y * 4 * g.V;
  out += (long)blockIdx.y * g.V;
  int x[4]; g_coords(g, s, x);
  double acc = 0;
  for (int mu = 0; mu < 4; ++mu) {
    long smu = g_up(g, s, mu, x[mu]);
    for (int nu = mu + 1; nu < 4; ++nu) {
      long snu = g_up(g, s, nu, x[nu]);
      Q p = qmul(qmul(g_ld(U, g, mu, s), g_ld(U, g, nu, smu)), qmul(qdag(g_ld(U, g, mu, snu)), qdag(g_ld(U, g, nu, s))));
#if KVU_GEO_MUTANT == 4
      acc += p.x;
#else
      if (coef) acc += coef[(long)g_plan(mu, nu) * g.V + s] * p.x;
      else acc += p.x;
#endif
    }
  }
  out[s] = acc;
}

// Boucles de Polyakov ½Tr Π_k U_mu(x + k e_mu) de la direction mu, une par site de base (x_mu = 0)
extern "C" __global__ void k_polyakov_geo(const Q* __restrict__ U, const long long* geo, int mu, double* out, long out_pas) {
  GeoG g = g_lire(geo);
  long n = g.V / g.l[mu];
  long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  U += (long)blockIdx.y * 4 * g.V;
  out += (long)blockIdx.y * out_pas;
  int x[4]; x[mu] = 0;
  int dirs[3], nd = 0;
  for (int d = 0; d < 4; ++d) if (d != mu) dirs[nd++] = d;
  x[dirs[2]] = (int)(j % g.l[dirs[2]]); long t = j / g.l[dirs[2]];
  x[dirs[1]] = (int)(t % g.l[dirs[1]]); x[dirs[0]] = (int)(t / g.l[dirs[1]]);
  long s = x[0] * g.st[0] + x[1] * g.st[1] + x[2] * g.st[2] + x[3] * g.st[3];
  Q P = {1, 0, 0, 0};
  for (long k = 0; k < g.l[mu]; ++k) { P = qmul(P, g_ld(U, g, mu, s)); s += (k == g.l[mu] - 1) ? 0 : g.st[mu]; }
  out[j] = P.x;
}
