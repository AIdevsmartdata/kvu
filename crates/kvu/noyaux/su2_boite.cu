// su2_boite.cu — mesures des boîtes de Dirichlet ℓ⁴ plongées dans un réseau de calcul périodique de côté Lc = ℓ + 2
// (propriétaire : WP-DIR).
//
// Assemblage : prélude du backend, puis su2_commun.h, puis ce fichier (module « boite », distinct du module « su2_ref64 »).
// Le texte est compilé tel quel par hiprtc (gfx1151) et par NVRTC (sm_120), SANS contraction multiplication-addition
// (OptionsCompil::reference_k1) : mêmes arrondis que la référence hôte `su2::boite::mesurer_hote`.
//
// Géométrie : la boîte occupe les coordonnées [0, ℓ]⁴ du réseau de calcul (Lg = ℓ + 2, pair). Toute plaquette mesurée a ses
// coordonnées dans [0, ℓ] : aucun repliement périodique n'est jamais utilisé (x + e_μ ≤ ℓ < Lg).
//
// ORDRES DE SOMMATION (contractuels, égaux à ceux de l'hôte, GATES-DIR §1 et crates/kvu/src/su2/boite.rs) :
//   P_c  : 24 plaquettes, plan ip = 0..5 (01 02 03 12 13 23), puis a ∈ {0,1}, puis b ∈ {0,1} (b le plus rapide), de coin
//          c − a e_μ − b e_ν ; somme séquentielle depuis 0.0 puis division par 24.0.
//   W2   : 6 boucles 2×2 de coin inférieur c − e_μ − e_ν (2 pas +μ, 2 pas +ν, 2 pas −μ, 2 pas −ν ; produit séquentiel de
//          gauche à droite depuis l'identité) ; somme séquentielle depuis 0.0 puis division par 6.0.
//   profil : chaque fil t somme les entrées t, t + 256, t + 512, … de la classe (ordre de la liste) puis arbre fixe à 256 fils
//          (identique à k_final de su2_ref64.cu) ; la moyenne (division par le nombre de plaquettes) est faite par l'hôte.
//
// NOMS extern "C" (utilisés par src/su2/boite_gpu.rs) :
//   k_boite_centre (const Q* U, int Lg, double* out, long out_pas, long idx)
//        grille (1, nch), bloc 32 : out[c·out_pas + 2·idx + {0,1}] = {P_c, W2} de la chaîne c (champ de la chaîne c : U + c·4V).
//   k_boite_profil (const Q* U, int Lg, const u32* entrees, const u32* debuts, double* out, long out_pas, long idx, int ncl)
//        grille (ncl, nch), bloc 256 : out[c·out_pas + idx·ncl + d] = Σ ½Tr des plaquettes de la classe d = blockIdx.x.
//        entrees[i] = (site de base)·8 + indice de plan ; debuts : ncl + 1 décalages.

// Plans (μ, ν) dans l'ordre 01 02 03 12 13 23 (sans table constante : le runtime n'en fournit pas).
__device__ inline int boite_mu(int ip) { return ip < 3 ? 0 : (ip < 5 ? 1 : 2); }
__device__ inline int boite_nu(int ip) { return ip < 3 ? ip + 1 : (ip == 3 ? 2 : 3); }

__device__ inline long boite_site(const Geo& g, const int x[4]) {
  return (((long)x[0] * g.L + x[1]) * g.L + x[2]) * g.L + x[3];
}

// ½Tr de la plaquette de base x dans le plan (mu, nu) : même association que k_plaq_site.
__device__ inline double boite_plaq(const Q* __restrict__ U, const Geo& g, int mu, int nu, const int x[4]) {
  long s = boite_site(g, x);
  long smu = s + g.st[mu];
  long snu = s + g.st[nu];
  Q p = qmul(qmul(ld(U, g, mu, s), ld(U, g, nu, smu)), qmul(qdag(ld(U, g, mu, snu)), qdag(ld(U, g, nu, s))));
  return p.x;
}

// ½Tr de la boucle a×a de coin inférieur x dans le plan (mu, nu) : produit séquentiel depuis l'identité.
__device__ double boite_boucle(const Q* __restrict__ U, const Geo& g, int mu, int nu, const int x0[4], int a) {
  Q q = {1.0, 0.0, 0.0, 0.0};
  int y[4] = {x0[0], x0[1], x0[2], x0[3]};
  for (int k = 0; k < a; ++k) { q = qmul(q, ld(U, g, mu, boite_site(g, y))); y[mu] += 1; }
  for (int k = 0; k < a; ++k) { q = qmul(q, ld(U, g, nu, boite_site(g, y))); y[nu] += 1; }
  for (int k = 0; k < a; ++k) { y[mu] -= 1; q = qmul(q, qdag(ld(U, g, mu, boite_site(g, y)))); }
  for (int k = 0; k < a; ++k) { y[nu] -= 1; q = qmul(q, qdag(ld(U, g, nu, boite_site(g, y)))); }
  return q.x;
}

extern "C" __global__ void k_boite_centre(const Q* __restrict__ U, int Lg, double* out, long out_pas, long idx) {
  __shared__ double sh[32];
  Geo g = mkgeo(Lg);
  const int c = blockIdx.y;
  U += (long)c * 4 * g.V;
  out += (long)c * out_pas + 2 * idx;
  const int ell = Lg - 2;
  const int ctr = ell / 2;
  const int t = threadIdx.x;
  double v = 0.0;
  if (t < 24) {
    // plaquette n° t : plan ip = t / 4, a = (t / 2) % 2, b = t % 2
    const int ip = t / 4, a = (t / 2) & 1, b = t & 1;
    const int mu = boite_mu(ip), nu = boite_nu(ip);
    int x[4] = {ctr, ctr, ctr, ctr};
    x[mu] -= a; x[nu] -= b;
    v = boite_plaq(U, g, mu, nu, x);
  } else if (t < 30) {
    const int ip = t - 24;
    const int mu = boite_mu(ip), nu = boite_nu(ip);
    int x[4] = {ctr, ctr, ctr, ctr};
    x[mu] -= 1; x[nu] -= 1;
    v = (ell >= 4) ? boite_boucle(U, g, mu, nu, x, 2) : 0.0;
  }
  sh[t] = v;
  __syncthreads();
  if (t == 0) {
    double s = 0.0;
    for (int i = 0; i < 24; ++i) s += sh[i];
    double w = 0.0;
    for (int i = 24; i < 30; ++i) w += sh[i];
    out[0] = s / 24.0;
    out[1] = w / 6.0;
  }
}

extern "C" __global__ void k_boite_profil(const Q* __restrict__ U, int Lg, const u32* __restrict__ entrees,
                                          const u32* __restrict__ debuts, double* out, long out_pas, long idx, int ncl) {
  __shared__ double sh[256];
  Geo g = mkgeo(Lg);
  const int d = blockIdx.x;
  const int c = blockIdx.y;
  U += (long)c * 4 * g.V;
  out += (long)c * out_pas + idx * ncl;
  const u32 i0 = debuts[d], i1 = debuts[d + 1];
  double a = 0.0;
  for (u32 i = i0 + threadIdx.x; i < i1; i += 256) {
    const u32 e = entrees[i];
    const long s = (long)(e >> 3);
    const int ip = (int)(e & 7u);
    const int mu = boite_mu(ip), nu = boite_nu(ip);
    int x[4];
    coords(g, s, x);
    a += boite_plaq(U, g, mu, nu, x);
  }
  sh[threadIdx.x] = a;
  __syncthreads();
  for (int w = 128; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[d] = sh[0];
}
