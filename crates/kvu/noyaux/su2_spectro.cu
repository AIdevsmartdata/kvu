// su2_spectro.cu : spectroscopie des opérateurs scalaires (boucles spatiales lissées APE) SUR CARTE, branche spectro/gpu-1120.
// Module autonome (aucune dépendance aux autres noyaux) ; compilé deux fois : KV_T = float (champ F32) ou double (champ F64).
// Conventions IDENTIQUES à `su2::spectro` (CPU) : site s = ((x0·l1 + x1)·l2 + x2)·l3 + x3, lien li = μ·V + s, quaternion (x,y,z,w)
// avec ½Tr U = x ; chemin = suite de pas ±1..3 sur les trois axes spatiaux (axes ≠ temps, croissants) ; lissage APE spatial
// (agrafes des plans spatiaux, projection V/|V|, mise à jour simultanée, liens de temps intacts).
// Les produits de lien sont en KV_T ; TOUTES les sommes (sites, images, temps) sont en double, à arbre FIXE (déterministe).
// Champ d'une chaîne c : décalage c·4V quaternions (c = blockIdx.z pour les noyaux de boucles, blockIdx.y pour le lissage).
// MUT (contrôles négatifs, jamais en production) : 1 = agrafe basse sans dagger ; 2 = lissage sans projection ; 3 = trace sans ½
// (facteur 2) ; 4 = somme spatiale tronquée (dernier site de la tranche oublié).
#ifndef KV_T
#define KV_T double
#endif
#ifndef MUT
#define MUT 0
#endif
typedef KV_T RT;
struct SQ { RT x, y, z, w; };

__device__ inline SQ sq_mul(SQ a, SQ b) {
  SQ c;
  c.x = a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w;
  c.y = a.x * b.y + b.x * a.y - (a.z * b.w - a.w * b.z);
  c.z = a.x * b.z + b.x * a.z - (a.w * b.y - a.y * b.w);
  c.w = a.x * b.w + b.x * a.w - (a.y * b.z - a.z * b.y);
  return c;
}
__device__ inline SQ sq_dag(SQ a) { SQ c = {a.x, -a.y, -a.z, -a.w}; return c; }
__device__ inline SQ sq_add(SQ a, SQ b) { SQ c = {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; return c; }
__device__ inline SQ sq_scale(SQ a, RT s) { SQ c = {a.x * s, a.y * s, a.z * s, a.w * s}; return c; }

struct GeoS { int l[4]; int V; int st[4]; };
__device__ inline GeoS gs_lire(const long long* __restrict__ geo) {
  GeoS g;
  for (int d = 0; d < 4; ++d) { g.l[d] = (int)geo[d]; g.st[d] = (int)geo[5 + d]; }
  g.V = (int)geo[4];
  return g;
}
__device__ inline int gs_x(const GeoS& g, int s, int d) { return (s / g.st[d]) % g.l[d]; }
__device__ inline int gs_up(const GeoS& g, int s, int d, int xd) { return (xd == g.l[d] - 1) ? s - (g.l[d] - 1) * g.st[d] : s + g.st[d]; }
__device__ inline int gs_dn(const GeoS& g, int s, int d, int xd) { return (xd == 0) ? s + (g.l[d] - 1) * g.st[d] : s - g.st[d]; }

// ---- lissage APE : out = APE(in), une itération. grille (ceil(4V/256), nch) ----
extern "C" __global__ void k_ape(const SQ* __restrict__ in, SQ* __restrict__ out, const long long* __restrict__ geo, int temps,
                                 double alpha) {
  GeoS g = gs_lire(geo);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= 4 * g.V) return;
  long off = (long)blockIdx.y * 4 * g.V;
  in += off; out += off;
  int mu = i / g.V, s = i - mu * g.V;
  if (mu == temps) { out[i] = in[i]; return; }
  int xm = gs_x(g, s, mu);
  int smu = gs_up(g, s, mu, xm);
  SQ a = {0, 0, 0, 0};
  for (int nu = 0; nu < 4; ++nu) {
    if (nu == mu || nu == temps) continue;
    int xn = gs_x(g, s, nu);
    int snu = gs_up(g, s, nu, xn);
    int sdn = gs_dn(g, s, nu, xn);
    int smu_dn = gs_dn(g, smu, nu, xn);
    SQ haut = sq_mul(sq_mul(in[nu * g.V + s], in[mu * g.V + snu]), sq_dag(in[nu * g.V + smu]));
#if MUT == 1
    SQ prem = in[nu * g.V + sdn];
#else
    SQ prem = sq_dag(in[nu * g.V + sdn]);
#endif
    SQ bas = sq_mul(sq_mul(prem, in[mu * g.V + sdn]), in[nu * g.V + smu_dn]);
    a = sq_add(a, sq_add(haut, bas));
  }
  SQ v = sq_add(sq_scale(in[i], (RT)(1.0 - alpha)), sq_scale(a, (RT)(alpha / 4.0)));
#if MUT == 2
  out[i] = v;
#else
  RT n = sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
  if (n > (RT)1e-30) out[i] = sq_scale(v, (RT)1.0 / n); else out[i] = in[i];
#endif
}

// ---- boucles : sommes par bloc. grille (chunks, T·nim, nch), bloc 256.
// Pour la tranche t, l'image im, le paquet de sites `chunk` : part[((c·T + t)·nim + im)·chunks + chunk][q] avec
// q = 0 : Σ_x ½Tr W ; q = 1 + ik·6 + d·2 + {0,1} : Σ_x ½Tr W · {cos, sin}(phase) (phase au CENTRE de l'image, table `ph`).
extern "C" __global__ void k_boucles(const SQ* __restrict__ U, const long long* __restrict__ geo, int temps, int nim,
                                     const int* __restrict__ pas, const int* __restrict__ off, const double* __restrict__ ph,
                                     int nk, int lmax, double* __restrict__ part, int nq) {
  __shared__ double sh[256];
  GeoS g = gs_lire(geo);
  const int tt = g.l[temps];
  const int t = (int)blockIdx.y / nim, im = (int)blockIdx.y % nim;
  const int c = (int)blockIdx.z;
  U += (long)c * 4 * g.V;
  int ax[3], na = 0;
  for (int d = 0; d < 4; ++d) if (d != temps) ax[na++] = d;
  const int v3 = g.V / tt;
  const int j = (int)(blockIdx.x * 256 + threadIdx.x);
  bool ok = j < v3;
#if MUT == 4
  ok = ok && (j < v3 - 1);
#endif
  int x[4] = {0, 0, 0, 0};
  int s = 0;
  double tr = 0.0;
  if (ok) {
    int r = j;
    x[temps] = t;
    x[ax[2]] = r % g.l[ax[2]]; r /= g.l[ax[2]];
    x[ax[1]] = r % g.l[ax[1]]; r /= g.l[ax[1]];
    x[ax[0]] = r;
    for (int d = 0; d < 4; ++d) s += x[d] * g.st[d];
    const int s0 = s;
    SQ p = {1, 0, 0, 0};
    for (int k = off[im]; k < off[im + 1]; ++k) {
      int st = pas[k];
      int a = ax[(st < 0 ? -st : st) - 1];
      int xa = gs_x(g, s, a);
      if (st > 0) { p = sq_mul(p, U[a * g.V + s]); s = gs_up(g, s, a, xa); }
      else        { s = gs_dn(g, s, a, xa); p = sq_mul(p, sq_dag(U[a * g.V + s])); }
    }
    (void)s0;
    tr = (double)p.x;
#if MUT == 3
    tr *= 2.0;
#endif
  }
  const long base = ((((long)c * tt + t) * nim + im) * (long)gridDim.x + blockIdx.x) * nq;
  for (int q = 0; q < nq; ++q) {
    double val = 0.0;
    if (ok) {
      if (q == 0) val = tr;
      else {
        int qq = q - 1, ik = qq / 6, r = qq % 6, d = r / 2, cs = r % 2;
        int xd = x[ax[d]];
        val = tr * ph[((((long)im * nk + ik) * 3 + d) * lmax + xd) * 2 + cs];
      }
    }
    sh[threadIdx.x] = val;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
      if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
      __syncthreads();
    }
    if (threadIdx.x == 0) part[base + q] = sh[0];
    __syncthreads();
  }
}

// ---- opérateurs : Sg[((c·nlev + lev)·nf + f)·T·nqo + t·nqo + q] = Σ_im W[f·nim + im] Σ_chunks part. grille (ceil(nf·T·nq/256), nch)
// nq = nombre de quantités de `part` (pas), nqo ≤ nq = nombre rendu dans Sg (les premières : q = 0 seul pour les représentations).
extern "C" __global__ void k_operateurs(const double* __restrict__ part, const double* __restrict__ W, int nim, int nf, int tt,
                                        int nq, int nqo, int chunks, int nlev, int lev, double* __restrict__ Sg) {
  const int c = (int)blockIdx.y;
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= nf * tt * nqo) return;
  const int f = i / (tt * nqo), r = i % (tt * nqo), t = r / nqo, q = r % nqo;
  double acc = 0.0;
  for (int im = 0; im < nim; ++im) {
    const double w = W[f * nim + im];
    if (w == 0.0) continue;
    double sm = 0.0;
    const long b = ((((long)c * tt + t) * nim + im) * chunks) * nq + q;
    for (int ch = 0; ch < chunks; ++ch) sm += part[b + (long)ch * nq];
    acc += w * sm;
  }
  Sg[((((long)c * nlev + lev) * nf + f) * tt + t) * nqo + q] = acc;
}

// ---- corrélateurs. Sg comme ci-dessus ; opidx[o] = lev·nf + f des nops opérateurs retenus (ordre = ordre de sortie).
// slot < 0 : impulsion nulle (q = 0), moyenne temporelle symétrisée ; slot = ik ≥ 0 : impulsion du cran ik, moyenne des 3 axes.
// grille (ceil(npaires·(taumax+1)/256), nch). Sortie out[(c·nslots + (slot+1))·npaires·(taumax+1) + pair·(taumax+1) + tau].
extern "C" __global__ void k_corr(const double* __restrict__ Sg, const int* __restrict__ opidx, int nops, int nfl, int tt, int nq,
                                  int taumax, int slot, int nslots, int nlev, double* __restrict__ out) {
  const int c = (int)blockIdx.y;
  const int npair = nops * (nops + 1) / 2;
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= npair * (taumax + 1)) return;
  const int pair = i / (taumax + 1), tau = i % (taumax + 1);
  // pair → (a ≤ b) dans l'ordre (0,0)…(0,n−1),(1,1)…
  int a = 0, rest = pair;
  while (rest >= nops - a) { rest -= nops - a; ++a; }
  const int b = a + rest;
  const double* S = Sg + (long)c * nlev * nfl * tt * nq;
  const long ia = (long)opidx[a] * tt * nq, ib = (long)opidx[b] * tt * nq;
  double acc = 0.0;
  if (slot < 0) {
    for (int t = 0; t < tt; ++t) {
      const int t2 = (t + tau) % tt;
      acc += S[ia + (long)t * nq] * S[ib + (long)t2 * nq] + S[ib + (long)t * nq] * S[ia + (long)t2 * nq];
    }
    acc /= (2.0 * tt);
  } else {
    for (int t = 0; t < tt; ++t) {
      const int t2 = (t + tau) % tt;
      for (int d = 0; d < 3; ++d) {
        const int q = 1 + slot * 6 + d * 2;
        const double ar = S[ia + (long)t * nq + q], ai = S[ia + (long)t * nq + q + 1];
        const double br = S[ib + (long)t2 * nq + q], bi = S[ib + (long)t2 * nq + q + 1];
        const double ar2 = S[ib + (long)t * nq + q], ai2 = S[ib + (long)t * nq + q + 1];
        const double br2 = S[ia + (long)t2 * nq + q], bi2 = S[ia + (long)t2 * nq + q + 1];
        acc += (ar * br + ai * bi) + (ar2 * br2 + ai2 * bi2);
      }
    }
    acc /= (2.0 * tt * 3.0);
  }
  out[((long)c * nslots + (slot + 1)) * npair * (taumax + 1) + i] = acc;
}

// ---- impulsions HORS AXE (`--impulsions-vecteurs`) : comme k_boucles, avec nv vecteurs k = (k₁,k₂,k₃) quelconques.
// q = 0 : Σ_x ½Tr W ; q = 1 + 2·iv + {0,1} : Σ_x ½Tr W · {cos, sin}(k·(x + c)) = partie réelle / imaginaire de
// tr · [e^{iθ₀(x₀)} e^{iθ₁(x₁)} e^{iθ₂(x₂)}] · e^{i k·c} ; tv[((iv·3 + d)·lmax + x)·2 + {0,1}] = (cos, sin)(k_d 2π x/ℓ_d),
// cen[(im·nv + iv)·2 + {0,1}] = (cos, sin)(k·c_image).

// Définie avec les autres opérateurs de superliens.
__device__ inline int gs_sh(const GeoS& g, int s, int d, int l);

struct PhaseOrb { double re, im; };

__device__ inline PhaseOrb orb_add(PhaseOrb a, PhaseOrb b) {
  return {a.re + b.re, a.im + b.im};
}

__device__ inline double orb_xor(double a, int delta) {
#ifdef KV_HIP
  return __shfl_xor(a, delta, 32);
#else
  return __shfl_xor_sync(0xffffffffu, a, delta, 32);
#endif
}

__device__ inline double orb_somme32(double a) {
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) a += orb_xor(a, delta);
  return a;
}

__device__ inline PhaseOrb orb_terme(const double* tr, const int (*coord)[256], int j, int sites,
                                     const double* tv, PhaseOrb centre) {
  if (j >= sites) return {0.0, 0.0};
  const double* a0 = tv + coord[0][j];
  const double* a1 = tv + coord[1][j];
  const double* a2 = tv + coord[2][j];
  const double pr = a0[0] * a1[0] - a0[1] * a1[1], pi = a0[0] * a1[1] + a0[1] * a1[0];
  const double qr = pr * a2[0] - pi * a2[1], qi = pr * a2[1] + pi * a2[0];
  const double w = tr[j];
  return {w * (qr * centre.re - qi * centre.im), w * (qr * centre.im + qi * centre.re)};
}

// Un seul partage des traces et des coordonnées, puis un sous-groupe de 32 lanes par projection.
// Chaque lane réduit ses huit sites de pas 32 en reproduisant les étages 128, 64, 32 de l'ancien
// arbre. Les échanges de registres reproduisent les étages 16, 8, 4, 2, 1 pour la lane 0.
// Sur un wavefront HIP de 64 lanes, les deux groupes demeurent indépendants (largeur explicite).
template <bool Bloc>
__device__ inline void boucles_orbites(const SQ* __restrict__ U, const long long* __restrict__ geo, int temps, int nim,
                                     const int* __restrict__ pas, const int* __restrict__ off, const double* __restrict__ tv,
                                     const double* __restrict__ cen, int nv, int lmax, double* __restrict__ part, int nq, int l,
                                     double* tr, int (*coord)[256]) {
  GeoS g = gs_lire(geo);
  const int tt = g.l[temps];
  const int t = (int)blockIdx.y / nim, im = (int)blockIdx.y % nim;
  const int c = (int)blockIdx.z;
  U += (long)c * 4 * g.V;
  int ax[3], na = 0;
  for (int d = 0; d < 4; ++d) if (d != temps) ax[na++] = d;
  const int tid = (int)threadIdx.x;
  const int debut = (int)blockIdx.x * 256;
  const int sites = g.V / tt - debut;
  tr[tid] = 0.0;
  if (tid < sites) {
    int r = debut + tid;
    const int x2 = r % g.l[ax[2]]; r /= g.l[ax[2]];
    const int x1 = r % g.l[ax[1]], x0 = r / g.l[ax[1]];
    coord[0][tid] = 2 * x0;
    coord[1][tid] = 2 * (lmax + x1);
    coord[2][tid] = 2 * (2 * lmax + x2);
    int s = t * g.st[temps] + x0 * g.st[ax[0]] + x1 * g.st[ax[1]] + x2 * g.st[ax[2]];
    SQ p = {1, 0, 0, 0};
    for (int k = off[im]; k < off[im + 1]; ++k) {
      const int st = pas[k];
      const int a = ax[(st < 0 ? -st : st) - 1];
      if (st > 0) {
        p = sq_mul(p, U[a * g.V + s]);
        if constexpr (Bloc) s = gs_sh(g, s, a, l);
        else s = gs_up(g, s, a, gs_x(g, s, a));
      } else {
        if constexpr (Bloc) s = gs_sh(g, s, a, -l);
        else s = gs_dn(g, s, a, gs_x(g, s, a));
        p = sq_mul(p, sq_dag(U[a * g.V + s]));
      }
    }
    tr[tid] = (double)p.x;
  }
  __syncthreads();

  const long base = ((((long)c * tt + t) * nim + im) * (long)gridDim.x + blockIdx.x) * nq;
  const int groupe = tid / 32, lane = tid % 32;
  if (groupe == 0) {
    const double a = (tr[lane] + tr[lane + 128]) + (tr[lane + 64] + tr[lane + 192]);
    const double b = (tr[lane + 32] + tr[lane + 160]) + (tr[lane + 96] + tr[lane + 224]);
    const double somme = orb_somme32(a + b);
    if (lane == 0) part[base] = somme;
  }
  // La projection nulle occupe la première place du groupe 0.
  for (int iv = (groupe + 7) % 8; iv < nv; iv += 8) {
    const double* phase = tv + (long)iv * 3 * lmax * 2;
    const double* pcentre = cen + ((long)im * nv + iv) * 2;
    const PhaseOrb centre = {pcentre[0], pcentre[1]};
    const PhaseOrb s04 = orb_add(orb_terme(tr, coord, lane,       sites, phase, centre),
                                 orb_terme(tr, coord, lane + 128, sites, phase, centre));
    const PhaseOrb s26 = orb_add(orb_terme(tr, coord, lane + 64,  sites, phase, centre),
                                 orb_terme(tr, coord, lane + 192, sites, phase, centre));
    const PhaseOrb s15 = orb_add(orb_terme(tr, coord, lane + 32,  sites, phase, centre),
                                 orb_terme(tr, coord, lane + 160, sites, phase, centre));
    const PhaseOrb s37 = orb_add(orb_terme(tr, coord, lane + 96,  sites, phase, centre),
                                 orb_terme(tr, coord, lane + 224, sites, phase, centre));
    const PhaseOrb somme = orb_add(orb_add(s04, s26), orb_add(s15, s37));
    const double re = orb_somme32(somme.re), imv = orb_somme32(somme.im);
    if (lane == 0) {
      part[base + 1 + 2 * iv] = re;
      part[base + 2 + 2 * iv] = imv;
    }
  }
}

extern "C" __global__ void k_boucles_v(const SQ* __restrict__ U, const long long* __restrict__ geo, int temps, int nim,
                                       const int* __restrict__ pas, const int* __restrict__ off, const double* __restrict__ tv,
                                       const double* __restrict__ cen, int nv, int lmax, double* __restrict__ part, int nq) {
  __shared__ double tr[256];
  __shared__ int coord[3][256];
  boucles_orbites<false>(U, geo, temps, nim, pas, off, tv, cen, nv, lmax, part, nq, 1, tr, coord);
}

// ---- corrélateur moyenné sur les nv vecteurs v0 … v0+nv−1 d'une orbite (plans q = 1 + 2·iv) : out[((c·nslots)+slot)·npaires·(taumax+1) + i].
// Même formule que k_corr (slot ≥ 0) avec la boucle sur les axes remplacée par la boucle sur les vecteurs ; norme 2·T·nv.
extern "C" __global__ void k_corr_orb(const double* __restrict__ Sg, const int* __restrict__ opidx, int nops, int nfl, int tt, int nq,
                                      int taumax, int slot, int nslots, int nlev, int v0, int nv, double* __restrict__ out) {
  const int c = (int)blockIdx.y;
  const int npair = nops * (nops + 1) / 2;
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= npair * (taumax + 1)) return;
  const int pair = i / (taumax + 1), tau = i % (taumax + 1);
  int a = 0, rest = pair;
  while (rest >= nops - a) { rest -= nops - a; ++a; }
  const int b = a + rest;
  const double* S = Sg + (long)c * nlev * nfl * tt * nq;
  const long ia = (long)opidx[a] * tt * nq, ib = (long)opidx[b] * tt * nq;
  double acc = 0.0;
  for (int t = 0; t < tt; ++t) {
    const int t2 = (t + tau) % tt;
    for (int iv = v0; iv < v0 + nv; ++iv) {
      const int q = 1 + iv * 2;
      const double ar = S[ia + (long)t * nq + q], ai = S[ia + (long)t * nq + q + 1];
      const double br = S[ib + (long)t2 * nq + q], bi = S[ib + (long)t2 * nq + q + 1];
      const double ar2 = S[ib + (long)t * nq + q], ai2 = S[ib + (long)t * nq + q + 1];
      const double br2 = S[ia + (long)t2 * nq + q], bi2 = S[ia + (long)t2 * nq + q + 1];
      acc += (ar * br + ai * bi) + (ar2 * br2 + ai2 * bi2);
    }
  }
  acc /= (2.0 * tt * (double)nv);
  out[((long)c * nslots + slot) * npair * (taumax + 1) + i] = acc;
}

// ---- algorithme à deux niveaux : cumul sur la carte (sans FMA : mêmes arrondis que le cumul hôte `deux_niveaux.rs`) ----
// k_cumul_somme : Sum[i] += S[i], i < n. grille (ceil(n/256))
extern "C" __global__ void k_cumul_somme(const double* __restrict__ S, double* __restrict__ Sum, long n) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Sum[i] = __dadd_rn(Sum[i], S[i]);
}

// paire de temps (t, t2) NON valide (même tranche épaisse intérieure) pour D = d (d = 0 : toutes valides)
__device__ static inline bool dn_invalide(int t, int t2, int d) {
  if (d == 0) return false;
  if (t % d == 0 || t2 % d == 0) return false;
  return (t / d) == (t2 / d);
}

// k_cumul_ord : out[((c*nslots)+(slot+1))*npair*(taumax+1) + i] += Σ_{t : paire (t,t+τ) non valide} (produit a·b + produit b·a)
// (mêmes indices et mêmes formules que k_corr, sans la normalisation). thread par (paire, τ), grille (ceil(npair·ntau/128), nch)
extern "C" __global__ void k_cumul_ord(const double* __restrict__ Sg, const int* __restrict__ opidx, int nops, int nfl, int tt, int nq,
                                       int taumax, int slot, int nslots, int nlev, int d, double* __restrict__ out) {
  const int c = (int)blockIdx.y;
  const int npair = nops * (nops + 1) / 2;
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= npair * (taumax + 1)) return;
  const int pair = i / (taumax + 1), tau = i % (taumax + 1);
  int a = 0, rest = pair;
  while (rest >= nops - a) { rest -= nops - a; ++a; }
  const int b = a + rest;
  const double* S = Sg + (long)c * nlev * nfl * tt * nq;
  const long ia = (long)opidx[a] * tt * nq, ib = (long)opidx[b] * tt * nq;
  double acc = 0.0;
  for (int t = 0; t < tt; ++t) {
    const int t2 = (t + tau) % tt;
    if (!dn_invalide(t, t2, d)) continue;
    if (slot < 0) {
      acc = __dadd_rn(acc, __dadd_rn(__dmul_rn(S[ia + (long)t * nq], S[ib + (long)t2 * nq]), __dmul_rn(S[ib + (long)t * nq], S[ia + (long)t2 * nq])));
    } else {
      for (int dd = 0; dd < 3; ++dd) {
        const int q = 1 + slot * 6 + dd * 2;
        const double ar = S[ia + (long)t * nq + q], ai = S[ia + (long)t * nq + q + 1];
        const double br = S[ib + (long)t2 * nq + q], bi = S[ib + (long)t2 * nq + q + 1];
        const double ar2 = S[ib + (long)t * nq + q], ai2 = S[ib + (long)t * nq + q + 1];
        const double br2 = S[ia + (long)t2 * nq + q], bi2 = S[ia + (long)t2 * nq + q + 1];
        const double u = __dadd_rn(__dmul_rn(ar, br), __dmul_rn(ai, bi));
        const double w = __dadd_rn(__dmul_rn(ar2, br2), __dmul_rn(ai2, bi2));
        acc = __dadd_rn(acc, __dadd_rn(u, w));
      }
    }
  }
  const long o = ((long)c * nslots + (slot + 1)) * npair * (taumax + 1) + i;
  out[o] = __dadd_rn(out[o], acc);
}

// ⟨S_op⟩_t : thread par (c, op). grille (ceil(nops/64), nch)
extern "C" __global__ void k_moy(const double* __restrict__ Sg, const int* __restrict__ opidx, int nops, int nfl, int tt, int nq,
                                 int nlev, double* __restrict__ out) {
  const int c = (int)blockIdx.y;
  const int o = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (o >= nops) return;
  const double* S = Sg + (long)c * nlev * nfl * tt * nq + (long)opidx[o] * tt * nq;
  double acc = 0.0;
  for (int t = 0; t < tt; ++t) acc += S[(long)t * nq];
  out[(long)c * nops + o] = acc / (double)tt;
}

// ---- plaquette : Σ_s Σ_{μ<ν} ½Tr P_{μν}(s) par bloc de 256 sites. grille (ceil(V/256), nch), sortie out[c·gridDim.x + blk]
extern "C" __global__ void k_plaq(const SQ* __restrict__ U, const long long* __restrict__ geo, double* __restrict__ out) {
  __shared__ double sh[256];
  GeoS g = gs_lire(geo);
  U += (long)blockIdx.y * 4 * g.V;
  const int s = (int)(blockIdx.x * 256 + threadIdx.x);
  double acc = 0.0;
  if (s < g.V) {
    int x[4];
    for (int d = 0; d < 4; ++d) x[d] = gs_x(g, s, d);
    for (int mu = 0; mu < 4; ++mu)
      for (int nu = mu + 1; nu < 4; ++nu) {
        int smu = gs_up(g, s, mu, x[mu]), snu = gs_up(g, s, nu, x[nu]);
        SQ p = sq_mul(sq_mul(U[mu * g.V + s], U[nu * g.V + smu]), sq_mul(sq_dag(U[mu * g.V + snu]), sq_dag(U[nu * g.V + s])));
        acc += (double)p.x;
      }
  }
  sh[threadIdx.x] = acc;
  __syncthreads();
  for (int w = 128; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[(long)blockIdx.y * gridDim.x + blockIdx.x] = sh[0];
}

// ---- plaquettes SPATIALES et TEMPORELLES séparées (réseau anisotrope, --xi0 ≠ 1 seulement) : Σ_s Σ_{μ<ν} ½Tr P_{μν}(s),
// plaquette temporelle ⇔ μ = temps ou ν = temps. grille (ceil(V/256), nch), sorties outS/outT[c·gridDim.x + blk]
extern "C" __global__ void k_plaq_st_an(const SQ* __restrict__ U, const long long* __restrict__ geo, int temps,
                                        double* __restrict__ outS, double* __restrict__ outT) {
  __shared__ double sh[2][256];
  GeoS g = gs_lire(geo);
  U += (long)blockIdx.y * 4 * g.V;
  const int s = (int)(blockIdx.x * 256 + threadIdx.x);
  double aS = 0.0, aT = 0.0;
  if (s < g.V) {
    int x[4];
    for (int d = 0; d < 4; ++d) x[d] = gs_x(g, s, d);
    for (int mu = 0; mu < 4; ++mu)
      for (int nu = mu + 1; nu < 4; ++nu) {
        int smu = gs_up(g, s, mu, x[mu]), snu = gs_up(g, s, nu, x[nu]);
        SQ p = sq_mul(sq_mul(U[mu * g.V + s], U[nu * g.V + smu]), sq_mul(sq_dag(U[mu * g.V + snu]), sq_dag(U[nu * g.V + s])));
        if (mu == temps || nu == temps) aT += (double)p.x; else aS += (double)p.x;
      }
  }
  sh[0][threadIdx.x] = aS;
  sh[1][threadIdx.x] = aT;
  __syncthreads();
  for (int w = 128; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) { sh[0][threadIdx.x] += sh[0][threadIdx.x + w]; sh[1][threadIdx.x] += sh[1][threadIdx.x + w]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { outS[(long)blockIdx.y * gridDim.x + blockIdx.x] = sh[0][0]; outT[(long)blockIdx.y * gridDim.x + blockIdx.x] = sh[1][0]; }
}

// ---- TORELONS (lignes de Polyakov enroulées lissées), voir su2::spectro (CPU) ----
// k_lignes : lig[(c·T + t)·ta·tb + a·tb + b] = ½Tr ∏_k U_w(x + kŵ), x_w = 0, x[temps] = t, x[tr0] = a, x[tr1] = b.
// grille (ceil(T·ta·tb/256), nch)
extern "C" __global__ void k_lignes(const SQ* __restrict__ U, const long long* __restrict__ geo, int w, int temps, int tr0, int tr1,
                                    double* __restrict__ lig) {
  GeoS g = gs_lire(geo);
  const int ta = g.l[tr0], tb = g.l[tr1], tt = g.l[temps];
  const int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (j >= tt * ta * tb) return;
  const int c = (int)blockIdx.y;
  U += (long)c * 4 * g.V;
  const int t = j / (ta * tb), a = (j / tb) % ta, b = j % tb;
  int x[4] = {0, 0, 0, 0};
  x[temps] = t; x[tr0] = a; x[tr1] = b; x[w] = 0;
  int s = 0;
  for (int d = 0; d < 4; ++d) s += x[d] * g.st[d];
  SQ p = {1, 0, 0, 0};
  for (int k = 0; k < g.l[w]; ++k) {
    p = sq_mul(p, U[w * g.V + s]);
    if (k != g.l[w] - 1) s += g.st[w];
  }
  lig[(long)c * tt * ta * tb + j] = (double)p.x;
}

// k_phi : phi[(((c·nlev + lev)·nmom + m)·T + t)·2 + {0,1}] = (1/N⊥) Σ_{a,b} lig[c,t,a,b]·{cos,sin}(2π(na a/ta + nb b/tb)),
// table ph[m·ta·tb·2 + (a·tb+b)·2 + {0,1}] calculée sur l'hôte. grille (nmom·T, nch), bloc 256.
extern "C" __global__ void k_phi(const double* __restrict__ lig, const double* __restrict__ ph, int nmom, int tt, int nperp, int nlev,
                                 int lev, double* __restrict__ phi) {
  __shared__ double sh[256];
  const int c = (int)blockIdx.y, m = (int)blockIdx.x / tt, t = (int)blockIdx.x % tt;
  const double* L = lig + ((long)c * tt + t) * nperp;
  const double* P = ph + (long)m * nperp * 2;
  for (int cs = 0; cs < 2; ++cs) {
    double acc = 0.0;
    for (int i = threadIdx.x; i < nperp; i += 256) acc += L[i] * P[2 * i + cs];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int w2 = 128; w2 > 0; w2 >>= 1) {
      if ((int)threadIdx.x < w2) sh[threadIdx.x] += sh[threadIdx.x + w2];
      __syncthreads();
    }
    if (threadIdx.x == 0) phi[((((long)c * nlev + lev) * nmom + m) * tt + t) * 2 + cs] = sh[0] / (double)nperp;
    __syncthreads();
  }
}

// ======================================================================================================================
// BLOCAGE DE TEPER (branche spectro/blocage-1120) : superliens de longueur ℓ = 2^b sur TOUS les sites (référence processeur :
// `su2::spectro::bloc`). Les noyaux ci-dessus ne sont pas modifiés (identité au bit des anciens opérateurs).
// MUT (contrôles négatifs) : 6 = blocage, agrafe basse sans dagger ; 7 = blocage, chemin direct décalé de 1 au lieu de ℓ ;
// 8 = boucles de superliens : pas de 1 site au lieu de ℓ.
// ======================================================================================================================
__device__ inline int gs_sh(const GeoS& g, int s, int d, int l) {
  const int xd = gs_x(g, s, d);
  int n = (xd + l) % g.l[d];
  if (n < 0) n += g.l[d];
  return s + (n - xd) * g.st[d];
}

// k_bloc : out = blocage(in), superliens ℓ → 2ℓ, poids p des agrafes allongées, projection V/|V|. grille (ceil(4V/256), nch)
extern "C" __global__ void k_bloc(const SQ* __restrict__ in, SQ* __restrict__ out, const long long* __restrict__ geo, int temps, int l,
                                  double p) {
  GeoS g = gs_lire(geo);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= 4 * g.V) return;
  long off = (long)blockIdx.y * 4 * g.V;
  in += off; out += off;
  int mu = i / g.V, s = i - mu * g.V;
  if (mu == temps) { out[i] = in[i]; return; }
#if MUT == 7
  const int s_mu = gs_sh(g, s, mu, 1);
#else
  const int s_mu = gs_sh(g, s, mu, l);
#endif
  const int s_2mu = gs_sh(g, s, mu, 2 * l);
  SQ direct = sq_mul(in[mu * g.V + s], in[mu * g.V + s_mu]);
  SQ a = {0, 0, 0, 0};
  for (int nu = 0; nu < 4; ++nu) {
    if (nu == mu || nu == temps) continue;
    const int s_nu = gs_sh(g, s, nu, l);
    const int s_dn = gs_sh(g, s, nu, -l);
    SQ haut = sq_mul(sq_mul(sq_mul(in[nu * g.V + s], in[mu * g.V + s_nu]), in[mu * g.V + gs_sh(g, s_nu, mu, l)]),
                     sq_dag(in[nu * g.V + s_2mu]));
#if MUT == 6
    SQ prem = in[nu * g.V + s_dn];
#else
    SQ prem = sq_dag(in[nu * g.V + s_dn]);
#endif
    SQ bas = sq_mul(sq_mul(sq_mul(prem, in[mu * g.V + s_dn]), in[mu * g.V + gs_sh(g, s_dn, mu, l)]),
                    in[nu * g.V + gs_sh(g, s_dn, mu, 2 * l)]);
    a = sq_add(a, sq_add(haut, bas));
  }
  SQ v = sq_add(direct, sq_scale(a, (RT)p));
  RT n = sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
  if (n > (RT)1e-30) out[i] = sq_scale(v, (RT)1.0 / n); else out[i] = direct;
}

// k_ape_l : lissage APE des superliens de longueur ℓ (agrafes de superliens). grille (ceil(4V/256), nch)
extern "C" __global__ void k_ape_l(const SQ* __restrict__ in, SQ* __restrict__ out, const long long* __restrict__ geo, int temps,
                                   double alpha, int l) {
  GeoS g = gs_lire(geo);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= 4 * g.V) return;
  long off = (long)blockIdx.y * 4 * g.V;
  in += off; out += off;
  int mu = i / g.V, s = i - mu * g.V;
  if (mu == temps) { out[i] = in[i]; return; }
  const int s_mu = gs_sh(g, s, mu, l);
  SQ a = {0, 0, 0, 0};
  for (int nu = 0; nu < 4; ++nu) {
    if (nu == mu || nu == temps) continue;
    const int s_nu = gs_sh(g, s, nu, l);
    const int s_dn = gs_sh(g, s, nu, -l);
    SQ haut = sq_mul(sq_mul(in[nu * g.V + s], in[mu * g.V + s_nu]), sq_dag(in[nu * g.V + s_mu]));
    SQ bas = sq_mul(sq_mul(sq_dag(in[nu * g.V + s_dn]), in[mu * g.V + s_dn]), in[nu * g.V + gs_sh(g, s_dn, mu, l)]);
    a = sq_add(a, sq_add(haut, bas));
  }
  SQ v = sq_add(sq_scale(in[i], (RT)(1.0 - alpha)), sq_scale(a, (RT)(alpha / 4.0)));
  RT n = sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
  if (n > (RT)1e-30) out[i] = sq_scale(v, (RT)1.0 / n); else out[i] = in[i];
}

// k_boucles_l : comme k_boucles, chaque pas valant un superlien de ℓ sites ; table `ph` bâtie sur l'hôte avec le centre ×ℓ.
extern "C" __global__ void k_boucles_l(const SQ* __restrict__ U, const long long* __restrict__ geo, int temps, int nim,
                                       const int* __restrict__ pas, const int* __restrict__ off, const double* __restrict__ ph,
                                       int nk, int lmax, double* __restrict__ part, int nq, int l) {
  __shared__ double sh[256];
  GeoS g = gs_lire(geo);
  const int tt = g.l[temps];
  const int t = (int)blockIdx.y / nim, im = (int)blockIdx.y % nim;
  const int c = (int)blockIdx.z;
  U += (long)c * 4 * g.V;
  int ax[3], na = 0;
  for (int d = 0; d < 4; ++d) if (d != temps) ax[na++] = d;
  const int v3 = g.V / tt;
  const int j = (int)(blockIdx.x * 256 + threadIdx.x);
  const bool ok = j < v3;
#if MUT == 8
  const int lp = 1;
#else
  const int lp = l;
#endif
  int x[4] = {0, 0, 0, 0};
  double tr = 0.0;
  if (ok) {
    int r = j;
    x[temps] = t;
    x[ax[2]] = r % g.l[ax[2]]; r /= g.l[ax[2]];
    x[ax[1]] = r % g.l[ax[1]]; r /= g.l[ax[1]];
    x[ax[0]] = r;
    int s = 0;
    for (int d = 0; d < 4; ++d) s += x[d] * g.st[d];
    SQ p = {1, 0, 0, 0};
    for (int k = off[im]; k < off[im + 1]; ++k) {
      int st = pas[k];
      int a = ax[(st < 0 ? -st : st) - 1];
      if (st > 0) { p = sq_mul(p, U[a * g.V + s]); s = gs_sh(g, s, a, lp); }
      else        { s = gs_sh(g, s, a, -lp); p = sq_mul(p, sq_dag(U[a * g.V + s])); }
    }
    tr = (double)p.x;
  }
  const long base = ((((long)c * tt + t) * nim + im) * (long)gridDim.x + blockIdx.x) * nq;
  for (int q = 0; q < nq; ++q) {
    double val = 0.0;
    if (ok) {
      if (q == 0) val = tr;
      else {
        int qq = q - 1, ik = qq / 6, r = qq % 6, d = r / 2, cs = r % 2;
        int xd = x[ax[d]];
        val = tr * ph[((((long)im * nk + ik) * 3 + d) * lmax + xd) * 2 + cs];
      }
    }
    sh[threadIdx.x] = val;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
      if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
      __syncthreads();
    }
    if (threadIdx.x == 0) part[base + q] = sh[0];
    __syncthreads();
  }
}

// k_lignes_l : lignes de Polyakov de superliens de longueur ℓ (ℓ divise L_w), moyenne sur les ℓ points de départ x_w = 0..ℓ−1 :
// lig[(c·T + t)·ta·tb + a·tb + b] = (1/ℓ) Σ_{x0<ℓ} ½Tr ∏_{k < L_w/ℓ} U_w(x0 + kℓ). grille (ceil(T·ta·tb/256), nch)
extern "C" __global__ void k_lignes_l(const SQ* __restrict__ U, const long long* __restrict__ geo, int w, int temps, int tr0, int tr1,
                                      int l, double* __restrict__ lig) {
  GeoS g = gs_lire(geo);
  const int ta = g.l[tr0], tb = g.l[tr1], tt = g.l[temps];
  const int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (j >= tt * ta * tb) return;
  const int c = (int)blockIdx.y;
  U += (long)c * 4 * g.V;
  const int t = j / (ta * tb), a = (j / tb) % ta, b = j % tb;
  if (l < 1) return;  // garde défensive (l'hôte passe ℓ = 2^b ≥ 2)
  const int nseg = g.l[w] / l;
  double tot = 0.0;
  for (int x0 = 0; x0 < l; ++x0) {
    int x[4] = {0, 0, 0, 0};
    x[temps] = t; x[tr0] = a; x[tr1] = b; x[w] = x0;
    int s = 0;
    for (int d = 0; d < 4; ++d) s += x[d] * g.st[d];
    SQ p = {1, 0, 0, 0};
    for (int k = 0; k < nseg; ++k) {
      p = sq_mul(p, U[w * g.V + s]);
      s = gs_sh(g, s, w, l);
    }
    tot += (double)p.x;
  }
  lig[(long)c * tt * ta * tb + j] = tot / (double)l;
}

// ======================================================================================================================
// PROFIL DU TUBE (--profil) : densité d'action ρ(x⊥,t) sommée le long de w, séparée électrique (plan contenant `temps`) / magnétique.
// Chaque plaquette (1 − ½Tr P) est REPARTIE à parts égales sur ses coins dans les directions non-w qu'elle couvre (transverses et
// temps) : la carte est centrée sur le réseau (symétrique sous réflexion) et Σ_{x⊥,t} (e+m) = Σ_plaquettes (1 − ½Tr P).
// rho[((c·2 + k)·T + t)·ta·tb + a·tb + b], k = 0 électrique, 1 magnétique. grille (ceil(T·ta·tb/128), nch)
// ======================================================================================================================
extern "C" __global__ void k_rho(const SQ* __restrict__ U, const long long* __restrict__ geo, int w, int temps, int tr0, int tr1,
                                 double* __restrict__ rho) {
  GeoS g = gs_lire(geo);
  const int ta = g.l[tr0], tb = g.l[tr1], tt = g.l[temps];
  const int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (j >= tt * ta * tb) return;
  const int c = (int)blockIdx.y;
  U += (long)c * 4 * g.V;
  const int t = j / (ta * tb), a = (j / tb) % ta, b = j % tb;
  int x[4] = {0, 0, 0, 0};
  x[temps] = t; x[tr0] = a; x[tr1] = b; x[w] = 0;
  int s0 = 0;
  for (int d = 0; d < 4; ++d) s0 += x[d] * g.st[d];
  const int dirs[4] = {w, temps, tr0, tr1};
  double acc[2] = {0.0, 0.0};
  for (int i = 0; i < 4; ++i)
    for (int k = i + 1; k < 4; ++k) {
      const int mu = dirs[i], nu = dirs[k];
      const int kind = (mu == temps || nu == temps) ? 0 : 1;
      int sd[2], nd = 0;
      if (mu != w) sd[nd++] = mu;
      if (nu != w) sd[nd++] = nu;
      const int ncomb = 1 << nd;
      const double wt = 1.0 / (double)ncomb;
      for (int comb = 0; comb < ncomb; ++comb) {
        int sb = s0;
        for (int q = 0; q < nd; ++q)
          if ((comb >> q) & 1) sb = gs_sh(g, sb, sd[q], -1);
        for (int xw = 0; xw < g.l[w]; ++xw) {
          const int s = (xw == 0) ? sb : gs_sh(g, sb, w, xw);
          const int smu = gs_sh(g, s, mu, 1), snu = gs_sh(g, s, nu, 1);
          SQ p = sq_mul(sq_mul(U[mu * g.V + s], U[nu * g.V + smu]), sq_mul(sq_dag(U[mu * g.V + snu]), sq_dag(U[nu * g.V + s])));
          acc[kind] += wt * (1.0 - (double)p.x);
        }
      }
    }
  const long n3 = (long)tt * ta * tb;
  rho[((long)c * 2 + 0) * n3 + j] = acc[0];
  rho[((long)c * 2 + 1) * n3 + j] = acc[1];
}

// k_boucles_l_v : k_boucles_v pour des superliens de ℓ sites (niveaux de blocage) ; `cen` bâti avec le centre ×ℓ (phase e^{i k·ℓ c}).
extern "C" __global__ void k_boucles_l_v(const SQ* __restrict__ U, const long long* __restrict__ geo, int temps, int nim,
                                       const int* __restrict__ pas, const int* __restrict__ off, const double* __restrict__ tv,
                                       const double* __restrict__ cen, int nv, int lmax, double* __restrict__ part, int nq, int l) {
  __shared__ double tr[256];
  __shared__ int coord[3][256];
  boucles_orbites<true>(U, geo, temps, nim, pas, off, tv, cen, nv, lmax, part, nq, l, tr, coord);
}


// ======================================================================================================================
// SECTEUR DE FLUX (--secteur-flux, `su2::spectro::flux`) : opérateurs enroulés dans DEUX directions d1, d2 (côtés égaux), sommes de
// tranche (impulsion totale nulle). Pas de superlien l (1 = niveaux APE, 2^b = blocage). Noyaux précédents inchangés.
// out[((c·nlev + lev)·T + t)·6 + q], q = 0 a+ = ½Tr[L1 L2], 1 a− = ½Tr[L1 L2†], 2 b+ (escalier (+1,+2)), 3 b− (escalier (+1,−2)),
// 4 T1 = ½Tr L1, 5 T2 = ½Tr L2 ; sommes sur les points de la tranche de temps t. grille (T, nch), bloc 256.
// ======================================================================================================================
extern "C" __global__ void k_flux(const SQ* __restrict__ U, const long long* __restrict__ geo, int d1, int d2, int temps, int l,
                                  int nlev, int lev, double* __restrict__ out) {
  __shared__ double sh[256];
  GeoS g = gs_lire(geo);
  const int tt = g.l[temps];
  const int t = (int)blockIdx.x, c = (int)blockIdx.y;
  U += (long)c * 4 * g.V;
  int sp[3], k = 0;
  for (int d = 0; d < 4; ++d) if (d != temps) sp[k++] = d;
  const int n3 = g.l[sp[0]] * g.l[sp[1]] * g.l[sp[2]];
  const int n = g.l[d1] / l;
  double acc[6] = {0, 0, 0, 0, 0, 0};
  for (int i = (int)threadIdx.x; i < n3; i += 256) {
    int x[4] = {0, 0, 0, 0};
    x[temps] = t;
    x[sp[2]] = i % g.l[sp[2]];
    x[sp[1]] = (i / g.l[sp[2]]) % g.l[sp[1]];
    x[sp[0]] = i / (g.l[sp[2]] * g.l[sp[1]]);
    int s = 0;
    for (int d = 0; d < 4; ++d) s += x[d] * g.st[d];
    SQ l1 = {1, 0, 0, 0}, l2 = {1, 0, 0, 0}, bp = {1, 0, 0, 0}, bm = {1, 0, 0, 0};
    int s1 = s, s2 = s, spp = s, smm = s;
    for (int j = 0; j < n; ++j) {
      l1 = sq_mul(l1, U[d1 * g.V + s1]);
      s1 = gs_sh(g, s1, d1, l);
      l2 = sq_mul(l2, U[d2 * g.V + s2]);
      s2 = gs_sh(g, s2, d2, l);
      bp = sq_mul(bp, U[d1 * g.V + spp]);
      spp = gs_sh(g, spp, d1, l);
      bp = sq_mul(bp, U[d2 * g.V + spp]);
      spp = gs_sh(g, spp, d2, l);
      bm = sq_mul(bm, U[d1 * g.V + smm]);
      smm = gs_sh(g, smm, d1, l);
      smm = gs_sh(g, smm, d2, -l);
      bm = sq_mul(bm, sq_dag(U[d2 * g.V + smm]));
    }
    acc[0] += (double)sq_mul(l1, l2).x;
    acc[1] += (double)sq_mul(l1, sq_dag(l2)).x;
    acc[2] += (double)bp.x;
    acc[3] += (double)bm.x;
    acc[4] += (double)l1.x;
    acc[5] += (double)l2.x;
  }
  for (int q = 0; q < 6; ++q) {
    sh[threadIdx.x] = acc[q];
    __syncthreads();
    for (int w2 = 128; w2 > 0; w2 >>= 1) {
      if ((int)threadIdx.x < w2) sh[threadIdx.x] += sh[threadIdx.x + w2];
      __syncthreads();
    }
    if (threadIdx.x == 0) out[((((long)c * nlev + lev) * tt) + t) * 6 + q] = sh[0];
    __syncthreads();
  }
}

// (1,1,1) : deux escaliers positifs, ordres (a,b,c) et (a,c,b) pour les axes spatiaux croissants.
// L est commun et divisible par l ; L/l répétitions ferment chaque chemin sur le tore.
// Grille (T, nch), bloc 256 ; out[(chaîne * nlev + lev) * T + t] : moyenne des deux demi-traces, sommée sur la tranche.
extern "C" __global__ void k_flux_111(const SQ* __restrict__ U, const long long* __restrict__ geo, int temps, int l,
                                      int nlev, int lev, double* __restrict__ out) {
  __shared__ double sh[256];
  const GeoS g = gs_lire(geo);
  const int t = (int)blockIdx.x, chaine = (int)blockIdx.y;
  U += (long)chaine * 4 * g.V;
  int sp[3], k = 0;
  for (int d = 0; d < 4; ++d) if (d != temps) sp[k++] = d;
  const int a = sp[0], b = sp[1], c = sp[2];
  const int L = g.l[a], n = L / l;
  double acc = 0.0;
  for (int i = (int)threadIdx.x; i < g.V / g.l[temps]; i += 256) {
    int s = t * g.st[temps] + (i / (L * L)) * g.st[a] + ((i / L) % L) * g.st[b] + (i % L) * g.st[c];
    SQ abc = {1, 0, 0, 0}, acb = {1, 0, 0, 0};
    // Les deux motifs ont les mêmes départs et arrivées à chaque répétition.
    for (int j = 0; j < n; ++j) {
      const SQ ua = U[a * g.V + s];
      const int sa = gs_sh(g, s, a, l);
      const int sab = gs_sh(g, sa, b, l), sac = gs_sh(g, sa, c, l);
      abc = sq_mul(sq_mul(sq_mul(abc, ua), U[b * g.V + sa]), U[c * g.V + sab]);
      acb = sq_mul(sq_mul(sq_mul(acb, ua), U[c * g.V + sa]), U[b * g.V + sac]);
      s = gs_sh(g, sab, c, l);
    }
    acc += 0.5 * ((double)abc.x + (double)acb.x);
  }
  sh[threadIdx.x] = acc;
  __syncthreads();
  for (int w = 128; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[((long)chaine * nlev + lev) * g.l[temps] + t] = sh[0];
}
