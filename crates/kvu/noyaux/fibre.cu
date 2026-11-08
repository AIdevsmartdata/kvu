// fibre.cu — mesure de fibre sur GPU (propriétaire : T6). Source UNIQUE pour HIP (hiprtc) et CUDA (NVRTC).
//
// Assemblage : prélude du backend, puis su2_commun.h (Q, qmul, qdag, Geo, mkgeo, coords, nb_up, u32, u64),
// puis ce fichier. Aucune directive #include ici.
//
// Ce que le port calcule (= cb.mesure / cb.mesure_libre / cb.block_sums de noyau2) :
//   pv[p]    = ½ Re Tr U_p, plaquettes dans l'ordre loops_plaquettes (plan (mu<nu) majeur, puis site) ;
//   gp[p,i,a] = ½ s_i Re Tr(X_a Q_i), X_a = iσ_a/√2, gradients invariants à droite des 4 liens ;
//   F        = Wf · pv, sommation compensée de Neumaier par segments fixes de chaque ligne de Wf ;
//   D        = Σ_{éléments de fibre, a} ligne ⊗ ligne (54×54) ;
//   blk      = sommes de blocs 2⁴ des plaquettes, arrondies en float32.
//
// DÉTERMINISME : D est la somme de partiels de TRANCHES FIXES (`tranche` éléments de fibre, constante
// côté hôte), sommés dans l'ordre croissant des tranches. Ni le nombre de blocs lancés, ni leur ordre
// d'exécution, ni la taille du bloc ne changent un seul bit du résultat : chaque case de la matrice a
// UN propriétaire qui accumule les lignes dans l'ordre (élément croissant, puis a = 0, 1, 2).
// Ni atomique, ni arbre dépendant de la géométrie de lancement.
//
// ARITHMÉTIQUE : mêmes expressions, dans le même ordre, que la référence CPU Rust (src/fibre/cpu.rs).
// Compilé sans contraction FMA (OptionsCompil::reference_k1 : -ffp-contract=off / --fmad=false), le
// résultat est identique à l'octet à la référence CPU (gate FIB-3).
//
// CHAMP : FP64 (défaut) ou FP32 promu en double (FB_CHAMP_F32). Le champ d'une chaîne c commence à
// c·4V·4 réels : l'hôte passe déjà le pointeur décalé.

#ifdef FB_CHAMP_F32
typedef float fb_reel;
#else
typedef double fb_reel;
#endif

#ifndef FB_NF
#define FB_NF 54        // nombre de fonctionnelles de Fourier (6 plans × 9)
#endif
#define FB_NSMAX 27     // colonnes non nulles d'une ligne creuse : 3 plans contenant mu × 9
#define FB_LD 28        // pas des lignes en mémoire partagée (NSMAX + 1 : évite les conflits de banque)
#define FB_EB 32        // éléments de fibre traités par sous-lot
#define FB_BLOC 256     // fils par bloc de k_D_partiel
#define FB_HORS 255     // cmap : ligne de Wf hors du support
#define FB_S2I 0.7071067811865476   // 1/√2 (le littéral de cb.py)

// ------------------------------------------------------------------------------------------
// Utilitaires
// ------------------------------------------------------------------------------------------
__device__ inline Q fb_lien(const fb_reel* __restrict__ U, long e) {
  Q c = {(double)U[4 * e], (double)U[4 * e + 1], (double)U[4 * e + 2], (double)U[4 * e + 3]};
  return c;
}

// Plan (m, n), m < n, d'indice k (ordre de cb.Geometry.planes : 01 02 03 12 13 23).
__device__ inline void fb_plan(int k, int* m, int* n) {
  *m = (k < 3) ? 0 : ((k < 5) ? 1 : 2);
  *n = (k < 3) ? k + 1 : ((k < 5) ? k - 1 : 3);
}

// ------------------------------------------------------------------------------------------
// Plaquettes et gradients : une plaquette par fil
// ------------------------------------------------------------------------------------------
// Liens de la plaquette p (plan (m, n), site x) : e0 = m·V + x, e1 = n·V + x+m, e2 = m·V + x+n,
// e3 = n·V + x ; signes + + − −. Préfixes P1 = W0, P2 = W0 W1 W2⁺, Up = P2 W3⁺ ; Q_i = P_i⁺ Up P_i
// (Q_3 = Up⁺ Up Up, comme cb._k_plaq). Gradient : g[i][a] = −s_i/√2 · composante (a+1) de Q_i.
// `gp` nul : seules les valeurs pv sont écrites (mesure de F seule, sans gradients).
extern "C" __global__ void k_plaq_grad(const fb_reel* __restrict__ U, int L, double* __restrict__ pv,
                                       double* __restrict__ gp) {
  Geo g = mkgeo(L);
  long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= 6 * g.V) return;
  int k = (int)(p / g.V);
  long x = p - (long)k * g.V;
  int m, n;
  fb_plan(k, &m, &n);
  int xs[4];
  coords(g, x, xs);
  long e0 = (long)m * g.V + x;
  long e1 = (long)n * g.V + nb_up(g, x, m, xs[m]);
  long e2 = (long)m * g.V + nb_up(g, x, n, xs[n]);
  long e3 = (long)n * g.V + x;
  Q w0 = fb_lien(U, e0), w1 = fb_lien(U, e1), w2 = qdag(fb_lien(U, e2)), w3 = qdag(fb_lien(U, e3));
  Q p1 = w0;
  Q p2 = qmul(qmul(p1, w1), w2);
  Q up = qmul(p2, w3);
  pv[p] = up.x;
  if (gp == 0) return;                 // gp nul : valeurs seules (mesure légère de F)
  Q q0 = up;
  Q q1 = qmul(qmul(qdag(p1), up), p1);
  Q q2 = qmul(qmul(qdag(p2), up), p2);
  Q q3 = qmul(qmul(qdag(up), up), up);
  double* gg = gp + p * 12;
  double c0 = 1.0 * FB_S2I, c1 = 1.0 * FB_S2I, c2 = -1.0 * FB_S2I, c3 = -1.0 * FB_S2I;
  gg[0] = -c0 * q0.y;  gg[1] = -c0 * q0.z;  gg[2] = -c0 * q0.w;
  gg[3] = -c1 * q1.y;  gg[4] = -c1 * q1.z;  gg[5] = -c1 * q1.w;
  gg[6] = -c2 * q2.y;  gg[7] = -c2 * q2.z;  gg[8] = -c2 * q2.w;
  gg[9] = -c3 * q3.y;  gg[10] = -c3 * q3.z; gg[11] = -c3 * q3.w;
}

// ------------------------------------------------------------------------------------------
// F = Wf · pv : somme compensée de Neumaier par SEGMENTS FIXES (erreur ≈ 1 ulp de F)
// ------------------------------------------------------------------------------------------
// Une ligne de Wf compte ≈ L⁴ termes (un million à L = 32) : un seul fil par ligne serait presque
// sériel (54 fils). Chaque ligne est donc coupée en segments de `segment` termes CONSÉCUTIFS (constante
// côté hôte, jamais le nombre de blocs ; multiple de FB_FBLOC).
// Étape 1, k_F_seg : UN BLOC (FB_FBLOC fils) par segment. Le fil t somme les termes t, t + FB_FBLOC,
// t + 2·FB_FBLOC, … du segment (lectures coalescées) par Neumaier ; puis le fil 0 combine les paires
// (s_t, c_t) dans l'ordre t = 0 … FB_FBLOC − 1, par Neumaier sur la suite s_0, c_0, s_1, c_1, …
// Étape 2, k_F_fin : un fil par ligne combine les paires (s, c) des segments dans l'ordre, de même.
// Le résultat ne dépend que de `segment` et de FB_FBLOC (jamais de la grille de lancement).
// seg_ptr[i] = numéro du premier segment de la ligne i (nf + 1 entrées, croissant).
#define FB_FBLOC 256    // fils par bloc de k_F_seg (= largeur des voies de la somme)

__device__ inline void fb_neumaier(double* s, double* c, double x) {
  double t = *s + x;
  if (fabs(*s) >= fabs(x)) *c += (*s - t) + x;
  else *c += (x - t) + *s;
  *s = t;
}

extern "C" __global__ __launch_bounds__(FB_FBLOC) void k_F_seg(
    const u64* __restrict__ ptr, const u32* __restrict__ idx, const double* __restrict__ val,
    const double* __restrict__ pv, const u32* __restrict__ seg_ptr, int nf, int segment,
    double* __restrict__ seg_s, double* __restrict__ seg_c) {
  __shared__ double ss[FB_FBLOC], sc[FB_FBLOC];
  const u32 g = blockIdx.x;
  const int t = threadIdx.x;
  int lo = 0, hi = nf;                       // dernière ligne i telle que seg_ptr[i] <= g
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (seg_ptr[mid] <= g) lo = mid; else hi = mid;
  }
  const u64 q0 = ptr[lo] + (u64)(g - seg_ptr[lo]) * (u64)segment;
  u64 q1 = q0 + (u64)segment;
  if (q1 > ptr[lo + 1]) q1 = ptr[lo + 1];
  double s = 0.0, c = 0.0;
  for (u64 q = q0 + t; q < q1; q += FB_FBLOC) fb_neumaier(&s, &c, val[q] * pv[idx[q]]);
  ss[t] = s;
  sc[t] = c;
  __syncthreads();
  if (t == 0) {
    double s0 = 0.0, c0 = 0.0;
    for (int j = 0; j < FB_FBLOC; ++j) {
      fb_neumaier(&s0, &c0, ss[j]);
      fb_neumaier(&s0, &c0, sc[j]);
    }
    seg_s[g] = s0;
    seg_c[g] = c0;
  }
}

extern "C" __global__ void k_F_fin(const u32* __restrict__ seg_ptr, const double* __restrict__ seg_s,
                                   const double* __restrict__ seg_c, int nf, double* __restrict__ F) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nf) return;
  double s = 0.0, c = 0.0;
  for (u32 g = seg_ptr[i]; g < seg_ptr[i + 1]; ++g) {
    fb_neumaier(&s, &c, seg_s[g]);
    fb_neumaier(&s, &c, seg_c[g]);
  }
  F[i] = s + c;
}

// ------------------------------------------------------------------------------------------
// D : lignes de fibre creuses, partiels de tranches fixes
// ------------------------------------------------------------------------------------------
// Gradient du lien e projeté sur Wf, accumulé dans la ligne `ligne` (indexée par la position dans le
// support) pour la composante a : ligne[cmap[r]] += gp[p,pos,a] · Wf[r,p], plaquettes dans l'ordre de lp.
__device__ inline void fb_grad_lien(const double* __restrict__ gp, const u32* __restrict__ lp,
                                    const u32* __restrict__ wt_ptr, const unsigned char* __restrict__ wt_idx,
                                    const double* __restrict__ wt_val, const unsigned char* __restrict__ cm,
                                    long e, int a, double* ligne) {
  for (int j = 0; j < 6; ++j) {
    u32 x = lp[6 * e + j];
    long p = x >> 2;
    int pos = x & 3;
    double g = gp[(4 * p + pos) * 3 + a];
    for (u32 q = wt_ptr[p]; q < wt_ptr[p + 1]; ++q) ligne[cm[wt_idx[q]]] += g * wt_val[q];
  }
}

// Un bloc = une tranche de `tranche` éléments de fibre d'UNE direction mu. Éléments de la direction :
// d'abord les liens libres (métrique 1), puis les paires (e1, e2) de la fibre :
//   ligne_a = (∇_{e1,a} − Σ_b R_ba ∇_{e2,b}) / √2,  R_ba = ½ Re Tr(σ_b U₁⁺ σ_a U₁).
// En mode libre (libre != 0), tous les V liens de la direction sont des éléments libres (lfree ignoré).
// Écrit le partiel (FB_NF × FB_NF, symétrique, ligne majeure) de la tranche `tr0 + blockIdx.x`.
//   sup  : (4, 1 + FB_NSMAX) octets : [mu][0] = nombre de colonnes, puis les lignes de Wf du support ;
//   cmap : (4, 256) octets : position dans le support, ou FB_HORS.
extern "C" __global__ __launch_bounds__(FB_BLOC) void k_D_partiel(
    const fb_reel* __restrict__ U, const double* __restrict__ gp, const u32* __restrict__ lp,
    const u32* __restrict__ wt_ptr, const unsigned char* __restrict__ wt_idx,
    const double* __restrict__ wt_val, const u32* __restrict__ lfree, const u32* __restrict__ e1t,
    const u32* __restrict__ e2t, const unsigned char* __restrict__ sup, const unsigned char* __restrict__ cmap,
    int L, int libre, int nfree_mu, int npair_mu, int tranche, int tr_par_mu, int tr0,
    double* __restrict__ partiel) {
  __shared__ double sR[FB_EB * 3][FB_LD];           // lignes du sous-lot : (élément, a) → colonnes du support
  __shared__ double sG2[FB_EB][3][FB_LD];           // ∇_{e2,b} des paires du sous-lot
  const int tid = threadIdx.x;
  const int tr = tr0 + blockIdx.x;
  const int mu = tr / tr_par_mu;
  const int itr = tr - mu * tr_par_mu;
  const long V = (long)L * L * L * L;
  const int n_mu = libre ? (int)V : nfree_mu + npair_mu;
  const int t0 = itr * tranche;
  const int t1 = (t0 + tranche < n_mu) ? t0 + tranche : n_mu;
  const int ns = sup[mu * (1 + FB_NSMAX)];
  const unsigned char* cm = cmap + mu * 256;
  double* sortie = partiel + (long)blockIdx.x * FB_NF * FB_NF;

  // Partiel mis à zéro : seules les cases du support sont ensuite écrites par leur propriétaire.
  for (int i = tid; i < FB_NF * FB_NF; i += FB_BLOC) sortie[i] = 0.0;

  // Propriétaire des cases (ii, jj ≥ ii) du support : la case numéro c est tenue par le fil c mod 256.
  const int ncell = ns * (ns + 1) / 2;
  int cii[2] = {0, 0}, cjj[2] = {0, 0};
  for (int s = 0; s < 2; ++s) {
    int c = tid + s * FB_BLOC;
    if (c < ncell) {
      int ii = 0, r = c;
      while (r >= ns - ii) { r -= ns - ii; ++ii; }
      cii[s] = ii; cjj[s] = ii + r;
    }
  }
  double acc[2] = {0.0, 0.0};

  for (int sous = t0; sous < t1; sous += FB_EB) {
    const int nel = (t1 - sous < FB_EB) ? t1 - sous : FB_EB;

    // Phase A : ∇_{e2,b} des paires (élément le, composante b), un fil par (le, b).
    for (int w = tid; w < nel * 3; w += FB_BLOC) {
      int le = w / 3, b = w - 3 * le;
      int t = sous + le;
      for (int c = 0; c < FB_LD; ++c) sG2[le][b][c] = 0.0;
      if (!libre && t >= nfree_mu) {
        long k = (long)mu * npair_mu + (t - nfree_mu);
        fb_grad_lien(gp, lp, wt_ptr, wt_idx, wt_val, cm, (long)e2t[k], b, sG2[le][b]);
      }
    }
    __syncthreads();

    // Phase B : lignes (élément le, composante a), un fil par ligne.
    for (int w = tid; w < nel * 3; w += FB_BLOC) {
      int le = w / 3, a = w - 3 * le;
      int t = sous + le;
      double* ligne = sR[w];
      for (int c = 0; c < FB_LD; ++c) ligne[c] = 0.0;
      if (libre) {
        fb_grad_lien(gp, lp, wt_ptr, wt_idx, wt_val, cm, (long)mu * V + t, a, ligne);
      } else if (t < nfree_mu) {
        fb_grad_lien(gp, lp, wt_ptr, wt_idx, wt_val, cm, (long)lfree[(long)mu * nfree_mu + t], a, ligne);
      } else {
        long k = (long)mu * npair_mu + (t - nfree_mu);
        long e1 = e1t[k];
        fb_grad_lien(gp, lp, wt_ptr, wt_idx, wt_val, cm, e1, a, ligne);
        // R_ba = composante b de U₁⁺ (iσ_a) U₁ (quaternion pur), pour la colonne a.
        Q ea = {0.0, 0.0, 0.0, 0.0};
        if (a == 0) ea.y = 1.0; else if (a == 1) ea.z = 1.0; else ea.w = 1.0;
        Q u = fb_lien(U, e1);
        Q r = qmul(qmul(qdag(u), ea), u);
        for (int c = 0; c < ns; ++c) {
          double m = r.y * sG2[le][0][c] + r.z * sG2[le][1][c] + r.w * sG2[le][2][c];
          ligne[c] = (ligne[c] - m) * FB_S2I;
        }
      }
    }
    __syncthreads();

    // Phase C : chaque propriétaire accumule ligne ⊗ ligne dans l'ordre des lignes (le, a).
    for (int s = 0; s < 2; ++s) {
      if (tid + s * FB_BLOC < ncell) {
        double a_ = acc[s];
        for (int r = 0; r < nel * 3; ++r) a_ += sR[r][cii[s]] * sR[r][cjj[s]];
        acc[s] = a_;
      }
    }
    __syncthreads();
  }

  // Les propriétaires écrivent leur case et sa symétrique (le zéro-remplissage est visible après la barrière).
  __syncthreads();
  for (int s = 0; s < 2; ++s) {
    if (tid + s * FB_BLOC < ncell) {
      int ci = sup[mu * (1 + FB_NSMAX) + 1 + cii[s]], cj = sup[mu * (1 + FB_NSMAX) + 1 + cjj[s]];
      sortie[ci * FB_NF + cj] = acc[s];
      sortie[cj * FB_NF + ci] = acc[s];
    }
  }
}

// D := D + Σ_{partiels, ordre croissant} : un fil par case. `d` doit valoir l'accumulateur courant
// (zéro au premier lot) : les lots successifs reproduisent la somme séquentielle sur toutes les tranches.
extern "C" __global__ void k_D_final(const double* __restrict__ partiel, int npart, double* __restrict__ d) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= FB_NF * FB_NF) return;
  double s = d[c];
  for (int t = 0; t < npart; ++t) s += partiel[(long)t * FB_NF * FB_NF + c];
  d[c] = s;
}

// ------------------------------------------------------------------------------------------
// blk = block_sums(pv).astype(float32) : un fil par case de sortie (plan, bloc 2⁴)
// ------------------------------------------------------------------------------------------
// Ordre de numpy pour sum(axis=(2,4,6,8)) : les deux sites voisins selon x3 d'abord (une paire),
// puis accumulation des 8 paires dans l'ordre (d0, d1, d2), à partir de 0.
extern "C" __global__ void k_blk(const double* __restrict__ pv, int L, float* __restrict__ blk) {
  const int lc = L / 2;
  const long nb = (long)lc * lc * lc * lc;
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 6 * nb) return;
  long k = i / nb, b = i - k * nb;
  long b3 = b % lc, b2 = (b / lc) % lc, b1 = (b / ((long)lc * lc)) % lc, b0 = b / ((long)lc * lc * lc);
  const long V = (long)L * L * L * L;
  double acc = 0.0;
  for (int d0 = 0; d0 < 2; ++d0)
    for (int d1 = 0; d1 < 2; ++d1)
      for (int d2 = 0; d2 < 2; ++d2) {
        long s = ((((2 * b0 + d0) * L + 2 * b1 + d1) * L + 2 * b2 + d2) * L) + 2 * b3;
        acc += pv[k * V + s] + pv[k * V + s + 1];
      }
  blk[i] = (float)acc;
}
