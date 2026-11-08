// Assemblé après su2_commun.h, sans contraction multiplication-addition.
// Tables en mémoire globale en lecture seule : crate::gpu n'expose pas la mémoire constante.
static const int RG_MAXN = 24;

__device__ inline Q rg_lire(const Q* __restrict__ U, const Geo& g,
                          const int origine[4], const int dep[4],
                          int axe_translation, int mu) {
  long site = 0;
  for (int m = 0; m < 4; ++m) {
    long x = ((long)origine[m] + dep[m] + (m == axe_translation ? 3 : 0)) % g.L;
    if (x < 0) x += g.L;
    site = site * g.L + x;
  }
  return U[(long)mu * g.V + site];
}

__device__ inline Q rg_produit(const Q* __restrict__ U, const Geo& g,
                             const int origine[4], const unsigned* __restrict__ steps,
                             int debut, int fin, int axe_translation) {
  Q p = {1.0, 0.0, 0.0, 0.0};
  for (int k = debut; k < fin; ++k) {
    unsigned mot = steps[k];
    int dep[4];
    for (int m = 0; m < 4; ++m) dep[m] = (int)((mot >> (4 * m)) & 15u) - 8;
    Q u = rg_lire(U, g, origine, dep, axe_translation, (int)((mot >> 16) & 3u));
    p = qmul(p, (mot & (1u << 18)) ? u : qdag(u));
  }
  return p;
}

__device__ inline Q rg_segment(const Q* __restrict__ U, const Geo& g,
                             const int origine[4], int mu, int ir) {
  int dep[4];
  for (int m = 0; m < 4; ++m) { dep[m] = ir % 3; ir /= 3; }
  Q p = {1.0, 0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k) {
    p = qmul(p, rg_lire(U, g, origine, dep, -1, mu));
    ++dep[mu];
  }
  return p;
}

__device__ inline void rg_log_q(Q w, double ell[3], double& theta, bool& ambigue) {
  double norme = hypot(hypot(w.y, w.z), w.w);
  theta = atan2(norme, w.x);
  double facteur;
  if (norme < 1e-6) {
    double n2 = norme * norme;
    facteur = 1.0 + n2 / 6.0 + 7.0 * n2 * n2 / 360.0;
  } else {
    facteur = theta / norme;
  }
  ell[0] = facteur * w.y;
  ell[1] = facteur * w.z;
  ell[2] = facteur * w.w;
  ambigue = M_PI - theta < 1e-6;
}

__device__ inline Q rg_exp_q(const double s[3]) {
  double norme = hypot(hypot(s[0], s[1]), s[2]);
  double facteur;
  if (norme < 1e-6) {
    double n2 = norme * norme;
    facteur = 1.0 - n2 / 6.0 + n2 * n2 / 120.0;
  } else {
    facteur = sin(norme) / norme;
  }
  Q q = {cos(norme), facteur * s[0], facteur * s[1], facteur * s[2]};
  return q;
}

extern "C" __global__ void k_rg_bloc(
    const Q* __restrict__ U, int L,
    const unsigned* __restrict__ steps, const int* __restrict__ ch_ir,
    const int* __restrict__ ch_pas, const double* __restrict__ poids, int ir_ref,
    Q* __restrict__ out, unsigned* __restrict__ hist,
    unsigned* __restrict__ cnt, double* __restrict__ tmax) {
  long li = (long)blockIdx.x * blockDim.x + threadIdx.x;
  Geo gc = mkgeo(L / 3);
  if (li >= 4 * gc.V) return;
  Geo g = mkgeo(L);
  int mu = (int)(li / gc.V);
  int origine[4];
  coords(gc, li % gc.V, origine);
  for (int m = 0; m < 4; ++m) origine[m] *= 3;
  Q u0 = rg_segment(U, g, origine, mu, ir_ref);
  Q inverse = qdag(u0);
  double s[3] = {0.0, 0.0, 0.0};
  unsigned n_hors = 0, n_ambigues = 0;
  double theta_max = 0.0;
  if (hist) {
    for (int c = 0; c < 64; ++c) hist[li * 64 + c] = 0;
  }
  Q depart[RG_MAXN], retour[RG_MAXN];
  for (int ir = 0; ir < 81; ++ir) {
    int debut = ch_ir[ir], n = ch_ir[ir + 1] - debut;
    for (int i = 0; i < n; ++i) {
      int c = debut + i;
      depart[i] = rg_produit(U, g, origine, steps, ch_pas[c], ch_pas[c + 1], -1);
      retour[i] = qdag(rg_produit(U, g, origine, steps, ch_pas[c], ch_pas[c + 1], mu));
    }
    Q segment = rg_segment(U, g, origine, mu, ir);
    for (int i = 0; i < n; ++i) {
      Q pre = qmul(qmul(inverse, depart[i]), segment);
      for (int j = 0; j < n; ++j) {
        double ell[3], theta;
        bool ambigue;
        rg_log_q(qmul(pre, retour[j]), ell, theta, ambigue);
        for (int m = 0; m < 3; ++m) s[m] += poids[ir] * ell[m];
        if (hist) {
          int classe = (int)floor(theta * 64.0 / M_PI);
          if (classe > 63) classe = 63;
          ++hist[li * 64 + classe];
          n_hors += theta > M_PI / 2.0;
          n_ambigues += ambigue;
          theta_max = fmax(theta_max, theta);
        }
      }
    }
  }
  out[li] = qmul(u0, rg_exp_q(s));
  if (hist) {
    cnt[li * 2] = n_hors;
    cnt[li * 2 + 1] = n_ambigues;
    tmax[li] = theta_max;
  }
}
