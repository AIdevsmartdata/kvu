// obs_flot.cu — propriétaire : T7. Flot de Wilson RK3 (Lüscher 2010) en quaternions : force Z,
// exponentielle exacte de su(2), densités d'énergie (plaquette, trèfle) et charge topologique.
//
// ASSEMBLAGE : après su2_commun.h ET obs_wilson.cu (qui fournit obs_ld, obs_pos, obs_dot, obs_arbre,
// ObsQF, k_obs_final). Aucun #include.
//
// Le flot travaille TOUJOURS sur une COPIE résidente en FP64 du champ (k_obs_copie / _f32) :
// le champ source n'est jamais écrit. Les forces Z0, Z1, Z2 sont des tampons FP64 de 3 doubles
// par lien (composantes de l'algèbre su(2), indice 3*li + k, li = mu*V + s).
//
// Conventions identiques à src/observables/wilson_flow.rs (coefficients RK3, signe de z_field,
// réunitarisation après chaque mise à jour) et à src/obs/ref_cpu.rs :
//   Z      = -Ta(U . A) = -(m.y, m.z, m.w) avec m = U . A (A = somme des 6 agrafes) ;
//   étape 0 : w = 1/4 dt Z0 ;
//   étape 1 : w = dt (8/9 Z1 - 17/36 Z0) ;
//   étape 2 : w = dt (3/4 Z2 + 17/36 Z0 - 8/9 Z1) ;
//   V <- exp(w) V, exp(0, w) = (cos|w|, sin|w|/|w| w), puis V <- V / |V|.
// Les expressions flottantes sont écrites dans le même ordre que la référence CPU ; compiler sans
// contraction (reference_k1) pour retrouver le CPU à l'arrondi près.

// Copie du champ d'une chaîne vers le tampon de travail FP64 (n = 4V liens).
extern "C" __global__ void k_obs_copie(const Q* __restrict__ src, Q* __restrict__ dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}
extern "C" __global__ void k_obs_copie_f32(const ObsQF* __restrict__ src, Q* __restrict__ dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = obs_ld(src, i);
}

// Force : un fil par lien (4V fils).
extern "C" __global__ void k_obs_z(const Q* __restrict__ V, int Lg, double* __restrict__ Z) {
  Geo g = mkgeo(Lg);
  long li = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (li >= 4 * g.V) return;
  int mu = (int)(li / g.V);
  long s = li % g.V;
  int x[4];
  coords(g, s, x);
  Q A = staple(V, g, mu, s, x);
  Q m = qmul(V[li], A);
  Z[3 * li + 0] = -m.y;
  Z[3 * li + 1] = -m.z;
  Z[3 * li + 2] = -m.w;
}

// exp(0, w) : exponentielle exacte de su(2) (série de sin(th)/th en dessous de 1e-8).
__device__ inline Q obs_exp_su2(double w0, double w1, double w2) {
  double th = sqrt(w0 * w0 + w1 * w1 + w2 * w2);
  double sinc = (th < 1e-8) ? 1.0 - th * th / 6.0 : sin(th) / th;
  Q e = {cos(th), sinc * w0, sinc * w1, sinc * w2};
  return e;
}

// Mise à jour V <- exp(w) V, un fil par lien. `etape` = 0, 1, 2.
extern "C" __global__ void k_obs_maj(Q* __restrict__ V, long n, const double* __restrict__ Z0,
                                     const double* __restrict__ Z1, const double* __restrict__ Z2, int etape,
                                     double dt) {
  long li = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (li >= n) return;
  double w[3];
  for (int k = 0; k < 3; ++k) {
    double z0 = Z0[3 * li + k];
    if (etape == 0) {
      w[k] = z0 * (dt * 0.25);
    } else if (etape == 1) {
      w[k] = (8.0 / 9.0 * Z1[3 * li + k] + (-17.0 / 36.0) * z0) * dt;
    } else {
      w[k] = ((3.0 / 4.0 * Z2[3 * li + k] + 17.0 / 36.0 * z0) * 1.0 + (-8.0 / 9.0) * Z1[3 * li + k]) * dt;
    }
  }
  Q e = obs_exp_su2(w[0], w[1], w[2]);
  Q u = qmul(e, V[li]);
  V[li] = qscale(u, 1.0 / qnorm(u));
}

// Densités par site : somme des 6 plaquettes, énergie de trèfle, densité topologique.
//
// Feuilles du trèfle d'un plan (mu, nu) autour de x (décalages (a, b) = x + a e_mu + b e_nu) :
//   F1 (mu,+nu)  : U_mu(0,0) U_nu(1,0) U_mu(0,1)^+ U_nu(0,0)^+
//   F2 (nu,-mu)  : U_nu(0,0) U_mu(-1,1)^+ U_nu(-1,0)^+ U_mu(-1,0)
//   F3 (-mu,-nu) : U_mu(-1,0)^+ U_nu(-1,-1)^+ U_mu(-1,-1) U_nu(0,-1)
//   F4 (-nu,mu)  : U_nu(0,-1)^+ U_mu(0,-1) U_nu(1,-1) U_mu(0,0)^+
// g_munu = (partie vectorielle de F1 + F2 + F3 + F4) / 4 ;  E(x) = 2 sum_{mu<nu} |g_munu|^2 ;
// q(x) = (1/2pi^2) (g01.g23 - g02.g13 + g03.g12).
__device__ inline void obs_trefle_plan(const Q* __restrict__ V, const Geo& g, const int x[4], int mu, int nu,
                                       double gv[3]) {
  long V4 = g.V;
  #define LK(d, a, b) V[(long)(d) * V4 + obs_pos(g, x, mu, (a), nu, (b))]
  Q f1 = qmul(qmul(LK(mu, 0, 0), LK(nu, 1, 0)), qmul(qdag(LK(mu, 0, 1)), qdag(LK(nu, 0, 0))));
  Q f2 = qmul(qmul(LK(nu, 0, 0), qdag(LK(mu, -1, 1))), qmul(qdag(LK(nu, -1, 0)), LK(mu, -1, 0)));
  Q f3 = qmul(qmul(qdag(LK(mu, -1, 0)), qdag(LK(nu, -1, -1))), qmul(LK(mu, -1, -1), LK(nu, 0, -1)));
  Q f4 = qmul(qmul(qdag(LK(nu, 0, -1)), LK(mu, 0, -1)), qmul(LK(nu, 1, -1), qdag(LK(mu, 0, 0))));
  #undef LK
  Q s = qadd(qadd(qadd(f1, f2), f3), f4);
  gv[0] = 0.25 * s.y;
  gv[1] = 0.25 * s.z;
  gv[2] = 0.25 * s.w;
}

extern "C" __global__ void k_obs_densites(const Q* __restrict__ V, int Lg, double* __restrict__ part) {
  Geo g = mkgeo(Lg);
  __shared__ double sh[OBS_BLOC];
  long s = (long)blockIdx.x * OBS_BLOC + threadIdx.x;
  bool ok = s < g.V;
  double plaq = 0.0, ener = 0.0, topo = 0.0;
  if (ok) {
    int x[4];
    coords(g, s, x);
    for (int mu = 0; mu < 4; ++mu) {                       // plaquettes (même expression que k_plaq_site)
      long smu = nb_up(g, s, mu, x[mu]);
      for (int nu = mu + 1; nu < 4; ++nu) {
        long snu = nb_up(g, s, nu, x[nu]);
        Q p = qmul(qmul(ld(V, g, mu, s), ld(V, g, nu, smu)), qmul(qdag(ld(V, g, mu, snu)), qdag(ld(V, g, nu, s))));
        plaq += p.x;
      }
    }
    double gp[6][3];
    int p = 0;
    for (int mu = 0; mu < 4; ++mu)
      for (int nu = mu + 1; nu < 4; ++nu) obs_trefle_plan(V, g, x, mu, nu, gp[p++]);
    double e = 0.0;
    for (int k = 0; k < 6; ++k) e += gp[k][0] * gp[k][0] + gp[k][1] * gp[k][1] + gp[k][2] * gp[k][2];
    ener = 2.0 * e;
    // plans dans l'ordre 01, 02, 03, 12, 13, 23 : compléments 23, 13, 12 ; signes +, -, +
    double q = 0.0;
    q += 1.0 * (gp[0][0] * gp[5][0] + gp[0][1] * gp[5][1] + gp[0][2] * gp[5][2]);
    q += -1.0 * (gp[1][0] * gp[4][0] + gp[1][1] * gp[4][1] + gp[1][2] * gp[4][2]);
    q += 1.0 * (gp[2][0] * gp[3][0] + gp[2][1] * gp[3][1] + gp[2][2] * gp[3][2]);
    topo = (1.0 / (2.0 * OBS_PI * OBS_PI)) * q;
  }
  sh[threadIdx.x] = plaq;
  obs_arbre(sh);
  if (threadIdx.x == 0) part[(long)blockIdx.x * 3 + 0] = sh[0];
  __syncthreads();
  sh[threadIdx.x] = ener;
  obs_arbre(sh);
  if (threadIdx.x == 0) part[(long)blockIdx.x * 3 + 1] = sh[0];
  __syncthreads();
  sh[threadIdx.x] = topo;
  obs_arbre(sh);
  if (threadIdx.x == 0) part[(long)blockIdx.x * 3 + 2] = sh[0];
}


// WP19 : topologie améliorée et cooling. Ajout seul au module historique « obs ».
__device__ inline double topo_plaq_site(const Q* V, const Geo& g, long s, const int x[4]) {
  double somme = 0.0;
  for (int mu = 0; mu < 4; ++mu) {
    long smu = nb_up(g, s, mu, x[mu]);
    for (int nu = mu + 1; nu < 4; ++nu) {
      long snu = nb_up(g, s, nu, x[nu]);
      Q p = qmul(qmul(ld(V, g, mu, s), ld(V, g, nu, smu)),
                 qmul(qdag(ld(V, g, mu, snu)), qdag(ld(V, g, nu, s))));
      somme += p.x;
    }
  }
  return somme;
}

// Même produit séquentiel depuis l'identité que topo::trefle_mxm / produit_chemin,
// y compris m=1 ; mêmes feuilles que obs_trefle_plan.
__device__ inline void topo_trefle_mxm(const Q* V, const Geo& g, const int x[4],
                                      int mu, int nu, int m, double gv[3]) {
  const int dirs[4] = {mu, nu, mu, nu};
  const int sens[4] = {1, 1, -1, -1};
  Q somme = {0.0, 0.0, 0.0, 0.0};
  for (int feuille = 0; feuille < 4; ++feuille) {
    Q produit = {1.0, 0.0, 0.0, 0.0};
    int a = 0, b = 0;
    for (int cote = 0; cote < 4; ++cote) {
      int j = (feuille + cote) % 4;
      int d = dirs[j], signe = sens[j];
      for (int pas = 0; pas < m; ++pas) {
        if (signe < 0) {
          if (d == mu) --a; else --b;
        }
        Q lien = V[(long)d * g.V + obs_pos(g, x, mu, a, nu, b)];
        produit = qmul(produit, signe > 0 ? lien : qdag(lien));
        if (signe > 0) {
          if (d == mu) ++a; else ++b;
        }
      }
    }
    somme = qadd(somme, produit);
  }
  gv[0] = 0.25 * somme.y;
  gv[1] = 0.25 * somme.z;
  gv[2] = 0.25 * somme.w;
}

__device__ inline double topo_densite_q(const double g[6][3]) {
  double q = 0.0;
  q += g[0][0] * g[5][0] + g[0][1] * g[5][1] + g[0][2] * g[5][2];
  q += -(g[1][0] * g[4][0] + g[1][1] * g[4][1] + g[1][2] * g[4][2]);
  q += g[2][0] * g[3][0] + g[2][1] * g[3][1] + g[2][2] * g[3][2];
  return (1.0 / (2.0 * OBS_PI * OBS_PI)) * q;
}

extern "C" __global__ void k_topo_densites(const Q* V, int Lg, double* part) {
  Geo g = mkgeo(Lg);
  __shared__ double sh[OBS_BLOC];
  long s = (long)blockIdx.x * OBS_BLOC + threadIdx.x;
  double valeurs[3] = {0.0, 0.0, 0.0};
  if (s < g.V) {
    int x[4];
    coords(g, s, x);
    valeurs[0] = topo_plaq_site(V, g, s, x);
    double g1[6][3], ameliore[6][3];
    int p = 0;
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = mu + 1; nu < 4; ++nu, ++p) {
        double g2[3], g3[3];
        topo_trefle_mxm(V, g, x, mu, nu, 1, g1[p]);
        topo_trefle_mxm(V, g, x, mu, nu, 2, g2);
        topo_trefle_mxm(V, g, x, mu, nu, 3, g3);
        for (int k = 0; k < 3; ++k)
          ameliore[p][k] = (3.0 / 2.0) * g1[p][k] / 1.0
                        + (-3.0 / 5.0) * g2[k] / 4.0
                        + (1.0 / 10.0) * g3[k] / 9.0;
      }
    }
    valeurs[1] = topo_densite_q(g1);
    valeurs[2] = topo_densite_q(ameliore);
  }
  for (int k = 0; k < 3; ++k) {
    sh[threadIdx.x] = valeurs[k];
    obs_arbre(sh);
    if (threadIdx.x == 0) part[(long)blockIdx.x * 3 + k] = sh[0];
    __syncthreads();
  }
}

extern "C" __global__ void k_topo_plaq(const Q* V, int Lg, double* part) {
  Geo g = mkgeo(Lg);
  __shared__ double sh[OBS_BLOC];
  long s = (long)blockIdx.x * OBS_BLOC + threadIdx.x;
  double plaq = 0.0;
  if (s < g.V) {
    int x[4];
    coords(g, s, x);
    plaq = topo_plaq_site(V, g, s, x);
  }
  sh[threadIdx.x] = plaq;
  obs_arbre(sh);
  if (threadIdx.x == 0) part[blockIdx.x] = sh[0];
}

__device__ inline Q topo_maj_cooling(Q w, int dagger) {
  double norme = qnorm(w);
  if (norme == 0.0) {
    Q un = {1.0, 0.0, 0.0, 0.0};
    return un;
  }
  return qscale(dagger ? qdag(w) : w, 1.0 / norme);
}

// Pas de restrict : lecture et écriture en place. Les classes (mu, parité)
// sont indépendantes pour L pair ; aucun lien lu n'est écrit par un autre fil.
extern "C" __global__ void k_topo_cool(Q* V, int Lg, int mu, int parite, int dagger) {
  Geo g = mkgeo(Lg);
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= g.V) return;
  int x[4];
  coords(g, s, x);
  if (((x[0] + x[1] + x[2] + x[3]) & 1) != parite) return;
  long smu = nb_up(g, s, mu, x[mu]);
  Q w = {0.0, 0.0, 0.0, 0.0};
  for (int nu = 0; nu < 4; ++nu) {
    if (nu == mu) continue;
    long snu = nb_up(g, s, nu, x[nu]);
    long sdn = nb_dn(g, s, nu, x[nu]);
    long smudn = nb_dn(g, smu, nu, x[nu]);
    Q haut = qmul(qmul(ld(V, g, nu, smu), qdag(ld(V, g, mu, snu))), qdag(ld(V, g, nu, s)));
    Q bas = qmul(qmul(qdag(ld(V, g, nu, smudn)), qdag(ld(V, g, mu, sdn))), ld(V, g, nu, sdn));
    w = qadd(w, qadd(haut, bas));
  }
  V[(long)mu * g.V + s] = topo_maj_cooling(w, dagger);
}

extern "C" __global__ void k_topo_cool_id(const Q* W, Q* U, long n, int dagger) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) U[i] = topo_maj_cooling(W[i], dagger);
}
