// obs_wilson.cu — propriétaire : T7. Boucles de Wilson R x T, boucles de Polyakov, corrélateur de
// Polyakov, et briques communes de l'observable GPU (lecture F64 ou F32 promu, réductions FP64).
//
// ASSEMBLAGE : prélude du backend, su2_commun.h (Q, qmul, qdag, Geo, mkgeo, nb_up, nb_dn, coords,
// ld, staple), PUIS ce fichier, PUIS obs_flot.cu. Le texte ne contient aucun #include.
//
// Conventions (celles de su2_hip.cpp) : quaternion U = a0 + i(a1 s1 + a2 s2 + a3 s3) stocké
// (x, y, z, w) ; lien li = mu*V + s, s = ((x0*L + x1)*L + x2)*L + x3 ; 1/2 Tr U = x ;
// 1/2 Tr (A B^+) = A.x B.x + A.y B.y + A.z B.z + A.w B.w (produit scalaire).
//
// Champ F32 : quatre float par lien (x, y, z, w), même indexation. Il est PROMU en double à la
// lecture : tous les produits et toutes les sommes sont en FP64.
//
// Réductions : arbre fixe de 256 fils par bloc (même schéma que k_reduce / k_final de su2_hip),
// puis somme finale à arbre fixe. Le résultat est déterministe d'un lancement à l'autre.
// Les noms de ce fichier commencent par obs_ ou k_obs_ : aucune collision avec su2_ref64.cu.

#define OBS_BLOC 256
// Nombre maximal de longueurs R traitées par un fil (tableau de produits en registres / pile locale).
#define OBS_MAXR 32
#define OBS_PI 3.14159265358979323846

// Lien F32 : quatre float contigus (pas de dépendance au type vectoriel du backend).
struct ObsQF { float x, y, z, w; };

// ------------------------------------------------------------------------------------------
// Lecture d'un lien : F64 tel quel, F32 promu en double
// ------------------------------------------------------------------------------------------
__device__ inline Q obs_ld(const Q* __restrict__ U, long li) { return U[li]; }
__device__ inline Q obs_ld(const ObsQF* __restrict__ U, long li) {
  ObsQF f = U[li];
  Q q = {(double)f.x, (double)f.y, (double)f.z, (double)f.w};
  return q;
}

// Site x + a e_mu + b e_nu (a, b signés, périodique). x = coordonnées du site de départ.
__device__ inline long obs_pos(const Geo& g, const int x[4], int mu, int a, int nu, int b) {
  int c[4] = {x[0], x[1], x[2], x[3]};
  c[mu] = (((x[mu] + a) % g.L) + g.L) % g.L;
  c[nu] = (((x[nu] + b) % g.L) + g.L) % g.L;
  return (((long)c[0] * g.L + c[1]) * g.L + c[2]) * g.L + c[3];
}

__device__ inline double obs_dot(Q a, Q b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

// Réduction de 256 valeurs (une par fil) à arbre fixe ; résultat dans sh[0] (valable pour le fil 0).
// Tous les fils du bloc DOIVENT appeler cette fonction (barrières).
__device__ inline void obs_arbre(double* sh) {
  __syncthreads();
  for (int w = OBS_BLOC / 2; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
    __syncthreads();
  }
}

// ------------------------------------------------------------------------------------------
// Boucles de Wilson : un bloc = 256 sites consécutifs, une orientation (mu, nu) par lancement.
//
// Pour le site x, la boucle R x T (R pas selon mu, T pas selon nu) vaut
//   W = 1/2 Tr [ P1 P2^+ ] = P1 . P2     avec
//   P1(r,t) = A_r(x) B_t(x + r mu)        (d'abord mu, puis nu)
//   P2(r,t) = B_t(x) A_r(x + t nu)        (d'abord nu, puis mu)
// où A_r(y) = produit de r liens selon mu depuis y et B_t(y) = produit de t liens selon nu depuis y.
// Deux récurrences évitent de refaire les produits : P1 se prolonge d'un lien en t, P2 d'un lien en r.
// Pour t = 1..tmax (boucle externe), le fil garde P1(r, t) pour tous les r dans un tableau local,
// et recalcule P2(r, t) en r croissant (un produit par pas).
//
// Sortie : part[bloc*rmax*tmax + (r-1)*tmax + (t-1)] = somme du bloc de W(r,t) (arbre fixe).
// ------------------------------------------------------------------------------------------
template <class TL>
__device__ void obs_wilson_corps(const TL* __restrict__ U, int Lg, int mu, int nu, int rmax, int tmax,
                                 double* __restrict__ part) {
  Geo g = mkgeo(Lg);
  __shared__ double sh[OBS_BLOC];
  long s = (long)blockIdx.x * OBS_BLOC + threadIdx.x;
  bool ok = s < g.V;
  int x[4] = {0, 0, 0, 0};
  if (ok) coords(g, s, x);
  Q Pr[OBS_MAXR];
  if (ok) {
    Q a = {1.0, 0.0, 0.0, 0.0};
    for (int r = 1; r <= rmax; ++r) {                       // A_r(x)
      a = qmul(a, obs_ld(U, (long)mu * g.V + obs_pos(g, x, mu, r - 1, nu, 0)));
      Pr[r - 1] = a;
    }
  }
  Q B = {1.0, 0.0, 0.0, 0.0};                                // B_t(x)
  for (int t = 1; t <= tmax; ++t) {
    if (ok) {
      for (int r = 1; r <= rmax; ++r)                        // P1(r,t) = P1(r,t-1) U_nu(x + r mu + (t-1) nu)
        Pr[r - 1] = qmul(Pr[r - 1], obs_ld(U, (long)nu * g.V + obs_pos(g, x, mu, r, nu, t - 1)));
      B = qmul(B, obs_ld(U, (long)nu * g.V + obs_pos(g, x, mu, 0, nu, t - 1)));
    }
    Q P2 = B;                                                // P2(0,t) = B_t(x)
    for (int r = 1; r <= rmax; ++r) {
      double w = 0.0;
      if (ok) {
        P2 = qmul(P2, obs_ld(U, (long)mu * g.V + obs_pos(g, x, mu, r - 1, nu, t)));   // U_mu(x + t nu + (r-1) mu)
        w = obs_dot(Pr[r - 1], P2);
      }
      sh[threadIdx.x] = w;
      obs_arbre(sh);
      if (threadIdx.x == 0) part[(long)blockIdx.x * rmax * tmax + (long)(r - 1) * tmax + (t - 1)] = sh[0];
      __syncthreads();
    }
  }
}

extern "C" __global__ void k_obs_wilson(const Q* U, int Lg, int mu, int nu, int rmax, int tmax, double* part) {
  obs_wilson_corps<Q>(U, Lg, mu, nu, rmax, tmax, part);
}
extern "C" __global__ void k_obs_wilson_f32(const ObsQF* U, int Lg, int mu, int nu, int rmax, int tmax,
                                            double* part) {
  obs_wilson_corps<ObsQF>(U, Lg, mu, nu, rmax, tmax, part);
}

// ------------------------------------------------------------------------------------------
// Somme finale à arbre fixe : un bloc par sortie idx = blockIdx.x,
//   out[idx] = scale * somme_i in[i*stride + idx]   (i = 0..n-1)
// Avec stride = 1, idx = 0 et un seul bloc, c'est exactement k_final de su2_hip.
// ------------------------------------------------------------------------------------------
extern "C" __global__ void k_obs_final(const double* __restrict__ in, long n, long stride, double scale,
                                       double* __restrict__ out) {
  __shared__ double sh[OBS_BLOC];
  double a = 0.0;
  for (long i = threadIdx.x; i < n; i += OBS_BLOC) a += in[i * stride + blockIdx.x];
  sh[threadIdx.x] = a;
  __syncthreads();
  for (int w = OBS_BLOC / 2; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = sh[0] * scale;
}

// ------------------------------------------------------------------------------------------
// Boucles de Polyakov : ell[j] = 1/2 Tr prod_k U_dir(x, k) pour chaque site transverse j.
// Même numérotation que k_polyakov de su2_hip : les 3 directions transverses sont prises par
// ordre croissant ; x[dirs[2]] = j mod L, x[dirs[1]] = (j/L) mod L, x[dirs[0]] = j/L^2.
// ------------------------------------------------------------------------------------------
template <class TL>
__device__ void obs_polyakov_corps(const TL* __restrict__ U, int Lg, int dir, double* __restrict__ out) {
  Geo g = mkgeo(Lg);
  long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long L3 = g.V / g.L;
  if (j >= L3) return;
  int x[4];
  x[dir] = 0;
  int dirs[3], nd = 0;
  for (int d = 0; d < 4; ++d) if (d != dir) dirs[nd++] = d;
  x[dirs[2]] = (int)(j % g.L);
  long t = j / g.L;
  x[dirs[1]] = (int)(t % g.L);
  x[dirs[0]] = (int)(t / g.L);
  long s = ((long)(x[0] * g.L + x[1]) * g.L + x[2]) * g.L + x[3];
  Q P = {1.0, 0.0, 0.0, 0.0};
  for (int k = 0; k < g.L; ++k) {
    P = qmul(P, obs_ld(U, (long)dir * g.V + s));
    s += (k == g.L - 1) ? 0 : g.st[dir];
  }
  out[j] = P.x;
}

extern "C" __global__ void k_obs_polyakov(const Q* U, int Lg, int dir, double* out) {
  obs_polyakov_corps<Q>(U, Lg, dir, out);
}
extern "C" __global__ void k_obs_polyakov_f32(const ObsQF* U, int Lg, int dir, double* out) {
  obs_polyakov_corps<ObsQF>(U, Lg, dir, out);
}

// Corrélateur : pour r = 0..rmax, somme sur j de ell[j] (ell[j + r e_0] + ell[j + r e_1] + ell[j + r e_2]),
// les trois axes étant ceux du réseau transverse (c0, c1, c2) = (j/L^2, (j/L) mod L, j mod L).
// Sortie : part[bloc*(rmax+1) + r], sommes de bloc à arbre fixe.
extern "C" __global__ void k_obs_pcorr(const double* __restrict__ ell, int Lg, int rmax,
                                       double* __restrict__ part) {
  __shared__ double sh[OBS_BLOC];
  long L = Lg;
  long L3 = L * L * L;
  long j = (long)blockIdx.x * OBS_BLOC + threadIdx.x;
  bool ok = j < L3;
  long c[3] = {0, 0, 0};
  if (ok) { c[0] = j / (L * L); c[1] = (j / L) % L; c[2] = j % L; }
  for (int r = 0; r <= rmax; ++r) {
    double v = 0.0;
    if (ok) {
      double e0 = ell[j];
      long j0 = (((c[0] + r) % L) * L + c[1]) * L + c[2];
      long j1 = (c[0] * L + ((c[1] + r) % L)) * L + c[2];
      long j2 = (c[0] * L + c[1]) * L + ((c[2] + r) % L);
      v = e0 * ell[j0] + e0 * ell[j1] + e0 * ell[j2];
    }
    sh[threadIdx.x] = v;
    obs_arbre(sh);
    if (threadIdx.x == 0) part[(long)blockIdx.x * (rmax + 1) + r] = sh[0];
    __syncthreads();
  }
}
