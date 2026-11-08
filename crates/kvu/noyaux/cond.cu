// cond.cu — bain thermique conditionnel « fin sachant grossier » (propriétaire : T5).
//
// Assemblé APRÈS su2_commun.h (Philox4x32-10, u01, Q et son algèbre, Geo/mkgeo/coords, staple) : ce fichier
// n'en redéfinit rien. Port GPU de noyau2/cb.py (sweep, _k_pairs_prep/fin, _k_free_prep/fin, _hb_rapide).
//
// Rappel de la décimation. Pour chaque direction mu, une « paire » est (x, x+mu) avec x_nu pair pour tout
// nu != mu et x_mu pair ; son produit U_mu(x) U_mu(x+mu) = V est FIGÉ. Les autres liens de direction mu dont
// les coordonnées transverses ne sont pas toutes paires sont « libres ». Un balayage (= cb.sweep) est :
//   réunitarisation ; pour mu = 0..3 : paires ; puis pour mu = 0..3 et parité p = 0, 1 : liens libres.
//
// Deux modes :
//   ref_rng : k_cond_*_prep calcule l'agrafe effective, Vn = A/k et alpha = beta k ; l'HÔTE tire (x0, r, v)
//             avec le rejeu exact du générateur numpy ; k_cond_*_fin écrit les liens. Mêmes tirages que cb.py.
//   rapide  : k_cond_*_rapide fait tout sur le GPU. Philox à compteur (li, sweep_id, tirage, TAG_COND), li =
//             indice LOCAL du lien (mu·V + s ; pour une paire, celui de son premier lien), clé = graine.
//
// Conventions : champ de quaternions (x=a0, y=a1, z=a2, w=a3), lien li = mu·V + s ; la matrice
// [[a0 + i a3, a2 + i a1], [-a2 + i a1, a0 - i a3]] est celle de cb.py. Les pointeurs de champ visent la
// chaîne 0 (le moteur conditionnel pilote une chaîne à la fois). Aucune FMA n'est supposée : le code est
// compilé -ffp-contract=off / --fmad=false pour rester à 1e-12 de cb.py (numba fastmath=False).

enum : u32 { TAG_COND = 0x434F4E44u };   // « COND »
#define COND_PLAFOND 1000000u            // essais maximaux d'un tirage de x0 (comme K1)

// ------------------------------------------------------------------------------------------
// Réunitarisation : q <- q / |q| (= _k_reunit)
// ------------------------------------------------------------------------------------------
extern "C" __global__ void k_cond_reunit(Q* U, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Q q = U[i];
  double nn = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  Q c = {q.x / nn, q.y / nn, q.z / nn, q.w / nn};
  U[i] = c;
}

// ------------------------------------------------------------------------------------------
// Briques communes
// ------------------------------------------------------------------------------------------

// k = sqrt(max(|A|^2, 1e-300)), Vn = A / k, alpha = beta k (= _prep1)
__device__ inline void cond_preparer(Q a, double beta, Q* vn, double* alpha) {
  double n2 = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
  double k = sqrt(fmax(n2, 1e-300));
  Q v = {a.x / k, a.y / k, a.z / k, a.w / k};
  *vn = v;
  *alpha = beta * k;
}

// A_eff = A1 + A2^+ V^+ : agrafes aux deux liens de la paire (en s1 et s2 = s1 + mu), V figé.
__device__ inline Q cond_agrafe_paire(const Q* U, const Geo& g, int mu, long s1, long s2, Q v) {
  int x1[4], x2[4];
  coords(g, s1, x1);
  coords(g, s2, x2);
  Q a1 = staple(U, g, mu, s1, x1);
  Q a2 = staple(U, g, mu, s2, x2);
  return qadd(a1, qmul(qdag(a2), qdag(v)));
}

__device__ inline Q cond_agrafe_libre(const Q* U, const Geo& g, int mu, long s) {
  int x[4];
  coords(g, s, x);
  return staple(U, g, mu, s, x);
}

// U1 = W Vn^+ avec W = (x0, r v0, r v1, r v2) (= _hb_U) ; tir = (x0, r, v0, v1, v2) pour l'élément i
__device__ inline Q cond_lien_depuis_tirage(const double* tir, long i, Q vn) {
  const double* t = tir + 5 * i;
  Q w = {t[0], t[1] * t[2], t[1] * t[3], t[1] * t[4]};
  return qmul(w, qdag(vn));
}

// ------------------------------------------------------------------------------------------
// Mode ref_rng : préparation (agrafes -> Vn, alpha) puis fin (tirages de l'hôte -> liens)
// ------------------------------------------------------------------------------------------
extern "C" __global__ void k_cond_paires_prep(const Q* U, int Lg, int mu, const u32* p1, const u32* p1pm, const Q* Vmu,
                                              double beta, Q* Vn, double* alpha, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Geo g = mkgeo(Lg);
  Q a = cond_agrafe_paire(U, g, mu, (long)p1[i], (long)p1pm[i], Vmu[i]);
  Q vn; double al;
  cond_preparer(a, beta, &vn, &al);
  Vn[i] = vn;
  alpha[i] = al;
}

extern "C" __global__ void k_cond_paires_fin(Q* U, int Lg, int mu, const u32* p1, const u32* p1pm, const Q* Vmu,
                                             const double* tir, const Q* Vn, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Geo g = mkgeo(Lg);
  Q u1 = cond_lien_depuis_tirage(tir, i, Vn[i]);
  Q u2 = qmul(qdag(u1), Vmu[i]);                    // U2 = U1^+ V
  U[(long)mu * g.V + (long)p1[i]] = u1;
  U[(long)mu * g.V + (long)p1pm[i]] = u2;
}

extern "C" __global__ void k_cond_libres_prep(const Q* U, int Lg, int mu, const u32* sites, double beta, Q* Vn,
                                              double* alpha, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Geo g = mkgeo(Lg);
  Q a = cond_agrafe_libre(U, g, mu, (long)sites[i]);
  Q vn; double al;
  cond_preparer(a, beta, &vn, &al);
  Vn[i] = vn;
  alpha[i] = al;
}

extern "C" __global__ void k_cond_libres_fin(Q* U, int Lg, int mu, const u32* sites, const double* tir, const Q* Vn,
                                             long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Geo g = mkgeo(Lg);
  U[(long)mu * g.V + (long)sites[i]] = cond_lien_depuis_tirage(tir, i, Vn[i]);
}

// ------------------------------------------------------------------------------------------
// Mode rapide : tirage de x0 ~ sqrt(1 - x0^2) exp(alpha x0) sur [-1, 1], sur le GPU
//   alpha >= 2     : Kennedy–Pendleton (2 appels Philox par essai, comme K1) ;
//   1e-12 <= alpha < 2 : proposition exponentielle de cb (1 + log1p(-u (-expm1(-2 alpha)))/alpha) + rejet ;
//   alpha < 1e-12  : proposition uniforme (limite exacte à O(alpha) ; beta = 0 admis).
// ------------------------------------------------------------------------------------------
__device__ inline U4 cond_ph(u32 li, u32 sweep_id, u32& draw, u32 k0, u32 k1) {
  return philox4x32_10(U4{{li, sweep_id, draw++, TAG_COND}}, k0, k1);
}

__device__ double cond_x0(double alpha, u32 li, u32 sweep_id, u32& draw, u32 k0, u32 k1, u32& ntry, bool& ok) {
  ntry = 0;
  ok = false;
  if (alpha >= 2.0) {
    while (ntry < COND_PLAFOND) {
      U4 r1 = cond_ph(li, sweep_id, draw, k0, k1);
      U4 r2 = cond_ph(li, sweep_id, draw, k0, k1);
      double a1 = u01(r1.v[0], r1.v[1]), a2 = u01(r1.v[2], r1.v[3]);
      double a3 = u01(r2.v[0], r2.v[1]), a4 = u01(r2.v[2], r2.v[3]);
      double c = cos(2 * M_PI * a2);
      double lam2 = -(log(a1) + c * c * log(a3)) / (2.0 * alpha);
      ++ntry;
      if (a4 * a4 <= 1.0 - lam2) { ok = true; return 1.0 - 2.0 * lam2; }
    }
  } else {
    double em = -expm1(-2.0 * alpha);
    while (ntry < COND_PLAFOND) {
      U4 r1 = cond_ph(li, sweep_id, draw, k0, k1);
      double a1 = u01(r1.v[0], r1.v[1]), a2 = u01(r1.v[2], r1.v[3]);
      double x = (alpha < 1e-12) ? 2.0 * a1 - 1.0 : 1.0 + log1p(-a1 * em) / alpha;
      ++ntry;
      if (a2 < sqrt(fmax(1.0 - x * x, 0.0))) { ok = true; return x; }
    }
  }
  return 1.0;
}

// Direction uniforme par (cos theta, phi) (un appel Philox) : W = (x0, r sin t cos p, r sin t sin p, r cos t)
__device__ inline Q cond_w(double x0, u32 li, u32 sweep_id, u32& draw, u32 k0, u32 k1) {
  U4 rd = cond_ph(li, sweep_id, draw, k0, k1);
  double ct = 2.0 * u01(rd.v[0], rd.v[1]) - 1.0;
  double ph = 2 * M_PI * u01(rd.v[2], rd.v[3]);
  double st = sqrt(fmax(0.0, 1.0 - ct * ct));
  double rr = sqrt(fmax(0.0, 1.0 - x0 * x0));
  Q w = {x0, rr * st * cos(ph), rr * st * sin(ph), rr * ct};
  return w;
}

// Nouveau lien depuis l'agrafe effective a ; ok = faux si le plafond d'essais est atteint.
__device__ inline Q cond_lien_rapide(Q a, double beta, u32 li, u32 sweep_id, u32 k0, u32 k1, u32& ntry, bool& ok) {
  Q vn; double alpha;
  cond_preparer(a, beta, &vn, &alpha);
  u32 draw = 0;
  double x0 = cond_x0(alpha, li, sweep_id, draw, k0, k1, ntry, ok);
  if (!ok) { Q id = {1, 0, 0, 0}; return id; }
  Q w = cond_w(x0, li, sweep_id, draw, k0, k1);
  return qmul(w, qdag(vn));
}

extern "C" __global__ void k_cond_paires_rapide(Q* U, int Lg, int mu, const u32* p1, const u32* p1pm, const Q* Vmu,
                                                double beta, u32 k0, u32 k1, u32 sweep_id, u32* diag, u64* ncap,
                                                long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Geo g = mkgeo(Lg);
  long s1 = (long)p1[i], s2 = (long)p1pm[i];
  Q v = Vmu[i];
  Q a = cond_agrafe_paire(U, g, mu, s1, s2, v);
  u32 ntry; bool ok;
  long li = (long)mu * g.V + s1;
  Q u1 = cond_lien_rapide(a, beta, (u32)li, sweep_id, k0, k1, ntry, ok);
  if (diag) diag[li] = ok ? ntry : 0u;
  if (!ok) { atomicAdd(ncap, 1ull); return; }       // lien inchangé, compté (attendu : 0)
  U[li] = u1;
  U[(long)mu * g.V + s2] = qmul(qdag(u1), v);       // U2 = U1^+ V
}

extern "C" __global__ void k_cond_libres_rapide(Q* U, int Lg, int mu, const u32* sites, double beta, u32 k0, u32 k1,
                                                u32 sweep_id, u32* diag, u64* ncap, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Geo g = mkgeo(Lg);
  long s = (long)sites[i];
  Q a = cond_agrafe_libre(U, g, mu, s);
  u32 ntry; bool ok;
  long li = (long)mu * g.V + s;
  Q q = cond_lien_rapide(a, beta, (u32)li, sweep_id, k0, k1, ntry, ok);
  if (diag) diag[li] = ok ? ntry : 0u;
  if (!ok) { atomicAdd(ncap, 1ull); return; }
  U[li] = q;
}

// ------------------------------------------------------------------------------------------
// Contrôles : out[0] = max |V(U) - Vfix| (par paire, réduit ensuite sur l'hôte), out[1] = |q|^2 - 1 maximal.
// Une valeur par paire / par lien ; la réduction (max) est faite à l'hôte pour rester déterministe.
// ------------------------------------------------------------------------------------------
extern "C" __global__ void k_cond_vdev(const Q* U, int Lg, int mu, const u32* p1, const u32* p1pm, const Q* Vmu,
                                       double* out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Geo g = mkgeo(Lg);
  Q a = U[(long)mu * g.V + (long)p1[i]];
  Q b = U[(long)mu * g.V + (long)p1pm[i]];
  Q v = qmul(a, b);
  Q f = Vmu[i];
  out[i] = fmax(fmax(fabs(v.x - f.x), fabs(v.y - f.y)), fmax(fabs(v.z - f.z), fabs(v.w - f.w)));
}

extern "C" __global__ void k_cond_unite(const Q* U, long n, double* out) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Q q = U[i];
  out[i] = fabs(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w - 1.0);
}

// ------------------------------------------------------------------------------------------
// Test de la loi du tirage rapide (gate EQC-2, loi de x0) : n tirages indépendants pour UN alpha.
// Compteur (i, 0x7E57, tirage, TAG_COND) ; out[i] = x0. Les tirages au plafond sont comptés dans ncap.
// ------------------------------------------------------------------------------------------
extern "C" __global__ void k_cond_test_x0(double alpha, long n, u32 k0, u32 k1, double* out, u64* ncap) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u32 draw = 0, ntry; bool ok;
  double x0 = cond_x0(alpha, (u32)i, 0x7E57u, draw, k0, k1, ntry, ok);
  if (!ok) atomicAdd(ncap, 1ull);
  out[i] = x0;
}
