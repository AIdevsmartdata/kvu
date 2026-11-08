// su2_ref64.cu — chemin FP64 de RÉFÉRENCE du générateur SU(2) unifié (propriétaire : T3).
//
// Assemblage : prélude du backend, puis su2_commun.h, su2_ref64.cu, su2_bord.cu (dans cet ordre ;
// aucun #include entre fichiers). Le texte est compilé tel quel par hiprtc (gfx1151) et par NVRTC (sm_120).
//
// ARITHMÉTIQUE INCHANGÉE : les expressions flottantes (agrafe, Kennedy-Pendleton, Creutz, direction
// (cos θ, φ), overrelaxation, plaquette, Polyakov, arbre de réduction) sont celles de
// src/hip/su2_sweep.hip (lot K1) = su2_hip.cpp, dans le MÊME ordre d'évaluation. Seule la STRUCTURE change :
//   (1) LOTS. blockIdx.y = indice de chaîne c. Le champ de la chaîne c commence à U + c·4V ; les tableaux
//       par chaîne betas[], k0s[], k1s[], hot[] donnent β, la clé Philox (graine basse, haute) et le départ ;
//       n_cap[] et n_tries[] sont des compteurs PAR CHAÎNE. Le compteur Philox utilise l'indice de lien
//       LOCAL li = mu·V + s : une chaîne d'un lot donne donc les mêmes octets que la même chaîne seule
//       (invariant LOT-1).
//   (2) GEL (R2-3). `int gel` : si non nul, les liens SPATIAUX U_mu(x), mu ∈ {1,2,3}, de l'hyperplan x0 = 0
//       ne sont ni mis à jour par le bain thermique ni par l'overrelaxation. Le test est placé EXACTEMENT
//       comme dans su2_bord.cu : juste après site_of, avant tout calcul.
//   (3) diag_ntry : pointeur de diagnostic facultatif (nul en production), 4V entrées par chaîne.
//   (4) K1_MUTANT_METRO (jamais en production) : contrôle négatif, Metropolis naïf sur x0.
//
// NOMS extern "C" (documentés ici, utilisés par src/su2/ref64.rs) :
//   k_init       (Q* U, int Lg, const int* hot, const u32* k0s, const u32* k1s)
//                grille (ceil(4V/256), nch)
//   k_test_x0    (double alpha, long n, u32 k0, u32 k1, double* out, unsigned long long* n_cap)
//                (pas de lot : n tirages indépendants, étiquette de sweep 0x7E57, comme K1)
//   k_heatbath   (Q* U, int Lg, int mu, int parity, const double* betas, const u32* k0s, const u32* k1s,
//                 u32 sweep_id, unsigned long long* n_cap, unsigned long long* n_tries, u32* diag_ntry, int gel,
//                 const unsigned char* masque)
//                grille (ceil(V/2/256), nch)
//   k_overrelax  (Q* U, int Lg, int mu, int parity, int gel, const unsigned char* masque)
//                grille (ceil(V/2/256), nch)
//   masque : octet par lien (indice local li = mu·V + s, partagé par toutes les chaînes du lot) ; un octet non nul
//            gèle le lien (ni bain thermique ni overrelaxation) ; pointeur nul = pas de masque (comportement antérieur,
//            mêmes octets de sortie). Le test est placé juste après le calcul de li, avant tout calcul flottant.
//   k_plaq_site  (const Q* U, int Lg, double* out)                          grille (ceil(V/256), nch)
//   k_reduce     (const double* in, long n, double* out)                    grille (ceil(V/256), nch), n = V
//   k_final      (const double* in, long n, long in_pas, double scale, double* out, long out_pas)
//                grille (1, nch), bloc 256 : in += c·in_pas, out += c·out_pas
//   k_polyakov   (const Q* U, int Lg, double* out)                          grille (ceil(4L³/256), nch)

extern "C" __global__ void k_init(Q* U, int Lg, const int* hot, const u32* k0s, const u32* k1s) {
  Geo g = mkgeo(Lg);
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 4 * g.V) return;
  const int c = blockIdx.y;
  U += (long)c * 4 * g.V;                    // champ de la chaîne c
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

// Tirage exact de x0 selon sqrt(1-x0^2) exp(alpha x0) sur [-1,1] :
// Kennedy-Pendleton si alpha >= 2, Creutz sinon. draw = compteur de tirages Philox (incrémenté).
__device__ double sample_x0(double alpha, u32 li, u32 sweep_id, u32& draw, u32 k0, u32 k1, u32& ntry, bool& ok) {
  const u32 CAP = 1000000u;
  double x0 = 1.0; ok = false; ntry = 0;
  if (alpha >= 2.0) {
    while (ntry < CAP) {
      U4 r1 = philox4x32_10(U4{{li, sweep_id, draw++, TAG_HB}}, k0, k1);
      U4 r2 = philox4x32_10(U4{{li, sweep_id, draw++, TAG_HB}}, k0, k1);
      double a1 = u01(r1.v[0], r1.v[1]), a2 = u01(r1.v[2], r1.v[3]);
      double a3 = u01(r2.v[0], r2.v[1]), a4 = u01(r2.v[2], r2.v[3]);
      double c = cos(2 * M_PI * a2);
      double lam2 = -(log(a1) + c * c * log(a3)) / (2.0 * alpha);
      ++ntry;
      if (a4 * a4 <= 1.0 - lam2) { x0 = 1.0 - 2.0 * lam2; ok = true; break; }
    }
  } else {
    double e2 = exp(-2.0 * alpha);
    while (ntry < CAP) {
      U4 r1 = philox4x32_10(U4{{li, sweep_id, draw++, TAG_HB}}, k0, k1);
      double a1 = u01(r1.v[0], r1.v[1]), a2 = u01(r1.v[2], r1.v[3]);
      double y = 1.0 + log(e2 + a1 * (1.0 - e2)) / alpha;
      ++ntry;
      if (a2 * a2 <= 1.0 - y * y) { x0 = y; ok = true; break; }
    }
  }
  return x0;
}

#ifdef K1_MUTANT_METRO
// MUTANT (contrôle négatif, jamais en production) : UN pas de Metropolis naïf sur x0 depuis x_cur,
// proposition uniforme sur [-1,1], rapport sqrt(1-y^2) exp(alpha y) / idem(x_cur).
// Chaîne de Markov encore valide (bilan détaillé), mais le tirage n'est PAS exact.
__device__ double sample_x0_mut(double alpha, double x_cur, u32 li, u32 sweep_id, u32& draw, u32 k0, u32 k1,
                                u32& ntry, bool& ok) {
  U4 r1 = philox4x32_10(U4{{li, sweep_id, draw++, TAG_HB}}, k0, k1);
  double y = 2.0 * u01(r1.v[0], r1.v[1]) - 1.0;
  double pc = sqrt(fmax(0.0, 1.0 - x_cur * x_cur));
  double py = sqrt(fmax(0.0, 1.0 - y * y)) * exp(alpha * (y - x_cur));
  ntry = 1; ok = true;
  return (u01(r1.v[2], r1.v[3]) * pc <= py) ? y : x_cur;
}
#endif

// Test unitaire : n tirages indépendants de x0 à alpha fixé (même fonction que le bain thermique).
extern "C" __global__ void k_test_x0(double alpha, long n, u32 k0, u32 k1, double* out, unsigned long long* n_cap) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u32 draw = 0, ntry = 0; bool ok = false;
#ifdef K1_MUTANT_METRO
  double x0 = sample_x0_mut(alpha, 1.0, (u32)i, 0x7E57u, draw, k0, k1, ntry, ok);
#else
  double x0 = sample_x0(alpha, (u32)i, 0x7E57u, draw, k0, k1, ntry, ok);
#endif
  if (!ok) atomicAdd(n_cap, 1ull);
  out[i] = x0;
}

extern "C" __global__ void k_heatbath(Q* U, int Lg, int mu, int parity, const double* betas, const u32* k0s,
                                      const u32* k1s, u32 sweep_id, unsigned long long* n_cap,
                                      unsigned long long* n_tries, u32* diag_ntry, int gel,
                                      const unsigned char* __restrict__ masque) {
  Geo g = mkgeo(Lg);
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= g.V / 2) return;
  const int c = blockIdx.y;                  // chaîne du lot
  U += (long)c * 4 * g.V;
  if (diag_ntry) diag_ntry += (long)c * 4 * g.V;
  const double beta = betas[c];
  const u32 k0 = k0s[c], k1 = k1s[c];
  int x[4];
  long s = site_of(g, t, parity, x);
  if (gel && mu != 0 && x[0] == 0) return;   // lien de bord gelé
  long li = (long)mu * g.V + s;              // indice de lien LOCAL (compteur Philox identique à la chaîne seule)
  if (masque && masque[li]) return;          // lien gelé par le masque (boîte à bord fixé)
  Q A = staple(U, g, mu, s, x);
  double k = qnorm(A);
  u32 draw = 0;
  Q W;
  if (!(k > 1e-300)) {   // agrafe nulle : mesure de Haar (jamais observé en pratique)
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
#ifdef K1_MUTANT_METRO
  double x0 = sample_x0_mut(alpha, qmul(U[li], Vv).x, (u32)li, sweep_id, draw, k0, k1, ntry, ok);
#else
  double x0 = sample_x0(alpha, (u32)li, sweep_id, draw, k0, k1, ntry, ok);
#endif
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

extern "C" __global__ void k_overrelax(Q* U, int Lg, int mu, int parity, int gel,
                                       const unsigned char* __restrict__ masque) {
  Geo g = mkgeo(Lg);
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= g.V / 2) return;
  U += (long)blockIdx.y * 4 * g.V;           // champ de la chaîne
  int x[4];
  long s = site_of(g, t, parity, x);
  if (gel && mu != 0 && x[0] == 0) return;   // lien de bord gelé
  long li = (long)mu * g.V + s;
  if (masque && masque[li]) return;          // lien gelé par le masque (boîte à bord fixé)
  Q A = staple(U, g, mu, s, x);
  double k = qnorm(A);
  if (!(k > 1e-300)) return;
  Q Vd = qdag(qscale(A, 1.0 / k));
  Q Un = qmul(qmul(Vd, qdag(U[li])), Vd);   // U' = V^+ U^+ V^+  (ReTr(U'A) = ReTr(UA))
  U[li] = qscale(Un, 1.0 / qnorm(Un));
}

// somme par site des 6 plaquettes 1/2 ReTr U_p ; out : V valeurs par chaîne
extern "C" __global__ void k_plaq_site(const Q* __restrict__ U, int Lg, double* out) {
  Geo g = mkgeo(Lg);
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= g.V) return;
  U += (long)blockIdx.y * 4 * g.V;
  out += (long)blockIdx.y * g.V;
  int x[4]; coords(g, s, x);
  double acc = 0;
  for (int mu = 0; mu < 4; ++mu) {
    long smu = nb_up(g, s, mu, x[mu]);
    for (int nu = mu + 1; nu < 4; ++nu) {
      long snu = nb_up(g, s, nu, x[nu]);
      Q p = qmul(qmul(ld(U, g, mu, s), ld(U, g, nu, smu)), qmul(qdag(ld(U, g, mu, snu)), qdag(ld(U, g, nu, s))));
      acc += p.x;
    }
  }
  out[s] = acc;
}

// réduction déterministe par bloc (256 threads) ; chaque chaîne a n entrées et gridDim.x sorties
extern "C" __global__ void k_reduce(const double* __restrict__ in, long n, double* out) {
  __shared__ double sh[256];
  in += (long)blockIdx.y * n;
  out += (long)blockIdx.y * gridDim.x;
  long i = (long)blockIdx.x * 256 + threadIdx.x;
  sh[threadIdx.x] = (i < n) ? in[i] : 0.0;
  __syncthreads();
  for (int w = 128; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = sh[0];
}

// Somme finale déterministe (1 bloc de 256 par chaîne) : chaque thread somme in[t], in[t+256], ... dans
// l'ordre, puis arbre fixe ; écrit sum*scale dans *out. Chaîne c : in += c·in_pas, out += c·out_pas.
extern "C" __global__ void k_final(const double* __restrict__ in, long n, long in_pas, double scale, double* out,
                                   long out_pas) {
  __shared__ double sh[256];
  in += (long)blockIdx.y * in_pas;
  out += (long)blockIdx.y * out_pas;
  double a = 0;
  for (long i = threadIdx.x; i < n; i += 256) a += in[i];
  sh[threadIdx.x] = a;
  __syncthreads();
  for (int w = 128; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = sh[0] * scale;
}

// Boucles de Polyakov 1/2 Tr prod_{k} U_mu(x + k e_mu), pour chaque mu et chaque site transverse ;
// out : 4 L³ valeurs par chaîne
extern "C" __global__ void k_polyakov(const Q* __restrict__ U, int Lg, double* out) {
  Geo g = mkgeo(Lg);
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long L3 = g.V / g.L;
  if (i >= 4 * L3) return;
  U += (long)blockIdx.y * 4 * g.V;
  out += (long)blockIdx.y * 4 * L3;
  int mu = (int)(i / L3);
  long j = i % L3;
  int x[4]; x[mu] = 0;
  int dirs[3], nd = 0;
  for (int d = 0; d < 4; ++d) if (d != mu) dirs[nd++] = d;
  x[dirs[2]] = (int)(j % g.L); long t = j / g.L;
  x[dirs[1]] = (int)(t % g.L); x[dirs[0]] = (int)(t / g.L);
  long s = ((long)(x[0] * g.L + x[1]) * g.L + x[2]) * g.L + x[3];
  Q P = {1, 0, 0, 0};
  for (int k = 0; k < g.L; ++k) { P = qmul(P, ld(U, g, mu, s)); s += (k == g.L - 1) ? 0 : g.st[mu]; }
  out[i] = P.x;
}
