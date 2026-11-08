// su2_bord.cu — bord gelé R2-3 : pose du bord, jauge de l'hyperplan, plaquettes par tranche (propriétaire : T3).
//
// Port de reference/su2_bord/su2_bord.cu (sha256 0a01a43e…, binaire 5af3ea49 de la production S2) : mêmes
// expressions, mêmes étiquettes Philox, mêmes emplacements de compteur. À assembler APRÈS su2_commun.h
// (qui fournit Philox, Q, Geo, coords, nb_up, ld…) et su2_ref64.cu.
//
// Différences STRUCTURELLES seulement :
//   - la Geo n'est plus passée par valeur (avec son champ `freeze`) : on passe L, Geo est reconstruite par
//     mkgeo ; le gel est l'argument `gel` de k_heatbath / k_overrelax (su2_ref64.cu) ;
//   - k_plaq_st et k_tranche sont des LOTS (blockIdx.y = chaîne) ; k_set_bord, k_gen_jauge, k_app_jauge
//     travaillent sur UNE chaîne par lancement (le pointeur est décalé de c·4V par l'appelant), car chaque
//     chaîne a son propre bord et sa propre jauge.
//
// Étiquettes et emplacements (identiques à su2_bord.cu, NE PAS CHANGER : bit-identité) :
//   TAG_BORD  = "BORD", slot 0xFFFFFFFE : liens du bord « chaud » ;
//   TAG_JAUGE = "JAUG", slot 0xFFFFFFFD : transformation de jauge de l'hyperplan x0 = 0.
//
// NOMS extern "C" (utilisés par src/su2/ref64.rs) :
//   k_set_bord   (Q* U, int Lg, int mode, u32 k0, u32 k1, const Q* src)   grille (ceil(3L³/256), 1)
//                mode 0 = froid, 1 = chaud (clé k0,k1), 2 = copie de src (même indice de lien)
//   k_gen_jauge  (Q* G, int Lg, u32 k0, u32 k1)                           grille (ceil(L³/256), 1)
//   k_app_jauge  (Q* U, int Lg, const Q* G)                               grille (ceil(4V/256), 1)
//   k_plaq_st    (const Q* U, int Lg, double* outS, double* outT)         grille (ceil(V/256), nch)
//   k_tranche    (const double* in, int Lg, double scale, double* out, long out_pas)
//                grille (L, nch), bloc 256 : un bloc par tranche t ; chaîne c : in += c·V, out += c·out_pas

enum : u32 { TAG_BORD = 0x424F5244u, TAG_JAUGE = 0x4A415547u };

// ------------------------------------------------------------------------------------------
// Ajouts R2-3 : bord gelé, jauge du bord, mesures par tranche
// ------------------------------------------------------------------------------------------
__device__ inline Q haar_q(u32 i, u32 slot, u32 tag, u32 k0, u32 k1) {
  U4 r1 = philox4x32_10(U4{{i, slot, 0u, tag}}, k0, k1), r2 = philox4x32_10(U4{{i, slot, 1u, tag}}, k0, k1);
  double u1 = u01(r1.v[0], r1.v[1]), u2 = u01(r1.v[2], r1.v[3]);
  double u3 = u01(r2.v[0], r2.v[1]), u4 = u01(r2.v[2], r2.v[3]);
  double ra = sqrt(-2.0 * log(u1)), rb = sqrt(-2.0 * log(u3));
  Q q = {ra * cos(2 * M_PI * u2), ra * sin(2 * M_PI * u2), rb * cos(2 * M_PI * u4), rb * sin(2 * M_PI * u4)};
  return qscale(q, 1.0 / qnorm(q));
}

// Pose du bord : liens U_mu(x), mu = 1..3, x0 = 0 (sites s in [0, L^3) car x0 est la coordonnée la plus lente).
// mode 0 = cold (identité), 1 = hot (Haar, clé k0,k1, étiquette TAG_BORD), 2 = copie depuis src (même indice).
extern "C" __global__ void k_set_bord(Q* U, int Lg, int mode, u32 k0, u32 k1, const Q* __restrict__ src) {
  Geo g = mkgeo(Lg);
  long L3 = g.V / g.L;
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * L3) return;
  int mu = 1 + (int)(i / L3); long s = i % L3;
  long li = (long)mu * g.V + s;
  if (mode == 0) { Q e = {1, 0, 0, 0}; U[li] = e; }
  else if (mode == 1) U[li] = haar_q((u32)li, 0xFFFFFFFEu, TAG_BORD, k0, k1);
  else U[li] = src[li];
}

// Transformation de jauge aléatoire (Haar) sur les sites de l'hyperplan x0 = 0.
extern "C" __global__ void k_gen_jauge(Q* G, int Lg, u32 k0, u32 k1) {
  Geo g = mkgeo(Lg);
  long L3 = g.V / g.L;
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L3) return;
  G[i] = haar_q((u32)i, 0xFFFFFFFDu, TAG_JAUGE, k0, k1);
}

// U_mu(x) -> g(x) U_mu(x) g(x+mu)^+ avec g = G sur x0 = 0, g = 1 ailleurs. Chaque lien est traité par
// un seul thread et ne lit que G : pas de course.
extern "C" __global__ void k_app_jauge(Q* U, int Lg, const Q* __restrict__ G) {
  Geo g = mkgeo(Lg);
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 4 * g.V) return;
  int mu = (int)(i / g.V); long s = i % g.V;
  int x[4]; coords(g, s, x);
  long L3 = g.V / g.L;
  Q u = U[i];
  if (x[0] == 0) u = qmul(G[s], u);                       // extrémité de départ dans l'hyperplan
  long sm = nb_up(g, s, mu, x[mu]);
  int xm0 = (mu == 0) ? ((x[0] + 1) % g.L) : x[0];
  if (xm0 == 0) u = qmul(u, qdag(G[sm % L3]));            // extrémité d'arrivée dans l'hyperplan
  U[i] = qscale(u, 1.0 / qnorm(u));
}

// Par site : somme des 3 plaquettes spatiales (plans 12, 13, 23) et des 3 temporelles (01, 02, 03), 1/2 ReTr.
// Lot : chaîne c = blockIdx.y ; outS et outT ont V valeurs par chaîne.
extern "C" __global__ void k_plaq_st(const Q* __restrict__ U, int Lg, double* outS, double* outT) {
  Geo g = mkgeo(Lg);
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= g.V) return;
  U += (long)blockIdx.y * 4 * g.V;
  outS += (long)blockIdx.y * g.V;
  outT += (long)blockIdx.y * g.V;
  int x[4]; coords(g, s, x);
  double aS = 0, aT = 0;
  for (int mu = 0; mu < 4; ++mu) {
    long smu = nb_up(g, s, mu, x[mu]);
    for (int nu = mu + 1; nu < 4; ++nu) {
      long snu = nb_up(g, s, nu, x[nu]);
      Q p = qmul(qmul(ld(U, g, mu, s), ld(U, g, nu, smu)), qmul(qdag(ld(U, g, mu, snu)), qdag(ld(U, g, nu, s))));
      if (mu == 0) aT += p.x; else aS += p.x;
    }
  }
  outS[s] = aS; outT[s] = aT;
}

// Un bloc par tranche t (sites contigus [t L^3, (t+1) L^3)) ; somme déterministe ; out[t] = somme * scale.
// Lot : chaîne c = blockIdx.y ; in += c·V ; out += c·out_pas.
extern "C" __global__ void k_tranche(const double* __restrict__ in, int Lg, double scale, double* out,
                                     long out_pas) {
  Geo g = mkgeo(Lg);
  __shared__ double sh[256];
  in += (long)blockIdx.y * g.V;
  out += (long)blockIdx.y * out_pas;
  long L3 = g.V / g.L; long base = (long)blockIdx.x * L3;
  double a = 0;
  for (long i = threadIdx.x; i < L3; i += 256) a += in[base + i];
  sh[threadIdx.x] = a;
  __syncthreads();
  for (int w = 128; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = sh[0] * scale;
}
