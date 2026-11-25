// su2_feuille.cu : noyaux de la campagne « une boîte » (branche campagne/une-boite), assemblés APRÈS su2_geo.cu dans le module
// « su2_boite_une » = su2_commun.h + su2_ref64.cu + su2_geo.cu + ce fichier. Aucun fichier existant n'est modifié.
//
// k_heatbath_boite : copie exacte de k_heatbath_geo, à UNE différence près : un β nul rend le lien de Haar (le bain thermique
//   historique est singulier à β = 0 ; ici le lien est tiré de la mesure de Haar avec les mêmes compteurs Philox que la branche
//   « agrafe nulle »). À β > 0 le texte exécuté est celui de k_heatbath_geo : mêmes octets (porte de la campagne).
// k_plaq_site_feuille : par site, S_tot(s) = Σ_{μ<ν} ½ReTr U_p(s) (somme NON signée, brute) et S_feuille(s) = même somme restreinte
//   aux plaquettes marquées dans `feuille` (un octet par (plan, coin) : feuille[p·V + s] ≠ 0). S_reste = S_tot − S_feuille.
//   Les deux tableaux sont réduits ensuite par k_reduce / k_final (arbre fixe, accumulateurs F64).
extern "C" __global__ void k_heatbath_boite(Q* U, const long long* geo, int mu, int parity, const double* betas,
                                            const u32* k0s, const u32* k1s, u32 sweep_id, unsigned long long* n_cap,
                                            unsigned long long* n_tries, u32* diag_ntry, const double* __restrict__ coef,
                                            const unsigned char* __restrict__ masque) {
  GeoG g = g_lire(geo);
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= g_nsel(geo, parity, g.V)) return;
  const int c = blockIdx.y;
  U += (long)c * 4 * g.V;
  if (diag_ntry) diag_ntry += (long)c * 4 * g.V;
  const double beta = betas[c];
  const u32 k0 = k0s[c], k1 = k1s[c];
  int x[4];
  long s = g_site_sel(g, geo, t, parity, x);
  long li = (long)mu * g.V + s;
  if (masque && masque[li]) return;
  Q A = g_agrafe(U, g, mu, s, x, coef);
  double k = qnorm(A);
  u32 draw = 0;
  Q W;
  if (!(k > 1e-300) || beta == 0.0) {   // agrafe nulle OU beta nul : mesure de Haar
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
  double alpha = beta * k;
  u32 ntry = 0; bool ok = false;
  double x0 = sample_x0(alpha, (u32)li, sweep_id, draw, k0, k1, ntry, ok);
  atomicAdd(n_tries + c, (unsigned long long)ntry);
  if (diag_ntry) diag_ntry[li] = ok ? ntry : 0u;
  if (!ok) { atomicAdd(n_cap + c, 1ull); return; }
  U4 rd = philox4x32_10(U4{{(u32)li, sweep_id, draw++, TAG_HB}}, k0, k1);
  double ct = 2.0 * u01(rd.v[0], rd.v[1]) - 1.0;
  double ph = 2 * M_PI * u01(rd.v[2], rd.v[3]);
  double st = sqrt(fmax(0.0, 1.0 - ct * ct));
  double rr = sqrt(fmax(0.0, 1.0 - x0 * x0));
  W.x = x0; W.y = rr * st * cos(ph); W.z = rr * st * sin(ph); W.w = rr * ct;
  Q Un = qmul(W, qdag(Vv));
  U[li] = qscale(Un, 1.0 / qnorm(Un));
}

extern "C" __global__ void k_plaq_site_feuille(const Q* __restrict__ U, const long long* geo,
                                               const unsigned char* __restrict__ feuille, double* out_tot,
                                               double* out_fe) {
  GeoG g = g_lire(geo);
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= g.V) return;
  U += (long)blockIdx.y * 4 * g.V;
  out_tot += (long)blockIdx.y * g.V;
  out_fe += (long)blockIdx.y * g.V;
  int x[4]; g_coords(g, s, x);
  double tot = 0, fe = 0;
  for (int mu = 0; mu < 4; ++mu) {
    long smu = g_up(g, s, mu, x[mu]);
    for (int nu = mu + 1; nu < 4; ++nu) {
      long snu = g_up(g, s, nu, x[nu]);
      Q p = qmul(qmul(g_ld(U, g, mu, s), g_ld(U, g, nu, smu)), qmul(qdag(g_ld(U, g, mu, snu)), qdag(g_ld(U, g, nu, s))));
      tot += p.x;
      if (feuille && feuille[(long)g_plan(mu, nu) * g.V + s]) fe += p.x;
    }
  }
  out_tot[s] = tot;
  out_fe[s] = fe;
}
