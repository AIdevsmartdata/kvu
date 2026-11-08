// cond3.cu — facteur 3 : fibre dure de DEC (T3-D). Assemblé APRÈS su2_commun.h et cond.cu (qui fournissent Philox, Q et son algèbre, Geo,
// staple, cond_lien_rapide) : ce fichier n'en redéfinit rien.
//
// Convention : staple() de su2_commun.h rend l'agrafe A = K† (K = agrafe du SPEC_T3 : Σ_{p∋ℓ} ½ Tr U_p = ½ Tr(U_ℓ K†)) ; cond_lien_rapide(A, …)
// tire U = X W avec W = K/|K| (poids exp(β ½ Tr(U K†))). Ligne ē = (U₁, U₂, U₃), V = U₁U₂U₃ figé :
//   U₁ : K_eff = K₁ + V K₃† U₂†  donc  A_eff = A₁ + U₂ A₃† V†     (A = K†)
//   U₂ : K_eff = K₂ + U₁† V K₃†  donc  A_eff = A₂ + A₃† V† U₁
//   U₃ = U₂† U₁† V asservi. Surrelaxation U → W U† W, W = K/|K| = A†/|A|.
// Même ordre d'évaluation que le code CPU (cond3/su2.rs) ; compilé sans contraction FMA (OptionsCompil::reference_k1).

typedef unsigned short u16;

__device__ inline Q t3_renorm(Q q) {
  double n = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  Q c = {q.x / n, q.y / n, q.z / n, q.w / n};
  return c;
}

__device__ inline Q t3_surrelaxer(Q a, Q u) {
  double n2 = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
  if (n2 < 1e-300) return u;
  double n = sqrt(n2);
  Q k = qdag(a);
  Q w = {k.x / n, k.y / n, k.z / n, k.w / n};
  return qmul(qmul(w, qdag(u)), w);
}

// R(U) y = partie vectorielle de U (0, y) U†
__device__ inline void t3_adj(Q u, const double y[3], double out[3]) {
  Q v = {0.0, y[0], y[1], y[2]};
  Q w = qmul(qmul(u, v), qdag(u));
  out[0] = w.y; out[1] = w.z; out[2] = w.w;
}

// ------------------------------------------------------------------------------------------
// Liens libres : bain thermique et surrelaxation (sites = liste de sites de la direction mu et d'une parité)
// ------------------------------------------------------------------------------------------
extern "C" __global__ void k_t3_libres_hb(Q* U, int Lg, int mu, const u32* sites, double beta, u32 k0, u32 k1, u32 sweep,
                                          u32* diag, u64* ncap, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Geo g = mkgeo(Lg);
  long s = (long)sites[i];
  int x[4];
  coords(g, s, x);
  Q a = staple(U, g, mu, s, x);
  u32 ntry; bool ok;
  long li = (long)mu * g.V + s;
  Q q = cond_lien_rapide(a, beta, (u32)li, sweep, k0, k1, ntry, ok);
  if (diag) diag[li] = ok ? ntry : 0u;
  if (!ok) { atomicAdd(ncap, 1ull); return; }
  U[li] = t3_renorm(q);
}

extern "C" __global__ void k_t3_libres_or(Q* U, int Lg, int mu, const u32* sites, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Geo g = mkgeo(Lg);
  long s = (long)sites[i];
  int x[4];
  coords(g, s, x);
  Q a = staple(U, g, mu, s, x);
  long li = (long)mu * g.V + s;
  U[li] = t3_renorm(t3_surrelaxer(a, U[li]));
}

// ------------------------------------------------------------------------------------------
// Lignes : U₁ (quelle = 1) ou U₂ (quelle = 2) à agrafe effective, U₃ asservi. hb = 1 : bain thermique ; 0 : surrelaxation.
// ls1/ls2/ls3 : sites des trois liens de chaque ligne de la direction mu ; Vl : V figé de ces lignes.
// ------------------------------------------------------------------------------------------
extern "C" __global__ void k_t3_lignes(Q* U, int Lg, int mu, const u32* ls1, const u32* ls2, const u32* ls3, const Q* Vl, int quelle,
                                       int hb, double beta, u32 k0, u32 k1, u32 sweep, u32* diag, u64* ncap, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Geo g = mkgeo(Lg);
  long s1 = (long)ls1[i], s2 = (long)ls2[i], s3 = (long)ls3[i];
  int x1[4], x2[4], x3[4];
  coords(g, s1, x1); coords(g, s2, x2); coords(g, s3, x3);
  Q a1 = staple(U, g, mu, s1, x1);
  Q a2 = staple(U, g, mu, s2, x2);
  Q a3 = staple(U, g, mu, s3, x3);
  Q u1 = U[(long)mu * g.V + s1], u2 = U[(long)mu * g.V + s2];
  Q v = Vl[i];
  Q aeff = (quelle == 1) ? qadd(a1, qmul(qmul(u2, qdag(a3)), qdag(v)))
                         : qadd(a2, qmul(qmul(qdag(a3), qdag(v)), u1));
  long cible = (quelle == 1) ? s1 : s2;
  long li = (long)mu * g.V + cible;
  Q un;
  if (hb) {
    u32 ntry; bool ok;
    un = cond_lien_rapide(aeff, beta, (u32)li, sweep, k0, k1, ntry, ok);
    if (diag) diag[li] = ok ? ntry : 0u;
    if (!ok) { atomicAdd(ncap, 1ull); un = U[li]; }
  } else {
    un = t3_surrelaxer(aeff, U[li]);
  }
  Q n1 = (quelle == 1) ? un : u1;
  Q n2 = (quelle == 1) ? u2 : un;
  Q u3 = qmul(qmul(qdag(n2), qdag(n1)), v);
  U[li] = t3_renorm(un);
  U[(long)mu * g.V + s3] = t3_renorm(u3);
}

// ------------------------------------------------------------------------------------------
// Mesure : observables des faces (obuf[t·nf + f] = ½ Tr U_p : disposition TRANSPOSÉE, écritures et lectures groupées) et scores moyens des
// sondes (sbuf[(s·3 + c)·lot + t])
// ------------------------------------------------------------------------------------------
extern "C" __global__ void k_t3_obs(const Q* U, int Lg, const u32* fmu, const u32* fnu, const u32* fsite, double* obuf, int t, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Geo g = mkgeo(Lg);
  long s = (long)fsite[i];
  int mu = (int)fmu[i], nu = (int)fnu[i];
  int x[4];
  coords(g, s, x);
  long smu = nb_up(g, s, mu, x[mu]);
  long snu = nb_up(g, s, nu, x[nu]);
  Q a = ld(U, g, mu, s), b = ld(U, g, nu, smu), c = ld(U, g, mu, snu), d = ld(U, g, nu, s);
  Q p = qmul(qmul(a, b), qmul(qdag(c), qdag(d)));
  obuf[(long)t * n + i] = p.x;
}

extern "C" __global__ void k_t3_scores(const Q* U, int Lg, const u32* pmu, const u32* ps1, const u32* ps2, const u32* ps3, double beta,
                                       double* sbuf, int lot, int t, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Geo g = mkgeo(Lg);
  int mu = (int)pmu[i];
  long s1 = (long)ps1[i], s2 = (long)ps2[i], s3 = (long)ps3[i];
  int x1[4], x2[4], x3[4];
  coords(g, s1, x1); coords(g, s2, x2); coords(g, s3, x3);
  Q a1 = staple(U, g, mu, s1, x1);
  Q a2 = staple(U, g, mu, s2, x2);
  Q a3 = staple(U, g, mu, s3, x3);
  Q u1 = U[(long)mu * g.V + s1], u2 = U[(long)mu * g.V + s2], u3 = U[(long)mu * g.V + s3];
  Q m1 = qmul(u1, a1);      // U₁ K₁†
  Q m2 = qmul(u2, a2);
  Q m3 = qmul(u3, a3);
  double y2[3] = {m2.y, m2.z, m2.w}, y3[3] = {m3.y, m3.z, m3.w};
  double r2[3], r3[3];
  t3_adj(u1, y2, r2);
  t3_adj(qmul(u1, u2), y3, r3);
  double s1v[3] = {beta * m1.y, beta * m1.z, beta * m1.w};
  for (int c = 0; c < 3; ++c) {
    double sc2 = beta * r2[c], sc3 = beta * r3[c];
    sbuf[(i * 3 + c) * (long)lot + t] = (s1v[c] + sc2 + sc3) / 3.0;
  }
}

// ------------------------------------------------------------------------------------------
// Accumulation par lot : sos[q·3 + c] += Σ_t obuf[t·nf + face(q)] · sbuf[(sonde(q)·3 + c)·lot + t]
// (4 sommes partielles par composante, puis (s0 + s1) + (s2 + s3), reste séquentiel : même ordre que le CPU).
// ------------------------------------------------------------------------------------------
extern "C" __global__ void k_t3_acc(double* sos, const double* obuf, const double* sbuf, const u32* pface, const u32* pprobe, int lot,
                                    int nb, long nf, long np) {
  long q = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= np) return;
  const double* ocol = obuf + (long)pface[q];
  long s = (long)pprobe[q];
  const double* s0 = sbuf + (s * 3 + 0) * (long)lot;
  const double* s1 = sbuf + (s * 3 + 1) * (long)lot;
  const double* s2 = sbuf + (s * 3 + 2) * (long)lot;
  // 4 sommes partielles par composante (comme le CPU) : indice t = 4 j + r
  double a[3][4] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
  int i = 0;
  for (; i + 4 <= nb; i += 4) {
    for (int r = 0; r < 4; ++r) {
      double o = ocol[(long)(i + r) * nf];
      a[0][r] += o * s0[i + r];
      a[1][r] += o * s1[i + r];
      a[2][r] += o * s2[i + r];
    }
  }
  double r0 = (a[0][0] + a[0][1]) + (a[0][2] + a[0][3]);
  double r1 = (a[1][0] + a[1][1]) + (a[1][2] + a[1][3]);
  double r2 = (a[2][0] + a[2][1]) + (a[2][2] + a[2][3]);
  for (; i < nb; ++i) {
    double o = ocol[(long)i * nf];
    r0 += o * s0[i]; r1 += o * s1[i]; r2 += o * s2[i];
  }
  sos[q * 3 + 0] += r0;
  sos[q * 3 + 1] += r1;
  sos[q * 3 + 2] += r2;
}

// Σ_t O par face et Σ_t σ par sonde et composante (ordre séquentiel en t, comme l'accumulation par itération du CPU).
extern "C" __global__ void k_t3_sommes_o(double* so, const double* obuf, int nb, long nf) {
  long f = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= nf) return;
  double a = so[f];
  for (int t = 0; t < nb; ++t) a += obuf[(long)t * nf + f];
  so[f] = a;
}

// Séries des plaquettes suivies : oser[k·lot + t] = obuf[t·nf + idx[k]]
extern "C" __global__ void k_t3_serie_o(double* oser, const double* obuf, const u32* idx, int lot, int nb, long nf, long nk) {
  long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nk * nb) return;
  long k = j / nb; int t = (int)(j % nb);
  oser[k * (long)lot + t] = obuf[(long)t * nf + (long)idx[k]];
}

extern "C" __global__ void k_t3_sommes_s(double* ss, const double* sbuf, int lot, int nb, long nk) {
  long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nk) return;
  double a = ss[k];
  for (int t = 0; t < nb; ++t) a += sbuf[k * (long)lot + t];
  ss[k] = a;
}
