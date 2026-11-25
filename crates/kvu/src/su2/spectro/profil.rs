//! PROFIL du tube de flux enroulé (« IRM du tube ») : accumulation hôte de
//! C(Δ⊥;τ) = Σ_{x⊥,t} ⟨Φ(x⊥,t) Φ(x⊥,t+τ) ρ(x⊥+Δ⊥, t+τ/2)⟩ / Σ⟨ΦΦ⟩ − ⟨ρ⟩, ρ = densité d'action (électrique k=0 / magnétique k=1)
//! sommée le long de la direction d'enroulement. τ impair : ρ moyenné sur les deux tranches du milieu.
//! Les cartes viennent de la carte graphique (noyaux `k_lignes*`, `k_rho`) ; ce module est pur (testable sur processeur).

/// Cartes d'UNE configuration d'UNE réplique : `phi` [t][a][b], `rho` [k][t][a][b].
#[derive(Clone, Debug)]
pub struct Cartes {
    pub phi: Vec<f64>,
    pub rho: Vec<f64>,
}

/// Sommes sur un paquet de configurations (le paquet est l'unité du jackknife).
#[derive(Clone, Debug)]
pub struct Accu {
    pub ta: usize,
    pub tb: usize,
    pub tt: usize,
    pub taus: Vec<usize>,
    pub ncfg: usize,
    /// Σ_cfg Σ_{x⊥,t} ρ_k
    pub r: [f64; 2],
    /// [itau] Σ_cfg Σ_{x,t} Φ(x,t)Φ(x,t+τ)
    pub d: Vec<f64>,
    /// [itau][k][da·tb+db] Σ_cfg Σ_{x,t} Φ(x,t)Φ(x,t+τ) ρ̄_k(x+Δ, t+τ/2)
    pub n: Vec<f64>,
}

impl Accu {
    /// Précondition (garantie par l'appelant : `TorelonGpu::nouveau` refuse `--profil` hors 1 ≤ τ ≤ tau_max, et tau_max < ℓ_temps) :
    /// `tt > 0` et `1 ≤ τ < tt` pour chaque `tau`. Ce constructeur public ne le vérifie pas ; `t + τ` ne déborde un `usize`
    /// que si `tt - 1 + τ` est représentable, ce que le domaine ci-dessus (valeurs de la taille d'un réseau) garantit.
    pub fn nouveau(ta: usize, tb: usize, tt: usize, taus: &[usize]) -> Accu {
        Accu { ta, tb, tt, taus: taus.to_vec(), ncfg: 0, r: [0.0; 2], d: vec![0.0; taus.len()], n: vec![0.0; taus.len() * 2 * ta * tb] }
    }
    pub fn raz(&mut self) {
        self.ncfg = 0;
        self.r = [0.0; 2];
        self.d.iter_mut().for_each(|x| *x = 0.0);
        self.n.iter_mut().for_each(|x| *x = 0.0);
    }
    pub fn longueur_record(&self) -> usize {
        3 + self.taus.len() * (1 + 2 * self.ta * self.tb)
    }
    /// [ncfg, R_e, R_m, puis par τ : D, N_e[ta·tb], N_m[ta·tb]]
    pub fn record(&self) -> Vec<f64> {
        let nab = self.ta * self.tb;
        let mut v = vec![self.ncfg as f64, self.r[0], self.r[1]];
        for it in 0..self.taus.len() {
            v.push(self.d[it]);
            v.extend_from_slice(&self.n[(it * 2) * nab..(it * 2 + 2) * nab]);
        }
        v
    }
    #[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
    pub fn ajouter(&mut self, c: &Cartes) {
        let (ta, tb, tt) = (self.ta, self.tb, self.tt);
        let nab = ta * tb;
        assert_eq!(c.phi.len(), tt * nab);
        assert_eq!(c.rho.len(), 2 * tt * nab);
        self.ncfg += 1;
        for k in 0..2 {
            self.r[k] += c.rho[k * tt * nab..(k + 1) * tt * nab].iter().sum::<f64>();
        }
        let mut s = vec![0f64; nab];
        let mut rb = vec![0f64; nab];
        for (it, &tau) in self.taus.iter().enumerate() {
            for t in 0..tt {
                let (p0, p1) = (&c.phi[t * nab..(t + 1) * nab], &c.phi[((t + tau) % tt) * nab..((t + tau) % tt + 1) * nab]);
                let mut dd = 0.0;
                for i in 0..nab {
                    s[i] = p0[i] * p1[i];
                    dd += s[i];
                }
                self.d[it] += dd;
                for k in 0..2 {
                    let base = k * tt * nab;
                    let (t1, t2) = ((t + tau / 2) % tt, (t + tau.div_ceil(2)) % tt);
                    for i in 0..nab {
                        rb[i] = 0.5 * (c.rho[base + t1 * nab + i] + c.rho[base + t2 * nab + i]);
                    }
                    let dest = &mut self.n[(it * 2 + k) * nab..(it * 2 + k + 1) * nab];
                    for a in 0..ta {
                        for b in 0..tb {
                            let i = a * tb + b;
                            let si = s[i];
                            for da in 0..ta {
                                let row = ((a + da) % ta) * tb;
                                let drow = da * tb;
                                for db in 0..tb {
                                    dest[drow + db] += si * rb[row + (b + db) % tb];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    // même convention que `sq_mul` du noyau
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + b[0] * a[1] - (a[2] * b[3] - a[3] * b[2]),
        a[0] * b[2] + b[0] * a[2] - (a[3] * b[1] - a[1] * b[3]),
        a[0] * b[3] + b[0] * a[3] - (a[1] * b[2] - a[2] * b[1]),
    ]
}

/// Référence processeur de ρ à partir du champ hôte `q[(mu·V + s)·4 + k]` (quaternions, x = ½Tr) : [k][t][a][b], mêmes conventions que `k_rho`.
pub fn rho_cpu(d: [usize; 4], q: &[f64], w: usize, temps: usize, tr: [usize; 2]) -> Vec<f64> {
    let v: usize = d.iter().product();
    let st = [d[1] * d[2] * d[3], d[2] * d[3], d[3], 1];
    let site = |x: &[usize; 4]| (0..4).map(|i| x[i] * st[i]).sum::<usize>();
    let mul = qmul;
    let dag = |a: [f64; 4]| [a[0], -a[1], -a[2], -a[3]];
    let lk = |mu: usize, x: &[usize; 4]| {
        let s = site(x);
        let o = (mu * v + s) * 4;
        [q[o], q[o + 1], q[o + 2], q[o + 3]]
    };
    let (ta, tb, tt) = (d[tr[0]], d[tr[1]], d[temps]);
    let mut rho = vec![0f64; 2 * tt * ta * tb];
    let dirs = [w, temps, tr[0], tr[1]];
    for x0 in 0..d[w] {
        for t in 0..tt {
            for a in 0..ta {
                for b in 0..tb {
                    let mut x = [0usize; 4];
                    x[w] = x0;
                    x[temps] = t;
                    x[tr[0]] = a;
                    x[tr[1]] = b;
                    // plaquette de base x, plan (mu,nu) ; répartie sur ses coins non-w
                    for i in 0..4 {
                        for kk in i + 1..4 {
                            let (mu, nu) = (dirs[i], dirs[kk]);
                            let kind = if mu == temps || nu == temps { 0 } else { 1 };
                            let mut xm = x;
                            xm[mu] = (x[mu] + 1) % d[mu];
                            let mut xn = x;
                            xn[nu] = (x[nu] + 1) % d[nu];
                            let p = mul(mul(lk(mu, &x), lk(nu, &xm)), mul(dag(lk(mu, &xn)), dag(lk(nu, &x))));
                            let e = 1.0 - p[0];
                            let sd: Vec<usize> = [mu, nu].iter().copied().filter(|&m| m != w).collect();
                            let wt = e / (1usize << sd.len()) as f64;
                            for comb in 0..(1usize << sd.len()) {
                                let mut y = x;
                                for (qq, &dd) in sd.iter().enumerate() {
                                    if (comb >> qq) & 1 == 1 {
                                        y[dd] = (y[dd] + 1) % d[dd];
                                    }
                                }
                                rho[(kind * tt + y[temps]) * ta * tb + y[tr[0]] * tb + y[tr[1]]] += wt;
                            }
                        }
                    }
                }
            }
        }
    }
    rho
}

/// Référence processeur de Φ(x⊥,t) = ½Tr ∏_k U_w(x + k ŵ) : [t][a][b].
pub fn phi_cpu(d: [usize; 4], q: &[f64], w: usize, temps: usize, tr: [usize; 2]) -> Vec<f64> {
    let v: usize = d.iter().product();
    let st = [d[1] * d[2] * d[3], d[2] * d[3], d[3], 1];
    let (ta, tb, tt) = (d[tr[0]], d[tr[1]], d[temps]);
    let mut out = vec![0f64; tt * ta * tb];
    for t in 0..tt {
        for a in 0..ta {
            for b in 0..tb {
                let mut x = [0usize; 4];
                x[temps] = t;
                x[tr[0]] = a;
                x[tr[1]] = b;
                let mut p = [1.0, 0.0, 0.0, 0.0];
                for k in 0..d[w] {
                    x[w] = k;
                    let s: usize = (0..4).map(|i| x[i] * st[i]).sum();
                    let o = (w * v + s) * 4;
                    let u = [q[o], q[o + 1], q[o + 2], q[o + 3]];
                    p = qmul(p, u);
                }
                out[(t * ta + a) * tb + b] = p[0];
            }
        }
    }
    out
}
