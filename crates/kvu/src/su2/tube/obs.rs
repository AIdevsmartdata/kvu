//! Observables du tube tordu (toutes sous torsion, sommes déterministes par blocs de sites).
//!
//! - plaquette moyenne par plan : (1/V) Σ_x z_p ½Tr U_p, plans dans l'ordre 01, 02, 03, 12, 13, 23 ;
//! - V = Σ_p (1 − z_p ½Tr U_p) (S = βV) ;
//! - boucles de Wilson R×T dans le plan (a, b) (R le long de a, T le long de b), moyennées sur tous les sites :
//!   brute W = ½Tr W(C) et CORRIGÉE W̃ = (Π_{p ⊂ C, plan (a,b)} z_p) ½Tr W(C) (convention de lib_tube.py `wilson_tilde`).
//!   W̃ ne dépend pas de la position de la pile (un déplacement de pile change à la fois le signe des liens traversés et
//!   l'ensemble des z_p intérieurs) ; quand C entoure la pile du plan (a,b), W et W̃ diffèrent d'un signe. Sans torsion dans le
//!   plan (a,b), W̃ = W. Le produit des z_p intérieurs est calculé par sommes préfixes sur le plan déplié (R ≤ ℓ_a, T ≤ ℓ_b) ;
//! - Polyakov brut dans chaque direction : moyenne de P, de |P| et de P² sur les sites de base ;
//! - fraction de colonnes de vortex du plan (a, b) : sites (x_a, x_b) où Σ_{autres coordonnées} z_p ½Tr U_p < 0.

use super::reseau::{indice_plan, plaquette, polyakov_lignes, GeoA, Pile, PLANS};
use crate::heritage::su2_hb::{qdag, qmul, Q};
use rayon::prelude::*;

/// Taille des blocs de sites des sommes déterministes (et, pour l'estimation de mémoire du CLI, de leurs sommes partielles).
pub const BLOC: usize = 4096;

/// Somme déterministe de `k` composantes sur les sites : blocs fixes de BLOC sites (parallèles), puis somme des blocs dans
/// l'ordre.
fn somme_sites<F>(v: usize, k: usize, f: F) -> Vec<f64>
where
    F: Fn(usize, &mut [f64]) + Sync,
{
    let parts: Vec<Vec<f64>> = (0..v.div_ceil(BLOC))
        .into_par_iter()
        .map(|b| {
            let mut acc = vec![0.0; k];
            for s in b * BLOC..((b + 1) * BLOC).min(v) {
                f(s, &mut acc);
            }
            acc
        })
        .collect();
    let mut out = vec![0.0; k];
    for p in parts {
        for (o, x) in out.iter_mut().zip(p) {
            *o += x;
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct ParamObs {
    /// Plan des boucles de Wilson et des colonnes de vortex (a < b).
    pub plan: (usize, usize),
    pub rmax: usize,
    pub tmax: usize,
}

impl ParamObs {
    /// « ab:RMAX:TMAX » (ex. « 01:8:8 »).
    pub fn analyser(t: &str, g: &GeoA) -> crate::commun::Resultat<ParamObs> {
        use crate::commun::Erreur;
        let p: Vec<&str> = t.split(':').collect();
        if p.len() != 3 || p[0].len() != 2 {
            return Err(Erreur::Config(format!(
                "observables du tube : « {t} » (attendu ab:RMAX:TMAX, ex. 01:8:8)"
            )));
        }
        let a = (p[0].as_bytes()[0] as char).to_digit(10).unwrap_or(9) as usize;
        let b = (p[0].as_bytes()[1] as char).to_digit(10).unwrap_or(9) as usize;
        if a > 3 || b > 3 || a == b {
            return Err(Erreur::Config(format!("plan « {} » invalide", p[0])));
        }
        let (a, b) = (a.min(b), a.max(b));
        let rmax: usize = p[1]
            .parse()
            .map_err(|_| Erreur::Config(format!("RMAX « {} »", p[1])))?;
        let tmax: usize = p[2]
            .parse()
            .map_err(|_| Erreur::Config(format!("TMAX « {} »", p[2])))?;
        if rmax < 1 || tmax < 1 || rmax > g.d[a] / 2 || tmax > g.d[b] / 2 {
            return Err(Erreur::Config(format!(
                "boucles R ≤ {rmax}, T ≤ {tmax} : 1 ≤ R ≤ ℓ_{a}/2 = {}, 1 ≤ T ≤ ℓ_{b}/2 = {} requis",
                g.d[a] / 2,
                g.d[b] / 2
            )));
        }
        Ok(ParamObs {
            plan: (a, b),
            rmax,
            tmax,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ObsTube {
    pub plaq_plans: [f64; 6],
    pub v: f64,
    /// W̃(R,T), indice (R−1)·tmax + (T−1).
    pub wilson: Vec<f64>,
    /// W brut (sans le produit des z intérieurs).
    pub wilson_brut: Vec<f64>,
    /// par direction : (⟨P⟩, ⟨|P|⟩, ⟨P²⟩) bruts.
    pub poly: [[f64; 3]; 4],
    pub frac_vortex: f64,
}

/// Plaquettes moyennes par plan et V.
pub fn plaquettes_et_v(g: &GeoA, u: &[Q], pile: &Pile) -> ([f64; 6], f64) {
    let s = somme_sites(g.v, 6, |s, acc| {
        let x = g.coords(s);
        for (p, &(mu, nu)) in PLANS.iter().enumerate() {
            acc[p] += pile.z(p, s) * plaquette(g, u, s, &x, mu, nu)[0];
        }
    });
    let mut pl = [0.0; 6];
    let mut v = 0.0;
    for p in 0..6 {
        pl[p] = s[p] / g.v as f64;
        v += g.v as f64 - s[p];
    }
    (pl, v)
}

/// Lignes de liens : out[r−1][s] = U_dir(s) U_dir(s+1) … U_dir(s+(r−1)), r = 1..lmax.
fn lignes(g: &GeoA, u: &[Q], dir: usize, lmax: usize) -> Vec<Vec<Q>> {
    let mut out: Vec<Vec<Q>> = Vec::with_capacity(lmax);
    out.push(u[dir * g.v..(dir + 1) * g.v].to_vec());
    for r in 1..lmax {
        let prev = &out[r - 1];
        let nouv: Vec<Q> = (0..g.v)
            .into_par_iter()
            .map(|s| qmul(prev[s], u[dir * g.v + g.decaler(s, dir, r)]))
            .collect();
        out.push(nouv);
    }
    out
}

/// Boucles de Wilson (W̃, W brut) dans le plan (a, b), moyennes sur les V sites de coin.
pub fn boucles_wilson(g: &GeoA, u: &[Q], pile: &Pile, par: &ParamObs) -> (Vec<f64>, Vec<f64>) {
    let (a, b) = par.plan;
    let (rmax, tmax) = (par.rmax, par.tmax);
    let la = lignes(g, u, a, rmax);
    let lb = lignes(g, u, b, tmax);
    // sommes préfixes des z < 0 du plan (a,b), par tranche des deux autres coordonnées, sur le plan déplié 2ℓa × 2ℓb
    let p = indice_plan(a, b);
    let (da, db) = (g.d[a], g.d[b]);
    let autres: Vec<usize> = (0..4).filter(|&k| k != a && k != b).collect();
    let (c, e) = (autres[0], autres[1]);
    let w = 2 * db + 1;
    let taille = (2 * da + 1) * w;
    let ntr = g.d[c] * g.d[e];
    let mut pref = vec![0i32; ntr * taille];
    for xc in 0..g.d[c] {
        for xe in 0..g.d[e] {
            let tr = xc * g.d[e] + xe;
            let base = tr * taille;
            for i in 0..2 * da {
                for j in 0..2 * db {
                    let mut x = [0usize; 4];
                    x[a] = i % da;
                    x[b] = j % db;
                    x[c] = xc;
                    x[e] = xe;
                    let neg = i32::from(pile.z(p, g.site(&x)) < 0.0);
                    pref[base + (i + 1) * w + j + 1] =
                        neg + pref[base + i * w + j + 1] + pref[base + (i + 1) * w + j]
                            - pref[base + i * w + j];
                }
            }
        }
    }
    let n = rmax * tmax;
    let s = somme_sites(g.v, 2 * n, |s, acc| {
        let x = g.coords(s);
        let tr = x[c] * g.d[e] + x[e];
        let base = tr * taille;
        let (xa, xb) = (x[a], x[b]);
        for r in 1..=rmax {
            let s_r = g.decaler(s, a, r);
            for t in 1..=tmax {
                let s_t = g.decaler(s, b, t);
                let xx = qmul(la[r - 1][s], lb[t - 1][s_r]);
                let yy = qmul(qdag(la[r - 1][s_t]), qdag(lb[t - 1][s]));
                let re = xx[0] * yy[0] - xx[1] * yy[1] - xx[2] * yy[2] - xx[3] * yy[3];
                let cnt = pref[base + (xa + r) * w + xb + t]
                    - pref[base + xa * w + xb + t]
                    - pref[base + (xa + r) * w + xb]
                    + pref[base + xa * w + xb];
                let z = if cnt % 2 == 0 { 1.0 } else { -1.0 };
                let k = (r - 1) * tmax + (t - 1);
                acc[k] += z * re;
                acc[n + k] += re;
            }
        }
    });
    let inv = 1.0 / g.v as f64;
    (
        s[..n].iter().map(|x| x * inv).collect(),
        s[n..].iter().map(|x| x * inv).collect(),
    )
}

/// Polyakov bruts : (⟨P⟩, ⟨|P|⟩, ⟨P²⟩) par direction.
pub fn polyakov(g: &GeoA, u: &[Q]) -> [[f64; 3]; 4] {
    let mut out = [[0.0; 3]; 4];
    for (mu, o) in out.iter_mut().enumerate() {
        let l = polyakov_lignes(g, u, mu);
        let n = l.len() as f64;
        *o = [
            l.iter().sum::<f64>() / n,
            l.iter().map(|x| x.abs()).sum::<f64>() / n,
            l.iter().map(|x| x * x).sum::<f64>() / n,
        ];
    }
    out
}

/// Fraction des colonnes (x_a, x_b) dont Σ_{autres} z_p ½Tr U_p (plan (a,b)) est négative.
pub fn fraction_vortex(g: &GeoA, u: &[Q], pile: &Pile, a: usize, b: usize) -> f64 {
    let p = indice_plan(a, b);
    let vals: Vec<f64> = (0..g.v)
        .into_par_iter()
        .map(|s| {
            let x = g.coords(s);
            pile.z(p, s) * plaquette(g, u, s, &x, a, b)[0]
        })
        .collect();
    let mut col = vec![0.0; g.d[a] * g.d[b]];
    for (s, v) in vals.iter().enumerate() {
        let x = g.coords(s);
        col[x[a] * g.d[b] + x[b]] += v;
    }
    col.iter().filter(|&&x| x < 0.0).count() as f64 / col.len() as f64
}

pub fn mesurer(g: &GeoA, u: &[Q], pile: &Pile, par: &ParamObs) -> ObsTube {
    let (plaq_plans, v) = plaquettes_et_v(g, u, pile);
    let (wilson, wilson_brut) = boucles_wilson(g, u, pile, par);
    ObsTube {
        plaq_plans,
        v,
        wilson,
        wilson_brut,
        poly: polyakov(g, u),
        frac_vortex: fraction_vortex(g, u, pile, par.plan.0, par.plan.1),
    }
}

/// En-tête CSV des observables (après `measurement_index,total_steps`).
pub fn entete(par: &ParamObs) -> String {
    let mut h = vec!["V".to_string()];
    for &(m, n) in PLANS.iter() {
        h.push(format!("plaq{m}{n}"));
    }
    for d in 0..4 {
        h.push(format!("P{d}"));
        h.push(format!("absP{d}"));
        h.push(format!("P{d}sq"));
    }
    h.push(format!("frac_vortex{}{}", par.plan.0, par.plan.1));
    for r in 1..=par.rmax {
        for t in 1..=par.tmax {
            h.push(format!("Wt_{r}x{t}"));
        }
    }
    for r in 1..=par.rmax {
        for t in 1..=par.tmax {
            h.push(format!("W_{r}x{t}"));
        }
    }
    h.join(",")
}

pub fn ligne(o: &ObsTube) -> String {
    use crate::heritage::su2_hb::g17;
    let mut v = vec![g17(o.v)];
    v.extend(o.plaq_plans.iter().map(|&x| g17(x)));
    for p in &o.poly {
        v.extend(p.iter().map(|&x| g17(x)));
    }
    v.push(g17(o.frac_vortex));
    v.extend(o.wilson.iter().map(|&x| g17(x)));
    v.extend(o.wilson_brut.iter().map(|&x| g17(x)));
    v.join(",")
}

/// Moyenne et erreur par blocs : nb blocs égaux de t = ⌊n/nb⌋ valeurs ; moyenne ET erreur portent sur les MÊMES nb·t premières
/// valeurs (le reste n − nb·t est écarté des deux). Erreur = écart type des moyennes de blocs / √nb, SANS temps
/// d'autocorrélation.
pub fn moyenne_erreur(x: &[f64], nb: usize) -> (f64, f64) {
    let n = x.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let nb = nb.min(n);
    if nb < 2 {
        return (x.iter().sum::<f64>() / n as f64, f64::NAN);
    }
    let t = n / nb;
    let x = &x[..nb * t];
    let m = x.iter().sum::<f64>() / (nb * t) as f64;
    let bm: Vec<f64> = (0..nb)
        .map(|k| x[k * t..(k + 1) * t].iter().sum::<f64>() / t as f64)
        .collect();
    let mb = bm.iter().sum::<f64>() / nb as f64;
    let var = bm.iter().map(|y| (y - mb) * (y - mb)).sum::<f64>() / (nb as f64 * (nb as f64 - 1.0));
    (m, var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heritage::su2_hb::{qnorm, qscale};
    use crate::su2::torsion::Torsion;
    use crate::su2::tube::reseau::ChaineTube;

    /// Transformation de jauge aléatoire U_μ(x) → g(x) U_μ(x) g(x+μ)†.
    fn jauge(g: &GeoA, u: &[Q], graine: u64) -> Vec<Q> {
        let mut c = ChaineTube::nouvelle(g.clone(), graine);
        c.init(true);
        let gs: Vec<Q> = c.u[..g.v].to_vec();
        let mut out = u.to_vec();
        for mu in 0..4 {
            for s in 0..g.v {
                let x = g.coords(s);
                let t = g.nb_up(s, mu, x[mu]);
                let r = qmul(qmul(gs[s], u[mu * g.v + s]), qdag(gs[t]));
                out[mu * g.v + s] = qscale(r, 1.0 / qnorm(r));
            }
        }
        out
    }

    #[test]
    fn invariance_de_jauge_et_vide() {
        let g = GeoA::nouvelle([6, 4, 3, 2]).unwrap();
        let pile = Pile::depuis_torsion(&g, &Torsion::analyser("01,23").unwrap());
        let par = ParamObs {
            plan: (0, 1),
            rmax: 3,
            tmax: 2,
        };
        let mut c = ChaineTube::nouvelle(g.clone(), 9);
        c.init(true);
        for _ in 0..3 {
            c.pas(1.5, 1, &pile);
        }
        let o1 = mesurer(&g, &c.u, &pile, &par);
        let u2 = jauge(&g, &c.u, 77);
        let o2 = mesurer(&g, &u2, &pile, &par);
        assert!((o1.v - o2.v).abs() < 1e-9);
        assert!((o1.v - c.action(&pile)).abs() < 1e-9);
        for (x, y) in o1.wilson.iter().zip(&o2.wilson) {
            assert!((x - y).abs() < 1e-12);
        }
        for d in 0..4 {
            assert!((o1.poly[d][1] - o2.poly[d][1]).abs() < 1e-12);
        }
        assert_eq!(o1.frac_vortex, o2.frac_vortex);
    }

    #[test]
    fn mince_wilson_corrige() {
        // vortex mince : W brut = 1 partout ; W̃ = −1 pour les boucles qui contiennent la pile (coin (ℓ0−1, ℓ1−1)) du plan 01.
        let (l, ff) = (6usize, 2usize);
        let g = GeoA::nouvelle([l, l, ff, ff]).unwrap();
        let pile = Pile::depuis_torsion(&g, &Torsion::analyser("01,23").unwrap());
        let mut c = ChaineTube::nouvelle(g.clone(), 1);
        c.vide(true, true);
        let par = ParamObs {
            plan: (0, 1),
            rmax: 3,
            tmax: 3,
        };
        let (wt, w) = boucles_wilson(&g, &c.u, &pile, &par);
        for r in 1..=3 {
            for t in 1..=3 {
                let k = (r - 1) * 3 + (t - 1);
                assert!((w[k] - 1.0).abs() < 1e-14);
                // fraction des coins dont le rectangle contient la plaquette de pile : r·t/ℓ²
                let attendu = 1.0 - 2.0 * (r * t) as f64 / (l * l) as f64;
                assert!(
                    (wt[k] - attendu).abs() < 1e-12,
                    "R={r} T={t} {} {}",
                    wt[k],
                    attendu
                );
            }
        }
        // une colonne sur ℓ² est un vortex
        assert!((fraction_vortex(&g, &c.u, &pile, 0, 1) - 1.0 / (l * l) as f64).abs() < 1e-14);
        // fibre tordue : Polyakov de fibre nul en moyenne au vide
        let pp = polyakov(&g, &c.u);
        assert!(pp[2][1].abs() < 1e-14 && pp[3][1].abs() < 1e-14);
    }

    #[test]
    fn blocs() {
        let x: Vec<f64> = (0..100).map(|i| (i % 2) as f64).collect();
        let (m, e) = moyenne_erreur(&x, 10);
        assert!((m - 0.5).abs() < 1e-15);
        assert!(e.abs() < 1e-15);
    }
}
