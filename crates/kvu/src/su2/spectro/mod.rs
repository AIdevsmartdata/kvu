//! [EXPÉRIMENTAL] Spectroscopie SU(2) sur boîtes rectangulaires : torelons (lignes de Polyakov enroulées lissées) et GEVP.
//!
//! Conventions (voir `docs/v0.2/notes/spectro.md` et `ETAT_DES_LIEUX.md` du chantier) :
//! - réseau et liens de `su2::tube` : site s = ((x0·ℓ1 + x1)·ℓ2 + x2)·ℓ3 + x3, lien li = μ·V + s, quaternion unitaire q,
//!   ½Tr U = q[0], `qmul` = produit matriciel dans l'ordre ;
//! - AXE DU TEMPS `temps` et direction d'enroulement `w` au choix (distincts) ; les deux axes restants sont les axes
//!   TRANSVERSES `a < b` ;
//! - lissage APE spatial dans la tranche de temps (liens des trois directions ≠ temps, agrafes des plans spatiaux seulement) :
//!   V = (1 − α) U + (α/4) Σ_{4 agrafes} puis projection sur SU(2) (V/|V|), mise à jour simultanée à chaque itération ; les
//!   liens de temps ne sont jamais modifiés ;
//! - opérateur : P_w(x⊥, t) = ½ Tr ∏_{x_w} Ũ_w, φ_s(t; p) = (1/N⊥) Σ_{x⊥} P_w(x⊥, t) e^{i p·x⊥} avec p = 2π (n_a/ℓ_a, n_b/ℓ_b) ;
//! - corrélateur par configuration, symétrisé : C_{ss'}(τ; p) = (1/2T) Σ_t Re[φ_s(t)* φ_s'(t+τ) + φ_s'(t)* φ_s(t+τ)],
//!   t + τ modulo T (temps périodique). La moyenne sur les configurations, l'erreur et le GEVP sont dans
//!   `scripts/spectro/analyse_spectro.py` (versionné avec le module).
//! Aucun ⟨φ⟩ n'est soustrait ici : `phi_moyen` rapporte ⟨φ_s⟩ par configuration.
#![allow(clippy::doc_lazy_continuation)] // style seulement : mise en forme des listes de la documentation

pub mod cli;
pub mod cubique;
pub mod gpu;
pub mod flux;
pub mod gpu_torelon;
pub mod profil;
pub mod bloc;
pub mod neutre;
pub mod deux_niveaux;

use crate::commun::{Erreur, Resultat};
use crate::heritage::su2_hb::{qadd, qdag, qmul, qnorm, qscale, Q};
use crate::su2::tube::reseau::GeoA;
use rayon::prelude::*;

pub const UN: Q = [1.0, 0.0, 0.0, 0.0];

/// Contrôles négatifs (mutants), jamais en production : chacun doit faire échouer au moins un test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutant {
    Aucun,
    /// la ligne s'enroule le long du premier axe transverse au lieu de `w`
    MauvaiseDirection,
    /// le produit omet le dernier lien de la ligne
    LienOublie,
    /// la séparation en temps du corrélateur est décalée d'un cran (τ + 1 au lieu de τ)
    SeparationDecalee,
}

impl Mutant {
    pub fn analyser(t: Option<&str>) -> Resultat<Mutant> {
        match t {
            None | Some("aucun") => Ok(Mutant::Aucun),
            Some("mauvaise-direction") => Ok(Mutant::MauvaiseDirection),
            Some("lien-oublie") => Ok(Mutant::LienOublie),
            Some("separation-decalee") => Ok(Mutant::SeparationDecalee),
            Some(a) => Err(Erreur::Config(format!(
                "--mutant « {a} » inconnu (mauvaise-direction|lien-oublie|separation-decalee)"
            ))),
        }
    }
}

/// Axes d'une mesure : temps, enroulement, deux axes transverses (croissants).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Axes {
    pub w: usize,
    pub temps: usize,
    pub tr: [usize; 2],
}

impl Axes {
    pub fn nouveaux(w: usize, temps: usize) -> Resultat<Axes> {
        if w > 3 || temps > 3 || w == temps {
            return Err(Erreur::Config(format!(
                "axes : enroulement {w} et temps {temps} doivent être deux axes distincts de 0..3"
            )));
        }
        let r: Vec<usize> = (0..4).filter(|&d| d != w && d != temps).collect();
        Ok(Axes { w, temps, tr: [r[0], r[1]] })
    }
}

/// Une itération de lissage APE spatial (liens de temps inchangés), mise à jour simultanée.
pub fn ape_iteration(g: &GeoA, u: &[Q], temps: usize, alpha: f64) -> Vec<Q> {
    ape_iteration_gen(g, u, temps, alpha, false)
}

/// Même fonction ; `faux` = mutant (agrafe basse sans le dagger du premier lien : la covariance de jauge est perdue).
pub fn ape_iteration_gen(g: &GeoA, u: &[Q], temps: usize, alpha: f64, faux: bool) -> Vec<Q> {
    let mut out = u.to_vec();
    let spatiaux: Vec<usize> = (0..4).filter(|&d| d != temps).collect();
    for &mu in &spatiaux {
        let nouv: Vec<Q> = (0..g.v)
            .into_par_iter()
            .map(|s| {
                let x = g.coords(s);
                let ld = |d: usize, site: usize| u[d * g.v + site];
                let smu = g.nb_up(s, mu, x[mu]);
                let mut a: Q = [0.0; 4];
                for &nu in &spatiaux {
                    if nu == mu {
                        continue;
                    }
                    let snu = g.nb_up(s, nu, x[nu]);
                    let sdn = g.nb_dn(s, nu, x[nu]);
                    let smu_dn = g.nb_dn(smu, nu, x[nu]);
                    // agrafe haute U_ν(x) U_μ(x+ν) U_ν(x+μ)†, basse U_ν(x−ν)† U_μ(x−ν) U_ν(x−ν+μ)
                    let haut = qmul(qmul(ld(nu, s), ld(mu, snu)), qdag(ld(nu, smu)));
                    let prem = if faux { ld(nu, sdn) } else { qdag(ld(nu, sdn)) };
                    let bas = qmul(qmul(prem, ld(mu, sdn)), ld(nu, smu_dn));
                    a = qadd(a, qadd(haut, bas));
                }
                let v = qadd(qscale(ld(mu, s), 1.0 - alpha), qscale(a, alpha / 4.0));
                let n = qnorm(v);
                if n > 1e-300 {
                    qscale(v, 1.0 / n)
                } else {
                    ld(mu, s)
                }
            })
            .collect();
        out[mu * g.v..(mu + 1) * g.v].copy_from_slice(&nouv);
    }
    out
}

/// n itérations de lissage APE.
pub fn lisser(g: &GeoA, u: &[Q], temps: usize, alpha: f64, n: usize) -> Vec<Q> {
    let mut c = u.to_vec();
    for _ in 0..n {
        c = ape_iteration(g, &c, temps, alpha);
    }
    c
}

/// Axe effectif de la ligne (mutant MauvaiseDirection : premier axe transverse).
fn axe_ligne(ax: &Axes, m: Mutant) -> usize {
    if m == Mutant::MauvaiseDirection {
        ax.tr[0]
    } else {
        ax.w
    }
}

/// ½Tr ∏_{k} U_w(x + k ŵ), x_w = 0 (mutant LienOublie : omet le dernier lien).
pub fn trace_ligne(g: &GeoA, u: &[Q], w: usize, base: usize, m: Mutant) -> f64 {
    let n = if m == Mutant::LienOublie { g.d[w] - 1 } else { g.d[w] };
    let mut s = base;
    let mut p = UN;
    for k in 0..n {
        p = qmul(p, u[w * g.v + s]);
        if k != g.d[w] - 1 {
            s += g.st[w];
        }
    }
    p[0]
}

/// Toutes les lignes P_w(x⊥, t), indexées [t][a][b] (a, b = coordonnées sur les axes transverses).
pub fn lignes_points(g: &GeoA, u: &[Q], ax: &Axes, m: Mutant) -> Vec<f64> {
    let w = axe_ligne(ax, m);
    let (ta, tb, tt) = (g.d[ax.tr[0]], g.d[ax.tr[1]], g.d[ax.temps]);
    // pour MauvaiseDirection, les « transverses » de la boucle sont tout de même ceux de `ax` (indexation inchangée) ;
    // la base est le site (x_w = 0) pour les coordonnées (a, b, t) : si w' est l'un des axes de l'indexation, il est mis à 0.
    (0..tt * ta * tb)
        .into_par_iter()
        .map(|j| {
            let t = j / (ta * tb);
            let a = (j / tb) % ta;
            let b = j % tb;
            let mut x = [0usize; 4];
            x[ax.temps] = t;
            x[ax.tr[0]] = a;
            x[ax.tr[1]] = b;
            x[ax.w] = 0;
            x[w] = 0;
            trace_ligne(g, u, w, g.site(&x), m)
        })
        .collect()
}

/// φ_s(t; p) pour une liste d'impulsions (n_a, n_b), normalisé par N⊥ ; résultat [impulsion][t] = (Re, Im).
/// Somme séquentielle dans chaque tranche (déterministe au bit près).
pub fn phi(g: &GeoA, u: &[Q], ax: &Axes, moments: &[[i32; 2]], m: Mutant) -> Vec<Vec<[f64; 2]>> {
    let (ta, tb, tt) = (g.d[ax.tr[0]], g.d[ax.tr[1]], g.d[ax.temps]);
    let p = lignes_points(g, u, ax, m);
    let nperp = (ta * tb) as f64;
    moments
        .iter()
        .map(|&[na, nb]| {
            (0..tt)
                .map(|t| {
                    let (mut re, mut im) = (0.0, 0.0);
                    for a in 0..ta {
                        for b in 0..tb {
                            let ph = 2.0
                                * std::f64::consts::PI
                                * (na as f64 * a as f64 / ta as f64 + nb as f64 * b as f64 / tb as f64);
                            let v = p[(t * ta + a) * tb + b];
                            re += v * ph.cos();
                            im += v * ph.sin();
                        }
                    }
                    [re / nperp, im / nperp]
                })
                .collect()
        })
        .collect()
}

/// Paires (s ≤ s') dans l'ordre de stockage : (0,0), (0,1) … (0,n−1), (1,1) …
pub fn paires(n: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for a in 0..n {
        for b in a..n {
            v.push((a, b));
        }
    }
    v
}

/// Corrélateur symétrisé moyenné sur t : `phis[niveau][t]` (complexe) → C[paire][τ], τ = 0..=tau_max.
/// Mutant SeparationDecalee : τ + 1 au lieu de τ.
pub fn correlateur(phis: &[Vec<[f64; 2]>], tau_max: usize, m: Mutant) -> Vec<Vec<f64>> {
    let tt = phis[0].len();
    let dec = if m == Mutant::SeparationDecalee { 1 } else { 0 };
    paires(phis.len())
        .into_iter()
        .map(|(a, b)| {
            (0..=tau_max)
                .map(|tau| {
                    let mut acc = 0.0;
                    for t in 0..tt {
                        let t2 = (t + tau + dec) % tt;
                        let (x, y) = (phis[a][t], phis[b][t2]);
                        let (x2, y2) = (phis[b][t], phis[a][t2]);
                        acc += x[0] * y[0] + x[1] * y[1] + x2[0] * y2[0] + x2[1] * y2[1];
                    }
                    acc / (2.0 * tt as f64)
                })
                .collect()
        })
        .collect()
}

/// ⟨φ_s⟩ moyenné sur t, (Re, Im) — rapporté tel quel, jamais soustrait.
pub fn phi_moyen(phis: &[Vec<[f64; 2]>]) -> Vec<[f64; 2]> {
    phis.iter()
        .map(|f| {
            let n = f.len() as f64;
            [f.iter().map(|v| v[0]).sum::<f64>() / n, f.iter().map(|v| v[1]).sum::<f64>() / n]
        })
        .collect()
}

/// Corrélateur « ponctuel » ⟨P(x)P(x + r ê_i)⟩ de l'ancienne mesure `obs polyakov_corr`, sur un CUBE : lignes enroulées le
/// long de `ax.w`, séparation r le long des trois axes (temps et deux transverses), moyenne sur les sites et les trois axes,
/// r = 0..L/2, sommes `k_reduce`/`k_final` comme `ref_cpu::correlateur_polyakov` ; sert au test d'accord exact.
pub fn polyakov_corr_cube(g: &GeoA, u: &[Q], ax: &Axes, m: Mutant) -> Vec<f64> {
    let l = g.d[ax.w];
    let pts = lignes_points(g, u, ax, m); // [t][a][b]
    let l3 = l * l * l;
    // réordonnées comme `ref_cpu::polyakov_sites` : axes ≠ w croissants (d0 < d1 < d2), d2 le plus rapide
    let dirs: Vec<usize> = (0..4).filter(|&d| d != ax.w).collect();
    let ell: Vec<f64> = (0..l3)
        .map(|j| {
            let mut x = [0usize; 4];
            x[dirs[2]] = j % l;
            x[dirs[1]] = (j / l) % l;
            x[dirs[0]] = j / (l * l);
            pts[(x[ax.temps] * l + x[ax.tr[0]]) * l + x[ax.tr[1]]]
        })
        .collect();
    let decal = |j: usize, axe: usize, r: usize| -> usize {
        let mut c = [j / (l * l), (j / l) % l, j % l];
        c[axe] = (c[axe] + r) % l;
        (c[0] * l + c[1]) * l + c[2]
    };
    (0..=l / 2)
        .map(|r| {
            let par_site: Vec<f64> = (0..l3)
                .map(|j| ell[j] * ell[decal(j, 0, r)] + ell[j] * ell[decal(j, 1, r)] + ell[j] * ell[decal(j, 2, r)])
                .collect();
            let k = crate::heritage::su2_hb::k_reduce(&par_site);
            crate::heritage::su2_hb::k_final(&k, 1.0 / (3.0 * l3 as f64))
        })
        .collect()
}

/// Transformation de jauge g(x) : U_μ(x) → g(x) U_μ(x) g(x+μ̂)†, g de Haar aléatoires (graine donnée).
pub fn transformer_jauge(g: &GeoA, u: &mut [Q], graine: u64) {
    let (k0, k1) = crate::su2::tube::reseau::cle_philox(graine);
    let gs: Vec<Q> = (0..g.v)
        .map(|s| crate::su2::cpu::haar_q(s as u32, 0xABCD, 7, k0, k1))
        .collect();
    let copie = u.to_vec();
    for mu in 0..4 {
        for s in 0..g.v {
            let x = g.coords(s);
            let s2 = g.nb_up(s, mu, x[mu]);
            u[mu * g.v + s] = qmul(qmul(gs[s], copie[mu * g.v + s]), qdag(gs[s2]));
        }
    }
}

/// Paramètres déclarés d'une mesure.
#[derive(Clone, Debug)]
pub struct ParamMesure {
    pub axes: Axes,
    /// nombres d'itérations APE (croissants), un opérateur par niveau
    pub niveaux: Vec<usize>,
    pub alpha: f64,
    pub moments: Vec<[i32; 2]>,
    pub tau_max: usize,
    pub mutant: Mutant,
}

/// Résultat d'une mesure sur une configuration.
#[derive(Clone, Debug)]
pub struct Mesure {
    /// [impulsion][paire][τ]
    pub corr: Vec<Vec<Vec<f64>>>,
    /// [impulsion][niveau] = ⟨φ⟩_t (Re, Im)
    pub phi_moyen: Vec<Vec<[f64; 2]>>,
    /// [niveau][τ] = (1/N⊥T) Σ_{x⊥,t} P(x⊥,t) P(x⊥,t+τ) : corrélateur PONCTUEL (même x⊥), pour le test de couplage fort
    pub ponctuel: Vec<Vec<f64>>,
}

/// Corrélateur ponctuel ⟨P(x⊥,t)P(x⊥,t+τ)⟩ à partir des lignes [t][a][b].
pub fn ponctuel(g: &GeoA, lignes: &[f64], ax: &Axes, tau_max: usize) -> Vec<f64> {
    let (ta, tb, tt) = (g.d[ax.tr[0]], g.d[ax.tr[1]], g.d[ax.temps]);
    (0..=tau_max)
        .map(|tau| {
            let mut s = 0.0;
            for t in 0..tt {
                let t2 = (t + tau) % tt;
                for ab in 0..ta * tb {
                    s += lignes[t * ta * tb + ab] * lignes[t2 * ta * tb + ab];
                }
            }
            s / (tt * ta * tb) as f64
        })
        .collect()
}

/// Mesure complète : lissage par niveaux croissants, φ, corrélateurs symétrisés, ⟨φ⟩.
pub fn mesurer(g: &GeoA, u: &[Q], p: &ParamMesure) -> Resultat<Mesure> {
    if p.niveaux.is_empty() || p.niveaux.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Erreur::Config("niveaux de lissage : liste croissante non vide attendue".into()));
    }
    if p.tau_max >= g.d[p.axes.temps] {
        return Err(Erreur::Config("tau_max doit être < ℓ_temps".into()));
    }
    let mut courant = u.to_vec();
    let mut fait = 0usize;
    // phis[impulsion][niveau][t]
    let mut phis: Vec<Vec<Vec<[f64; 2]>>> = vec![Vec::new(); p.moments.len()];
    let mut ponc: Vec<Vec<f64>> = Vec::new();
    for &n in &p.niveaux {
        for _ in fait..n {
            courant = ape_iteration(g, &courant, p.axes.temps, p.alpha);
        }
        fait = n;
        ponc.push(ponctuel(g, &lignes_points(g, &courant, &p.axes, p.mutant), &p.axes, p.tau_max));
        let ph = phi(g, &courant, &p.axes, &p.moments, p.mutant);
        for (i, f) in ph.into_iter().enumerate() {
            phis[i].push(f);
        }
    }
    let corr = phis.iter().map(|f| correlateur(f, p.tau_max, p.mutant)).collect();
    let phi_moyen = phis.iter().map(|f| self::phi_moyen(f)).collect();
    Ok(Mesure { corr, phi_moyen, ponctuel: ponc })
}

#[cfg(test)]
mod tests;
