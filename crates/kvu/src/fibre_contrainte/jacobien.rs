//! Jacobien analytique `J = DŪ` de la carte de Bałaban (L_b = 3, SU(2)), forme fermée de `docs/FIBRE_COAIRE.md` §3.3, et gradient
//! de `½ log det(J Jᵀ)` par passe arrière à la main (WP12a, `FIB-JAC`, `FIB-LOGDET`).
//!
//! # Conventions (WP11 §1)
//! - perturbation **à gauche** `U_ℓ → exp(i ε_ℓ·σ) U_ℓ`, `ε_ℓ ∈ ℝ³` orthonormées (SU(2) = S³ de rayon 1) ;
//! - `Ū_c → exp(i Δ_c·σ) Ū_c`, `Δ_c = Σ_ℓ J_{cℓ} ε_ℓ` ; `J` est la matrice 3n × 3N de blocs 3×3 `J_{cℓ}` ;
//! - `R(A)` : `A (i ε·σ) A† = i (R(A) ε)·σ` ; logarithme `ω(W)` de `rg::bloc::log_q` (demi-angle) ;
//! - lien grossier `c = μ·V_c + site(z)`, lien fin `μ·V + site(x)` (comme `rg::bloc`).
//!
//! # Formule (WP11 §3.3, réécrite dans le repère de `U₀`)
//! Avec `Y_k = U₀⁻¹ P*_k`, `N_{Γ,k} = R(Y_k)`, `L_Γ = L_g(ω_Γ)`, `E = E_g(ω̄)` :
//! `J_{cℓ} = R₀ [ N0_ℓ + E Λ_ℓ ]`, `Λ_ℓ = Σ_Γ w L_Γ (N_Γ(ℓ) − N0_ℓ)`, `N_Γ(ℓ) = Σ_k s_k N_{Γ,k}`.
//! C'est exactement `J = (1 − B_tot) T₀ + Σ_Γ B_Γ T_Γ`, `B_Γ = w R₀ E L_Γ R₀ᵀ`.
//!
//! # Gradient (passe arrière, à `W = G⁻¹ J` figé)
//! `∇ ½ log det(J Jᵀ) = ∇_U h`, `h = Σ_c Σ_ℓ ⟨W_{cℓ}, J_{cℓ}(U)⟩` (`G = J Jᵀ`). Dérivée : `δR(P) = −2 [η×] R(P)`, `δω_Γ = L_Γ ν_Γ`,
//! dérivées fermées de `L_g` et `E_g` (WP11 §3.2). Le terme « rotation du repère » `Σ_ℓ J̃_ℓ V_ℓᵀ = 1` s'annule (`J Wᵀ = 1`).
//! La somme sur les `10 473` boucles est factorisée : un chemin d'aller `i`, un segment `r` et un chemin de retour `j` ne se
//! couplent que par `L_{ij}` ; les accumulations par `i` et par `j` sont faites avant de toucher aux 3×3 des liens.
//!
//! Déterminisme : chaque ligne `c` est calculée indépendamment (ordre fixe des sommes) ; les sommes entre lignes sont
//! faites en série dans l'ordre de `c`. Le résultat ne dépend donc pas du nombre de fils.

#![allow(clippy::needless_range_loop)]

use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul};
use crate::rg::bloc::log_q;
use crate::rg::geometrie::{Geometrie, Table, SITES_BLOC};
use rayon::prelude::*;
use std::collections::HashMap;

pub type Q = [f64; 4];
/// Matrice 3×3 en ordre ligne.
pub type M3 = [f64; 9];
const IDQ: Q = [1.0, 0.0, 0.0, 0.0];
const ZERO: M3 = [0.0; 9];
const ID3: M3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

// ------------------------------------------------------------------------------------------
// Mutants (compilés en test seulement)
// ------------------------------------------------------------------------------------------

/// Mutants de l'instrument (jamais compilés hors des tests).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Mutant {
    #[default]
    Aucun,
    /// FIB-JAC : dérivée de `log` remplacée par l'identité (`L_g := 1` dans `J`).
    #[cfg(test)]
    LIdentite,
    /// FIB-LOGDET : termes en `∂E_g` retirés du gradient.
    #[cfg(test)]
    SansDerivE,
    /// FIB-LOGDET : termes en `∂L_g` retirés du gradient.
    #[cfg(test)]
    SansDerivL,
    /// FIB-LOGDET : termes en `∂R` (rotations de transport) retirés du gradient.
    #[cfg(test)]
    SansDerivR,
    /// FIB-LOGDET : les trois familles de dérivées de J (∂R, ∂L_g, ∂E_g) retirées ensemble.
    #[cfg(test)]
    SansDerivees,
}

/// Réglages de l'évaluation (le seul champ est le mutant, `Aucun` en production).
#[derive(Clone, Copy, Debug, Default)]
pub struct Reglages {
    pub mutant: Mutant,
}

// ------------------------------------------------------------------------------------------
// Petites matrices 3×3
// ------------------------------------------------------------------------------------------

#[inline]
fn mm(a: &M3, b: &M3) -> M3 {
    let mut r = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
        }
    }
    r
}
/// aᵀ b
#[inline]
fn mtm(a: &M3, b: &M3) -> M3 {
    let mut r = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[3 * i + j] = a[i] * b[j] + a[3 + i] * b[3 + j] + a[6 + i] * b[6 + j];
        }
    }
    r
}
/// a bᵀ
#[inline]
fn mmt(a: &M3, b: &M3) -> M3 {
    let mut r = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[3 * i + j] = a[3 * i] * b[3 * j] + a[3 * i + 1] * b[3 * j + 1] + a[3 * i + 2] * b[3 * j + 2];
        }
    }
    r
}
#[inline]
fn add(a: &mut M3, b: &M3) {
    for k in 0..9 {
        a[k] += b[k];
    }
}
#[inline]
fn sub(a: &mut M3, b: &M3) {
    for k in 0..9 {
        a[k] -= b[k];
    }
}
#[inline]
fn axpy(a: &mut M3, s: f64, b: &M3) {
    for k in 0..9 {
        a[k] += s * b[k];
    }
}
#[inline]
fn scale(a: &M3, s: f64) -> M3 {
    let mut r = *a;
    for x in r.iter_mut() {
        *x *= s;
    }
    r
}
/// tr(a b)
#[inline]
fn tr_prod(a: &M3, b: &M3) -> f64 {
    let mut t = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            t += a[3 * i + j] * b[3 * j + i];
        }
    }
    t
}
/// a v
#[inline]
fn mv(a: &M3, v: [f64; 3]) -> [f64; 3] {
    [
        a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
        a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
        a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
    ]
}
/// aᵀ v
#[inline]
fn mtv(a: &M3, v: [f64; 3]) -> [f64; 3] {
    [
        a[0] * v[0] + a[3] * v[1] + a[6] * v[2],
        a[1] * v[0] + a[4] * v[1] + a[7] * v[2],
        a[2] * v[0] + a[5] * v[1] + a[8] * v[2],
    ]
}
#[inline]
fn vadd(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn vaxpy(a: &mut [f64; 3], s: f64, b: [f64; 3]) {
    a[0] += s * b[0];
    a[1] += s * b[1];
    a[2] += s * b[2];
}
/// a(X)_c = Σ_ab ε_abc X_ab
#[inline]
fn axial(x: &M3) -> [f64; 3] {
    [x[5] - x[7], x[6] - x[2], x[1] - x[3]]
}
#[inline]
fn croix(v: [f64; 3]) -> M3 {
    [0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0]
}
#[inline]
fn outer(a: [f64; 3], b: [f64; 3]) -> M3 {
    [
        a[0] * b[0], a[0] * b[1], a[0] * b[2], a[1] * b[0], a[1] * b[1], a[1] * b[2], a[2] * b[0], a[2] * b[1], a[2] * b[2],
    ]
}

/// R(q) : `(q0² − |v|²) 1 − 2 q0 [v×] + 2 v vᵀ`.
#[inline]
pub fn rad(q: Q) -> M3 {
    let v = [q[1], q[2], q[3]];
    let d = q[0] * q[0] - (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    let mut r = [0.0; 9];
    let c = croix(v);
    let o = outer(v, v);
    for k in 0..9 {
        r[k] = -2.0 * q[0] * c[k] + 2.0 * o[k];
    }
    r[0] += d;
    r[4] += d;
    r[8] += d;
    r
}

// ------------------------------------------------------------------------------------------
// L_g, E_g et leurs dérivées (WP11 §3.1-3.2)
// ------------------------------------------------------------------------------------------

// Séries en θ² (indice n ↔ θ^{2n}), générées par SymPy (scratch/wp12a/proto/gen_series.py).
const SER_F: [f64; 10] = [1.0, -0.3333333333333333, -0.022222222222222223, -0.0021164021164021165, -0.00021164021164021165, -2.1377799155576935e-05, -2.1644042808063972e-06, -2.1925947851873778e-07, -2.2214608789979678e-08, -2.2507846516808994e-09];
const SER_FP: [f64; 10] = [-0.6666666666666666, -0.08888888888888889, -0.012698412698412698, -0.0016931216931216932, -0.00021377799155576933, -2.5972851369676765e-05, -3.069632699262329e-06, -3.5543374063967485e-07, -4.0514123730256185e-08, -4.561030240918436e-09];
const SER_H: [f64; 10] = [0.3333333333333333, 0.022222222222222223, 0.0021164021164021165, 0.00021164021164021165, 2.1377799155576935e-05, 2.1644042808063972e-06, 2.1925947851873778e-07, 2.2214608789979678e-08, 2.2507846516808994e-09, 2.2805151204592183e-10];
const SER_HP: [f64; 10] = [0.044444444444444446, 0.008465608465608466, 0.0012698412698412698, 0.00017102239324461548, 2.1644042808063973e-05, 2.631113742224853e-06, 3.110045230597155e-07, 3.601255442689439e-08, 4.1049272168265925e-09, 4.621286519800525e-10];
const SER_S: [f64; 10] = [1.0, -0.6666666666666666, 0.13333333333333333, -0.012698412698412698, 0.0007054673721340388, -2.565335898669232e-05, 6.577784355562133e-07, -1.2529113058213587e-08, 1.8425166262078804e-10, -2.1549902060910882e-12];
const SER_SP: [f64; 10] = [-1.3333333333333333, 0.5333333333333333, -0.0761904761904762, 0.005643738977072311, -0.0002565335898669232, 7.89334122667456e-06, -1.7540758281499022e-07, 2.9480266019326086e-09, -3.878982370963959e-11, 4.104743249697311e-13];
const SER_K: [f64; 10] = [0.6666666666666666, -0.13333333333333333, 0.012698412698412698, -0.0007054673721340388, 2.565335898669232e-05, -6.577784355562133e-07, 1.2529113058213587e-08, -1.8425166262078804e-10, 2.1549902060910882e-12, -2.0523716248486557e-14];
const SER_KP: [f64; 10] = [-0.26666666666666666, 0.050793650793650794, -0.004232804232804233, 0.00020522687189353857, -6.5777843555621335e-06, 1.5034935669856304e-07, -2.5795232766910326e-09, 3.447984329745741e-11, -3.69426892472758e-13, 3.244856323871392e-15];
const SER_M: [f64; 10] = [1.0, -0.3333333333333333, 0.044444444444444446, -0.0031746031746031746, 0.00014109347442680775, -4.275559831115387e-06, 9.39683479366019e-08, -1.5661391322766984e-09, 2.0472406957865337e-11, -2.1549902060910883e-13];
const SER_MP: [f64; 10] = [-0.6666666666666666, 0.17777777777777778, -0.01904761904761905, 0.001128747795414462, -4.275559831115387e-05, 1.1276201752392229e-06, -2.1925947851873777e-08, 3.275585113258454e-10, -3.878982370963959e-12, 3.731584772452101e-14];

/// Seuil de bascule série / forme fermée (θ).
const SEUIL_SERIE: f64 = 0.25;

#[inline]
fn horner(c: &[f64; 10], t2: f64) -> f64 {
    let mut s = 0.0;
    for k in (0..10).rev() {
        s = s * t2 + c[k];
    }
    s
}

/// (f, h, f'/θ, h'/θ) : `L_g = f 1 + h ω ωᵀ + [ω×]`, `f = θ cot θ`, `h = (1 − f)/θ²`.
fn coeffs_l(th: f64) -> (f64, f64, f64, f64) {
    if th < SEUIL_SERIE {
        let t2 = th * th;
        return (horner(&SER_F, t2), horner(&SER_H, t2), horner(&SER_FP, t2), horner(&SER_HP, t2));
    }
    let f = th / th.tan();
    let h = (1.0 - f) / (th * th);
    let fd = 1.0 / th.tan() - th / (th.sin() * th.sin());
    let fp = fd / th;
    let hp = (-fd / (th * th) - 2.0 * (1.0 - f) / (th * th * th)) / th;
    (f, h, fp, hp)
}

/// (s, k, m, s'/θ, k'/θ, m'/θ) : `E_g = s 1 + k ω ωᵀ − m [ω×]`, `s = sin 2θ/(2θ)`, `k = (1 − s)/θ²`, `m = sin²θ/θ²`.
fn coeffs_e(th: f64) -> (f64, f64, f64, f64, f64, f64) {
    if th < SEUIL_SERIE {
        let t2 = th * th;
        return (
            horner(&SER_S, t2),
            horner(&SER_K, t2),
            horner(&SER_M, t2),
            horner(&SER_SP, t2),
            horner(&SER_KP, t2),
            horner(&SER_MP, t2),
        );
    }
    let s = (2.0 * th).sin() / (2.0 * th);
    let sn = th.sin() / th;
    let m = sn * sn;
    let k = (1.0 - s) / (th * th);
    let sd = (2.0 * th).cos() / th - (2.0 * th).sin() / (2.0 * th * th);
    let md = 2.0 * th.sin() * (th * th.cos() - th.sin()) / (th * th * th);
    let kd = -sd / (th * th) - 2.0 * (1.0 - s) / (th * th * th);
    (s, k, m, sd / th, kd / th, md / th)
}

#[inline]
fn norme(w: [f64; 3]) -> f64 {
    (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt()
}

/// `L_g(ω)` : `log(e^{iη·σ} W) = i(ω + L_g η)·σ`.
pub fn lg(w: [f64; 3]) -> M3 {
    let (f, h, _, _) = coeffs_l(norme(w));
    let mut r = croix(w);
    let o = outer(w, w);
    for k in 0..9 {
        r[k] += h * o[k];
    }
    r[0] += f;
    r[4] += f;
    r[8] += f;
    r
}

/// `E_g(ω) = L_g⁻¹` : `e^{i(ω+δ)·σ} = e^{i(E_g δ)·σ} e^{iω·σ}`.
pub fn eg(w: [f64; 3]) -> M3 {
    let (s, k, m, _, _, _) = coeffs_e(norme(w));
    let mut r = scale(&croix(w), -m);
    let o = outer(w, w);
    for i in 0..9 {
        r[i] += k * o[i];
    }
    r[0] += s;
    r[4] += s;
    r[8] += s;
    r
}

/// `∂L_g/∂ω_a` (a = 0, 1, 2), WP11 §3.2.
pub fn dlg(w: [f64; 3]) -> [M3; 3] {
    let (_, h, fp, hp) = coeffs_l(norme(w));
    let o = outer(w, w);
    std::array::from_fn(|a| {
        let mut e = [0.0; 3];
        e[a] = 1.0;
        let mut r = croix(e);
        let ew = outer(e, w);
        let we = outer(w, e);
        for k in 0..9 {
            r[k] += hp * w[a] * o[k] + h * (ew[k] + we[k]);
        }
        r[0] += fp * w[a];
        r[4] += fp * w[a];
        r[8] += fp * w[a];
        r
    })
}

/// `∂E_g/∂ω_a` (a = 0, 1, 2), WP11 §3.2.
pub fn deg(w: [f64; 3]) -> [M3; 3] {
    let (_, k, m, sp, kp, mp) = coeffs_e(norme(w));
    let o = outer(w, w);
    let cw = croix(w);
    std::array::from_fn(|a| {
        let mut e = [0.0; 3];
        e[a] = 1.0;
        let ce = croix(e);
        let ew = outer(e, w);
        let we = outer(w, e);
        let mut r = [0.0; 9];
        for i in 0..9 {
            r[i] = kp * w[a] * o[i] + k * (ew[i] + we[i]) - mp * w[a] * cw[i] - m * ce[i];
        }
        r[0] += sp * w[a];
        r[4] += sp * w[a];
        r[8] += sp * w[a];
        r
    })
}

// ------------------------------------------------------------------------------------------
// Réseau et gabarit des chemins (relatifs à l'origine 3z du lien grossier)
// ------------------------------------------------------------------------------------------

/// Gabarit d'un lien grossier de direction `mu` : « slots » = liens fins relatifs touchés (clé `(dep, ν)`).
pub struct Gabarit {
    pub nb_slots: usize,
    /// (déplacement relatif à l'origine 3z, direction ν) de chaque slot.
    pub cles: Vec<([i8; 4], u8)>,
    /// `[p][k]` : slot du pas k du chemin d'aller p.
    pub chemin_slot: Vec<Vec<usize>>,
    pub chemin_signe: Vec<Vec<i8>>,
    /// `[p][m]` : slot du pas m du chemin p translaté de 3 e_μ (chemin de retour).
    pub retour_slot: Vec<Vec<usize>>,
    /// `[ir][k]` : slot du k-ième lien du segment x → x + 3 e_μ (x = site local ir).
    pub seg_slot: Vec<[usize; 3]>,
    /// slots des 3 liens du chemin de référence `Γ₀`.
    pub ref_slot: [usize; 3],
}

fn construire_gabarit(table: &Table, mu: usize) -> Gabarit {
    let mut map: HashMap<([i8; 4], u8), usize> = HashMap::new();
    let mut cles: Vec<([i8; 4], u8)> = Vec::new();
    let mut slot = |dep: [i8; 4], nu: u8| -> usize {
        *map.entry((dep, nu)).or_insert_with(|| {
            cles.push((dep, nu));
            cles.len() - 1
        })
    };
    let seg0 = table.segment_droit(mu);
    let ref_slot = [slot(seg0[0].dep, seg0[0].mu), slot(seg0[1].dep, seg0[1].mu), slot(seg0[2].dep, seg0[2].mu)];
    let mut chemin_slot = Vec::new();
    let mut chemin_signe = Vec::new();
    let mut retour_slot = Vec::new();
    for ch in &table.chemins {
        chemin_slot.push(ch.iter().map(|p| slot(p.dep, p.mu)).collect::<Vec<_>>());
        chemin_signe.push(ch.iter().map(|p| p.signe).collect::<Vec<_>>());
        retour_slot.push(
            ch.iter()
                .map(|p| {
                    let mut d = p.dep;
                    d[mu] += 3;
                    slot(d, p.mu)
                })
                .collect::<Vec<_>>(),
        );
    }
    let mut seg_slot = Vec::new();
    for ir in 0..SITES_BLOC {
        let s = table.segment(mu, ir);
        seg_slot.push([slot(s[0].dep, s[0].mu), slot(s[1].dep, s[1].mu), slot(s[2].dep, s[2].mu)]);
    }
    Gabarit { nb_slots: cles.len(), cles, chemin_slot, chemin_signe, retour_slot, seg_slot, ref_slot }
}

/// Géométrie d'un réseau fin L⁴ (L multiple de 3, L ≥ 6) et de ses liens grossiers.
pub struct Reseau {
    pub l: usize,
    pub lc: usize,
    pub v: usize,
    pub vc: usize,
    pub n_liens: usize,
    pub n_c: usize,
    pub table: Table,
    pub gab: [Gabarit; 4],
}

impl Reseau {
    pub fn nouveau(l: usize) -> Resultat<Reseau> {
        if l < 6 || !l.is_multiple_of(3) {
            return Err(Erreur::Config(format!(
                "fibre_contrainte : L = {l} : multiple de 3 supérieur ou égal à 6 requis (le voisinage d'un lien grossier s'enroule sinon)"
            )));
        }
        let table = Table::nouvelle(Geometrie::TroisCentre);
        let gab = std::array::from_fn(|mu| construire_gabarit(&table, mu));
        let lc = l / 3;
        Ok(Reseau { l, lc, v: l.pow(4), vc: lc.pow(4), n_liens: 4 * l.pow(4), n_c: 4 * lc.pow(4), table, gab })
    }

    /// (z, μ) du lien grossier `c`.
    pub fn coords_c(&self, c: usize) -> ([usize; 4], usize) {
        let mu = c / self.vc;
        let mut s = c % self.vc;
        let mut z = [0; 4];
        for m in (0..4).rev() {
            z[m] = s % self.lc;
            s /= self.lc;
        }
        (z, mu)
    }

    pub fn indice_c(&self, z: [usize; 4], mu: usize) -> usize {
        mu * self.vc + ((z[0] * self.lc + z[1]) * self.lc + z[2]) * self.lc + z[3]
    }

    fn id_lien(&self, origine: [usize; 4], cle: &([i8; 4], u8)) -> u32 {
        let l = self.l as isize;
        let mut x = [0usize; 4];
        for m in 0..4 {
            x[m] = (origine[m] as isize + cle.0[m] as isize).rem_euclid(l) as usize;
        }
        (cle.1 as usize * self.v + ((x[0] * self.l + x[1]) * self.l + x[2]) * self.l + x[3]) as u32
    }

    /// Identifiants globaux des liens fins des slots de la ligne `c`.
    pub fn liens_de(&self, c: usize) -> Vec<u32> {
        let (z, mu) = self.coords_c(c);
        let o = z.map(|v| 3 * v);
        self.gab[mu].cles.iter().map(|k| self.id_lien(o, k)).collect()
    }
}

// ------------------------------------------------------------------------------------------
// Contexte d'une ligne : produits partiels de tous les chemins
// ------------------------------------------------------------------------------------------

struct Ctx {
    r0: M3,
    n0: [M3; 3],
    ni: Vec<Vec<M3>>,
    nseg: Vec<[M3; 3]>,
    base: Vec<Q>,
    rbase: Vec<M3>,
    retinv0: Vec<Q>,
    rret: Vec<Vec<M3>>,
}

#[inline]
fn facteur(q: Q, s: i8) -> Q {
    if s > 0 {
        q
    } else {
        qdag(q)
    }
}

impl Ctx {
    fn nouveau(res: &Reseau, champ: &ChampHote, c: usize) -> (Ctx, Vec<u32>) {
        let (_, mu) = res.coords_c(c);
        let gab = &res.gab[mu];
        let liens = res.liens_de(c);
        let q: Vec<Q> = liens.iter().map(|&i| champ.lien(i as usize)).collect();
        // référence Γ₀
        let mut u0 = IDQ;
        let mut t0pre = [IDQ; 3];
        for k in 0..3 {
            t0pre[k] = u0;
            u0 = qmul(u0, q[gab.ref_slot[k]]);
        }
        let u0inv = qdag(u0);
        let r0 = rad(u0);
        let n0 = [rad(qmul(u0inv, t0pre[0])), rad(qmul(u0inv, t0pre[1])), rad(qmul(u0inv, t0pre[2]))];
        // segments
        let mut segq = [IDQ; SITES_BLOC];
        let mut segpre = [[IDQ; 3]; SITES_BLOC];
        for ir in 0..SITES_BLOC {
            let mut p = IDQ;
            for k in 0..3 {
                segpre[ir][k] = p;
                p = qmul(p, q[gab.seg_slot[ir][k]]);
            }
            segq[ir] = p;
        }
        let np = res.table.nb_chemins();
        let mut ni = Vec::with_capacity(np);
        let mut nseg = Vec::with_capacity(np);
        let mut base = Vec::with_capacity(np);
        let mut rbase = Vec::with_capacity(np);
        let mut retinv0 = Vec::with_capacity(np);
        let mut rret = Vec::with_capacity(np);
        for ir in 0..SITES_BLOC {
            for p in res.table.debut[ir]..res.table.debut[ir + 1] {
                // chemin d'aller
                let sl = &gab.chemin_slot[p];
                let sg = &gab.chemin_signe[p];
                let mut pre = IDQ;
                let mut v = Vec::with_capacity(sl.len());
                for k in 0..sl.len() {
                    let avant = pre;
                    pre = qmul(pre, facteur(q[sl[k]], sg[k]));
                    let pstar = if sg[k] > 0 { avant } else { pre };
                    v.push(rad(qmul(u0inv, pstar)));
                }
                ni.push(v);
                let yend = qmul(u0inv, pre);
                nseg.push([
                    rad(qmul(yend, segpre[ir][0])),
                    rad(qmul(yend, segpre[ir][1])),
                    rad(qmul(yend, segpre[ir][2])),
                ]);
                let b = qmul(yend, segq[ir]);
                base.push(b);
                rbase.push(rad(b));
                // chemin de retour (translaté), suffixes
                let rs = &gab.retour_slot[p];
                let mm_ = rs.len();
                let mut suf = vec![IDQ; mm_ + 1];
                for m in (0..mm_).rev() {
                    suf[m] = qmul(facteur(q[rs[m]], sg[m]), suf[m + 1]);
                }
                retinv0.push(qdag(suf[0]));
                let mut rr = Vec::with_capacity(mm_);
                for m in 0..mm_ {
                    let idx = if sg[m] > 0 { m } else { m + 1 };
                    rr.push(rad(qdag(suf[idx])));
                }
                rret.push(rr);
            }
        }
        (Ctx { r0, n0, ni, nseg, base, rbase, retinv0, rret }, liens)
    }
}

// ------------------------------------------------------------------------------------------
// Ligne de J
// ------------------------------------------------------------------------------------------

/// Ligne `c` de `J` : blocs 3×3 `J_{cℓ}` pour les liens fins du voisinage, et données de la passe arrière.
pub struct LigneJ {
    pub c: usize,
    /// identifiants globaux des liens (un par slot).
    pub liens: Vec<u32>,
    /// `J_{cℓ}` (repère d'origine), un par slot.
    pub j: Vec<M3>,
    lam: Vec<M3>,
    wbar: [f64; 3],
    lw: M3,
    e: M3,
    r0: M3,
    /// vrai si un logarithme de boucle est à moins de 1e-6 de la coupure (θ ≈ π) : évaluation refusée par `Jac::evaluer`.
    pub ambigu: bool,
}

fn evaluer_ligne(res: &Reseau, champ: &ChampHote, c: usize, reg: &Reglages) -> LigneJ {
    let (ctx, liens) = Ctx::nouveau(res, champ, c);
    let (_, mu) = res.coords_c(c);
    let gab = &res.gab[mu];
    let ns = gab.nb_slots;
    let np = res.table.nb_chemins();
    let mut lam = vec![ZERO; ns];
    let mut wbar = [0.0; 3];
    let mut lw = ZERO;
    let mut ambigu = false;
    let mut ai = vec![ZERO; np];
    let mut cj = vec![ZERO; np];
    for ir in 0..SITES_BLOC {
        let w = res.table.poids(ir);
        let (a, b) = (res.table.debut[ir], res.table.debut[ir + 1]);
        for p in a..b {
            ai[p] = ZERO;
            cj[p] = ZERO;
        }
        for i in a..b {
            for j in a..b {
                let (om, _, amb) = log_q(qmul(ctx.base[i], ctx.retinv0[j]));
                ambigu |= amb;
                for m in 0..3 {
                    wbar[m] += w * om[m];
                }
                #[allow(unused_mut)]
                let mut l = lg(om);
                #[cfg(test)]
                if reg.mutant == Mutant::LIdentite {
                    l = ID3;
                }
                let _ = reg;
                add(&mut ai[i], &l);
                add(&mut cj[j], &mm(&l, &ctx.rbase[i]));
            }
        }
        for i in a..b {
            add(&mut lw, &scale(&ai[i], w));
            let bi = scale(&ai[i], w);
            for k in 0..gab.chemin_slot[i].len() {
                axpy(&mut lam[gab.chemin_slot[i][k]], gab.chemin_signe[i][k] as f64, &mm(&bi, &ctx.ni[i][k]));
            }
            for k in 0..3 {
                add(&mut lam[gab.seg_slot[ir][k]], &mm(&bi, &ctx.nseg[i][k]));
            }
        }
        for j in a..b {
            let bj = scale(&cj[j], w);
            for m in 0..gab.retour_slot[j].len() {
                // pas de retour : signe du pas de la boucle = − signe d'origine
                axpy(&mut lam[gab.retour_slot[j][m]], -(gab.chemin_signe[j][m] as f64), &mm(&bj, &ctx.rret[j][m]));
            }
        }
    }
    for k in 0..3 {
        sub(&mut lam[gab.ref_slot[k]], &mm(&lw, &ctx.n0[k]));
    }
    let e = eg(wbar);
    let mut jt = vec![ZERO; ns];
    for s in 0..ns {
        jt[s] = mm(&e, &lam[s]);
    }
    for k in 0..3 {
        add(&mut jt[gab.ref_slot[k]], &ctx.n0[k]);
    }
    let j: Vec<M3> = jt.iter().map(|m| mm(&ctx.r0, m)).collect();
    LigneJ { c, liens, j, lam, wbar, lw, e, r0: ctx.r0, ambigu }
}

// ------------------------------------------------------------------------------------------
// Passe arrière : contribution d'une ligne au gradient de h
// ------------------------------------------------------------------------------------------

fn grad_ligne(res: &Reseau, champ: &ChampHote, lj: &LigneJ, w: &[M3], reg: &Reglages) -> Vec<[f64; 3]> {
    let (ctx, _) = Ctx::nouveau(res, champ, lj.c);
    let (_, mu) = res.coords_c(lj.c);
    let gab = &res.gab[mu];
    let ns = gab.nb_slots;
    let np = res.table.nb_chemins();
    #[cfg(test)]
    let (sans_e, sans_l, sans_r) = (
        matches!(reg.mutant, Mutant::SansDerivE | Mutant::SansDerivees),
        matches!(reg.mutant, Mutant::SansDerivL | Mutant::SansDerivees),
        matches!(reg.mutant, Mutant::SansDerivR | Mutant::SansDerivees),
    );
    #[cfg(not(test))]
    let (sans_e, sans_l, sans_r) = {
        let _ = reg;
        (false, false, false)
    };
    let r0 = &lj.r0;
    let e = &lj.e;
    let v: Vec<M3> = w.iter().map(|x| mtm(r0, x)).collect();
    let um: Vec<M3> = v.iter().map(|x| mtm(e, x)).collect();
    let mut psi = ZERO;
    for s in 0..ns {
        add(&mut psi, &mmt(&lj.lam[s], &v[s]));
    }
    let de = deg(lj.wbar);
    let ebar: [f64; 3] = if sans_e { [0.0; 3] } else { [tr_prod(&de[0], &psi), tr_prod(&de[1], &psi), tr_prod(&de[2], &psi)] };
    let mut f0 = ZERO;
    for k in 0..3 {
        add(&mut f0, &mmt(&ctx.n0[k], &um[gab.ref_slot[k]]));
    }
    let mut fi = vec![ZERO; np];
    let mut fs = vec![ZERO; np];
    let mut gj = vec![ZERO; np];
    for ir in 0..SITES_BLOC {
        for p in res.table.debut[ir]..res.table.debut[ir + 1] {
            for k in 0..gab.chemin_slot[p].len() {
                axpy(&mut fi[p], gab.chemin_signe[p][k] as f64, &mmt(&ctx.ni[p][k], &um[gab.chemin_slot[p][k]]));
            }
            for k in 0..3 {
                add(&mut fs[p], &mmt(&ctx.nseg[p][k], &um[gab.seg_slot[ir][k]]));
            }
            for m in 0..gab.retour_slot[p].len() {
                axpy(&mut gj[p], -(gab.chemin_signe[p][m] as f64), &mmt(&ctx.rret[p][m], &um[gab.retour_slot[p][m]]));
            }
        }
    }
    let mut g = vec![[0.0f64; 3]; ns];
    let mut ctot = [0.0f64; 3];
    for ir in 0..SITES_BLOC {
        let wt = res.table.poids(ir);
        let (a, b) = (res.table.debut[ir], res.table.debut[ir + 1]);
        for i in a..b {
            let rb = &ctx.rbase[i];
            let mut sai = ZERO;
            let mut cin = [0.0f64; 3];
            let mut fbase = fi[i];
            add(&mut fbase, &fs[i]);
            sub(&mut fbase, &f0);
            for j in a..b {
                let (om, _, _) = log_q(qmul(ctx.base[i], ctx.retinv0[j]));
                let l = lg(om);
                add(&mut sai, &l);
                let mut fij = fbase;
                add(&mut fij, &mm(rb, &gj[j]));
                let dv: [f64; 3] = if sans_l {
                    [0.0; 3]
                } else {
                    let dl = dlg(om);
                    [tr_prod(&dl[0], &fij), tr_prod(&dl[1], &fij), tr_prod(&dl[2], &fij)]
                };
                let cend = {
                    let mut c = mtv(&l, vadd(ebar, dv));
                    for x in c.iter_mut() {
                        *x *= wt;
                    }
                    c
                };
                let mut suf = cend;
                let mut tot = cend;
                let mj = gab.retour_slot[j].len();
                for m in 0..mj {
                    let sp = -(gab.chemin_signe[j][m] as f64);
                    let sl = gab.retour_slot[j][m];
                    let nmat = mm(rb, &ctx.rret[j][m]);
                    let x = mmt(&mtm(&l, &um[sl]), &nmat);
                    let mut cc = axial(&x);
                    let fac = if sans_r { 0.0 } else { 2.0 * wt * sp };
                    for t in cc.iter_mut() {
                        *t *= fac;
                    }
                    let mut cm = suf;
                    if sp < 0.0 {
                        cm = vadd(cm, cc);
                    }
                    let gv = mtv(&ctx.rret[j][m], mtv(rb, cm));
                    vaxpy(&mut g[sl], sp, gv);
                    suf = vadd(suf, cc);
                    tot = vadd(tot, cc);
                }
                cin = vadd(cin, tot);
            }
            // parties aller et segment (sommées sur j)
            let mut cs = [[0.0f64; 3]; 3];
            for k in 0..3 {
                let x = mmt(&mtm(&sai, &um[gab.seg_slot[ir][k]]), &ctx.nseg[i][k]);
                let mut cc = axial(&x);
                let fac = if sans_r { 0.0 } else { 2.0 * wt };
                for t in cc.iter_mut() {
                    *t *= fac;
                }
                cs[k] = cc;
            }
            let mut suf = cin;
            for k in 0..3 {
                suf = vadd(suf, cs[k]);
            }
            let mut total = suf;
            let lk = gab.chemin_slot[i].len();
            let mut cg = vec![[0.0f64; 3]; lk];
            for k in 0..lk {
                let x = mmt(&mtm(&sai, &um[gab.chemin_slot[i][k]]), &ctx.ni[i][k]);
                let mut cc = axial(&x);
                let fac = if sans_r { 0.0 } else { 2.0 * wt * gab.chemin_signe[i][k] as f64 };
                for t in cc.iter_mut() {
                    *t *= fac;
                }
                cg[k] = cc;
            }
            for k in (0..lk).rev() {
                let s = gab.chemin_signe[i][k] as f64;
                let mut cm = suf;
                if s < 0.0 {
                    cm = vadd(cm, cg[k]);
                }
                vaxpy(&mut g[gab.chemin_slot[i][k]], s, mtv(&ctx.ni[i][k], cm));
                suf = vadd(suf, cg[k]);
                total = vadd(total, cg[k]);
            }
            let mut sufs = cin;
            for k in (0..3).rev() {
                vaxpy(&mut g[gab.seg_slot[ir][k]], 1.0, mtv(&ctx.nseg[i][k], sufs));
                sufs = vadd(sufs, cs[k]);
            }
            ctot = vadd(ctot, total);
        }
    }
    // références
    let mut c0 = [[0.0f64; 3]; 3];
    for k in 0..3 {
        let s = gab.ref_slot[k];
        let xv = mmt(&v[s], &ctx.n0[k]);
        let xl = mmt(&mtm(&lj.lw, &um[s]), &ctx.n0[k]);
        let a1 = axial(&xv);
        let a2 = axial(&xl);
        let fac = if sans_r { 0.0 } else { 2.0 };
        c0[k] = [fac * (a1[0] - a2[0]), fac * (a1[1] - a2[1]), fac * (a1[2] - a2[2])];
        ctot = vadd(ctot, c0[k]);
    }
    for m0 in 0..3 {
        let mut s = [0.0f64; 3];
        for k0 in (m0 + 1)..3 {
            s = vadd(s, c0[k0]);
        }
        let d = [s[0] - ctot[0], s[1] - ctot[1], s[2] - ctot[2]];
        vaxpy(&mut g[gab.ref_slot[m0]], 1.0, mtv(&ctx.n0[m0], d));
    }
    let _ = ID3;
    g
}

// ------------------------------------------------------------------------------------------
// Jacobien complet
// ------------------------------------------------------------------------------------------

/// `J = DŪ` complet (une ligne par lien grossier) et index inverse lien fin → (ligne, slot).
pub struct Jac {
    pub n_c: usize,
    pub n_liens: usize,
    pub lignes: Vec<LigneJ>,
    /// pour chaque lien fin : (c, slot), triés par c croissant.
    pub par_lien: Vec<Vec<(u32, u32)>>,
}

impl Jac {
    /// Évalue `J` en `champ` ; les lignes sont calculées en parallèle dans `pool` (résultat indépendant du nombre de fils).
    pub fn evaluer(res: &Reseau, champ: &ChampHote, pool: &rayon::ThreadPool, reg: &Reglages) -> Resultat<Jac> {
        if champ.l != res.l {
            return Err(Erreur::Config(format!("jacobien : champ L = {} pour un réseau L = {}", champ.l, res.l)));
        }
        let lignes: Vec<LigneJ> = pool.install(|| (0..res.n_c).into_par_iter().map(|c| evaluer_ligne(res, champ, c, reg)).collect());
        if let Some(l) = lignes.iter().find(|l| l.ambigu) {
            return Err(Erreur::Validation(format!("jacobien : logarithme ambigu (θ à moins de 1e-6 de π) au lien grossier {}", l.c)));
        }
        let mut par_lien: Vec<Vec<(u32, u32)>> = vec![Vec::new(); res.n_liens];
        for lj in &lignes {
            for (s, &l) in lj.liens.iter().enumerate() {
                par_lien[l as usize].push((lj.c as u32, s as u32));
            }
        }
        Ok(Jac { n_c: res.n_c, n_liens: res.n_liens, lignes, par_lien })
    }

    /// `J v`, v de longueur 3N (ordre `3·lien + a`), résultat de longueur 3n (ordre `3·c + a`).
    pub fn jvp(&self, v: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; 3 * self.n_c];
        for lj in &self.lignes {
            let mut acc = [0.0; 3];
            for (s, &l) in lj.liens.iter().enumerate() {
                let x = mv(&lj.j[s], [v[3 * l as usize], v[3 * l as usize + 1], v[3 * l as usize + 2]]);
                acc = vadd(acc, x);
            }
            y[3 * lj.c..3 * lj.c + 3].copy_from_slice(&acc);
        }
        y
    }

    /// `Jᵀ u`, u de longueur 3n, résultat de longueur 3N.
    pub fn vjp(&self, u: &[f64]) -> Vec<f64> {
        let mut z = vec![0.0; 3 * self.n_liens];
        for lj in &self.lignes {
            let uc = [u[3 * lj.c], u[3 * lj.c + 1], u[3 * lj.c + 2]];
            for (s, &l) in lj.liens.iter().enumerate() {
                let x = mtv(&lj.j[s], uc);
                for a in 0..3 {
                    z[3 * l as usize + a] += x[a];
                }
            }
        }
        z
    }

    /// `G = J Jᵀ` par blocs : ligne c = liste triée de `(c', G_{cc'})` (blocs non nuls structurellement), symétrique.
    pub fn jjt_blocs(&self) -> Vec<Vec<(u32, M3)>> {
        let mut lignes: Vec<std::collections::BTreeMap<u32, M3>> = vec![Default::default(); self.n_c];
        for l in 0..self.n_liens {
            let t = &self.par_lien[l];
            for &(c1, s1) in t {
                let a = &self.lignes[c1 as usize].j[s1 as usize];
                for &(c2, s2) in t {
                    let b = &self.lignes[c2 as usize].j[s2 as usize];
                    let e = lignes[c1 as usize].entry(c2).or_insert(ZERO);
                    add(e, &mmt(a, b));
                }
            }
        }
        lignes.into_iter().map(|m| m.into_iter().collect()).collect()
    }

    /// `G` dense, ordre ligne, 3n × 3n.
    pub fn jjt_dense(&self) -> Vec<f64> {
        let n = 3 * self.n_c;
        let mut g = vec![0.0; n * n];
        for (c1, ligne) in self.jjt_blocs().iter().enumerate() {
            for (c2, b) in ligne {
                for i in 0..3 {
                    for j in 0..3 {
                        g[(3 * c1 + i) * n + 3 * (*c2 as usize) + j] = b[3 * i + j];
                    }
                }
            }
        }
        g
    }

    /// Blocs de `W = G⁻¹ J` aux emplacements de `J` à partir de `G⁻¹` dense (3n × 3n, ordre ligne) :
    /// `W_{cℓ} = Σ_{c'} (G⁻¹)_{cc'} J_{c'ℓ}`.
    pub fn w_depuis_ginv(&self, ginv: &[f64], pool: &rayon::ThreadPool) -> Vec<Vec<M3>> {
        let n = 3 * self.n_c;
        pool.install(|| {
            self.lignes
                .par_iter()
                .map(|lj| {
                    lj.liens
                        .iter()
                        .map(|&l| {
                            let mut acc = ZERO;
                            for &(c2, s2) in &self.par_lien[l as usize] {
                                let jb = &self.lignes[c2 as usize].j[s2 as usize];
                                let mut gb = ZERO;
                                for i in 0..3 {
                                    for j in 0..3 {
                                        gb[3 * i + j] = ginv[(3 * lj.c + i) * n + 3 * c2 as usize + j];
                                    }
                                }
                                add(&mut acc, &mm(&gb, jb));
                            }
                            acc
                        })
                        .collect()
                })
                .collect()
        })
    }

    /// Gradient de `½ log det(J Jᵀ)` par rapport aux perturbations gauches `ε_ℓ` (3N réels, `3·lien + a`), à `W = G⁻¹ J` donné.
    pub fn grad_demi_logdet(&self, res: &Reseau, champ: &ChampHote, w: &[Vec<M3>], pool: &rayon::ThreadPool, reg: &Reglages) -> Vec<f64> {
        let parts: Vec<Vec<[f64; 3]>> =
            pool.install(|| self.lignes.par_iter().zip(w.par_iter()).map(|(lj, wl)| grad_ligne(res, champ, lj, wl, reg)).collect());
        let mut g = vec![0.0; 3 * self.n_liens];
        for (lj, p) in self.lignes.iter().zip(&parts) {
            for (s, &l) in lj.liens.iter().enumerate() {
                for a in 0..3 {
                    g[3 * l as usize + a] += p[s][a];
                }
            }
        }
        g
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::heritage::su2_hb::qmul;
    use crate::rg::bloc::{exp_q, ubar_lien};
    use crate::su2::cpu::haar_q;

    pub(crate) fn pool(n: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap()
    }

    /// Champ « proche de l'identité » : exp(amp · gaussienne) (Philox), pour des angles de log modérés.
    pub(crate) fn champ_amp(l: usize, amp: f64, graine: u32) -> ChampHote {
        let mut q = Vec::with_capacity(16 * l.pow(4));
        for i in 0..4 * l.pow(4) {
            let h = haar_q(i as u32, 3, graine, 7, 11);
            // direction de Haar, angle réduit : exp(amp * h_vec/|h_vec| * facteur)
            let nv = (h[1] * h[1] + h[2] * h[2] + h[3] * h[3]).sqrt();
            let th = h[0].clamp(-1.0, 1.0).acos(); // angle ∈ [0, π]
            let w = amp * th / std::f64::consts::PI * 2.0;
            let u = exp_q([w * h[1] / nv, w * h[2] / nv, w * h[3] / nv]);
            q.extend_from_slice(&u);
        }
        ChampHote { l, q }
    }

    /// Perturbation gauche U_ℓ → exp(i s ε_ℓ·σ) U_ℓ.
    pub(crate) fn perturber(champ: &ChampHote, eps: &[f64], s: f64) -> ChampHote {
        let mut c = champ.clone();
        for i in 0..champ.q.len() / 4 {
            let e = exp_q([s * eps[3 * i], s * eps[3 * i + 1], s * eps[3 * i + 2]]);
            let u = qmul(e, champ.lien(i));
            c.q[4 * i..4 * i + 4].copy_from_slice(&u);
        }
        c
    }

    /// Ū de tous les liens grossiers (ordre c).
    pub(crate) fn ubar_tous(res: &Reseau, champ: &ChampHote) -> Vec<Q> {
        (0..res.n_c)
            .map(|c| {
                let (z, mu) = res.coords_c(c);
                ubar_lien(champ, &res.table, z, mu).0
            })
            .collect()
    }

    fn directions(n: usize, graine: u64) -> Vec<f64> {
        let mut s = graine;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let a = ((s >> 11) as f64) / ((1u64 << 53) as f64);
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let b = ((s >> 11) as f64) / ((1u64 << 53) as f64);
                (-2.0 * (a + 1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
            })
            .collect()
    }

    #[test]
    fn series_et_formes_fermees_se_raccordent() {
        // continuité à la bascule θ = SEUIL_SERIE (valeurs et dérivées)
        for &d in &[1e-9] {
            let (a, b) = (SEUIL_SERIE - d, SEUIL_SERIE + d);
            let (fa, ha, fpa, hpa) = coeffs_l(a);
            let (fb, hb, fpb, hpb) = coeffs_l(b);
            for (x, y) in [(fa, fb), (ha, hb), (fpa, fpb), (hpa, hpb)] {
                assert!((x - y).abs() < 1e-8, "{x} {y}");
            }
            let ea = coeffs_e(a);
            let eb = coeffs_e(b);
            for (x, y) in [(ea.0, eb.0), (ea.1, eb.1), (ea.2, eb.2), (ea.3, eb.3), (ea.4, eb.4), (ea.5, eb.5)] {
                assert!((x - y).abs() < 1e-8, "{x} {y}");
            }
        }
    }

    #[test]
    fn lg_eg_inverses_et_derivees_par_differences() {
        for w in [[0.0, 0.0, 0.0], [0.01, -0.02, 0.03], [0.2, 0.1, -0.1], [0.7, -0.4, 0.5], [1.5, 1.0, -1.2], [2.0, 1.4, 0.5]] {
            let l = lg(w);
            let e = eg(w);
            let p = mm(&l, &e);
            for k in 0..9 {
                assert!((p[k] - ID3[k]).abs() < 1e-12, "{w:?} {p:?}");
            }
            let dl = dlg(w);
            let de = deg(w);
            for a in 0..3 {
                let h = 1e-6;
                let (mut wp, mut wm) = (w, w);
                wp[a] += h;
                wm[a] -= h;
                for (f, d) in [(lg as fn([f64; 3]) -> M3, &dl[a]), (eg as fn([f64; 3]) -> M3, &de[a])] {
                    let (fp, fm) = (f(wp), f(wm));
                    for k in 0..9 {
                        let fd = (fp[k] - fm[k]) / (2.0 * h);
                        assert!((fd - d[k]).abs() < 1e-7 * (1.0 + d[k].abs()), "{w:?} a={a} k={k}: {fd} {}", d[k]);
                    }
                }
            }
        }
    }

    #[test]
    fn ubar_local_nul_a_froid_et_j_est_q() {
        let res = Reseau::nouveau(6).unwrap();
        let champ = ChampHote { l: 6, q: (0..4 * 6usize.pow(4)).flat_map(|_| IDQ).collect() };
        let jac = Jac::evaluer(&res, &champ, &pool(2), &Reglages::default()).unwrap();
        // à U = 1 : J = Q ⊗ 1₃ ; chaque bloc est un multiple de l'identité
        for lj in &jac.lignes {
            for b in &lj.j {
                assert!(b[1].abs() < 1e-13 && b[2].abs() < 1e-13 && b[3].abs() < 1e-13 && (b[0] - b[4]).abs() < 1e-13 && (b[4] - b[8]).abs() < 1e-13, "{b:?}");
            }
        }
    }

    #[test]
    fn jacobien_contre_richardson_l6() {
        let res = Reseau::nouveau(6).unwrap();
        let champ = champ_amp(6, 0.8, 5);
        let jac = Jac::evaluer(&res, &champ, &pool(4), &Reglages::default()).unwrap();
        let ub0 = ubar_tous(&res, &champ);
        let n = 3 * res.n_liens;
        let mut pire: f64 = 0.0;
        for t in 0..3 {
            let mut d = directions(n, 100 + t);
            let nrm = d.iter().map(|x| x * x).sum::<f64>().sqrt();
            d.iter_mut().for_each(|x| *x /= nrm);
            let jv = jac.jvp(&d);
            let dc = |h: f64| -> Vec<f64> {
                let (p, m) = (perturber(&champ, &d, h), perturber(&champ, &d, -h));
                let (up, um) = (ubar_tous(&res, &p), ubar_tous(&res, &m));
                let mut out = Vec::new();
                for c in 0..res.n_c {
                    let wp = log_q(qmul(up[c], qdag(ub0[c]))).0;
                    let wm = log_q(qmul(um[c], qdag(ub0[c]))).0;
                    for a in 0..3 {
                        out.push((wp[a] - wm[a]) / (2.0 * h));
                    }
                }
                out
            };
            let (a, b) = (dc(5e-4), dc(1e-3));
            let fd: Vec<f64> = a.iter().zip(&b).map(|(x, y)| (4.0 * x - y) / 3.0).collect();
            let num: f64 = fd.iter().zip(&jv).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
            let den: f64 = jv.iter().map(|x| x * x).sum::<f64>().sqrt();
            pire = pire.max(num / den);
        }
        assert!(pire < 1e-8, "écart relatif {pire:e}");
    }

    #[test]
    fn mutant_l_identite_est_rouge() {
        let res = Reseau::nouveau(6).unwrap();
        let champ = champ_amp(6, 0.8, 5);
        let reg = Reglages { mutant: Mutant::LIdentite };
        let jac = Jac::evaluer(&res, &champ, &pool(4), &reg).unwrap();
        let ub0 = ubar_tous(&res, &champ);
        let n = 3 * res.n_liens;
        let mut d = directions(n, 100);
        let nrm = d.iter().map(|x| x * x).sum::<f64>().sqrt();
        d.iter_mut().for_each(|x| *x /= nrm);
        let jv = jac.jvp(&d);
        let h = 1e-4;
        let (up, um) = (ubar_tous(&res, &perturber(&champ, &d, h)), ubar_tous(&res, &perturber(&champ, &d, -h)));
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for c in 0..res.n_c {
            let wp = log_q(qmul(up[c], qdag(ub0[c]))).0;
            let wm = log_q(qmul(um[c], qdag(ub0[c]))).0;
            for a in 0..3 {
                let fd = (wp[a] - wm[a]) / (2.0 * h);
                num += (fd - jv[3 * c + a]).powi(2);
                den += jv[3 * c + a].powi(2);
            }
        }
        assert!((num / den).sqrt() > 1e-3, "le mutant doit être rouge : {:e}", (num / den).sqrt());
    }

    /// Cholesky dense minimale (tests seulement) : renvoie G⁻¹ et log det G.
    pub(crate) fn inverse_dense(g: &[f64], n: usize) -> (Vec<f64>, f64) {
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = g[i * n + j];
                for k in 0..j {
                    s -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    assert!(s > 0.0, "G non définie positive");
                    l[i * n + i] = s.sqrt();
                } else {
                    l[i * n + j] = s / l[j * n + j];
                }
            }
        }
        let logdet = 2.0 * (0..n).map(|i| l[i * n + i].ln()).sum::<f64>();
        let mut inv = vec![0.0; n * n];
        for col in 0..n {
            let mut y = vec![0.0; n];
            for i in 0..n {
                let mut s = if i == col { 1.0 } else { 0.0 };
                for k in 0..i {
                    s -= l[i * n + k] * y[k];
                }
                y[i] = s / l[i * n + i];
            }
            for i in (0..n).rev() {
                let mut s = y[i];
                for k in i + 1..n {
                    s -= l[k * n + i] * inv[k * n + col];
                }
                inv[i * n + col] = s / l[i * n + i];
            }
        }
        (inv, logdet)
    }

    pub(crate) fn demi_logdet(res: &Reseau, champ: &ChampHote, p: &rayon::ThreadPool) -> f64 {
        let jac = Jac::evaluer(res, champ, p, &Reglages::default()).unwrap();
        let n = 3 * res.n_c;
        0.5 * inverse_dense(&jac.jjt_dense(), n).1
    }

    #[test]
    fn gradient_logdet_contre_richardson_l6() {
        let res = Reseau::nouveau(6).unwrap();
        let p = pool(4);
        let champ = champ_amp(6, 0.8, 5);
        let jac = Jac::evaluer(&res, &champ, &p, &Reglages::default()).unwrap();
        let n = 3 * res.n_c;
        let (ginv, _) = inverse_dense(&jac.jjt_dense(), n);
        let w = jac.w_depuis_ginv(&ginv, &p);
        let g = jac.grad_demi_logdet(&res, &champ, &w, &p, &Reglages::default());
        let nl = 3 * res.n_liens;
        let mut pire: f64 = 0.0;
        let mut dirs = Vec::new();
        for t in 0..3 {
            let mut d = directions(nl, 200 + t);
            let nrm = d.iter().map(|x| x * x).sum::<f64>().sqrt();
            d.iter_mut().for_each(|x| *x /= nrm);
            let dc = |h: f64| (demi_logdet(&res, &perturber(&champ, &d, h), &p) - demi_logdet(&res, &perturber(&champ, &d, -h), &p)) / (2.0 * h);
            let fd = (4.0 * dc(5e-4) - dc(1e-3)) / 3.0;
            let an: f64 = g.iter().zip(&d).map(|(a, b)| a * b).sum();
            pire = pire.max((an - fd).abs() / fd.abs());
            dirs.push((d, fd));
        }
        assert!(pire < 1e-6, "écart relatif {pire:e}");
        // mutants : chacun doit s'écarter de ≥ 1e-3 sur au moins une direction
        for m in [Mutant::SansDerivE, Mutant::SansDerivL, Mutant::SansDerivR] {
            let gm = jac.grad_demi_logdet(&res, &champ, &w, &p, &Reglages { mutant: m });
            let mut max: f64 = 0.0;
            for (d, fd) in &dirs {
                let an: f64 = gm.iter().zip(d).map(|(a, b)| a * b).sum();
                max = max.max((an - fd).abs() / fd.abs());
            }
            assert!(max > 1e-3, "mutant {m:?} non détecté : {max:e}");
        }
    }

    #[test]
    fn independance_au_nombre_de_fils() {
        let res = Reseau::nouveau(6).unwrap();
        let champ = champ_amp(6, 0.8, 9);
        let n = 3 * res.n_c;
        let mut sorties = Vec::new();
        for fils in [1usize, 3, 8] {
            let p = pool(fils);
            let jac = Jac::evaluer(&res, &champ, &p, &Reglages::default()).unwrap();
            let (ginv, ld) = inverse_dense(&jac.jjt_dense(), n);
            let w = jac.w_depuis_ginv(&ginv, &p);
            let g = jac.grad_demi_logdet(&res, &champ, &w, &p, &Reglages::default());
            let bits: Vec<u64> = g.iter().map(|x| x.to_bits()).chain(std::iter::once(ld.to_bits())).collect();
            sorties.push(bits);
        }
        assert!(sorties[0] == sorties[1] && sorties[0] == sorties[2]);
    }
}
