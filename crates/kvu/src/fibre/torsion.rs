//! Lecture TORDUE de la saturation v3 (port de `torsion.py`, chantier brancheC_saturation_v3_20261004 ; copie de
//! référence : `reference/saturation_v3/torsion.py`). Propriétaire : T6 (correctif du vérificateur, 04/10 :
//! `cond chaine` ne produisait pas Ft et Dtw, donc S3 v3 ne pouvait pas migrer : torsion.py ne peut pas les
//! recalculer à partir de F.npy, D.npy et blk.npy).
//!
//! Famille 𝒜 de 18 fonctionnelles, dans l'ordre de torsion.py : S1 (6 lignes, 1/L² sur le plan p), T+ (6 lignes,
//! e^{+ψ(x)}/L²), T− (6 lignes, e^{−ψ(x)}/L²), avec ψ(x) = (c/L)·dist_tore(x, 0), c = [`C_TW`] = 1 (gelé).
//! - Ft (18) = 𝒜 · pv ;
//! - Dtw = (D0, Dm, Dp) (3 × 18 × 18) : Gram du gradient de fibre de 𝒜, non pondéré, pondéré par e^{−2ψ} et par
//!   e^{+2ψ} au site de la coordonnée de fibre (lien libre (μ, x) → x ; paire → site de e1).
//!
//! Calcul sur l'hôte (référence rayon), à partir des plaquettes et gradients du champ (calculés par le GPU s'il y en
//! a un, par la référence CPU sinon) : F par sommes compensées sur segments fixes, D par tranches fixes, donc
//! indépendants du nombre de fils.

use super::cpu::{f_neumaier, matrices_d_tordues};
use super::plan::PlanFibre;
use super::poids::PoidsFourier;
use crate::commun::{Erreur, Resultat};
use crate::cond::GeometrieCond;

/// Constante de torsion c (ψ = c/L · distance au site 0), gelée par le PLAN de la saturation v3.
pub const C_TW: f64 = 1.0;
/// Nombre de fonctionnelles de la famille tordue.
pub const NF_TW: usize = 18;

/// ψ(x) = (c/L) · (Σ_i min(x_i, L − x_i)²)^{1/2} pour chaque site (ordre C : s = ((x0 L + x1) L + x2) L + x3).
pub fn psi_sites(l: usize, c: f64) -> Vec<f64> {
    let v = l.pow(4);
    (0..v)
        .map(|s| {
            let x = [s / (l * l * l), (s / (l * l)) % l, (s / l) % l, s % l];
            let d2: f64 = x.iter().map(|&xi| (xi.min(l - xi) as f64).powi(2)).sum();
            (c / l as f64) * d2.sqrt()
        })
        .collect()
}

/// 𝒜 en CSR par ligne (18 × 6V) : [S1 p = 0..5, T+ p = 0..5, T− p = 0..5] (= `poids_tw` de torsion.py).
pub fn poids_tw(l: usize, c: f64) -> PoidsFourier {
    let v = l.pow(4);
    let psi = psi_sites(l, c);
    let l2 = (l * l) as f64;
    let mut ptr = vec![0u64];
    let mut idx = Vec::with_capacity(NF_TW * v);
    let mut val = Vec::with_capacity(NF_TW * v);
    for famille in 0..3 {
        for p in 0..6 {
            for (s, &ps) in psi.iter().enumerate() {
                let b = match famille {
                    0 => 1.0,
                    1 => ps.exp(),
                    _ => (-ps).exp(),
                };
                idx.push((p * v + s) as u32);
                val.push(b / l2);
            }
            ptr.push(idx.len() as u64);
        }
    }
    PoidsFourier { nf: NF_TW, ptr, idx, val }
}

/// Mesureur de la famille tordue pour un L donné (tables construites une fois).
pub struct MesureurTordu {
    plan: PlanFibre,
    w: PoidsFourier,
    psi: Vec<f64>,
    l: usize,
}

impl MesureurTordu {
    pub fn nouveau(geo: &GeometrieCond) -> Resultat<MesureurTordu> {
        let w = poids_tw(geo.l, C_TW);
        let plan = PlanFibre::nouveau(geo, &w)?;
        Ok(MesureurTordu { psi: psi_sites(geo.l, C_TW), plan, w, l: geo.l })
    }

    pub fn l(&self) -> usize {
        self.l
    }

    /// Ft (18) à partir des plaquettes pv (6V).
    pub fn f_tw(&self, pv: &[f64]) -> Resultat<Vec<f64>> {
        if pv.len() != 6 * self.plan.v {
            return Err(Erreur::Config(format!("torsion : pv de {} valeurs pour L = {}", pv.len(), self.l)));
        }
        Ok(f_neumaier(&self.w, pv))
    }

    /// Dtw = (D0, Dm, Dp), 3 × 18 × 18 ligne majeure, à partir du champ `q` (16V réels) et des gradients `gp`
    /// (6V × 4 × 3) de ce champ.
    pub fn d_tw(&self, q: &[f64], gp: &[f64]) -> Resultat<Vec<f64>> {
        let v = self.plan.v;
        if q.len() != 16 * v || gp.len() != 72 * v {
            return Err(Erreur::Config(format!(
                "torsion : champ de {} réels et gradients de {} pour L = {}",
                q.len(),
                gp.len(),
                self.l
            )));
        }
        Ok(matrices_d_tordues(&self.plan, q, gp, &self.psi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psi_comme_torsion_py() {
        let p = psi_sites(4, 1.0);
        assert_eq!(p[0], 0.0);
        // x = (0,0,0,1) : d = 1 ; x = (0,0,0,3) : min(3, 1) = 1 ; x = (2,2,2,2) : d = 4
        assert!((p[1] - 0.25).abs() < 1e-15);
        assert!((p[3] - 0.25).abs() < 1e-15);
        let s = ((2 * 4 + 2) * 4 + 2) * 4 + 2;
        assert!((p[s] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn famille_de_18_lignes_dans_l_ordre() {
        let w = poids_tw(4, 1.0);
        let v = 256;
        assert_eq!(w.nf, 18);
        assert_eq!(w.ptr.len(), 19);
        assert!(w.ptr.windows(2).all(|d| d[1] - d[0] == v as u64));
        // ligne 7 = T+ du plan 1 : colonne 1·V + s, poids e^{ψ}/L²
        let psi = psi_sites(4, 1.0);
        let k = w.ptr[7] as usize + 5;
        assert_eq!(w.idx[k] as usize, v + 5);
        assert!((w.val[k] - psi[5].exp() / 16.0).abs() < 1e-16);
        // ligne 0 = S1 du plan 0 : 1/L²
        assert_eq!(w.val[0], 1.0 / 16.0);
    }
}
