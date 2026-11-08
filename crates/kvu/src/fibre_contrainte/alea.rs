//! Aléa à compteur (Philox4x32-10) de la fibre contrainte : un tirage est une fonction pure de (graine, a, b, c, étiquette),
//! donc indépendante du nombre de fils et de l'ordre d'exécution (FIB-DET, FIB-FILS).
//!
//! Convention des compteurs : `a` = numéro de chaîne, `b` = numéro de trajectoire (route B) ou de balayage (route A),
//! `c` = indice de tirage dans la trajectoire/le balayage (indice de lien, de composante, d'essai…), `tag` = usage.

use crate::commun::Graine;
use crate::heritage::su2_hb::{philox4x32_10, u01};

/// Impulsions initiales N(0, M) de la route B (composante `c`).
pub const TAG_IMPULSION: u32 = 0x4642_0001;
/// Tirage d'acceptation de Metropolis (une valeur par trajectoire ou par proposition).
pub const TAG_ACCEPTATION: u32 = 0x4642_0002;
/// Proposition locale de la route A (3 gaussiennes par proposition).
pub const TAG_PROPOSITION: u32 = 0x4642_0003;
/// Configuration de départ (jouets).
pub const TAG_DEPART: u32 = 0x4642_0004;
/// Réglages et réplicas des outils de statistique (tests).
pub const TAG_OUTIL: u32 = 0x4642_0005;

/// Flux d'aléa lié à une graine 64 bits (clé Philox = (graine basse, graine haute), comme `Graine::cle`).
#[derive(Clone, Copy, Debug)]
pub struct Flux {
    k0: u32,
    k1: u32,
}

impl Flux {
    pub fn nouveau(graine: u64) -> Flux {
        let (k0, k1) = Graine(graine).cle();
        Flux { k0, k1 }
    }

    /// Les quatre mots Philox du compteur `[a, b, c, tag]`.
    #[inline]
    pub fn mots(&self, a: u32, b: u32, c: u32, tag: u32) -> [u32; 4] {
        philox4x32_10([a, b, c, tag], self.k0, self.k1)
    }

    /// Uniforme dans ]0, 1[ (52 bits).
    #[inline]
    pub fn uniforme(&self, a: u32, b: u32, c: u32, tag: u32) -> f64 {
        let r = self.mots(a, b, c, tag);
        u01(r[0], r[1])
    }

    /// Gaussienne N(0, 1) (Box–Muller sur les deux paires de mots du compteur).
    #[inline]
    pub fn gauss(&self, a: u32, b: u32, c: u32, tag: u32) -> f64 {
        let r = self.mots(a, b, c, tag);
        let (u1, u2) = (u01(r[0], r[1]), u01(r[2], r[3]));
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tirages_purs_et_distincts() {
        let f = Flux::nouveau(26_102_612_001);
        assert_eq!(f.gauss(1, 2, 3, TAG_IMPULSION).to_bits(), f.gauss(1, 2, 3, TAG_IMPULSION).to_bits());
        assert_ne!(f.gauss(1, 2, 3, TAG_IMPULSION).to_bits(), f.gauss(1, 2, 4, TAG_IMPULSION).to_bits());
        assert_ne!(f.uniforme(1, 2, 3, TAG_ACCEPTATION).to_bits(), f.uniforme(1, 2, 3, TAG_PROPOSITION).to_bits());
        let g = Flux::nouveau(26_102_612_002);
        assert_ne!(f.uniforme(0, 0, 0, TAG_DEPART).to_bits(), g.uniforme(0, 0, 0, TAG_DEPART).to_bits());
    }

    #[test]
    fn moments_gaussiens() {
        let f = Flux::nouveau(7);
        let n = 200_000;
        let (mut s1, mut s2) = (0.0, 0.0);
        for i in 0..n {
            let x = f.gauss(0, 0, i, TAG_OUTIL);
            s1 += x;
            s2 += x * x;
        }
        let (m, v) = (s1 / n as f64, s2 / n as f64);
        assert!(m.abs() < 0.01 && (v - 1.0).abs() < 0.02, "{m} {v}");
    }
}
