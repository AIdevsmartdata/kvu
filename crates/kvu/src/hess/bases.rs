//! Bases, coordonnées et indexation de la Hessienne (WP13).
//!
//! Deux BASES de l'algèbre su(2) et deux dispositions d'indices :
//! - `ym4` : X_a = iσ_a/√2 (S1, `ym4_hess`), indice `3(μV + x) + a` ;
//! - `s2`  : X_a = iσ_a/2 (S2, `lat.py`, coordonnées `a`), indice `3(4x + μ) + a`.
//!
//! Relation établie par le calcul avant le gel de GATES v4-T2 (`scratch/pilote-t2/conv/conv_lat_ym4.py`) :
//! `H_ym4 = 2 H_s2` après la permutation `3(μV + x) + a ↔ 3(4x + μ) + a`.
//! `kvu hess` rend la Hessienne de S AVEC son β (pas H/β).
//!
//! Deux jeux de COORDONNÉES : `gauche` (U → exp(i a·σ/2) U), `additive` (U = exp(i A·σ/2), A → A + ζ,
//! convention de `lat.py::hess_su2_additive`).

use crate::commun::{Erreur, Resultat};
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Base {
    Ym4,
    S2,
}

impl Base {
    pub fn nom(self) -> &'static str {
        match self {
            Base::Ym4 => "ym4",
            Base::S2 => "s2",
        }
    }
    /// Facteur de la Hessienne de la base par rapport à la base ym4 : H_base = facteur · P H_ym4 Pᵀ.
    pub fn facteur_hessienne(self) -> f64 {
        match self {
            Base::Ym4 => 1.0,
            Base::S2 => 0.5,
        }
    }
}

impl FromStr for Base {
    type Err = Erreur;
    fn from_str(s: &str) -> Resultat<Self> {
        match s {
            "ym4" => Ok(Base::Ym4),
            "s2" => Ok(Base::S2),
            _ => Err(Erreur::Config(format!("base « {s} » inconnue (ym4 | s2)"))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coord {
    Gauche,
    Additive,
}

impl Coord {
    pub fn nom(self) -> &'static str {
        match self {
            Coord::Gauche => "gauche",
            Coord::Additive => "additive",
        }
    }
}

impl FromStr for Coord {
    type Err = Erreur;
    fn from_str(s: &str) -> Resultat<Self> {
        match s {
            "gauche" => Ok(Coord::Gauche),
            "additive" => Ok(Coord::Additive),
            _ => Err(Erreur::Config(format!("coordonnées « {s} » inconnues (gauche | additive)"))),
        }
    }
}

/// Réseau hypercubique périodique L⁴ et indexation. Le lien « s2 » `l = 4x + μ` est l'indice de lien de `lat.py` ;
/// le lien « champ » `e = μV + x` est celui de `ChampHote` et de `ym4_hess`.
#[derive(Clone, Debug)]
pub struct Reseau {
    pub l: usize,
    pub v: usize,
    /// Nombre de liens 4V.
    pub nl: usize,
    /// Nombre de degrés de liberté 12V.
    pub n: usize,
    /// `avant[μ][x]` = site x + μ̂.
    pub avant: [Vec<usize>; 4],
    /// `arriere[μ][x]` = site x − μ̂.
    pub arriere: [Vec<usize>; 4],
}

impl Reseau {
    pub fn nouveau(l: usize) -> Resultat<Reseau> {
        if l < 2 {
            return Err(Erreur::Config(format!("L = {l} : L ≥ 2 requis")));
        }
        let v = l.checked_pow(4).ok_or_else(|| Erreur::Config("L trop grand".into()))?;
        let mut avant: [Vec<usize>; 4] = Default::default();
        let mut arriere: [Vec<usize>; 4] = Default::default();
        for mu in 0..4 {
            avant[mu] = Vec::with_capacity(v);
            arriere[mu] = Vec::with_capacity(v);
            for x in 0..v {
                let c = Self::coordonnees_de(x, l);
                avant[mu].push(Self::site_de(Self::decaler(c, mu, 1, l), l));
                arriere[mu].push(Self::site_de(Self::decaler(c, mu, -1, l), l));
            }
        }
        Ok(Reseau { l, v, nl: 4 * v, n: 12 * v, avant, arriere })
    }

    pub fn coordonnees_de(mut s: usize, l: usize) -> [usize; 4] {
        let mut x = [0; 4];
        for m in (0..4).rev() {
            x[m] = s % l;
            s /= l;
        }
        x
    }

    pub fn site_de(x: [usize; 4], l: usize) -> usize {
        ((x[0] * l + x[1]) * l + x[2]) * l + x[3]
    }

    pub fn decaler(mut x: [usize; 4], mu: usize, delta: isize, l: usize) -> [usize; 4] {
        x[mu] = (x[mu] as isize + delta).rem_euclid(l as isize) as usize;
        x
    }

    /// Lien « champ » `e = μV + x` d'un lien « s2 » `l = 4x + μ`.
    pub fn champ_de_s2(&self, l: usize) -> usize {
        (l % 4) * self.v + l / 4
    }

    /// Lien « s2 » d'un lien « champ ».
    pub fn s2_de_champ(&self, e: usize) -> usize {
        4 * (e % self.v) + e / self.v
    }

    /// Vecteur ym4 (indice 3(μV+x)+a) → vecteur s2 (3(4x+μ)+a), sans changement d'échelle.
    pub fn ym4_vers_s2(&self, src: &[f64], dst: &mut [f64]) {
        for e in 0..self.nl {
            let l = self.s2_de_champ(e);
            dst[3 * l..3 * l + 3].copy_from_slice(&src[3 * e..3 * e + 3]);
        }
    }

    /// Vecteur s2 → vecteur ym4.
    pub fn s2_vers_ym4(&self, src: &[f64], dst: &mut [f64]) {
        for l in 0..self.nl {
            let e = self.champ_de_s2(l);
            dst[3 * e..3 * e + 3].copy_from_slice(&src[3 * l..3 * l + 3]);
        }
    }

    /// Plaquettes (x, μ<ν) dans l'ordre de `lat.Geo` (x croissant, puis les couples μ<ν) : liens « s2 »
    /// (x,μ), (x+μ̂,ν), (x+ν̂,μ), (x,ν), signes +,+,−,−.
    pub fn plaquettes_s2(&self) -> Vec<[usize; 4]> {
        let mut p = Vec::with_capacity(6 * self.v);
        for x in 0..self.v {
            for mu in 0..4 {
                for nu in mu + 1..4 {
                    p.push([4 * x + mu, 4 * self.avant[mu][x] + nu, 4 * self.avant[nu][x] + mu, 4 * x + nu]);
                }
            }
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_aller_retour() {
        let r = Reseau::nouveau(3).unwrap();
        for l in 0..r.nl {
            assert_eq!(r.s2_de_champ(r.champ_de_s2(l)), l);
        }
        let src: Vec<f64> = (0..r.n).map(|i| i as f64).collect();
        let mut a = vec![0.0; r.n];
        let mut b = vec![0.0; r.n];
        r.s2_vers_ym4(&src, &mut a);
        r.ym4_vers_s2(&a, &mut b);
        assert_eq!(src, b);
        // 3(μV + x) + a ↔ 3(4x + μ) + a
        let (mu, x) = (2, 5);
        assert_eq!(a[3 * (mu * r.v + x) + 1], src[3 * (4 * x + mu) + 1]);
    }

    #[test]
    fn voisins_periodiques() {
        let r = Reseau::nouveau(2).unwrap();
        for mu in 0..4 {
            for x in 0..r.v {
                assert_eq!(r.arriere[mu][r.avant[mu][x]], x);
            }
        }
        assert!(Reseau::nouveau(1).is_err());
    }

    #[test]
    fn plaquettes_comme_lat() {
        let r = Reseau::nouveau(2).unwrap();
        let p = r.plaquettes_s2();
        assert_eq!(p.len(), 6 * 16);
        assert_eq!(p[0], [0, 4 * r.avant[0][0] + 1, 4 * r.avant[1][0], 1]);
    }
}
