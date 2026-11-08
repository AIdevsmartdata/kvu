//! Géométrie de la décimation (= cb.Geometry / lib_brancheC.masks_dec). Propriétaire : T5.
//! Utilisée aussi par la mesure de fibre (T6) : les ORDRES ci-dessous sont gelés.
//!
//! Décimation : pour chaque direction μ, une « paire » est (x, x+μ) avec x_ν pair pour tout ν ≠ μ et
//! x_μ pair ; son produit U_μ(x) U_μ(x+μ) = V est figé. Les liens de direction μ dont les coordonnées
//! transverses ne sont pas toutes paires sont « libres » ; ceux de la seconde moitié des paires
//! (x_μ impair, transverses paires) sont les « seconds liens » (pair2), exclus des libres.

use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qmul, Q};

/// Tables d'indices d'un réseau L⁴ (L pair ≥ 4), sites aplatis en ordre C (s = ((x0 L + x1) L + x2) L + x3),
/// liens li = mu·V + s. Tous les ordres sont ceux de numpy `flatnonzero` (croissant en s).
#[derive(Clone, Debug, PartialEq)]
pub struct GeometrieCond {
    pub l: usize,
    pub v: usize,
    /// p1[mu] : sites x de pair1 (x_ν pair ∀ν≠μ et x_μ pair), ordre croissant ; npair = V/16 par mu.
    pub p1: [Vec<u32>; 4],
    /// p1pm[mu][i] = x + e_mu pour x = p1[mu][i].
    pub p1pm: [Vec<u32>; 4],
    /// fr[mu][p] : sites des liens libres de direction mu et de parité p (Σx mod 2), ordre croissant.
    pub fr: [[Vec<u32>; 2]; 4],
    /// free_all[mu] : tous les sites libres de direction mu, ordre croissant.
    pub free_all: [Vec<u32>; 4],
    /// Fibre (ordre de fiber_map) : liens libres mu·V + s (mu majeur), puis paires (e1, e2).
    pub fib_free: Vec<u32>,
    pub fib_e1: Vec<u32>,
    pub fib_e2: Vec<u32>,
    /// Dimension de la fibre : 3·|fib_free| + 3·|fib_e1|.
    pub n_fib: usize,
}

/// Coordonnées (x0, x1, x2, x3) du site `s`.
#[inline]
fn coords(s: usize, l: usize) -> [usize; 4] {
    [s / (l * l * l), (s / (l * l)) % l, (s / l) % l, s % l]
}

/// Voisin x + e_d (périodique) du site `s`.
#[inline]
fn voisin_haut(s: usize, d: usize, l: usize) -> usize {
    let pas = [l * l * l, l * l, l, 1][d];
    if coords(s, l)[d] == l - 1 {
        s - (l - 1) * pas
    } else {
        s + pas
    }
}

impl GeometrieCond {
    /// = cb.Geometry(L) (partie utile au bain conditionnel et à la fibre). L pair ≥ 4.
    pub fn nouvelle(l: usize) -> Resultat<GeometrieCond> {
        if l < 4 || !l.is_multiple_of(2) {
            return Err(Erreur::Config(format!("GeometrieCond : L = {l} (L pair ≥ 4 requis)")));
        }
        if l > 80 {
            // les indices de lien mu·V + s doivent tenir dans u32 : 4·L⁴ < 2³²
            return Err(Erreur::Config(format!("GeometrieCond : L = {l} > 80 (indices de lien hors de u32)")));
        }
        let v = l * l * l * l;
        let mut p1: [Vec<u32>; 4] = Default::default();
        let mut p1pm: [Vec<u32>; 4] = Default::default();
        let mut fr: [[Vec<u32>; 2]; 4] = Default::default();
        let mut free_all: [Vec<u32>; 4] = Default::default();
        for s in 0..v {
            let x = coords(s, l);
            let parite = (x[0] + x[1] + x[2] + x[3]) % 2;
            for mu in 0..4 {
                let transverses_paires = (0..4).filter(|&nu| nu != mu).all(|nu| x[nu].is_multiple_of(2));
                if transverses_paires {
                    if x[mu].is_multiple_of(2) {
                        p1[mu].push(s as u32);
                        p1pm[mu].push(voisin_haut(s, mu, l) as u32);
                    }
                } else {
                    free_all[mu].push(s as u32);
                    fr[mu][parite].push(s as u32);
                }
            }
        }
        let mut fib_free = Vec::new();
        let mut fib_e1 = Vec::new();
        let mut fib_e2 = Vec::new();
        for mu in 0..4 {
            let base = (mu * v) as u32;
            fib_free.extend(free_all[mu].iter().map(|&s| base + s));
            fib_e1.extend(p1[mu].iter().map(|&s| base + s));
            fib_e2.extend(p1pm[mu].iter().map(|&s| base + s));
        }
        let n_fib = 3 * fib_free.len() + 3 * fib_e1.len();
        Ok(GeometrieCond { l, v, p1, p1pm, fr, free_all, fib_free, fib_e1, fib_e2, n_fib })
    }

    /// Nombre de paires par direction (V/16).
    pub fn npair(&self) -> usize {
        self.p1[0].len()
    }
}

/// V figé : (4 × npair) quaternions U_mu(x) U_mu(x+mu) aux x de p1 (= cb.V_pairs), index mu·npair + i.
pub fn v_paires(c: &ChampHote, g: &GeometrieCond) -> Vec<[f64; 4]> {
    let np = g.npair();
    let mut out: Vec<Q> = Vec::with_capacity(4 * np);
    for mu in 0..4 {
        for i in 0..np {
            let a = c.lien(mu * g.v + g.p1[mu][i] as usize);
            let b = c.lien(mu * g.v + g.p1pm[mu][i] as usize);
            out.push(qmul(a, b));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effectifs_et_dimension_de_fibre() {
        for l in [4usize, 6, 8] {
            let g = GeometrieCond::nouvelle(l).unwrap();
            let v = l.pow(4);
            assert_eq!(g.npair(), v / 16);
            for mu in 0..4 {
                assert_eq!(g.p1[mu].len(), v / 16);
                assert_eq!(g.p1pm[mu].len(), v / 16);
                // libres = tous les liens de direction mu sauf les 2 liens de chaque paire
                assert_eq!(g.free_all[mu].len(), v - 2 * (v / 16));
                assert_eq!(g.fr[mu][0].len() + g.fr[mu][1].len(), g.free_all[mu].len());
            }
            assert_eq!(g.fib_free.len(), 4 * (v - v / 8));
            assert_eq!(g.n_fib, 3 * g.fib_free.len() + 3 * g.fib_e1.len());
            // dimension de la fibre = 3 × (4V − 4·V/16) = 12V − 3V/4
            assert_eq!(g.n_fib, 3 * (4 * v - 4 * (v / 16)));
        }
    }

    #[test]
    fn ordres_croissants_et_partition_des_liens() {
        let g = GeometrieCond::nouvelle(6).unwrap();
        let v = g.v;
        for mu in 0..4 {
            assert!(g.p1[mu].windows(2).all(|w| w[0] < w[1]));
            assert!(g.free_all[mu].windows(2).all(|w| w[0] < w[1]));
            for p in 0..2 {
                assert!(g.fr[mu][p].windows(2).all(|w| w[0] < w[1]));
            }
        }
        // chaque lien est exactement : premier lien de paire, second lien de paire ou libre
        let mut vu = vec![0u8; 4 * v];
        for mu in 0..4 {
            for i in 0..g.npair() {
                vu[mu * v + g.p1[mu][i] as usize] += 1;
                vu[mu * v + g.p1pm[mu][i] as usize] += 1;
            }
            for &s in &g.free_all[mu] {
                vu[mu * v + s as usize] += 1;
            }
        }
        assert!(vu.iter().all(|&c| c == 1));
    }

    #[test]
    fn parite_des_liens_libres() {
        let g = GeometrieCond::nouvelle(4).unwrap();
        for mu in 0..4 {
            for p in 0..2 {
                for &s in &g.fr[mu][p] {
                    let x = coords(s as usize, 4);
                    assert_eq!((x[0] + x[1] + x[2] + x[3]) % 2, p);
                }
            }
        }
    }

    #[test]
    fn voisin_periodique() {
        assert_eq!(voisin_haut(3, 3, 4), 0);
        assert_eq!(voisin_haut(0, 3, 4), 1);
        assert_eq!(voisin_haut(3 * 64, 0, 4), 0);
        assert_eq!(voisin_haut(0, 0, 4), 64);
    }

    #[test]
    fn l_invalide_refuse() {
        assert!(GeometrieCond::nouvelle(2).is_err());
        assert!(GeometrieCond::nouvelle(5).is_err());
        assert!(GeometrieCond::nouvelle(82).is_err());
    }

    #[test]
    fn v_paires_du_champ_froid_est_l_identite() {
        let g = GeometrieCond::nouvelle(4).unwrap();
        let c = ChampHote::froid(4).unwrap();
        let v = v_paires(&c, &g);
        assert_eq!(v.len(), 4 * g.npair());
        assert!(v.iter().all(|q| *q == [1.0, 0.0, 0.0, 0.0]));
    }
}
