//! Tables d'indices des plaquettes et des liens (= `cb.Geometry.E`, `lp_plaq`, `lp_pos`).
//!
//! Ordre des plaquettes (`loops_plaquettes`) : plan (mu < nu) majeur, puis site. La plaquette p du
//! plan (m, n) au site x a pour liens, dans l'ordre : e0 = m·V + x, e1 = n·V + x+m, e2 = m·V + x+n,
//! e3 = n·V + x ; signes + + − −. Chaque lien appartient à exactement 6 plaquettes, listées dans
//! l'ordre de l'argsort STABLE de l'aplatissement de E (indice plaquette croissant).

use crate::commun::{Erreur, Resultat};

/// Les 6 plans (mu < nu), dans l'ordre de `cb.Geometry.planes`.
pub const PLANS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Pas de site selon la direction d (ordre C) : L^(3−d).
#[inline]
fn pas(l: usize, d: usize) -> usize {
    l.pow(3 - d as u32)
}

/// Voisin vers le haut du site s dans la direction d (conditions périodiques).
#[inline]
pub fn voisin_haut(l: usize, s: usize, d: usize) -> usize {
    let st = pas(l, d);
    let xd = (s / st) % l;
    if xd == l - 1 {
        s - (l - 1) * st
    } else {
        s + st
    }
}

/// Les 4 liens (e0..e3) de la plaquette `p` (indice dans 0..6V).
#[inline]
pub fn liens_plaquette(l: usize, p: usize) -> [u32; 4] {
    let v = l * l * l * l;
    let (m, n) = PLANS[p / v];
    let x = p % v;
    [
        (m * v + x) as u32,
        (n * v + voisin_haut(l, x, m)) as u32,
        (m * v + voisin_haut(l, x, n)) as u32,
        (n * v + x) as u32,
    ]
}

/// Plaquettes incidentes à chaque lien : `lp[6·e + j] = 4·p + pos` (p = plaquette, pos = position du lien
/// dans la plaquette), ordre = argsort stable de E aplati. 4V lignes de 6 entrées.
pub fn plaquettes_par_lien(l: usize) -> Resultat<Vec<u32>> {
    if l < 2 || !l.is_multiple_of(2) {
        return Err(Erreur::Config(format!("fibre : L = {l} : L pair ≥ 2 requis")));
    }
    let v = l * l * l * l;
    if 24 * v > u32::MAX as usize {
        return Err(Erreur::Config(format!(
            "fibre : L = {l} trop grand pour des indices u32"
        )));
    }
    let mut remplis = vec![0u8; 4 * v];
    let mut lp = vec![0u32; 4 * v * 6];
    for p in 0..6 * v {
        let e = liens_plaquette(l, p);
        for (pos, &lien) in e.iter().enumerate() {
            let lien = lien as usize;
            let j = remplis[lien] as usize;
            lp[6 * lien + j] = (4 * p + pos) as u32;
            remplis[lien] += 1;
        }
    }
    if remplis.iter().any(|&c| c != 6) {
        return Err(Erreur::Config(
            "fibre : chaque lien devrait appartenir à 6 plaquettes".into(),
        ));
    }
    Ok(lp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_plaquettes_par_lien_et_ordre_croissant() {
        let l = 4;
        let lp = plaquettes_par_lien(l).unwrap();
        let v = 256;
        assert_eq!(lp.len(), 4 * v * 6);
        for e in 0..4 * v {
            let s = &lp[6 * e..6 * e + 6];
            assert!(s.windows(2).all(|w| w[0] < w[1]), "ordre de l'argsort stable");
            for &x in s {
                let (p, pos) = ((x / 4) as usize, (x % 4) as usize);
                assert_eq!(liens_plaquette(l, p)[pos] as usize, e);
            }
        }
    }

    #[test]
    fn plaquette_000_liens_connus() {
        // L = 4, plan (0,1), site 0 : e0 = lien 0 du site 0 ; e1 = lien 1 de x0+1 ; e2 = lien 0 de x1+1 ; e3 = lien 1 du site 0.
        let e = liens_plaquette(4, 0);
        assert_eq!(e, [0, 256 + 64, 16, 256]);
    }

    #[test]
    fn voisin_periodique() {
        let l = 4;
        // site (3,0,0,0) = 3·64 ; voisin vers le haut en d = 0 : (0,0,0,0).
        assert_eq!(voisin_haut(l, 3 * 64, 0), 0);
        assert_eq!(voisin_haut(l, 3, 3), 0);
        assert_eq!(voisin_haut(l, 0, 3), 1);
    }

    #[test]
    fn l_impair_refuse() {
        assert!(plaquettes_par_lien(3).is_err());
    }
}
