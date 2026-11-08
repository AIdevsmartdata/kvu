//! Paires (plaquette observée p, sonde ē), tranches de distance b et listes de paires par sonde (T3-D, SPEC_T3 §1.9, §4.3).
//!
//! Distance : r = |c_p − m_ē| à l'image minimale composante par composante, b = ⌊r + 1/2⌋ ; calcul en arithmétique ENTIÈRE
//! (centres doublés), donc aucune ambiguïté de bord. Les listes sont en CSR par sonde : pour la sonde `s`, les paires sont
//! `debut[s]..debut[s + 1]`, avec l'indice LOCAL de la plaquette (position dans `faces`) et la tranche.

use super::geom::{tranche_de_s, Geom};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Paires {
    /// Plaquettes observées (indices `pi` du réseau fin), ordre fixé.
    pub faces: Vec<u32>,
    /// Sondes (liens grossiers `ci`), ordre fixé.
    pub sondes: Vec<u32>,
    pub bmin: u32,
    pub bmax: u32,
    /// CSR par sonde.
    pub debut: Vec<u64>,
    pub face_loc: Vec<u32>,
    pub tranche: Vec<u8>,
    /// Effectifs de paires par tranche b = 0..=B_COMPTE (TOUTES les paires, fenêtre ou non).
    pub effectifs: Vec<u64>,
    /// Paires (indices q) dont la plaquette contient un lien de la ligne de la sonde : seules à avoir un terme E[X(O)] non nul.
    pub px: Vec<u32>,
}

/// (paires (face locale, tranche) de la fenêtre, effectifs par tranche) d'une sonde.
type ListeSonde = (Vec<(u32, u8)>, Vec<u64>);

/// Plus grande tranche comptée dans `effectifs`.
pub const B_COMPTE: usize = 64;

impl Paires {
    /// Liste des paires de la fenêtre bmin ⩽ b ⩽ bmax.
    pub fn nouvelles(g: &Geom, faces: Vec<u32>, sondes: Vec<u32>, bmin: u32, bmax: u32) -> Paires {
        let cp2: Vec<[i32; 4]> = faces.iter().map(|&pi| g.centre2_plaquette(pi as usize)).collect();
        let par_sonde: Vec<ListeSonde> = sondes
            .par_iter()
            .map(|&ci| {
                let me2 = g.milieu2_gros(ci as usize);
                let mut liste = Vec::new();
                let mut eff = vec![0u64; B_COMPTE + 1];
                for (fl, c) in cp2.iter().enumerate() {
                    let b = tranche_de_s(g.dist2_doublee(c, &me2));
                    if (b as usize) <= B_COMPTE {
                        eff[b as usize] += 1;
                    }
                    if b >= bmin && b <= bmax {
                        liste.push((fl as u32, b as u8));
                    }
                }
                (liste, eff)
            })
            .collect();
        let total: usize = par_sonde.iter().map(|p| p.0.len()).sum();
        let mut debut = Vec::with_capacity(sondes.len() + 1);
        let mut face_loc = Vec::with_capacity(total);
        let mut tranche = Vec::with_capacity(total);
        let mut effectifs = vec![0u64; B_COMPTE + 1];
        debut.push(0u64);
        for (liste, eff) in &par_sonde {
            for &(f, b) in liste {
                face_loc.push(f);
                tranche.push(b);
            }
            debut.push(face_loc.len() as u64);
            for (i, e) in eff.iter().enumerate() {
                effectifs[i] += e;
            }
        }
        // paires dont la plaquette contient un lien de la ligne (tranche ⩽ 1 : une plaquette qui contient un lien de la ligne est à r ⩽ 1,12)
        let mut px = Vec::new();
        for (s, &ci) in sondes.iter().enumerate() {
            let ligne = g.lignes[ci as usize];
            for q in debut[s] as usize..debut[s + 1] as usize {
                if tranche[q] > 1 {
                    continue;
                }
                let pi = faces[face_loc[q] as usize] as usize;
                let l = g.liens_plaquette(pi / g.v, pi % g.v);
                if l.iter().any(|x| ligne.contains(x)) {
                    px.push(q as u32);
                }
            }
        }
        Paires { faces, sondes, bmin, bmax, debut, face_loc, tranche, effectifs, px }
    }

    /// (plaquette pi, sonde ci) de la paire q.
    pub fn paire(&self, q: usize) -> (u32, u32) {
        let s = self.debut.partition_point(|&d| d <= q as u64) - 1;
        (self.faces[self.face_loc[q] as usize], self.sondes[s])
    }

    pub fn npaires(&self) -> usize {
        self.face_loc.len()
    }

    /// Nombre de paires de la fenêtre par tranche (b = 0..=bmax).
    pub fn effectifs_fenetre(&self) -> Vec<u64> {
        let mut e = vec![0u64; self.bmax as usize + 1];
        for &b in &self.tranche {
            e[b as usize] += 1;
        }
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effectifs_g05_g06() {
        // ORACLES G0.5 (L = 24) et G0.6 (L = 12), faces × sondes, tranches b = 1..
        let g12 = Geom::nouvelle(4, 12).unwrap();
        let p12 = Paires::nouvelles(&g12, g12.faces(), g12.sondes_paires(), 2, 6);
        let att12 = [(1usize, 1152u64), (2, 6528), (3, 21_120), (4, 57_984), (5, 103_296), (6, 160_896)];
        for (b, n) in att12 {
            assert_eq!(p12.effectifs[b], n, "L = 12, b = {b}");
        }
        let g24 = Geom::nouvelle(4, 24).unwrap();
        let p24 = Paires::nouvelles(&g24, g24.faces(), g24.sondes_paires(), 2, 10);
        let att24 = [
            (1usize, 18_432u64), (2, 104_448), (3, 337_920), (4, 927_744), (5, 1_652_736),
            (6, 2_758_656), (7, 4_859_904), (8, 6_862_848), (9, 9_480_192), (10, 13_694_976),
        ];
        for (b, n) in att24 {
            assert_eq!(p24.effectifs[b], n, "L = 24, b = {b}");
        }
        let fen = p24.effectifs_fenetre();
        for (b, n) in att24 {
            if b >= 2 {
                assert_eq!(fen[b], n);
            }
        }
        assert_eq!(p24.npaires() as u64, att24.iter().filter(|x| x.0 >= 2).map(|x| x.1).sum::<u64>());
    }
}
