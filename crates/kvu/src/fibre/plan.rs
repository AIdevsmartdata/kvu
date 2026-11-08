//! Plan de mesure : tables dérivées de (géométrie, poids) construites une fois par L et partagées par la
//! référence CPU et le chemin GPU (le GPU les téléverse telles quelles).
//!
//! - Wf transposée (CSR par plaquette) : pour chaque plaquette, les (ligne de Wf, poids) non nuls ;
//! - support par direction : colonnes de Wf atteintes par les plaquettes d'un lien de direction mu
//!   (≤ 27 pour pmax² = 1 : 3 plans × 9 fonctionnelles) ;
//! - plaquettes par lien (6 par lien), listes de liens libres et de paires de la fibre.

use super::poids::PoidsFourier;
use super::tables::{plaquettes_par_lien, PLANS};
use crate::commun::{Erreur, Resultat};
use crate::cond::GeometrieCond;

/// Nombre d'éléments de fibre par tranche : taille FIXE d'un partiel de D, donc le résultat ne dépend
/// ni du nombre de fils (CPU) ni de la géométrie de lancement (GPU).
pub const TRANCHE: usize = 512;

/// Mesure conditionnelle (fibre de dec : liens libres + paires) ou libre (tous les liens, métrique 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeFibre {
    Conditionnel,
    Libre,
}

/// Valeur de remplissage de `cmap` pour une ligne de Wf hors du support de la direction.
pub const HORS_SUPPORT: u8 = 255;

pub(crate) struct PlanFibre {
    pub l: usize,
    pub v: usize,
    pub nf: usize,
    /// Wf transposée en CSR par plaquette : (ligne de Wf, poids), lignes croissantes.
    pub wt_ptr: Vec<u32>,
    pub wt_idx: Vec<u8>,
    pub wt_val: Vec<f64>,
    /// Lignes de Wf du support de la direction mu, croissantes.
    pub sup: [Vec<u8>; 4],
    /// cmap[mu][r] = position de r dans sup[mu], ou HORS_SUPPORT.
    pub cmap: [Vec<u8>; 4],
    /// Plaquettes incidentes : lp[6 e + j] = 4 p + pos.
    pub lp: Vec<u32>,
    /// Liens libres (mu majeur), puis paires (e1, e2) de la fibre.
    pub lfree: Vec<u32>,
    pub e1: Vec<u32>,
    pub e2: Vec<u32>,
    /// Éléments libres et paires PAR direction (uniformes).
    pub nfree_mu: usize,
    pub npair_mu: usize,
}

impl PlanFibre {
    pub fn nouveau(geo: &GeometrieCond, w: &PoidsFourier) -> Resultat<PlanFibre> {
        let l = geo.l;
        let v = l * l * l * l;
        if geo.v != v {
            return Err(Erreur::Config(format!(
                "fibre : géométrie incohérente (V = {} pour L = {l})",
                geo.v
            )));
        }
        if w.nf == 0 || w.nf > 254 {
            return Err(Erreur::Config(format!(
                "fibre : {} lignes de Wf (1..=254 attendues)",
                w.nf
            )));
        }
        if w.ptr.len() != w.nf + 1 || w.idx.len() != w.val.len() || *w.ptr.last().unwrap() as usize != w.idx.len() {
            return Err(Erreur::Config("fibre : CSR de Wf incohérent".into()));
        }
        if w.idx.iter().any(|&c| c as usize >= 6 * v) {
            return Err(Erreur::Config(format!("fibre : colonne de Wf hors de 0..6V (V = {v})")));
        }
        if 54 * v > u32::MAX as usize {
            return Err(Erreur::Config("fibre : L trop grand pour les indices u32".into()));
        }
        // Transposition : comptage, puis remplissage par ligne croissante (⇒ lignes croissantes par plaquette).
        let mut wt_ptr = vec![0u32; 6 * v + 1];
        for &c in &w.idx {
            wt_ptr[c as usize + 1] += 1;
        }
        for p in 0..6 * v {
            wt_ptr[p + 1] += wt_ptr[p];
        }
        let nnz = w.idx.len();
        let mut suivant: Vec<u32> = wt_ptr[..6 * v].to_vec();
        let mut wt_idx = vec![0u8; nnz];
        let mut wt_val = vec![0f64; nnz];
        for r in 0..w.nf {
            for q in w.ptr[r] as usize..w.ptr[r + 1] as usize {
                let p = w.idx[q] as usize;
                let k = suivant[p] as usize;
                suivant[p] += 1;
                wt_idx[k] = r as u8;
                wt_val[k] = w.val[q];
            }
        }
        // Supports par direction.
        let mut sup: [Vec<u8>; 4] = Default::default();
        let mut cmap: [Vec<u8>; 4] = Default::default();
        for mu in 0..4 {
            let mut present = vec![false; w.nf];
            for (k, &(m, n)) in PLANS.iter().enumerate() {
                if m == mu || n == mu {
                    for p in k * v..(k + 1) * v {
                        for q in wt_ptr[p] as usize..wt_ptr[p + 1] as usize {
                            present[wt_idx[q] as usize] = true;
                        }
                    }
                }
            }
            let mut cm = vec![HORS_SUPPORT; w.nf];
            for (r, &pr) in present.iter().enumerate() {
                if pr {
                    cm[r] = sup[mu].len() as u8;
                    sup[mu].push(r as u8);
                }
            }
            cmap[mu] = cm;
        }
        let lp = plaquettes_par_lien(l)?;
        let nfree_mu = geo.fib_free.len() / 4;
        let npair_mu = geo.fib_e1.len() / 4;
        if nfree_mu * 4 != geo.fib_free.len()
            || npair_mu * 4 != geo.fib_e1.len()
            || geo.fib_e2.len() != geo.fib_e1.len()
        {
            return Err(Erreur::Config(
                "fibre : la fibre doit avoir autant d'éléments par direction".into(),
            ));
        }
        Ok(PlanFibre {
            l,
            v,
            nf: w.nf,
            wt_ptr,
            wt_idx,
            wt_val,
            sup,
            cmap,
            lp,
            lfree: geo.fib_free.clone(),
            e1: geo.fib_e1.clone(),
            e2: geo.fib_e2.clone(),
            nfree_mu,
            npair_mu,
        })
    }

    /// Éléments de fibre par direction : libres + paires (conditionnel) ou V liens (libre).
    pub fn n_elem_mu(&self, mode: ModeFibre) -> usize {
        match mode {
            ModeFibre::Conditionnel => self.nfree_mu + self.npair_mu,
            ModeFibre::Libre => self.v,
        }
    }

    /// Tranches par direction.
    pub fn tranches_mu(&self, mode: ModeFibre) -> usize {
        self.n_elem_mu(mode).div_ceil(TRANCHE)
    }

    /// Nombre total de tranches (4 directions).
    pub fn tranches(&self, mode: ModeFibre) -> usize {
        4 * self.tranches_mu(mode)
    }

    /// Plus grand support (colonnes d'une ligne creuse).
    pub fn support_max(&self) -> usize {
        self.sup.iter().map(Vec::len).max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::poids::poids_fourier;
    use super::*;
    use crate::cond::GeometrieCond;

    fn plan4() -> PlanFibre {
        PlanFibre::nouveau(&GeometrieCond::nouvelle(4).unwrap(), &poids_fourier(4, 1)).unwrap()
    }

    #[test]
    fn transposition_conserve_les_entrees() {
        let w = poids_fourier(4, 1);
        let p = plan4();
        assert_eq!(p.wt_idx.len(), w.idx.len());
        assert_eq!(*p.wt_ptr.last().unwrap() as usize, w.idx.len());
        let somme: f64 = w.val.iter().sum();
        let somme_t: f64 = p.wt_val.iter().sum();
        assert!((somme - somme_t).abs() < 1e-12);
        // lignes croissantes par plaquette
        for q in 0..6 * 256 {
            let s = &p.wt_idx[p.wt_ptr[q] as usize..p.wt_ptr[q + 1] as usize];
            assert!(s.windows(2).all(|x| x[0] < x[1]));
        }
    }

    #[test]
    fn supports_de_27_colonnes() {
        let p = plan4();
        for mu in 0..4 {
            assert_eq!(p.sup[mu].len(), 27, "3 plans × 9 fonctionnelles");
            assert!(p.sup[mu].windows(2).all(|x| x[0] < x[1]));
            for (i, &r) in p.sup[mu].iter().enumerate() {
                assert_eq!(p.cmap[mu][r as usize] as usize, i);
            }
            // un plan sans mu n'est pas dans le support
            for (k, &(m, n)) in PLANS.iter().enumerate() {
                let dedans = p.sup[mu].iter().any(|&r| r as usize / 9 == k);
                assert_eq!(dedans, m == mu || n == mu);
            }
        }
        assert_eq!(p.support_max(), 27);
    }

    #[test]
    fn tranches_et_comptes() {
        let p = plan4();
        assert_eq!(p.nfree_mu, 224);
        assert_eq!(p.npair_mu, 16);
        assert_eq!(p.n_elem_mu(ModeFibre::Conditionnel), 240);
        assert_eq!(p.n_elem_mu(ModeFibre::Libre), 256);
        assert_eq!(p.tranches(ModeFibre::Conditionnel), 4);
        assert_eq!(p.tranches(ModeFibre::Libre), 4);
    }

    #[test]
    fn csr_incoherent_refuse() {
        let g = GeometrieCond::nouvelle(4).unwrap();
        let mut w = poids_fourier(4, 1);
        w.idx[0] = 6 * 256;
        assert!(PlanFibre::nouveau(&g, &w).is_err());
        let mut vide = poids_fourier(4, 1);
        vide.nf = 0;
        assert!(PlanFibre::nouveau(&g, &vide).is_err());
    }
}
