//! Tables compilées des géométries régulières C0/C1 ; les tables explicites C2 relèvent de B07.
//!
//! CONTRAT des consommateurs (CPU, noyaux GPU de B10, intrication de B07) — ne change qu'avec l'accord du chef d'orchestre :
//!
//! - Sites et liens : s = ((x0·ℓ1 + x1)·ℓ2 + x2)·ℓ3 + x3, `pas` = [ℓ1ℓ2ℓ3, ℓ2ℓ3, ℓ3, 1], lien li = μ·V + s (de x vers x + μ̂),
//!   voisins périodiques `voisin_haut` / `voisin_bas`.
//! - Action C1 : S = β Σ_{s, p} (1 − c(p, s) ½ Re Tr U_p(s)), où U_p(s) est la plaquette du plan p = (μ, ν), μ < ν, de COIN s
//!   (U_μ(s) U_ν(s+μ̂) U_μ(s+ν̂)† U_ν(s)†), p dans l'ordre `PLANS` = 01, 02, 03, 12, 13, 23, et c(p, s) = `coef[p·V + s]`
//!   (6V réels, décalage p·V ; 1 partout si `coef` = None). Le β GLOBAL n'est PAS dans `coef` : c = β_plan(p) × z(p, s).
//! - Agrafe du lien (μ, s) : pour chaque ν ≠ μ, plaquette « haute » de coin s, coefficient `coef[plan(μ,ν)·V + s]`, et
//!   plaquette « basse » de coin s − ν̂, coefficient `coef[plan(μ,ν)·V + voisin_bas(s, ν)]` (comme `geom::Pile` et
//!   `su2::torsion::agrafe_tordue`).
//! - Masque : 4V octets indexés par li ; 1 = lien gelé (ni bain thermique ni surrelaxation).
//! - Classes : `classes[k]` est une liste de SITES (indices s croissants). Garantie : pour une direction μ FIXÉE, les liens
//!   (μ, s), s dans une même classe, ne partagent aucune plaquette (deux sites voisins n'ont jamais la même couleur) et peuvent
//!   être mis à jour en parallèle. Les liens de directions différentes d'un même site NE sont PAS indépendants. Ordre d'un
//!   balayage : `for μ in 0..4 { for classe in classes { … } }`, barrière entre deux classes et entre deux directions (ordre
//!   de `Cpu64` à côtés pairs : parité 0 puis 1). `coloriage` = `Parite` (2 classes, tous côtés pairs) ou `Glouton` (au plus
//!   9 classes, côtés impairs : CPU seulement ; les noyaux GPU de B10 refusent les côtés impairs).
//! - C2 (plaquettes explicites, construites par l'intrication) : entrée d'une plaquette `e[k] = (li << 1) | dague`
//!   (dague = 1 : lien parcouru à rebours) ; incidences `cases` : 6 cases par lien li, case `6·li + 2·i(μ, ν) + bas` (i = rang
//!   de ν parmi les directions ≠ μ, bas = 0 plaquette haute, 1 plaquette basse), `MAXCASE` entrées `(id << 2) | position`
//!   (id = indice de la plaquette, position = rang du lien dans `e`), case vide = `NUL`. Capacités (vérifiées par
//!   `GeoSpec::valider` pour un bord recollé) : 4V ≤ `LIMITE_LIENS_C2` = 2³¹ ; majorant des plaquettes de l'UNION des
//!   deux géométries de l'interpolation, 2 × (6V + 3·s·ℓ1ℓ2ℓ3) ≤ `LIMITE_PLAQUETTES_C2` = 2³⁰ − 1 (aucune incidence ne vaut
//!   `NUL`) ; l'intrication contrôle en outre le nombre RÉEL de plaquettes avant l'encodage (B07).

use super::{Bord, Chemin, GeoA, GeoSpec, Pile, SourceMasque};
use crate::commun::{Erreur, Resultat};
use crate::su2::boite::{masque_boite, GeometrieBoite};

/// Nombre d'entrées d'une case d'incidence (C2) : 2 boucles de couture × 2 géométries.
pub const MAXCASE: usize = 4;
/// Case d'incidence vide (C2).
pub const NUL: u32 = u32::MAX;
/// Nombre maximal de liens (4V) d'un réseau C2 : `(li << 1) | dague` tient sur 32 bits.
pub const LIMITE_LIENS_C2: usize = 1 << 31;
/// Nombre maximal de plaquettes d'un réseau C2 : `(id << 2) | position` tient sur 32 bits sans atteindre `NUL`.
pub const LIMITE_PLAQUETTES_C2: usize = (1 << 30) - 1;

/// Plaquette explicite construite par l'intrication (B07) ; mêmes champs et mêmes dérivations que `intrication::Plaq`
/// (470ba63), pour qu'elle la remplace par `pub use crate::geom::Plaq;`. Encodage de `e` : voir le contrat du module.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plaq {
    pub e: [u32; 4],
    pub mu: u8,
    pub nu: u8,
    /// Poids dans la géométrie 0 (α = 0).
    pub w0: f64,
    /// Poids dans la géométrie 1 (α = 1).
    pub w1: f64,
}

/// Plaquettes et incidences explicites construites par l'intrication (B07) ; encodages : voir le contrat du module.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaquettesExplicites {
    pub plaqs: Vec<Plaq>,
    /// 6 cases par lien, `MAXCASE` entrées `(id << 2) | position` chacune, `NUL` si vide.
    pub cases: Vec<[u32; MAXCASE]>,
    /// Plaquettes dont w0 ≠ w1 (support de S₁ − S₀).
    pub diff: Vec<u32>,
}

/// Coloriage des sites : `Parite` (tous côtés pairs, 2 classes), `Glouton` (côtés impairs : désormais le coloriage à trois classes de `GeoA::couleurs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coloriage {
    Parite,
    Glouton,
}

/// Tables d'une géométrie C0 ou C1 ; contrat d'emploi : voir la documentation du module.
#[derive(Clone, Debug, PartialEq)]
pub struct Tables {
    /// Côtés STOCKÉS (ℓ0', ℓ1, ℓ2, ℓ3).
    pub cotes: [usize; 4],
    /// [ℓ1ℓ2ℓ3, ℓ2ℓ3, ℓ3, 1].
    pub pas: [usize; 4],
    pub v: usize,
    /// c(p, s) = β_plan(p) × z(p, s) de la plaquette de COIN s du plan p : `coef[p·V + s]` (6V) ; None ⇒ 1 partout.
    /// Le β global n'y est pas.
    pub coef: Option<Vec<f64>>,
    /// Un octet par lien li = μ·V + s : 1 = gelé, 0 = libre ; None ⇒ aucun lien gelé.
    pub masque: Option<Vec<u8>>,
    /// C2 seulement (None en C0/C1).
    pub explicite: Option<PlaquettesExplicites>,
    /// Classes de SITES ; indépendance garantie pour les liens d'une MÊME direction (voir le contrat du module).
    pub classes: Vec<Vec<usize>>,
    pub coloriage: Coloriage,
}

impl Tables {
    pub fn compiler(spec: &GeoSpec) -> Resultat<Tables> {
        spec.valider()?;
        if spec.chemin() == Chemin::C2 {
            return Err(Erreur::NonImplemente(
                "tables C2 : plaquettes explicites construites par l'intrication (B07)",
            ));
        }
        let cotes = spec.cotes_stockees();
        // Les produits sont bornés par la validation de 4·V.
        let geo_a = GeoA::nouvelle(cotes)?;
        let v = geo_a.v;
        let pas = [
            cotes[1] * cotes[2] * cotes[3],
            cotes[2] * cotes[3],
            cotes[3],
            1,
        ];
        let coef = if spec.torsion.est_triviale() && spec.beta_plans.is_none() {
            None
        } else {
            let mut z = Pile::depuis_torsion(&geo_a, &spec.torsion).z;
            if let Some(beta) = &spec.beta_plans {
                for (bloc, &b) in z.chunks_exact_mut(v).zip(beta) {
                    if b != 1.0 {
                        for facteur in bloc {
                            *facteur *= b;
                        }
                    }
                }
            }
            Some(z)
        };
        let mut masque = None;
        for (mu, bord) in spec.bords.iter().enumerate() {
            if !matches!(bord, Bord::Fixe { .. }) {
                continue;
            }
            let liens = masque.get_or_insert_with(|| vec![0u8; 4 * v]);
            for s in 0..v {
                if geo_a.coords(s)[mu] == 0 {
                    for (nu, bloc) in liens.chunks_exact_mut(v).enumerate() {
                        if nu != mu {
                            bloc[s] |= 1;
                        }
                    }
                }
            }
        }
        if let Some(SourceMasque::Boite { ell, .. }) = &spec.masque {
            let boite = masque_boite(&GeometrieBoite::nouvelle(*ell)?);
            if let Some(liens) = &mut masque {
                for (lien, gele) in liens.iter_mut().zip(boite) {
                    *lien |= gele;
                }
            } else {
                masque = Some(boite);
            }
        }
        Ok(Tables {
            cotes,
            pas,
            v,
            coef,
            masque,
            explicite: None,
            classes: geo_a.couleurs(),
            coloriage: if geo_a.tous_pairs() {
                Coloriage::Parite
            } else {
                Coloriage::Glouton
            },
        })
    }

    #[inline]
    pub fn coords(&self, s: usize) -> [usize; 4] {
        [
            s / self.pas[0],
            (s / self.pas[1]) % self.cotes[1],
            (s / self.pas[2]) % self.cotes[2],
            s % self.cotes[3],
        ]
    }

    #[inline]
    pub fn site(&self, x: &[usize; 4]) -> usize {
        x.iter().zip(self.pas).map(|(&xi, pas)| xi * pas).sum()
    }

    #[inline]
    pub fn voisin_haut(&self, s: usize, mu: usize) -> usize {
        if self.coords(s)[mu] == self.cotes[mu] - 1 {
            s - (self.cotes[mu] - 1) * self.pas[mu]
        } else {
            s + self.pas[mu]
        }
    }

    #[inline]
    pub fn voisin_bas(&self, s: usize, mu: usize) -> usize {
        if self.coords(s)[mu] == 0 {
            s + (self.cotes[mu] - 1) * self.pas[mu]
        } else {
            s - self.pas[mu]
        }
    }

    #[inline]
    pub fn lien(&self, mu: usize, s: usize) -> usize {
        mu * self.v + s
    }

    /// Pour chaque site : voisins hauts μ = 0..3, puis voisins bas μ = 0..3.
    pub fn voisins(&self) -> Vec<[u32; 8]> {
        (0..self.v)
            .map(|s| {
                std::array::from_fn(|i| {
                    if i < 4 {
                        self.voisin_haut(s, i) as u32
                    } else {
                        self.voisin_bas(s, i - 4) as u32
                    }
                })
            })
            .collect()
    }

    #[inline]
    pub fn coef(&self, plan: usize, s: usize) -> f64 {
        self.coef
            .as_ref()
            .map_or(1.0, |coef| coef[plan * self.v + s])
    }

    #[inline]
    pub fn gele(&self, li: usize) -> bool {
        self.masque.as_ref().is_some_and(|masque| masque[li] != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{Couture, RegionSpec, SourceBord, Torsion};

    #[test]
    fn replique_explicite_non_compilee() {
        let spec = GeoSpec::repliques(4, 6, 2, 1, 2);
        assert_eq!(spec.cotes_stockees(), [12, 4, 4, 4]);
        assert_eq!(spec.v(), 12 * 4 * 4 * 4);
        assert_eq!(Tables::compiler(&spec).unwrap_err().code_sortie(), 9);
        assert_eq!(
            spec.bords[0],
            Bord::Recolle {
                repliques: 2,
                region: RegionSpec::Tranche { axe: 1, largeur: 2 },
                couture: Couture::Symetrique,
            }
        );
    }

    #[test]
    fn hyperplan_fixe_et_refus_de_deux_bords_fixes() {
        let mut spec = GeoSpec::anisotrope([2, 3, 4, 5], 3);
        spec.bords[2] = Bord::Fixe {
            valeurs: SourceBord::ParChaine,
        };
        let tables = Tables::compiler(&spec).unwrap();
        for s in 0..tables.v {
            let x = tables.coords(s);
            for nu in 0..4 {
                assert_eq!(tables.gele(tables.lien(nu, s)), nu != 2 && x[2] == 0);
            }
        }
        // O6 : deux hyperplans fixes (valeurs indéfinies à leur intersection) sont refusés en v0.2.
        spec.bords[0] = Bord::Fixe {
            valeurs: SourceBord::Froid,
        };
        assert_eq!(Tables::compiler(&spec).unwrap_err().code_sortie(), 2);
    }

    #[test]
    fn coefficients_du_coin_anisotrope() {
        let spec = GeoSpec::anisotrope([2, 3, 4, 5], 0)
            .avec_torsion(Torsion::analyser("01,23").unwrap())
            .avec_beta_plans([1.0, 2.0, 0.5, 3.0, 0.25, 1.5]);
        let tables = Tables::compiler(&spec).unwrap();
        let beta = spec.beta_plans.unwrap();
        for (p, &(mu, nu)) in crate::geom::PLANS.iter().enumerate() {
            for s in 0..tables.v {
                let x = tables.coords(s);
                let coin = x[mu] == tables.cotes[mu] - 1 && x[nu] == tables.cotes[nu] - 1;
                let attendu = if coin && spec.torsion.tordu(mu, nu) {
                    -beta[p]
                } else {
                    beta[p]
                };
                assert_eq!(tables.coef(p, s), attendu);
            }
        }
    }
}
