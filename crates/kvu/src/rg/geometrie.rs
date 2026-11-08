//! Chemins locaux du bloc S1, ordonnés comme les permutations de Python.

use crate::commun::Erreur;
use std::str::FromStr;

pub const PAS: usize = 3;
pub const SITES_BLOC: usize = 81;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Geometrie {
    TroisCentre,
    TroisCoin,
}

impl Geometrie {
    pub fn nom(self) -> &'static str {
        match self {
            Self::TroisCentre => "3centre",
            Self::TroisCoin => "3coin",
        }
    }
}

impl FromStr for Geometrie {
    type Err = Erreur;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "3centre" => Ok(Self::TroisCentre),
            "3coin" => Ok(Self::TroisCoin),
            _ => Err(Erreur::Config(format!(
                "géométrie inconnue : {s} (3centre ou 3coin attendu)"
            ))),
        }
    }
}

/// Base du lien lu, relativement à l'origine du bloc, et sens de parcours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pas {
    pub dep: [i8; 4],
    pub mu: u8,
    pub signe: i8,
}

pub struct Table {
    pub geometrie: Geometrie,
    pub ref_site: [usize; 4],
    pub chemins: Vec<Vec<Pas>>,
    pub debut: Vec<usize>,
}

impl Table {
    pub fn nouvelle(g: Geometrie) -> Table {
        let ref_site = match g {
            Geometrie::TroisCentre => [1; 4],
            Geometrie::TroisCoin => [0; 4],
        };
        let mut table = Self {
            geometrie: g,
            ref_site,
            chemins: Vec::new(),
            debut: Vec::with_capacity(82),
        };
        for ir in 0..SITES_BLOC {
            table.debut.push(table.chemins.len());
            let r = Self::site_local(ir);
            let axes: Vec<usize> = (0..4).filter(|&m| r[m] != ref_site[m]).collect();
            // Choix successifs dans l'ordre des axes : ordre itertools.permutations.
            ordonner(&axes, &mut Vec::new(), ref_site, r, &mut table.chemins);
        }
        table.debut.push(table.chemins.len());
        table
    }

    pub fn nb_chemins(&self) -> usize {
        self.chemins.len()
    }

    pub fn nb_boucles(&self) -> usize {
        self.debut.windows(2).map(|d| (d[1] - d[0]).pow(2)).sum()
    }

    pub fn poids(&self, ir: usize) -> f64 {
        let n = self.debut[ir + 1] - self.debut[ir];
        1.0 / (81.0 * (n * n) as f64)
    }

    pub fn segment(&self, mu: usize, ir: usize) -> [Pas; 3] {
        let r = Self::site_local(ir);
        std::array::from_fn(|k| {
            let mut dep = r.map(|v| v as i8);
            dep[mu] += k as i8;
            Pas {
                dep,
                mu: mu as u8,
                signe: 1,
            }
        })
    }

    pub fn segment_droit(&self, mu: usize) -> [Pas; 3] {
        self.segment(mu, Self::ir_de(self.ref_site))
    }

    pub fn site_local(ir: usize) -> [usize; 4] {
        std::array::from_fn(|m| (ir / PAS.pow(m as u32)) % PAS)
    }

    pub fn ir_de(r: [usize; 4]) -> usize {
        r[0] + 3 * r[1] + 9 * r[2] + 27 * r[3]
    }

    pub fn nb_triviales(&self, mu: usize) -> usize {
        (0..SITES_BLOC)
            .filter(|&ir| {
                let r = Self::site_local(ir);
                (0..4).all(|m| {
                    if m == mu {
                        r[m].abs_diff(self.ref_site[m]) <= 1
                    } else {
                        r[m] == self.ref_site[m]
                    }
                })
            })
            .count()
    }
}

fn ordonner(
    axes: &[usize],
    ordre: &mut Vec<usize>,
    reference: [usize; 4],
    r: [usize; 4],
    sortie: &mut Vec<Vec<Pas>>,
) {
    if ordre.len() == axes.len() {
        let mut position = reference.map(|v| v as i8);
        let mut chemin = Vec::new();
        for &mu in ordre.iter() {
            let delta = r[mu] as i8 - position[mu];
            let signe = delta.signum();
            for _ in 0..delta.unsigned_abs() {
                if signe < 0 {
                    position[mu] -= 1;
                }
                chemin.push(Pas {
                    dep: position,
                    mu: mu as u8,
                    signe,
                });
                if signe > 0 {
                    position[mu] += 1;
                }
            }
        }
        sortie.push(chemin);
        return;
    }
    for &mu in axes {
        if !ordre.contains(&mu) {
            ordre.push(mu);
            ordonner(axes, ordre, reference, r, sortie);
            ordre.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comptages_centre() {
        let t = Table::nouvelle(Geometrie::TroisCentre);
        assert_eq!(t.nb_chemins(), 633);
        assert_eq!(t.nb_boucles(), 10_473);
        assert_eq!(t.debut.len(), 82);
        for mu in 0..4 {
            assert_eq!(t.nb_triviales(mu), 3);
        }
    }

    #[test]
    fn chemins_segments_et_permutations() {
        for g in [Geometrie::TroisCentre, Geometrie::TroisCoin] {
            let t = Table::nouvelle(g);
            assert_eq!(
                t.ref_site,
                if g == Geometrie::TroisCentre {
                    [1; 4]
                } else {
                    [0; 4]
                }
            );
            for ir in 0..81 {
                let r = Table::site_local(ir);
                assert_eq!(Table::ir_de(r), ir);
                let k = (0..4).filter(|&m| r[m] != t.ref_site[m]).count();
                let chemins = &t.chemins[t.debut[ir]..t.debut[ir + 1]];
                assert_eq!(chemins.len(), (1..=k).product::<usize>());
                assert_eq!(t.poids(ir), 1.0 / (81.0 * chemins.len().pow(2) as f64));
                for (i, p) in chemins.iter().enumerate() {
                    assert!(!chemins[..i].contains(p));
                    let mut x = t.ref_site.map(|v| v as i8);
                    let mut ordre = Vec::new();
                    for pas in p {
                        let mu = pas.mu as usize;
                        if ordre.last() != Some(&mu) {
                            ordre.push(mu);
                        }
                        if pas.signe < 0 {
                            x[mu] -= 1;
                        }
                        assert_eq!(pas.dep, x);
                        if pas.signe > 0 {
                            x[mu] += 1;
                        }
                    }
                    assert_eq!(x, r.map(|v| v as i8));
                    assert_eq!(ordre.len(), k);
                }
                let ordres: Vec<Vec<u8>> = chemins
                    .iter()
                    .map(|p| {
                        let mut o: Vec<u8> = p.iter().map(|s| s.mu).collect();
                        o.dedup();
                        o
                    })
                    .collect();
                assert!(ordres.windows(2).all(|w| w[0] < w[1]));
            }
            for mu in 0..4 {
                for (i, p) in t.segment_droit(mu).iter().enumerate() {
                    let mut x = t.ref_site.map(|v| v as i8);
                    x[mu] += i as i8;
                    assert_eq!(
                        *p,
                        Pas {
                            dep: x,
                            mu: mu as u8,
                            signe: 1
                        }
                    );
                }
                assert_eq!(t.segment(mu, 80)[2].dep[mu], 4);
            }
        }
    }

    #[test]
    fn noms() {
        for g in [Geometrie::TroisCentre, Geometrie::TroisCoin] {
            assert_eq!(g.nom().parse::<Geometrie>().unwrap(), g);
        }
        assert!("autre".parse::<Geometrie>().is_err());
    }
}
