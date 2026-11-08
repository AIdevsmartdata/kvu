//! Blocages de facteur 2 du script `n3_blocage_beta_eff.py` (référence de RG-N3), en quaternions SU(2).
//!
//! - `dec` : U′_μ(x′) = U_μ(2x′) U_μ(2x′ + μ) (décimation) ;
//! - `ape` : V_μ(x) = Proj[U_μ(x) + c Σ_ν agrafes†] (c = 0,5, toutes les liaisons lissées à partir de la configuration d'origine), puis
//!   U′_μ(x′) = V_μ(2x′) V_μ(2x′ + μ).
//!
//! ATTENTION : `ape` n'est PAS le témoin de Swendsen de WP6 (`rg::swendsen::bloquer_b2`, port de `block_config`), qui lisse le produit des
//! deux liens par six chemins de quatre liens ; les deux schémas donnent des configurations différentes (écart rapporté par RG-N3, INFO).

use super::sd::agrafes;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul};
use crate::rg::bloc::{coordonnees, deplacer, lire, valider_champ};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchemaScript {
    Dec,
    Ape,
}

/// Un pas de blocage de facteur 2 (L pair ≥ 2, ≥ 4 pour `ape`).
pub fn bloquer_script(champ: &ChampHote, schema: SchemaScript, c: f64) -> Resultat<ChampHote> {
    valider_champ(champ)?;
    let l = champ.l;
    if l < 2 || !l.is_multiple_of(2) || !c.is_finite() {
        return Err(Erreur::Config(format!("L = {l} (pair ≥ 2) ou c = {c} invalide")));
    }
    let v = l.pow(4);
    // V_μ(x) pour tous les liens (ape) ou U (dec)
    let lisse: Vec<[f64; 4]> = match schema {
        SchemaScript::Dec => champ.q.chunks_exact(4).map(|q| [q[0], q[1], q[2], q[3]]).collect(),
        SchemaScript::Ape => {
            if l < 3 {
                return Err(Erreur::Config("ape : L ≥ 4 requis".into()));
            }
            let mut w = Vec::with_capacity(4 * v);
            for mu in 0..4 {
                for s in 0..v {
                    let x = coordonnees(s, l);
                    let a = qdag(agrafes(champ, mu, x));
                    let u = champ.lien(mu * v + s);
                    let m: [f64; 4] = std::array::from_fn(|i| u[i] + c * a[i]);
                    let k = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2] + m[3] * m[3]).sqrt().max(1e-150);
                    w.push(std::array::from_fn(|i| m[i] / k));
                }
            }
            w
        }
    };
    let champ_lisse = ChampHote { l, q: lisse.iter().flatten().copied().collect() };
    let lc = l / 2;
    let vc = lc.pow(4);
    let mut q = vec![0.0; 16 * vc];
    for mu in 0..4 {
        for sc in 0..vc {
            let y = coordonnees(sc, lc).map(|t| 2 * t);
            let p = qmul(lire(&champ_lisse, y, mu), lire(&champ_lisse, deplacer(y, mu, 1, l), mu));
            let o = 4 * (mu * vc + sc);
            q[o..o + 4].copy_from_slice(&p);
        }
    }
    Ok(ChampHote { l: lc, q })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn champ_froid_reste_froid() {
        for s in [SchemaScript::Dec, SchemaScript::Ape] {
            let b = bloquer_script(&ChampHote::froid(4).unwrap(), s, 0.5).unwrap();
            assert_eq!(b.l, 2);
            assert!(b.q.chunks_exact(4).all(|q| (q[0] - 1.0).abs() < 1e-14 && q[1..].iter().all(|x| x.abs() < 1e-14)));
        }
    }
}
