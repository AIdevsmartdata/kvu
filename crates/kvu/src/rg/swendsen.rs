//! Témoin de Swendsen b = 2 et plaquette SU(2).

use super::bloc::{coordonnees, deplacer, lire, valider_champ, Q};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul};

fn produit4(a: Q, b: Q, c: Q, d: Q) -> Q {
    qmul(qmul(qmul(a, b), c), d)
}

pub fn bloquer_b2(champ: &ChampHote, c: f64) -> Resultat<ChampHote> {
    if champ.l < 2 || !champ.l.is_multiple_of(2) {
        return Err(Erreur::Config(format!(
            "L = {} : côté pair ≥ 2 requis pour b = 2",
            champ.l
        )));
    }
    if !c.is_finite() {
        return Err(Erreur::Config(
            "le coefficient de Swendsen doit être fini".into(),
        ));
    }
    valider_champ(champ)?;
    let lc = champ.l / 2;
    let vc = lc.pow(4);
    let mut q = vec![0.0; 16 * vc];
    for mu in 0..4 {
        for site in 0..vc {
            let y = coordonnees(site, lc).map(|v| 2 * v);
            let y_mu = deplacer(y, mu, 1, champ.l);
            let y_2mu = deplacer(y, mu, 2, champ.l);
            let mut s = qmul(lire(champ, y, mu), lire(champ, y_mu, mu));
            if c != 0.0 {
                for nu in 0..4 {
                    if nu == mu {
                        continue;
                    }
                    let y_nu = deplacer(y, nu, 1, champ.l);
                    let y_nu_mu = deplacer(y_nu, mu, 1, champ.l);
                    let fwd = produit4(
                        lire(champ, y, nu),
                        lire(champ, y_nu, mu),
                        lire(champ, y_nu_mu, mu),
                        qdag(lire(champ, y_2mu, nu)),
                    );
                    let y_mnu = deplacer(y, nu, -1, champ.l);
                    let y_mnu_mu = deplacer(y_mnu, mu, 1, champ.l);
                    let y_mnu_2mu = deplacer(y_mnu, mu, 2, champ.l);
                    let bwd = produit4(
                        qdag(lire(champ, y_mnu, nu)),
                        lire(champ, y_mnu, mu),
                        lire(champ, y_mnu_mu, mu),
                        lire(champ, y_mnu_2mu, nu),
                    );
                    for i in 0..4 {
                        s[i] += c * (fwd[i] + bwd[i]);
                    }
                }
            }
            let norme = s[0].hypot(s[1]).hypot(s[2]).hypot(s[3]);
            let u = if norme < 1e-300 {
                [1.0, 0.0, 0.0, 0.0]
            } else {
                s.map(|v| v / norme)
            };
            q[4 * (mu * vc + site)..4 * (mu * vc + site) + 4].copy_from_slice(&u);
        }
    }
    Ok(ChampHote { l: lc, q })
}

/// Moyenne de la demi-trace réelle sur les six plans et tous les sites.
pub fn plaquette_moyenne(champ: &ChampHote) -> f64 {
    let mut somme = 0.0;
    for s in 0..champ.v() {
        let x = coordonnees(s, champ.l);
        for mu in 0..4 {
            for nu in mu + 1..4 {
                let p = produit4(
                    lire(champ, x, mu),
                    lire(champ, deplacer(x, mu, 1, champ.l), nu),
                    qdag(lire(champ, deplacer(x, nu, 1, champ.l), mu)),
                    qdag(lire(champ, x, nu)),
                );
                somme += p[0];
            }
        }
    }
    somme / (6 * champ.v()) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rg::bloc::{
        site,
        tests::{aleatoire, froid, jauges, proche, transformer},
    };

    #[test]
    fn froid_et_plaquette() {
        let champ = froid(8);
        for c in [0.0, 0.1, 0.5] {
            assert_eq!(bloquer_b2(&champ, c).unwrap(), froid(4));
        }
        assert_eq!(plaquette_moyenne(&champ), 1.0);
        assert_eq!(bloquer_b2(&froid(2), 0.1).unwrap(), froid(1));
    }

    #[test]
    fn coefficient_nul_liens_droits() {
        let champ = aleatoire(8);
        let sortie = bloquer_b2(&champ, 0.0).unwrap();
        for mu in 0..4 {
            for s in 0..4_usize.pow(4) {
                let y = coordonnees(s, 4).map(|v| 2 * v);
                let u = qmul(lire(&champ, y, mu), lire(&champ, deplacer(y, mu, 1, 8), mu));
                proche(sortie.lien(mu * 4_usize.pow(4) + s), u, 1e-14);
            }
        }
    }

    #[test]
    fn covariance_l8_et_plaquette() {
        let champ = aleatoire(8);
        let g = jauges(8);
        let transforme = transformer(&champ, &g);
        assert!((plaquette_moyenne(&champ) - plaquette_moyenne(&transforme)).abs() < 1e-14);
        for c in [0.0, 0.1, -0.2] {
            let a = bloquer_b2(&champ, c).unwrap();
            let b = bloquer_b2(&transforme, c).unwrap();
            for mu in 0..4 {
                for s in 0..4_usize.pow(4) {
                    let y = coordonnees(s, 4).map(|v| 2 * v);
                    let yp = deplacer(y, mu, 2, 8);
                    let attendu = qmul(
                        qmul(g[site(y, 8)], a.lien(mu * 4_usize.pow(4) + s)),
                        qdag(g[site(yp, 8)]),
                    );
                    proche(b.lien(mu * 4_usize.pow(4) + s), attendu, 1e-13);
                }
            }
        }
    }

    #[test]
    fn dimensions_et_coefficient_invalides() {
        for l in [0, 1, 3, 9] {
            assert!(matches!(bloquer_b2(&froid(l), 0.1), Err(Erreur::Config(_))));
        }
        assert!(bloquer_b2(&froid(2), f64::NAN).is_err());
        let mut champ = froid(2);
        champ.q[5] = f64::INFINITY;
        assert!(bloquer_b2(&champ, 0.1).is_err());
    }
}
