//! Plaquette 1×1 et rectangle 1×2, avec sommation séquentielle déterministe.

use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul};

pub type Q = [f64; 4];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observables {
    pub plaquette: f64,
    pub rectangle: f64,
}

/// Moyennes des traces réduites sur les six plans et douze rectangles orientés.
pub fn plaquette_et_rectangle(champ: &ChampHote) -> Resultat<Observables> {
    if champ.l < 3 {
        return Err(Erreur::Config("Les observables MCRG exigent L ≥ 3".into()));
    }
    let v = champ
        .l
        .checked_pow(4)
        .ok_or_else(|| Erreur::Config("Volume L⁴ trop grand".into()))?;
    let longueur = v
        .checked_mul(16)
        .ok_or_else(|| Erreur::Config("Nombre de composantes trop grand".into()))?;
    if champ.q.len() != longueur {
        return Err(Erreur::Config(format!(
            "Champ de {} composantes, {longueur} attendues",
            champ.q.len()
        )));
    }
    if champ.q.iter().any(|q| !q.is_finite()) {
        return Err(Erreur::Config(
            "Le champ contient des composantes non finies".into(),
        ));
    }
    let mut plaquette = 0.0;
    let mut rectangle = 0.0;
    // Ordre fixe : directions croissantes, puis sites croissants, sans réduction parallèle.
    for mu in 0..4 {
        for nu in 0..4 {
            if mu == nu {
                continue;
            }
            let mut somme_p = 0.0;
            let mut somme_r = 0.0;
            for s in 0..v {
                if mu < nu {
                    somme_p += plaquette_site(champ, v, s, mu, nu);
                }
                somme_r += rectangle_site(champ, v, s, mu, nu);
            }
            plaquette += somme_p / v as f64;
            rectangle += somme_r / v as f64;
        }
    }
    let observables = Observables {
        plaquette: plaquette / 6.0,
        rectangle: rectangle / 12.0,
    };
    if !observables.plaquette.is_finite() || !observables.rectangle.is_finite() {
        return Err(Erreur::Config(
            "Débordement lors du calcul des observables".into(),
        ));
    }
    Ok(observables)
}

/// Translation périodique de +e_mu, avec s = ((x0 L + x1) L + x2) L + x3.
fn suivant(s: usize, mu: usize, l: usize) -> usize {
    let pas = l.pow((3 - mu) as u32);
    if (s / pas) % l == l - 1 {
        s - (l - 1) * pas
    } else {
        s + pas
    }
}

fn produit(liens: &[Q]) -> Q {
    liens.iter().copied().fold([1.0, 0.0, 0.0, 0.0], qmul)
}

fn plaquette_site(champ: &ChampHote, v: usize, s: usize, mu: usize, nu: usize) -> f64 {
    let s_mu = suivant(s, mu, champ.l);
    let s_nu = suivant(s, nu, champ.l);
    produit(&[
        champ.lien(mu * v + s),
        champ.lien(nu * v + s_mu),
        qdag(champ.lien(mu * v + s_nu)),
        qdag(champ.lien(nu * v + s)),
    ])[0]
}

fn rectangle_site(champ: &ChampHote, v: usize, s: usize, mu: usize, nu: usize) -> f64 {
    let s_mu = suivant(s, mu, champ.l);
    let s_nu = suivant(s, nu, champ.l);
    let s_mu_nu = suivant(s_mu, nu, champ.l);
    let s_2nu = suivant(s_nu, nu, champ.l);
    produit(&[
        champ.lien(mu * v + s),
        champ.lien(nu * v + s_mu),
        champ.lien(nu * v + s_mu_nu),
        qdag(champ.lien(mu * v + s_2nu)),
        qdag(champ.lien(nu * v + s_nu)),
        qdag(champ.lien(nu * v + s)),
    ])[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn uniforme(graine: &mut u64) -> f64 {
        *graine = graine
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Intervalle ouvert, pour le logarithme de Box–Muller.
        (((*graine >> 12) as f64) + 0.5) / (1_u64 << 52) as f64
    }

    fn haar(graine: &mut u64) -> Q {
        let mut q = [0.0; 4];
        for paire in q.chunks_exact_mut(2) {
            let rayon = (-2.0 * uniforme(graine).ln()).sqrt();
            let angle = TAU * uniforme(graine);
            paire[0] = rayon * angle.cos();
            paire[1] = rayon * angle.sin();
        }
        let norme = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        q.map(|v| v / norme)
    }

    fn aleatoire(l: usize, graine: &mut u64) -> ChampHote {
        ChampHote {
            l,
            q: (0..4 * l.pow(4)).flat_map(|_| haar(graine)).collect(),
        }
    }

    fn transformer(champ: &ChampHote, g: &[Q]) -> ChampHote {
        let v = champ.v();
        let mut transforme = champ.clone();
        for mu in 0..4 {
            for (s, &gx) in g.iter().enumerate() {
                let lien = qmul(
                    qmul(gx, champ.lien(mu * v + s)),
                    qdag(g[suivant(s, mu, champ.l)]),
                );
                transforme.q[4 * (mu * v + s)..4 * (mu * v + s + 1)].copy_from_slice(&lien);
            }
        }
        transforme
    }

    fn verifier_unite(champ: &ChampHote) {
        let obs = plaquette_et_rectangle(champ).expect("Champ valide");
        assert!((obs.plaquette - 1.0).abs() < 1e-14, "{obs:?}");
        assert!((obs.rectangle - 1.0).abs() < 1e-14, "{obs:?}");
    }

    #[test]
    fn froid_jauge_pure_et_abelien_constant() {
        let froid = ChampHote::froid(4).expect("L pair");
        assert_eq!(
            plaquette_et_rectangle(&froid).expect("Champ froid"),
            Observables {
                plaquette: 1.0,
                rectangle: 1.0
            }
        );
        let mut graine = 42;
        let g: Vec<_> = (0..froid.v()).map(|_| haar(&mut graine)).collect();
        verifier_unite(&transformer(&froid, &g));
        let mut abelien = froid;
        let a = 0.73_f64;
        let v = abelien.v();
        for lien in abelien.q[..4 * v].chunks_exact_mut(4) {
            lien.copy_from_slice(&[a.cos(), 0.0, 0.0, a.sin()]);
        }
        verifier_unite(&abelien);
        // Le calcul accepte aussi un volume impair, sans passer par froid().
        let impair = ChampHote {
            l: 3,
            q: (0..4 * 3_usize.pow(4))
                .flat_map(|_| [1.0, 0.0, 0.0, 0.0])
                .collect(),
        };
        verifier_unite(&impair);
    }

    #[test]
    fn invariance_de_jauge_et_determinisme() {
        let mut graine = 9876;
        let champ = aleatoire(4, &mut graine);
        let g: Vec<_> = (0..champ.v()).map(|_| haar(&mut graine)).collect();
        let avant = plaquette_et_rectangle(&champ).expect("Champ SU(2)");
        assert!((-1.0..=1.0).contains(&avant.plaquette));
        assert!((-1.0..=1.0).contains(&avant.rectangle));
        let apres = plaquette_et_rectangle(&transformer(&champ, &g)).expect("Champ transformé");
        assert!((avant.plaquette - apres.plaquette).abs() < 1e-14);
        assert!((avant.rectangle - apres.rectangle).abs() < 1e-14);
        assert_eq!(avant, plaquette_et_rectangle(&champ).expect("Même champ"));
    }

    // Référence géométrique indépendante : on parcourt un chemin signé en coordonnées.
    fn boucle_naive(champ: &ChampHote, s: usize, chemin: &[(usize, bool)]) -> f64 {
        let l = champ.l;
        let mut x = [s / l.pow(3), (s / l.pow(2)) % l, (s / l) % l, s % l];
        let mut q = [1.0, 0.0, 0.0, 0.0];
        for &(mu, positif) in chemin {
            if !positif {
                x[mu] = (x[mu] + l - 1) % l;
            }
            let site = ((x[0] * l + x[1]) * l + x[2]) * l + x[3];
            let lien = champ.lien(mu * champ.v() + site);
            q = qmul(q, if positif { lien } else { qdag(lien) });
            if positif {
                x[mu] = (x[mu] + 1) % l;
            }
        }
        q[0]
    }

    #[test]
    fn cinq_sites_six_plans_et_moyennes_naives() {
        let champ = aleatoire(4, &mut 1234);
        let v = champ.v();
        let mut somme_p = 0.0;
        let mut somme_r = 0.0;
        for mu in 0..4 {
            for nu in 0..4 {
                if mu == nu {
                    continue;
                }
                let chemin_p = [(mu, true), (nu, true), (mu, false), (nu, false)];
                let chemin_r = [
                    (mu, true),
                    (nu, true),
                    (nu, true),
                    (mu, false),
                    (nu, false),
                    (nu, false),
                ];
                if mu < nu {
                    for s in [0, 3, 15, 63, 255] {
                        let attendu = boucle_naive(&champ, s, &chemin_p);
                        assert!((plaquette_site(&champ, v, s, mu, nu) - attendu).abs() < 1e-15);
                    }
                    somme_p += (0..v)
                        .map(|s| boucle_naive(&champ, s, &chemin_p))
                        .sum::<f64>()
                        / v as f64;
                }
                somme_r += (0..v)
                    .map(|s| boucle_naive(&champ, s, &chemin_r))
                    .sum::<f64>()
                    / v as f64;
            }
        }
        let obs = plaquette_et_rectangle(&champ).expect("Champ valide");
        assert!((obs.plaquette - somme_p / 6.0).abs() < 1e-14);
        assert!((obs.rectangle - somme_r / 12.0).abs() < 1e-14);
    }

    #[test]
    fn configurations_invalides() {
        for champ in [
            ChampHote { l: 2, q: vec![] },
            ChampHote { l: 4, q: vec![] },
            ChampHote {
                l: usize::MAX,
                q: vec![],
            },
        ] {
            assert!(matches!(
                plaquette_et_rectangle(&champ),
                Err(Erreur::Config(_))
            ));
        }
        for invalide in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
            let mut champ = ChampHote::froid(4).expect("L pair");
            champ.q.fill(invalide);
            assert!(matches!(
                plaquette_et_rectangle(&champ),
                Err(Erreur::Config(_))
            ));
        }
    }
}
