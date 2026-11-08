//! Action de Wilson et dérivées dans les coordonnées gauches orthonormées de SU(2).

use super::jacobien::Q;
use crate::commun::ChampHote;
use crate::heritage::su2_hb::{qdag, qmul, staple, Geo};
use rayon::prelude::*;

fn quaternions(champ: &ChampHote) -> Vec<Q> {
    champ
        .q
        .chunks_exact(4)
        .map(|q| [q[0], q[1], q[2], q[3]])
        .collect()
}

/// Somme compensée dans l'ordre fixe (plan μ < ν, puis site).
fn somme_plaquettes(champ: &ChampHote) -> f64 {
    assert!(champ.l >= 2, "L ≥ 2 requis pour les plaquettes");
    let geo = Geo::new(champ.l);
    let mut somme = 0.0;
    let mut correction = 0.0;
    for mu in 0..4 {
        for nu in mu + 1..4 {
            for s in 0..geo.v {
                let x = geo.coords(s);
                let smu = geo.nb_up(s, mu, x[mu]);
                let snu = geo.nb_up(s, nu, x[nu]);
                let p = qmul(
                    qmul(champ.lien(mu * geo.v + s), champ.lien(nu * geo.v + smu)),
                    qmul(
                        qdag(champ.lien(mu * geo.v + snu)),
                        qdag(champ.lien(nu * geo.v + s)),
                    ),
                );
                let terme = p[0] - correction;
                let suivante = somme + terme;
                correction = (suivante - somme) - terme;
                somme = suivante;
            }
        }
    }
    somme
}

/// Moyenne de ½ Re tr U_p sur les 6V plaquettes, dans un ordre fixe.
pub fn plaquette_moyenne(champ: &ChampHote) -> f64 {
    somme_plaquettes(champ) / (6 * champ.v()) as f64
}

/// S = β (6V − Σ_p ½ Re tr U_p), avec le même ordre de somme que la moyenne.
pub fn action_wilson(champ: &ChampHote, beta: f64) -> f64 {
    beta * ((6 * champ.v()) as f64 - somme_plaquettes(champ))
}

/// Gradient pour U_ℓ → exp(i ε_ℓ·σ) U_ℓ, indice 3·li + a.
/// Chaque lien est indépendant et conserve l'ordre des six agrafes du générateur.
pub fn grad_action(champ: &ChampHote, beta: f64, pool: &rayon::ThreadPool) -> Vec<f64> {
    assert!(champ.l >= 2, "L ≥ 2 requis pour les agrafes");
    let geo = Geo::new(champ.l);
    let u = quaternions(champ);
    let mut gradient = vec![0.0; 12 * geo.v];
    pool.install(|| {
        gradient
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(li, sortie)| {
                let s = li % geo.v;
                let a = staple(&u, &geo, li / geo.v, s, &geo.coords(s));
                let m = qmul(u[li], a);
                for j in 0..3 {
                    sortie[j] = beta * m[j + 1];
                }
            });
    });
    gradient
}

/// Variation locale d'action lors du remplacement du lien `li` par `nouveau`.
/// Pour des remplacements répétés, extraire une fois les quaternions et utiliser
/// [`delta_action_lien_q`] afin d'éviter une copie du champ à chaque appel.
pub fn delta_action_lien(champ: &ChampHote, geo: &Geo, li: usize, nouveau: Q, beta: f64) -> f64 {
    assert_eq!(champ.l, geo.l, "géométrie incompatible avec le champ");
    delta_action_lien_q(&quaternions(champ), geo, li, nouveau, beta)
}

/// Même variation locale, sur un tampon de quaternions déjà extrait (li = μV + s).
pub fn delta_action_lien_q(u: &[Q], geo: &Geo, li: usize, nouveau: Q, beta: f64) -> f64 {
    assert!(geo.l >= 2, "L ≥ 2 requis pour les agrafes");
    assert_eq!(u.len(), 4 * geo.v, "longueur du tampon de quaternions");
    assert!(li < u.len(), "indice de lien hors du champ");
    let s = li % geo.v;
    let a = staple(u, geo, li / geo.v, s, &geo.coords(s));
    -beta * (qmul(nouveau, a)[0] - qmul(u[li], a)[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heritage::su2_hb::measure_quat;
    use crate::rg::bloc::exp_q;
    use crate::su2::cpu::haar_q;

    fn pool(fils: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(fils)
            .build()
            .unwrap()
    }

    fn haar(l: usize) -> ChampHote {
        ChampHote {
            l,
            q: (0..4 * l.pow(4))
                .flat_map(|i| haar_q(i as u32, 0, 23, 17, 41))
                .collect(),
        }
    }

    fn perturber(champ: &ChampHote, direction: &[f64], pas: f64) -> ChampHote {
        let mut resultat = champ.clone();
        for (li, lien) in resultat.q.chunks_exact_mut(4).enumerate() {
            let epsilon = std::array::from_fn(|a| pas * direction[3 * li + a]);
            lien.copy_from_slice(&qmul(exp_q(epsilon), champ.lien(li)));
        }
        resultat
    }

    #[test]
    fn froid_haar_et_geometries_impaires() {
        for l in [2, 3, 4, 6] {
            let froid = ChampHote {
                l,
                q: (0..4 * l.pow(4))
                    .flat_map(|_| [1.0, 0.0, 0.0, 0.0])
                    .collect(),
            };
            assert_eq!(plaquette_moyenne(&froid), 1.0);
            assert_eq!(action_wilson(&froid, 2.3), 0.0);
            assert!(grad_action(&froid, 2.3, &pool(1)).iter().all(|&x| x == 0.0));
        }
        let moyenne = plaquette_moyenne(&haar(6));
        assert!(moyenne.abs() < 0.01, "moyenne Haar : {moyenne}");
    }

    #[test]
    fn accord_avec_la_mesure_du_generateur() {
        let champ = haar(4);
        let reference = pool(4).install(|| measure_quat(&Geo::new(4), &quaternions(&champ))[0]);
        assert!((plaquette_moyenne(&champ) - reference).abs() < 1e-13);
    }

    #[test]
    fn gradient_identique_a_loctet_selon_les_fils() {
        let champ = haar(6);
        let reference = grad_action(&champ, 2.3, &pool(1));
        for fils in [4, 8] {
            let gradient = grad_action(&champ, 2.3, &pool(fils));
            assert!(reference
                .iter()
                .zip(gradient)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn gradient_par_differences_finies_richardson() {
        let pool = pool(3);
        for l in [4, 6] {
            let champ = haar(l);
            let beta = 1.0;
            let gradient = grad_action(&champ, beta, &pool);
            let mut erreur_carree = 0.0;
            let mut norme_carree = 0.0;
            for tirage in 0..3 {
                let mut direction: Vec<_> = (0..4 * champ.v())
                    .flat_map(|li| {
                        let q = haar_q(li as u32, tirage, 37, 31, 53);
                        [q[1], q[2], q[3]]
                    })
                    .collect();
                let norme = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
                for x in &mut direction {
                    *x /= norme;
                }
                let analytique = gradient
                    .iter()
                    .zip(&direction)
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                let derivees: [f64; 3] = std::array::from_fn(|k| {
                    let h = 1e-3 / (1 << k) as f64;
                    (action_wilson(&perturber(&champ, &direction, h), beta)
                        - action_wilson(&perturber(&champ, &direction, -h), beta))
                        / (2.0 * h)
                });
                let premier = (4.0 * derivees[1] - derivees[0]) / 3.0;
                let second = (4.0 * derivees[2] - derivees[1]) / 3.0;
                let richardson = (16.0 * second - premier) / 15.0;
                erreur_carree += (richardson - analytique).powi(2);
                norme_carree += analytique * analytique;
            }
            // Norme relative du vecteur des trois dérivées directionnelles :
            // une projection presque nulle ne doit pas servir de dénominateur.
            // À L=6, l'action extensive vaut environ 7800 : son arrondi FP64
            // est amplifié par les différences à h/4, même avec une somme compensée.
            let relatif = (erreur_carree / norme_carree).sqrt();
            eprintln!("gradient Richardson L={l}, écart relatif en norme : {relatif:e}");
            assert!(relatif <= 1e-9, "L={l} : écart relatif {relatif:e}");
        }
    }

    #[test]
    fn variations_locales_sur_deux_cents_liens() {
        // L=2 vérifie aussi les agrafes qui traversent deux fois une direction
        // périodique ; L=3 contrôle l'absence d'hypothèse de parité.
        for l in [2, 3, 4] {
            let mut champ = haar(l);
            let geo = Geo::new(l);
            let mut u = quaternions(&champ);
            let beta = 1.0;
            let mut etat = 0x7b91_4b07_u64;
            for tirage in 0..200 {
                etat = etat.wrapping_mul(6364136223846793005).wrapping_add(1);
                let li = ((etat >> 32) as usize) % u.len();
                let nouveau = haar_q(li as u32, tirage, 43, 59, 61);
                let ancien = action_wilson(&champ, beta);
                let delta = delta_action_lien(&champ, &geo, li, nouveau, beta);
                assert_eq!(
                    delta.to_bits(),
                    delta_action_lien_q(&u, &geo, li, nouveau, beta).to_bits()
                );
                champ.q[4 * li..4 * li + 4].copy_from_slice(&nouveau);
                u[li] = nouveau;
                let mesure = action_wilson(&champ, beta) - ancien;
                assert!(
                    (delta - mesure).abs() <= 1e-12,
                    "L={l}, lien={li} : local={delta}, global={mesure}, écart={:e}",
                    (delta - mesure).abs()
                );
            }
        }
    }
}
