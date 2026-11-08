//! Fixation de Landau réseau (max Σ Re tr U) par sur-relaxation en damier : port de `lat.py::landau_fix` (S2).
//!
//! Même algorithme : pour chaque parité p ∈ {0, 1} des sites, g_x = (h_x)† / |h_x| avec h_x = Σ_μ [U_{x,μ} + U_{x−μ,μ}†], puis g_x^ω (ω = 1,7 par défaut), U_{x,μ} ← g U_{x,μ},
//! U_{x−μ,μ} ← U_{x−μ,μ} g† ; critère (toutes les 20 itérations) θ = moyenne sur les sites de |Σ_μ [A_{x,μ} − A_{x−μ,μ}]|² avec A = sin(θ_lien) n̂ (partie vectorielle du quaternion).
//! Les sites d'une même parité ne se touchent pas : les mises à jour sont indépendantes (parallélisme sans réduction), résultat identique à l'octet quel que soit le nombre de fils.

use super::bases::Reseau;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul};
use rayon::prelude::*;

pub struct SortieLandau {
    pub champ: ChampHote,
    pub theta: f64,
    pub iterations: usize,
    pub converge: bool,
}

fn puissance(g: [f64; 4], omega: f64) -> [f64; 4] {
    let nv = (g[1] * g[1] + g[2] * g[2] + g[3] * g[3]).sqrt();
    if nv < 1e-300 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let theta = nv.atan2(g[0].clamp(-1.0, 1.0));
    let (s, c) = (omega * theta).sin_cos();
    [c, s * g[1] / nv, s * g[2] / nv, s * g[3] / nv]
}

/// Critère de Landau θ.
pub fn critere(r: &Reseau, q: &[f64]) -> f64 {
    let lien = |mu: usize, x: usize| &q[4 * (mu * r.v + x)..4 * (mu * r.v + x) + 4];
    let mut somme = 0.0;
    for x in 0..r.v {
        let mut d = [0.0; 3];
        for mu in 0..4 {
            let a = lien(mu, x);
            let b = lien(mu, r.arriere[mu][x]);
            for c in 0..3 {
                d[c] += a[1 + c] - b[1 + c];
            }
        }
        somme += d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
    somme / r.v as f64
}

pub fn fixer(pool: &rayon::ThreadPool, champ: &ChampHote, tol: f64, iter_max: usize, omega: f64) -> Resultat<SortieLandau> {
    let r = Reseau::nouveau(champ.l)?;
    if champ.q.len() != 16 * r.v {
        return Err(Erreur::Config("longueur du champ invalide".into()));
    }
    let mut q = champ.q.clone();
    let parite: Vec<usize> = (0..r.v).map(|x| Reseau::coordonnees_de(x, champ.l).iter().sum::<usize>() % 2).collect();
    let sites: [Vec<usize>; 2] = [
        (0..r.v).filter(|&x| parite[x] == 0).collect(),
        (0..r.v).filter(|&x| parite[x] == 1).collect(),
    ];
    let mut theta = f64::INFINITY;
    let mut fait = 0usize;
    for it in 0..iter_max {
        fait = it;
        for p in 0..2 {
            // g_x pour tous les sites de la parité (lecture seule de q), puis application
            let gs: Vec<[f64; 4]> = pool.install(|| {
                sites[p]
                    .par_iter()
                    .map(|&x| {
                        let mut h = [0.0; 4];
                        for mu in 0..4 {
                            let a = &q[4 * (mu * r.v + x)..4 * (mu * r.v + x) + 4];
                            let xm = r.arriere[mu][x];
                            let b = &q[4 * (mu * r.v + xm)..4 * (mu * r.v + xm) + 4];
                            h[0] += a[0] + b[0];
                            for c in 1..4 {
                                h[c] += a[c] - b[c];
                            }
                        }
                        let norme = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2] + h[3] * h[3]).sqrt().max(1e-150);
                        let g = [h[0] / norme, -h[1] / norme, -h[2] / norme, -h[3] / norme];
                        puissance(g, omega)
                    })
                    .collect()
            });
            for (k, &x) in sites[p].iter().enumerate() {
                let g = gs[k];
                let gd = qdag(g);
                for mu in 0..4 {
                    let o = 4 * (mu * r.v + x);
                    let u = qmul(g, [q[o], q[o + 1], q[o + 2], q[o + 3]]);
                    q[o..o + 4].copy_from_slice(&u);
                    let xm = r.arriere[mu][x];
                    let om = 4 * (mu * r.v + xm);
                    let um = qmul([q[om], q[om + 1], q[om + 2], q[om + 3]], gd);
                    q[om..om + 4].copy_from_slice(&um);
                }
            }
        }
        if it % 20 == 0 || it == iter_max - 1 {
            theta = critere(&r, &q);
            if theta < tol {
                break;
            }
        }
    }
    let converge = theta < tol;
    Ok(SortieLandau { champ: ChampHote { l: champ.l, q }, theta, iterations: fait, converge })
}
