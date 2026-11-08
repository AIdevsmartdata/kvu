//! Hessienne des plaquettes SU(2), sans matrice, dans la base iσ/√2.
//!
//! Les liens sont rangés par direction puis site en ordre C. Les deux passes
//! séparent les contributions des boucles de leur somme par lien : aucune
//! réduction flottante ne dépend de l'ordonnancement des fils.
#![forbid(unsafe_code)]

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const NCH: usize = 256;
type Matrice = [[f64; 3]; 3];
type Contributions = [[f64; 3]; 4];

/// Tables partagées : six plaquettes par site et six occurrences par lien.
pub struct GeoPlaq {
    l: usize,
    lid: Vec<[usize; 4]>,
    occ: Vec<[usize; 6]>,
    signes: [f64; 4],
}

impl GeoPlaq {
    /// Géométrie mise en cache pour L ≥ 2.
    ///
    /// Panique si L < 2 ou si les dimensions débordent `usize`.
    pub fn pour(l: usize) -> Arc<GeoPlaq> {
        static CACHE: OnceLock<Mutex<HashMap<usize, Arc<GeoPlaq>>>> = OnceLock::new();
        assert!(l >= 2, "la taille L doit être au moins 2");
        let mut cache = CACHE
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap_or_else(|empoisonne| empoisonne.into_inner());
        Arc::clone(
            cache
                .entry(l)
                .or_insert_with(|| Arc::new(Self::construire(l))),
        )
    }

    fn construire(l: usize) -> Self {
        let volume = l
            .checked_pow(4)
            .filter(|v| v.checked_mul(16).is_some())
            .expect("les dimensions de la géométrie débordent usize");
        let pas = [l * l * l, l * l, l, 1];
        let suivant = |site: usize, direction: usize| {
            if (site / pas[direction]) % l + 1 == l {
                site - (l - 1) * pas[direction]
            } else {
                site + pas[direction]
            }
        };
        let mut lid = vec![[0; 4]; 6 * volume];
        let mut occ = vec![[0; 6]; 4 * volume];
        let mut nombres = vec![0; 4 * volume];
        let signes = [1.0, 1.0, -1.0, -1.0];
        let mut t = 0;
        for mu in 0..4 {
            for nu in mu + 1..4 {
                for site in 0..volume {
                    lid[t * volume + site] = [
                        mu * volume + site,
                        nu * volume + suivant(site, mu),
                        mu * volume + suivant(site, nu),
                        nu * volume + site,
                    ];
                }
                // Ordre np.nonzero(dirs == mu) : t, puis slot, puis site.
                for (s, _) in signes.iter().enumerate() {
                    for site in 0..volume {
                        let boucle = t * volume + site;
                        let e = lid[boucle][s];
                        occ[e][nombres[e]] = 4 * boucle + s;
                        nombres[e] += 1;
                    }
                }
                t += 1;
            }
        }
        Self {
            l,
            lid,
            occ,
            signes,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutantHv {
    Aucun,
    /// Retire uniquement B_ii = (u0/2) I de chaque occurrence :
    /// H_mutant v = H v − (Σ_occ u0/2) v, lien par lien.
    MemeLienRetire,
}

#[derive(Clone, Default)]
struct BoucleCache {
    a: [Matrice; 4],
    u: [f64; 4],
}

/// Opérateur au point q : A_s = signe_s R(P_s) et U_l sont précalculés.
/// Les tampons de contributions sont propres à chaque appel, donc les appels
/// simultanés sur le même opérateur sont possibles.
pub struct OperateurGauche {
    geo: Arc<GeoPlaq>,
    boucles: Vec<BoucleCache>,
}

impl OperateurGauche {
    /// Construit le cache dans le pool fourni, sans vérifier la norme de q.
    pub fn nouveau(pool: &rayon::ThreadPool, q: &[f64], l: usize) -> Result<Self, String> {
        if l < 2 {
            return Err("la taille L doit être au moins 2".into());
        }
        let longueur = l
            .checked_pow(4)
            .and_then(|v| v.checked_mul(16))
            .ok_or_else(|| "les dimensions demandées débordent usize".to_string())?;
        if q.len() != longueur {
            return Err(format!(
                "longueur de q incorrecte : attendue {longueur}, reçue {}",
                q.len()
            ));
        }
        let geo = GeoPlaq::pour(l);
        let mut boucles = vec![BoucleCache::default(); geo.lid.len()];
        pool.install(|| {
            boucles
                .par_chunks_mut(NCH)
                .enumerate()
                .for_each(|(ch, troncon)| {
                    for (position, cache) in troncon.iter_mut().enumerate() {
                        let liens = geo.lid[ch * NCH + position];
                        let mut produit = [1.0, 0.0, 0.0, 0.0];
                        for (s, &e) in liens.iter().enumerate() {
                            let mut facteur = [q[4 * e], q[4 * e + 1], q[4 * e + 2], q[4 * e + 3]];
                            let prefixe;
                            if geo.signes[s] > 0.0 {
                                prefixe = produit;
                                produit = produit_quaternion(produit, facteur);
                            } else {
                                for valeur in &mut facteur[1..] {
                                    *valeur = -*valeur;
                                }
                                produit = produit_quaternion(produit, facteur);
                                prefixe = produit;
                            }
                            for b in 0..3 {
                                let mut base = [0.0; 3];
                                base[b] = 1.0;
                                let colonne = rotation(prefixe, base);
                                for (a, valeur) in colonne.into_iter().enumerate() {
                                    cache.a[s][a][b] = geo.signes[s] * valeur;
                                }
                            }
                        }
                        cache.u = produit;
                    }
                });
        });
        Ok(Self { geo, boucles })
    }

    pub fn l(&self) -> usize {
        self.geo.l
    }

    /// Nombre de composantes tangentes : 12 L⁴.
    pub fn n(&self) -> usize {
        3 * self.geo.occ.len()
    }

    /// Écrase y par H v (β = 1). Panique si v ou y n'a pas n composantes.
    pub fn hv(&self, pool: &rayon::ThreadPool, v: &[f64], y: &mut [f64], mutant: MutantHv) {
        assert_eq!(v.len(), self.n(), "longueur incorrecte du vecteur tangent");
        assert_eq!(y.len(), self.n(), "longueur incorrecte de la sortie");
        let mut contributions = vec![[[0.0; 3]; 4]; self.boucles.len()];
        pool.install(|| {
            contributions
                .par_chunks_mut(NCH)
                .enumerate()
                .for_each(|(ch, troncon)| {
                    for (position, sortie) in troncon.iter_mut().enumerate() {
                        let indice = ch * NCH + position;
                        let cache = &self.boucles[indice];
                        let [u0, u1, u2, u3] = cache.u;
                        let mut w = [[0.0; 3]; 4];
                        let mut total = [0.0; 3];
                        for (s, ws) in w.iter_mut().enumerate() {
                            let e = self.geo.lid[indice][s];
                            let [x0, x1, x2] = [v[3 * e], v[3 * e + 1], v[3 * e + 2]];
                            for a in 0..3 {
                                ws[a] = cache.a[s][a][0] * x0
                                    + cache.a[s][a][1] * x1
                                    + cache.a[s][a][2] * x2;
                                total[a] += ws[a];
                            }
                        }
                        let mut prefixe = [0.0; 3];
                        for s in 0..4 {
                            let z0 = 2.0 * prefixe[0] + w[s][0] - total[0];
                            let z1 = 2.0 * prefixe[1] + w[s][1] - total[1];
                            let z2 = 2.0 * prefixe[2] + w[s][2] - total[2];
                            let y0 = u0 * total[0] + (z1 * u3 - z2 * u2);
                            let y1 = u0 * total[1] + (z2 * u1 - z0 * u3);
                            let y2 = u0 * total[2] + (z0 * u2 - z1 * u1);
                            for (b, valeur) in sortie[s].iter_mut().enumerate() {
                                *valeur = 0.5
                                    * (cache.a[s][0][b] * y0
                                        + cache.a[s][1][b] * y1
                                        + cache.a[s][2][b] * y2);
                            }
                            for (a, valeur) in prefixe.iter_mut().enumerate() {
                                *valeur += w[s][a];
                            }
                        }
                    }
                });
        });
        self.sommer_liens(pool, &contributions, y);
        if mutant == MutantHv::MemeLienRetire {
            pool.install(|| {
                y.par_chunks_mut(3 * NCH)
                    .enumerate()
                    .for_each(|(ch, troncon)| {
                        for (position, sortie) in troncon.chunks_mut(3).enumerate() {
                            let e = ch * NCH + position;
                            let mut diagonale = 0.0;
                            for &j in &self.geo.occ[e] {
                                diagonale += 0.5 * self.boucles[j / 4].u[0];
                            }
                            for a in 0..3 {
                                sortie[a] -= diagonale * v[3 * e + a];
                            }
                        }
                    })
            });
        }
    }

    /// Écrase g par ∇S dans la même base et le même ordre des occurrences.
    /// Panique si g n'a pas n composantes.
    pub fn gradient(&self, pool: &rayon::ThreadPool, g: &mut [f64]) {
        assert_eq!(g.len(), self.n(), "longueur incorrecte du gradient");
        let mut contributions = vec![[[0.0; 3]; 4]; self.boucles.len()];
        pool.install(|| {
            contributions
                .par_chunks_mut(NCH)
                .enumerate()
                .for_each(|(ch, troncon)| {
                    for (position, sortie) in troncon.iter_mut().enumerate() {
                        let cache = &self.boucles[ch * NCH + position];
                        for (s, slot) in sortie.iter_mut().enumerate() {
                            for (b, valeur) in slot.iter_mut().enumerate() {
                                *valeur = std::f64::consts::FRAC_1_SQRT_2
                                    * (cache.a[s][0][b] * cache.u[1]
                                        + cache.a[s][1][b] * cache.u[2]
                                        + cache.a[s][2][b] * cache.u[3]);
                            }
                        }
                    }
                })
        });
        self.sommer_liens(pool, &contributions, g);
    }

    /// Action Σ_l (1 − a0(U_l)), sommée séquentiellement par indice de boucle.
    pub fn action(&self) -> f64 {
        let mut action = 0.0;
        for cache in &self.boucles {
            action += 1.0 - cache.u[0];
        }
        action
    }

    fn sommer_liens(
        &self,
        pool: &rayon::ThreadPool,
        contributions: &[Contributions],
        sortie: &mut [f64],
    ) {
        pool.install(|| {
            sortie
                .par_chunks_mut(3 * NCH)
                .enumerate()
                .for_each(|(ch, troncon)| {
                    for (position, lien) in troncon.chunks_mut(3).enumerate() {
                        let mut somme = [0.0; 3];
                        for &j in &self.geo.occ[ch * NCH + position] {
                            for a in 0..3 {
                                somme[a] += contributions[j / 4][j % 4][a];
                            }
                        }
                        for a in 0..3 {
                            lien[a] = 0.0 + somme[a];
                        }
                    }
                })
        });
    }
}

fn produit_quaternion(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [a0, a1, a2, a3] = a;
    let [b0, b1, b2, b3] = b;
    [
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + b0 * a1 - (a2 * b3 - a3 * b2),
        a0 * b2 + b0 * a2 - (a3 * b1 - a1 * b3),
        a0 * b3 + b0 * a3 - (a1 * b2 - a2 * b1),
    ]
}

fn rotation(q: [f64; 4], z: [f64; 3]) -> [f64; 3] {
    let [q0, p1, p2, p3] = q;
    let [z0, z1, z2] = z;
    let a = q0 * q0 - (p1 * p1 + p2 * p2 + p3 * p3);
    let d = 2.0 * (p1 * z0 + p2 * z1 + p3 * z2);
    let c0 = p2 * z2 - p3 * z1;
    let c1 = p3 * z0 - p1 * z2;
    let c2 = p1 * z1 - p2 * z0;
    let g = -2.0 * q0;
    [
        a * z0 + d * p1 + g * c0,
        a * z1 + d * p2 + g * c1,
        a * z2 + d * p3 + g * c2,
    ]
}
