//! LOBPCG généralisé déterministe, avec images des opérateurs conservées.
use super::dense::{cholesky, eigh_generalise, norme2, pdscal, resoudre_lt};
use rayon::prelude::*;

/// Opérateur linéaire ; la sortie doit être entièrement écrasée.
pub trait Op: Sync {
    fn n(&self) -> usize;
    fn appliquer(&self, x: &[f64], y: &mut [f64]);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cote {
    Min,
    Max,
}

pub struct Params {
    pub bloc: usize,
    pub tol: f64,
    pub iter_max: usize,
    pub graine: u64,
    pub cote: Cote,
    pub echelle: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            bloc: 4,
            tol: 1e-10,
            iter_max: 2000,
            graine: 0,
            cote: Cote::Min,
            echelle: 1.0,
        }
    }
}

pub struct Sortie {
    pub valeurs: Vec<f64>,
    pub vecteur: Vec<f64>,
    pub residu_abs: f64,
    pub residu_rel: f64,
    pub iterations: usize,
    pub converge: bool,
    pub historique: Vec<(usize, f64)>,
}

#[derive(Clone)]
struct Colonne {
    x: Vec<f64>,
    ax: Vec<f64>,
    bx: Vec<f64>,
}

fn projeter(proj: &dyn Op, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    proj.appliquer(x, &mut y);
    y
}

fn images(a: &dyn Op, b: &dyn Op, proj: &dyn Op, x: Vec<f64>, signe: f64) -> Colonne {
    let mut tampon = vec![0.0; x.len()];
    a.appliquer(&x, &mut tampon);
    let mut ax = projeter(proj, &tampon);
    for v in &mut ax {
        *v *= signe;
    }
    b.appliquer(&x, &mut tampon);
    let bx = projeter(proj, &tampon);
    Colonne { x, ax, bx }
}

fn combinaison(base: &[Colonne], coefficients: &[f64]) -> Colonne {
    let n = base[0].x.len();
    let mut resultat = Colonne {
        x: vec![0.0; n],
        ax: vec![0.0; n],
        bx: vec![0.0; n],
    };
    // Les sommes portent sur les colonnes, toujours dans le même ordre.
    for debut in (0..base.len()).step_by(4096) {
        for i in 0..n {
            let mut x = 0.0;
            let mut ax = 0.0;
            let mut bx = 0.0;
            for j in debut..base.len().min(debut + 4096) {
                x += coefficients[j] * base[j].x[i];
                ax += coefficients[j] * base[j].ax[i];
                bx += coefficients[j] * base[j].bx[i];
            }
            resultat.x[i] += x;
            resultat.ax[i] += ax;
            resultat.bx[i] += bx;
        }
    }
    resultat
}

fn gram(base: &[Colonne], avec_a: bool) -> Vec<f64> {
    let n = base.len();
    let mut g = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let image_i = if avec_a { &base[i].ax } else { &base[i].bx };
            let image_j = if avec_a { &base[j].ax } else { &base[j].bx };
            let v = 0.5 * (pdscal(None, &base[i].x, image_j) + pdscal(None, &base[j].x, image_i));
            g[i * n + j] = v;
            g[j * n + i] = v;
        }
    }
    g
}

/// Deux passages de Cholesky-QR limitent la perte d'orthogonalité des résidus.
fn orthonormaliser(
    pool: &rayon::ThreadPool,
    mut base: Vec<Colonne>,
) -> Result<Vec<Colonne>, String> {
    let n = base.len();
    for colonne in &mut base {
        let carre = pdscal(None, &colonne.x, &colonne.bx);
        if !carre.is_finite() || carre <= 0.0 {
            return Err("Gram de B non défini positif sur l'espace projeté".into());
        }
        let facteur = 1.0 / carre.sqrt();
        for v in colonne
            .x
            .iter_mut()
            .chain(&mut colonne.ax)
            .chain(&mut colonne.bx)
        {
            *v *= facteur;
        }
    }
    for _ in 0..2 {
        let mut g = gram(&base, false);
        cholesky(&mut g, n)
            .map_err(|i| format!("Gram de B non défini positif ou base dépendante : pivot {i}"))?;
        if (0..n).any(|i| g[i * n + i] < 1e-7) {
            return Err("Base mal conditionnée au cours de Cholesky-QR".into());
        }
        base = pool.install(|| {
            (0..n)
                .into_par_iter()
                .map(|j| {
                    let mut coefficients = vec![0.0; n];
                    coefficients[j] = 1.0;
                    resoudre_lt(&g, n, &mut coefficients);
                    combinaison(&base, &coefficients)
                })
                .collect()
        });
    }
    Ok(base)
}

type Ritz = Result<(Vec<f64>, Vec<Colonne>, Vec<f64>), String>;

fn ritz(pool: &rayon::ThreadPool, base: &[Colonne], bloc: usize) -> Ritz {
    let n = base.len();
    let (valeurs, vecteurs) = eigh_generalise(gram(base, true), gram(base, false), n, true)?;
    if valeurs.iter().any(|v| !v.is_finite()) {
        return Err("Valeur de Ritz non finie".into());
    }
    let v = vecteurs.ok_or("Vecteurs de Ritz absents")?;
    let colonnes = pool.install(|| {
        (0..bloc)
            .into_par_iter()
            .map(|j| {
                let coefficients: Vec<_> = (0..n).map(|i| v[i * n + j]).collect();
                combinaison(base, &coefficients)
            })
            .collect()
    });
    Ok((valeurs[..bloc].to_vec(), colonnes, v))
}

fn aleatoire(etat: &mut u64) -> f64 {
    *etat = etat.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *etat;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    2.0 * ((z >> 11) as f64 * (1.0 / 9007199254740992.0)) - 1.0
}

/// Valeurs extrêmes du pencil projeté, par blocs X, W = T R et P de Knyazev.
/// Le pool explicite n'exécute que des calculs indépendants sur des colonnes.
pub fn lobpcg(
    pool: &rayon::ThreadPool,
    a: &dyn Op,
    b: &dyn Op,
    proj: &dyn Op,
    prec: Option<&dyn Op>,
    p: &Params,
) -> Result<Sortie, String> {
    let n = a.n();
    if p.bloc == 0 || p.bloc > n {
        return Err("Taille de bloc invalide".into());
    }
    if !p.echelle.is_finite() || p.echelle <= 0.0 {
        return Err("L'échelle doit être finie et strictement positive".into());
    }
    if !p.tol.is_finite() || p.tol <= 0.0 {
        return Err("La tolérance doit être finie et strictement positive".into());
    }
    if b.n() != n || proj.n() != n || prec.is_some_and(|t| t.n() != n) {
        return Err("Dimensions incompatibles des opérateurs".into());
    }
    let signe = if p.cote == Cote::Min { 1.0 } else { -1.0 };
    let mut etat = p.graine;
    let depart: Vec<Vec<f64>> = (0..p.bloc)
        .map(|_| (0..n).map(|_| aleatoire(&mut etat)).collect())
        .collect();
    let depart: Vec<_> = pool.install(|| {
        depart
            .into_par_iter()
            .map(|x| images(a, b, proj, projeter(proj, &x), signe))
            .collect()
    });
    let depart = orthonormaliser(pool, depart)
        .map_err(|e| format!("Départ : B non définie positive ou image trop petite ; {e}"))?;
    let (mut valeurs, mut x, _) = ritz(pool, &depart, p.bloc)?;
    let mut directions: Vec<Colonne> = Vec::new();
    let mut historique = Vec::new();
    for iteration in 0..=p.iter_max {
        let residus: Vec<Vec<f64>> = x
            .iter()
            .zip(&valeurs)
            .map(|(colonne, &lambda)| {
                colonne
                    .ax
                    .iter()
                    .zip(&colonne.bx)
                    .map(|(&av, &bv)| av - lambda * bv)
                    .collect()
            })
            .collect();
        let absolus: Vec<_> = residus.iter().map(|r| norme2(None, r)).collect();
        let relatifs: Vec<_> = x
            .iter()
            .zip(&absolus)
            .map(|(c, &r)| r / (p.echelle * norme2(None, &c.x)))
            .collect();
        if relatifs.iter().any(|r| !r.is_finite()) {
            return Err("Résidu non fini".into());
        }
        let maximum = relatifs.iter().copied().fold(0.0, f64::max);
        historique.push((iteration, maximum));
        if maximum <= p.tol || iteration == p.iter_max {
            // Fixation du signe sans arbitraire dépendant de l'ordonnancement.
            let mut vecteur = x[0].x.clone();
            let pivot = vecteur
                .iter()
                .max_by(|u, v| u.abs().total_cmp(&v.abs()))
                .copied()
                .unwrap_or(0.0);
            if pivot < 0.0 {
                for v in &mut vecteur {
                    *v = -*v;
                }
            }
            // Contrôle final explicite : les arrondis des images conservées
            // ne doivent pas fausser le résidu annoncé après de longues suites.
            let mut finalise = images(a, b, proj, vecteur, signe);
            let carre_b = pdscal(None, &finalise.x, &finalise.bx);
            if !carre_b.is_finite() || carre_b <= 0.0 {
                return Err("B non définie positive lors du contrôle final".into());
            }
            let facteur = 1.0 / carre_b.sqrt();
            for v in &mut finalise.x {
                *v *= facteur;
            }
            let residu: Vec<_> = finalise
                .ax
                .iter()
                .zip(&finalise.bx)
                .map(|(&av, &bv)| facteur * (av - valeurs[0] * bv))
                .collect();
            let residu_abs = norme2(None, &projeter(proj, &residu));
            let residu_rel = residu_abs / (p.echelle * norme2(None, &finalise.x));
            if !residu_rel.is_finite() {
                return Err("Résidu final non fini".into());
            }
            return Ok(Sortie {
                valeurs: valeurs.iter().map(|v| signe * v).collect(),
                vecteur: finalise.x,
                residu_abs,
                residu_rel,
                iterations: iteration,
                converge: maximum <= p.tol && residu_rel <= p.tol,
                historique,
            });
        }
        // Verrouillage souple : les colonnes déjà convergées restent dans X.
        let nouveaux: Vec<_> = residus
            .into_iter()
            .zip(&relatifs)
            .filter_map(|(r, &rel)| (rel > p.tol).then_some(r))
            .collect();
        let w: Vec<_> = pool.install(|| {
            nouveaux
                .into_par_iter()
                .map(|r| {
                    let w = match prec {
                        Some(t) => projeter(proj, &projeter(t, &r)),
                        None => projeter(proj, &r),
                    };
                    images(a, b, proj, w, signe)
                })
                .collect()
        });
        let mut candidats = x.clone();
        candidats.extend(w.iter().cloned());
        candidats.extend(
            directions
                .iter()
                .zip(&relatifs)
                .filter(|(_, rel)| **rel > p.tol)
                .map(|(direction, _)| direction.clone()),
        );
        let base = match orthonormaliser(pool, candidats) {
            Ok(base) => base,
            Err(_) => {
                let mut candidats = x.clone();
                candidats.extend(w.iter().cloned());
                match orthonormaliser(pool, candidats) {
                    Ok(base) => base,
                    Err(_) => {
                        // Dans un petit espace ou près d'une multiplicité,
                        // plusieurs résidus peuvent être dépendants. Retirer
                        // seulement ces colonnes évite un redémarrage immobile.
                        let mut base = orthonormaliser(pool, x.clone())?;
                        for colonne in w {
                            let mut essai = base.clone();
                            essai.push(colonne);
                            if let Ok(completee) = orthonormaliser(pool, essai) {
                                base = completee;
                            }
                        }
                        base
                    }
                }
            }
        };
        // P est la partie hors de X de la combinaison de Ritz dans la base
        // d'origine. Cholesky-QR conserve le drapeau des sous-espaces emboîtés :
        // ses premières colonnes engendrent exactement l'ancien X.
        let (nouvelles_valeurs, nouveau_x, v) = ritz(pool, &base, p.bloc)?;
        directions = if base.len() > p.bloc {
            pool.install(|| {
                (0..p.bloc)
                    .into_par_iter()
                    .map(|j| {
                        let coefficients: Vec<_> = (p.bloc..base.len())
                            .map(|i| v[i * base.len() + j])
                            .collect();
                        combinaison(&base[p.bloc..], &coefficients)
                    })
                    .collect()
            })
        } else {
            Vec::new()
        };
        valeurs = nouvelles_valeurs;
        x = nouveau_x;
    }
    unreachable!("La boucle rend toujours une sortie")
}
