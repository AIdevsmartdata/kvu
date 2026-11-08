//! Instrument de réponse WP22-B : différences centrées et agrégats déterministes.
use super::bloc::{coordonnees, exp_q, log_q, site, valider_champ};
use super::lissage::{bloquer_operateur, OptionsOperateur, Variante};
use super::operateur::Operateur;
use super::support;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub const SCHEMA: &str = "kvu-rg-reponse/1";
pub const PAS_DEFAUT: f64 = 1e-3;
pub const LIENS_BLOC: usize = 324;
pub const LARGEUR_CLASSE: f64 = 0.5;
pub const U: f64 = f64::EPSILON / 2.0;
pub type Matrice = [[f64; 3]; 3];

/// Ordre : composante de sortie, puis axe perturbé.
#[derive(Clone, Debug)]
pub struct Mesure {
    pub matrices: Vec<Matrice>,
    pub nulles_exactes: Vec<bool>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Paire {
    pub lien_fin: usize,
    pub lien_grossier: usize,
    pub d_p: u32,
    pub r: f64,
    pub norme: f64,
    pub erreur: f64,
    pub nulle_exacte: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct Agregat {
    pub n_paires: usize,
    pub n_nulles_exactes: usize,
    pub max: f64,
    pub moyenne: f64,
    pub mediane: f64,
    pub q90: f64,
    pub erreur_estimee_max: f64,
}

pub fn liens_bloc(l: usize, k: usize) -> Resultat<Vec<usize>> {
    if !(1..=LIENS_BLOC).contains(&k) || l < 3 || !l.is_multiple_of(3) {
        return Err(Erreur::Config(
            "L multiple de 3 ≥ 3 et --echantillon dans [1, 324] requis".into(),
        ));
    }
    let mut liens = Vec::with_capacity(LIENS_BLOC);
    for mu in 0..4 {
        for i in 0..81 {
            liens.push(mu * l.pow(4) + site(coordonnees(i, 3), l));
        }
    }
    Ok((0..k).map(|j| liens[j * LIENS_BLOC / k]).collect())
}
pub fn valider_pas(h: f64) -> Resultat<()> {
    if !h.is_finite() || h <= 0.0 || h / 4.0 == 0.0 || !(8.0 * U / h).is_finite() {
        return Err(Erreur::Config(
            "--pas doit être strictement positif, fini et représentable à h/4".into(),
        ));
    }
    Ok(())
}
pub fn norme(m: &Matrice) -> f64 {
    m.iter().flatten().map(|x| x * x).sum::<f64>().sqrt()
}
pub fn ecart(a: &Matrice, b: &Matrice) -> f64 {
    let d = std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] - b[i][j]));
    norme(&d)
}
fn calcul(c: &ChampHote, op: &Operateur, variante: Variante) -> Resultat<ChampHote> {
    let v = bloquer_operateur(c, op, &OptionsOperateur { fils: 1, variante })?.grossier;
    if v.q.iter().any(|x| !x.is_finite()) {
        return Err(Erreur::Validation("RG-QL-REP : sortie non finie".into()));
    }
    Ok(v)
}
/// Toutes les sorties perturbées sont des champs grossiers complets ; l'opérateur utilise un seul fil.
pub fn mesurer(
    c: &ChampHote,
    v0: &ChampHote,
    op: &Operateur,
    variante: Variante,
    li: usize,
    pas: &[f64],
) -> Resultat<Mesure> {
    let n = 4 * v0.v();
    let mut matrices = vec![[[0.0; 3]; 3]; n * pas.len()];
    let mut nulles_exactes = vec![true; n];
    let mut perturbe = c.clone();
    for (ih, &h) in pas.iter().enumerate() {
        valider_pas(h)?;
        for a in 0..3 {
            let mut deltas = vec![[[0.0; 3]; 2]; n];
            for (is, signe) in [1.0, -1.0].into_iter().enumerate() {
                let mut axe = [0.0; 3];
                axe[a] = signe * h;
                perturbe.q[4 * li..4 * li + 4].copy_from_slice(&qmul(exp_q(axe), c.lien(li)));
                let vp = calcul(&perturbe, op, variante)?;
                for j in 0..n {
                    let q = vp.lien(j);
                    nulles_exactes[j] &= q
                        .iter()
                        .zip(v0.lien(j))
                        .all(|(x, y)| x.to_bits() == y.to_bits());
                    deltas[j][is] = log_q(qmul(q, qdag(v0.lien(j)))).0;
                }
            }
            for j in 0..n {
                for b in 0..3 {
                    matrices[ih * n + j][b][a] = (deltas[j][0][b] - deltas[j][1][b]) / (2.0 * h);
                }
            }
        }
    }
    if matrices.iter().flatten().flatten().any(|x| !x.is_finite()) {
        return Err(Erreur::Config("réponse non finie : pas inadéquat".into()));
    }
    // Un zéro au bit est rendu comme +0.0, indépendamment de l'arrondi de V₀ V₀†.
    for (j, &nulle) in nulles_exactes.iter().enumerate() {
        if nulle {
            for ih in 0..pas.len() {
                matrices[ih * n + j] = [[0.0; 3]; 3];
            }
        }
    }
    Ok(Mesure {
        matrices,
        nulles_exactes,
    })
}
/// Pool externe sur les liens fins ; collecte dans l'ordre des indices d'entrée.
pub fn table(
    c: &ChampHote,
    op: &Operateur,
    variante: Variante,
    liens: &[usize],
    pas: &[f64],
    fils: usize,
) -> Resultat<Vec<Mesure>> {
    valider_champ(c)?;
    if c.l < 3
        || !c.l.is_multiple_of(3)
        || fils == 0
        || pas.is_empty()
        || liens.iter().any(|&li| li >= 4 * c.v())
    {
        return Err(Erreur::Config(
            "géométrie, liens, pas ou nombre de fils invalides".into(),
        ));
    }
    for &h in pas {
        valider_pas(h)?;
    }
    let v0 = calcul(c, op, variante)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(fils)
        .build()
        .map_err(|e| Erreur::Config(format!("pool réponse : {e}")))?;
    pool.install(|| {
        liens
            .par_iter()
            .map(|&li| mesurer(c, &v0, op, variante, li, pas))
            .collect()
    })
}

pub fn attache(op: &Operateur, z: [usize; 4]) -> [usize; 4] {
    z.map(|x| 3 * x + usize::from(op.est_trois_centre()))
}
pub fn chemin(op: &Operateur, l: usize, z: [usize; 4], mu: usize) -> [usize; 3] {
    if !op.est_trois_centre() {
        return support::chemin_droit(l, z, mu);
    }
    let a = attache(op, z);
    std::array::from_fn(|k| {
        let mut x = a;
        x[mu] = (x[mu] + k) % l;
        mu * l.pow(4) + site(x, l)
    })
}
/// Distance du milieu du lien fin au segment grossier, sur le tore.
pub fn distance_euclidienne(l: usize, li: usize, a: [usize; 4], mu: usize) -> f64 {
    let v = l.pow(4);
    let nu = li / v;
    let x = coordonnees(li % v, l);
    (0..4)
        .map(|d| {
            let delta = x[d] as f64 + if d == nu { 0.5 } else { 0.0 } - a[d] as f64;
            let minimal = (delta + l as f64 / 2.0).rem_euclid(l as f64) - l as f64 / 2.0;
            let t = if d == mu {
                [-1.0, 0.0, 1.0]
                    .into_iter()
                    .map(|k| {
                        let q = minimal + k * l as f64;
                        (q - q.clamp(0.0, 3.0)).abs()
                    })
                    .fold(f64::INFINITY, f64::min)
            } else {
                minimal.abs()
            };
            t * t
        })
        .sum::<f64>()
        .sqrt()
}
pub fn agreger(paires: &[&Paire]) -> Agregat {
    assert!(!paires.is_empty());
    let mut valeurs: Vec<f64> = paires.iter().map(|p| p.norme).collect();
    valeurs.sort_by(f64::total_cmp);
    let n = valeurs.len();
    Agregat {
        n_paires: n,
        n_nulles_exactes: paires.iter().filter(|p| p.nulle_exacte).count(),
        max: valeurs[n - 1],
        moyenne: valeurs.iter().sum::<f64>() / n as f64,
        mediane: valeurs[(n - 1) / 2],
        q90: valeurs[(9 * n).div_ceil(10) - 1],
        erreur_estimee_max: paires.iter().map(|p| p.erreur).fold(0.0, f64::max),
    }
}
#[derive(Clone, Copy, Debug)]
pub struct OptionsReponse {
    pub hors_famille: bool,
    pub echantillon: usize,
    pub pas: f64,
    pub fils: usize,
    pub detail: bool,
}
pub fn document(
    c: &ChampHote,
    op: &Operateur,
    options: &OptionsReponse,
    sha: &str,
) -> Resultat<Value> {
    let OptionsReponse {
        hors_famille,
        echantillon: k,
        pas: h,
        fils,
        detail,
    } = *options;
    op.verifier_famille(hors_famille)?;
    valider_pas(h)?;
    let liens = liens_bloc(c.l, k)?;
    let mesures = table(c, op, Variante::Normale, &liens, &[h, h / 2.0], fils)?;
    let lc = c.l / 3;
    let vc = lc.pow(4);
    let n = 4 * vc;
    let mut paires = Vec::with_capacity(k * n);
    for (&li, mesure) in liens.iter().zip(mesures) {
        let distances = support::distances(c.l, &[li], usize::MAX);
        for j in 0..n {
            let mu = j / vc;
            let z = coordonnees(j % vc, lc);
            let d_p = chemin(op, c.l, z, mu)
                .iter()
                .map(|&i| distances[i])
                .min()
                .unwrap();
            let r = distance_euclidienne(c.l, li, attache(op, z), mu);
            paires.push(Paire {
                lien_fin: li,
                lien_grossier: j,
                d_p,
                r,
                norme: norme(&mesure.matrices[n + j]),
                erreur: ecart(&mesure.matrices[j], &mesure.matrices[n + j]) / 3.0 + 8.0 * U / h,
                nulle_exacte: mesure.nulles_exactes[j],
            });
        }
    }
    let mut dp: BTreeMap<u32, Vec<&Paire>> = BTreeMap::new();
    let mut r: BTreeMap<usize, Vec<&Paire>> = BTreeMap::new();
    for p in &paires {
        dp.entry(p.d_p).or_default().push(p);
        r.entry((p.r / LARGEUR_CLASSE).floor() as usize)
            .or_default()
            .push(p);
    }
    let mut doc = json!({ "format": SCHEMA, "operateur": op.parametres_json(), "hors_famille": hors_famille,
        "support": {"rayon": op.rayon_support(), "definition": support::definition(op)},
        "pas": h, "l": c.l, "sha256_entree": sha, "n_liens_fins": k, "n_liens_grossiers": n,
        "par_d_p": dp.into_iter().map(|(distance, ps)| json!({"d_p": distance, "statistiques": agreger(&ps)})).collect::<Vec<_>>(),
        "par_r": r.into_iter().map(|(classe, ps)| json!({"classe": classe, "r_min": classe as f64 * LARGEUR_CLASSE, "r_max": (classe + 1) as f64 * LARGEUR_CLASSE, "statistiques": agreger(&ps)})).collect::<Vec<_>>(),
        "version": crate::cli::version_texte().replace('\n', " | ") });
    if detail {
        doc["paires"] = json!(paires);
    }
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;
    pub(super) fn froid(l: usize) -> ChampHote {
        let mut q = vec![0.0; 16 * l.pow(4)];
        for lien in q.chunks_exact_mut(4) {
            lien[0] = 1.0;
        }
        ChampHote { l, q }
    }
    #[test]
    fn distances_segment_et_images_minimales() {
        let li = |x, nu| nu * 9usize.pow(4) + site(x, 9);
        assert_eq!(distance_euclidienne(9, li([0; 4], 0), [0; 4], 0), 0.0);
        assert_eq!(distance_euclidienne(9, li([8, 0, 0, 0], 0), [0; 4], 0), 0.5);
        assert_eq!(distance_euclidienne(9, li([0, 8, 0, 0], 1), [0; 4], 0), 0.5);
        assert_eq!(distance_euclidienne(9, li([4, 0, 0, 0], 0), [0; 4], 0), 1.5);
        assert_eq!(
            distance_euclidienne(9, li([8, 0, 0, 0], 0), [6, 0, 0, 0], 0),
            0.0
        );
        assert_eq!(
            chemin(&Operateur::TroisCentre, 9, [0; 4], 0)[0],
            site([1; 4], 9)
        );
    }
    #[test]
    fn quantiles_et_echantillon_canoniques() {
        let p: Vec<_> = (1..=10)
            .map(|i| Paire {
                lien_fin: 0,
                lien_grossier: 0,
                d_p: 0,
                r: 0.0,
                norme: i as f64,
                erreur: i as f64 / 10.0,
                nulle_exacte: i == 1,
            })
            .collect();
        let a = agreger(&p.iter().collect::<Vec<_>>());
        assert_eq!((a.mediane, a.q90, a.moyenne, a.max), (5.0, 9.0, 5.5, 10.0));
        assert_eq!(a.n_nulles_exactes, 1);
        let b = liens_bloc(9, 324).unwrap();
        let e = liens_bloc(9, 16).unwrap();
        assert!(b.windows(2).all(|w| w[0] < w[1]));
        for j in 0..16 {
            assert_eq!(e[j], b[j * 324 / 16]);
        }
        assert!(liens_bloc(9, 0).is_err() && liens_bloc(9, 325).is_err());
        for h in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(valider_pas(h).is_err());
        }
    }
    #[test]
    fn dec_froid_matrices_et_zeros_au_bit() {
        let c = froid(9);
        let liens = liens_bloc(9, 4).unwrap();
        let t = table(
            &c,
            &Operateur::Dec,
            Variante::Normale,
            &liens,
            &[PAS_DEFAUT, PAS_DEFAUT / 2.0],
            2,
        )
        .unwrap();
        let vc = 81;
        for (&li, m) in liens.iter().zip(t) {
            for j in 0..324 {
                let dedans =
                    chemin(&Operateur::Dec, 9, coordonnees(j % vc, 3), j / vc).contains(&li);
                assert_eq!(m.nulles_exactes[j], !dedans);
                let n = norme(&m.matrices[324 + j]);
                if dedans {
                    assert!((n - 3.0f64.sqrt()).abs() < 1e-9);
                } else {
                    assert_eq!(n.to_bits(), 0.0f64.to_bits());
                }
            }
        }
    }
}
