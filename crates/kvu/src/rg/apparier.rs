//! `kvu rg apparier` : appariement Monte Carlo du groupe de renormalisation (MCRG, méthode de Swendsen à deux réseaux).
//!
//! Principe. Pour un β_haut donné, on bloque `n` fois (facteur `b`, carte de Swendsen `bloquer_b2` de coefficient `c`) les
//! configurations d'un ensemble de Wilson à β_haut sur un réseau fin de côté L ; on mesure sur le réseau bloqué les
//! observables O = (plaquette P, rectangle 1×2 R, chaise C), moyennes de ½ Tr sur toutes les boucles de chaque type. Sur des
//! ensembles de Wilson de côté L/b à une grille de β′, bloqués `n − 1` fois, on mesure les mêmes observables, de sorte
//! que les deux réseaux de comparaison aient le MÊME côté final. On cherche β′ tel que O_gros(β′) = O_fin(β_haut). Le pas du
//! flot de Wilson est Δβ = β_haut − β′.
//!
//! Estimation. Pour chaque observable, ajustement polynomial (degré ≤ 2) de la moyenne grossière en β′ sur la grille, puis
//! inversion. Les trois solutions (P, R, C) sont combinées par moyenne pondérée (poids = 1 / variance bootstrap de chacune,
//! figés sur l'échantillon complet) ; leur dispersion est rapportée comme incertitude de schéma (si la mesure bloquée n'est
//! pas celle d'une action de Wilson, P, R, C ne donnent pas le même β′). Les erreurs viennent du bootstrap de configurations
//! stratifié par chaîne du module `rg::action::bootstrap`, appliqué À LA FOIS à l'ensemble fin et à chaque ensemble de la
//! grille (l'incertitude de l'ajustement grossier est donc incluse).
//!
//! ```text
//! kvu rg apparier --fin DIR [--filtre-fin MOTIF] --grossiers DIR [--filtre-gros MOTIF] [--b 2] [--c 1.1] [--pas 2]
//!                 [--bootstrap-b 400] [--bootstrap-bloc 1] [--graine S] [--sortie DIR] [--fils N]
//! ```
//! `--b 1` : blocage identité (porte de cohérence : deux ensembles de Wilson de même côté, Δβ doit être nul).

use super::action::bootstrap::{ecart_type, plans_bootstrap, resumer};
use super::action::boucles::{produit_chemin, TableFormes, TypeBoucle};
use super::action::cli::{decouvrir_chaines, ChaineTrouvee};
use super::domaine::ecrire_json;
use super::lissage::Voisinage;
use super::swendsen::bloquer_b2;
use crate::commun::{code, ChampHote, Erreur, Resultat};
use crate::heritage::manifest::sha256_fichier;
use crate::stockage::npy::lire_champ;
use crate::stockage::provenance::ecrire_manifeste_fichier;
use clap::{Arg, ArgMatches, Command};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Moteur consigné dans les manifestes de sortie.
const MOTEUR: &str = "kvu-rg-apparier";
pub const SCHEMA_APPARIER: &str = "kvu-rg-apparier/1";
/// Observables retenues : plaquette, rectangle, chaise.
pub const NOMS_OBS: [&str; 3] = ["P", "R", "C"];
/// Côté minimal du réseau final sur lequel on mesure (boucles de la table : étendue ≤ 2).
pub const L_FINAL_MIN: usize = 4;
/// Variantes (degré, demi-fenêtre) de l'erreur d'interpolation (amendement 1 du préenregistrement).
pub const VARIANTES_INTERP: [(usize, f64); 4] = [(2, 0.25), (2, 0.35), (3, 0.35), (3, 0.5)];

// ---------------------------------------------------------------------------------------------
// Observables et blocage
// ---------------------------------------------------------------------------------------------

/// (P, R, C) : moyenne de ½ Tr U_boucle sur toutes les boucles (tous sites, orientations et formes) de chaque type.
pub fn observables(champ: &ChampHote, table: &TableFormes) -> Resultat<[f64; 3]> {
    if champ.l < L_FINAL_MIN {
        return Err(Erreur::Config(format!(
            "observables : L = {} < {L_FINAL_MIN}",
            champ.l
        )));
    }
    if champ.q.iter().any(|x| !x.is_finite()) {
        return Err(Erreur::Config("observables : champ non fini".into()));
    }
    let vois = Voisinage::nouveau(champ.l);
    let v = champ.v();
    let types = [TypeBoucle::P, TypeBoucle::R, TypeBoucle::C];
    let mut somme = [0.0f64; 3];
    let mut compte = [0u64; 3];
    for mu in 0..4 {
        for s in 0..v {
            for (k, &ty) in types.iter().enumerate() {
                for forme in table.formes(mu, ty) {
                    somme[k] += produit_chemin(champ, &vois, s, forme)[0];
                    compte[k] += 1;
                }
            }
        }
    }
    Ok(std::array::from_fn(|k| somme[k] / compte[k] as f64))
}

/// `n` pas de blocage de facteur `b` ∈ {1, 2} (b = 1 : identité ; b = 2 : Swendsen de coefficient `c`).
pub fn bloquer_n(champ: &ChampHote, b: usize, c: f64, n: usize) -> Resultat<ChampHote> {
    let mut u = champ.clone();
    match b {
        1 => Ok(u),
        2 => {
            for _ in 0..n {
                u = bloquer_b2(&u, c)?;
            }
            Ok(u)
        }
        _ => Err(Erreur::Config(format!(
            "--b {b} : seuls 1 (identité) et 2 (Swendsen) sont implémentés"
        ))),
    }
}

/// Côté après `n` blocages de facteur `b`.
pub fn cote_final(l: usize, b: usize, n: usize) -> Result<usize, String> {
    if b == 1 {
        return Ok(l);
    }
    let mut l2 = l;
    for _ in 0..n {
        if !l2.is_multiple_of(b) {
            return Err(format!("L = {l} non divisible par {b}^{n}"));
        }
        l2 /= b;
    }
    Ok(l2)
}

// ---------------------------------------------------------------------------------------------
// Statistique : échantillons, ajustement, inversion, bootstrap
// ---------------------------------------------------------------------------------------------

/// Observables par configuration, par chaîne.
#[derive(Clone, Debug)]
pub struct Echantillon {
    pub chaines: Vec<Vec<[f64; 3]>>,
}

impl Echantillon {
    pub fn effectifs(&self) -> Vec<usize> {
        self.chaines.iter().map(|c| c.len()).collect()
    }
    pub fn plat(&self) -> Vec<[f64; 3]> {
        self.chaines.iter().flatten().copied().collect()
    }
    pub fn moyenne(&self) -> [f64; 3] {
        moyenne_plan(&self.plat(), None)
    }
}

fn moyenne_plan(vals: &[[f64; 3]], plan: Option<&[usize]>) -> [f64; 3] {
    let mut s = [0.0; 3];
    let n = match plan {
        Some(p) => {
            for &i in p {
                for k in 0..3 {
                    s[k] += vals[i][k];
                }
            }
            p.len()
        }
        None => {
            for v in vals {
                for k in 0..3 {
                    s[k] += v[k];
                }
            }
            vals.len()
        }
    };
    s.map(|x| x / n as f64)
}

/// Polynôme y = Σ_k c_k (x − x0)^k.
#[derive(Clone, Debug)]
pub struct Poly {
    pub x0: f64,
    pub c: Vec<f64>,
}

impl Poly {
    pub fn eval(&self, x: f64) -> f64 {
        let t = x - self.x0;
        self.c.iter().rev().fold(0.0, |a, &ck| a * t + ck)
    }
}

/// Moindres carrés non pondérés de degré `deg` (≤ points − 1), équations normales centrées, élimination de Gauss avec pivot.
#[allow(clippy::needless_range_loop)] // indices explicites : l'ordre des opérations flottantes est préenregistré
pub fn ajuster(x: &[f64], y: &[f64], deg: usize) -> Resultat<Poly> {
    let m = x.len();
    if m != y.len() || m == 0 {
        return Err(Erreur::Config("ajustement : tailles incohérentes".into()));
    }
    let deg = deg.min(m - 1);
    let x0 = x.iter().sum::<f64>() / m as f64;
    let d = deg + 1;
    let mut a = vec![vec![0.0f64; d + 1]; d];
    for (&xi, &yi) in x.iter().zip(y) {
        let t = xi - x0;
        for r in 0..d {
            for s in 0..d {
                a[r][s] += t.powi((r + s) as i32);
            }
            a[r][d] += yi * t.powi(r as i32);
        }
    }
    for col in 0..d {
        let piv = (col..d)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())
            .unwrap();
        if a[piv][col].abs() < 1e-300 {
            return Err(Erreur::Config(
                "ajustement : système singulier (points confondus ?)".into(),
            ));
        }
        a.swap(col, piv);
        for r in 0..d {
            if r != col {
                let f = a[r][col] / a[col][col];
                for s in col..=d {
                    a[r][s] -= f * a[col][s];
                }
            }
        }
    }
    Ok(Poly {
        x0,
        c: (0..d).map(|r| a[r][d] / a[r][r]).collect(),
    })
}

/// Racine de poly(β′) = cible dans [lo, hi] par bissection (signe opposé aux bornes exigé), None sinon.
pub fn inverser(p: &Poly, cible: f64, lo: f64, hi: f64) -> Option<f64> {
    let (mut a, mut b) = (lo, hi);
    let (mut fa, fb) = (p.eval(a) - cible, p.eval(b) - cible);
    if !(fa.is_finite() && fb.is_finite()) {
        return None;
    }
    // Racines exactement aux bornes : rendues telles quelles (la bissection ci-dessous les perdrait, `fa == 0` ne bornant rien).
    if fa == 0.0 {
        return Some(a);
    }
    if fb == 0.0 {
        return Some(b);
    }
    // Comparaison de SIGNES (le produit fa·fb peut sous-dépasser vers 0 et masquer un intervalle sans racine).
    if (fa < 0.0) == (fb < 0.0) {
        return None;
    }
    for _ in 0..200 {
        let m = 0.5 * (a + b);
        let fm = p.eval(m) - cible;
        if fm == 0.0 {
            return Some(m);
        }
        if (fa < 0.0) != (fm < 0.0) {
            b = m;
        } else {
            a = m;
            fa = fm;
        }
    }
    Some(0.5 * (a + b))
}

/// Solution sur un jeu de moyennes : (β′_P, β′_R, β′_C), None si une inversion échoue.
/// La fenêtre de recherche déborde la grille de `marge` × son étendue (extrapolation signalée à part).
fn resoudre_moyennes(
    betas: &[f64],
    moy_gros: &[[f64; 3]],
    cible: [f64; 3],
    degre: usize,
    marge: f64,
) -> Option<[f64; 3]> {
    let lo = betas.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = betas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w = hi - lo;
    let mut out = [0.0; 3];
    for k in 0..3 {
        let y: Vec<f64> = moy_gros.iter().map(|m| m[k]).collect();
        let p = ajuster(betas, &y, degre).ok()?;
        out[k] = inverser(&p, cible[k], lo - marge * w, hi + marge * w)?;
    }
    Some(out)
}

#[derive(Clone, Debug)]
pub struct Appariement {
    pub beta_haut: f64,
    /// β′ de chaque observable (P, R, C) sur l'échantillon complet, et leurs erreurs bootstrap.
    pub beta_obs: [f64; 3],
    pub err_obs: [f64; 3],
    /// Combinaison pondérée et son erreur bootstrap.
    pub beta_comb: f64,
    pub err_comb: f64,
    pub poids: [f64; 3],
    /// Dispersion de schéma : max − min des trois β′ (échantillon complet).
    pub dispersion: f64,
    /// Tension P/R/C : max sur les paires (β′_i − β′_j) / σ(β′_i − β′_j) (σ par bootstrap).
    pub z_max_paires: f64,
    pub ic95_comb: [f64; 2],
    pub replicats_valides: usize,
    pub replicats_total: usize,
    pub hors_grille: bool,
    pub degre: usize,
}

impl Appariement {
    pub fn delta_beta(&self) -> f64 {
        self.beta_haut - self.beta_comb
    }
}

/// Degré maximal de l'ajustement en β′ (au-delà : équations normales mal conditionnées, aucun usage préenregistré).
pub const DEGRE_MAX: usize = 8;

/// Valide `--degre` (1 ≤ degré ≤ [`DEGRE_MAX`]) ; rend le degré.
pub fn valider_degre(degre: usize) -> Resultat<usize> {
    if (1..=DEGRE_MAX).contains(&degre) {
        Ok(degre)
    } else {
        Err(Erreur::Config(format!(
            "--degre {degre} : entier entre 1 et {DEGRE_MAX} requis"
        )))
    }
}

/// Appariement complet avec bootstrap (stratifié par chaîne, indépendant pour le fin et chaque point de la grille).
#[allow(clippy::too_many_arguments)] // signature publique préenregistrée (appelée telle quelle par les tests et les portes)
pub fn apparier(
    beta_haut: f64,
    fin: &Echantillon,
    gros: &[(f64, Echantillon)],
    degre: usize,
    fenetre: f64,
    b_boot: usize,
    bloc: usize,
    graine: u64,
) -> Resultat<Appariement> {
    if gros.len() < 2 {
        return Err(Erreur::Config(
            "appariement : au moins 2 valeurs de β′ dans la grille".into(),
        ));
    }
    if gros.iter().any(|g| !g.0.is_finite()) {
        return Err(Erreur::Config(
            "appariement : β′ non fini dans la grille".into(),
        ));
    }
    if !(beta_haut.is_finite() && fenetre.is_finite() && fenetre >= 0.0) {
        return Err(Erreur::Config(
            "appariement : β_haut fini et fenêtre finie ≥ 0 requis".into(),
        ));
    }
    let besoin = valider_degre(degre)? + 2;
    if fin
        .chaines
        .iter()
        .chain(gros.iter().flat_map(|g| g.1.chaines.iter()))
        .flatten()
        .flatten()
        .any(|x| !x.is_finite())
    {
        return Err(Erreur::Config(
            "appariement : observable non finie dans un ensemble".into(),
        ));
    }
    let mut gros: Vec<(f64, Echantillon)> = gros.to_vec();
    gros.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let betas_tous: Vec<f64> = gros.iter().map(|g| g.0).collect();
    let betas = betas_tous.clone();
    if betas.windows(2).any(|w| w[1] - w[0] < 1e-9) {
        return Err(Erreur::Config(
            "appariement : β′ en double dans la grille".into(),
        ));
    }
    let marge = 0.25;
    let cible0 = fin.moyenne();
    // Fenêtre locale (fenetre > 0) : pilote sur toute la grille, puis on ne garde que les points à moins de `fenetre` du pilote
    // (indices FIGÉS pour tous les réplicats). Le biais d'interpolation d'un polynôme global sur une grille large était la cause
    // du rouge des portes G1/G2 (amendement 1).
    if fenetre > 0.0 {
        let moy_t: Vec<[f64; 3]> = gros.iter().map(|g| g.1.moyenne()).collect();
        // Pilote ROBUSTE : croisement par interpolation linéaire par morceaux (un polynôme global sur une grille large peut ne pas
        // croiser la cible). Premier croisement trouvé par observable, moyenné.
        let mut centres = Vec::new();
        for k in 0..3 {
            let mut trouve = false;
            for j in 0..gros.len() - 1 {
                let (y0, y1) = (moy_t[j][k] - cible0[k], moy_t[j + 1][k] - cible0[k]);
                if y0 == 0.0 || y0 * y1 < 0.0 {
                    let (x0, x1) = (betas_tous[j], betas_tous[j + 1]);
                    centres.push(x0 + (x1 - x0) * y0 / (y0 - y1));
                    trouve = true;
                    break;
                }
            }
            // Égalité exacte au DERNIER point de grille : la boucle ci-dessus ne regarde y0 que sur les points qui ont un suivant.
            if !trouve && moy_t[gros.len() - 1][k] == cible0[k] {
                centres.push(betas_tous[gros.len() - 1]);
            }
        }
        if centres.is_empty() {
            return Err(Erreur::Config(
                "appariement : pas de croisement pilote dans la grille".into(),
            ));
        }
        let centre = centres.iter().sum::<f64>() / centres.len() as f64;
        gros.retain(|g| (g.0 - centre).abs() <= fenetre);
        if gros.len() < besoin {
            return Err(Erreur::Config(format!("appariement : {} points de grille dans la fenêtre ±{fenetre} autour de {centre:.4} (≥ {besoin} requis)", gros.len())));
        }
    }
    let betas: Vec<f64> = gros.iter().map(|g| g.0).collect();
    let moy0: Vec<[f64; 3]> = gros.iter().map(|g| g.1.moyenne()).collect();
    let sol0 = resoudre_moyennes(&betas, &moy0, cible0, degre, marge)
        .ok_or_else(|| Erreur::Config(format!("appariement : pas de solution dans la fenêtre de la grille (cible {cible0:?}, β′ ∈ {betas:?}) : élargir la grille")))?;
    let hors = sol0
        .iter()
        .any(|&b| b < betas_tous[0] - 1e-12 || b > *betas_tous.last().unwrap() + 1e-12);

    let plat_fin = fin.plat();
    let plans_fin = plans_bootstrap(&fin.effectifs(), bloc, b_boot, graine);
    let plats: Vec<Vec<[f64; 3]>> = gros.iter().map(|g| g.1.plat()).collect();
    let plans_gros: Vec<Vec<Vec<usize>>> = gros
        .iter()
        .enumerate()
        .map(|(j, g)| {
            plans_bootstrap(
                &g.1.effectifs(),
                bloc,
                b_boot,
                graine.wrapping_add(1 + j as u64),
            )
        })
        .collect();
    let reps: Vec<Option<[f64; 3]>> = (0..b_boot)
        .into_par_iter()
        .map(|r| {
            let cible = moyenne_plan(&plat_fin, Some(&plans_fin[r]));
            let moy: Vec<[f64; 3]> = (0..gros.len())
                .map(|j| moyenne_plan(&plats[j], Some(&plans_gros[j][r])))
                .collect();
            resoudre_moyennes(&betas, &moy, cible, degre, marge)
        })
        .collect();
    let valides: Vec<[f64; 3]> = reps.iter().flatten().copied().collect();
    if valides.len() < 20 {
        return Err(Erreur::Config(format!(
            "appariement : seulement {} réplicats valides sur {b_boot}",
            valides.len()
        )));
    }
    let col = |k: usize| -> Vec<f64> { valides.iter().map(|r| r[k]).collect() };
    let err_obs: [f64; 3] = std::array::from_fn(|k| ecart_type(&col(k)));
    // Variance bootstrap nulle ou non finie ⇒ poids 1/σ² infinis puis ∞/∞ = NaN dans la combinaison : ajustement non identifiable.
    for k in 0..3 {
        if !(err_obs[k].is_finite() && err_obs[k] > 0.0) {
            return Err(Erreur::Config(format!(
                "appariement : écart-type bootstrap de β′_{} = {} (fini et > 0 requis pour pondérer P, R, C) : ajustement non identifiable (échantillons constants, trop peu de configurations ou fenêtre trop étroite ?)",
                NOMS_OBS[k], err_obs[k]
            )));
        }
    }
    let poids: [f64; 3] = std::array::from_fn(|k| 1.0 / (err_obs[k] * err_obs[k]));
    let psum: f64 = poids.iter().sum();
    let comb = |s: &[f64; 3]| (0..3).map(|k| poids[k] * s[k]).sum::<f64>() / psum;
    let beta_comb = comb(&sol0);
    let reps_comb: Vec<f64> = valides.iter().map(comb).collect();
    let st = resumer(beta_comb, &reps_comb);
    let mut z_max = 0.0f64;
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let diffs: Vec<f64> = valides.iter().map(|r| r[i] - r[j]).collect();
        let s = ecart_type(&diffs);
        // Deux observables aux solutions identiques à chaque réplicat : tension nulle (0/0 ignoré par `max`, comme avant).
        if s == 0.0 && sol0[i] == sol0[j] {
            continue;
        }
        if !(s.is_finite() && s > 0.0) {
            return Err(Erreur::Config(format!(
                "appariement : écart-type bootstrap de β′_{} − β′_{} = {s} (fini et > 0 requis pour la tension P/R/C)",
                NOMS_OBS[i], NOMS_OBS[j]
            )));
        }
        z_max = z_max.max(((sol0[i] - sol0[j]) / s).abs());
    }
    let dispersion = sol0.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - sol0.iter().cloned().fold(f64::INFINITY, f64::min);
    let sortis = [
        beta_comb, st.erreur, st.ic95[0], st.ic95[1], dispersion, z_max,
    ]
    .into_iter()
    .chain(sol0)
    .chain(poids);
    if sortis.into_iter().any(|x| !x.is_finite()) {
        return Err(Erreur::Config(
            "appariement : résultat non fini (NaN ou infini) : refus de le consigner".into(),
        ));
    }
    Ok(Appariement {
        beta_haut,
        beta_obs: sol0,
        err_obs,
        beta_comb,
        err_comb: st.erreur,
        poids,
        dispersion,
        z_max_paires: z_max,
        ic95_comb: st.ic95,
        replicats_valides: valides.len(),
        replicats_total: b_boot,
        hors_grille: hors,
        degre,
    })
}

// ---------------------------------------------------------------------------------------------
// Lecture des ensembles
// ---------------------------------------------------------------------------------------------

fn mesurer_chaine(
    ch: &ChaineTrouvee,
    b: usize,
    c: f64,
    n: usize,
    table: &TableFormes,
) -> Resultat<Vec<[f64; 3]>> {
    ch.configs
        .par_iter()
        .map(|p| {
            let u = lire_champ(p)?;
            if u.l != ch.l {
                return Err(Erreur::Format(format!(
                    "{} : L = {} ≠ {} du manifeste",
                    p.display(),
                    u.l,
                    ch.l
                )));
            }
            observables(&bloquer_n(&u, b, c, n)?, table)
        })
        .collect()
}

/// Groupe les chaînes par β (tolérance 1e-9) et mesure chaque groupe.
fn mesurer_groupes(
    chaines: &[ChaineTrouvee],
    b: usize,
    c: f64,
    n: usize,
    table: &TableFormes,
) -> Resultat<Vec<(f64, usize, Echantillon)>> {
    let mut groupes: Vec<(f64, usize, Vec<&ChaineTrouvee>)> = Vec::new();
    for ch in chaines {
        match groupes.iter_mut().find(|g| (g.0 - ch.beta).abs() < 1e-9) {
            Some(g) => {
                if g.1 != ch.l {
                    return Err(Erreur::Config(format!(
                        "β = {} : côtés mélangés ({} et {})",
                        ch.beta, g.1, ch.l
                    )));
                }
                g.2.push(ch)
            }
            None => groupes.push((ch.beta, ch.l, vec![ch])),
        }
    }
    groupes
        .into_iter()
        .map(|(beta, l, chs)| {
            let chaines = chs
                .iter()
                .map(|ch| mesurer_chaine(ch, b, c, n, table))
                .collect::<Resultat<Vec<_>>>()?;
            Ok((beta, l, Echantillon { chaines }))
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Choix de c (amendement 2) : critère « dispersion P/R/C minimale » — PAS le critère complet de Tomboulis–Velytsky
// ---------------------------------------------------------------------------------------------

/// Point (c, dispersion P/R/C en β′, Δβ, erreur de Δβ) issu d'un `apparier.json`.
#[derive(Clone, Copy, Debug)]
pub struct PointC {
    pub c: f64,
    pub dispersion: f64,
    pub delta: f64,
    pub err: f64,
}

#[derive(Clone, Debug)]
pub struct ChoixC {
    pub c_etoile: f64,
    pub delta_etoile: f64,
    /// Erreur sur Δβ(c*) : interpolation des erreurs ⊕ pente × résolution de c* (`RESOLUTION_C`).
    pub err_delta: f64,
    pub dispersion_min: f64,
    /// Le minimum est au bord de la liste (pas d'extrapolation) ou la parabole n'est pas convexe.
    pub drapeau: Option<&'static str>,
}

/// Incertitude déclarée sur c* (demi-pas de la liste {0,15 ; 0,2 ; 0,25 ; 0,3 ; 0,35}).
pub const RESOLUTION_C: f64 = 0.025;

/// Mode de sélection ; `MutantArgmax` et `MutantDelta` n'existent que pour les tests négatifs.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ModeC {
    Normal,
    #[cfg(test)]
    MutantArgmax,
    #[cfg(test)]
    MutantDelta,
}

pub fn choisir_c_mode(pts: &[PointC], mode: ModeC) -> Resultat<ChoixC> {
    if pts.len() < 3 {
        return Err(Erreur::Config(
            "choix de c : au moins 3 valeurs de c requises".into(),
        ));
    }
    if pts.iter().any(|q| {
        ![q.c, q.dispersion, q.delta, q.err]
            .iter()
            .all(|x| x.is_finite())
            || q.err < 0.0
    }) {
        return Err(Erreur::Config(
            "choix de c : valeur non finie (ou erreur négative) dans un point".into(),
        ));
    }
    let mut p = pts.to_vec();
    p.sort_by(|a, b| a.c.partial_cmp(&b.c).unwrap());
    if p.windows(2).any(|w| w[1].c - w[0].c < 1e-9) {
        return Err(Erreur::Config("choix de c : c en double".into()));
    }
    let cle = |q: &PointC| -> f64 {
        match mode {
            ModeC::Normal => q.dispersion,
            #[cfg(test)]
            ModeC::MutantArgmax => -q.dispersion,
            #[cfg(test)]
            ModeC::MutantDelta => q.delta.abs(),
        }
    };
    let i = (0..p.len())
        .min_by(|&a, &b| cle(&p[a]).partial_cmp(&cle(&p[b])).unwrap())
        .unwrap();
    if i == 0 || i == p.len() - 1 {
        return Ok(ChoixC {
            c_etoile: p[i].c,
            delta_etoile: p[i].delta,
            err_delta: p[i].err,
            dispersion_min: p[i].dispersion,
            drapeau: Some("minimum au bord de la liste de c : c* non encadré"),
        });
    }
    let (a, b, c) = (p[i - 1], p[i], p[i + 1]);
    // parabole de la clé en c (points non forcément équidistants) : sommet de l'interpolation de Lagrange
    let (k0, k1, k2) = (cle(&a), cle(&b), cle(&c));
    let den = (a.c - b.c) * (a.c - c.c) * (b.c - c.c);
    let aa = (c.c * (k1 - k0) + b.c * (k0 - k2) + a.c * (k2 - k1)) / den;
    let bb = (c.c * c.c * (k0 - k1) + b.c * b.c * (k2 - k0) + a.c * a.c * (k1 - k2)) / den;
    let (mut c_et, mut drap) = (b.c, None);
    if aa > 0.0 {
        c_et = (-bb / (2.0 * aa)).clamp(a.c, c.c);
    } else {
        drap = Some("parabole non convexe : c* = point de dispersion minimale");
    }
    // Δβ(c*) par interpolation de Lagrange quadratique, erreur par poids de Lagrange + pente × résolution
    let l = |x: f64| -> [f64; 3] {
        [
            (x - b.c) * (x - c.c) / ((a.c - b.c) * (a.c - c.c)),
            (x - a.c) * (x - c.c) / ((b.c - a.c) * (b.c - c.c)),
            (x - a.c) * (x - b.c) / ((c.c - a.c) * (c.c - b.c)),
        ]
    };
    let w = l(c_et);
    let delta = w[0] * a.delta + w[1] * b.delta + w[2] * c.delta;
    let err_int =
        (w[0] * w[0] * a.err * a.err + w[1] * w[1] * b.err * b.err + w[2] * w[2] * c.err * c.err)
            .sqrt();
    let h = 1e-4;
    let (wp, wm) = (l(c_et + h), l(c_et - h));
    let pente = ((wp[0] - wm[0]) * a.delta + (wp[1] - wm[1]) * b.delta + (wp[2] - wm[2]) * c.delta)
        / (2.0 * h);
    let err = (err_int * err_int + (pente * RESOLUTION_C).powi(2)).sqrt();
    let disp = {
        let wd = w;
        wd[0] * a.dispersion + wd[1] * b.dispersion + wd[2] * c.dispersion
    };
    Ok(ChoixC {
        c_etoile: c_et,
        delta_etoile: delta,
        err_delta: err,
        dispersion_min: disp,
        drapeau: drap,
    })
}

pub fn choisir_c(pts: &[PointC]) -> Resultat<ChoixC> {
    choisir_c_mode(pts, ModeC::Normal)
}

/// Clés de `parametres` qui peuvent légitimement différer d'une entrée de `choisir-c` à l'autre (le seul paramètre balayé).
const PARAMETRES_VARIABLES: [&str; 1] = ["c"];

/// Les entrées de `choisir-c` doivent décrire LE MÊME problème à `c` près : même schéma, mêmes paramètres (volumes, facteur b,
/// nombres de blocages, degré, fenêtre, bootstrap, graine), mêmes ensembles (β_haut, nombres de chaînes et de configurations
/// du fin et de chaque point de la grille). Sans cela, la « courbe en c » mélangerait des problèmes différents.
pub fn verifier_compatibilite(docs: &[(String, Value)]) -> Resultat<()> {
    for (nom, v) in docs {
        if v["format"].as_str() != Some(SCHEMA_APPARIER) {
            return Err(Erreur::Format(format!(
                "{nom} : format « {} » ≠ {SCHEMA_APPARIER}",
                v["format"].as_str().unwrap_or("absent")
            )));
        }
    }
    let Some((nom0, ref0)) = docs.first() else {
        return Ok(());
    };
    let params = |v: &Value, nom: &str| -> Resultat<Vec<(String, Value)>> {
        let o = v["parametres"]
            .as_object()
            .ok_or_else(|| Erreur::Format(format!("{nom} : « parametres » absent")))?;
        Ok(o.iter()
            .filter(|(k, _)| !PARAMETRES_VARIABLES.contains(&k.as_str()))
            .map(|(k, x)| (k.clone(), x.clone()))
            .collect())
    };
    let p0 = params(ref0, nom0)?;
    for (nom, v) in &docs[1..] {
        let p = params(v, nom)?;
        if p != p0 {
            let get = |l: &[(String, Value)], k: &str| {
                l.iter().find(|(k2, _)| k2 == k).map(|(_, x)| x.clone())
            };
            let mut diff: Vec<String> = p0
                .iter()
                .chain(p.iter())
                .map(|(k, _)| k.clone())
                .filter(|k| get(&p0, k) != get(&p, k))
                .collect();
            diff.dedup();
            return Err(Erreur::Config(format!("{nom} : paramètres incompatibles avec {nom0} (seul « c » peut différer ; différence : {diff:?})")));
        }
        for cle in ["n_chaines", "n_configs"] {
            if v["fin"][cle] != ref0["fin"][cle] {
                return Err(Erreur::Config(format!(
                    "{nom} : fin.{cle} = {} ≠ {} de {nom0} : ensembles différents",
                    v["fin"][cle], ref0["fin"][cle]
                )));
            }
        }
        let (g, g0) = (v["grossiers"].as_array(), ref0["grossiers"].as_array());
        let (Some(g), Some(g0)) = (g, g0) else {
            return Err(Erreur::Format(format!("{nom} : « grossiers » absent")));
        };
        if g.len() != g0.len() {
            return Err(Erreur::Config(format!(
                "{nom} : {} points de grille ≠ {} de {nom0}",
                g.len(),
                g0.len()
            )));
        }
        for (a, b) in g.iter().zip(g0) {
            let memes_betas = match (a["beta_prime"].as_f64(), b["beta_prime"].as_f64()) {
                (Some(x), Some(y)) => (x - y).abs() <= 1e-9,
                _ => false,
            };
            if !memes_betas
                || ["L", "n_chaines", "n_configs"]
                    .iter()
                    .any(|k| a[*k] != b[*k])
            {
                return Err(Erreur::Config(format!("{nom} : grille des ensembles grossiers différente de celle de {nom0} (β′ = {}, L = {})", a["beta_prime"], a["L"])));
            }
        }
    }
    Ok(())
}

/// Premier chemin JSON portant un `null` (ce que `serde_json` écrit pour NaN et ±∞), hors clés de `permis`.
fn premier_null(v: &Value, chemin: &str, permis: &[&str]) -> Option<String> {
    match v {
        Value::Null => Some(chemin.to_string()),
        Value::Array(a) => a
            .iter()
            .enumerate()
            .find_map(|(i, x)| premier_null(x, &format!("{chemin}[{i}]"), permis)),
        Value::Object(o) => o
            .iter()
            .filter(|(k, _)| !permis.contains(&k.as_str()))
            .find_map(|(k, x)| premier_null(x, &format!("{chemin}.{k}"), permis)),
        _ => None,
    }
}

/// Refuse d'écrire un document dont un nombre est NaN ou infini (il serait sérialisé en `null`, avec rc = 0).
fn exiger_finis(doc: &Value, permis: &[&str]) -> Resultat<()> {
    match premier_null(doc, "", permis) {
        Some(ch) => Err(Erreur::Config(format!(
            "valeur non finie (NaN ou infini) en « {ch} » : refus d'écrire un JSON à `null`"
        ))),
        None => Ok(()),
    }
}

/// Prépare le dossier de sortie (créé s'il est neuf) et refuse d'écraser.
fn preparer_sortie(sortie: &Path, nom_fichier: &str) -> Resultat<PathBuf> {
    let fichier = sortie.join(nom_fichier);
    if fichier.exists() {
        return Err(Erreur::Config(format!(
            "{} existe : refus d'écraser",
            fichier.display()
        )));
    }
    Ok(fichier)
}

fn creer_dossier(sortie: &Path) -> Resultat<()> {
    std::fs::create_dir_all(sortie)
        .map_err(|e| Erreur::io(format!("création de {}", sortie.display()), e))
}

fn choisir_c_cmd(m: &ArgMatches) -> Resultat<i32> {
    let liste = parse::<String>(m, "entrees")?;
    let sortie = PathBuf::from(parse::<String>(m, "sortie")?);
    let fichier = preparer_sortie(&sortie, "choix_c.json")?;
    let mut pts = Vec::new();
    let mut docs: Vec<(String, Value)> = Vec::new();
    let mut entrees: Vec<Value> = Vec::new();
    let mut beta = f64::NAN;
    for f in liste.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let t = std::fs::read_to_string(f).map_err(|e| Erreur::io(format!("lecture {f}"), e))?;
        let v: Value =
            serde_json::from_str(&t).map_err(|e| Erreur::Format(format!("{f} : {e}")))?;
        let g = |p: &Value| {
            p.as_f64()
                .ok_or_else(|| Erreur::Format(format!("{f} : champ numérique absent")))
        };
        let b = g(&v["fin"]["beta_haut"])?;
        if beta.is_nan() {
            beta = b;
        } else if (beta - b).abs() > 1e-9 {
            return Err(Erreur::Config(format!("{f} : β_haut {b} ≠ {beta}")));
        }
        pts.push(PointC {
            c: g(&v["parametres"]["c"])?,
            dispersion: g(&v["resultat"]["dispersion_schema"])?,
            delta: g(&v["resultat"]["delta_beta"])?,
            err: g(&v["resultat"]["erreur_delta_beta"])?,
        });
        entrees.push(json!({"chemin": f, "sha256": sha256_fichier(Path::new(f)).map_err(|e| Erreur::io(format!("sha256 {f}"), e))?}));
        docs.push((f.to_string(), v));
    }
    verifier_compatibilite(&docs)?;
    let r = choisir_c(&pts)?;
    let tab: Vec<Value> = pts.iter().map(|p| json!({"c": p.c, "dispersion": p.dispersion, "delta_beta": p.delta, "erreur": p.err})).collect();
    let doc = json!({"format": "kvu-rg-choix-c/1", "critere": "dispersion P/R/C minimale en β′ (PAS le critère complet de Tomboulis-Velytsky)", "beta_haut": beta, "points": tab,
                "c_etoile": r.c_etoile, "delta_beta_c_etoile": r.delta_etoile, "erreur_delta_beta": r.err_delta, "dispersion_min": r.dispersion_min,
                "resolution_c": RESOLUTION_C, "drapeau": r.drapeau});
    exiger_finis(&doc, &["drapeau"])?;
    creer_dossier(&sortie)?;
    ecrire_json(&fichier, &doc)?;
    let argv: Vec<String> = std::env::args().collect();
    ecrire_manifeste_fichier(
        &fichier,
        MOTEUR,
        &argv,
        &[("entrees", Value::Array(entrees))],
    )?;
    println!("INFO RG-CHOIX-C β_haut = {beta} : c* = {:.4} ; Δβ(c*) = {:.4} ± {:.4} ; dispersion {:.4} ; drapeau : {}", r.c_etoile, r.delta_etoile, r.err_delta, r.dispersion_min, r.drapeau.unwrap_or("aucun"));
    Ok(code::OK)
}

// ---------------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------------

pub fn sous_commandes() -> Vec<Command> {
    vec![Command::new("choisir-c")
        .about("Choix de c par dispersion P/R/C minimale à partir de plusieurs apparier.json (même β_haut, c différents)")
        .arg(Arg::new("entrees").long("entrees").required(true).num_args(1).help("liste (virgules) de fichiers apparier.json"))
        .arg(Arg::new("sortie").long("sortie").num_args(1).default_value(".")),
        Command::new("apparier")
        .about("MCRG (Swendsen, deux réseaux) : β′ de Wilson dont le réseau grossier reproduit les observables P, R, C du réseau fin bloqué")
        .arg(Arg::new("fin").long("fin").required(true).num_args(1).help("dossier d'une chaîne ou parent de chaînes du réseau fin (un seul β)"))
        .arg(Arg::new("filtre-fin").long("filtre-fin").num_args(1))
        .arg(Arg::new("grossiers").long("grossiers").required(true).num_args(1).help("parent de chaînes de Wilson du réseau grossier (plusieurs β′, une grille)"))
        .arg(Arg::new("filtre-gros").long("filtre-gros").num_args(1))
        .arg(Arg::new("b").long("b").num_args(1).default_value("2").help("facteur de blocage : 1 (identité, porte de cohérence) | 2 (Swendsen)"))
        .arg(Arg::new("c").long("c").num_args(1).default_value("1.1").help("coefficient d'agrafes de Swendsen (b = 2)"))
        .arg(Arg::new("pas").long("pas").num_args(1).default_value("2").help("nombre n de blocages du réseau fin (le grossier en reçoit n − 1)"))
        .arg(Arg::new("degre").long("degre").num_args(1).default_value("2").help("degré de l'ajustement en β′"))
        .arg(Arg::new("fenetre").long("fenetre").num_args(1).default_value("0.3").help("demi-largeur de la fenêtre locale autour du β′ pilote (0 = toute la grille)"))
        .arg(Arg::new("bootstrap-b").long("bootstrap-b").num_args(1).default_value("400"))
        .arg(Arg::new("bootstrap-bloc").long("bootstrap-bloc").num_args(1).default_value("1"))
        .arg(Arg::new("graine").long("graine").num_args(1).default_value("20261110"))
        .arg(Arg::new("sortie").long("sortie").num_args(1).default_value(".").help("dossier de sortie (apparier.json, refus d'écraser)"))
        .arg(Arg::new("fils").long("fils").num_args(1).help("fils (défaut RAYON_NUM_THREADS ou 4)"))]
}

fn parse<T: std::str::FromStr>(m: &ArgMatches, nom: &str) -> Resultat<T> {
    let t = m
        .get_one::<String>(nom)
        .ok_or_else(|| Erreur::Config(format!("--{nom} requis")))?;
    t.parse::<T>()
        .map_err(|_| Erreur::Config(format!("--{nom} : « {t} » invalide")))
}

pub fn executer(nom: &str, m: &ArgMatches) -> Resultat<i32> {
    match nom {
        "apparier" => apparier_cmd(m),
        "choisir-c" => choisir_c_cmd(m),
        _ => Err(Erreur::Config(format!(
            "sous-commande « {nom} » inconnue du module apparier"
        ))),
    }
}

/// Bornes des entiers de la ligne de commande (O10) : validées AVANT toute mesure, sans arithmétique débordante.
pub const PAS_MAX: usize = 16;
pub const BOOTSTRAP_MIN: usize = 20;
pub const BOOTSTRAP_MAX: usize = 1_000_000;
pub const BLOC_MAX: usize = 1_000_000;
pub const FILS_MAX: usize = 4096;

/// Paramètres validés de `rg apparier`.
#[derive(Clone, Copy, Debug)]
pub struct ParamsApparier {
    pub b: usize,
    pub c: f64,
    pub n: usize,
    pub degre: usize,
    pub fenetre: f64,
    pub bb: usize,
    pub bloc: usize,
    pub graine: u64,
}

pub fn valider_params(p: &ParamsApparier) -> Resultat<()> {
    if !matches!(p.b, 1 | 2) {
        return Err(Erreur::Config(format!(
            "--b {} : seuls 1 (identité) et 2 (Swendsen) sont implémentés",
            p.b
        )));
    }
    if !p.c.is_finite() {
        return Err(Erreur::Config(format!("--c {} : nombre fini requis", p.c)));
    }
    if !(1..=PAS_MAX).contains(&p.n) {
        return Err(Erreur::Config(format!(
            "--pas {} : entier entre 1 et {PAS_MAX} requis",
            p.n
        )));
    }
    valider_degre(p.degre)?;
    if !(p.fenetre.is_finite() && p.fenetre >= 0.0) {
        return Err(Erreur::Config(format!(
            "--fenetre {} : nombre fini ≥ 0 requis",
            p.fenetre
        )));
    }
    if !(BOOTSTRAP_MIN..=BOOTSTRAP_MAX).contains(&p.bb) {
        return Err(Erreur::Config(format!(
            "--bootstrap-b {} : entier entre {BOOTSTRAP_MIN} et {BOOTSTRAP_MAX} requis",
            p.bb
        )));
    }
    if !(1..=BLOC_MAX).contains(&p.bloc) {
        return Err(Erreur::Config(format!(
            "--bootstrap-bloc {} : entier entre 1 et {BLOC_MAX} requis",
            p.bloc
        )));
    }
    Ok(())
}

/// `--fils` (ou RAYON_NUM_THREADS, défaut 4) ; 0 refusé comme dans `rg action`.
pub fn valider_fils(texte: Option<&str>, env: Option<&str>) -> Resultat<usize> {
    let n = match texte {
        Some(t) => t
            .parse::<usize>()
            .map_err(|_| Erreur::Config(format!("--fils « {t} » invalide")))?,
        None => env.and_then(|t| t.parse().ok()).unwrap_or(4),
    };
    if !(1..=FILS_MAX).contains(&n) {
        return Err(Erreur::Config(format!(
            "--fils {n} : entier entre 1 et {FILS_MAX} requis"
        )));
    }
    Ok(n)
}

fn apparier_cmd(m: &ArgMatches) -> Resultat<i32> {
    let p = ParamsApparier {
        b: parse(m, "b")?,
        c: parse(m, "c")?,
        n: parse(m, "pas")?,
        degre: parse(m, "degre")?,
        fenetre: parse(m, "fenetre")?,
        bb: parse(m, "bootstrap-b")?,
        bloc: parse(m, "bootstrap-bloc")?,
        graine: parse(m, "graine")?,
    };
    valider_params(&p)?;
    let ParamsApparier {
        b,
        c,
        n,
        degre,
        fenetre,
        bb,
        bloc,
        graine,
    } = p;
    let fils = valider_fils(
        m.get_one::<String>("fils").map(|s| s.as_str()),
        std::env::var("RAYON_NUM_THREADS").ok().as_deref(),
    )?;
    let sortie = PathBuf::from(parse::<String>(m, "sortie")?);
    let fichier = preparer_sortie(&sortie, "apparier.json")?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(fils)
        .build()
        .map_err(|e| Erreur::Config(e.to_string()))?;
    let table = TableFormes::construire()?;
    let ch_fin = decouvrir_chaines(
        Path::new(&parse::<String>(m, "fin")?),
        m.get_one::<String>("filtre-fin").map(|s| s.as_str()),
    )?;
    let ch_gros = decouvrir_chaines(
        Path::new(&parse::<String>(m, "grossiers")?),
        m.get_one::<String>("filtre-gros").map(|s| s.as_str()),
    )?;
    let (fin_g, gros_g) = pool.install(|| -> Resultat<_> {
        Ok((
            mesurer_groupes(&ch_fin, b, c, n, &table)?,
            mesurer_groupes(&ch_gros, b, c, n.saturating_sub(1), &table)?,
        ))
    })?;
    if fin_g.len() != 1 {
        return Err(Erreur::Config(format!(
            "--fin doit porter un seul β ({} trouvés) : utiliser --filtre-fin",
            fin_g.len()
        )));
    }
    let (beta_haut, l_fin, ech_fin) = fin_g.into_iter().next().unwrap();
    let lf = cote_final(l_fin, b, n).map_err(Erreur::Config)?;
    for (beta, l, _) in &gros_g {
        let lg = cote_final(*l, b, n - 1).map_err(Erreur::Config)?;
        if lg != lf {
            return Err(Erreur::Config(format!(
                "côtés finaux différents : fin {l_fin} → {lf} ; grossier β′ = {beta} : {l} → {lg}"
            )));
        }
    }
    if lf < L_FINAL_MIN {
        return Err(Erreur::Config(format!("côté final {lf} < {L_FINAL_MIN}")));
    }
    let gros: Vec<(f64, Echantillon)> = gros_g.iter().map(|g| (g.0, g.2.clone())).collect();
    // Tout le calcul statistique (bootstrap parallèle compris) tourne dans le pool demandé par --fils (O11).
    let a =
        pool.install(|| apparier(beta_haut, &ech_fin, &gros, degre, fenetre, bb, bloc, graine))?;
    // Erreur d'interpolation (amendement 1) : demi-étendue de β′ sur les variantes (degré, fenêtre) déclarées ; les variantes
    // infaisables (trop peu de points dans la fenêtre) sont listées et ignorées.
    let mut variantes: Vec<Value> = Vec::new();
    let mut valeurs = vec![a.beta_comb];
    for &(d, w) in VARIANTES_INTERP.iter() {
        match pool.install(|| apparier(beta_haut, &ech_fin, &gros, d, w, bb, bloc, graine)) {
            Ok(v) => {
                valeurs.push(v.beta_comb);
                variantes.push(json!({"degre": d, "fenetre": w, "beta_prime": v.beta_comb, "erreur_stat": v.err_comb}));
            }
            Err(e) => variantes.push(json!({"degre": d, "fenetre": w, "refusee": e.to_string()})),
        }
    }
    let err_interp = 0.5
        * (valeurs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - valeurs.iter().cloned().fold(f64::INFINITY, f64::min));
    let err_tot = (a.err_comb * a.err_comb + err_interp * err_interp).sqrt();
    let tab: Vec<Value> = gros_g
        .iter()
        .map(|(beta, l, e)| {
            let moy = e.moyenne();
            let plat = e.plat();
            let err: Vec<f64> = (0..3).map(|k| ecart_type(&plat.iter().map(|v| v[k]).collect::<Vec<_>>()) / (plat.len() as f64).sqrt()).collect();
            json!({"beta_prime": beta, "L": l, "n_chaines": e.chaines.len(), "n_configs": plat.len(), "moyenne": moy, "erreur_naive": err})
        })
        .collect();
    let plat_f = ech_fin.plat();
    let err_f: Vec<f64> = (0..3)
        .map(|k| {
            ecart_type(&plat_f.iter().map(|v| v[k]).collect::<Vec<_>>())
                / (plat_f.len() as f64).sqrt()
        })
        .collect();
    let doc = json!({
        "format": SCHEMA_APPARIER,
        "parametres": {"b": b, "c": c, "pas_fin": n, "pas_grossier": n - 1, "degre": degre, "fenetre": fenetre, "bootstrap_b": bb, "bootstrap_bloc": bloc, "graine": graine,
                       "L_fin": l_fin, "L_final": lf, "observables": NOMS_OBS},
        "fin": {"beta_haut": beta_haut, "n_chaines": ech_fin.chaines.len(), "n_configs": plat_f.len(), "moyenne": ech_fin.moyenne(), "erreur_naive": err_f},
        "grossiers": tab,
        "resultat": {
            "beta_prime_P_R_C": a.beta_obs, "erreur_P_R_C": a.err_obs, "poids": a.poids,
            "beta_prime": a.beta_comb, "erreur_beta_prime_stat": a.err_comb, "erreur_interpolation": err_interp, "erreur_beta_prime": err_tot, "variantes_interpolation": variantes, "ic95_beta_prime_stat": a.ic95_comb,
            "delta_beta": a.delta_beta(), "erreur_delta_beta": err_tot,
            "dispersion_schema": a.dispersion, "z_max_paires_PRC": a.z_max_paires,
            "replicats_valides": a.replicats_valides, "replicats_total": a.replicats_total,
            "solution_hors_grille": a.hors_grille,
        },
    });
    exiger_finis(&doc, &[])?;
    creer_dossier(&sortie)?;
    ecrire_json(&fichier, &doc)?;
    let argv: Vec<String> = std::env::args().collect();
    ecrire_manifeste_fichier(
        &fichier,
        MOTEUR,
        &argv,
        &[
            ("parametres", doc["parametres"].clone()),
            (
                "entrees",
                json!({"fin": parse::<String>(m, "fin")?, "filtre_fin": m.get_one::<String>("filtre-fin"), "grossiers": parse::<String>(m, "grossiers")?, "filtre_gros": m.get_one::<String>("filtre-gros")}),
            ),
        ],
    )?;
    println!(
        "INFO RG-APPARIER β_haut = {beta_haut} : β′ = {:.5} ± {:.5} (stat {:.5}, interp {:.5}) (P {:.5}, R {:.5}, C {:.5}) ; Δβ = {:.5} ± {:.5} ; dispersion {:.5} ; z_max P/R/C {:.2} ; hors grille : {}",
        a.beta_comb, err_tot, a.err_comb, err_interp, a.beta_obs[0], a.beta_obs[1], a.beta_obs[2], a.delta_beta(), err_tot, a.dispersion, a.z_max_paires, a.hors_grille
    );
    Ok(code::OK)
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rg::action::bootstrap::RngSplitMix64;
    use crate::rg::bloc::tests::{aleatoire, froid, jauges, transformer};
    use crate::rg::flot::mcrg2::plaquette_et_rectangle;
    use crate::rg::swendsen::plaquette_moyenne;

    fn gauss(rng: &mut RngSplitMix64) -> f64 {
        let u1 = (rng.tirer_u64() >> 11) as f64 / (1u64 << 53) as f64 + 1e-18;
        let u2 = (rng.tirer_u64() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Modèle jouet : O_k(β′) = a_k + b_k β′ + d_k β′², bruit gaussien σ par configuration.
    fn jouet(beta: f64, nch: usize, ncfg: usize, sigma: f64, graine: u64) -> Echantillon {
        let mut rng = RngSplitMix64::nouveau(graine);
        let a = [0.10, 0.02, 0.01];
        let bq = [0.20, 0.12, 0.08];
        let d = [-0.010, -0.008, -0.005];
        Echantillon {
            chaines: (0..nch)
                .map(|_| {
                    (0..ncfg)
                        .map(|_| {
                            std::array::from_fn(|k| {
                                a[k] + bq[k] * beta + d[k] * beta * beta + sigma * gauss(&mut rng)
                            })
                        })
                        .collect()
                })
                .collect(),
        }
    }

    #[test]
    fn observables_champ_froid_egal_un() {
        let t = TableFormes::construire().unwrap();
        let o = observables(&froid(4), &t).unwrap();
        for x in o {
            assert!((x - 1.0).abs() < 1e-14, "{o:?}");
        }
        // Le blocage de Swendsen d'un champ froid est froid (b = 2, tout c), aussi après deux pas.
        let o2 = observables(&bloquer_n(&froid(16), 2, 1.1, 2).unwrap(), &t).unwrap();
        for x in o2 {
            assert!((x - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn observables_p_r_concordent_avec_mcrg2() {
        let t = TableFormes::construire().unwrap();
        let u = aleatoire(6);
        let o = observables(&u, &t).unwrap();
        let ref_ = plaquette_et_rectangle(&u).unwrap();
        assert!(
            (o[0] - ref_.plaquette).abs() < 1e-13,
            "P {} vs {}",
            o[0],
            ref_.plaquette
        );
        assert!(
            (o[1] - ref_.rectangle).abs() < 1e-13,
            "R {} vs {}",
            o[1],
            ref_.rectangle
        );
        assert!((o[0] - plaquette_moyenne(&u)).abs() < 1e-13);
    }

    #[test]
    fn observables_et_blocage_invariants_de_jauge() {
        let t = TableFormes::construire().unwrap();
        let u = aleatoire(8);
        let v = transformer(&u, &jauges(8));
        let (ou, ov) = (observables(&u, &t).unwrap(), observables(&v, &t).unwrap());
        for k in 0..3 {
            assert!((ou[k] - ov[k]).abs() < 1e-12);
        }
        // Après blocage (Swendsen est covariant de jauge) les observables bloquées sont invariantes.
        let (bu, bv) = (
            bloquer_n(&u, 2, 1.1, 1).unwrap(),
            bloquer_n(&v, 2, 1.1, 1).unwrap(),
        );
        let (ou, ov) = (observables(&bu, &t).unwrap(), observables(&bv, &t).unwrap());
        for k in 0..3 {
            assert!((ou[k] - ov[k]).abs() < 1e-11, "{ou:?} {ov:?}");
        }
    }

    #[test]
    fn haar_bloque_sans_agrafes_est_haar() {
        // c = 0 : le produit de deux liens de Haar indépendants est de Haar : moyennes nulles à 5σ (σ ≈ 1/√(2 · nb de boucles)).
        let t = TableFormes::construire().unwrap();
        let o = observables(&bloquer_n(&aleatoire(16), 2, 0.0, 2).unwrap(), &t).unwrap();
        for x in o {
            assert!(x.abs() < 0.02, "{o:?}");
        }
    }

    #[test]
    fn ajustement_exact_et_inversion() {
        let x = [1.0, 1.5, 2.0, 2.5, 3.0];
        let y: Vec<f64> = x
            .iter()
            .map(|t: &f64| 0.3 - 0.2 * t + 0.05 * t * t)
            .collect();
        let p = ajuster(&x, &y, 2).unwrap();
        for &xi in &x {
            assert!((p.eval(xi) - (0.3 - 0.2 * xi + 0.05 * xi * xi)).abs() < 1e-12);
        }
        // inversion sur une branche monotone
        let y2: Vec<f64> = x
            .iter()
            .map(|t: &f64| 0.1 + 0.2 * t - 0.01 * t * t)
            .collect();
        let p2 = ajuster(&x, &y2, 2).unwrap();
        let cible = 0.1 + 0.2 * 2.2 - 0.01 * 2.2 * 2.2;
        assert!((inverser(&p2, cible, 0.5, 3.5).unwrap() - 2.2).abs() < 1e-10);
        assert!(inverser(&p2, 5.0, 0.5, 3.5).is_none());
    }

    /// O2 : racines exactement aux bornes (la bissection d'origine renvoyait 1 pour p(x) = x, cible 0, [0, 1]).
    #[test]
    fn inversion_racine_exacte_aux_bornes() {
        let p = Poly {
            x0: 0.0,
            c: vec![0.0, 1.0],
        };
        assert_eq!(
            inverser(&p, 0.0, 0.0, 1.0),
            Some(0.0),
            "racine à gauche perdue"
        );
        assert_eq!(inverser(&p, 1.0, 0.0, 1.0), Some(1.0), "racine à droite");
        assert_eq!(inverser(&p, 0.0, -1.0, 0.0), Some(0.0));
        assert_eq!(
            inverser(&p, 2.0, 0.0, 1.0),
            None,
            "pas de racine dans l'intervalle"
        );
        // sous-dépassement du produit fa·fb : intervalle sans racine, tous deux petits et de même signe
        let q = Poly {
            x0: 0.0,
            c: vec![1e-200, 1e-200],
        };
        assert_eq!(inverser(&q, 0.0, 0.0, 1.0), None);
    }

    /// O2 : le croisement pilote tombe EXACTEMENT sur le dernier point de grille (fin = copie du dernier ensemble de la grille).
    #[test]
    fn pilote_egalite_au_dernier_point_de_grille() {
        let grille: Vec<(f64, Echantillon)> = (0..7)
            .map(|j| {
                (
                    2.0 + 0.1 * j as f64,
                    jouet(2.0 + 0.1 * j as f64, 4, 50, 0.01, 100 + j as u64),
                )
            })
            .collect();
        let fin = grille[6].1.clone();
        let a = apparier(2.6, &fin, &grille, 2, 0.35, 100, 1, 7)
            .expect("le croisement au dernier point doit être trouvé");
        assert!((a.beta_comb - 2.6).abs() < 0.05, "β′ = {}", a.beta_comb);
    }

    /// N1 (seconde relecture) : deux observables aux solutions identiques mais fluctuantes ⇒ tension nulle, PAS un refus.
    #[test]
    fn solutions_identiques_tension_nulle_acceptee() {
        let ens = |v: &[f64]| Echantillon {
            chaines: vec![v.iter().map(|&x| [x, x, x]).collect()],
        };
        let grille = vec![
            (1.0, ens(&[0.0, 0.0, 0.0, 0.0])),
            (3.0, ens(&[1.0, 1.0, 1.0, 1.0])),
        ];
        let fin = ens(&[0.25, 0.75, 0.25, 0.75]);
        let a = apparier(2.0, &fin, &grille, 1, 0.0, 64, 1, 5)
            .expect("variance des différences nulle mais identifiable");
        assert_eq!(a.z_max_paires, 0.0);
        assert!(a.err_comb > 0.0);
    }

    /// O3 : ensembles constants (b = 1, configurations froides répétées) ⇒ variance bootstrap nulle ⇒ refus explicite
    /// (avant la correction : poids ∞, ∞/∞ = NaN, `null` dans le JSON et rc = 0).
    #[test]
    fn variance_bootstrap_nulle_refusee() {
        let cst = |beta: f64| Echantillon {
            chaines: vec![vec![[0.1 + 0.2 * beta, 0.02 + 0.12 * beta, 0.01 + 0.08 * beta]; 8]; 2],
        };
        let grille = vec![(1.0, cst(1.0)), (3.0, cst(3.0))];
        let fin = cst(2.0);
        let e = apparier(2.0, &fin, &grille, 2, 0.0, 32, 1, 1)
            .expect_err("variance nulle : doit être refusé");
        assert!(e.to_string().contains("non identifiable"), "{e}");
    }

    #[test]
    fn entrees_non_finies_ou_degre_hors_bornes_refuses() {
        let grille: Vec<(f64, Echantillon)> = (0..5)
            .map(|j| {
                (
                    2.0 + 0.1 * j as f64,
                    jouet(2.0 + 0.1 * j as f64, 2, 20, 0.01, 10 + j as u64),
                )
            })
            .collect();
        let fin = jouet(2.2, 2, 20, 0.01, 77);
        assert!(apparier(2.2, &fin, &grille, 2, f64::NAN, 100, 1, 3).is_err());
        assert!(apparier(f64::NAN, &fin, &grille, 2, 0.0, 100, 1, 3).is_err());
        assert!(
            apparier(2.2, &fin, &grille, usize::MAX, 0.3, 100, 1, 3).is_err(),
            "degre + 2 débordait"
        );
        assert!(apparier(2.2, &fin, &grille, 0, 0.0, 100, 1, 3).is_err());
        let mut g2 = grille.clone();
        g2[1].0 = f64::NAN;
        assert!(
            apparier(2.2, &fin, &g2, 2, 0.0, 100, 1, 3).is_err(),
            "β′ NaN : panique du tri avant la correction"
        );
        let mut f2 = fin.clone();
        f2.chaines[0][0][1] = f64::NAN;
        assert!(apparier(2.2, &f2, &grille, 2, 0.0, 100, 1, 3).is_err());
        assert!(
            apparier(2.2, &fin, &grille, 2, 0.0, 100, 1, 3).is_ok(),
            "le cas valide reste accepté"
        );
    }

    fn params_ok() -> ParamsApparier {
        ParamsApparier {
            b: 2,
            c: 0.25,
            n: 2,
            degre: 2,
            fenetre: 0.3,
            bb: 400,
            bloc: 1,
            graine: 1,
        }
    }

    /// O10 : bornes des entiers et des flottants de la ligne de commande (chaque cas échouait ou débordait avant).
    #[test]
    fn parametres_cli_bornes_validees() {
        assert!(valider_params(&params_ok()).is_ok());
        let mauvais: Vec<ParamsApparier> = vec![
            ParamsApparier {
                degre: usize::MAX,
                ..params_ok()
            },
            ParamsApparier {
                degre: 0,
                ..params_ok()
            },
            ParamsApparier {
                degre: DEGRE_MAX + 1,
                ..params_ok()
            },
            ParamsApparier {
                n: 0,
                ..params_ok()
            },
            ParamsApparier {
                n: usize::MAX,
                ..params_ok()
            },
            ParamsApparier {
                b: 0,
                ..params_ok()
            },
            ParamsApparier {
                b: 3,
                ..params_ok()
            },
            ParamsApparier {
                c: f64::NAN,
                ..params_ok()
            },
            ParamsApparier {
                c: f64::INFINITY,
                ..params_ok()
            },
            ParamsApparier {
                fenetre: f64::NAN,
                ..params_ok()
            },
            ParamsApparier {
                fenetre: -0.1,
                ..params_ok()
            },
            ParamsApparier {
                bb: 0,
                ..params_ok()
            },
            ParamsApparier {
                bb: usize::MAX,
                ..params_ok()
            },
            ParamsApparier {
                bloc: 0,
                ..params_ok()
            },
            ParamsApparier {
                bloc: usize::MAX,
                ..params_ok()
            },
        ];
        for p in mauvais {
            assert!(valider_params(&p).is_err(), "{p:?} doit être refusé");
        }
    }

    /// O11 : `--fils 0` refusé (rayon l'interprétait comme « tous les cœurs ») ; le résultat ne dépend pas du nombre de fils.
    #[test]
    fn fils_zero_refuse_et_resultat_independant_du_pool() {
        assert!(valider_fils(Some("0"), None).is_err());
        assert!(valider_fils(Some("x"), None).is_err());
        assert!(valider_fils(Some("99999"), None).is_err());
        assert_eq!(valider_fils(Some("3"), Some("8")).unwrap(), 3);
        assert_eq!(valider_fils(None, Some("8")).unwrap(), 8);
        assert_eq!(valider_fils(None, None).unwrap(), 4);
        let grille: Vec<(f64, Echantillon)> = (0..5)
            .map(|j| {
                (
                    2.0 + 0.1 * j as f64,
                    jouet(2.0 + 0.1 * j as f64, 2, 20, 0.01, 10 + j as u64),
                )
            })
            .collect();
        let fin = jouet(2.2, 2, 20, 0.01, 77);
        let r = |f: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(f)
                .build()
                .unwrap();
            let a = pool
                .install(|| apparier(2.2, &fin, &grille, 2, 0.0, 100, 1, 3))
                .unwrap();
            (a.beta_comb, a.err_comb)
        };
        assert_eq!(r(1), r(4));
    }

    /// Document `apparier.json` minimal pour les tests de compatibilité de `choisir-c`.
    fn doc_apparier(c: f64) -> Value {
        json!({"format": SCHEMA_APPARIER,
               "parametres": {"b": 2, "c": c, "pas_fin": 2, "pas_grossier": 1, "degre": 2, "fenetre": 0.3, "bootstrap_b": 400, "bootstrap_bloc": 1, "graine": 7, "L_fin": 16, "L_final": 4, "observables": NOMS_OBS},
               "fin": {"beta_haut": 3.0, "n_chaines": 4, "n_configs": 160},
               "grossiers": [{"beta_prime": 2.0, "L": 8, "n_chaines": 4, "n_configs": 200}, {"beta_prime": 2.1, "L": 8, "n_chaines": 4, "n_configs": 200}]})
    }

    /// O8 : un seul paramètre peut varier (c) ; volume, facteur b, nombre de blocages, ensembles : refus.
    #[test]
    fn choisir_c_refuse_les_entrees_incompatibles() {
        let nom = |i: usize| format!("e{i}.json");
        let ok = vec![
            (nom(0), doc_apparier(0.2)),
            (nom(1), doc_apparier(0.25)),
            (nom(2), doc_apparier(0.3)),
        ];
        assert!(verifier_compatibilite(&ok).is_ok());
        type Mutation = Box<dyn Fn(&mut Value)>;
        let mutations: Vec<Mutation> = vec![
            Box::new(|d| d["parametres"]["L_fin"] = json!(24)),
            Box::new(|d| d["parametres"]["L_final"] = json!(6)),
            Box::new(|d| d["parametres"]["b"] = json!(1)),
            Box::new(|d| d["parametres"]["pas_fin"] = json!(3)),
            Box::new(|d| d["parametres"]["pas_grossier"] = json!(2)),
            Box::new(|d| d["parametres"]["bootstrap_b"] = json!(200)),
            Box::new(|d| d["parametres"]["graine"] = json!(8)),
            Box::new(|d| d["parametres"]["supplementaire"] = json!(1)),
            Box::new(|d| d["fin"]["n_configs"] = json!(80)),
            Box::new(|d| d["grossiers"][1]["L"] = json!(12)),
            Box::new(|d| d["grossiers"][1]["beta_prime"] = json!(2.2)),
            Box::new(|d| d["grossiers"][0]["n_configs"] = json!(100)),
            Box::new(|d| {
                d["grossiers"].as_array_mut().unwrap().pop();
            }),
            Box::new(|d| d["format"] = json!("autre/1")),
        ];
        for (i, mu) in mutations.iter().enumerate() {
            let mut d = doc_apparier(0.3);
            mu(&mut d);
            let docs = vec![
                (nom(0), doc_apparier(0.2)),
                (nom(1), doc_apparier(0.25)),
                (nom(2), d),
            ];
            assert!(
                verifier_compatibilite(&docs).is_err(),
                "la mutation {i} doit être refusée"
            );
        }
    }

    /// O3 : un NaN sérialisé devient `null` ; la garde le voit (sauf `drapeau`, légitimement nul).
    #[test]
    fn garde_de_finitude_du_json() {
        assert!(exiger_finis(&json!({"a": [1.0, 2.0], "b": {"c": 3.0}}), &[]).is_ok());
        assert!(exiger_finis(&json!({"a": [1.0, f64::NAN]}), &[]).is_err());
        assert!(exiger_finis(&json!({"b": {"c": f64::INFINITY}}), &[]).is_err());
        assert!(exiger_finis(&json!({"drapeau": null}), &["drapeau"]).is_ok());
        assert!(exiger_finis(&json!({"drapeau": null, "x": f64::NAN}), &["drapeau"]).is_err());
    }

    #[test]
    fn choisir_c_refuse_points_non_finis() {
        let mut p = pts_parabole(0.2);
        p[2].dispersion = f64::NAN;
        assert!(
            choisir_c(&p).is_err(),
            "NaN : panique du tri avant la correction"
        );
        let mut q = pts_parabole(0.2);
        q[1].err = f64::INFINITY;
        assert!(choisir_c(&q).is_err());
    }

    #[test]
    fn sortie_neuve_creee_et_refus_d_ecraser() {
        let base = std::env::temp_dir().join(format!("kvu_apparier_test_{}", std::process::id()));
        let neuf = base.join("a").join("b");
        let f = preparer_sortie(&neuf, "apparier.json").unwrap();
        creer_dossier(&neuf).unwrap();
        std::fs::write(&f, "{}").unwrap();
        assert!(
            preparer_sortie(&neuf, "apparier.json").is_err(),
            "refus d'écraser"
        );
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// O7 (NOTE, estimateur préenregistré NON modifié) : si les trois Δβ(c) sont parfaitement corrélés d'erreur commune σ, l'erreur
    /// vraie de l'interpolée est |Σ w_i| σ = σ (les poids de Lagrange somment à 1), alors que `choisir_c` rend √(Σ w_i²) σ.
    /// Test `#[ignore]` : il DOCUMENTE l'écart (voir docs/v0.2/notes/apparier.md), il ne garde rien. Lancer avec `--ignored --nocapture`.
    #[test]
    #[ignore = "démonstration chiffrée de l'objection O7, pas une porte"]
    fn note_o7_covariances_ignorees_entre_delta_beta_de_c() {
        let sigma = 0.01;
        // c* = 0,22, voisins {0,15 ; 0,2 ; 0,25} : poids de Lagrange (−0,12 ; 0,84 ; 0,28), ceux de la relecture.
        let pts: Vec<PointC> = [0.15, 0.2, 0.25, 0.3, 0.35]
            .iter()
            .map(|&c| PointC {
                c,
                dispersion: 0.01 + 2.0 * (c - 0.22f64).powi(2),
                delta: 0.3,
                err: sigma,
            })
            .collect();
        let r = choisir_c(&pts).unwrap();
        let (a, b, c) = (0.15, 0.2, 0.25);
        let x = r.c_etoile;
        let w = [
            (x - b) * (x - c) / ((a - b) * (a - c)),
            (x - a) * (x - c) / ((b - a) * (b - c)),
            (x - a) * (x - b) / ((c - a) * (c - b)),
        ];
        let vraie_rho1 = w.iter().sum::<f64>().abs() * sigma;
        let somme_carres = w.iter().map(|v| v * v).sum::<f64>().sqrt();
        println!("O7 : c* = {x:.4} ; poids = {w:?} ; erreur rendue = {:.6} ; erreur vraie (ρ = 1) = {vraie_rho1:.6} ; rapport = {:.4}", r.err_delta, r.err_delta / vraie_rho1);
        assert!(
            (r.err_delta - somme_carres * sigma).abs() < 1e-12,
            "la pente est nulle (Δβ constant) : l'erreur rendue est la somme en quadrature"
        );
        // cas d'école de la revue : poids (−0,12 ; 0,84 ; 0,28) (c* = 0,22, voisins 0,15 ; 0,2 ; 0,25) ⇒ √0,7984 = 0,8935
        assert!((x - 0.22).abs() < 1e-9, "c* = {x}");
        for (wi, attendu) in w.iter().zip([-0.12, 0.84, 0.28]) {
            assert!((wi - attendu).abs() < 1e-9, "poids {w:?}");
        }
        assert!(
            (r.err_delta / vraie_rho1 - 0.7984f64.sqrt()).abs() < 1e-9,
            "rapport {}",
            r.err_delta / vraie_rho1
        );
    }

    /// PORTE DE COHERENCE (peut échouer) : le blocage est l'identité, le « fin » est un ensemble indépendant au même β que
    /// l'un des points de la grille ⇒ Δβ = 0 à 3σ ; et un ensemble fin à β décalé de 0,08 ⇒ Δβ retrouvé à 3σ.
    #[test]
    fn porte_identite_delta_beta_nul_et_decalage_retrouve() {
        let grille: Vec<(f64, Echantillon)> = (0..7)
            .map(|j| {
                (
                    2.0 + 0.1 * j as f64,
                    jouet(2.0 + 0.1 * j as f64, 4, 50, 0.01, 100 + j as u64),
                )
            })
            .collect();
        let fin = jouet(2.3, 4, 50, 0.01, 999);
        let a = apparier(2.3, &fin, &grille, 2, 0.0, 300, 1, 7).unwrap();
        assert!(
            a.delta_beta().abs() < 3.0 * a.err_comb,
            "Δβ = {} ± {}",
            a.delta_beta(),
            a.err_comb
        );
        assert!(
            a.err_comb > 0.0 && a.err_comb < 0.05,
            "err = {}",
            a.err_comb
        );
        assert!(
            !a.hors_grille && a.z_max_paires < 4.0,
            "z = {}",
            a.z_max_paires
        );
        // fin « décalé » : ensemble produit à β = 2.38 mais déclaré à β_haut = 2.3 ⇒ Δβ = 2.3 − 2.38 = −0.08
        let fin2 = jouet(2.38, 4, 50, 0.01, 1234);
        let a2 = apparier(2.3, &fin2, &grille, 2, 0.0, 300, 1, 7).unwrap();
        assert!(
            (a2.delta_beta() + 0.08).abs() < 3.0 * a2.err_comb,
            "Δβ = {} ± {}",
            a2.delta_beta(),
            a2.err_comb
        );
        assert!(
            a2.delta_beta().abs() > 3.0 * a2.err_comb,
            "la porte doit distinguer un décalage de 0,08"
        );
    }

    #[test]
    fn bootstrap_deterministe_et_solution_hors_grille_signalee() {
        let grille: Vec<(f64, Echantillon)> = (0..5)
            .map(|j| {
                (
                    2.0 + 0.1 * j as f64,
                    jouet(2.0 + 0.1 * j as f64, 2, 40, 0.01, 10 + j as u64),
                )
            })
            .collect();
        let fin = jouet(2.45, 2, 40, 0.01, 77); // un peu au-dessus de la grille [2.0, 2.4]
        let (a, b) = (
            apparier(2.45, &fin, &grille, 2, 0.0, 100, 1, 3).unwrap(),
            apparier(2.45, &fin, &grille, 2, 0.0, 100, 1, 3).unwrap(),
        );
        assert_eq!(a.beta_comb, b.beta_comb);
        assert_eq!(a.err_comb, b.err_comb);
        assert!(a.hors_grille, "β′ = {}", a.beta_comb);
    }

    #[test]
    fn fenetre_locale_corrige_le_biais_d_interpolation() {
        // Modèle à terme cubique marqué : O(β′) = 0.5·tanh(β′ − 2.2) + 0.5. Grille large 1.0…3.4. Fin simulé exactement à 2.2.
        let f = |b: f64| 0.5 * (b - 2.2).tanh() + 0.5;
        let mk = |b: f64, g: u64| {
            let mut e = jouet(b, 4, 50, 0.002, g);
            for ch in e.chaines.iter_mut() {
                for v in ch.iter_mut() {
                    let bruit = v[0] - (0.10 + 0.20 * b - 0.010 * b * b); // bruit du jouet (même graine)
                    *v = [f(b) + bruit, f(b) * 0.5 + bruit, f(b) * 0.2 + bruit];
                }
            }
            e
        };
        let grille: Vec<(f64, Echantillon)> = (0..13)
            .map(|j| {
                (
                    1.0 + 0.2 * j as f64,
                    mk(1.0 + 0.2 * j as f64, 50 + j as u64),
                )
            })
            .collect();
        let fin = mk(2.2, 4242);
        let global = apparier(2.2, &fin, &grille, 2, 0.0, 200, 1, 5).unwrap();
        let local = apparier(2.2, &fin, &grille, 3, 0.45, 200, 1, 5).unwrap();
        assert!(
            local.delta_beta().abs() < 3.0 * local.err_comb.max(1e-4),
            "local Δβ = {} ± {}",
            local.delta_beta(),
            local.err_comb
        );
        assert!(
            global.delta_beta().abs() > local.delta_beta().abs(),
            "le polynôme global doit être plus biaisé"
        );
    }

    fn pts_parabole(c0: f64) -> Vec<PointC> {
        [0.15, 0.2, 0.25, 0.3, 0.35]
            .iter()
            .map(|&c| PointC {
                c,
                dispersion: 0.01 + 2.0 * (c - c0).powi(2),
                delta: 0.5 - 1.2 * c + 0.8 * c * c,
                err: 0.004,
            })
            .collect()
    }

    #[test]
    fn choix_c_retrouve_le_minimum_et_interpole_delta() {
        for c0 in [0.19, 0.237, 0.31] {
            let r = choisir_c(&pts_parabole(c0)).unwrap();
            assert!(
                (r.c_etoile - c0).abs() < 1e-9,
                "c* = {} attendu {c0}",
                r.c_etoile
            );
            assert!((r.delta_etoile - (0.5 - 1.2 * c0 + 0.8 * c0 * c0)).abs() < 1e-9);
            assert!(r.drapeau.is_none());
            assert!(r.err_delta > 0.0 && r.err_delta < 0.05);
        }
    }

    #[test]
    fn choix_c_minimum_au_bord_est_signale_sans_extrapolation() {
        let r = choisir_c(&pts_parabole(0.05)).unwrap();
        assert!(r.drapeau.is_some() && (r.c_etoile - 0.15).abs() < 1e-12);
        let r2 = choisir_c(&pts_parabole(0.6)).unwrap();
        assert!(r2.drapeau.is_some() && (r2.c_etoile - 0.35).abs() < 1e-12);
    }

    /// MUTANTS (doivent ÉCHOUER l'oracle) : sélection par maximum de dispersion, ou par |Δβ| minimal.
    #[test]
    fn choix_c_les_mutants_ratent_l_oracle() {
        let pts = pts_parabole(0.237);
        let bon = choisir_c_mode(&pts, ModeC::Normal).unwrap();
        assert!((bon.c_etoile - 0.237).abs() < 1e-9);
        for mode in [ModeC::MutantArgmax, ModeC::MutantDelta] {
            let m = choisir_c_mode(&pts, mode).unwrap();
            assert!(
                (m.c_etoile - 0.237).abs() > 0.02 || m.drapeau.is_some(),
                "le mutant {mode:?} ne doit pas passer l'oracle : c* = {}",
                m.c_etoile
            );
        }
    }

    #[test]
    fn choix_c_refus_peu_de_points_ou_doublons() {
        assert!(choisir_c(&pts_parabole(0.2)[..2]).is_err());
        let mut p = pts_parabole(0.2);
        p[1].c = p[0].c;
        assert!(choisir_c(&p).is_err());
    }

    #[test]
    fn refus_cote_incompatible() {
        assert_eq!(cote_final(16, 2, 2), Ok(4));
        assert_eq!(cote_final(8, 2, 1), Ok(4));
        assert!(cote_final(18, 2, 2).is_err());
        assert_eq!(cote_final(8, 1, 5), Ok(8));
    }
}
