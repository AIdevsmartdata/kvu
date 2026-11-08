//! Outils communs des gates de WP12a (compilés en test seulement) : lignes de verdict, champs de test, directions, Richardson.

use super::alea::{Flux, TAG_OUTIL};
use super::jacobien::{tests::pool, Reseau, Q};
use crate::commun::ChampHote;
use crate::heritage::su2_hb::{qdag, qmul, CpuSu2};
use crate::rg::bloc::{log_q, ubar_lien};
use crate::su2::cpu::haar_q;
use std::path::PathBuf;

/// Écrit une ligne de verdict (`PASS|FAIL|SKIP ID détail`) sur la sortie standard.
pub(crate) fn ligne(etat: &str, id: &str, detail: &str) {
    println!("{etat} {id} {detail}");
}

/// Racine des fixtures WP6 (9⁴).
pub(crate) fn dossier_fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("validation/rg/fixtures")
}

pub(crate) fn champ_haar(l: usize, graine: u32) -> ChampHote {
    ChampHote { l, q: (0..4 * l.pow(4)).flat_map(|i| haar_q(i as u32, 0, graine, 17, 41)).collect() }
}

/// Configuration de Wilson à β par bain thermique + 4 sur-relaxations (générateur CPU de référence), `pas` pas.
pub(crate) fn champ_wilson(l: usize, beta: f64, graine: u64, pas: usize) -> ChampHote {
    let mut g = CpuSu2::new(l, graine).unwrap();
    g.init(true);
    for _ in 0..pas {
        g.step(beta, 4);
    }
    ChampHote { l, q: g.flat() }
}

/// Direction gaussienne normalisée de longueur `n` (flux de graine, compteur de direction `k`).
pub(crate) fn direction(n: usize, graine: u64, k: u32) -> Vec<f64> {
    let f = Flux::nouveau(graine);
    let mut d: Vec<f64> = (0..n).map(|i| f.gauss(0, k, i as u32, TAG_OUTIL)).collect();
    let nrm = d.iter().map(|x| x * x).sum::<f64>().sqrt();
    d.iter_mut().for_each(|x| *x /= nrm);
    d
}

/// Dérivée directionnelle par Richardson à trois niveaux (h, h/2, h/4) des différences centrées `d(h)` :
/// R1 = (4 d(h/2) − d(h))/3, R2 = (4 d(h/4) − d(h/2))/3, résultat (16 R2 − R1)/15.
pub(crate) fn richardson3(d: impl Fn(f64) -> f64, h: f64) -> f64 {
    let (a, b, c) = (d(h), d(h / 2.0), d(h / 4.0));
    let r1 = (4.0 * b - a) / 3.0;
    let r2 = (4.0 * c - b) / 3.0;
    (16.0 * r2 - r1) / 15.0
}

/// Même schéma pour un vecteur.
pub(crate) fn richardson3_vec(d: impl Fn(f64) -> Vec<f64>, h: f64) -> Vec<f64> {
    let (a, b, c) = (d(h), d(h / 2.0), d(h / 4.0));
    (0..a.len())
        .map(|i| {
            let r1 = (4.0 * b[i] - a[i]) / 3.0;
            let r2 = (4.0 * c[i] - b[i]) / 3.0;
            (16.0 * r2 - r1) / 15.0
        })
        .collect()
}

/// Ū de tous les liens grossiers, en parallèle (ordre c).
pub(crate) fn ubar_pool(res: &Reseau, champ: &ChampHote, fils: usize) -> Vec<Q> {
    use rayon::prelude::*;
    let p = pool(fils);
    p.install(|| {
        (0..res.n_c)
            .into_par_iter()
            .map(|c| {
                let (z, mu) = res.coords_c(c);
                ubar_lien(champ, &res.table, z, mu).0
            })
            .collect()
    })
}

/// Transformation de jauge U_{x,μ} → g(x) U_{x,μ} g(x+μ)† avec g de Haar en chaque site.
pub(crate) fn jauge_de_haar(champ: &ChampHote, graine: u32) -> ChampHote {
    let l = champ.l;
    let v = champ.v();
    let g: Vec<Q> = (0..v).map(|s| haar_q(s as u32, 5, graine, 29, 43)).collect();
    let mut r = champ.clone();
    for mu in 0..4 {
        for s in 0..v {
            let mut x = [s / (l * l * l), (s / (l * l)) % l, (s / l) % l, s % l];
            x[mu] = (x[mu] + 1) % l;
            let t = ((x[0] * l + x[1]) * l + x[2]) * l + x[3];
            let u = qmul(qmul(g[s], champ.lien(mu * v + s)), qdag(g[t]));
            r.q[4 * (mu * v + s)..4 * (mu * v + s) + 4].copy_from_slice(&u);
        }
    }
    r
}

/// Écart `ω(Ū′ Ū⁻¹)` aux liens grossiers : vecteur 3n (pour Richardson de la carte).
pub(crate) fn delta_omega(u1: &[Q], u0: &[Q]) -> Vec<f64> {
    let mut out = Vec::with_capacity(3 * u1.len());
    for c in 0..u1.len() {
        out.extend_from_slice(&log_q(qmul(u1[c], qdag(u0[c]))).0);
    }
    out
}

/// Entrée WP11 (GATES v4-T2 partie 1 §1.0) : sha256 de `docs/FIBRE_COAIRE.md` et de `validation/fibre_coaire/sorties/SHA256SUMS`
/// contre les valeurs consignées au journal avant le premier run. `Some(motif)` si l'entrée n'est pas conforme (la gate [WP11] émet
/// alors `SKIP <ID> ENTREE-WP11`).
pub(crate) fn controle_entree_wp11() -> Option<String> {
    use crate::heritage::manifest::sha256_fichier;
    const DOC: &str = "ff179136e9b9d6acea5e752f8e85609158b89706564f7fbeb3e3a5292e184dd1";
    const SOMMES: &str = "243e317f09da8534b94a6f326ef9f6de4d1efc2b793aea102def0e038c81c9b9";
    let racine = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let verifs = [(racine.join("../../docs/FIBRE_COAIRE.md"), DOC), (racine.join("validation/fibre_coaire/sorties/SHA256SUMS"), SOMMES)];
    for (chemin, attendu) in verifs {
        match sha256_fichier(&chemin) {
            Ok(h) if h == attendu => {}
            Ok(h) => return Some(format!("{} : sha256 {h} ≠ {attendu}", chemin.display())),
            Err(e) => return Some(format!("{} illisible : {e}", chemin.display())),
        }
    }
    None
}

/// Écrit `SKIP <ID> ENTREE-WP11 …` et rend vrai si l'entrée WP11 n'est pas conforme.
pub(crate) fn sauter_si_sans_wp11(id: &str) -> bool {
    match controle_entree_wp11() {
        Some(m) => {
            ligne("SKIP", id, &format!("ENTREE-WP11 : {m}"));
            true
        }
        None => false,
    }
}

/// Empreinte sha256 des octets (little-endian) du champ : recette des configurations de test générées dans le processus.
pub(crate) fn empreinte(champ: &ChampHote) -> String {
    let octets: Vec<u8> = champ.q.iter().flat_map(|x| x.to_le_bytes()).collect();
    crate::heritage::su2_hb::sha256_hex(&octets)
}

#[test]
fn entree_wp11() {
    match controle_entree_wp11() {
        None => ligne("PASS", "ENTREE-WP11", "docs/FIBRE_COAIRE.md et sorties/SHA256SUMS conformes aux sha256 consignés au journal avant le premier run"),
        Some(m) => ligne("SKIP", "ENTREE-WP11", &m),
    }
    assert!(controle_entree_wp11().is_none());
}

/// Empreinte sha256 d'une liste de quaternions (cibles V).
pub(crate) fn empreinte_quaternions(v: &[Q]) -> String {
    let octets: Vec<u8> = v.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
    crate::heritage::su2_hb::sha256_hex(&octets)
}
