//! Aides des tests et des gates de `cond` : lecture des fixtures de cb.py (dossier `tests/donnees/cond/`,
//! ou celui de la variable `KVU_COND_FIXTURES` pour le jeu complet, temporaire, produit par
//! `validation/cond/gen_fixtures.py`). Caché de la documentation : ce n'est pas une API de production.
#![doc(hidden)]

use super::npy_min;
use crate::commun::{Erreur, Resultat};
use std::path::{Path, PathBuf};

/// Dossier des fixtures commitées (L = 4, petites).
///
/// `KVU_DOSSIER_CRATE` (build.rs) vaut `CARGO_MANIFEST_DIR` sous `cargo test`, et le chemin remappé (`/kvu/crates/kvu`)
/// sous `scripts/build_auto.sh` : un `env!("CARGO_MANIFEST_DIR")` ici rendait le binaire dépendant de la LONGUEUR du
/// chemin de build (la fonction est retirée à l'édition des liens, mais sa table d'exceptions reste ; lot t1-prelude).
/// Seuls les tests (non remappés) l'appellent.
pub fn dossier_commite() -> PathBuf {
    PathBuf::from(env!("KVU_DOSSIER_CRATE")).join("tests/donnees/cond")
}

/// Dossier des fixtures étendues (jeu complet à L = 8, temporaire), s'il est demandé.
pub fn dossier_complet() -> Option<PathBuf> {
    std::env::var_os("KVU_COND_FIXTURES").map(PathBuf::from).filter(|p| p.join("index.json").exists())
}

/// Un cas de balayage `ref_rng` de cb.py : entrée, β, graine, sortie après 1 balayage, états PCG64.
#[derive(Clone, Debug)]
pub struct CasBalayage {
    pub l: usize,
    pub cfg: String,
    pub beta: f64,
    pub graine: u64,
    pub fichier_cfg: PathBuf,
    pub fichier_out: PathBuf,
    pub fichier_vp: PathBuf,
    /// (état, incrément) de `rng.bit_generator.state` après 1 puis 5 balayages.
    pub etat1: (u128, u128),
    pub etat5: (u128, u128),
}

pub fn lire_index(dossier: &Path) -> Resultat<serde_json::Value> {
    let p = dossier.join("index.json");
    let t = std::fs::read_to_string(&p).map_err(|e| Erreur::io(format!("lecture {}", p.display()), e))?;
    serde_json::from_str(&t).map_err(|e| Erreur::Format(format!("{} : {e}", p.display())))
}

pub fn u128_de(v: &serde_json::Value) -> Resultat<u128> {
    v.as_str()
        .ok_or_else(|| Erreur::Format("entier 128 bits attendu en texte".into()))?
        .parse::<u128>()
        .map_err(|e| Erreur::Format(e.to_string()))
}

pub fn u64_de(v: &serde_json::Value) -> Resultat<u64> {
    v.as_str()
        .ok_or_else(|| Erreur::Format("entier 64 bits attendu en texte".into()))?
        .parse::<u64>()
        .map_err(|e| Erreur::Format(e.to_string()))
}

/// Tous les cas de balayage du dossier (L donné, ou tous si `l` est None).
pub fn cas_balayages(dossier: &Path, l: Option<usize>) -> Resultat<Vec<CasBalayage>> {
    let idx = lire_index(dossier)?;
    let mut out = Vec::new();
    for c in idx["cas"].as_array().ok_or_else(|| Erreur::Format("index.json : cas absents".into()))? {
        let cl = c["L"].as_u64().unwrap_or(0) as usize;
        if l.is_some_and(|x| x != cl) {
            continue;
        }
        let f = |k: &str| dossier.join(c[k].as_str().unwrap_or(""));
        out.push(CasBalayage {
            l: cl,
            cfg: c["cfg"].as_str().unwrap_or("").to_string(),
            beta: c["beta"].as_f64().unwrap_or(f64::NAN),
            graine: c["seed"].as_u64().unwrap_or(0),
            fichier_cfg: f("cfg_file"),
            fichier_out: f("out"),
            fichier_vp: f("vp"),
            etat1: (u128_de(&c["etat1"])?, u128_de(&c["inc1"])?),
            etat5: (u128_de(&c["etat5"])?, u128_de(&c["inc5"])?),
        });
    }
    Ok(out)
}

/// Lit un tableau `float64` du dossier.
pub fn f64_de(dossier: &Path, nom: &str) -> Resultat<Vec<f64>> {
    npy_min::lire(&dossier.join(nom))?.f64()
}

/// Lit un tableau d'indices du dossier.
pub fn indices_de(dossier: &Path, nom: &str) -> Resultat<Vec<u32>> {
    npy_min::lire(&dossier.join(nom))?.indices_u32()
}

/// Plus grand écart absolu entre deux vecteurs de même longueur.
pub fn ecart_max(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "longueurs différentes");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}
