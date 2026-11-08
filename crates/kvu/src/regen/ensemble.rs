// SPDX-License-Identifier: Apache-2.0
//! Classement d'un ensemble (répertoire de chaînes) : SIMULATION, lecture seule (`kvu store classer`).
//!
//! Une « chaîne » est le répertoire le plus haut contenant un `manifest.json` ; ce qui n'est sous aucune chaîne (journaux,
//! scripts, ensembles sans manifeste) est compté à part (« hors chaînes », classe C : sans manifeste, non régénérable).
//! Rien n'est écrit, déplacé ni supprimé.

use super::adaptateurs::{adapter_ou_classer_c, Hachage};
use super::classer::{Classe, Contexte};
use super::manifeste::parcourir;
use crate::commun::Resultat;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Une chaîne classée.
#[derive(Clone, Debug)]
pub struct Chaine {
    pub nom: String,
    pub chemin: PathBuf,
    pub source: Option<String>,
    pub classe: Classe,
    pub raisons: Vec<String>,
    pub fichiers: usize,
    pub octets: u64,
}

/// Rapport de classement d'un ensemble.
#[derive(Clone, Debug)]
pub struct RapportEnsemble {
    pub racine: PathBuf,
    pub chaines: Vec<Chaine>,
    pub hors_fichiers: usize,
    pub hors_octets: u64,
    pub total_fichiers: usize,
    pub total_octets: u64,
}

impl RapportEnsemble {
    /// (nombre de chaînes, octets) par classe ; la zone « hors chaînes » est comptée en C (octets seulement).
    pub fn par_classe(&self) -> BTreeMap<Classe, (usize, u64)> {
        let mut m: BTreeMap<Classe, (usize, u64)> = BTreeMap::new();
        for cl in [Classe::A, Classe::B, Classe::C] {
            m.insert(cl, (0, 0));
        }
        for c in &self.chaines {
            let e = m.entry(c.classe).or_insert((0, 0));
            e.0 += 1;
            e.1 += c.octets;
        }
        m.entry(Classe::C).or_insert((0, 0)).1 += self.hors_octets;
        m
    }

    /// Classe dominante en octets (égalité : la moins favorable).
    pub fn classe_dominante(&self) -> Classe {
        let p = self.par_classe();
        [Classe::C, Classe::B, Classe::A]
            .into_iter()
            .max_by_key(|c| p[c].1)
            .unwrap_or(Classe::C)
    }

    pub fn en_json(&self) -> Value {
        let p = self.par_classe();
        let classes: serde_json::Map<String, Value> = p
            .iter()
            .map(|(c, (n, o))| (c.nom().to_string(), json!({"chaines": n, "octets": o})))
            .collect();
        json!({
            "racine": self.racine.to_string_lossy(),
            "total_fichiers": self.total_fichiers,
            "total_octets": self.total_octets,
            "classe_dominante": self.classe_dominante().nom(),
            "classes": classes,
            "hors_chaines": {"fichiers": self.hors_fichiers, "octets": self.hors_octets},
            "chaines": self.chaines.iter().map(|c| json!({
                "nom": c.nom, "source": c.source, "classe": c.classe.nom(), "raisons": c.raisons,
                "fichiers": c.fichiers, "octets": c.octets,
            })).collect::<Vec<_>>(),
        })
    }
}

/// Répertoires de chaînes sous `racine` (le plus haut répertoire contenant `manifest.json`, sans descendre plus bas), triés.
/// Si `racine` contient elle-même un `manifest.json`, c'est la seule chaîne.
pub fn trouver_chaines(racine: &Path) -> Resultat<Vec<PathBuf>> {
    let mut sortie = Vec::new();
    chercher(racine, &mut sortie)?;
    sortie.sort();
    Ok(sortie)
}

fn chercher(dir: &Path, sortie: &mut Vec<PathBuf>) -> Resultat<()> {
    if dir.join("manifest.json").is_file() {
        sortie.push(dir.to_path_buf());
        return Ok(());
    }
    let it = match std::fs::read_dir(dir) {
        Ok(i) => i,
        Err(e) => return Err(crate::commun::Erreur::io(format!("lecture du répertoire {}", dir.display()), e)),
    };
    for entree in it.flatten() {
        let ft = match entree.file_type() {
            Ok(f) => f,
            Err(_) => continue,
        };
        if ft.is_dir() {
            chercher(&entree.path(), sortie)?;
        }
    }
    Ok(())
}

/// Classe toutes les chaînes de `racine` (simulation : lecture seule).
pub fn classer_ensemble(racine: &Path, ctx: &mut Contexte) -> Resultat<RapportEnsemble> {
    let tout = parcourir(racine)?;
    let total_fichiers = tout.len();
    let total_octets: u64 = tout.iter().map(|f| f.octets).sum();
    let mut chaines = Vec::new();
    for dir in trouver_chaines(racine)? {
        let (v, classe, raisons) = adapter_ou_classer_c(&dir.join("manifest.json"), ctx, Hachage::Omettre);
        let prefixe = format!("{}/", dir.to_string_lossy().trim_end_matches('/'));
        let (nf, no) = tout
            .iter()
            .filter(|f| f.chemin.to_string_lossy().starts_with(&prefixe))
            .fold((0usize, 0u64), |(n, o), f| (n + 1, o + f.octets));
        let nom = dir.strip_prefix(racine).map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
        let nom = if nom.is_empty() { dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default() } else { nom };
        chaines.push(Chaine {
            nom,
            chemin: dir,
            source: v.as_ref().and_then(|x| x["origine"]["schema"].as_str()).map(str::to_string),
            classe,
            raisons,
            fichiers: nf,
            octets: no,
        });
    }
    let dans_chaines_f: usize = chaines.iter().map(|c| c.fichiers).sum();
    let dans_chaines_o: u64 = chaines.iter().map(|c| c.octets).sum();
    Ok(RapportEnsemble {
        racine: racine.to_path_buf(),
        chaines,
        hors_fichiers: total_fichiers - dans_chaines_f,
        hors_octets: total_octets - dans_chaines_o,
        total_fichiers,
        total_octets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classe_un_ensemble_jouet_sans_rien_ecrire() {
        let d = std::env::temp_dir().join(format!("kvu_ens_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let c1 = d.join("L8_b2.3_c1");
        std::fs::create_dir_all(&c1).unwrap();
        std::fs::write(c1.join("manifest.json"), br#"{"schema":1,"engine":"autre","statut":"COMPLETE"}"#).unwrap();
        std::fs::write(c1.join("cfg_000001.npy"), vec![0u8; 100]).unwrap();
        std::fs::write(d.join("notes.txt"), vec![0u8; 7]).unwrap();
        let avant = parcourir(&d).unwrap();
        let r = classer_ensemble(&d, &mut Contexte::nouveau()).unwrap();
        assert_eq!(r.chaines.len(), 1);
        assert_eq!(r.chaines[0].classe, Classe::C);
        assert_eq!(r.chaines[0].fichiers, 2);
        assert_eq!(r.hors_octets, 7);
        assert_eq!(r.total_octets, r.chaines[0].octets + 7);
        assert_eq!(parcourir(&d).unwrap(), avant, "classer n'écrit rien");
        let j = r.en_json();
        assert_eq!(j["classes"]["C"]["chaines"], 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn ensemble_sans_manifeste_est_entierement_hors_chaines() {
        let d = std::env::temp_dir().join(format!("kvu_ens2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("sous")).unwrap();
        std::fs::write(d.join("sous/x.npy"), vec![1u8; 33]).unwrap();
        let r = classer_ensemble(&d, &mut Contexte::nouveau()).unwrap();
        assert!(r.chaines.is_empty());
        assert_eq!(r.hors_octets, 33);
        assert_eq!(r.classe_dominante(), Classe::C);
        let _ = std::fs::remove_dir_all(&d);
    }
}
