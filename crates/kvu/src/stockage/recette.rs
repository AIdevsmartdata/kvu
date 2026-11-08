// SPDX-License-Identifier: Apache-2.0
//! Recette de provenance autonome du dépôt kvu, utilisable aussi par build.rs.

use crate::heritage::manifest::{json_str_py, sha256_fichier, sha256_hex};
use std::fs;
use std::io;
use std::path::Path;

#[allow(dead_code)]
pub const NOM_RECETTE: &str = "kvu-source/1";

#[allow(dead_code)]
pub const DOSSIERS_KVU_SOURCE_1: &[&str] = &["crates/kvu/src", "crates/kvu/noyaux"];

#[allow(dead_code)]
pub const FICHIERS_KVU_SOURCE_1: &[&str] = &[
    "Cargo.lock", "Cargo.toml", "crates/kvu/Cargo.toml", "crates/kvu/build.rs", "rust-toolchain.toml"
];

#[allow(dead_code)]
pub const TEXTE_RECETTE: &str = "kvu-source/1 : sha256(json.dumps({chemin: sha256}, sort_keys=True)) sur crates/kvu/src/** + crates/kvu/noyaux/** + Cargo.lock + Cargo.toml + crates/kvu/Cargo.toml + crates/kvu/build.rs + rust-toolchain.toml ; injecte par crates/kvu/build.rs ; reproductible par crates/kvu/validation/historique/source_hash_kvu.py";

#[allow(dead_code)]
fn lister_dossier(dossier: &Path, relatif: &str, fichiers: &mut Vec<String>) -> io::Result<()> {
    for entree in fs::read_dir(dossier)? {
        let entree = entree?;
        let nom = entree.file_name();
        let nom = nom.to_str().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "nom de fichier non UTF-8")
        })?;
        let chemin = format!("{relatif}/{nom}");
        let nature = entree.file_type()?;
        if nature.is_dir() {
            lister_dossier(&entree.path(), &chemin, fichiers)?;
        } else if nature.is_file() || (nature.is_symlink() && entree.path().is_file()) {
            // Les liens vers un dossier ne sont pas parcourus ; ceux vers un fichier
            // gardent leur chemin relatif et seront lus par leur contenu pointé.
            fichiers.push(chemin);
        }
    }
    Ok(())
}

/// Chemins relatifs triés et sans doublon ; chaque entrée obligatoire doit exister.
#[allow(dead_code)]
pub fn fichiers_kvu_source_1(racine: &Path) -> io::Result<Vec<String>> {
    let mut fichiers = Vec::new();
    for dossier in DOSSIERS_KVU_SOURCE_1 {
        lister_dossier(&racine.join(dossier), dossier, &mut fichiers)?;
    }
    for fichier in FICHIERS_KVU_SOURCE_1 {
        let nature = fs::metadata(racine.join(fichier))
            .map_err(|e| io::Error::new(e.kind(), format!("{fichier} : {e}")))?;
        if !nature.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("fichier régulier requis : {fichier}"),
            ));
        }
        fichiers.push((*fichier).to_owned());
    }
    fichiers.sort();
    fichiers.dedup();
    Ok(fichiers)
}

/// SHA-256 du dictionnaire sérialisé comme json.dumps(sort_keys=True), et son effectif.
#[allow(dead_code)]
pub fn hash_kvu_source_1(racine: &Path) -> io::Result<(String, usize)> {
    let fichiers = fichiers_kvu_source_1(racine)?;
    let mut json = String::from("{");
    for (indice, fichier) in fichiers.iter().enumerate() {
        let empreinte = sha256_fichier(&racine.join(fichier))
            .map_err(|e| io::Error::new(e.kind(), format!("{fichier} : {e}")))?;
        if indice > 0 {
            json.push_str(", ");
        }
        // Le sérialiseur historique laisse DEL brut ; Python échappe aussi U+007F.
        json.push_str(&json_str_py(fichier).replace('\u{7f}', "\\u007f"));
        json.push_str(": ");
        json.push_str(&json_str_py(&empreinte));
    }
    json.push('}');
    Ok((sha256_hex(json.as_bytes()), fichiers.len()))
}
