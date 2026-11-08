// SPDX-License-Identifier: Apache-2.0
//! Utilitaires partagés du module `regen` : date UTC, hachage de fichiers, parcours d'arbre en lecture seule.

use crate::commun::{Erreur, Resultat};
use crate::heritage::manifest::{sha256_fichier, sha256_hex};
use std::fs;
use std::path::{Path, PathBuf};

/// Schéma des manifestes de régénération.
pub const SCHEMA_REGEN: &str = "kvu-regen/1";

/// Vrai si `s` est un sha256 en hexadécimal minuscule (64 caractères).
pub fn est_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// sha256 d'un fichier avec contexte d'erreur.
pub fn sha_fichier(p: &Path) -> Resultat<String> {
    sha256_fichier(p).map_err(|e| Erreur::io(format!("sha256 de {}", p.display()), e))
}

/// sha256 d'un tampon.
pub fn sha_octets(b: &[u8]) -> String {
    sha256_hex(b)
}

/// Date UTC « AAAA-MM-JJThh:mm:ssZ » à partir de secondes Unix (algorithme civil de Howard Hinnant, pur et testé).
pub fn date_iso(secondes: u64) -> String {
    let jours = (secondes / 86_400) as i64;
    let reste = secondes % 86_400;
    let z = jours + 719_468;
    let ere = z.div_euclid(146_097);
    let jde = z.rem_euclid(146_097);
    let ane = (jde - jde / 1460 + jde / 36_524 - jde / 146_096) / 365;
    let a = ane + ere * 400;
    let jda = jde - (365 * ane + ane / 4 - ane / 100);
    let mp = (5 * jda + 2) / 153;
    let j = jda - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let a = if m <= 2 { a + 1 } else { a };
    format!(
        "{a:04}-{m:02}-{j:02}T{:02}:{:02}:{:02}Z",
        reste / 3600,
        (reste % 3600) / 60,
        reste % 60
    )
}

/// Date UTC courante.
pub fn maintenant_iso() -> String {
    let s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    date_iso(s)
}

/// Un fichier d'un arbre : chemin relatif (séparateur `/`) et taille.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FichierArbre {
    pub relatif: String,
    pub chemin: PathBuf,
    pub octets: u64,
}

/// Liste récursive des fichiers réguliers sous `racine` (liens symboliques NON suivis, comptés pour leur propre taille
/// de lien comme `du -sb` ne les suit pas ; seuls les fichiers réguliers sont rendus), triée par chemin relatif.
/// Lecture seule : aucune écriture.
pub fn parcourir(racine: &Path) -> Resultat<Vec<FichierArbre>> {
    let mut sortie = Vec::new();
    parcourir_dans(racine, "", &mut sortie)?;
    sortie.sort_by(|a, b| a.relatif.cmp(&b.relatif));
    Ok(sortie)
}

fn parcourir_dans(dir: &Path, prefixe: &str, sortie: &mut Vec<FichierArbre>) -> Resultat<()> {
    let it = fs::read_dir(dir).map_err(|e| Erreur::io(format!("lecture du répertoire {}", dir.display()), e))?;
    for entree in it {
        let entree = entree.map_err(|e| Erreur::io(format!("lecture du répertoire {}", dir.display()), e))?;
        let nom = entree.file_name().to_string_lossy().into_owned();
        let rel = if prefixe.is_empty() { nom.clone() } else { format!("{prefixe}/{nom}") };
        let meta = match fs::symlink_metadata(entree.path()) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.is_dir() {
            parcourir_dans(&entree.path(), &rel, sortie)?;
        } else if meta.is_file() {
            sortie.push(FichierArbre { relatif: rel, chemin: entree.path(), octets: meta.len() });
        }
    }
    Ok(())
}

/// Vrai si `enfant` est `parent` ou se trouve sous `parent` (comparaison lexicale sur chemins canoniques quand ils existent,
/// sinon sur le plus long préfixe existant : sert à refuser une sortie sous un ensemble existant).
pub fn est_sous(enfant: &Path, parent: &Path) -> bool {
    let canon = |p: &Path| -> PathBuf {
        let mut reste = Vec::new();
        let mut courant = p.to_path_buf();
        loop {
            if let Ok(c) = fs::canonicalize(&courant) {
                let mut c = c;
                for r in reste.iter().rev() {
                    c.push(r);
                }
                return c;
            }
            match (courant.file_name().map(|n| n.to_os_string()), courant.parent()) {
                (Some(n), Some(pp)) => {
                    reste.push(n);
                    courant = pp.to_path_buf();
                }
                _ => return p.to_path_buf(),
            }
        }
    };
    canon(enfant).starts_with(canon(parent))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_iso_valeurs_connues() {
        assert_eq!(date_iso(0), "1970-01-01T00:00:00Z");
        assert_eq!(date_iso(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(date_iso(1_761_413_130), "2025-10-25T17:25:30Z");
        assert_eq!(date_iso(1_803_662_400), "2027-02-26T17:20:00Z");
    }

    #[test]
    fn sha256_format() {
        assert!(est_sha256(&"a".repeat(64)));
        assert!(!est_sha256(&"A".repeat(64)));
        assert!(!est_sha256(&"a".repeat(63)));
        assert!(!est_sha256(&"g".repeat(64)));
    }

    #[test]
    fn est_sous_detecte_les_descendants_meme_inexistants() {
        let base = std::env::temp_dir();
        assert!(est_sous(&base.join("nouveau/profond"), &base));
        assert!(!est_sous(&base, &base.join("nouveau")));
        assert!(est_sous(&base, &base));
    }
}
