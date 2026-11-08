//! Compression zstd FACULTATIVE par le binaire système `zstd` (aucune dépendance de crate). Propriétaire : T8.
//!
//! Niveau 3, sans fil auxiliaire (`-3 --single-thread --no-asyncio`). Les quaternions en double précision compressent peu (5 à 10 %) :
//! le vrai levier est le format `q64` et l'éclaircissement, la compression reste facultative.
//! Une compression n'efface JAMAIS l'original avant que `zstd -t` ait validé le fichier compressé.
//!
//! Pourquoi « sans fil » : sous un plafond de tâches (`TasksMax`, `pids.max`), `zstd -T4` échoue (code 104, « Failed creating
//! I/O thread pool ») parce qu'il crée des fils de compression ET, depuis zstd 1.5.4, des pools d'E/S asynchrones.
//! `--single-thread` (zstd ≥ 1.3.4) supprime les premiers, `--no-asyncio` (zstd ≥ 1.5.4) les seconds ; les deux sont passés à
//! la compression, au test et à la décompression. Un zstd plus ancien ne connaît pas `--no-asyncio` (et n'a pas de pool
//! d'E/S) : les options sont donc sondées une fois par processus ([`options_sans_fil`]).

use crate::commun::{Erreur, Resultat};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::OnceLock;

const NIVEAU: &str = "-3";
// `--single-thread` désactive les workers de compression, mais pas les pools d'I/O de zstd.
// Les deux options sont donc requises, y compris lors du test et de la décompression.
const FILS: &str = "--single-thread";
const IO_SYNCHRONE: &str = "--no-asyncio";

/// Vrai si le `zstd` du PATH accepte `options` (sonde : `zstd <options> -V`, sans fichier ni fil).
fn zstd_accepte(options: &[&str]) -> bool {
    Command::new("zstd")
        .args(options)
        .arg("-V")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Options « sans fil auxiliaire » que le `zstd` du PATH accepte, sondées une fois par processus :
/// `--single-thread --no-asyncio` (zstd ≥ 1.5.4), sinon `--single-thread` seul (zstd 1.3.4 à 1.5.3, sans pool d'E/S),
/// sinon aucune (zstd absent ou très ancien : l'appel suivant rendra l'erreur de zstd lui-même).
pub fn options_sans_fil() -> &'static [&'static str] {
    static OPTIONS: OnceLock<&'static [&'static str]> = OnceLock::new();
    OPTIONS.get_or_init(|| {
        const DEUX: &[&str] = &[FILS, IO_SYNCHRONE];
        const UNE: &[&str] = &[FILS];
        if zstd_accepte(DEUX) {
            DEUX
        } else if zstd_accepte(UNE) {
            UNE
        } else {
            &[]
        }
    })
}

/// Vrai si `zstd --version` répond.
pub fn disponible() -> bool {
    Command::new("zstd")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Chemin du compressé : `chemin` + « .zst » (l'extension s'ajoute au nom complet, `cfg.npy` → `cfg.npy.zst`).
pub fn chemin_compresse(chemin: &Path) -> PathBuf {
    ajouter_suffixe(chemin, ".zst")
}

fn ajouter_suffixe(chemin: &Path, suffixe: &str) -> PathBuf {
    let mut s = chemin.as_os_str().to_os_string();
    s.push(suffixe);
    PathBuf::from(s)
}

/// Vrai si le nom du fichier se termine par `.zst`.
pub fn est_zst(chemin: &Path) -> bool {
    chemin.extension().is_some_and(|e| e == "zst")
}

fn erreur_zstd(contexte: &str, sortie: &std::process::Output) -> Erreur {
    let msg = String::from_utf8_lossy(&sortie.stderr);
    Erreur::io(
        contexte.to_string(),
        std::io::Error::other(format!("zstd a rendu {} : {}", sortie.status, msg.trim())),
    )
}

/// Teste l'intégrité d'un fichier compressé (`zstd -t`, qui vérifie aussi la somme de contrôle du contenu).
pub fn tester(chemin: &Path) -> Resultat<()> {
    let sortie = Command::new("zstd")
        .arg("-t")
        .args(options_sans_fil())
        .args(["-q", "--"])
        .arg(chemin)
        .output()
        .map_err(|e| Erreur::io("lancement de zstd -t", e))?;
    if sortie.status.success() {
        Ok(())
    } else {
        Err(erreur_zstd(&format!("zstd -t {}", chemin.display()), &sortie))
    }
}

/// Compresse `chemin` en `chemin.zst` (niveau 3, sans fil auxiliaire), vérifie (zstd -t) puis supprime l'original.
///
/// Atomique : écriture dans `chemin.zst.part`, test, fsync, renommage, fsync du répertoire, et seulement
/// ensuite suppression de l'original. En cas d'échec, l'original est intact et le `.part` supprimé.
pub fn compresser(chemin: &Path) -> Resultat<PathBuf> {
    let meta = std::fs::symlink_metadata(chemin).map_err(|e| Erreur::io(format!("stat {}", chemin.display()), e))?;
    if !meta.is_file() {
        return Err(Erreur::Config(format!(
            "{} n'est pas un fichier régulier",
            chemin.display()
        )));
    }
    let sortie = chemin_compresse(chemin);
    if std::fs::symlink_metadata(&sortie).is_ok() {
        return Err(Erreur::Config(format!(
            "{} existe déjà : refus d'écraser",
            sortie.display()
        )));
    }
    let part = ajouter_suffixe(&sortie, ".part");
    let _ = std::fs::remove_file(&part);

    let resultat = (|| -> Resultat<()> {
        let r = Command::new("zstd")
            .arg(NIVEAU)
            .args(options_sans_fil())
            .args(["-q", "-o"])
            .arg(&part)
            .arg("--")
            .arg(chemin)
            .output()
            .map_err(|e| Erreur::io("lancement de zstd", e))?;
        if !r.status.success() {
            return Err(erreur_zstd(&format!("zstd {}", chemin.display()), &r));
        }
        tester(&part)?;
        let f = std::fs::File::open(&part).map_err(|e| Erreur::io(format!("ouverture {}", part.display()), e))?;
        f.sync_all()
            .map_err(|e| Erreur::io(format!("fsync {}", part.display()), e))?;
        std::fs::rename(&part, &sortie).map_err(|e| Erreur::io(format!("renommage {}", part.display()), e))?;
        Ok(())
    })();
    if let Err(e) = resultat {
        let _ = std::fs::remove_file(&part);
        return Err(e);
    }
    synchroniser_repertoire(&sortie);
    std::fs::remove_file(chemin)
        .map_err(|e| Erreur::io(format!("suppression de l'original {}", chemin.display()), e))?;
    synchroniser_repertoire(&sortie);
    Ok(sortie)
}

/// fsync du répertoire parent de `chemin` (échec ignoré : certains systèmes de fichiers le refusent).
pub(crate) fn synchroniser_repertoire(chemin: &Path) {
    let dir = chemin
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if let Ok(f) = std::fs::File::open(dir) {
        let _ = f.sync_all();
    }
}

/// Flux de lecture sur la sortie de `zstd -dc` : décompression à la volée, mémoire bornée.
///
/// À lire jusqu'à la fin, puis [`FluxZstd::terminer`] vérifie le code de retour de zstd (un fichier
/// tronqué ou corrompu n'est détecté qu'à ce moment-là). Détruit sans être terminé, le processus est tué.
pub struct FluxZstd {
    enfant: Option<Child>,
    sortie: ChildStdout,
    chemin: PathBuf,
}

/// Lance `zstd -dc` sur `chemin` et rend le flux de la sortie décompressée.
pub fn ouvrir_flux(chemin: &Path) -> Resultat<FluxZstd> {
    let mut enfant = Command::new("zstd")
        .arg("-dc")
        .args(options_sans_fil())
        .args(["-q", "--"])
        .arg(chemin)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| Erreur::io(format!("lancement de zstd -dc {}", chemin.display()), e))?;
    let sortie = enfant.stdout.take().expect("stdout demandé en tube");
    Ok(FluxZstd {
        enfant: Some(enfant),
        sortie,
        chemin: chemin.to_path_buf(),
    })
}

impl Read for FluxZstd {
    fn read(&mut self, tampon: &mut [u8]) -> std::io::Result<usize> {
        self.sortie.read(tampon)
    }
}

impl FluxZstd {
    /// Attend la fin de zstd et vérifie son code de retour. À appeler après la lecture complète.
    pub fn terminer(mut self) -> Resultat<()> {
        let enfant = self.enfant.take().expect("terminer consomme le flux");
        let sortie = enfant
            .wait_with_output()
            .map_err(|e| Erreur::io("attente de zstd", e))?;
        if sortie.status.success() {
            Ok(())
        } else {
            Err(erreur_zstd(&format!("zstd -dc {}", self.chemin.display()), &sortie))
        }
    }
}

impl Drop for FluxZstd {
    fn drop(&mut self) {
        if let Some(mut e) = self.enfant.take() {
            let _ = e.kill();
            let _ = e.wait();
        }
    }
}

/// Décompresse entièrement en mémoire (pour les petits fichiers ; les gros se lisent par [`ouvrir_flux`]).
pub fn decompresser_en_memoire(chemin: &Path) -> Resultat<Vec<u8>> {
    let mut flux = ouvrir_flux(chemin)?;
    let mut v = Vec::new();
    flux.read_to_end(&mut v)
        .map_err(|e| Erreur::io(format!("lecture de zstd -dc {}", chemin.display()), e))?;
    flux.terminer()?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffixes() {
        assert_eq!(
            chemin_compresse(Path::new("/a/cfg_000001.npy")),
            PathBuf::from("/a/cfg_000001.npy.zst")
        );
        assert!(est_zst(Path::new("/a/x.npy.zst")));
        assert!(!est_zst(Path::new("/a/x.npy")));
        assert!(!est_zst(Path::new("/a/zst")));
    }

    #[test]
    fn options_sans_fil_coherentes() {
        // Quel que soit le zstd installé : « --no-asyncio » n'est jamais passé sans « --single-thread »,
        // et un zstd ≥ 1.5.4 (celui des machines de référence) reçoit les deux options.
        let o = options_sans_fil();
        assert!(o.is_empty() || o[0] == FILS, "{o:?}");
        assert!(o.len() <= 2);
        if zstd_accepte(&[FILS, IO_SYNCHRONE]) {
            assert_eq!(o, &[FILS, IO_SYNCHRONE]);
        }
        assert!(!zstd_accepte(&["--option-qui-n-existe-pas"]));
    }
}
