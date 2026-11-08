// build.rs du crate kvu (tâche T8, adapté par WP1 au dépôt autonome) : empreinte des sources de la recette
// `kvu-source/1` (DECISIONS D2).
//
// Calcule, avec la MÊME implémentation que celle du crate (src/stockage/recette.rs, std + heritage/manifest.rs,
// inclus ci-dessous par #[path]), l'empreinte
//   sha256(json.dumps({chemin: sha256(fichier)}, sort_keys=True))
// des fichiers de la recette : crates/kvu/{src,noyaux}/** + Cargo.lock, Cargo.toml, crates/kvu/Cargo.toml,
// crates/kvu/build.rs, rust-toolchain.toml (listes : constantes de recette.rs, relues par le recalcul Python
// validation/historique/source_hash_kvu.py).
//
// Variables injectées pour rustc (les noms gardent le suffixe _UNIFIE, hérité, pour la compatibilité des manifestes) :
//   KEVINOTRON_SOURCE_HASH_UNIFIE       l'empreinte (reproductible par validation/historique/source_hash_kvu.py)
//   KEVINOTRON_SOURCE_NFICHIERS_UNIFIE  nombre de fichiers couverts
//   KVU_DOSSIER_CRATE                  dossier du crate, après remappage des chemins demandé à rustc
//   KEV_GIT_HEAD                        commit de l'arbre compilé, « +sale » s'il est modifié (voir
//                                       src/stockage/etat_source.rs : git si l'arbre en est un, sinon la
//                                       déclaration KEV_GIT_HEAD vérifiée par KEV_SOURCE_HASH_ATTENDU)
// Une source illisible fait ÉCHOUER la compilation : pas de binaire sans empreinte.

// Copie gelée du moteur historique (SHA-256, JSON à la Python) : ses remarques de style ne concernent pas ce fichier.
// `recette.rs` l'appelle sous le chemin `crate::heritage::manifest`, qu'on reproduit ici.
#[allow(dead_code, clippy::all)]
#[path = "src/heritage/manifest.rs"]
pub mod manifest_heritage;
mod heritage {
    pub use super::manifest_heritage as manifest;
}

// Recette kvu-source/1 : listes de fichiers et calcul (une seule implémentation, partagée avec le crate).
#[allow(dead_code)]
#[path = "src/stockage/recette.rs"]
mod recette;

// Décision pure du git_head embarqué (testée dans le crate).
#[allow(dead_code)]
#[path = "src/stockage/etat_source.rs"]
mod etat_source;

use std::path::{Path, PathBuf};
use std::process::Command;

/// Applique la dernière règle dont l'ancien préfixe couvre le chemin, composant par composant.
/// Chaque règle porte sur le chemin d'origine (les remappages ne s'enchaînent pas).
fn remapper(chemin: &str, encodes: &str) -> String {
    let mut resultat = chemin.to_string();
    let mut args = encodes.split('\x1f');
    while let Some(arg) = args.next() {
        let regle = if arg == "--remap-path-prefix" {
            args.next()
        } else {
            arg.strip_prefix("--remap-path-prefix=")
        };
        if let Some((ancien, nouveau)) = regle.and_then(|r| r.split_once('=')) {
            if let Ok(reste) = Path::new(chemin).strip_prefix(ancien) {
                resultat = if reste.as_os_str().is_empty() {
                    nouveau.to_string()
                } else {
                    Path::new(nouveau).join(reste).to_string_lossy().into_owned()
                };
            }
        }
    }
    resultat
}

/// Sortie d'une commande git dans `racine`, None si git est absent ou si ce n'est pas un dépôt.
fn git(racine: &Path, args: &[&str]) -> Option<String> {
    let o = Command::new("git").arg("-C").arg(racine).args(args).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

/// (HEAD, sale) de l'arbre, et les fichiers git à surveiller pour relancer ce script quand HEAD bouge.
fn etat_git(racine: &Path) -> Option<(String, bool, Vec<PathBuf>)> {
    let head = git(racine, &["rev-parse", "HEAD"])?;
    let mut chemins: Vec<&str> = Vec::new();
    chemins.extend(recette::DOSSIERS_KVU_SOURCE_1);
    chemins.extend(recette::FICHIERS_KVU_SOURCE_1);
    let mut args = vec!["status", "--porcelain", "--untracked-files=all", "--"];
    args.extend(chemins);
    let sale = !git(racine, &args)?.is_empty();
    let mut surveiller = Vec::new();
    if let Some(d) = git(racine, &["rev-parse", "--absolute-git-dir"]) {
        surveiller.push(PathBuf::from(&d).join("HEAD"));
        surveiller.push(PathBuf::from(&d).join("index"));
    }
    if let (Some(commun), Some(reference)) =
        (git(racine, &["rev-parse", "--git-common-dir"]), git(racine, &["symbolic-ref", "-q", "HEAD"]))
    {
        let commun = racine.join(commun);
        surveiller.push(commun.join(reference));
        surveiller.push(commun.join("packed-refs"));
    }
    Some((head, sale, surveiller))
}

fn main() {
    let dossier_crate = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let encodes = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    println!("cargo:rustc-env=KVU_DOSSIER_CRATE={}", remapper(&dossier_crate, &encodes));
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    let ici = PathBuf::from(dossier_crate);
    let racine = ici.join("..").join("..").canonicalize().unwrap_or_else(|e| panic!("racine du dépôt : {e}"));

    let (h, n) = recette::hash_kvu_source_1(&racine)
        .unwrap_or_else(|e| panic!("source_hash (recette {}) : {e}", recette::NOM_RECETTE));
    println!("cargo:rustc-env=KEVINOTRON_SOURCE_HASH_UNIFIE={h}");
    println!("cargo:rustc-env=KEVINOTRON_SOURCE_NFICHIERS_UNIFIE={n}");

    println!("cargo:rerun-if-env-changed=KEV_GIT_HEAD");
    println!("cargo:rerun-if-env-changed=KEV_SOURCE_HASH_ATTENDU");
    let declare = std::env::var("KEV_GIT_HEAD").ok();
    let attendu = std::env::var("KEV_SOURCE_HASH_ATTENDU").ok();
    let etat = etat_git(&racine);
    if let Some((_, _, surveiller)) = &etat {
        for f in surveiller.iter().filter(|f| f.exists()) {
            println!("cargo:rerun-if-changed={}", f.display());
        }
    }
    let git_etat = etat.as_ref().map(|(h, s, _)| (h.as_str(), *s));
    if let (Some((h, _)), Some(d)) = (git_etat, declare.as_deref()) {
        let commit = d.trim().split('+').next().unwrap_or("");
        if !commit.is_empty() && !h.starts_with(commit) {
            println!("cargo:warning=KEV_GIT_HEAD={d} ne correspond pas à HEAD {h} : HEAD est embarqué");
        }
    }
    if let Some(g) = etat_source::git_head_embarque(git_etat, declare.as_deref(), attendu.as_deref(), &h) {
        println!("cargo:rustc-env=KEV_GIT_HEAD={g}");
    }

    // Dossiers surveillés récursivement par cargo (ajout, retrait ou modification relancent ce script),
    // puis les fichiers de la liste racine de la recette.
    for d in recette::DOSSIERS_KVU_SOURCE_1 {
        println!("cargo:rerun-if-changed={}", racine.join(d).display());
    }
    for f in recette::FICHIERS_KVU_SOURCE_1 {
        println!("cargo:rerun-if-changed={}", racine.join(f).display());
    }
}
