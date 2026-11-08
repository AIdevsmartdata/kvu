//! Stockage et provenance : NPY (+ zstd facultatif), manifeste (binaire + noyaux embarqués +
//! sources compilées), garde SSD, store (du, gc, vérifier, exporter), fetch par le câble.
//! Propriétaire : T8.
//!
//! - [`garde`]       garde SSD (statvfs), projection de la taille d'un run ;
//! - [`npy`]         NPY c16 (compat su2_hip) et q64, tableaux f64/f32/u32, écriture atomique, conversions en flux ;
//! - [`compression`] zstd par le binaire système (`-3 -T4`, `zstd -t` avant de supprimer l'original) ;
//! - [`archive`]     index `kvu-archive/1` des dumps archivés, lecteur et vérificateur ;
//! - [`provenance`]  manifeste `kevinotron-unifie/1` au-dessus du contrat K8 (recette de source `kvu-source/1`) ;
//! - [`recette`]     recette `kvu-source/1` : listes de fichiers et empreinte (std seulement, partagée avec `build.rs`) ;
//! - [`store`]       du, gc (simulation par défaut), vérifier, lister, exporter ;
//! - [`fetch`]       rapatriement par le câble 2,5 Gbit/s avec vérification des sha256 ;
//! - [`etat_source`] git_head embarqué (« +sale » pour un arbre modifié), décision partagée avec `build.rs` ;
//! - [`cli`]         sous-commandes `store` et `fetch`.

pub mod archive;
pub mod cli;
pub mod compression;
pub mod etat_source;
pub mod fetch;
pub mod garde;
pub mod npy;
pub mod provenance;
pub mod recette;
pub mod store;

pub use garde::GardeSsd;
pub use npy::FormatChamp;
pub use provenance::Manifeste;

/// Répertoire de travail des tests unitaires : `$KVU_SCRATCH` ou `~/.cache/kvu/scratch/stockage`, puis `<nom>_<pid>_<n>`
/// (jamais dans /tmp, qui peut être en RAM), supprimé à la destruction.
#[cfg(test)]
pub(crate) struct Scratch(std::path::PathBuf);

#[cfg(test)]
impl Scratch {
    pub(crate) fn chemin(&self) -> &std::path::Path {
        &self.0
    }
}

#[cfg(test)]
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
pub(crate) fn scratch_tests(nom: &str) -> Scratch {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let base = std::env::var_os("KVU_SCRATCH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::path::PathBuf::from("."));
            home.join(".cache/kvu/scratch/stockage")
        });
    let d = base.join(format!(
        "{nom}_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("création du scratch de test");
    Scratch(d)
}
