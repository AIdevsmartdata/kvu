// SPDX-License-Identifier: Apache-2.0
//! Modules hérités du moteur historique, copiés À L'OCTET dans ce dépôt (import depuis le commit `0a7a2de`
//! du dépôt historique, sha256 d'origine dans `SOURCES_IMPORTEES.md`, racine du dépôt).
//!
//! GELÉ. Les quatre copies (`ffi.rs`, `wrap.rs`, `su2_hb.rs`, `manifest.rs`) ne sont modifiées par aucune tâche ;
//! la gate WP1-AUTONOME compare leur sha256 à celui de leur origine. Aucun `#[path]` ne sort du dépôt.
//!
//! - `ffi`, `wrap`       : dlopen de libamdhip64 + libhiprtc, enveloppes RAII (M1, 10/07) ;
//! - `su2_hb`           : référence CPU FP64 du bain thermique SU(2) (K1, 26/09), bit-compatible
//!   avec su2_sweep.hip à l'ULP des transcendantes près ;
//! - `manifest`         : SHA-256, recettes de source_hash, répertoire de run atomique (K8).

// Couche HIP historique (le runtime unifié `crate::gpu` s'appuie dessus pour le backend HIP).
// `wrap.rs` importe `super::ffi` : les deux fichiers sont donc frères dans ce module.
#[allow(clippy::all)]
pub mod ffi;
#[allow(clippy::all)]
pub mod wrap;

#[allow(clippy::all)]
pub mod su2_hb;

#[allow(clippy::all)]
pub mod manifest;
