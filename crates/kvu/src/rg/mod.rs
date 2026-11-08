//! Module RG par blocs : bloc de Bałaban L_b = 3 (définition gelée par S1), témoin de Swendsen b = 2, rapport de domaine.
//! Propriétaire : WP6 (ARCHITECTURE §4.1-4.2, GATES v1 §4 et v3 §v3.3).
//!
//! - [`geometrie`] tables de chemins (10 473 boucles par lien grossier, 633 chemins, 81 segments), géométries `3centre` et `3coin` (mutant) ;
//! - [`bloc`]      Ū(z, μ) = U(Γ₀) exp(Σ w log(U(Γ₀)⁻¹ U(Γ ∪ [x, x+3e_μ] ∪ Γ′⁻¹))) en quaternions, log/exp exacts, statistiques de θ ;
//! - [`swendsen`]  blocage de Swendsen b = 2 (port de `block_b2`, témoin de schéma) et plaquette moyenne ;
//! - [`domaine`]   rapport de domaine (RG-DOMAINE) ;
//! - [`gates`]     gates RG-* exécutables (`kvu rg gates`) ;
//! - [`gpu`]       noyau GPU HIP/CUDA du bloc (WP7), [`gates_gpu`] RG-GPU / G-DET / RG-HIP-CUDA, [`en_ligne`] mesure `bloc:3` de `su2 gen` ;
//! - [`flot`]      estimateurs de couplage effectif `sd`, `plaq`, `mcrg2`, étalonnages, τ_int, jackknife (WP8) ;
//! - [`operateur`], [`lissage`], [`support`] : opérateurs de blocage quasi-locaux `dec`, `ape`, `flot`, `agr` (WP22,
//!   `docs/BLOCAGE_QUASI_LOCAL.md`), leurs supports exacts ; [`gates_ql`] gates RG-QL-* (`kvu rg ql-gates`) ;
//! - [`cli`]       sous-commande `rg`.

pub mod bloc;
pub mod cli;
pub mod domaine;
pub mod en_ligne;
pub mod action;
pub mod flot;
pub mod flux;
pub mod gates;
pub mod gates_ql;
pub mod gates_reponse;
pub mod gates_gpu;
pub mod geometrie;
pub mod gpu;
pub mod lissage;
pub mod operateur;
pub mod support;
pub mod reponse;
pub mod swendsen;
