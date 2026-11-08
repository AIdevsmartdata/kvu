//! Estimateurs de couplage effectif et flot de blocs (`rg flot`). Propriétaire : WP8 (ARCHITECTURE §4.3, GATES v1 §4 et v3 §v3.3).
//!
//! - [`ape`]      blocages `dec` et `ape` (facteur 2) du script `n3_blocage_beta_eff.py` (référence de RG-N3) ;
//! - [`bessel`]   rapports I₂/I₁ et I₁/I₀ exacts ;
//! - [`sd`]       estimateur Schwinger–Dyson à un lien (port de `n3_blocage_beta_eff.py`) ;
//! - [`plaq`]     courbe d'étalonnage ⟨P⟩, ⟨R⟩(β ; L) et inversion monotone, sans extrapolation ;
//! - [`mcrg2`]    observables plaquette 1×1 et rectangle 1×2 d'une configuration ;
//! - [`tau`]      τ_int à fenêtre de Sokal (c = 6), port de `analyse_plaquette.py` ;
//! - [`jackknife`] jackknife par groupe (une chaîne = un groupe).

pub mod ape;
pub mod bessel;
pub mod cellule;
pub mod cli;
pub mod etalonner;
pub mod gates;
pub mod jackknife;
pub mod mcrg2;
pub mod plaq;
pub mod sd;
pub mod tau;

pub mod extensions;
pub mod etalonnage_gates;
