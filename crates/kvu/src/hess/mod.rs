//! Hessienne de l'action de Wilson SU(2) sur fonds réels (WP13, tranche 2) : H·v sans matrice, DŪ·v par nombres duaux,
//! espaces (libre, ker B, Landau, T-cov, ker DŪ) et λ_min généralisé par LOBPCG F64.
//! Propriétaire : WP13 (ARCHITECTURE §6, GATES v4-T2 partie 1 §1.3).
//!
//! - [`landau`]  fixation de Landau réseau (port de `lat.py::landau_fix`) ;
//! - [`hv`]      H·v de `ym4_hess` (plaquettes, base iσ/√2), gradient, action ; résultat indépendant du nombre de fils ;
//! - [`bases`]   bases ym4/s2, coordonnées gauche/additive, indexation du réseau ;
//! - [`additif`] coordonnées additives de S2 (D(A), dD, correction de commutateur) ;
//! - [`ops`]     références Q0 et Q_cov, contraintes des espaces (transcrites de `lat.py`) ;
//! - [`dual`]    DŪ·v et jacobien dense de la carte de Bałaban par nombres duaux ;
//! - [`dense`]   algèbre dense F64 déterministe (Cholesky, eigh, SVD de Jacobi) ;
//! - [`lobpcg`]  LOBPCG généralisé F64 déterministe ;
//! - [`espaces`] projecteurs explicites des espaces ;
//! - [`hamilt`]  Hessienne avec β, références, `lmin` ;
//! - [`cli`]     sous-commande `hess`.

// Code numérique à indices explicites (transcription de lat.py et des noyaux numba) : ces lints de style n'apportent rien ici.
#![allow(clippy::needless_range_loop, clippy::should_implement_trait, clippy::too_many_arguments, clippy::type_complexity)]

pub mod additif;
pub mod bases;
pub mod cli;
pub mod dense;
pub mod dual;
pub mod espaces;
pub mod hamilt;
pub mod hv;
pub mod landau;
pub mod lobpcg;
pub mod ops;
