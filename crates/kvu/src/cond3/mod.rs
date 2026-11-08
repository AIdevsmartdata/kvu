//! cond3 : bain thermique conditionnel de FACTEUR 3 sur la fibre dure de DEC, et noyau de réponse de la moyenne conditionnelle
//! (test T3-D du programme YM4, SPEC_T3 : `chantiers/T3_moyenne_conditionnelle_2026-11/`). Extension de [`crate::cond`] (facteur 2).
//!
//! Mesure Wilson(β) sachant DEC(U) = V figé, DEC(U)(X, μ) = U(3X, μ) U(3X + e_μ, μ) U(3X + 2e_μ, μ) :
//! pour chaque ligne (U₁, U₂, U₃), U₃ = U₂† U₁† V est asservi ; les autres liens sont libres. Le même code (géométrie, ordonnancement,
//! mesure, estimation) sert à SU(2) ([`su2`]) et au U(1) gaussien non compact ([`u1`]), en dimension 2 ou 4 : c'est la règle des jouets
//! du SPEC (§6.3), un oracle sur un jouet codé à part ne testerait rien.

// Code numérique : plusieurs tableaux indexés par la même variable de boucle (clarté de la correspondance avec le SPEC).
#![allow(clippy::needless_range_loop)]

pub mod cli;
pub mod geom;
pub mod gpu;
pub mod mesure;
pub mod modele;
pub mod paires;
pub mod reduction;
pub mod run;
pub mod su2;
pub mod tube;
pub mod u1;

pub use geom::Geom;
pub use modele::{Modele, Ordonnancement};
