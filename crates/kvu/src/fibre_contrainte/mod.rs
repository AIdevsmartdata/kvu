//! Échantillonneur de fibre contrainte `μ(dU | Ū(U) = V)` (WP12a) : route A (pénalité) et route B (RATTLE sur SU(2)^N).
//! Références : `docs/FIBRE_COAIRE.md` (WP11), `ARCHITECTURE.md` §5, `GATES.md` GATES v4-T2 partie 1 §1.1.
//!
//! - [`jacobien`] `J = DŪ` analytique, `J v`, `Jᵀ u`, `J Jᵀ` par blocs, gradient de `½ log det(J Jᵀ)` (API commune de la fibre).

pub mod jacobien;
pub mod chol;
pub mod action;
pub mod alea;
pub mod stat;
pub mod rattle;
pub mod jouets;
pub mod modele_su2;
pub mod penalite;
pub mod manifeste;
pub mod cli;
#[cfg(test)]
mod gates_commun;
#[cfg(test)]
mod gates_jacobien;
#[cfg(test)]
mod gates_penalite;
#[cfg(test)]
mod gates_rattle;
#[cfg(test)]
mod gates_su2;
#[cfg(test)]
mod gates_newton;
