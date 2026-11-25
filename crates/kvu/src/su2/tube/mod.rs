//! [EXPÉRIMENTAL] Interface et formats susceptibles de changer ; résultats à confirmer indépendamment (statut « proposition »,
//! `parametres.statut = "experimental"` dans les manifestes ; note `docs/v0.2/notes/tube.md`).
//!
//! Module « tube tordu » (11/11/2026) : SU(2) Wilson sur un tore ANISOTROPE (ℓ0, ℓ1, ℓ2, ℓ3), typiquement deux directions
//! longues et une petite fibre (ℓ, ℓ, L, L), avec torsion de 't Hooft (pile de coin, convention kvu M4 = `lib_tube.py`), CPU F64.
//!
//! - `reseau` : géométrie anisotrope, coloriage (parité si tous les côtés sont pairs, glouton sinon), pile z en table,
//!   chaîne (bain thermique, OR, refroidissement) bit-identique à `Cpu64` à côtés égaux ;
//! - `obs` : plaquettes par plan, V = Σ_p (1 − z_p ½Tr U_p), boucles de Wilson R×T (brutes et corrigées de la pile),
//!   Polyakov, fraction de colonnes de vortex ;
//! - `cli` : `su2 gen --cotes …`, `su2 tube smin`, `su2 tube apparie`.
//!
//! Les noyaux GPU, `Cpu64` et la boucle `su2::run` ne sont pas modifiés : `su2 gen` sans `--cotes` est inchangé.

pub mod cli;
pub mod domaine;
pub mod obs;
pub mod reseau;

#[cfg(test)]
mod couture_tests;
