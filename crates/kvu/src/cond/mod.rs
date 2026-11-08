//! Bain thermique conditionnel « fin sachant grossier » (port GPU de noyau2/cb.py). Propriétaire : T5.
//!
//! Mesure Wilson(β) sachant dec(U) = V figé : pour chaque paire (x, x+μ) de la décimation
//! (x_ν pair ∀ν≠μ ; x_μ pair), U₁U₂ = V ; les autres liens sont libres.
//! - mode `RefRng` : tirages IDENTIQUES à cb.sweep(mode='ref_rng') (numpy PCG64 + heatbath_su2
//!   rejoué à l'hôte par [`numpy_rng`]) ; agrafes et mises à jour sur GPU ; concordance ≤ 1e-12
//!   par balayage avec cb.py ;
//! - mode `Rapide` : tout sur GPU, Philox à compteur (li, sweep, tirage, ETIQUETTE_COND) ;
//!   même loi (gates EQC-2), pas les mêmes tirages.

pub mod chaine;
pub mod cli;
pub mod cpu;
pub mod essais;
pub mod geometrie;
pub mod gpu;
pub mod mesure;
pub mod moteur;
pub mod npy_min;
pub mod numpy_rng;
pub mod numpy_tables;
pub mod reprise;
pub mod ressources;

pub use geometrie::GeometrieCond;
pub use moteur::{ModeCond, MoteurCond, SourceAlea};
pub use numpy_rng::GenerateurNumpy;
