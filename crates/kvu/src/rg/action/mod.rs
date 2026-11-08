//! Module `rg action` : couplages d'une action effective tronquée (plaquette, rectangle,
//! chaise, torsade) estimés par identités exactes de Schwinger–Dyson sur des configurations
//! bloquées.
//!
//! L'action tronquée est S_β(U) = − Σ_{γ ∈ F} β_γ Σ_{C ∈ 𝒞_γ} ½ Tr U_C ; deux estimateurs
//! sont fournis : la pseudo-vraisemblance à un lien (PL, maximum de la loi conditionnelle de
//! von Mises–Fisher) et l'équation de Schwinger–Dyson linéaire (SD). Les incertitudes sont
//! calculées par jackknife. Un échantillonneur de Gibbs de l'action tronquée sert d'instrument
//! de contrôle.

pub mod boucles;
pub mod cli;
pub mod estimation;
pub mod gates;
pub mod gibbs;

/// Schéma du JSON produit par `rg action`.
pub const SCHEMA_ACTION: &str = "kvu-rg-action/1";
/// Taille des tranches de parallélisation (liens).
pub const TRANCHE: usize = 4096;
/// Seuil de la norme R4 (critère de trou).
pub const SEUIL_R4: f64 = 4.0;
/// Seuil du critère P+R (|β_P| + 5|β_R|).
pub const SEUIL_PR: f64 = 2.0 / 9.0;
/// Côté minimal du réseau sur lequel on estime (les boucles sont simples pour L ≥ 3,
/// on prend une marge).
pub const L_MIN_ESTIMATION: usize = 4;
