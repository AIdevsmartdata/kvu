//! Mesure du flux par face de bloc 3×3 : angle propre Θ = 2 arccos(½ Tr W) et
//! flux cohérent Φ_coh (somme vectorielle des angles des plaquettes transportées,
//! non repliée). Portage Rust de `desert_debut_2026-11` (`boucles.py`, `traiter.py`)
//! et de la référence indépendante `reference_flux_py.py`.
//!
//! Deux sous-commandes : `kvu rg flux` (statistiques d'une configuration ou d'une
//! liste) et `kvu rg flux-gates` (gates de validation). Voir `docs/FLUX_BLOC.md`.

pub mod cli;
pub mod faces;
pub mod gates;

/// Identifiant de schéma des documents JSON produits par `kvu rg flux`.
pub const SCHEMA_FLUX: &str = "kvu-rg-flux/1";
/// Taille des tranches fixes de parallélisme : 4096 sites consécutifs du réseau,
/// séparément pour chaque plan, avant filtrage du pavage. Les statistiques sont
/// calculées tranche par tranche, puis les résultats partiels sont combinés
/// séquentiellement dans l'ordre (plan, premier site) :
/// le résultat est identique À L'OCTET quel que soit `--fils`.
pub const TRANCHE: usize = 4096;
/// Nombre de classes de l'histogramme de α = arccos(x) sur [0, π].
pub const NBINS_ALPHA: usize = 3600;
/// Plans dans l'ordre canonique : (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
pub const PLANS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
/// Seuils sur x = ½ Tr W : (nom, seuil, strict). `true` = « < », `false` = « ≤ ».
/// Valeurs exactes données par la spécification (repr Python de -√½, -cos(π/32)…).
pub const SEUILS_X: [(&str, f64, bool); 5] = [
    ("T1_pi", 0.0, true),
    // −√½ : la constante de std a les mêmes bits que le repr Python −0.7071067811865476 (vérifié : 0x1.6a09e667f3bcdp-1).
    ("T2_3pi2", -std::f64::consts::FRAC_1_SQRT_2, true),
    ("C32", -0.9951847266721969, false),
    ("C16", -0.9807852804032304, false),
    ("C8", -0.9238795325112867, false),
];
/// Seuils sur Φ_coh : (nom, seuil, strict). `true` = « > », `false` = « ≥ ».
pub const SEUILS_COH: [(&str, f64, bool); 4] = [
    ("Q1_2", std::f64::consts::PI, true),
    ("Q3_4", 1.5 * std::f64::consts::PI, true),
    ("Q31_32", 2.0 * std::f64::consts::PI * 31.0 / 32.0, false),
    ("Q1", 2.0 * std::f64::consts::PI, false),
];
/// Seuils de cohérence C = Φ_coh / S pour les comptes conjoints.
pub const COH_MIN: [(&str, f64); 3] = [("0.5", 0.5), ("0.8", 0.8), ("0.95", 0.95)];
