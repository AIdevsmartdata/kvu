//! Rapport de domaine du bloc (RG-DOMAINE, GATES v3 §v3.3). Propriétaire : WP6.
//!
//! Pour chaque lien grossier et chacune de ses 10 473 boucles, W = U(Γ₀)⁻¹ U(Γ ∪ [x, x+3e_μ] ∪ Γ′⁻¹) = (a₀, **a**), θ = atan2(|**a**|, a₀) ∈ [0, π]
//! (branche principale de log W). Branche ambiguë : π − θ < 1e-6. Rapport : histogramme de θ (64 classes sur [0, π]), fraction
//! f = #(θ > π/2) / #boucles, θ max, nombre de branches ambiguës. Étiquette (rapportée, jamais un FAIL de l'instrument) :
//! `DANS_DOMAINE` ssi f ≤ 1e-3 et 0 branche ambiguë, sinon `HORS_DOMAINE`.

use super::bloc::{StatsTheta, N_CLASSES};
use super::geometrie::Geometrie;
use crate::commun::{Erreur, Resultat};
use serde_json::{json, Value};
use std::path::Path;

/// Seuil de fraction hors domaine pour l'étiquette `DANS_DOMAINE` (GATES v3 §v3.3).
pub const SEUIL_FRACTION: f64 = 1e-3;
/// Fraction attendue sous la mesure de Haar : ½ × 10 470 / 10 473 (P(θ > π/2) = ½ ; 3 boucles triviales par lien grossier).
pub const F_HAAR_ATTENDU: f64 = 0.49986;
/// Tolérance de l'instrument sur la configuration de Haar.
pub const TOLERANCE_HAAR: f64 = 0.01;

/// Étiquette de domaine d'un ensemble de statistiques.
pub fn etiquette(s: &StatsTheta) -> &'static str {
    if s.fraction_hors() <= SEUIL_FRACTION && s.n_ambigues == 0 {
        "DANS_DOMAINE"
    } else {
        "HORS_DOMAINE"
    }
}

/// Bloc JSON « theta » du manifeste et des rapports (entiers et maximum : indépendant du nombre de fils).
pub fn theta_json(s: &StatsTheta) -> Value {
    json!({
        "classes": N_CLASSES,
        "bornes": [0.0, std::f64::consts::PI],
        "histogramme": s.histogramme.to_vec(),
        "n_boucles": s.n,
        "n_hors": s.n_hors,
        "fraction_hors": s.fraction_hors(),
        "theta_max": s.theta_max,
        "n_ambigues": s.n_ambigues,
        "etiquette": etiquette(s),
    })
}

/// Rapport RG-DOMAINE d'une configuration.
pub fn rapport(nom: &str, sha256_entree: &str, l: usize, geometrie: Geometrie, s: &StatsTheta) -> Value {
    json!({
        "format": "kvu-rg-domaine/1",
        "nom": nom,
        "sha256_entree": sha256_entree,
        "l": l,
        "pas_de_bloc": super::geometrie::PAS,
        "geometrie": geometrie.nom(),
        "theta": theta_json(s),
    })
}

/// Écrit un document JSON (retour à la ligne final), atomiquement (`.part` puis renommage).
pub fn ecrire_json(chemin: &Path, doc: &Value) -> Resultat<()> {
    let texte = serde_json::to_string_pretty(doc).map_err(|e| Erreur::Format(e.to_string()))? + "\n";
    let mut part = chemin.as_os_str().to_os_string();
    part.push(".part");
    let part = std::path::PathBuf::from(part);
    std::fs::write(&part, texte).map_err(|e| Erreur::io(format!("écriture {}", part.display()), e))?;
    std::fs::rename(&part, chemin).map_err(|e| Erreur::io(format!("renommage {}", chemin.display()), e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn etiquettes() {
        let mut s = StatsTheta::vide();
        for _ in 0..10_000 {
            s.ajouter(0.3, false, false);
        }
        assert_eq!(etiquette(&s), "DANS_DOMAINE");
        s.ajouter(2.0, false, false); // 1 sur 10 001 ≤ 1e-3
        assert_eq!(etiquette(&s), "DANS_DOMAINE");
        for _ in 0..20 {
            s.ajouter(2.0, false, false);
        }
        assert_eq!(etiquette(&s), "HORS_DOMAINE");
        let mut a = StatsTheta::vide();
        a.ajouter(std::f64::consts::PI, true, false);
        assert_eq!(etiquette(&a), "HORS_DOMAINE");
    }

    #[test]
    fn json_stable() {
        let mut s = StatsTheta::vide();
        s.ajouter(0.1, false, false);
        let a = rapport("x", "abc", 12, Geometrie::TroisCentre, &s);
        assert_eq!(a["theta"]["histogramme"].as_array().map(|v| v.len()), Some(64));
        assert_eq!(a["geometrie"], "3centre");
        assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&a.clone()).unwrap());
    }
}
