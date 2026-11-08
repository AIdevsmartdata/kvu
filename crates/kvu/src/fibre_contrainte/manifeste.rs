//! Recette native de régénération, sans donnée dépendant de l'exécution.

use crate::commun::{Erreur, Resultat};
use crate::heritage::manifest::sha256_hex;
use crate::stockage::{provenance, GardeSsd};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Écrit un manifeste déterministe, validé avant toute écriture.
pub fn ecrire_manifeste_regen(
    sortie: &Path,
    commande: &[String],
    parametres: Value,
    entrees: &[(String, String)],
    fichiers: &[(String, String, u64)],
) -> Resultat<PathBuf> {
    let profil = crate::machine::detecter();
    let versions: serde_json::Map<String, Value> = profil
        .versions
        .iter()
        .filter(|(k, _)| k.as_str() == "rustc")
        .map(|(k, v)| (k.clone(), json!(v)))
        .collect();
    let graines =
        json!({"cle": parametres["graine"], "etiquettes": ["philox4x32-10", "fibre-contrainte/1"]});
    // serde_json sans preserve_order : Map est un BTreeMap, y compris les objets imbriqués.
    let recette = json!({"commande": commande, "parametres": parametres, "entrees": entrees, "graines": graines});
    let noyaux: Vec<Value> = crate::noyaux::empreintes()
        .into_iter()
        .map(|(nom, sha)| json!({"nom": nom, "sha256": sha}))
        .collect();
    let doc = json!({
        "schema": "kvu-regen/1",
        "origine": {"schema": "kvu-regen/1", "manifeste": "manifest_regen.json",
            "sha256_manifeste": sha256_hex(recette.to_string().as_bytes())},
        "binaire_sha256": provenance::sha256_binaire()?, "binaire_provenance": "manifeste",
        "git_head": provenance::git_head(),
        "source_hash": provenance::source_hash_unifiee().unwrap_or(&"0".repeat(64)),
        "noyaux_embarques": noyaux, "modules_compiles": [], "backend": "cpu", "arch": null,
        "versions": versions, "machine": {"nom": profil.machine.nom(), "cpu": null, "gpu": null, "ram": null},
        "graines": graines, "parametres": parametres, "commande": commande,
        "entrees": entrees.iter().map(|(n, s)| json!({"nom": n, "sha256": s})).collect::<Vec<_>>(),
        "fichiers": fichiers.iter().map(|(n, s, o)| json!({"nom": n, "sha256": s, "octets": o})).collect::<Vec<_>>(),
        "fichiers_hachage": "disque", "duree_s": null, "statut": "COMPLETE", "classe_regen": "C",
        "regen_verifiee": {"date": null, "machine": null, "resultat": "non_verifiee"}
    });
    let erreurs = crate::regen::schema::valider_manifeste(&doc);
    if !erreurs.is_empty() {
        return Err(Erreur::Validation(format!(
            "manifeste de fibre : {}",
            erreurs.join(" ; ")
        )));
    }
    let texte =
        serde_json::to_string_pretty(&doc).map_err(|e| Erreur::Format(e.to_string()))? + "\n";
    let chemin = sortie.join("manifest_regen.json");
    GardeSsd::machine().verifier(&chemin, texte.len() as u64)?;
    std::fs::write(&chemin, texte).map_err(|e| Erreur::io("écriture du manifeste de fibre", e))?;
    Ok(chemin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fibre_contrainte_manifeste_deterministe_et_valide() {
        let scratch = crate::stockage::scratch_tests("fibre_contrainte_manifeste");
        let p = json!({"groupe": "SU(2)", "L": 2, "beta": 2.0, "start": "cold", "therm": 0,
            "nmeas": 2, "skip": 1, "nor": 0, "precision": "f64", "options": "ref-k1", "graine": 42});
        let cmd = vec!["fibre".into(), "echantillonner".into()];
        let chemin = ecrire_manifeste_regen(scratch.chemin(), &cmd, p.clone(), &[], &[]).unwrap();
        let a = std::fs::read(&chemin).unwrap();
        ecrire_manifeste_regen(scratch.chemin(), &cmd, p, &[], &[]).unwrap();
        assert_eq!(a, std::fs::read(&chemin).unwrap());
        let mut doc: Value = serde_json::from_slice(&a).unwrap();
        assert!(crate::regen::schema::valider_manifeste(&doc).is_empty());
        doc.as_object_mut().unwrap().remove("graines");
        assert!(!crate::regen::schema::valider_manifeste(&doc).is_empty());
        assert!(ecrire_manifeste_regen(scratch.chemin(), &cmd, json!({}), &[], &[]).is_err());
    }
}
