// SPDX-License-Identifier: Apache-2.0
//! Schéma embarqué et contraintes de vocabulaire du manifeste d'expérience.

use super::{
    identifiant, prefixe, secondes_unix, statuts_carnet, tableau, texte, verifier_relatif,
};
use serde_json::Value;

pub const SCHEMA_TEXTE: &str = include_str!("kvu-experience-1.schema.json");

/// Réutilise le validateur de `regen` et complète les relations entre champs.
pub fn valider_experience(instance: &Value) -> Vec<String> {
    let schema = match serde_json::from_str(SCHEMA_TEXTE) {
        Ok(schema) => schema,
        Err(e) => return vec![format!("schéma embarqué illisible : {e}")],
    };
    let mut erreurs = match crate::regen::schema::valider(&schema, instance) {
        Ok(erreurs) => erreurs,
        Err(e) => return vec![e],
    };
    if !identifiant(texte(instance, "id")) {
        erreurs.push("/id : identifiant invalide".into());
    }
    for cle in ["titre", "question"] {
        if texte(instance, cle).trim().is_empty() {
            erreurs.push(format!("/{cle} : texte non vide attendu"));
        }
    }
    let carnet_id = texte(instance, "carnet_id");
    if !carnet_id.starts_with("EXP-") || !identifiant(carnet_id) {
        erreurs.push("/carnet_id : identifiant EXP- attendu".into());
    }
    if let Some(p) = instance.get("chemin_projet").and_then(Value::as_str) {
        if let Err(e) = verifier_relatif(p, true) {
            erreurs.push(format!("/chemin_projet : {e}"));
        }
    }
    for (i, p) in tableau(instance, "prereg").iter().enumerate() {
        let id = texte(p, "id");
        if !id.starts_with('P') || id.len() < 2 || !id[1..].bytes().all(|b| b.is_ascii_digit()) {
            erreurs.push(format!("/prereg/{i}/id : P suivi d'un entier attendu"));
        }
        if let Err(e) = secondes_unix(texte(p, "gele_le")) {
            erreurs.push(format!("/prereg/{i}/gele_le : {e}"));
        }
        if p.get("hachage_consigne").is_some() != p.get("hachage_consigne_sha256").is_some() {
            erreurs.push(format!(
                "/prereg/{i} : chemin et empreinte du hachage consigné requis ensemble"
            ));
        }
    }
    for cle in ["cree_le", "modifie_le"] {
        if let Err(e) = secondes_unix(texte(&instance["horodatage"], cle)) {
            erreurs.push(format!("/horodatage/{cle} : {e}"));
        }
    }
    for (i, ancien) in tableau(&instance["verdict"], "historique")
        .iter()
        .enumerate()
    {
        if let Err(e) = secondes_unix(texte(ancien, "remplace_le")) {
            erreurs.push(format!("/verdict/historique/{i}/remplace_le : {e}"));
        }
    }
    for (i, p) in tableau(instance, "carnet").iter().enumerate() {
        let typ = texte(p, "type");
        let id = texte(p, "id");
        for cle in ["titre", "enonce"] {
            if texte(p, cle).trim().is_empty() {
                erreurs.push(format!("/carnet/{i}/{cle} : texte non vide attendu"));
            }
        }
        if !statuts_carnet(typ).contains(&texte(p, "statut")) {
            erreurs.push(format!(
                "/carnet/{i}/statut : vocabulaire interdit pour {typ}"
            ));
        }
        if !identifiant(id) || !prefixe(typ).is_some_and(|pre| id.starts_with(pre)) {
            erreurs.push(format!("/carnet/{i}/id : préfixe incompatible avec {typ}"));
        }
        if let Err(e) = secondes_unix(texte(p, "date")) {
            erreurs.push(format!("/carnet/{i}/date : {e}"));
        }
        for lien in tableau(p, "liens") {
            if !lien.as_str().is_some_and(identifiant) {
                erreurs.push(format!("/carnet/{i}/liens : identifiant invalide"));
            }
        }
    }
    erreurs
}
