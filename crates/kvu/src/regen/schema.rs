// SPDX-License-Identifier: Apache-2.0

//! Validation du sous-ensemble de JSON Schema employé par les manifestes.

use serde_json::Value;
use std::cmp::Ordering;

pub const SCHEMA_TEXTE: &str = include_str!("kvu-regen-1.schema.json");

/// Analyse le schéma embarqué, dont la syntaxe est également vérifiée par les tests.
pub fn schema() -> Value {
    serde_json::from_str(SCHEMA_TEXTE)
        .expect("le schéma JSON embarqué kvu-regen/1 doit être syntaxiquement valide")
}

/// Renvoie les erreurs de l'instance, ou une erreur d'usage du schéma.
///
/// Le schéma entier est contrôlé avant l'instance : un mot-clé inconnu dans une
/// propriété absente ou une branche conditionnelle inactive reste une erreur.
pub fn valider(schema: &Value, instance: &Value) -> Result<Vec<String>, String> {
    verifier_schema(schema, "")?;
    let mut erreurs = Vec::new();
    valider_noeud(schema, instance, "", &mut erreurs);
    Ok(erreurs)
}

/// Valide un manifeste sans transformer une erreur d'usage en panique.
pub fn valider_manifeste(instance: &Value) -> Vec<String> {
    match valider(&schema(), instance) {
        Ok(erreurs) => erreurs,
        Err(message) => vec![message],
    }
}

/// Énumère les champs obligatoires, en représentant chaque tableau par l'élément 0.
/// Les branches `if` et `then` ne participent pas à cette énumération.
pub fn champs_obligatoires(schema: &Value) -> Vec<String> {
    let mut chemins = Vec::new();
    collecter_champs(schema, "", &mut chemins);
    chemins.sort();
    chemins.dedup();
    chemins
}

/// Produit une copie par champ obligatoire présent, en retirant ce seul champ.
pub fn mutants_champ_retire(schema: &Value, instance: &Value) -> Vec<(String, Value)> {
    let mut mutants = Vec::new();
    for chemin in champs_obligatoires(schema) {
        if let Some((parent, segment)) = chemin.rsplit_once('/') {
            let nom = segment.replace("~1", "/").replace("~0", "~");
            let mut copie = instance.clone();
            if let Some(objet) = copie.pointer_mut(parent).and_then(Value::as_object_mut) {
                if objet.remove(&nom).is_some() {
                    mutants.push((chemin, copie));
                }
            }
        }
    }
    mutants
}

fn joindre(chemin: &str, segment: &str) -> String {
    let segment = segment.replace('~', "~0").replace('/', "~1");
    format!("{chemin}/{segment}")
}

fn diagnostic(chemin: &str, raison: &str) -> String {
    let chemin = if chemin.is_empty() { "/" } else { chemin };
    format!("{chemin} : {raison}")
}

fn type_connu(nom: &str) -> bool {
    matches!(
        nom,
        "integer" | "number" | "string" | "boolean" | "object" | "array" | "null"
    )
}

fn verifier_schema(schema: &Value, chemin: &str) -> Result<(), String> {
    let objet = schema
        .as_object()
        .ok_or_else(|| diagnostic(chemin, "un schéma doit être un objet"))?;
    for (mot, valeur) in objet {
        let emplacement = joindre(chemin, mot);
        let forme_valide = match mot.as_str() {
            "$schema" | "$id" | "title" | "description" | "const" => true,
            "type" => match valeur {
                Value::String(nom) => type_connu(nom),
                Value::Array(noms) => {
                    !noms.is_empty() && noms.iter().all(|nom| nom.as_str().is_some_and(type_connu))
                }
                _ => false,
            },
            "required" => valeur
                .as_array()
                .is_some_and(|noms| noms.iter().all(Value::is_string)),
            "properties" => {
                let proprietes = valeur.as_object().ok_or_else(|| {
                    diagnostic(&emplacement, "properties doit être un objet de schémas")
                })?;
                for (nom, enfant) in proprietes {
                    verifier_schema(enfant, &joindre(&emplacement, nom))?;
                }
                true
            }
            "additionalProperties" => valeur.is_boolean(),
            "items" | "if" | "then" => {
                verifier_schema(valeur, &emplacement)?;
                true
            }
            "enum" => valeur.is_array(),
            "minimum" => valeur.is_number(),
            "minItems" | "minLength" => valeur.as_u64().is_some(),
            "format" => valeur.as_str() == Some("sha256"),
            _ => {
                return Err(diagnostic(
                    &emplacement,
                    &format!("mot-clé non géré : {mot}"),
                ));
            }
        };
        if !forme_valide {
            return Err(diagnostic(
                &emplacement,
                &format!("valeur non gérée ou incorrecte pour le mot-clé {mot}"),
            ));
        }
    }
    Ok(())
}

fn correspond_au_type(nom: &str, instance: &Value) -> bool {
    match nom {
        // Les conversions entières de serde_json excluent les flottants et booléens.
        "integer" => instance.as_u64().is_some() || instance.as_i64().is_some(),
        "number" => instance.is_number(),
        "string" => instance.is_string(),
        "boolean" => instance.is_boolean(),
        "object" => instance.is_object(),
        "array" => instance.is_array(),
        "null" => instance.is_null(),
        _ => false,
    }
}

fn entier(valeur: &Value) -> Option<i128> {
    valeur
        .as_i64()
        .map(i128::from)
        .or_else(|| valeur.as_u64().map(i128::from))
}

fn comparer_entier_flottant(nombre: i128, seuil: f64) -> Ordering {
    // Les entiers JSON tiennent dans i64/u64. La conversion saturante vers i128
    // garde donc le bon ordre même pour un flottant situé hors de cet intervalle.
    match nombre.cmp(&(seuil as i128)) {
        Ordering::Equal if seuil.fract() > 0.0 => Ordering::Less,
        Ordering::Equal if seuil.fract() < 0.0 => Ordering::Greater,
        ordre => ordre,
    }
}

fn comparer_nombres(gauche: &Value, droite: &Value) -> Option<Ordering> {
    match (entier(gauche), entier(droite)) {
        (Some(nombre), Some(seuil)) => Some(nombre.cmp(&seuil)),
        (Some(nombre), None) => droite
            .as_f64()
            .map(|seuil| comparer_entier_flottant(nombre, seuil)),
        (None, Some(seuil)) => gauche
            .as_f64()
            .map(|nombre| comparer_entier_flottant(seuil, nombre).reverse()),
        (None, None) => gauche.as_f64()?.partial_cmp(&droite.as_f64()?),
    }
}

fn egaux(gauche: &Value, droite: &Value) -> bool {
    match (gauche, droite) {
        // L'égalité numérique autorise 2 et 2.0, sans assimiler true à 1.
        (Value::Number(_), Value::Number(_)) => {
            comparer_nombres(gauche, droite) == Some(Ordering::Equal)
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| egaux(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(nom, valeur)| b.get(nom).is_some_and(|autre| egaux(valeur, autre)))
        }
        _ => gauche == droite,
    }
}

fn valider_noeud(schema: &Value, instance: &Value, chemin: &str, erreurs: &mut Vec<String>) {
    if let Some(types) = schema.get("type") {
        let conforme = match types {
            Value::String(nom) => correspond_au_type(nom, instance),
            Value::Array(noms) => noms.iter().any(|nom| {
                nom.as_str()
                    .is_some_and(|nom| correspond_au_type(nom, instance))
            }),
            _ => false,
        };
        if !conforme {
            erreurs.push(diagnostic(chemin, &format!("type attendu : {types}")));
        }
    }
    if let Some(choix) = schema.get("enum").and_then(Value::as_array) {
        if !choix.iter().any(|choix| egaux(choix, instance)) {
            erreurs.push(diagnostic(chemin, "valeur absente de l'énumération"));
        }
    }
    if let Some(constante) = schema.get("const") {
        if !egaux(instance, constante) {
            erreurs.push(diagnostic(
                chemin,
                "valeur différente de la constante attendue",
            ));
        }
    }
    if let Some(objet) = instance.as_object() {
        if let Some(requis) = schema.get("required").and_then(Value::as_array) {
            for nom in requis.iter().filter_map(Value::as_str) {
                if !objet.contains_key(nom) {
                    erreurs.push(diagnostic(
                        &joindre(chemin, nom),
                        "champ obligatoire absent",
                    ));
                }
            }
        }
        let proprietes = schema.get("properties").and_then(Value::as_object);
        let supplementaires_interdits =
            schema.get("additionalProperties").and_then(Value::as_bool) == Some(false);
        for (nom, valeur) in objet {
            let enfant = joindre(chemin, nom);
            if let Some(regle) = proprietes.and_then(|champs| champs.get(nom)) {
                valider_noeud(regle, valeur, &enfant, erreurs);
            } else if supplementaires_interdits {
                erreurs.push(diagnostic(&enfant, "champ supplémentaire interdit"));
            }
        }
    }
    if let Some(elements) = instance.as_array() {
        if let Some(minimum) = schema.get("minItems").and_then(Value::as_u64) {
            if (elements.len() as u128) < u128::from(minimum) {
                erreurs.push(diagnostic(
                    chemin,
                    &format!("au moins {minimum} élément(s) attendu(s)"),
                ));
            }
        }
        if let Some(regle) = schema.get("items") {
            for (indice, valeur) in elements.iter().enumerate() {
                valider_noeud(
                    regle,
                    valeur,
                    &joindre(chemin, &indice.to_string()),
                    erreurs,
                );
            }
        }
    }
    if instance.is_number() {
        if let Some(minimum) = schema.get("minimum") {
            if comparer_nombres(instance, minimum) == Some(Ordering::Less) {
                erreurs.push(diagnostic(chemin, &format!("nombre inférieur à {minimum}")));
            }
        }
    }
    if let Some(texte) = instance.as_str() {
        if let Some(minimum) = schema.get("minLength").and_then(Value::as_u64) {
            if (texte.chars().count() as u128) < u128::from(minimum) {
                erreurs.push(diagnostic(
                    chemin,
                    &format!("au moins {minimum} caractère(s) attendu(s)"),
                ));
            }
        }
        if schema.get("format").and_then(Value::as_str) == Some("sha256")
            && (texte.len() != 64
                || !texte
                    .bytes()
                    .all(|octet| octet.is_ascii_digit() || (b'a'..=b'f').contains(&octet)))
        {
            erreurs.push(diagnostic(
                chemin,
                "sha256 attendu : exactement 64 chiffres hexadécimaux minuscules",
            ));
        }
    }
    if let Some(condition) = schema.get("if") {
        let mut erreurs_condition = Vec::new();
        valider_noeud(condition, instance, chemin, &mut erreurs_condition);
        if erreurs_condition.is_empty() {
            if let Some(consequence) = schema.get("then") {
                valider_noeud(consequence, instance, chemin, erreurs);
            }
        }
    }
}

fn collecter_champs(schema: &Value, chemin: &str, chemins: &mut Vec<String>) {
    if let Some(requis) = schema.get("required").and_then(Value::as_array) {
        for nom in requis.iter().filter_map(Value::as_str) {
            chemins.push(joindre(chemin, nom));
        }
    }
    if let Some(proprietes) = schema.get("properties").and_then(Value::as_object) {
        for (nom, enfant) in proprietes {
            collecter_champs(enfant, &joindre(chemin, nom), chemins);
        }
    }
    if let Some(elements) = schema.get("items") {
        collecter_champs(elements, &joindre(chemin, "0"), chemins);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn manifeste() -> Value {
        let sha256 = "a".repeat(64);
        json!({
            "schema": "kvu-regen/1",
            "origine": {
                "schema": "su2_hip/1", "manifeste": "origine.json",
                "sha256_manifeste": sha256
            },
            "binaire_sha256": sha256,
            "git_head": "0123456789abcdef",
            "source_hash": sha256,
            "noyaux_embarques": [{"nom": "su2", "sha256": sha256}],
            "modules_compiles": [{
                "source_sha256": sha256, "options": "-O3", "arguments": ["-O3"],
                "objet_sha256": sha256
            }],
            "backend": "hip", "arch": "gfx1151",
            "versions": {"hip": "7.0"},
            "machine": {"nom": "synthétique", "cpu": "CPU", "gpu": "gfx1151", "ram": 64},
            "graines": {"cle": 42, "etiquettes": ["initialisation"]},
            "parametres": {
                "groupe": "SU(2)", "L": 12, "beta": 2.5, "start": "hot",
                "therm": 10, "nmeas": 20, "skip": 2, "nor": 1,
                "precision": "f64", "options": "ref-k1"
            },
            "commande": ["kvu", "su2"],
            "entrees": [{"nom": "initial.bin", "sha256": sha256}],
            "fichiers": [{"nom": "mesures.bin", "sha256": sha256, "octets": 128}],
            "duree_s": 1.5, "statut": "COMPLETE", "classe_regen": "A",
            "regen_verifiee": {"date": null, "machine": null, "resultat": "non_verifiee"}
        })
    }

    fn verifier_rejet(chemin: &str, valeur: Value) {
        let mut instance = manifeste();
        *instance
            .pointer_mut(chemin)
            .expect("le champ du témoin existe") = valeur;
        let erreurs = valider_manifeste(&instance);
        assert!(
            erreurs
                .iter()
                .any(|erreur| erreur.starts_with(&format!("{chemin} : "))),
            "mutation de {chemin} non détectée au bon chemin : {erreurs:?}"
        );
    }

    #[test]
    fn schema_embarque_analysable_et_champs_racine_declares() {
        let valeur: Value = serde_json::from_str(SCHEMA_TEXTE).unwrap();
        assert_eq!(schema(), valeur);
        let requis = valeur["required"].as_array().unwrap();
        for nom in ["schema", "origine", "regen_verifiee"] {
            assert!(
                requis.contains(&json!(nom)),
                "champ obligatoire absent : {nom}"
            );
        }
        assert!(valider(&valeur, &manifeste()).is_ok());
    }

    #[test]
    fn manifeste_synthetique_accepte() {
        assert_eq!(valider_manifeste(&manifeste()), Vec::<String>::new());
    }

    #[test]
    fn tous_les_mutants_champ_retire_sont_rejetes() {
        let regle = schema();
        let temoin = manifeste();
        let mutants = mutants_champ_retire(&regle, &temoin);
        assert!(
            mutants.len() >= 20,
            "au moins vingt mutants sont nécessaires"
        );
        assert_eq!(mutants.len(), champs_obligatoires(&regle).len());
        for (chemin, mutant) in mutants {
            assert!(
                mutant.pointer(&chemin).is_none(),
                "champ conservé : {chemin}"
            );
            let erreurs = valider(&regle, &mutant).unwrap();
            assert!(!erreurs.is_empty(), "champ retiré non détecté : {chemin}");
            assert!(
                erreurs
                    .iter()
                    .any(|erreur| erreur.starts_with(&format!("{chemin} : "))),
                "chemin absent du diagnostic : {chemin}, {erreurs:?}"
            );
        }
        assert_eq!(temoin, manifeste(), "le témoin doit rester intact");
    }

    #[test]
    fn sha256_trop_court_rejete() {
        verifier_rejet("/binaire_sha256", json!("a".repeat(63)));
    }

    #[test]
    fn sha256_majuscule_rejete() {
        verifier_rejet("/binaire_sha256", json!(format!("{}A", "a".repeat(63))));
    }

    #[test]
    fn sha256_trop_long_rejete() {
        verifier_rejet("/source_hash", json!("a".repeat(65)));
    }

    #[test]
    fn sha256_non_hexadecimal_rejete() {
        verifier_rejet("/source_hash", json!(format!("{}g", "a".repeat(63))));
    }

    #[test]
    fn classe_d_rejetee() {
        verifier_rejet("/classe_regen", json!("D"));
    }

    #[test]
    fn taille_booleenne_rejetee() {
        verifier_rejet("/parametres/L", json!(true));
    }

    #[test]
    fn taille_un_rejetee() {
        verifier_rejet("/parametres/L", json!(1));
    }

    #[test]
    fn taille_flottante_rejetee() {
        verifier_rejet("/parametres/L", json!(12.0));
    }

    #[test]
    fn statut_absent_rejete() {
        let mut instance = manifeste();
        instance.as_object_mut().unwrap().remove("statut");
        assert!(valider_manifeste(&instance).contains(&"/statut : champ obligatoire absent".into()));
    }

    #[test]
    fn commande_vide_rejetee() {
        verifier_rejet("/commande", json!([]));
    }

    #[test]
    fn classe_a_binaire_null_rejete() {
        verifier_rejet("/binaire_sha256", Value::Null);
    }

    #[test]
    fn classe_b_binaire_null_rejete() {
        let mut instance = manifeste();
        instance["classe_regen"] = json!("B");
        instance["binaire_sha256"] = Value::Null;
        assert!(!valider_manifeste(&instance).is_empty());
    }

    #[test]
    fn classe_c_binaire_null_accepte() {
        let mut instance = manifeste();
        instance["classe_regen"] = json!("C");
        instance["binaire_sha256"] = Value::Null;
        assert!(valider_manifeste(&instance).is_empty());
    }

    #[test]
    fn classe_a_source_null_rejetee() {
        verifier_rejet("/source_hash", Value::Null);
    }

    #[test]
    fn classe_a_graine_null_rejetee() {
        verifier_rejet("/graines/cle", Value::Null);
    }

    #[test]
    fn classe_c_source_et_graine_null_acceptees() {
        let mut instance = manifeste();
        instance["classe_regen"] = json!("C");
        instance["source_hash"] = Value::Null;
        instance["graines"]["cle"] = Value::Null;
        assert!(valider_manifeste(&instance).is_empty());
    }

    #[test]
    fn mot_cle_inconnu_signale() {
        let erreur = valider(&json!({"maximum": 3}), &json!(2)).unwrap_err();
        assert!(erreur.contains("maximum"));
    }

    #[test]
    fn mot_cle_inconnu_dans_propriete_absente_signale() {
        assert!(valider(
            &json!({"properties": {"absent": {"pattern": "x"}}}),
            &json!({})
        )
        .is_err());
    }

    #[test]
    fn mot_cle_inconnu_dans_items_vide_signale() {
        assert!(valider(&json!({"items": {"maximum": 2}}), &json!([])).is_err());
    }

    #[test]
    fn mot_cle_inconnu_dans_then_inactif_signale() {
        let regle = json!({"if": {"const": 1}, "then": {"maximum": 2}});
        assert!(valider(&regle, &json!(0)).is_err());
    }

    #[test]
    fn annotations_ignorees_et_const_null_respectee() {
        let regle = json!({"$schema": "annotation", "$id": "id", "title": "titre",
            "description": "texte", "const": null});
        assert!(valider(&regle, &Value::Null).unwrap().is_empty());
        assert!(!valider(&regle, &json!(0)).unwrap().is_empty());
    }

    #[test]
    fn entier_distingue_flottants_et_booleens() {
        let regle = json!({"type": "integer"});
        for valeur in [json!(0), json!(-1), json!(u64::MAX), json!(i64::MIN)] {
            assert!(
                valider(&regle, &valeur).unwrap().is_empty(),
                "entier rejeté : {valeur}"
            );
        }
        for valeur in [json!(2.0), json!(true), json!(false), json!("2")] {
            assert!(
                !valider(&regle, &valeur).unwrap().is_empty(),
                "faux entier accepté : {valeur}"
            );
        }
    }

    #[test]
    fn nombre_accepte_entier_et_flottant_exclut_booleen() {
        let regle = json!({"type": "number"});
        for valeur in [json!(2), json!(2.0), json!(-0.5)] {
            assert!(valider(&regle, &valeur).unwrap().is_empty());
        }
        assert!(!valider(&regle, &json!(true)).unwrap().is_empty());
    }

    #[test]
    fn chaque_type_simple_respecte() {
        for (nom, valeur) in [
            ("string", json!("texte")),
            ("boolean", json!(true)),
            ("object", json!({})),
            ("array", json!([])),
            ("null", Value::Null),
        ] {
            let regle = json!({"type": nom});
            assert!(
                valider(&regle, &valeur).unwrap().is_empty(),
                "type rejeté : {nom}"
            );
            assert!(
                !valider(&regle, &json!(17)).unwrap().is_empty(),
                "type ignoré : {nom}"
            );
        }
    }

    #[test]
    fn liste_de_types_acceptee() {
        let regle = json!({"type": ["string", "null"]});
        for valeur in [json!("texte"), Value::Null] {
            assert!(valider(&regle, &valeur).unwrap().is_empty());
        }
        assert!(!valider(&regle, &json!(false)).unwrap().is_empty());
    }

    #[test]
    fn mots_cles_specifiques_ignores_pour_autres_types() {
        let regle = json!({"required": ["x"], "properties": {"x": {"const": 0}},
            "additionalProperties": false, "items": {"const": 0}, "minItems": 3,
            "minimum": 10, "minLength": 2, "format": "sha256"});
        for valeur in [Value::Null, json!(false)] {
            assert!(valider(&regle, &valeur).unwrap().is_empty());
        }
    }

    #[test]
    fn longueur_mesuree_en_caracteres() {
        let regle = json!({"minLength": 2});
        assert!(!valider(&regle, &json!("é")).unwrap().is_empty());
        assert!(valider(&regle, &json!("é🦀")).unwrap().is_empty());
    }

    #[test]
    fn minimum_inclusif_et_precision_entiere_preservee() {
        let regle = json!({"minimum": 2});
        assert!(valider(&regle, &json!(2)).unwrap().is_empty());
        assert!(valider(&regle, &json!(2.0)).unwrap().is_empty());
        assert!(!valider(&regle, &json!(1.5)).unwrap().is_empty());
        let regle = json!({"minimum": 9_007_199_254_740_993_u64});
        assert!(!valider(&regle, &json!(9_007_199_254_740_992_u64))
            .unwrap()
            .is_empty());
        assert!(!valider(&json!({"minimum": u64::MAX}), &json!(-1))
            .unwrap()
            .is_empty());
        assert!(!valider(&regle, &json!(9_007_199_254_740_992.0_f64))
            .unwrap()
            .is_empty());
        assert!(!valider(
            &json!({"minimum": 18_446_744_073_709_551_616.0_f64}),
            &json!(u64::MAX)
        )
        .unwrap()
        .is_empty());
        assert!(valider(&json!({"minimum": -0.5}), &json!(0))
            .unwrap()
            .is_empty());
        assert!(!valider(&json!({"minimum": 0.5}), &json!(0))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn proprietes_supplementaires_interdites() {
        let regle =
            json!({"properties": {"x": {"type": "integer"}}, "additionalProperties": false});
        assert!(valider(&regle, &json!({"x": 1})).unwrap().is_empty());
        assert_eq!(
            valider(&regle, &json!({"x": 1, "y": 2})).unwrap(),
            vec!["/y : champ supplémentaire interdit"]
        );
        assert!(
            !valider(&json!({"additionalProperties": false}), &json!({"x": 1}))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn proprietes_supplementaires_autorisees_par_defaut_ou_explicitement() {
        for regle in [json!({}), json!({"additionalProperties": true})] {
            assert!(valider(&regle, &json!({"x": 1})).unwrap().is_empty());
        }
    }

    #[test]
    fn items_tous_valides_et_chemin_indice() {
        let regle = json!({"items": {"type": "integer"}});
        assert!(valider(&regle, &json!([1, 2])).unwrap().is_empty());
        let erreurs = valider(&regle, &json!([1, "deux"])).unwrap();
        assert_eq!(erreurs.len(), 1);
        assert!(erreurs[0].starts_with("/1 : "));
    }

    #[test]
    fn enumeration_et_constante() {
        let regle = json!({"enum": ["A", "C"]});
        assert!(valider(&regle, &json!("C")).unwrap().is_empty());
        assert!(!valider(&regle, &json!("B")).unwrap().is_empty());
        let regle = json!({"const": {"valeurs": [2, 0]}});
        assert!(valider(&regle, &json!({"valeurs": [2.0, -0.0]}))
            .unwrap()
            .is_empty());
        assert!(!valider(&regle, &json!({"valeurs": [2, false]}))
            .unwrap()
            .is_empty());
        assert!(valider(&json!({"enum": [2]}), &json!(2.0))
            .unwrap()
            .is_empty());
        assert!(!valider(&json!({"enum": [1]}), &json!(true))
            .unwrap()
            .is_empty());
        assert!(!valider_manifeste(&{
            let mut instance = manifeste();
            instance["schema"] = json!("autre/1");
            instance
        })
        .is_empty());
    }

    #[test]
    fn if_sans_then_et_then_sans_if_n_imposent_rien() {
        assert!(valider(&json!({"if": {"const": 1}}), &json!(0))
            .unwrap()
            .is_empty());
        assert!(valider(&json!({"then": {"const": 1}}), &json!(0))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn champs_obligatoires_tries_et_chemins_tableaux() {
        let chemins = champs_obligatoires(&schema());
        assert!(chemins.windows(2).all(|paire| paire[0] < paire[1]));
        for chemin in [
            "/machine/gpu",
            "/fichiers/0/octets",
            "/modules_compiles/0/arguments",
        ] {
            assert!(
                chemins.iter().any(|valeur| valeur == chemin),
                "chemin absent : {chemin}"
            );
        }
    }

    #[test]
    fn champs_sous_if_then_non_explores() {
        let regle = json!({"required": ["x"], "if": {"required": ["condition"]},
            "then": {"required": ["consequence"]},
            "properties": {"facultatif": {"required": ["interne"]}}});
        assert_eq!(
            champs_obligatoires(&regle),
            vec!["/facultatif/interne", "/x"]
        );
    }

    #[test]
    fn mutants_ignorent_les_champs_absents_et_les_tableaux_vides() {
        let regle = json!({"required": ["absent"], "properties": {
            "liste": {"items": {"required": ["x"]}},
            "objet": {"required": ["x", "absent"]}
        }});
        let instance = json!({"liste": [], "objet": {"x": 1}});
        assert_eq!(
            mutants_champ_retire(&regle, &instance),
            vec![("/objet/x".into(), json!({"liste": [], "objet": {}}))]
        );
    }

    #[test]
    fn mutant_tableau_ne_retire_que_le_champ_du_premier_element() {
        let regle = json!({"properties": {"liste": {"items": {"required": ["x"]}}}});
        let instance = json!({"liste": [{"x": 1}, {"x": 2}]});
        assert_eq!(
            mutants_champ_retire(&regle, &instance),
            vec![("/liste/0/x".into(), json!({"liste": [{}, {"x": 2}]}))]
        );
    }

    #[test]
    fn chemins_json_echappes_et_mutants_coherents() {
        let regle = json!({"required": ["a/b~c"], "properties": {
            "a/b~c": {"required": ["", "~1"]}
        }});
        let instance = json!({"a/b~c": {"": 1, "~1": 2}});
        assert_eq!(
            champs_obligatoires(&regle),
            vec!["/a~1b~0c", "/a~1b~0c/", "/a~1b~0c/~01"]
        );
        let mutants = mutants_champ_retire(&regle, &instance);
        assert_eq!(mutants.len(), 3);
        for (chemin, mutant) in mutants {
            assert!(
                mutant.pointer(&chemin).is_none(),
                "champ non retiré : {chemin}"
            );
            assert!(valider(&regle, &mutant)
                .unwrap()
                .iter()
                .any(|erreur| erreur.starts_with(&format!("{chemin} : "))));
        }
    }

    #[test]
    fn erreur_racine_identifiee() {
        assert!(valider(&json!({"type": "object"}), &Value::Null).unwrap()[0].starts_with("/ : "));
    }

    #[test]
    fn usages_non_geres_signales() {
        for regle in [
            json!({"format": "date"}),
            json!({"type": "inconnu"}),
            json!({"additionalProperties": {}}),
            json!({"items": []}),
            json!({"required": "x"}),
            json!({"minItems": -1}),
            json!(true),
        ] {
            assert!(
                valider(&regle, &Value::Null).is_err(),
                "usage accepté : {regle}"
            );
        }
    }

    #[test]
    fn sha256_chiffres_et_lettres_minuscules_acceptes() {
        let regle = json!({"format": "sha256"});
        assert!(valider(&regle, &json!("0123456789abcdef".repeat(4)))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn accumulation_des_erreurs_independantes() {
        let regle = json!({"required": ["x", "y"], "properties": {"z": {"type": "number"}}});
        let erreurs = valider(&regle, &json!({"z": false})).unwrap();
        assert_eq!(erreurs.len(), 3);
    }
}
