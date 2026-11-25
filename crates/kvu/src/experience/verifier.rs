// SPDX-License-Identifier: Apache-2.0
//! Contrôles V1 à V10, en lecture seule, avec constats contextualisés.

use super::modele::{classe_experience, classe_machine, construire_run, graines_distinctes};
use super::*;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Serialize)]
pub struct Constat {
    pub controle: u8,
    pub message: String,
    pub ok: bool,
    /// Constat non rouge qui interdit un « vert » plein (ex. gel déclaré, non observé).
    pub avertissement: bool,
}

fn rouge(constats: &mut Vec<Constat>, controle: u8, message: impl Into<String>) {
    constats.push(Constat {
        controle,
        message: message.into(),
        ok: false,
        avertissement: false,
    });
}

fn empreinte(racine: &Path, chemin: &str, attendu: &str, constats: &mut Vec<Constat>) {
    match resoudre(racine, chemin).and_then(|p| sha_fichier(&p)) {
        Ok(obtenu) if obtenu == attendu => {}
        Ok(obtenu) => rouge(
            constats,
            2,
            format!("{chemin} : sha256 attendu {attendu}, obtenu {obtenu}"),
        ),
        Err(e) => rouge(constats, 2, e.to_string()),
    }
}

fn empreinte_piece(racine: &Path, piece: &Value, constats: &mut Vec<Constat>) {
    empreinte(
        racine,
        texte(piece, "chemin"),
        texte(piece, "sha256"),
        constats,
    );
}

/// Vérifie les pièces présentes, sans corriger ni réécrire le manifeste.
pub fn verifier(racine: &Path) -> Resultat<Vec<Constat>> {
    let racine = canonique(racine)?;
    let racine = racine.as_path();
    let mut constats = Vec::new();
    let v = match lire_json(&racine.join(MANIFESTE)) {
        Ok(v) => v,
        Err(e) => {
            rouge(&mut constats, 1, e.to_string());
            return Ok(constats);
        }
    };
    for erreur in schema::valider_experience(&v) {
        rouge(&mut constats, 1, erreur);
    }
    for p in tableau(&v, "prereg") {
        empreinte_piece(racine, p, &mut constats);
        if let Some(chemin) = p.get("hachage_consigne").and_then(Value::as_str) {
            empreinte(
                racine,
                chemin,
                texte(p, "hachage_consigne_sha256"),
                &mut constats,
            );
        }
    }
    for cle in ["garde", "sha256sums"] {
        if let Some(piece) = v.get("anteriorite").and_then(|a| a.get(cle)) {
            empreinte_piece(racine, piece, &mut constats);
        }
    }
    for run in tableau(&v, "runs") {
        empreinte(
            racine,
            texte(run, "manifeste"),
            texte(run, "sha256"),
            &mut constats,
        );
        if let Some(sortie) = run.get("sortie") {
            empreinte_piece(racine, sortie, &mut constats);
        }
    }
    for porte in tableau(&v, "portes") {
        empreinte_piece(racine, &porte["source"], &mut constats);
    }

    for cle in ["prereg", "runs", "portes", "carnet"] {
        let mut ids = BTreeSet::new();
        for piece in tableau(&v, cle) {
            let id = texte(piece, "id");
            if !ids.insert(id) {
                rouge(&mut constats, 3, format!("{cle} : id {id} dupliqué"));
            }
        }
    }
    for p in tableau(&v, "prereg") {
        if let Some(chemin) = p.get("hachage_consigne").and_then(Value::as_str) {
            match resoudre(racine, chemin).and_then(|p| lire_texte(&p)) {
                Ok(contenu) if contenu.split_whitespace().next() == p["sha256"].as_str() => {}
                Ok(_) => rouge(
                    &mut constats,
                    3,
                    format!(
                        "{chemin} : premier mot différent du sha256 de {}",
                        texte(p, "id")
                    ),
                ),
                Err(e) => rouge(&mut constats, 3, e.to_string()),
            }
        }
    }
    let preregs: BTreeMap<_, _> = tableau(&v, "prereg")
        .iter()
        .map(|p| (texte(p, "id"), p))
        .collect();
    // PREREG racine = P0, créé par `init` : aucun run ne peut le précéder, quels que soient les PREREG
    // auxquels on l'a rattaché ; un addendum ne peut pas non plus être gelé avant lui.
    let racine_gel = tableau(&v, "prereg")
        .iter()
        .find(|p| texte(p, "id") == "P0")
        .and_then(|p| Some((secondes_unix(texte(p, "gele_le")).ok()?, texte(p, "id"))));
    if racine_gel.is_none() {
        rouge(
            &mut constats,
            4,
            "PREREG racine P0 absent ou sans date de gel valide",
        );
    }
    if let Some((gel, pid)) = racine_gel {
        for p in tableau(&v, "prereg")
            .iter()
            .filter(|p| texte(p, "id") != "P0")
        {
            if secondes_unix(texte(p, "gele_le")).is_ok_and(|g| g < gel) {
                rouge(
                    &mut constats,
                    4,
                    format!(
                        "PREREG {} gelé avant le PREREG racine {pid}",
                        texte(p, "id")
                    ),
                );
            }
        }
    }
    for run in tableau(&v, "runs") {
        let id = texte(run, "id");
        let debut = secondes_unix(texte(run, "debut"));
        match debut {
            Err(e) => rouge(&mut constats, 4, format!("run {id} : {e}")),
            Ok(debut) => {
                let cites: Vec<&str> = tableau(run, "prereg")
                    .iter()
                    .filter_map(Value::as_str)
                    .collect();
                if let Some((gel, pid)) = racine_gel {
                    if debut < gel && !cites.contains(&pid) {
                        rouge(
                            &mut constats,
                            4,
                            format!(
                                "run {id} avant le PREREG racine {pid} : écart {} secondes",
                                gel - debut
                            ),
                        );
                    }
                }
                for p in tableau(&v, "prereg") {
                    let pid = texte(p, "id");
                    if p["change_un_critere"] == true
                        && !cites.contains(&pid)
                        && secondes_unix(texte(p, "gele_le")).is_ok_and(|gel| debut >= gel)
                    {
                        rouge(
                            &mut constats,
                            4,
                            format!("run {id} postérieur à {pid} (critère changé) sans le citer"),
                        );
                    }
                }
                for pid in tableau(run, "prereg") {
                    let pid = pid.as_str().unwrap_or("");
                    match preregs.get(pid) {
                        None => rouge(&mut constats, 4, format!("run {id} : PREREG {pid} inconnu")),
                        Some(p) => match secondes_unix(texte(p, "gele_le")) {
                            Ok(gel) if debut < gel => rouge(
                                &mut constats,
                                4,
                                format!(
                                    "run {id} avant PREREG {pid} : écart {} secondes",
                                    gel - debut
                                ),
                            ),
                            Ok(_) => {}
                            Err(e) => rouge(&mut constats, 4, format!("PREREG {pid} : {e}")),
                        },
                    }
                }
            }
        }
    }
    for p in tableau(&v, "prereg") {
        let id = texte(p, "id");
        let remplace = p.get("remplace").and_then(Value::as_str);
        if p["change_un_critere"] == true {
            match remplace.and_then(|id| preregs.get(id)) {
                Some(ancien) if p["chemin"] != ancien["chemin"] && remplace != Some(id) => {}
                _ => rouge(
                    &mut constats,
                    5,
                    format!("{id} : critère changé sans PREREG remplacé dans un fichier différent"),
                ),
            }
        } else if remplace.is_some_and(|id| !preregs.contains_key(id)) {
            rouge(
                &mut constats,
                5,
                format!("{id} : préenregistrement remplacé inconnu"),
            );
        }
    }
    let mut graines = BTreeMap::new();
    for run in tableau(&v, "runs") {
        if tableau(run, "graines").is_empty() {
            rouge(
                &mut constats,
                6,
                format!(
                    "run {} : graine non déclarée (aucune graine lue)",
                    texte(run, "id")
                ),
            );
        }
        for graine in tableau(run, "graines").iter().filter_map(Value::as_u64) {
            if let Some(ancien) = graines.insert(graine, texte(run, "id")) {
                rouge(
                    &mut constats,
                    6,
                    format!(
                        "graine {graine} répétée : runs {ancien} et {}",
                        texte(run, "id")
                    ),
                );
            }
        }
    }
    if v["graines"]["toutes_distinctes"].as_bool() != Some(graines_distinctes(&v)) {
        rouge(&mut constats, 6, "graines.toutes_distinctes incohérent");
    }
    let principale = texte(&v, "machine_principale");
    let mut machines = BTreeMap::new();
    for machine in tableau(&v, "machines") {
        let nom = texte(machine, "nom");
        if machines.insert(nom, machine).is_some() {
            rouge(&mut constats, 7, format!("machine {nom} dupliquée"));
        }
        let classe = classe_machine(nom, principale);
        if texte(machine, "classe") != classe {
            rouge(
                &mut constats,
                7,
                format!("machine {nom} : classe {classe} attendue"),
            );
        }
    }
    let classe = classe_experience(&v);
    if texte(&v["verdict"], "classe") != classe {
        rouge(
            &mut constats,
            7,
            format!("verdict.classe : {classe} attendue"),
        );
    }
    let mut binaires = BTreeSet::new();
    for b in tableau(&v["code"], "binaires") {
        if !binaires.insert(b.to_string()) {
            rouge(&mut constats, 7, "code.binaires contient un doublon");
        }
    }
    for run in tableau(&v, "runs") {
        let id = texte(run, "id");
        if !machines.contains_key(texte(run, "machine")) {
            rouge(
                &mut constats,
                7,
                format!("run {id} : machine inconnue {}", texte(run, "machine")),
            );
        }
        if !tableau(&v["code"], "binaires")
            .iter()
            .any(|b| b["sha256"] == run["binaire_sha256"])
        {
            rouge(
                &mut constats,
                7,
                format!("run {id} : binaire absent de code.binaires"),
            );
        }
    }
    for porte in tableau(&v, "portes") {
        if porte["statut"] == "rouge_attendu" && texte(porte, "mutant").trim().is_empty() {
            rouge(
                &mut constats,
                8,
                format!(
                    "porte {} : rouge_attendu exige un mutant",
                    texte(porte, "id")
                ),
            );
        }
    }
    for p in tableau(&v, "carnet") {
        for source in tableau(p, "sources") {
            if let Err(e) = verifier_source(racine, source.as_str().unwrap_or("")) {
                rouge(&mut constats, 9, format!("{} : {e}", texte(p, "id")));
            }
        }
    }
    for run in tableau(&v, "runs") {
        let id = texte(run, "id");
        match (
            secondes_unix(texte(run, "debut")),
            secondes_unix(texte(run, "fin")),
        ) {
            (Ok(debut), Ok(fin)) if fin >= debut => {}
            (Ok(_), Ok(_)) => rouge(
                &mut constats,
                10,
                format!("run {id} : fin antérieure au début"),
            ),
            _ => rouge(&mut constats, 10, format!("run {id} : dates invalides")),
        }
        if run["rc"].as_i64().is_none() {
            rouge(&mut constats, 10, format!("run {id} : rc entier attendu"));
        }
        let chemin = texte(run, "manifeste");
        match resoudre(racine, chemin).and_then(|p| lire_json(&p)) {
            Err(e) => rouge(&mut constats, 10, format!("run {id} : {e}")),
            Ok(original) => {
                let preregs: Vec<String> = tableau(run, "prereg")
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect();
                match construire_run(racine, chemin, &original, texte(run, "machine"), &preregs) {
                    Err(e) => rouge(&mut constats, 10, format!("run {id} : {e}")),
                    Ok((lu, binaire, description)) => {
                        for cle in [
                            "id",
                            "schema",
                            "hote",
                            "debut",
                            "fin",
                            "rc",
                            "graines",
                            "binaire_sha256",
                            "geometrie",
                            "parametres",
                            "sortie",
                        ] {
                            let conforme = if matches!(cle, "debut" | "fin") {
                                secondes_unix(texte(&lu, cle)).ok()
                                    == secondes_unix(texte(run, cle)).ok()
                            } else {
                                lu[cle] == run[cle]
                            };
                            if !conforme {
                                rouge(
                                    &mut constats,
                                    10,
                                    format!("run {id} : champ {cle} différent du manifeste lu"),
                                );
                            }
                        }
                        if !tableau(&v["code"], "binaires").contains(&binaire) {
                            rouge(
                                &mut constats,
                                7,
                                format!(
                                    "run {id} : métadonnées du binaire différentes du manifeste lu"
                                ),
                            );
                        }
                        if let Some(machine) = machines.get(texte(run, "machine")) {
                            for cle in ["hote", "backend", "arch"] {
                                if machine[cle] != description[cle] {
                                    rouge(
                                        &mut constats,
                                        7,
                                        format!("run {id} : machine, champ {cle} incohérent"),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // V11 : historique du verdict chaîné par hachage, dates croissantes, tête annoncée par `verdict.chaine`.
    let chaine_presente = v["verdict"].get("chaine").is_some();
    let liens_presents = tableau(&v["verdict"], "historique")
        .iter()
        .any(|e| e.get("precedent").is_some());
    if !chaine_presente && liens_presents {
        rouge(
            &mut constats,
            11,
            "historique du verdict partiellement chaîné (liens présents sans verdict.chaine) : corruption possible, aucune migration",
        );
    }
    let mut attendu = EMPREINTE_NULLE.to_owned();
    let mut derniere_date = 0;
    for (i, e) in tableau(&v["verdict"], "historique").iter().enumerate() {
        if chaine_presente && texte(e, "precedent") != attendu {
            rouge(
                &mut constats,
                11,
                format!("historique du verdict : entrée {i} ne prolonge pas la précédente (chaîne rompue)"),
            );
        }
        match secondes_unix(texte(e, "remplace_le")) {
            Ok(t) if t >= derniere_date => derniere_date = t,
            _ => rouge(
                &mut constats,
                11,
                format!("historique du verdict : entrée {i} : date absente ou décroissante"),
            ),
        }
        attendu = empreinte_json(e);
    }
    if chaine_presente && texte(&v["verdict"], "chaine") != attendu {
        rouge(
            &mut constats,
            11,
            "verdict.chaine ne désigne pas la dernière entrée de l'historique (historique tronqué ou modifié)",
        );
    }
    for (controle, message) in avertissements(&v) {
        // Un manifeste partiellement chaîné est déjà rouge : pas d'avertissement de migration en plus.
        if controle == 11 && !chaine_presente && liens_presents {
            continue;
        }
        constats.push(Constat {
            controle,
            message,
            ok: false,
            avertissement: true,
        });
    }
    for controle in 1..=11 {
        if !constats.iter().any(|c| c.controle == controle) {
            constats.push(Constat {
                controle,
                message: "contrôle satisfait".into(),
                ok: true,
                avertissement: false,
            });
        }
    }
    constats.sort_by_key(|c| c.controle);
    Ok(constats)
}
