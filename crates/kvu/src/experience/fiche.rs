// SPDX-License-Identifier: Apache-2.0
//! Fiche Markdown déterministe, produite uniquement à partir du manifeste.

use super::modele::classe_experience;
use super::*;

pub(super) fn rendre(v: &Value) -> String {
    let classe = classe_experience(v);
    let mut lignes = vec![
        format!("# {} : {}", texte(v, "id"), texte(v, "titre")),
        String::new(),
        texte(v, "question").into(),
        String::new(),
        format!(
            "{} run(s), {} machine(s), classe {classe}.",
            tableau(v, "runs").len(),
            tableau(v, "machines").len()
        ),
        String::new(),
        "## Préenregistrements".into(),
        String::new(),
    ];
    for p in tableau(v, "prereg") {
        lignes.push(format!(
            "- {} : {} ; gel {} ({}) ; sha256 `{}`.",
            texte(p, "id"),
            texte(p, "chemin"),
            texte(p, "gele_le"),
            texte(p, "gele_source"),
            texte(p, "sha256")
        ));
    }
    lignes.extend([String::new(), "## Machines et runs".into(), String::new()]);
    for machine in tableau(v, "machines") {
        lignes.push(format!(
            "- Machine {} : classe {}, hôte {}, backend {}, architecture {}.",
            texte(machine, "nom"),
            texte(machine, "classe"),
            texte(machine, "hote"),
            texte(machine, "backend"),
            texte(machine, "arch")
        ));
    }
    // Un résumé par (machine, PREREG cités) : le détail des runs est dans EXPERIENCE.json.
    let mut groupes: std::collections::BTreeMap<String, (usize, String, String, usize)> =
        std::collections::BTreeMap::new();
    for run in tableau(v, "runs") {
        let cle = format!(
            "machine {}, PREREG {}",
            texte(run, "machine"),
            run["prereg"]
        );
        let g = groupes.entry(cle).or_insert((
            0,
            texte(run, "debut").into(),
            texte(run, "fin").into(),
            0,
        ));
        g.0 += 1;
        if texte(run, "debut") < g.1.as_str() {
            g.1 = texte(run, "debut").into();
        }
        if texte(run, "fin") > g.2.as_str() {
            g.2 = texte(run, "fin").into();
        }
        g.3 += usize::from(run["rc"] != 0);
    }
    for (cle, (n, debut, fin, rc)) in groupes {
        lignes.push(format!(
            "- Runs {cle} : {n} run(s), du {debut} au {fin}, rc non nul : {rc}."
        ));
    }
    lignes.extend([String::new(), "## Portes".into(), String::new()]);
    for porte in tableau(v, "portes") {
        lignes.push(format!(
            "- {} : {} ; {} ; seuil {}, valeur {} ; source {}.",
            texte(porte, "id"),
            texte(porte, "statut"),
            texte(porte, "enonce"),
            porte["seuil"],
            porte["valeur"],
            texte(&porte["source"], "chemin")
        ));
    }
    lignes.extend([
        String::new(),
        "## Verdict".into(),
        String::new(),
        format!(
            "{} ; provisoire : {} ; classe {classe}.",
            texte(&v["verdict"], "statut"),
            v["verdict"]["provisoire"]
        ),
        texte(&v["verdict"], "texte").into(),
    ]);
    for chiffre in tableau(&v["verdict"], "chiffres") {
        lignes.push(format!(
            "- {} = {} {} ; source {}.",
            texte(chiffre, "nom"),
            chiffre["valeur"],
            texte(chiffre, "statut_lecture"),
            texte(chiffre, "source")
        ));
    }
    let avertissements = avertissements(v);
    if !avertissements.is_empty() {
        lignes.extend([String::new(), "## Avertissements".into(), String::new()]);
        lignes.extend(
            avertissements
                .iter()
                .map(|(_, a)| format!("- AVERTISSEMENT : {a}")),
        );
    }
    lignes.extend([String::new(), "## Réserves".into(), String::new()]);
    for reserve in tableau(v, "reserves") {
        lignes.push(format!("- {}", reserve.as_str().unwrap_or("")));
    }
    lignes.push(String::new());
    lignes.join("\n")
}
