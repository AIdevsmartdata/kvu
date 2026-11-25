// SPDX-License-Identifier: Apache-2.0
//! Export compatible avec le carnet : en-têtes YAML et versions immuables.

use super::*;
use crate::regen::manifeste::sha_octets;
use std::collections::BTreeSet;

const DOSSIERS: &[&str] = &[
    "experiences",
    "resultats",
    "erreurs",
    "decisions",
    "hypotheses",
    "pistes",
    "faits",
];

fn dossier(typ: &str) -> Resultat<&'static str> {
    match typ {
        "experience" => Ok("experiences"),
        "resultat" => Ok("resultats"),
        "erreur" => Ok("erreurs"),
        "decision" => Ok("decisions"),
        "hypothese" => Ok("hypotheses"),
        "piste" => Ok("pistes"),
        "fait" => Ok("faits"),
        _ => Err(Erreur::Validation(format!(
            "type de carnet inconnu : {typ}"
        ))),
    }
}

fn chaine(s: &str) -> Resultat<String> {
    serde_json::to_string(s).map_err(|e| Erreur::Format(e.to_string()))
}

fn joindre_source(projet: &str, chemin: &str, ligne: usize) -> String {
    if projet == "." {
        format!("{chemin}:{ligne}")
    } else {
        format!("{projet}/{chemin}:{ligne}")
    }
}

fn entree_experience(v: &Value) -> Value {
    let verdict = &v["verdict"];
    let statut = if tableau(v, "runs").is_empty() {
        "prereg".into()
    } else if texte(verdict, "statut") == "en_cours" || verdict["provisoire"] == true {
        "en_cours".into()
    } else {
        format!("verdict:{}", texte(verdict, "statut"))
    };
    let fiche = super::fiche::rendre(v);
    let corps = fiche
        .split_once('\n')
        .map(|(_, suite)| suite.trim_start())
        .unwrap_or("");
    let liens: Vec<_> = tableau(v, "carnet")
        .iter()
        .map(|p| texte(p, "id"))
        .collect();
    json!({
        "type": "experience", "id": v["carnet_id"], "titre": v["titre"],
        "date": tableau(v, "prereg").first().map(|p| texte(p, "gele_le")).unwrap_or(""),
        "statut": statut, "enonce": corps, "sources": [format!(
            "{}:1",
            tableau(v, "prereg").first().map(|p| texte(p, "chemin")).unwrap_or("")
        )],
        "liens": liens
    })
}

struct Export<'a> {
    experience: &'a Experience,
    destination: &'a Path,
    empreinte: String,
}

impl Export<'_> {
    fn rendre(&self, entree: &Value, id: &str, precedent: Option<&str>) -> Resultat<String> {
        let v = &self.experience.manifeste;
        let projet = v
            .get("chemin_projet")
            .and_then(Value::as_str)
            .unwrap_or(".");
        verifier_relatif(projet, true)?;
        let date = normaliser_date(texte(entree, "date"))?;
        let mut lignes = vec![
            "---".into(),
            format!("id: {id}"),
            format!("titre: {}", chaine(texte(entree, "titre"))?),
            format!("date: {}", chaine(&date[..16])?),
            format!("type: {}", texte(entree, "type")),
            format!("statut: {}", chaine(texte(entree, "statut"))?),
            "sources:".into(),
        ];
        for source in tableau(entree, "sources") {
            let source = verifier_source(&self.experience.racine, source.as_str().unwrap_or(""))?;
            lignes.push(format!(
                "  - {}",
                chaine(&joindre_source(projet, &source.chemin, source.debut))?
            ));
            if source.fin != source.debut {
                lignes.push(format!(
                    "  - {}",
                    chaine(&joindre_source(projet, &source.chemin, source.fin))?
                ));
            }
        }
        let mut liens: Vec<String> = tableau(entree, "liens")
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        if let Some(precedent) = precedent {
            liens.push(precedent.into());
        }
        liens.sort();
        liens.dedup();
        lignes.push(format!(
            "liens: {}",
            serde_json::to_string(&liens).map_err(|e| Erreur::Format(e.to_string()))?
        ));
        lignes.push(format!("experience: {}", texte(v, "carnet_id")));
        lignes.push(format!("manifeste_sha256: {}", self.empreinte));
        if let Some(precedent) = precedent {
            lignes.push(format!("remplace: {precedent}"));
        }
        if let Some(tomberait) = entree.get("tomberait").and_then(Value::as_str) {
            lignes.push(format!("tomberait: {}", chaine(tomberait)?));
        }
        lignes.extend([
            "---".into(),
            format!(
                "# {id} : {}",
                texte(entree, "titre").replace(['\n', '\r'], " ")
            ),
            String::new(),
            texte(entree, "enonce").into(),
            String::new(),
        ]);
        Ok(lignes.join("\n"))
    }

    fn preparer(&self, entree: &Value) -> Resultat<Option<(PathBuf, String)>> {
        let base = texte(entree, "id");
        if !identifiant(base) {
            return Err(Erreur::Validation(format!(
                "identifiant de carnet invalide : {base}"
            )));
        }
        let dir = self.destination.join(dossier(texte(entree, "type"))?);
        let mut n = 1usize;
        let mut precedent = None;
        loop {
            let id = if n == 1 {
                base.to_owned()
            } else {
                format!("{base}-v{n}")
            };
            let fichier = dir.join(format!("{id}.md"));
            match fs::read_to_string(&fichier) {
                Ok(existant) => {
                    let candidat = self.rendre(entree, &id, precedent.as_deref())?;
                    let suivant = dir.join(format!("{base}-v{}.md", n + 1));
                    let a_suivant = suivant.try_exists().map_err(|e| {
                        Erreur::io(format!("recherche de {}", suivant.display()), e)
                    })?;
                    if !a_suivant && sans_empreinte(&existant) == sans_empreinte(&candidat) {
                        return Ok(None);
                    }
                    precedent = Some(id);
                    n += 1;
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    let contenu = self.rendre(entree, &id, precedent.as_deref())?;
                    return Ok(Some((fichier, contenu)));
                }
                Err(e) => {
                    return Err(Erreur::io(format!("lecture de {}", fichier.display()), e));
                }
            }
        }
    }
}

fn sans_empreinte(s: &str) -> String {
    let mut entete = false;
    s.split_inclusive('\n')
        .enumerate()
        .filter_map(|(i, ligne)| {
            if ligne.trim_end_matches('\n') == "---" && (i == 0 || entete) {
                entete = !entete;
                return Some(ligne);
            }
            if entete && ligne.starts_with("manifeste_sha256:") {
                None
            } else {
                Some(ligne)
            }
        })
        .collect()
}

/// Prépare tous les contenus et leurs liens avant la première écriture.
pub fn exporter(experience: &Experience, destination: &Path) -> Resultat<usize> {
    let constats = verifier(&experience.racine)?;
    if constats.iter().any(|c| !c.ok && !c.avertissement) {
        let erreurs: Vec<_> = constats
            .iter()
            .filter(|c| !c.ok && !c.avertissement)
            .map(|c| format!("ROUGE V{} {}", c.controle, c.message))
            .collect();
        return Err(Erreur::Validation(erreurs.join(" ; ")));
    }
    let export = Export {
        experience,
        destination,
        empreinte: sha_octets(&lire_octets(&experience.racine.join(MANIFESTE))?),
    };
    let mut entrees = tableau(&experience.manifeste, "carnet").to_vec();
    entrees.push(entree_experience(&experience.manifeste));
    entrees.sort_by(|a, b| texte(a, "id").cmp(texte(b, "id")));
    let mut ids: BTreeSet<String> = entrees.iter().map(|e| texte(e, "id").to_owned()).collect();
    for nom in DOSSIERS {
        let dir = destination.join(nom);
        match fs::read_dir(&dir) {
            Ok(fichiers) => {
                for fichier in fichiers {
                    let fichier = fichier
                        .map_err(|e| Erreur::io(format!("lecture de {}", dir.display()), e))?;
                    let nom = fichier.file_name();
                    if let Some(id) = nom.to_str().and_then(|s| s.strip_suffix(".md")) {
                        ids.insert(id.into());
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(Erreur::io(format!("lecture de {}", dir.display()), e)),
        }
    }
    let mut fichiers = Vec::new();
    let mut destinations = BTreeSet::new();
    for entree in &entrees {
        for lien in tableau(entree, "liens").iter().filter_map(Value::as_str) {
            if !ids.contains(lien) {
                return Err(Erreur::Validation(format!(
                    "{} : lien de carnet inconnu {lien}",
                    texte(entree, "id")
                )));
            }
        }
        if let Some((p, contenu)) = export.preparer(entree)? {
            if !destinations.insert(p.clone()) {
                return Err(Erreur::Validation(format!(
                    "collision d'export : {}",
                    p.display()
                )));
            }
            fichiers.push((p, contenu));
        }
    }
    for nom in DOSSIERS {
        let dir = destination.join(nom);
        fs::create_dir_all(&dir)
            .map_err(|e| Erreur::io(format!("création de {}", dir.display()), e))?;
    }
    for (p, contenu) in &fichiers {
        if p.try_exists()
            .map_err(|e| Erreur::io(format!("recherche de {}", p.display()), e))?
        {
            return Err(Erreur::Validation(format!(
                "refus d'écraser {}",
                p.display()
            )));
        }
        ecrire_sans_ecraser(p, contenu.as_bytes())?;
    }
    Ok(fichiers.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seule_empreinte_de_l_entete_est_ignoree() {
        let a = "---\nmanifeste_sha256: abc\n---\nmanifeste_sha256: corps\n";
        let b = "---\nmanifeste_sha256: def\n---\nmanifeste_sha256: corps\n";
        assert_eq!(sans_empreinte(a), sans_empreinte(b));
        assert_ne!(
            sans_empreinte(a),
            sans_empreinte(&b.replace("corps", "autre"))
        );
    }
}
