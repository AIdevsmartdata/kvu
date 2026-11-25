// SPDX-License-Identifier: Apache-2.0
//! Construction du manifeste et opérations d'ajout sans écrasement des pièces.

use super::*;
use std::collections::BTreeSet;
use std::time::UNIX_EPOCH;

pub(crate) struct OptionsPrereg<'a> {
    pub fichier: &'a str,
    pub hachage_consigne: Option<&'a str>,
    pub gele_le: Option<&'a str>,
    pub change_un_critere: bool,
    pub remplace: Option<&'a str>,
    pub sections: Vec<String>,
}

pub(crate) struct OptionsInit<'a> {
    pub id: &'a str,
    pub titre: &'a str,
    pub question: &'a str,
    pub carnet_id: Option<&'a str>,
    pub chemin_projet: &'a str,
    pub machine_principale: &'a str,
    pub commit: &'a str,
    pub branche: &'a str,
    pub garde: Option<&'a str>,
    pub sha256sums: Option<&'a str>,
    pub prereg: OptionsPrereg<'a>,
}

/// État d'une expérience ; les pièces citées restent dans leur dossier d'origine.
pub struct Experience {
    pub(super) racine: PathBuf,
    pub(super) manifeste: Value,
}

impl Experience {
    pub fn charger(racine: &Path) -> Resultat<Self> {
        let racine = canonique(racine)?;
        let mut manifeste = lire_json(&racine.join(MANIFESTE))?;
        valider(&manifeste)?;
        normaliser_dates(&mut manifeste)?;
        chainer_si_absent(&mut manifeste)?;
        marquer_horloge(&mut manifeste);
        Ok(Self { racine, manifeste })
    }

    pub(crate) fn initialiser(racine: &Path, o: OptionsInit<'_>) -> Resultat<()> {
        fs::create_dir_all(racine)
            .map_err(|e| Erreur::io(format!("création de {}", racine.display()), e))?;
        let racine_canonique = canonique(racine)?;
        let racine = racine_canonique.as_path();
        if racine
            .join(MANIFESTE)
            .try_exists()
            .map_err(|e| Erreur::io("recherche du manifeste existant", e))?
        {
            return Err(Erreur::Validation("EXPERIENCE.json existe déjà".into()));
        }
        verifier_relatif(o.chemin_projet, true)?;
        let p0 = construire_prereg(racine, "P0", o.prereg)?;
        let mut anteriorite = json!({});
        for (cle, fichier) in [("garde", o.garde), ("sha256sums", o.sha256sums)] {
            if let Some(fichier) = fichier {
                anteriorite[cle] = piece(racine, fichier)?;
            }
        }
        let date = horloge()?;
        let carnet_id = o
            .carnet_id
            .map(str::to_owned)
            .unwrap_or_else(|| format!("EXP-{}", o.id));
        let manifeste = json!({
            "schema": SCHEMA,
            "id": o.id,
            "titre": o.titre,
            "question": o.question,
            "carnet_id": carnet_id,
            "chemin_projet": o.chemin_projet,
            "machine_principale": o.machine_principale,
            "prereg": [p0],
            "anteriorite": anteriorite,
            "code": {"commit": o.commit, "branche": o.branche, "binaires": []},
            "machines": [],
            "runs": [],
            "graines": {"toutes_distinctes": true},
            "donnees": {},
            "portes": [],
            "verdict": {
                "statut": "en_cours", "provisoire": true, "classe": "A",
                "texte": "", "chiffres": [], "historique": [], "chaine": EMPREINTE_NULLE
            },
            "reserves": [],
            "incidents": [],
            "carnet": [],
            "index": {},
            "auteurs": {"relecteurs": []},
            "horodatage": {"cree_le": date, "modifie_le": date}
        });
        let mut manifeste = manifeste;
        marquer_horloge(&mut manifeste);
        valider(&manifeste)?;
        ecrire_atomique(&racine.join(MANIFESTE), &json_octets(&manifeste)?)
    }

    fn sauver(&mut self) -> Resultat<()> {
        self.manifeste["horodatage"]["modifie_le"] = json!(horloge()?);
        marquer_horloge(&mut self.manifeste);
        valider(&self.manifeste)?;
        ecrire_atomique(&self.racine.join(MANIFESTE), &json_octets(&self.manifeste)?)
    }

    pub(crate) fn ajouter_prereg(&mut self, o: OptionsPrereg<'_>) -> Resultat<String> {
        let mut n = 0usize;
        while tableau(&self.manifeste, "prereg")
            .iter()
            .any(|p| texte(p, "id") == format!("P{n}"))
        {
            n += 1;
        }
        let id = format!("P{n}");
        let prereg = construire_prereg(&self.racine, &id, o)?;
        if let Some(remplace) = prereg["remplace"].as_str() {
            let ancien = tableau(&self.manifeste, "prereg")
                .iter()
                .find(|p| texte(p, "id") == remplace)
                .ok_or_else(|| {
                    Erreur::Validation(format!("préenregistrement remplacé inconnu : {remplace}"))
                })?;
            if prereg["change_un_critere"] == true && prereg["chemin"] == ancien["chemin"] {
                return Err(Erreur::Validation(
                    "un critère changé exige un nouveau fichier de PREREG".into(),
                ));
            }
        }
        liste_mut(&mut self.manifeste, "prereg")?.push(prereg);
        self.sauver()?;
        Ok(id)
    }

    pub(crate) fn ajouter_runs(
        &mut self,
        machine: &str,
        preregs: &[String],
        entrees: &[String],
    ) -> Resultat<(usize, usize)> {
        if machine.trim().is_empty() {
            return Err(Erreur::Config("--machine : nom vide".into()));
        }
        let mut preregs = preregs.to_vec();
        if preregs.is_empty() {
            preregs.push("P0".into());
        }
        preregs.sort();
        preregs.dedup();
        for id in &preregs {
            if !tableau(&self.manifeste, "prereg")
                .iter()
                .any(|p| texte(p, "id") == id)
            {
                return Err(Erreur::Validation(format!(
                    "préenregistrement inconnu : {id}"
                )));
            }
        }
        let fichiers = trouver_manifestes(&self.racine, entrees)?;
        let (mut ajoutes, mut ignores) = (0, 0);
        for chemin in fichiers {
            let original = lire_json(&resoudre(&self.racine, &chemin)?)?;
            if texte(&original, "schema") != "kevinotron-unifie/1" {
                ignores += 1;
                continue;
            }
            let (run, binaire, mut description) =
                construire_run(&self.racine, &chemin, &original, machine, &preregs)?;
            let id = texte(&run, "id");
            if let Some(ancien) = tableau(&self.manifeste, "runs")
                .iter()
                .find(|r| texte(r, "id") == id)
            {
                if ancien != &run {
                    return Err(Erreur::Validation(format!(
                        "run {id} déjà présent avec un contenu différent"
                    )));
                }
                ignores += 1;
                continue;
            }
            description["classe"] = json!(classe_machine(
                machine,
                texte(&self.manifeste, "machine_principale")
            ));
            if let Some(ancienne) = tableau(&self.manifeste, "machines")
                .iter()
                .find(|m| texte(m, "nom") == machine)
            {
                if ancienne != &description {
                    return Err(Erreur::Validation(format!(
                        "machine {machine} : hôte, backend ou architecture différents"
                    )));
                }
            } else {
                liste_mut(&mut self.manifeste, "machines")?.push(description);
            }
            if !tableau(&self.manifeste["code"], "binaires").contains(&binaire) {
                liste_mut(&mut self.manifeste["code"], "binaires")?.push(binaire);
            }
            liste_mut(&mut self.manifeste, "runs")?.push(run);
            ajoutes += 1;
        }
        if ajoutes > 0 {
            liste_mut(&mut self.manifeste, "runs")?
                .sort_by(|a, b| texte(a, "id").cmp(texte(b, "id")));
            liste_mut(&mut self.manifeste, "machines")?
                .sort_by(|a, b| texte(a, "nom").cmp(texte(b, "nom")));
            liste_mut(&mut self.manifeste["code"], "binaires")?.sort_by_key(Value::to_string);
            self.manifeste["graines"]["toutes_distinctes"] =
                json!(graines_distinctes(&self.manifeste));
            self.manifeste["verdict"]["classe"] = json!(classe_experience(&self.manifeste));
            self.sauver()?;
        }
        Ok((ajoutes, ignores))
    }

    pub(crate) fn porte(&mut self, mut porte: Value) -> Resultat<()> {
        refuser_id(&self.manifeste, "portes", texte(&porte, "id"))?;
        let source = texte(&porte, "source").to_owned();
        porte["source"] = piece(&self.racine, &source)?;
        if porte["statut"] == "rouge_attendu" && texte(&porte, "mutant").trim().is_empty() {
            return Err(Erreur::Validation(
                "rouge_attendu exige un mutant non vide".into(),
            ));
        }
        liste_mut(&mut self.manifeste, "portes")?.push(porte);
        self.sauver()
    }

    pub(crate) fn verdict(&mut self, mut verdict: Value, reserves: &[String]) -> Resultat<()> {
        for chiffre in tableau(&verdict, "chiffres") {
            let source = texte(chiffre, "source");
            if source.contains(':') {
                verifier_source(&self.racine, source)?;
            } else {
                let p = resoudre(&self.racine, source)?;
                fs::metadata(&p)
                    .map_err(|e| Erreur::io(format!("lecture de {}", p.display()), e))?;
            }
        }
        let ancien = &self.manifeste["verdict"];
        let mut historique = tableau(ancien, "historique").to_vec();
        let mut etat = ancien.clone();
        let objet = etat
            .as_object_mut()
            .ok_or_else(|| Erreur::Validation("verdict : objet attendu".into()))?;
        objet.remove("historique");
        // La provenance de migration reste portée par le verdict COURANT ; elle n'entre pas dans les entrées archivées.
        let migre = objet.remove("migre_le");
        // Chaînage : l'entrée archivée porte l'empreinte de la précédente (celle que le verdict courant
        // annonçait), et le nouveau verdict annonce l'empreinte de l'entrée qu'on vient d'archiver.
        let precedent = objet
            .remove("chaine")
            .unwrap_or_else(|| json!(EMPREINTE_NULLE));
        objet.insert("precedent".into(), precedent);
        objet.insert("remplace_le".into(), json!(horloge()?));
        verdict["chaine"] = json!(empreinte_json(&etat));
        if let Some(migre) = migre {
            verdict["migre_le"] = migre;
        }
        historique.push(etat);
        verdict["historique"] = json!(historique);
        verdict["classe"] = json!(classe_experience(&self.manifeste));
        self.manifeste["verdict"] = verdict;
        if !reserves.is_empty() {
            if self.manifeste.get("reserves").is_none() {
                self.manifeste["reserves"] = json!([]);
            }
            liste_mut(&mut self.manifeste, "reserves")?.extend(reserves.iter().map(|s| json!(s)));
        }
        self.sauver()
    }

    pub(crate) fn proposer(&mut self, mut proposition: Value) -> Resultat<()> {
        refuser_id(&self.manifeste, "carnet", texte(&proposition, "id"))?;
        for source in tableau(&proposition, "sources") {
            verifier_source(&self.racine, source.as_str().unwrap_or(""))?;
        }
        proposition["date"] = json!(horloge()?);
        liste_mut(&mut self.manifeste, "carnet")?.push(proposition);
        self.sauver()
    }

    pub(crate) fn fiche(&self) -> Resultat<()> {
        let contenu = super::fiche::rendre(&self.manifeste);
        ecrire_atomique(&self.racine.join("EXPERIENCE.md"), contenu.as_bytes())
    }
}

fn valider(manifeste: &Value) -> Resultat<()> {
    let erreurs = schema::valider_experience(manifeste);
    if erreurs.is_empty() {
        Ok(())
    } else {
        Err(Erreur::Validation(format!(
            "ROUGE V1 {}",
            erreurs.join(" ; ")
        )))
    }
}

/// Migration d'un manifeste antérieur au chaînage : les entrées d'historique reçoivent leur `precedent` et le
/// verdict sa `chaine`. Le chaînage n'atteste l'historique qu'à partir de cet instant.
fn chainer_si_absent(v: &mut Value) -> Resultat<()> {
    // Migration réservée aux historiques ENTIÈREMENT dépourvus de chaînage ; un état partiel reste tel quel
    // (V11 le rend rouge) et aucun lien existant n'est jamais remplacé.
    let a_des_liens = tableau(&v["verdict"], "historique")
        .iter()
        .any(|e| e.get("precedent").is_some());
    if v["verdict"].get("chaine").is_some() || a_des_liens {
        return Ok(());
    }
    let mut precedent = EMPREINTE_NULLE.to_owned();
    for e in liste_mut(&mut v["verdict"], "historique")? {
        e["precedent"] = json!(precedent);
        precedent = empreinte_json(e);
    }
    v["verdict"]["chaine"] = json!(precedent);
    v["verdict"]["migre_le"] = json!(horloge()?);
    Ok(())
}

fn normaliser_champ(v: &mut Value, cle: &str) -> Resultat<()> {
    let date = normaliser_date(texte(v, cle))?;
    v[cle] = json!(date);
    Ok(())
}

fn normaliser_dates(v: &mut Value) -> Resultat<()> {
    for p in liste_mut(v, "prereg")? {
        normaliser_champ(p, "gele_le")?;
    }
    for run in liste_mut(v, "runs")? {
        normaliser_champ(run, "debut")?;
        normaliser_champ(run, "fin")?;
    }
    for cle in ["cree_le", "modifie_le"] {
        normaliser_champ(&mut v["horodatage"], cle)?;
    }
    for proposition in liste_mut(v, "carnet")? {
        normaliser_champ(proposition, "date")?;
    }
    Ok(())
}

fn refuser_id(manifeste: &Value, tableau_id: &str, id: &str) -> Resultat<()> {
    if !identifiant(id) {
        return Err(Erreur::Validation(format!("identifiant invalide : {id}")));
    }
    if tableau(manifeste, tableau_id)
        .iter()
        .any(|p| texte(p, "id") == id)
    {
        return Err(Erreur::Validation(format!(
            "{tableau_id} : id {id} déjà présent"
        )));
    }
    Ok(())
}

fn construire_prereg(racine: &Path, id: &str, o: OptionsPrereg<'_>) -> Resultat<Value> {
    if o.change_un_critere && o.remplace.is_none() {
        return Err(Erreur::Validation(
            "--change-un-critere exige --remplace Pn".into(),
        ));
    }
    let mut prereg = piece(racine, o.fichier)?;
    prereg["id"] = json!(id);
    prereg["change_un_critere"] = json!(o.change_un_critere);
    prereg["remplace"] = json!(o.remplace);
    if !o.sections.is_empty() {
        prereg["sections"] = json!(o.sections);
    }
    let fichier_date = if let Some(fichier) = o.hachage_consigne {
        let hachage = piece(racine, fichier)?;
        let chemin = texte(&hachage, "chemin").to_owned();
        let contenu = lire_texte(&racine.join(&chemin))?;
        if contenu.split_whitespace().next() != prereg["sha256"].as_str() {
            return Err(Erreur::Validation(format!(
                "{chemin} : premier mot différent du sha256 du PREREG"
            )));
        }
        prereg["hachage_consigne"] = json!(chemin);
        prereg["hachage_consigne_sha256"] = hachage["sha256"].clone();
        chemin
    } else {
        texte(&prereg, "chemin").to_owned()
    };
    let (date, source) = if let Some(iso) = o.gele_le {
        (normaliser_date(iso)?, "declare")
    } else {
        let p = racine.join(fichier_date);
        let mtime = fs::metadata(&p)
            .and_then(|m| m.modified())
            .map_err(|e| Erreur::io(format!("date de modification de {}", p.display()), e))?;
        let secondes = mtime
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Erreur::Validation("mtime antérieur à l'époque Unix".into()))?
            .as_secs();
        if secondes > DERNIERE_SECONDE {
            return Err(Erreur::Validation(
                "mtime hors de la plage des dates ISO".into(),
            ));
        }
        (date_iso(secondes), "mtime")
    };
    prereg["gele_le"] = json!(date);
    prereg["gele_source"] = json!(source);
    Ok(prereg)
}

fn trouver_manifestes(racine: &Path, entrees: &[String]) -> Resultat<Vec<String>> {
    let mut fichiers = BTreeSet::new();
    for entree in entrees {
        let p = Path::new(entree);
        let p = if p.is_absolute() {
            p.to_path_buf()
        } else {
            racine.join(p)
        };
        let meta =
            fs::metadata(&p).map_err(|e| Erreur::io(format!("lecture de {}", p.display()), e))?;
        if meta.is_dir() {
            let dir = fs::read_dir(&p)
                .map_err(|e| Erreur::io(format!("lecture de {}", p.display()), e))?;
            for enfant in dir {
                let enfant =
                    enfant.map_err(|e| Erreur::io(format!("lecture de {}", p.display()), e))?;
                let type_fichier = enfant
                    .file_type()
                    .map_err(|e| Erreur::io(format!("type de {}", enfant.path().display()), e))?;
                if type_fichier.is_file()
                    && enfant
                        .file_name()
                        .to_str()
                        .is_some_and(|s| s.ends_with(".manifest.json"))
                {
                    fichiers.insert(relatif(racine, &enfant.path())?);
                }
            }
        } else if meta.is_file() {
            fichiers.insert(relatif(racine, &p)?);
        } else {
            return Err(Erreur::Validation(format!(
                "{} : fichier ou dossier attendu",
                p.display()
            )));
        }
    }
    Ok(fichiers.into_iter().collect())
}

pub(super) fn construire_run(
    racine: &Path,
    chemin: &str,
    original: &Value,
    machine: &str,
    preregs: &[String],
) -> Resultat<(Value, Value, Value)> {
    let p = resoudre(racine, chemin)?;
    let nom = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let id = nom
        .strip_suffix(".manifest.json")
        .filter(|id| identifiant(id))
        .ok_or_else(|| Erreur::Validation(format!("{chemin} : nom <id>.manifest.json attendu")))?;
    let argv = original
        .get("argv")
        .and_then(Value::as_array)
        .ok_or_else(|| Erreur::Validation(format!("{chemin} : argv absent ou non tabulaire")))?;
    let argv: Vec<String> = argv
        .iter()
        .map(|v| {
            v.as_str().map(str::to_owned).ok_or_else(|| {
                Erreur::Validation(format!("{chemin} : argv doit contenir des chaînes"))
            })
        })
        .collect::<Resultat<_>>()?;
    let arguments = analyser_argv(&argv)?;
    if original["rc"].as_i64().is_none() {
        return Err(Erreur::Validation(format!("{chemin} : rc entier attendu")));
    }
    let mut run = json!({
        "id": id,
        "manifeste": chemin,
        "sha256": sha_fichier(&p)?,
        "schema": original["schema"],
        "machine": machine,
        "hote": original["hote"],
        "debut": date_unix(&original["debut_unix"], "debut_unix")?,
        "fin": date_unix(&original["fin_unix"], "fin_unix")?,
        "rc": original["rc"],
        "graines": arguments.graines,
        "binaire_sha256": original["binaire_sha256"],
        "geometrie": arguments.geometrie,
        "parametres": arguments.parametres,
        "prereg": preregs
    });
    let sortie = p.with_file_name(format!("{id}.json"));
    if sortie
        .try_exists()
        .map_err(|e| Erreur::io(format!("recherche de {}", sortie.display()), e))?
    {
        let rel = relatif(racine, &sortie)?;
        run["sortie"] = piece(racine, &rel)?;
    }
    let binaire = json!({
        "sha256": original["binaire_sha256"],
        "source_hash": original["source_hash"],
        "version": original["version"],
        "arch": original["arch"],
        "backend": original["backend"],
        "versions": original["versions"]
    });
    let description = json!({
        "nom": machine, "hote": original["hote"],
        "backend": original["backend"], "arch": original["arch"]
    });
    Ok((run, binaire, description))
}

pub(super) fn classe_machine(nom: &str, principale: &str) -> &'static str {
    if nom == principale {
        "A"
    } else {
        "B"
    }
}

pub(super) fn classe_experience(manifeste: &Value) -> &'static str {
    let principale = texte(manifeste, "machine_principale");
    if tableau(manifeste, "machines")
        .iter()
        .any(|m| texte(m, "nom") != principale)
    {
        "B"
    } else {
        "A"
    }
}

pub(super) fn graines_distinctes(manifeste: &Value) -> bool {
    let mut graines = BTreeSet::new();
    tableau(manifeste, "runs").iter().all(|r| {
        !tableau(r, "graines").is_empty()
            && tableau(r, "graines")
                .iter()
                .all(|g| g.as_u64().is_some_and(|g| graines.insert(g)))
    })
}
