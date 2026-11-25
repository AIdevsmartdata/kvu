// SPDX-License-Identifier: Apache-2.0
//! Interface `kvu experience` : acquisition des pièces et contrôles documentaires.

use super::modele::{OptionsInit, OptionsPrereg};
use super::*;
use crate::commun::{code, NOM_PROFIL_ACEMAGIC};
use clap::{Arg, ArgAction, ArgMatches, Command};

fn option(nom: &'static str) -> Arg {
    Arg::new(nom).long(nom).value_name("VALEUR")
}

fn requis(nom: &'static str) -> Arg {
    option(nom).required(true)
}

fn multiple(nom: &'static str) -> Arg {
    option(nom).action(ArgAction::Append)
}

fn gel(commande: Command) -> Command {
    commande
        .arg(option("hachage-consigne").value_name("FICHIER"))
        .arg(
            option("gele-le")
                .value_name("ISO")
                .help("Instant de gel déclaré ; sinon mtime du hachage consigné ou du PREREG"),
        )
}

pub fn commandes() -> Vec<Command> {
    let init = gel(Command::new("init")
        .about("Crée EXPERIENCE.json et gèle le PREREG P0")
        .arg(requis("id"))
        .arg(requis("titre"))
        .arg(requis("question"))
        .arg(requis("prereg").value_name("FICHIER"))
        .arg(option("machine-principale").default_value(NOM_PROFIL_ACEMAGIC))
        .arg(option("chemin-projet").default_value("."))
        .arg(option("commit").default_value("non_declare"))
        .arg(option("branche").default_value("non_declare"))
        .arg(option("garde").value_name("FICHIER"))
        .arg(option("sha256sums").value_name("FICHIER"))
        .arg(option("carnet-id")));
    let prereg = gel(Command::new("ajouter-prereg")
        .about("Ajoute un préenregistrement dans un nouveau fichier")
        .arg(Arg::new("fichier").required(true).value_name("FICHIER"))
        .arg(
            Arg::new("change-un-critere")
                .long("change-un-critere")
                .action(ArgAction::SetTrue),
        )
        .arg(option("remplace").value_name("Pn"))
        .arg(multiple("sections").num_args(1..).value_delimiter(',')));
    let runs = Command::new("ajouter-run")
        .about("Lit des manifestes de run, sans lancer de calcul")
        .arg(requis("machine"))
        .arg(
            multiple("prereg")
                .num_args(1..)
                .value_delimiter(',')
                .help("Ids Pn, séparés par des espaces ou des virgules (défaut P0)"),
        )
        .arg(
            Arg::new("fichiers")
                .num_args(1..)
                .value_name("FICHIER_OU_DOSSIER"),
        );
    let porte = Command::new("porte")
        .about("Consigne une porte dont l'id doit être neuf")
        .arg(Arg::new("id").required(true))
        .arg(requis("enonce"))
        .arg(requis("seuil").allow_hyphen_values(true))
        .arg(requis("valeur").allow_hyphen_values(true))
        .arg(requis("statut").value_parser(["vert", "rouge", "rouge_attendu", "non_lance"]))
        .arg(requis("source").value_name("FICHIER"))
        .arg(option("mutant"));
    let verdict = Command::new("verdict")
        .about("Archive l'état courant puis consigne le nouveau verdict")
        .arg(requis("statut").value_parser([
            "en_cours",
            "non_concluant",
            "compatible",
            "refute",
            "indetermine",
        ]))
        .arg(
            Arg::new("provisoire")
                .long("provisoire")
                .action(ArgAction::SetTrue),
        )
        .arg(requis("texte"))
        .arg(multiple("chiffre").value_name("NOM=VALEUR@SOURCE:STATUT"))
        .arg(multiple("reserve"));
    let proposer = Command::new("proposer")
        .about("Consigne une proposition du carnet sans promotion automatique")
        .arg(Arg::new("type").required(true).value_parser([
            "resultat",
            "erreur",
            "decision",
            "hypothese",
            "piste",
            "fait",
        ]))
        .arg(Arg::new("id").required(true))
        .arg(requis("titre"))
        .arg(requis("statut"))
        .arg(requis("enonce"))
        .arg(multiple("source").required(true).num_args(1..))
        .arg(multiple("lien").num_args(1..).value_delimiter(','))
        .arg(option("tomberait"));
    let sous_commandes = vec![
        init,
        prereg,
        runs,
        porte,
        verdict,
        proposer,
        Command::new("verifier")
            .about("Contrôle V1 à V11 ; code 8 si un constat est ROUGE")
            .arg(Arg::new("json").long("json").action(ArgAction::SetTrue)),
        Command::new("fiche").about("Écrit la fiche lisible EXPERIENCE.md"),
        Command::new("carnet")
            .about("Exporte le carnet avec versionnage, sans écrasement")
            .arg(requis("vers").value_name("DIR")),
    ];
    vec![Command::new("experience")
        .about("Préenregistrements, runs, portes et carnet d'une expérience")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .arg(
            option("racine")
                .value_name("DIR")
                .default_value(".")
                .global(true),
        )
        .subcommands(sous_commandes)]
}

fn valeur<'a>(m: &'a ArgMatches, cle: &str) -> Resultat<&'a str> {
    m.get_one::<String>(cle)
        .map(String::as_str)
        .ok_or_else(|| Erreur::Config(format!("argument requis : {cle}")))
}

fn facultatif<'a>(m: &'a ArgMatches, cle: &str) -> Option<&'a str> {
    m.get_one::<String>(cle).map(String::as_str)
}

fn valeurs(m: &ArgMatches, cle: &str) -> Vec<String> {
    m.get_many::<String>(cle)
        .map(|v| v.cloned().collect())
        .unwrap_or_default()
}

fn options_prereg(m: &ArgMatches, initial: bool) -> Resultat<OptionsPrereg<'_>> {
    Ok(OptionsPrereg {
        fichier: valeur(m, if initial { "prereg" } else { "fichier" })?,
        hachage_consigne: facultatif(m, "hachage-consigne"),
        gele_le: facultatif(m, "gele-le"),
        change_un_critere: !initial && m.get_flag("change-un-critere"),
        remplace: if initial {
            None
        } else {
            facultatif(m, "remplace")
        },
        sections: if initial {
            Vec::new()
        } else {
            valeurs(m, "sections")
        },
    })
}

fn scalaire(s: &str) -> Value {
    serde_json::from_str::<Value>(s)
        .ok()
        .filter(|v| !v.is_array() && !v.is_object())
        .unwrap_or_else(|| json!(s))
}

fn chiffre(s: &str) -> Resultat<Value> {
    let invalide = || Erreur::Config(format!("chiffre attendu NOM=VALEUR@SOURCE:STATUT : {s}"));
    let (nom, reste) = s.split_once('=').ok_or_else(invalide)?;
    let (nombre, reste) = reste.split_once('@').ok_or_else(invalide)?;
    let (source, statut) = reste.rsplit_once(':').ok_or_else(invalide)?;
    let valeur = serde_json::from_str::<Value>(nombre).map_err(|_| invalide())?;
    if nom.trim().is_empty()
        || !valeur.is_number()
        || !["[calcul exact]", "[numérique]", "[lecture]", "[estimé]"].contains(&statut)
    {
        return Err(invalide());
    }
    if source.contains(':') {
        analyser_source(source)?;
    } else {
        verifier_relatif(source, false)?;
    }
    Ok(json!({
        "nom": nom, "valeur": valeur, "source": source, "statut_lecture": statut
    }))
}

fn ajouter_runs(racine: &Path, m: &ArgMatches) -> Resultat<i32> {
    // Clap peut absorber les fichiers qui suivent la liste variable des Pn.
    // Leur suffixe non numérique les distingue des identifiants de PREREG.
    let mut preregs = Vec::new();
    let mut fichiers = valeurs(m, "fichiers");
    for valeur in valeurs(m, "prereg") {
        if valeur
            .strip_prefix('P')
            .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
        {
            preregs.push(valeur);
        } else {
            fichiers.push(valeur);
        }
    }
    if fichiers.is_empty() {
        return Err(Erreur::Config(
            "ajouter-run : au moins un fichier ou dossier est requis".into(),
        ));
    }
    let mut experience = Experience::charger(racine)?;
    let (ajoutes, ignores) = experience.ajouter_runs(valeur(m, "machine")?, &preregs, &fichiers)?;
    println!("Runs : {ajoutes} ajouté(s), {ignores} ignoré(s)");
    Ok(code::OK)
}

pub fn executer(nom: &str, m: &ArgMatches) -> Resultat<i32> {
    if nom != "experience" {
        return Err(Erreur::Config(format!(
            "module experience : commande inconnue {nom}"
        )));
    }
    let (action, m) = m
        .subcommand()
        .ok_or_else(|| Erreur::Config("experience : sous-commande requise".into()))?;
    let racine = Path::new(valeur(m, "racine")?);
    // Toute commande qui écrit prend le verrou avant de lire l'état (aucune mise à jour perdue).
    let _verrou = if action == "verifier" {
        None
    } else {
        std::fs::create_dir_all(racine)
            .map_err(|e| Erreur::io(format!("création de {}", racine.display()), e))?;
        Some(verrouiller(racine)?)
    };
    if action == "init" {
        Experience::initialiser(
            racine,
            OptionsInit {
                id: valeur(m, "id")?,
                titre: valeur(m, "titre")?,
                question: valeur(m, "question")?,
                carnet_id: facultatif(m, "carnet-id"),
                chemin_projet: valeur(m, "chemin-projet")?,
                machine_principale: valeur(m, "machine-principale")?,
                commit: valeur(m, "commit")?,
                branche: valeur(m, "branche")?,
                garde: facultatif(m, "garde"),
                sha256sums: facultatif(m, "sha256sums"),
                prereg: options_prereg(m, true)?,
            },
        )?;
        println!(
            "Expérience initialisée : {}",
            racine.join(MANIFESTE).display()
        );
        return Ok(code::OK);
    }
    if action == "verifier" {
        let constats = verifier(racine)?;
        let rouge = constats.iter().any(|c| !c.ok && !c.avertissement);
        if m.get_flag("json") {
            let sortie = serde_json::to_string_pretty(&constats)
                .map_err(|e| Erreur::Format(e.to_string()))?;
            println!("{sortie}");
        } else {
            for c in constats {
                println!(
                    "{} V{} {}",
                    if c.ok {
                        "OK"
                    } else if c.avertissement {
                        "AVERTISSEMENT"
                    } else {
                        "ROUGE"
                    },
                    c.controle,
                    c.message
                );
            }
        }
        return Ok(if rouge { code::VALIDATION } else { code::OK });
    }
    if action == "ajouter-run" {
        return ajouter_runs(racine, m);
    }
    let mut experience = Experience::charger(racine)?;
    match action {
        "ajouter-prereg" => {
            let id = experience.ajouter_prereg(options_prereg(m, false)?)?;
            println!("Préenregistrement ajouté : {id}");
        }
        "porte" => experience.porte(json!({
            "id": valeur(m, "id")?, "enonce": valeur(m, "enonce")?,
            "seuil": scalaire(valeur(m, "seuil")?), "valeur": scalaire(valeur(m, "valeur")?),
            "statut": valeur(m, "statut")?, "source": valeur(m, "source")?,
            "mutant": facultatif(m, "mutant")
        }))?,
        "verdict" => {
            let chiffres = valeurs(m, "chiffre")
                .iter()
                .map(|s| chiffre(s))
                .collect::<Resultat<Vec<_>>>()?;
            experience.verdict(
                json!({
                    "statut": valeur(m, "statut")?, "provisoire": m.get_flag("provisoire"),
                    "texte": valeur(m, "texte")?, "chiffres": chiffres
                }),
                &valeurs(m, "reserve"),
            )?;
        }
        "proposer" => {
            let mut proposition = json!({
                "type": valeur(m, "type")?, "id": valeur(m, "id")?,
                "titre": valeur(m, "titre")?, "statut": valeur(m, "statut")?,
                "enonce": valeur(m, "enonce")?, "sources": valeurs(m, "source"),
                "liens": valeurs(m, "lien")
            });
            if let Some(tomberait) = facultatif(m, "tomberait") {
                proposition["tomberait"] = json!(tomberait);
            }
            experience.proposer(proposition)?;
        }
        "fiche" => experience.fiche()?,
        "carnet" => {
            let n = carnet::exporter(&experience, Path::new(valeur(m, "vers")?))?;
            for (controle, message) in avertissements(&experience.manifeste) {
                eprintln!("AVERTISSEMENT V{controle} {message}");
            }
            println!("Carnet : {n} nouveau(x) fichier(s)");
        }
        autre => {
            return Err(Erreur::Config(format!(
                "experience : sous-commande inconnue {autre}"
            )));
        }
    }
    Ok(code::OK)
}
