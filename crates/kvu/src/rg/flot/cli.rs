//! Sous-commandes `rg flot`, `rg etalonner`, `rg flot-gates` (WP8). Enregistrées par `src/rg/cli.rs` (deux lignes).
//!
//! ```text
//! kvu rg flot --ensemble DIR [--filtre MOTIF] [--schema balaban|swendsen] [--pas k] [--estimateurs sd,plaq,mcrg2]
//!             [--courbe etal_L4.json]… [--sortie DIR] [--fils N] [--operateur 3centre|dec|ape:<α>:<n>|flot:<t>[:<ε>]|agr:<c>]
//!             [--hors-famille]   (WP22 : opérateur de chaque pas de bloc, schéma balaban seulement)
//! kvu rg etalonner --L 4|6 [--independante] [--facteur-beta 1.05] [--sortie FICHIER.json] [--fils N]
//! kvu rg flot-gates --donnees D [--grandl D] [--etalonnages D] [--sans-mutants] [--seul ID,…] [--json] [--certifier [FICHIER]] [--fils N]
//! ```
//! `flot` écrit `<sortie>/flot.json` (refuse d'écraser) ; son contenu ne dépend ni de la durée ni du nombre de fils.

use super::cellule::{flot_options, lire_courbes, Estimateurs, ParamFlot, Schema};
use super::etalonner::{en_json, etalonner, ParamEtalonnage};
use super::extensions::{analyser_bins, OptionsFlot};
use super::gates::{batterie, Contexte};
use super::sd::RapportBessel;
use crate::commun::{code, Erreur, Resultat};
use crate::groupes::gates::{certifiable, tout_vert, LigneGate, Verdict};
use crate::heritage::manifest::sha256_fichier;
use clap::{Arg, ArgAction, ArgMatches, Command};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const SCHEMA_CERT_FLOT: &str = "kvu-rg-flot-cert/1";
pub const CERT_DEFAUT: &str = "validation/rg_flot/rg_flot.json";

pub fn sous_commandes() -> Vec<Command> {
    let fils = || Arg::new("fils").long("fils").num_args(1).help("nombre de fils (défaut : RAYON_NUM_THREADS ou 4) ; les sorties n'en dépendent pas");
    vec![
        Command::new("flot")
            .about("Estimateurs de couplage effectif (sd, plaq, mcrg2) d'un ensemble après k pas de bloc : flot.json")
            .arg(Arg::new("ensemble").long("ensemble").required(true).num_args(1).help("dossier d'une chaîne (cfg_*.npy + manifest.json) ou parent de chaînes"))
            .arg(Arg::new("backend").long("backend").num_args(1).default_value("cpu").help("cpu | hip | cuda | auto (GPU : schéma balaban seulement)"))
            .arg(Arg::new("bins").long("bins").num_args(1).help("tailles ≥ 1 des blocs consécutifs, séparées par des virgules, sans doublon (ex. 4,8)"))
            .arg(Arg::new("detail-configs").long("detail-configs").action(ArgAction::SetTrue).help("statistiques de chaque configuration, dans l'ordre des chaînes et fichiers"))
            .arg(Arg::new("filtre").long("filtre").num_args(1).help("motif (jokers *) des dossiers de chaînes"))
            .arg(Arg::new("schema").long("schema").num_args(1).default_value("balaban").help("balaban (L_b = 3) | swendsen (b = 2, c = 0,5) | ape, dec (b = 2, script n3)"))
            .arg(Arg::new("pas").long("pas").num_args(1).default_value("1").help("nombre de pas de bloc"))
            .arg(Arg::new("estimateurs").long("estimateurs").num_args(1).default_value("sd,plaq,mcrg2"))
            .arg(Arg::new("courbe").long("courbe").num_args(1).action(ArgAction::Append).help("courbe d'étalonnage (JSON de `rg etalonner`), répétable"))
            .arg(Arg::new("sortie").long("sortie").num_args(1).default_value(".").help("dossier de sortie de flot.json"))
            .arg(Arg::new("mutant-u1").long("mutant-u1").action(ArgAction::SetTrue).help("MUTANT : rapport I1/I0 au lieu de I2/I1 (contrôle négatif)"))
            .arg(Arg::new("operateur").long("operateur").num_args(1).default_value("3centre").help("opérateur de chaque pas de bloc (schéma balaban) : 3centre (défaut, inchangé) | dec | ape:<α>:<n> | flot:<t>[:<ε>] | agr:<c>"))
            .arg(Arg::new("hors-famille").long("hors-famille").action(ArgAction::SetTrue).help("accepte un opérateur dont le rayon de lissage dépasse le bloc (mutant)"))
            .arg(fils()),
        Command::new("etalonner")
            .about("Courbe d'étalonnage ⟨P⟩, ⟨R⟩(β ; L) de Wilson (protocole gelé de WP8) : chaînes CPU, mesures en ligne")
            .arg(Arg::new("L").long("L").required(true).num_args(1).help("côté du volume grossier : 4 ou 6 (wp8) ; 4 ou 8 (wp9)"))
            .arg(Arg::new("protocole").long("protocole").num_args(1).default_value("wp8").help("wp8 (gelé) | wp9 (31 β, L = 4 ou 8)"))
            .arg(Arg::new("independante").long("independante").action(ArgAction::SetTrue).help("une chaîne indépendante par point (c = 9), pour RG-PLAQ"))
            .arg(Arg::new("milieux").long("milieux").action(ArgAction::SetTrue).conflicts_with_all(["independante", "facteur-beta"]).help("wp9 : une chaîne c = 8 à chacun des 30 milieux de la grille"))
            .arg(Arg::new("facteur-beta").long("facteur-beta").num_args(1).default_value("1.0").help("MUTANT : simule à facteur·β"))
            .arg(Arg::new("sortie").long("sortie").required(true).num_args(1))
            .arg(fils()),
        Command::new("etalonnage-gates")
            .about("Gates RG-PLAQ d'un étalonnage wp9 et d'une chaîne indépendante")
            .arg(Arg::new("courbe").long("courbe").required(true).num_args(1))
            .arg(Arg::new("independante").long("independante").required(true).num_args(1))
            .arg(Arg::new("mutant").long("mutant").num_args(1))
            .arg(Arg::new("comparer").long("comparer").num_args(1))
            .arg(Arg::new("milieux").long("milieux").num_args(1).help("chaînes wp9 aux 30 milieux, validation directe de PCHIP pour P et R"))
            .arg(Arg::new("json").long("json").action(ArgAction::SetTrue)),
        Command::new("flot-gates")
            .about("Gates RG-SD, RG-PLAQ, RG-N3, RG-TAU, RG-SCHEMA (rapporté) : une ligne PASS|FAIL|SKIP|INFO par gate")
            .arg(Arg::new("donnees").long("donnees").num_args(1).default_value("tests/donnees/rg_flot"))
            .arg(Arg::new("grandl").long("grandl").num_args(1).help("racine des ensembles grandL (…/ensembles/production) ; absent ⇒ SKIP"))
            .arg(Arg::new("etalonnages").long("etalonnages").num_args(1).help("dossier de etal_L4.json, indep_L4.json, indep_mut_L4.json, etal_L6.json"))
            .arg(Arg::new("seul").long("seul").num_args(1).help("préfixes d'ID à exécuter (liste à virgules)"))
            .arg(Arg::new("sans-mutants").long("sans-mutants").action(ArgAction::SetTrue))
            .arg(Arg::new("json").long("json").action(ArgAction::SetTrue))
            .arg(Arg::new("certifier").long("certifier").num_args(0..=1).default_missing_value(CERT_DEFAUT).help("écrit le certificat (défaut validation/rg_flot/rg_flot.json) ; refusé si une ligne est FAIL ou SKIP"))
            .arg(fils()),
    ]
}

fn texte(m: &ArgMatches, nom: &str) -> Resultat<String> {
    m.get_one::<String>(nom).cloned().ok_or_else(|| Erreur::Config(format!("--{nom} requis")))
}

fn nombre_fils(m: &ArgMatches) -> Resultat<usize> {
    let n = match m.get_one::<String>("fils") {
        Some(t) => t.parse::<usize>().map_err(|_| Erreur::Config(format!("--fils « {t} » invalide")))?,
        None => std::env::var("RAYON_NUM_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(4),
    };
    if n == 0 {
        return Err(Erreur::Config("--fils doit être ≥ 1".into()));
    }
    Ok(n)
}

pub fn executer(nom: &str, m: &ArgMatches) -> Resultat<i32> {
    match nom {
        "flot" => flot_cmd(m),
        "etalonner" => etalonner_cmd(m),
        "flot-gates" => gates_cmd(m),
        "etalonnage-gates" => etalonnage_gates_cmd(m),
        _ => Err(Erreur::Config(format!("sous-commande « {nom} » inconnue du module rg flot"))),
    }
}

fn ecrire_neuf(chemin: &Path, doc: &Value) -> Resultat<()> {
    ecrire_neuf_fin(chemin, doc, None)
}

/// `serde_json::Map` trie les clés : déplacer explicitement la provenance WP9 à la fin du texte.
fn json_cle_finale(doc: &Value, cle: &str) -> Resultat<String> {
    let mut tete = doc.clone();
    let objet = tete.as_object_mut().ok_or_else(|| Erreur::Format("objet JSON attendu".into()))?;
    let valeur = objet.remove(cle).ok_or_else(|| Erreur::Format(format!("clé {cle} absente")))?;
    let mut fin = serde_json::Map::new();
    fin.insert(cle.into(), valeur);
    let serialiser = |v: &Value| serde_json::to_string_pretty(v).map_err(|e| Erreur::Format(e.to_string()));
    let prefixe = serialiser(&tete)?;
    let suffixe = serialiser(&Value::Object(fin))?;
    if tete.as_object().is_some_and(serde_json::Map::is_empty) {
        return Ok(suffixe + "\n");
    }
    Ok(format!("{},{}\n", &prefixe[..prefixe.len() - 2], &suffixe[1..]))
}

fn ecrire_neuf_fin(chemin: &Path, doc: &Value, cle: Option<&str>) -> Resultat<()> {
    if chemin.exists() {
        return Err(Erreur::Config(format!("{} existe déjà : refus d'écraser (supprimer à la main si voulu)", chemin.display())));
    }
    if let Some(d) = chemin.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(d).map_err(|e| Erreur::io("création du dossier de sortie", e))?;
    }
    if let Some(cle) = cle.filter(|cle| doc.get(*cle).is_some()) {
        let texte = json_cle_finale(doc, cle)?;
        let mut part = chemin.as_os_str().to_os_string();
        part.push(".part");
        let part = PathBuf::from(part);
        std::fs::write(&part, texte).map_err(|e| Erreur::io(format!("écriture {}", part.display()), e))?;
        std::fs::rename(&part, chemin).map_err(|e| Erreur::io(format!("renommage {}", chemin.display()), e))
    } else {
        crate::rg::domaine::ecrire_json(chemin, doc)
    }
}

fn flot_cmd(m: &ArgMatches) -> Resultat<i32> {
    let courbes_f: Vec<PathBuf> = m.get_many::<String>("courbe").map(|v| v.map(PathBuf::from).collect()).unwrap_or_default();
    let (courbes, fichiers) = lire_courbes(&courbes_f)?;
    let par = ParamFlot {
        schema: Schema::analyser(&texte(m, "schema")?)?,
        pas: texte(m, "pas")?.parse().map_err(|_| Erreur::Config("--pas invalide".into()))?,
        estimateurs: Estimateurs::analyser(&texte(m, "estimateurs")?)?,
        rapport: if m.get_flag("mutant-u1") { RapportBessel::I1SurI0 } else { RapportBessel::I2SurI1 },
        courbes,
        fils: nombre_fils(m)?,
    };
    let bins_tailles = m.get_one::<String>("bins").map(|s| analyser_bins(s)).transpose()?.unwrap_or_default();
    let options = OptionsFlot {
        backend: texte(m, "backend")?,
        bins: bins_tailles.first().copied(),
        bins_tailles,
        detail_configs: m.get_flag("detail-configs"),
        operateur: crate::rg::operateur::Operateur::analyser(&texte(m, "operateur")?)?,
        hors_famille: m.get_flag("hors-famille"),
    };
    let doc = flot_options(Path::new(&texte(m, "ensemble")?), m.get_one::<String>("filtre").map(String::as_str), &par, &fichiers, &options)?;
    let sortie = PathBuf::from(texte(m, "sortie")?).join("flot.json");
    ecrire_neuf_fin(&sortie, &doc, Some("calcul"))?;
    let n = doc["cellules"].as_array().map_or(0, Vec::len);
    println!("INFO RG-FLOT {n} cellule(s) écrites dans {}", sortie.display());
    Ok(code::OK)
}

fn etalonner_cmd(m: &ArgMatches) -> Resultat<i32> {
    let l: usize = texte(m, "L")?.parse().map_err(|_| Erreur::Config("--L invalide".into()))?;
    let mut par = match texte(m, "protocole")?.as_str() {
        "wp8" => ParamEtalonnage::gele(l),
        "wp9" => ParamEtalonnage::wp9(l),
        _ => return Err(Erreur::Config("--protocole attendu : wp8|wp9".into())),
    }.map_err(Erreur::Config)?;
    par.independante = m.get_flag("independante");
    if m.get_flag("milieux") {
        par.activer_milieux().map_err(Erreur::Config)?;
    }
    par.facteur_beta = texte(m, "facteur-beta")?.parse().map_err(|_| Erreur::Config("--facteur-beta invalide".into()))?;
    if par.facteur_beta != 1.0 && !par.independante {
        return Err(Erreur::Config("--facteur-beta (mutant) exige --independante".into()));
    }
    let chaines = etalonner(&par, nombre_fils(m)?)?;
    let doc = en_json(&par, &chaines)?;
    let sortie = PathBuf::from(texte(m, "sortie")?);
    ecrire_neuf_fin(&sortie, &doc, Some("protocole"))?;
    println!("INFO RG-ETALONNER {} chaînes (L = {l}, {} β) écrites dans {}", chaines.len(), par.betas.len(), sortie.display());
    Ok(code::OK)
}

fn gates_cmd(m: &ArgMatches) -> Resultat<i32> {
    let seul: Vec<String> = m.get_one::<String>("seul").map(|s| s.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or_default();
    let mutants = !m.get_flag("sans-mutants");
    if m.get_one::<String>("certifier").is_some() && (!seul.is_empty() || !mutants) {
        return Err(Erreur::Config("--certifier exige la batterie complète (ni --seul, ni --sans-mutants)".into()));
    }
    let ctx = Contexte {
        donnees: PathBuf::from(texte(m, "donnees")?),
        grandl: m.get_one::<String>("grandl").map(PathBuf::from),
        etalonnages: m.get_one::<String>("etalonnages").map(PathBuf::from),
        fils: nombre_fils(m)?,
        mutants,
        seul,
    };
    let t0 = std::time::Instant::now();
    let lignes = batterie(&ctx)?;
    for l in &lignes {
        println!("{l}");
    }
    let c = |v: Verdict| lignes.iter().filter(|x| x.verdict == v).count();
    let (np, nf, ns) = (c(Verdict::Pass), c(Verdict::Fail), c(Verdict::Skip));
    println!("INFO RG-FLOT-BILAN : {np} PASS, {nf} FAIL, {ns} SKIP, {:.1} s, {} fils", t0.elapsed().as_secs_f64(), ctx.fils);
    if m.get_flag("json") {
        let doc = json!({"tout_vert": tout_vert(&lignes), "pass": np, "fail": nf, "skip": ns, "lignes": lignes});
        println!("{}", serde_json::to_string(&doc).map_err(|e| Erreur::Format(e.to_string()))?);
    }
    if let Some(f) = m.get_one::<String>("certifier") {
        if certifiable(&lignes, "cpu").is_none() {
            return Err(Erreur::Validation("certificat refusé : au moins une ligne FAIL ou SKIP".into()));
        }
        let sha_bin = std::env::current_exe().ok().and_then(|p| sha256_fichier(&p).ok()).unwrap_or_else(|| "inconnu".into());
        let verdicts: Vec<Value> = lignes.iter().map(|x: &LigneGate| json!({"verdict": x.verdict.to_string(), "id": x.id, "detail": x.detail})).collect();
        let doc = json!({
            "schema": SCHEMA_CERT_FLOT,
            "horodatage": crate::regen::manifeste::maintenant_iso(),
            "binaire_sha256": sha_bin,
            "version": crate::cli::version_texte().replace('\n', " | "),
            "backend": "cpu",
            "tout_vert": true,
            "verdicts": verdicts,
        });
        let p = PathBuf::from(f);
        ecrire_neuf(&p, &doc)?;
        println!("INFO RG-FLOT-CERT certificat écrit : {}", p.display());
    }
    Ok(if nf == 0 && ns == 0 { code::OK } else { code::VALIDATION })
}

fn etalonnage_gates_cmd(m: &ArgMatches) -> Resultat<i32> {
    let courbe = PathBuf::from(texte(m, "courbe")?);
    let independante = PathBuf::from(texte(m, "independante")?);
    let mutant = m.get_one::<String>("mutant").map(PathBuf::from);
    let comparer = m.get_one::<String>("comparer").map(PathBuf::from);
    let milieux = m.get_one::<String>("milieux").map(PathBuf::from);
    let lignes = super::etalonnage_gates::batterie_milieux(&courbe, &independante, mutant.as_deref(), comparer.as_deref(), milieux.as_deref())?;
    for l in &lignes { println!("{l}"); }
    let ok = lignes.iter().all(|l| l.verdict != Verdict::Fail);
    if m.get_flag("json") {
        println!("{}", json!({"tout_vert": ok, "lignes": lignes}));
    }
    Ok(if ok { code::OK } else { code::VALIDATION })
}

#[cfg(test)]
mod tests_wp9 {
    use super::*;

    #[test]
    fn rg_wp9_serialisation_cle_finale() {
        for cle in ["calcul", "protocole"] {
            let doc = json!({"a": 1, cle: {"backend": "hip"}, "z": [1, 2]});
            let texte = json_cle_finale(&doc, cle).unwrap();
            assert_eq!(serde_json::from_str::<Value>(&texte).unwrap(), doc);
            assert!(texte.find(&format!("\"{cle}\"")).unwrap() > texte.find("\"z\"").unwrap());
            assert!(texte.ends_with("\n}\n"));
            let seul = json!({cle: "wp9"});
            assert_eq!(serde_json::from_str::<Value>(&json_cle_finale(&seul, cle).unwrap()).unwrap(), seul);
        }
    }
}
