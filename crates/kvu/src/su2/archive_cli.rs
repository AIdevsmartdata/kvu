//! Sous-commande `kvu su2 archive` : vérifier, indexer, extraire et lister les
//! archives `kvu-archive/1`.

use crate::commun::{Erreur, Resultat};
use crate::stockage::archive;
use crate::stockage::{npy, FormatChamp, GardeSsd};
use crate::su2::run::{ConfigRun, NOM_CONFIG_RUN};
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::path::{Path, PathBuf};

/// Argument `--archive` de `su2 gen`.
pub fn arg_archive() -> clap::Arg {
    Arg::new("archive")
        .long("archive")
        .action(ArgAction::SetTrue)
        .help("archive les dumps compressés dans archive.json (kvu-archive/1)")
}

/// Sous-commande `archive` : verifier | indexer | extraire | lister.
pub fn sous_commande() -> clap::Command {
    let dir = || Arg::new("DIR").required(true).help("répertoire du run");
    Command::new("archive")
        .about("Archive des dumps : verifier | indexer | extraire | lister")
        .subcommand_required(true)
        .subcommand(Command::new("verifier").about("vérifie l'intégrité de archive.json").arg(dir()))
        .subcommand(Command::new("indexer").about("(re)construit archive.json d'un run à dumps compressés").arg(dir()))
        .subcommand(
            Command::new("extraire")
                .about("vérifie puis écrit une configuration en .npy")
                .arg(dir())
                .arg(Arg::new("indice").long("indice").required(true).num_args(1).help("indice de mesure"))
                .arg(Arg::new("chaine").long("chaine").num_args(1).help("chaîne du lot (défaut 0)"))
                .arg(Arg::new("sortie").long("sortie").required(true).num_args(1).help("fichier .npy à écrire"))
                .arg(Arg::new("format").long("format").num_args(1).help("c16 | q64 (défaut : format de l'archive)")),
        )
        .subcommand(Command::new("lister").about("liste les configurations indexées").arg(dir()))
}

fn texte<'a>(m: &'a ArgMatches, nom: &str) -> Option<&'a str> {
    m.get_one::<String>(nom).map(|s| s.as_str())
}

fn chemin(m: &ArgMatches, nom: &str) -> Resultat<PathBuf> {
    let t = texte(m, nom)
        .ok_or_else(|| Erreur::Config(format!("--{nom} requis")))?;
    Ok(PathBuf::from(t))
}

fn nombre<T: std::str::FromStr>(m: &ArgMatches, nom: &str) -> Resultat<T> {
    let t = texte(m, nom).ok_or_else(|| Erreur::Config(format!("--{nom} requis")))?;
    t.parse::<T>()
        .map_err(|_| Erreur::Config(format!("--{nom} : « {t} » n'est pas un nombre valide")))
}

fn verifier_presence_run(dir: &Path) -> Resultat<()> {
    if !dir.join(NOM_CONFIG_RUN).is_file() {
        return Err(Erreur::Config(format!(
            "{} : pas un run su2 ({NOM_CONFIG_RUN} absent)",
            dir.display()
        )));
    }
    Ok(())
}

fn nombre_configurations(index: &archive::IndexArchive) -> usize {
    index.chaines.iter().map(|c| c.configurations.len()).sum()
}

pub fn executer_archive(m: &clap::ArgMatches) -> Resultat<i32> {
    match m.subcommand() {
        Some(("verifier", v)) => {
            let dir = chemin(v, "DIR")?;
            verifier_presence_run(&dir)?;
            let rapport = archive::verifier(&dir)?;
            for defaut in &rapport.defauts {
                println!("DEFAUT {} {}", defaut.nature, defaut.fichier);
            }
            Ok(if rapport.defauts.is_empty() {
                0
            } else {
                crate::commun::code::VALIDATION
            })
        }
        Some(("indexer", i)) => {
            let dir = chemin(i, "DIR")?;
            verifier_presence_run(&dir)?;
            let index = archive::indexer_run(&dir)?;
            println!("archive.json : {} configurations indexées", nombre_configurations(&index));
            Ok(0)
        }
        Some(("extraire", e)) => {
            let dir = chemin(e, "DIR")?;
            // O3 : `extraire` n'exige pas config_run.json ; seuls `indexer` et `verifier` l'exigent.
            // Intégrateur (M1 r1) : un index illisible est une corruption (rc 8, cas (h) de A4), pas une erreur d'usage.
            let index = archive::lire_index(&dir)
                .map_err(|_| Erreur::Validation(format!("{} : archive.json : index illisible ou invalide", dir.display())))?;
            let indice = nombre::<u64>(e, "indice")?;
            let chaine = match texte(e, "chaine") {
                Some(_) => nombre::<usize>(e, "chaine")?,
                None => 0,
            };
            let sortie = chemin(e, "sortie")?;
            if sortie.exists() {
                return Err(Erreur::Config(format!(
                    "{} existe déjà : refus d'écraser",
                    sortie.display()
                )));
            }
            let format_nom = match texte(e, "format") {
                Some(f) => f.to_string(),
                None => index.format_champ.clone(),
            };
            let format: FormatChamp = format_nom.parse()?;
            let champ = archive::extraire(&dir, chaine, indice)?;
            npy::ecrire_champ(&sortie, &champ, format, &GardeSsd::machine())?;
            println!("{} : configuration {indice} extraite", sortie.display());
            Ok(0)
        }
        Some(("lister", l)) => {
            let dir = chemin(l, "DIR")?;
            // O3 : `lister` n'exige pas config_run.json.
            let index = archive::lire_index(&dir)?;
            for chaine in &index.chaines {
                for entree in &chaine.configurations {
                    println!(
                        "{} {} {} {} {}",
                        chaine.chaine, entree.indice, entree.balayages, entree.fichier, entree.sha256_contenu
                    );
                }
            }
            Ok(0)
        }
        _ => Err(Erreur::Config(
            "archive : sous-commande attendue (verifier | indexer | extraire | lister)".to_string(),
        )),
    }
}

/// Valide les contraintes de `--archive` avant un run, force la compression zstd et rend `true`
/// si le mode archive est actif.
pub fn preparer_gen(m: &ArgMatches, cfg: &mut ConfigRun, _reprendre: bool) -> Resultat<bool> {
    let demande = m.get_flag("archive");
    let archive_existante = cfg.outdir.join(archive::NOM_INDEX).is_file();
    let actif = demande || archive_existante;
    if !actif {
        return Ok(false);
    }
    if cfg.dump_every == 0 {
        return Err(Erreur::Config("--archive exige --dump-every N ≥ 1".to_string()));
    }
    if cfg.bord.is_some() {
        return Err(Erreur::Config("--archive est incompatible avec --bord".to_string()));
    }
    if !crate::stockage::compression::disponible() {
        return Err(Erreur::Config(
            "--archive exige le binaire système `zstd`, absent".to_string(),
        ));
    }
    cfg.zstd = true;
    Ok(true)
}

/// Appelé après `run::executer_jusqua` : reconstruit `archive.json` si le mode archive est actif.
pub fn apres_gen(cfg: &ConfigRun, actif: bool) -> Resultat<()> {
    if !actif {
        return Ok(());
    }
    let index = archive::indexer_run(&cfg.outdir)?;
    eprintln!(
        "su2 gen : archive.json : {} configurations indexées",
        nombre_configurations(&index)
    );
    Ok(())
}
