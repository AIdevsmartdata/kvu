//! `kevinotron valide` : suite de validation (régression des 13 gates historiques G-H01..G-H13,
//! lois FP32/FP64, équivalence conditionnelle, observables, bord, lots, reprise, provenance).
//! Propriétaire : T9.
//!
//! La suite est un script (`validation/gates_unifie.sh`) qui orchestre les scripts de chaque tâche
//! (`validation/<tâche>/gates_*.sh`) et les gates historiques (`validation/historique/`), sur
//! la machine locale puis, par le câble du profil (`cable` de `kvu.toml`), sur l'autre machine. Ce module :
//! 1. retrouve le script, le lance avec l'environnement du contrat (voir ci-dessous) ;
//! 2. lit les lignes normalisées `GATE<TAB>statut<TAB>machine<TAB>nom<TAB>détail` ;
//! 3. ajoute un FAIL pour chaque gate attendue (catalogue §6) que le script n'a pas émise ;
//! 4. rend un tableau ou le JSON gelé `{"gates":[…],"fail":n,"skip":n}` ; rc 0 seulement si
//!    aucun FAIL et aucun SKIP.
//!
//! Contrat d'environnement fourni aux scripts : `KVU_MODE` (rapide|complet), `KVU_GATE` (motif de
//! gate, vide = toutes), `KVU_MACHINES` (liste), `KVU_BIN` (binaire kevinotron courant),
//! `KVU_RACINE` (racine du dépôt), `KVU_SANS_GPU=1` (mode rapide : ne touche ni GPU ni autre machine).

use super::catalogue::{self, Gate, Mode, Statut};
use super::usage;
use crate::commun::{Erreur, Resultat};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const NOM_SCRIPT: &str = "gates_unifie.sh";

pub fn commandes() -> Vec<clap::Command> {
    use clap::{Arg, ArgAction, Command};
    vec![Command::new("valide")
        .about("Suite de validation : --rapide | --complet | --gate NOM ; --json ; --dry-run")
        .long_about(
            "Lance validation/gates_unifie.sh : les gates de ARCHITECTURE_UNIFIEE.md §6, dont les 13 gates\n\
             historiques G-H01..G-H13, sur la machine locale puis (par le câble du profil) sur l'autre machine.\n\
             Un SKIP n'est pas vert : rc 0 seulement s'il n'y a ni FAIL ni SKIP. rc 8 sinon.",
        )
        .arg(
            Arg::new("rapide")
                .long("rapide")
                .action(ArgAction::SetTrue)
                .help("gates CPU seulement (défaut)"),
        )
        .arg(
            Arg::new("complet")
                .long("complet")
                .action(ArgAction::SetTrue)
                .help("toutes les gates, deux machines"),
        )
        .arg(
            Arg::new("gate")
                .long("gate")
                .value_name("NOM")
                .help("une gate (ou un préfixe : G-H0)"),
        )
        .arg(
            Arg::new("machines")
                .long("machines")
                .value_name("LISTE")
                .help("liste de machines du kvu.toml, séparées par des virgules (défaut : selon le mode)"),
        )
        .arg(
            Arg::new("json")
                .long("json")
                .action(ArgAction::SetTrue)
                .help("sortie JSON gelée"),
        )
        .arg(
            Arg::new("dry-run")
                .long("dry-run")
                .action(ArgAction::SetTrue)
                .help("affiche le plan sans rien lancer"),
        )
        .group(
            clap::ArgGroup::new("mode")
                .args(["rapide", "complet", "gate"])
                .multiple(false),
        )]
}

/// Plan d'une validation (pur, testable).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub mode: Mode,
    pub motif: Option<String>,
    pub machines: Vec<String>,
    pub json: bool,
}

/// Machines reconnaissables du fichier de configuration (`kvu.toml`), dans l'ordre de reconnaissance ; la machine
/// locale en fait toujours partie (sans fichier, seule la machine locale existe).
fn machines_connues(local: &str) -> Vec<String> {
    let mut v: Vec<String> = crate::config::actuelle()
        .machines_reconnaissables()
        .iter()
        .map(|m| m.nom().to_string())
        .collect();
    if !v.iter().any(|m| m == local) {
        v.push(local.to_string());
    }
    v
}

/// Construit le plan depuis les options. `local` = nom de la machine courante.
pub fn plan_depuis(m: &clap::ArgMatches, local: &str) -> Resultat<Plan> {
    let motif = m.get_one::<String>("gate").cloned();
    let mode = if m.get_flag("complet") || motif.is_some() {
        Mode::Complet
    } else {
        Mode::Rapide
    };
    let machines = match m.get_one::<String>("machines") {
        Some(l) => {
            let v: Vec<String> = l
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let connues = machines_connues(local);
            for x in &v {
                if !connues.contains(x) {
                    return Err(usage(format!("machine « {x} » inconnue ({})", connues.join("|"))));
                }
            }
            if v.is_empty() {
                return Err(usage("--machines : liste vide"));
            }
            v
        }
        None => match mode {
            Mode::Rapide => vec![local.to_string()],
            Mode::Complet => machines_connues(local),
        },
    };
    Ok(Plan {
        mode,
        motif,
        machines,
        json: m.get_flag("json"),
    })
}

/// Environnement transmis au script.
pub fn environnement(plan: &Plan, bin: &Path, racine: &Path) -> Vec<(String, String)> {
    let mut v = vec![
        ("KVU_MODE".to_string(), plan.mode.nom().to_string()),
        ("KVU_GATE".to_string(), plan.motif.clone().unwrap_or_default()),
        ("KVU_MACHINES".to_string(), plan.machines.join(",")),
        ("KVU_BIN".to_string(), bin.display().to_string()),
        ("KVU_RACINE".to_string(), racine.display().to_string()),
    ];
    if plan.mode == Mode::Rapide {
        v.push(("KVU_SANS_GPU".to_string(), "1".to_string()));
    }
    v
}

/// Emplacements candidats du script, dans l'ordre : variable `KVU_VALIDATION_DIR`, ancêtres de
/// l'exécutable, ancêtres du répertoire courant, puis le répertoire des sources compilées.
pub fn candidats_script(env_dir: Option<&str>, exe: Option<&Path>, cwd: Option<&Path>) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(d) = env_dir.filter(|d| !d.is_empty()) {
        v.push(Path::new(d).join(NOM_SCRIPT));
    }
    for base in [exe, cwd].into_iter().flatten() {
        for anc in base.ancestors() {
            v.push(anc.join("crates/kvu/validation").join(NOM_SCRIPT));
            v.push(anc.join("validation").join(NOM_SCRIPT));
        }
    }
    // Sous build_auto.sh, le chemin est remappé (/kvu/crates/kvu, qui n'existe pas) : ce candidat
    // est simplement sauté ; `kvu valide` trouve le script par KVU_VALIDATION_DIR, le dossier courant
    // ou l'exécutable.
    v.push(
        Path::new(env!("KVU_DOSSIER_CRATE"))
            .join("validation")
            .join(NOM_SCRIPT),
    );
    v
}

/// Premier candidat qui existe.
pub fn trouver_script() -> Resultat<PathBuf> {
    let env_dir = std::env::var("KVU_VALIDATION_DIR").ok();
    let exe = std::env::current_exe().ok();
    let cwd = std::env::current_dir().ok();
    let cands = candidats_script(env_dir.as_deref(), exe.as_deref(), cwd.as_deref());
    cands.iter().find(|p| p.is_file()).cloned().ok_or_else(|| {
        Erreur::Config(format!(
            "{NOM_SCRIPT} introuvable (définir KVU_VALIDATION_DIR) ; essayé : {}",
            cands.len()
        ))
    })
}

/// Racine du dépôt déduite de l'emplacement du script (`…/crates/kvu/validation/gates_unifie.sh`).
pub fn racine_depuis_script(script: &Path) -> PathBuf {
    let dossier = script.parent().unwrap_or(Path::new("."));
    // validation/ -> kvu/ -> crates/ -> racine
    dossier
        .ancestors()
        .nth(3)
        .filter(|r| r.join("Cargo.toml").exists())
        .unwrap_or(dossier)
        .to_path_buf()
}

/// Lance le script, rend (gates émises, code de sortie du script).
pub fn lancer(script: &Path, env: &[(String, String)], progression: bool) -> Resultat<(Vec<Gate>, i32)> {
    let mut cmd = Command::new("bash");
    cmd.arg(script).stdout(Stdio::piped()).stderr(Stdio::inherit());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut enfant = cmd
        .spawn()
        .map_err(|e| Erreur::io(format!("lancement de {}", script.display()), e))?;
    let sortie = enfant.stdout.take().expect("stdout capturé");
    let mut gates = Vec::new();
    for ligne in BufReader::new(sortie).lines() {
        let ligne = ligne.map_err(|e| Erreur::io("lecture de la sortie du script", e))?;
        match catalogue::analyser_ligne(&ligne) {
            Some(g) => {
                if progression {
                    eprintln!("[{}] {} {} {}", g.machine, g.statut.nom(), g.nom, g.detail);
                }
                gates.push(g);
            }
            None => {
                if progression {
                    eprintln!("{ligne}");
                }
            }
        }
    }
    let st = enfant.wait().map_err(|e| Erreur::io("attente du script", e))?;
    Ok((gates, st.code().unwrap_or(-1)))
}

/// Un SKIP qui ne compte pas : une ligne dont le nom n'est pas celui d'une gate (« enfant : lancé seul »), ou une gate du
/// catalogue qui n'est pas attendue sur cette machine (« propre à CUDA » vu sur une machine HIP). Un SKIP d'une gate attendue,
/// lui, reste un SKIP, donc rouge. Les FAIL ne sont jamais ignorés.
pub fn skip_ignorable(g: &Gate) -> Option<&'static str> {
    if g.statut != Statut::Skip {
        return None;
    }
    let ressemble_a_une_gate = g
        .nom
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
        && g.nom.chars().any(|c| c == '-' || c.is_ascii_digit());
    if !ressemble_a_une_gate {
        return Some("la ligne SKIP ne nomme pas une gate");
    }
    match catalogue::CATALOGUE.iter().find(|a| a.couverte_par(&g.nom)) {
        Some(a) if !a.machines.contains(&g.machine.as_str()) => Some("gate non attendue sur cette machine"),
        _ => None,
    }
}

/// Rapport final : émises (sans les SKIP ignorables, listés sur stderr), puis manquantes (+ un FAIL « orchestrateur » si le
/// script a échoué sans FAIL).
pub fn finaliser(plan: &Plan, emises: Vec<Gate>, rc_script: i32) -> Vec<Gate> {
    let mut gardees = Vec::with_capacity(emises.len());
    for g in emises {
        match skip_ignorable(&g) {
            Some(raison) => eprintln!("[ignoré] SKIP {} ({}) : {} — {}", g.nom, g.machine, g.detail, raison),
            None => gardees.push(g),
        }
    }
    let mut toutes = catalogue::completer(
        gardees,
        &catalogue::attendues(plan.mode, &plan.machines, plan.motif.as_deref()),
    );
    let (_, fails, _) = catalogue::totaux(&toutes);
    if rc_script != 0 && fails == 0 {
        toutes.push(Gate {
            nom: NOM_SCRIPT.to_string(),
            machine: plan.machines.first().cloned().unwrap_or_default(),
            statut: Statut::Fail,
            detail: format!("le script s'est terminé avec rc = {rc_script} sans signaler de FAIL"),
        });
    }
    toutes
}

pub fn executer(nom: &str, m: &clap::ArgMatches) -> Resultat<i32> {
    debug_assert_eq!(nom, "valide");
    let local = crate::machine::detection::detecter_rapide().machine.nom();
    let plan = plan_depuis(m, local)?;
    let script = trouver_script()?;
    let bin = std::env::current_exe().map_err(|e| Erreur::io("exécutable courant", e))?;
    let racine = racine_depuis_script(&script);
    let env = environnement(&plan, &bin, &racine);
    if m.get_flag("dry-run") {
        println!("plan de validation (rien n'est lancé)");
        println!("  script   : {}", script.display());
        println!("  mode     : {}", plan.mode.nom());
        println!("  machines : {}", plan.machines.join(","));
        println!("  gate     : {}", plan.motif.as_deref().unwrap_or("toutes"));
        let att = catalogue::attendues(plan.mode, &plan.machines, plan.motif.as_deref());
        println!("  gates attendues : {}", att.len());
        for (k, v) in &env {
            println!("  env {k}={v}");
        }
        return Ok(crate::commun::code::OK);
    }
    let (emises, rc) = lancer(&script, &env, true)?;
    let toutes = finaliser(&plan, emises, rc);
    if plan.json {
        println!("{}", catalogue::json(&toutes));
    } else {
        print!("{}", catalogue::tableau(&toutes));
    }
    Ok(catalogue::code_de(&toutes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches(args: &str) -> clap::ArgMatches {
        let cmd = commandes().remove(0);
        cmd.try_get_matches_from(args.split_whitespace()).unwrap()
    }

    #[test]
    fn mode_par_defaut_rapide_sur_la_machine_locale() {
        let p = plan_depuis(&matches("valide"), "acemagic").unwrap();
        assert_eq!(
            p,
            Plan {
                mode: Mode::Rapide,
                motif: None,
                machines: vec!["acemagic".into()],
                json: false
            }
        );
    }

    #[test]
    fn complet_vise_les_deux_machines() {
        let p = plan_depuis(&matches("valide --complet --json"), "acemagic").unwrap();
        assert_eq!(p.mode, Mode::Complet);
        assert_eq!(p.machines, vec!["acemagic", "gamer"]);
        assert!(p.json);
    }

    #[test]
    fn gate_implique_complet_et_garde_le_motif() {
        let p = plan_depuis(&matches("valide --gate G-H06"), "acemagic").unwrap();
        assert_eq!(p.mode, Mode::Complet);
        assert_eq!(p.motif.as_deref(), Some("G-H06"));
    }

    #[test]
    fn options_exclusives_et_machines_validees() {
        assert!(commandes()
            .remove(0)
            .try_get_matches_from(["valide", "--rapide", "--complet"])
            .is_err());
        assert!(plan_depuis(&matches("valide --machines pc"), "acemagic").is_err());
        let p = plan_depuis(&matches("valide --complet --machines gamer"), "acemagic").unwrap();
        assert_eq!(p.machines, vec!["gamer"]);
    }

    #[test]
    fn environnement_du_contrat() {
        let p = Plan {
            mode: Mode::Rapide,
            motif: Some("G-H01".into()),
            machines: vec!["acemagic".into()],
            json: false,
        };
        let e = environnement(&p, Path::new("/b/kevinotron"), Path::new("/r"));
        let get = |k: &str| e.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
        assert_eq!(get("KVU_MODE"), Some("rapide"));
        assert_eq!(get("KVU_GATE"), Some("G-H01"));
        assert_eq!(get("KVU_BIN"), Some("/b/kevinotron"));
        assert_eq!(get("KVU_SANS_GPU"), Some("1"));
        let c = Plan {
            mode: Mode::Complet,
            ..p
        };
        assert!(environnement(&c, Path::new("/b"), Path::new("/r"))
            .iter()
            .all(|(k, _)| k != "KVU_SANS_GPU"));
    }

    #[test]
    fn candidats_dans_lordre() {
        let c = candidats_script(Some("/x"), Some(Path::new("/a/b/kvu")), Some(Path::new("/c")));
        assert_eq!(c[0], Path::new("/x/gates_unifie.sh"));
        assert!(c.contains(&PathBuf::from("/a/crates/kvu/validation/gates_unifie.sh")));
        assert!(c.last().unwrap().ends_with("validation/gates_unifie.sh"));
        let sans = candidats_script(None, None, None);
        assert_eq!(sans.len(), 1);
    }

    #[test]
    fn le_script_du_depot_existe_et_est_trouve_par_les_sources() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("validation")
            .join(NOM_SCRIPT);
        assert!(p.is_file(), "{} absent", p.display());
        let r = racine_depuis_script(&p);
        assert!(r.join("Cargo.toml").exists(), "racine {} sans Cargo.toml", r.display());
    }

    #[test]
    fn finaliser_ajoute_les_gates_manquantes_et_le_fail_orchestrateur() {
        let plan = Plan {
            mode: Mode::Rapide,
            motif: Some("G-H01".into()),
            machines: vec!["acemagic".into()],
            json: true,
        };
        // rien d'émis, script rc 0 : la gate attendue manque => FAIL
        let t = finaliser(&plan, vec![], 0);
        assert_eq!(catalogue::totaux(&t).1, 1);
        // script en échec sans FAIL émis : un FAIL orchestrateur s'ajoute (en plus du manquant ici)
        let ok = vec![Gate {
            nom: "G-H01".into(),
            machine: "acemagic".into(),
            statut: Statut::Pass,
            detail: String::new(),
        }];
        let t = finaliser(&plan, ok, 3);
        assert!(
            t.iter().any(|g| g.nom == NOM_SCRIPT && g.statut == Statut::Fail),
            "{t:?}"
        );
        let ok = vec![Gate {
            nom: "G-H01".into(),
            machine: "acemagic".into(),
            statut: Statut::Pass,
            detail: String::new(),
        }];
        assert_eq!(catalogue::code_de(&finaliser(&plan, ok, 0)), 0);
    }

    #[test]
    fn skip_ignorables_et_skip_qui_comptent() {
        let sk = |nom: &str, machine: &str| Gate {
            nom: nom.into(),
            machine: machine.into(),
            statut: Statut::Skip,
            detail: String::new(),
        };
        // ligne qui ne nomme pas une gate
        assert!(skip_ignorable(&sk("enfant", "acemagic")).is_some());
        assert!(skip_ignorable(&sk(":", "gamer")).is_some());
        // gate propre au gamer vue sur acemagic : non applicable
        assert!(skip_ignorable(&sk("G-BIT-CUDA", "acemagic")).is_some());
        // la même sur le gamer : un vrai SKIP, rouge
        assert!(skip_ignorable(&sk("G-BIT-CUDA", "gamer")).is_none());
        // gate attendue partout : un SKIP compte
        assert!(skip_ignorable(&sk("LOI-2", "acemagic")).is_none());
        // un FAIL n'est jamais ignoré
        let f = Gate {
            statut: Statut::Fail,
            ..sk("enfant", "acemagic")
        };
        assert!(skip_ignorable(&f).is_none());
    }
}
