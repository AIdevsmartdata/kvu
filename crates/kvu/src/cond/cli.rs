//! Sous-commande `cond` (chaine, sweep, tirage). Propriétaire : T5.
//!
//! ```text
//! kevinotron cond chaine --cfg X.npy --beta 2.3 --graine 2709 --mode ref_rng|rapide
//!            [--ntherm 100 --nsweep 6000 --every 2 --deach 5 --ckpt 500] [--backend auto|hip|cuda|cpu]
//!            --out DIR | --reprendre DIR
//! kevinotron cond sweep  --cfg X.npy --beta 2.3 --graine 0 --mode ref_rng|rapide [--n 1] --out Y.npy
//!            [--format q64|c16] [--backend …]
//! kevinotron cond tirage --alpha 5 --n 400000 --graine 1 --out x.npy [--backend …]
//! ```
//! Les gates GPU passent SOUS le verrou de la machine (`flock`) ; `cond chaine` le détecte (ancêtre) ou le prend.

use super::chaine::{executer_avec, ConfigChaine, Issue, OptionsChaine};
use super::moteur::{ModeCond, MoteurCond, SourceAlea};
use super::{geometrie::v_paires, GeometrieCond};
use crate::commun::{BackendKind, Erreur, Graine, Resultat};
use crate::gpu::Appareil;
use super::ressources;
use crate::heritage::manifest::sha256_fichier;
use crate::stockage::provenance::ecrire_manifeste_fichier;
use crate::stockage::{npy, FormatChamp, GardeSsd};
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::path::PathBuf;
use std::sync::Arc;

fn arg_backend() -> Arg {
    Arg::new("backend")
        .long("backend")
        .default_value("auto")
        .value_parser(["auto", "hip", "cuda", "cpu"])
        .help("auto : GPU de la machine (backend du profil), sinon CPU ; cpu : référence rayon")
}

fn arg_mode() -> Arg {
    Arg::new("mode")
        .long("mode")
        .default_value("ref_rng")
        .value_parser(["ref_rng", "rapide"])
        .help("ref_rng : tirages numpy de cb.py rejoués (loi de référence) ; rapide : Philox sur GPU, même loi")
}

fn arg_fils() -> Arg {
    Arg::new("fils")
        .long("fils")
        .value_parser(clap::value_parser!(usize))
        .help("fils rayon (référence CPU, tirages numpy)")
}

pub fn commandes() -> Vec<Command> {
    let chaine = Command::new("chaine")
        .about("Chaîne conditionnelle fin|grossier (cadence de chaine_v3) : F.npy, D.npy, blk.npy, controles.json")
        .arg(Arg::new("cfg").long("cfg").help("configuration fine de départ (NPY c16 ou q64)"))
        .arg(Arg::new("beta").long("beta").value_parser(clap::value_parser!(f64)))
        .arg(Arg::new("graine").long("graine").value_parser(clap::value_parser!(u64)))
        .arg(arg_mode())
        .arg(Arg::new("ntherm").long("ntherm").default_value("100").value_parser(clap::value_parser!(u64)))
        .arg(Arg::new("nsweep").long("nsweep").default_value("6000").value_parser(clap::value_parser!(u64)))
        .arg(Arg::new("every").long("every").default_value("2").value_parser(clap::value_parser!(u64)))
        .arg(Arg::new("deach").long("deach").default_value("5").value_parser(clap::value_parser!(u64)))
        .arg(Arg::new("ckpt").long("ckpt").default_value("500").value_parser(clap::value_parser!(u64)))
        .arg(Arg::new("out").long("out").help("répertoire de sortie NEUF"))
        .arg(
            Arg::new("reprendre")
                .long("reprendre")
                .help("répertoire d'une chaîne interrompue (la configuration est relue du manifeste)"),
        )
        .arg(
            Arg::new("arret-apres-ckpt")
                .long("arret-apres-ckpt")
                .hide(true)
                .value_parser(clap::value_parser!(u64))
                .help("TEST : s'arrête après l'écriture du N-ième point de reprise"),
        )
        .arg(
            Arg::new("sans-journal")
                .long("sans-journal")
                .action(ArgAction::SetTrue)
                .help("pas de progression sur stderr"),
        )
        .arg(
            Arg::new("tordu")
                .long("tordu")
                .action(ArgAction::SetTrue)
                .help("lecture tordue de chaine_v3 (torsion.py, c = 1) : Ft.npy (nmes, 18) et Dtw.npy (nD, 3, 18, 18)"),
        )
        .arg(arg_backend())
        .arg(arg_fils());
    let sweep = Command::new("sweep")
        .about("N balayages conditionnels depuis une configuration (V = dec(cfg) figé) ; écrit la configuration finale")
        .arg(Arg::new("cfg").long("cfg").required(true))
        .arg(Arg::new("beta").long("beta").required(true).value_parser(clap::value_parser!(f64)))
        .arg(Arg::new("graine").long("graine").required(true).value_parser(clap::value_parser!(u64)))
        .arg(arg_mode())
        .arg(Arg::new("n").long("n").default_value("1").value_parser(clap::value_parser!(u64)))
        .arg(Arg::new("out").long("out").required(true).help("fichier NPY de la configuration finale"))
        .arg(Arg::new("format").long("format").default_value("q64").value_parser(["q64", "c16"]))
        .arg(arg_backend())
        .arg(arg_fils());
    let tirage = Command::new("tirage")
        .about("Loi du tirage rapide de x0 pour un α (gate EQC-2) : n tirages Philox, écrits en NPY")
        .arg(Arg::new("alpha").long("alpha").required(true).value_parser(clap::value_parser!(f64)))
        .arg(Arg::new("n").long("n").default_value("400000").value_parser(clap::value_parser!(usize)))
        .arg(Arg::new("graine").long("graine").default_value("1").value_parser(clap::value_parser!(u64)))
        .arg(Arg::new("out").long("out").required(true))
        .arg(arg_backend())
        .arg(arg_fils());
    vec![Command::new("cond")
        .about("Bain thermique conditionnel « fin sachant grossier » : chaine | sweep | tirage")
        .subcommand_required(true)
        .subcommand(chaine)
        .subcommand(sweep)
        .subcommand(tirage)
        .subcommand(crate::cond3::cli::commande())]
}

/// Ouvre le GPU demandé (`auto` : celui de la machine ; `cpu` : aucun ; `hip` / `cuda` : forcé, erreur s'il manque).
///
/// `auto` passe par `gpu::ouvrir_auto` (détection de la machine, politique de T1).
pub fn ouvrir_backend(nom: &str) -> Resultat<Option<Appareil>> {
    match nom {
        "cpu" => Ok(None),
        "auto" => crate::gpu::ouvrir_auto(),
        autre => Ok(Some(crate::gpu::ouvrir(autre.parse::<BackendKind>()?, 0)?)),
    }
}

/// Fils rayon : `--fils N`, sinon `RAYON_NUM_THREADS` s'il est posé, sinon le plafond de la politique de la machine
/// (clé `fils_max` du profil : on ne prend jamais tous les cœurs d'une machine partagée).
fn fixer_fils(m: &ArgMatches) {
    let voulu = match m.get_one::<usize>("fils") {
        Some(n) => Some(*n),
        None if std::env::var_os("RAYON_NUM_THREADS").is_some() => None,
        None => Some(crate::machine::politique(&crate::machine::detection::detecter_rapide()).fils_max.max(1)),
    };
    if let Some(n) = voulu {
        // erreur ignorée : le pool global peut déjà exister (tests)
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    }
}

fn mode_de(m: &ArgMatches) -> ModeCond {
    match m.get_one::<String>("mode").map(String::as_str) {
        Some("rapide") => ModeCond::Rapide,
        _ => ModeCond::RefRng,
    }
}

fn requis<T: Clone + Send + Sync + 'static>(m: &ArgMatches, nom: &str) -> Resultat<T> {
    m.get_one::<T>(nom).cloned().ok_or_else(|| Erreur::Config(format!("--{nom} est requis")))
}

pub fn executer(_nom: &str, m: &ArgMatches) -> Resultat<i32> {
    match m.subcommand() {
        Some(("chaine", s)) => cmd_chaine(s),
        Some(("sweep", s)) => cmd_sweep(s),
        Some(("tirage", s)) => cmd_tirage(s),
        Some(("facteur3", s)) => crate::cond3::cli::executer(s),
        _ => Err(Erreur::Config("cond : sous-commande attendue (chaine | sweep | tirage)".into())),
    }
}

fn config_chaine(m: &ArgMatches) -> Resultat<(ConfigChaine, bool)> {
    if let Some(dir) = m.get_one::<String>("reprendre") {
        let p = PathBuf::from(dir).join("manifest.json");
        let t = std::fs::read_to_string(&p).map_err(|e| Erreur::io(format!("lecture {}", p.display()), e))?;
        let v: serde_json::Value =
            serde_json::from_str(&t).map_err(|e| Erreur::Format(format!("{} : {e}", p.display())))?;
        let mut cfg: ConfigChaine = serde_json::from_value(v["config"].clone())
            .map_err(|e| Erreur::Format(format!("{} : configuration illisible : {e}", p.display())))?;
        cfg.out = PathBuf::from(dir);
        if let Some(c) = m.get_one::<u64>("ckpt") {
            if m.value_source("ckpt") == Some(clap::parser::ValueSource::CommandLine) {
                cfg.ckpt = *c;
            }
        }
        return Ok((cfg, true));
    }
    let out: String = requis(m, "out").map_err(|_| Erreur::Config("--out DIR ou --reprendre DIR est requis".into()))?;
    Ok((
        ConfigChaine {
            cfg: PathBuf::from(requis::<String>(m, "cfg")?),
            beta: requis(m, "beta")?,
            graine: requis(m, "graine")?,
            mode: mode_de(m),
            ntherm: requis(m, "ntherm")?,
            nsweep: requis(m, "nsweep")?,
            every: requis(m, "every")?,
            deach: requis(m, "deach")?,
            ckpt: requis(m, "ckpt")?,
            out: PathBuf::from(out),
            tordu: m.get_flag("tordu"),
        },
        false,
    ))
}

fn cmd_chaine(m: &ArgMatches) -> Resultat<i32> {
    fixer_fils(m);
    let (cfg, reprendre) = config_chaine(m)?;
    let app = ouvrir_backend(m.get_one::<String>("backend").map(String::as_str).unwrap_or("auto"))?;
    let opts = OptionsChaine {
        arret_apres_ckpt: m.get_one::<u64>("arret-apres-ckpt").copied(),
        mesure: None,
        journal: !m.get_flag("sans-journal"),
    };
    match executer_avec(app.as_ref(), &cfg, reprendre, opts)? {
        Issue::Terminee(r) => {
            println!("{}", serde_json::to_string(&r).map_err(|e| Erreur::Format(e.to_string()))?);
            Ok(crate::commun::code::OK)
        }
        Issue::Interrompue { ckpt } => {
            eprintln!(
                "arrêt volontaire après le point de reprise n° {ckpt} (--reprendre {} pour continuer)",
                cfg.out.display()
            );
            Ok(crate::commun::code::OK)
        }
    }
}

fn cmd_sweep(m: &ArgMatches) -> Resultat<i32> {
    fixer_fils(m);
    let entree = PathBuf::from(requis::<String>(m, "cfg")?);
    let sortie = PathBuf::from(requis::<String>(m, "out")?);
    let format = if m.get_one::<String>("format").map(String::as_str) == Some("c16") {
        FormatChamp::C16
    } else {
        FormatChamp::Q64
    };
    let app = ouvrir_backend(m.get_one::<String>("backend").map(String::as_str).unwrap_or("auto"))?;
    // Avant tout calcul : admission mémoire (rc 5), garde SSD de la sortie (rc 7), puis verrou GPU (le GPU de la
    // production n'est jamais touché sans lui).
    let (_, l_entete) = npy::inspecter_champ(&entree)?;
    ressources::admettre(
        &format!("cond sweep L={l_entete}"),
        ressources::besoin_memoire(l_entete, 0, 0, 0, app.is_some()),
        app.as_ref(),
    )?;
    GardeSsd::machine().verifier(&sortie, npy::octets_champ(l_entete, format) + (1 << 20))?;
    let verrou = ressources::prendre_verrou(app.as_ref())?;
    let champ = npy::lire_champ(&entree)?;
    let l = champ.l;
    let geo = Arc::new(GeometrieCond::nouvelle(l)?);
    let vfix = v_paires(&champ, &geo);
    let alea = SourceAlea::depuis_graine(mode_de(m), requis(m, "graine")?);
    let mut moteur = MoteurCond::nouveau(app.as_ref(), geo, &champ, vfix, requis(m, "beta")?, alea)?;
    let n: u64 = requis(m, "n")?;
    for _ in 0..n {
        moteur.sweep()?;
    }
    let fin = moteur.telecharger()?;
    let (vdev, unitarite) = moteur.controles()?;
    npy::ecrire_champ(&sortie, &fin, format, &GardeSsd::machine())?;
    let alea_json = serde_json::to_value(moteur.alea()).map_err(|e| Erreur::Format(e.to_string()))?;
    let resultat = serde_json::json!({"vdev": vdev, "unitarite": unitarite, "balayages": n, "backend": moteur.backend().nom(), "alea": alea_json});
    // Provenance de la configuration écrite (`<sortie>.manifest.json`) : binaire, git_head, noyaux, backend effectif,
    // module compilé, entrée et paramètres.
    let sha_entree = sha256_fichier(&entree).map_err(|e| Erreur::io(format!("sha256 {}", entree.display()), e))?;
    let argv: Vec<String> = std::env::args().collect();
    ecrire_manifeste_fichier(
        &sortie,
        "kvu cond sweep",
        &argv,
        &[
            ("backend", serde_json::json!(moteur.backend().nom())),
            ("verrou", serde_json::json!(verrou.etat)),
            ("modules_compiles", serde_json::json!(moteur.provenance().map(|p| vec![p.clone()]).unwrap_or_default())),
            ("entree", serde_json::json!({"chemin": entree, "sha256": sha_entree, "L": l})),
            ("parametres", serde_json::json!({"beta": requis::<f64>(m, "beta")?, "graine": requis::<u64>(m, "graine")?,
                                               "mode": mode_de(m), "balayages": n, "format": format.nom()})),
            ("resultat", resultat.clone()),
        ],
    )?;
    println!("{resultat}");
    Ok(crate::commun::code::OK)
}

fn cmd_tirage(m: &ArgMatches) -> Resultat<i32> {
    fixer_fils(m);
    let app = ouvrir_backend(m.get_one::<String>("backend").map(String::as_str).unwrap_or("auto"))?;
    let alpha: f64 = requis(m, "alpha")?;
    let n: usize = requis(m, "n")?;
    let graine = Graine(requis(m, "graine")?);
    let besoin = crate::machine::admission::BesoinMemoire {
        hote: 16 * n as u64,
        gpu: if app.is_some() { 16 * n as u64 } else { 0 },
    };
    ressources::admettre(&format!("cond tirage n={n}"), besoin, app.as_ref())?;
    GardeSsd::machine().verifier(&PathBuf::from(requis::<String>(m, "out")?), 8 * n as u64 + 128)?;
    let _verrou = ressources::prendre_verrou(app.as_ref())?;
    let (x, plafond) = match &app {
        Some(a) => super::gpu::tirer_x0(a, alpha, n, graine)?,
        None => super::cpu::tirer_x0(alpha, n, graine),
    };
    npy::ecrire_f64(&PathBuf::from(requis::<String>(m, "out")?), &[n], &x, &GardeSsd::machine())?;
    println!(
        "{}",
        serde_json::json!({"alpha": alpha, "n": n, "plafond": plafond, "backend": app.as_ref().map(|a| a.backend().nom()).unwrap_or("cpu")})
    );
    Ok(crate::commun::code::OK)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyser(args: &[&str]) -> Result<ArgMatches, clap::Error> {
        let mut v = vec!["kvu"];
        v.extend_from_slice(args);
        commandes().remove(0).try_get_matches_from(v)
    }

    #[test]
    fn commande_cond_et_sous_commandes() {
        let c = commandes();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].get_name(), "cond");
        let noms: Vec<&str> = c[0].get_subcommands().map(|s| s.get_name()).collect();
        assert_eq!(noms, vec!["chaine", "sweep", "tirage", "facteur3"]);
    }

    #[test]
    fn valeurs_par_defaut_de_la_cadence_v3() {
        let m = analyser(&["chaine", "--cfg", "x.npy", "--beta", "2.3", "--graine", "1", "--out", "o"]).unwrap();
        let (cfg, reprendre) = config_chaine(m.subcommand_matches("chaine").unwrap()).unwrap();
        assert!(!reprendre);
        assert_eq!((cfg.ntherm, cfg.nsweep, cfg.every, cfg.deach, cfg.ckpt), (100, 6000, 2, 5, 500));
        assert_eq!(cfg.mode, ModeCond::RefRng);
    }

    #[test]
    fn sans_out_ni_reprendre_refuse() {
        let m = analyser(&["chaine", "--cfg", "x.npy", "--beta", "2.3", "--graine", "1"]).unwrap();
        assert!(config_chaine(m.subcommand_matches("chaine").unwrap()).is_err());
    }

    #[test]
    fn mode_inconnu_refuse() {
        assert!(analyser(&["chaine", "--mode", "metropolis"]).is_err());
        assert!(analyser(&[
            "sweep",
            "--cfg",
            "x",
            "--beta",
            "1",
            "--graine",
            "0",
            "--out",
            "y",
            "--backend",
            "vulkan"
        ])
        .is_err());
    }
}
