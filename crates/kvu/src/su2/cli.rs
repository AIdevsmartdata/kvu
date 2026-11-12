//! Sous-commande `su2` (gen, archive, test-hb, selftest). Propriétaire : T3.
//!
//! ```text
//! su2 gen --L 16 --beta 2.3[,2.8…] --graine S[,S2…] [--chaines K] [--graine-pas P] --start hot|cold
//!         --therm N --nmeas N --skip 1 --nor 4 [--precision f64|f32] [--backend auto|hip|cuda|cpu]
//!         [--options ref-k1|compat-prod|rapide] [--dump-every N --dump-initial --format c16|q64 --zstd]
//!         [--archive] [--mesures SPEC] [--reprise-every N] [--torsion PLANS]
//!         [--bord --bc cold|hot|eq --bc-graine G --jauge G --pre N --bloc N] --outdir DIR | --reprendre DIR
//! su2 archive verifier DIR | indexer DIR | extraire DIR --indice M [--chaine C] --sortie F [--format c16|q64] | lister DIR
//! su2 test-hb ALPHA N --graine S [--precision …] [--backend …] [--options …]    (ligne TESTHB de su2_hip)
//! su2 selftest                                                                   (KAT Philox, bornes u01)
//! su2 boite --ell L --beta B (--bords … | --chaines …) …                         (boîtes de Dirichlet à bord fixé : voir boite_cli)
//! su2 boite-obs --cfg X.npy --ell L                                               (observables d'une configuration de boîte)
//! ```
//! Les noms historiques `--seed` et `--bc-seed` sont acceptés (alias) : les lignes de commande de su2_hip et de
//! su2_bord fonctionnent telles quelles. Défaut des options numériques : F64 ⇒ `ref-k1`, F32 ⇒ `rapide`.
//! `--torsion PLANS` : « aucune » ou liste de paires (ex. `01,23`) ; CPU F64 uniquement, incompatible avec bord, mesures, GPU.

use super::run::{self, ConfigBord, ConfigRun, MesureEnLigne, OptionsRun};
use super::torsion::Torsion;
use super::{ConfigLot, ParamChaine};
use crate::commun::{BackendKind, Depart, Erreur, Graine, Precision, Resultat};
use crate::gpu::{Appareil, OptionsCompil};
use crate::heritage::su2_hb::{self, g17};
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::path::{Path, PathBuf};

pub fn commandes() -> Vec<clap::Command> {
    vec![Command::new("su2")
        .about("Générateur SU(2) unifié : gen | archive | test-hb | boite | boite-obs | selftest")
        .subcommand_required(true)
        .subcommand(gen())
        .subcommand(super::archive_cli::sous_commande())
        .subcommand(test_hb())
        .subcommand(super::boite_cli::boite())
        .subcommand(super::boite_cli::boite_obs())
        .subcommand(Command::new("selftest").about("Auto-test : vecteurs de Philox4x32-10 (Random123) et bornes de u01"))]
}

fn arg(nom: &'static str, aide: &'static str) -> Arg {
    Arg::new(nom).long(nom).help(aide).num_args(1)
}

fn communs(c: Command) -> Command {
    c.arg(arg("precision", "f64 (référence) | f32 (rapide)").default_value("f64"))
        .arg(arg("backend", "auto | hip | cuda | cpu (un backend forcé mais absent est une erreur)").default_value("auto"))
        .arg(arg("options", "ref-k1 | compat-prod | rapide (défaut : f64 → ref-k1, f32 → rapide)"))
        .arg(Arg::new("mutant").long("mutant").hide(true).num_args(1).help("contrôle négatif : metro"))
}

fn gen() -> Command {
    communs(Command::new("gen").about("Génère un ensemble (lot de chaînes), périodique ou à bord gelé R2-3"))
        .arg(arg("L", "côté du réseau (pair)").default_value("8"))
        .arg(arg("beta", "β, ou liste β1,β2,… (un par chaîne)").default_value("2.3"))
        .arg(arg("graine", "graine 64 bits, ou liste (une par chaîne)").alias("seed").default_value("1"))
        .arg(arg("chaines", "nombre de chaînes du lot (défaut : longueur des listes)"))
        .arg(arg("graine-pas", "graine de la chaîne i = graine + i·PAS (lot avec une seule graine)"))
        .arg(arg("start", "hot | cold").default_value("hot"))
        .arg(arg("therm", "pas de thermalisation").default_value("1000"))
        .arg(arg("nmeas", "nombre de mesures").default_value("100"))
        .arg(arg("skip", "pas entre deux mesures").default_value("1"))
        .arg(arg("nor", "sweeps d'overrelaxation par pas").default_value("4"))
        .arg(arg("dump-every", "un dump cfg_NNNNNN.npy tous les N indices de mesure (0 = jamais)").default_value("0"))
        .arg(Arg::new("dump-initial").long("dump-initial").action(ArgAction::SetTrue).help("écrit cfg_000000.npy"))
        .arg(arg("format", "format des dumps : c16 | q64").default_value("c16"))
        .arg(Arg::new("zstd").long("zstd").action(ArgAction::SetTrue).help("compresse les dumps"))
        .arg(super::archive_cli::arg_archive())
        .arg(arg("mesures", "mesures en ligne, ex. wilson:R4xT4:10,flot:0.01:3:20"))
        .arg(arg("reprise-every", "point de reprise tous les N pas (0 = jamais)").default_value("0"))
        .arg(arg("torsion", "torsion de 't Hooft : « aucune » ou liste de paires (ex: 01,23) ; CPU F64 uniquement"))
        .arg(Arg::new("bord").long("bord").action(ArgAction::SetTrue).help("mode R2-3 : liens spatiaux de x0 = 0 gelés"))
        .arg(arg("bc", "bord : cold | hot | eq"))
        .arg(arg("bc-graine", "graine du bord").alias("bc-seed").default_value("0"))
        .arg(arg("jauge", "graine de la jauge aléatoire du bord (contrôle nul C2)"))
        .arg(arg("pre", "pas de pré-thermalisation périodique (bord)").default_value("2000"))
        .arg(arg("bloc", "pas par bloc (bord)").default_value("100"))
        .arg(arg("outdir", "répertoire de sortie (NEUF)"))
        .arg(arg("reprendre", "reprend le run du répertoire DIR (point de reprise)"))
        .arg(Arg::new("arret-apres").long("arret-apres").hide(true).num_args(1).help("tests : s'arrête après N pas, avec point de reprise"))
        .arg(Arg::new("fenetre-s").long("fenetre-s").hide(true).num_args(1).help("WP10 : s'arrête proprement (point de reprise, `--reprendre`) après ce nombre de secondes"))
        .arg(arg("source-hash", "accepté pour compatibilité avec su2_hip (ignoré : la provenance est dans manifest.json)"))
}

fn test_hb() -> Command {
    communs(Command::new("test-hb").about("Test du tirage de x0 : n tirages à α fixé, moments m1 et m2 (ligne TESTHB)"))
        .arg(Arg::new("alpha").required(true).help("α"))
        .arg(Arg::new("n").required(true).help("nombre de tirages"))
        .arg(arg("graine", "graine").alias("seed").default_value("1"))
}

// ------------------------------------------------------------------------------------------
// Analyse des options
// ------------------------------------------------------------------------------------------

fn texte<'a>(m: &'a ArgMatches, nom: &str) -> Option<&'a str> {
    m.get_one::<String>(nom).map(|s| s.as_str())
}

fn nombre<T: std::str::FromStr>(m: &ArgMatches, nom: &str) -> Resultat<T> {
    let t = texte(m, nom).ok_or_else(|| Erreur::Config(format!("--{nom} requis")))?;
    t.parse::<T>().map_err(|_| Erreur::Config(format!("--{nom} : « {t} » n'est pas un nombre valide")))
}

/// Liste séparée par des virgules.
pub fn liste<T: std::str::FromStr>(s: &str, nom: &str) -> Resultat<Vec<T>> {
    s.split(',')
        .map(|t| t.trim().parse::<T>().map_err(|_| Erreur::Config(format!("--{nom} : « {t} » n'est pas valide"))))
        .collect()
}

/// Ouvre l'appareil demandé : auto ⇒ choix de la machine (None si CPU), cpu ⇒ None, hip/cuda ⇒ forcé.
pub fn ouvrir_backend(nom: &str) -> Resultat<Option<Appareil>> {
    match nom.trim().to_ascii_lowercase().as_str() {
        "auto" => crate::gpu::ouvrir_auto(),
        "cpu" => Ok(None),
        "hip" => crate::gpu::ouvrir(BackendKind::Hip, 0).map(Some),
        "cuda" => crate::gpu::ouvrir(BackendKind::Cuda, 0).map(Some),
        autre => Err(Erreur::Config(format!("--backend « {autre} » inconnu (auto|hip|cuda|cpu)"))),
    }
}

/// Options de compilation selon `--options` et la précision, pour le backend effectif.
pub fn options_compil(nom: Option<&str>, precision: Precision, backend: BackendKind) -> Resultat<OptionsCompil> {
    let nom = nom.unwrap_or(match precision {
        Precision::F64 => "ref-k1",
        Precision::F32 => "rapide",
    });
    match nom {
        "ref-k1" => Ok(OptionsCompil::reference_k1()),
        "compat-prod" => Ok(OptionsCompil::compat_production(backend)),
        "rapide" => Ok(OptionsCompil::rapide()),
        autre => Err(Erreur::Config(format!("--options « {autre} » inconnu (ref-k1|compat-prod|rapide)"))),
    }
}

fn mutant(m: &ArgMatches) -> Resultat<bool> {
    match texte(m, "mutant") {
        None => Ok(false),
        Some("metro") => Ok(true),
        Some(autre) => Err(Erreur::Config(format!("--mutant « {autre} » inconnu (metro)"))),
    }
}

/// Paramètres des chaînes à partir des listes β et graines.
pub fn chaines_depuis_listes(betas: &[f64], graines: &[u64], k: Option<usize>, pas: Option<u64>, depart: Depart)
    -> Resultat<Vec<ParamChaine>> {
    let k = k.unwrap_or_else(|| betas.len().max(graines.len()));
    if k == 0 {
        return Err(Erreur::Config("aucune chaîne".into()));
    }
    if betas.len() != 1 && betas.len() != k {
        return Err(Erreur::Config(format!("{} valeurs de β pour {k} chaînes (1 ou {k} attendues)", betas.len())));
    }
    if graines.len() != 1 && graines.len() != k {
        return Err(Erreur::Config(format!("{} graines pour {k} chaînes (1 ou {k} attendues)", graines.len())));
    }
    if graines.len() == 1 && k > 1 && pas.is_none() {
        return Err(Erreur::Config(format!(
            "{k} chaînes avec une seule graine : donner {k} graines ou --graine-pas PAS (graine + i·PAS)"
        )));
    }
    Ok((0..k)
        .map(|i| ParamChaine {
            beta: if betas.len() == 1 { betas[0] } else { betas[i] },
            graine: Graine(if graines.len() == 1 { graines[0].wrapping_add(i as u64 * pas.unwrap_or(0)) } else { graines[i] }),
            depart,
        })
        .collect())
}

/// Construit la configuration d'un run depuis les options de `su2 gen` pour le backend effectif `backend`.
pub fn config_run(m: &ArgMatches, backend: BackendKind) -> Resultat<ConfigRun> {
    let precision: Precision = texte(m, "precision").unwrap_or("f64").parse()?;
    let depart = match texte(m, "start") {
        Some("hot") | None => Depart::Chaud,
        Some("cold") => Depart::Froid,
        Some(autre) => return Err(Erreur::Config(format!("--start « {autre} » inconnu (hot|cold)"))),
    };
    let betas: Vec<f64> = liste(texte(m, "beta").unwrap_or("2.3"), "beta")?;
    let graines: Vec<u64> = liste(texte(m, "graine").unwrap_or("1"), "graine")?;
    let k = match texte(m, "chaines") {
        Some(_) => Some(nombre::<usize>(m, "chaines")?),
        None => None,
    };
    let pas = match texte(m, "graine-pas") {
        Some(_) => Some(nombre::<u64>(m, "graine-pas")?),
        None => None,
    };
    let lot = ConfigLot {
        l: nombre(m, "L")?,
        chaines: chaines_depuis_listes(&betas, &graines, k, pas, depart)?,
        nor: nombre(m, "nor")?,
        precision,
        options: options_compil(texte(m, "options"), precision, backend)?,
        mutant_metro: mutant(m)?,
    };
    let bord = if m.get_flag("bord") {
        let bc = texte(m, "bc").ok_or_else(|| Erreur::Config("--bord exige --bc cold|hot|eq".into()))?;
        Some(ConfigBord {
            bc: bc.to_string(),
            bc_graine: nombre(m, "bc-graine")?,
            jauge: match texte(m, "jauge") {
                Some(_) => Some(nombre(m, "jauge")?),
                None => None,
            },
            pre: nombre(m, "pre")?,
            bloc: nombre(m, "bloc")?,
        })
    } else {
        None
    };
    let outdir = texte(m, "outdir").ok_or_else(|| Erreur::Config("--outdir requis (ou --reprendre DIR)".into()))?;
    let cfg = ConfigRun {
        lot,
        therm: nombre(m, "therm")?,
        nmeas: nombre(m, "nmeas")?,
        skip: nombre(m, "skip")?,
        dump_every: nombre(m, "dump-every")?,
        dump_initial: m.get_flag("dump-initial"),
        format_dump: texte(m, "format").unwrap_or("c16").to_string(),
        zstd: m.get_flag("zstd"),
        reprise_every: nombre(m, "reprise-every")?,
        outdir: PathBuf::from(outdir),
        bord,
    };
    run::valider_run(&cfg)?;
    Ok(cfg)
}

// ------------------------------------------------------------------------------------------
// Exécution
// ------------------------------------------------------------------------------------------

pub fn executer(nom: &str, m: &clap::ArgMatches) -> Resultat<i32> {
    if nom != "su2" {
        return Err(Erreur::Config(format!("sous-commande « {nom} » inconnue pour le module su2")));
    }
    match m.subcommand() {
        Some(("archive", g)) => super::archive_cli::executer_archive(g),
        Some(("gen", g)) => executer_gen(g),
        Some(("test-hb", g)) => executer_test_hb(g),
        Some(("boite", g)) => super::boite_cli::executer_boite(g),
        Some(("boite-obs", g)) => super::boite_cli::executer_boite_obs(g),
        Some(("selftest", _)) => Ok(selftest()),
        _ => Err(Erreur::Config("su2 : sous-commande attendue (gen | archive | test-hb | boite | boite-obs | selftest)".into())),
    }
}

fn executer_gen(g: &ArgMatches) -> Resultat<i32> {
    // M4 : déterminer la torsion EFFECTIVE avant toute ouverture de backend.
    let torsion_cli = texte(g, "torsion").map(Torsion::analyser).transpose()?;
    let (cfg_reprise, opts) = match texte(g, "reprendre") {
        Some(d) => {
            let dir = Path::new(d);
            let opts = run::charger_options(dir)?;
            if let Some(t) = torsion_cli {
                if t != opts.torsion {
                    return Err(Erreur::Config(format!(
                        "--torsion {} diffère de la torsion du run {} ({})",
                        t.texte(), dir.display(), opts.torsion.texte()
                    )));
                }
            }
            (Some(run::charger_config(dir)?), opts)
        }
        None => (None, OptionsRun { torsion: torsion_cli.unwrap_or_default() }),
    };
    let backend_demande = texte(g, "backend").unwrap_or("auto").trim().to_ascii_lowercase();
    if !opts.torsion.est_triviale() {
        // CPU explicite requis : même « auto » est refusé, sans sonder/ouvrir de GPU.
        if backend_demande != "cpu" {
            return Err(Erreur::Config("torsion de 't Hooft : les noyaux GPU ne portent pas la torsion (--backend cpu requis)".into()));
        }
        let precision: Precision = texte(g, "precision").unwrap_or("f64").parse()?;
        if precision != Precision::F64 || cfg_reprise.as_ref().is_some_and(|c| c.lot.precision != Precision::F64) {
            return Err(Erreur::Config("torsion de 't Hooft incompatible avec --precision f32 (CPU F64 requis)".into()));
        }
        if g.get_flag("bord") || cfg_reprise.as_ref().is_some_and(|c| c.bord.is_some()) {
            return Err(Erreur::Config("torsion de 't Hooft incompatible avec --bord".into()));
        }
        if texte(g, "mesures").is_some() {
            return Err(Erreur::Config("torsion de 't Hooft incompatible avec --mesures (mesures en ligne)".into()));
        }
        let beta_zero = match &cfg_reprise {
            Some(cfg) => cfg.lot.chaines.iter().any(|c| c.beta == 0.0),
            None => liste::<f64>(texte(g, "beta").unwrap_or("2.3"), "beta")?.contains(&0.0),
        };
        if beta_zero {
            return Err(Erreur::Config("torsion de 't Hooft : β = 0 interdit (bain thermique historique singulier)".into()));
        }
    }

    let app = ouvrir_backend(&backend_demande)?;
    let backend = app.as_ref().map_or(BackendKind::Cpu, |a| a.backend());
    let arret = match texte(g, "arret-apres") {
        Some(_) => Some(nombre::<u64>(g, "arret-apres")?),
        None => None,
    };
    if texte(g, "fenetre-s").is_some() {
        let f: f64 = nombre(g, "fenetre-s")?;
        if !(f.is_finite() && f > 0.0) {
            return Err(Erreur::Config(format!("--fenetre-s : « {f} » invalide (secondes > 0)")));
        }
        run::fixer_fenetre(f);
    }
    let (mut cfg, reprendre) = match cfg_reprise {
        Some(cfg) => (cfg, true),
        None => (config_run(g, backend)?, false),
    };
    let archive = super::archive_cli::preparer_gen(g, &mut cfg, reprendre)?;
    // 0.1.1 (PUB-01) : à la reprise, une --mesures absente est RECHARGÉE de la spécification enregistrée (comme la torsion) ;
    // une --mesures présente est comparée à celle-ci par le run, avant toute écriture.
    let spec_mesures: Option<String> = match (texte(g, "reprendre"), texte(g, "mesures")) {
        (Some(d), None) => run::charger_spec_mesures(Path::new(d))?.map(|e| {
            eprintln!("su2 gen : reprise : --mesures « {} » rechargée de {}", e.spec, run::NOM_FICHIER_MESURES);
            e.spec
        }),
        // une --mesures ÉQUIVALENTE à la spécification enregistrée est remplacée par celle-ci (noms des sorties inchangés) ;
        // une --mesures différente est refusée par le run
        (Some(d), Some(m)) => match run::charger_spec_mesures(Path::new(d))? {
            Some(e) if run::specs_mesures_equivalentes(m, &e.spec) => Some(e.spec),
            _ => Some(m.to_string()),
        },
        (_, m) => m.map(str::to_string),
    };
    let mut mesures: Vec<Box<dyn MesureEnLigne>> = match spec_mesures.as_deref() {
        Some(spec) => crate::obs::enligne::analyser(spec)?,
        None => Vec::new(),
    };
    let (r, termine) =
        run::executer_avec_mesures(app.as_ref(), &cfg, &opts, &mut mesures, spec_mesures.as_deref(), reprendre, arret)?;
    super::archive_cli::apres_gen(&cfg, archive)?;
    // WP7 : un run terminé qui portait `bloc:3` écrit son manifeste kvu-regen/1 (`parametres.mesures`).
    if termine {
        if let Some(spec) = spec_mesures.as_deref().filter(|s| s.split(',').any(|e| e.trim().starts_with("bloc:"))) {
            let p = crate::rg::en_ligne::ecrire_manifeste_regen(&cfg.outdir, spec)?;
            eprintln!("su2 gen : manifeste kvu-regen/1 écrit : {}", p.display());
        }
    }
    eprintln!(
        "su2 gen : {} — {} pas, {} mesures, {} dumps, plafond {}, {:.2} s",
        if termine { "terminé (COMPLETE)" } else { "interrompu (point de reprise écrit)" },
        r.pas,
        r.mesures,
        r.dumps,
        r.plafond,
        r.secondes
    );
    Ok(0)
}

fn executer_test_hb(m: &ArgMatches) -> Resultat<i32> {
    let alpha: f64 = nombre(m, "alpha")?;
    let n: usize = nombre(m, "n")?;
    let graine = Graine(nombre(m, "graine")?);
    let precision: Precision = texte(m, "precision").unwrap_or("f64").parse()?;
    let app = ouvrir_backend(texte(m, "backend").unwrap_or("auto"))?;
    let backend = app.as_ref().map_or(BackendKind::Cpu, |a| a.backend());
    let mutant = mutant(m)?;
    if alpha.is_nan() || alpha <= 0.0 || n == 0 {
        return Err(Erreur::Config("test-hb : α > 0 et N ≥ 1 requis".into()));
    }
    let petit_lot = |l: usize| -> Resultat<ConfigLot> {
        Ok(ConfigLot {
            l,
            chaines: vec![ParamChaine { beta: 1.0, graine, depart: Depart::Froid }],
            nor: 0,
            precision,
            options: options_compil(texte(m, "options"), precision, backend)?,
            mutant_metro: mutant,
        })
    };
    let (tirages, plafond) = match (&app, precision) {
        (None, Precision::F64) => su2_hb::test_x0_cpu(alpha, n, graine.0, mutant),
        (Some(a), Precision::F64) => super::ref64::Ref64Gpu::nouveau(a, petit_lot(2)?)?.test_x0(alpha, n, graine)?,
        (Some(a), Precision::F32) => super::rapide32::Rapide32Gpu::nouveau(a, petit_lot(4)?)?.test_x0(alpha, n, graine.0)?,
        (None, Precision::F32) => {
            return Err(Erreur::Config("précision f32 : un GPU est requis (le CPU ne calcule qu'en f64)".into()));
        }
    };
    let (mut m1, mut m2, mut mn, mut mx) = (0.0f64, 0.0f64, 1.0f64, -1.0f64);
    for &x in &tirages {
        m1 += x;
        m2 += x * x;
        mn = mn.min(x);
        mx = mx.max(x);
    }
    m1 /= n as f64;
    m2 /= n as f64;
    println!("TESTHB alpha={} n={} m1={} m2={} min={} max={} plafond={}", g17(alpha), n, g17(m1), g17(m2), g17(mn), g17(mx), plafond);
    Ok(if plafond != 0 { 6 } else { 0 })
}

/// Vecteurs d'essai Random123 de Philox4x32-10 : (compteur, clé, résultat).
pub const KAT_PHILOX: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
    ([0, 0, 0, 0], [0, 0], [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]),
    ([0xffffffff; 4], [0xffffffff, 0xffffffff], [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]),
    ([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344], [0xa4093822, 0x299f31d0], [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]),
];

/// Auto-test hôte : mêmes lignes que `su2_hip --selftest`. Rend 0 si tout passe.
pub fn selftest() -> i32 {
    let mut mauvais = 0;
    for (t, (c, k, attendu)) in KAT_PHILOX.iter().enumerate() {
        let r = su2_hb::philox4x32_10(*c, k[0], k[1]);
        let ok = r == *attendu;
        println!(
            "SELFTEST philox KAT {t} : {} ({:08x} {:08x} {:08x} {:08x})",
            if ok { "PASS" } else { "FAIL" },
            r[0],
            r[1],
            r[2],
            r[3]
        );
        mauvais += i32::from(!ok);
    }
    let (lo, hi) = (su2_hb::u01(0, 0), su2_hb::u01(0xffff_ffff, 0xffff_ffff));
    let ok = lo > 0.0 && hi < 1.0;
    println!("SELFTEST u01 bornes : {} ({lo:.3e}, 1-{:.3e})", if ok { "PASS" } else { "FAIL" }, 1.0 - hi);
    mauvais += i32::from(!ok);
    println!("SELFTEST {}", if mauvais == 0 { "OK" } else { "ECHEC" });
    i32::from(mauvais != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyser(args: &[&str]) -> Resultat<(ConfigRun, bool)> {
        let mut argv = vec!["kvu", "su2"];
        argv.extend_from_slice(args);
        let m = Command::new("kvu").subcommands(commandes()).try_get_matches_from(argv).map_err(|e| Erreur::Config(e.to_string()))?;
        let (_, su2) = m.subcommand().ok_or_else(|| Erreur::Config("su2".into()))?;
        let (_, g) = su2.subcommand().ok_or_else(|| Erreur::Config("gen".into()))?;
        let cfg = config_run(g, BackendKind::Cpu)?;
        Ok((cfg, g.get_flag("bord")))
    }

    #[test]
    fn la_ligne_de_commande_de_su2_hip_est_acceptee() {
        let (c, bord) = analyser(&[
            "gen", "--L", "6", "--beta", "2.8", "--seed", "27092900006", "--start", "hot", "--therm", "0", "--nmeas", "20",
            "--skip", "1", "--nor", "4", "--dump-every", "1", "--dump-initial", "--source-hash", "ed1e0fce", "--outdir", "/tmp/x",
        ])
        .unwrap();
        assert!(!bord);
        assert_eq!(c.lot.l, 6);
        assert_eq!(c.lot.chaines.len(), 1);
        assert_eq!(c.lot.chaines[0].graine, Graine(27092900006));
        assert_eq!(c.lot.chaines[0].depart, Depart::Chaud);
        assert!(c.dump_initial && c.dump_every == 1 && c.skip == 1 && c.lot.nor == 4);
        assert_eq!(c.lot.options, OptionsCompil::reference_k1(), "défaut F64 : ref-k1");
    }

    #[test]
    fn la_ligne_de_commande_de_su2_bord_est_acceptee() {
        let (c, bord) = analyser(&[
            "gen", "--bord", "--L", "8", "--beta", "2.5", "--seed", "28092800002", "--bc", "eq", "--bc-seed", "28092800004",
            "--jauge", "28092800005", "--pre", "200", "--therm", "200", "--nmeas", "1000", "--bloc", "100", "--nor", "4",
            "--outdir", "/tmp/y",
        ])
        .unwrap();
        assert!(bord);
        let b = c.bord.unwrap();
        assert_eq!((b.bc.as_str(), b.bc_graine, b.jauge, b.pre, b.bloc), ("eq", 28092800004, Some(28092800005), 200, 100));
    }

    #[test]
    fn lots_listes_et_graine_pas() {
        let (c, _) = analyser(&["gen", "--L", "4", "--beta", "2.3,2.8,3.5", "--graine", "10,20,30", "--outdir", "/tmp/z"]).unwrap();
        assert_eq!(c.lot.chaines.iter().map(|x| x.beta).collect::<Vec<_>>(), vec![2.3, 2.8, 3.5]);
        assert_eq!(c.lot.chaines.iter().map(|x| x.graine.0).collect::<Vec<_>>(), vec![10, 20, 30]);
        let (c, _) = analyser(&["gen", "--L", "4", "--beta", "2.3", "--graine", "100", "--chaines", "3", "--graine-pas", "7", "--outdir", "/tmp/z"]).unwrap();
        assert_eq!(c.lot.chaines.iter().map(|x| x.graine.0).collect::<Vec<_>>(), vec![100, 107, 114]);
        // une seule graine pour plusieurs chaînes sans pas : refus
        assert!(analyser(&["gen", "--L", "4", "--beta", "2.3,2.8", "--graine", "100", "--outdir", "/tmp/z"]).is_err());
        // longueurs incohérentes
        assert!(analyser(&["gen", "--L", "4", "--beta", "2.3,2.8", "--graine", "1,2,3", "--outdir", "/tmp/z"]).is_err());
    }

    #[test]
    fn options_numeriques_par_defaut_et_explicites() {
        let (c, _) = analyser(&["gen", "--L", "4", "--options", "compat-prod", "--outdir", "/tmp/z"]).unwrap();
        assert_eq!(c.lot.options, OptionsCompil::compat_production(BackendKind::Cpu));
        assert!(analyser(&["gen", "--L", "4", "--options", "bizarre", "--outdir", "/tmp/z"]).is_err());
        assert_eq!(options_compil(None, Precision::F32, BackendKind::Hip).unwrap(), OptionsCompil::rapide());
    }

    #[test]
    fn erreurs_d_usage() {
        assert!(analyser(&["gen", "--L", "5", "--outdir", "/tmp/z"]).is_err(), "L impair");
        assert!(analyser(&["gen", "--L", "4"]).is_err(), "outdir manquant");
        assert!(analyser(&["gen", "--L", "4", "--beta", "abc", "--outdir", "/tmp/z"]).is_err());
        assert!(analyser(&["gen", "--L", "4", "--bord", "--outdir", "/tmp/z"]).is_err(), "bord sans --bc");
        assert!(analyser(&["gen", "--L", "4", "--format", "zip", "--outdir", "/tmp/z"]).is_err());
        assert!(analyser(&["gen", "--L", "4", "--start", "tiede", "--outdir", "/tmp/z"]).is_err());
    }

    #[test]
    fn selftest_passe() {
        assert_eq!(selftest(), 0);
    }

    #[test]
    fn backend_cpu_et_inconnu() {
        assert!(ouvrir_backend("cpu").unwrap().is_none());
        assert!(matches!(ouvrir_backend("vulkan"), Err(Erreur::Config(_))));
    }
}
