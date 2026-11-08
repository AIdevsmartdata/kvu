// SPDX-License-Identifier: Apache-2.0
//! Sous-commandes `su2 boite` et `su2 boite-obs` (boîtes de Dirichlet SU(2) à bord fixé). Propriétaire : WP-DIR.
//!
//! ```text
//! su2 boite --ell L --beta B  (--bords n1,n2,… [--repliques R] [--graine G] [--graine-pas P] | --chaines n1:g1,n2:g2,…)
//!           [--bord-npy FICHIER …] --therm N --nmeas N [--skip 1] [--nor 2] [--options ref-k1|compat-prod] [--precision f64]
//!           [--backend auto|hip|cuda|cpu] [--profil-tous 10] [--therm-source 2000] [--dump-final]
//!           [--mutant masque-ignore|masque-ignore-bain|masque-ignore-surrelax] --outdir DIR
//! su2 boite-obs --cfg X.npy --ell L [--backend auto|hip|cuda|cpu]      (observables d'une configuration NPY de réseau de calcul)
//! ```
//! Bords : cold, gauge, z2full, z2_01, flux, haar, therm, et les bords lus `npy:<h8>` (`--bord-npy`, répétable : 8 premiers
//! hexadécimaux du sha256 du fichier ; `npy` seul s'il n'y a qu'un fichier ; docs/BORD_NPY.md). Avec `--bords`, la chaîne
//! c = indice_bord·R + réplique porte la graine G + c·P (défauts G = 1, P = 1, R = 1). Un lot = un (ℓ, β) ; les chaînes sont
//! numérotées dans l'ordre donné ; `--chaines` coupe chaque élément au DERNIER « : » (`npy:1a2b3c4d:7`).

use super::boite::GeometrieBoite;
use super::boite_npy::{self, BordNpy};
use super::boite_run::{self, ChaineBoite, ConfigBoite};
use super::AppMasque;
use crate::commun::{BackendKind, Erreur, Precision, Resultat};
use crate::heritage::su2_hb::g17;
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::path::PathBuf;

fn arg(nom: &'static str, aide: &'static str) -> Arg {
    Arg::new(nom).long(nom).help(aide).num_args(1)
}

/// `su2 boite` : définition clap.
pub fn boite() -> Command {
    Command::new("boite")
        .about("Boîtes de Dirichlet ℓ⁴ SU(2) à bord fixé : lot de chaînes (bord, graine), P_c, W₂ et profil par couche")
        .arg(arg("ell", "côté ℓ de la boîte (pair ≥ 2)").required(true))
        .arg(arg("beta", "β = 4/g²").required(true))
        .arg(arg("bords", "liste de bords : cold,gauge,z2full,z2_01,flux,haar,therm (avec --repliques)"))
        .arg(arg("repliques", "répliques par bord (avec --bords)").default_value("1"))
        .arg(arg("graine", "graine de base (avec --bords) : chaîne c = G + c·PAS").alias("seed").default_value("1"))
        .arg(arg("graine-pas", "pas des graines (avec --bords)").default_value("1"))
        .arg(arg("chaines", "chaînes explicites BORD:GRAINE,BORD:GRAINE,… (exclusif de --bords)"))
        .arg(
            Arg::new("bord-npy")
                .long("bord-npy")
                .num_args(1)
                .action(ArgAction::Append)
                .value_name("FICHIER")
                .help(
                    "bord arbitraire lu dans un fichier NPY (q64 ou c16, réseau de calcul de côté ℓ + 2 ; seuls les liens fixés sont lus) : \
                     nom de bord npy:<8 premiers hexadécimaux du sha256>, ou npy s'il est seul ; répétable (docs/BORD_NPY.md)",
                ),
        )
        .arg(arg("therm", "pas de thermalisation (sans mesure)").default_value("1000"))
        .arg(arg("nmeas", "nombre de mesures").default_value("1000"))
        .arg(arg("skip", "pas entre deux mesures").default_value("1"))
        .arg(arg("nor", "surrelaxations par pas").default_value("2"))
        .arg(arg("options", "ref-k1 (défaut : FMA interdite, égalité avec la référence hôte) | compat-prod").default_value("ref-k1"))
        .arg(arg("precision", "f64 seulement (le masque n'existe pas en f32)").default_value("f64"))
        .arg(arg("backend", "auto | hip | cuda | cpu").default_value("auto"))
        .arg(arg("profil-tous", "profil par couche tous les N mesures (0 = jamais)").default_value("10"))
        .arg(arg("therm-source", "pas de thermalisation des sources thermiques").default_value("2000"))
        .arg(Arg::new("dump-final").long("dump-final").action(ArgAction::SetTrue).help("écrit cNNN/config_finale.npy (q64, réseau de calcul)"))
        .arg(Arg::new("mutant").long("mutant").hide(true).num_args(1).help("contrôle négatif : masque-ignore | masque-ignore-bain | masque-ignore-surrelax"))
        .arg(arg("outdir", "répertoire de sortie (NEUF)").required(true))
}

/// `su2 boite-obs` : définition clap.
pub fn boite_obs() -> Command {
    Command::new("boite-obs")
        .about("Observables (P_c, W₂, profil) d'une configuration NPY de réseau de calcul d'une boîte ℓ⁴ (hôte ou GPU)")
        .arg(arg("cfg", "fichier NPY (q64 ou c16) de côté ℓ + 2").required(true))
        .arg(arg("ell", "côté ℓ de la boîte").required(true))
        .arg(arg("backend", "auto | hip | cuda | cpu").default_value("cpu"))
}

fn texte<'a>(m: &'a ArgMatches, nom: &str) -> Option<&'a str> {
    m.get_one::<String>(nom).map(|s| s.as_str())
}

fn nombre<T: std::str::FromStr>(m: &ArgMatches, nom: &str) -> Resultat<T> {
    let t = texte(m, nom).ok_or_else(|| Erreur::Config(format!("--{nom} requis")))?;
    t.parse::<T>().map_err(|_| Erreur::Config(format!("--{nom} : « {t} » n'est pas un nombre valide")))
}

/// `BORD:GRAINE,BORD:GRAINE,…` → chaînes (pure, testée). Chaque élément est coupé au DERNIER « : » (bord lu `npy:<h8>:GRAINE`).
pub fn analyser_chaines(spec: &str) -> Resultat<Vec<ChaineBoite>> {
    spec.split(',')
        .map(|t| {
            let t = t.trim();
            let (b, g) = t
                .rsplit_once(':')
                .ok_or_else(|| Erreur::Config(format!("--chaines : « {t} » n'est pas de la forme BORD:GRAINE")))?;
            let graine: u64 =
                g.trim().parse().map_err(|_| Erreur::Config(format!("--chaines : graine « {g} » invalide (entier ≥ 0)")))?;
            Ok(ChaineBoite { bord: b.trim().to_string(), graine })
        })
        .collect()
}

/// `--bords` × `--repliques` → chaînes : c = indice_bord·R + réplique, graine = G + c·P (pure, testée).
pub fn chaines_depuis_bords(bords: &str, repliques: usize, graine: u64, pas: u64) -> Resultat<Vec<ChaineBoite>> {
    if repliques == 0 {
        return Err(Erreur::Config("--repliques ≥ 1 requis".into()));
    }
    let noms: Vec<&str> = bords.split(',').map(|b| b.trim()).collect();
    if noms.iter().any(|b| b.is_empty()) {
        return Err(Erreur::Config("--bords : nom de bord vide".into()));
    }
    let mut v = Vec::new();
    for (i, b) in noms.iter().enumerate() {
        for r in 0..repliques {
            let c = (i * repliques + r) as u64;
            v.push(ChaineBoite { bord: (*b).to_string(), graine: graine.wrapping_add(c.wrapping_mul(pas)) });
        }
    }
    Ok(v)
}

fn mutant(m: &ArgMatches) -> Resultat<AppMasque> {
    match texte(m, "mutant") {
        None => Ok(AppMasque::Tout),
        Some("masque-ignore") => Ok(AppMasque::IgnoreTout),
        Some("masque-ignore-bain") => Ok(AppMasque::IgnoreBain),
        Some("masque-ignore-surrelax") => Ok(AppMasque::IgnoreSurrelax),
        Some(autre) => Err(Erreur::Config(format!(
            "--mutant « {autre} » inconnu (masque-ignore|masque-ignore-bain|masque-ignore-surrelax)"
        ))),
    }
}

/// Configuration d'un lot depuis les options de `su2 boite` (les bords lus sont validés puis oubliés : voir
/// [`config_et_bords_depuis_args`]).
pub fn config_depuis_args(m: &ArgMatches) -> Resultat<ConfigBoite> {
    config_et_bords_depuis_args(m).map(|(c, _)| c)
}

/// Configuration d'un lot et ses bords lus (`--bord-npy`, dans l'ordre des options) : fichiers lus et validés, nom `npy` nu
/// résolu, puis [`boite_run::valider_boite_bords`]. Tout refus a lieu avant la création du répertoire de sortie.
pub fn config_et_bords_depuis_args(m: &ArgMatches) -> Resultat<(ConfigBoite, Vec<BordNpy>)> {
    let precision: Precision = texte(m, "precision").unwrap_or("f64").parse()?;
    if precision != Precision::F64 {
        return Err(Erreur::Config("boîte : --precision f64 requise (le masque de liens gelés n'existe pas en f32)".into()));
    }
    let mut chaines = match (texte(m, "chaines"), texte(m, "bords")) {
        (Some(_), Some(_)) => return Err(Erreur::Config("--chaines et --bords s'excluent".into())),
        (None, None) => return Err(Erreur::Config("--chaines ou --bords requis".into())),
        (Some(s), None) => analyser_chaines(s)?,
        (None, Some(b)) => chaines_depuis_bords(b, nombre(m, "repliques")?, nombre(m, "graine")?, nombre(m, "graine-pas")?)?,
    };
    let ell: usize = nombre(m, "ell")?;
    let fichiers: Vec<PathBuf> = m.get_many::<String>("bord-npy").map(|v| v.map(PathBuf::from).collect()).unwrap_or_default();
    let bords = if fichiers.is_empty() {
        Vec::new()
    } else {
        let g = GeometrieBoite::nouvelle(ell)?;
        fichiers.iter().map(|f| boite_npy::lire_bord_npy(f, &g)).collect::<Resultat<Vec<_>>>()?
    };
    boite_npy::resoudre_noms(&mut chaines, &bords)?;
    let cfg = ConfigBoite {
        ell,
        beta: nombre(m, "beta")?,
        chaines,
        therm: nombre(m, "therm")?,
        nmeas: nombre(m, "nmeas")?,
        skip: nombre(m, "skip")?,
        nor: nombre(m, "nor")?,
        options: texte(m, "options").unwrap_or("ref-k1").to_string(),
        profil_tous: nombre(m, "profil-tous")?,
        therm_source: nombre(m, "therm-source")?,
        dump_final: m.get_flag("dump-final"),
        mutant: mutant(m)?,
        outdir: PathBuf::from(texte(m, "outdir").ok_or_else(|| Erreur::Config("--outdir requis".into()))?),
    };
    boite_run::valider_boite_bords(&cfg, &bords)?;
    Ok((cfg, bords))
}

/// `su2 boite` : exécution.
pub fn executer_boite(m: &ArgMatches) -> Resultat<i32> {
    let (cfg, bords) = config_et_bords_depuis_args(m)?;
    let app = super::cli::ouvrir_backend(texte(m, "backend").unwrap_or("auto"))?;
    let argv: Vec<String> = std::env::args().collect();
    let r = boite_run::executer_boite_bords(app.as_ref(), &cfg, &bords, &argv)?;
    eprintln!(
        "su2 boite : terminé (COMPLETE) — ℓ = {}, {} chaîne(s), {} pas, {:.2} s ({:.3} ms par pas et par chaîne), bord intact : {}",
        cfg.ell, r.chaines, r.pas, r.secondes, r.ms_par_pas_et_chaine, r.bord_intact
    );
    Ok(0)
}

/// `su2 boite-obs` : exécution (une ligne JSON, nombres au format %.17g).
pub fn executer_boite_obs(m: &ArgMatches) -> Resultat<i32> {
    let ell: usize = nombre(m, "ell")?;
    let chemin = PathBuf::from(texte(m, "cfg").ok_or_else(|| Erreur::Config("--cfg requis".into()))?);
    let app = super::cli::ouvrir_backend(texte(m, "backend").unwrap_or("cpu"))?;
    let backend = app.as_ref().map_or(BackendKind::Cpu, |a| a.backend());
    // un GPU n'est jamais utilisé hors du verrou de sa machine (déjà tenu par un ancêtre : flock externe ⇒ pas de second verrou)
    let _verrou = match &app {
        Some(_) => Some(super::run::prendre_verrou()?),
        None => None,
    };
    let o = boite_run::observables_fichier(app.as_ref(), &chemin, ell)?;
    let profil: Vec<String> = o.profil.iter().map(|v| g17(*v)).collect();
    let w2 = if o.w2.is_nan() { "null".to_string() } else { g17(o.w2) };
    println!(
        "{{\"ell\": {ell}, \"backend\": \"{}\", \"pc\": {}, \"w2\": {w2}, \"profil\": [{}]}}",
        backend.nom(),
        g17(o.pc),
        profil.join(", ")
    );
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyser(args: &[&str]) -> Resultat<ConfigBoite> {
        let mut argv = vec!["kvu", "boite"];
        argv.extend_from_slice(args);
        let m = Command::new("kvu")
            .subcommand(boite())
            .try_get_matches_from(argv)
            .map_err(|e| Erreur::Config(e.to_string()))?;
        let (_, b) = m.subcommand().ok_or_else(|| Erreur::Config("boite".into()))?;
        config_depuis_args(b)
    }

    #[test]
    fn chaines_explicites_et_bords() {
        let c = analyser(&["--ell", "8", "--beta", "2.5", "--chaines", "cold:5,z2full:9", "--outdir", "/tmp/x"]).unwrap();
        assert_eq!(c.chaines, vec![ChaineBoite { bord: "cold".into(), graine: 5 }, ChaineBoite { bord: "z2full".into(), graine: 9 }]);
        let c = analyser(&["--ell", "4", "--beta", "2.5", "--bords", "cold,haar", "--repliques", "3", "--graine", "100", "--graine-pas", "7", "--outdir", "/tmp/x"])
            .unwrap();
        let g: Vec<(String, u64)> = c.chaines.iter().map(|x| (x.bord.clone(), x.graine)).collect();
        assert_eq!(
            g,
            vec![
                ("cold".to_string(), 100),
                ("cold".to_string(), 107),
                ("cold".to_string(), 114),
                ("haar".to_string(), 121),
                ("haar".to_string(), 128),
                ("haar".to_string(), 135)
            ]
        );
        assert_eq!(c.options, "ref-k1");
        assert_eq!((c.therm, c.nmeas, c.skip, c.nor, c.profil_tous, c.therm_source), (1000, 1000, 1, 2, 10, 2000));
    }

    #[test]
    fn erreurs_d_usage() {
        let ok = ["--beta", "2.5", "--outdir", "/tmp/x"];
        let avec = |extra: &[&str]| -> Resultat<ConfigBoite> {
            let mut v = extra.to_vec();
            v.extend_from_slice(&ok);
            analyser(&v)
        };
        assert!(avec(&["--ell", "5", "--chaines", "cold:1"]).is_err(), "ℓ impair");
        assert!(avec(&["--ell", "4"]).is_err(), "ni --chaines ni --bords");
        assert!(avec(&["--ell", "4", "--chaines", "cold:1", "--bords", "cold"]).is_err(), "les deux");
        assert!(avec(&["--ell", "4", "--chaines", "tiede:1"]).is_err(), "bord inconnu");
        assert!(avec(&["--ell", "4", "--chaines", "cold"]).is_err(), "graine absente");
        assert!(avec(&["--ell", "4", "--chaines", "cold:x"]).is_err(), "graine non entière");
        assert!(avec(&["--ell", "4", "--chaines", "cold:1", "--precision", "f32"]).is_err(), "f32 refusé");
        assert!(avec(&["--ell", "4", "--chaines", "cold:1", "--options", "rapide"]).is_err());
        assert!(avec(&["--ell", "4", "--chaines", "cold:1", "--mutant", "bizarre"]).is_err());
        assert!(avec(&["--ell", "4", "--chaines", "cold:1", "--nmeas", "0"]).is_err());
        assert!(avec(&["--ell", "4", "--bords", "cold", "--repliques", "0"]).is_err());
        assert!(avec(&["--ell", "4", "--chaines", "cold:1", "--mutant", "masque-ignore-bain"]).is_ok());
    }

    #[test]
    fn mutants_connus() {
        for (nom, attendu) in [
            ("masque-ignore", AppMasque::IgnoreTout),
            ("masque-ignore-bain", AppMasque::IgnoreBain),
            ("masque-ignore-surrelax", AppMasque::IgnoreSurrelax),
        ] {
            let c = analyser(&["--ell", "4", "--beta", "2.5", "--chaines", "cold:1", "--outdir", "/tmp/x", "--mutant", nom]).unwrap();
            assert_eq!(c.mutant, attendu);
            assert_eq!(c.mutant.nom(), nom);
        }
    }
}
