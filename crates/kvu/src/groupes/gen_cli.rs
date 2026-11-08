//! `kvu gen --groupe sp4` : générateur d'ensembles de groupe générique (CPU), Metropolis + sur-relaxation (pas v4.0/v4.4). Propriétaire : WP16.
//!
//! ```text
//! kvu gen --groupe sp4 --L 8 --beta 8 --graine S [--start hot|cold] [--therm N] [--nmeas N] [--skip K] [--dump-every N]
//!         --outdir DIR [--certificats DIR]
//! ```
//! `su2 gen` reste le chemin SU(2) bit-identique à (A). Le binaire refuse un groupe sans certificat `cpu` vert (G-CERT, code 2).
//! Sortie (DIR, NEUF) : `measurements.csv` (indice, plaquette ; l'indice 0 est l'état après thermalisation), `manifest.json`, `COMPLETE`,
//! et `cfg_NNNNNN.npy` (complex128, C-order (μ, x₀…x₃, i, j), liens li = μV + s) tous les `--dump-every` indices (0 = jamais).

use super::cli::{exiger_certificat, DOSSIER_CERT_DEFAUT};
use super::generique::{Mat, Reseau};
use super::sp4::Sp4;
use crate::commun::{Depart, Erreur, Resultat};
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn commande() -> Command {
    let a = |n: &'static str, h: &'static str| Arg::new(n).long(n).help(h).num_args(1);
    Command::new("gen")
        .about("Générateur d'ensembles d'un groupe générique (sp4) : Metropolis + sur-relaxation, CPU ; refuse un groupe sans certificat")
        .arg(a("groupe", "sp4").required(true))
        .arg(a("L", "côté du réseau (pair)").default_value("8"))
        .arg(a("beta", "β (action S = β Σ_p (1 − Re χ_F(U_p)/d_F))").default_value("8"))
        .arg(a("graine", "graine 64 bits").alias("seed").default_value("1"))
        .arg(a("start", "hot | cold").default_value("hot"))
        .arg(a("therm", "pas de thermalisation (adaptation de ε par blocs de 50, puis gelé)").default_value("1000"))
        .arg(a("nmeas", "nombre de mesures").default_value("100"))
        .arg(a("skip", "pas entre deux mesures").default_value("1"))
        .arg(a("dump-every", "un dump cfg_NNNNNN.npy tous les N indices de mesure (0 = jamais)").default_value("0"))
        .arg(a("outdir", "répertoire de sortie (NEUF)").required(true))
        .arg(a("certificats", "dossier des certificats").default_value(DOSSIER_CERT_DEFAUT))
        .arg(Arg::new("dump-initial").long("dump-initial").action(ArgAction::SetTrue).help("écrit cfg_000000.npy (état après thermalisation)"))
}

fn nombre<T: std::str::FromStr>(m: &ArgMatches, nom: &str) -> Resultat<T> {
    let t = m.get_one::<String>(nom).ok_or_else(|| Erreur::Config(format!("--{nom} requis")))?;
    t.parse::<T>().map_err(|_| Erreur::Config(format!("--{nom} : « {t} » n'est pas un nombre valide")))
}

fn npy_header(l: usize, d: usize) -> Vec<u8> {
    let mut hd = format!("{{'descr': '<c16', 'fortran_order': False, 'shape': (4, {l}, {l}, {l}, {l}, {d}, {d}), }}");
    let tot = 10 + hd.len() + 1;
    hd.push_str(&" ".repeat((64 - tot % 64) % 64));
    hd.push('\n');
    let mut out = vec![0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0];
    out.extend_from_slice(&(hd.len() as u16).to_le_bytes());
    out.extend_from_slice(hd.as_bytes());
    out
}

/// Écrit le champ en NPY complex128 via un fichier .part renommé.
pub fn ecrire_npy<const D: usize>(chemin: &Path, l: usize, liens: &[Mat<D>]) -> std::io::Result<()> {
    let tmp = chemin.with_extension("npy.part");
    {
        let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp)?);
        f.write_all(&npy_header(l, D))?;
        for m in liens {
            for i in 0..D {
                for j in 0..D {
                    f.write_all(&m.re[i][j].to_le_bytes())?;
                    f.write_all(&m.im[i][j].to_le_bytes())?;
                }
            }
        }
        f.flush()?;
        f.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    }
    std::fs::rename(&tmp, chemin)
}

pub fn executer(m: &ArgMatches) -> Resultat<i32> {
    let nom = m.get_one::<String>("groupe").cloned().unwrap_or_default();
    if nom != "sp4" {
        return Err(Erreur::Config(format!("groupe « {nom} » refusé : pas de noyau générique au registre (sp4 seulement ; g2 : lot WP18) ; G-CERT")));
    }
    let certs = PathBuf::from(m.get_one::<String>("certificats").cloned().unwrap_or_else(|| DOSSIER_CERT_DEFAUT.into()));
    let cert = exiger_certificat(&nom, &certs)?;
    let l: usize = nombre(m, "L")?;
    let beta: f64 = nombre(m, "beta")?;
    let graine: u64 = nombre(m, "graine")?;
    let therm: u64 = nombre(m, "therm")?;
    let nmeas: u64 = nombre(m, "nmeas")?;
    let skip: u64 = nombre(m, "skip")?;
    let dump: u64 = nombre(m, "dump-every")?;
    if skip == 0 {
        return Err(Erreur::Config("--skip doit être ≥ 1".into()));
    }
    let depart = match m.get_one::<String>("start").map(|s| s.as_str()) {
        Some("hot") => Depart::Chaud,
        Some("cold") => Depart::Froid,
        autre => return Err(Erreur::Config(format!("--start « {autre:?} » : hot | cold"))),
    };
    let out = PathBuf::from(m.get_one::<String>("outdir").cloned().unwrap_or_default());
    if out.exists() {
        return Err(Erreur::Config(format!("--outdir {} existe déjà (répertoire NEUF exigé)", out.display())));
    }
    std::fs::create_dir_all(&out).map_err(|e| Erreur::io("création du répertoire de sortie", e))?;
    let g = Sp4::nouveau();
    let mut r = Reseau::<4>::nouveau(&g, l, beta, graine, depart);
    let t0 = std::time::Instant::now();
    r.thermaliser(&g, therm);
    let eps = r.eps;
    let mut csv = String::from("indice,plaquette\n");
    csv.push_str(&format!("0,{:.17e}\n", r.plaquette()));
    if m.get_flag("dump-initial") {
        ecrire_npy(&out.join("cfg_000000.npy"), l, &r.liens).map_err(|e| Erreur::io("dump initial", e))?;
    }
    for i in 1..=nmeas {
        for _ in 0..skip {
            r.pas(&g);
        }
        csv.push_str(&format!("{i},{:.17e}\n", r.plaquette()));
        if dump > 0 && i % dump == 0 {
            ecrire_npy(&out.join(format!("cfg_{i:06}.npy")), l, &r.liens).map_err(|e| Erreur::io("dump", e))?;
        }
    }
    std::fs::write(out.join("measurements.csv"), csv).map_err(|e| Erreur::io("measurements.csv", e))?;
    let cpu = cert.certificats.iter().find(|e| e.backend == "cpu");
    let manifeste = serde_json::json!({
        "schema": "kvu-gen/1", "groupe": nom, "L": l, "beta": beta, "graine": graine,
        "depart": if depart == Depart::Chaud { "hot" } else { "cold" },
        "therm": therm, "nmeas": nmeas, "skip": skip, "dump_every": dump, "eps_gele": eps,
        "pas_total": r.pas_faits, "secondes": t0.elapsed().as_secs_f64(),
        "version": crate::cli::version_texte().replace('\n', " | "),
        "certificat_cpu": cpu.map(|e| serde_json::json!({"horodatage": e.horodatage, "binaire_sha256": e.binaire_sha256})),
    });
    std::fs::write(out.join("manifest.json"), serde_json::to_string_pretty(&manifeste).map_err(|e| Erreur::Format(e.to_string()))? + "\n").map_err(|e| Erreur::io("manifest.json", e))?;
    std::fs::write(out.join("COMPLETE"), "").map_err(|e| Erreur::io("COMPLETE", e))?;
    println!("INFO gen {nom} : L = {l}, β = {beta}, graine {graine}, {nmeas} mesures, ε gelé {eps:.6}, {:.1} s -> {}", t0.elapsed().as_secs_f64(), out.display());
    Ok(0)
}
