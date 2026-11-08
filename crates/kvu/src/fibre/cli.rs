//! Sous-commande `fibre` : `fibre mesurer --cfg X.npy [--libre] [--backend auto|hip|cuda|cpu] --out DIR`.
//!
//! Écrit `F.npy` (nF), `D.npy` (nF, nF) et, hors mode `--libre`, `blk.npy` (6, (L/2)⁴, float32), plus le manifeste
//! de provenance (`manifest.json`, COMPLETE en dernier) dans un répertoire NEUF ; garde SSD et admission mémoire
//! AVANT toute écriture. Sur
//! GPU, le verrou de la machine est pris (sauf s'il est déjà tenu par un ancêtre) et le GPU est rendu à
//! la sortie. Le backend `auto` suit `machine::choisir_backend`.

use super::npy::{ecrire_f32, ecrire_f64, lire_champ};
use super::{poids_fourier, Mesureur, PoidsFourier};
use crate::commun::{BackendKind, Erreur, Resultat};
use crate::cond::GeometrieCond;
use crate::heritage::manifest::sha256_fichier;
use crate::machine::admission::{admettre, octets_champ, BesoinMemoire};
use crate::stockage::{GardeSsd, Manifeste};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

pub fn commandes() -> Vec<clap::Command> {
    vec![clap::Command::new("fibre")
        .about("Mesure de fibre (F, D, blk) d'une configuration")
        .subcommand_required(true)
        .subcommand(crate::fibre_contrainte::cli::commande())
        .subcommand(
            clap::Command::new("mesurer")
                .about("Mesure F (54), D (54×54) et blk d'une configuration SU(2) (NPY c16 ou q64)")
                .arg(
                    clap::Arg::new("cfg")
                        .long("cfg")
                        .required(true)
                        .value_name("X.npy")
                        .help("configuration"),
                )
                .arg(
                    clap::Arg::new("libre")
                        .long("libre")
                        .action(clap::ArgAction::SetTrue)
                        .help("sans conditionnement : tous les liens libres (témoin Haar G10) ; pas de blk"),
                )
                .arg(
                    clap::Arg::new("backend")
                        .long("backend")
                        .default_value("auto")
                        .value_parser(["auto", "hip", "cuda", "cpu"])
                        .help("auto = backend de la machine ; cpu = référence Rust"),
                )
                .arg(
                    clap::Arg::new("out")
                        .long("out")
                        .required(true)
                        .value_name("DIR")
                        .help("répertoire de sortie"),
                ),
        )]
}

/// Choix du backend : `cpu` ⇒ None ; `auto` ⇒ celui de la machine ; sinon forcé (erreur si absent).
fn choisir(nom: &str) -> Resultat<BackendKind> {
    match nom {
        "auto" => crate::machine::choisir_backend(&crate::machine::detecter()),
        autre => autre.parse(),
    }
}

pub fn executer(nom: &str, m: &clap::ArgMatches) -> Resultat<i32> {
    if nom != "fibre" {
        return Err(Erreur::Config(format!("fibre : sous-commande inconnue « {nom} »")));
    }
    match m.subcommand() {
        Some(("echantillonner", sm)) => crate::fibre_contrainte::cli::executer(sm),
        Some(("mesurer", sm)) => {
            let cfg = PathBuf::from(sm.get_one::<String>("cfg").expect("requis"));
            let out = PathBuf::from(sm.get_one::<String>("out").expect("requis"));
            let libre = sm.get_flag("libre");
            let backend = choisir(sm.get_one::<String>("backend").expect("défaut"))?;
            mesurer(&cfg, &out, libre, backend)
        }
        _ => Err(Erreur::Config("fibre : sous-commande attendue (mesurer|echantillonner)".into())),
    }
}

/// Mémoire estimée d'une mesure à côté L : champ hôte (et tampon de décodage), plaquettes et gradients
/// (78 réels par site) sur l'hôte en CPU, ou sur le GPU avec le champ téléversé.
pub fn besoin_memoire(l: usize, gpu: bool) -> BesoinMemoire {
    let v = (l as u64).pow(4);
    let champ = octets_champ(l, 8);
    let plaquettes = 78 * 8 * v;
    BesoinMemoire {
        hote: 2 * champ + if gpu { 0 } else { plaquettes },
        gpu: if gpu { champ + plaquettes + (1 << 20) } else { 0 },
    }
}

/// Octets écrits : F (nF), D (nF²), blk (6 (L/2)⁴ float32), en-têtes.
pub fn projection_octets(l: usize, nf: usize, libre: bool) -> u64 {
    let blk = if libre { 0 } else { 4 * 6 * (l as u64 / 2).pow(4) };
    (8 * nf + 8 * nf * nf) as u64 + blk + 3 * 128
}

fn mesurer(cfg: &std::path::Path, out: &std::path::Path, libre: bool, backend: BackendKind) -> Resultat<i32> {
    // Avant tout octet écrit et avant le GPU : admission mémoire, puis manifeste (répertoire NEUF, garde SSD avec
    // la projection : refus rc 7 sans rien créer).
    let (_, l) = crate::stockage::npy::inspecter_champ(cfg)?;
    let gpu_unifie = backend == BackendKind::Hip
        && crate::machine::detecter()
            .gpus
            .iter()
            .any(|g| g.backend == BackendKind::Hip && crate::machine::detection::arch_memoire_unifiee(&g.arch));
    admettre(
        &format!("fibre mesurer L={l} ({})", backend.nom()),
        besoin_memoire(l, backend != BackendKind::Cpu),
        gpu_unifie,
    )?;
    let w = poids_fourier(l, 1);
    let argv: Vec<String> = std::env::args().collect();
    let mut man = Manifeste::ouvrir_avec_garde(
        out,
        "kvu fibre mesurer",
        &argv,
        &GardeSsd::machine(),
        projection_octets(l, w.nf, libre),
    )?;
    match mesurer_dans(cfg, out, libre, backend, &w, &mut man) {
        Ok(()) => {
            man.terminer(0)?;
            Ok(crate::commun::code::OK)
        }
        Err(e) => {
            let _ = man.echouer(e.code_sortie(), &e.to_string());
            Err(e)
        }
    }
}

fn mesurer_dans(
    cfg: &std::path::Path,
    out: &std::path::Path,
    libre: bool,
    backend: BackendKind,
    w: &PoidsFourier,
    man: &mut Manifeste,
) -> Resultat<()> {
    let champ = lire_champ(cfg)?;
    let l = champ.l;
    let sha = sha256_fichier(cfg).map_err(|e| Erreur::io(format!("sha256 {}", cfg.display()), e))?;
    let geo = Arc::new(GeometrieCond::nouvelle(l)?);
    // Verrou GPU (jamais pour le CPU), tenu jusqu'à la fin de la mesure.
    let verrou = if backend == BackendKind::Cpu {
        None
    } else {
        let pol = crate::machine::politique(&crate::machine::detection::detecter_rapide());
        Some(crate::machine::VerrouGpu::nouveau(&pol.verrou_gpu).acquerir(Duration::from_secs(pol.attente_verrou_s))?)
    };
    let app = if backend == BackendKind::Cpu {
        None
    } else {
        Some(crate::gpu::ouvrir(backend, 0)?)
    };
    let mut mes = Mesureur::nouveau(app.as_ref(), geo, w)?;
    let nf = w.nf;
    let (f, d, blk) = if libre {
        let (f, d) = mes.mesurer_libre(None, 0, Some(&champ))?;
        (f, d, None)
    } else {
        let r = mes.mesurer(None, 0, Some(&champ))?;
        (r.f, r.d, Some(r.blk))
    };
    ecrire_f64(&out.join("F.npy"), &[nf], &f)?;
    ecrire_f64(&out.join("D.npy"), &[nf, nf], &d)?;
    man.fichier(&out.join("F.npy"))?;
    man.fichier(&out.join("D.npy"))?;
    if let Some(b) = blk {
        ecrire_f32(&out.join("blk.npy"), &[6, (l / 2).pow(4)], &b)?;
        man.fichier(&out.join("blk.npy"))?;
    }
    man.champ("backend", &serde_json::json!(backend.nom()).to_string())?;
    if let Some(a) = &app {
        man.champ("arch", &serde_json::json!(a.arch()).to_string())?;
    }
    if let Some(v) = &verrou {
        man.verrou(v.mode())?;
    }
    man.ajouter_compil(&mes.provenance())?;
    man.champ(
        "mesure",
        &serde_json::json!({"mode": if libre { "libre" } else { "conditionnel" }, "L": l, "nF": nf, "pmax2": 1,
                            "entree": {"chemin": cfg, "sha256": sha}})
        .to_string(),
    )?;
    eprintln!(
        "fibre mesurer : L = {l}, backend {}, {} → {}",
        backend.nom(),
        if libre { "libre" } else { "conditionnel" },
        out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn la_commande_se_declare() {
        let c = commandes();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].get_name(), "fibre");
        let m = c[0]
            .clone()
            .try_get_matches_from([
                "fibre",
                "mesurer",
                "--cfg",
                "x.npy",
                "--out",
                "d",
                "--libre",
                "--backend",
                "cpu",
            ])
            .unwrap();
        let (n, sm) = m.subcommand().unwrap();
        assert_eq!(n, "mesurer");
        assert!(sm.get_flag("libre"));
        assert_eq!(sm.get_one::<String>("backend").unwrap(), "cpu");
    }

    #[test]
    fn cfg_et_out_requis() {
        let c = commandes();
        assert!(c[0]
            .clone()
            .try_get_matches_from(["fibre", "mesurer", "--out", "d"])
            .is_err());
        assert!(c[0]
            .clone()
            .try_get_matches_from(["fibre", "mesurer", "--cfg", "x"])
            .is_err());
    }

    #[test]
    fn sortie_cpu_de_bout_en_bout() {
        // champ froid L = 4 écrit en q64, mesure CPU : D = 0 et blk = 4.
        // scratch sur le disque (la garde SSD refuse /tmp, un tmpfs en RAM)
        let scratch = crate::stockage::scratch_tests("fibre_cli");
        let d = scratch.chemin().to_path_buf();
        let c = crate::commun::ChampHote::froid(4).unwrap();
        ecrire_f64(&d.join("c.npy"), &[4, 4, 4, 4, 4, 4], &c.q).unwrap();
        let rc = mesurer(&d.join("c.npy"), &d.join("sortie"), false, BackendKind::Cpu).unwrap();
        assert_eq!(rc, 0);
        let (forme, dd) = super::super::npy::lire_f64(&d.join("sortie/D.npy")).unwrap();
        assert_eq!(forme, vec![54, 54]);
        assert!(dd.iter().all(|&x| x == 0.0));
        assert!(d.join("sortie/F.npy").exists() && d.join("sortie/blk.npy").exists());
        let rc = mesurer(&d.join("c.npy"), &d.join("sortie_libre"), true, BackendKind::Cpu).unwrap();
        assert_eq!(rc, 0);
        assert!(!d.join("sortie_libre/blk.npy").exists());
        assert!(d.join("sortie/manifest.json").exists() && d.join("sortie/COMPLETE").exists());
    }
}
