//! Tests de la sous-commande `obs` (wilson, polyakov, flot) sur le CPU : fichiers produits, valeurs,
//! ensembles, erreurs d'usage. La configuration est un NPY c16 écrit par le lecteur historique.

use clap::Command;
use kvu::commun::{ChampHote, Erreur};
use kvu::heritage::su2_hb::{write_npy_quat, CpuSu2};
use kvu::obs::cli::{commandes, executer};
use kvu::obs::flot::ParamsFlot;
use kvu::obs::ref_cpu;
use kvu::obs::wilson::ParamsWilson;
use std::path::{Path, PathBuf};

fn dossier_test(nom: &str) -> PathBuf {
    let base = option_env!("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let d = base.join(format!("kvu_obs_cli_{nom}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn ecrire_config(chemin: &Path, l: usize, graine: u64, n_pas: usize) -> ChampHote {
    let mut c = CpuSu2::new(l, graine).unwrap();
    c.init(true);
    for _ in 0..n_pas {
        c.step(2.3, 2);
    }
    let flat = c.flat();
    std::fs::create_dir_all(chemin.parent().unwrap()).unwrap();
    write_npy_quat(chemin, l, &flat).unwrap();
    ChampHote { l, q: flat }
}

/// Analyse `args` (après « kvu obs ») et exécute la commande.
fn lancer(args: &[&str]) -> Result<i32, Erreur> {
    let app = Command::new("kvu").subcommands(commandes());
    let mut argv = vec!["kvu", "obs"];
    argv.extend_from_slice(args);
    let m = app
        .try_get_matches_from(argv)
        .map_err(|e| Erreur::Config(e.to_string()))?;
    let (nom, sm) = m.subcommand().expect("sous-commande obs");
    executer(nom, sm)
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Relit un NPY f64 (v1.0) : (forme, données).
fn lire_npy_f64(chemin: &Path) -> (Vec<usize>, Vec<f64>) {
    let o = std::fs::read(chemin).unwrap();
    assert_eq!(&o[..6], b"\x93NUMPY");
    let n = u16::from_le_bytes([o[8], o[9]]) as usize;
    let entete = std::str::from_utf8(&o[10..10 + n]).unwrap();
    assert!(
        entete.contains("'<f8'") && entete.contains("'fortran_order': False"),
        "{entete}"
    );
    let forme: Vec<usize> = entete[entete.find("'shape': (").unwrap() + 10..]
        .split(')')
        .next()
        .unwrap()
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    let donnees = o[10 + n..]
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (forme, donnees)
}

#[test]
fn obs_wilson_cfg_cpu() {
    let d = dossier_test("wilson");
    let cfg = d.join("cfg_000000.npy");
    let c = ecrire_config(&cfg, 4, 5, 6);
    let sortie = d.join("sortie");
    let rc = lancer(&[
        "wilson",
        "--cfg",
        &s(&cfg),
        "--out",
        &s(&sortie),
        "--rmax",
        "3",
        "--tmax",
        "2",
        "--toutes",
        "--backend",
        "cpu",
    ])
    .unwrap();
    assert_eq!(rc, 0);
    let attendu = ref_cpu::table_wilson(
        &c,
        &ParamsWilson {
            r_max: 3,
            t_max: 2,
            dir_temps: 3,
            toutes_orientations: true,
        },
    )
    .unwrap();
    let (forme, w) = lire_npy_f64(&sortie.join("wilson.npy"));
    assert_eq!(forme, vec![3, 2]);
    assert_eq!(w, attendu.w, "wilson.npy doit contenir la table (r_max, t_max) exacte");
    let csv = std::fs::read_to_string(sortie.join("wilson.csv")).unwrap();
    let lignes: Vec<&str> = csv.lines().collect();
    assert_eq!(lignes[0], "r,t,W");
    assert_eq!(lignes.len(), 1 + 6);
    let v: f64 = lignes[2].split(',').nth(2).unwrap().parse().unwrap(); // r = 1, t = 2
    assert_eq!(v, attendu.get(1, 2));
}

#[test]
fn obs_polyakov_cfg_cpu() {
    let d = dossier_test("polyakov");
    let cfg = d.join("cfg.npy");
    let c = ecrire_config(&cfg, 4, 6, 6);
    let sortie = d.join("sortie");
    assert_eq!(
        lancer(&["polyakov", "--cfg", &s(&cfg), "--out", &s(&sortie), "--backend", "cpu"]).unwrap(),
        0
    );
    let attendu = ref_cpu::correlateur_polyakov(&c, 3).unwrap();
    let csv = std::fs::read_to_string(sortie.join("polyakov_corr.csv")).unwrap();
    let lignes: Vec<&str> = csv.lines().collect();
    assert_eq!(lignes[0], "r,C");
    assert_eq!(lignes.len(), 1 + attendu.len());
    for (r, a) in attendu.iter().enumerate() {
        let v: f64 = lignes[1 + r].split(',').nth(1).unwrap().parse().unwrap();
        assert_eq!(v, *a);
    }
    let moy: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(sortie.join("polyakov_moyen.json")).unwrap()).unwrap();
    assert_eq!(moy["P3"].as_f64().unwrap(), ref_cpu::polyakov_moyen(&c).unwrap()[3]);
}

#[test]
fn obs_flot_cfg_cpu() {
    let d = dossier_test("flot");
    let cfg = d.join("cfg.npy");
    let c = ecrire_config(&cfg, 6, 101, 30);
    let sortie = d.join("sortie");
    let rc = lancer(&[
        "flot",
        "--cfg",
        &s(&cfg),
        "--out",
        &s(&sortie),
        "--dt",
        "0.05",
        "--tmax",
        "2",
        "--backend",
        "cpu",
    ])
    .unwrap();
    assert_eq!(rc, 0);
    let p = ParamsFlot {
        dt: 0.05,
        t_max: 2.0,
        ..ParamsFlot::default()
    };
    let attendu = ref_cpu::flot(&c, &p).unwrap();
    let csv = std::fs::read_to_string(sortie.join("flot.csv")).unwrap();
    let lignes: Vec<&str> = csv.lines().collect();
    assert_eq!(lignes[0], "t,E_plaq,E_trefle,t2E,Q");
    assert_eq!(lignes.len(), 1 + attendu.t.len());
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(sortie.join("flot.json")).unwrap()).unwrap();
    assert_eq!(json["t0"].as_f64().unwrap(), attendu.t0.unwrap());
    assert_eq!(json["w0"].as_f64().unwrap(), attendu.w0.unwrap());
    assert_eq!(json["cible_t2e"].as_f64().unwrap(), 0.3);
}

#[test]
fn obs_ensemble_un_sous_repertoire_par_configuration() {
    let d = dossier_test("ensemble");
    let ens = d.join("ens");
    ecrire_config(&ens.join("cfg_000000.npy"), 4, 1, 4);
    ecrire_config(&ens.join("cfg_000010.npy"), 4, 2, 4);
    ecrire_config(&ens.join("c001").join("cfg_000020.npy"), 4, 3, 4);
    std::fs::write(ens.join("measurements.csv"), "x\n").unwrap();
    let sortie = d.join("sortie");
    assert_eq!(
        lancer(&[
            "wilson",
            "--ensemble",
            &s(&ens),
            "--out",
            &s(&sortie),
            "--rmax",
            "1",
            "--tmax",
            "1",
            "--backend",
            "cpu"
        ])
        .unwrap(),
        0
    );
    for sous in ["cfg_000000", "cfg_000010", "c001_cfg_000020"] {
        assert!(
            sortie.join(sous).join("wilson.npy").exists(),
            "{sous} : wilson.npy absent ({:?})",
            std::fs::read_dir(&sortie).map(|r| r.flatten().map(|e| e.file_name()).collect::<Vec<_>>())
        );
    }
}

#[test]
fn obs_erreurs_d_usage() {
    let d = dossier_test("erreurs");
    let cfg = d.join("cfg.npy");
    ecrire_config(&cfg, 4, 1, 2);
    let sortie = s(&d.join("sortie"));
    // ni --cfg ni --ensemble
    let e = lancer(&["wilson", "--out", &sortie, "--backend", "cpu"]).unwrap_err();
    assert_eq!(e.code_sortie(), 2, "{e}");
    // les deux à la fois
    let e = lancer(&[
        "wilson",
        "--cfg",
        &s(&cfg),
        "--ensemble",
        "x",
        "--out",
        &sortie,
        "--backend",
        "cpu",
    ])
    .unwrap_err();
    assert_eq!(e.code_sortie(), 2, "{e}");
    // --out manquant : refus de clap converti en erreur de configuration
    assert!(lancer(&["wilson", "--cfg", &s(&cfg)]).is_err());
    // paramètre non numérique
    let e = lancer(&[
        "wilson",
        "--cfg",
        &s(&cfg),
        "--out",
        &sortie,
        "--rmax",
        "abc",
        "--backend",
        "cpu",
    ])
    .unwrap_err();
    assert_eq!(e.code_sortie(), 2, "{e}");
    // paramètres de flot invalides
    let e = lancer(&[
        "flot",
        "--cfg",
        &s(&cfg),
        "--out",
        &sortie,
        "--dt",
        "0",
        "--backend",
        "cpu",
    ])
    .unwrap_err();
    assert_eq!(e.code_sortie(), 2, "{e}");
    // ensemble introuvable
    let e = lancer(&[
        "wilson",
        "--ensemble",
        "/inexistant/ens",
        "--out",
        &sortie,
        "--backend",
        "cpu",
    ])
    .unwrap_err();
    assert_eq!(e.code_sortie(), 2, "{e}");
    // backend inconnu
    let e = lancer(&["wilson", "--cfg", &s(&cfg), "--out", &sortie, "--backend", "vulkan"]).unwrap_err();
    assert_eq!(e.code_sortie(), 2, "{e}");
    // configuration illisible : erreur de format, pas de panique
    std::fs::write(d.join("faux.npy"), b"pas un npy").unwrap();
    assert!(lancer(&[
        "wilson",
        "--cfg",
        &s(&d.join("faux.npy")),
        "--out",
        &sortie,
        "--backend",
        "cpu"
    ])
    .is_err());
    // mauvaise sous-commande du module
    assert!(executer(
        "autre",
        &commandes()[0]
            .clone()
            .get_matches_from(["obs", "wilson", "--cfg", "x", "--out", "y"])
    )
    .is_err());
}
