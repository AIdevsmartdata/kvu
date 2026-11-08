//! WP3 : dé-câblage des machines (`kvu.toml`). Ajouts seulement : les tests `machine_*` existants ne changent pas.
//!
//! Gates couvertes : MACH-INCONNUE, MACH-LOCATION, WP3-SURCHARGE (priorité variable > fichier > défaut),
//! rejet des fichiers malformés (rc 2, fichier:ligne:colonne), architecture lue sur le périphérique.
//! Les tests qui lancent le binaire utilisent un HOME jetable, jamais le verrou GPU, jamais de contexte GPU.

use kvu::commun::{BackendKind, Machine};
use kvu::machine::detection::{decider_backend, GpuDetecte, ProfilMachine};
use kvu::machine::politique::politique_avec;
use kvu::machine::sonde_cuda::sm_depuis;
use kvu::machine::sonde_hip::extraire_arch;
use kvu::stockage::fetch::hote_cable_depuis;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn dossier(nom: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("kvu_wp3_{nom}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Lance `kvu` avec un environnement propre (HOME jetable, GPU masqués), cwd = `cwd`.
fn kvu(args: &[&str], cwd: &Path, home: &Path, env: &[(&str, &str)]) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_kvu"));
    c.env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", home)
        .env("HIP_VISIBLE_DEVICES", "-1")
        .env("CUDA_VISIBLE_DEVICES", "-1")
        .current_dir(cwd)
        .args(args);
    for (k, v) in env {
        c.env(k, v);
    }
    c.output().expect("lancement de kvu")
}

fn doctor(cwd: &Path, home: &Path, env: &[(&str, &str)]) -> serde_json::Value {
    let o = kvu(&["doctor", "--json"], cwd, home, env);
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    serde_json::from_slice(&o.stdout).expect("JSON de doctor")
}

fn gpu(backend: BackendKind, nom: &str, arch: &str) -> GpuDetecte {
    GpuDetecte {
        backend,
        nom: nom.into(),
        arch: arch.into(),
        memoire_go: 16.0,
        ratio_fp32_fp64: 2,
        memoire_unifiee: false,
    }
}

fn profil(machine: Machine, gpus: Vec<GpuDetecte>) -> ProfilMachine {
    ProfilMachine {
        machine,
        hote: "conteneur-1234".into(),
        cpu_modele: "cpu quelconque".into(),
        fils_logiques: 32,
        ram_totale_go: 256.0,
        ram_disponible_go: 200.0,
        gpus,
        versions: vec![],
        surcharge_refusee: None,
    }
}

// ---------------------------------------------------------------------------------------------
// MACH-INCONNUE : kvu.toml absent => profil `inconnue`, prudent, message explicite
// ---------------------------------------------------------------------------------------------

#[test]
fn mach_inconnue_sans_kvu_toml_cpu_ssd20_ram8_mem8g() {
    let d = dossier("inconnue");
    let v = doctor(&d, &d, &[]);
    assert_eq!(v["machine"], "inconnue");
    assert!(v["config"]["source"].is_null(), "aucun fichier : défauts compilés");
    assert_eq!(v["backend"]["choisi"], "cpu", "aucun GPU visible : CPU");
    assert_eq!(v["politique"]["seuil_ssd_go"], 20.0);
    assert_eq!(v["politique"]["ram_min_go"], 8.0);
    assert_eq!(v["politique"]["mem_max"], "8G");
    assert!(v["politique"]["pair_cable"].is_null() && v["politique"]["cpus"].is_null());
    let recos = v["recommandations"].to_string();
    assert!(recos.contains("profil inconnue") && recos.contains("aucun kvu.toml"), "{recos}");
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn mach_inconnue_avec_fichier_sans_motifs_ne_reconnait_aucune_machine() {
    let d = dossier("inconnue2");
    std::fs::write(d.join("kvu.toml"), "[profils.inconnue]\nssd_min_go = 20\n").unwrap();
    let v = doctor(&d, &d, &[]);
    assert_eq!(v["machine"], "inconnue");
    assert_eq!(v["config"]["source"], d.join("kvu.toml").display().to_string());
    std::fs::remove_dir_all(d).ok();
}

// ---------------------------------------------------------------------------------------------
// WP3-SURCHARGE : variable KEVINOTRON_* > fichier > défaut `inconnue` (une assertion par clé surchargeable)
// ---------------------------------------------------------------------------------------------

/// Premier cœur autorisé pour ce processus de test (lu dans /proc ; les gates tournent épinglées, ex. cœurs 12-19).
fn coeur_permis() -> u32 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|t| {
            let l = t.lines().find(|l| l.starts_with("Cpus_allowed_list:"))?;
            l.split(':').nth(1)?.trim().split([',', '-']).next()?.parse().ok()
        })
        .unwrap_or(0)
}

fn fichier_complet() -> String {
    let c = coeur_permis();
    FICHIER_COMPLET.replace("0-1", &format!("{c}-{}", c + 1))
}

const FICHIER_COMPLET: &str = "\
[profils.inconnue]
backend = \"auto\"
verrou = \"/tmp/kvu_essai_verrou.lock\"
attente_verrou_s = 11
fenetre_banc_s = 22
nice = 7
cpus = \"0-1\"
fils_max = 3
mem_max = \"5G\"
ram_min_go = 9.5
ssd_min_go = 77
alerte = \"\"
precision = \"f32\"
store = \"~/magasin\"
file = \"~/ma_file\"
cable = \"cable-essai\"
";

#[test]
fn surcharge_le_fichier_l_emporte_sur_le_defaut() {
    let d = dossier("fichier");
    std::fs::write(d.join("kvu.toml"), fichier_complet()).unwrap();
    let v = doctor(&d, &d, &[]);
    let p = &v["politique"];
    assert_eq!(p["verrou_gpu"], "/tmp/kvu_essai_verrou.lock");
    assert_eq!((p["attente_verrou_s"].as_u64(), p["fenetre_banc_s"].as_u64()), (Some(11), Some(22)));
    assert_eq!(p["nice"], 7);
    assert_eq!(p["cpus"], format!("{}-{}", coeur_permis(), coeur_permis() + 1));
    assert_eq!(p["fils_max"], 3);
    assert_eq!(p["mem_max"], "5G");
    assert_eq!(p["ram_min_go"], 9.5);
    assert_eq!(p["seuil_ssd_go"], 77.0);
    assert_eq!(p["precision_conseillee"], "f32");
    assert_eq!(p["store"], d.join("magasin").display().to_string());
    assert_eq!(p["file"], d.join("ma_file").display().to_string());
    assert_eq!(p["pair_cable"], "cable-essai");
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn surcharge_la_variable_l_emporte_sur_le_fichier() {
    let d = dossier("variable");
    std::fs::write(d.join("kvu.toml"), fichier_complet()).unwrap();
    let magasin = d.join("autre_magasin").display().to_string();
    let file = d.join("autre_file").display().to_string();
    let v = doctor(
        &d,
        &d,
        &[
            ("KEVINOTRON_SSD_MIN_GO", "55"),
            ("KEVINOTRON_RAM_MIN_GO", "4"),
            ("KEVINOTRON_STORE", &magasin),
            ("KEVINOTRON_FILE", &file),
            ("KEVINOTRON_CPUS", &coeur_permis().to_string()),
        ],
    );
    let p = &v["politique"];
    assert_eq!(p["seuil_ssd_go"], 55.0, "KEVINOTRON_SSD_MIN_GO");
    assert_eq!(p["ram_min_go"], 4.0, "KEVINOTRON_RAM_MIN_GO");
    assert_eq!(p["store"], magasin.as_str(), "KEVINOTRON_STORE");
    assert_eq!(p["file"], file.as_str(), "KEVINOTRON_FILE");
    assert_eq!(p["cpus"], coeur_permis().to_string(), "KEVINOTRON_CPUS restreint l'affinité du profil");
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn surcharge_backend_et_profil_et_config() {
    let d = dossier("backend");
    std::fs::write(d.join("kvu.toml"), fichier_complet()).unwrap();
    // KEVINOTRON_BACKEND : un backend forcé indisponible donne rc 3, sans repli silencieux
    let o = kvu(&["doctor", "--json"], &d, &d, &[("KEVINOTRON_BACKEND", "hip")]);
    assert_eq!(o.status.code(), Some(3), "{}", String::from_utf8_lossy(&o.stdout));
    let o = kvu(&["doctor", "--json"], &d, &d, &[("KEVINOTRON_BACKEND", "cuda")]);
    assert_eq!(o.status.code(), Some(3));
    // KEVINOTRON_PROFIL / KEVINOTRON_MACHINE : un profil générique est accepté partout
    let v = doctor(&d, &d, &[("KEVINOTRON_PROFIL", "location")]);
    assert_eq!(v["machine"], "location");
    assert_eq!(v["config"]["profil"], "location");
    let v = doctor(&d, &d, &[("KEVINOTRON_MACHINE", "location")]);
    assert_eq!(v["machine"], "location", "ancien nom de la variable conservé");
    let v = doctor(&d, &d, &[("KEVINOTRON_PROFIL", "inconnue"), ("KEVINOTRON_MACHINE", "location")]);
    assert_eq!(v["machine"], "inconnue", "KEVINOTRON_PROFIL prime sur KEVINOTRON_MACHINE");
    // une faute de frappe est refusée (rc 2), pas ignorée
    let o = kvu(&["doctor", "--json"], &d, &d, &[("KEVINOTRON_PROFIL", "cray")]);
    assert_ne!(o.status.code(), Some(0));
    // KEVINOTRON_CONFIG : un autre fichier que ./kvu.toml
    let autre = d.join("autre.toml");
    std::fs::write(&autre, "[profils.inconnue]\nssd_min_go = 31\n").unwrap();
    let v = doctor(&d, &d, &[("KEVINOTRON_CONFIG", &autre.display().to_string())]);
    assert_eq!(v["politique"]["seuil_ssd_go"], 31.0);
    assert_eq!(v["config"]["source"], autre.display().to_string());
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn surcharge_cable_variable_puis_profil_puis_vide() {
    assert_eq!(hote_cable_depuis(Some(" h-env "), Some("h-profil".into())), "h-env");
    assert_eq!(hote_cable_depuis(Some("  "), Some("h-profil".into())), "h-profil");
    assert_eq!(hote_cable_depuis(None, Some("h-profil".into())), "h-profil");
    assert_eq!(hote_cable_depuis(None, None), "", "pas de câble : le fetch refuse");
}

// ---------------------------------------------------------------------------------------------
// Fichiers malformés : rc 2 avec fichier, ligne et colonne
// ---------------------------------------------------------------------------------------------

fn doctor_rejete(contenu: &str) -> String {
    let d = dossier("rejet");
    std::fs::write(d.join("kvu.toml"), contenu).unwrap();
    let o = kvu(&["doctor", "--json"], &d, &d, &[]);
    std::fs::remove_dir_all(&d).ok();
    assert_eq!(o.status.code(), Some(2), "rc 2 attendu : {}", String::from_utf8_lossy(&o.stdout));
    assert!(o.stdout.is_empty(), "rien sur stdout quand la configuration est invalide");
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn fichier_malforme_rc2_avec_ligne_et_colonne() {
    let e = doctor_rejete("[profils.inconnue]\nssd_min_go = 5\nssd_min_go = 6\n");
    assert!(e.contains("kvu.toml:3:1") && e.contains("dupliquée"), "{e}");
    let e = doctor_rejete("[profils.inconnue]\nstore = \"ouvert\n");
    assert!(e.contains("kvu.toml:2:9") && e.contains("non fermée"), "{e}");
    let e = doctor_rejete("[profils.cray]\n");
    assert!(e.contains("kvu.toml:1:1") && e.contains("table inconnue"), "{e}");
    let e = doctor_rejete("[profils.inconnue]\nnice = \"dix\"\n");
    assert!(e.contains("kvu.toml:2:1") && e.contains("entier attendu"), "{e}");
}

#[test]
fn fichier_designe_par_la_variable_et_absent_rc2() {
    let d = dossier("absent");
    let o = kvu(&["doctor", "--json"], &d, &d, &[("KEVINOTRON_CONFIG", &d.join("rien.toml").display().to_string())]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("rien.toml"));
    std::fs::remove_dir_all(d).ok();
}

// ---------------------------------------------------------------------------------------------
// MACH-LOCATION : profil générique de machine louée, architecture lue sur le périphérique
// ---------------------------------------------------------------------------------------------

fn pas_d_env(_: &str) -> Option<String> {
    None
}

#[test]
fn location_pas_de_cable_verrou_local_store_sous_home() {
    let p = profil(Machine::Location, vec![]);
    let pol = politique_avec(&p, &pas_d_env, Path::new("/home/loueur"));
    assert_eq!(pol.pair_cable, None, "pas de câble, donc pas de ssh");
    assert_eq!(pol.verrou_gpu, PathBuf::from("/tmp/kvu_gpu.lock"));
    assert_eq!(pol.store, PathBuf::from("/home/loueur/kevinotron_store"));
    assert_eq!(pol.alerte_moniteur, None);
    assert_eq!(pol.cpus, None);
}

#[test]
fn location_sur_un_materiel_connu_garde_son_verrou_et_ses_planchers() {
    // Sur le second poste, `location` ne doit pas contourner son verrou ni sa garde SSD (vérificateur du 04/10).
    let mut p = profil(Machine::Location, vec![gpu(BackendKind::Cuda, "GPU", "sm_120")]);
    p.hote = "pc-nvidia-exemple".into();
    let pol = politique_avec(&p, &pas_d_env, Path::new("/home/x"));
    assert_eq!(pol.verrou_gpu, PathBuf::from("/tmp/gamer_gpu.lock"));
    assert!(pol.seuil_ssd_go >= 50.0 && pol.ram_min_go >= 6.0);
    assert_eq!(pol.cpus.as_deref(), Some("12-19"));
}

#[test]
fn location_choisit_le_gpu_visible_cuda_et_hip() {
    let cuda = profil(Machine::Location, vec![gpu(BackendKind::Cuda, "NVIDIA GeForce RTX 5060 Ti", "sm_120")]);
    let (b, raison) = decider_backend(&cuda, "auto", true).unwrap();
    assert_eq!(b, BackendKind::Cuda);
    assert!(raison.contains("sm_120"), "{raison}");
    let hip = profil(Machine::Location, vec![gpu(BackendKind::Hip, "AMD Instinct MI300X", "gfx942")]);
    let (b, raison) = decider_backend(&hip, "auto", false).unwrap();
    assert_eq!(b, BackendKind::Hip);
    assert!(raison.contains("gfx942"), "{raison}");
    // CUDA présent mais binaire sans la feature : erreur avec conseil (rc 3), jamais de repli CPU
    assert_eq!(decider_backend(&cuda, "auto", false).unwrap_err().code_sortie(), 3);
}

#[test]
fn architecture_cuda_derivee_de_la_capacite() {
    for ((maj, min), attendu) in [
        ((7, 5), "sm_75"),
        ((8, 0), "sm_80"),
        ((8, 6), "sm_86"),
        ((8, 9), "sm_89"),
        ((9, 0), "sm_90"),
        ((12, 0), "sm_120"),
    ] {
        assert_eq!(sm_depuis(maj, min), attendu);
    }
}

#[test]
fn architecture_hip_lue_dans_la_structure_du_peripherique() {
    for nom in ["gfx1151", "gfx1100", "gfx90a", "gfx942", "gfx1030"] {
        let mut t = vec![0u8; 64];
        t.extend_from_slice(format!("{nom}:sramecc+:xnack-").as_bytes());
        t.extend_from_slice(&[0; 32]);
        assert_eq!(extraire_arch(&t).as_deref(), Some(nom), "arch lue sur le tampon, jamais écrite en dur");
    }
}
