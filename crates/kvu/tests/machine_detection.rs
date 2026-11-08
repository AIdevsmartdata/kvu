//! Tests de la détection, du choix du backend et de la politique (tâche T1).
//! CPU : analyse de /proc simulée (tests/donnees/machine), profils simulés, tableau §3.
//! GPU : la détection réelle (#[ignore = "GPU"]) imprime PASS/FAIL/SKIP.

use kvu::commun::{BackendKind, Erreur, Machine, Precision};
use kvu::machine::cli::{backend_attendu, etat_json, rapport_doctor};
use kvu::machine::detection::*;
use kvu::machine::politique::{politique_avec, ALERTE_ACEMAGIC};
use kvu::machine::sonde_cuda::{candidats_nvrtc, decoder_version_cuda, sm_depuis};
use kvu::machine::sonde_hip::{
    bibliotheque_chargee, chaine_c, decoder_version_hip, extraire_arch, version_rocm, versions_rocm,
};
use std::path::{Path, PathBuf};

fn donnees(machine: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/donnees/machine")
        .join(machine)
}

fn gpu_hip() -> GpuDetecte {
    GpuDetecte {
        backend: BackendKind::Hip,
        nom: "AMD Radeon 8060S".into(),
        arch: "gfx1151".into(),
        memoire_go: 111.5,
        ratio_fp32_fp64: 16,
        memoire_unifiee: true,
    }
}

fn gpu_cuda() -> GpuDetecte {
    GpuDetecte {
        backend: BackendKind::Cuda,
        nom: "NVIDIA GeForce RTX 5060 Ti".into(),
        arch: "sm_120".into(),
        memoire_go: 15.5,
        ratio_fp32_fp64: 64,
        memoire_unifiee: false,
    }
}

fn profil(machine: Machine, gpus: Vec<GpuDetecte>) -> ProfilMachine {
    ProfilMachine {
        machine,
        hote: "h".into(),
        cpu_modele: "cpu".into(),
        fils_logiques: 8,
        ram_totale_go: 32.0,
        ram_disponible_go: 20.0,
        gpus,
        versions: vec![],
        surcharge_refusee: None,
    }
}

fn pas_d_env(_: &str) -> Option<String> {
    None
}

// ---------------------------------------------------------------------------------------------
// /proc simulé
// ---------------------------------------------------------------------------------------------

#[test]
fn analyse_proc_acemagic() {
    let p = profil_depuis(&donnees("acemagic"), vec![], vec![], None);
    assert_eq!(p.hote, "acemagic");
    assert_eq!(p.cpu_modele, "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S");
    assert_eq!(p.fils_logiques, 32);
    assert_eq!(p.machine, Machine::Acemagic);
    assert!((p.ram_totale_go - 120_909_552.0 * 1024.0 / (1u64 << 30) as f64).abs() < 1e-9);
    assert!((p.ram_disponible_go - 111_230_000.0 * 1024.0 / (1u64 << 30) as f64).abs() < 1e-9);
}

#[test]
fn analyse_proc_gamer() {
    let p = profil_depuis(&donnees("gamer"), vec![], vec![], None);
    assert_eq!(p.hote, HOTE_GAMER);
    assert_eq!(p.fils_logiques, 20);
    assert_eq!(p.machine, Machine::Gamer);
    assert!(p.ram_totale_go > 30.0 && p.ram_totale_go < 31.0);
}

#[test]
fn analyse_proc_inconnue_et_sans_memavailable() {
    let p = profil_depuis(&donnees("inconnue"), vec![], vec![], None);
    assert_eq!(p.machine, Machine::Inconnue);
    assert_eq!(p.fils_logiques, 8);
    // MemAvailable absent : MemFree + Buffers + Cached = 5 500 000 kB.
    assert!((p.ram_disponible_go - 5_500_000.0 * 1024.0 / (1u64 << 30) as f64).abs() < 1e-9);
}

#[test]
fn racine_proc_absente_ne_panique_pas() {
    let p = profil_depuis(Path::new("/chemin/qui/nexiste/pas"), vec![], vec![], None);
    assert_eq!(p.machine, Machine::Inconnue);
    assert_eq!(p.fils_logiques, 1);
    assert_eq!(p.ram_totale_go, 0.0);
    assert!(p.hote.is_empty() && p.cpu_modele.is_empty());
}

#[test]
fn analyse_cpuinfo_et_meminfo_cas_limites() {
    assert_eq!(analyser_cpuinfo(""), (String::new(), 1));
    let (m, n) = analyser_cpuinfo("processor : 0\nmodel name : A\nprocessor : 1\nmodel name : B\n");
    assert_eq!((m.as_str(), n), ("A", 2));
    assert_eq!(analyser_meminfo(""), (0.0, 0.0));
    assert_eq!(analyser_meminfo("MemTotal: abc kB\n"), (0.0, 0.0));
    let (t, d) = analyser_meminfo("MemTotal: 1048576 kB\nMemAvailable: 524288 kB\n");
    assert_eq!((t, d), (1.0, 0.5));
}

#[test]
fn surcharge_de_machine() {
    // Vérificateur du 04/10 : `=gamer` sur acemagic faisait tourner HIP sous /tmp/gamer_gpu.lock (aucune
    // exclusion avec la production d'acemagic). Une machine CONNUE différente du matériel est REFUSÉE.
    let p = profil_depuis(&donnees("acemagic"), vec![], vec![], Some("gamer"));
    assert_eq!(p.machine, Machine::Acemagic, "la machine du matériel est gardée");
    assert!(p.surcharge_refusee.as_deref().is_some_and(|m| m.contains("gamer") && m.contains("acemagic")));
    let p = profil_depuis(&donnees("gamer"), vec![], vec![], Some("acemagic"));
    assert_eq!(p.machine, Machine::Gamer);
    assert!(p.surcharge_refusee.is_some());
    // « inconnue » reste admise partout ; une machine non identifiée peut se faire passer pour une autre.
    let p = profil_depuis(&donnees("acemagic"), vec![], vec![], Some("inconnue"));
    assert_eq!((p.machine, p.surcharge_refusee.is_none()), (Machine::Inconnue, true));
    let p = profil_depuis(&donnees("inconnue"), vec![], vec![], Some("gamer"));
    assert_eq!((p.machine, p.surcharge_refusee.is_none()), (Machine::Gamer, true));
    // Surcharge invalide : REFUSÉE (avant le 04/10 elle était ignorée sans erreur).
    let q = profil_depuis(&donnees("acemagic"), vec![], vec![], Some("pc-de-mamie"));
    assert_eq!(q.machine, Machine::Acemagic);
    assert!(q.surcharge_refusee.as_deref().is_some_and(|m| m.contains("pc-de-mamie")));
    assert!(q
        .versions
        .iter()
        .any(|(k, v)| k == "avertissement" && v.contains("pc-de-mamie")));
    assert_eq!(controler_surcharge(Machine::Acemagic, Some("acemagic")).unwrap(), Machine::Acemagic);
    assert_eq!(controler_surcharge(Machine::Acemagic, None).unwrap(), Machine::Acemagic);
    assert_eq!(controler_surcharge(Machine::Gamer, Some("xyz")).unwrap_err().code_sortie(), 2);
    // Surcharge vide : sans effet.
    assert_eq!(
        profil_depuis(&donnees("gamer"), vec![], vec![], Some("  ")).machine,
        Machine::Gamer
    );
}

// ---------------------------------------------------------------------------------------------
// Identification
// ---------------------------------------------------------------------------------------------

#[test]
fn identification_par_hote_cpu_ou_gpu() {
    use Machine::*;
    assert_eq!(identifier_machine("acemagic", "", &[]), Acemagic);
    assert_eq!(
        identifier_machine("x", "AMD Ryzen AI Max+ 395 w/ Radeon", &[]),
        Acemagic
    );
    assert_eq!(identifier_machine("x", "", &[gpu_hip()]), Acemagic);
    assert_eq!(identifier_machine(HOTE_GAMER, "", &[]), Gamer);
    assert_eq!(identifier_machine("x", "Intel(R) Core(TM) i5-14600KF", &[]), Gamer);
    assert_eq!(identifier_machine("x", "", &[gpu_cuda()]), Gamer);
    assert_eq!(identifier_machine("x", "Xeon", &[]), Inconnue);
    // Un autre GPU CUDA ne fait pas un gamer.
    let mut autre = gpu_cuda();
    autre.arch = "sm_80".into();
    assert_eq!(identifier_machine("x", "Xeon", &[autre]), Inconnue);
}

#[test]
fn ratios_fp32_fp64() {
    assert_eq!(ratio_fp32_fp64("gfx1151", "AMD Radeon 8060S"), 16);
    assert_eq!(ratio_fp32_fp64("sm_120", "NVIDIA GeForce RTX 5060 Ti"), 64);
    assert_eq!(ratio_fp32_fp64("sm_89", "NVIDIA GeForce RTX 4090"), 64);
    assert_eq!(ratio_fp32_fp64("sm_80", "NVIDIA A100"), 2);
    assert_eq!(ratio_fp32_fp64("gfx90a", "AMD Instinct MI210"), 2);
}

// ---------------------------------------------------------------------------------------------
// Sondes : décodages purs
// ---------------------------------------------------------------------------------------------

#[test]
fn decodage_des_versions() {
    assert_eq!(decoder_version_cuda(13020), "13.2");
    assert_eq!(decoder_version_cuda(12080), "12.8");
    assert_eq!(decoder_version_cuda(0), "");
    assert_eq!(sm_depuis(12, 0), "sm_120");
    assert_eq!(sm_depuis(8, 9), "sm_89");
    assert_eq!(decoder_version_hip(70_000_000 + 1_400_000 + 60_850), "7.14.60850");
    assert_eq!(decoder_version_hip(0), "");
}

#[test]
fn extraction_de_larchitecture_dans_un_tampon_de_proprietes() {
    let mut t = vec![0u8; 2048];
    t[..16].copy_from_slice(b"AMD Radeon 8060S");
    // gcnArchName quelque part dans la structure, avec options de fonctionnalités.
    t[600..600 + 23].copy_from_slice(b"gfx1151:sramecc-:xnack-");
    assert_eq!(extraire_arch(&t).as_deref(), Some("gfx1151"));
    assert_eq!(chaine_c(&t[..256]), "AMD Radeon 8060S");
    // « gfx » seul ou suivi d'une lettre : pas une architecture.
    let mut u = vec![0u8; 64];
    u[..6].copy_from_slice(b"gfxabc");
    assert_eq!(extraire_arch(&u), None);
    assert_eq!(extraire_arch(&[0u8; 32]), None);
    let mut v = vec![0u8; 64];
    v[..7].copy_from_slice(b"gfx90a\0");
    assert_eq!(extraire_arch(&v).as_deref(), Some("gfx90a"));
}

fn dossier_temp(nom: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("kvu_{nom}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn version_rocm_lue_dans_opt() {
    let opt = dossier_temp("opt");
    for (d, v) in [("rocm714", "7.14.0"), ("rocm71", "7.1.0"), ("autre", "9.9.9")] {
        std::fs::create_dir_all(opt.join(d).join(".info")).unwrap();
        std::fs::write(opt.join(d).join(".info/version"), format!("{v}\n")).unwrap();
    }
    assert_eq!(version_rocm(&opt).as_deref(), Some("7.14.0"));
    // « rocm » seul (lien usuel) est prioritaire.
    std::fs::create_dir_all(opt.join("rocm/.info")).unwrap();
    std::fs::write(opt.join("rocm/.info/version"), "7.2.0").unwrap();
    assert_eq!(version_rocm(&opt).as_deref(), Some("7.2.0"));
    assert_eq!(version_rocm(&opt.join("absent")), None);
    std::fs::remove_dir_all(&opt).unwrap();
}

#[test]
fn candidats_nvrtc_ordre_ld_library_path_puis_usr_local() {
    let racine = dossier_temp("nvrtc");
    let ld = racine.join("ld");
    std::fs::create_dir_all(&ld).unwrap();
    for f in [
        "libnvrtc.so",
        "libnvrtc.so.13",
        "libnvrtc-builtins.so",
        "libnvrtc_static.a",
    ] {
        std::fs::write(ld.join(f), b"").unwrap();
    }
    let ul = racine.join("local");
    for v in ["cuda-12.4", "cuda-13.3"] {
        std::fs::create_dir_all(ul.join(v).join("lib64")).unwrap();
        std::fs::write(ul.join(v).join("lib64/libnvrtc.so.13"), b"").unwrap();
    }
    std::fs::create_dir_all(ul.join("autre/lib64")).unwrap();
    std::fs::write(ul.join("autre/lib64/libnvrtc.so"), b"").unwrap();
    let c = candidats_nvrtc(&format!("{}:/nexiste/pas", ld.display()), &ul);
    let noms: Vec<String> = c
        .iter()
        .map(|p| p.strip_prefix(&racine).unwrap_or(p).display().to_string())
        .collect();
    assert_eq!(
        noms,
        vec![
            "ld/libnvrtc.so.13",
            "ld/libnvrtc.so",
            "local/cuda-13.3/lib64/libnvrtc.so.13",
            "local/cuda-12.4/lib64/libnvrtc.so.13",
            "libnvrtc.so"
        ]
    );
    std::fs::remove_dir_all(&racine).unwrap();
}

// ---------------------------------------------------------------------------------------------
// Choix du backend (profils simulés)
// ---------------------------------------------------------------------------------------------

fn dec(p: &ProfilMachine, demande: &str, cuda: bool) -> Result<BackendKind, Erreur> {
    decider_backend(p, demande, cuda).map(|(b, _)| b)
}

#[test]
fn auto_acemagic_hip_gamer_cuda() {
    let ace = profil(Machine::Acemagic, vec![gpu_hip()]);
    let gam = profil(Machine::Gamer, vec![gpu_cuda()]);
    assert_eq!(dec(&ace, "", false).unwrap(), BackendKind::Hip);
    assert_eq!(dec(&ace, "auto", true).unwrap(), BackendKind::Hip);
    assert_eq!(dec(&gam, "auto", true).unwrap(), BackendKind::Cuda);
    let (_, raison) = decider_backend(&gam, "auto", true).unwrap();
    assert!(raison.contains("gamer") && raison.contains("sm_120"), "{raison}");
}

#[test]
fn auto_gamer_sans_feature_cuda_est_une_erreur_avec_conseil() {
    let gam = profil(Machine::Gamer, vec![gpu_cuda()]);
    let e = dec(&gam, "auto", false).unwrap_err();
    assert_eq!(e.code_sortie(), 3);
    let t = e.to_string();
    assert!(t.contains("build_auto.sh"), "le conseil de recompiler manque : {t}");
}

#[test]
fn auto_inconnue_premier_gpu_utilisable_sinon_cpu() {
    let inc_cuda = profil(Machine::Inconnue, vec![gpu_cuda()]);
    assert_eq!(dec(&inc_cuda, "auto", true).unwrap(), BackendKind::Cuda);
    assert_eq!(dec(&inc_cuda, "auto", false).unwrap_err().code_sortie(), 3);
    // CUDA non pilotable + HIP présent : le HIP est le premier GPU utilisable.
    let mixte = profil(Machine::Inconnue, vec![gpu_cuda(), gpu_hip()]);
    assert_eq!(dec(&mixte, "auto", false).unwrap(), BackendKind::Hip);
    // Aucun GPU : CPU, et seulement dans ce cas.
    let nu = profil(Machine::Inconnue, vec![]);
    assert_eq!(dec(&nu, "auto", true).unwrap(), BackendKind::Cpu);
    // Machine connue sans GPU visible : CPU, avec un avertissement dans la raison.
    let (b, raison) = decider_backend(&profil(Machine::Acemagic, vec![]), "auto", true).unwrap();
    assert_eq!(b, BackendKind::Cpu);
    assert!(raison.contains("vérifier"), "{raison}");
}

#[test]
fn backend_force_indisponible_donne_rc3() {
    let ace = profil(Machine::Acemagic, vec![gpu_hip()]);
    let gam = profil(Machine::Gamer, vec![gpu_cuda()]);
    assert_eq!(dec(&ace, "cuda", true).unwrap_err().code_sortie(), 3);
    assert_eq!(dec(&gam, "hip", true).unwrap_err().code_sortie(), 3);
    // cuda forcé, GPU présent mais binaire sans la feature.
    assert_eq!(dec(&gam, "cuda", false).unwrap_err().code_sortie(), 3);
    assert!(dec(&gam, "cuda", false)
        .unwrap_err()
        .to_string()
        .contains("build_auto.sh"));
    assert_eq!(
        dec(&profil(Machine::Inconnue, vec![]), "hip", true)
            .unwrap_err()
            .code_sortie(),
        3
    );
}

#[test]
fn backend_force_disponible_et_cpu_toujours_possible() {
    let ace = profil(Machine::Acemagic, vec![gpu_hip()]);
    assert_eq!(dec(&ace, "HIP", false).unwrap(), BackendKind::Hip);
    assert_eq!(dec(&ace, " cpu ", false).unwrap(), BackendKind::Cpu);
    assert_eq!(
        dec(&profil(Machine::Gamer, vec![gpu_cuda()]), "cuda", true).unwrap(),
        BackendKind::Cuda
    );
    assert_eq!(
        dec(&profil(Machine::Inconnue, vec![]), "cpu", true).unwrap(),
        BackendKind::Cpu
    );
}

#[test]
fn backend_invalide_est_une_erreur_dusage() {
    let e = dec(&profil(Machine::Acemagic, vec![gpu_hip()]), "vulkan", true).unwrap_err();
    assert_eq!(e.code_sortie(), 2);
}

#[test]
fn memoire_gpu_et_l_max() {
    let p = profil(Machine::Gamer, vec![gpu_cuda()]);
    assert_eq!(memoire_gpu_go(&p, BackendKind::Cuda), 15.5);
    assert_eq!(memoire_gpu_go(&p, BackendKind::Hip), 0.0);
    // 16 Go : moitié = 8 Go ; F32 : 16·L⁴·4·2 octets ≤ 8 Go ⇒ L⁴ ≤ 6,7e7 ⇒ L ≤ 90.
    assert_eq!(l_max_memoire(16.0, 4), 90);
    assert!(l_max_memoire(16.0, 8) < l_max_memoire(16.0, 4));
    assert_eq!(l_max_memoire(0.0, 4), 0);
    // Cohérence avec §2 : un champ F32 à L = 64 pèse 1,07 Go.
    assert!((16.0 * 64f64.powi(4) * 4.0 / 1e9 - 1.07).abs() < 0.01);
}

// ---------------------------------------------------------------------------------------------
// Politique (tableau §3)
// ---------------------------------------------------------------------------------------------

#[test]
fn politique_acemagic() {
    let p = politique_avec(&profil(Machine::Acemagic, vec![]), &pas_d_env, Path::new("/home/x"));
    assert_eq!(p.verrou_gpu, Path::new("/tmp/kevinotron_gpu.lock"));
    assert_eq!((p.attente_verrou_s, p.fenetre_banc_s), (7200, 600));
    assert_eq!((p.nice, p.cpus.clone()), (10, None));
    assert_eq!(p.fils_max, 8);
    assert_eq!(p.mem_max.as_deref(), Some("24G"));
    assert_eq!((p.ram_min_go, p.seuil_ssd_go), (25.0, 100.0));
    assert_eq!(p.alerte_moniteur, Some(PathBuf::from(ALERTE_ACEMAGIC)));
    assert_eq!(p.precision_conseillee, Precision::F32);
    assert_eq!(p.store, Path::new("/chemin/vers/projet/kevinotron_store"));
    assert_eq!(p.file, Path::new("/home/x/.local/state/kevinotron/file"));
    assert_eq!(p.pair_cable.as_deref(), Some("gamer-eth"));
}

#[test]
fn politique_gamer() {
    let p = politique_avec(&profil(Machine::Gamer, vec![]), &pas_d_env, Path::new("/home/utilisateur"));
    assert_eq!(p.verrou_gpu, Path::new("/tmp/gamer_gpu.lock"));
    assert_eq!(p.nice, 5);
    assert_eq!(p.cpus.as_deref(), Some("12-19"));
    assert_eq!(p.fils_max, 8);
    assert_eq!(p.mem_max.as_deref(), Some("12G"));
    assert_eq!((p.ram_min_go, p.seuil_ssd_go), (6.0, 50.0));
    assert_eq!(p.alerte_moniteur, None);
    assert_eq!(p.precision_conseillee, Precision::F32);
    assert_eq!(p.store, Path::new("/home/utilisateur/kevinotron_store"));
    assert_eq!(p.pair_cable, None);
}

#[test]
fn politique_inconnue() {
    let p = politique_avec(&profil(Machine::Inconnue, vec![]), &pas_d_env, Path::new("/home/z"));
    assert_eq!(p.verrou_gpu, Path::new("/tmp/kevinotron_gpu.lock"));
    assert_eq!((p.nice, p.fils_max), (10, 4));
    assert_eq!(p.mem_max.as_deref(), Some("8G"));
    assert_eq!((p.ram_min_go, p.seuil_ssd_go), (8.0, 20.0));
    assert_eq!(p.precision_conseillee, Precision::F64);
    assert_eq!(p.store, Path::new("/home/z/kevinotron_store"));
    assert_eq!(p.cpus, None);
}

#[test]
fn politique_surcharges_denvironnement() {
    let env = |k: &str| match k {
        "KEVINOTRON_SSD_MIN_GO" => Some("12,5".to_string()),
        "KEVINOTRON_RAM_MIN_GO" => Some("3,5".to_string()),
        "KEVINOTRON_STORE" => Some(" /srv/store ".to_string()),
        "KEVINOTRON_FILE" => Some("/srv/file".to_string()),
        _ => None,
    };
    let p = politique_avec(&profil(Machine::Acemagic, vec![]), &env, Path::new("/home/x"));
    assert_eq!(p.seuil_ssd_go, 12.5);
    assert_eq!(p.ram_min_go, 3.5);
    // valeur invalide : ignorée (défaut d'acemagic, 25 Go)
    let mauvaise = |k: &str| (k == "KEVINOTRON_RAM_MIN_GO").then(|| "-1".to_string());
    assert_eq!(politique_avec(&profil(Machine::Acemagic, vec![]), &mauvaise, Path::new("/h")).ram_min_go, 25.0);
    assert_eq!(p.store, Path::new("/srv/store"));
    assert_eq!(p.file, Path::new("/srv/file"));
}

/// Sous `KEVINOTRON_MACHINE=inconnue` sur une machine connue, le verrou, l'affinité et les planchers restent ceux
/// du MATÉRIEL : le GPU réellement utilisé garde son exclusion avec la production.
#[test]
fn politique_securite_suit_le_materiel() {
    let p = profil_depuis(&donnees("gamer"), vec![], vec![], Some("inconnue"));
    assert_eq!(p.machine, Machine::Inconnue);
    let pol = politique_avec(&p, &|_| None, Path::new("/h"));
    assert_eq!(pol.verrou_gpu, Path::new("/tmp/gamer_gpu.lock"));
    assert_eq!(pol.cpus.as_deref(), Some("12-19"));
    assert_eq!(pol.seuil_ssd_go, 50.0, "le plus strict des deux seuils (gamer 50, inconnue 20)");
    assert_eq!(pol.ram_min_go, 8.0, "le plus strict des deux planchers (gamer 6, inconnue 8)");
    let p = profil_depuis(&donnees("acemagic"), vec![], vec![], Some("inconnue"));
    let pol = politique_avec(&p, &|_| None, Path::new("/h"));
    assert_eq!(pol.verrou_gpu, Path::new("/tmp/kevinotron_gpu.lock"));
    assert_eq!((pol.seuil_ssd_go, pol.ram_min_go), (100.0, 25.0));
    // une surcharge explicite des seuils reste possible (tests hermétiques)
    let env = |k: &str| (k == "KEVINOTRON_SSD_MIN_GO").then(|| "0".to_string());
    assert_eq!(politique_avec(&p, &env, Path::new("/h")).seuil_ssd_go, 0.0);
}

/// KEVINOTRON_CPUS ne peut que RESTREINDRE l'affinité de la politique (jamais l'élargir aux cœurs P du gamer).
#[test]
fn surcharge_des_cpus_restreint_seulement() {
    let gamer = profil_depuis(&donnees("gamer"), vec![], vec![], None);
    let env = |v: &'static str| move |k: &str| (k == "KEVINOTRON_CPUS").then(|| v.to_string());
    assert_eq!(politique_avec(&gamer, &env("14-15"), Path::new("/h")).cpus.as_deref(), Some("14-15"));
    assert_eq!(politique_avec(&gamer, &env("0-19"), Path::new("/h")).cpus.as_deref(), Some("12-19"));
    assert_eq!(politique_avec(&gamer, &env("0-3"), Path::new("/h")).cpus.as_deref(), Some("12-19"), "disjoint : ignoré");
    assert_eq!(politique_avec(&gamer, &env("x"), Path::new("/h")).cpus.as_deref(), Some("12-19"), "invalide : ignoré");
    let ace = profil_depuis(&donnees("acemagic"), vec![], vec![], None);
    assert_eq!(politique_avec(&ace, &|_| None, Path::new("/h")).cpus, None);
    assert_eq!(politique_avec(&ace, &env("2,4-5"), Path::new("/h")).cpus.as_deref(), Some("2,4-5"));
}

#[test]
fn surcharge_invalide_du_seuil_ssd_est_ignoree() {
    for mauvais in ["abc", "-5", "NaN", "inf", ""] {
        let env = move |k: &str| (k == "KEVINOTRON_SSD_MIN_GO").then(|| mauvais.to_string());
        let p = politique_avec(&profil(Machine::Gamer, vec![]), &env, Path::new("/h"));
        assert_eq!(p.seuil_ssd_go, 50.0, "valeur {mauvais:?}");
    }
    // 0 est une valeur légale (désactive la marge, choix explicite).
    let env = |k: &str| (k == "KEVINOTRON_SSD_MIN_GO").then(|| "0".to_string());
    assert_eq!(
        politique_avec(&profil(Machine::Gamer, vec![]), &env, Path::new("/h")).seuil_ssd_go,
        0.0
    );
}

// ---------------------------------------------------------------------------------------------
// etat --json (contrat §4.11) et doctor
// ---------------------------------------------------------------------------------------------

#[test]
fn etat_json_contrat_gele() {
    let p = profil_depuis(&donnees("gamer"), vec![], vec![], None);
    let dir = dossier_temp("etat");
    let env = |k: &str| (k == "KEVINOTRON_FILE").then(|| dir.display().to_string());
    let pol = politique_avec(&p, &env, Path::new("/h"));
    let v = etat_json(&p, &pol);
    let obj = v.as_object().unwrap();
    let mut cles: Vec<&str> = obj.keys().map(String::as_str).collect();
    cles.sort_unstable();
    let mut attendu = vec![
        "machine",
        "backend",
        "gpu_verrou",
        "ssd_libre_go",
        "ssd_seuil_go",
        "ram_dispo_go",
        "charge1",
        "file",
        "alerte_moniteur",
        "horodatage",
    ];
    attendu.sort_unstable();
    assert_eq!(cles, attendu);
    assert_eq!(v["machine"], "gamer");
    assert_eq!(v["backend"], "cuda");
    assert!(v["gpu_verrou"]["pris"].is_boolean());
    assert!(v["gpu_verrou"].as_object().unwrap().contains_key("pid"));
    for k in ["attente", "en_cours", "echec"] {
        assert_eq!(v["file"][k], 0, "{k}");
    }
    assert_eq!(v["file"].as_object().unwrap().len(), 4, "attente, en_cours, echec, orphelins");
    assert_eq!(v["ssd_seuil_go"], 50.0);
    assert!(v["ssd_libre_go"].as_f64().unwrap() > 0.0);
    assert!(v["ram_dispo_go"].as_f64().unwrap() > 0.0);
    assert!(v["charge1"].as_f64().unwrap() >= 0.0);
    assert!(v["alerte_moniteur"].is_boolean());
    let h = v["horodatage"].as_str().unwrap();
    assert!(h.len() == 20 && h.ends_with('Z') && h.contains('T'), "{h}");
    // Une ligne : la sérialisation compacte ne contient pas de retour à la ligne.
    assert!(!v.to_string().contains('\n'));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn etat_json_en_moins_de_200_ms_sans_contexte_gpu() {
    let debut = std::time::Instant::now();
    let p = detecter_rapide();
    let pol = kvu::machine::politique(&p);
    let v = etat_json(&p, &pol);
    let ms = debut.elapsed().as_millis();
    assert!(v.is_object());
    assert!(ms < 200, "etat a pris {ms} ms");
}

#[test]
fn etat_backend_attendu() {
    assert_eq!(backend_attendu(Machine::Acemagic, ""), "hip");
    assert_eq!(backend_attendu(Machine::Gamer, "auto"), "cuda");
    assert_eq!(backend_attendu(Machine::Inconnue, ""), "auto");
    assert_eq!(backend_attendu(Machine::Gamer, "CPU"), "cpu");
    assert_eq!(backend_attendu(Machine::Acemagic, "hip"), "hip");
}

#[test]
fn doctor_json_sur_profil_simule() {
    let mut p = profil_depuis(
        &donnees("acemagic"),
        vec![gpu_hip()],
        vec![("rocm".into(), "7.14.0".into())],
        None,
    );
    p.machine = Machine::Acemagic;
    let dir = dossier_temp("doctor");
    let env = |k: &str| match k {
        "KEVINOTRON_FILE" => Some(dir.join("file").display().to_string()),
        "KEVINOTRON_STORE" => Some(dir.join("store").display().to_string()),
        _ => None,
    };
    let pol = politique_avec(&p, &env, Path::new("/h"));
    let (v, _rc) = rapport_doctor(&p, &pol);
    assert_eq!(v["machine"], "acemagic");
    assert_eq!(v["gpus"][0]["arch"], "gfx1151");
    assert_eq!(v["gpus"][0]["ratio_fp32_fp64"], 16);
    assert_eq!(v["gpus"][0]["memoire_unifiee"], true);
    assert_eq!(v["versions"]["rocm"], "7.14.0");
    assert_eq!(v["politique"]["seuil_ssd_go"], 100.0);
    assert_eq!(v["politique"]["nice"], 10);
    for cle in [
        "cpu",
        "ram",
        "backend",
        "verrou_gpu",
        "ssd",
        "store",
        "file",
        "cache",
        "recommandations",
        "alerte_moniteur",
    ] {
        assert!(!v[cle].is_null(), "clé {cle} absente");
    }
    // Les recommandations donnent la précision et le L maximal.
    let recos = v["recommandations"].to_string();
    assert!(recos.contains("f32") && recos.contains("L maximal"), "{recos}");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn doctor_json_cuda_force_sur_acemagic_donne_rc3() {
    // Pas de variable d'environnement ici (tests parallèles) : on passe par la décision pure.
    let ace = profil(Machine::Acemagic, vec![gpu_hip()]);
    assert_eq!(decider_backend(&ace, "cuda", true).unwrap_err().code_sortie(), 3);
}

// ---------------------------------------------------------------------------------------------
// Détection réelle (ne panique jamais, rapide) et GPU
// ---------------------------------------------------------------------------------------------

#[test]
fn detecter_ne_panique_pas_et_reste_rapide() {
    let debut = std::time::Instant::now();
    let p = detecter();
    let ms = debut.elapsed().as_millis();
    assert!(p.fils_logiques >= 1);
    assert!(p.ram_totale_go > 0.0);
    assert!(!p.hote.is_empty());
    assert!(ms < 2000, "detecter() a pris {ms} ms");
    // Cohérence : tout GPU HIP a une arch « gfx… », tout GPU CUDA une arch « sm_… ».
    for g in &p.gpus {
        match g.backend {
            BackendKind::Hip => assert!(g.arch.starts_with("gfx"), "{g:?}"),
            BackendKind::Cuda => assert!(g.arch.starts_with("sm_"), "{g:?}"),
            BackendKind::Cpu => panic!("un GPU ne peut pas avoir le backend cpu"),
        }
    }
    println!(
        "detecter() : {ms} ms, machine {}, {} GPU",
        p.machine.nom(),
        p.gpus.len()
    );
}

#[test]
fn detecter_rapide_sans_gpu() {
    let p = detecter_rapide();
    assert!(p.gpus.is_empty());
    assert!(p.fils_logiques >= 1);
}

#[test]
#[ignore = "GPU"]
fn detection_reelle_sur_cette_machine() {
    let p = detecter();
    println!("profil : {p:#?}");
    let attendu = match p.hote.as_str() {
        "acemagic" => Some(("acemagic", "hip", "gfx1151")),
        HOTE_GAMER => Some(("gamer", "cuda", "sm_120")),
        _ => None,
    };
    let Some((machine, backend, arch)) = attendu else {
        println!("SKIP DETECTION-REELLE : hôte {} inconnu", p.hote);
        return;
    };
    let ok_machine = p.machine.nom() == machine;
    let gpu = p.gpus.iter().find(|g| g.backend.nom() == backend);
    let ok_gpu = gpu.is_some_and(|g| g.arch == arch && g.memoire_go > 1.0 && !g.nom.is_empty());
    let ok_versions = if backend == "hip" {
        // `rocm` n'est consignée que si l'installation CHARGÉE la déclare (/opt/rocm*/.info) ; le runtime des paquets
        // du système n'en a pas : seule `hip_runtime` est exigée (une installation /opt non chargée est
        // `rocm_opt_non_chargee`, jamais `rocm`).
        p.versions.iter().any(|(k, _)| k == "hip_runtime")
    } else {
        p.versions.iter().any(|(k, _)| k == "cuda_pilote") && p.versions.iter().any(|(k, _)| k == "nvrtc")
    };
    let choix = choisir_backend(&p);
    let ok_choix = match (&choix, cfg!(feature = "cuda") || backend == "hip") {
        (Ok(b), true) => b.nom() == backend,
        (Err(e), false) => e.code_sortie() == 3,
        _ => false,
    };
    let ok = ok_machine && ok_gpu && ok_versions && ok_choix;
    println!(
        "{} DETECTION-REELLE {machine}/{backend}/{arch} (machine {ok_machine}, gpu {ok_gpu}, versions {ok_versions}, choix {ok_choix})",
        if ok { "PASS" } else { "FAIL" }
    );
    assert!(ok);
}

/// Vérificateur du 04/10 : `doctor` affichait « rocm 7.14.0 » (/opt/rocm714, présent mais NON chargé) alors
/// que le runtime chargé était le 7.1 du système. La version vient désormais de l'installation chargée.
#[test]
fn version_rocm_de_l_installation_chargee() {
    let maps = "7f00-7f10 r-xp 00000000 103:02 123 /usr/lib/x86_64-linux-gnu/libamdhip64.so.7.1.52801\n\
                7f20-7f30 r-xp 00000000 103:02 456 /usr/lib/x86_64-linux-gnu/libc.so.6\n";
    let lib = bibliotheque_chargee(maps).expect("libamdhip64 dans maps");
    assert!(lib.ends_with("libamdhip64.so.7.1.52801"));
    assert_eq!(bibliotheque_chargee("7f20-7f30 r-xp 0 0:0 1 /usr/lib/libc.so.6\n"), None);

    let racine = std::env::temp_dir().join(format!("kvu_rocm_chargee_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&racine);
    let opt = racine.join("opt");
    std::fs::create_dir_all(opt.join("rocm714/.info")).unwrap();
    std::fs::create_dir_all(opt.join("rocm714/lib")).unwrap();
    std::fs::write(opt.join("rocm714/.info/version"), "7.14.0\n").unwrap();
    // bibliothèque du système (sans .info) : l'installation de /opt est signalée NON chargée
    let systeme = racine.join("usr/lib/libamdhip64.so.7");
    let v = versions_rocm(Some(&systeme), &opt);
    assert_eq!(v, vec![("rocm_opt_non_chargee".to_string(), "7.14.0".to_string())]);
    // bibliothèque de /opt/rocm714 : c'est bien sa version
    let v = versions_rocm(Some(&opt.join("rocm714/lib/libamdhip64.so")), &opt);
    assert_eq!(v, vec![("rocm".to_string(), "7.14.0".to_string())]);
    let _ = std::fs::remove_dir_all(&racine);
}
