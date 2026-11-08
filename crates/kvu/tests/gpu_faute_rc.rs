//! Code de sortie de kvu quand le GPU fait une faute d'exécution (lot gpu-rc), SANS GPU.
//!
//! Contexte : le 01/11/2026 une faute mémoire du GPU (« Queue error - HSA_STATUS_ERROR_MEMORY_FAULT », puis
//! « hipStreamSynchronize: an illegal memory access was encountered (700) ») a été prise pour un `rc = 0`. Ces essais lancent le VRAI
//! binaire `kvu`, son vrai backend HIP, contre un FAUX runtime HIP (`tests/faux_hip/faux_hip.c`, compilé ici puis trouvé par
//! `LD_LIBRARY_PATH` : aucun calcul, aucun GPU) qui rend les codes d'erreur d'un runtime fautif, et vérifient le code de sortie du
//! processus, le message de stderr et le manifeste. Ils vérifient aussi l'injection de pannes de kvu lui-même
//! (`KVU_INJECTION_FAUTE_GPU`), qui emprunte le même chemin d'erreur.
//!
//! Les essais « de base » (commandes qui lisent leurs résultats) passaient déjà avant le lot : une erreur relue à la synchronisation ou
//! à la copie donnait le code 3. Ce que le lot ajoute, et que seuls les essais « sortie » et « manifeste » exigent : une faute qu'aucun
//! appel de la commande ne relit n'est plus perdue (dernière synchronisation de l'appareil, contrôle de sortie du processus), et le
//! manifeste d'un run repris ne dit plus « terminé, rc 0 » à côté de la raison d'un échec.
//!
//! Le faux runtime est aussi un DÉTECTEUR DE DURÉE DE VIE : il compte les opérations asynchrones encore en vol et signale (`viol=`) tout
//! `hipFree` ou `hipModuleUnload` qui arrive avant qu'une attente de l'appareil les ait terminées. Sur un vrai GPU, un tampon libéré sous un
//! noyau en cours (faute de page, client TCP) ou un module déchargé sous un noyau en cours (client SQC) est une faute que le processus peut ne
//! jamais voir, puisque tout est démonté après l'écriture des résultats. Chaque tampon et chaque module de kvu attend donc l'appareil avant
//! d'être libéré : aucun essai ne doit relever de violation.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Texte de la faute que rend le runtime à la synchronisation (incident du 01/11/2026).
const FAUTE_700: &str = "an illegal memory access was encountered (700)";

/// Compile le faux runtime (une fois par processus de test) et rend son dossier. Les deux noms que kvu charge (`libamdhip64`,
/// `libhiprtc`) y sont, sous tous les noms de la liste de candidats : aucun ne peut mener au vrai runtime.
fn faux_runtime() -> &'static Path {
    static DOSSIER: OnceLock<PathBuf> = OnceLock::new();
    DOSSIER.get_or_init(|| {
        let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("faux_hip_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/faux_hip/faux_hip.c");
        let so = d.join("libamdhip64.so");
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let o = Command::new(&cc)
            .args(["-shared", "-fPIC", "-O1", "-Wall", "-o"])
            .arg(&so)
            .arg(&source)
            .output()
            .unwrap_or_else(|e| panic!("compilateur C « {cc} » introuvable : {e}"));
        assert!(o.status.success(), "compilation du faux runtime : {}", String::from_utf8_lossy(&o.stderr));
        for nom in ["libhiprtc.so", "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5", "libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so.5"] {
            fs::copy(&so, d.join(nom)).unwrap();
        }
        // Garde-fou : le faux runtime doit se charger ICI, sinon kvu retomberait sur le vrai runtime de la machine (et toucherait un vrai
        // GPU) au lieu d'échouer. Le chargement du test lui-même ne touche à aucun GPU : le faux runtime n'a que des fonctions factices.
        // SAFETY : bibliothèque compilée à l'instant depuis tests/faux_hip/faux_hip.c.
        let lib = unsafe { libloading::Library::new(&so) }.unwrap_or_else(|e| panic!("le faux runtime ne se charge pas : {e}"));
        // SAFETY : symbole de la même bibliothèque ; signature C `int hipInit(unsigned)`.
        let init: libloading::Symbol<unsafe extern "C" fn(u32) -> i32> =
            unsafe { lib.get(b"hipInit\0") }.expect("hipInit absent du faux runtime");
        assert_eq!(unsafe { init(0) }, 0);
        std::mem::forget(lib);
        d
    })
}

/// Appels du faux runtime, comptés à sa sortie.
#[derive(Clone, Copy, Debug)]
struct Comptes {
    /// `hipStreamSynchronize` : les attentes de la commande, plus celle de la destruction de l'appareil.
    sync: u64,
    /// `hipDeviceSynchronize` : l'attente qui précède chaque libération de tampon, chaque déchargement de module, et la destruction de l'appareil.
    devsync: u64,
    dtoh: u64,
    htod: u64,
    launch: u64,
    free: u64,
    unload: u64,
    /// `hipFree` ou `hipModuleUnload` arrivés avec des opérations asynchrones encore en vol.
    viol: u64,
}

/// Résultat d'un lancement de kvu contre le faux runtime.
struct Essai {
    rc: i32,
    out: String,
    err: String,
    comptes: Option<Comptes>,
}

impl Essai {
    fn comptes(&self) -> Comptes {
        self.comptes.expect("le faux runtime n'a pas été chargé (fichier de comptes absent)")
    }
    fn syncs(&self) -> u64 {
        self.comptes().sync
    }
}

fn lire_comptes(chemin: &Path) -> Option<Comptes> {
    let t = fs::read_to_string(chemin).ok()?;
    let ligne = t.lines().next()?;
    let champ = |nom: &str| -> u64 {
        ligne
            .split_whitespace()
            .find_map(|m| m.strip_prefix(&format!("{nom}=")).and_then(|v| v.parse().ok()))
            .unwrap_or_else(|| panic!("comptes illisibles : {ligne}"))
    };
    Some(Comptes {
        sync: champ("sync"),
        devsync: champ("devsync"),
        dtoh: champ("dtoh"),
        htod: champ("htod"),
        launch: champ("launch"),
        free: champ("free"),
        unload: champ("unload"),
        viol: champ("viol"),
    })
}

/// Lance `kvu <args>` avec le faux runtime. `panne` : `FAUX_HIP_PANNE` ; `extra` : variables d'environnement en plus.
fn lancer(args: &[&str], panne: Option<&str>, extra: &[(&str, &str)]) -> Essai {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dossier = faux_runtime();
    let comptes = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("faux_hip_comptes_{}_{}.txt", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
    let _ = fs::remove_file(&comptes);
    let ld = match std::env::var("LD_LIBRARY_PATH") {
        Ok(v) if !v.is_empty() => format!("{}:{v}", dossier.display()),
        _ => dossier.display().to_string(),
    };
    let mut c = Command::new(env!("CARGO_BIN_EXE_kvu"));
    c.args(args)
        .env("LD_LIBRARY_PATH", ld)
        .env("KEVINOTRON_CACHE_DISABLE", "1")
        // une architecture à mémoire non unifiée : l'admission mémoire ne dépend pas de la machine qui exécute l'essai
        .env("KEVINOTRON_HIP_ARCH", "gfx90a")
        .env("FAUX_HIP_COMPTES", &comptes)
        .env_remove("FAUX_HIP_PANNE")
        .env_remove("KVU_INJECTION_FAUTE_GPU");
    if let Some(p) = panne {
        c.env("FAUX_HIP_PANNE", p);
    }
    for (k, v) in extra {
        c.env(k, v);
    }
    let o = c.output().expect("lancement de kvu");
    let essai = Essai {
        rc: o.status.code().unwrap_or(-1),
        out: String::from_utf8_lossy(&o.stdout).into_owned(),
        err: String::from_utf8_lossy(&o.stderr).into_owned(),
        comptes: lire_comptes(&comptes),
    };
    let _ = fs::remove_file(&comptes);
    essai
}

/// Commande GPU la plus courte : un noyau, deux lectures. Chemin HIP réel de kvu (allocation, compilation, lancement, copies).
const TEST_HB: &[&str] = &["su2", "test-hb", "1.0", "64", "--backend", "hip"];

fn test_hb(panne: Option<&str>, extra: &[(&str, &str)]) -> Essai {
    lancer(TEST_HB, panne, extra)
}

/// Variables qui rendent un run `su2 gen` indépendant de l'hôte et de la production : pas de verrou GPU de la machine (le faux
/// runtime ne partage rien), pas de seuil de disque ni de mémoire.
const ENV_RUN: &[(&str, &str)] =
    &[("KEVINOTRON_VERROU", "ignorer"), ("KEVINOTRON_SSD_MIN_GO", "0"), ("KEVINOTRON_RAM_MIN_GO", "0")];

fn dossier_essai(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("gpu_faute_rc").join(format!("{nom}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.parent().unwrap()).unwrap();
    d
}

fn manifeste(d: &Path) -> serde_json::Value {
    serde_json::from_str(&fs::read_to_string(d.join("manifest.json")).unwrap()).unwrap()
}

// ------------------------------------------------------------------------------------------------------------------
// Témoin
// ------------------------------------------------------------------------------------------------------------------

#[test]
fn temoin_sans_faute_kvu_rend_0_et_le_faux_runtime_est_bien_celui_qui_sert() {
    let e = test_hb(None, &[]);
    assert_eq!(e.rc, 0, "stdout : {}\nstderr : {}", e.out, e.err);
    assert!(e.out.contains("TESTHB alpha=1 n=64"), "{}", e.out);
    assert!(e.err.is_empty(), "stderr non vide sans faute : {}", e.err);
    // le noyau a été lancé et les résultats relus par le faux runtime (zéros) : ce n'est pas un vrai GPU
    let c = e.comptes();
    assert!(c.dtoh == 2 && c.launch == 1, "{c:?}");
    // les rangs d'injection des essais suivants (`synchronisation:5`, `copie:3`, `liberation:2`) doivent exister dans cette commande
    assert!(c.sync >= 5 && c.htod + c.dtoh >= 3 && c.devsync >= 2, "la commande n'a plus assez d'appels pour les essais d'injection : {c:?}");
    // aucun tampon ni module n'est libéré avant que l'appareil ait fini, et CHAQUE libération est précédée de son attente
    assert_eq!(c.viol, 0, "libération sous un noyau en vol : {c:?} ; stderr : {}", e.err);
    assert!(c.free >= 1 && c.unload >= 1, "{c:?}");
    assert!(c.devsync > c.free + c.unload, "chaque libération doit être précédée d'une attente de l'appareil : {c:?}");
}

// ------------------------------------------------------------------------------------------------------------------
// Fautes du runtime, rendues à la commande (comportement de base, conservé)
// ------------------------------------------------------------------------------------------------------------------

#[test]
fn faute_d_un_noyau_relevee_a_la_synchronisation_donne_3_et_un_message_clair() {
    // le noyau « fait sa faute » en s'exécutant ; le premier appel qui attend le GPU la rapporte (incident du 01/11 19:51)
    let e = test_hb(Some("apres_launch:1:700"), &[]);
    assert_eq!(e.rc, 3, "stdout : {}\nstderr : {}", e.out, e.err);
    assert!(e.err.contains("Queue error - HSA_STATUS_ERROR_MEMORY_FAULT"), "{}", e.err);
    assert!(e.err.contains(&format!("kvu su2 : GPU : hipStreamSynchronize: {FAUTE_700}")), "{}", e.err);
    assert!(e.err.contains("faute d'exécution GPU (code de sortie 3)"), "{}", e.err);
    assert!(!e.out.contains("TESTHB"), "aucun résultat ne doit sortir d'une commande fautive : {}", e.out);
}

#[test]
fn faute_relevee_a_la_copie_donne_3() {
    // incident du 26/10 (WP10, R54-b16-c2) : c'est la copie device -> hôte qui rapporte la faute
    let e = test_hb(Some("dtoh:1:700"), &[]);
    assert_eq!(e.rc, 3, "stdout : {}\nstderr : {}", e.out, e.err);
    assert!(e.err.contains(&format!("kvu su2 : GPU : hipMemcpyDtoHAsync : {FAUTE_700}")), "{}", e.err);
    assert!(!e.out.contains("TESTHB"), "{}", e.out);
}

#[test]
fn echec_de_lancement_donne_3() {
    let e = test_hb(Some("launch:1:719"), &[]);
    assert_eq!(e.rc, 3, "stdout : {}\nstderr : {}", e.out, e.err);
    assert!(e.err.contains("kvu su2 : GPU : hipModuleLaunchKernel: unspecified launch failure (719)"), "{}", e.err);
    assert!(!e.out.contains("TESTHB"), "{}", e.out);
}

#[test]
fn faute_collante_donne_3_une_seule_fois_sans_boucler() {
    // une fois le runtime en faute, chaque appel suivant échoue : la commande s'arrête à la première erreur
    let e = test_hb(Some("apres_launch:1:700:collant"), &[]);
    assert_eq!(e.rc, 3, "{}", e.err);
    assert_eq!(e.err.matches("kvu su2 : GPU :").count(), 1, "{}", e.err);
}

// ------------------------------------------------------------------------------------------------------------------
// Ce que le lot ajoute : la faute qu'aucun appel de la commande ne relit
// ------------------------------------------------------------------------------------------------------------------

#[test]
fn faute_que_seule_la_derniere_attente_de_l_appareil_revele_donne_3() {
    // Témoin : la commande se déroule sans faute ; sa DERNIÈRE attente de l'appareil (hipDeviceSynchronize) est celle de la destruction de
    // l'appareil, après toutes les libérations.
    let temoin = test_hb(None, &[]);
    let derniere = temoin.comptes().devsync;
    // La faute n'apparaît qu'à cette dernière attente : la commande a déjà lu ses résultats, affiché sa ligne TESTHB, libéré ses tampons
    // et rendrait 0 si rien ne relisait l'erreur.
    let e = test_hb(Some(&format!("devsync:{derniere}:700")), &[]);
    assert!(e.out.contains("TESTHB alpha=1 n=64"), "la commande avait fini normalement : {}", e.out);
    assert_eq!(e.rc, 3, "faute passée sous silence : rc {} ; stderr : {}", e.rc, e.err);
    assert!(e.err.contains("FAUTE D'EXÉCUTION GPU relevée pendant la commande, qui rendait pourtant 0"), "{}", e.err);
    assert!(e.err.contains(&format!("fin d'appareil HIP : hipDeviceSynchronize : {FAUTE_700}")), "{}", e.err);
    assert!(e.err.contains("code de sortie forcé à 3"), "{}", e.err);
}

#[test]
fn faute_collante_a_la_derniere_attente_n_est_signalee_qu_une_fois() {
    // Même scénario, faute collante : tout ce qui suit la panne échoue aussi ; le code reste 3 et la faute est écrite une seule fois
    let temoin = test_hb(None, &[]);
    let derniere = temoin.comptes().devsync;
    let e = test_hb(Some(&format!("devsync:{derniere}:700:collant")), &[]);
    assert_eq!(e.rc, 3, "{}", e.err);
    assert_eq!(e.err.matches("FAUTE D'EXÉCUTION GPU relevée").count(), 1, "{}", e.err);
}

// ------------------------------------------------------------------------------------------------------------------
// Libération des tampons et déchargement des modules : toujours APRÈS l'attente de l'appareil
// ------------------------------------------------------------------------------------------------------------------

#[test]
fn faute_a_l_attente_qui_precede_la_liberation_d_un_tampon_donne_3() {
    // La première attente de l'appareil est celle qui précède la libération du premier tampon, après que la commande a fini et affiché
    // son résultat : un noyau encore en cours à ce moment-là est la faute de page que rien d'autre ne rapporte.
    let e = test_hb(Some("devsync:1:700"), &[]);
    assert!(e.out.contains("TESTHB alpha=1 n=64"), "la commande avait fini normalement : {}", e.out);
    assert_eq!(e.rc, 3, "faute passée sous silence : rc {} ; stderr : {}", e.rc, e.err);
    assert!(e.err.contains(&format!("avant hipFree : hipDeviceSynchronize : {FAUTE_700}")), "{}", e.err);
    assert!(e.err.contains("FAUTE D'EXÉCUTION GPU relevée pendant la commande, qui rendait pourtant 0"), "{}", e.err);
}

#[test]
fn faute_a_l_attente_qui_precede_le_dechargement_du_module_donne_3() {
    // Le déchargement du module est précédé de sa propre attente : on la vise par son rang (la dernière de la commande hors destruction
    // de l'appareil est celle du module ou d'un tampon libéré après lui ; on les essaie toutes, une par une : chacune doit donner 3).
    let temoin = test_hb(None, &[]);
    let c = temoin.comptes();
    assert!(c.unload >= 1 && c.devsync > c.free + c.unload, "{c:?}");
    for rang in 1..=c.devsync {
        let e = test_hb(Some(&format!("devsync:{rang}:700")), &[]);
        assert_eq!(e.rc, 3, "attente n° {rang} sur {} : faute passée sous silence ; stderr : {}", c.devsync, e.err);
        assert!(e.err.contains(&format!("hipDeviceSynchronize : {FAUTE_700}")), "attente n° {rang} : {}", e.err);
    }
}

#[test]
fn injection_de_liberation_donne_3() {
    let temoin = test_hb(None, &[]);
    let c = temoin.comptes();
    // le rang de la première attente et celui de la dernière (destruction de l'appareil) : le chemin d'erreur est le même
    for rang in [1, 2, c.free + c.unload + 1] {
        let spec = format!("liberation:{rang}");
        let e = test_hb(None, &[("KVU_INJECTION_FAUTE_GPU", spec.as_str())]);
        assert!(e.out.contains("TESTHB alpha=1 n=64"), "{spec} : la commande avait fini normalement : {}", e.out);
        assert_eq!(e.rc, 3, "{spec} : faute passée sous silence ; stderr : {}", e.err);
        assert!(e.err.contains(&format!("[injection KVU_INJECTION_FAUTE_GPU={spec}]")), "{spec} : {}", e.err);
        assert!(e.err.contains("FAUTE D'EXÉCUTION GPU relevée pendant la commande, qui rendait pourtant 0"), "{spec} : {}", e.err);
        assert_eq!(e.comptes().viol, 0, "{spec} : {:?}", e.comptes());
    }
    // un rang au-delà des libérations de la commande : rien n'est injecté
    let e = test_hb(None, &[("KVU_INJECTION_FAUTE_GPU", "liberation:999")]);
    assert_eq!(e.rc, 0, "{}", e.err);
}

/// `cond facteur3 repliques` (T3-D) sur le faux GPU : une cellule SU(2) de L = 6, thermalisation de Wilson, deux répliques.
fn cond3(sortie: &Path) -> Vec<String> {
    [
        "cond", "facteur3", "repliques", "--groupe", "su2", "--d", "4", "--l", "6", "--beta", "2.3", "--wilson", "--therm", "4", "--espacement", "2",
        "--graine-a", "1", "--graine-b", "2", "--n-dec", "4", "--n-mes", "20", "--backend", "hip", "--sortie",
    ]
    .iter()
    .map(|s| s.to_string())
    .chain([sortie.to_str().unwrap().to_string()])
    .collect()
}

#[test]
fn une_erreur_en_cours_de_commande_ne_libere_aucun_tampon_sous_un_noyau_en_vol() {
    // Témoin : la commande va au bout ; elle synchronise avant de rendre ses tampons.
    let dt = dossier_essai("cond3_temoin");
    let t = lancer(&refs(&cond3(&dt)), None, ENV_RUN);
    assert_eq!(t.rc, 0, "stdout : {}\nstderr : {}", t.out, t.err);
    assert_eq!(t.comptes().viol, 0, "{:?}", t.comptes());
    // Un lancement échoue au 200e : l'erreur court-circuite la fin de la commande, et le déroulement des `Drop` libère les tampons alors que
    // 199 noyaux lancés avant elle ne sont pas attendus (sans attente préalable, `hipFree` partirait sous des noyaux en vol : faute de page).
    let d = dossier_essai("cond3_faute");
    let e = lancer(&refs(&cond3(&d)), Some("launch:200:719"), ENV_RUN);
    assert_eq!(e.rc, 3, "stdout : {}\nstderr : {}", e.out, e.err);
    assert!(e.err.contains("kvu cond : GPU : hipModuleLaunchKernel: unspecified launch failure (719)"), "{}", e.err);
    let c = e.comptes();
    assert_eq!(c.viol, 0, "libération sous un noyau en vol pendant le déroulement après l'erreur : {c:?} ; stderr : {}", e.err);
    assert!(c.free >= 10 && c.devsync > c.free + c.unload, "chaque libération doit être précédée de son attente : {c:?}");
    // les mêmes essais, faute à la première opération qui attend l'appareil après un noyau fautif : la même garantie
    let d2 = dossier_essai("cond3_faute_sync");
    let e2 = lancer(&refs(&cond3(&d2)), Some("apres_launch:150:700"), ENV_RUN);
    assert_eq!(e2.rc, 3, "stdout : {}\nstderr : {}", e2.out, e2.err);
    assert_eq!(e2.comptes().viol, 0, "{:?} ; stderr : {}", e2.comptes(), e2.err);
}

// ------------------------------------------------------------------------------------------------------------------
// Injection de pannes de kvu (KVU_INJECTION_FAUTE_GPU) : même chemin d'erreur, sans faute du runtime
// ------------------------------------------------------------------------------------------------------------------

#[test]
fn injection_au_lancement_a_la_synchronisation_et_a_la_copie_donne_3() {
    for (spec, texte) in [
        ("lancement:1", "unspecified launch failure (719)"),
        ("synchronisation:2", FAUTE_700),
        ("synchronisation:5", FAUTE_700),
        ("copie:1", FAUTE_700),
        ("copie:3", FAUTE_700),
    ] {
        let e = test_hb(None, &[("KVU_INJECTION_FAUTE_GPU", spec)]);
        assert_eq!(e.rc, 3, "{spec} : stdout : {}\nstderr : {}", e.out, e.err);
        assert!(e.err.contains(&format!("[injection KVU_INJECTION_FAUTE_GPU={spec}]")), "{spec} : {}", e.err);
        assert!(e.err.contains(texte), "{spec} : {}", e.err);
        assert!(e.err.contains("kvu su2 : GPU :"), "{spec} : {}", e.err);
        assert!(!e.out.contains("TESTHB"), "{spec} : aucun résultat ne doit sortir : {}", e.out);
    }
}

#[test]
fn injection_a_un_rang_jamais_atteint_laisse_la_commande_se_derouler() {
    // une panne programmée plus loin que la commande ne va : rien n'est injecté (et rien n'est simulé non plus)
    let e = test_hb(None, &[("KVU_INJECTION_FAUTE_GPU", "lancement:99")]);
    assert_eq!(e.rc, 0, "stdout : {}\nstderr : {}", e.out, e.err);
    assert!(e.out.contains("TESTHB alpha=1 n=64"), "{}", e.out);
}

#[test]
fn injection_de_sortie_sur_une_commande_avec_gpu_donne_3() {
    let e = test_hb(None, &[("KVU_INJECTION_FAUTE_GPU", "sortie")]);
    assert!(e.out.contains("TESTHB alpha=1 n=64"), "la commande avait fini normalement : {}", e.out);
    assert_eq!(e.rc, 3, "stderr : {}", e.err);
    assert!(e.err.contains("[injection KVU_INJECTION_FAUTE_GPU=sortie] contrôle de sortie"), "{}", e.err);
}

#[test]
fn injection_de_sortie_sur_une_commande_sans_gpu_donne_3() {
    // `su2 selftest` n'ouvre aucun GPU et rend 0 : le contrôle de sortie du processus ne dépend pas d'un appareil
    let sans = Command::new(env!("CARGO_BIN_EXE_kvu")).args(["su2", "selftest"]).env_remove("KVU_INJECTION_FAUTE_GPU").output().unwrap();
    assert_eq!(sans.status.code(), Some(0), "témoin : {}", String::from_utf8_lossy(&sans.stderr));
    let o = Command::new(env!("CARGO_BIN_EXE_kvu"))
        .args(["su2", "selftest"])
        .env("KVU_INJECTION_FAUTE_GPU", "sortie")
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(3), "stderr : {}", String::from_utf8_lossy(&o.stderr));
    assert!(String::from_utf8_lossy(&o.stdout).contains("SELFTEST OK"), "la commande avait réussi");
    let e = String::from_utf8_lossy(&o.stderr);
    assert!(e.contains("FAUTE D'EXÉCUTION GPU relevée pendant la commande, qui rendait pourtant 0"), "{e}");
    assert!(e.contains("code de sortie forcé à 3"), "{e}");
}

#[test]
fn injection_mal_ecrite_refuse_la_commande_en_usage_sans_l_executer() {
    for mauvais in ["bidule", "lancement:0", "synchronisation:x", "sortie:2"] {
        let o = Command::new(env!("CARGO_BIN_EXE_kvu"))
            .args(["su2", "selftest"])
            .env("KVU_INJECTION_FAUTE_GPU", mauvais)
            .output()
            .unwrap();
        let (out, err) = (String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
        assert_eq!(o.status.code(), Some(2), "{mauvais} : stderr : {err}");
        assert!(err.contains("KVU_INJECTION_FAUTE_GPU"), "{mauvais} : {err}");
        assert!(!out.contains("SELFTEST"), "{mauvais} : la commande ne doit pas avoir été exécutée : {out}");
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Run su2 gen : manifeste honnête après une faute, puis clôture à la reprise
// ------------------------------------------------------------------------------------------------------------------

/// Arguments d'un petit run `su2 gen` (L = 2, 20 pas, point de reprise tous les 5 pas) vers `outdir`.
fn gen(outdir: &Path) -> Vec<String> {
    ["su2", "gen", "--backend", "hip", "--L", "2", "--beta", "2.3", "--graine", "1", "--therm", "10", "--nmeas", "10", "--skip", "1", "--nor", "1", "--reprise-every", "5", "--outdir"]
        .iter()
        .map(|s| s.to_string())
        .chain([outdir.to_str().unwrap().to_string()])
        .collect()
}

fn refs(v: &[String]) -> Vec<&str> {
    v.iter().map(String::as_str).collect()
}

#[test]
fn run_interrompu_par_une_faute_gpu_laisse_un_manifeste_echec_puis_se_cloture_termine_a_la_reprise() {
    let d = dossier_essai("gen_faute");
    // fenêtre 1 : le noyau fait sa faute au 300e lancement (en cours de thermalisation, après un point de reprise)
    let e1 = lancer(&refs(&gen(&d)), Some("apres_launch:300:700"), ENV_RUN);
    assert_eq!(e1.rc, 3, "stdout : {}\nstderr : {}", e1.out, e1.err);
    assert!(e1.err.contains(&format!("kvu su2 : GPU : hipStreamSynchronize: {FAUTE_700}")), "{}", e1.err);
    let m1 = manifeste(&d);
    assert_eq!(m1["statut"], "echec", "{m1}");
    assert_eq!(m1["rc"], 3, "{m1}");
    assert!(m1["raison_echec"].as_str().unwrap().contains("illegal memory access"), "{m1}");
    assert!(!d.join("COMPLETE").exists(), "COMPLETE ne s'écrit pas après une faute");
    assert!(d.join("reprise.json").exists(), "un point de reprise antérieur à la faute doit exister");

    // fenêtre 2 : reprise sans faute ; le run va au bout
    let reprise = ["su2", "gen", "--backend", "hip", "--reprendre", d.to_str().unwrap()];
    let e2 = lancer(&reprise, None, ENV_RUN);
    assert_eq!(e2.rc, 0, "stdout : {}\nstderr : {}", e2.out, e2.err);
    assert!(d.join("COMPLETE").exists());
    let m2 = manifeste(&d);
    assert_eq!(m2["statut"], "termine", "{m2}");
    assert_eq!(m2["rc"], 0, "{m2}");
    // « termine, rc 0 » ne reste pas à côté de la raison d'un échec ; l'historique de la faute est conservé
    assert!(m2.get("raison_echec").is_none(), "manifeste contradictoire (termine, rc 0 ET raison d'échec) : {m2}");
    let anterieures = m2["fautes_anterieures"].as_array().unwrap_or_else(|| panic!("historique des fautes absent : {m2}"));
    assert_eq!(anterieures.len(), 1, "{m2}");
    assert_eq!(anterieures[0]["rc"], 3);
    assert!(anterieures[0]["raison_echec"].as_str().unwrap().contains("illegal memory access"), "{m2}");
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn faute_a_la_derniere_synchronisation_avant_la_cloture_du_run_ne_s_ecrit_pas_termine() {
    // Témoin : nombre de synchronisations de flux d'un run complet ; la dernière est la synchronisation de fin de run, avant la clôture du
    // manifeste (celle de la destruction de l'appareil est une attente de tout l'appareil, comptée à part).
    let dt = dossier_essai("gen_temoin");
    let temoin = lancer(&refs(&gen(&dt)), None, ENV_RUN);
    assert_eq!(temoin.rc, 0, "stdout : {}\nstderr : {}", temoin.out, temoin.err);
    assert!(dt.join("COMPLETE").exists());
    let n = temoin.syncs();
    fs::remove_dir_all(&dt).unwrap();

    let d = dossier_essai("gen_derniere_sync");
    let e = lancer(&refs(&gen(&d)), Some(&format!("sync:{n}:700")), ENV_RUN);
    assert_eq!(e.rc, 3, "stdout : {}\nstderr : {}", e.out, e.err);
    let m = manifeste(&d);
    assert_eq!(m["statut"], "echec", "le run ne doit pas être clos « termine » : {m}");
    assert_eq!(m["rc"], 3, "{m}");
    assert!(!d.join("COMPLETE").exists(), "COMPLETE ne s'écrit pas quand la dernière synchronisation échoue");
    fs::remove_dir_all(&d).unwrap();
}
