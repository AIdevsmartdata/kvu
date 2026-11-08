//! Détection de la machine et des GPU, choix du backend. Propriétaire : T1.
//!
//! `detecter()` lit /proc (hôte, CPU, RAM) puis sonde les GPU par dlopen (voir `sonde_hip` et
//! `sonde_cuda`) sans créer de contexte. Elle ne panique jamais : un champ illisible reste vide.
//! Toutes les décisions (identification de la machine, choix du backend) sont des fonctions PURES
//! de leurs entrées, donc testables sur des profils simulés.

use super::{sonde_cuda, sonde_hip};
use crate::commun::{BackendKind, Erreur, Machine, Resultat};
use std::path::Path;

/// Un GPU vu par les sondes (dlopen des bibliothèques runtime, sans créer de contexte lourd).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GpuDetecte {
    pub backend: BackendKind,
    /// Nom commercial (« AMD Radeon 8060S », « NVIDIA GeForce RTX 5060 Ti »).
    pub nom: String,
    /// Architecture cible de compilation : « gfx1151 », « sm_120 ».
    pub arch: String,
    pub memoire_go: f64,
    /// Rapport débit FP32 / débit FP64 (16 pour gfx1151, 64 pour sm_120 grand public).
    pub ratio_fp32_fp64: u32,
    /// Mémoire unifiée avec l'hôte (APU) : la GTT compte, pas une VRAM dédiée.
    pub memoire_unifiee: bool,
}

/// Profil complet de la machine courante. Ne panique jamais ; un champ inconnu reste vide.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProfilMachine {
    pub machine: Machine,
    pub hote: String,
    pub cpu_modele: String,
    pub fils_logiques: usize,
    pub ram_totale_go: f64,
    pub ram_disponible_go: f64,
    pub gpus: Vec<GpuDetecte>,
    /// Versions runtime : ("rocm", "7.1.x"), ("hip_runtime", ..), ("cuda_pilote", "13.2"), ("nvrtc", "13.3")…
    pub versions: Vec<(String, String)>,
    /// Surcharge KEVINOTRON_MACHINE REFUSÉE (valeur invalide, ou machine différente de celle du matériel) :
    /// le profil garde la machine du matériel et la ligne de commande refuse de s'exécuter (rc 2).
    #[serde(default)]
    pub surcharge_refusee: Option<String>,
}

pub use super::compat_tests_existants::HOTE_GAMER;
const GIB: f64 = (1u64 << 30) as f64;

// ------------------------------------------------------------------------------------------
// Analyse de /proc (pures)
// ------------------------------------------------------------------------------------------

/// (modèle du premier « model name », nombre de lignes « processor »).
pub fn analyser_cpuinfo(texte: &str) -> (String, usize) {
    let mut modele = String::new();
    let mut fils = 0;
    for ligne in texte.lines() {
        let Some((cle, val)) = ligne.split_once(':') else {
            continue;
        };
        match cle.trim() {
            "processor" => fils += 1,
            "model name" if modele.is_empty() => modele = val.trim().to_string(),
            _ => {}
        }
    }
    (modele, fils.max(1))
}

/// (MemTotal, MemAvailable) en Gio. MemAvailable absent (vieux noyau) : MemFree + Buffers + Cached.
pub fn analyser_meminfo(texte: &str) -> (f64, f64) {
    let lire = |cle: &str| -> Option<f64> {
        texte.lines().find_map(|l| {
            let (k, v) = l.split_once(':')?;
            (k.trim() == cle)
                .then(|| v.split_whitespace().next()?.parse::<f64>().ok())
                .flatten()
        })
    };
    let ko_en_go = |ko: f64| ko * 1024.0 / GIB;
    let total = lire("MemTotal").map(ko_en_go).unwrap_or(0.0);
    let dispo = lire("MemAvailable")
        .or_else(|| Some(lire("MemFree")? + lire("Buffers").unwrap_or(0.0) + lire("Cached").unwrap_or(0.0)))
        .map(ko_en_go)
        .unwrap_or(0.0);
    (total, dispo)
}

/// RAM disponible en Gio d'après /proc/meminfo (0 si illisible).
pub fn ram_disponible_go() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .map(|t| analyser_meminfo(&t).1)
        .unwrap_or(0.0)
}

/// Rapport FP32/FP64 : 16 pour gfx1151, 64 pour sm_120 et les GeForce, 2 par défaut.
pub fn ratio_fp32_fp64(arch: &str, nom: &str) -> u32 {
    if arch == "gfx1151" {
        16
    } else if arch == "sm_120" || nom.to_ascii_lowercase().contains("geforce") {
        64
    } else {
        2
    }
}

/// Identification de la machine : l'hôte, OU le CPU, OU un GPU caractéristique, selon les motifs des profils de
/// `kvu.toml` (clés `hotes`, `cpu`, `gpu_arch`). Aucun motif ne correspond : profil `inconnue`.
pub fn identifier_machine(hote: &str, cpu: &str, gpus: &[GpuDetecte]) -> Machine {
    let archs: Vec<&str> = gpus.iter().map(|g| g.arch.as_str()).collect();
    crate::config::actuelle().identifier(hote, cpu, &archs)
}

/// Mémoire GPU partagée avec l'hôte (APU) : l'allocation d'un tampon GPU consomme la RAM du système.
pub fn arch_memoire_unifiee(arch: &str) -> bool {
    arch == "gfx1151"
}

/// Contrôle PUR de la surcharge KEVINOTRON_PROFIL (ou KEVINOTRON_MACHINE) face à la machine identifiée par le matériel.
///
/// La politique (verrou GPU, affinité, seuils) doit suivre le GPU RÉELLEMENT utilisé : un profil de machine
/// différent du matériel ferait tourner le GPU sous le verrou de l'autre machine, donc SANS exclusion avec la
/// production (vérificateur du 04/10). Règles :
/// - absente ou vide ⇒ la machine du matériel ;
/// - invalide ⇒ refus (une faute de frappe ne doit pas passer en silence) ;
/// - un profil générique (`inconnue`, `location`) ⇒ accepté partout (politique prudente ; le verrou et l'affinité
///   restent ceux du matériel, voir `politique_avec`) ;
/// - une machine CONNUE différente de celle du matériel ⇒ refus (seule une machine non identifiée peut se
///   faire passer pour une autre).
pub fn controler_surcharge(materielle: Machine, surcharge: Option<&str>) -> Resultat<Machine> {
    let Some(s) = surcharge.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(materielle);
    };
    let demandee: Machine = s.parse().map_err(|_| {
        Erreur::Config(format!(
            "KEVINOTRON_MACHINE=« {s} » invalide : valeurs {}",
            Machine::noms_possibles()
        ))
    })?;
    if demandee == materielle || demandee.est_generique() || materielle.est_generique() {
        return Ok(demandee);
    }
    Err(Erreur::Config(format!(
        "KEVINOTRON_MACHINE={} sur la machine {} : le verrou GPU, l'affinité et la garde SSD ne suivraient plus le \
         GPU réellement utilisé (refus ; seuls les profils génériques « {} » et « {} » sont admis ici)",
        demandee.nom(),
        materielle.nom(),
        Machine::Inconnue.nom(),
        Machine::Location.nom()
    )))
}

/// Lit un fichier de /proc sous `racine` (un chemin simulé dans les tests).
fn lire(racine: &Path, rel: &str) -> String {
    std::fs::read_to_string(racine.join(rel)).unwrap_or_default()
}

/// Construit un profil depuis une racine « /proc » simulée (`racine/cpuinfo`, `racine/meminfo`,
/// `racine/sys/kernel/hostname`), des GPU et versions donnés, et une surcharge éventuelle de la
/// machine (KEVINOTRON_MACHINE). Fonction de base de `detecter` : testée sur des fichiers fixes.
pub fn profil_depuis(
    racine: &Path,
    gpus: Vec<GpuDetecte>,
    mut versions: Vec<(String, String)>,
    surcharge: Option<&str>,
) -> ProfilMachine {
    let hote = lire(racine, "sys/kernel/hostname").trim().to_string();
    let (cpu_modele, fils_logiques) = analyser_cpuinfo(&lire(racine, "cpuinfo"));
    let (ram_totale_go, ram_disponible_go) = analyser_meminfo(&lire(racine, "meminfo"));
    let materielle = identifier_machine(&hote, &cpu_modele, &gpus);
    let (machine, surcharge_refusee) = match controler_surcharge(materielle, surcharge) {
        Ok(m) => (m, None),
        Err(e) => {
            let message = match e {
                Erreur::Config(m) => m,
                autre => autre.to_string(),
            };
            versions.push(("avertissement".to_string(), format!("KEVINOTRON_MACHINE refusée : {message}")));
            (materielle, Some(message))
        }
    };
    ProfilMachine {
        machine,
        hote,
        cpu_modele,
        fils_logiques,
        ram_totale_go,
        ram_disponible_go,
        gpus,
        versions,
        surcharge_refusee,
    }
}

/// Machine identifiée par le MATÉRIEL (hôte, CPU, GPU), sans tenir compte de KEVINOTRON_MACHINE.
pub fn machine_materielle(p: &ProfilMachine) -> Machine {
    identifier_machine(&p.hote, &p.cpu_modele, &p.gpus)
}

/// Profil SANS sonde GPU (hôte, CPU, RAM seulement) : quelques millisecondes, aucun dlopen.
/// Utilisé par `etat --json` (< 200 ms). L'identification par GPU n'y joue pas.
pub fn detecter_rapide() -> ProfilMachine {
    profil_depuis(
        Path::new("/proc"),
        Vec::new(),
        Vec::new(),
        surcharge_profil_env().as_deref(),
    )
}

/// Profil demandé par l'environnement : `KEVINOTRON_PROFIL`, à défaut `KEVINOTRON_MACHINE` (ancien nom conservé).
pub fn surcharge_profil_env() -> Option<String> {
    ["KEVINOTRON_PROFIL", "KEVINOTRON_MACHINE"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .find(|v| !v.trim().is_empty())
}

/// Détecte la machine (nom d'hôte, modèle de CPU, GPU) et les GPU disponibles.
/// Surcharge : KEVINOTRON_PROFIL (ou KEVINOTRON_MACHINE) = nom d'un profil.
pub fn detecter() -> ProfilMachine {
    let (mut gpus, mut versions) = sonde_hip::sonder();
    let (gpus_cuda, versions_cuda) = sonde_cuda::sonder();
    gpus.extend(gpus_cuda);
    versions.extend(versions_cuda);
    // Les runtimes sondés ont pu créer des fils sur tous les CPU : ils reprennent l'affinité du processus.
    super::processus::reappliquer_affinite();
    profil_depuis(
        Path::new("/proc"),
        gpus,
        versions,
        surcharge_profil_env().as_deref(),
    )
}

// ------------------------------------------------------------------------------------------
// Choix du backend
// ------------------------------------------------------------------------------------------

/// Vrai si ce binaire sait piloter CUDA (feature `cuda`).
pub fn cuda_compile() -> bool {
    cfg!(feature = "cuda")
}

fn gpu_de(p: &ProfilMachine, b: BackendKind) -> Option<&GpuDetecte> {
    p.gpus.iter().find(|g| g.backend == b)
}

/// Décision pure : (backend, raison). `demande` = valeur de KEVINOTRON_BACKEND (« » = auto) ;
/// `cuda_ok` = ce binaire sait piloter CUDA.
///
/// - forcé mais indisponible ⇒ `Erreur::Gpu` (jamais de repli silencieux) ;
/// - auto : le backend du profil (`backend` de `kvu.toml`), sinon le premier GPU utilisable ;
/// - un GPU présent que ce binaire ne sait pas piloter ⇒ `Erreur::Gpu` avec le conseil de recompiler ;
/// - le CPU n'est choisi en auto que s'il n'y a AUCUN GPU.
pub fn decider_backend(p: &ProfilMachine, demande: &str, cuda_ok: bool) -> Resultat<(BackendKind, String)> {
    let conseil = "recompiler avec scripts/build_auto.sh (machine CUDA : --features unifie-cuda, ou --crate kvu)";
    let demande = demande.trim().to_ascii_lowercase();
    match demande.as_str() {
        "" | "auto" => decider_auto(p, cuda_ok, conseil),
        "cpu" => Ok((BackendKind::Cpu, "forcé par KEVINOTRON_BACKEND=cpu".to_string())),
        "hip" => match gpu_de(p, BackendKind::Hip) {
            Some(g) => Ok((
                BackendKind::Hip,
                format!("forcé par KEVINOTRON_BACKEND=hip ({} {})", g.nom, g.arch),
            )),
            None => Err(Erreur::Gpu(
                "KEVINOTRON_BACKEND=hip mais aucun GPU HIP n'est visible (ROCm absent ou GPU inutilisable)".into(),
            )),
        },
        "cuda" => match gpu_de(p, BackendKind::Cuda) {
            None => Err(Erreur::Gpu(
                "KEVINOTRON_BACKEND=cuda mais aucun GPU CUDA n'est visible (pilote NVIDIA absent ou GPU inutilisable)"
                    .into(),
            )),
            Some(_) if !cuda_ok => Err(Erreur::Gpu(format!(
                "KEVINOTRON_BACKEND=cuda : ce binaire est compilé sans la feature cuda ; {conseil}"
            ))),
            Some(g) => Ok((
                BackendKind::Cuda,
                format!("forcé par KEVINOTRON_BACKEND=cuda ({} {})", g.nom, g.arch),
            )),
        },
        autre => Err(Erreur::Config(format!(
            "KEVINOTRON_BACKEND=« {autre} » : valeurs auto|hip|cuda|cpu"
        ))),
    }
}

fn decider_auto(p: &ProfilMachine, cuda_ok: bool, conseil: &str) -> Resultat<(BackendKind, String)> {
    if p.gpus.is_empty() {
        let note = if p.machine == Machine::Inconnue {
            ""
        } else {
            " (machine connue : vérifier ROCm / le pilote NVIDIA)"
        };
        return Ok((BackendKind::Cpu, format!("aucun GPU visible : calcul CPU{note}")));
    }
    let utilisable = |g: &GpuDetecte| g.backend == BackendKind::Hip || (g.backend == BackendKind::Cuda && cuda_ok);
    // Le backend préféré du profil d'abord (`backend` de kvu.toml), puis le premier GPU utilisable.
    let profil = crate::config::actuelle().profil(p.machine);
    if profil.backend == BackendKind::Cpu.nom() {
        return Ok((BackendKind::Cpu, format!("profil {} : backend cpu", p.machine.nom())));
    }
    let prefere = profil.backend_prefere();
    if let Some(b) = prefere {
        if let Some(g) = gpu_de(p, b) {
            if utilisable(g) {
                return Ok((
                    b,
                    format!("machine {} : {} {} ({})", p.machine.nom(), g.nom, g.arch, b.nom()),
                ));
            }
            return Err(Erreur::Gpu(format!(
                "GPU {} {} détecté mais ce binaire ne sait pas le piloter (compilé sans la feature cuda) : {conseil}",
                g.nom, g.arch
            )));
        }
    }
    if let Some(g) = p.gpus.iter().find(|g| utilisable(g)) {
        return Ok((
            g.backend,
            format!("premier GPU utilisable : {} {} ({})", g.nom, g.arch, g.backend.nom()),
        ));
    }
    let g = &p.gpus[0];
    Err(Erreur::Gpu(format!(
        "GPU {} {} détecté mais ce binaire ne sait pas le piloter (compilé sans la feature cuda) : {conseil}",
        g.nom, g.arch
    )))
}

/// Choix du backend : KEVINOTRON_BACKEND=auto|hip|cuda|cpu (défaut auto = cascade du profil :
/// backend du profil (`kvu.toml`), sinon le premier GPU disponible, sinon CPU).
/// Un backend FORCÉ mais indisponible est une erreur (jamais de repli silencieux).
pub fn choisir_backend(p: &ProfilMachine) -> Resultat<BackendKind> {
    explication_backend(p).map(|(b, _)| b)
}

/// Comme `choisir_backend`, avec la raison du choix (pour `doctor`).
pub fn explication_backend(p: &ProfilMachine) -> Resultat<(BackendKind, String)> {
    decider_backend(
        p,
        &std::env::var("KEVINOTRON_BACKEND").unwrap_or_default(),
        cuda_compile(),
    )
}

/// Mémoire GPU utile (Go) du premier GPU du backend, 0 s'il n'y en a pas.
pub fn memoire_gpu_go(p: &ProfilMachine, b: BackendKind) -> f64 {
    gpu_de(p, b).map_or(0.0, |g| g.memoire_go)
}

/// Plus grand L pair dont le champ SU(2) (+ un tampon de sauvegarde ×2) tient dans la moitié de la
/// mémoire `memoire_go`. Champ : 16·L⁴ réels (4 octets en F32, 8 en F64). Conseil de `doctor`.
pub fn l_max_memoire(memoire_go: f64, octets_par_reel: u32) -> usize {
    let budget = memoire_go * GIB * 0.5;
    let mut l = 2usize;
    loop {
        let suivant = l + 2;
        let besoin = 16.0 * (suivant as f64).powi(4) * octets_par_reel as f64 * 2.0;
        if besoin > budget {
            return if 16.0 * (l as f64).powi(4) * octets_par_reel as f64 * 2.0 > budget {
                0
            } else {
                l
            };
        }
        l = suivant;
    }
}
