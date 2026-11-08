//! Sonde HIP (AMD) : nombre de GPU, nom, architecture (gcnArchName), mémoire, versions.
//!
//! On s'appuie sur `crate::heritage::ffi::hiplib()` pour `hipInit` et `hipGetDeviceCount`. Les
//! symboles absents de `ffi.rs` (propriétés du GPU, version du runtime) sont chargés ICI par
//! libloading : le moteur historique n'est jamais modifié. Aucun contexte GPU n'est créé.
//! La sonde ne panique jamais : toute erreur donne « aucun GPU HIP ».

use super::detection::{arch_memoire_unifiee, ratio_fp32_fp64, GpuDetecte};
use crate::commun::BackendKind;
use std::ffi::{c_char, c_int, c_void};
use std::path::Path;

type HipErr = c_int;
type FnNom = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> HipErr;
type FnMemoire = unsafe extern "C" fn(*mut usize, c_int) -> HipErr;
type FnProprietes = unsafe extern "C" fn(*mut c_void, c_int) -> HipErr;
type FnVersion = unsafe extern "C" fn(*mut c_int) -> HipErr;

/// Candidats dlopen de la bibliothèque runtime (mêmes noms que `heritage::ffi`).
const CANDIDATS: &[&str] = &[
    "libamdhip64.so",
    "libamdhip64.so.7",
    "libamdhip64.so.6",
    "libamdhip64.so.5",
    "/opt/rocm/lib/libamdhip64.so",
    "/opt/rocm/lib/libamdhip64.so.7",
];

/// Taille du tampon qui reçoit `hipDeviceProp_t` (≈ 1,5 Ko réels) : largement surdimensionné,
/// car on ne dépend pas de la disposition exacte de la structure (voir `extraire_arch`).
const TAILLE_PROPRIETES: usize = 16 * 1024;

/// Décode `hipRuntimeGetVersion` (ROCm ≥ 6 : major·10⁷ + minor·10⁵ + patch).
pub fn decoder_version_hip(v: i32) -> String {
    if v <= 0 {
        return String::new();
    }
    if v >= 10_000_000 {
        format!("{}.{}.{}", v / 10_000_000, (v / 100_000) % 100, v % 100_000)
    } else {
        // Ancien schéma (HIP 5.x) : major·10⁷ impossible, version = major·100 + minor….
        format!("{}.{}", v / 100, v % 100)
    }
}

/// Cherche « gfx… » dans le tampon rendu par hipGetDeviceProperties (gcnArchName est une chaîne C
/// quelque part dans la structure). On ne dépend donc pas de son décalage, qui varie avec la
/// version de ROCm. Rend « gfx1151 » (sans les options « :sramecc+:xnack- »).
pub fn extraire_arch(tampon: &[u8]) -> Option<String> {
    let motif = b"gfx";
    let mut debut = 0;
    while let Some(pos) = tampon[debut..].windows(3).position(|w| w == motif) {
        let i = debut + pos;
        let fin = tampon[i + 3..]
            .iter()
            .position(|c| !c.is_ascii_alphanumeric())
            .map_or(tampon.len(), |p| i + 3 + p);
        let mot = &tampon[i..fin];
        // « gfx » suivi d'au moins un chiffre hexadécimal : « gfx1151 », « gfx90a », « gfx942 ».
        if mot.len() >= 6 && mot[3].is_ascii_digit() {
            return Some(String::from_utf8_lossy(mot).into_owned());
        }
        debut = i + 3;
    }
    None
}

/// Chaîne C → String, tronquée au premier zéro.
pub fn chaine_c(tampon: &[u8]) -> String {
    let fin = tampon.iter().position(|&c| c == 0).unwrap_or(tampon.len());
    String::from_utf8_lossy(&tampon[..fin]).trim().to_string()
}

/// Version de ROCm lue dans `/opt/rocm*/.info/version` (`/opt/rocm` d'abord, puis par ordre décroissant).
pub fn version_rocm(racine_opt: &Path) -> Option<String> {
    let mut dossiers: Vec<_> = std::fs::read_dir(racine_opt)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("rocm"))
        .map(|e| e.path())
        .collect();
    // « rocm » seul d'abord (lien symbolique usuel), puis rocm714 avant rocm71 (ordre décroissant).
    dossiers.sort_by_key(|d| {
        let nom = d
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        (nom != "rocm", std::cmp::Reverse(nom))
    });
    dossiers.into_iter().find_map(|d| {
        let t = std::fs::read_to_string(d.join(".info/version")).ok()?;
        let t = t.trim();
        (!t.is_empty()).then(|| t.to_string())
    })
}

/// Chemin de la bibliothèque `libamdhip64` réellement chargée, lu dans le contenu de `/proc/self/maps`.
pub fn bibliotheque_chargee(maps: &str) -> Option<std::path::PathBuf> {
    maps.lines().find_map(|l| {
        let chemin = l.split_whitespace().nth(5)?;
        let nom = Path::new(chemin).file_name()?.to_string_lossy().into_owned();
        nom.starts_with("libamdhip64.so").then(|| std::path::PathBuf::from(chemin))
    })
}

/// Version ROCm de l'installation qui CONTIENT la bibliothèque chargée : premier ancêtre de `lib` qui a un
/// `.info/version`. None si la bibliothèque vient du système (paquets sans `.info`) : la version utile est
/// alors `hip_runtime`. (Avant le 04/10, `doctor` lisait le premier `/opt/rocm*` venu : « rocm 7.14.0 »
/// alors que le runtime chargé était le 7.1 du système.)
pub fn version_rocm_de(lib: &Path) -> Option<String> {
    lib.ancestors().skip(1).find_map(|d| {
        let t = std::fs::read_to_string(d.join(".info/version")).ok()?;
        let t = t.trim();
        (!t.is_empty()).then(|| t.to_string())
    })
}

/// Versions ROCm à consigner : `rocm` = celle de l'installation chargée (si elle en déclare une) ; sinon, une
/// installation présente sous `racine_opt` mais NON chargée est signalée comme telle (`rocm_opt_non_chargee`).
pub fn versions_rocm(lib_chargee: Option<&Path>, racine_opt: &Path) -> Vec<(String, String)> {
    if let Some(v) = lib_chargee.and_then(version_rocm_de) {
        return vec![("rocm".to_string(), v)];
    }
    match version_rocm(racine_opt) {
        Some(v) => vec![("rocm_opt_non_chargee".to_string(), v)],
        None => Vec::new(),
    }
}

fn ouvrir_bibliotheque() -> Option<&'static libloading::Library> {
    for nom in CANDIDATS {
        // SAFETY : chargement d'une bibliothèque système ; ses initialiseurs sont ceux de ROCm.
        if let Ok(lib) = unsafe { libloading::Library::new(*nom) } {
            // Fuite volontaire : une poignée par processus (comme heritage::ffi).
            return Some(Box::leak(Box::new(lib)));
        }
    }
    None
}

/// Sonde les GPU HIP. Rend (gpus, versions). Vide si ROCm est absent ou inutilisable.
pub fn sonder() -> (Vec<GpuDetecte>, Vec<(String, String)>) {
    let Ok(hip) = crate::heritage::ffi::hiplib() else {
        return (Vec::new(), versions_rocm(None, Path::new("/opt")));
    };
    // La bibliothèque est chargée : sa version ROCm est celle de SON installation, pas du premier /opt/rocm*.
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    let mut versions = versions_rocm(bibliotheque_chargee(&maps).as_deref(), Path::new("/opt"));
    // SAFETY : signatures de la table HipLib ; hipInit(0) est idempotent et ne crée pas de contexte.
    let n = unsafe {
        if (hip.hipInit)(0) != 0 {
            return (Vec::new(), versions);
        }
        let mut n: c_int = 0;
        if (hip.hipGetDeviceCount)(&mut n) != 0 || n <= 0 {
            return (Vec::new(), versions);
        }
        n
    };
    let Some(lib) = ouvrir_bibliotheque() else {
        return (Vec::new(), versions);
    };
    // SAFETY : types de fonctions conformes à hip_runtime_api.h.
    let (f_nom, f_mem, f_prop, f_ver) = unsafe {
        (
            lib.get::<FnNom>(b"hipDeviceGetName\0").ok().map(|s| *s),
            lib.get::<FnMemoire>(b"hipDeviceTotalMem\0").ok().map(|s| *s),
            lib.get::<FnProprietes>(b"hipGetDevicePropertiesR0600\0")
                .or_else(|_| lib.get::<FnProprietes>(b"hipGetDeviceProperties\0"))
                .ok()
                .map(|s| *s),
            lib.get::<FnVersion>(b"hipRuntimeGetVersion\0").ok().map(|s| *s),
        )
    };
    if let Some(f) = f_ver {
        let mut v: c_int = 0;
        // SAFETY : pointeur de sortie valide.
        if unsafe { f(&mut v) } == 0 {
            let d = decoder_version_hip(v);
            if !d.is_empty() {
                versions.push(("hip_runtime".to_string(), d));
            }
        }
    }
    let mut gpus = Vec::new();
    for dev in 0..n {
        let mut nom = String::new();
        let mut arch = String::new();
        let mut memoire_go = 0.0;
        if let Some(f) = f_prop {
            // u64 : alignement suffisant pour toute structure C.
            let mut tampon = vec![0u64; TAILLE_PROPRIETES / 8];
            // SAFETY : tampon de 16 Ko, bien plus grand que hipDeviceProp_t.
            if unsafe { f(tampon.as_mut_ptr().cast(), dev) } == 0 {
                // SAFETY : réinterprétation d'un Vec<u64> en octets, même durée de vie.
                let octets = unsafe { std::slice::from_raw_parts(tampon.as_ptr().cast::<u8>(), TAILLE_PROPRIETES) };
                arch = extraire_arch(octets).unwrap_or_default();
                // hipDeviceProp_t commence par `char name[256]`.
                nom = chaine_c(&octets[..256]);
            }
        }
        if nom.is_empty() {
            if let Some(f) = f_nom {
                let mut buf = [0u8; 256];
                // SAFETY : tampon de 256 octets, longueur passée.
                if unsafe { f(buf.as_mut_ptr().cast(), 256, dev) } == 0 {
                    nom = chaine_c(&buf);
                }
            }
        }
        if let Some(f) = f_mem {
            let mut o: usize = 0;
            // SAFETY : pointeur de sortie valide.
            if unsafe { f(&mut o, dev) } == 0 {
                memoire_go = o as f64 / (1u64 << 30) as f64;
            }
        }
        if arch.is_empty() {
            // Sans architecture, on ne peut pas compiler : le GPU est signalé, mais « inconnu ».
            arch = "gfx?".to_string();
        }
        gpus.push(GpuDetecte {
            backend: BackendKind::Hip,
            ratio_fp32_fp64: ratio_fp32_fp64(&arch, &nom),
            memoire_unifiee: arch_memoire_unifiee(&arch),
            nom,
            arch,
            memoire_go,
        });
    }
    (gpus, versions)
}
