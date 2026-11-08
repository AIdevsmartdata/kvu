//! Sonde CUDA (NVIDIA) SANS cudarc : elle marche aussi dans un binaire compilé sans la feature
//! `cuda`. On charge `libcuda.so.1` par libloading (cuInit, cuDeviceGetCount, cuDeviceGetName,
//! cuDeviceGetAttribute pour la compute capability, cuDeviceTotalMem_v2, cuDriverGetVersion) et
//! NVRTC par dlopen (LD_LIBRARY_PATH, puis /usr/local/cuda*/lib64) pour sa version.
//! Aucun contexte CUDA n'est créé. La sonde ne panique jamais.

use super::detection::{ratio_fp32_fp64, GpuDetecte};
use crate::commun::BackendKind;
use std::ffi::{c_char, c_int, c_uint};
use std::path::{Path, PathBuf};

type CuResult = c_int;
type FnInit = unsafe extern "C" fn(c_uint) -> CuResult;
type FnCompte = unsafe extern "C" fn(*mut c_int) -> CuResult;
type FnObtenir = unsafe extern "C" fn(*mut c_int, c_int) -> CuResult;
type FnNom = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> CuResult;
type FnAttribut = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CuResult;
type FnMemoire = unsafe extern "C" fn(*mut usize, c_int) -> CuResult;
type FnVersionPilote = unsafe extern "C" fn(*mut c_int) -> CuResult;
type FnNvrtcVersion = unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int;

/// CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / MINOR (cuda.h).
const ATTR_CC_MAJEUR: c_int = 75;
const ATTR_CC_MINEUR: c_int = 76;

const CANDIDATS_CUDA: &[&str] = &["libcuda.so.1", "libcuda.so", "/usr/lib/x86_64-linux-gnu/libcuda.so.1"];

/// « sm_120 » depuis la compute capability (12, 0).
pub fn sm_depuis(majeur: i32, mineur: i32) -> String {
    format!("sm_{majeur}{mineur}")
}

/// Décode `cuDriverGetVersion` : 13020 → « 13.2 ».
pub fn decoder_version_cuda(v: i32) -> String {
    if v <= 0 {
        return String::new();
    }
    format!("{}.{}", v / 1000, (v % 1000) / 10)
}

/// Fichiers `libnvrtc.so*` d'un dossier (hors libnvrtc-builtins et bibliothèques statiques),
/// triés par nom décroissant : libnvrtc.so.13.3.33 avant libnvrtc.so.13 avant libnvrtc.so.
pub fn nvrtc_dans(dossier: &Path) -> Vec<PathBuf> {
    let Ok(entrees) = std::fs::read_dir(dossier) else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = entrees
        .filter_map(|e| e.ok())
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.starts_with("libnvrtc.so")
        })
        .map(|e| e.path())
        .collect();
    v.sort();
    v.reverse();
    v
}

/// Candidats NVRTC dans l'ordre voulu : LD_LIBRARY_PATH, puis /usr/local/cuda*/lib64 (version la plus
/// récente d'abord), puis le chargeur système en dernier recours.
pub fn candidats_nvrtc(ld_library_path: &str, usr_local: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    for d in ld_library_path.split(':').filter(|d| !d.is_empty()) {
        v.extend(nvrtc_dans(Path::new(d)));
    }
    if let Ok(entrees) = std::fs::read_dir(usr_local) {
        let mut cuda: Vec<PathBuf> = entrees
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("cuda")))
            .collect();
        cuda.sort();
        cuda.reverse();
        for d in cuda {
            v.extend(nvrtc_dans(&d.join("lib64")));
        }
    }
    v.push(PathBuf::from("libnvrtc.so"));
    v
}

/// Version de NVRTC (« 13.3 »), par dlopen + nvrtcVersion ; None si absent.
pub fn version_nvrtc() -> Option<String> {
    let ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    for c in candidats_nvrtc(&ld, Path::new("/usr/local")) {
        // SAFETY : chargement d'une bibliothèque CUDA ; la poignée est libérée en sortie de boucle.
        let Ok(lib) = (unsafe { libloading::Library::new(&c) }) else {
            continue;
        };
        // SAFETY : signature de nvrtcVersion (nvrtc.h).
        let Ok(f) = (unsafe { lib.get::<FnNvrtcVersion>(b"nvrtcVersion\0") }) else {
            continue;
        };
        let (mut maj, mut min) = (0, 0);
        // SAFETY : pointeurs de sortie valides.
        if unsafe { f(&mut maj, &mut min) } == 0 && maj > 0 {
            return Some(format!("{maj}.{min}"));
        }
    }
    None
}

/// Sonde les GPU CUDA. Rend (gpus, versions). Vide si le pilote est absent ou ne répond pas.
pub fn sonder() -> (Vec<GpuDetecte>, Vec<(String, String)>) {
    let mut versions = Vec::new();
    let mut lib = None;
    for nom in CANDIDATS_CUDA {
        // SAFETY : chargement du pilote NVIDIA.
        if let Ok(l) = unsafe { libloading::Library::new(*nom) } {
            lib = Some(l);
            break;
        }
    }
    let Some(lib) = lib else { return (Vec::new(), versions) };
    // SAFETY : types de fonctions conformes à cuda.h (versions « _v2 » pour la mémoire).
    let (f_init, f_compte, f_get, f_nom, f_attr, f_mem, f_ver) = unsafe {
        (
            lib.get::<FnInit>(b"cuInit\0").ok().map(|s| *s),
            lib.get::<FnCompte>(b"cuDeviceGetCount\0").ok().map(|s| *s),
            lib.get::<FnObtenir>(b"cuDeviceGet\0").ok().map(|s| *s),
            lib.get::<FnNom>(b"cuDeviceGetName\0").ok().map(|s| *s),
            lib.get::<FnAttribut>(b"cuDeviceGetAttribute\0").ok().map(|s| *s),
            lib.get::<FnMemoire>(b"cuDeviceTotalMem_v2\0").ok().map(|s| *s),
            lib.get::<FnVersionPilote>(b"cuDriverGetVersion\0").ok().map(|s| *s),
        )
    };
    let (Some(f_init), Some(f_compte), Some(f_get), Some(f_attr)) = (f_init, f_compte, f_get, f_attr) else {
        return (Vec::new(), versions);
    };
    // SAFETY : cuInit(0) ne crée pas de contexte.
    if unsafe { f_init(0) } != 0 {
        return (Vec::new(), versions);
    }
    let mut n: c_int = 0;
    // SAFETY : pointeur de sortie valide.
    if unsafe { f_compte(&mut n) } != 0 || n <= 0 {
        return (Vec::new(), versions);
    }
    if let Some(f) = f_ver {
        let mut v: c_int = 0;
        // SAFETY : pointeur de sortie valide.
        if unsafe { f(&mut v) } == 0 && v > 0 {
            versions.push(("cuda_pilote".to_string(), decoder_version_cuda(v)));
        }
    }
    if let Some(v) = version_nvrtc() {
        versions.push(("nvrtc".to_string(), v));
    }
    let mut gpus = Vec::new();
    for ordinal in 0..n {
        let mut dev: c_int = 0;
        // SAFETY : pointeur de sortie valide.
        if unsafe { f_get(&mut dev, ordinal) } != 0 {
            continue;
        }
        let (mut maj, mut min) = (0, 0);
        // SAFETY : pointeurs de sortie valides.
        let cc_ok = unsafe { f_attr(&mut maj, ATTR_CC_MAJEUR, dev) == 0 && f_attr(&mut min, ATTR_CC_MINEUR, dev) == 0 };
        let arch = if cc_ok { sm_depuis(maj, min) } else { "sm_?".to_string() };
        let mut nom = String::new();
        if let Some(f) = f_nom {
            let mut buf = [0u8; 256];
            // SAFETY : tampon de 256 octets, longueur passée.
            if unsafe { f(buf.as_mut_ptr().cast(), 256, dev) } == 0 {
                nom = super::sonde_hip::chaine_c(&buf);
            }
        }
        let mut memoire_go = 0.0;
        if let Some(f) = f_mem {
            let mut o: usize = 0;
            // SAFETY : pointeur de sortie valide.
            if unsafe { f(&mut o, dev) } == 0 {
                memoire_go = o as f64 / (1u64 << 30) as f64;
            }
        }
        gpus.push(GpuDetecte {
            backend: BackendKind::Cuda,
            ratio_fp32_fp64: ratio_fp32_fp64(&arch, &nom),
            memoire_unifiee: false,
            nom,
            arch,
            memoire_go,
        });
    }
    (gpus, versions)
}
