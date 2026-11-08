//! Backend CUDA du runtime unifié (architecture `sm_XY` lue sur le périphérique). Propriétaire : T2. Feature `cuda`.
//!
//! - pilote : `cudarc::driver` (contexte primaire, flux par défaut = flux nul, `alloc_zeros`, copies) ;
//! - compilation : NVRTC par `cudarc::nvrtc::{result, sys}`, avec `-arch=sm_XY` (préfixe `sm_`) pour obtenir
//!   un **CUBIN** (`nvrtcGetCUBIN`) et non du PTX : le pilote d'un poste de développement (CUDA 13.2) est plus ancien que
//!   NVRTC 13.3 et refuserait un PTX trop récent ; un CUBIN sm_120 se charge sans compilation JIT ;
//! - chargement : `Ptx::from_binary(cubin)` ; lancement : `launch_builder` (qui appelle `result::launch_kernel`)
//!   avec des emplacements de 8 octets construits depuis `&[Arg]`.
//!
//! cudarc charge libcuda et libnvrtc à la première utilisation et **panique** si elles sont absentes ;
//! `ouvrir` sonde donc les bibliothèques d'abord et convertit toute panique en `Erreur::Gpu`.

use super::{Arg, Lancement};
use crate::commun::{BackendKind, Erreur, Resultat};
use cudarc::driver::{
    result, sys, CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{result as nvrtc_result, sys as nvrtc_sys, Ptx};
use std::ffi::CString;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

fn gpu(e: impl std::fmt::Display) -> Erreur {
    Erreur::Gpu(e.to_string())
}

/// Bibliothèques à sonder avant de laisser cudarc les charger (il paniquerait en leur absence).
const CANDIDATS_PILOTE: &[&str] = &["libcuda.so.1", "libcuda.so"];
const CANDIDATS_NVRTC: &[&str] = &["libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2"];

fn sonder(candidats: &[&str], quoi: &str) -> Resultat<()> {
    let mut erreurs = Vec::new();
    for c in candidats {
        // SAFETY : chargement d'une bibliothèque système sans exécution de code utilisateur.
        match unsafe { libloading::Library::new(*c) } {
            Ok(l) => {
                std::mem::forget(l); // une poignée par processus
                return Ok(());
            }
            Err(e) => erreurs.push(format!("{c} : {e}")),
        }
    }
    Err(Erreur::Gpu(format!(
        "{quoi} introuvable ({}) — LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 ?",
        erreurs.join(" | ")
    )))
}

/// Exécute `f` en convertissant une panique de cudarc (bibliothèque absente) en erreur GPU.
fn sans_panique<T>(quoi: &str, f: impl FnOnce() -> Resultat<T>) -> Resultat<T> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        Err(Erreur::Gpu(format!(
            "{quoi} : panique de cudarc (bibliothèque absente ?)"
        )))
    })
}

/// Un GPU CUDA ouvert : contexte primaire, flux nul, architecture lue sur l'appareil.
pub(crate) struct CudaInterne {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) flux: Arc<CudaStream>,
    pub(crate) arch: String,
    pub(crate) nom: String,
}

/// Attend TOUT le travail du contexte (`cuCtxSynchronize`) puis relit les erreurs que cudarc a consignées pendant des `Drop` (libération d'un tampon,
/// déchargement d'un module : sans résultat à rendre, elles restent dans le contexte et ne sont relues que par l'appel suivant, qui n'existe plus
/// en fin de commande).
fn attendre_l_appareil(ctx: &CudaContext) -> Resultat<()> {
    ctx.bind_to_thread().map_err(gpu)?;
    ctx.synchronize().map_err(gpu)?;
    ctx.check_err().map_err(gpu)
}

impl Drop for CudaInterne {
    /// Dernière attente de l'appareil, AVANT la libération du contexte : une faute des derniers noyaux, que rien n'a relue depuis, est
    /// consignée dans le registre de santé du processus (si l'erreur décrit une faute d'exécution ; le point d'entrée refuse alors de
    /// rendre 0) au lieu de se perdre dans la libération des tampons. Une panique de cudarc est contenue.
    fn drop(&mut self) {
        super::sante::controle_destruction("fin d'appareil CUDA", || attendre_l_appareil(&self.ctx));
    }
}

/// Module CUDA chargé. Il attend l'appareil AVANT d'être déchargé (`cuModuleUnload` ne l'attend pas : décharger un module sous un noyau en
/// cours fait une faute de lecture d'instructions que rien ne rapporte). Chaque `FonctionCuda` en garde une référence : le module n'est
/// déchargé qu'après la dernière fonction, et toujours après cette attente (`module` est libéré après `Drop::drop`).
pub(crate) struct ModuleCudaInterne {
    module: Arc<CudaModule>,
    ctx: Arc<CudaContext>,
}

impl Drop for ModuleCudaInterne {
    fn drop(&mut self) {
        super::sante::controle_destruction("avant cuModuleUnload", || attendre_l_appareil(&self.ctx));
    }
}

pub(crate) type ModuleCuda = Arc<ModuleCudaInterne>;

pub(crate) struct FonctionCuda {
    pub(crate) f: CudaFunction,
    /// Garde le module (et son attente avant déchargement) en vie aussi longtemps que la fonction.
    pub(crate) _module: ModuleCuda,
}

/// Tampon CUDA : octets bruts (libérés au Drop de `CudaSlice`) et adresse device.
pub(crate) struct TamponCuda {
    pub(crate) _s: CudaSlice<u8>,
    pub(crate) ptr: u64,
}

/// Arguments NVRTC refusés par la version installée, ou dont le comportement est à connaître, sont validés
/// par le test GPU RT-2. Voir aussi `OptionsCompil::arguments`.
impl CudaInterne {
    pub(crate) fn ouvrir(index: usize) -> Resultat<Self> {
        sonder(CANDIDATS_PILOTE, "libcuda")?;
        sonder(CANDIDATS_NVRTC, "libnvrtc")?;
        sans_panique("ouverture CUDA", || {
            let ctx = CudaContext::new(index).map_err(gpu)?;
            let (maj, min) = ctx.compute_capability().map_err(gpu)?;
            // Architecture LUE SUR LE PÉRIPHÉRIQUE ; `arch_cuda` du profil peut la forcer.
            let arch = crate::config::arch_surcharge(crate::commun::BackendKind::Cuda)
                .unwrap_or_else(|| crate::machine::sonde_cuda::sm_depuis(maj, min));
            let nom = ctx.name().map_err(gpu)?;
            let flux = ctx.default_stream();
            // Première utilisation de NVRTC : échoue ici (et non au premier noyau) s'il manque.
            let (mut a, mut b) = (0, 0);
            // SAFETY : pointeurs vers des entiers locaux.
            unsafe { nvrtc_sys::nvrtcVersion(&mut a, &mut b) }
                .result()
                .map_err(gpu)?;
            Ok(CudaInterne { ctx, flux, arch, nom })
        })
    }

    fn lier(&self) -> Resultat<()> {
        self.ctx.bind_to_thread().map_err(gpu)
    }

    pub(crate) fn versions(&self) -> Vec<(String, String)> {
        let mut v = Vec::new();
        let mut pilote = 0i32;
        // SAFETY : pointeur vers un entier local.
        if unsafe { sys::cuDriverGetVersion(&mut pilote) }.result().is_ok() {
            v.push((
                "cuda_pilote".to_string(),
                format!("{}.{}", pilote / 1000, (pilote % 1000) / 10),
            ));
        }
        let (mut a, mut b) = (0, 0);
        // SAFETY : pointeurs vers des entiers locaux.
        if unsafe { nvrtc_sys::nvrtcVersion(&mut a, &mut b) }.result().is_ok() {
            v.push(("nvrtc".to_string(), format!("{a}.{b}")));
        }
        v
    }

    /// Chaîne qui entre dans la clé du cache : tout changement de NVRTC invalide les CUBIN.
    pub(crate) fn version_compilateur(&self) -> String {
        self.versions()
            .iter()
            .filter(|(k, _)| k == "nvrtc")
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(";")
    }

    pub(crate) fn memoire_libre_octets(&self) -> Resultat<u64> {
        self.lier()?;
        let (libre, _total) = result::mem_get_info().map_err(gpu)?;
        Ok(libre as u64)
    }

    /// CUBIN produit par NVRTC. Le journal de compilation est renvoyé en entier en cas d'échec.
    pub(crate) fn compiler_objet(&self, nom: &str, texte: &str, arguments: &[String]) -> Resultat<Vec<u8>> {
        let src = CString::new(texte).map_err(|e| Erreur::Config(format!("source « {nom} » : octet NUL ({e})")))?;
        let nom_c = CString::new(format!("{nom}.cu")).map_err(|e| Erreur::Config(format!("nom « {nom} » : {e}")))?;
        sans_panique("compilation NVRTC", || {
            let prog = nvrtc_result::create_program(&src, Some(&nom_c)).map_err(gpu)?;
            // SAFETY : `prog` vient de create_program ; il est détruit une seule fois, à la fin.
            let resultat = unsafe { self.compiler_programme(prog, arguments) };
            let _ = unsafe { nvrtc_result::destroy_program(prog) };
            resultat
        })
    }

    /// # Safety
    /// `prog` doit être un programme NVRTC valide, non encore détruit.
    unsafe fn compiler_programme(&self, prog: nvrtc_sys::nvrtcProgram, arguments: &[String]) -> Resultat<Vec<u8>> {
        let rc = nvrtc_result::compile_program(prog, arguments);
        let journal = match nvrtc_result::get_program_log(prog) {
            Ok(l) => {
                let octets: Vec<u8> = l.iter().map(|&c| c as u8).collect();
                String::from_utf8_lossy(&octets).trim_end_matches('\0').to_string()
            }
            Err(_) => String::new(),
        };
        if let Err(e) = rc {
            return Err(Erreur::Compilation {
                backend: BackendKind::Cuda,
                journal: format!("nvrtcCompileProgram : {e:?}\n--- journal NVRTC ---\n{journal}"),
            });
        }
        if !journal.trim().is_empty() {
            eprintln!("# journal NVRTC :\n{journal}");
        }
        let mut taille = 0usize;
        nvrtc_sys::nvrtcGetCUBINSize(prog, &mut taille).result().map_err(gpu)?;
        if taille == 0 {
            return Err(Erreur::Gpu(
                "NVRTC n'a pas produit de CUBIN (architecture « compute_XY » au lieu de « sm_XY » ?)".to_string(),
            ));
        }
        let mut cubin = vec![0u8; taille];
        nvrtc_sys::nvrtcGetCUBIN(prog, cubin.as_mut_ptr() as *mut std::ffi::c_char)
            .result()
            .map_err(gpu)?;
        Ok(cubin)
    }

    pub(crate) fn charger(&self, cubin: &[u8]) -> Resultat<ModuleCuda> {
        self.lier()?;
        let module = self.ctx.load_module(Ptx::from_binary(cubin.to_vec())).map_err(gpu)?;
        Ok(Arc::new(ModuleCudaInterne { module, ctx: Arc::clone(&self.ctx) }))
    }

    pub(crate) fn fonction(&self, module: &ModuleCuda, nom: &str) -> Resultat<FonctionCuda> {
        self.lier()?;
        Ok(FonctionCuda {
            f: module.module.load_function(nom).map_err(gpu)?,
            _module: Arc::clone(module),
        })
    }

    pub(crate) fn alloc(&self, octets: usize) -> Resultat<TamponCuda> {
        self.lier()?;
        let s = self.flux.alloc_zeros::<u8>(octets.max(1)).map_err(gpu)?;
        let ptr = {
            let (p, _garde) = s.device_ptr(&self.flux);
            p
        };
        Ok(TamponCuda { _s: s, ptr })
    }

    pub(crate) fn ecrire(&self, dst: &TamponCuda, decalage_octets: usize, src: &[u8]) -> Resultat<()> {
        self.lier()?;
        // SAFETY : la borne de destination est vérifiée par l'appelant ; copie synchrone sur le flux nul.
        unsafe { result::memcpy_htod_sync(dst.ptr + decalage_octets as u64, src) }.map_err(gpu)
    }

    pub(crate) fn lire(&self, src: &TamponCuda, decalage_octets: usize, dst: &mut [u8]) -> Resultat<()> {
        self.lier()?;
        // SAFETY : la borne de source est vérifiée par l'appelant ; copie synchrone sur le flux nul.
        unsafe { result::memcpy_dtoh_sync(dst, src.ptr + decalage_octets as u64) }.map_err(gpu)
    }

    pub(crate) fn copier(&self, src: &TamponCuda, dst: &TamponCuda, octets: usize) -> Resultat<()> {
        self.lier()?;
        // SAFETY : les deux tampons ont au moins `octets` octets (vérifié par l'appelant).
        unsafe { result::memcpy_dtod_async(dst.ptr, src.ptr, octets, self.flux.cu_stream()) }.map_err(gpu)
    }

    pub(crate) fn zero(&self, dst: &TamponCuda, octets: usize) -> Resultat<()> {
        self.lier()?;
        // SAFETY : `dst` a au moins `octets` octets.
        unsafe { result::memset_d8_async(dst.ptr, 0, octets, self.flux.cu_stream()) }.map_err(gpu)
    }

    pub(crate) fn lancer(&self, f: &FonctionCuda, cfg: &Lancement, args: &[Arg]) -> Resultat<()> {
        self.lier()?;
        // Un emplacement de 8 octets par argument ; `launch_builder` en transmet les adresses à cuLaunchKernel
        // (le CUfunction brut n'est pas exposé par cudarc : le lancement passe donc par son constructeur,
        // qui appelle `result::launch_kernel`).
        let emplacements: Vec<u64> = args.iter().map(bits).collect();
        let mut lanceur = self.flux.launch_builder(&f.f);
        for e in &emplacements {
            lanceur.arg(e);
        }
        let config = LaunchConfig {
            grid_dim: (cfg.grille[0], cfg.grille[1], cfg.grille[2]),
            block_dim: (cfg.bloc[0], cfg.bloc[1], cfg.bloc[2]),
            shared_mem_bytes: cfg.shmem,
        };
        // SAFETY : fonction issue d'un module vivant ; les emplacements vivent jusqu'à la fin de l'appel
        // (cuLaunchKernel copie les arguments avant de rendre la main) ; lancement asynchrone sur le flux nul.
        unsafe { lanceur.launch(config) }.map(|_| ()).map_err(gpu)
    }

    pub(crate) fn synchroniser(&self) -> Resultat<()> {
        self.lier()?;
        self.flux.synchronize().map_err(gpu)
    }
}

/// Valeur d'un argument dans un emplacement de 8 octets (little-endian) : un paramètre de 4 octets lit
/// l'octet bas de l'emplacement, ce qui est correct sur x86_64.
pub(crate) fn bits(a: &Arg) -> u64 {
    match *a {
        Arg::Tampon(t) => t.ptr,
        Arg::Nul => 0,
        Arg::I32(v) => v as u32 as u64,
        Arg::U32(v) => v as u64,
        Arg::I64(v) => v as u64,
        Arg::U64(v) => v,
        Arg::F32(v) => v.to_bits() as u64,
        Arg::F64(v) => v.to_bits(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::TamponBrut;

    #[test]
    fn emplacements_des_arguments() {
        assert_eq!(bits(&Arg::Nul), 0);
        assert_eq!(bits(&Arg::I32(-1)), 0xFFFF_FFFF);
        assert_eq!(bits(&Arg::U32(7)), 7);
        assert_eq!(bits(&Arg::I64(-1)), u64::MAX);
        assert_eq!(bits(&Arg::F32(1.0)), 0x3F80_0000);
        assert_eq!(bits(&Arg::F64(1.0)), 0x3FF0_0000_0000_0000);
        assert_eq!(bits(&Arg::Tampon(TamponBrut { ptr: 0xDEAD_BEEF })), 0xDEAD_BEEF);
    }
}
