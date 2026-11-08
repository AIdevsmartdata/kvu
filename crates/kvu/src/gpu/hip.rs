//! Backend HIP du runtime unifié (architecture `gfxNNNN` lue sur le périphérique). Propriétaire : T2.
//!
//! S'appuie sur `crate::heritage::{ffi, wrap}` (dlopen de libamdhip64 et libhiprtc, enveloppes RAII).
//! Les symboles que `ffi.rs` ne lie pas (`hipMemGetInfo`, `hipGetDevicePropertiesR0600`,
//! `hipRuntimeGetVersion`, `hiprtcVersion`, `hipDeviceGetName`) sont chargés ici par libloading.

use super::{Arg, Lancement};
use crate::commun::{BackendKind, Erreur, Resultat};
use crate::heritage::ffi::{hipDeviceptr_t, hip_err_str, HIP_SUCCESS};
use crate::heritage::wrap::{HipContext, HipFunction, HipModule, HipSlice, HipStream, KArg, LaunchArgs};
use std::ffi::{c_char, c_void};
use std::sync::{Arc, OnceLock};

/// Poignée HIP déclarée transmissible entre fils : le runtime HIP est sûr vis-à-vis des fils, et le
/// runtime unifié n'utilise qu'un flux (le flux nul) : l'ordre des opérations y est total.
pub(crate) struct Mt<T>(pub(crate) T);
// SAFETY : voir ci-dessus (poignées HIP : pointeurs opaques du runtime, pas d'état partagé côté Rust).
unsafe impl<T> Send for Mt<T> {}
unsafe impl<T> Sync for Mt<T> {}

/// Taille du tampon qui reçoit `hipDeviceProp_tR0600` (la structure fait 1472 octets sur ROCm 7.x).
const TAILLE_PROPRIETES: usize = 8192;
/// Décalage de `gcnArchName` dans `hipDeviceProp_tR0600` (mesuré avec offsetof sur ROCm 7.14).
const DECALAGE_GCN_ARCH: usize = 1160;
/// Longueur de `name` et de `gcnArchName`.
const LONGUEUR_CHAMP: usize = 256;

const CANDIDATS_HIP: &[&str] = &[
    "libamdhip64.so",
    "libamdhip64.so.7",
    "libamdhip64.so.6",
    "libamdhip64.so.5",
    "/opt/rocm/lib/libamdhip64.so",
    "/opt/rocm/lib/libamdhip64.so.7",
];
const CANDIDATS_HIPRTC: &[&str] = &[
    "libhiprtc.so",
    "libhiprtc.so.7",
    "libhiprtc.so.6",
    "libhiprtc.so.5",
    "/opt/rocm/lib/libhiprtc.so",
    "/opt/rocm/lib/libhiprtc.so.7",
];

/// Symboles HIP absents de `ffi.rs`.
struct Extras {
    mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> i32,
    runtime_version: unsafe extern "C" fn(*mut i32) -> i32,
    device_name: unsafe extern "C" fn(*mut c_char, i32, i32) -> i32,
    /// `hipGetDevicePropertiesR0600` (ROCm ≥ 6) : absent sur un runtime plus ancien.
    device_props: Option<unsafe extern "C" fn(*mut c_void, i32) -> i32>,
    hiprtc_version: unsafe extern "C" fn(*mut i32, *mut i32) -> i32,
}

fn ouvrir_premiere(candidats: &[&str]) -> Result<&'static libloading::Library, String> {
    let mut erreurs = Vec::new();
    for nom in candidats {
        match unsafe { libloading::Library::new(*nom) } {
            // Fuite volontaire : une poignée par processus (même modèle que ffi.rs).
            Ok(lib) => return Ok(Box::leak(Box::new(lib))),
            Err(e) => erreurs.push(format!("{nom} : {e}")),
        }
    }
    Err(erreurs.join(" | "))
}

macro_rules! symbole {
    ($lib:expr, $nom:literal) => {
        unsafe {
            *$lib
                .get(concat!($nom, "\0").as_bytes())
                .map_err(|e| format!("dlsym {} : {e}", $nom))?
        }
    };
}

fn charger_extras() -> Result<Extras, String> {
    let hip = ouvrir_premiere(CANDIDATS_HIP)?;
    let rtc = ouvrir_premiere(CANDIDATS_HIPRTC)?;
    Ok(Extras {
        mem_get_info: symbole!(hip, "hipMemGetInfo"),
        runtime_version: symbole!(hip, "hipRuntimeGetVersion"),
        device_name: symbole!(hip, "hipDeviceGetName"),
        device_props: unsafe { hip.get(b"hipGetDevicePropertiesR0600\0").ok().map(|s| *s) },
        hiprtc_version: symbole!(rtc, "hiprtcVersion"),
    })
}

fn extras() -> Result<&'static Extras, String> {
    static E: OnceLock<Result<Extras, String>> = OnceLock::new();
    match E.get_or_init(charger_extras) {
        Ok(e) => Ok(e),
        Err(m) => Err(m.clone()),
    }
}

/// Texte C (terminé par NUL) d'un champ de longueur fixe.
fn texte_c(octets: &[u8]) -> String {
    let fin = octets.iter().position(|&b| b == 0).unwrap_or(octets.len());
    String::from_utf8_lossy(&octets[..fin]).to_string()
}

/// Décode `HIP_VERSION` (major·10⁷ + minor·10⁵ + patch) : « 7.1.12345 ».
pub(crate) fn decoder_version_hip(v: i32) -> String {
    let v = v.max(0);
    format!("{}.{}.{}", v / 10_000_000, (v / 100_000) % 100, v % 100_000)
}

/// « gfx1151:sramecc+:xnack- » → « gfx1151 ».
pub(crate) fn arch_nue(gcn: &str) -> String {
    gcn.split(':').next().unwrap_or("").trim().to_string()
}

fn gpu(e: String) -> Erreur {
    Erreur::Gpu(e)
}

/// Un GPU HIP ouvert : contexte, flux nul (flux par défaut), architecture cible.
pub(crate) struct HipInterne {
    pub(crate) ctx: Arc<HipContext>,
    pub(crate) flux: Mt<HipStream>,
    pub(crate) index: usize,
    pub(crate) arch: String,
    pub(crate) nom: String,
    extras: &'static Extras,
}

/// Attend TOUT le travail de l'appareil du contexte `ctx` (`hipDeviceSynchronize`) et rend l'erreur du runtime, code de retour compris :
/// c'est l'attente qui précède toute libération (`hipFree`, `hipModuleUnload`) et la destruction de l'appareil.
fn attendre_l_appareil(ctx: &HipContext) -> Resultat<()> {
    // SAFETY : appels FFI sans pointeur ; le périphérique courant est propre à chaque fil, donc posé avant l'attente.
    let rc = unsafe { (ctx.lib.hipSetDevice)(ctx.dev) };
    if rc != HIP_SUCCESS {
        return Err(Erreur::Gpu(format!("hipSetDevice : {}", hip_err_str(ctx.lib, rc))));
    }
    // SAFETY : appel FFI sans pointeur.
    let rc = unsafe { (ctx.lib.hipDeviceSynchronize)() };
    if rc != HIP_SUCCESS {
        return Err(Erreur::Gpu(format!("hipDeviceSynchronize : {}", hip_err_str(ctx.lib, rc))));
    }
    Ok(())
}

impl Drop for HipInterne {
    /// Dernière attente de l'appareil : une faute des derniers noyaux, que rien n'a relue depuis, est consignée dans le registre de
    /// santé du processus (si l'erreur décrit une faute d'exécution) au lieu de se perdre dans `hipFree`, dont le code de retour est
    /// ignoré. Le point d'entrée refuse alors de rendre 0.
    fn drop(&mut self) {
        super::sante::controle_destruction("fin d'appareil HIP", || attendre_l_appareil(&self.ctx));
    }
}

/// Module HIP chargé (partagé avec les fonctions qui en sont issues). Il attend l'appareil AVANT d'être déchargé : décharger un module
/// pendant qu'un noyau qui en vient s'exécute encore fait une faute de lecture d'instructions (client SQC) que rien ne rapporte.
pub(crate) struct ModuleHipInterne {
    /// Déchargé (`hipModuleUnload`) après `Drop::drop`, c'est-à-dire après l'attente.
    module: Mt<HipModule>,
    ctx: Arc<HipContext>,
}

impl Drop for ModuleHipInterne {
    fn drop(&mut self) {
        super::sante::controle_destruction("avant hipModuleUnload", || attendre_l_appareil(&self.ctx));
    }
}

pub(crate) type ModuleHip = Arc<ModuleHipInterne>;

pub(crate) struct FonctionHip {
    pub(crate) f: HipFunction,
    pub(crate) _module: ModuleHip,
}

/// Tampon HIP : octets bruts. Il attend l'appareil AVANT d'être libéré par `HipSlice::drop` (`hipFree`) : libérer un tampon sous un noyau en
/// cours fait une faute de page (client TCP) que rien ne rapporte, et l'ordre des `Drop` de Rust ne l'exclut pas (une erreur qui
/// court-circuite la fin d'une commande libère ses tampons sans avoir synchronisé).
pub(crate) struct TamponHip {
    /// Libéré après `Drop::drop`, c'est-à-dire après l'attente.
    s: Mt<HipSlice<u8>>,
    ctx: Arc<HipContext>,
}

impl Drop for TamponHip {
    fn drop(&mut self) {
        super::sante::controle_destruction("avant hipFree", || attendre_l_appareil(&self.ctx));
    }
}

impl TamponHip {
    pub(crate) fn ptr(&self) -> u64 {
        self.s.0.device_ptr() as u64
    }
}

impl HipInterne {
    pub(crate) fn ouvrir(index: usize) -> Resultat<Self> {
        let extras = extras().map_err(|e| gpu(format!("HIP indisponible : {e}")))?;
        let ctx = HipContext::new(index as i32).map_err(gpu)?;
        let flux = Mt(ctx.default_stream());
        let (mut nom, mut gcn) = (String::new(), String::new());
        if let Some(props) = extras.device_props {
            let mut tampon = vec![0u8; TAILLE_PROPRIETES];
            // SAFETY : le tampon est plus grand que la structure ; le runtime n'écrit que dans la structure.
            let rc = unsafe { props(tampon.as_mut_ptr() as *mut c_void, index as i32) };
            if rc == HIP_SUCCESS {
                nom = texte_c(&tampon[..LONGUEUR_CHAMP]);
                gcn = arch_nue(&texte_c(&tampon[DECALAGE_GCN_ARCH..DECALAGE_GCN_ARCH + LONGUEUR_CHAMP]));
            }
        }
        if nom.is_empty() {
            let mut b = vec![0u8; LONGUEUR_CHAMP];
            // SAFETY : tampon de 256 octets, longueur transmise.
            if unsafe { (extras.device_name)(b.as_mut_ptr() as *mut c_char, LONGUEUR_CHAMP as i32, index as i32) }
                == HIP_SUCCESS
            {
                nom = texte_c(&b);
            }
        }
        // Priorité : KEVINOTRON_HIP_ARCH (historique), puis `arch_hip` du profil, puis l'architecture LUE SUR L'APPAREIL.
        // Jamais de valeur écrite en dur : sans architecture lisible, on refuse (une arch fausse donnerait un code faux).
        let arch = std::env::var("KEVINOTRON_HIP_ARCH")
            .ok()
            .filter(|a| !a.trim().is_empty())
            .or_else(|| crate::config::arch_surcharge(crate::commun::BackendKind::Hip))
            .or(Some(gcn).filter(|a| !a.is_empty()))
            .ok_or_else(|| {
                gpu(
                    "architecture HIP illisible sur le périphérique : poser KEVINOTRON_HIP_ARCH ou `arch_hip` du profil"
                        .to_string(),
                )
            })?;
        Ok(HipInterne {
            ctx,
            flux,
            index,
            arch,
            nom,
            extras,
        })
    }

    /// Rend le contexte courant sur le fil appelant (le périphérique courant HIP est propre à chaque fil).
    fn lier(&self) -> Resultat<()> {
        // SAFETY : appel FFI sans pointeur.
        let rc = unsafe { (self.ctx.lib.hipSetDevice)(self.index as i32) };
        self.verifier(rc, "hipSetDevice")
    }

    fn verifier(&self, rc: i32, quoi: &str) -> Resultat<()> {
        if rc == HIP_SUCCESS {
            Ok(())
        } else {
            Err(Erreur::Gpu(format!("{quoi} : {}", hip_err_str(self.ctx.lib, rc))))
        }
    }

    pub(crate) fn versions(&self) -> Vec<(String, String)> {
        let mut v = Vec::new();
        let mut hip = 0i32;
        // SAFETY : pointeur vers un entier local.
        if unsafe { (self.extras.runtime_version)(&mut hip) } == HIP_SUCCESS {
            v.push(("hip_runtime".to_string(), decoder_version_hip(hip)));
        }
        let (mut maj, mut min) = (0i32, 0i32);
        if unsafe { (self.extras.hiprtc_version)(&mut maj, &mut min) } == 0 {
            v.push(("hiprtc".to_string(), format!("{maj}.{min}")));
        }
        v
    }

    /// Chaîne qui entre dans la clé du cache : tout changement de compilateur invalide les objets.
    pub(crate) fn version_compilateur(&self) -> String {
        self.versions()
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(";")
    }

    pub(crate) fn memoire_libre_octets(&self) -> Resultat<u64> {
        self.lier()?;
        let (mut libre, mut total) = (0usize, 0usize);
        // SAFETY : pointeurs vers des entiers locaux.
        let rc = unsafe { (self.extras.mem_get_info)(&mut libre, &mut total) };
        self.verifier(rc, "hipMemGetInfo")?;
        Ok(libre as u64)
    }

    /// Code objet HSACO produit par hiprtc.
    pub(crate) fn compiler_objet(&self, nom: &str, texte: &str, arguments: &[String]) -> Resultat<Vec<u8>> {
        self.lier()?;
        let opts: Vec<&str> = arguments.iter().map(String::as_str).collect();
        self.ctx
            .compile_hip(texte, &format!("{nom}.hip"), &opts)
            .map_err(|journal| {
                if journal.starts_with("hiprtcCompileProgram") {
                    Erreur::Compilation {
                        backend: BackendKind::Hip,
                        journal,
                    }
                } else {
                    Erreur::Gpu(journal)
                }
            })
    }

    pub(crate) fn charger(&self, code: &[u8]) -> Resultat<ModuleHip> {
        self.lier()?;
        Ok(Arc::new(ModuleHipInterne {
            module: Mt(self.ctx.load_module(code).map_err(gpu)?),
            ctx: Arc::clone(&self.ctx),
        }))
    }

    pub(crate) fn fonction(&self, module: &ModuleHip, nom: &str) -> Resultat<FonctionHip> {
        self.lier()?;
        let f = module.module.0.load_function(nom).map_err(gpu)?;
        Ok(FonctionHip {
            f,
            _module: Arc::clone(module),
        })
    }

    pub(crate) fn alloc(&self, octets: usize) -> Resultat<TamponHip> {
        self.lier()?;
        let s = self.flux.0.alloc_zeros::<u8>(octets.max(1)).map_err(gpu)?;
        Ok(TamponHip { s: Mt(s), ctx: Arc::clone(&self.ctx) })
    }

    pub(crate) fn ecrire(
        &self,
        dst: &TamponHip,
        decalage_octets: usize,
        src: *const u8,
        octets: usize,
    ) -> Resultat<()> {
        self.lier()?;
        let d = (dst.ptr() + decalage_octets as u64) as hipDeviceptr_t;
        // SAFETY : `src` pointe sur `octets` octets valides de l'hôte ; la borne de `d` est vérifiée par l'appelant.
        let rc = unsafe { (self.ctx.lib.hipMemcpyHtoDAsync)(d, src as *const c_void, octets, self.flux.0.raw()) };
        self.verifier(rc, "hipMemcpyHtoDAsync")?;
        self.synchroniser()
    }

    pub(crate) fn lire(&self, src: &TamponHip, decalage_octets: usize, dst: *mut u8, octets: usize) -> Resultat<()> {
        self.lier()?;
        let s = (src.ptr() + decalage_octets as u64) as hipDeviceptr_t;
        // SAFETY : `dst` pointe sur `octets` octets inscriptibles de l'hôte ; la borne de `s` est vérifiée par l'appelant.
        let rc = unsafe { (self.ctx.lib.hipMemcpyDtoHAsync)(dst as *mut c_void, s, octets, self.flux.0.raw()) };
        self.verifier(rc, "hipMemcpyDtoHAsync")?;
        self.synchroniser()
    }

    pub(crate) fn copier(&self, src: &TamponHip, dst: &TamponHip, octets: usize) -> Resultat<()> {
        self.lier()?;
        // SAFETY : les deux tampons ont au moins `octets` octets (vérifié par l'appelant).
        let rc = unsafe {
            (self.ctx.lib.hipMemcpyDtoDAsync)(
                dst.ptr() as hipDeviceptr_t,
                src.ptr() as hipDeviceptr_t,
                octets,
                self.flux.0.raw(),
            )
        };
        self.verifier(rc, "hipMemcpyDtoDAsync")
    }

    pub(crate) fn zero(&self, dst: &TamponHip, octets: usize) -> Resultat<()> {
        self.lier()?;
        // SAFETY : `dst` a au moins `octets` octets.
        let rc = unsafe { (self.ctx.lib.hipMemsetD8Async)(dst.ptr() as hipDeviceptr_t, 0, octets, self.flux.0.raw()) };
        self.verifier(rc, "hipMemsetD8Async")
    }

    pub(crate) fn lancer(&self, f: &FonctionHip, cfg: &Lancement, args: &[Arg]) -> Resultat<()> {
        self.lier()?;
        let kargs: Vec<KArg> = args.iter().map(karg).collect();
        let la = LaunchArgs {
            grid: (cfg.grille[0], cfg.grille[1], cfg.grille[2]),
            block: (cfg.bloc[0], cfg.bloc[1], cfg.bloc[2]),
            shmem: cfg.shmem,
        };
        self.flux.0.launch(&f.f, &la, &kargs).map_err(gpu)
    }

    pub(crate) fn synchroniser(&self) -> Resultat<()> {
        self.lier()?;
        self.flux.0.synchronize().map_err(gpu)
    }
}

/// `Arg` → argument HIP. Chaque valeur occupe un emplacement de 8 octets (little-endian) : un entier de
/// 32 bits dans l'octet bas d'un emplacement de 64 est correct sur x86_64.
pub(crate) fn karg(a: &Arg) -> KArg {
    match *a {
        Arg::Tampon(t) => KArg::Ptr(t.ptr as *mut c_void),
        Arg::Nul => KArg::Ptr(std::ptr::null_mut()),
        Arg::I32(v) => KArg::I32(v),
        Arg::U32(v) => KArg::U32(v),
        Arg::I64(v) => KArg::U64(v as u64),
        Arg::U64(v) => KArg::U64(v),
        Arg::F32(v) => KArg::F32(v),
        Arg::F64(v) => KArg::F64(v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_hip_decodee() {
        assert_eq!(decoder_version_hip(70_251_831), "7.2.51831");
        assert_eq!(decoder_version_hip(60_000_000), "6.0.0");
        assert_eq!(decoder_version_hip(-1), "0.0.0");
    }

    #[test]
    fn arch_sans_fonctionnalites() {
        assert_eq!(arch_nue("gfx1151:sramecc+:xnack-"), "gfx1151");
        assert_eq!(arch_nue("gfx1100"), "gfx1100");
        assert_eq!(arch_nue(""), "");
    }

    #[test]
    fn texte_c_coupe_au_nul() {
        assert_eq!(texte_c(b"abc\0def"), "abc");
        assert_eq!(texte_c(b"abc"), "abc");
    }
}
