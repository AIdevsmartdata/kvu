//! Runtime GPU unifié HIP/CUDA à source unique. Propriétaire : T2.
//!
//! - HIP : s'appuie sur `crate::heritage::{ffi, wrap}` (dlopen libamdhip64 + libhiprtc) ;
//!   code objet produit par hiprtc avec `--offload-arch=<arch>` explicite.
//! - CUDA (feature `cuda`) : cudarc 0.19 (driver) + NVRTC ; on produit du CUBIN
//!   (`nvrtcGetCUBIN`, `-arch=sm_120`) et non du PTX : le pilote d'un poste de développement (CUDA 13.2) est plus
//!   ancien que NVRTC 13.3 et refuserait un PTX trop récent.
//! - Les sources passent TOUTES par `assembler` (prélude + defines + fichiers du registre) ;
//!   l'empreinte sha256 de la source assemblée et des options entre dans le manifeste.
//! - Cache disque des codes objets : $KEVINOTRON_CACHE_DIR ou ~/.cache/kevinotron/kvu/,
//!   plafonné (défaut 256 Mo, éviction LRU), écriture atomique, entrée corrompue = recompilation.
//!
//! Sémantique : API synchrone sur un flux unique (le flux par défaut). `lancer` est asynchrone ;
//! `lire` et `ecrire` synchronisent ; `copier` et `zero` sont ordonnés sur le flux.
//! Les noyaux sont appelés par leur nom `extern "C"`.
//!
//! Erreurs d'exécution : un lancement est asynchrone, donc la faute d'un noyau n'est connue qu'à la synchronisation ou à la
//! copie suivante. Toute erreur du runtime reçue par `lancer`, `synchroniser`, `lire`, `ecrire`, `copier` ou `zero` est rendue
//! à l'appelant (`Erreur::Gpu`, code de sortie 3) ET consignée dans le registre du processus ([`sante`]) ; chaque tampon et chaque
//! module attend l'appareil avant d'être libéré ou déchargé (jamais de libération sous un noyau en vol, quel que soit l'ordre des `Drop`),
//! et un appareil qu'on détruit fait une dernière attente ; le point d'entrée refuse de rendre 0 si le registre contient une faute.

pub mod cache;
#[cfg(feature = "cuda")]
mod cuda;
mod hip;
pub mod sante;
pub mod source;

pub use cache::Cache;
pub use source::{assembler, essayer_assembler, FpContract, OptionsCompil, ProvenanceCompil, SourceAssemblee};

use crate::commun::{BackendKind, Erreur, Resultat};
use crate::heritage::manifest::sha256_hex;
use std::marker::PhantomData;
use std::sync::Arc;

/// Types copiables tels quels entre hôte et GPU.
///
/// # Safety
/// Le type doit être `Copy`, sans pointeur ni octet de bourrage non initialisé.
pub unsafe trait Pod: Copy + Send + Sync + 'static {}
unsafe impl Pod for f64 {}
unsafe impl Pod for f32 {}
unsafe impl Pod for u64 {}
unsafe impl Pod for u32 {}
unsafe impl Pod for i64 {}
unsafe impl Pod for i32 {}
unsafe impl Pod for u8 {}
unsafe impl Pod for [f64; 4] {}
unsafe impl Pod for [f32; 4] {}

/// État interne d'un appareil (HIP ou CUDA).
pub(crate) enum Interne {
    Hip(hip::HipInterne),
    #[cfg(feature = "cuda")]
    Cuda(cuda::CudaInterne),
}

/// Un GPU ouvert : contexte + flux par défaut. Clonable (Arc).
#[derive(Clone)]
pub struct Appareil {
    pub(crate) interne: Arc<Interne>,
}

/// Mémoire device d'un tampon (propriétaire : libérée à la destruction).
enum Memoire {
    Hip(hip::TamponHip),
    #[cfg(feature = "cuda")]
    Cuda(cuda::TamponCuda),
}

/// Tampon device typé. Libéré à la destruction.
pub struct Tampon<T: Pod> {
    pub(crate) n: usize,
    /// Adresse device du premier élément.
    pub(crate) ptr: u64,
    _memoire: Memoire,
    pub(crate) _t: PhantomData<T>,
}

/// D'où vient le code objet d'un module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origine {
    /// Relu du cache disque : aucun compilateur n'a tourné.
    Cache,
    /// Compilé par hiprtc / NVRTC (puis écrit dans le cache s'il est actif).
    Compilation,
}

enum ModuleInterne {
    Hip(hip::ModuleHip),
    #[cfg(feature = "cuda")]
    Cuda(cuda::ModuleCuda),
}

/// Code objet chargé (module HIP / CUDA).
pub struct Module {
    /// sha256 du code objet (HSACO / CUBIN), pour le manifeste.
    pub sha256_code: String,
    provenance: ProvenanceCompil,
    origine: Origine,
    interne: ModuleInterne,
}

impl Module {
    /// Provenance complète (source, options, arguments, code objet) pour le manifeste.
    pub fn provenance(&self) -> &ProvenanceCompil {
        &self.provenance
    }
    /// Code objet relu du cache ou compilé ?
    pub fn origine(&self) -> Origine {
        self.origine
    }
}

enum FonctionInterne {
    Hip(hip::FonctionHip),
    #[cfg(feature = "cuda")]
    Cuda(cuda::FonctionCuda),
}

/// Point d'entrée d'un noyau `extern "C" __global__`.
pub struct Fonction {
    pub nom: String,
    interne: FonctionInterne,
}

/// Géométrie de lancement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lancement {
    pub grille: [u32; 3],
    pub bloc: [u32; 3],
    pub shmem: u32,
}

impl Lancement {
    /// ceil(n / bloc) blocs de `bloc` fils (1D).
    pub fn lineaire(n: usize, bloc: u32) -> Lancement {
        let g = n.div_ceil(bloc as usize).max(1) as u32;
        Lancement {
            grille: [g, 1, 1],
            bloc: [bloc, 1, 1],
            shmem: 0,
        }
    }
    /// 2D : x = ceil(n / bloc), y = `ny` (une ligne par chaîne d'un lot).
    pub fn lineaire_lot(n: usize, ny: usize, bloc: u32) -> Lancement {
        let g = n.div_ceil(bloc as usize).max(1) as u32;
        Lancement {
            grille: [g, ny as u32, 1],
            bloc: [bloc, 1, 1],
            shmem: 0,
        }
    }
    /// Même géométrie avec `shmem` octets de mémoire partagée dynamique.
    pub fn avec_shmem(mut self, shmem: u32) -> Lancement {
        self.shmem = shmem;
        self
    }
    fn verifier(&self) -> Resultat<()> {
        if self.grille.contains(&0) || self.bloc.contains(&0) {
            return Err(Erreur::Config(format!(
                "lancement invalide : grille {:?}, bloc {:?} (dimension nulle)",
                self.grille, self.bloc
            )));
        }
        Ok(())
    }
}

/// Vue non typée d'un tampon, pour les arguments de noyau.
#[derive(Clone, Copy)]
pub struct TamponBrut {
    pub(crate) ptr: u64,
}

/// Argument de noyau, dans l'ordre de la signature C.
#[derive(Clone, Copy)]
pub enum Arg {
    Tampon(TamponBrut),
    /// Pointeur nul (ex. diagnostic facultatif).
    Nul,
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
}

impl<T: Pod> Tampon<T> {
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn octets(&self) -> usize {
        self.n * std::mem::size_of::<T>()
    }
    /// Argument de noyau pointant sur le début du tampon.
    pub fn arg(&self) -> Arg {
        Arg::Tampon(TamponBrut { ptr: self.ptr })
    }
    /// Argument pointant sur l'élément `decalage` (lots : chaîne c ⇒ c · taille_par_chaîne).
    ///
    /// # Panics
    /// Si `decalage` dépasse la taille du tampon (le décalage `len()` est permis : fin de tampon).
    pub fn arg_decale(&self, decalage: usize) -> Arg {
        assert!(
            decalage <= self.n,
            "arg_decale : décalage {decalage} > longueur {}",
            self.n
        );
        Arg::Tampon(TamponBrut {
            ptr: self.ptr + (decalage * std::mem::size_of::<T>()) as u64,
        })
    }
}

/// Ouvre le GPU `index` du backend demandé (Cpu ⇒ erreur Config).
pub fn ouvrir(backend: BackendKind, index: usize) -> Resultat<Appareil> {
    let interne = match backend {
        BackendKind::Hip => Interne::Hip(hip::HipInterne::ouvrir(index)?),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => Interne::Cuda(cuda::CudaInterne::ouvrir(index)?),
        #[cfg(not(feature = "cuda"))]
        BackendKind::Cuda => {
            return Err(Erreur::Gpu(
                "backend cuda non compilé : reconstruire kvu avec --features cuda".to_string(),
            ))
        }
        BackendKind::Cpu => return Err(Erreur::Config("le backend cpu n'a pas de GPU à ouvrir".to_string())),
    };
    // Les fils internes du runtime (créés sur tous les CPU) reprennent l'affinité obligatoire du processus.
    crate::machine::processus::reappliquer_affinite();
    Ok(Appareil {
        interne: Arc::new(interne),
    })
}

/// Ouvre le GPU du backend choisi par `machine::choisir_backend` ; None si le choix est CPU.
pub fn ouvrir_auto() -> Resultat<Option<Appareil>> {
    let profil = crate::machine::detecter();
    match crate::machine::choisir_backend(&profil)? {
        BackendKind::Cpu => Ok(None),
        b => ouvrir(b, 0).map(Some),
    }
}

/// Vue en octets d'une tranche de `Pod`.
fn octets_de<T: Pod>(s: &[T]) -> &[u8] {
    // SAFETY : `Pod` garantit l'absence de pointeur et d'octet de bourrage non initialisé.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn octets_de_mut<T: Pod>(s: &mut [T]) -> &mut [u8] {
    // SAFETY : idem ; toute suite d'octets est une valeur valide d'un `Pod`.
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, std::mem::size_of_val(s)) }
}

/// Consigne dans le registre du processus l'erreur du runtime que porte `r` (les autres erreurs, de configuration par exemple, ne
/// sont pas des fautes d'exécution), puis rend `r` tel quel.
fn suivre<T>(origine: &str, r: Resultat<T>) -> Resultat<T> {
    if let Err(e) = &r {
        sante::globale().consigner(origine, e);
    }
    r
}

/// Vérifie qu'une plage [decalage, decalage + len) tient dans un tampon de `n` éléments.
fn verifier_plage(quoi: &str, n: usize, decalage: usize, len: usize) -> Resultat<()> {
    match decalage.checked_add(len) {
        Some(fin) if fin <= n => Ok(()),
        _ => Err(Erreur::Config(format!(
            "{quoi} : plage [{decalage}, {decalage}+{len}) hors du tampon de {n} éléments"
        ))),
    }
}

impl Appareil {
    pub fn backend(&self) -> BackendKind {
        match &*self.interne {
            Interne::Hip(_) => BackendKind::Hip,
            #[cfg(feature = "cuda")]
            Interne::Cuda(_) => BackendKind::Cuda,
        }
    }
    /// « gfx1151 », « sm_120 ».
    pub fn arch(&self) -> String {
        match &*self.interne {
            Interne::Hip(h) => h.arch.clone(),
            #[cfg(feature = "cuda")]
            Interne::Cuda(c) => c.arch.clone(),
        }
    }
    pub fn nom(&self) -> String {
        match &*self.interne {
            Interne::Hip(h) => h.nom.clone(),
            #[cfg(feature = "cuda")]
            Interne::Cuda(c) => c.nom.clone(),
        }
    }
    /// Versions runtime pour le manifeste (hip_runtime, hiprtc, cuda_pilote, nvrtc…).
    pub fn versions(&self) -> Vec<(String, String)> {
        match &*self.interne {
            Interne::Hip(h) => h.versions(),
            #[cfg(feature = "cuda")]
            Interne::Cuda(c) => c.versions(),
        }
    }
    /// Chaîne de version du compilateur, ingrédient de la clé du cache.
    pub fn version_compilateur(&self) -> String {
        match &*self.interne {
            Interne::Hip(h) => h.version_compilateur(),
            #[cfg(feature = "cuda")]
            Interne::Cuda(c) => c.version_compilateur(),
        }
    }
    /// Mémoire libre selon le runtime (sur une APU à mémoire unifiée, c'est la réserve vue par `hipMemGetInfo`,
    /// pas la GTT : voir `machine::doctor` pour la mémoire effectivement allouable).
    pub fn memoire_libre_octets(&self) -> Resultat<u64> {
        match &*self.interne {
            Interne::Hip(h) => h.memoire_libre_octets(),
            #[cfg(feature = "cuda")]
            Interne::Cuda(c) => c.memoire_libre_octets(),
        }
    }

    /// Compile (ou relit du cache) une source assemblée POUR CE backend.
    /// Cache : celui de l'environnement (`Cache::depuis_env`), absent si KEVINOTRON_CACHE_DISABLE=1.
    pub fn compiler(&self, src: &SourceAssemblee, opts: &OptionsCompil) -> Resultat<Module> {
        self.compiler_avec_cache(src, opts, Cache::depuis_env().as_ref())
    }

    /// Comme [`Appareil::compiler`], avec un cache explicite (`None` : toujours compiler).
    pub fn compiler_avec_cache(
        &self,
        src: &SourceAssemblee,
        opts: &OptionsCompil,
        cache: Option<&Cache>,
    ) -> Resultat<Module> {
        let backend = self.backend();
        if src.backend != backend {
            return Err(Erreur::Config(format!(
                "source « {} » assemblée pour {} mais l'appareil est {} : assembler avec le bon prélude",
                src.nom,
                src.backend.nom(),
                backend.nom()
            )));
        }
        let arch = self.arch();
        let arguments = opts.arguments(backend, &arch);
        let version = self.version_compilateur();
        let cle = cache::cle(&src.texte, &arguments, &arch, &version);
        let ext = match backend {
            BackendKind::Hip => "hsaco",
            _ => "cubin",
        };

        // Un code objet relu du cache mais que le chargeur refuse est évincé, puis on recompile une fois.
        if let Some(c) = cache {
            if let Some(entree) = c.lire(&src.nom, &arch, &cle, ext) {
                match self.charger(&entree.code) {
                    Ok(interne) => {
                        return Ok(self.module(src, opts, arguments, entree.meta.sha256_code, Origine::Cache, interne))
                    }
                    Err(e) => {
                        eprintln!(
                            "# cache : {} refusé au chargement ({e}) : évincé, recompilation",
                            src.nom
                        );
                        let _ = std::fs::remove_file(c.chemin_code(&src.nom, &arch, &cle, ext));
                        let _ = std::fs::remove_file(c.chemin_meta(&src.nom, &arch, &cle));
                    }
                }
            }
        }
        let code = match &*self.interne {
            Interne::Hip(h) => h.compiler_objet(&src.nom, &src.texte, &arguments)?,
            #[cfg(feature = "cuda")]
            Interne::Cuda(c) => c.compiler_objet(&src.nom, &src.texte, &arguments)?,
        };
        let sha = sha256_hex(&code);
        if let Some(c) = cache {
            let meta = cache::MetaEntree {
                format: cache::VERSION_FORMAT,
                nom: src.nom.clone(),
                arch: arch.clone(),
                cle: cle.clone(),
                version_compilateur: version,
                arguments: arguments.clone(),
                sha256_source: src.sha256.clone(),
                sha256_code: sha.clone(),
                taille_code: code.len() as u64,
            };
            if let Err(e) = c.ecrire(&src.nom, &arch, &cle, ext, &code, meta) {
                eprintln!("# cache : écriture impossible ({e}) : le module reste utilisable");
            }
        }
        let interne = self.charger(&code)?;
        Ok(self.module(src, opts, arguments, sha, Origine::Compilation, interne))
    }

    fn charger(&self, code: &[u8]) -> Resultat<ModuleInterne> {
        Ok(match &*self.interne {
            Interne::Hip(h) => ModuleInterne::Hip(h.charger(code)?),
            #[cfg(feature = "cuda")]
            Interne::Cuda(c) => ModuleInterne::Cuda(c.charger(code)?),
        })
    }

    fn module(
        &self,
        src: &SourceAssemblee,
        opts: &OptionsCompil,
        arguments: Vec<String>,
        sha: String,
        origine: Origine,
        interne: ModuleInterne,
    ) -> Module {
        Module {
            sha256_code: sha.clone(),
            provenance: ProvenanceCompil {
                nom: src.nom.clone(),
                backend: self.backend(),
                arch: self.arch(),
                sha256_source: src.sha256.clone(),
                parties: src.parties.clone(),
                options: opts.clone(),
                arguments,
                sha256_code: sha,
            },
            origine,
            interne,
        }
    }

    pub fn fonction(&self, module: &Module, nom: &str) -> Resultat<Fonction> {
        let interne = match (&*self.interne, &module.interne) {
            (Interne::Hip(h), ModuleInterne::Hip(m)) => FonctionInterne::Hip(h.fonction(m, nom)?),
            #[cfg(feature = "cuda")]
            (Interne::Cuda(c), ModuleInterne::Cuda(m)) => FonctionInterne::Cuda(c.fonction(m, nom)?),
            #[cfg(feature = "cuda")]
            _ => return Err(Erreur::Config("module d'un autre backend que l'appareil".to_string())),
        };
        Ok(Fonction {
            nom: nom.to_string(),
            interne,
        })
    }

    /// Tampon de `n` éléments mis à zéro (synchrone).
    pub fn alloc<T: Pod>(&self, n: usize) -> Resultat<Tampon<T>> {
        let octets = n.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            Erreur::Config(format!(
                "alloc : {n} éléments de {} octets débordent usize",
                std::mem::size_of::<T>()
            ))
        })?;
        let (ptr, memoire) = match &*self.interne {
            Interne::Hip(h) => {
                let t = h.alloc(octets)?;
                (t.ptr(), Memoire::Hip(t))
            }
            #[cfg(feature = "cuda")]
            Interne::Cuda(c) => {
                let t = c.alloc(octets)?;
                (t.ptr, Memoire::Cuda(t))
            }
        };
        self.synchroniser()?; // la mise à zéro est ordonnée sur le flux : on la termine avant de rendre le tampon
        Ok(Tampon {
            n,
            ptr,
            _memoire: memoire,
            _t: PhantomData,
        })
    }

    pub fn ecrire<T: Pod>(&self, dst: &mut Tampon<T>, decalage: usize, src: &[T]) -> Resultat<()> {
        verifier_plage("ecrire", dst.n, decalage, src.len())?;
        let (off, octets) = (decalage * std::mem::size_of::<T>(), octets_de(src));
        if octets.is_empty() {
            return Ok(());
        }
        sante::globale().avant(sante::Point::Copie, "ecrire")?;
        let r = match (&*self.interne, &dst._memoire) {
            (Interne::Hip(h), Memoire::Hip(m)) => h.ecrire(m, off, octets.as_ptr(), octets.len()),
            #[cfg(feature = "cuda")]
            (Interne::Cuda(c), Memoire::Cuda(m)) => c.ecrire(m, off, octets),
            #[cfg(feature = "cuda")]
            _ => Err(Erreur::Config("tampon d'un autre backend que l'appareil".to_string())),
        };
        suivre("ecrire", r)
    }

    pub fn lire<T: Pod>(&self, src: &Tampon<T>, decalage: usize, dst: &mut [T]) -> Resultat<()> {
        verifier_plage("lire", src.n, decalage, dst.len())?;
        let off = decalage * std::mem::size_of::<T>();
        let octets = octets_de_mut(dst);
        if octets.is_empty() {
            return Ok(());
        }
        sante::globale().avant(sante::Point::Copie, "lire")?;
        let r = match (&*self.interne, &src._memoire) {
            (Interne::Hip(h), Memoire::Hip(m)) => h.lire(m, off, octets.as_mut_ptr(), octets.len()),
            #[cfg(feature = "cuda")]
            (Interne::Cuda(c), Memoire::Cuda(m)) => c.lire(m, off, octets),
            #[cfg(feature = "cuda")]
            _ => Err(Erreur::Config("tampon d'un autre backend que l'appareil".to_string())),
        };
        suivre("lire", r)
    }

    pub fn copier<T: Pod>(&self, src: &Tampon<T>, dst: &mut Tampon<T>) -> Resultat<()> {
        if src.n != dst.n {
            return Err(Erreur::Config(format!(
                "copier : tailles différentes ({} ≠ {} éléments)",
                src.n, dst.n
            )));
        }
        if src.n == 0 {
            return Ok(());
        }
        sante::globale().avant(sante::Point::Copie, "copier")?;
        let r = match (&*self.interne, &src._memoire, &dst._memoire) {
            (Interne::Hip(h), Memoire::Hip(s), Memoire::Hip(d)) => h.copier(s, d, src.octets()),
            #[cfg(feature = "cuda")]
            (Interne::Cuda(c), Memoire::Cuda(s), Memoire::Cuda(d)) => c.copier(s, d, src.octets()),
            #[cfg(feature = "cuda")]
            _ => Err(Erreur::Config("tampon d'un autre backend que l'appareil".to_string())),
        };
        suivre("copier", r)
    }

    pub fn zero<T: Pod>(&self, dst: &mut Tampon<T>) -> Resultat<()> {
        if dst.n == 0 {
            return Ok(());
        }
        let r = match (&*self.interne, &dst._memoire) {
            (Interne::Hip(h), Memoire::Hip(m)) => h.zero(m, dst.octets()),
            #[cfg(feature = "cuda")]
            (Interne::Cuda(c), Memoire::Cuda(m)) => c.zero(m, dst.octets()),
            #[cfg(feature = "cuda")]
            _ => Err(Erreur::Config("tampon d'un autre backend que l'appareil".to_string())),
        };
        suivre("zero", r)
    }

    /// Lance un noyau sur le flux par défaut (asynchrone).
    ///
    /// Contrôle par lots : [`sante::LANCEMENTS_MAX_SANS_CONTROLE`] lancements de suite sans aucune synchronisation en forcent une
    /// ici, pour qu'une faute d'un noyau soit relevée au plus tard après ce nombre de lancements, quel que soit l'appelant.
    pub fn lancer(&self, f: &Fonction, cfg: &Lancement, args: &[Arg]) -> Resultat<()> {
        cfg.verifier()?;
        let sante = sante::globale();
        sante.avant(sante::Point::Lancement, "lancer")?;
        let r = match (&*self.interne, &f.interne) {
            (Interne::Hip(h), FonctionInterne::Hip(fh)) => h.lancer(fh, cfg, args),
            #[cfg(feature = "cuda")]
            (Interne::Cuda(c), FonctionInterne::Cuda(fc)) => c.lancer(fc, cfg, args),
            #[cfg(feature = "cuda")]
            _ => Err(Erreur::Config("fonction d'un autre backend que l'appareil".to_string())),
        };
        suivre("lancer", r)?;
        if sante.lancement_compte() {
            self.synchroniser()?;
        }
        Ok(())
    }

    /// Attend la fin de tout le travail lancé et rend l'erreur d'exécution éventuelle (faute d'un noyau, mémoire illégale).
    pub fn synchroniser(&self) -> Resultat<()> {
        let sante = sante::globale();
        sante.avant(sante::Point::Synchronisation, "synchroniser")?;
        let r = match &*self.interne {
            Interne::Hip(h) => h.synchroniser(),
            #[cfg(feature = "cuda")]
            Interne::Cuda(c) => c.synchroniser(),
        };
        if r.is_ok() {
            sante.controle_fait();
        }
        suivre("synchroniser", r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plages() {
        assert!(verifier_plage("t", 10, 0, 10).is_ok());
        assert!(verifier_plage("t", 10, 4, 6).is_ok());
        assert!(verifier_plage("t", 10, 10, 0).is_ok());
        assert!(verifier_plage("t", 10, 5, 6).is_err());
        assert!(verifier_plage("t", 10, usize::MAX, 2).is_err());
    }

    #[test]
    fn octets_des_pod() {
        let x = [1.0f64, -2.0];
        let o = octets_de(&x);
        assert_eq!(o.len(), 16);
        assert_eq!(&o[..8], &1.0f64.to_ne_bytes());
        let mut y = [0u32; 2];
        octets_de_mut(&mut y).copy_from_slice(&[1, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(y, [1, 2]);
    }

    #[test]
    fn lancement_lineaire() {
        assert_eq!(Lancement::lineaire(1000, 256).grille, [4, 1, 1]);
        assert_eq!(Lancement::lineaire(0, 256).grille, [1, 1, 1]);
        assert_eq!(Lancement::lineaire_lot(512, 7, 256).grille, [2, 7, 1]);
        assert!(Lancement {
            grille: [0, 1, 1],
            bloc: [1, 1, 1],
            shmem: 0
        }
        .verifier()
        .is_err());
        assert!(Lancement::lineaire(1, 64).verifier().is_ok());
    }

    #[test]
    fn ouvrir_cpu_est_une_erreur_de_config() {
        assert!(matches!(ouvrir(BackendKind::Cpu, 0), Err(Erreur::Config(_))));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_non_compile_est_une_erreur_gpu() {
        assert!(matches!(ouvrir(BackendKind::Cuda, 0), Err(Erreur::Gpu(_))));
    }
}
