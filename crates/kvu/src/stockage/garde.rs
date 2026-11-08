//! Garde SSD : aucune écriture ne doit laisser moins de `seuil_go` Go libres. Propriétaire : T8.
//!
//! - `libre_go` interroge `statvfs` (déclaration `extern "C"` minimale, aucune dépendance de crate) ;
//! - `verifier` refuse AVANT d'écrire le moindre octet (code de sortie 7) ;
//! - `projeter` estime la taille totale d'un run pour le refuser dès le lancement.
//!
//! Unité : 1 Go = 1 Gio = 2³⁰ octets, PARTOUT (seuils, libre, projections), comme `df -BG`, la file de jobs,
//! `doctor` et `etat`. (Jusqu'au 04/10 la garde comptait 10⁹ octets : le même disque valait 718,8 Go dans un
//! manifeste et 670,5 Go dans `doctor`, pour un même seuil « 100 Go ».)

use super::npy::{octets_champ, FormatChamp};
use crate::commun::{Erreur, Machine, Resultat};
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

/// Octets dans un Go (= Gio, comme `df -BG`).
pub const OCTETS_PAR_GO: f64 = (1u64 << 30) as f64;

/// Réserve fixe comptée par run dans la projection : manifeste, journaux, `.part` en vol (1 Mio).
pub const OCTETS_FIXES_RUN: u64 = 1 << 20;
/// Réserve comptée par chaîne dans la projection : `therm.csv`, `measurements.csv`, blocs (4 Mio).
pub const OCTETS_CSV_PAR_CHAINE: u64 = 4 << 20;

/// Disposition de `struct statvfs` de la glibc sur Linux 64 bits (x86_64 et aarch64) : `unsigned long`
/// = 64 bits, `fsblkcnt_t` et `fsfilcnt_t` = 64 bits, six entiers de réserve.
#[repr(C)]
struct Statvfs {
    f_bsize: u64,
    f_frsize: u64,
    f_blocks: u64,
    f_bfree: u64,
    f_bavail: u64,
    f_files: u64,
    f_ffree: u64,
    f_favail: u64,
    f_fsid: u64,
    f_flag: u64,
    f_namemax: u64,
    spare: [i32; 6],
}

extern "C" {
    fn statvfs(chemin: *const c_char, tampon: *mut Statvfs) -> c_int;
}

/// Seuils par défaut (Go libres après écriture) : clé `ssd_min_go` du profil de `kvu.toml`.
/// Surcharge : KEVINOTRON_SSD_MIN_GO.
#[derive(Clone, Debug, PartialEq)]
pub struct GardeSsd {
    pub seuil_go: f64,
}

/// Seuil par défaut d'une machine : clé `ssd_min_go` de son profil (`kvu.toml`).
pub fn seuil_defaut(m: Machine) -> f64 {
    crate::config::actuelle().profil(m).ssd_min_go
}

/// Lit KEVINOTRON_SSD_MIN_GO (nombre de Go, ≥ 0). Une valeur illisible est ignorée.
fn seuil_depuis_env() -> Option<f64> {
    let v = std::env::var("KEVINOTRON_SSD_MIN_GO").ok()?;
    let x: f64 = v.trim().parse().ok()?;
    (x.is_finite() && x >= 0.0).then_some(x)
}

impl GardeSsd {
    pub fn nouvelle(seuil_go: f64) -> GardeSsd {
        GardeSsd { seuil_go }
    }

    /// Garde d'une machine donnée, avec ses seuils par défaut (sans surcharge d'environnement).
    pub fn pour(m: Machine) -> GardeSsd {
        GardeSsd {
            seuil_go: seuil_defaut(m),
        }
    }

    /// Garde de la machine courante : KEVINOTRON_SSD_MIN_GO, sinon le seuil de la politique de
    /// `crate::machine` (profil de `kvu.toml`).
    pub fn machine() -> GardeSsd {
        if let Some(s) = seuil_depuis_env() {
            return GardeSsd { seuil_go: s };
        }
        let profil = crate::machine::detection::detecter_rapide();
        GardeSsd {
            seuil_go: crate::machine::politique(&profil).seuil_ssd_go,
        }
    }

    /// Go libres du système de fichiers qui contient `chemin` (statvfs ; remonte au premier parent existant).
    pub fn libre_go(chemin: &Path) -> Resultat<f64> {
        let existant = premier_parent_existant(chemin);
        let c = CString::new(existant.as_os_str().as_bytes())
            .map_err(|_| Erreur::Config(format!("chemin avec octet nul : {}", chemin.display())))?;
        let mut t = std::mem::MaybeUninit::<Statvfs>::zeroed();
        // SAFETY : `c` est une chaîne C valide ; `t` est un tampon de la taille de `struct statvfs`.
        let rc = unsafe { statvfs(c.as_ptr(), t.as_mut_ptr()) };
        if rc != 0 {
            return Err(Erreur::io(
                format!("statvfs {}", existant.display()),
                std::io::Error::last_os_error(),
            ));
        }
        // SAFETY : statvfs a rendu 0, le tampon est initialisé.
        let t = unsafe { t.assume_init() };
        let frsize = if t.f_frsize != 0 { t.f_frsize } else { t.f_bsize };
        Ok(t.f_bavail as f64 * frsize as f64 / OCTETS_PAR_GO)
    }

    /// Décision pure : `libre_go` Go libres, une écriture de `octets` octets sur `chemin`.
    /// Erreur SsdPlein si libre − octets/2³⁰ < seuil. Testable sans disque.
    pub fn evaluer(&self, chemin: &Path, libre_go: f64, octets: u64) -> Resultat<()> {
        let besoin_go = octets as f64 / OCTETS_PAR_GO;
        if libre_go - besoin_go < self.seuil_go {
            return Err(Erreur::SsdPlein {
                chemin: chemin.to_path_buf(),
                libre_go,
                besoin_go,
                seuil_go: self.seuil_go,
            });
        }
        Ok(())
    }

    /// Erreur SsdPlein si libre − octets/2³⁰ < seuil. À appeler AVANT chaque écriture lourde,
    /// et une fois au début d'un run avec la projection totale (`projeter`).
    pub fn verifier(&self, chemin: &Path, octets: u64) -> Resultat<()> {
        let libre = Self::libre_go(chemin)?;
        self.evaluer(chemin, libre, octets)
    }

    /// Marge actuelle au seuil (Go) : libre − seuil. Négative si la machine est déjà sous le seuil.
    pub fn marge_go(&self, chemin: &Path) -> Resultat<f64> {
        Ok(Self::libre_go(chemin)? - self.seuil_go)
    }
}

/// Premier ancêtre (le chemin lui-même compris) qui existe ; « . » pour un chemin relatif sans parent existant.
fn premier_parent_existant(chemin: &Path) -> std::path::PathBuf {
    let mut p = chemin;
    loop {
        if !p.as_os_str().is_empty() && std::fs::metadata(p).is_ok() {
            return p.to_path_buf();
        }
        match p.parent() {
            Some(q) if !q.as_os_str().is_empty() => p = q,
            _ => {
                return if chemin.is_absolute() {
                    std::path::PathBuf::from("/")
                } else {
                    std::path::PathBuf::from(".")
                }
            }
        }
    }
}

/// Taille projetée d'un run (octets) : dumps × taille d'une configuration (format, L) + CSV + reprises.
///
/// Par chaîne : `ndumps` fichiers du format demandé, `nreprises` copies du champ brut en double précision
/// (128·L⁴ octets : `reprise.champs` atomique = 2 copies à l'instant du remplacement) et
/// [`OCTETS_CSV_PAR_CHAINE`] de CSV ; plus [`OCTETS_FIXES_RUN`] pour le run.
/// Saturante : un dépassement de u64 rend `u64::MAX` (donc un refus).
pub fn projeter(l: usize, nchaines: usize, ndumps: u64, format: FormatChamp, nreprises: u64) -> u64 {
    let dump = octets_champ(l, format);
    let reprise = 128u64.saturating_mul((l as u64).saturating_pow(4));
    let par_chaine = dump
        .saturating_mul(ndumps)
        .saturating_add(reprise.saturating_mul(nreprises))
        .saturating_add(OCTETS_CSV_PAR_CHAINE);
    par_chaine
        .saturating_mul(nchaines as u64)
        .saturating_add(OCTETS_FIXES_RUN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statvfs_a_la_taille_de_la_glibc() {
        assert_eq!(std::mem::size_of::<Statvfs>(), 112);
    }

    #[test]
    fn seuils_par_machine() {
        assert_eq!(seuil_defaut(Machine::Acemagic), 100.0);
        assert_eq!(seuil_defaut(Machine::Gamer), 50.0);
        assert_eq!(seuil_defaut(Machine::Inconnue), 20.0);
        assert_eq!(GardeSsd::pour(Machine::Gamer).seuil_go, 50.0);
    }

    #[test]
    fn decision_pure_aux_bornes() {
        let g = GardeSsd::nouvelle(100.0);
        let p = Path::new("/x");
        // 160 Go libres, 60 Go (Gio) demandés : 100 restent, pile au seuil -> accepté
        let soixante = 60u64 << 30;
        assert!(g.evaluer(p, 160.0, soixante).is_ok());
        // un octet de plus -> refusé
        match g.evaluer(p, 160.0, soixante + 1) {
            Err(Erreur::SsdPlein {
                libre_go,
                besoin_go,
                seuil_go,
                ..
            }) => {
                assert_eq!(libre_go, 160.0);
                assert!(besoin_go > 60.0);
                assert_eq!(seuil_go, 100.0);
            }
            autre => panic!("refus attendu : {autre:?}"),
        }
        assert_eq!(g.evaluer(p, 90.0, 0).unwrap_err().code_sortie(), 7);
    }

    /// Vérificateur du 04/10 : la garde comptait 10⁹ octets par Go, la file et `doctor` 2³⁰ ; une seule unité.
    #[test]
    fn un_go_vaut_un_gio_comme_df_et_la_file() {
        assert_eq!(OCTETS_PAR_GO, 1_073_741_824.0);
        let g = GardeSsd::nouvelle(100.0);
        // 101 Go libres, 1 Gio demandé : 100 restent -> accepté ; 1 Gio + 1 octet -> refusé
        assert!(g.evaluer(Path::new("/x"), 101.0, 1 << 30).is_ok());
        assert!(g.evaluer(Path::new("/x"), 101.0, (1 << 30) + 1).is_err());
    }

    #[test]
    fn libre_go_remonte_au_premier_parent() {
        let a = GardeSsd::libre_go(Path::new("/tmp")).unwrap();
        let b = GardeSsd::libre_go(Path::new("/tmp/inexistant_kvu_t8/profond/encore")).unwrap();
        assert!((a - b).abs() < 5.0, "même système de fichiers : {a} contre {b}");
        assert!(GardeSsd::libre_go(Path::new("relatif/inexistant")).unwrap() > 0.0);
    }

    #[test]
    fn projection_c16_l64_20_dumps() {
        let p = projeter(64, 1, 20, FormatChamp::C16, 0);
        let attendu = 20 * (256u64 * 64u64.pow(4) + 128) + OCTETS_CSV_PAR_CHAINE + OCTETS_FIXES_RUN;
        assert_eq!(p, attendu);
        assert!((p as f64 / 1e9 - 85.9).abs() < 0.2, "{p}");
        // q64 : la moitié
        let q = projeter(64, 1, 20, FormatChamp::Q64, 0);
        assert!(q < p / 2 + (1 << 24));
        // saturation
        assert_eq!(
            projeter(usize::MAX, usize::MAX, u64::MAX, FormatChamp::C16, u64::MAX),
            u64::MAX
        );
    }

    #[test]
    fn projection_compte_chaines_et_reprises() {
        let seule = projeter(8, 1, 10, FormatChamp::Q64, 0);
        let quatre = projeter(8, 4, 10, FormatChamp::Q64, 0);
        assert!(quatre > 3 * (seule - OCTETS_FIXES_RUN));
        let avec_reprise = projeter(8, 1, 10, FormatChamp::Q64, 2);
        assert_eq!(avec_reprise - seule, 2 * 128 * 8u64.pow(4));
    }
}
