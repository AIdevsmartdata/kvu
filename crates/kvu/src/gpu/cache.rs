//! Cache disque des codes objets compilés (HSACO, CUBIN). Propriétaire : T2.
//!
//! Disposition (un « entrée » = trois noms de fichier sous le même radical) :
//!
//! ```text
//! <répertoire>/<nom>-<arch>-<sha16>.hsaco | .cubin     code objet
//! <répertoire>/<nom>-<arch>-<sha16>.json               métadonnées (sha256 et taille du code, clé, version)
//! ```
//!
//! - **Clé** = sha256(version du format ‖ texte assemblé ‖ arguments ‖ arch ‖ version du compilateur).
//!   `<sha16>` en est le préfixe de 16 caractères hexadécimaux.
//! - **Écriture atomique** : chaque fichier est écrit sous un nom temporaire puis renommé. Le JSON est publié
//!   AVANT le code objet : l'apparition du code objet est le point de validation, un lecteur concurrent ne voit
//!   donc jamais une entrée à moitié écrite.
//! - **Entrée corrompue** : code absent du JSON, de taille nulle, de taille ou de sha256 différents de ceux
//!   consignés ⇒ évincée (les deux fichiers supprimés) et traitée comme un manque (donc recompilée).
//! - **Plafond** : `KEVINOTRON_CACHE_MAX_MO` (défaut 256) ; après chaque écriture, les entrées les plus
//!   anciennes (date de modification du code objet, rafraîchie à chaque lecture réussie) sont évincées
//!   jusqu'à passer sous le plafond : éviction LRU.
//! - `KEVINOTRON_CACHE_DISABLE=1` désactive le cache ; `KEVINOTRON_CACHE_DIR` choisit le répertoire.

use crate::commun::{Erreur, Resultat};
use crate::heritage::manifest::sha256_hex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

/// Taille maximale du cache (octets) ; KEVINOTRON_CACHE_MAX_MO surcharge (défaut 256).
pub const TAILLE_MAX_DEFAUT: u64 = 256 * 1024 * 1024;

/// Version du format disque : elle entre dans la clé, donc un changement invalide tout le cache.
pub const VERSION_FORMAT: u32 = 1;

/// Répertoire du cache : $KEVINOTRON_CACHE_DIR, sinon $XDG_CACHE_HOME/kevinotron/kvu, sinon ~/.cache/kevinotron/kvu.
pub fn repertoire() -> Option<PathBuf> {
    repertoire_depuis(|k| std::env::var(k).ok())
}

/// Version pure de [`repertoire`] : `env` donne la valeur d'une variable d'environnement.
/// Une variable vide est ignorée.
pub fn repertoire_depuis(env: impl Fn(&str) -> Option<String>) -> Option<PathBuf> {
    let non_vide = |k: &str| env(k).filter(|v| !v.is_empty());
    if let Some(d) = non_vide("KEVINOTRON_CACHE_DIR") {
        return Some(PathBuf::from(d));
    }
    if let Some(x) = non_vide("XDG_CACHE_HOME") {
        return Some(PathBuf::from(x).join("kevinotron").join("kvu"));
    }
    non_vide("HOME").map(|h| PathBuf::from(h).join(".cache").join("kevinotron").join("kvu"))
}

/// Plafond en octets d'après la valeur de KEVINOTRON_CACHE_MAX_MO (absente ou invalide ⇒ défaut).
/// `0` est une valeur valide : le cache n'est alors jamais conservé.
pub fn taille_max_depuis(valeur: Option<&str>) -> u64 {
    match valeur.and_then(|v| v.trim().parse::<u64>().ok()) {
        Some(mo) => mo.saturating_mul(1024 * 1024),
        None => TAILLE_MAX_DEFAUT,
    }
}

/// KEVINOTRON_CACHE_DISABLE actif ? (toute valeur non vide autre que « 0 »).
pub fn desactive_depuis(valeur: Option<&str>) -> bool {
    matches!(valeur.map(str::trim), Some(v) if !v.is_empty() && v != "0")
}

/// Clé de cache (64 caractères hexadécimaux). Chaque ingrédient est préfixé par sa longueur : deux
/// découpages différents du même flux d'octets ne peuvent pas donner la même clé.
pub fn cle(texte: &str, arguments: &[String], arch: &str, version_compilateur: &str) -> String {
    let mut flux: Vec<u8> = Vec::with_capacity(texte.len() + 256);
    let mut champ = |octets: &[u8]| {
        flux.extend_from_slice(&(octets.len() as u64).to_le_bytes());
        flux.extend_from_slice(octets);
    };
    champ(&VERSION_FORMAT.to_le_bytes());
    champ(texte.as_bytes());
    champ(&(arguments.len() as u64).to_le_bytes());
    for a in arguments {
        champ(a.as_bytes());
    }
    champ(arch.as_bytes());
    champ(version_compilateur.as_bytes());
    sha256_hex(&flux)
}

/// Métadonnées d'une entrée (fichier .json voisin du code objet).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetaEntree {
    pub format: u32,
    pub nom: String,
    pub arch: String,
    pub cle: String,
    pub version_compilateur: String,
    pub arguments: Vec<String>,
    pub sha256_source: String,
    pub sha256_code: String,
    pub taille_code: u64,
}

/// Entrée lue et vérifiée.
#[derive(Clone, Debug)]
pub struct EntreeLue {
    pub code: Vec<u8>,
    pub meta: MetaEntree,
}

/// Cache disque : un répertoire et un plafond.
#[derive(Clone, Debug)]
pub struct Cache {
    dir: PathBuf,
    max_octets: u64,
}

static COMPTEUR_TMP: AtomicU64 = AtomicU64::new(0);

impl Cache {
    pub fn nouveau(dir: impl Into<PathBuf>, max_octets: u64) -> Self {
        Cache {
            dir: dir.into(),
            max_octets,
        }
    }

    /// Cache décrit par l'environnement ; `None` si KEVINOTRON_CACHE_DISABLE est actif ou si aucun
    /// répertoire ne peut être déterminé.
    pub fn depuis_env() -> Option<Self> {
        if desactive_depuis(std::env::var("KEVINOTRON_CACHE_DISABLE").ok().as_deref()) {
            return None;
        }
        let max = taille_max_depuis(std::env::var("KEVINOTRON_CACHE_MAX_MO").ok().as_deref());
        repertoire().map(|d| Cache::nouveau(d, max))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn max_octets(&self) -> u64 {
        self.max_octets
    }

    /// Radical de fichier : `<nom>-<arch>-<sha16>`. Les caractères hors [A-Za-z0-9_.] sont remplacés par « _ ».
    pub fn radical(nom: &str, arch: &str, cle: &str) -> String {
        let propre = |s: &str| -> String {
            s.chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '.' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect()
        };
        format!("{}-{}-{}", propre(nom), propre(arch), &cle[..cle.len().min(16)])
    }

    pub fn chemin_code(&self, nom: &str, arch: &str, cle: &str, ext: &str) -> PathBuf {
        self.dir.join(format!("{}.{ext}", Self::radical(nom, arch, cle)))
    }

    pub fn chemin_meta(&self, nom: &str, arch: &str, cle: &str) -> PathBuf {
        self.dir.join(format!("{}.json", Self::radical(nom, arch, cle)))
    }

    /// Relit une entrée. `None` = manque (absente, ou corrompue et alors évincée).
    /// Une lecture réussie rafraîchit la date de modification (ordre LRU).
    pub fn lire(&self, nom: &str, arch: &str, cle: &str, ext: &str) -> Option<EntreeLue> {
        let p_code = self.chemin_code(nom, arch, cle, ext);
        let p_meta = self.chemin_meta(nom, arch, cle);
        if !p_code.exists() {
            return None; // manque franc (ou entrée en cours de publication) : rien à évincer
        }
        match Self::verifier(&p_code, &p_meta, cle) {
            Some(e) => {
                rafraichir(&p_code);
                Some(e)
            }
            None => {
                let _ = fs::remove_file(&p_code);
                let _ = fs::remove_file(&p_meta);
                None
            }
        }
    }

    fn verifier(p_code: &Path, p_meta: &Path, cle: &str) -> Option<EntreeLue> {
        let code = fs::read(p_code).ok()?;
        if code.is_empty() {
            return None;
        }
        let meta: MetaEntree = serde_json::from_slice(&fs::read(p_meta).ok()?).ok()?;
        let ok = meta.format == VERSION_FORMAT
            && meta.cle == cle
            && meta.taille_code == code.len() as u64
            && meta.sha256_code == sha256_hex(&code);
        ok.then_some(EntreeLue { code, meta })
    }

    /// Écrit une entrée de façon atomique puis applique le plafond. Le JSON est renommé avant le code objet.
    pub fn ecrire(
        &self,
        nom: &str,
        arch: &str,
        cle: &str,
        ext: &str,
        code: &[u8],
        mut meta: MetaEntree,
    ) -> Resultat<()> {
        if code.is_empty() {
            return Err(Erreur::Format(format!("cache : code objet vide pour {nom}")));
        }
        fs::create_dir_all(&self.dir)
            .map_err(|e| Erreur::io(format!("création du cache {}", self.dir.display()), e))?;
        meta.format = VERSION_FORMAT;
        meta.cle = cle.to_string();
        meta.sha256_code = sha256_hex(code);
        meta.taille_code = code.len() as u64;
        let json = serde_json::to_vec_pretty(&meta).map_err(|e| Erreur::Format(format!("cache : JSON : {e}")))?;
        self.ecrire_atomique(&self.chemin_meta(nom, arch, cle), &json)?;
        self.ecrire_atomique(&self.chemin_code(nom, arch, cle, ext), code)?;
        self.appliquer_plafond();
        Ok(())
    }

    fn ecrire_atomique(&self, cible: &Path, octets: &[u8]) -> Resultat<()> {
        let n = COMPTEUR_TMP.fetch_add(1, Ordering::Relaxed);
        let tmp = self.dir.join(format!(
            ".tmp-{}-{n}-{}",
            std::process::id(),
            cible.file_name().unwrap().to_string_lossy()
        ));
        fs::write(&tmp, octets).map_err(|e| Erreur::io(format!("écriture {}", tmp.display()), e))?;
        fs::rename(&tmp, cible).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            Erreur::io(format!("renommage vers {}", cible.display()), e)
        })
    }

    /// Entrées présentes : (radical, extension du code, taille totale code + json, date du code).
    /// Les fichiers temporaires et les JSON orphelins ne sont pas des entrées.
    fn entrees(&self) -> Vec<(String, String, u64, SystemTime)> {
        let Ok(rd) = fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut v = Vec::new();
        for e in rd.flatten() {
            let chemin = e.path();
            let (Some(stem), Some(ext)) = (chemin.file_stem(), chemin.extension()) else {
                continue;
            };
            let (stem, ext) = (stem.to_string_lossy().to_string(), ext.to_string_lossy().to_string());
            if ext != "hsaco" && ext != "cubin" {
                continue;
            }
            let Ok(md) = e.metadata() else { continue };
            let json = fs::metadata(self.dir.join(format!("{stem}.json")))
                .map(|m| m.len())
                .unwrap_or(0);
            v.push((
                stem,
                ext,
                md.len() + json,
                md.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            ));
        }
        v
    }

    /// Taille totale occupée par les entrées (codes objets + JSON), en octets.
    pub fn taille_totale(&self) -> u64 {
        self.entrees().iter().map(|e| e.2).sum()
    }

    /// Évince les entrées les plus anciennes jusqu'à ce que la taille totale tienne dans le plafond.
    /// Retourne le nombre d'entrées évincées. Nettoie aussi les temporaires abandonnés (plus d'une heure).
    pub fn appliquer_plafond(&self) -> usize {
        self.nettoyer_temporaires();
        let mut v = self.entrees();
        let mut total: u64 = v.iter().map(|e| e.2).sum();
        if total <= self.max_octets {
            return 0;
        }
        // Plus ancien d'abord ; à date égale, ordre alphabétique (déterministe).
        v.sort_by(|a, b| a.3.cmp(&b.3).then_with(|| a.0.cmp(&b.0)));
        let mut evinces = 0;
        for (stem, ext, taille, _) in v {
            if total <= self.max_octets {
                break;
            }
            let _ = fs::remove_file(self.dir.join(format!("{stem}.{ext}")));
            let _ = fs::remove_file(self.dir.join(format!("{stem}.json")));
            total = total.saturating_sub(taille);
            evinces += 1;
        }
        evinces
    }

    fn nettoyer_temporaires(&self) {
        let Ok(rd) = fs::read_dir(&self.dir) else { return };
        for e in rd.flatten() {
            if !e.file_name().to_string_lossy().starts_with(".tmp-") {
                continue;
            }
            let vieux = e
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|d| d.as_secs() > 3600);
            if vieux {
                let _ = fs::remove_file(e.path());
            }
        }
    }
}

/// Met la date de modification à « maintenant » (ordre LRU). Échec silencieux : sans conséquence.
fn rafraichir(p: &Path) {
    if let Ok(f) = fs::OpenOptions::new().write(true).open(p) {
        let _ = f.set_modified(SystemTime::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repertoire_priorites() {
        let e = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
        };
        assert_eq!(
            repertoire_depuis(e(&[
                ("KEVINOTRON_CACHE_DIR", "/x"),
                ("XDG_CACHE_HOME", "/y"),
                ("HOME", "/h")
            ])),
            Some(PathBuf::from("/x"))
        );
        assert_eq!(
            repertoire_depuis(e(&[("XDG_CACHE_HOME", "/y"), ("HOME", "/h")])),
            Some(PathBuf::from("/y/kevinotron/kvu"))
        );
        assert_eq!(
            repertoire_depuis(e(&[("HOME", "/h")])),
            Some(PathBuf::from("/h/.cache/kevinotron/kvu"))
        );
        assert_eq!(
            repertoire_depuis(e(&[("KEVINOTRON_CACHE_DIR", ""), ("HOME", "/h")])),
            Some(PathBuf::from("/h/.cache/kevinotron/kvu"))
        );
        assert_eq!(repertoire_depuis(e(&[])), None);
    }

    #[test]
    fn plafond_et_desactivation() {
        assert_eq!(taille_max_depuis(None), TAILLE_MAX_DEFAUT);
        assert_eq!(taille_max_depuis(Some("64")), 64 << 20);
        assert_eq!(taille_max_depuis(Some(" 1 ")), 1 << 20);
        assert_eq!(taille_max_depuis(Some("0")), 0);
        assert_eq!(taille_max_depuis(Some("abc")), TAILLE_MAX_DEFAUT);
        assert_eq!(taille_max_depuis(Some("-3")), TAILLE_MAX_DEFAUT);
        assert!(desactive_depuis(Some("1")));
        assert!(desactive_depuis(Some("oui")));
        assert!(!desactive_depuis(Some("0")));
        assert!(!desactive_depuis(Some("")));
        assert!(!desactive_depuis(None));
    }

    #[test]
    fn cle_sans_ambiguite_de_decoupage() {
        let a = cle("ab", &["c".into()], "x", "v");
        let b = cle("a", &["bc".into()], "x", "v");
        assert_ne!(a, b);
        assert_eq!(a.len(), 64);
        assert_eq!(a, cle("ab", &["c".into()], "x", "v"));
    }

    #[test]
    fn radical_assaini() {
        assert_eq!(
            Cache::radical("su2/ref 64", "sm_120", "0123456789abcdef0123"),
            "su2_ref_64-sm_120-0123456789abcdef"
        );
    }
}
