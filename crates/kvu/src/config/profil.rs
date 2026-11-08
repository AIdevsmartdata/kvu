//! Profil de machine de `kvu.toml` : clés admises, valeurs par défaut (celles du profil `inconnue`), validation.
//!
//! Les défauts compilés ci-dessous sont ceux du profil prudent `inconnue` et d'AUCUN autre profil : tout ce qui
//! distingue une machine connue (hôtes, GPU, chemins, seuils) est écrit dans `kvu.toml`.

use super::toml_min::{Entree, SousTable, Valeur};
use crate::commun::BackendKind;
use std::path::{Path, PathBuf};

/// Réglages d'un profil. Les champs textuels vides valent « absent » (pas d'affinité, pas de câble, etc.).
#[derive(Clone, Debug, PartialEq)]
pub struct ProfilConfig {
    /// Noms d'hôte exacts qui reconnaissent la machine.
    pub hotes: Vec<String>,
    /// Sous-chaînes (sans tenir compte de la casse) du modèle de CPU qui reconnaissent la machine.
    pub cpu: Vec<String>,
    /// Architectures de GPU visibles (« gfxNNNN », « sm_XY ») qui reconnaissent la machine.
    pub gpu_arch: Vec<String>,
    /// Backend préféré en mode auto : `auto` (premier GPU utilisable), `hip`, `cuda` ou `cpu`.
    pub backend: String,
    /// Fichier de verrou GPU.
    pub verrou: String,
    pub attente_verrou_s: u64,
    pub fenetre_banc_s: u64,
    pub nice: i32,
    /// Affinité CPU imposée (« 12-19 ») ; vide : aucune.
    pub cpus: String,
    pub fils_max: usize,
    /// Plafond mémoire systemd-run (« 24G ») ; vide : aucun.
    pub mem_max: String,
    pub ram_min_go: f64,
    pub ssd_min_go: f64,
    /// Fichier d'alerte du moniteur de charge ; vide : aucun.
    pub alerte: String,
    /// Précision conseillée : `f32` ou `f64`.
    pub precision: String,
    /// Racine du store d'ensembles (« ~/ » = répertoire personnel).
    pub store: String,
    /// Racine de la file de jobs.
    pub file: String,
    /// Hôte du câble direct vers l'autre machine ; vide : aucun câble (ni ssh).
    pub cable: String,
    /// Binaire de référence de `kvu bench` ; vide : aucun.
    pub reference_bench: String,
    /// Surcharge de l'architecture HIP à compiler ; vide : lue sur le périphérique.
    pub arch_hip: String,
    /// Surcharge de l'architecture CUDA (« sm_XY ») ; vide : lue sur le périphérique.
    pub arch_cuda: String,
}

/// Clés admises dans un profil (toute autre clé est rejetée avec sa position).
pub const CLES: &[&str] = &[
    "hotes",
    "cpu",
    "gpu_arch",
    "backend",
    "verrou",
    "attente_verrou_s",
    "fenetre_banc_s",
    "nice",
    "cpus",
    "fils_max",
    "mem_max",
    "ram_min_go",
    "ssd_min_go",
    "alerte",
    "precision",
    "store",
    "file",
    "cable",
    "reference_bench",
    "arch_hip",
    "arch_cuda",
];

/// Erreur de validation sémantique avec la position de la clé fautive.
#[derive(Clone, Debug, PartialEq)]
pub struct ErreurProfil {
    pub ligne: usize,
    pub colonne: usize,
    pub message: String,
}

impl ProfilConfig {
    /// Défauts compilés : le profil prudent `inconnue` (CPU/GPU détectés, garde SSD 20 Go, RAM 8 Go, MemoryMax 8G).
    pub fn defauts_inconnue() -> ProfilConfig {
        ProfilConfig {
            hotes: Vec::new(),
            cpu: Vec::new(),
            gpu_arch: Vec::new(),
            backend: "auto".to_string(),
            verrou: "/tmp/kevinotron_gpu.lock".to_string(),
            attente_verrou_s: 7200,
            fenetre_banc_s: 600,
            nice: 10,
            cpus: String::new(),
            fils_max: 4,
            mem_max: "8G".to_string(),
            ram_min_go: 8.0,
            ssd_min_go: 20.0,
            alerte: String::new(),
            precision: "f64".to_string(),
            store: "~/kevinotron_store".to_string(),
            file: "~/.local/state/kevinotron/file".to_string(),
            cable: String::new(),
            reference_bench: String::new(),
            arch_hip: String::new(),
            arch_cuda: String::new(),
        }
    }

    /// Le profil porte-t-il au moins un motif de reconnaissance ?
    pub fn reconnaissable(&self) -> bool {
        !(self.hotes.is_empty() && self.cpu.is_empty() && self.gpu_arch.is_empty())
    }

    /// Vrai si l'hôte, le CPU ou une architecture de GPU correspond à un motif du profil.
    pub fn correspond(&self, hote: &str, cpu: &str, archs: &[&str]) -> bool {
        let cpu_maj = cpu.to_ascii_uppercase();
        let hote = hote.trim();
        self.hotes.iter().any(|h| h == hote)
            || self
                .cpu
                .iter()
                .any(|m| !m.is_empty() && cpu_maj.contains(&m.to_ascii_uppercase()))
            || archs.iter().any(|a| self.gpu_arch.iter().any(|g| g == a))
    }

    /// Chemin (`~/` = répertoire personnel) ; None si le texte est vide.
    pub fn chemin_optionnel(texte: &str, home: &Path) -> Option<PathBuf> {
        let t = texte.trim();
        (!t.is_empty()).then(|| developper_chemin(t, home))
    }

    /// Texte optionnel : vide ⇒ None.
    pub fn texte_optionnel(texte: &str) -> Option<String> {
        let t = texte.trim();
        (!t.is_empty()).then(|| t.to_string())
    }

    /// Backend préféré du profil quand il désigne un GPU précis (`hip`, `cuda`).
    pub fn backend_prefere(&self) -> Option<BackendKind> {
        match self.backend.as_str() {
            "hip" => Some(BackendKind::Hip),
            "cuda" => Some(BackendKind::Cuda),
            _ => None,
        }
    }

    /// Construit un profil depuis une table TOML : les clés absentes gardent les défauts `inconnue`.
    pub fn depuis_table(table: &SousTable) -> Result<ProfilConfig, ErreurProfil> {
        let mut p = ProfilConfig::defauts_inconnue();
        for (cle, e) in &table.entrees {
            let err = |m: String| ErreurProfil {
                ligne: e.ligne,
                colonne: e.colonne,
                message: m,
            };
            match cle.as_str() {
                "hotes" => p.hotes = tableau(cle, e)?,
                "cpu" => p.cpu = tableau(cle, e)?,
                "gpu_arch" => p.gpu_arch = tableau(cle, e)?,
                "backend" => {
                    let b = chaine(cle, e)?.to_ascii_lowercase();
                    if !matches!(b.as_str(), "auto" | "hip" | "cuda" | "cpu") {
                        return Err(err(format!("backend=« {b} » : valeurs auto|hip|cuda|cpu")));
                    }
                    p.backend = b;
                }
                "verrou" => p.verrou = non_vide(cle, e)?,
                "attente_verrou_s" => p.attente_verrou_s = entier_positif(cle, e)? as u64,
                "fenetre_banc_s" => p.fenetre_banc_s = entier_positif(cle, e)? as u64,
                "nice" => {
                    let n = entier(cle, e)?;
                    if !(-20..=19).contains(&n) {
                        return Err(err(format!("nice={n} : entier entre -20 et 19 attendu")));
                    }
                    p.nice = n as i32;
                }
                "cpus" => {
                    let c = chaine(cle, e)?;
                    if !c.trim().is_empty() && crate::machine::processus::analyser_liste_cpus(&c).is_err() {
                        return Err(err(format!("cpus=« {c} » : liste de cœurs invalide (ex. « 12-19 »)")));
                    }
                    p.cpus = c.trim().to_string();
                }
                "fils_max" => {
                    let n = entier_positif(cle, e)?;
                    if n == 0 {
                        return Err(err("fils_max : au moins 1 fil".to_string()));
                    }
                    p.fils_max = n as usize;
                }
                "mem_max" => {
                    let m = chaine(cle, e)?;
                    if !m.trim().is_empty() && !plafond_valide(m.trim()) {
                        return Err(err(format!("mem_max=« {m} » : forme « 24G » (entier + K, M, G ou T) attendue")));
                    }
                    p.mem_max = m.trim().to_string();
                }
                "ram_min_go" => p.ram_min_go = reel_positif(cle, e)?,
                "ssd_min_go" => p.ssd_min_go = reel_positif(cle, e)?,
                "alerte" => p.alerte = chaine(cle, e)?,
                "precision" => {
                    let c = chaine(cle, e)?.to_ascii_lowercase();
                    if c.parse::<crate::commun::Precision>().is_err() {
                        return Err(err(format!("precision=« {c} » : valeurs f64|f32")));
                    }
                    p.precision = c;
                }
                "store" => p.store = non_vide(cle, e)?,
                "file" => p.file = non_vide(cle, e)?,
                "cable" => p.cable = chaine(cle, e)?,
                "reference_bench" => p.reference_bench = chaine(cle, e)?,
                "arch_hip" => p.arch_hip = chaine(cle, e)?.trim().to_string(),
                "arch_cuda" => p.arch_cuda = chaine(cle, e)?.trim().to_string(),
                autre => {
                    return Err(err(format!(
                        "clé inconnue « {autre} » dans le profil (clés admises : {})",
                        CLES.join(", ")
                    )))
                }
            }
        }
        Ok(p)
    }
}

/// Développe `~/` et `~` en répertoire personnel ; tout autre chemin est gardé tel quel.
pub fn developper_chemin(texte: &str, home: &Path) -> PathBuf {
    if texte == "~" {
        home.to_path_buf()
    } else if let Some(reste) = texte.strip_prefix("~/") {
        home.join(reste)
    } else {
        PathBuf::from(texte)
    }
}

fn plafond_valide(m: &str) -> bool {
    let (chiffres, unite) = m.split_at(m.len().saturating_sub(1));
    !chiffres.is_empty()
        && chiffres.chars().all(|c| c.is_ascii_digit())
        && matches!(unite, "K" | "M" | "G" | "T" | "k" | "m" | "g" | "t")
}

fn type_attendu(cle: &str, e: &Entree, attendu: &str) -> ErreurProfil {
    ErreurProfil {
        ligne: e.ligne,
        colonne: e.colonne,
        message: format!("« {cle} » : {attendu} attendu"),
    }
}

fn chaine(cle: &str, e: &Entree) -> Result<String, ErreurProfil> {
    match &e.valeur {
        Valeur::Chaine(s) => Ok(s.clone()),
        _ => Err(type_attendu(cle, e, "chaîne")),
    }
}

fn non_vide(cle: &str, e: &Entree) -> Result<String, ErreurProfil> {
    let s = chaine(cle, e)?;
    if s.trim().is_empty() {
        return Err(type_attendu(cle, e, "chaîne non vide"));
    }
    Ok(s.trim().to_string())
}

fn entier(cle: &str, e: &Entree) -> Result<i64, ErreurProfil> {
    match &e.valeur {
        Valeur::Entier(n) => Ok(*n),
        _ => Err(type_attendu(cle, e, "entier")),
    }
}

fn entier_positif(cle: &str, e: &Entree) -> Result<i64, ErreurProfil> {
    let n = entier(cle, e)?;
    if n < 0 {
        return Err(type_attendu(cle, e, "entier positif ou nul"));
    }
    Ok(n)
}

fn reel_positif(cle: &str, e: &Entree) -> Result<f64, ErreurProfil> {
    let x = match &e.valeur {
        Valeur::Flottant(x) => *x,
        Valeur::Entier(n) => *n as f64,
        _ => return Err(type_attendu(cle, e, "nombre")),
    };
    if !x.is_finite() || x < 0.0 {
        return Err(type_attendu(cle, e, "nombre fini positif ou nul"));
    }
    Ok(x)
}

fn tableau(cle: &str, e: &Entree) -> Result<Vec<String>, ErreurProfil> {
    match &e.valeur {
        Valeur::Tableau(v) => Ok(v.clone()),
        _ => Err(type_attendu(cle, e, "tableau de chaînes")),
    }
}
