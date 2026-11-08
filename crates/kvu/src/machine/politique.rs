//! Politique de ressources par machine. Propriétaire : T1.

use super::detection::{machine_materielle, ProfilMachine};
use super::processus::{analyser_liste_cpus, formater_liste_cpus, intersection};
use crate::commun::{Machine, Precision};
use crate::config::{developper_chemin, ProfilConfig};
use std::path::{Path, PathBuf};

pub use super::compat_tests_existants::ALERTE_ACEMAGIC;

/// Règles d'exécution d'un job sur la machine courante (valeurs du profil de `kvu.toml`, surchargées par KEVINOTRON_*).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Politique {
    /// Fichier de verrou GPU du profil (clé `verrou`).
    pub verrou_gpu: PathBuf,
    /// Attente maximale du verrou (s). Défaut 7200.
    pub attente_verrou_s: u64,
    /// Fenêtre maximale d'un banc GPU sous verrou (s). Défaut 600.
    pub fenetre_banc_s: u64,
    pub nice: i32,
    /// Affinité CPU imposée par le profil (clé `cpus`) ; None sans affinité.
    pub cpus: Option<String>,
    /// Fils rayon maximum.
    pub fils_max: usize,
    /// Plafond mémoire systemd-run des jobs > 4 Go (clé `mem_max`) ; None : aucun plafond.
    pub mem_max: Option<String>,
    /// RAM disponible minimale pour lancer un job (Go) ; KEVINOTRON_RAM_MIN_GO.
    pub ram_min_go: f64,
    /// Garde SSD (Go libres minimum après écriture) ; KEVINOTRON_SSD_MIN_GO.
    pub seuil_ssd_go: f64,
    /// Fichier d'alerte du moniteur de charge (clé `alerte`) s'il existe.
    pub alerte_moniteur: Option<PathBuf>,
    /// Précision conseillée pour la génération de production (les gates de bit-identité exigent F64).
    pub precision_conseillee: Precision,
    /// Racine du store d'ensembles (KEVINOTRON_STORE).
    pub store: PathBuf,
    /// Racine de la file de jobs (KEVINOTRON_FILE).
    pub file: PathBuf,
    /// Hôte du câble direct vers l'autre machine (clé `cable`) ; None : aucun câble, aucun ssh.
    pub pair_cable: Option<String>,
}

/// Politique de la machine `p` (valeurs du profil de `kvu.toml`, puis surcharges d'environnement KEVINOTRON_SSD_MIN_GO, KEVINOTRON_RAM_MIN_GO, KEVINOTRON_CPUS,
/// KEVINOTRON_STORE, KEVINOTRON_FILE ; priorité : variable > fichier > défauts de `inconnue`).
pub fn politique(p: &ProfilMachine) -> Politique {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    politique_avec(p, &|k| std::env::var(k).ok(), &home)
}

/// Cœur pur de `politique` : `env` donne les variables d'environnement, `home` le dossier personnel.
///
/// Sécurité : quand la machine du profil (surcharge `inconnue`) diffère de celle du MATÉRIEL, le verrou GPU,
/// l'affinité et le moniteur restent ceux du matériel, et les planchers (SSD, RAM) sont les plus stricts des
/// deux : le GPU réellement utilisé garde son exclusion avec la production (vérificateur du 04/10).
pub fn politique_avec(p: &ProfilMachine, env: &dyn Fn(&str) -> Option<String>, home: &Path) -> Politique {
    let mut pol = defauts(p.machine, home);
    let materielle = machine_materielle(p);
    if materielle != Machine::Inconnue && materielle != p.machine {
        let sure = defauts(materielle, home);
        pol.verrou_gpu = sure.verrou_gpu;
        pol.cpus = sure.cpus;
        pol.alerte_moniteur = sure.alerte_moniteur;
        pol.seuil_ssd_go = pol.seuil_ssd_go.max(sure.seuil_ssd_go);
        pol.ram_min_go = pol.ram_min_go.max(sure.ram_min_go);
    }
    surcharger(&mut pol, env);
    pol
}

/// Valeurs du profil de la machine `m` dans `kvu.toml` (défauts compilés de `inconnue` si le profil est absent).
fn defauts(m: Machine, home: &Path) -> Politique {
    let c = crate::config::actuelle().profil(m);
    Politique {
        verrou_gpu: developper_chemin(&c.verrou, home),
        attente_verrou_s: c.attente_verrou_s,
        fenetre_banc_s: c.fenetre_banc_s,
        nice: c.nice,
        cpus: ProfilConfig::texte_optionnel(&c.cpus),
        fils_max: c.fils_max,
        mem_max: ProfilConfig::texte_optionnel(&c.mem_max),
        ram_min_go: c.ram_min_go,
        seuil_ssd_go: c.ssd_min_go,
        alerte_moniteur: ProfilConfig::chemin_optionnel(&c.alerte, home),
        // Valeur validée au chargement de la configuration ; F64 (la plus sûre) si elle ne se relit pas.
        precision_conseillee: c.precision.parse().unwrap_or(Precision::F64),
        store: developper_chemin(&c.store, home),
        file: developper_chemin(&c.file, home),
        pair_cable: ProfilConfig::texte_optionnel(&c.cable),
    }
}

/// Surcharges d'environnement. Une valeur invalide est IGNORÉE (on garde le défaut) : mieux vaut le seuil sûr
/// du tableau qu'un seuil absurde (négatif, NaN) qui désactiverait la garde SSD.
fn surcharger(pol: &mut Politique, env: &dyn Fn(&str) -> Option<String>) {
    if let Some(v) = env("KEVINOTRON_SSD_MIN_GO").and_then(|s| s.trim().replace(',', ".").parse::<f64>().ok()) {
        if v.is_finite() && v >= 0.0 {
            pol.seuil_ssd_go = v;
        }
    }
    // Même règle pour la RAM minimale d'admission des jobs de la file (tests hermétiques, ou décision consciente
    // sur une machine chargée ; le ram-guard du système reste actif).
    if let Some(v) = env("KEVINOTRON_RAM_MIN_GO").and_then(|s| s.trim().replace(',', ".").parse::<f64>().ok()) {
        if v.is_finite() && v >= 0.0 {
            pol.ram_min_go = v;
        }
    }
    if let Some(v) = env("KEVINOTRON_STORE").filter(|s| !s.trim().is_empty()) {
        pol.store = PathBuf::from(v.trim());
    }
    if let Some(v) = env("KEVINOTRON_FILE").filter(|s| !s.trim().is_empty()) {
        pol.file = PathBuf::from(v.trim());
    }
    // KEVINOTRON_CPUS ne peut que RESTREINDRE l'affinité obligatoire du profil (jamais rendre des cœurs réservés) ;
    // sans affinité dans le profil, elle en pose une.
    if let Some(Ok(voulus)) = env("KEVINOTRON_CPUS").map(|v| analyser_liste_cpus(&v)) {
        let retenus = match pol.cpus.as_deref().map(analyser_liste_cpus) {
            None => voulus,
            Some(Ok(base)) => intersection(&base, &voulus),
            Some(Err(_)) => Vec::new(),
        };
        if !retenus.is_empty() {
            pol.cpus = Some(formater_liste_cpus(&retenus));
        }
    }
}
