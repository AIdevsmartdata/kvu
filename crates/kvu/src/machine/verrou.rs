//! Verrou GPU par machine (flock), sans interblocage avec un `flock` externe. Propriétaire : T1.
//!
//! Tout run GPU prend le verrou de sa machine, SAUF s'il est déjà tenu par un ANCÊTRE : c'est le
//! cas de `flock -w 7200 /tmp/kevinotron_gpu.lock kevinotron …` (usage des chantiers). Sans cette
//! détection, le `flock` interne attendrait celui de son propre parent jusqu'à expiration.
//! Détection : /proc/locks donne (pid, maj:min:inode) de chaque flock ; on compare à l'inode et au
//! périphérique du fichier de verrou, puis on remonte la chaîne des ppid (/proc/<pid>/stat).
//!
//! Mode PARTAGÉ (lot WP21, docs/MACHINES.md §8) : sous `gpu_partage` (variables `GPU_PARTAGE=1` et
//! `GPU_PARTAGE_VERROU_TENU` = même fichier que notre verrou), le verrou est pris en LOCK_SH : l'enveloppe
//! tient déjà le fichier en partagé par un `flock -s` éphémère (PID mort dans /proc/locks, donc pas
//! d'ancêtre reconnu) et une prise exclusive attendrait derrière son propre parent.

use crate::commun::{Erreur, Resultat};
use std::fs::{File, OpenOptions, TryLockError};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Pas de la boucle d'attente (try_lock + sommeil).
const PAS_ATTENTE: Duration = Duration::from_millis(500);

/// Verrou exclusif du GPU de la machine (fichier de la politique).
pub struct VerrouGpu {
    pub chemin: PathBuf,
}

/// Comment le verrou est tenu (consigné dans le manifeste : `verrou: pris_ici|ancetre|partage|ignore`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeVerrou {
    PrisIci,
    Ancetre,
    /// LOCK_SH pris par CE processus sous `gpu_partage` (GPU_PARTAGE=1, même fichier).
    Partage,
    Ignore,
}

impl ModeVerrou {
    pub fn nom(self) -> &'static str {
        match self {
            ModeVerrou::PrisIci => "pris_ici",
            ModeVerrou::Ancetre => "ancetre",
            ModeVerrou::Partage => "partage",
            ModeVerrou::Ignore => "ignore",
        }
    }
}

/// Variable posée par `gpu_partage` (v2.1) : `1` sous l'enveloppe.
pub const ENV_GPU_PARTAGE: &str = "GPU_PARTAGE";
/// Variable posée par `gpu_partage` (v2.1) : chemin du verrou qu'elle tient en partagé.
pub const ENV_GPU_PARTAGE_VERROU_TENU: &str = "GPU_PARTAGE_VERROU_TENU";

/// Règle du mode partagé (pure) : `GPU_PARTAGE` vaut `1` (espaces ignorés) ET le fichier tenu par l'enveloppe
/// est le même que notre verrou (identités `(st_dev, st_ino)` connues et égales).
pub fn partage_demande(gpu_partage: Option<&str>, identite_tenu: Option<(u64, u64)>, identite_verrou: Option<(u64, u64)>) -> bool {
    gpu_partage.map(str::trim) == Some("1") && identite_tenu.is_some() && identite_tenu == identite_verrou
}

/// Garde RAII : le verrou est rendu à la destruction (ou n'a jamais été pris, s'il était
/// déjà tenu par un processus ancêtre).
pub struct GardeVerrou {
    pub chemin: PathBuf,
    /// Vrai si CE processus tient le verrou (exclusif, ou partagé sous `gpu_partage`) ; faux si un ancêtre le
    /// tenait déjà (ex. `flock -w 7200 /tmp/kevinotron_gpu.lock kevinotron …`) — détecté par /proc/locks.
    pub pris_ici: bool,
    mode: ModeVerrou,
    /// Descripteur qui porte le flock ; sa fermeture (Drop) rend le verrou.
    _fichier: Option<File>,
}

impl GardeVerrou {
    /// `pris_ici`, `ancetre`, `partage` ou `ignore`, pour le manifeste.
    pub fn mode(&self) -> &'static str {
        self.mode.nom()
    }
}

impl Drop for GardeVerrou {
    fn drop(&mut self) {
        if let Some(f) = &self._fichier {
            let _ = f.unlock();
        }
    }
}

/// Une ligne FLOCK de /proc/locks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntreeLock {
    pub pid: u32,
    pub majeur: u32,
    pub mineur: u32,
    pub inode: u64,
}

/// Analyse /proc/locks. Format : `1: FLOCK  ADVISORY  WRITE 2302 00:3f:78 0 EOF`
/// (majeur:mineur en hexadécimal, inode en décimal). Les verrous POSIX/OFD et les lignes
/// illisibles (« -> » des bloqués, etc.) sont ignorés.
pub fn analyser_locks(texte: &str) -> Vec<EntreeLock> {
    texte
        .lines()
        .filter_map(|l| {
            let champs: Vec<&str> = l.split_whitespace().collect();
            // Les lignes de processus bloqués commencent par « 1: -> FLOCK … » : un jeton en plus.
            let debut = champs.iter().position(|c| *c == "FLOCK")?;
            let pid: u32 = champs.get(debut + 3)?.parse().ok()?;
            let mut it = champs.get(debut + 4)?.split(':');
            let majeur = u32::from_str_radix(it.next()?, 16).ok()?;
            let mineur = u32::from_str_radix(it.next()?, 16).ok()?;
            let inode: u64 = it.next()?.parse().ok()?;
            if debut > 0 && champs[debut - 1] == "->" {
                return None; // ligne d'un processus qui ATTEND le verrou
            }
            Some(EntreeLock {
                pid,
                majeur,
                mineur,
                inode,
            })
        })
        .collect()
}

/// (majeur, mineur) d'un `st_dev` Linux (encodage glibc `gnu_dev_major/minor`).
pub fn majeur_mineur(dev: u64) -> (u32, u32) {
    let majeur = (((dev >> 8) & 0xfff) | ((dev >> 32) & !0xfff)) as u32;
    let mineur = ((dev & 0xff) | ((dev >> 12) & !0xff)) as u32;
    (majeur, mineur)
}

/// ppid lu dans le contenu de /proc/<pid>/stat. Le nom du processus (entre parenthèses) peut
/// contenir des espaces et des parenthèses : on coupe après la DERNIÈRE « ) ».
pub fn ppid_depuis_stat(stat: &str) -> Option<u32> {
    let apres = &stat[stat.rfind(')')? + 1..];
    // « S 1234 … » : l'état, puis le ppid.
    apres.split_whitespace().nth(1)?.parse().ok()
}

/// Chaîne des ancêtres de `pid` (parent, grand-parent… jusqu'à 1 inclus), lue sous `proc_racine`.
/// Borne de sécurité : 4096 niveaux (cycle impossible en vrai, possible dans un fichier simulé).
pub fn ancetres(proc_racine: &Path, pid: u32) -> Vec<u32> {
    let mut chaine = Vec::new();
    let mut courant = pid;
    while courant > 1 && chaine.len() < 4096 {
        let Ok(stat) = std::fs::read_to_string(proc_racine.join(courant.to_string()).join("stat")) else {
            break;
        };
        let Some(pp) = ppid_depuis_stat(&stat) else { break };
        if pp == 0 || chaine.contains(&pp) {
            break;
        }
        chaine.push(pp);
        courant = pp;
    }
    chaine
}

/// Vrai si KEVINOTRON_VERROU=ignorer (tests CPU seulement).
pub fn verrou_ignore() -> bool {
    std::env::var("KEVINOTRON_VERROU").is_ok_and(|v| v.trim().eq_ignore_ascii_case("ignorer"))
}

impl VerrouGpu {
    pub fn nouveau(chemin: &Path) -> Self {
        VerrouGpu {
            chemin: chemin.to_path_buf(),
        }
    }

    /// Ouvre le fichier de verrou comme `flock(1)` : lecture seule si possible (le flock n'exige
    /// pas l'écriture, et /tmp est « sticky » : créer un fichier d'un autre utilisateur échouerait),
    /// création seulement s'il n'existe pas.
    fn ouvrir(&self) -> Resultat<File> {
        match OpenOptions::new().read(true).open(&self.chemin) {
            Ok(f) => Ok(f),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&self.chemin)
                .map_err(|e| Erreur::io(format!("création du verrou {}", self.chemin.display()), e)),
            Err(e) => Err(Erreur::io(format!("ouverture du verrou {}", self.chemin.display()), e)),
        }
    }

    /// (périphérique, inode) du fichier de verrou, s'il existe.
    fn identite(&self) -> Option<(u32, u32, u64)> {
        let m = std::fs::metadata(&self.chemin).ok()?;
        let (maj, min) = majeur_mineur(m.dev());
        Some((maj, min, m.ino()))
    }

    /// PID des détenteurs actuels du verrou, d'après /proc/locks (vide si libre ou illisible).
    pub fn detenteurs(&self) -> Vec<u32> {
        let Some((maj, min, ino)) = self.identite() else {
            return Vec::new();
        };
        let Ok(texte) = std::fs::read_to_string("/proc/locks") else {
            return Vec::new();
        };
        analyser_locks(&texte)
            .into_iter()
            .filter(|e| e.majeur == maj && e.mineur == min && e.inode == ino)
            .map(|e| e.pid)
            .collect()
    }

    /// Vrai si un ancêtre de CE processus tient le verrou.
    pub fn tenu_par_ancetre(&self) -> bool {
        let det = self.detenteurs();
        !det.is_empty() && {
            let anc = ancetres(Path::new("/proc"), std::process::id());
            det.iter().any(|p| anc.contains(p))
        }
    }

    /// Vrai si le mode partagé s'applique à CE verrou, le fichier étant ouvert (règle [`partage_demande`]). Écrit une ligne
    /// d'avertissement sur stderr si `GPU_PARTAGE=1` est posé sans correspondance (la prise reste alors exclusive).
    fn condition_partage(&self, fichier: &File) -> bool {
        let gpu_partage = std::env::var(ENV_GPU_PARTAGE).ok();
        let tenu = std::env::var_os(ENV_GPU_PARTAGE_VERROU_TENU).filter(|t| !t.is_empty());
        let identite_tenu = tenu.as_ref().and_then(|t| std::fs::metadata(t).ok()).map(|m| (m.dev(), m.ino()));
        let identite_verrou = fichier.metadata().ok().map(|m| (m.dev(), m.ino()));
        let partage = partage_demande(gpu_partage.as_deref(), identite_tenu, identite_verrou);
        if !partage && gpu_partage.as_deref().map(str::trim) == Some("1") {
            eprintln!(
                "kvu : GPU_PARTAGE=1 mais le verrou tenu par gpu_partage ({}) n'est pas le verrou de kvu ({}) : prise exclusive",
                tenu.as_ref().map_or_else(|| "absent".to_string(), |t| Path::new(t).display().to_string()),
                self.chemin.display()
            );
        }
        partage
    }

    fn garde(&self, mode: ModeVerrou, fichier: Option<File>) -> GardeVerrou {
        GardeVerrou {
            chemin: self.chemin.clone(),
            pris_ici: matches!(mode, ModeVerrou::PrisIci | ModeVerrou::Partage),
            mode,
            _fichier: fichier,
        }
    }

    fn delai_depasse(&self, attente: Duration) -> Erreur {
        let det = self.detenteurs();
        Erreur::Verrou(format!(
            "{} non obtenu en {} s (détenteur : {})",
            self.chemin.display(),
            attente.as_secs(),
            if det.is_empty() {
                "inconnu".to_string()
            } else {
                det.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(",")
            }
        ))
    }

    /// Prend le verrou (flock LOCK_EX, attente bornée), SAUF s'il est déjà tenu par un ancêtre :
    /// on lit /proc/locks (inode du fichier) et la chaîne des parents (/proc/<pid>/stat).
    /// KEVINOTRON_VERROU=ignorer le saute (tests CPU seulement, consigné dans le manifeste).
    /// Sous `gpu_partage` (règle [`partage_demande`]) : `ancetre` si un ancêtre vivant tient déjà le verrou, sinon LOCK_SH
    /// (mode `partage`).
    ///
    /// Boucle try_lock + sommeil de 0,5 s jusqu'au délai ; au-delà : `Erreur::Verrou` (rc 6).
    pub fn acquerir(&self, attente: Duration) -> Resultat<GardeVerrou> {
        if verrou_ignore() {
            return Ok(self.garde(ModeVerrou::Ignore, None));
        }
        let fichier = self.ouvrir()?;
        if self.condition_partage(&fichier) {
            self.acquerir_partage(fichier, attente)
        } else {
            self.acquerir_exclusif_ouvert(fichier, attente)
        }
    }

    /// Prise EXCLUSIVE exigée (bancs) : sous `gpu_partage` (règle [`partage_demande`]), refus immédiat (`Erreur::Verrou`,
    /// rc 6) au lieu d'attendre un verrou exclusif que l'enveloppe parente empêche ; sinon comme [`VerrouGpu::acquerir`].
    pub fn acquerir_exclusif(&self, attente: Duration) -> Resultat<GardeVerrou> {
        if verrou_ignore() {
            return Ok(self.garde(ModeVerrou::Ignore, None));
        }
        let fichier = self.ouvrir()?;
        if self.condition_partage(&fichier) {
            return Err(Erreur::Verrou(format!(
                "exclusivité requise : sous gpu_partage (GPU_PARTAGE=1), {} est tenu en partagé par l'enveloppe ; \
                 lancer sous flock -w 7200 {} (fenêtre exclusive)",
                self.chemin.display(),
                self.chemin.display()
            )));
        }
        self.acquerir_exclusif_ouvert(fichier, attente)
    }

    /// Chemin exclusif (comportement d'avant WP21, inchangé) sur le fichier déjà ouvert.
    fn acquerir_exclusif_ouvert(&self, fichier: File, attente: Duration) -> Resultat<GardeVerrou> {
        let echeance = Instant::now() + attente;
        loop {
            match fichier.try_lock() {
                Ok(()) => return Ok(self.garde(ModeVerrou::PrisIci, Some(fichier))),
                Err(TryLockError::WouldBlock) => {
                    if self.tenu_par_ancetre() {
                        return Ok(self.garde(ModeVerrou::Ancetre, None));
                    }
                    let restant = echeance.saturating_duration_since(Instant::now());
                    if restant.is_zero() {
                        return Err(self.delai_depasse(attente));
                    }
                    std::thread::sleep(PAS_ATTENTE.min(restant));
                }
                Err(TryLockError::Error(e)) => {
                    return Err(Erreur::io(format!("flock {}", self.chemin.display()), e));
                }
            }
        }
    }

    /// Chemin partagé (sous `gpu_partage`) : un ancêtre vivant qui tient déjà le verrou ⇒ `ancetre` (pas de seconde prise,
    /// comme avant) ; sinon LOCK_SH ⇒ `partage` ; un détenteur exclusif non ancêtre fait attendre jusqu'au délai (rc 6).
    fn acquerir_partage(&self, fichier: File, attente: Duration) -> Resultat<GardeVerrou> {
        let echeance = Instant::now() + attente;
        loop {
            if self.tenu_par_ancetre() {
                return Ok(self.garde(ModeVerrou::Ancetre, None));
            }
            match fichier.try_lock_shared() {
                Ok(()) => return Ok(self.garde(ModeVerrou::Partage, Some(fichier))),
                Err(TryLockError::WouldBlock) => {
                    let restant = echeance.saturating_duration_since(Instant::now());
                    if restant.is_zero() {
                        return Err(self.delai_depasse(attente));
                    }
                    std::thread::sleep(PAS_ATTENTE.min(restant));
                }
                Err(TryLockError::Error(e)) => {
                    return Err(Erreur::io(format!("flock -s {}", self.chemin.display()), e));
                }
            }
        }
    }

    /// Vrai si un processus quelconque tient le verrou (lecture de /proc/locks, pour doctor/etat).
    pub fn est_pris(&self) -> bool {
        !self.detenteurs().is_empty()
    }

    /// PID du détenteur (le premier), pour `etat --json`.
    pub fn proprietaire(&self) -> Option<u32> {
        self.detenteurs().first().copied()
    }
}
