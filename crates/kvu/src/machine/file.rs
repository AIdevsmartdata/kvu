//! File de jobs sur les deux machines (répertoire, renommages atomiques). Propriétaire : T1.
//!
//! Arborescence : `<racine>/{attente,en_cours,fini,echec,annule}/<id>.json` et `journaux/<id>.log`.
//! - Écriture d'un job : fichier temporaire, fsync, rename (ou hard_link sans écrasement à la création).
//! - TOUTE transition est un `rename` : deux workers qui visent le même job ne peuvent pas réussir
//!   tous les deux (le second voit le fichier source disparu).
//! - Le worker ne tue jamais rien. Il vérifie les ressources AVANT de lancer, sinon le job reste en
//!   attente et la raison est consignée au journal.

use super::detection::ram_disponible_go;
use super::politique::Politique;
use super::verrou::VerrouGpu;
use crate::commun::{Erreur, Machine, Resultat};
use std::ffi::{c_char, c_int, CString};
use std::io::Write;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Âge (s) du fichier d'alerte du moniteur en dessous duquel aucun job ne démarre.
pub const ALERTE_RECENTE_S: f64 = 600.0;
const GIB: f64 = (1u64 << 30) as f64;

/// État d'un job : le sous-répertoire de la file qui contient son fichier JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EtatJob {
    Attente,
    EnCours,
    Fini,
    Echec,
    Annule,
}

impl EtatJob {
    pub const TOUS: [EtatJob; 5] = [
        EtatJob::Attente,
        EtatJob::EnCours,
        EtatJob::Fini,
        EtatJob::Echec,
        EtatJob::Annule,
    ];

    /// Nom du sous-répertoire de la file.
    pub fn dossier(self) -> &'static str {
        match self {
            EtatJob::Attente => "attente",
            EtatJob::EnCours => "en_cours",
            EtatJob::Fini => "fini",
            EtatJob::Echec => "echec",
            EtatJob::Annule => "annule",
        }
    }
}

/// Un job : une commande kevinotron (ou toute commande) et ses besoins.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Job {
    /// AAAAMMJJ-HHMMSS-<machine>-<n> (ordre lexicographique = ordre d'arrivée).
    pub id: String,
    pub machine: Machine,
    pub commande: Vec<String>,
    pub repertoire: PathBuf,
    /// Besoin du GPU : le worker prend le verrou de la machine pour toute la durée du job.
    pub gpu: bool,
    /// Plafond mémoire (« 8G ») : systemd-run --scope -p MemoryMax=… ; sans lui, celui de la politique.
    pub mem_max: Option<String>,
    pub fils: usize,
    /// Plus grand = plus prioritaire ; à priorité égale, ordre d'arrivée.
    pub priorite: i32,
    /// Sortie estimée (Go) : la garde SSD refuse le lancement si libre − estimation < seuil.
    pub sortie_estimee_go: f64,
    pub etat: EtatJob,
    pub rc: Option<i32>,
    pub pid: Option<u32>,
    pub cree_unix: f64,
    pub debut_unix: Option<f64>,
    pub fin_unix: Option<f64>,
    pub journal: Option<PathBuf>,
    /// Worker qui a pris le job : pid et instant de démarrage (champ 22 de /proc/<pid>/stat, en tops depuis le
    /// boot ; protège contre la réutilisation des pid).
    #[serde(default)]
    pub worker_pid: Option<u32>,
    #[serde(default)]
    pub worker_debut: Option<u64>,
    /// Instant de démarrage du processus du job (`pid`).
    #[serde(default)]
    pub pid_debut: Option<u64>,
    /// /proc/sys/kernel/random/boot_id à la prise : un autre boot ⇒ plus rien ne tourne.
    #[serde(default)]
    pub boot_id: Option<String>,
    /// Job rangé en échec parce que son worker ET son processus ont disparu sans écrire de rc (SIGKILL, reboot).
    #[serde(default)]
    pub orphelin: bool,
}

impl Job {
    /// Job minimal : id vide (attribué à l'ajout), sans besoin particulier.
    pub fn nouveau(machine: Machine, commande: Vec<String>) -> Job {
        Job {
            id: String::new(),
            machine,
            commande,
            repertoire: PathBuf::new(),
            gpu: false,
            mem_max: None,
            fils: 0,
            priorite: 0,
            sortie_estimee_go: 0.0,
            etat: EtatJob::Attente,
            rc: None,
            pid: None,
            cree_unix: 0.0,
            debut_unix: None,
            fin_unix: None,
            journal: None,
            worker_pid: None,
            worker_debut: None,
            pid_debut: None,
            boot_id: None,
            orphelin: false,
        }
    }
}

/// Résultat détaillé de `prendre_suivant_detail` : le job pris et, pour chaque job laissé en
/// attente, `(id, raison)`.
#[derive(Debug, Default)]
pub struct Prise {
    pub job: Option<Job>,
    pub refus: Vec<(String, String)>,
}

// ------------------------------------------------------------------------------------------
// Temps, identifiants
// ------------------------------------------------------------------------------------------

/// Secondes Unix actuelles.
pub fn maintenant_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// (année, mois, jour) civils UTC depuis un nombre de jours depuis 1970 (algorithme de H. Hinnant).
fn civil_depuis_jours(jours: i64) -> (i64, u32, u32) {
    let z = jours + 719_468;
    let ere = z.div_euclid(146_097);
    let jde = z.rem_euclid(146_097);
    let ae = (jde - jde / 1460 + jde / 36_524 - jde / 146_096) / 365;
    let a = ae + ere * 400;
    let jda = jde - (365 * ae + ae / 4 - ae / 100);
    let mp = (5 * jda + 2) / 153;
    let jour = (jda - (153 * mp + 2) / 5 + 1) as u32;
    let mois = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (a + i64::from(mois <= 2), mois, jour)
}

fn decouper(unix: f64) -> (i64, u32, u32, u32, u32, u32) {
    let s = unix.floor() as i64;
    let (y, m, d) = civil_depuis_jours(s.div_euclid(86_400));
    let reste = s.rem_euclid(86_400);
    (
        y,
        m,
        d,
        (reste / 3600) as u32,
        ((reste % 3600) / 60) as u32,
        (reste % 60) as u32,
    )
}

/// « 2026-10-04T12:34:56Z » (UTC).
pub fn horodatage_iso(unix: f64) -> String {
    let (y, mo, d, h, mi, s) = decouper(unix);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// Début d'identifiant « AAAAMMJJ-HHMMSS » (UTC).
pub fn prefixe_id(unix: f64) -> String {
    let (y, mo, d, h, mi, s) = decouper(unix);
    format!("{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}")
}

// ------------------------------------------------------------------------------------------
// Processus vivants, boot (jobs orphelins)
// ------------------------------------------------------------------------------------------

/// Délai pendant lequel un job sans worker consigné (format d'avant le 04/10, ou fenêtre entre la prise et
/// l'écriture du worker) n'est jamais déclaré orphelin.
pub const GRACE_ORPHELIN_S: f64 = 60.0;

/// Instant de démarrage (champ 22 de /proc/<pid>/stat, tops depuis le boot) d'après le contenu du fichier ;
/// None pour un processus terminé mais pas encore récolté (état Z ou X : il ne fera plus rien).
pub fn debut_depuis_stat(stat: &str) -> Option<u64> {
    let apres = &stat[stat.rfind(')')? + 1..];
    let mut mots = apres.split_whitespace();
    // après « ) » : état (champ 3) … starttime (champ 22) ⇒ 20ᵉ mot.
    if matches!(mots.next()?, "Z" | "X") {
        return None;
    }
    mots.nth(18)?.parse().ok()
}

/// Instant de démarrage du processus `pid`, None s'il n'existe pas.
pub fn debut_processus(pid: u32) -> Option<u64> {
    debut_depuis_stat(&std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?)
}

/// Vrai si le processus `pid` existe ET (si `debut` est connu) a démarré à cet instant (pas un pid réutilisé).
pub fn processus_vivant(pid: u32, debut: Option<u64>) -> bool {
    match (debut_processus(pid), debut) {
        (None, _) => false,
        (Some(_), None) => true,
        (Some(d), Some(attendu)) => d == attendu,
    }
}

/// Identifiant du boot courant.
pub fn boot_id() -> Option<String> {
    std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Décision PURE : le job EN COURS est-il orphelin (plus aucun worker ni processus pour le terminer) ?
///
/// - autre boot ⇒ orphelin ;
/// - worker vivant ⇒ non ;
/// - sans worker consigné et fichier récent (< [`GRACE_ORPHELIN_S`]) ⇒ non (fenêtre de prise, ancien format) ;
/// - processus du job encore vivant ⇒ non (il finira ; son rc sera perdu, mais rien n'est tué ni relancé) ;
/// - sinon ⇒ orphelin.
pub fn est_orphelin(
    job: &Job,
    boot_courant: Option<&str>,
    vivant: &dyn Fn(u32, Option<u64>) -> bool,
    age_fichier_s: f64,
) -> bool {
    if let (Some(b), Some(c)) = (job.boot_id.as_deref(), boot_courant) {
        if b != c {
            return true;
        }
    }
    match job.worker_pid {
        Some(w) if vivant(w, job.worker_debut) => return false,
        None if age_fichier_s < GRACE_ORPHELIN_S => return false,
        _ => {}
    }
    !job.pid.is_some_and(|p| vivant(p, job.pid_debut))
}

// ------------------------------------------------------------------------------------------
// Espace disque (statvfs local, sans dépendance libc)
// ------------------------------------------------------------------------------------------

#[repr(C)]
struct StatVfs {
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
    _reserve: [i32; 6],
}

extern "C" {
    fn statvfs(chemin: *const c_char, tampon: *mut StatVfs) -> c_int;
}

/// Espace libre (Gio) du système de fichiers qui contient `chemin` (ou son premier ancêtre existant),
/// comme `df --output=avail`. None si illisible.
pub fn libre_go(chemin: &Path) -> Option<f64> {
    let mut p = if chemin.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        chemin.to_path_buf()
    };
    while !p.exists() {
        if !p.pop() || p.as_os_str().is_empty() {
            p = PathBuf::from(".");
            break;
        }
    }
    let c = CString::new(p.as_os_str().as_bytes()).ok()?;
    let mut sv = StatVfs {
        f_bsize: 0,
        f_frsize: 0,
        f_blocks: 0,
        f_bfree: 0,
        f_bavail: 0,
        f_files: 0,
        f_ffree: 0,
        f_favail: 0,
        f_fsid: 0,
        f_flag: 0,
        f_namemax: 0,
        _reserve: [0; 6],
    };
    // SAFETY : chaîne C valide, structure conforme à `struct statvfs` de glibc/musl x86-64 et aarch64.
    if unsafe { statvfs(c.as_ptr(), &mut sv) } != 0 {
        return None;
    }
    Some(sv.f_bavail as f64 * sv.f_frsize as f64 / GIB)
}

// ------------------------------------------------------------------------------------------
// Admissibilité (pure)
// ------------------------------------------------------------------------------------------

/// Raison pour laquelle `job` ne peut PAS démarrer maintenant, ou None. Étiquette entre crochets
/// (ram, alerte, ssd) = clé de déduplication du journal.
pub fn raison_refus(
    job: &Job,
    pol: &Politique,
    ram_dispo_go: f64,
    age_alerte_s: Option<f64>,
    libre_ssd_go: Option<f64>,
) -> Option<String> {
    if ram_dispo_go < pol.ram_min_go {
        return Some(format!(
            "[ram] RAM disponible {ram_dispo_go:.1} Go < {:.1} Go requis",
            pol.ram_min_go
        ));
    }
    if let Some(age) = age_alerte_s {
        if age < ALERTE_RECENTE_S {
            return Some(format!(
                "[alerte] moniteur de charge en alerte depuis {age:.0} s (< {ALERTE_RECENTE_S:.0} s)"
            ));
        }
    }
    match libre_ssd_go {
        None => Some("[ssd] espace libre illisible : refus par prudence".to_string()),
        Some(libre) if libre - job.sortie_estimee_go < pol.seuil_ssd_go => Some(format!(
            "[ssd] libre {libre:.1} Go − sortie estimée {:.1} Go < seuil {:.0} Go",
            job.sortie_estimee_go, pol.seuil_ssd_go
        )),
        Some(_) => None,
    }
}

/// Âge (s) du fichier d'alerte du moniteur, None s'il est absent.
fn age_alerte(pol: &Politique) -> Option<f64> {
    let m = std::fs::metadata(pol.alerte_moniteur.as_ref()?).ok()?;
    let modif = m.modified().ok()?;
    Some(SystemTime::now().duration_since(modif).map_or(0.0, |d| d.as_secs_f64()))
}

// ------------------------------------------------------------------------------------------
// Commande lancée (pure)
// ------------------------------------------------------------------------------------------

/// Commande réellement lancée, PURE (testable sans rien exécuter) :
/// [systemd-run --scope -p MemoryMax=… --quiet] nice -n N [taskset -c CPUS] [flock -w T VERROU] <commande…>.
///
/// MemoryMax est posé pour TOUT job dès que le job ou la politique en donne un (clé `mem_max` du profil) : la
/// taille de la SORTIE disque estimée ne dit rien de la RAM d'un job (vérificateur du 04/10 : avant, un job
/// sans `--mem` dont la sortie estimée tenait sous 4 Go tournait sans plafond, quelle que soit sa RAM).
pub fn construire_commande(job: &Job, pol: &Politique) -> Vec<String> {
    construire_commande_ex(job, pol, false)
}

/// Variante qui sait que le verrou GPU est DÉJÀ tenu par un ancêtre du worker (`queue run` lancé
/// sous un `flock` externe) : on n'enveloppe alors pas dans un second `flock`, qui bloquerait.
pub fn construire_commande_ex(job: &Job, pol: &Politique, verrou_deja_tenu: bool) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    if let Some(m) = job.mem_max.clone().or_else(|| pol.mem_max.clone()) {
        v.extend(["systemd-run", "--scope", "-p"].map(String::from));
        v.push(format!("MemoryMax={m}"));
        v.push("--quiet".to_string());
    }
    v.extend(["nice".to_string(), "-n".to_string(), pol.nice.to_string()]);
    if let Some(c) = &pol.cpus {
        v.extend(["taskset".to_string(), "-c".to_string(), c.clone()]);
    }
    if job.gpu && !verrou_deja_tenu {
        v.extend([
            "flock".to_string(),
            "-w".to_string(),
            pol.attente_verrou_s.to_string(),
            pol.verrou_gpu.display().to_string(),
        ]);
    }
    v.extend(job.commande.iter().cloned());
    v
}

/// Échappe un mot pour un shell POSIX (apostrophes).
pub fn echapper_shell(mot: &str) -> String {
    if !mot.is_empty()
        && mot
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_-./=:,@%+".contains(c))
    {
        return mot.to_string();
    }
    format!("'{}'", mot.replace('\'', "'\\''"))
}

/// Binaire kevinotron distant : KEVINOTRON_BIN_DISTANT, sinon ~/kevinotron_dev/bin/kevinotron.
pub fn binaire_distant() -> String {
    std::env::var("KEVINOTRON_BIN_DISTANT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "~/kevinotron_dev/bin/kevinotron".to_string())
}

/// Ligne de commande ssh pour `ajouter_distant` (pure, testable) :
/// `ssh -o BatchMode=yes <hôte> "<bin> queue add --machine … -- CMD…"`, arguments échappés.
pub fn commande_ssh(job: &Job, pol: &Politique, bin: &str) -> Resultat<Vec<String>> {
    let hote = pol.pair_cable.clone().ok_or_else(|| {
        Erreur::Config("aucun câble vers l'autre machine dans la politique (pair_cable absent)".to_string())
    })?;
    // « ~/… » doit rester développable par le shell distant.
    let bin_echappe = match bin.strip_prefix("~/") {
        Some(reste) => format!("~/{}", echapper_shell(reste)),
        None => echapper_shell(bin),
    };
    let mut r = vec![
        bin_echappe,
        "queue".to_string(),
        "add".to_string(),
        "--machine".to_string(),
        job.machine.nom().to_string(),
    ];
    if job.gpu {
        r.push("--gpu".to_string());
    }
    if let Some(m) = &job.mem_max {
        r.extend(["--mem".to_string(), echapper_shell(m)]);
    }
    if job.fils > 0 {
        r.extend(["--fils".to_string(), job.fils.to_string()]);
    }
    r.extend(["--priorite".to_string(), job.priorite.to_string()]);
    r.extend(["--sortie-go".to_string(), format!("{}", job.sortie_estimee_go)]);
    if !job.repertoire.as_os_str().is_empty() {
        r.extend([
            "--repertoire".to_string(),
            echapper_shell(&job.repertoire.display().to_string()),
        ]);
    }
    r.push("--".to_string());
    r.extend(job.commande.iter().map(|c| echapper_shell(c)));
    Ok(vec![
        "ssh".to_string(),
        "-o".to_string(),
        "BatchMode=yes".to_string(),
        hote,
        r.join(" "),
    ])
}

/// Ajoute un job dans la file de l'AUTRE machine par le câble (ssh vers l'hôte `cable` du profil, `kevinotron queue add …`).
pub fn ajouter_distant(job: &Job, pol: &Politique) -> Resultat<String> {
    ajouter_distant_avec(job, pol, "ssh", &binaire_distant())
}

/// Cœur de `ajouter_distant` : `programme` (« ssh » ; un faux script dans les tests) et `bin` distant.
pub fn ajouter_distant_avec(job: &Job, pol: &Politique, programme: &str, bin: &str) -> Resultat<String> {
    let argv = commande_ssh(job, pol, bin)?;
    let sortie = std::process::Command::new(programme)
        .args(&argv[1..])
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| Erreur::io(format!("lancement de {programme}"), e))?;
    if !sortie.status.success() {
        let err = String::from_utf8_lossy(&sortie.stderr);
        return Err(Erreur::io(
            format!("{programme} {} (rc {:?})", argv[3], sortie.status.code()),
            std::io::Error::other(err.trim().to_string()),
        ));
    }
    let texte = String::from_utf8_lossy(&sortie.stdout);
    texte
        .lines()
        .map(str::trim)
        .rfind(|l| !l.is_empty())
        .map(str::to_string)
        .ok_or_else(|| Erreur::Format("réponse vide de queue add distant (identifiant attendu)".to_string()))
}

// ------------------------------------------------------------------------------------------
// La file
// ------------------------------------------------------------------------------------------

/// File de jobs locale : <racine>/{attente,en_cours,fini,echec,annule}/<id>.json.
pub struct File {
    pub racine: PathBuf,
}

static COMPTEUR_TMP: AtomicU64 = AtomicU64::new(0);

fn io(ctx: impl Into<String>, e: std::io::Error) -> Erreur {
    Erreur::io(ctx, e)
}

/// fsync du dossier (rend le rename durable).
fn synchroniser_dossier(d: &Path) {
    if let Ok(f) = std::fs::File::open(d) {
        let _ = f.sync_all();
    }
}

/// Fichier temporaire frère de `cible` (même dossier ⇒ rename atomique), nom caché.
fn nom_temporaire(cible: &Path) -> PathBuf {
    let n = COMPTEUR_TMP.fetch_add(1, Ordering::Relaxed);
    let base = cible
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    cible.with_file_name(format!(".{base}.{}.{n}.tmp", std::process::id()))
}

/// Écrit `job` dans un temporaire fsyncé et rend son chemin.
fn ecrire_temporaire(cible: &Path, job: &Job) -> Resultat<PathBuf> {
    let tmp = nom_temporaire(cible);
    let json = serde_json::to_vec_pretty(job).map_err(|e| Erreur::Format(format!("sérialisation du job : {e}")))?;
    let mut f = std::fs::File::create(&tmp).map_err(|e| io(format!("création de {}", tmp.display()), e))?;
    f.write_all(&json)
        .and_then(|_| f.write_all(b"\n"))
        .and_then(|_| f.sync_all())
        .map_err(|e| io(format!("écriture de {}", tmp.display()), e))?;
    Ok(tmp)
}

/// Écriture atomique : temporaire, fsync, rename.
fn ecrire_atomique(cible: &Path, job: &Job) -> Resultat<()> {
    let tmp = ecrire_temporaire(cible, job)?;
    std::fs::rename(&tmp, cible).map_err(|e| io(format!("rename vers {}", cible.display()), e))?;
    if let Some(d) = cible.parent() {
        synchroniser_dossier(d);
    }
    Ok(())
}

fn lire_job(chemin: &Path) -> Resultat<Job> {
    let t = std::fs::read_to_string(chemin).map_err(|e| io(format!("lecture de {}", chemin.display()), e))?;
    serde_json::from_str(&t).map_err(|e| Erreur::Format(format!("job {} illisible : {e}", chemin.display())))
}

impl File {
    pub fn ouvrir(racine: &Path) -> Resultat<File> {
        for d in EtatJob::TOUS.iter().map(|e| e.dossier()).chain(["journaux"]) {
            let p = racine.join(d);
            std::fs::create_dir_all(&p).map_err(|e| io(format!("création de {}", p.display()), e))?;
        }
        Ok(File {
            racine: racine.to_path_buf(),
        })
    }

    fn chemin(&self, etat: EtatJob, id: &str) -> PathBuf {
        self.racine.join(etat.dossier()).join(format!("{id}.json"))
    }

    /// Chemin du journal d'un job.
    pub fn chemin_journal(&self, id: &str) -> PathBuf {
        self.racine.join("journaux").join(format!("{id}.log"))
    }

    /// Ajoute une ligne horodatée au journal du job.
    pub fn journaliser(&self, id: &str, message: &str) {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.chemin_journal(id))
        {
            let _ = writeln!(f, "[{}] {message}", horodatage_iso(maintenant_unix()));
        }
    }

    fn existe_quelque_part(&self, id: &str) -> bool {
        EtatJob::TOUS.iter().any(|e| self.chemin(*e, id).exists())
    }

    /// Ajoute un job (écriture atomique dans attente/) ; rend son id. Un id vide est attribué :
    /// AAAAMMJJ-HHMMSS-<machine>-<n>, n croissant dans la seconde. La publication utilise un
    /// `hard_link` (jamais d'écrasement) : deux ajouts simultanés obtiennent deux ids différents.
    pub fn ajouter(&self, mut job: Job) -> Resultat<String> {
        let maintenant = maintenant_unix();
        job.etat = EtatJob::Attente;
        if job.cree_unix == 0.0 {
            job.cree_unix = maintenant;
        }
        if job.commande.is_empty() {
            return Err(Erreur::Config("job sans commande".to_string()));
        }
        let id_impose = (!job.id.is_empty()).then(|| job.id.clone());
        if let Some(id) = &id_impose {
            if id.contains('/') || id.starts_with('.') {
                return Err(Erreur::Config(format!("identifiant de job invalide « {id} »")));
            }
        }
        let prefixe = format!("{}-{}", prefixe_id(job.cree_unix), job.machine.nom());
        for n in 1..100_000u32 {
            let id = id_impose.clone().unwrap_or_else(|| format!("{prefixe}-{n:04}"));
            if self.existe_quelque_part(&id) {
                if id_impose.is_some() {
                    return Err(Erreur::Config(format!("le job {id} existe déjà")));
                }
                continue;
            }
            job.id = id.clone();
            let cible = self.chemin(EtatJob::Attente, &id);
            let tmp = ecrire_temporaire(&cible, &job)?;
            let publie = std::fs::hard_link(&tmp, &cible);
            let _ = std::fs::remove_file(&tmp);
            match publie {
                Ok(()) => {
                    synchroniser_dossier(&self.racine.join("attente"));
                    return Ok(id);
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && id_impose.is_none() => continue,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    return Err(Erreur::Config(format!("le job {id} existe déjà")))
                }
                Err(e) => return Err(io(format!("publication de {}", cible.display()), e)),
            }
        }
        Err(Erreur::Ressources("trop de jobs dans la même seconde".to_string()))
    }

    /// Jobs d'un état, triés par id. Les fichiers illisibles sont ignorés (voir `invalides`).
    pub fn lister_etat(&self, etat: EtatJob) -> Resultat<Vec<Job>> {
        let dossier = self.racine.join(etat.dossier());
        let mut jobs = Vec::new();
        for chemin in noms_json(&dossier)? {
            if let Ok(mut j) = lire_job(&chemin) {
                j.etat = etat;
                jobs.push(j);
            }
        }
        jobs.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(jobs)
    }

    /// Tous les jobs (tous états), triés par id.
    pub fn lister(&self) -> Resultat<Vec<Job>> {
        let mut tous = Vec::new();
        for e in EtatJob::TOUS {
            tous.extend(self.lister_etat(e)?);
        }
        tous.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(tous)
    }

    /// Fichiers .json de la file qui ne se lisent pas comme un job (à examiner à la main).
    pub fn invalides(&self) -> Vec<PathBuf> {
        let mut v = Vec::new();
        for e in EtatJob::TOUS {
            if let Ok(noms) = noms_json(&self.racine.join(e.dossier())) {
                v.extend(noms.into_iter().filter(|p| lire_job(p).is_err()));
            }
        }
        v
    }

    /// Nombre de jobs par état (sans lire les JSON : rapide, pour `etat --json`).
    pub fn compter(&self, etat: EtatJob) -> usize {
        noms_json(&self.racine.join(etat.dossier())).map_or(0, |v| v.len())
    }

    /// Jobs de en_cours/ dont le worker et le processus ont disparu (SIGKILL, reboot). Lecture seule.
    pub fn orphelins(&self) -> Vec<Job> {
        let boot = boot_id();
        let Ok(chemins) = noms_json(&self.racine.join(EtatJob::EnCours.dossier())) else {
            return Vec::new();
        };
        let mut v: Vec<Job> = chemins
            .iter()
            .filter_map(|c| {
                let job = lire_job(c).ok()?;
                let age = std::fs::metadata(c)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| SystemTime::now().duration_since(t).ok())
                    .map_or(0.0, |d| d.as_secs_f64());
                est_orphelin(&job, boot.as_deref(), &processus_vivant, age).then_some(job)
            })
            .collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }

    /// Range les jobs orphelins en échec (`orphelin: true`, rc absent) ; rend leurs ids. Rien n'est relancé :
    /// un job interrompu a pu écrire une partie de ses sorties (le relancer est une décision humaine).
    pub fn recuperer_orphelins(&self) -> Resultat<Vec<String>> {
        let mut faits = Vec::new();
        for mut job in self.orphelins() {
            let en_cours = self.chemin(EtatJob::EnCours, &job.id);
            let echec = self.chemin(EtatJob::Echec, &job.id);
            job.etat = EtatJob::Echec;
            job.orphelin = true;
            job.fin_unix = Some(maintenant_unix());
            // Écrit à côté, puis UN rename vers echec/ : si un autre worker l'a déjà rangé, on s'arrête là.
            let tmp = ecrire_temporaire(&en_cours, &job)?;
            match std::fs::rename(&en_cours, &echec) {
                Ok(()) => {
                    let r = std::fs::rename(&tmp, &echec);
                    let _ = std::fs::remove_file(&tmp);
                    r.map_err(|e| io(format!("rangement de l'orphelin {}", job.id), e))?;
                    synchroniser_dossier(&self.racine.join(EtatJob::Echec.dossier()));
                    self.journaliser(
                        &job.id,
                        &format!(
                            "orphelin : worker {} et processus {} disparus sans rc ; rangé en échec (non relancé)",
                            job.worker_pid.map_or("?".to_string(), |p| p.to_string()),
                            job.pid.map_or("?".to_string(), |p| p.to_string())
                        ),
                    );
                    faits.push(job.id.clone());
                }
                Err(e) => {
                    let _ = std::fs::remove_file(&tmp);
                    if e.kind() != std::io::ErrorKind::NotFound {
                        return Err(io(format!("rangement de l'orphelin {}", job.id), e));
                    }
                }
            }
        }
        Ok(faits)
    }

    /// Annule un job EN ATTENTE (rename attente/ → annule/). Un job en cours n'est jamais tué.
    pub fn annuler(&self, id: &str) -> Resultat<()> {
        match std::fs::rename(self.chemin(EtatJob::Attente, id), self.chemin(EtatJob::Annule, id)) {
            Ok(()) => {
                self.journaliser(id, "annulé");
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if self.chemin(EtatJob::EnCours, id).exists() {
                    Err(Erreur::Config(format!(
                        "le job {id} est en cours : rien n'est tué, attendre sa fin"
                    )))
                } else if self.existe_quelque_part(id) {
                    Err(Erreur::Config(format!("le job {id} n'est plus en attente")))
                } else {
                    Err(Erreur::Config(format!("job {id} inconnu")))
                }
            }
            Err(e) => Err(io(format!("annulation de {id}"), e)),
        }
    }

    /// Prend le prochain job admissible (priorité, ressources, garde SSD, moniteur) par
    /// renommage atomique attente/ → en_cours/ ; None si rien d'admissible.
    pub fn prendre_suivant(&self, pol: &Politique) -> Resultat<Option<Job>> {
        Ok(self.prendre_suivant_detail(pol)?.job)
    }

    /// Comme `prendre_suivant`, avec la raison de chaque job laissé en attente. Ordre : priorité
    /// décroissante, puis id croissant. Un job refusé n'empêche pas un suivant plus petit de passer
    /// (chaque refus est journalisé, une fois par raison).
    pub fn prendre_suivant_detail(&self, pol: &Politique) -> Resultat<Prise> {
        // Un worker tué (SIGKILL, reboot) laissait son job dans en_cours/ pour toujours : tout worker range
        // d'abord les orphelins en échec.
        self.recuperer_orphelins()?;
        let mut candidats = self.lister_etat(EtatJob::Attente)?;
        candidats.sort_by(|a, b| b.priorite.cmp(&a.priorite).then_with(|| a.id.cmp(&b.id)));
        let ram = ram_disponible_go();
        let alerte = age_alerte(pol);
        let mut prise = Prise::default();
        for mut job in candidats {
            let dossier = if job.repertoire.as_os_str().is_empty() {
                self.racine.clone()
            } else {
                job.repertoire.clone()
            };
            if let Some(raison) = raison_refus(&job, pol, ram, alerte, libre_go(&dossier)) {
                self.journaliser_attente(&job.id, &raison);
                prise.refus.push((job.id.clone(), raison));
                continue;
            }
            // La transition : un seul rename réussit, même avec plusieurs workers.
            match std::fs::rename(
                self.chemin(EtatJob::Attente, &job.id),
                self.chemin(EtatJob::EnCours, &job.id),
            ) {
                Ok(()) => {
                    job.etat = EtatJob::EnCours;
                    job.debut_unix = Some(maintenant_unix());
                    job.journal = Some(self.chemin_journal(&job.id));
                    job.worker_pid = Some(std::process::id());
                    job.worker_debut = debut_processus(std::process::id());
                    job.boot_id = boot_id();
                    // Mise à jour du contenu dans en_cours/ (nous seuls possédons ce fichier).
                    ecrire_atomique(&self.chemin(EtatJob::EnCours, &job.id), &job)?;
                    self.journaliser(&job.id, "pris par un worker");
                    prise.job = Some(job);
                    return Ok(prise);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue, // pris par un autre worker
                Err(e) => return Err(io(format!("prise du job {}", job.id), e)),
            }
        }
        Ok(prise)
    }

    /// Journalise « en attente : raison » sauf si la dernière ligne a déjà la même étiquette.
    fn journaliser_attente(&self, id: &str, raison: &str) {
        let etiquette = raison.split(']').next().unwrap_or("");
        let derniere = std::fs::read_to_string(self.chemin_journal(id))
            .ok()
            .and_then(|t| t.lines().last().map(str::to_string));
        if derniere.is_some_and(|l| l.contains("en attente : ") && l.contains(etiquette)) {
            return;
        }
        self.journaliser(id, &format!("en attente : {raison}"));
    }

    /// Exécute un job pris (enveloppe de `construire_commande`), écrit rc et journal, le range
    /// dans fini/ (rc 0) ou echec/. Rend le code de retour de la commande. Ne tue jamais rien.
    pub fn executer(&self, mut job: Job, pol: &Politique) -> Resultat<i32> {
        let en_cours = self.chemin(EtatJob::EnCours, &job.id);
        if !en_cours.exists() {
            return Err(Erreur::Config(format!(
                "le job {} n'a pas été pris (absent de en_cours/)",
                job.id
            )));
        }
        // Verrou déjà tenu par un ancêtre du worker : pas de second flock (interblocage).
        let deja = job.gpu && VerrouGpu::nouveau(&pol.verrou_gpu).tenu_par_ancetre();
        let argv = construire_commande_ex(&job, pol, deja);
        let journal = self.chemin_journal(&job.id);
        let mut j = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&journal)
            .map_err(|e| io(format!("journal {}", journal.display()), e))?;
        let _ = writeln!(
            j,
            "[{}] lancement : {}",
            horodatage_iso(maintenant_unix()),
            argv.join(" ")
        );
        let sortie = j.try_clone().map_err(|e| io("duplication du journal", e))?;
        let erreur = j.try_clone().map_err(|e| io("duplication du journal", e))?;
        let mut cmd = std::process::Command::new(&argv[0]);
        cmd.args(&argv[1..])
            .stdin(std::process::Stdio::null())
            .stdout(sortie)
            .stderr(erreur);
        if job.repertoire.is_dir() {
            cmd.current_dir(&job.repertoire);
        }
        if job.fils > 0 {
            cmd.env("RAYON_NUM_THREADS", job.fils.min(pol.fils_max.max(1)).to_string());
        }
        let rc = match cmd.spawn() {
            Ok(mut enfant) => {
                job.pid = Some(enfant.id());
                job.pid_debut = debut_processus(enfant.id());
                ecrire_atomique(&en_cours, &job)?;
                match enfant.wait() {
                    Ok(st) => st.code().unwrap_or_else(|| 128 + st.signal().unwrap_or(0)),
                    Err(e) => {
                        let _ = writeln!(j, "attente du processus impossible : {e}");
                        1
                    }
                }
            }
            Err(e) => {
                let _ = writeln!(j, "lancement impossible ({}) : {e}", argv[0]);
                127
            }
        };
        job.rc = Some(rc);
        job.fin_unix = Some(maintenant_unix());
        job.etat = if rc == 0 { EtatJob::Fini } else { EtatJob::Echec };
        let _ = writeln!(j, "[{}] fin : rc = {rc}", horodatage_iso(maintenant_unix()));
        ecrire_atomique(&en_cours, &job)?;
        std::fs::rename(&en_cours, self.chemin(job.etat, &job.id))
            .map_err(|e| io(format!("rangement du job {}", job.id), e))?;
        Ok(rc)
    }

    /// Boucle de worker : prend et exécute les jobs admissibles. `vider` : s'arrête quand plus rien
    /// n'est admissible ; sinon attend `pause` et recommence. `max` borne le nombre de jobs.
    pub fn boucle(
        &self,
        pol: &Politique,
        vider: bool,
        pause: Duration,
        max: Option<usize>,
    ) -> Resultat<Vec<(String, i32)>> {
        let mut faits = Vec::new();
        loop {
            if max.is_some_and(|m| faits.len() >= m) {
                return Ok(faits);
            }
            match self.prendre_suivant(pol)? {
                Some(job) => {
                    let id = job.id.clone();
                    let rc = self.executer(job, pol)?;
                    faits.push((id, rc));
                }
                None if vider => return Ok(faits),
                None => std::thread::sleep(pause),
            }
        }
    }
}

/// Chemins des *.json (hors fichiers cachés/temporaires) d'un dossier.
fn noms_json(dossier: &Path) -> Resultat<Vec<PathBuf>> {
    let mut v = Vec::new();
    for e in std::fs::read_dir(dossier).map_err(|e| io(format!("lecture de {}", dossier.display()), e))? {
        let e = e.map_err(|e| io(format!("lecture de {}", dossier.display()), e))?;
        let n = e.file_name().to_string_lossy().into_owned();
        if !n.starts_with('.') && n.ends_with(".json") {
            v.push(e.path());
        }
    }
    Ok(v)
}
