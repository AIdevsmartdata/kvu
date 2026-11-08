//! Store d'ensembles : du, gc (dry-run par défaut), vérifier, lister, exporter, éclaircir.
//! Propriétaire : T8.
//!
//! Un **ensemble** est un enfant direct de la racine du store (un répertoire de run, ou un répertoire qui en
//! contient). Une **unité** est un répertoire qui contient un `manifest.json` (contrat K8) : c'est l'unité de
//! décision de `gc` (COMPLETE ou non, pid du processus, âge).
//!
//! Garanties de `gc` (testées, SSD-2) :
//! - simulation par défaut : RIEN n'est supprimé ni modifié sans `appliquer` ;
//! - n'agit que sous la racine (chemins canoniques re-vérifiés avant chaque action) ;
//! - les liens symboliques ne sont JAMAIS suivis (ni parcourus, ni supprimés) ;
//! - un répertoire qui contient un fichier `.garder` est intouchable, sous-arbre compris ;
//! - le `manifest.json` et le `COMPLETE` d'un run ne sont jamais réécrits : chaque modification d'un fichier
//!   consigné s'inscrit dans le journal `ECLAIRCISSEMENT.jsonl` de l'unité, et `verifier` en tient compte ;
//! - c16 → q64 n'est publié qu'après vérification de l'aller-retour exact (voir `npy::convertir_champ`).

use super::compression;
use super::garde::GardeSsd;
use super::npy::{self, FormatChamp};
use crate::commun::{Erreur, Resultat};
use crate::heritage::manifest::{
    creer_repertoire_neuf, hex, maintenant_unix, nom_hote, sha256_fichier, Sha256, NOM_COMPLETE, NOM_MANIFESTE,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

/// Fichier qui rend un répertoire (et son sous-arbre) intouchable par `gc`.
pub const NOM_GARDER: &str = ".garder";
/// Journal des modifications faites par `gc` dans une unité (une ligne JSON par action).
pub const NOM_JOURNAL: &str = "ECLAIRCISSEMENT.jsonl";
/// Âge par défaut au-delà duquel un run orphelin (sans COMPLETE, pid mort) peut être supprimé : 24 h.
pub const AGE_ORPHELIN_DEFAUT_S: f64 = 24.0 * 3600.0;

/// Politique d'éclaircissement d'un ensemble (jamais appliquée sans `--appliquer`).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PolitiqueEclaircissement {
    /// Garder un dump sur `garder_chaque` (indices multiples), plus le premier et le dernier.
    /// 0 ou 1 : aucun éclaircissement.
    pub garder_chaque: u64,
    /// Convertir les c16 en q64 (moitié de la place, conversion exacte et vérifiée).
    pub convertir_q64: bool,
    pub compresser_zstd: bool,
    /// Supprimer les points de reprise d'un run COMPLETE.
    pub purger_reprises: bool,
}

impl PolitiqueEclaircissement {
    /// Politique prudente : seules les reprises d'un run COMPLETE sont purgées.
    pub fn prudente() -> Self {
        PolitiqueEclaircissement {
            garder_chaque: 1,
            convertir_q64: false,
            compresser_zstd: false,
            purger_reprises: true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LigneDu {
    pub chemin: PathBuf,
    pub octets: u64,
    pub fichiers: u64,
    pub complete: bool,
    /// Âge du dernier fichier (s).
    pub age_s: f64,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RapportGc {
    pub applique: bool,
    /// (chemin, octets, raison) de chaque action prévue ou faite.
    pub actions: Vec<(PathBuf, u64, String)>,
    /// Octets libérés (faits si `applique`, estimés sinon).
    pub octets_liberes: u64,
    /// (chemin, raison) de ce que `gc` a laissé en l'état et qu'il a fallu décider : `.garder`, run vivant…
    pub ignores: Vec<(PathBuf, String)>,
}

/// Une ligne de `store lister`.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LigneListe {
    pub id: String,
    pub chemin: PathBuf,
    pub complete: bool,
    pub garder: bool,
    pub octets: u64,
    pub fichiers: u64,
    pub age_s: f64,
    pub machine: Option<String>,
    pub moteur: Option<String>,
    pub rc: Option<i64>,
    pub debut_unix: Option<f64>,
}

/// Un écart constaté par `verifier`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Ecart {
    /// Fichier concerné (chemin complet).
    pub chemin: PathBuf,
    /// « manquant », « sha256 », « complete » ou « manifeste ».
    pub nature: String,
    pub detail: String,
}

/// Résultat d'une vérification de sha256 d'un ensemble.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RapportVerif {
    pub unites: u64,
    pub fichiers_verifies: u64,
    pub octets_verifies: u64,
    pub ecarts: Vec<Ecart>,
    /// Unités sans COMPLETE (run en cours, tué, ou en échec).
    pub sans_complete: Vec<PathBuf>,
}

impl RapportVerif {
    pub fn ok(&self) -> bool {
        self.ecarts.is_empty()
    }

    /// Erreur de validation (code de sortie 8) au moindre écart ou si une unité n'a pas de COMPLETE.
    pub fn en_resultat(&self, id: &str) -> Resultat<()> {
        if let Some(e) = self.ecarts.first() {
            return Err(Erreur::Validation(format!(
                "ensemble « {id} » : {} écart(s), par exemple {} ({}) : {}",
                self.ecarts.len(),
                e.chemin.display(),
                e.nature,
                e.detail
            )));
        }
        if let Some(s) = self.sans_complete.first() {
            return Err(Erreur::Validation(format!(
                "ensemble « {id} » : {} n'a pas de COMPLETE",
                s.display()
            )));
        }
        Ok(())
    }
}

/// Résultat d'un export.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RapportExport {
    pub destination: PathBuf,
    pub fichiers: u64,
    pub convertis: u64,
    pub octets: u64,
}

pub struct Store {
    pub racine: PathBuf,
}

// ------------------------------------------------------------------------------------------
// Parcours de répertoires (sans jamais suivre un lien symbolique)
// ------------------------------------------------------------------------------------------

struct Fic {
    chemin: PathBuf,
    octets: u64,
    mtime: f64,
}

fn io_ctx<'a>(ctx: &'a str, p: &'a Path) -> impl Fn(std::io::Error) -> Erreur + 'a {
    move |e| Erreur::io(format!("{ctx} {}", p.display()), e)
}

fn mtime_s(m: &fs::Metadata) -> f64 {
    m.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn est_lien(p: &Path) -> bool {
    fs::symlink_metadata(p)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
}

fn est_fichier_regulier(p: &Path) -> bool {
    fs::symlink_metadata(p)
        .map(|m| m.file_type().is_file())
        .unwrap_or(false)
}

fn est_repertoire_reel(p: &Path) -> bool {
    fs::symlink_metadata(p).map(|m| m.file_type().is_dir()).unwrap_or(false)
}

/// Entrées d'un répertoire, triées par nom ; un répertoire disparu pendant le parcours est une liste vide.
fn entrees(dir: &Path) -> Resultat<Vec<(PathBuf, fs::FileType)>> {
    let rd = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(io_ctx("lecture du répertoire", dir)(e)),
    };
    let mut v = Vec::new();
    for e in rd {
        let e = e.map_err(io_ctx("lecture du répertoire", dir))?;
        if let Ok(ft) = e.file_type() {
            v.push((e.path(), ft));
        }
    }
    v.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(v)
}

/// Tous les fichiers réguliers sous `dir` (récursif, liens symboliques ignorés).
fn parcourir(dir: &Path, sortie: &mut Vec<Fic>) -> Resultat<()> {
    for (p, ft) in entrees(dir)? {
        if ft.is_dir() {
            parcourir(&p, sortie)?;
        } else if ft.is_file() {
            if let Ok(m) = fs::symlink_metadata(&p) {
                sortie.push(Fic {
                    chemin: p,
                    octets: m.len(),
                    mtime: mtime_s(&m),
                });
            }
        }
    }
    Ok(())
}

/// Fichiers réguliers « propres » à une unité : sous-arbres des unités imbriquées et des répertoires
/// protégés par `.garder` exclus.
fn fichiers_unite(dir: &Path, sortie: &mut Vec<Fic>) -> Resultat<()> {
    for (p, ft) in entrees(dir)? {
        if ft.is_dir() {
            if est_fichier_regulier(&p.join(NOM_GARDER)) || est_fichier_regulier(&p.join(NOM_MANIFESTE)) {
                continue;
            }
            fichiers_unite(&p, sortie)?;
        } else if ft.is_file() {
            if let Ok(m) = fs::symlink_metadata(&p) {
                sortie.push(Fic {
                    chemin: p,
                    octets: m.len(),
                    mtime: mtime_s(&m),
                });
            }
        }
    }
    Ok(())
}

fn ligne_du(chemin: &Path) -> Resultat<LigneDu> {
    let meta = fs::metadata(chemin).map_err(io_ctx("stat", chemin))?;
    let mut fics = Vec::new();
    let (dir, complete) = if meta.is_dir() {
        parcourir(chemin, &mut fics)?;
        (true, est_fichier_regulier(&chemin.join(NOM_COMPLETE)))
    } else {
        fics.push(Fic {
            chemin: chemin.to_path_buf(),
            octets: meta.len(),
            mtime: mtime_s(&meta),
        });
        (false, false)
    };
    let recent = fics
        .iter()
        .map(|f| f.mtime)
        .fold(if dir { mtime_s(&meta) } else { 0.0 }, f64::max);
    Ok(LigneDu {
        chemin: chemin.to_path_buf(),
        octets: fics.iter().map(|f| f.octets).sum(),
        fichiers: fics.len() as u64,
        complete,
        age_s: (maintenant_unix() - recent).max(0.0),
    })
}

// ------------------------------------------------------------------------------------------
// Unités (runs) et répertoires protégés
// ------------------------------------------------------------------------------------------

struct Unite {
    dir: PathBuf,
    complete: bool,
    pid: Option<u64>,
    hote: Option<String>,
}

fn lire_manifeste(dir: &Path) -> Option<Value> {
    serde_json::from_str(&fs::read_to_string(dir.join(NOM_MANIFESTE)).ok()?).ok()
}

/// Cherche les unités et les répertoires protégés sous `dir` (le répertoire `dir` lui-même n'est jamais une unité).
fn explorer(dir: &Path, unites: &mut Vec<Unite>, protegees: &mut Vec<PathBuf>) -> Resultat<()> {
    for (p, ft) in entrees(dir)? {
        if !ft.is_dir() {
            continue;
        }
        if est_fichier_regulier(&p.join(NOM_GARDER)) {
            protegees.push(p);
            continue;
        }
        if est_fichier_regulier(&p.join(NOM_MANIFESTE)) {
            let m = lire_manifeste(&p);
            unites.push(Unite {
                complete: est_fichier_regulier(&p.join(NOM_COMPLETE)),
                pid: m.as_ref().and_then(|m| m.get("pid")).and_then(Value::as_u64),
                hote: m
                    .as_ref()
                    .and_then(|m| m.get("hote"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
                dir: p.clone(),
            });
        }
        explorer(&p, unites, protegees)?;
    }
    Ok(())
}

/// Vrai si le processus du run est mort : même hôte, pid renseigné, /proc/<pid> absent.
fn pid_mort(u: &Unite) -> bool {
    match (u.pid, u.hote.as_deref()) {
        (Some(pid), Some(h)) if h == nom_hote() => !Path::new(&format!("/proc/{pid}")).exists(),
        _ => false,
    }
}

// ------------------------------------------------------------------------------------------
// Journal d'éclaircissement
// ------------------------------------------------------------------------------------------

/// Lecture du journal d'une unité : par nom relatif, fichier supprimé ? et sha256 des versions converties.
#[derive(Default)]
struct Journal {
    supprimes: HashSet<String>,
    apres: HashMap<String, Vec<String>>,
    /// Répertoire (relatif à l'unité, « . » pour sa racine) -> plus grand k déjà appliqué par l'éclaircissement.
    eclaircis: HashMap<String, u64>,
}

fn lire_journal(dir: &Path) -> Journal {
    let mut j = Journal::default();
    let Ok(texte) = fs::read_to_string(dir.join(NOM_JOURNAL)) else {
        return j;
    };
    for l in texte.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(v) = serde_json::from_str::<Value>(l) else {
            continue;
        };
        let Some(nom) = v.get("fichier").and_then(Value::as_str) else {
            continue;
        };
        match v.get("action").and_then(Value::as_str) {
            Some("supprime") => {
                j.supprimes.insert(nom.to_string());
                if let (Some(k), Some(rep)) = (
                    v.get("eclaircissement_k").and_then(Value::as_u64),
                    v.get("repertoire").and_then(Value::as_str),
                ) {
                    let e = j.eclaircis.entry(rep.to_string()).or_insert(0);
                    *e = (*e).max(k);
                }
            }
            Some("convertit") => {
                if let Some(s) = v.get("apres").and_then(|a| a.get("sha256")).and_then(Value::as_str) {
                    j.apres.entry(nom.to_string()).or_default().push(s.to_string());
                }
            }
            _ => {}
        }
    }
    j
}

fn ajouter_au_journal(dir: &Path, entree: &Value) -> Resultat<()> {
    let chemin = dir.join(NOM_JOURNAL);
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&chemin)
        .map_err(io_ctx("ouverture du journal", &chemin))?;
    writeln!(f, "{entree}").map_err(io_ctx("écriture du journal", &chemin))?;
    f.sync_all().map_err(io_ctx("fsync du journal", &chemin))
}

/// Répertoire relatif à l'unité, « . » pour la racine de l'unité.
fn rel_repertoire(unite: &Path, dir: &Path) -> String {
    let r = relatif(unite, dir);
    if r.is_empty() {
        ".".to_string()
    } else {
        r
    }
}

fn relatif(dir: &Path, p: &Path) -> String {
    p.strip_prefix(dir).unwrap_or(p).to_string_lossy().into_owned()
}

// ------------------------------------------------------------------------------------------
// Dumps
// ------------------------------------------------------------------------------------------

/// Indice N d'un fichier `cfg_NNNNNN.npy` ou `cfg_NNNNNN.npy.zst`.
fn indice_dump(nom: &str) -> Option<u64> {
    let r = nom.strip_prefix("cfg_")?;
    let r = r.strip_suffix(".npy.zst").or_else(|| r.strip_suffix(".npy"))?;
    if r.is_empty() || !r.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    r.parse().ok()
}

/// Flux de lecture hachant ce qui passe : sha256 du contenu décompressé d'un `.zst`.
fn sha256_contenu(chemin: &Path) -> Resultat<String> {
    let mut h = Sha256::new();
    let mut tampon = vec![0u8; 1 << 20];
    if compression::est_zst(chemin) {
        let mut flux = compression::ouvrir_flux(chemin)?;
        loop {
            let n = flux.read(&mut tampon).map_err(io_ctx("lecture de zstd -dc", chemin))?;
            if n == 0 {
                break;
            }
            h.update(&tampon[..n]);
        }
        flux.terminer()?;
        Ok(hex(&h.finalize()))
    } else {
        sha256_fichier(chemin).map_err(io_ctx("sha256", chemin))
    }
}

// ------------------------------------------------------------------------------------------
// Vérification des sha256 d'un répertoire (utilisée par `Store::verifier` et par `fetch`)
// ------------------------------------------------------------------------------------------

/// Vérifie les sha256 de toutes les unités sous `dir` (ou de `dir` lui-même s'il en est une) : lien
/// `COMPLETE` ↔ `manifest.json`, puis chaque fichier consigné dans `fichiers`, en tenant compte du journal
/// d'éclaircissement (fichier supprimé, converti, compressé). Les unités protégées par `.garder` sont vérifiées
/// aussi (lecture seule). Ne modifie rien.
pub fn verifier_repertoire(dir: &Path) -> Resultat<RapportVerif> {
    if !est_repertoire_reel(dir) {
        return Err(Erreur::Config(format!("{} n'est pas un répertoire", dir.display())));
    }
    let mut unites = Vec::new();
    if est_fichier_regulier(&dir.join(NOM_MANIFESTE)) {
        let m = lire_manifeste(dir);
        unites.push(Unite {
            dir: dir.to_path_buf(),
            complete: est_fichier_regulier(&dir.join(NOM_COMPLETE)),
            pid: m.as_ref().and_then(|m| m.get("pid")).and_then(Value::as_u64),
            hote: None,
        });
    }
    explorer_tout(dir, &mut unites)?;
    if unites.is_empty() {
        return Err(Erreur::Format(format!(
            "aucun {NOM_MANIFESTE} sous {} : rien à vérifier",
            dir.display()
        )));
    }
    let mut r = RapportVerif::default();
    for u in &unites {
        r.unites += 1;
        verifier_unite(u, &mut r)?;
    }
    Ok(r)
}

/// Comme `explorer`, mais traverse aussi les répertoires `.garder` (la vérification est en lecture seule).
fn explorer_tout(dir: &Path, unites: &mut Vec<Unite>) -> Resultat<()> {
    for (p, ft) in entrees(dir)? {
        if !ft.is_dir() {
            continue;
        }
        if est_fichier_regulier(&p.join(NOM_MANIFESTE)) {
            unites.push(Unite {
                complete: est_fichier_regulier(&p.join(NOM_COMPLETE)),
                pid: None,
                hote: None,
                dir: p.clone(),
            });
        }
        explorer_tout(&p, unites)?;
    }
    Ok(())
}

fn verifier_unite(u: &Unite, r: &mut RapportVerif) -> Resultat<()> {
    let manifeste = u.dir.join(NOM_MANIFESTE);
    let texte = fs::read_to_string(&manifeste).map_err(io_ctx("lecture", &manifeste))?;
    let Ok(m) = serde_json::from_str::<Value>(&texte) else {
        r.ecarts.push(Ecart {
            chemin: manifeste,
            nature: "manifeste".into(),
            detail: "JSON illisible".into(),
        });
        return Ok(());
    };
    if u.complete {
        let complete = u.dir.join(NOM_COMPLETE);
        let c = fs::read_to_string(&complete).map_err(io_ctx("lecture", &complete))?;
        let attendu = c
            .lines()
            .find_map(|l| l.strip_prefix("manifest_sha256="))
            .map(str::trim)
            .unwrap_or("");
        let reel = crate::heritage::manifest::sha256_hex(texte.as_bytes());
        if attendu != reel {
            r.ecarts.push(Ecart {
                chemin: manifeste.clone(),
                nature: "complete".into(),
                detail: format!("COMPLETE annonce {attendu}, manifest.json vaut {reel}"),
            });
        }
    } else {
        r.sans_complete.push(u.dir.clone());
    }
    let journal = lire_journal(&u.dir);
    let Some(liste) = m.get("fichiers").and_then(Value::as_array) else {
        return Ok(());
    };
    for e in liste {
        let (Some(nom), Some(sha)) = (
            e.get("nom").and_then(Value::as_str),
            e.get("sha256").and_then(Value::as_str),
        ) else {
            r.ecarts.push(Ecart {
                chemin: manifeste.clone(),
                nature: "manifeste".into(),
                detail: format!("entrée de fichier mal formée : {e}"),
            });
            continue;
        };
        let chemin = if Path::new(nom).is_absolute() {
            PathBuf::from(nom)
        } else {
            u.dir.join(nom)
        };
        let mut acceptes: Vec<&str> = vec![sha];
        if let Some(a) = journal.apres.get(nom) {
            acceptes.extend(a.iter().map(String::as_str));
        }
        let mut zst = chemin.as_os_str().to_os_string();
        zst.push(".zst");
        let zst = PathBuf::from(zst);
        let present = if est_fichier_regulier(&chemin) {
            Some(chemin.clone())
        } else if est_fichier_regulier(&zst) {
            Some(zst)
        } else {
            None
        };
        match present {
            Some(p) => {
                let obtenu = sha256_contenu(&p)?;
                if acceptes.contains(&obtenu.as_str()) {
                    r.fichiers_verifies += 1;
                    r.octets_verifies += fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                } else {
                    r.ecarts.push(Ecart {
                        chemin: p,
                        nature: "sha256".into(),
                        detail: format!("attendu {sha}, obtenu {obtenu}"),
                    });
                }
            }
            None if journal.supprimes.contains(nom) => {}
            None => r.ecarts.push(Ecart {
                chemin,
                nature: "manquant".into(),
                detail: "fichier consigné absent".into(),
            }),
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------
// Copies atomiques (export)
// ------------------------------------------------------------------------------------------

fn copier_flux(mut src: impl Read, dst: &Path, octets_prevus: u64, garde: &GardeSsd) -> Resultat<u64> {
    let mut ec = npy::Ecriture::creer(dst, octets_prevus, garde, false)?;
    let mut tampon = vec![0u8; 1 << 22];
    let mut total = 0u64;
    loop {
        let n = src
            .read(&mut tampon)
            .map_err(|e| Erreur::io(format!("lecture pour {}", dst.display()), e))?;
        if n == 0 {
            break;
        }
        ec.ecrire(&tampon[..n])?;
        total += n as u64;
    }
    ec.fermer()?;
    ec.publier()?;
    Ok(total)
}

/// Chemin sans `.zst` final.
fn sans_zst(p: &Path) -> PathBuf {
    if compression::est_zst(p) {
        p.with_extension("")
    } else {
        p.to_path_buf()
    }
}

impl Store {
    pub fn ouvrir(racine: &Path) -> Resultat<Store> {
        if !racine.is_dir() {
            return Err(Erreur::Config(format!(
                "le store {} n'existe pas (ou n'est pas un répertoire)",
                racine.display()
            )));
        }
        Ok(Store {
            racine: racine.to_path_buf(),
        })
    }

    fn racine_canonique(&self) -> Resultat<PathBuf> {
        fs::canonicalize(&self.racine).map_err(io_ctx("chemin canonique de", &self.racine))
    }

    /// Chemin d'un ensemble désigné par son identifiant : relatif, sans `..`, sous la racine.
    pub fn chemin_ensemble(&self, id: &str) -> Resultat<PathBuf> {
        let p = Path::new(id);
        if id.is_empty() || p.is_absolute() || !p.components().all(|c| matches!(c, Component::Normal(_))) {
            return Err(Erreur::Config(format!(
                "identifiant d'ensemble invalide « {id} » (chemin relatif au store, sans « .. »)"
            )));
        }
        let complet = self.racine.join(p);
        let canon = fs::canonicalize(&complet)
            .map_err(|_| Erreur::Config(format!("ensemble « {id} » introuvable dans {}", self.racine.display())))?;
        if !canon.starts_with(self.racine_canonique()?) {
            return Err(Erreur::Config(format!(
                "ensemble « {id} » hors du store (lien symbolique ?) : refusé"
            )));
        }
        Ok(canon)
    }

    /// Occupation par ensemble (et des répertoires externes passés en `--chemin`), trié décroissant.
    /// Les chemins externes ne sont que LUS.
    pub fn du(&self, autres: &[PathBuf]) -> Resultat<Vec<LigneDu>> {
        let mut lignes = Vec::new();
        for (p, ft) in entrees(&self.racine)? {
            if ft.is_symlink() {
                continue;
            }
            lignes.push(ligne_du(&p)?);
        }
        for a in autres {
            lignes.push(ligne_du(a)?);
        }
        lignes.sort_by(|a, b| b.octets.cmp(&a.octets).then_with(|| a.chemin.cmp(&b.chemin)));
        Ok(lignes)
    }

    /// Une ligne par ensemble (enfant direct du store), avec les champs utiles du manifeste.
    pub fn lister(&self) -> Resultat<Vec<LigneListe>> {
        let mut v = Vec::new();
        for (p, ft) in entrees(&self.racine)? {
            if !ft.is_dir() {
                continue;
            }
            let du = ligne_du(&p)?;
            let m = lire_manifeste(&p);
            let get = |k: &str| m.as_ref().and_then(|m| m.get(k));
            v.push(LigneListe {
                id: p
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                complete: du.complete,
                garder: est_fichier_regulier(&p.join(NOM_GARDER)),
                octets: du.octets,
                fichiers: du.fichiers,
                age_s: du.age_s,
                machine: get("machine").and_then(Value::as_str).map(str::to_string),
                moteur: get("moteur").and_then(Value::as_str).map(str::to_string),
                rc: get("rc").and_then(Value::as_i64),
                debut_unix: get("debut_unix").and_then(Value::as_f64),
                chemin: p,
            });
        }
        v.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(v)
    }

    /// gc : runs sans COMPLETE dont le pid est mort et plus vieux que `age_min_s`, reprises
    /// orphelines, puis éclaircissement. N'agit QUE sous `racine`, et seulement si `appliquer`.
    /// Un ensemble contenant `.garder` n'est jamais touché.
    pub fn gc(&self, p: &PolitiqueEclaircissement, age_min_s: f64, appliquer: bool) -> Resultat<RapportGc> {
        self.gc_avec_garde(p, age_min_s, appliquer, &GardeSsd::machine())
    }

    /// Comme [`Store::gc`], avec la garde SSD explicite (la conversion c16 → q64 écrit avant de remplacer).
    pub fn gc_avec_garde(
        &self,
        p: &PolitiqueEclaircissement,
        age_min_s: f64,
        appliquer: bool,
        garde: &GardeSsd,
    ) -> Resultat<RapportGc> {
        let racine = self.racine_canonique()?;
        let maison = std::env::var_os("HOME")
            .map(PathBuf::from)
            .and_then(|h| fs::canonicalize(h).ok());
        if racine.parent().is_none() || maison.as_deref() == Some(racine.as_path()) {
            return Err(Erreur::Config(format!(
                "gc refusé : {} n'est pas une racine de store plausible",
                racine.display()
            )));
        }
        let mut rapport = RapportGc {
            applique: appliquer,
            ..RapportGc::default()
        };
        let mut unites = Vec::new();
        let mut protegees = Vec::new();
        if est_fichier_regulier(&racine.join(NOM_GARDER)) {
            rapport.ignores.push((
                racine.clone(),
                format!("{NOM_GARDER} à la racine : store entier protégé"),
            ));
            return Ok(rapport);
        }
        explorer(&racine, &mut unites, &mut protegees)?;
        for pr in &protegees {
            rapport.ignores.push((pr.clone(), format!("protégé par {NOM_GARDER}")));
        }
        let maintenant = maintenant_unix();
        let mut supprimes: Vec<PathBuf> = Vec::new();

        for u in &unites {
            if supprimes.iter().any(|s| u.dir.starts_with(s)) {
                continue;
            }
            if !u.complete {
                self.gc_orphelin(
                    u,
                    &unites,
                    &protegees,
                    age_min_s,
                    maintenant,
                    appliquer,
                    &racine,
                    &mut rapport,
                    &mut supprimes,
                )?;
                continue;
            }
            self.gc_complete(u, p, age_min_s, maintenant, appliquer, &racine, garde, &mut rapport)?;
        }
        Ok(rapport)
    }

    /// Unité sans COMPLETE : supprimée si pid mort ET plus vieille que `age_min_s` ET sans rien de précieux dedans.
    #[allow(clippy::too_many_arguments)]
    fn gc_orphelin(
        &self,
        u: &Unite,
        unites: &[Unite],
        protegees: &[PathBuf],
        age_min_s: f64,
        maintenant: f64,
        appliquer: bool,
        racine: &Path,
        rapport: &mut RapportGc,
        supprimes: &mut Vec<PathBuf>,
    ) -> Resultat<()> {
        if !pid_mort(u) {
            let raison = match (u.pid, u.hote.as_deref()) {
                (Some(_), Some(h)) if h == nom_hote() => "run sans COMPLETE dont le processus vit encore",
                _ => "run sans COMPLETE : pid ou hôte inconnu, laissé en place",
            };
            rapport.ignores.push((u.dir.clone(), raison.into()));
            return Ok(());
        }
        let mut tout = Vec::new();
        parcourir(&u.dir, &mut tout)?;
        let recent = tout.iter().map(|f| f.mtime).fold(0.0, f64::max);
        if maintenant - recent <= age_min_s {
            rapport.ignores.push((
                u.dir.clone(),
                format!("run sans COMPLETE trop récent (moins de {:.0} h)", age_min_s / 3600.0),
            ));
            return Ok(());
        }
        let precieux = protegees.iter().any(|p| p.starts_with(&u.dir))
            || unites
                .iter()
                .any(|v| v.dir != u.dir && v.dir.starts_with(&u.dir) && v.complete);
        if precieux {
            rapport.ignores.push((
                u.dir.clone(),
                "contient un sous-ensemble COMPLETE ou protégé : laissé en place".into(),
            ));
            return Ok(());
        }
        let octets: u64 = tout.iter().map(|f| f.octets).sum();
        rapport.actions.push((u.dir.clone(), octets, "run-orphelin".into()));
        rapport.octets_liberes += octets;
        supprimes.push(u.dir.clone());
        if appliquer {
            self.verifier_sous_racine(racine, &u.dir)?;
            fs::remove_dir_all(&u.dir).map_err(io_ctx("suppression de", &u.dir))?;
        }
        Ok(())
    }

    /// Unité COMPLETE : reprises, fichiers partiels, éclaircissement, conversion, compression.
    #[allow(clippy::too_many_arguments)]
    fn gc_complete(
        &self,
        u: &Unite,
        p: &PolitiqueEclaircissement,
        age_min_s: f64,
        maintenant: f64,
        appliquer: bool,
        racine: &Path,
        garde_ssd: &GardeSsd,
        rapport: &mut RapportGc,
    ) -> Resultat<()> {
        let mut fics = Vec::new();
        fichiers_unite(&u.dir, &mut fics)?;
        let mut a_supprimer: BTreeMap<PathBuf, (u64, String)> = BTreeMap::new();

        // 2. reprises d'un run COMPLETE, et fichiers partiels abandonnés
        for f in &fics {
            let nom = f
                .chemin
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if p.purger_reprises && nom.starts_with("reprise") {
                a_supprimer.insert(f.chemin.clone(), (f.octets, "reprise".into()));
            } else if nom.ends_with(".part") && maintenant - f.mtime > age_min_s {
                a_supprimer.insert(f.chemin.clone(), (f.octets, "partiel-orphelin".into()));
            }
        }

        // 3. éclaircissement : un dump sur k par répertoire, premier et dernier gardés
        let mut par_repertoire: BTreeMap<PathBuf, Vec<(u64, &Fic)>> = BTreeMap::new();
        for f in &fics {
            let nom = f
                .chemin
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if let (Some(i), Some(parent)) = (indice_dump(&nom), f.chemin.parent()) {
                par_repertoire.entry(parent.to_path_buf()).or_default().push((i, f));
            }
        }
        let mut gardes: Vec<&Fic> = Vec::new();
        let journal = lire_journal(&u.dir);
        for (repertoire, dumps) in par_repertoire.iter_mut() {
            dumps.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.chemin.cmp(&b.1.chemin)));
            let n = dumps.len();
            // Idempotence : un répertoire déjà éclairci avec ce k (ou un k plus grand) n'est pas éclairci à nouveau ;
            // un k strictement plus grand éclaircit ce qui reste.
            let deja = journal
                .eclaircis
                .get(&rel_repertoire(&u.dir, repertoire))
                .copied()
                .unwrap_or(0);
            let k = if p.garder_chaque <= deja { 0 } else { p.garder_chaque };
            for (pos, (_, f)) in dumps.iter().enumerate() {
                let garde = k < 2 || pos == 0 || pos == n - 1 || (pos as u64).is_multiple_of(k);
                if garde {
                    gardes.push(f);
                } else {
                    a_supprimer.insert(
                        f.chemin.clone(),
                        (f.octets, format!("eclaircissement 1/{}", p.garder_chaque)),
                    );
                }
            }
        }

        for (chemin, (octets, raison)) in &a_supprimer {
            rapport.actions.push((chemin.clone(), *octets, raison.clone()));
            rapport.octets_liberes += octets;
            if appliquer {
                self.verifier_sous_racine(racine, chemin)?;
                ajouter_au_journal(
                    &u.dir,
                    &json!({
                        "horodatage": maintenant_unix(), "action": "supprime", "fichier": relatif(&u.dir, chemin),
                        "octets": octets, "raison": raison,
                        "eclaircissement_k": raison.starts_with("eclaircissement").then_some(p.garder_chaque),
                        "repertoire": chemin.parent().map(|d| rel_repertoire(&u.dir, d)),
                    }),
                )?;
                fs::remove_file(chemin).map_err(io_ctx("suppression de", chemin))?;
            }
        }

        // 4. conversion c16 -> q64 (vérifiée), puis compression facultative des dumps conservés
        let zstd_ok = p.compresser_zstd && compression::disponible();
        for f in gardes {
            if compression::est_zst(&f.chemin) {
                continue;
            }
            let chemin = f.chemin.clone();
            if p.convertir_q64 {
                match npy::inspecter_champ(&chemin) {
                    Ok((FormatChamp::C16, l)) => {
                        if appliquer {
                            self.verifier_sous_racine(racine, &chemin)?;
                            let nom = relatif(&u.dir, &chemin);
                            let r = npy::convertir_champ_avec(
                                &chemin,
                                &chemin,
                                FormatChamp::Q64,
                                garde_ssd,
                                true,
                                &mut |r| {
                                    ajouter_au_journal(
                                        &u.dir,
                                        &json!({
                                            "horodatage": maintenant_unix(), "action": "convertit", "fichier": nom,
                                            "format_avant": "c16", "format_apres": "q64",
                                            "avant": {"octets": f.octets},
                                            "apres": {"octets": r.octets, "sha256": r.sha256},
                                        }),
                                    )
                                },
                            );
                            match r {
                                Ok(r) => {
                                    let gain = f.octets.saturating_sub(r.octets);
                                    rapport.actions.push((chemin.clone(), gain, "conversion-q64".into()));
                                    rapport.octets_liberes += gain;
                                }
                                Err(e) => rapport
                                    .ignores
                                    .push((chemin.clone(), format!("conversion q64 refusée : {e}"))),
                            }
                        } else {
                            let gain = f.octets.saturating_sub(npy::octets_champ(l, FormatChamp::Q64));
                            rapport.actions.push((chemin.clone(), gain, "conversion-q64".into()));
                            rapport.octets_liberes += gain;
                        }
                    }
                    Ok(_) => {}
                    Err(e) => rapport
                        .ignores
                        .push((chemin.clone(), format!("illisible, laissé en place : {e}"))),
                }
            }
            if zstd_ok && est_fichier_regulier(&chemin) {
                if appliquer {
                    let nom = relatif(&u.dir, &chemin);
                    ajouter_au_journal(
                        &u.dir,
                        &json!({
                            "horodatage": maintenant_unix(), "action": "compresse", "fichier": nom,
                            "apres": {"fichier": format!("{nom}.zst")},
                        }),
                    )?;
                    let avant = fs::metadata(&chemin).map(|m| m.len()).unwrap_or(0);
                    match compression::compresser(&chemin) {
                        Ok(z) => {
                            let apres = fs::metadata(&z).map(|m| m.len()).unwrap_or(avant);
                            let gain = avant.saturating_sub(apres);
                            rapport.actions.push((z, gain, "compression-zstd".into()));
                            rapport.octets_liberes += gain;
                        }
                        Err(e) => rapport
                            .ignores
                            .push((chemin.clone(), format!("compression refusée : {e}"))),
                    }
                } else {
                    rapport
                        .actions
                        .push((chemin.clone(), 0, "compression-zstd (gain non estimé)".into()));
                }
            }
        }
        Ok(())
    }

    /// Garde-fou avant toute modification : `chemin` est sous la racine canonique, n'est pas un lien
    /// symbolique, et son répertoire parent (canonique) est sous la racine.
    fn verifier_sous_racine(&self, racine: &Path, chemin: &Path) -> Resultat<()> {
        if est_lien(chemin) {
            return Err(Erreur::Validation(format!(
                "{} est un lien symbolique : jamais suivi, jamais modifié",
                chemin.display()
            )));
        }
        let parent = chemin.parent().unwrap_or(chemin);
        let canon = fs::canonicalize(parent).map_err(io_ctx("chemin canonique de", parent))?;
        if !canon.starts_with(racine) || !chemin.starts_with(racine) || chemin == racine {
            return Err(Erreur::Validation(format!(
                "{} est hors de la racine du store {} : refus",
                chemin.display(),
                racine.display()
            )));
        }
        Ok(())
    }

    /// Revérifie les sha256 du manifeste d'un ensemble.
    pub fn verifier(&self, id: &str) -> Resultat<()> {
        self.verifier_rapport(id)?.en_resultat(id)
    }

    /// Comme [`Store::verifier`], avec le rapport détaillé (rien n'est une erreur ici sauf une E/S).
    pub fn verifier_rapport(&self, id: &str) -> Resultat<RapportVerif> {
        verifier_repertoire(&self.chemin_ensemble(id)?)
    }

    /// Exporte un ensemble vers un répertoire NEUF `destination` : les dumps `cfg_*.npy[.zst]` sont écrits
    /// en `.npy` du `format` demandé (conversion en flux, aller-retour vérifié), le reste est copié ; les
    /// reprises et les `.part` sont omis. Garde SSD avant chaque écriture. Consigne `EXPORT.json`.
    pub fn exporter(
        &self,
        id: &str,
        destination: &Path,
        format: FormatChamp,
        garde: &GardeSsd,
    ) -> Resultat<RapportExport> {
        let source = self.chemin_ensemble(id)?;
        let mut fics = Vec::new();
        parcourir(&source, &mut fics)?;
        let a_exporter: Vec<&Fic> = fics
            .iter()
            .filter(|f| {
                let nom = f
                    .chemin
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                !nom.starts_with("reprise") && !nom.ends_with(".part")
            })
            .collect();
        let total: u64 = a_exporter.iter().map(|f| f.octets).sum();
        garde.verifier(destination, total / 2)?;
        creer_repertoire_neuf(destination)
            .map_err(|e| Erreur::Config(format!("export vers {} : {e}", destination.display())))?;
        let mut rapport = RapportExport {
            destination: destination.to_path_buf(),
            ..RapportExport::default()
        };
        let mut consignes = Vec::new();
        for f in a_exporter {
            let rel = f.chemin.strip_prefix(&source).unwrap_or(&f.chemin).to_path_buf();
            let nom = f
                .chemin
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let est_dump = indice_dump(&nom).is_some();
            let dst = if est_dump {
                sans_zst(&destination.join(&rel))
            } else {
                destination.join(&rel)
            };
            if let Some(parent) = dst.parent() {
                fs::create_dir_all(parent).map_err(io_ctx("création de", parent))?;
            }
            let octets = if est_dump {
                let (fmt, _) = npy::inspecter_champ(&f.chemin)?;
                if fmt == format {
                    let src: Box<dyn Read> = if compression::est_zst(&f.chemin) {
                        Box::new(compression::ouvrir_flux(&f.chemin)?)
                    } else {
                        Box::new(fs::File::open(&f.chemin).map_err(io_ctx("ouverture", &f.chemin))?)
                    };
                    copier_flux(src, &dst, f.octets, garde)?
                } else {
                    rapport.convertis += 1;
                    npy::convertir_champ(&f.chemin, &dst, format, garde, true)?.octets
                }
            } else {
                let src = fs::File::open(&f.chemin).map_err(io_ctx("ouverture", &f.chemin))?;
                copier_flux(src, &dst, f.octets, garde)?
            };
            rapport.fichiers += 1;
            rapport.octets += octets;
            consignes.push(json!({"nom": rel.to_string_lossy(), "octets": octets, "dump": est_dump}));
        }
        let resume = json!({
            "source": id, "format": format.nom(), "horodatage": maintenant_unix(), "hote": nom_hote(),
            "fichiers": consignes, "convertis": rapport.convertis,
        });
        let chemin = destination.join("EXPORT.json");
        copier_flux(resume.to_string().as_bytes(), &chemin, 4096, garde)?;
        Ok(rapport)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_de_dumps() {
        assert_eq!(indice_dump("cfg_000250.npy"), Some(250));
        assert_eq!(indice_dump("cfg_000250.npy.zst"), Some(250));
        assert_eq!(indice_dump("cfg_.npy"), None);
        assert_eq!(indice_dump("cfg_12a.npy"), None);
        assert_eq!(indice_dump("blk.npy"), None);
        assert_eq!(indice_dump("cfg_000001.npy.part"), None);
    }

    #[test]
    fn politique_prudente() {
        let p = PolitiqueEclaircissement::prudente();
        assert!(p.purger_reprises && !p.convertir_q64 && p.garder_chaque < 2);
    }
}
