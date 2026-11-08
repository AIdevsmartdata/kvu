//! Index d'archive `kvu-archive/1` des configurations SU(2), lecteur et vérificateur.
//!
//! Un index est une fonction pure du contenu d'un dossier de run `su2 gen --archive` :
//! le reconstruire redonne exactement le même fichier à l'octet (sur la même machine,
//! pour la même version de `zstd`). Il n'y a ni horodatage, ni durée, ni champ volatile.

use crate::commun::{ChampHote, Depart, Erreur, Resultat};
use crate::heritage::manifest::{self, Sha256};
use crate::stockage::{compression, npy};
use crate::su2::run::ConfigRun;
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const SCHEMA_ARCHIVE: &str = "kvu-archive/1";
pub const NOM_INDEX: &str = "archive.json";
const NOM_COMPLETE: &str = "COMPLETE";
const NOM_CONFIG_RUN: &str = "config_run.json";
const NOM_MANIFESTE: &str = "manifest.json";

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EntreeArchive {
    pub fichier: String,
    pub indice: u64,
    pub balayages: u64,
    pub octets: u64,
    pub sha256: String,
    pub octets_contenu: u64,
    pub sha256_contenu: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChaineArchive {
    pub chaine: usize,
    pub dossier: String,
    pub beta: f64,
    pub graine: u64,
    pub depart: String,
    pub configurations: Vec<EntreeArchive>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IndexArchive {
    pub format: String,
    pub l: usize,
    pub nor: usize,
    pub precision: String,
    pub therm: u64,
    pub nmeas: u64,
    pub skip: u64,
    pub dump_every: u64,
    pub dump_initial: bool,
    pub format_champ: String,
    pub compression: String,
    pub provenance: Option<serde_json::Value>,
    pub complet: bool,
    pub chaines: Vec<ChaineArchive>,
    pub note: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Defaut {
    pub fichier: String,
    pub nature: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RapportVerification {
    pub n_configurations: usize,
    pub n_verifiees: usize,
    pub defauts: Vec<Defaut>,
}

#[derive(Clone, Debug)]
struct ChaineConfig {
    beta: f64,
    graine: u64,
    depart: String,
}

#[derive(Clone, Debug)]
struct ChampsConfig {
    l: usize,
    nor: usize,
    precision: String,
    therm: u64,
    nmeas: u64,
    skip: u64,
    dump_every: u64,
    dump_initial: bool,
    format_champ: String,
    zstd: bool,
    chaines: Vec<ChaineConfig>,
}

fn io(contexte: &str, chemin: &Path, e: std::io::Error) -> Erreur {
    Erreur::io(format!("{contexte} {}", chemin.display()), e)
}

fn lire_json_option(chemin: &Path) -> Resultat<Option<Value>> {
    if !chemin.is_file() {
        return Ok(None);
    }
    let texte = fs::read_to_string(chemin).map_err(|e| io("lecture de", chemin, e))?;
    let valeur: Value = serde_json::from_str(&texte)
        .map_err(|e| Erreur::Format(format!("{} illisible : {e}", chemin.display())))?;
    Ok(Some(valeur))
}

fn charger_config_run(outdir: &Path) -> Resultat<Value> {
    let chemin = outdir.join(NOM_CONFIG_RUN);
    if !chemin.is_file() {
        return Err(Erreur::Config(format!(
            "{} : pas un run su2 ({NOM_CONFIG_RUN} absent)",
            outdir.display()
        )));
    }
    let texte = fs::read_to_string(&chemin).map_err(|e| io("lecture de", &chemin, e))?;
    serde_json::from_str(&texte)
        .map_err(|e| Erreur::Format(format!("{} illisible : {e}", chemin.display())))
}

fn champs_config(valeur: &Value) -> Resultat<ChampsConfig> {
    // Valider toute la structure du run, y compris les booléens obligatoires :
    // un champ absent ou mal typé ne doit jamais devenir silencieusement `false`.
    let cfg: ConfigRun = serde_json::from_value(valeur.clone())
        .map_err(|e| Erreur::Format(format!("{NOM_CONFIG_RUN} : configuration invalide : {e}")))?;
    if cfg.lot.chaines.is_empty() {
        return Err(Erreur::Format(format!(
            "{NOM_CONFIG_RUN} : aucune chaîne dans le lot"
        )));
    }
    let chaines = cfg.lot.chaines.iter().map(|chaine| ChaineConfig {
        beta: chaine.beta,
        graine: chaine.graine.0,
        depart: match chaine.depart {
            Depart::Chaud => "chaud",
            Depart::Froid => "froid",
        }
        .to_string(),
    }).collect();

    Ok(ChampsConfig {
        l: cfg.lot.l,
        nor: cfg.lot.nor,
        precision: cfg.lot.precision.nom().to_string(),
        therm: cfg.therm,
        nmeas: cfg.nmeas,
        skip: cfg.skip,
        dump_every: cfg.dump_every,
        dump_initial: cfg.dump_initial,
        format_champ: cfg.format_dump,
        zstd: cfg.zstd,
        chaines,
    })
}

fn dossier_chaine(outdir: &Path, nchaines: usize, chaine: usize) -> PathBuf {
    crate::su2::run::repertoire_chaine(outdir, nchaines, chaine)
}

fn indice_config(base: &str) -> Option<u64> {
    let n = base.strip_prefix("cfg_")?;
    if n.len() != 6 || !n.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    n.parse().ok()
}

/// Vrai si le nom est un candidat `cfg_*` (compressé ou non, canonique ou non)
/// ou un résidu `.part` — tout ce qui doit être signalé `non_indexe` par `verifier`.
fn est_candidat_non_indexe(nom: &str) -> bool {
    nom.ends_with(".part") || (nom.starts_with("cfg_") && (nom.ends_with(".npy") || nom.ends_with(".npy.zst")))
}

fn provenance_depuis_manifeste(outdir: &Path) -> Resultat<Option<Value>> {
    let manifeste = lire_json_option(&outdir.join(NOM_MANIFESTE))?;
    let Some(man) = manifeste else {
        return Ok(None);
    };
    let copier = |nom: &str| man.get(nom).cloned().unwrap_or(Value::Null);
    Ok(Some(serde_json::json!({
        "binaire": copier("binaire"),
        "binaire_sha256": copier("binaire_sha256"),
        "argv": copier("argv"),
        "source_hash": copier("source_hash"),
        "git_head": copier("git_head"),
        "version": copier("version"),
        "hote": copier("hote"),
    })))
}

fn relatif(outdir: &Path, chemin: &Path) -> String {
    match chemin.strip_prefix(outdir) {
        Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
        Err(_) => chemin
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| chemin.to_string_lossy().into_owned()),
    }
}

fn sha_manifest_pour_chemin(
    manifeste: Option<&Value>,
    outdir: &Path,
    chemin: &Path,
) -> Resultat<Option<String>> {
    let Some(man) = manifeste else {
        return Ok(None);
    };
    let Some(fichiers) = man.get("fichiers").and_then(Value::as_array) else {
        return Ok(None);
    };
    let rel = relatif(outdir, chemin);
    for f in fichiers {
        let Some(nom) = f.get("nom").and_then(Value::as_str) else {
            continue;
        };
        let nom_normalise = nom.replace('\\', "/");
        // Intégrateur (M1 r1) : correspondance EXACTE du chemin relatif au dossier de sortie (nom consigné par le manifeste) ;
        // r1 acceptait le seul nom de base, et dans un lot c001/cfg_000005 prenait l'empreinte de c000/cfg_000005 (fausse corruption).
        if nom_normalise == rel {
            // Intégrateur (M1 r3, objection S1 d'Astra) : une entrée LISTÉE sans empreinte valide (absente, null ou mal typée) est une
            // erreur de format, jamais une absence d'entrée (sinon la concordance exigée au §2.2 est contournée).
            return match f.get("sha256").and_then(Value::as_str) {
                Some(sha) => Ok(Some(sha.to_string())),
                None => Err(Erreur::Format(format!(
                    "{NOM_MANIFESTE} : l'entrée « {nom} » n'a pas d'empreinte sha256 valide"
                ))),
            };
        }
    }
    Ok(None)
}

fn hash_contenu(chemin: &Path) -> Resultat<(u64, String)> {
    let mut flux = compression::ouvrir_flux(chemin)?;
    let mut sha = Sha256::new();
    let mut total = 0u64;
    let mut tampon = [0u8; 1 << 20];
    loop {
        match flux.read(&mut tampon) {
            Ok(0) => break,
            Ok(n) => {
                sha.update(&tampon[..n]);
                total += n as u64;
            }
            Err(e) => {
                return Err(Erreur::io(
                    format!("lecture de zstd -dc {}", chemin.display()),
                    e,
                ))
            }
        }
    }
    flux.terminer()?;
    Ok((total, manifest::hex(&sha.finalize())))
}

/// Balayages au dump d'indice `indice` : 0 pour l'indice 0, `therm + indice·skip` sinon.
/// Les opérations sont contrôlées : tout débordement rend une `Erreur::Validation`.
fn balayages_pour_indice(therm: u64, skip: u64, indice: u64) -> Resultat<u64> {
    if indice == 0 {
        return Ok(0);
    }
    let produit = indice
        .checked_mul(skip)
        .ok_or_else(|| Erreur::Validation(format!("balayages : produit indice·skip débordé (indice={indice}, skip={skip})")))?;
    therm
        .checked_add(produit)
        .ok_or_else(|| Erreur::Validation(format!("balayages : somme therm + indice·skip débordée (therm={therm})")))
}

fn construire_entree(
    chemin: &Path,
    indice: u64,
    balayages: u64,
    outdir: &Path,
    manifeste: Option<&Value>,
) -> Resultat<EntreeArchive> {
    let octets = fs::metadata(chemin)
        .map_err(|e| io("stat de", chemin, e))?
        .len();
    let sha256 = manifest::sha256_fichier(chemin).map_err(|e| io("sha256 de", chemin, e))?;
    if let Some(attendu) = sha_manifest_pour_chemin(manifeste, outdir, chemin)? {
        if attendu != sha256 {
            return Err(Erreur::Validation(format!(
                "{} : le sha256 du manifeste diffère de celui du fichier (corruption depuis l'écriture)",
                relatif(outdir, chemin)
            )));
        }
    }
    let (octets_contenu, sha256_contenu) = hash_contenu(chemin)?;
    let fichier = chemin
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .ok_or_else(|| Erreur::Config(format!("{} : nom de fichier invalide", chemin.display())))?;
    // Le nom construit est canonique par construction (`decouvrir_indices` ne renvoie que des indices canoniques) ;
    // on vérifie tout de même l'invariant pour prévenir toute régression.
    let attendu = format!("cfg_{indice:06}.npy.zst");
    if fichier != attendu {
        return Err(Erreur::Validation(format!(
            "{} : nom de fichier non canonique pour l'indice {indice}",
            chemin.display()
        )));
    }
    Ok(EntreeArchive {
        fichier,
        indice,
        balayages,
        octets,
        sha256,
        octets_contenu,
        sha256_contenu,
    })
}

/// Découvre les indices des configurations d'un dossier de chaîne, en ignorant les résidus `.part`,
/// et REFUSE tout fichier `cfg_*.npy` non compressé (anomalie d'archive).
fn decouvrir_indices(dir: &Path) -> Resultat<Vec<u64>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut resultat = Vec::new();
    let entrees = fs::read_dir(dir).map_err(|e| io("lecture du répertoire", dir, e))?;
    let mut noms: Vec<String> = entrees
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    noms.sort();
    for nom in noms {
        if nom.ends_with(".part") {
            continue;
        }
        // Tout fichier cfg_*.npy non compressé est une anomalie bloquante.
        if nom.starts_with("cfg_") && nom.ends_with(".npy") {
            return Err(Erreur::Config(format!(
                "{} : dump non compressé dans une archive ; relancer avec --zstd",
                dir.join(&nom).display()
            )));
        }
        if let Some(base) = nom.strip_suffix(".npy.zst") {
            if let Some(indice) = indice_config(base) {
                resultat.push(indice);
            }
        }
    }
    Ok(resultat)
}

fn valider_indice(indice: u64, cfg: &ChampsConfig) -> Resultat<()> {
    if indice == 0 {
        if !cfg.dump_initial {
            return Err(Erreur::Validation(
                "indice 0 présent sans --dump-initial".to_string(),
            ));
        }
    } else if !indice.is_multiple_of(cfg.dump_every) {
        return Err(Erreur::Validation(format!(
            "indice {indice} non multiple de --dump-every {}",
            cfg.dump_every
        )));
    }
    Ok(())
}

fn valider_index(idx: &IndexArchive) -> Resultat<()> {
    if idx.format != SCHEMA_ARCHIVE {
        return Err(Erreur::Format(format!(
            "{NOM_INDEX} : schéma « {} » inattendu (attendu {SCHEMA_ARCHIVE})",
            idx.format
        )));
    }
    if idx.compression != "zstd" {
        return Err(Erreur::Format(format!(
            "{NOM_INDEX} : compression « {} » inattendue (attendu zstd)",
            idx.compression
        )));
    }
    if !matches!(idx.precision.as_str(), "f64" | "f32") {
        return Err(Erreur::Format(format!(
            "{NOM_INDEX} : précision « {} » invalide",
            idx.precision
        )));
    }
    if !matches!(idx.format_champ.as_str(), "c16" | "q64") {
        return Err(Erreur::Format(format!(
            "{NOM_INDEX} : format de champ « {} » invalide",
            idx.format_champ
        )));
    }
    if idx.dump_every == 0 {
        return Err(Erreur::Format(format!(
            "{NOM_INDEX} : dump_every doit être ≥ 1"
        )));
    }
    if idx.chaines.is_empty() {
        return Err(Erreur::Format(format!(
            "{NOM_INDEX} : aucune chaîne dans l'index"
        )));
    }
    let une_chaine = idx.chaines.len() == 1;
    for (position, chaine) in idx.chaines.iter().enumerate() {
        if chaine.chaine != position {
            return Err(Erreur::Format(format!(
                "{NOM_INDEX} : chaîne {} alors que {position} était attendu",
                chaine.chaine
            )));
        }
        let dossier_attendu = if une_chaine {
            ".".to_string()
        } else {
            format!("c{position:03}")
        };
        if chaine.dossier != dossier_attendu {
            return Err(Erreur::Format(format!(
                "{NOM_INDEX} : dossier « {} » invalide pour la chaîne {position}",
                chaine.dossier
            )));
        }
        // Intégrateur (M1 r3, objection S2 d'Astra) : domaine de `depart` (§2.2 : « chaud » | « froid »).
        if !matches!(chaine.depart.as_str(), "chaud" | "froid") {
            return Err(Erreur::Format(format!(
                "{NOM_INDEX} : départ « {} » invalide pour la chaîne {position} (chaud | froid)",
                chaine.depart
            )));
        }
        let mut precedent = None;
        for entree in &chaine.configurations {
            valider_indice(entree.indice, &ChampsConfig {
                l: idx.l,
                nor: idx.nor,
                precision: idx.precision.clone(),
                therm: idx.therm,
                nmeas: idx.nmeas,
                skip: idx.skip,
                dump_every: idx.dump_every,
                dump_initial: idx.dump_initial,
                format_champ: idx.format_champ.clone(),
                zstd: true,
                chaines: Vec::new(),
            })?;
            let indice_decode = entree.fichier.strip_suffix(".npy.zst").and_then(indice_config);
            if indice_decode != Some(entree.indice) {
                return Err(Erreur::Format(format!(
                    "{NOM_INDEX} : fichier « {} » non canonique pour l'indice {} (six chiffres exacts requis)",
                    entree.fichier, entree.indice
                )));
            }
            let balayages_attendu = balayages_pour_indice(idx.therm, idx.skip, entree.indice)?;
            if entree.balayages != balayages_attendu {
                return Err(Erreur::Format(format!(
                    "{NOM_INDEX} : balayages {} pour l'indice {} (attendu {balayages_attendu})",
                    entree.balayages, entree.indice
                )));
            }
            if let Some(prec) = precedent {
                if prec >= entree.indice {
                    return Err(Erreur::Format(format!(
                        "{NOM_INDEX} : configurations non triées ou indice dupliqué dans la chaîne {}",
                        chaine.chaine
                    )));
                }
            }
            precedent = Some(entree.indice);
        }
    }
    Ok(())
}

/// Construit l'index `kvu-archive/1` à partir des fichiers présents, de `config_run.json`
/// et de `manifest.json`. N'écrit rien.
pub fn construire_index(outdir: &Path) -> Resultat<IndexArchive> {
    let config = charger_config_run(outdir)?;
    let cfg = champs_config(&config)?;
    if !cfg.zstd {
        return Err(Erreur::Config(
            "l'archivage exige un run compressé ; relancer avec --zstd".to_string(),
        ));
    }
    if cfg.dump_every == 0 {
        return Err(Erreur::Config("--archive exige --dump-every N ≥ 1".to_string()));
    }
    if !config.get("bord").is_none_or(|b| b.is_null()) {
        return Err(Erreur::Config("--archive est incompatible avec --bord".to_string()));
    }
    if !compression::disponible() {
        return Err(Erreur::Config(
            "--archive exige le binaire système `zstd`, absent".to_string(),
        ));
    }
    if !matches!(cfg.format_champ.as_str(), "c16" | "q64") {
        return Err(Erreur::Format(format!(
            "{NOM_CONFIG_RUN} : format de dump « {} » inconnu (c16|q64)",
            cfg.format_champ
        )));
    }
    if !matches!(cfg.precision.as_str(), "f64" | "f32") {
        return Err(Erreur::Format(format!(
            "{NOM_CONFIG_RUN} : précision « {} » invalide",
            cfg.precision
        )));
    }

    let manifeste = lire_json_option(&outdir.join(NOM_MANIFESTE))?;
    let provenance = provenance_depuis_manifeste(outdir)?;
    let complet = outdir.join(NOM_COMPLETE).is_file();
    let nchaines = cfg.chaines.len();
    let mut chaines = Vec::with_capacity(nchaines);

    for c in 0..nchaines {
        let dossier = dossier_chaine(outdir, nchaines, c);
        let mut indices = decouvrir_indices(&dossier)?;
        indices.sort_unstable();
        let mut configurations = Vec::with_capacity(indices.len());
        for indice in indices {
            valider_indice(indice, &cfg)?;
            let nom = format!("cfg_{indice:06}.npy.zst");
            let chemin = dossier.join(&nom);
            let balayages = balayages_pour_indice(cfg.therm, cfg.skip, indice)?;
            configurations.push(construire_entree(
                &chemin,
                indice,
                balayages,
                outdir,
                manifeste.as_ref(),
            )?);
        }
        chaines.push(ChaineArchive {
            chaine: c,
            dossier: if nchaines == 1 {
                ".".to_string()
            } else {
                format!("c{c:03}")
            },
            beta: cfg.chaines[c].beta,
            graine: cfg.chaines[c].graine,
            depart: cfg.chaines[c].depart.clone(),
            configurations,
        });
    }

    Ok(IndexArchive {
        format: SCHEMA_ARCHIVE.to_string(),
        l: cfg.l,
        nor: cfg.nor,
        precision: cfg.precision,
        therm: cfg.therm,
        nmeas: cfg.nmeas,
        skip: cfg.skip,
        dump_every: cfg.dump_every,
        dump_initial: cfg.dump_initial,
        format_champ: cfg.format_champ,
        compression: "zstd".to_string(),
        provenance,
        complet,
        chaines,
        note: "index reconstruit depuis les fichiers ; vérifier avec kvu su2 archive verifier".to_string(),
    })
}

/// Construit l'index puis l'écrit dans `outdir/archive.json` de façon atomique.
pub fn indexer_run(outdir: &Path) -> Resultat<IndexArchive> {
    let index = construire_index(outdir)?;
    let mut texte = serde_json::to_string_pretty(&index)
        .map_err(|e| Erreur::Format(format!("sérialisation de {NOM_INDEX} : {e}")))?;
    texte.push('\n');
    let chemin = outdir.join(NOM_INDEX);
    manifest::ecrire_atomique(&chemin, texte.as_bytes())
        .map_err(|e| io("écriture atomique de", &chemin, e))?;
    Ok(index)
}

/// Lit `outdir/archive.json` et valide le schéma, le tri, l'unicité des indices
/// et la liaison canonique fichier ↔ indice.
pub fn lire_index(outdir: &Path) -> Resultat<IndexArchive> {
    let chemin = outdir.join(NOM_INDEX);
    let texte = fs::read_to_string(&chemin).map_err(|e| io("lecture de", &chemin, e))?;
    let index: IndexArchive = serde_json::from_str(&texte)
        .map_err(|e| Erreur::Format(format!("{} illisible : {e}", chemin.display())))?;
    valider_index(&index)?;
    Ok(index)
}

fn chemin_entree(outdir: &Path, chaine: &ChaineArchive, fichier: &str) -> PathBuf {
    if chaine.dossier == "." {
        outdir.join(fichier)
    } else {
        outdir.join(&chaine.dossier).join(fichier)
    }
}

fn ajouter_defaut(defauts: &mut Vec<Defaut>, defaut: Defaut) {
    if !defauts.contains(&defaut) {
        defauts.push(defaut);
    }
}

fn defaut(fichier: impl Into<String>, nature: impl Into<String>) -> Defaut {
    Defaut {
        fichier: fichier.into(),
        nature: nature.into(),
    }
}

/// Vérifie tous les fichiers indexés et signale tous les défauts rencontrés.
pub fn verifier(outdir: &Path) -> Resultat<RapportVerification> {
    let mut defauts: Vec<Defaut> = Vec::new();
    let index = match lire_index(outdir) {
        Ok(index) => Some(index),
        Err(_) => {
            ajouter_defaut(&mut defauts, defaut(NOM_INDEX, "index"));
            None
        }
    };

    // config_run.json : une configuration présente mais invalide produit un défaut `index`.
    let chemin_config = outdir.join(NOM_CONFIG_RUN);
    let mut config_invalide = false;
    let config_valeur = if chemin_config.is_file() {
        match fs::read_to_string(&chemin_config) {
            Ok(texte) => match serde_json::from_str::<Value>(&texte) {
                Ok(v) => Some(v),
                Err(_) => {
                    config_invalide = true;
                    None
                }
            },
            Err(_) => {
                config_invalide = true;
                None
            }
        }
    } else {
        None
    };
    if config_invalide {
        ajouter_defaut(&mut defauts, defaut(NOM_INDEX, "index"));
    }

    let config_champs = config_valeur.as_ref().and_then(|v| match champs_config(v) {
        Ok(c) => Some(c),
        Err(_) => {
            config_invalide = true;
            None
        }
    });
    if config_invalide {
        ajouter_defaut(&mut defauts, defaut(NOM_INDEX, "index"));
    }

    if let (Some(idx), Some(cfg)) = (&index, &config_champs) {
        let mut incoherent = idx.l != cfg.l || idx.chaines.len() != cfg.chaines.len();
        for (a, b) in idx.chaines.iter().zip(&cfg.chaines) {
            if a.beta != b.beta || a.graine != b.graine {
                incoherent = true;
            }
        }
        if incoherent {
            ajouter_defaut(&mut defauts, defaut(NOM_INDEX, "index"));
        }
    }

    let n_configurations = index
        .as_ref()
        .map(|i| i.chaines.iter().map(|c| c.configurations.len()).sum())
        .unwrap_or(0);
    let mut n_verifiees = 0usize;
    let mut connus: BTreeSet<String> = BTreeSet::new();

    if let Some(idx) = &index {
        for chaine in &idx.chaines {
            for entree in &chaine.configurations {
                let chemin = chemin_entree(outdir, chaine, &entree.fichier);
                connus.insert(relatif(outdir, &chemin));
            }
        }

        for chaine in &idx.chaines {
            for entree in &chaine.configurations {
                let chemin = chemin_entree(outdir, chaine, &entree.fichier);
                let rel = relatif(outdir, &chemin);
                let mut defauts_entree = false;

                if !chemin.is_file() {
                    ajouter_defaut(&mut defauts, defaut(rel, "absent"));
                    continue;
                }

                let octets = match fs::metadata(&chemin) {
                    Ok(m) => m.len(),
                    Err(_) => {
                        ajouter_defaut(&mut defauts, defaut(&rel, "taille"));
                        defauts_entree = true;
                        0
                    }
                };
                if octets != entree.octets {
                    ajouter_defaut(&mut defauts, defaut(&rel, "taille"));
                    defauts_entree = true;
                }

                let sha_fichier = match manifest::sha256_fichier(&chemin) {
                    Ok(s) => Some(s),
                    Err(_) => {
                        ajouter_defaut(&mut defauts, defaut(&rel, "sha256"));
                        defauts_entree = true;
                        None
                    }
                };
                let sha_ok = sha_fichier.as_deref() == Some(entree.sha256.as_str());
                if !sha_ok {
                    ajouter_defaut(&mut defauts, defaut(&rel, "sha256"));
                    defauts_entree = true;
                }

                let contenu_ok = match hash_contenu(&chemin) {
                    Ok((octets_contenu, sha_contenu)) => {
                        let ok = octets_contenu == entree.octets_contenu
                            && sha_contenu == entree.sha256_contenu;
                        if !ok {
                            ajouter_defaut(&mut defauts, defaut(&rel, "contenu"));
                            defauts_entree = true;
                        }
                        ok
                    }
                    Err(_) => {
                        ajouter_defaut(&mut defauts, defaut(&rel, "contenu"));
                        defauts_entree = true;
                        false
                    }
                };

                if contenu_ok {
                    let champ = npy::lire_champ(&chemin);
                    match champ {
                        Ok(c) if c.l == idx.l => {}
                        _ => {
                            ajouter_defaut(&mut defauts, defaut(&rel, "champ"));
                            defauts_entree = true;
                        }
                    }
                }

                if !defauts_entree {
                    n_verifiees += 1;
                }
            }
        }

        // Anomalies : résidus .part et candidats cfg_* non indexés.
        for chaine in &idx.chaines {
            let dossier = if chaine.dossier == "." {
                outdir.to_path_buf()
            } else {
                outdir.join(&chaine.dossier)
            };
            if !dossier.is_dir() {
                continue;
            }
            let entrees = match fs::read_dir(&dossier) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let mut noms: Vec<String> = entrees
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            noms.sort();
            for nom in noms {
                if !est_candidat_non_indexe(&nom) {
                    continue;
                }
                let rel = if chaine.dossier == "." {
                    nom.clone()
                } else {
                    format!("{}/{}", chaine.dossier, nom)
                };
                if !connus.contains(&rel) {
                    ajouter_defaut(&mut defauts, defaut(rel, "non_indexe"));
                }
            }
        }
    }

    defauts.sort_by(|a, b| a.fichier.cmp(&b.fichier).then_with(|| a.nature.cmp(&b.nature)));
    Ok(RapportVerification {
        n_configurations,
        n_verifiees,
        defauts,
    })
}

/// Vérifie les deux empreintes, puis relit et rend la configuration demandée.
///
/// Un indice absent de l'index ou une chaîne hors lot est une erreur de configuration
/// (code 2) ; un fichier corrompu ou un index invalide est une erreur de validation (code 8).
pub fn extraire(outdir: &Path, chaine: usize, indice: u64) -> Resultat<ChampHote> {
    let index = lire_index(outdir)
        .map_err(|_| Erreur::Validation(format!("{NOM_INDEX} : index illisible ou invalide")))?;
    let chaine_index = index
        .chaines
        .iter()
        .find(|c| c.chaine == chaine)
        .ok_or_else(|| Erreur::Config(format!("chaîne {chaine} hors du lot")))?;
    let entree = chaine_index
        .configurations
        .iter()
        .find(|e| e.indice == indice)
        .ok_or_else(|| Erreur::Config(format!("indice {indice} absent de {NOM_INDEX}")))?;
    let chemin = chemin_entree(outdir, chaine_index, &entree.fichier);
    let rel = relatif(outdir, &chemin);

    if !chemin.is_file() {
        return Err(Erreur::Validation(format!("{rel} : absent")));
    }
    let octets = fs::metadata(&chemin)
        .map_err(|e| Erreur::io(format!("stat de {}", chemin.display()), e))?
        .len();
    if octets != entree.octets {
        return Err(Erreur::Validation(format!("{rel} : taille")));
    }
    let sha_fichier = manifest::sha256_fichier(&chemin)
        .map_err(|e| Erreur::io(format!("sha256 de {}", chemin.display()), e))?;
    if sha_fichier != entree.sha256 {
        return Err(Erreur::Validation(format!("{rel} : sha256")));
    }
    let (octets_contenu, sha_contenu) = hash_contenu(&chemin)
        .map_err(|_| Erreur::Validation(format!("{rel} : contenu")))?;
    if octets_contenu != entree.octets_contenu || sha_contenu != entree.sha256_contenu {
        return Err(Erreur::Validation(format!("{rel} : contenu")));
    }
    let champ = npy::lire_champ(&chemin)
        .map_err(|_| Erreur::Validation(format!("{rel} : champ")))?;
    if champ.l != index.l {
        return Err(Erreur::Validation(format!("{rel} : champ")));
    }
    Ok(champ)
}
