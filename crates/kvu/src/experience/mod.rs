// SPDX-License-Identifier: Apache-2.0
//! Traçabilité des expériences : préenregistrements gelés, runs lus, portes et carnet.
//!
//! Le manifeste `kvu-experience/1` conserve les empreintes des pièces originales.
//! Les dates de gel proviennent de leur mtime, sauf déclaration explicite ; les
//! faits techniques proviennent des manifestes de run. Aucun jugement physique
//! ni statut de preuve n'est produit. Les écritures passent par un temporaire
//! voisin et un renommage ; un ajout identique ne change aucun octet.

pub mod carnet;
pub mod cli;
pub mod schema;

mod fiche;
mod modele;
mod verifier;

pub use modele::Experience;
pub use verifier::{verifier, Constat};

use crate::commun::{Erreur, Resultat};
use crate::regen::manifeste::{date_iso, maintenant_iso, sha_fichier};
use serde_json::{json, Map, Value};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const SCHEMA: &str = "kvu-experience/1";
pub const MANIFESTE: &str = "EXPERIENCE.json";
const DERNIERE_SECONDE: u64 = 253_402_300_799;
static TEMPORAIRES: AtomicU64 = AtomicU64::new(0);

/// Empreinte nulle : précédent de la première entrée de l'historique du verdict.
/// Plus grand lot de chaînes développé depuis un manifeste de run (borne anti-explosion mémoire).
pub const MAX_CHAINES: u64 = 1_000_000;

pub const EMPREINTE_NULLE: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Verrou exclusif sur `.EXPERIENCE.lock` : tenu tant que la valeur vit, pour qu'aucune mise à jour du
/// manifeste ne soit perdue par une écriture concurrente (charger, modifier, renommer = une seule section critique).
pub(crate) struct Verrou(#[allow(dead_code)] fs::File);

/// Drapeaux d'ouverture du verrou : O_NOFOLLOW (jamais de lien suivi) et O_NONBLOCK (jamais bloqué par un fichier
/// spécial). Valeurs du noyau Linux par architecture ; ailleurs, seuls les contrôles du descripteur s'appliquent.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const DRAPEAUX_VERROU: i32 = 0o400000 | 0o4000;
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const DRAPEAUX_VERROU: i32 = 0o100000 | 0o4000;
#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
const DRAPEAUX_VERROU: i32 = 0;

/// Vrai si les drapeaux d'ouverture sans suivi de lien ni blocage sont connus pour cette cible.
pub(crate) const VERROU_PRIS_EN_CHARGE: bool = DRAPEAUX_VERROU != 0;

pub(crate) fn verrouiller(racine: &Path) -> Resultat<Verrou> {
    // Pas de protection dégradée en silence : sans drapeaux connus (cible autre que Linux x86_64/aarch64), on refuse.
    if !VERROU_PRIS_EN_CHARGE {
        return Err(Erreur::Validation(
            "verrou d'expérience non pris en charge sur cette cible (O_NOFOLLOW/O_NONBLOCK inconnus) : écriture refusée".into(),
        ));
    }
    let chemin = racine.join(".EXPERIENCE.lock");
    let refus = |raison: &str| {
        Erreur::Validation(format!("{} : verrou refusé ({raison})", chemin.display()))
    };
    let ouvrir = |creer: bool| {
        let mut o = OpenOptions::new();
        o.write(true);
        if creer {
            // O_CREAT|O_EXCL ne suit jamais un lien symbolique, même pendant.
            o.create_new(true);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            o.custom_flags(DRAPEAUX_VERROU);
        }
        o.open(&chemin)
    };
    let f = match ouvrir(true) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            ouvrir(false).map_err(|e| refus(&format!("ouverture sans suivi de lien : {e}")))?
        }
        Err(e) => return Err(Erreur::io(format!("ouverture de {}", chemin.display()), e)),
    };
    let meta = f
        .metadata()
        .map_err(|e| Erreur::io(format!("lecture de {}", chemin.display()), e))?;
    if !meta.is_file() {
        return Err(refus("pas un fichier régulier"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let nom = fs::symlink_metadata(&chemin).map_err(|_| refus("chemin disparu"))?;
        if nom.file_type().is_symlink() || nom.dev() != meta.dev() || nom.ino() != meta.ino() {
            return Err(refus("le chemin n'est plus le fichier ouvert"));
        }
    }
    f.lock()
        .map_err(|e| Erreur::io(format!("verrou de {}", chemin.display()), e))?;
    Ok(Verrou(f))
}

/// Empreinte canonique d'une entrée JSON (clés triées, forme compacte) : maillon de la chaîne du verdict.
pub(crate) fn empreinte_json(v: &Value) -> String {
    crate::regen::manifeste::sha_octets(&serde_json::to_vec(v).unwrap_or_default())
}

/// Avertissements qui ne rendent pas le manifeste rouge mais interdisent un « vert » plein :
/// un gel DÉCLARÉ n'a pas été observé sur un fichier.
pub(crate) fn avertissements(v: &Value) -> Vec<(u8, String)> {
    let mut sortie: Vec<(u8, String)> = tableau(v, "prereg")
        .iter()
        .filter(|p| texte(p, "gele_source") == "declare")
        .map(|p| {
            (
                4,
                format!(
                    "gel déclaré (non observé) pour {} : la date {} n'est pas celle d'un fichier",
                    texte(p, "id"),
                    texte(p, "gele_le")
                ),
            )
        })
        .collect();
    if v["horodatage"]["horloge_imposee"] == true {
        sortie.push((
            4,
            "dates écrites sous horloge imposée (KVU_EXPERIENCE_HORLOGE) : non probantes".into(),
        ));
    }
    // Provenance de la migration du chaînage : permanente, l'historique antérieur n'est pas attesté.
    if let Some(date) = v["verdict"].get("migre_le").and_then(Value::as_str) {
        sortie.push((
            11,
            format!(
                "historique du verdict chaîné par migration le {date} : l'historique antérieur n'est pas attesté"
            ),
        ));
    } else if v["verdict"].get("chaine").is_none() {
        sortie.push((
            11,
            "historique du verdict non chaîné (format antérieur) : toute commande d'écriture le chaîne à partir de l'état présent".into(),
        ));
    }
    sortie
}

/// Convertit une date civile avec secondes, suffixe Z ou décalage ±hh:mm.
/// Les instants représentables vont de 1970 à 9999, à la seconde UTC.
pub fn secondes_unix(iso: &str) -> Resultat<u64> {
    let invalide = || Erreur::Validation(format!("date ISO invalide : {iso}"));
    if !iso.is_ascii() || !matches!(iso.len(), 20 | 25) {
        return Err(invalide());
    }
    let b = iso.as_bytes();
    if b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return Err(invalide());
    }
    let nombre = |a: usize, z: usize| -> Resultat<i64> {
        let s = &iso[a..z];
        if !s.bytes().all(|c| c.is_ascii_digit()) {
            return Err(invalide());
        }
        s.parse().map_err(|_| invalide())
    };
    let (annee, mois, jour) = (nombre(0, 4)?, nombre(5, 7)?, nombre(8, 10)?);
    let (heure, minute, seconde) = (nombre(11, 13)?, nombre(14, 16)?, nombre(17, 19)?);
    let annee_civile = annee as u32;
    let bissextile = annee_civile.is_multiple_of(4)
        && (!annee_civile.is_multiple_of(100) || annee_civile.is_multiple_of(400));
    let jours_mois = match mois {
        4 | 6 | 9 | 11 => 30,
        2 if bissextile => 29,
        2 => 28,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => return Err(invalide()),
    };
    if annee == 0 || !(1..=jours_mois).contains(&jour) || heure > 23 || minute > 59 || seconde > 59
    {
        return Err(invalide());
    }
    let decalage = if iso.len() == 20 {
        if b[19] != b'Z' {
            return Err(invalide());
        }
        0
    } else {
        if !matches!(b[19], b'+' | b'-') || b[22] != b':' {
            return Err(invalide());
        }
        let (h, m) = (nombre(20, 22)?, nombre(23, 25)?);
        if h > 23 || m > 59 {
            return Err(invalide());
        }
        (h * 3600 + m * 60) * if b[19] == b'+' { 1 } else { -1 }
    };
    let a = annee - i64::from(mois <= 2);
    let ere = a.div_euclid(400);
    let ane = a - ere * 400;
    let mp = mois + if mois > 2 { -3 } else { 9 };
    let jda = (153 * mp + 2) / 5 + jour - 1;
    let jours = ere * 146_097 + ane * 365 + ane / 4 - ane / 100 + jda - 719_468;
    let secondes = jours * 86_400 + heure * 3600 + minute * 60 + seconde - decalage;
    u64::try_from(secondes)
        .ok()
        .filter(|s| *s <= DERNIERE_SECONDE)
        .ok_or_else(invalide)
}

fn normaliser_date(iso: &str) -> Resultat<String> {
    Ok(date_iso(secondes_unix(iso)?))
}

/// Vrai si l'horloge est IMPOSÉE par l'environnement (outil de test) : les dates écrites ne sont alors pas probantes.
pub(crate) fn horloge_imposee() -> bool {
    std::env::var_os("KVU_EXPERIENCE_HORLOGE").is_some()
}

/// Trace collante : une fois posé, `horodatage.horloge_imposee` n'est jamais remis à faux.
pub(crate) fn marquer_horloge(v: &mut Value) {
    if horloge_imposee() {
        v["horodatage"]["horloge_imposee"] = json!(true);
    }
}

fn horloge() -> Resultat<String> {
    match std::env::var("KVU_EXPERIENCE_HORLOGE") {
        Ok(iso) => normaliser_date(&iso),
        Err(std::env::VarError::NotPresent) => Ok(maintenant_iso()),
        Err(_) => Err(Erreur::Config(
            "KVU_EXPERIENCE_HORLOGE doit être une date UTF-8".into(),
        )),
    }
}

fn date_unix(valeur: &Value, champ: &str) -> Resultat<String> {
    if let Some(s) = valeur.as_u64().filter(|s| *s <= DERNIERE_SECONDE) {
        return Ok(date_iso(s));
    }
    let s = valeur
        .as_f64()
        .filter(|s| s.is_finite() && *s >= 0.0 && *s < (DERNIERE_SECONDE + 1) as f64)
        .ok_or_else(|| Erreur::Validation(format!("{champ} : secondes Unix attendues")))?;
    Ok(date_iso(s.floor() as u64))
}

fn lire_octets(p: &Path) -> Resultat<Vec<u8>> {
    fs::read(p).map_err(|e| Erreur::io(format!("lecture de {}", p.display()), e))
}

fn lire_texte(p: &Path) -> Resultat<String> {
    fs::read_to_string(p).map_err(|e| Erreur::io(format!("lecture de {}", p.display()), e))
}

fn lire_json(p: &Path) -> Resultat<Value> {
    serde_json::from_slice(&lire_octets(p)?)
        .map_err(|e| Erreur::Validation(format!("ROUGE V1 {} : {e}", p.display())))
}

fn json_octets(v: &Value) -> Resultat<Vec<u8>> {
    let mut s = serde_json::to_vec_pretty(v).map_err(|e| Erreur::Format(e.to_string()))?;
    s.push(b'\n');
    Ok(s)
}

/// Écrit un fichier NEUF sans jamais en écraser un existant, même apparu entre-temps : temporaire complet puis
/// `hard_link` (qui échoue si la cible existe), puis suppression du temporaire.
pub(crate) fn ecrire_sans_ecraser(p: &Path, octets: &[u8]) -> Resultat<()> {
    let nom = p
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| Erreur::Validation("nom de fichier non UTF-8".into()))?;
    let n = TEMPORAIRES.fetch_add(1, Ordering::Relaxed);
    let t = p.with_file_name(format!(".{nom}.{}.{n}.nouveau", std::process::id()));
    ecrire_atomique(&t, octets)?;
    let r = fs::hard_link(&t, p);
    let _ = fs::remove_file(&t);
    r.map_err(|e| {
        if e.kind() == std::io::ErrorKind::AlreadyExists {
            Erreur::Validation(format!("refus d'écraser {}", p.display()))
        } else {
            Erreur::io(format!("création de {}", p.display()), e)
        }
    })
}

fn ecrire_atomique(p: &Path, octets: &[u8]) -> Resultat<()> {
    let dossier = p
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let nom = p
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| Erreur::Validation("nom de fichier non UTF-8".into()))?;
    let (temporaire, mut fichier) = loop {
        let n = TEMPORAIRES.fetch_add(1, Ordering::Relaxed);
        let t = dossier.join(format!(".{nom}.{}.{n}.tmp", std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&t) {
            Ok(f) => break (t, f),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(Erreur::io(format!("création de {}", t.display()), e)),
        }
    };
    let resultat = (|| {
        fichier
            .write_all(octets)
            .map_err(|e| Erreur::io(format!("écriture de {}", temporaire.display()), e))?;
        fichier
            .sync_all()
            .map_err(|e| Erreur::io(format!("synchronisation de {}", temporaire.display()), e))?;
        drop(fichier);
        fs::rename(&temporaire, p)
            .map_err(|e| Erreur::io(format!("renommage vers {}", p.display()), e))
    })();
    if resultat.is_err() {
        let _ = fs::remove_file(&temporaire);
    }
    resultat
}

fn verifier_relatif(chemin: &str, point_admis: bool) -> Resultat<()> {
    if point_admis && chemin == "." {
        return Ok(());
    }
    if chemin.is_empty()
        || chemin.contains(['\\', ':', '\n', '\r'])
        || chemin
            .split('/')
            .any(|s| s.is_empty() || s == "." || s == "..")
        || Path::new(chemin)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(Erreur::Validation(format!(
            "chemin relatif interdit : {chemin}"
        )));
    }
    Ok(())
}

fn canonique(p: &Path) -> Resultat<PathBuf> {
    fs::canonicalize(p).map_err(|e| Erreur::io(format!("résolution de {}", p.display()), e))
}

fn relatif(racine: &Path, p: &Path) -> Resultat<String> {
    let racine = canonique(racine)?;
    if p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(Erreur::Validation(format!(
            "chemin contenant .. : {}",
            p.display()
        )));
    }
    let p = if p.is_absolute() {
        p.to_path_buf()
    } else {
        racine.join(p)
    };
    let p = canonique(&p)?;
    let rel = p
        .strip_prefix(&racine)
        .map_err(|_| Erreur::Validation(format!("fichier hors de la racine : {}", p.display())))?;
    let rel = rel
        .to_str()
        .ok_or_else(|| Erreur::Validation("chemin non UTF-8".into()))?;
    verifier_relatif(rel, false)?;
    Ok(rel.to_owned())
}

fn resoudre(racine: &Path, chemin: &str) -> Resultat<PathBuf> {
    verifier_relatif(chemin, false)?;
    let p = racine.join(chemin);
    relatif(racine, &p)?;
    Ok(p)
}

fn piece(racine: &Path, chemin: &str) -> Resultat<Value> {
    let chemin = relatif(racine, Path::new(chemin))?;
    Ok(json!({
        "sha256": sha_fichier(&racine.join(&chemin))?,
        "chemin": chemin
    }))
}

fn texte<'a>(v: &'a Value, cle: &str) -> &'a str {
    v.get(cle).and_then(Value::as_str).unwrap_or("")
}

fn tableau<'a>(v: &'a Value, cle: &str) -> &'a [Value] {
    v.get(cle)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

fn liste_mut<'a>(v: &'a mut Value, cle: &str) -> Resultat<&'a mut Vec<Value>> {
    v.get_mut(cle)
        .and_then(Value::as_array_mut)
        .ok_or_else(|| Erreur::Validation(format!("{cle} : tableau attendu")))
}

fn identifiant(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        && !matches!(id, "." | "..")
}

fn prefixe(type_carnet: &str) -> Option<&'static str> {
    match type_carnet {
        "resultat" => Some("RES-"),
        "erreur" => Some("ERR-"),
        "decision" => Some("DEC-"),
        "hypothese" => Some("HYP-"),
        "piste" => Some("PIS-"),
        "fait" => Some("FAI-"),
        _ => None,
    }
}

fn statuts_carnet(type_carnet: &str) -> &'static [&'static str] {
    match type_carnet {
        "resultat" => &[
            "numerique",
            "proposition_une_famille",
            "retrograde",
            "echec",
        ],
        "fait" => &["lecture", "numerique", "une_famille"],
        "erreur" => &[
            "a_corriger",
            "corrigee",
            "garde_fabriquee",
            "sans_consequence",
        ],
        "decision" => &["prise", "en_attente"],
        "hypothese" => &["ouverte", "conditionnelle", "en_test", "refutee", "retiree"],
        "piste" => &["ouverte", "bornee", "close", "gelee"],
        _ => &[],
    }
}

/// Une source avec une ligne ou une plage inclusive, toujours relative à l'expérience.
#[derive(Debug, PartialEq, Eq)]
pub struct SourceLignes {
    pub chemin: String,
    pub debut: usize,
    pub fin: usize,
}

pub fn analyser_source(s: &str) -> Resultat<SourceLignes> {
    let invalide =
        || Erreur::Validation(format!("source attendue chemin:ligne ou chemin:a-b : {s}"));
    let (chemin, plage) = s.rsplit_once(':').ok_or_else(invalide)?;
    verifier_relatif(chemin, false)?;
    let (a, b) = plage.split_once('-').unwrap_or((plage, plage));
    if a.is_empty()
        || b.is_empty()
        || !a.bytes().all(|c| c.is_ascii_digit())
        || !b.bytes().all(|c| c.is_ascii_digit())
    {
        return Err(invalide());
    }
    let (debut, fin) = (
        a.parse::<usize>().map_err(|_| invalide())?,
        b.parse::<usize>().map_err(|_| invalide())?,
    );
    if debut == 0 || fin < debut {
        return Err(invalide());
    }
    Ok(SourceLignes {
        chemin: chemin.into(),
        debut,
        fin,
    })
}

fn verifier_source(racine: &Path, s: &str) -> Resultat<SourceLignes> {
    let source = analyser_source(s)?;
    let octets = lire_octets(&resoudre(racine, &source.chemin)?)?;
    let lignes = octets.iter().filter(|b| **b == b'\n').count()
        + usize::from(!octets.is_empty() && octets.last() != Some(&b'\n'));
    if source.fin > lignes {
        return Err(Erreur::Validation(format!(
            "{s} : ligne hors bornes (1..{lignes})"
        )));
    }
    Ok(source)
}

#[derive(Debug, PartialEq)]
pub struct ArgumentsRun {
    pub graines: Vec<u64>,
    pub geometrie: Value,
    pub parametres: Map<String, Value>,
}

/// Lit les options longues d'argv sans interpréter une commande ni la lancer.
pub fn analyser_argv(argv: &[String]) -> Resultat<ArgumentsRun> {
    let mut graines: Vec<u64> = Vec::new();
    let mut parametres = Map::new();
    let mut brut: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    let mut dimensions = Map::new();
    let mut i = 1;
    while i < argv.len() {
        let Some(option) = argv[i].strip_prefix("--") else {
            i += 1;
            continue;
        };
        if option.is_empty() {
            break;
        }
        let (cle, valeur) = if let Some((cle, valeur)) = option.split_once('=') {
            (cle, Some(valeur))
        } else if argv.get(i + 1).is_some_and(|s| !s.starts_with("--")) {
            i += 1;
            (option, Some(argv[i].as_str()))
        } else {
            (option, None)
        };
        match cle {
            "graine" | "seed" => {
                let s = valeur.ok_or_else(|| Erreur::Validation("--graine sans valeur".into()))?;
                // Même analyse de liste que `su2 gen` (virgules, espaces autour des éléments).
                graines.extend(
                    crate::su2::cli::liste::<u64>(s, "graine")
                        .map_err(|e| Erreur::Validation(format!("graine invalide : {e}")))?,
                );
            }
            "L" | "Lt" | "Ls" => {
                let n = valeur
                    .and_then(|s| s.parse::<u64>().ok())
                    .filter(|n| *n > 0)
                    .ok_or_else(|| {
                        Erreur::Validation(format!("--{cle} : côté entier positif attendu"))
                    })?;
                dimensions.insert(cle.into(), json!(n));
            }
            "outdir" | "format" | "dump-every" | "zstd" => {}
            _ => {
                if let Some(s) = valeur {
                    brut.insert(cle.to_owned(), s.to_owned());
                }
                let v = match valeur {
                    Some(s) => serde_json::from_str::<Value>(s)
                        .ok()
                        .filter(Value::is_number)
                        .unwrap_or_else(|| json!(s)),
                    None => json!(true),
                };
                parametres.insert(cle.into(), v);
            }
        }
        i += 1;
    }
    // `su2 gen` : graine par défaut 1 et chaînes du lot résolues par la fonction MÊME de `su2 gen`
    // (`su2::cli::chaines_depuis_listes`, mêmes conversions, validations et règles). Pour toute autre
    // commande, une graine non lue reste une absence (V6 la refuse).
    if argv.get(1).map(String::as_str) == Some("su2")
        && argv.get(2).map(String::as_str) == Some("gen")
    {
        if graines.is_empty() {
            graines.push(1);
        }
        let betas: Vec<f64> = match brut.get("beta") {
            Some(s) => s
                .split(',')
                .map(|b| {
                    b.trim()
                        .parse::<f64>()
                        .map_err(|_| Erreur::Validation(format!("beta invalide : {b}")))
                })
                .collect::<Resultat<_>>()?,
            None => vec![2.3],
        };
        let entier = |cle: &str| -> Resultat<Option<u64>> {
            brut.get(cle)
                .map(|s| {
                    s.trim()
                        .parse::<u64>()
                        .map_err(|_| Erreur::Validation(format!("--{cle} : entier attendu : {s}")))
                })
                .transpose()
        };
        let k = entier("chaines")?;
        // Borne sur le nombre EFFECTIF de chaînes, y compris celui déduit des listes (β, graines).
        let effectif = k.unwrap_or(betas.len().max(graines.len()) as u64);
        if effectif > MAX_CHAINES {
            return Err(Erreur::Validation(format!(
                "{effectif} chaînes dépassent {MAX_CHAINES} : refusé (expansion bornée)"
            )));
        }
        let pas = entier("graine-pas")?;
        let k = k.map(|k| k as usize);
        let lot = crate::su2::cli::chaines_depuis_listes(
            &betas,
            &graines,
            k,
            pas,
            crate::commun::Depart::Chaud,
        )
        .map_err(|e| Erreur::Validation(format!("graines de su2 gen : {e}")))?;
        graines = lot.iter().map(|c| c.graine.0).collect();
    }
    let cotes = if dimensions.contains_key("Lt") || dimensions.contains_key("Ls") {
        let lt = dimensions
            .get("Lt")
            .ok_or_else(|| Erreur::Validation("--Lt manque pour --Ls".into()))?;
        let ls = dimensions
            .get("Ls")
            .ok_or_else(|| Erreur::Validation("--Ls manque pour --Lt".into()))?;
        vec![lt.clone(), ls.clone(), ls.clone(), ls.clone()]
    } else if let Some(l) = dimensions.get("L") {
        vec![l.clone(); 4]
    } else {
        Vec::new()
    };
    Ok(ArgumentsRun {
        graines,
        geometrie: json!({"cotes": cotes}),
        parametres,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_aller_retour_et_decalages() {
        for s in [0, 951_782_400, 1_761_413_130, DERNIERE_SECONDE] {
            assert_eq!(secondes_unix(&date_iso(s)).unwrap(), s);
        }
        assert_eq!(secondes_unix("1970-01-01T01:00:00+01:00").unwrap(), 0);
        assert_eq!(
            normaliser_date("2000-02-28T23:00:00-01:00").unwrap(),
            "2000-02-29T00:00:00Z"
        );
        for s in [
            "1900-02-29T00:00:00Z",
            "2100-02-29T00:00:00Z",
            "2026-02-30T00:00:00Z",
            "2026-01-01T24:00:00Z",
            "2026-01-01T00:00:00+24:00",
            "é",
            "1969-12-31T23:59:59Z",
        ] {
            assert!(secondes_unix(s).is_err(), "{s}");
        }
    }

    #[test]
    fn plages_inclusives_et_chemins() {
        assert_eq!(
            analyser_source("notes/a.md:2-4").unwrap(),
            SourceLignes {
                chemin: "notes/a.md".into(),
                debut: 2,
                fin: 4,
            }
        );
        assert_eq!(analyser_source("a:1").unwrap().fin, 1);
        for s in ["a:0", "a:4-2", "a:1-", "../a:1", "a:1-2-3", "a:-1"] {
            assert!(analyser_source(s).is_err(), "{s}");
        }
    }

    #[test]
    fn verrou_pris_en_charge_sur_les_cibles_connues() {
        // Linux x86_64 / aarch64 : drapeaux connus ; ailleurs `verrouiller` refuse (erreur claire).
        let connue = cfg!(all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ));
        assert_eq!(VERROU_PRIS_EN_CHARGE, connue);
    }

    #[test]
    fn graines_figees_a_la_frontiere_de_l_arithmetique_enveloppante_de_la_production() {
        let a = |v: &[&str]| {
            analyser_argv(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>())
                .unwrap()
                .graines
        };
        // multiplication enveloppante : 2·pas = 2 (mod 2^64)
        assert_eq!(
            a(&[
                "kvu",
                "su2",
                "gen",
                "--graine",
                "1",
                "--chaines",
                "3",
                "--graine-pas",
                "9223372036854775809"
            ]),
            [1, 9_223_372_036_854_775_810, 3]
        );
        // addition enveloppante au sommet de u64
        assert_eq!(
            a(&[
                "kvu",
                "su2",
                "gen",
                "--graine",
                "18446744073709551615",
                "--chaines",
                "2",
                "--graine-pas",
                "1"
            ]),
            [u64::MAX, 0]
        );
        assert_eq!(
            a(&[
                "kvu",
                "su2",
                "gen",
                "--graine",
                "10",
                "--chaines",
                "3",
                "--graine-pas",
                "5"
            ]),
            [10, 15, 20]
        );
    }

    #[test]
    fn graine_effective_su2_gen_defaut_alias_et_lot() {
        let a = |v: &[&str]| analyser_argv(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(a(&["kvu", "su2", "gen", "--L", "8"]).unwrap().graines, [1]);
        assert_eq!(
            a(&["kvu", "su2", "gen", "--seed", "7"]).unwrap().graines,
            [7]
        );
        assert_eq!(
            a(&[
                "kvu",
                "su2",
                "gen",
                "--graine",
                "10",
                "--chaines",
                "3",
                "--graine-pas",
                "5"
            ])
            .unwrap()
            .graines,
            [10, 15, 20]
        );
        assert!(a(&["kvu", "su2", "gen", "--graine", "10", "--chaines", "3"]).is_err());
        assert!(a(&["kvu", "rg", "correlateur"]).unwrap().graines.is_empty());
    }

    #[test]
    fn arguments_numeriques_geometrie_et_exclusions() {
        let argv = [
            "kvu",
            "su2",
            "gen",
            "--L",
            "8",
            "--beta=2.3",
            "--graine",
            "9",
            "--outdir",
            "sortie",
            "--zstd",
            "--therm",
            "10",
            "--alpha",
            "-0.5",
        ];
        let r = analyser_argv(&argv.map(str::to_owned)).unwrap();
        assert_eq!(r.graines, [9]);
        assert_eq!(r.geometrie, json!({"cotes": [8, 8, 8, 8]}));
        assert_eq!(r.parametres["beta"], json!(2.3));
        assert_eq!(r.parametres["alpha"], json!(-0.5));
        assert!(!r.parametres.contains_key("outdir"));
        let r =
            analyser_argv(&["kvu", "--Lt=4", "--Ls=8", "--graine=1,2"].map(str::to_owned)).unwrap();
        assert_eq!(r.geometrie["cotes"], json!([4, 8, 8, 8]));
        assert_eq!(r.graines, [1, 2]);
        assert!(analyser_argv(&["kvu", "--Lt=4"].map(str::to_owned)).is_err());
    }
}
