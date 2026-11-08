// SPDX-License-Identifier: Apache-2.0
//! Adaptateurs : les manifestes existants sont lus TELS QUELS (jamais modifiés) et traduits en `kvu-regen/1`.
//!
//! - `su2_hip/1` (`engine = "su2_hip"`, ex. `wilson_gpu_grandL_20260926/ensembles/production/*/manifest.json`) : pas de
//!   `binaire_sha256` dans le manifeste ; l'adaptateur fournit l'ancre de la publication (A) `d62ff7d2…`
//!   ([`ANCRE_HIP_BINAIRE`], `tests/anchors.sha256`, ligne `hip binary`) avec `binaire_provenance = "ancre_publication"` ;
//!   options numériques = `compat-prod` ; `source_hash` = sha256 de `su2_hip.cpp` ; pas de hachage des fichiers dans le
//!   manifeste (calculé sur disque à la demande : `fichiers_hachage = "disque"`).
//! - `su2_gen/1` (`engine = "kevinotron su2_gen (lot K1)"`, ex. `ym4_mass_gap_0510/ens/*/manifest.json`) : `argv` complet,
//!   `binaire_sha256`, `source_hash`, `sorties_sha256` + `csv_sha256` (donc `fichiers_hachage = "manifeste"`) ; options
//!   numériques `ref-k1` (FMA interdite : `-ffp-contract=off`). L'`argv` historique (`su2_gen --hip --seed …`) est traduit en
//!   ligne de commande `kvu su2 gen` ; l'original est gardé dans `commande_origine`.
//!
//! La machine de production est INFÉRÉE de l'architecture GPU : gfx1151 ⇒ profil de la machine principale,
//! sm_120 ⇒ profil de la machine secondaire ; champ `inferee: true`.

use super::classer::{classer, Classe, Contexte};
use super::manifeste::{est_sha256, parcourir, sha_fichier, sha_octets, SCHEMA_REGEN};
use crate::commun::{Erreur, Resultat, NOM_PROFIL_ACEMAGIC, NOM_PROFIL_GAMER, NOM_PROFIL_INCONNUE};
use serde_json::{json, Map, Value};
use std::path::Path;

/// Version des adaptateurs (consignée dans l'origine de chaque manifeste produit).
pub const NOM_ADAPTATEUR: &str = "kvu-regen-adaptateurs/1";

/// sha256 du binaire `su2_hip` de la publication (A) : ligne `hip binary` de `A/tests/anchors.sha256`.
pub const ANCRE_HIP_BINAIRE: &str = "d62ff7d2a9cfc861e8c02faf4b1e284aaed7da8a2b9cd42f7aec14d335af93cf";

/// Formes de manifestes reconnues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    SuHip,
    SuGen,
    Regen,
}

impl Source {
    pub fn nom(self) -> &'static str {
        match self {
            Source::SuHip => "su2_hip/1",
            Source::SuGen => "su2_gen/1",
            Source::Regen => "kvu-regen/1",
        }
    }
}

/// Faut-il hacher les fichiers de la chaîne sur disque quand le manifeste ne porte pas leurs sha256 ?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hachage {
    /// Lit chaque fichier (rapide pour un L = 12 : 100 Mo ; long pour un ensemble entier).
    Calculer,
    /// `sha256 = null`, `fichiers_hachage = "non_calcule"` (classement/audit seulement ; `regen` refuse).
    Omettre,
}

/// Reconnaît la forme d'un manifeste JSON (sans rien lire d'autre).
pub fn reconnaitre(m: &Value) -> Option<Source> {
    if m.get("schema").and_then(Value::as_str) == Some(SCHEMA_REGEN) {
        return Some(Source::Regen);
    }
    let engine = m.get("engine").and_then(Value::as_str)?;
    if engine == "su2_hip" {
        Some(Source::SuHip)
    } else if engine.starts_with("kevinotron su2_gen") {
        Some(Source::SuGen)
    } else {
        None
    }
}

fn fmt_err(m: impl Into<String>) -> Erreur {
    Erreur::Format(m.into())
}

fn champ_u64(m: &Value, k: &str) -> Resultat<u64> {
    m.get(k).and_then(Value::as_u64).ok_or_else(|| fmt_err(format!("champ « {k} » absent ou non entier")))
}

fn champ_f64(m: &Value, k: &str) -> Resultat<f64> {
    m.get(k).and_then(Value::as_f64).ok_or_else(|| fmt_err(format!("champ « {k} » absent ou non numérique")))
}

fn champ_txt<'a>(m: &'a Value, k: &str) -> Resultat<&'a str> {
    m.get(k).and_then(Value::as_str).ok_or_else(|| fmt_err(format!("champ « {k} » absent ou non textuel")))
}

/// Premier mot de `texte` qui commence par « gfx » ou « sm_ » (éventuellement derrière `=`), lettres/chiffres/`_`.
pub fn trouver_arch(texte: &str) -> Option<String> {
    for mot in texte.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        if (mot.starts_with("gfx") && mot.len() > 3) || (mot.starts_with("sm_") && mot.len() > 3) {
            return Some(mot.to_string());
        }
    }
    None
}

/// Machine de production déduite de l'architecture GPU.
/// table historique des manifestes existants (architecture -> profil) ; noms issus des variantes de Machine (commun)
pub fn machine_depuis_arch(arch: Option<&str>) -> &'static str {
    match arch {
        Some("gfx1151") => NOM_PROFIL_ACEMAGIC,
        Some("sm_120") => NOM_PROFIL_GAMER,
        _ => NOM_PROFIL_INCONNUE,
    }
}

fn backend_depuis_texte(texte: &str) -> Resultat<&'static str> {
    let premier = texte.split_whitespace().next().unwrap_or("").to_ascii_lowercase();
    match premier.as_str() {
        "hip" => Ok("hip"),
        "cuda" => Ok("cuda"),
        autre => Err(fmt_err(format!("backend « {autre} » non reconnu dans « {texte} »"))),
    }
}

fn depart(m: &Value) -> Resultat<&'static str> {
    match champ_txt(m, "start")?.to_ascii_lowercase().as_str() {
        "hot" => Ok("hot"),
        "cold" => Ok("cold"),
        autre => Err(fmt_err(format!("start « {autre} » inconnu"))),
    }
}

/// Fichiers de données d'un répertoire de chaîne (NPY, CSV, zst), triés, avec sha256 selon `hachage`.
fn fichiers_disque(rep: &Path, hachage: Hachage) -> Resultat<Vec<Value>> {
    let mut v = Vec::new();
    for f in parcourir(rep)? {
        if f.relatif.contains('/') {
            continue;
        }
        let donnees = [".npy", ".csv", ".zst"].iter().any(|e| f.relatif.ends_with(e));
        if !donnees {
            continue;
        }
        let sha = match hachage {
            Hachage::Calculer => Value::String(sha_fichier(&f.chemin)?),
            Hachage::Omettre => Value::Null,
        };
        v.push(json!({"nom": f.relatif, "sha256": sha, "octets": f.octets}));
    }
    Ok(v)
}

fn statut(m: &Value, rep: &Path) -> String {
    match m.get("statut").and_then(Value::as_str) {
        Some(s) => s.to_string(),
        None if rep.join("COMPLETE").is_file() => "COMPLETE".to_string(),
        None => "INCOMPLETE".to_string(),
    }
}

/// Lit le manifeste `manifeste`, le reconnaît, le traduit en `kvu-regen/1`, le classe (A/B/C) avec `ctx`.
/// Aucune écriture. Un manifeste non reconnu ou non traduisible ⇒ `Err` (le classement le range alors en C).
pub fn adapter(manifeste: &Path, ctx: &mut Contexte, hachage: Hachage) -> Resultat<Value> {
    let octets = std::fs::read(manifeste).map_err(|e| Erreur::io(format!("lecture de {}", manifeste.display()), e))?;
    let m: Value = serde_json::from_slice(&octets).map_err(|e| fmt_err(format!("{} : JSON invalide : {e}", manifeste.display())))?;
    let rep = manifeste.parent().unwrap_or_else(|| Path::new("."));
    let sha_m = sha_octets(&octets);
    let mut out = match reconnaitre(&m) {
        Some(Source::Regen) => return Ok(m),
        Some(Source::SuHip) => adapter_hip(&m, rep, hachage)?,
        Some(Source::SuGen) => adapter_gen(&m, rep)?,
        None => return Err(fmt_err(format!("{} : manifeste non reconnu (ni su2_hip/1 ni su2_gen/1 ni kvu-regen/1)", manifeste.display()))),
    };
    let source = reconnaitre(&m).map_or("?", Source::nom);
    out["origine"] = json!({
        "schema": source,
        "manifeste": manifeste.to_string_lossy(),
        "sha256_manifeste": sha_m,
        "adaptateur": NOM_ADAPTATEUR,
    });
    let cl = classer(&out, ctx);
    out["classe_regen"] = Value::String(cl.classe.nom().to_string());
    out["raisons_classe"] = json!(cl.raisons);
    Ok(out)
}

fn adapter_hip(m: &Value, rep: &Path, hachage: Hachage) -> Resultat<Value> {
    let l = champ_u64(m, "l")?;
    let beta = champ_f64(m, "beta")?;
    let graine = champ_u64(m, "seed")?;
    let therm = champ_u64(m, "therm_steps")?;
    let nmeas = champ_u64(m, "measurements")?;
    let skip = champ_u64(m, "skip_steps")?;
    let nor = champ_u64(m, "or_per_step")?;
    let dump = champ_u64(m, "dump_every")?;
    let start = depart(m)?;
    let source_hash = champ_txt(m, "source_hash")?;
    if !est_sha256(source_hash) {
        return Err(fmt_err("source_hash n'est pas un sha256"));
    }
    let texte_backend = champ_txt(m, "backend")?;
    let backend = backend_depuis_texte(texte_backend)?;
    let arch = trouver_arch(texte_backend);
    let precision = if texte_backend.contains("FP64") { "f64" } else { return Err(fmt_err("précision non FP64 dans « backend »")) };
    let fichiers = fichiers_disque(rep, hachage)?;
    let commande: Vec<String> = [
        "su2", "gen", "--backend", backend, "--precision", precision, "--options", "compat-prod", "--L", &l.to_string(),
        "--beta", &beta.to_string(), "--graine", &graine.to_string(), "--start", start, "--therm", &therm.to_string(),
        "--nmeas", &nmeas.to_string(), "--skip", &skip.to_string(), "--nor", &nor.to_string(), "--dump-every", &dump.to_string(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    Ok(json!({
        "schema": SCHEMA_REGEN,
        "binaire_sha256": ANCRE_HIP_BINAIRE,
        "binaire_provenance": "ancre_publication",
        "git_head": "non_enregistre",
        "source_hash": source_hash,
        "noyaux_embarques": [{"nom": "su2_hip.cpp", "sha256": source_hash}],
        "modules_compiles": [],
        "backend": backend,
        "arch": arch,
        "versions": {},
        "machine": {"nom": machine_depuis_arch(arch.as_deref()), "cpu": null, "gpu": arch, "ram": null, "inferee": true},
        "graines": {"cle": graine, "etiquettes": [champ_txt(m, "rng")?]},
        "parametres": {
            "groupe": "SU(2)", "L": l, "beta": beta, "start": start, "therm": therm, "nmeas": nmeas, "skip": skip, "nor": nor,
            "precision": precision, "options": "compat-prod", "dump_every": dump, "bord": false
        },
        "commande": commande,
        "entrees": [],
        "fichiers": fichiers,
        "fichiers_hachage": if hachage == Hachage::Calculer { "disque" } else { "non_calcule" },
        "duree_s": null,
        "statut": statut(m, rep),
        "regen_verifiee": {"date": null, "machine": null, "resultat": "non_verifiee"},
    }))
}

/// Drapeaux de l'`argv` historique de `su2_gen` → drapeaux de `kvu su2 gen`.
fn traduire_argv_gen(argv: &[String]) -> Resultat<(Vec<String>, Map<String, Value>)> {
    let mut cmd: Vec<String> = Vec::new();
    let mut vu = Map::new();
    let mut i = 1; // argv[0] = binaire
    while i < argv.len() {
        let a = argv[i].as_str();
        match a {
            "--hip" => {
                vu.insert("hip".into(), Value::Bool(true));
                i += 1;
            }
            "--L" | "--beta" | "--seed" | "--start" | "--therm" | "--nmeas" | "--skip" | "--nor" | "--dump-every" | "--outdir" => {
                let v = argv.get(i + 1).ok_or_else(|| fmt_err(format!("argv : « {a} » sans valeur")))?;
                vu.insert(a.trim_start_matches("--").to_string(), Value::String(v.clone()));
                i += 2;
            }
            "--dump-initial" => {
                vu.insert("dump-initial".into(), Value::Bool(true));
                i += 1;
            }
            autre => return Err(fmt_err(format!("argv : argument « {autre} » sans équivalent kvu (non traduisible)"))),
        }
    }
    if vu.get("hip") != Some(&Value::Bool(true)) {
        return Err(fmt_err("argv sans --hip : backend non traduisible"));
    }
    let g = |k: &str| -> Resultat<String> {
        vu.get(k).and_then(Value::as_str).map(str::to_string).ok_or_else(|| fmt_err(format!("argv sans --{k}")))
    };
    for (k, v) in [("--backend", "hip".to_string()), ("--precision", "f64".to_string()), ("--options", "ref-k1".to_string())] {
        cmd.push(k.to_string());
        cmd.push(v);
    }
    for (kvu, hist) in [("--L", "L"), ("--beta", "beta"), ("--graine", "seed"), ("--start", "start"), ("--therm", "therm"),
        ("--nmeas", "nmeas"), ("--skip", "skip"), ("--nor", "nor"), ("--dump-every", "dump-every")] {
        cmd.push(kvu.to_string());
        cmd.push(g(hist)?);
    }
    if vu.get("dump-initial") == Some(&Value::Bool(true)) {
        cmd.push("--dump-initial".to_string());
    }
    let mut full = vec!["su2".to_string(), "gen".to_string()];
    full.extend(cmd);
    Ok((full, vu))
}

fn adapter_gen(m: &Value, rep: &Path) -> Resultat<Value> {
    let argv: Vec<String> = m
        .get("argv")
        .and_then(Value::as_array)
        .ok_or_else(|| fmt_err("champ « argv » absent"))?
        .iter()
        .map(|v| v.as_str().map(str::to_string).ok_or_else(|| fmt_err("argv : élément non textuel")))
        .collect::<Resultat<_>>()?;
    let (commande, vu) = traduire_argv_gen(&argv)?;
    let valeur = |k: &str| vu.get(k).and_then(Value::as_str).unwrap_or("");
    let l = champ_u64(m, "l")?;
    if valeur("L").parse::<u64>().ok() != Some(l) {
        return Err(fmt_err("argv et champ « l » incohérents"));
    }
    let graine = champ_u64(m, "seed")?;
    if valeur("seed").parse::<u64>().ok() != Some(graine) {
        return Err(fmt_err("argv et champ « seed » incohérents"));
    }
    let beta: f64 = valeur("beta").parse().map_err(|_| fmt_err("argv : --beta non numérique"))?;
    if (beta - champ_f64(m, "beta")?).abs() > 1e-12 {
        return Err(fmt_err("argv et champ « beta » incohérents"));
    }
    let start = depart(m)?;
    let source_hash = champ_txt(m, "source_hash")?;
    let binaire = champ_txt(m, "binaire_sha256")?;
    if !est_sha256(source_hash) || !est_sha256(binaire) {
        return Err(fmt_err("source_hash ou binaire_sha256 n'est pas un sha256"));
    }
    let texte_backend = champ_txt(m, "backend")?;
    let backend = backend_depuis_texte(texte_backend)?;
    let arch = trouver_arch(texte_backend);
    if !texte_backend.contains("-ffp-contract=off") || !texte_backend.contains("FP64") {
        return Err(fmt_err("options numériques non ref-k1 (FMA non interdite) ou précision non FP64"));
    }
    let options_compil = texte_backend
        .split_once("(options ")
        .and_then(|(_, r)| r.rsplit_once(')'))
        .map(|(o, _)| o.to_string())
        .unwrap_or_default();
    let mut noyaux: Vec<Value> = Vec::new();
    if let Some(src) = m.get("sources_sha256").and_then(Value::as_object) {
        for (nom, sha) in src {
            let sha = sha.as_str().filter(|s| est_sha256(s)).ok_or_else(|| fmt_err(format!("sources_sha256[{nom}] n'est pas un sha256")))?;
            noyaux.push(json!({"nom": nom, "sha256": sha}));
        }
    }
    let sha_noyau = |nom: &str| -> Option<String> {
        noyaux.iter().find(|n| n["nom"] == nom).and_then(|n| n["sha256"].as_str()).map(str::to_string)
    };
    let modules = match sha_noyau("src/hip/su2_sweep.hip") {
        Some(s) => json!([{
            "source_sha256": s, "options": options_compil,
            "arguments": options_compil.split_whitespace().collect::<Vec<_>>(), "objet_sha256": null
        }]),
        None => json!([]),
    };
    let mut fichiers: Vec<Value> = Vec::new();
    for (cle, tableau) in [("sorties_sha256", true), ("csv_sha256", true)] {
        let _ = tableau;
        let Some(o) = m.get(cle).and_then(Value::as_object) else {
            return Err(fmt_err(format!("champ « {cle} » absent")));
        };
        for (nom, sha) in o {
            let sha = sha.as_str().filter(|s| est_sha256(s)).ok_or_else(|| fmt_err(format!("{cle}[{nom}] n'est pas un sha256")))?;
            let meta = std::fs::metadata(rep.join(nom)).map_err(|e| Erreur::io(format!("fichier listé absent : {}", rep.join(nom).display()), e))?;
            fichiers.push(json!({"nom": nom, "sha256": sha, "octets": meta.len()}));
        }
    }
    fichiers.sort_by(|a, b| a["nom"].as_str().cmp(&b["nom"].as_str()));
    Ok(json!({
        "schema": SCHEMA_REGEN,
        "binaire_sha256": binaire,
        "binaire_provenance": "manifeste",
        "git_head": m.get("git_head").and_then(Value::as_str).unwrap_or("non_enregistre"),
        "source_hash": source_hash,
        "noyaux_embarques": noyaux,
        "modules_compiles": modules,
        "backend": backend,
        "arch": arch,
        "versions": {},
        "machine": {"nom": machine_depuis_arch(arch.as_deref()), "cpu": null, "gpu": arch, "ram": null, "inferee": true},
        "graines": {"cle": graine, "etiquettes": [m.get("rng").and_then(Value::as_str).unwrap_or("")]},
        "parametres": {
            "groupe": "SU(2)", "L": l, "beta": beta, "start": start,
            "therm": champ_u64(m, "therm_steps")?, "nmeas": champ_u64(m, "measurements")?,
            "skip": champ_u64(m, "skip_steps")?, "nor": champ_u64(m, "or_per_step")?,
            "precision": "f64", "options": "ref-k1", "dump_every": champ_u64(m, "dump_every")?,
            "bord": false
        },
        "commande": commande,
        "commande_origine": argv,
        "entrees": [],
        "fichiers": fichiers,
        "fichiers_hachage": "manifeste",
        "duree_s": m.get("wall_s").and_then(Value::as_f64),
        "statut": statut(m, rep),
        "regen_verifiee": {"date": null, "machine": null, "resultat": "non_verifiee"},
    }))
}

/// Classe l'état de l'adaptation d'un manifeste : `Ok(classe)` ou `C` avec la raison si le manifeste n'est pas traduisible.
pub fn adapter_ou_classer_c(manifeste: &Path, ctx: &mut Contexte, hachage: Hachage) -> (Option<Value>, Classe, Vec<String>) {
    match adapter(manifeste, ctx, hachage) {
        Ok(v) => {
            let classe = match v["classe_regen"].as_str() {
                Some("A") => Classe::A,
                Some("B") => Classe::B,
                _ => Classe::C,
            };
            let raisons = v["raisons_classe"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            (Some(v), classe, raisons)
        }
        Err(e) => (None, Classe::C, vec![format!("manifeste non adaptable : {e}")]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jouet_hip(dir: &Path) -> std::path::PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("cfg_000100.npy"), b"npy1").unwrap();
        std::fs::write(dir.join("measurements.csv"), b"a,b\n1,2\n").unwrap();
        std::fs::write(dir.join("therm.csv"), b"t\n").unwrap();
        std::fs::write(dir.join("COMPLETE"), b"ok").unwrap();
        let m = dir.join("manifest.json");
        std::fs::write(
            &m,
            br#"{"schema":1,"engine":"su2_hip","source_hash":"ed1e0fce255cb15d640ab603b3af0b6c50075bbb65252ce5d7e351abd3061e30","backend":"HIP gfx1151 FP64","rng":"Philox4x32-10","l":12,"beta":2.5,"seed":26095612011,"start":"Hot","therm_steps":5000,"measurements":2000,"skip_steps":1,"or_per_step":4,"dump_every":100,"statut":"COMPLETE"}"#,
        )
        .unwrap();
        m
    }

    fn tmp(nom: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("kvu_adapt_{nom}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn hip_adapte_est_valide_au_schema() {
        let d = tmp("hip");
        let m = jouet_hip(&d);
        let mut ctx = Contexte::nouveau();
        let v = adapter(&m, &mut ctx, Hachage::Calculer).unwrap();
        let erreurs = super::super::schema::valider_manifeste(&v);
        assert!(erreurs.is_empty(), "{erreurs:?}");
        assert_eq!(v["binaire_sha256"], ANCRE_HIP_BINAIRE);
        assert_eq!(v["machine"]["nom"], "acemagic");
        assert_eq!(v["fichiers"].as_array().unwrap().len(), 3);
        assert_eq!(v["fichiers_hachage"], "disque");
        assert_eq!(v["origine"]["schema"], "su2_hip/1");
        // la chaîne ancrée de la publication (L=12, β=2.5, graine 26095612011) est C même avec G-BIT verte
        let mut avec = Contexte::nouveau().avec_gbit("G-BIT-HIP-PROD");
        let v = adapter(&m, &mut avec, Hachage::Omettre).unwrap();
        assert_eq!(v["classe_regen"], "C");
        assert_eq!(v["fichiers_hachage"], "non_calcule");
        assert!(v["fichiers"][0]["sha256"].is_null());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn hip_hors_ancre_est_a_avec_preuve_et_c_sans() {
        let d = tmp("hip2");
        let m = jouet_hip(&d);
        let txt = std::fs::read_to_string(&m).unwrap().replace("26095612011", "5");
        std::fs::write(&m, txt).unwrap();
        let sans = adapter(&m, &mut Contexte::nouveau(), Hachage::Omettre).unwrap();
        assert_eq!(sans["classe_regen"], "C");
        let avec = adapter(&m, &mut Contexte::nouveau().avec_gbit("G-BIT-HIP-PROD"), Hachage::Omettre).unwrap();
        assert_eq!(avec["classe_regen"], "A", "{avec}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn argv_gen_traduit_et_refuse_l_inconnu() {
        let a: Vec<String> = "/b/su2_gen --hip --L 16 --beta 2.45 --seed 22 --start hot --therm 1000 --nmeas 100 --skip 10 --nor 4 --dump-every 1 --outdir /x"
            .split_whitespace().map(String::from).collect();
        let (cmd, _) = traduire_argv_gen(&a).unwrap();
        assert_eq!(
            cmd.join(" "),
            "su2 gen --backend hip --precision f64 --options ref-k1 --L 16 --beta 2.45 --graine 22 --start hot --therm 1000 --nmeas 100 --skip 10 --nor 4 --dump-every 1"
        );
        let mut b = a.clone();
        b.push("--mutant".into());
        b.push("metro".into());
        assert!(traduire_argv_gen(&b).is_err());
        let sans_hip: Vec<String> = a.iter().filter(|s| *s != "--hip").cloned().collect();
        assert!(traduire_argv_gen(&sans_hip).is_err());
    }

    #[test]
    fn arch_et_machine() {
        assert_eq!(trouver_arch("HIP gfx1151 FP64").as_deref(), Some("gfx1151"));
        assert_eq!(trouver_arch("HIP hiprtc --offload-arch=gfx1151 FP64 (options …)").as_deref(), Some("gfx1151"));
        assert_eq!(trouver_arch("CUDA sm_120 FP64").as_deref(), Some("sm_120"));
        assert_eq!(machine_depuis_arch(Some("gfx1151")), "acemagic");
        assert_eq!(machine_depuis_arch(Some("sm_120")), "gamer");
        assert_eq!(machine_depuis_arch(None), "inconnue");
    }

    #[test]
    fn manifeste_inconnu_est_classe_c() {
        let d = tmp("inconnu");
        std::fs::create_dir_all(&d).unwrap();
        let m = d.join("manifest.json");
        std::fs::write(&m, br#"{"schema":1,"engine":"autre","statut":"COMPLETE"}"#).unwrap();
        let (v, classe, raisons) = adapter_ou_classer_c(&m, &mut Contexte::nouveau().avec_gbit("G-BIT-HIP-PROD"), Hachage::Omettre);
        assert!(v.is_none());
        assert_eq!(classe, Classe::C);
        assert!(raisons[0].contains("non adaptable"));
        let _ = std::fs::remove_dir_all(&d);
    }
}
