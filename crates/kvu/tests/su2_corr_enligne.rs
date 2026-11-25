//! Portes E1–E5 et régressions de reprise de `corr` dans `su2 gen` (B08, c08c, 11/11/2026).
//! Les tests CPU appartiennent à la suite normale ; les tests GPU sont `#[ignore = "GPU"]`.
//! Un test GPU demandé échoue si le GPU manque. À lancer sous le verrou de la machine :
//!
//! ```text
//! acemagic : flock -w 7200 /tmp/kevinotron_gpu.lock \
//!            cargo test -p kvu --release --test su2_corr_enligne \
//!            -- --ignored --nocapture --test-threads=1
//! gamer    : flock -w 7200 /tmp/gamer_gpu.lock \
//!            cargo test -p kvu --release --features cuda --test su2_corr_enligne \
//!            -- --ignored --nocapture --test-threads=1
//! ```

use kvu::commun::{ChampHote, Depart, Erreur, Graine, Precision, Resultat};
use kvu::gpu::{Appareil, Arg, Lancement, OptionsCompil, ProvenanceCompil};
use kvu::heritage::manifest::sha256_fichier;
use kvu::machine::admission::BesoinMemoire;
use kvu::obs::gpu::ModuleObs;
use kvu::rg::corr_enligne::{NOM_CSV, NOM_JSON};
use kvu::rg::correlateur::nombre_de_valeurs;
use kvu::su2::run::{self, ConfigRun, MesureEnLigne, OptionsRun, NOM_FICHIER_MESURES};
use kvu::su2::{ChampGpu, ConfigLot, ParamChaine, TamponChamp};
use serde_json::{json, Value};
use std::cell::Cell;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::rc::Rc;

const SPEC_CPU: &str = "corr:brut+flot/0.02/3:1";
const SPEC_GPU: &str = "corr:brut+flot/0.02/50:1";

fn kvu(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kvu"))
        .args(args)
        .env("RAYON_NUM_THREADS", "2")
        // Le hors ligne GPU ouvre « auto » : lui donner le même backend qu'aux runs explicites.
        .env("KEVINOTRON_BACKEND", backend_gpu())
        .output()
        .expect("lancement de kvu")
}

fn dossier(nom: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("su2_corr_{nom}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("création du répertoire temporaire");
    dir
}

#[allow(clippy::too_many_arguments)]
fn gen(
    backend: &str,
    l: usize,
    therm: u64,
    nmeas: u64,
    skip: u64,
    dump: bool,
    sortie: &Path,
    extra: &[&str],
) -> Output {
    let l = l.to_string();
    let therm = therm.to_string();
    let nmeas = nmeas.to_string();
    let skip = skip.to_string();
    let mut args = vec![
        "su2",
        "gen",
        "--backend",
        backend,
        "--L",
        &l,
        "--beta",
        "2.3",
        "--therm",
        &therm,
        "--nmeas",
        &nmeas,
        "--skip",
        &skip,
        "--format",
        "q64",
        "--outdir",
        sortie.to_str().expect("chemin UTF-8 de sortie"),
    ];
    if !extra.contains(&"--graine") {
        args.extend_from_slice(&["--graine", "26111108"]);
    }
    if dump && !extra.contains(&"--dump-every") {
        args.extend_from_slice(&["--dump-every", "1"]);
    }
    args.extend_from_slice(extra);
    kvu(&args)
}

fn reprendre(backend: &str, dir: &Path, extra: &[&str]) -> Output {
    let mut args = vec![
        "su2",
        "gen",
        "--backend",
        backend,
        "--reprendre",
        dir.to_str().expect("chemin UTF-8 de reprise"),
    ];
    args.extend_from_slice(extra);
    kvu(&args)
}

fn reussi(sortie: &Output) {
    assert_eq!(
        sortie.status.code(),
        Some(0),
        "kvu en échec :\n{}\n{}",
        String::from_utf8_lossy(&sortie.stdout),
        String::from_utf8_lossy(&sortie.stderr),
    );
}

fn octets(p: &Path) -> Vec<u8> {
    fs::read(p).unwrap_or_else(|e| panic!("lecture de {} : {e}", p.display()))
}

fn lire_json(p: &Path) -> Value {
    serde_json::from_slice(&octets(p)).unwrap_or_else(|e| panic!("JSON de {} : {e}", p.display()))
}

fn fichiers_physiques(dir: &Path) -> Vec<String> {
    let mut noms: Vec<String> = fs::read_dir(dir)
        .expect("lecture du répertoire de run")
        .map(|e| {
            e.expect("entrée du répertoire")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|n| n.starts_with("cfg_") || n == "therm.csv" || n == "measurements.csv")
        .collect();
    noms.sort();
    noms
}

fn comparer_physiques(a: &Path, b: &Path, dumps_min: usize) -> usize {
    let noms = fichiers_physiques(a);
    assert_eq!(
        noms,
        fichiers_physiques(b),
        "listes des fichiers physiques différentes"
    );
    assert!(noms.iter().any(|n| n == "therm.csv"), "therm.csv absent");
    assert!(
        noms.iter().any(|n| n == "measurements.csv"),
        "measurements.csv absent"
    );
    let dumps = noms.iter().filter(|n| n.starts_with("cfg_")).count();
    assert!(
        dumps >= dumps_min,
        "{dumps} dumps au lieu d'au moins {dumps_min}"
    );
    for nom in &noms {
        assert_eq!(octets(&a.join(nom)), octets(&b.join(nom)), "{nom} diffère");
    }
    noms.len()
}

fn hors_ligne(dir: &Path, lissages: &str, mesure: &str) -> Value {
    let nom = dir
        .file_name()
        .expect("nom du répertoire de chaîne")
        .to_string_lossy();
    let sortie = dir
        .parent()
        .expect("parent du répertoire de chaîne")
        .join(format!("{nom}_hors_ligne_{mesure}.json"));
    let o = kvu(&[
        "rg",
        "correlateur",
        "--ens",
        dir.to_str().expect("chemin UTF-8 de chaîne"),
        "--lissage",
        lissages,
        "--rmax",
        "0",
        "--mesure",
        mesure,
        "--fils",
        "2",
        "--sortie",
        sortie.to_str().expect("chemin UTF-8 hors ligne"),
    ]);
    reussi(&o);
    lire_json(&sortie)
}

fn renommer(v: &mut Value, therm: u64, skip: u64) {
    for chaine in v["chaines"].as_array_mut().expect("liste des chaînes") {
        for cfg in chaine["configs"]
            .as_array_mut()
            .expect("liste des configurations")
        {
            let nom = cfg["cfg"].as_str().expect("nom de configuration");
            let indice = nom
                .strip_prefix("cfg_")
                .and_then(|s| {
                    s.strip_suffix(".npy.zst")
                        .or_else(|| s.strip_suffix(".npy"))
                })
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or_else(|| panic!("nom de configuration inattendu : {nom}"));
            assert!(
                nom == format!("cfg_{indice:06}.npy") || nom == format!("cfg_{indice:06}.npy.zst"),
                "nom de configuration inattendu : {nom}",
            );
            cfg["cfg"] = json!(format!("pas_{:06}", therm + indice * skip));
        }
    }
}

/// Même structure et mêmes textes ; les nombres sont comparés par les bits de leur f64.
fn memes_bits(a: &Value, b: &Value) -> Result<usize, String> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let x = x
                .as_f64()
                .ok_or_else(|| format!("nombre A illisible : {x}"))?;
            let y = y
                .as_f64()
                .ok_or_else(|| format!("nombre B illisible : {y}"))?;
            if x.to_bits() == y.to_bits() {
                Ok(1)
            } else {
                Err(format!(
                    "réels différents : {x} ({:016x}) et {y} ({:016x})",
                    x.to_bits(),
                    y.to_bits()
                ))
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            x.iter().zip(y).enumerate().try_fold(0, |n, (i, (a, b))| {
                memes_bits(a, b)
                    .map(|m| n + m)
                    .map_err(|e| format!("[{i}] : {e}"))
            })
        }
        (Value::Object(x), Value::Object(y)) if x.len() == y.len() => {
            x.iter().try_fold(0, |n, (cle, a)| {
                let b = y.get(cle).ok_or_else(|| format!("clé « {cle} » absente"))?;
                memes_bits(a, b)
                    .map(|m| n + m)
                    .map_err(|e| format!("{cle} : {e}"))
            })
        }
        (Value::Null, Value::Null) => Ok(0),
        (Value::Bool(x), Value::Bool(y)) if x == y => Ok(0),
        (Value::String(x), Value::String(y)) if x == y => Ok(0),
        _ => Err(format!("structure ou textes différents : {a} et {b}")),
    }
}

#[test]
fn comparaison_au_bit_distingue_zero_signe() {
    assert!(memes_bits(&json!(0.0), &json!(-0.0)).is_err());
    assert!(memes_bits(&json!({"a": []}), &json!({"b": []})).is_err());
    assert!(memes_bits(&json!([1]), &json!([1, 2])).is_err());
    let v = json!({"a": [0.0, -0.0, f64::from_bits(1)], "b": "texte"});
    assert_eq!(memes_bits(&v, &v).expect("comparaison réflexive"), 3);
    println!("\nPASS COMPARAISON-AU-BIT : 3 nombres, 3 différences détectées");
}

fn ecart_max(a: &Value, b: &Value) -> f64 {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let x = x.as_f64().expect("réel A");
            let y = y.as_f64().expect("réel B");
            assert!(
                x.is_finite() && y.is_finite(),
                "valeurs non finies : {x} et {y}"
            );
            if x == y {
                0.0
            } else {
                (x - y).abs() / x.abs().max(y.abs())
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "longueurs différentes");
            x.iter()
                .zip(y)
                .map(|(a, b)| ecart_max(a, b))
                .fold(0.0, f64::max)
        }
        (Value::Object(x), Value::Object(y)) => {
            assert_eq!(
                x.keys().collect::<Vec<_>>(),
                y.keys().collect::<Vec<_>>(),
                "clés différentes"
            );
            x.iter()
                .map(|(k, a)| ecart_max(a, &y[k]))
                .fold(0.0, f64::max)
        }
        (Value::Null, _) | (_, Value::Null) => {
            panic!("valeur nulle, mesure non finie ou absente : {a} et {b}")
        }
        _ => {
            assert_eq!(a, b, "structure ou textes différents");
            0.0
        }
    }
}

fn backend_gpu() -> &'static str {
    match std::env::var("KEVINOTRON_BACKEND").as_deref() {
        Ok("hip") => "hip",
        Ok("cuda") => "cuda",
        _ if cfg!(feature = "cuda") => "cuda",
        _ => "hip",
    }
}

fn exiger_gpu() -> Appareil {
    match kvu::su2::cli::ouvrir_backend(backend_gpu()) {
        Ok(Some(app)) => app,
        Ok(None) => panic!("FAIL : test GPU demandé, aucun GPU"),
        Err(e) => panic!("FAIL : test GPU demandé, aucun GPU : {e}"),
    }
}

#[test]
fn e1_e2_cpu() {
    let d = dossier("e1_e2_cpu");
    let sans = d.join("sans");
    let avec = d.join("avec");
    reussi(&gen("cpu", 4, 6, 4, 2, true, &sans, &[]));
    reussi(&gen(
        "cpu",
        4,
        6,
        4,
        2,
        true,
        &avec,
        &["--mesures", SPEC_CPU],
    ));
    let fichiers = comparer_physiques(&sans, &avec, 4);
    println!("\nPASS E1-CPU : {fichiers} fichiers comparés, écart 0 octet");

    let en_ligne = lire_json(&avec.join(NOM_JSON));
    let mut hors = hors_ligne(&avec, "brut,flot:0.02:3", "cpu");
    renommer(&mut hors, 6, 2);
    let compares = memes_bits(&en_ligne, &hors).expect("égalité au bit en ligne/hors ligne CPU");
    let csv = String::from_utf8(octets(&avec.join(NOM_CSV))).expect("CSV UTF-8");
    assert_eq!(csv.lines().count(), 1 + 4 * 2, "nombre de lignes du CSV");
    assert!(avec.join("COMPLETE").exists(), "COMPLETE absent");
    let manifeste = lire_json(&avec.join("manifest.json"));
    let liste = manifeste["fichiers"]
        .as_array()
        .expect("liste des fichiers du manifeste");
    for nom in [NOM_JSON, NOM_CSV] {
        let f = liste
            .iter()
            .find(|f| f["nom"].as_str() == Some(nom))
            .unwrap_or_else(|| panic!("{nom} absent du manifeste"));
        let sha = sha256_fichier(&avec.join(nom)).expect("calcul du sha256");
        assert_eq!(sha.len(), 64, "longueur du sha256");
        assert!(
            sha.bytes().all(|b| b.is_ascii_hexdigit()),
            "sha256 non hexadécimal"
        );
        assert_eq!(f["sha256"].as_str(), Some(sha.as_str()), "sha256 de {nom}");
    }
    let n = nombre_de_valeurs(&en_ligne["chaines"]);
    assert!(
        n >= 4 * 2 * (4 + 2 * 4),
        "pas assez de valeurs mesurées : {n}"
    );
    println!("\nPASS E2-CPU : {n} valeurs, {compares} nombres au bit, 2 empreintes vérifiées");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

fn determinisme(backend: &str, l: usize, spec: &str, nom: &str) {
    let d = dossier(nom);
    // Le nom final est identique, car il figure dans le champ JSON « chaine ».
    let a = d.join("a/chaine");
    let b = d.join("b/chaine");
    reussi(&gen(backend, l, 6, 4, 2, true, &a, &["--mesures", spec]));
    reussi(&gen(backend, l, 6, 4, 2, true, &b, &["--mesures", spec]));
    for nom in [NOM_CSV, NOM_JSON] {
        assert_eq!(
            octets(&a.join(nom)),
            octets(&b.join(nom)),
            "déterminisme de {nom}"
        );
    }
    println!("\nPASS E4-{backend} : 2 fichiers comparés, écart 0 octet");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn e4_cpu_determinisme() {
    determinisme("cpu", 4, SPEC_CPU, "e4_cpu");
}

fn reprise_identique(backend: &str, l: usize, spec: &str, nom: &str) {
    let d = dossier(nom);
    let a = d.join("a/chaine");
    let b = d.join("b/chaine");
    reussi(&gen(
        backend,
        l,
        6,
        6,
        2,
        true,
        &a,
        &["--reprise-every", "2", "--mesures", spec],
    ));
    reussi(&gen(
        backend,
        l,
        6,
        6,
        2,
        true,
        &b,
        &[
            "--reprise-every",
            "2",
            "--mesures",
            spec,
            "--arret-apres",
            "10",
        ],
    ));
    assert!(!b.join("COMPLETE").exists(), "COMPLETE écrit avant la fin");
    assert!(!b.join(NOM_JSON).exists(), "JSON écrit avant la fin");
    let chemin = b.join(NOM_CSV);
    let texte = String::from_utf8(octets(&chemin)).expect("CSV UTF-8");
    let derniere = texte.lines().last().expect("dernière mesure du CSV");
    let (_, suite) = derniere.split_once(',').expect("premier champ pas du CSV");
    let parasites = format!("12,{suite}\n12,brut,0,0.5");
    fs::OpenOptions::new()
        .append(true)
        .open(&chemin)
        .expect("ouverture en ajout du CSV")
        .write_all(parasites.as_bytes())
        .expect("injection des lignes postérieures et partielles");
    // Préciser le backend à la reprise pour garder le même chemin numérique.
    let o = kvu(&[
        "su2",
        "gen",
        "--backend",
        backend,
        "--reprendre",
        b.to_str().expect("chemin UTF-8 de reprise"),
        "--mesures",
        spec,
    ]);
    reussi(&o);
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("octet(s) retiré(s)"),
        "troncature de reprise non signalée"
    );
    let physiques = comparer_physiques(&a, &b, 6);
    for nom in [NOM_CSV, NOM_JSON] {
        assert_eq!(
            octets(&a.join(nom)),
            octets(&b.join(nom)),
            "reprise de {nom}"
        );
    }
    assert!(b.join("COMPLETE").exists(), "COMPLETE absent après reprise");
    let fichiers = physiques + 2;
    println!(
        "\nPASS E5-{backend} : {fichiers} fichiers comparés, écart 0 octet, \
         2 lignes parasites retirées"
    );
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn e5_cpu_reprise() {
    reprise_identique("cpu", 4, SPEC_CPU, "e5_cpu");
}

#[test]
fn e5_cpu_reprise_spec_differente_refusee() {
    let d = dossier("e5_cpu_spec");
    let b = d.join("chaine");
    reussi(&gen(
        "cpu",
        4,
        6,
        6,
        2,
        true,
        &b,
        &[
            "--reprise-every",
            "2",
            "--mesures",
            "corr:brut:1",
            "--arret-apres",
            "10",
        ],
    ));
    let avant = octets(&b.join(NOM_CSV));
    let o = kvu(&[
        "su2",
        "gen",
        "--backend",
        "cpu",
        "--reprendre",
        b.to_str().expect("chemin UTF-8 de reprise"),
        "--mesures",
        SPEC_CPU,
    ]);
    assert_eq!(
        o.status.code(),
        Some(2),
        "spécification différente acceptée"
    );
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("diffère de la spécification du run"),
        "garde de spécification absente de l'erreur : {}",
        String::from_utf8_lossy(&o.stderr),
    );
    assert_eq!(
        octets(&b.join(NOM_CSV)),
        avant,
        "CSV tronqué malgré le refus"
    );
    assert!(!b.join("COMPLETE").exists(), "COMPLETE malgré le refus");
    assert!(!b.join(NOM_JSON).exists(), "JSON malgré le refus");
    println!("\nPASS E5-CPU-SPEC : 1 reprise refusée, 0 JSON, 0 COMPLETE");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

fn interrompre(backend: &str, l: usize, dir: &Path, spec: Option<&str>) {
    let mut extra = vec!["--reprise-every", "2", "--arret-apres", "10"];
    if let Some(spec) = spec {
        extra.extend_from_slice(&["--mesures", spec]);
    }
    reussi(&gen(backend, l, 6, 6, 2, true, dir, &extra));
    assert!(!dir.join("COMPLETE").exists(), "COMPLETE avant la fin");
    assert!(!dir.join(NOM_JSON).exists(), "synthèse avant la clôture");
}

fn paire_reprise(d: &Path, backend: &str, l: usize, spec: &str) -> (PathBuf, PathBuf) {
    let a = d.join("a/chaine");
    let b = d.join("b/chaine");
    reussi(&gen(
        backend,
        l,
        6,
        6,
        2,
        true,
        &a,
        &["--reprise-every", "2", "--mesures", spec],
    ));
    interrompre(backend, l, &b, Some(spec));
    (a, b)
}

fn comparer_corr(a: &Path, b: &Path) {
    for nom in [NOM_CSV, NOM_JSON] {
        assert_eq!(octets(&a.join(nom)), octets(&b.join(nom)), "{nom} diffère");
    }
    assert!(b.join("COMPLETE").exists(), "COMPLETE après clôture absent");
}

#[test]
fn arret_brutal_corr() {
    let d = dossier("arret_brutal_corr");
    // Le nom final figure dans le JSON du corrélateur.
    let a = d.join("a/chaine");
    let b = d.join("b/chaine");
    reussi(&gen(
        "cpu",
        4,
        6,
        6,
        2,
        false,
        &a,
        &["--reprise-every", "2", "--mesures", SPEC_CPU],
    ));
    reussi(&gen(
        "cpu",
        4,
        6,
        6,
        2,
        false,
        &b,
        &[
            "--reprise-every",
            "2",
            "--mesures",
            SPEC_CPU,
            "--arret-apres",
            "10",
        ],
    ));
    let etat = lire_json(&b.join("reprise.json"));
    assert_eq!(etat["pas_faits"].as_u64(), Some(10));
    assert_eq!(etat["mesures_faites"].as_u64(), Some(3)); // indice de la PROCHAINE mesure (m + 1, su2::run::point_si_du)
    assert!(!b.join("COMPLETE").exists(), "COMPLETE avant la fin");
    assert!(!b.join(NOM_JSON).exists(), "JSON avant la clôture");
    let texte = String::from_utf8(octets(&a.join(NOM_CSV))).expect("CSV témoin UTF-8");
    let lignes: String = texte
        .split_inclusive('\n')
        .filter(|ligne| ligne.split(',').next() == Some("12"))
        .collect();
    assert_eq!(
        lignes.lines().count(),
        2,
        "deux lissages attendus au pas 12"
    );
    assert!(lignes.ends_with('\n'), "ligne témoin partielle");
    let chemin = b.join(NOM_CSV);
    assert!(octets(&chemin).ends_with(b"\n"), "CSV interrompu partiel");
    {
        let mut f = fs::OpenOptions::new()
            .append(true)
            .open(&chemin)
            .expect("ouverture en ajout du CSV interrompu");
        f.write_all(lignes.as_bytes())
            .expect("injection des deux lignes du pas 12");
        f.write_all(b"12,0")
            .expect("injection du fragment sans fin de ligne");
        f.sync_all()
            .expect("synchronisation de l'arrêt brutal simulé");
    }
    reussi(&reprendre("cpu", &b, &[]));
    comparer_corr(&a, &b);
    assert_eq!(comparer_physiques(&a, &b, 0), 2, "aucun dump fin attendu");
    println!(
        "\nPASS ARRET-BRUTAL-CORR : 2 lignes et 1 fragment injectés, \
         4 fichiers identiques, 0 octet différent"
    );
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

fn reprise_sans_mesures_recharge(backend: &str, l: usize, spec: &str, nom: &str) {
    let d = dossier(nom);
    let (a, b) = paire_reprise(&d, backend, l, spec);
    let o = reprendre(backend, &b, &[]);
    reussi(&o);
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("rechargée"),
        "rechargement absent : {}",
        String::from_utf8_lossy(&o.stderr)
    );
    let n = comparer_physiques(&a, &b, 6) + 2;
    comparer_corr(&a, &b);
    println!("\nPASS E5-{backend}-RECHARGE : {n} fichiers identiques à l'octet");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn e5_cpu_reprise_sans_mesures_recharge() {
    reprise_sans_mesures_recharge("cpu", 4, SPEC_CPU, "e5_cpu_recharge");
}

#[test]
fn e5_cpu_reprise_periode_ou_mutant_change_refuse() {
    let d = dossier("e5_cpu_periode_mutant");
    let b = d.join("chaine");
    interrompre("cpu", 4, &b, Some(SPEC_CPU));
    let avant = octets(&b.join(NOM_CSV));
    for spec in [
        "corr:brut+flot/0.02/3:2",
        "corr:brut+flot/0.02/3:1:mutant=direction",
    ] {
        let o = reprendre("cpu", &b, &["--mesures", spec]);
        assert_eq!(o.status.code(), Some(2), "modification acceptée : {spec}");
        assert!(String::from_utf8_lossy(&o.stderr).contains("diffère de la spécification du run"));
        assert_eq!(octets(&b.join(NOM_CSV)), avant, "CSV modifié pour {spec}");
        assert!(!b.join("COMPLETE").exists());
        assert!(!b.join(NOM_JSON).exists());
    }
    println!("\nPASS E5-CPU-PERIODE-MUTANT : 2 refus code 2, 0 octet modifié");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn e5_cpu_run_sans_mesures_repris_avec_refuse() {
    let d = dossier("e5_cpu_ajout_mesures");
    let b = d.join("chaine");
    interrompre("cpu", 4, &b, None);
    let o = reprendre("cpu", &b, &["--mesures", "corr:brut:1"]);
    assert_eq!(o.status.code(), Some(2), "ajout de mesures accepté");
    assert!(String::from_utf8_lossy(&o.stderr).contains("créé sans --mesures"));
    assert!(!b.join(NOM_CSV).exists());
    assert!(!b.join("COMPLETE").exists());
    println!("\nPASS E5-CPU-AJOUT-MESURES : 1 refus code 2, 0 CSV, 0 COMPLETE");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

fn simuler_run_ancien(dir: &Path) {
    fs::remove_file(dir.join(NOM_FICHIER_MESURES)).expect("suppression du compagnon du run ancien");
    let chemin = dir.join("manifest.json");
    let mut man = lire_json(&chemin);
    let fichiers = man["fichiers"]
        .as_array_mut()
        .expect("fichiers du manifeste");
    let avant = fichiers.len();
    fichiers.retain(|f| f["nom"].as_str() != Some(NOM_FICHIER_MESURES));
    assert_eq!(avant, fichiers.len() + 1, "entrée du compagnon non retirée");
    fs::write(
        &chemin,
        serde_json::to_vec_pretty(&man).expect("sérialisation du manifeste ancien"),
    )
    .expect("écriture du manifeste sans entrée du compagnon");
}

#[test]
fn e5_cpu_run_ancien() {
    let d = dossier("e5_cpu_ancien");
    let (a, b) = paire_reprise(&d, "cpu", 4, SPEC_CPU);
    simuler_run_ancien(&b);
    let avant = octets(&b.join(NOM_CSV));
    let o = reprendre("cpu", &b, &[]);
    assert_eq!(o.status.code(), Some(2), "omission acceptée sur run ancien");
    let erreur = String::from_utf8_lossy(&o.stderr);
    assert!(
        erreur.contains("--reprendre") && erreur.contains(SPEC_CPU),
        "{erreur}"
    );
    let o = reprendre("cpu", &b, &["--mesures", "corr:brut:1"]);
    assert_eq!(o.status.code(), Some(2), "autre spécification acceptée");
    assert_eq!(octets(&b.join(NOM_CSV)), avant, "CSV modifié par les refus");
    reussi(&reprendre("cpu", &b, &["--mesures", SPEC_CPU]));
    comparer_corr(&a, &b);

    let sans_argv = d.join("sans_argv/chaine");
    interrompre("cpu", 4, &sans_argv, Some(SPEC_CPU));
    simuler_run_ancien(&sans_argv);
    let p = sans_argv.join("manifest.json");
    let mut man = lire_json(&p);
    assert!(man
        .as_object_mut()
        .expect("objet manifeste")
        .remove("argv")
        .is_some());
    fs::write(
        &p,
        serde_json::to_vec_pretty(&man).expect("sérialisation du manifeste"),
    )
    .expect("écriture du manifeste sans argv");
    let avant = octets(&sans_argv.join(NOM_CSV));
    let o = reprendre("cpu", &sans_argv, &[]);
    assert_eq!(
        o.status.code(),
        Some(2),
        "argv illisible accepté sans --mesures"
    );
    assert_eq!(octets(&sans_argv.join(NOM_CSV)), avant);
    reussi(&reprendre("cpu", &sans_argv, &["--mesures", SPEC_CPU]));
    comparer_corr(&a, &sans_argv);
    println!("\nPASS E5-CPU-ANCIEN : 3 refus code 2, 2 reprises, 4 sorties identiques à l'octet");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn e5_cpu_trou_ancien_refuse() {
    let d = dossier("e5_cpu_trou");
    let b = d.join("chaine");
    interrompre("cpu", 4, &b, Some(SPEC_CPU));
    let chemin = b.join(NOM_CSV);
    let csv = String::from_utf8(octets(&chemin)).expect("CSV UTF-8");
    let mut lignes = csv.split_inclusive('\n');
    let entete = lignes.next().expect("en-tête CSV");
    let _ = lignes.next().expect("première ligne de données");
    fs::write(&chemin, format!("{entete}{}", lignes.collect::<String>()))
        .expect("suppression de la première ligne de données");
    let o = reprendre("cpu", &b, &[]);
    assert_ne!(o.status.code(), Some(0), "trou accepté");
    let erreur = String::from_utf8_lossy(&o.stderr);
    assert!(
        erreur.contains("calendrier") || erreur.contains("trou"),
        "{erreur}"
    );
    assert!(!b.join("COMPLETE").exists(), "COMPLETE malgré le trou");
    assert!(!b.join(NOM_JSON).exists(), "JSON malgré le trou");
    println!("\nPASS E5-CPU-TROU : 1 trou refusé, 0 JSON, 0 COMPLETE");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn o8_cloture_echoue_vraiment() {
    let d = dossier("o8_cloture");
    let b = d.join("chaine");
    reussi(&gen(
        "cpu",
        4,
        6,
        4,
        2,
        false,
        &b,
        &[
            "--reprise-every",
            "2",
            "--mesures",
            SPEC_CPU,
            "--arret-apres",
            "14",
        ],
    ));
    let etat = lire_json(&b.join("reprise.json"));
    assert_eq!(
        etat["pas_faits"].as_u64(),
        Some(14),
        "dernier point attendu"
    );
    assert_eq!(
        etat["mesures_faites"].as_u64(),
        Some(5), // m + 1 > nmeas = 4
        "aucune mesure restante"
    );
    assert!(!b.join("COMPLETE").exists(), "COMPLETE avant la clôture");
    assert!(!b.join(NOM_JSON).exists(), "JSON avant la clôture");
    let chemin = b.join(NOM_CSV);
    let texte = String::from_utf8(octets(&chemin)).expect("CSV UTF-8");
    let mut lignes: Vec<String> = texte.lines().map(str::to_owned).collect();
    assert_eq!(lignes.len(), 9, "quatre mesures et deux lissages attendus");
    let ligne = lignes.get_mut(2).expect("deuxième ligne de données");
    let mut champs: Vec<String> = ligne.split(',').map(str::to_owned).collect();
    assert!(champs.len() >= 4, "quatrième champ absent");
    assert!(
        champs[3]
            .parse::<f64>()
            .expect("valeur initiale du quatrième champ")
            .is_finite(),
        "valeur déjà non finie avant corruption"
    );
    champs[3] = "nan".into();
    *ligne = champs.join(",");
    let mut corrompu = lignes.join("\n");
    corrompu.push('\n');
    fs::write(&chemin, corrompu).expect("corruption d'une valeur de la deuxième ligne");
    let avant_corr = octets(&chemin);
    let avant_therm = octets(&b.join("therm.csv"));
    let avant_mesures = octets(&b.join("measurements.csv"));
    let o = reprendre("cpu", &b, &[]);
    assert_ne!(
        o.status.code(),
        Some(0),
        "clôture réussie malgré la valeur nan"
    );
    let erreur = String::from_utf8_lossy(&o.stderr);
    assert!(
        erreur.contains("fini") || erreur.contains("clôture"),
        "{erreur}"
    );
    assert_eq!(
        octets(&chemin),
        avant_corr,
        "mesures corr refaites à la reprise finale"
    );
    assert_eq!(
        octets(&b.join("therm.csv")),
        avant_therm,
        "thermalisation refaite"
    );
    assert_eq!(
        octets(&b.join("measurements.csv")),
        avant_mesures,
        "mesure refaite"
    );
    assert!(
        !b.join("COMPLETE").exists(),
        "COMPLETE malgré l'échec de clôture"
    );
    assert!(!b.join(NOM_JSON).exists(), "JSON malgré l'échec de clôture");
    println!(
        "\nPASS O8-CLOTURE : 14 pas déjà faits, 4 mesures déjà faites, \
         1 valeur nan refusée, 0 JSON, 0 COMPLETE"
    );
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn o9_cpu_json_perime_invalide() {
    let d = dossier("o9_cpu_json_perime");
    let (a, b) = paire_reprise(&d, "cpu", 4, SPEC_CPU);
    fs::write(b.join(NOM_JSON), b"{\"synthese\":\"fausse\"}\n")
        .expect("injection de la synthèse périmée");
    reussi(&reprendre("cpu", &b, &["--arret-apres", "12"]));
    assert!(
        !b.join(NOM_JSON).exists(),
        "JSON périmé conservé à l'interruption"
    );
    assert!(!b.join("COMPLETE").exists());
    reussi(&reprendre("cpu", &b, &[]));
    comparer_corr(&a, &b);
    println!("\nPASS O9-CPU : 1 JSON périmé retiré, 2 sorties finales identiques à l'octet");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn cpu_bord_et_mesures_refuse() {
    let d = dossier("cpu_bord_mesures");
    let sortie = d.join("chaine");
    let o = gen(
        "cpu",
        4,
        2,
        4,
        1,
        false,
        &sortie,
        &[
            "--bord",
            "--bc",
            "cold",
            "--bloc",
            "2",
            "--mesures",
            "corr:brut:1",
        ],
    );
    assert_eq!(o.status.code(), Some(2), "bord avec mesures accepté");
    let erreur = String::from_utf8_lossy(&o.stderr);
    assert!(
        erreur.contains("--bord") && erreur.contains("--mesures"),
        "{erreur}"
    );
    assert!(!sortie.exists(), "répertoire créé malgré le refus");
    println!("\nPASS CPU-BORD-MESURES : 1 refus code 2, 0 répertoire créé");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

fn periode_et_sans_dump(backend: &str, l: usize, spec: &str, lissages: &str, nom: &str) {
    let d = dossier(nom);
    let avec = d.join("avec/chaine");
    let sans = d.join("sans/chaine");
    reussi(&gen(
        backend,
        l,
        6,
        6,
        2,
        true,
        &avec,
        &["--dump-every", "2", "--mesures", spec],
    ));
    reussi(&gen(
        backend,
        l,
        6,
        6,
        2,
        false,
        &sans,
        &["--mesures", spec],
    ));
    let mesure = if backend == "cpu" { "cpu" } else { "gpu" };
    let mut hors = hors_ligne(&avec, lissages, mesure);
    renommer(&mut hors, 6, 2);
    let en_ligne = lire_json(&avec.join(NOM_JSON));
    let n = memes_bits(&en_ligne, &hors).expect("égalité au bit avec période 2");
    let configs = en_ligne["chaines"][0]["configs"]
        .as_array()
        .expect("configurations");
    let noms: Vec<&str> = configs
        .iter()
        .map(|c| c["cfg"].as_str().expect("nom"))
        .collect();
    assert_eq!(noms, ["pas_000010", "pas_000014", "pas_000018"]);
    comparer_corr(&avec, &sans);
    assert!(!fichiers_physiques(&sans)
        .iter()
        .any(|n| n.starts_with("cfg_")));
    let csv = String::from_utf8(octets(&sans.join(NOM_CSV))).expect("CSV UTF-8");
    assert_eq!(csv.lines().count(), 7, "3 pas et 2 lissages attendus");
    println!("\nPASS {backend}-PERIODE-2-SANS-DUMP : {n} nombres au bit, 2 fichiers identiques");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn cpu_periode_2_et_sans_dump() {
    periode_et_sans_dump(
        "cpu",
        4,
        "corr:brut+flot/0.02/3:2",
        "brut,flot:0.02:3",
        "cpu_periode_2",
    );
}

fn deux_chaines(
    backend: &str,
    l: usize,
    spec: &str,
    lissages: &str,
    periode: u64,
    sans_dump: bool,
    nom: &str,
) {
    let d = dossier(nom);
    let avec = d.join("avec/ensemble");
    let periode_texte = periode.to_string();
    reussi(&gen(
        backend,
        l,
        6,
        6,
        2,
        true,
        &avec,
        &[
            "--graine",
            "26111108,26111109",
            "--dump-every",
            &periode_texte,
            "--mesures",
            spec,
        ],
    ));
    let sans = d.join("sans/ensemble");
    if sans_dump {
        reussi(&gen(
            backend,
            l,
            6,
            6,
            2,
            false,
            &sans,
            &["--graine", "26111108,26111109", "--mesures", spec],
        ));
    }
    let mut n = 0;
    for (c, nom) in ["c000", "c001"].iter().enumerate() {
        let chaine = avec.join(nom);
        let hl = d.join(format!("hl_c{c}/{nom}"));
        fs::create_dir_all(&hl).expect("création du répertoire hors ligne");
        fs::write(
            hl.join("manifest.json"),
            json!({"betas": [2.3], "l": l}).to_string(),
        )
        .expect("écriture du manifeste minimal");
        let dumps: Vec<String> = fichiers_physiques(&chaine)
            .into_iter()
            .filter(|n| n.starts_with("cfg_"))
            .collect();
        assert_eq!(
            dumps.len() as u64,
            6 / periode,
            "nombre de dumps par chaîne"
        );
        for nom in dumps {
            fs::copy(chaine.join(&nom), hl.join(&nom)).expect("copie du dump hors ligne");
        }
        let mesure = if backend == "cpu" { "cpu" } else { "gpu" };
        let mut hors = hors_ligne(&hl, lissages, mesure);
        renommer(&mut hors, 6, 2);
        let en_ligne = lire_json(&chaine.join(NOM_JSON));
        n += memes_bits(&en_ligne, &hors).expect("égalité au bit pour chaque chaîne");
        if sans_dump {
            let autre = sans.join(nom);
            for fichier in [NOM_CSV, NOM_JSON] {
                assert_eq!(octets(&chaine.join(fichier)), octets(&autre.join(fichier)));
            }
            assert!(!fichiers_physiques(&autre)
                .iter()
                .any(|n| n.starts_with("cfg_")));
        }
    }
    assert!(avec.join("COMPLETE").exists());
    if sans_dump {
        assert!(sans.join("COMPLETE").exists());
    }
    println!("\nPASS {backend}-DEUX-CHAINES : 2 chaînes, période {periode}, {n} nombres au bit");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn cpu_deux_chaines() {
    deux_chaines(
        "cpu",
        4,
        SPEC_CPU,
        "brut,flot:0.02:3",
        1,
        false,
        "cpu_deux_chaines",
    );
}

fn verifier_manifeste_corr(dir: &Path) -> usize {
    // Relire le document publié, sans dépendre d'un manifeste conservé en mémoire par l'API.
    let man = lire_json(&dir.join("manifest.json"));
    let fichiers = man["fichiers"]
        .as_array()
        .expect("fichiers du manifeste sur disque");
    for nom in [NOM_JSON, NOM_CSV, NOM_FICHIER_MESURES] {
        let entree = fichiers
            .iter()
            .find(|f| f["nom"].as_str() == Some(nom))
            .unwrap_or_else(|| panic!("{nom} absent du manifeste persisté"));
        let sha = sha256_fichier(&dir.join(nom)).expect("empreinte du fichier publié");
        assert_eq!(
            entree["sha256"].as_str(),
            Some(sha.as_str()),
            "empreinte de {nom}"
        );
    }
    let spec = lire_json(&dir.join(NOM_FICHIER_MESURES));
    assert_eq!(spec["spec"], SPEC_CPU, "spécification non normalisée");
    assert_eq!(spec["backend"], "cpu");
    3
}

#[test]
fn manifeste_persiste_les_syntheses() {
    let d = dossier("manifeste_persistant");
    let (a, b) = paire_reprise(&d, "cpu", 4, " corr:brut+flot/0.02/3:1 ");
    let n = verifier_manifeste_corr(&a);
    reussi(&reprendre("cpu", &b, &[]));
    let n = n + verifier_manifeste_corr(&b);
    println!("\nPASS MANIFESTE-PERSISTANT : {n} empreintes, 2 spécifications normalisées CPU");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn cpu_mutant_refuse() {
    let d = dossier("cpu_mutant");
    let sortie = d.join("chaine");
    let o = gen(
        "cpu",
        4,
        6,
        4,
        2,
        true,
        &sortie,
        &["--mesures", "corr:brut:1:mutant=direction"],
    );
    assert_ne!(o.status.code(), Some(0), "mutant accepté sur CPU");
    assert!(String::from_utf8_lossy(&o.stderr).contains("mutants de corr : GPU seulement"));
    assert!(
        !sortie.join("COMPLETE").exists(),
        "COMPLETE malgré le mutant CPU"
    );
    println!("\nPASS CPU-MUTANT : 1 mutant refusé, 0 COMPLETE");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn grammaire_cli_refusee() {
    let d = dossier("grammaire");
    for (i, spec) in [
        "corr:brut:0",
        "corr:flot/abc/5:1",
        "corr:brut:1,corr:brut:2",
    ]
    .iter()
    .enumerate()
    {
        let sortie = d.join(format!("invalide_{i}"));
        let o = gen("cpu", 4, 6, 4, 2, true, &sortie, &["--mesures", spec]);
        assert_ne!(o.status.code(), Some(0), "{spec} accepté");
        assert!(!sortie.exists(), "répertoire créé malgré {spec}");
    }
    println!("\nPASS GRAMMAIRE-CLI : 3 spécifications refusées, 0 répertoire de run créé");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

/// Mutant E1 : recopie délibérément un champ flotté dans le champ résident du générateur.
struct RecopieResident;

impl MesureEnLigne for RecopieResident {
    fn nom(&self) -> &str {
        "recopie_resident"
    }

    fn periode(&self) -> usize {
        1
    }

    #[allow(clippy::too_many_arguments)]
    fn mesurer(
        &mut self,
        app: Option<&Appareil>,
        champ: Option<&ChampGpu>,
        _hote: Option<&ChampHote>,
        chaine: usize,
        _pas: u64,
        _sortie: &Path,
    ) -> Resultat<()> {
        let app = app.ok_or_else(|| Erreur::Config("mutant E1 : appareil GPU absent".into()))?;
        let champ =
            champ.ok_or_else(|| Erreur::Config("mutant E1 : champ résident absent".into()))?;
        let module = ModuleObs::nouveau(app)?;
        let e = module.flot_prepare(champ, chaine)?;
        for _ in 0..5 {
            module.flot_pas(&e, 0.02)?;
        }
        let v = champ.v();
        let base = match &champ.tampon {
            TamponChamp::F64(t) => t.arg_decale(chaine * 16 * v),
            TamponChamp::F32(_) => {
                return Err(Erreur::Config("mutant E1 : champ F64 requis".into()));
            }
        };
        let f = module.fonction("k_obs_copie")?;
        app.lancer(
            &f,
            &Lancement::lineaire(4 * v, 256),
            &[e.champ_arg(), base, Arg::I64((4 * v) as i64)],
        )?;
        app.synchroniser()
    }
}

fn cfg_mutant_e1(outdir: &Path) -> ConfigRun {
    ConfigRun {
        lot: ConfigLot {
            l: 8,
            chaines: vec![ParamChaine {
                beta: 2.3,
                graine: Graine(26111108),
                depart: Depart::Chaud,
            }],
            nor: 4,
            precision: Precision::F64,
            options: OptionsCompil::reference_k1(),
            mutant_metro: false,
        },
        therm: 10,
        nmeas: 30,
        skip: 1,
        dump_every: 1,
        dump_initial: false,
        format_dump: "q64".into(),
        zstd: false,
        reprise_every: 0,
        outdir: outdir.to_path_buf(),
        bord: None,
    }
}

#[test]
#[ignore = "GPU"]
fn e1_gpu_non_perturbation() {
    drop(exiger_gpu());
    let backend = backend_gpu();
    let d = dossier("e1_gpu");
    for l in [8, 12] {
        let sans = d.join(format!("l{l}/sans"));
        let avec = d.join(format!("l{l}/avec"));
        reussi(&gen(backend, l, 10, 30, 1, true, &sans, &[]));
        reussi(&gen(
            backend,
            l,
            10,
            30,
            1,
            true,
            &avec,
            &["--mesures", SPEC_GPU],
        ));
        let n = comparer_physiques(&sans, &avec, 30);
        assert_eq!(n, 32, "30 dumps et 2 CSV attendus");
        println!("\nPASS E1-GPU : L={l}, 40 pas, {n} fichiers comparés, écart 0 octet");
    }
    let app = exiger_gpu();
    let sain = d.join("mutant/sain");
    let altere = d.join("mutant/altere");
    run::executer(Some(&app), &cfg_mutant_e1(&sain), &mut [], false).expect("run témoin E1");
    let mut mesures: Vec<Box<dyn MesureEnLigne>> = vec![Box::new(RecopieResident)];
    run::executer(Some(&app), &cfg_mutant_e1(&altere), &mut mesures, false).expect("run mutant E1");
    assert_ne!(
        octets(&sain.join("measurements.csv")),
        octets(&altere.join("measurements.csv")),
        "FAIL E1-GPU : mutant vert, la recopie du flot dans le champ résident n'est pas détectée",
    );
    println!("\nPASS E1-GPU : mutant rouge, 1 fichier measurements.csv différent, 40 pas");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
#[ignore = "GPU"]
fn e2_gpu_en_ligne_egal_hors_ligne() {
    drop(exiger_gpu());
    let d = dossier("e2_gpu");
    for l in [8, 12, 16] {
        let chaine = d.join(format!("l{l}/chaine"));
        reussi(&gen(
            backend_gpu(),
            l,
            10,
            6,
            2,
            true,
            &chaine,
            &["--mesures", SPEC_GPU],
        ));
        let en_ligne = lire_json(&chaine.join(NOM_JSON));
        let mut gpu = hors_ligne(&chaine, "brut,flot:0.02:50", "gpu");
        renommer(&mut gpu, 10, 2);
        let ecart_gpu = ecart_max(&en_ligne, &gpu);
        let compares = memes_bits(&en_ligne, &gpu).expect("égalité au bit en ligne/hors ligne GPU");
        let mut cpu = hors_ligne(&chaine, "brut,flot:0.02:50", "cpu");
        renommer(&mut cpu, 10, 2);
        let ecart_cpu = ecart_max(&en_ligne, &cpu);
        assert!(
            ecart_cpu <= 1e-10,
            "FAIL E2-GPU L={l} : écart CPU {ecart_cpu:e}"
        );
        let n = nombre_de_valeurs(&en_ligne["chaines"]);
        println!(
            "\nPASS E2-GPU : L={l}, {n} valeurs, {compares} nombres au bit, \
             écart GPU {ecart_gpu:e}, écart CPU {ecart_cpu:e}"
        );
    }
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
#[ignore = "GPU"]
fn e3_gpu_mutants_rouges() {
    drop(exiger_gpu());
    let d = dossier("e3_gpu");
    let reference = d.join("reference/chaine");
    reussi(&gen(
        backend_gpu(),
        8,
        10,
        4,
        2,
        true,
        &reference,
        &["--mesures", SPEC_GPU],
    ));
    let reference = lire_json(&reference.join(NOM_JSON));
    for mutant in ["pas-en-moins", "direction"] {
        let chaine = d.join(format!("{mutant}/chaine"));
        let spec = format!("{SPEC_GPU}:mutant={mutant}");
        reussi(&gen(
            backend_gpu(),
            8,
            10,
            4,
            2,
            true,
            &chaine,
            &["--mesures", &spec],
        ));
        let doc = lire_json(&chaine.join(NOM_JSON));
        assert_ne!(doc["mutant"], "Aucun", "mutant absent du JSON");
        let ecart = ecart_max(&reference["chaines"], &doc["chaines"]);
        assert!(
            ecart > 1e-6,
            "FAIL E3-GPU : mutant {mutant} vert, écart {ecart:e}"
        );
        if mutant == "pas-en-moins" {
            let configs = reference["chaines"][0]["configs"]
                .as_array()
                .expect("configurations de référence");
            let autres = doc["chaines"][0]["configs"]
                .as_array()
                .expect("configurations mutantes");
            assert_eq!(configs.len(), autres.len());
            let mut ecart_flot = 0.0_f64;
            for (a, b) in configs.iter().zip(autres) {
                assert_eq!(
                    ecart_max(&a["mesures"]["brut"], &b["mesures"]["brut"]),
                    0.0,
                    "brut modifié par le mutant"
                );
                ecart_flot = ecart_flot.max(ecart_max(
                    &a["mesures"]["flot:0.02:50"],
                    &b["mesures"]["flot:0.02:50"],
                ));
            }
            assert!(
                ecart_flot > 1e-6,
                "FAIL E3-GPU : flot inchangé, écart {ecart_flot:e}"
            );
        }
        let n = nombre_de_valeurs(&doc["chaines"]);
        println!("\nPASS E3-GPU : mutant {mutant} rouge, {n} valeurs, écart {ecart:e}");
    }
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
#[ignore = "GPU"]
fn e4_gpu_determinisme() {
    drop(exiger_gpu());
    determinisme(backend_gpu(), 8, SPEC_GPU, "e4_gpu");
}

#[test]
#[ignore = "GPU"]
fn e5_gpu_reprise() {
    drop(exiger_gpu());
    reprise_identique(backend_gpu(), 8, SPEC_GPU, "e5_gpu");
}

#[test]
#[ignore = "GPU"]
fn e2_gpu_f32() {
    drop(exiger_gpu());
    let d = dossier("e2_gpu_f32");
    let chaine = d.join("chaine");
    reussi(&gen(
        backend_gpu(),
        8,
        6,
        6,
        2,
        true,
        &chaine,
        &["--precision", "f32", "--mesures", SPEC_GPU],
    ));
    let en_ligne = lire_json(&chaine.join(NOM_JSON));
    let mut hors = hors_ligne(&chaine, "brut,flot:0.02:50", "gpu");
    renommer(&mut hors, 6, 2);
    let n = memes_bits(&en_ligne, &hors).expect("égalité au bit avec générateur F32");
    println!("\nPASS E2-GPU-F32 : L=8, {n} nombres comparés au bit");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
#[ignore = "GPU"]
fn e2_gpu_deux_chaines_periode_2_sans_dump() {
    drop(exiger_gpu());
    deux_chaines(
        backend_gpu(),
        8,
        "corr:brut+flot/0.02/50:2",
        "brut,flot:0.02:50",
        2,
        true,
        "e2_gpu_deux_chaines_periode_2",
    );
}

#[test]
#[ignore = "GPU"]
fn e5_gpu_reprise_sans_mesures_recharge() {
    drop(exiger_gpu());
    reprise_sans_mesures_recharge(backend_gpu(), 8, SPEC_GPU, "e5_gpu_recharge");
}

#[test]
#[ignore = "GPU"]
fn e5_gpu_moteur_change_refuse() {
    drop(exiger_gpu());
    let d = dossier("e5_gpu_moteur");
    let b = d.join("chaine");
    interrompre(backend_gpu(), 8, &b, Some(SPEC_GPU));
    let avant = octets(&b.join(NOM_CSV));
    let o = reprendre("cpu", &b, &[]);
    assert_eq!(o.status.code(), Some(2), "changement GPU vers CPU accepté");
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("moteur"),
        "garde du moteur absente : {}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert_eq!(
        octets(&b.join(NOM_CSV)),
        avant,
        "CSV modifié malgré le refus"
    );
    assert!(!b.join(NOM_JSON).exists());
    assert!(!b.join("COMPLETE").exists());
    println!("\nPASS E5-GPU-MOTEUR : 1 refus code 2, 0 octet modifié");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
#[ignore = "GPU"]
fn o6_gpu_provenance() {
    drop(exiger_gpu());
    let d = dossier("o6_gpu_provenance");
    let gpu = d.join("gpu");
    reussi(&gen(
        backend_gpu(),
        8,
        2,
        2,
        1,
        false,
        &gpu,
        &["--mesures", SPEC_GPU],
    ));
    let man = lire_json(&gpu.join("manifest.json"));
    let mesures = man["modules_mesures"]
        .as_array()
        .expect("modules de mesure GPU");
    assert!(!mesures.is_empty(), "provenance des mesures absente");
    for module in mesures {
        let sha = module["sha256_code"]
            .as_str()
            .expect("empreinte du code objet");
        assert_eq!(sha.len(), 64, "empreinte du module de mesure");
        assert!(sha.bytes().all(|b| b.is_ascii_hexdigit()));
    }
    let generateur = man["modules_compiles"]
        .as_array()
        .expect("modules du générateur");
    assert!(!generateur.is_empty(), "provenance du générateur perdue");
    let cpu = d.join("cpu");
    reussi(&gen(
        "cpu",
        4,
        2,
        2,
        1,
        false,
        &cpu,
        &["--mesures", SPEC_CPU],
    ));
    assert!(lire_json(&cpu.join("manifest.json"))
        .get("modules_mesures")
        .is_none());
    println!(
        "\nPASS O6-GPU : {} module(s) de mesure, {} module(s) du générateur, \
         0 module de mesure CPU",
        mesures.len(),
        generateur.len()
    );
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
#[ignore = "GPU"]
fn o6_gpu_provenance_conservee() {
    drop(exiger_gpu());
    let d = dossier("o6_gpu_provenance_conservee");
    let b = d.join("chaine");
    reussi(&gen(
        backend_gpu(),
        8,
        10,
        4,
        2,
        false,
        &b,
        &[
            "--reprise-every",
            "2",
            "--mesures",
            SPEC_GPU,
            "--arret-apres",
            "18",
        ],
    ));
    let etat = lire_json(&b.join("reprise.json"));
    assert_eq!(
        etat["pas_faits"].as_u64(),
        Some(18),
        "dernier point attendu"
    );
    assert_eq!(
        etat["mesures_faites"].as_u64(),
        Some(5), // m + 1 > nmeas = 4
        "aucune mesure restante"
    );
    assert!(!b.join("COMPLETE").exists(), "COMPLETE avant la clôture");
    assert!(!b.join(NOM_JSON).exists(), "JSON avant la clôture");
    let avant_corr = octets(&b.join(NOM_CSV));
    let avant_mesures = octets(&b.join("measurements.csv"));
    reussi(&reprendre(backend_gpu(), &b, &[]));
    assert!(b.join("COMPLETE").exists(), "COMPLETE après reprise absent");
    assert!(b.join(NOM_JSON).exists(), "JSON après reprise absent");
    assert_eq!(
        octets(&b.join(NOM_CSV)),
        avant_corr,
        "mesures corr refaites"
    );
    assert_eq!(
        octets(&b.join("measurements.csv")),
        avant_mesures,
        "mesure refaite"
    );
    let man = lire_json(&b.join("manifest.json"));
    let modules = man["modules_mesures"]
        .as_array()
        .expect("modules de mesure du manifeste final");
    assert!(
        !modules.is_empty(),
        "provenance des mesures perdue à la reprise finale"
    );
    println!(
        "\nPASS O6-GPU-PROVENANCE-CONSERVEE : 18 pas déjà faits, 0 mesure restante, \
         {} module(s) de mesure conservé(s)",
        modules.len()
    );
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

/// Délègue toute l'interface et observe l'allocation persistante au premier appel seulement.
struct MesureMemoire {
    interne: Box<dyn MesureEnLigne>,
    pic: Rc<Cell<Option<u64>>>,
}

impl MesureEnLigne for MesureMemoire {
    fn nom(&self) -> &str {
        self.interne.nom()
    }

    fn periode(&self) -> usize {
        self.interne.periode()
    }

    fn demarrer(&mut self, cfg: &ConfigRun, dirs: &[PathBuf], reprise: bool) -> Resultat<()> {
        self.interne.demarrer(cfg, dirs, reprise)
    }

    fn besoin_memoire(&self, cfg: &ConfigRun, gpu: bool) -> BesoinMemoire {
        self.interne.besoin_memoire(cfg, gpu)
    }

    fn provenance(&self) -> Vec<ProvenanceCompil> {
        self.interne.provenance()
    }

    fn terminer(&mut self, cfg: &ConfigRun, dirs: &[PathBuf]) -> Resultat<Vec<PathBuf>> {
        self.interne.terminer(cfg, dirs)
    }

    #[allow(clippy::too_many_arguments)]
    fn mesurer(
        &mut self,
        app: Option<&Appareil>,
        champ: Option<&ChampGpu>,
        hote: Option<&ChampHote>,
        chaine: usize,
        pas: u64,
        sortie: &Path,
    ) -> Resultat<()> {
        if self.pic.get().is_some() {
            return self.interne.mesurer(app, champ, hote, chaine, pas, sortie);
        }
        let appareil = app.ok_or_else(|| Erreur::Config("O7 : appareil GPU absent".into()))?;
        appareil.synchroniser()?;
        let avant = appareil.memoire_libre_octets()?;
        self.interne
            .mesurer(app, champ, hote, chaine, pas, sortie)?;
        appareil.synchroniser()?;
        let apres = appareil.memoire_libre_octets()?;
        self.pic.set(Some(avant.saturating_sub(apres)));
        Ok(())
    }
}

#[test]
#[ignore = "GPU"]
fn o7_gpu_admission_majore_le_pic() {
    let app = exiger_gpu();
    let d = dossier("o7_gpu_admission");
    for l in [16, 24, 32] {
        let sortie = d.join(format!("l{l}"));
        let mut cfg = cfg_mutant_e1(&sortie);
        cfg.lot.l = l;
        cfg.therm = 2;
        cfg.nmeas = 2;
        cfg.skip = 1;
        cfg.dump_every = 0;
        let mut construites = kvu::obs::enligne::analyser_avec(SPEC_GPU, None)
            .expect("construction de la mesure corr");
        assert_eq!(construites.len(), 1, "une seule mesure corr attendue");
        let interne = construites.pop().expect("mesure corr");
        let estimation = interne.besoin_memoire(&cfg, true).gpu;
        let pic = Rc::new(Cell::new(None));
        let mut mesures: Vec<Box<dyn MesureEnLigne>> = vec![Box::new(MesureMemoire {
            interne,
            pic: Rc::clone(&pic),
        })];
        let (_, termine) = run::executer_avec_mesures(
            Some(&app),
            &cfg,
            &OptionsRun::default(),
            &mut mesures,
            Some(SPEC_GPU),
            false,
            None,
        )
        .expect("run de mesure du pic mémoire");
        assert!(termine, "run mémoire interrompu");
        let pic = pic.get().expect("premier appel de mesurer observé");
        let statut = if estimation >= pic { "PASS" } else { "FAIL" };
        println!(
            "\n{statut} O7-GPU : L={l}, estimation {estimation} octets, pic mesuré {pic} octets"
        );
        assert!(
            estimation >= pic,
            "FAIL O7-GPU L={l} : {estimation} < {pic}"
        );
    }
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
#[ignore = "GPU"]
fn o7_vram_insuffisante_refusee_avant_ecriture() {
    let app = exiger_gpu();
    if kvu::machine::detection::arch_memoire_unifiee(&app.arch()) {
        println!("NON APPLICABLE (APU, mémoire unifiée)");
        return;
    }
    let libre = app.memoire_libre_octets().expect("mémoire vidéo libre");
    drop(app);
    let d = dossier("o7_vram_insuffisante");
    let sortie = d.join("chaine");
    assert!(!sortie.exists(), "répertoire de sortie déjà présent");
    let o = kvu(&[
        "su2",
        "gen",
        "--backend",
        backend_gpu(),
        "--L",
        "80",
        "--beta",
        "2.3",
        "--graine",
        "1",
        "--therm",
        "1",
        "--nmeas",
        "1",
        "--skip",
        "1",
        "--mesures",
        SPEC_GPU,
        "--outdir",
        sortie.to_str().expect("chemin UTF-8 de sortie"),
    ]);
    assert_eq!(
        o.status.code(),
        Some(5),
        "refus Ressources attendu avant le calcul : {}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(
        !sortie.exists(),
        "répertoire créé malgré le refus de mémoire vidéo"
    );
    println!(
        "\nPASS O7-VRAM : L=80, {libre} octets vidéo libres, 1 refus code 5, 0 répertoire créé"
    );
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

// ---------------------------------------------------------------------------------------------
// c08d : V3 (provenance après deux reprises) et V4 (lissages comparés typés)
// ---------------------------------------------------------------------------------------------

#[test]
fn v4_lissages_equivalents_acceptes_noms_conserves() {
    let d = dossier("v4_equivalents");
    let (a, b) = paire_reprise(&d, "cpu", 4, SPEC_CPU);
    let o = reprendre("cpu", &b, &["--mesures", "corr:brut+flot/0.020/03:1"]);
    reussi(&o);
    for nom in [NOM_CSV, NOM_JSON] {
        assert_eq!(
            octets(&a.join(nom)),
            octets(&b.join(nom)),
            "{nom} diffère du run d'une traite"
        );
    }
    println!("\nPASS V4 : flot/0.020/03 accepté pour flot/0.02/3, CSV et JSON identiques à l'octet (noms enregistrés)");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
#[ignore = "GPU"]
fn v3_gpu_provenance_apres_deux_reprises() {
    drop(exiger_gpu());
    let backend = backend_gpu();
    let d = dossier("v3_provenance");
    let b = d.join("chaine");
    // run neuf arrêté à la fin de la thermalisation (pas 10), AVANT toute mesure
    reussi(&gen(
        backend,
        8,
        10,
        4,
        2,
        false,
        &b,
        &[
            "--reprise-every",
            "2",
            "--arret-apres",
            "10",
            "--mesures",
            SPEC_GPU,
        ],
    ));
    assert!(!b.join(NOM_CSV).exists(), "une mesure a déjà été faite");
    // première reprise : toutes les mesures, arrêt au dernier point (pas 18)
    reussi(&reprendre(backend, &b, &["--arret-apres", "18"]));
    assert!(!b.join("COMPLETE").exists());
    // seconde reprise : aucune mesure restante
    reussi(&reprendre(backend, &b, &[]));
    assert!(b.join("COMPLETE").exists());
    let m = lire_json(&b.join("manifest.json"));
    let n = m["modules_mesures"].as_array().map_or(0, Vec::len);
    assert!(
        n > 0,
        "FAIL V3 : modules_mesures absent après deux reprises"
    );
    println!("\nPASS V3 : {n} module(s) de mesure conservé(s) après deux reprises (arrêt avant mesure, puis au dernier point)");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}
