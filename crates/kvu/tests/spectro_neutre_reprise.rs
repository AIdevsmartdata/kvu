//! `su2 spectro neutre --reprendre` avec impulsions HORS AXE (`--impulsions-vecteurs`) : un VRAI redémarrage de
//! processeur (sous-processus tué en cours de mesure, puis relancé avec `--reprendre`) doit rendre des sorties
//! identiques AU BIT à une exécution continue, y compris `neutre_o<k1>_<k2>_<k3>.bin`.
//!
//! Contrat vérifié ici :
//! 1. exécution continue de référence (dossier `d1`) ;
//! 2. exécution coupée (dossier `d2`) : le sous-processus est tué (SIGKILL) dès que la ligne `neutre: cfg 100/400`
//!    apparaît sur stderr, donc après l'écriture de `etat.bin`, des fichiers d'orbite et de `plaq.csv` ;
//! 3. reprise `--reprendre` dans `d2` : succès, stderr contient `neutre: cfg {n}/400` avec n = nmeas (elle est allée
//!    au bout) ;
//! 4. comparaison octet à octet de `etat.bin`, `neutre.bin`, `neutre_o2_1_0.bin`, `neutre_o1_1_1.bin`, `plaq.csv`
//!    (`manifeste.json` n'est PAS comparé : il est écrit par l'exécution coupée et n'est pas réécrit par la reprise) ;
//! 5. refus : `--reprendre` sur un dossier sans `etat.bin` (code 4, E/S) ; fichier d'orbite plus court que
//!    `neutre.bin` (code 2, configuration) ; dossier déjà rempli sans `--reprendre` (code 2, configuration).
//!
//! Réseau minuscule (6⁴, 400 configurations, ~1,5 s par exécution en `--release`) : la suite tient en quelques secondes.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Code de retour d'une erreur d'usage (`Erreur::Config`).
const RC_USAGE: i32 = 2;
/// Code de retour d'une erreur d'entrée/sortie (`Erreur::Io`).
const RC_ES: i32 = 4;
/// Configuration mesurée : la ligne de progression `neutre: cfg N/400` sert de point de coupure.
const NMEAS: u64 = 400;
/// Configuration à laquelle l'exécution coupée est tuée.
const CFG_COUPE: u64 = 100;
/// Impulsions hors axe de l'essai (orbites distinctes du groupe cubique).
const IMPULSIONS: &str = "2:1:0,1:1:1";
/// Fichiers comparés au bit entre exécution continue et reprise (ordre de la conception).
const FICHIERS: [&str; 5] = [
    "etat.bin",
    "neutre.bin",
    "neutre_o2_1_0.bin",
    "neutre_o1_1_1.bin",
    "plaq.csv",
];

fn dossier_test(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("kvu_spectro_neutre_reprise_{nom}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Arguments communs de l'essai ; `sortie` est le dossier de sortie, `reprendre` ajoute `--reprendre`.
fn args(sortie: &Path, reprendre: bool) -> Vec<String> {
    let mut a: Vec<String> = [
        "su2",
        "spectro",
        "neutre",
        "--cotes",
        "6,6,6,6",
        "--beta",
        "2.3",
        "--graine",
        "5",
        "--therm",
        "20",
        "--nmeas",
        "400",
        "--skip",
        "1",
        "--nor",
        "1",
        "--niveaux",
        "1,2",
        "--alpha",
        "0.5",
        "--tau-max",
        "3",
        "--impulsions-vecteurs",
        IMPULSIONS,
        "--sortie",
    ]
    .iter()
    .map(|x| x.to_string())
    .collect();
    a.push(s(sortie));
    if reprendre {
        a.push("--reprendre".into());
    }
    a
}

/// Sous-processus `kvu` isolé (variables KEVINOTRON_* / KVU_* retirées, seuils RAM et SSD nuls, 2 fils).
fn commande(a: &[String]) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_kvu"));
    for (k, _) in std::env::vars().filter(|(k, _)| k.starts_with("KEVINOTRON_") || k.starts_with("KVU_")) {
        c.env_remove(k);
    }
    c.env("KEVINOTRON_VERROU", "ignorer")
        .env("KEVINOTRON_RAM_MIN_GO", "0")
        .env("KEVINOTRON_SSD_MIN_GO", "0")
        .env("RAYON_NUM_THREADS", "2")
        .args(a)
        .stdin(Stdio::null());
    c
}

/// Exécution complète ; rend (code, stdout, stderr). Échoue si le délai est dépassé.
fn lancer(a: &[String], delai_s: u64) -> (i32, String, String) {
    let mut enfant = commande(a).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let lire = |mut f: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut b = Vec::new();
            let _ = f.read_to_end(&mut b);
            String::from_utf8_lossy(&b).into_owned()
        })
    };
    let t_out = lire(Box::new(enfant.stdout.take().unwrap()));
    let t_err = lire(Box::new(enfant.stderr.take().unwrap()));
    let debut = Instant::now();
    let rc = loop {
        if let Some(st) = enfant.try_wait().unwrap() {
            break st.code().unwrap_or(-1);
        }
        if debut.elapsed() > Duration::from_secs(delai_s) {
            let _ = enfant.kill();
            let _ = enfant.wait();
            panic!("kvu {:?} : délai de {delai_s} s dépassé", a);
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    (rc, t_out.join().unwrap(), t_err.join().unwrap())
}

/// Lance `kvu` et le tue (SIGKILL) dès que stderr porte la ligne `neutre: cfg {cfg}/{NMEAS}` : le processus meurt
/// donc en cours de mesure, après avoir écrit `etat.bin`, les fichiers d'orbite et `plaq.csv` jusqu'à `cfg`.
/// Rend le stderr lu jusque-là.
fn lancer_et_tuer(a: &[String], cfg: u64, delai_s: u64) -> String {
    let mut enfant = commande(a).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().unwrap();
    let err = enfant.stderr.take().unwrap();
    let mut lecteur = BufReader::new(err);
    let motif = format!("neutre: cfg {cfg}/{NMEAS}");
    let mut vu = String::new();
    let debut = Instant::now();
    let mut ligne = String::new();
    loop {
        ligne.clear();
        let n = lecteur.read_line(&mut ligne).unwrap();
        if n == 0 {
            let _ = enfant.wait();
            panic!("kvu {:?} : terminé avant la ligne « {motif} » ; stderr = {vu}", a);
        }
        vu.push_str(&ligne);
        if ligne.contains(&motif) {
            break;
        }
        if debut.elapsed() > Duration::from_secs(delai_s) {
            let _ = enfant.kill();
            let _ = enfant.wait();
            panic!("kvu {:?} : ligne « {motif} » absente après {delai_s} s ; stderr = {vu}", a);
        }
    }
    // SIGKILL : aucune chance que l'enfant écrive un octet de plus après ce point.
    enfant.kill().unwrap();
    let _ = enfant.wait();
    vu
}

/// Comparaison octet à octet ; le message nomme le fichier et le premier octet divergent.
fn comparer(a: &Path, b: &Path) {
    let x = std::fs::read(a).unwrap_or_else(|e| panic!("lecture {} : {e}", a.display()));
    let y = std::fs::read(b).unwrap_or_else(|e| panic!("lecture {} : {e}", b.display()));
    assert_eq!(
        x.len(),
        y.len(),
        "{} : {} octets contre {} pour {}",
        a.file_name().unwrap().to_string_lossy(),
        x.len(),
        y.len(),
        b.display()
    );
    if let Some(i) = x.iter().zip(y.iter()).position(|(p, q)| p != q) {
        panic!(
            "{} : premier octet divergent à l'offset {i} ({} contre {})",
            a.file_name().unwrap().to_string_lossy(),
            x[i],
            y[i]
        );
    }
}

fn taille(p: &Path) -> u64 {
    std::fs::metadata(p).unwrap_or_else(|e| panic!("{} : {e}", p.display())).len()
}

#[test]
fn reprise_hors_axe_identique_au_bit() {
    let racine = dossier_test("identique");
    let d1 = racine.join("d1");
    let d2 = racine.join("d2");

    // 1. référence : exécution continue.
    let (rc, out, err) = lancer(&args(&d1, false), 900);
    assert_eq!(rc, 0, "exécution de référence en échec ; stdout = {out} ; stderr = {err}");
    assert!(
        err.contains(&format!("neutre: cfg {NMEAS}/{NMEAS}")),
        "référence : progression finale absente ; stderr = {err}"
    );
    for f in FICHIERS {
        let p = d1.join(f);
        assert!(p.is_file(), "référence : {} manquant", p.display());
        assert!(taille(&p) > 0, "référence : {} vide", p.display());
    }

    // 2. exécution coupée en cours de mesure.
    let err_coupe = lancer_et_tuer(&args(&d2, false), CFG_COUPE, 900);
    assert!(
        err_coupe.contains(&format!("neutre: cfg {CFG_COUPE}/{NMEAS}")),
        "coupure : ligne de progression absente ; stderr = {err_coupe}"
    );
    assert!(
        !err_coupe.contains(&format!("neutre: cfg {NMEAS}/{NMEAS}")),
        "coupure : l'exécution est allée au bout, le test ne prouve rien ; stderr = {err_coupe}"
    );
    for f in ["etat.bin", "neutre.bin", "neutre_o2_1_0.bin", "neutre_o1_1_1.bin", "plaq.csv", "manifeste.json"] {
        let p = d2.join(f);
        assert!(p.is_file(), "coupure : {} manquant (rien à reprendre)", p.display());
    }
    // L'exécution coupée est bien partielle : ses fichiers sont plus courts que ceux de la référence.
    for f in ["neutre.bin", "neutre_o2_1_0.bin", "neutre_o1_1_1.bin"] {
        assert!(
            taille(&d2.join(f)) < taille(&d1.join(f)),
            "coupure : {} n'est pas partiel ({} octets contre {})",
            f,
            taille(&d2.join(f)),
            taille(&d1.join(f))
        );
    }
    let manifeste_coupe = std::fs::read(d2.join("manifeste.json")).unwrap();

    // 3. reprise.
    let (rc, out, err) = lancer(&args(&d2, true), 900);
    assert_eq!(rc, 0, "reprise en échec ; stdout = {out} ; stderr = {err}");
    assert!(
        err.contains(&format!("neutre: cfg {NMEAS}/{NMEAS}")),
        "reprise : elle n'est pas allée au bout ; stderr = {err}"
    );

    // 4. égalité au bit, fichier par fichier.
    for f in FICHIERS {
        comparer(&d1.join(f), &d2.join(f));
    }

    // 5. le manifeste de la reprise est celui de l'exécution coupée (non réécrit).
    assert_eq!(
        std::fs::read(d2.join("manifeste.json")).unwrap(),
        manifeste_coupe,
        "reprise : manifeste.json a été réécrit"
    );
}

#[test]
fn refus_reprise_sans_etat() {
    let racine = dossier_test("sans_etat");
    let d = racine.join("vide");
    let (rc, out, err) = lancer(&args(&d, true), 900);
    assert_eq!(rc, RC_ES, "reprise sans etat.bin : code {rc} ; stdout = {out} ; stderr = {err}");
    assert!(
        err.contains("etat.bin"),
        "reprise sans etat.bin : le message ne nomme pas etat.bin ; stderr = {err}"
    );
    assert!(
        !d.join("neutre.bin").exists(),
        "reprise sans etat.bin : des sorties ont été écrites"
    );
}

#[test]
fn refus_fichier_orbite_tronque() {
    let racine = dossier_test("orbite_court");
    let d1 = racine.join("d1");
    let d2 = racine.join("d2");
    let (rc, out, err) = lancer(&args(&d1, false), 900);
    assert_eq!(rc, 0, "exécution de référence en échec ; stdout = {out} ; stderr = {err}");
    copier_arbre(&d1, &d2);
    // `neutre_o2_1_0.bin` recoupé sous la taille de `neutre.bin` : la reprise doit refuser avant d'écrire.
    let court = d2.join("neutre_o2_1_0.bin");
    let f = std::fs::OpenOptions::new().write(true).open(&court).unwrap();
    f.set_len(taille(&d2.join("neutre.bin")) / 2).unwrap();
    drop(f);
    let avant = std::fs::read(d2.join("neutre.bin")).unwrap();
    let (rc, out, err) = lancer(&args(&d2, true), 900);
    assert_eq!(rc, RC_USAGE, "orbite tronquée : code {rc} ; stdout = {out} ; stderr = {err}");
    assert!(
        err.contains("neutre_o2_1_0.bin") && err.contains("reprise impossible"),
        "orbite tronquée : message inattendu ; stderr = {err}"
    );
    assert_eq!(
        std::fs::read(d2.join("neutre.bin")).unwrap(),
        avant,
        "orbite tronquée : neutre.bin a été modifié malgré le refus"
    );
}

#[test]
fn refus_dossier_deja_rempli_sans_reprendre() {
    let racine = dossier_test("deja_rempli");
    let d = racine.join("d");
    let (rc, out, err) = lancer(&args(&d, false), 900);
    assert_eq!(rc, 0, "première exécution en échec ; stdout = {out} ; stderr = {err}");
    let avant = std::fs::read(d.join("neutre.bin")).unwrap();
    let (rc, out, err) = lancer(&args(&d, false), 900);
    assert_eq!(rc, RC_USAGE, "dossier rempli sans --reprendre : code {rc} ; stdout = {out} ; stderr = {err}");
    assert!(
        err.contains("existe déjà") && err.contains("--reprendre"),
        "dossier rempli : message inattendu ; stderr = {err}"
    );
    assert_eq!(
        std::fs::read(d.join("neutre.bin")).unwrap(),
        avant,
        "dossier rempli : neutre.bin a été modifié malgré le refus"
    );
}

/// Copie récursive (fichiers seulement) de `src` vers `dst` (créé).
fn copier_arbre(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        let p = e.path();
        if p.is_file() {
            std::fs::copy(&p, dst.join(e.file_name())).unwrap();
        }
    }
}
