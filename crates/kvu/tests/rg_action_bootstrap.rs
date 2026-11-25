//!
//! Tests du bootstrap de configurations (stratifié par chaîne) de `kvu rg action`.
//!
//! Trois familles :
//! * l'utilitaire `fixture_regenerer_jackknife` (ignore) qui écrit l'ensemble synthétique
//!   L=4, 4 chaînes x 8 configurations de Haar (beta = 0) et le `action.json` de référence
//!   du mode jackknife (celui de la binaire de base) ;
//! * le test doré : en mode `--erreurs jackknife` (défaut), la sortie est identique à la
//!   fixture, champ `version` et chemin `ensemble` normalisés ;
//! * le test statistique pré-enregistré (section 4 du job) : sigma bootstrap de beta_P sur
//!   données de Haar à L=4 contre la borne de Cramer-Rao 2/sqrt(n*1536).

use kvu::commun::ChampHote;
use kvu::heritage::su2_hb::CpuSu2;
use kvu::stockage::npy::{ecrire_champ, FormatChamp};
use kvu::stockage::GardeSsd;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const L_FIXTURE: usize = 4;
const N_CHAINES: usize = 4;
const N_PAR_CHAINE: usize = 8;
/// Graine de départ des configurations de Haar ; la configuration c,i porte la graine
/// GRAINE_FIXTURE + 977*c + i (déterministe, indépendante du nombre de fils).
const GRAINE_FIXTURE: u64 = 2026_1109;

fn racine_tmp() -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).to_path_buf()
}

fn kvu(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kvu"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()
        .unwrap()
}

fn champ_haar(l: usize, graine: u64) -> ChampHote {
    let mut gen = CpuSu2::new(l, graine).unwrap();
    gen.init(true);
    ChampHote {
        l,
        q: gen.flat(),
    }
}

fn ecrire_manifeste(dossier: &Path, beta: f64, l: usize) {
    std::fs::write(
        dossier.join("manifest.json"),
        format!("{{\"beta\": {beta}, \"l\": {l}}}"),
    )
    .unwrap();
}

/// Écrit un ensemble `parent/c1..cN` de configurations de Haar (beta = 0), L = L_FIXTURE.
fn ecrire_ensemble_haar(parent: &Path, n_chaines: usize, n_par_chaine: usize) {
    std::fs::create_dir_all(parent).unwrap();
    for c in 1..=n_chaines {
        let dir = parent.join(format!("c{c}"));
        std::fs::create_dir_all(&dir).unwrap();
        ecrire_manifeste(&dir, 0.0, L_FIXTURE);
        for i in 1..=n_par_chaine {
            let champ = champ_haar(L_FIXTURE, GRAINE_FIXTURE + 977 * c as u64 + i as u64);
            ecrire_champ(
                &dir.join(format!("cfg_{i:06}.npy")),
                &champ,
                FormatChamp::Q64,
                &GardeSsd::nouvelle(0.0),
            )
            .unwrap();
        }
    }
}

/// Normalise un action.json : le chemin de l'ensemble et la version sont remplacés.
fn normaliser(doc: &mut Value) {
    let obj = doc.as_object_mut().unwrap();
    obj.insert("ensemble".into(), Value::String("<ENSEMBLE>".into()));
    obj.insert("version".into(), Value::String("<VERSION>".into()));
}

fn lire_json(chemin: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(chemin).unwrap()).unwrap()
}

fn chemin_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/donnees/rg_action/action_jackknife_L4.json")
}

/// Utilitaire (ignore) : régénère la fixture dorée du mode jackknife.
/// À lancer UNIQUEMENT avec la binaire de base (avant patch) :
/// `cargo test --release --test rg_action_bootstrap fixture_regenerer -- --ignored`
#[test]
#[ignore = "régénère la fixture dorée ; à exécuter avec la binaire de base"]
fn fixture_regenerer_jackknife() {
    let parent = racine_tmp().join("rg_boot_fixture_src");
    let _ = std::fs::remove_dir_all(&parent);
    ecrire_ensemble_haar(&parent, N_CHAINES, N_PAR_CHAINE);
    let sortie = racine_tmp().join("rg_boot_fixture_sortie_base");
    let _ = std::fs::remove_dir_all(&sortie);
    std::fs::create_dir_all(&sortie).unwrap();
    let out = kvu(&[
        "rg",
        "action",
        "--ensemble",
        parent.to_str().unwrap(),
        "--famille",
        "P",
        "--estimateurs",
        "pl,sd",
        "--sortie",
        sortie.to_str().unwrap(),
        "--fils",
        "1",
    ]);
    assert!(
        out.status.success(),
        "binaire de base en échec : {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut doc = lire_json(&sortie.join("action.json"));
    normaliser(&mut doc);
    let fixture = chemin_fixture();
    std::fs::create_dir_all(fixture.parent().unwrap()).unwrap();
    std::fs::write(
        &fixture,
        serde_json::to_string_pretty(&doc).unwrap() + "\n",
    )
    .unwrap();
    println!("fixture écrite : {}", fixture.display());
}


/// Lance `kvu rg action` sur `parent` (famille P) et renvoie le document JSON.
fn lancer_action(parent: &Path, sortie: &Path, estimateurs: &str, extra: &[&str]) -> (Value, f64) {
    let _ = std::fs::remove_dir_all(sortie);
    std::fs::create_dir_all(sortie).unwrap();
    let mut args = vec![
        "rg", "action", "--ensemble", parent.to_str().unwrap(), "--famille", "P",
        "--estimateurs", estimateurs, "--sortie", sortie.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    let t0 = std::time::Instant::now();
    let out = kvu(&args);
    let dt = t0.elapsed().as_secs_f64();
    assert!(out.status.success(), "kvu en échec : {}", String::from_utf8_lossy(&out.stderr));
    (lire_json(&sortie.join("action.json")), dt)
}

/// (a) Test doré (ajouté par le général, rejeu du job kvu-rg-action-bootstrap-3) : en mode par défaut (jackknife) la sortie
/// normalisée est IDENTIQUE à la fixture produite par la binaire de base. Échoue dès que la sortie jackknife dérive.
#[test]
fn dore_jackknife_identique_a_la_base() {
    let parent = racine_tmp().join("rg_boot_dore_src");
    let _ = std::fs::remove_dir_all(&parent);
    ecrire_ensemble_haar(&parent, N_CHAINES, N_PAR_CHAINE);
    let (mut doc, _) = lancer_action(&parent, &racine_tmp().join("rg_boot_dore_sortie"), "pl,sd", &["--fils", "1"]);
    normaliser(&mut doc);
    let obtenu = serde_json::to_string_pretty(&doc).unwrap() + "\n";
    let attendu = std::fs::read_to_string(chemin_fixture()).unwrap();
    assert!(obtenu == attendu, "la sortie jackknife a dérivé de la fixture de base");
}

/// (b) Test préenregistré (job kvu-rg-action-bootstrap-2, §PRE-REGISTERED) : Haar L = 4, famille P, 4 x 24 = 96 configurations,
/// B = 200, bloc 1, graine 20261109. PASS ssi |sigma_boot/sigma_CR - 1| <= 0,25, >= 190 réplicats valides, et deux exécutions à même
/// graine identiques. sigma_CR = 1/sqrt(384 n) (Var(1/2 Tr U) = 1/4 sous Haar ; variance sandwich de la PL à beta = 0 = borne de Fisher).
#[test]
fn bootstrap_haar_plancher_cramer_rao() {
    let parent = racine_tmp().join("rg_boot_cr_src");
    let _ = std::fs::remove_dir_all(&parent);
    ecrire_ensemble_haar(&parent, 4, 24);
    let opts = ["--fils", "4", "--erreurs", "bootstrap", "--bootstrap-b", "200", "--bootstrap-bloc", "1", "--graine", "20261109"];
    let (mut d1, t1) = lancer_action(&parent, &racine_tmp().join("rg_boot_cr_s1"), "pl", &opts);
    let (mut d2, _) = lancer_action(&parent, &racine_tmp().join("rg_boot_cr_s2"), "pl", &opts);
    let (dj, _) = lancer_action(&parent, &racine_tmp().join("rg_boot_cr_jk"), "pl", &["--fils", "4"]);
    let s = d1["pl"]["couplages"]["P"]["erreur"].as_f64().expect("erreur bootstrap de beta_P");
    let nrep = d1["pl"]["couplages"]["P"]["replicats"].as_array().map(|a| a.len()).unwrap_or(0);
    let sj = dj["pl"]["couplages"]["P"]["erreur"].as_f64().unwrap();
    let cr = 1.0 / (384.0f64 * 96.0).sqrt();
    println!("MESURE sigma_boot={s:.6e} sigma_jk4={sj:.6e} sigma_CR={cr:.6e} rapport_boot={:.4} rapport_jk={:.4} replicats={nrep} temps_B200_s={t1:.2}",
             s / cr, sj / cr);
    println!("BLOC_BOOTSTRAP {}", d1["bootstrap"]);
    normaliser(&mut d1);
    normaliser(&mut d2);
    assert!(d1 == d2, "deux exécutions à même graine diffèrent");
    assert!(nrep >= 190, "réplicats valides : {nrep}");
    assert!((s / cr - 1.0).abs() <= 0.25, "sigma_boot/sigma_CR = {}", s / cr);
}
