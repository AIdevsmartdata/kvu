//! Gates RG-* du bloc de Bałaban L_b = 3 (WP6) rejouées en test : les seuils sont ceux de GATES v1 §4 et v3 §v3.3 (constantes de
//! `kvu::rg::gates`, gelées au journal avant le premier run). Les gates qui exigent les ensembles grandL (RG-DOMAINE sur L12 β 2,3 et 16,
//! RG-SWENDSEN-L12) sont rejouées par `kvu rg gates --grandl …` ; ici elles rendent SKIP, ce qui n'est pas vert et n'est pas vérifié.

use kvu::rg::gates::{self, Contexte};
use std::path::{Path, PathBuf};

#[path = "donnees/hermetique/support.rs"]
mod support;

fn racine() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn contexte(seul: &[&str]) -> Contexte {
    // seuil de la garde SSD des tests (KVU_TEST_SSD_MIN_GO) : RG-FILS écrit un champ de test sous la garde de la machine
    support::appliquer_seuil_ssd_des_tests();
    Contexte {
        fixtures: racine().join("validation/rg/fixtures"),
        donnees: racine().join("tests/donnees/rg"),
        grandl: None,
        rapports: None,
        fils: 4,
        mutants: true,
        seul: seul.iter().map(|s| s.to_string()).collect(),
    }
}

fn verdicts(seul: &[&str]) -> Vec<(String, String)> {
    let l = gates::batterie(&contexte(seul)).expect("batterie");
    l.iter().map(|x| (x.verdict.to_string(), x.id.clone())).collect()
}

fn tous_pass(seul: &[&str], attendus: &[&str]) {
    let l = verdicts(seul);
    for a in attendus {
        assert!(l.iter().any(|(v, id)| id == a && v == "PASS"), "{a} n'est pas PASS : {l:?}");
    }
    assert!(l.iter().all(|(v, _)| v == "PASS" || v == "INFO" || v == "SKIP"), "ligne FAIL : {l:?}");
}

#[test]
fn fixtures_hachees_et_reg_py_avec_mutant() {
    tous_pass(&["RG-FIXTURES", "RG-REG-PY"], &["RG-FIXTURES", "RG-REG-PY", "RG-REG-PY-MUT"]);
}

#[test]
fn jauge_pur_abel_et_domaine_instrument() {
    tous_pass(&["RG-JAUGE", "RG-PUR", "RG-ABEL"], &["RG-JAUGE", "RG-PUR", "RG-ABEL"]);
    tous_pass(
        &["RG-DOMAINE-PUR-ABEL", "RG-DOMAINE-HAAR", "RG-DOMAINE-MUT"],
        &["RG-DOMAINE-PUR-ABEL", "RG-DOMAINE-HAAR", "RG-DOMAINE-MUT"],
    );
}

#[test]
fn hyper_384_elements_et_mutant_coin() {
    tous_pass(&["RG-HYPER"], &["RG-HYPER", "RG-HYPER-MUT"]);
    // les effectifs sont 24 permutations + 360 éléments avec réflexion
    let cfg = kvu::stockage::npy::lire_champ(&racine().join("validation/rg/fixtures/cfg_s7_q64.npy")).unwrap();
    let r = gates::hyper(&cfg, kvu::rg::geometrie::Geometrie::TroisCentre, 2).unwrap();
    assert_eq!((r.n_perm, r.n_refl), (24, 360));
}

#[test]
fn swendsen_contre_block_config_sur_8_puissance_4() {
    tous_pass(&["RG-SWENDSEN-L8"], &["RG-SWENDSEN-L8"]);
}

#[test]
fn independance_au_nombre_de_fils_par_cmp() {
    tous_pass(&["RG-FILS"], &["RG-FILS-9", "RG-FILS-12"]);
}

/// Mutation de l'instrument : une fixture altérée d'un octet est détectée par RG-FIXTURES (sha256) et fait changer RG-REG-PY.
#[test]
fn fixture_alteree_est_rejetee() {
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("rg_fixture_alteree_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    for e in std::fs::read_dir(racine().join("validation/rg/fixtures")).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), tmp.join(e.file_name())).unwrap();
    }
    let f = tmp.join("ubar_s11_q64.npy");
    let mut b = std::fs::read(&f).unwrap();
    let n = b.len();
    b[n - 9] ^= 0x10; // un bit de poids faible d'une composante
    std::fs::write(&f, &b).unwrap();
    let mut ctx = contexte(&["RG-FIXTURES"]);
    ctx.fixtures = tmp.clone();
    let l = gates::batterie(&ctx).unwrap();
    assert!(l.iter().any(|x| x.id == "RG-FIXTURES" && x.verdict.to_string() == "FAIL"), "{l:?}");
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn donnees_de_test_moins_de_5_mo_et_hachees() {
    let d = racine().join("tests/donnees/rg");
    let mut total = 0u64;
    for e in std::fs::read_dir(&d).unwrap() {
        total += e.unwrap().metadata().unwrap().len();
    }
    assert!(total <= 5 * 1024 * 1024, "{total} octets");
    let sums = std::fs::read_to_string(d.join("SHA256SUMS")).unwrap();
    let mut n = 0;
    for l in sums.lines() {
        let (h, nom) = l.split_once("  ").unwrap();
        assert_eq!(kvu::heritage::manifest::sha256_fichier(&d.join(nom)).unwrap(), h, "{nom}");
        n += 1;
    }
    assert!(n >= 4);
}
