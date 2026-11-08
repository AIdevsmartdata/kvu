//! Contrats CLI WP22-A2 : famille, invariants 3centre et batterie quasi-locale.
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Dossier(PathBuf);
impl Dossier {
    fn nouveau(nom: &str) -> Self {
        let p = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("rg_ql_cli_{nom}_{}", std::process::id()));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Dossier {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn kvu(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kvu"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()
        .unwrap()
}
fn rc(o: &Output, attendu: i32) {
    assert_eq!(
        o.status.code(),
        Some(attendu),
        "stdout : {}\nstderr : {}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
}
fn fixture() -> String {
    format!(
        "{}/validation/rg/fixtures/cfg_s7_q64.npy",
        env!("CARGO_MANIFEST_DIR")
    )
}
fn bloc(p: &Path, options: &[&str]) -> Output {
    let f = fixture();
    let mut a = vec![
        "rg",
        "bloc",
        "--entree",
        &f,
        "--sortie",
        p.to_str().unwrap(),
        "--fils",
        "1",
    ];
    a.extend(options);
    kvu(&a)
}
fn manifeste(p: &Path) -> PathBuf {
    kvu::rg::cli::chemin_manifeste(p)
}
fn doc(p: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}
fn verdicts(o: &Output) -> Vec<String> {
    String::from_utf8_lossy(&o.stdout)
        .lines()
        .filter(|l| ["PASS ", "FAIL ", "SKIP "].iter().any(|p| l.starts_with(p)))
        .map(str::to_string)
        .collect()
}

#[test]
fn famille_et_support_dans_le_manifeste() {
    let d = Dossier::nouveau("famille");
    let p = d.0.join("v.npy");
    rc(&bloc(&p, &["--operateur", "ape:0.5:14"]), 2);
    assert!(!p.exists() && !manifeste(&p).exists());
    assert_eq!(std::fs::read_dir(&d.0).unwrap().count(), 0);
    rc(
        &bloc(&p, &["--operateur", "ape:0.5:14", "--hors-famille"]),
        0,
    );
    let m = doc(&manifeste(&p));
    assert_eq!(m["format"], "kvu-rg-bloc-ql/1");
    assert_eq!(m["hors_famille"], true);
    assert_eq!(m["operateur"]["famille"]["dans_famille"], false);
    let p = d.0.join("ape13.npy");
    rc(&bloc(&p, &["--operateur", "ape:0.5:13"]), 0);
    assert_eq!(doc(&manifeste(&p))["support"]["n_liens_fins"], 25852);
}

#[test]
fn refus_de_geometrie_backend_et_specifications_invalides() {
    let d = Dossier::nouveau("refus");
    let p = d.0.join("refus.npy");
    for options in [
        vec!["--operateur", "dec", "--geometrie", "3coin"],
        vec!["--operateur", "dec", "--backend", "hip"],
    ] {
        rc(&bloc(&p, &options), 2);
    }
    for spec in ["ape:1:3", "flot:0.25:0.1", "inconnu"] {
        rc(&bloc(&p, &["--operateur", spec]), 2);
        rc(&bloc(&p, &["--operateur", spec, "--hors-famille"]), 2);
    }
    assert_eq!(std::fs::read_dir(&d.0).unwrap().count(), 0);
}

#[test]
fn trois_centre_explicite_conserve_les_octets() {
    let d = Dossier::nouveau("centre");
    let a = d.0.join("a.npy");
    let b = d.0.join("b.npy");
    rc(&bloc(&a, &[]), 0);
    rc(&bloc(&b, &["--operateur", "3centre"]), 0);
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    assert_eq!(
        std::fs::read(manifeste(&a)).unwrap(),
        std::fs::read(manifeste(&b)).unwrap()
    );
}

#[test]
fn flot_operateur_et_schema_incompatible() {
    let d = Dossier::nouveau("flot");
    let ens = d.0.join("ensemble");
    let chaine = ens.join("c1");
    std::fs::create_dir_all(&chaine).unwrap();
    std::fs::copy(fixture(), chaine.join("cfg_000100.npy")).unwrap();
    std::fs::write(chaine.join("manifest.json"), r#"{"beta": 2.3, "l": 9}"#).unwrap();
    let sortie = d.0.join("sortie");
    rc(
        &kvu(&[
            "rg",
            "flot",
            "--ensemble",
            ens.to_str().unwrap(),
            "--sortie",
            sortie.to_str().unwrap(),
            "--operateur",
            "ape:0.5:6",
            "--estimateurs",
            "sd",
            "--fils",
            "1",
        ]),
        0,
    );
    fn cles(v: &Value) {
        if let Some(o) = v.as_object() {
            assert!(!o.contains_key("domaine") && !o.contains_key("a_j"));
            for x in o.values() {
                cles(x);
            }
        }
        if let Some(a) = v.as_array() {
            for x in a {
                cles(x);
            }
        }
    }
    let m = doc(&sortie.join("flot.json"));
    assert!(m.get("operateur").is_some());
    cles(&m);
    let refus = d.0.join("refus");
    rc(
        &kvu(&[
            "rg",
            "flot",
            "--ensemble",
            ens.to_str().unwrap(),
            "--sortie",
            refus.to_str().unwrap(),
            "--operateur",
            "dec",
            "--schema",
            "swendsen",
        ]),
        2,
    );
    assert!(!refus.join("flot.json").exists());
}

#[test]
fn norm_trois_pass_et_verdicts_reproductibles() {
    let args = [
        "rg",
        "ql-gates",
        "--seul",
        "RG-QL-NORM",
        "--fils",
        "2",
        "--json",
    ];
    let a = kvu(&args);
    rc(&a, 0);
    let b = kvu(&args);
    rc(&b, 0);
    let lignes = verdicts(&a);
    assert_eq!(lignes.len(), 3);
    assert!(lignes.iter().all(|l| l.starts_with("PASS RG-QL-NORM-")));
    assert_eq!(lignes, verdicts(&b));
    let texte = String::from_utf8_lossy(&a.stdout);
    let json: Value = serde_json::from_str(texte.lines().last().unwrap()).unwrap();
    assert_eq!(json["pass"], 3);
    assert_eq!(json["fail"], 0);
    assert_eq!(json["skip"], 0);
}

#[test]
fn certificat_partiel_refuse_avant_toute_ecriture() {
    let d = Dossier::nouveau("certificat");
    let p = d.0.join("certificat.json");
    rc(
        &kvu(&["rg", "ql-gates", "--seul", "RG-QL-NORM", "--certifier"]),
        2,
    );
    for options in [
        vec!["--seul", "RG-QL-NORM"],
        vec!["--sans-mutants"],
        vec!["--seul", ""],
    ] {
        let mut args = vec!["rg", "ql-gates", "--certifier", p.to_str().unwrap()];
        args.extend(options);
        rc(&kvu(&args), 2);
        assert!(!p.exists());
    }
}

#[test]
fn skip_l12_erreur_interne_stable_et_erreur_fixture() {
    let o = kvu(&["rg", "ql-gates", "--seul", "RG-QL-REF-L12"]);
    rc(&o, 8);
    assert_eq!(verdicts(&o), ["SKIP RG-QL-REF-L12 --grandl absent"]);
    let d = Dossier::nouveau("erreurs");
    let mut reference = None;
    for nom in ["references_a", "references_b"] {
        let p = d.0.join(nom);
        std::fs::create_dir(&p).unwrap();
        let o = kvu(&[
            "rg",
            "ql-gates",
            "--seul",
            "RG-QL-REF-DEC",
            "--references",
            p.to_str().unwrap(),
        ]);
        rc(&o, 8);
        let lignes = verdicts(&o);
        assert_eq!(lignes.len(), 1);
        assert!(lignes[0].starts_with("FAIL RG-QL-REF-DEC "));
        assert!(!lignes[0].contains(d.0.to_str().unwrap()));
        if let Some(r) = &reference {
            assert_eq!(r, &lignes);
        } else {
            reference = Some(lignes);
        }
    }
    let absent = d.0.join("fixtures_absentes");
    let o = kvu(&[
        "rg",
        "ql-gates",
        "--seul",
        "RG-QL-REF-DEC",
        "--fixtures",
        absent.to_str().unwrap(),
    ]);
    rc(&o, 4);
    assert!(verdicts(&o).is_empty());
}

#[test]
#[ignore = "lent : batterie complète"]
fn batterie_complete_sans_grandl() {
    let o = kvu(&["rg", "ql-gates", "--fils", "6"]);
    rc(&o, 8);
    let ls = verdicts(&o);
    assert_eq!(ls.len(), 51);
    assert_eq!(ls.iter().filter(|l| l.starts_with("PASS ")).count(), 46);
    assert_eq!(
        ls.iter()
            .filter(|l| l.starts_with("SKIP ")
                && l.split_whitespace().nth(1).unwrap().ends_with("-L12"))
            .count(),
        5
    );
    let ids: std::collections::BTreeSet<_> = ls
        .iter()
        .map(|l| l.split_whitespace().nth(1).unwrap())
        .collect();
    assert_eq!(
        ids,
        kvu::rg::gates_ql::IDS_OBLIGATOIRES.into_iter().collect()
    );
}
