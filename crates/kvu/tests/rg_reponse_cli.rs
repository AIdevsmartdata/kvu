//! Contrats de l'instrument de réponse : JSON, déterminisme, refus avant lecture et protection de sortie.
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
struct Dossier(PathBuf);
impl Dossier {
    fn nouveau(nom: &str) -> Self {
        let p = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("rg_reponse_{nom}_{}", std::process::id()));
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
fn rc(o: &Output, n: i32) {
    assert_eq!(
        o.status.code(),
        Some(n),
        "stdout {}\nstderr {}",
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
#[test]
fn reponse_json_et_octets_identiques_selon_les_fils() {
    let d = Dossier::nouveau("fils");
    let mut docs = Vec::new();
    for n in ["1", "4"] {
        let p = d.0.join(format!("f{n}.json"));
        rc(
            &kvu(&[
                "rg",
                "reponse",
                "--entree",
                &fixture(),
                "--operateur",
                "dec",
                "--echantillon",
                "4",
                "--fils",
                n,
                "--detail",
                "--sortie",
                p.to_str().unwrap(),
            ]),
            0,
        );
        docs.push(std::fs::read(p).unwrap());
    }
    assert_eq!(docs[0], docs[1]);
    let doc: Value = serde_json::from_slice(&docs[0]).unwrap();
    assert_eq!(doc["format"], "kvu-rg-reponse/1");
    assert_eq!(doc["n_liens_fins"], 4);
    assert_eq!(doc["n_liens_grossiers"], 324);
    assert_eq!(doc["paires"].as_array().unwrap().len(), 4 * 324);
    for champ in ["par_d_p", "par_r"] {
        let n: u64 = doc[champ]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a["statistiques"]["n_paires"].as_u64().unwrap())
            .sum();
        assert_eq!(n, 4 * 324);
    }
    assert!(doc.get("fils").is_none() && doc.get("duree").is_none());
}
#[test]
fn refus_avant_lecture_et_aucun_ecrasement() {
    let d = Dossier::nouveau("refus");
    let p = d.0.join("sortie.json");
    for options in [
        vec!["--operateur", "ape:0.5:40"],
        vec!["--operateur", "dec", "--echantillon", "0"],
        vec!["--operateur", "dec", "--echantillon", "325"],
        vec!["--operateur", "dec", "--pas", "0"],
        vec!["--operateur", "dec", "--pas", "NaN"],
        vec!["--operateur", "dec", "--fils", "0"],
    ] {
        let mut a = vec![
            "rg",
            "reponse",
            "--entree",
            "absent.npy",
            "--sortie",
            p.to_str().unwrap(),
        ];
        a.extend(options);
        rc(&kvu(&a), 2);
        assert!(!p.exists());
    }
    std::fs::write(&p, b"temoin").unwrap();
    rc(
        &kvu(&[
            "rg",
            "reponse",
            "--entree",
            &fixture(),
            "--operateur",
            "dec",
            "--echantillon",
            "4",
            "--sortie",
            p.to_str().unwrap(),
        ]),
        2,
    );
    assert_eq!(std::fs::read(&p).unwrap(), b"temoin");
}
