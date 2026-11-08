//! `kvu rg` par la ligne de commande (WP6) : aide, refus d'usage, bloc d'une fixture avec manifeste, identité à l'octet selon les fils,
//! mutant `3coin` distinct, gates partielles.

use std::process::{Command, Output};

fn kvu(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kvu")).args(args).output().expect("lancement de kvu")
}

fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn fixture() -> String {
    format!("{}/validation/rg/fixtures/cfg_s11_q64.npy", env!("CARGO_MANIFEST_DIR"))
}

fn dossier(nom: &str) -> std::path::PathBuf {
    let d = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("rg_cli_{nom}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn aide_de_rg_et_de_ses_sous_commandes() {
    for args in [vec!["rg", "--help"], vec!["rg", "bloc", "--help"], vec!["rg", "swendsen", "--help"], vec!["rg", "domaine", "--help"], vec!["rg", "gates", "--help"]] {
        let o = kvu(&args);
        assert_eq!(o.status.code(), Some(0), "{args:?} : {}", err(&o));
        assert!(!out(&o).is_empty());
    }
}

#[test]
fn usages_invalides_rendent_2() {
    assert_eq!(kvu(&["rg"]).status.code(), Some(2));
    let d = dossier("usage");
    let s = d.join("s.npy");
    let s = s.to_str().unwrap();
    let o = kvu(&["rg", "bloc", "--entree", &fixture(), "--sortie", s, "--geometrie", "4coin"]);
    assert_eq!(o.status.code(), Some(2), "{}", err(&o));
    let o = kvu(&["rg", "bloc", "--entree", &fixture(), "--sortie", s, "--fils", "0"]);
    assert_eq!(o.status.code(), Some(2), "{}", err(&o));
    // L = 8 n'est pas un multiple de 3 : refus (Erreur::Config, code 2), aucune sortie écrite
    let o = kvu(&["rg", "bloc", "--entree", &format!("{}/tests/donnees/rg/cfg_su2_L8_b2.40_s24001.npy", env!("CARGO_MANIFEST_DIR")), "--sortie", s]);
    assert_eq!(o.status.code(), Some(2), "{}", err(&o));
    assert!(!d.join("s.npy").exists());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn bloc_ecrit_sortie_et_manifeste_identiques_selon_les_fils() {
    let d = dossier("fils");
    let mut sorties = Vec::new();
    for fils in ["1", "4", "16"] {
        let s = d.join(format!("s_{fils}.npy"));
        let o = kvu(&["rg", "bloc", "--entree", &fixture(), "--sortie", s.to_str().unwrap(), "--fils", fils]);
        assert_eq!(o.status.code(), Some(0), "{}", err(&o));
        assert!(out(&o).contains("L = 9 → 3"), "{}", out(&o));
        let man = std::fs::read(format!("{}.manifeste.json", s.display())).unwrap();
        sorties.push((std::fs::read(&s).unwrap(), man));
    }
    assert!(sorties.iter().all(|x| *x == sorties[0]), "sorties différentes selon les fils");
    let m: serde_json::Value = serde_json::from_slice(&sorties[0].1).unwrap();
    assert_eq!(m["format"], "kvu-rg-bloc/1");
    assert_eq!(m["pas_de_bloc"], 3);
    assert_eq!(m["geometrie"], "3centre");
    assert_eq!(m["n_boucles_par_lien"], 10473);
    assert_eq!(m["l_sortie"], 3);
    assert_eq!(m["theta"]["n_boucles"], 324 * 10473);
    assert_eq!(m["theta"]["histogramme"].as_array().unwrap().len(), 64);
    assert_eq!(m["sha256_entree"], kvu::heritage::manifest::sha256_fichier(std::path::Path::new(&fixture())).unwrap());
    assert_eq!(m["sha256_sortie"], kvu::heritage::manifest::sha256_hex(&sorties[0].0));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn le_mutant_3coin_donne_une_autre_sortie() {
    let d = dossier("coin");
    let (a, b) = (d.join("a.npy"), d.join("b.npy"));
    assert_eq!(kvu(&["rg", "bloc", "--entree", &fixture(), "--sortie", a.to_str().unwrap(), "--fils", "2"]).status.code(), Some(0));
    assert_eq!(kvu(&["rg", "bloc", "--entree", &fixture(), "--sortie", b.to_str().unwrap(), "--fils", "2", "--geometrie", "3coin"]).status.code(), Some(0));
    assert_ne!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    let m: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{}.manifeste.json", b.display())).unwrap()).unwrap();
    assert_eq!(m["geometrie"], "3coin");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn gates_partielles_lignes_bien_formees_et_certificat_refuse() {
    let racine = env!("CARGO_MANIFEST_DIR");
    let o = kvu(&[
        "rg", "gates", "--fixtures", &format!("{racine}/validation/rg/fixtures"), "--donnees", &format!("{racine}/tests/donnees/rg"),
        "--seul", "RG-PUR,RG-ABEL", "--json",
    ]);
    let s = out(&o);
    assert_eq!(o.status.code(), Some(0), "{s}\n{}", err(&o));
    for id in ["RG-PUR", "RG-ABEL"] {
        assert!(s.lines().any(|l| l.starts_with(&format!("PASS {id} "))), "{s}");
    }
    let j: serde_json::Value = serde_json::from_str(s.lines().find(|l| l.starts_with('{')).unwrap()).unwrap();
    assert_eq!(j["fail"], 0);
    // --certifier exige la batterie complète
    let o = kvu(&["rg", "gates", "--seul", "RG-PUR", "--certifier"]);
    assert_eq!(o.status.code(), Some(2), "{}", err(&o));
}

#[test]
fn domaine_rapporte_sans_jamais_echouer() {
    let o = kvu(&["rg", "domaine", "--entree", &fixture(), "--fils", "2"]);
    let s = out(&o);
    assert_eq!(o.status.code(), Some(0), "{s}\n{}", err(&o));
    assert!(s.contains("INFO RG-DOMAINE"), "{s}");
    assert!(s.contains("HORS_DOMAINE") || s.contains("DANS_DOMAINE"), "{s}");
}

#[test]
fn swendsen_c_nul_et_pas_invalide() {
    let d = dossier("sw");
    let s = d.join("s.npy");
    let entree = format!("{}/tests/donnees/rg/cfg_su2_L8_b2.40_s24001.npy", env!("CARGO_MANIFEST_DIR"));
    let o = kvu(&["rg", "swendsen", "--entree", &entree, "--sortie", s.to_str().unwrap(), "--c", "0.5", "--pas", "2"]);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    assert!(out(&o).contains("pas 2 → L = 2"), "{}", out(&o));
    let o = kvu(&["rg", "swendsen", "--entree", &entree, "--sortie", s.to_str().unwrap(), "--pas", "0"]);
    assert_eq!(o.status.code(), Some(2));
    let _ = std::fs::remove_dir_all(&d);
}
