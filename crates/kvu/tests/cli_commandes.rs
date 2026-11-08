//! Tests du binaire léger `kvu` : assemblage de la commande racine, aide, version, codes de
//! sortie d'usage, plans `--dry-run` (aucun GPU sollicité).

use std::process::{Command, Output};

fn kvu(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kvu"))
        .args(args)
        .output()
        .expect("lancement de kvu")
}

fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

const SOUS_COMMANDES: &[&str] = &[
    "doctor", "etat", "queue", "su2", "cond", "fibre", "obs", "store", "fetch", "valide", "bench",
];

#[test]
fn version_donne_version_source_hash_et_git_head() {
    let o = kvu(&["--version"]);
    assert_eq!(o.status.code(), Some(0));
    let s = out(&o);
    assert!(s.contains(env!("CARGO_PKG_VERSION")), "{s}");
    assert!(s.contains("source_hash (kvu-source/1) :"), "{s}");
    assert!(!s.contains("Unifiee"), "{s}");
    assert!(s.contains("git_head :"), "{s}");
    println!("PASS cli_version");
}

#[test]
fn aide_liste_les_onze_sous_commandes() {
    let o = kvu(&["--help"]);
    assert_eq!(o.status.code(), Some(0));
    let s = out(&o);
    for c in SOUS_COMMANDES {
        assert!(s.contains(c), "sous-commande {c} absente de l'aide :\n{s}");
    }
}

#[test]
fn chaque_sous_commande_a_une_aide() {
    for c in SOUS_COMMANDES {
        let o = kvu(&[c, "--help"]);
        assert_eq!(o.status.code(), Some(0), "{c} --help : {}", err(&o));
        assert!(!out(&o).is_empty(), "{c} --help vide");
    }
}

#[test]
fn usage_invalide_rend_2() {
    assert_eq!(kvu(&[]).status.code(), Some(2));
    assert_eq!(kvu(&["nimporte"]).status.code(), Some(2));
    let o = kvu(&["valide", "--option-inexistante"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(err(&o).contains("usage invalide"), "{}", err(&o));
}

#[test]
fn plan_de_bench_sans_gpu() {
    let o = kvu(&[
        "bench",
        "su2",
        "--L",
        "8,16",
        "--precision",
        "f64,f32",
        "--chaines",
        "1,16",
        "--dry-run",
    ]);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let s = out(&o);
    assert!(s.contains("plan du banc su2") && s.contains("fenêtres ≤ 600 s"), "{s}");
    assert_eq!(s.matches("  - su2 L=").count(), 8, "{s}");
    assert!(s.contains("su2 L=16 f32 K=16"));
}

#[test]
fn bench_refuse_les_parametres_invalides() {
    assert_eq!(kvu(&["bench", "su2", "--L", "7", "--dry-run"]).status.code(), Some(2));
    assert_eq!(
        kvu(&["bench", "su2", "--chaines", "0", "--dry-run"]).status.code(),
        Some(2)
    );
    assert_eq!(
        kvu(&["bench", "su2", "--budget-s", "abc", "--dry-run"]).status.code(),
        Some(2)
    );
    assert_eq!(
        kvu(&["bench", "su2", "--budget-s", "0", "--dry-run"]).status.code(),
        Some(2)
    );
    assert_eq!(
        kvu(&["bench", "cond", "--mode", "x", "--dry-run"]).status.code(),
        Some(2)
    );
    assert_eq!(kvu(&["bench"]).status.code(), Some(2));
}

#[test]
fn plans_de_bench_des_quatre_familles() {
    for (fam, motif) in [
        ("su2", "su2 L="),
        ("cond", "cond L="),
        ("fibre", "fibre L="),
        ("obs", "obs L="),
    ] {
        let o = kvu(&["bench", fam, "--L", "16", "--dry-run"]);
        assert_eq!(o.status.code(), Some(0), "{fam} : {}", err(&o));
        assert!(out(&o).contains(motif), "{fam} : {}", out(&o));
    }
}

#[test]
fn plan_de_validation_sans_rien_lancer() {
    let o = kvu(&["valide", "--complet", "--dry-run"]);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let s = out(&o);
    assert!(
        s.contains("mode     : complet") && s.contains("machines : acemagic,gamer"),
        "{s}"
    );
    // 13 gates historiques attendues sur chaque machine concernée, parmi les gates de §6
    assert!(s.contains("gates attendues :"), "{s}");
}
