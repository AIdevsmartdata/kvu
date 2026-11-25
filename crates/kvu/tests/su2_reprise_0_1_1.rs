//! Test demandé par le pilote (11/11/2026, B08) : un run CRÉÉ par le candidat public 0.1.1 (témoin
//! `tests/donnees/reprise_0_1_1/`, interrompu proprement au point de reprise, `--mesures wilson:R2xT2:1`) se reprend avec le
//! binaire 0.2, sans repasser `--mesures` (spécification rechargée de `mesures_en_ligne.json`) : sorties identiques à l'octet
//! au run d'une traite fait par le binaire 0.2 avec les mêmes paramètres. CPU seulement.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn kvu(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kvu"))
        .current_dir(dir)
        .args(args)
        .output()
        .expect("lancement de kvu")
}

fn dossier(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("su2_reprise_0_1_1_{nom}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).expect("création du répertoire temporaire");
    d
}

fn reussi(o: &Output) {
    assert_eq!(
        o.status.code(),
        Some(0),
        "kvu en échec :\n{}\n{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
}

#[test]
fn run_cree_par_la_0_1_1_repris_par_la_0_2() {
    let d = dossier("temoin");
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/reprise_0_1_1");
    let run = d.join("run");
    fs::create_dir_all(&run).unwrap();
    for e in fs::read_dir(&source).unwrap() {
        let e = e.unwrap();
        if e.file_name() != "LISEZMOI.md" {
            let copie = run.join(e.file_name());
            fs::copy(e.path(), &copie).unwrap();
            // c13b : `fs::copy` recopie les droits de la source ; un dépôt en lecture seule (export de recette) rendrait le
            // témoin non inscriptible, alors que la reprise doit y écrire.
            fs::set_permissions(&copie, fs::Permissions::from_mode(0o644)).unwrap();
        }
    }
    assert!(
        run.join("mesures_en_ligne.json").exists() && !run.join("COMPLETE").exists(),
        "témoin incomplet"
    );
    let o = kvu(
        &d,
        &["su2", "gen", "--backend", "cpu", "--reprendre", "run"],
    );
    reussi(&o);
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("rechargée"),
        "spécification non rechargée"
    );
    assert!(
        run.join("COMPLETE").exists(),
        "COMPLETE absent après reprise"
    );
    let reference = dossier("reference");
    reussi(&kvu(
        &reference,
        &[
            "su2",
            "gen",
            "--backend",
            "cpu",
            "--L",
            "4",
            "--beta",
            "2.3",
            "--graine",
            "26111112",
            "--start",
            "hot",
            "--therm",
            "6",
            "--nmeas",
            "6",
            "--skip",
            "2",
            "--reprise-every",
            "2",
            "--mesures",
            "wilson:R2xT2:1",
            "--outdir",
            "run",
        ],
    ));
    for f in ["wilson.csv", "measurements.csv", "therm.csv"] {
        assert_eq!(
            fs::read(run.join(f)).unwrap(),
            fs::read(reference.join("run").join(f)).unwrap(),
            "{f} diffère du run d'une traite"
        );
    }
    assert_eq!(
        fs::read_to_string(run.join("wilson.csv"))
            .unwrap()
            .lines()
            .count(),
        1 + 6 * 4
    );
    println!("\nPASS REPRISE-0.1.1→0.2 : témoin 0.1.1 repris sans --mesures, 3 fichiers identiques au run d'une traite (wilson 25 lignes)");
    let _ = fs::remove_dir_all(&d);
    let _ = fs::remove_dir_all(&reference);
}
