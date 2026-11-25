//! Entrées extrêmes de la ligne de commande : une erreur de configuration (code 2), jamais un panic ni un débordement silencieux.
use std::process::Command;

fn kvu(args: &[&str]) -> (i32, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_kvu")).args(args).output().expect("lancement de kvu");
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stderr).into_owned())
}

/// `su2 thermo mesure --paire-n/--paire-m` : les dimensions `m·n` et `2·n` sont calculées avec `checked_mul` (relecture B16).
#[test]
fn paire_n_et_paire_m_extremes_refuses_en_erreur_de_configuration() {
    for (n, m) in [("18446744073709551615", "1"), ("5", "9223372036854775807"), ("18446744073709551615", "18446744073709551615")] {
        let (code, err) = kvu(&[
            "su2", "thermo", "mesure", "--paire-n", n, "--paire-m", m, "--betas", "2.3", "--repliques", "2", "--therm", "1",
            "--nmeas", "2", "--graine", "1", "--sortie", env!("CARGO_TARGET_TMPDIR"),
        ]);
        assert_eq!(code, 2, "--paire-n {n} --paire-m {m} : code 2 attendu, stderr : {err}");
        assert!(err.contains("hors de portée"), "message attendu absent : {err}");
        assert!(!err.contains("panicked"), "panic : {err}");
    }
}

/// B2-14 (seconde lecture) : chaque côté est représentable mais le VOLUME ne l'est pas (65536⁴ = 2⁶⁴ sur 64 bits) :
/// `--paire-n` et `--cotes`/`--cotes-b` donnent une erreur de configuration, pas un débordement.
#[test]
fn volume_non_representable_refuse_en_erreur_de_configuration() {
    let sortie = env!("CARGO_TARGET_TMPDIR");
    let base = ["su2", "thermo", "mesure", "--betas", "2.3", "--repliques", "2", "--therm", "1", "--nmeas", "2", "--graine", "1", "--sortie", sortie];
    let essais: [&[&str]; 2] = [&["--paire-n", "65536", "--paire-m", "1"], &["--cotes", "65536,65536,65536,65536", "--cotes-b", "4,4,4,8"]];
    for extra in essais {
        let mut a: Vec<&str> = base.to_vec();
        a.extend_from_slice(extra);
        let (code, err) = kvu(&a);
        assert_eq!(code, 2, "{extra:?} : code 2 attendu, stderr : {err}");
        assert!(err.contains("hors de portée"), "message attendu absent : {err}");
        assert!(!err.contains("panicked"), "panic : {err}");
    }
}
