//! Scanner v0.3 (22/11/2026) : `su2 gen --cotes … --xi0`, refus de `--xi0` sans `--cotes`, `obs champs` sur une boîte non cubique.
//! Processeur seulement (les essais sur carte sont consignés dans `scanner-v03/RAPPORT.md`).

use std::path::{Path, PathBuf};
use std::process::Command;

fn kvu(args: &[&str], cwd: &Path) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_kvu"))
        .args(args)
        .current_dir(cwd)
        .env("KEVINOTRON_SSD_MIN_GO", "0")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .expect("lancement de kvu");
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

fn dossier(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("kvu_scanner_{nom}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn moyennes_ss_st(dir: &Path) -> (f64, f64, usize) {
    let (mut a, mut b, mut n) = (0.0, 0.0, 0usize);
    for c in std::fs::read_dir(dir).unwrap().flatten() {
        let f = c.path().join("plaq_ss_st.csv");
        if !f.is_file() {
            continue;
        }
        for l in std::fs::read_to_string(f).unwrap().lines().skip(2) {
            let v: Vec<f64> = l.split(',').skip(1).map(|x| x.parse().unwrap()).collect();
            a += v[0];
            b += v[1];
            n += 1;
        }
    }
    (a / n as f64, b / n as f64, n)
}

#[test]
fn xi0_sans_cotes_refuse() {
    let d = dossier("refus");
    let (rc, _, e) = kvu(&["su2", "gen", "--L", "4", "--xi0", "2", "--outdir", "x", "--backend", "cpu"], &d);
    assert_eq!(rc, 2, "{e}");
    assert!(e.contains("--xi0 n'existe qu'avec --cotes"), "{e}");
}

#[test]
fn cotes_backend_auto_refuse() {
    let d = dossier("auto");
    let (rc, _, e) = kvu(&["su2", "gen", "--cotes", "4,4,4,6", "--outdir", "x"], &d);
    assert_eq!(rc, 2, "{e}");
    assert!(e.contains("backend explicite"), "{e}");
}

#[test]
fn gen_xi0_processeur_et_champs_non_cubique() {
    let d = dossier("xi0");
    let (rc, _, e) = kvu(
        &["su2", "gen", "--cotes", "6,6,6,6", "--beta", "2.4", "--xi0", "2", "--graine", "1,2", "--backend", "cpu",
          "--therm", "300", "--nmeas", "150", "--outdir", "g"],
        &d,
    );
    assert_eq!(rc, 0, "{e}");
    let (ss, st, n) = moyennes_ss_st(&d.join("g"));
    assert_eq!(n, 300);
    // référence spectroscopie (β = 2,4 ; ξ₀ = 2 ; L = 6) : 0,5188 / 0,7639
    assert!((ss - 0.5188).abs() < 0.004, "ss = {ss}");
    assert!((st - 0.7639).abs() < 0.003, "st = {st}");
    let man = std::fs::read_to_string(d.join("g/manifest.json")).unwrap();
    assert!(man.contains("\"xi0\""), "ξ₀ absent du manifeste");
    let geo = std::fs::read_to_string(d.join("g/c000/geometrie.json")).unwrap();
    assert!(geo.contains("beta_plans") && geo.contains("0.5"), "poids de plans absents");
    // une boîte non cubique et anisotrope passe par obs champs
    let (rc, _, e) = kvu(
        &["su2", "gen", "--cotes", "4,6,6,8", "--beta", "2.3", "--backend", "cpu", "--therm", "20", "--nmeas", "1",
          "--dump-every", "1", "--outdir", "h"],
        &d,
    );
    assert_eq!(rc, 0, "{e}");
    let (rc, o, e) = kvu(&["obs", "champs", "--cfg", "h/cfg_000001.npy", "--out", "ch", "--porte-jauge"], &d);
    assert_eq!(rc, 0, "{o}\n{e}");
    assert!(o.contains("cotes=[4, 6, 6, 8]") && !o.contains("FAIL"), "{o}");
    let forme = std::fs::read(d.join("ch/charge_flot.npy")).unwrap();
    assert!(String::from_utf8_lossy(&forme[..128]).contains("(4, 6, 6, 8)"));
}
