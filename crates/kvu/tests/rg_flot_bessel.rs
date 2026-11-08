//! Rapports de Bessel exacts contre `mpmath` (60 chiffres) : GATES v3.3, seuil 1e-14 (relatif) ; WP8.
use kvu::rg::flot::bessel::{i1_sur_i0, i2_sur_i1};
use std::path::PathBuf;

fn table() -> Vec<(f64, f64, f64)> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/rg_flot/bessel_ratios.json");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    v["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| (e["x"].as_f64().unwrap(), e["r21"].as_f64().unwrap(), e["r10"].as_f64().unwrap()))
        .collect()
}

#[test]
fn rapports_contre_mpmath_a_1e_14() {
    let t = table();
    assert!(t.len() > 100);
    let (mut m21, mut m10) = (0.0_f64, 0.0_f64);
    for (x, r21, r10) in &t {
        m21 = m21.max(((i2_sur_i1(*x) - r21) / r21).abs());
        m10 = m10.max(((i1_sur_i0(*x) - r10) / r10).abs());
    }
    eprintln!("max relatif I2/I1 = {m21:e}, I1/I0 = {m10:e} sur {} points", t.len());
    assert!(m21 <= 1e-14, "I2/I1 : {m21:e}");
    assert!(m10 <= 1e-14, "I1/I0 : {m10:e}");
}

#[test]
fn monotonie_et_bornes() {
    let mut prec = 0.0;
    for i in 1..2000 {
        let x = i as f64 * 0.05;
        let r = i2_sur_i1(x);
        assert!(r > prec && r < 1.0);
        prec = r;
    }
}
