//! RG-N3 (instrument, hors grandL) : `sd` en mode « tableau-script » sur la configuration bloquée par le script
//! `n3_blocage_beta_eff.py` (schéma `ape`, c = 0,5) = β_eff du script à 1e-10 ; WP8.
use kvu::rg::flot::sd::{beta_sd, stats_sd, ModeRapport, RapportBessel};
use kvu::rg::swendsen::plaquette_moyenne;
use kvu::stockage::npy::lire_champ;
use std::path::PathBuf;

fn donnees(nom: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/rg_flot").join(nom)
}

#[test]
fn sd_tableau_script_egale_le_script_sur_la_config_bloquee() {
    let ref_: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(donnees("n3_ref_ape_L12_b2.3_c1_cfg002000.json")).unwrap()).unwrap();
    let attendu = ref_["niveaux"]["1"]["beta_eff"].as_f64().unwrap();
    let attendu_p = ref_["niveaux"]["1"]["plaq"].as_f64().unwrap();
    let c = lire_champ(&donnees("n3_ape_niv1_L6.npy")).unwrap();
    assert_eq!(c.l, 6);
    let s = stats_sd(&c).unwrap();
    let bt = beta_sd(&s, RapportBessel::I2SurI1, ModeRapport::TableauScript).unwrap();
    let be = beta_sd(&s, RapportBessel::I2SurI1, ModeRapport::Exact).unwrap();
    let p = plaquette_moyenne(&c);
    eprintln!("script {attendu:.15} ; tableau {bt:.15} (écart {:.3e}) ; exact {be:.15} (écart au script {:.3e}) ; P {p:.15} vs {attendu_p:.15}", (bt - attendu).abs(), (be - attendu).abs());
    assert!((bt - attendu).abs() <= 1e-10);
    assert!((p - attendu_p).abs() <= 1e-10);
}
