//! Gates d'instrument FIB-JAC, FIB-LOGDET (+ CT-9), FIB-CHOL (GATES v4-T2 partie 1 §1.1). Lourdes : `#[ignore]`, à lancer avec
//! `cargo test --release --lib fibre_contrainte::gates_jacobien -- --ignored --nocapture --test-threads=1`.

use super::chol::{cholesky_dense, logdet_lu, CholBlocs};
use super::gates_commun::*;
use super::stat::maxn;
use super::jacobien::{tests::{perturber, pool}, Jac, Mutant, Reglages, Reseau};
use crate::commun::ChampHote;
use crate::stockage::npy::lire_champ;

const NDIR: u32 = 20;
const FILS: usize = 8;

fn configs_jac() -> Vec<(String, ChampHote)> {
    let mut v = Vec::new();
    for graine in [7, 11, 13] {
        let c = lire_champ(&dossier_fixtures().join(format!("cfg_s{graine}_q64.npy"))).unwrap();
        v.push((format!("fixture9^4 s{graine}"), c));
    }
    v.push(("L=6 Haar (graine 26102612101)".into(), champ_haar(6, 26_102_612)));
    v.push(("L=6 Wilson beta=2.8 (graine 26102612102, 2000 pas)".into(), champ_wilson(6, 2.8, 26_102_612_102, 2000)));
    for (nom, c) in &v {
        ligne("INFO", "EMPREINTE", &format!("{nom} : sha256(octets) {}", empreinte(c)));
    }
    v
}

/// max sur les directions de ‖J v − Richardson‖₂ / ‖J v‖₂ pour la configuration donnée.
fn ecart_jac(res: &Reseau, champ: &ChampHote, reg: &Reglages, ndir: u32) -> f64 {
    let p = pool(FILS);
    let jac = Jac::evaluer(res, champ, &p, reg).unwrap();
    let ub0 = ubar_pool(res, champ, FILS);
    let mut pire: f64 = 0.0;
    for k in 0..ndir {
        let d = direction(3 * res.n_liens, 26_102_612_200 + k as u64, k);
        let jv = jac.jvp(&d);
        let fd = richardson3_vec(
            |h| {
                let up = ubar_pool(res, &perturber(champ, &d, h), FILS);
                let um = ubar_pool(res, &perturber(champ, &d, -h), FILS);
                let (a, b) = (delta_omega(&up, &ub0), delta_omega(&um, &ub0));
                a.iter().zip(&b).map(|(x, y)| (x - y) / (2.0 * h)).collect()
            },
            1e-3,
        );
        let num = fd.iter().zip(&jv).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
        let den = jv.iter().map(|x| x * x).sum::<f64>().sqrt();
        pire = maxn(pire, num / den);
    }
    pire
}

#[test]
#[ignore = "gate lourde FIB-JAC"]
fn fib_jac() {
    if sauter_si_sans_wp11("FIB-JAC") {
        return;
    }
    let mut pire: f64 = 0.0;
    let mut pire_mut: f64 = 0.0;
    let configs = configs_jac();
    for (nom, c) in &configs {
        let res = Reseau::nouveau(c.l).unwrap();
        let e = ecart_jac(&res, c, &Reglages::default(), NDIR);
        ligne("INFO", "FIB-JAC", &format!("{nom} : max ‖Δ‖₂/‖Jv‖₂ = {e:.3e} sur {NDIR} directions"));
        pire = maxn(pire, e);
    }
    // mutant « dérivée de log = identité » sur les configurations L = 6
    for (nom, c) in configs.iter().skip(3) {
        let res = Reseau::nouveau(c.l).unwrap();
        let e = ecart_jac(&res, c, &Reglages { mutant: Mutant::LIdentite }, 4);
        ligne("INFO", "FIB-JAC-MUTANT", &format!("{nom} : écart {e:.3e}"));
        pire_mut = maxn(pire_mut, e);
    }
    ligne(if pire <= 1e-8 { "PASS" } else { "FAIL" }, "FIB-JAC", &format!("max {pire:.3e} ≤ 1e-8 (Richardson h = 1e-3, h/2, h/4 ; {NDIR} directions ; 3 fixtures 9⁴ + L=6 Haar + L=6 Wilson β=2,8)"));
    ligne(if pire_mut >= 1e-3 { "PASS" } else { "FAIL" }, "FIB-JAC-MUTANT", &format!("mutant « dérivée de log = identité » : écart {pire_mut:.3e} ≥ 1e-3 (ROUGE exigé)"));
    assert!(pire <= 1e-8 && pire_mut >= 1e-3);
}

/// ½ log det(J Jᵀ) par Cholesky creux par blocs.
fn demi_logdet(res: &Reseau, champ: &ChampHote, p: &rayon::ThreadPool) -> f64 {
    let jac = Jac::evaluer(res, champ, p, &Reglages::default()).unwrap();
    0.5 * CholBlocs::factoriser(&jac.jjt_blocs()).unwrap().log_det()
}

/// (max sur les directions de |∇Φ·v − FD| / |FD| (relatif par direction, métrique du verdict), max de |∇Φ·v − FD| / ‖∇Φ_exact‖,
/// ‖∇Φ_exact‖). Le gradient du mutant est comparé aux mêmes différences finies ; le dénominateur de normalisation est celui du gradient EXACT.
fn ecarts_logdet(res: &Reseau, champ: &ChampHote, reg: &Reglages, ndir: u32) -> (f64, f64, f64) {
    let p = pool(FILS);
    let jac = Jac::evaluer(res, champ, &p, &Reglages::default()).unwrap();
    let chol = CholBlocs::factoriser(&jac.jjt_blocs()).unwrap();
    let ginv = chol.inverse_dense(&p);
    let w = jac.w_depuis_ginv(&ginv, &p);
    let g_exact = jac.grad_demi_logdet(res, champ, &w, &p, &Reglages::default());
    let g = jac.grad_demi_logdet(res, champ, &w, &p, reg);
    let gn = g_exact.iter().map(|x| x * x).sum::<f64>().sqrt();
    let (mut rel_dir, mut rel_norme): (f64, f64) = (0.0, 0.0);
    for k in 0..ndir {
        let d = direction(3 * res.n_liens, 26_102_612_300 + k as u64, k);
        let fd = richardson3(|h| (demi_logdet(res, &perturber(champ, &d, h), &p) - demi_logdet(res, &perturber(champ, &d, -h), &p)) / (2.0 * h), 1e-3);
        let an: f64 = g.iter().zip(&d).map(|(a, b)| a * b).sum();
        let r = if fd.is_finite() && fd.abs() > 1e-12 * gn { (an - fd).abs() / fd.abs() } else { f64::NAN };
        rel_dir = maxn(rel_dir, r);
        rel_norme = maxn(rel_norme, (an - fd).abs() / gn);
    }
    (rel_dir, rel_norme, gn)
}

#[test]
#[ignore = "gate lourde FIB-LOGDET"]
fn fib_logdet() {
    if sauter_si_sans_wp11("FIB-LOGDET") {
        return;
    }
    let mut pire: f64 = 0.0;
    let configs = configs_jac();
    for (nom, c) in &configs {
        let res = Reseau::nouveau(c.l).unwrap();
        let (e, en, gn) = ecarts_logdet(&res, c, &Reglages::default(), NDIR);
        ligne("INFO", "FIB-LOGDET", &format!("{nom} : max |∇Φ·v − FD| / |FD| = {e:.3e} (rapporté à ‖∇Φ‖ : {en:.3e} ; ‖∇Φ‖ = {gn:.3e}) sur {NDIR} directions"));
        pire = maxn(pire, e);
    }
    let ok = pire <= 1e-6;
    ligne(if ok { "PASS" } else { "FAIL" }, "FIB-LOGDET", &format!("max {pire:.3e} ≤ 1e-6 (écart relatif par direction |∇Φ·v − FD|/|FD| ; 20 directions ; mêmes configurations que FIB-JAC)"));
    // mutant de la gate : les trois familles de dérivées retirées ensemble (≥ 1e-3 exigé) ; les mutants partiels sont rapportés (INFO)
    let (nom, c) = &configs[4];
    let res = Reseau::nouveau(c.l).unwrap();
    for m in [Mutant::SansDerivE, Mutant::SansDerivL, Mutant::SansDerivR] {
        let (e, _, _) = ecarts_logdet(&res, c, &Reglages { mutant: m }, 6);
        ligne("INFO", "FIB-LOGDET-MUTANT-PARTIEL", &format!("{nom} : mutant {m:?} écart relatif par direction {e:.3e} (non exigé)"));
    }
    let (e, _, _) = ecarts_logdet(&res, c, &Reglages { mutant: Mutant::SansDerivees }, 6);
    let rouge = e >= 1e-3;
    ligne(if rouge { "PASS" } else { "FAIL" }, "FIB-LOGDET-MUTANT", &format!("{nom} : mutant « dérivées de J (∂R, ∂L_g, ∂E_g) retirées » écart relatif par direction {e:.3e} ≥ 1e-3 (ROUGE exigé)"));
    assert!(ok && rouge);
}

#[test]
#[ignore = "gate lourde CT-9"]
fn ct9_invariance_de_jauge() {
    if sauter_si_sans_wp11("CT-9") {
        return;
    }
    let p = pool(FILS);
    let mut pire: f64 = 0.0;
    for (nom, c) in configs_jac() {
        let res = Reseau::nouveau(c.l).unwrap();
        let (a, b) = (demi_logdet(&res, &c, &p), demi_logdet(&res, &jauge_de_haar(&c, 26_102_612), &p));
        let e = 2.0 * (a - b).abs(); // écart sur log det
        ligne("INFO", "CT-9", &format!("{nom} : log det = {:.12}, écart de jauge {e:.3e}", 2.0 * a));
        pire = maxn(pire, e);
    }
    ligne(if pire <= 1e-12 { "PASS" } else { "FAIL" }, "CT-9", &format!("écart de log det sous jauge de Haar : {pire:.3e} ≤ 1e-12"));
    assert!(pire <= 1e-12);
}

#[test]
#[ignore = "gate lourde FIB-CHOL"]
fn fib_chol() {
    let p = pool(FILS);
    let c = champ_wilson(6, 2.8, 26_102_612_103, 2000);
    ligne("INFO", "EMPREINTE", &format!("FIB-CHOL : U0 sha256 {}", empreinte(&c)));
    let res = Reseau::nouveau(6).unwrap();
    let jac = Jac::evaluer(&res, &c, &p, &Reglages::default()).unwrap();
    let blocs = jac.jjt_blocs();
    let chol = CholBlocs::factoriser(&blocs).unwrap();
    let n = 3 * res.n_c;
    let dense = jac.jjt_dense();
    let ld = cholesky_dense(&dense, n).unwrap();
    let lc = chol.facteur_dense();
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for i in 0..n * n {
        num += (ld[i] - lc[i]) * (ld[i] - lc[i]);
        den += ld[i] * ld[i];
    }
    let rel = (num / den).sqrt();
    let lu = logdet_lu(&dense, n).unwrap();
    let rel_ld = (chol.log_det() - lu).abs() / lu.abs();
    ligne(if rel <= 1e-12 { "PASS" } else { "FAIL" }, "FIB-CHOL-FACTEUR", &format!("facteur creux contre dense (Frobenius relatif) {rel:.3e} ≤ 1e-12 (permutation : identité, {} blocs non nuls sur {})", chol.nb_blocs_non_nuls(), res.n_c * (res.n_c + 1) / 2));
    ligne(if rel_ld <= 1e-10 { "PASS" } else { "FAIL" }, "FIB-CHOL-LOGDET", &format!("log det creux contre LU dense à pivot partiel : écart relatif {rel_ld:.3e} ≤ 1e-10 (log det = {:.10})", chol.log_det()));
    assert!(rel <= 1e-12 && rel_ld <= 1e-10);
}
