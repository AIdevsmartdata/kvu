//! RNG-NP : `GenerateurNumpy` contre numpy 2.5.3 (fixtures de validation/cond/gen_fixtures.py).
//! - SeedSequence, PCG64, `random()`, `standard_normal()` : identiques À L'OCTET (10⁵ tirages, graines
//!   {0, 1, 2709, 2⁶³−1}, empreinte sha256 du flux d'octets et état final du générateur) ;
//! - `hb_numpy` : même nombre de tirages (état final identique), direction `v` identique à l'octet,
//!   `x0` et `r` à quelques ULP (cb.py utilise SVML pour log1p/expm1 sur tableaux : voir numpy_rng.rs).
use kvu::cond::essais::{dossier_commite, f64_de, lire_index, u128_de, u64_de};
use kvu::cond::numpy_rng::{generate_state_u64, pool_seed_sequence};
use kvu::cond::GenerateurNumpy;
use kvu::heritage::su2_hb::sha256_hex;

fn octets(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn seed_sequence_et_etat_initial_pcg64() {
    let d = dossier_commite();
    for r in lire_index(&d).unwrap()["rng"].as_array().unwrap() {
        let graine = u64_de(&r["seed"]).unwrap();
        let pool: Vec<u32> = r["pool"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(pool_seed_sequence(graine).to_vec(), pool, "pool de SeedSequence({graine})");
        let mots: Vec<u64> = r["gen_state_u64"].as_array().unwrap().iter().map(|v| u64_de(v).unwrap()).collect();
        assert_eq!(generate_state_u64(graine, 4), mots, "generate_state({graine})");
        let g = GenerateurNumpy::depuis_graine(graine);
        assert_eq!(g.etat, u128_de(&r["etat0"]).unwrap(), "état PCG64 initial, graine {graine}");
        assert_eq!(g.inc, u128_de(&r["inc0"]).unwrap(), "incrément PCG64, graine {graine}");
    }
}

#[test]
fn random_identique_a_numpy_a_l_octet() {
    let d = dossier_commite();
    for r in lire_index(&d).unwrap()["rng"].as_array().unwrap() {
        let graine = u64_de(&r["seed"]).unwrap();
        let mut g = GenerateurNumpy::depuis_graine(graine);
        let u: Vec<f64> = (0..100_000).map(|_| g.random()).collect();
        let premiers = f64_de(&d, &format!("rng_{graine}_random_1000.npy")).unwrap();
        assert_eq!(bits(&u[..1000]), bits(&premiers), "1000 premiers random(), graine {graine}");
        assert_eq!(
            sha256_hex(&octets(&u)),
            r["sha_random"].as_str().unwrap(),
            "sha256 de 10^5 random(), graine {graine}"
        );
        assert_eq!((g.etat, g.inc), (u128_de(&r["etat_random"]).unwrap(), u128_de(&r["inc_random"]).unwrap()));
    }
}

#[test]
fn standard_normal_identique_a_numpy_a_l_octet() {
    let d = dossier_commite();
    for r in lire_index(&d).unwrap()["rng"].as_array().unwrap() {
        let graine = u64_de(&r["seed"]).unwrap();
        let mut g = GenerateurNumpy::depuis_graine(graine);
        let z: Vec<f64> = (0..100_000).map(|_| g.standard_normal()).collect();
        let premiers = f64_de(&d, &format!("rng_{graine}_normal_1000.npy")).unwrap();
        assert_eq!(bits(&z[..1000]), bits(&premiers), "1000 premières normales, graine {graine}");
        assert_eq!(
            sha256_hex(&octets(&z)),
            r["sha_normal"].as_str().unwrap(),
            "sha256 de 10^5 normales, graine {graine}"
        );
        assert_eq!((g.etat, g.inc), (u128_de(&r["etat_normal"]).unwrap(), u128_de(&r["inc_normal"]).unwrap()));
    }
}

#[test]
fn random_et_normal_entrelaces() {
    let d = dossier_commite();
    for r in lire_index(&d).unwrap()["rng"].as_array().unwrap() {
        let graine = u64_de(&r["seed"]).unwrap();
        let mut g = GenerateurNumpy::depuis_graine(graine);
        let mixte: Vec<f64> = (0..2000).map(|i| if i % 3 != 0 { g.random() } else { g.standard_normal() }).collect();
        assert_eq!(
            bits(&mixte),
            bits(&f64_de(&d, &format!("rng_{graine}_mixte_2000.npy")).unwrap()),
            "graine {graine}"
        );
        assert_eq!(g.etat, u128_de(&r["etat_mixte"]).unwrap());
    }
}

/// Ecart en ULP entre deux flottants de même signe (0 si égaux).
fn ulps(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
}

#[test]
fn hb_numpy_contre_cb_hb_numpy() {
    let d = dossier_commite();
    let idx = lire_index(&d).unwrap();
    let mut ulp_max_x = 0u64;
    let mut n_x = 0usize;
    let mut n_x_diff = 0usize;
    for h in idx["hb"].as_array().unwrap() {
        let nom = h["nom"].as_str().unwrap();
        let alpha = f64_de(&d, &format!("hb_{nom}_alpha.npy")).unwrap();
        let x0r = f64_de(&d, &format!("hb_{nom}_x0.npy")).unwrap();
        let rr = f64_de(&d, &format!("hb_{nom}_r.npy")).unwrap();
        let vr = f64_de(&d, &format!("hb_{nom}_v.npy")).unwrap();
        let mut g = GenerateurNumpy::depuis_graine(h["graine"].as_u64().unwrap());
        let (x0, r, v) = g.hb_numpy(&alpha).unwrap();
        // 1. mêmes décisions de rejet, donc même consommation du générateur
        assert_eq!(g.etat, u128_de(&h["etat"]).unwrap(), "{nom} : état PCG64 final (décisions de rejet)");
        assert_eq!(g.inc, u128_de(&h["inc"]).unwrap());
        // 2. direction : normales et normalisation exactes
        let vplat: Vec<f64> = v.iter().flatten().copied().collect();
        assert_eq!(bits(&vplat), bits(&vr), "{nom} : direction v à l'octet");
        // 3. x0 et r : quelques ULP (borne d'erreur SVML : |δx| ≤ 2e-15 ; δr ≤ |x|·δx / r + 1e-15)
        for i in 0..alpha.len() {
            n_x += 1;
            if x0[i] != x0r[i] {
                n_x_diff += 1;
            }
            ulp_max_x = ulp_max_x.max(ulps(x0[i], x0r[i]));
            assert!((x0[i] - x0r[i]).abs() <= 2e-15, "{nom}[{i}] alpha={} : x0 {} contre {}", alpha[i], x0[i], x0r[i]);
            let tol_r = 2e-15 * x0r[i].abs() / rr[i].max(1e-8) + 1e-15;
            assert!((r[i] - rr[i]).abs() <= tol_r, "{nom}[{i}] alpha={} : r {} contre {}", alpha[i], r[i], rr[i]);
        }
    }
    eprintln!("hb_numpy : {n_x} x0, {n_x_diff} différents de cb (SVML), écart maximal {ulp_max_x} ULP");
}

#[test]
fn hb_numpy_est_causal_et_enchaine() {
    // Deux appels successifs = deux appels de cb : l'état continue d'avancer.
    let mut g = GenerateurNumpy::depuis_graine(5);
    let a = g.hb_numpy(&[2.0, 5.0]).unwrap();
    let b = g.hb_numpy(&[2.0, 5.0]).unwrap();
    assert_ne!(a, b);
}
