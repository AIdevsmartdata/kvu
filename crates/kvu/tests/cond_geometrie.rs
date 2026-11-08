//! Géométrie de la décimation : indices IDENTIQUES à cb.Geometry (L = 4 et 8), V_pairs contre cb.V_pairs (L = 4).
use kvu::commun::ChampHote;
use kvu::cond::essais::{cas_balayages, dossier_commite, ecart_max, f64_de, indices_de};
use kvu::cond::{geometrie::v_paires, GeometrieCond};

fn verifier_geometrie(l: usize) {
    let d = dossier_commite();
    let g = GeometrieCond::nouvelle(l).unwrap();
    let np = g.npair();
    let p1 = indices_de(&d, &format!("geo_L{l}_p1.npy")).unwrap();
    let p1pm = indices_de(&d, &format!("geo_L{l}_p1pm.npy")).unwrap();
    assert_eq!(p1.len(), 4 * np);
    for mu in 0..4 {
        assert_eq!(g.p1[mu], p1[mu * np..(mu + 1) * np], "p1[{mu}] L={l}");
        assert_eq!(g.p1pm[mu], p1pm[mu * np..(mu + 1) * np], "p1pm[{mu}] L={l}");
        assert_eq!(
            g.free_all[mu],
            indices_de(&d, &format!("geo_L{l}_free_all_{mu}.npy")).unwrap(),
            "free_all[{mu}] L={l}"
        );
        for p in 0..2 {
            assert_eq!(
                g.fr[mu][p],
                indices_de(&d, &format!("geo_L{l}_fr_{mu}_{p}.npy")).unwrap(),
                "fr[{mu}][{p}] L={l}"
            );
        }
    }
    assert_eq!(g.fib_free, indices_de(&d, &format!("geo_L{l}_fib_free.npy")).unwrap(), "fib_free L={l}");
    assert_eq!(g.fib_e1, indices_de(&d, &format!("geo_L{l}_fib_e1.npy")).unwrap(), "fib_e1 L={l}");
    assert_eq!(g.fib_e2, indices_de(&d, &format!("geo_L{l}_fib_e2.npy")).unwrap(), "fib_e2 L={l}");
    let idx = kvu::cond::essais::lire_index(&d).unwrap();
    assert_eq!(g.n_fib as u64, idx["geo"][l.to_string()]["n_fib"].as_u64().unwrap(), "n_fib L={l}");
}

#[test]
fn indices_identiques_a_cb_geometry_l4() {
    verifier_geometrie(4);
}

#[test]
fn indices_identiques_a_cb_geometry_l8() {
    verifier_geometrie(8);
}

#[test]
fn v_paires_identique_a_cb_v_pairs() {
    let d = dossier_commite();
    let g = GeometrieCond::nouvelle(4).unwrap();
    let mut vus = std::collections::BTreeSet::new();
    for cas in cas_balayages(&d, Some(4)).unwrap() {
        if !vus.insert(cas.cfg.clone()) {
            continue;
        }
        let q = kvu::cond::npy_min::lire_champ_quaternions(&cas.fichier_cfg).unwrap();
        assert_eq!(q.0, 4);
        let champ = ChampHote { l: 4, q: q.1 };
        let v: Vec<f64> = v_paires(&champ, &g).iter().flatten().copied().collect();
        let attendu = f64_de(&d, cas.fichier_vp.file_name().unwrap().to_str().unwrap()).unwrap();
        let e = ecart_max(&v, &attendu);
        assert!(e <= 1e-14, "V_pairs {} : écart {e:e}", cas.cfg);
    }
    assert_eq!(vus.len(), 5, "5 configurations attendues (plate, pi, haar0-2)");
}
