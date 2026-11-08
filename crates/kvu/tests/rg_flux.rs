//! Tests d'acceptation du module `kvu rg flux` (M3).
//! Les champs et fonds abéliens sont construits dans les tests ; seules les
//! fixtures du dépôt sont utilisées pour la comparaison avec la référence Python.

use kvu::commun::{ChampHote, Erreur};
use kvu::rg::flux::faces::{
    angle_face, boucle_face, flux_coherent_faces, fond_abelien, grossier, stats_coherent,
    stats_faces, x_faces, Pavage, VarianteFlux,
};
use kvu::rg::operateur::Operateur;
use kvu::stockage::npy::lire_champ;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn racine_depot() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn fixture(rel: &str) -> PathBuf {
    racine_depot().join(rel)
}

fn champ_froid(l: usize) -> ChampHote {
    ChampHote::froid(l).unwrap()
}

fn champ_haar(l: usize, graine: u32) -> ChampHote {
    let v = l.pow(4);
    let mut q = vec![0.0; 16 * v];
    for i in 0..(4 * v) as u32 {
        let quat = kvu::su2::cpu::haar_q(i, 0, 0, graine, graine.wrapping_add(0x9e37_79b9));
        let o = 4 * i as usize;
        q[o..o + 4].copy_from_slice(&quat);
    }
    ChampHote { l, q }
}

fn coordonnees(mut s: usize, l: usize) -> [usize; 4] {
    let mut x = [0; 4];
    for m in (0..4).rev() {
        x[m] = s % l;
        s /= l;
    }
    x
}

fn site(x: [usize; 4], l: usize) -> usize {
    ((x[0] * l + x[1]) * l + x[2]) * l + x[3]
}

fn deplacer(mut x: [usize; 4], mu: usize, delta: isize, l: usize) -> [usize; 4] {
    x[mu] = (x[mu] as isize + delta).rem_euclid(l as isize) as usize;
    x
}

fn appliquer_jauge(champ: &ChampHote, graine: u32) -> ChampHote {
    let l = champ.l;
    let v = champ.v();
    let mut g = vec![[1.0_f64, 0.0, 0.0, 0.0]; v];
    for (s, gs) in g.iter_mut().enumerate() {
        *gs = kvu::su2::cpu::haar_q(s as u32, 1, 1, graine, graine.wrapping_add(0x6a09_e667));
    }
    let mut q = champ.q.clone();
    for mu in 0..4 {
        for s in 0..v {
            let x = coordonnees(s, l);
            let x_mu = deplacer(x, mu, 1, l);
            let s_mu = site(x_mu, l);
            let u = kvu::heritage::su2_hb::qmul(
                kvu::heritage::su2_hb::qmul(g[s], champ.lien(mu * v + s)),
                kvu::heritage::su2_hb::qdag(g[s_mu]),
            );
            let o = 4 * (mu * v + s);
            q[o..o + 4].copy_from_slice(&u);
        }
    }
    ChampHote { l, q }
}

/// Repli de la spécification : repli(t) = t si t ≤ 2π, 4π − t sinon.
fn repli(t: f64) -> f64 {
    if t <= 2.0 * std::f64::consts::PI {
        t
    } else {
        4.0 * std::f64::consts::PI - t
    }
}

/// T1 — Abélien brut et cohérent : TOUTES les faces (x, Θ_atan, Φ_coh, C).
#[test]
fn t1_abelien_brut_et_coherent() {
    let cas_plan01 = [
        (12, 1, (0, 1)),
        (12, 2, (0, 1)),
        (12, 3, (0, 1)),
        (12, 5, (0, 1)),
        (18, 14, (0, 1)),
        (18, 16, (0, 1)),
        (18, 17, (0, 1)),
        (18, 19, (0, 1)),
    ];
    let cas_extras = [
        (12, 1, (1, 3)),
        (12, 2, (1, 3)),
        (12, 3, (1, 3)),
        (12, 5, (1, 3)),
    ];
    for (l, m, plan) in cas_plan01.iter().chain(cas_extras.iter()) {
        let (champ, phi) = fond_abelien(*l, *m, *plan).unwrap();
        let v = champ.v();
        let p_actif = kvu::rg::flux::PLANS
            .iter()
            .position(|&(a, b)| (a, b) == *plan)
            .unwrap();

        for n in 1..=3 {
            let x = x_faces(&champ, n, 4).unwrap();
            let phi_n2 = (n * n) as f64 * phi;
            for p in 0..6 {
                for s in 0..v {
                    let idx = p * v + s;
                    let x_att = if p == p_actif {
                        (phi_n2 / 2.0).cos()
                    } else {
                        1.0
                    };
                    let tol = if p == p_actif { 1e-12 } else { 1e-15 };
                    assert!(
                        (x[idx] - x_att).abs() <= tol,
                        "L={l} m={m} plan={plan:?} n={n} face {p}/{s} : x={} attendu {x_att}",
                        x[idx]
                    );

                    let w = boucle_face(&champ, s, p, n);
                    let angle = angle_face(w);
                    let angle_att = if p == p_actif { repli(phi_n2) } else { 0.0 };
                    assert!(
                        (angle - angle_att).abs() <= 1e-12,
                        "L={l} m={m} n={n} face {p}/{s} : Θ_atan={angle} attendu {angle_att}"
                    );
                }
            }
            // Comptes exacts
            let stats = stats_faces(&champ, n, Pavage::Tous, false, 4).unwrap();
            let x_att_plan = (phi_n2 / 2.0).cos();
            let faces_plan = v as u64;
            let mut comptes_att = [0u64; 5];
            for (i, (_, seuil, strict)) in kvu::rg::flux::SEUILS_X.iter().enumerate() {
                let cond = if *strict {
                    x_att_plan < *seuil
                } else {
                    x_att_plan <= *seuil
                };
                if cond {
                    comptes_att[i] = faces_plan;
                }
            }
            assert_eq!(stats.comptes, comptes_att, "L={l} m={m} n={n} comptes");
        }

        // Cohérent n=3
        let (phi_vals, sabs_vals, _ambig) =
            flux_coherent_faces(&champ, 3, VarianteFlux::Normale, 4).unwrap();
        let phi_att = 9.0 * phi;
        for p in 0..6 {
            for s in 0..v {
                let idx = p * v + s;
                let att = if p == p_actif { phi_att } else { 0.0 };
                let tol = if p == p_actif { 1e-11 } else { 1e-14 };
                assert!(
                    (phi_vals[idx] - att).abs() <= tol,
                    "cohérent L={l} m={m} face {p}/{s} : Φ={} attendu {att}",
                    phi_vals[idx]
                );
                if p == p_actif {
                    let coh = phi_vals[idx] / sabs_vals[idx];
                    assert!((coh - 1.0).abs() <= 1e-12, "C face {p}/{s}");
                }
            }
        }
        let stats_coh = stats_coherent(&champ, 3, Pavage::Tous, VarianteFlux::Normale, 4).unwrap();
        let mut comptes_coh_att = [0u64; 4];
        for (j, (_, seuil, strict)) in kvu::rg::flux::SEUILS_COH.iter().enumerate() {
            let cond = if *strict {
                phi_att > *seuil
            } else {
                phi_att >= *seuil
            };
            if cond {
                comptes_coh_att[j] = v as u64;
            }
        }
        assert_eq!(stats_coh.comptes, comptes_coh_att);
    }
}

/// T2 — Abélien grossier : valeurs x de chaque plaquette grossière et pavage.
#[test]
fn t2_abelien_grossier_et_pavage() {
    for m in [14, 16, 17, 19] {
        let (champ, phi) = fond_abelien(18, m, (0, 1)).unwrap();
        let op = Operateur::TroisCentre;
        let champ_gr = grossier(&champ, &op, 1, 4).unwrap();
        let v_gr = champ_gr.v();
        let p_actif = 0; // plan (0,1)
        let x_att = (9.0 * phi / 2.0).cos();
        // Vérification des valeurs x par face
        let x_gr = x_faces(&champ_gr, 1, 4).unwrap();
        for p in 0..6 {
            for s in 0..v_gr {
                let idx = p * v_gr + s;
                let attendu = if p == p_actif { x_att } else { 1.0 };
                let tol = 1e-12;
                assert!(
                    (x_gr[idx] - attendu).abs() <= tol,
                    "grossier m={m} face {p}/{s} : x={} attendu {attendu}",
                    x_gr[idx]
                );
            }
        }
        // Comptes
        let stats_gr = stats_faces(&champ_gr, 1, Pavage::Tous, false, 4).unwrap();
        let faces_gr_plan = v_gr as u64;
        let mut comptes_att = [0u64; 5];
        for (i, (_, seuil, strict)) in kvu::rg::flux::SEUILS_X.iter().enumerate() {
            let cond = if *strict {
                x_att < *seuil
            } else {
                x_att <= *seuil
            };
            if cond {
                comptes_att[i] = faces_gr_plan;
            }
        }
        assert_eq!(stats_gr.comptes, comptes_att, "grossier m={m}");
    }

    for m in [1, 5] {
        let (champ, phi) = fond_abelien(12, m, (0, 1)).unwrap();
        let op = Operateur::TroisCentre;
        let champ_gr = grossier(&champ, &op, 1, 4).unwrap();
        let v_gr = champ_gr.v();
        let p_actif = 0;
        let x_att = (9.0 * phi / 2.0).cos();
        let x_gr = x_faces(&champ_gr, 1, 4).unwrap();
        for p in 0..6 {
            for s in 0..v_gr {
                let idx = p * v_gr + s;
                let attendu = if p == p_actif { x_att } else { 1.0 };
                assert!(
                    (x_gr[idx] - attendu).abs() <= 1e-12,
                    "grossier L=12 m={m} face {p}/{s}"
                );
            }
        }
        let stats_gr = stats_faces(&champ_gr, 1, Pavage::Tous, false, 4).unwrap();
        let faces_gr_plan = v_gr as u64;
        let mut comptes_att = [0u64; 5];
        for (i, (_, seuil, strict)) in kvu::rg::flux::SEUILS_X.iter().enumerate() {
            let cond = if *strict {
                x_att < *seuil
            } else {
                x_att <= *seuil
            };
            if cond {
                comptes_att[i] = faces_gr_plan;
            }
        }
        assert_eq!(stats_gr.comptes, comptes_att);
    }

    // Pavage 1 sur brut L=18 m=17
    let (champ, phi) = fond_abelien(18, 17, (0, 1)).unwrap();
    let stats_pavage = stats_faces(&champ, 3, Pavage::Blocs(1), false, 4).unwrap();
    assert_eq!(stats_pavage.n, 6 * 6u64.pow(4));
    let faces_bloc_plan = (18u64 / 3).pow(4);
    let x_att = (9.0 * phi / 2.0).cos();
    let mut comptes_att = [0u64; 5];
    for (i, (_, seuil, strict)) in kvu::rg::flux::SEUILS_X.iter().enumerate() {
        let cond = if *strict {
            x_att < *seuil
        } else {
            x_att <= *seuil
        };
        if cond {
            comptes_att[i] = faces_bloc_plan;
        }
    }
    assert_eq!(stats_pavage.comptes, comptes_att);
}

/// T3 — Froid et Haar : x = 1 sur chaque face froide, moyenne C ≈ 1/3, T1 ≈ 0,5.
#[test]
fn t3_froid_et_haar() {
    let froid = champ_froid(6);
    let x_froid = x_faces(&froid, 3, 1).unwrap();
    for &xi in &x_froid {
        assert!((xi - 1.0).abs() <= 1e-15, "x froid");
    }
    let stats_brut = stats_faces(&froid, 3, Pavage::Tous, false, 1).unwrap();
    let stats_coh = stats_coherent(&froid, 3, Pavage::Tous, VarianteFlux::Normale, 1).unwrap();
    assert_eq!(stats_brut.comptes, [0; 5]);
    assert_eq!(stats_coh.s_phi, 0.0);
    assert!((stats_coh.s_coh - stats_coh.n as f64).abs() <= 1e-12 * stats_coh.n as f64);

    let haar = champ_haar(6, 2611020301);
    let stats_haar_coh = stats_coherent(&haar, 3, Pavage::Tous, VarianteFlux::Normale, 2).unwrap();
    let coh_moy = stats_haar_coh.s_coh / stats_haar_coh.n as f64;
    assert!(coh_moy > 0.25 && coh_moy < 0.45, "coh_moy={coh_moy}");

    let stats_haar_brut = stats_faces(&haar, 3, Pavage::Tous, false, 2).unwrap();
    let f_t1 = stats_haar_brut.comptes[0] as f64 / stats_haar_brut.n as f64;
    assert!(f_t1 > 0.45 && f_t1 < 0.55, "f_T1={f_t1}");
}

/// Compare un StatsFaces calculé à la référence JSON, histogrammes inclus
/// (dans les DEUX sens), finitudes vérifiées.
fn comparison_stats_faces(calc: &kvu::rg::flux::faces::StatsFaces, ref_json: &Value) {
    let n_ref = ref_json["N"].as_u64().unwrap();
    assert_eq!(calc.n, n_ref);
    for f in ["sx", "sx2", "sTheta2", "sTheta4", "xmin"] {
        assert!(
            ref_json[f].as_f64().unwrap().is_finite(),
            "ref non finie : {f}"
        );
    }
    assert!([calc.sx, calc.sx2, calc.s_theta2, calc.s_theta4, calc.xmin]
        .into_iter()
        .all(f64::is_finite));
    let comptes_ref = ref_json["comptes"].as_object().unwrap();
    assert_eq!(comptes_ref.len(), kvu::rg::flux::SEUILS_X.len());
    for (i, (nom, _, _)) in kvu::rg::flux::SEUILS_X.iter().enumerate() {
        assert_eq!(
            calc.comptes[i],
            comptes_ref[*nom].as_u64().unwrap(),
            "{nom}"
        );
    }
    let rel = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1e-30);
    assert!(rel(calc.sx, ref_json["sx"].as_f64().unwrap()), "sx");
    assert!(rel(calc.sx2, ref_json["sx2"].as_f64().unwrap()), "sx2");
    assert!(
        rel(calc.s_theta2, ref_json["sTheta2"].as_f64().unwrap()),
        "sTheta2"
    );
    assert!(
        rel(calc.s_theta4, ref_json["sTheta4"].as_f64().unwrap()),
        "sTheta4"
    );
    // xmin : tolérance 1e-13 (arbitrage du chef de projet, écart résiduel c16 → quaternion).
    assert!(
        (calc.xmin - ref_json["xmin"].as_f64().unwrap()).abs() <= 1e-13,
        "xmin"
    );
    // Histogramme bidirectionnel
    if let Some(hist_ref) = ref_json
        .get("hist_alpha_nonvides")
        .and_then(|v| v.as_object())
    {
        let hist_calc = calc.hist_alpha.as_ref().expect("hist absent");
        // 1) classes non vides de la référence
        for (i, count) in hist_ref {
            let idx: usize = i.parse().unwrap();
            assert_eq!(hist_calc[idx], count.as_u64().unwrap(), "hist {i}");
        }
        // 2) pas de classes non vides en plus
        for (i, &c) in hist_calc.iter().enumerate() {
            if c > 0 {
                let ref_count = hist_ref
                    .get(&i.to_string())
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                assert_eq!(c, ref_count, "classe supplémentaire {i}");
            }
        }
    }
}

fn comparison_stats_coherent(calc: &kvu::rg::flux::faces::StatsCoherent, ref_json: &Value) {
    let n_ref = ref_json["N"].as_u64().unwrap();
    assert_eq!(calc.n, n_ref);
    for f in ["s_phi", "s_phi2", "s_coh", "phi_max"] {
        assert!(
            ref_json[f].as_f64().unwrap().is_finite(),
            "ref non finie : {f}"
        );
    }
    assert!([calc.s_phi, calc.s_phi2, calc.s_coh, calc.phi_max]
        .into_iter()
        .all(f64::is_finite));
    let comptes_ref = ref_json["comptes"].as_object().unwrap();
    assert_eq!(comptes_ref.len(), kvu::rg::flux::SEUILS_COH.len());
    for (j, (nom, _, _)) in kvu::rg::flux::SEUILS_COH.iter().enumerate() {
        assert_eq!(
            calc.comptes[j],
            comptes_ref[*nom].as_u64().unwrap(),
            "{nom}"
        );
    }
    if let Some(joint_ref) = ref_json
        .get("comptes_coherence")
        .and_then(|v| v.as_object())
    {
        assert_eq!(joint_ref.len(), kvu::rg::flux::SEUILS_COH.len());
        for (j, (nom, _, _)) in kvu::rg::flux::SEUILS_COH.iter().enumerate() {
            if let Some(cmin_obj) = joint_ref.get(*nom).and_then(|v| v.as_object()) {
                assert_eq!(cmin_obj.len(), kvu::rg::flux::COH_MIN.len());
                for (k, (cmin, _)) in kvu::rg::flux::COH_MIN.iter().enumerate() {
                    assert_eq!(
                        calc.comptes_coherence[j][k],
                        cmin_obj[*cmin].as_u64().unwrap(),
                        "{nom}/{cmin}"
                    );
                }
            } else {
                panic!("comptes_coherence manquant pour {nom}");
            }
        }
    } else {
        panic!("comptes_coherence absent");
    }
    let rel = |a: f64, b: f64| (a - b).abs() <= 1e-11 * b.abs().max(1e-30);
    assert!(
        rel(calc.s_phi, ref_json["s_phi"].as_f64().unwrap()),
        "s_phi"
    );
    assert!(
        rel(calc.s_phi2, ref_json["s_phi2"].as_f64().unwrap()),
        "s_phi2"
    );
    assert!(
        rel(calc.s_coh, ref_json["s_coh"].as_f64().unwrap()),
        "s_coh"
    );
    assert!(
        (calc.phi_max - ref_json["phi_max"].as_f64().unwrap()).abs() <= 1e-12,
        "phi_max"
    );
}

/// T4 — Référence Python, avec histogrammes bidirectionnels.
#[test]
fn t4_reference_python() {
    let doc: Value = serde_json::from_str(
        &std::fs::read_to_string(fixture("validation/rg_flux/reference_py.json")).unwrap(),
    )
    .unwrap();
    let fixtures = doc["fixtures"].as_array().unwrap();
    for entree in fixtures {
        let fichier = entree["fichier"].as_str().unwrap().to_string();
        let chemin_candidats = [
            fixture(&format!("tests/donnees/rg/{fichier}")),
            fixture(&format!("validation/rg/fixtures/{fichier}")),
        ];
        let chemin = chemin_candidats
            .iter()
            .find(|p| p.exists())
            .expect("fixture manquante");
        let champ = lire_champ(chemin).unwrap();

        if let Some(brut) = entree.get("brut").and_then(|v| v.as_object()) {
            for (cle, ref_stats) in brut {
                if let Some(n_str) = cle.strip_prefix('n') {
                    if let Ok(n) = n_str.parse::<usize>() {
                        let stats_calc = stats_faces(&champ, n, Pavage::Tous, true, 4).unwrap();
                        comparison_stats_faces(&stats_calc, ref_stats);
                    }
                }
            }
        }
        if let Some(coh) = entree.get("coherent").and_then(|v| v.as_object()) {
            for (cle, ref_stats) in coh {
                if let Some(n_str) = cle.strip_prefix('n') {
                    if let Ok(n) = n_str.parse::<usize>() {
                        let stats_calc =
                            stats_coherent(&champ, n, Pavage::Tous, VarianteFlux::Normale, 4)
                                .unwrap();
                        comparison_stats_coherent(&stats_calc, ref_stats);
                    }
                }
            }
        }
    }
}

/// T5 — Jauge et mutant SansTransport.
#[test]
fn t5_jauge_et_mutant() {
    let champ_base = champ_haar(6, 2611020301);
    let champ_jauge = appliquer_jauge(&champ_base, 2611020302);
    for n in [1, 2, 3] {
        let x_base = x_faces(&champ_base, n, 2).unwrap();
        let x_jauge = x_faces(&champ_jauge, n, 2).unwrap();
        let max_dx = x_base
            .iter()
            .zip(x_jauge.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_dx <= 1e-13, "max_dx n={n}");
        let stats_base = stats_faces(&champ_base, n, Pavage::Tous, false, 2).unwrap();
        let stats_jauge = stats_faces(&champ_jauge, n, Pavage::Tous, false, 2).unwrap();
        assert_eq!(stats_base.comptes, stats_jauge.comptes);
    }
    let (phi_base, _, _) = flux_coherent_faces(&champ_base, 3, VarianteFlux::Normale, 2).unwrap();
    let (phi_jauge, _, _) = flux_coherent_faces(&champ_jauge, 3, VarianteFlux::Normale, 2).unwrap();
    let max_dphi = phi_base
        .iter()
        .zip(phi_jauge.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(max_dphi <= 1e-12, "max_dphi");

    let (phi_mut_base, _, _) =
        flux_coherent_faces(&champ_base, 3, VarianteFlux::SansTransport, 2).unwrap();
    let (phi_mut_jauge, _, _) =
        flux_coherent_faces(&champ_jauge, 3, VarianteFlux::SansTransport, 2).unwrap();
    let max_mut = phi_mut_base
        .iter()
        .zip(phi_mut_jauge.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(max_mut > 1e-6, "mutant SansTransport attendu grand");
}

/// T6 — Grossier calculé = fixture ubar_s7_q64.
#[test]
fn t6_grossier_calcule_egal_fixture() {
    let cfg = fixture("validation/rg/fixtures/cfg_s7_q64.npy");
    let ubar = fixture("validation/rg/fixtures/ubar_s7_q64.npy");
    let champ_cfg = lire_champ(&cfg).unwrap();
    let champ_ubar = lire_champ(&ubar).unwrap();
    let op = Operateur::TroisCentre;
    let champ_gr = grossier(&champ_cfg, &op, 1, 4).unwrap();
    let stats_gr = stats_faces(&champ_gr, 1, Pavage::Tous, false, 4).unwrap();
    let stats_ubar = stats_faces(&champ_ubar, 1, Pavage::Tous, false, 4).unwrap();
    assert_eq!(stats_gr.comptes, stats_ubar.comptes);
    let rel = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1e-30);
    assert!(rel(stats_gr.sx, stats_ubar.sx));
    assert!(rel(stats_gr.sx2, stats_ubar.sx2));
    assert!(rel(stats_gr.s_theta2, stats_ubar.s_theta2));
    assert!(rel(stats_gr.s_theta4, stats_ubar.s_theta4));
}

/// Helper : exécute la sous-commande `flux` avec les arguments donnés.
fn executer_flux(args: &[&str]) -> Result<i32, Erreur> {
    let sous = kvu::rg::flux::cli::sous_commandes();
    let cmd_flux = sous
        .iter()
        .find(|c| c.get_name() == "flux")
        .unwrap()
        .clone();
    let matches = cmd_flux
        .try_get_matches_from(args)
        .map_err(|e| Erreur::Config(e.to_string()))?;
    kvu::rg::flux::cli::executer("flux", &matches)
}

/// Chaque test ne nettoie que le dossier qu'il a lui-même créé.
struct DossierTest(PathBuf);

impl DossierTest {
    fn nouveau(nom: &str) -> Self {
        static NUMERO: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let numero = NUMERO.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let chemin =
            std::env::temp_dir().join(format!("rg_flux_{nom}_{}_{numero}", std::process::id()));
        std::fs::create_dir(&chemin).unwrap();
        Self(chemin)
    }

    fn fichier(&self, nom: &str) -> PathBuf {
        self.0.join(nom)
    }
}

impl Drop for DossierTest {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn texte_jsonl(lignes: &[Value]) -> String {
    let mut texte = String::new();
    for ligne in lignes {
        texte.push_str(&ligne.to_string());
        texte.push('\n');
    }
    texte
}

fn reference_fixture<'a>(reference: &'a Value, nom: &str) -> &'a Value {
    reference["fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["fichier"] == nom)
        .expect("fixture absente de la référence indépendante")
}

fn verifier_cles(objet: &Value, cles: &[&str]) {
    let objet = objet.as_object().expect("objet attendu");
    assert_eq!(objet.len(), cles.len(), "clés : {objet:?}");
    for cle in cles {
        assert!(objet.contains_key(*cle), "clé {cle} absente");
    }
}

fn verifier_fractions(calc: &Value, reference: &Value) {
    let comptes = reference["comptes"].as_object().unwrap();
    let fractions = calc["fractions"].as_object().unwrap();
    assert_eq!(fractions.len(), comptes.len());
    let n = reference["N"].as_u64().unwrap() as f64;
    for (nom, compte) in comptes {
        let valeur = fractions[nom].as_f64().unwrap();
        assert!(valeur.is_finite());
        assert_eq!(
            valeur,
            compte.as_u64().unwrap() as f64 / n,
            "fraction {nom}"
        );
    }
}

/// Contrôle de la SÉRIALISATION CLI contre les références Python, sans recalcul
/// Rust des statistiques attendues. Les fractions découlent des comptes Python.
fn comparison_json_faces(calc: &Value, reference: &Value, histogramme: bool) {
    let mut cles = vec![
        "N",
        "comptes",
        "fractions",
        "sx",
        "sx2",
        "sTheta2",
        "sTheta4",
        "xmin",
    ];
    if histogramme {
        cles.push("hist_alpha_nonvides");
    }
    verifier_cles(calc, &cles);
    assert_eq!(calc["N"], reference["N"]);
    assert_eq!(calc["comptes"], reference["comptes"]);
    verifier_fractions(calc, reference);
    for cle in ["sx", "sx2", "sTheta2", "sTheta4", "xmin"] {
        let a = calc[cle].as_f64().unwrap();
        let b = reference[cle].as_f64().unwrap();
        assert!(a.is_finite() && b.is_finite(), "{cle}");
        let tolerance = if cle == "xmin" {
            1e-13
        } else {
            1e-12 * b.abs().max(1e-30)
        };
        assert!(
            (a - b).abs() <= tolerance,
            "{cle} : calcul={a}, référence={b}"
        );
    }
    if histogramme {
        assert_eq!(
            calc["hist_alpha_nonvides"],
            reference["hist_alpha_nonvides"]
        );
    }
}

fn comparison_json_coherent(calc: &Value, reference: &Value) {
    verifier_cles(
        calc,
        &[
            "N",
            "comptes",
            "fractions",
            "comptes_coherence",
            "s_phi",
            "s_phi2",
            "s_coh",
            "phi_max",
            "n_ambigues",
        ],
    );
    for cle in ["N", "comptes", "comptes_coherence"] {
        assert_eq!(calc[cle], reference[cle], "{cle}");
    }
    verifier_fractions(calc, reference);
    for cle in ["s_phi", "s_phi2", "s_coh", "phi_max"] {
        let a = calc[cle].as_f64().unwrap();
        let b = reference[cle].as_f64().unwrap();
        assert!(a.is_finite() && b.is_finite(), "{cle}");
        let tolerance = if cle == "phi_max" {
            1e-12
        } else {
            1e-11 * b.abs().max(1e-30)
        };
        assert!(
            (a - b).abs() <= tolerance,
            "{cle} : calcul={a}, référence={b}"
        );
    }
    assert!(calc["n_ambigues"].as_u64().is_some());
    if let Some(ambiguite) = reference.get("n_ambigues") {
        assert_eq!(&calc["n_ambigues"], ambiguite);
    }
}

/// T7 — CLI : statistiques complètes indépendantes, métadonnées et refus.
#[test]
fn t7_cli() {
    let dossier = DossierTest::nouveau("cli");
    let l8 = fixture("tests/donnees/rg/cfg_su2_L8_b2.40_s24001.npy");
    let cfg = fixture("validation/rg/fixtures/cfg_s7_q64.npy");
    let ubar = fixture("validation/rg/fixtures/ubar_s7_q64.npy");
    let reference: Value = serde_json::from_str(
        &std::fs::read_to_string(fixture("validation/rg_flux/reference_py.json")).unwrap(),
    )
    .unwrap();
    let ref_l8 = reference_fixture(&reference, "cfg_su2_L8_b2.40_s24001.npy");
    let ref_cfg = reference_fixture(&reference, "cfg_s7_q64.npy");
    let ref_ubar = reference_fixture(&reference, "ubar_s7_q64.npy");

    // L8 : grossier fourni ; L9 : grossier calculé. L9 possède deux tranches
    // physiques par plan, dont une incomplète. Comparer les octets avec 1/4 fils.
    for (entree, attendu, calculer, l) in [(&l8, ref_l8, false, 8), (&cfg, ref_cfg, true, 9)] {
        let sortie = dossier.fichier(&format!("configuration_L{l}.json"));
        let mut precedent = None;
        for fils in ["1", "4"] {
            let mut args = vec![
                "flux",
                "--entree",
                entree.to_str().unwrap(),
                "--n",
                "1,2,3",
                "--coherent",
                "3",
                "--histogramme",
                "--sortie",
                sortie.to_str().unwrap(),
                "--fils",
                fils,
            ];
            if calculer {
                args.push("--grossier");
            } else {
                args.extend(["--grossier-fichier", ubar.to_str().unwrap()]);
            }
            assert_eq!(executer_flux(&args).unwrap(), 0);
            let texte = std::fs::read_to_string(&sortie).unwrap();
            if let Some(avant) = &precedent {
                assert_eq!(avant, &texte, "déterminisme L{l}, 1 contre 4 fils");
            }
            let doc: Value = serde_json::from_str(&texte).unwrap();
            assert_eq!(doc["format"], kvu::rg::flux::SCHEMA_FLUX);
            assert_eq!(doc["entree"], entree.to_str().unwrap());
            assert_eq!(doc["sha256_entree"], attendu["sha256"]);
            assert_eq!(doc["L"], l);
            assert_eq!(doc["pavage"], "tous");
            verifier_cles(&doc["brut"], &["n1", "n2", "n3"]);
            for n in ["n1", "n2", "n3"] {
                comparison_json_faces(&doc["brut"][n], &attendu["brut"][n], true);
            }
            verifier_cles(&doc["coherent"], &["n3"]);
            comparison_json_coherent(&doc["coherent"]["n3"], &attendu["coherent"]["n3"]);
            assert_eq!(
                doc["grossier"]["source"],
                if calculer { "calcul" } else { "fichier" }
            );
            assert_eq!(doc["grossier"]["L"], 3);
            comparison_json_faces(&doc["grossier"]["n1"], &ref_ubar["brut"]["n1"], true);
            precedent = Some(texte);
            std::fs::remove_file(&sortie).unwrap();
        }
    }

    // Mode liste : chaque ligne porte toutes les métadonnées, avec des valeurs
    // distinctes pour détecter une mauvaise recopie ou une permutation de lignes.
    let travaux = [
        serde_json::json!({"kind":"fin", "path":l8, "cell":"L8", "chain":"c1", "idx":11, "L":8, "beta":2.4, "src":"source_l8"}),
        serde_json::json!({"kind":"gros1", "path":ubar, "cell":"ubar", "chain":"c2", "idx":22, "L":3, "beta":2.5, "src":"source_ubar"}),
        serde_json::json!({"kind":"fin", "path":cfg, "cell":"cfg", "chain":"c3", "idx":33, "L":9, "beta":2.6, "src":"source_cfg"}),
    ];
    let liste = dossier.fichier("liste.jsonl");
    let sortie_liste = dossier.fichier("lignes.jsonl");
    std::fs::write(&liste, texte_jsonl(&travaux)).unwrap();
    assert_eq!(
        executer_flux(&[
            "flux",
            "--liste",
            liste.to_str().unwrap(),
            "--coherent",
            "3",
            "--sortie",
            sortie_liste.to_str().unwrap(),
            "--fils",
            "4",
        ])
        .unwrap(),
        0
    );
    let texte = std::fs::read_to_string(&sortie_liste).unwrap();
    let lignes: Vec<Value> = texte
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lignes.len(), 3);
    for ((ligne, travail), attendu) in lignes.iter().zip(&travaux).zip([ref_l8, ref_ubar, ref_cfg])
    {
        for cle in ["kind", "cell", "chain", "idx", "L", "beta", "src"] {
            assert_eq!(ligne[cle], travail[cle], "métadonnée {cle}");
        }
        let fin = travail["kind"] == "fin";
        verifier_cles(
            &ligne["stats"],
            if fin {
                &["n1", "n2", "n3", "coherent_n3"]
            } else {
                &["n1"]
            },
        );
        for (n, ref_stats) in attendu["brut"].as_object().unwrap() {
            comparison_json_faces(&ligne["stats"][n], ref_stats, false);
        }
        if fin {
            comparison_json_coherent(&ligne["stats"]["coherent_n3"], &attendu["coherent"]["n3"]);
        }
    }

    // Chaque refus reçoit une entrée par ailleurs valide et sa propre sortie.
    // Les opérateurs sont testés sur L9, pour éviter un refus de divisibilité.
    let cas = [
        (&l8, vec!["--n", ""], "vide"),
        (&l8, vec!["--n", "0"], "≥ 1"),
        (&l8, vec!["--n", "1,x"], "entier"),
        (&l8, vec!["--n", "1,1"], "doublon"),
        (&l8, vec!["--coherent", ""], "vide"),
        (&l8, vec!["--coherent", "x"], "entier"),
        (&l8, vec!["--coherent", "0"], "≥ 1"),
        (&l8, vec!["--coherent", "3,3"], "doublon"),
        (&l8, vec!["--pavage", "5"], "pavage"),
        (&l8, vec!["--pavage", "0"], "multiple de 3"),
        (&l8, vec!["--grossier"], "non divisible"),
        // L9 est divisible par 3² ; L3 (ubar) permet le refus attendu pour K=2.
        (&ubar, vec!["--grossier", "--grossier-pas", "2"], "3^2"),
        (
            &cfg,
            vec!["--grossier", "--grossier-operateur", "inconnu"],
            "inconnu",
        ),
        (
            &cfg,
            vec!["--grossier", "--grossier-operateur", "ape:0.5:40"],
            "hors de la famille",
        ),
        (
            &cfg,
            vec!["--grossier", "--grossier-fichier", ubar.to_str().unwrap()],
            "",
        ),
    ];
    for (i, (entree, options, message)) in cas.iter().enumerate() {
        let sortie = dossier.fichier(&format!("refus_{i}.json"));
        let mut args = vec![
            "flux",
            "--entree",
            entree.to_str().unwrap(),
            "--sortie",
            sortie.to_str().unwrap(),
        ];
        args.extend(options.iter().copied());
        let erreur = executer_flux(&args).unwrap_err();
        assert_eq!(erreur.code_sortie(), 2, "cas {i}: {erreur}");
        assert!(erreur.to_string().contains(*message), "cas {i}: {erreur}");
        assert!(!sortie.exists(), "cas {i}: fichier créé");
    }

    let sortie_existante = dossier.fichier("existante.json");
    std::fs::write(&sortie_existante, "CONTENU ORIGINAL").unwrap();
    let erreur = executer_flux(&[
        "flux",
        "--entree",
        l8.to_str().unwrap(),
        "--n",
        "1",
        "--sortie",
        sortie_existante.to_str().unwrap(),
    ])
    .unwrap_err();
    assert_eq!(erreur.code_sortie(), 2);
    assert_eq!(
        std::fs::read_to_string(sortie_existante).unwrap(),
        "CONTENU ORIGINAL"
    );

    let erreur = executer_flux(&["flux", "--liste", liste.to_str().unwrap()]).unwrap_err();
    assert_eq!(erreur.code_sortie(), 2);
    let sortie = dossier.fichier("liste_refusee.jsonl");
    let erreur = executer_flux(&[
        "flux",
        "--liste",
        liste.to_str().unwrap(),
        "--n",
        "1",
        "--sortie",
        sortie.to_str().unwrap(),
    ])
    .unwrap_err();
    assert_eq!(erreur.code_sortie(), 2);
    assert!(!sortie.exists());

    // L présent mais non entier, négatif ou incohérent : même validateur que G-REPRO.
    for l_invalide in [
        serde_json::json!("huit"),
        serde_json::json!(8.5),
        serde_json::json!(-8),
        serde_json::json!(7),
    ] {
        let mut travail = travaux[0].clone();
        travail["L"] = l_invalide;
        std::fs::write(&liste, format!("{travail}\n")).unwrap();
        let erreur = executer_flux(&[
            "flux",
            "--liste",
            liste.to_str().unwrap(),
            "--sortie",
            sortie.to_str().unwrap(),
        ])
        .unwrap_err();
        assert!(matches!(&erreur, Erreur::Config(_)));
        assert_eq!(erreur.code_sortie(), 2);
        assert!(!sortie.exists());
    }

    // La fraction grossière doit aussi figurer dans la ligne INFO.
    let sortie = std::process::Command::new(env!("CARGO_BIN_EXE_kvu"))
        .args([
            "rg",
            "flux",
            "--entree",
            ubar.to_str().unwrap(),
            "--n",
            "1",
            "--grossier-fichier",
            ubar.to_str().unwrap(),
            "--fils",
            "1",
        ])
        .output()
        .unwrap();
    assert_eq!(sortie.status.code(), Some(0));
    let fraction = ref_ubar["brut"]["n1"]["comptes"]["T1_pi"].as_u64().unwrap() as f64
        / ref_ubar["brut"]["n1"]["N"].as_u64().unwrap() as f64;
    assert!(String::from_utf8_lossy(&sortie.stderr)
        .contains(&format!("grossier_n1_f_T1={fraction:.6}")));
}

/// T8 — Gates : sans --desert, G-REPRO SKIP, toutes les autres PASS ; mutants PASS.
/// Arbitrage du chef de projet : convention kvu, un SKIP n'est pas vert.
#[test]
fn t8_gates() {
    let lancer = |args: &[&str]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_kvu"))
            .args(args)
            .output()
            .unwrap()
    };
    let verifier = |sortie: &std::process::Output, avec_mutants: bool| {
        let texte = String::from_utf8_lossy(&sortie.stdout).into_owned();
        assert_eq!(
            sortie.status.code(),
            Some(1),
            "rc attendu 1 (G-REPRO SKIP) :\n{texte}"
        );
        let lignes: Vec<&str> = texte.lines().filter(|l| !l.trim().is_empty()).collect();
        let mut vu_repro = false;
        let mut vu_mut = false;
        for l in &lignes {
            if l.contains(" G-REPRO ") || l.starts_with("SKIP G-REPRO") {
                assert!(
                    l.starts_with("SKIP"),
                    "G-REPRO doit être SKIP sans --desert : {l}"
                );
                vu_repro = true;
            } else if l.starts_with("PASS") || l.starts_with("INFO") {
                if l.starts_with("PASS RG-FLUX-MUT") {
                    vu_mut = true;
                }
            } else {
                panic!("ligne non verte inattendue : {l}\n{texte}");
            }
        }
        assert!(vu_repro, "ligne G-REPRO absente :\n{texte}");
        assert_eq!(vu_mut, avec_mutants, "ligne RG-FLUX-MUT :\n{texte}");
        for id in [
            "RG-FLUX-ABELIEN",
            "RG-FLUX-FROID",
            "RG-FLUX-JAUGE",
            "RG-FLUX-REF-PY",
            "RG-FLUX-GROSSIER",
        ] {
            assert!(
                lignes.iter().any(|l| l.starts_with(&format!("PASS {id}"))),
                "{id} doit être PASS :\n{texte}"
            );
        }
    };
    verifier(
        &lancer(&["rg", "flux-gates", "--sans-mutants", "--fils", "4"]),
        false,
    );
    verifier(&lancer(&["rg", "flux-gates", "--fils", "4"]), true);
}

/// Une configuration réelle de fixture, sa paire fin/gros1 et le format
/// Python à sept champs : PASS positif et FAIL pour chaque altération.
#[test]
fn g_repro_format_python_reel_et_refus() {
    let dossier = DossierTest::nouveau("repro");
    let donnees = dossier.fichier("donnees");
    std::fs::create_dir(&donnees).unwrap();
    let cfg = dossier.fichier("cfg.npy");
    let ubar = dossier.fichier("ubar.npy");
    std::fs::copy(fixture("validation/rg/fixtures/cfg_s7_q64.npy"), &cfg).unwrap();
    std::fs::copy(fixture("validation/rg/fixtures/ubar_s7_q64.npy"), &ubar).unwrap();
    let reference: Value = serde_json::from_str(
        &std::fs::read_to_string(fixture("validation/rg_flux/reference_py.json")).unwrap(),
    )
    .unwrap();
    let travaux = vec![
        serde_json::json!({"kind":"fin", "path":cfg, "cell":"fixture", "chain":"c1", "idx":7, "L":9}),
        serde_json::json!({"kind":"gros1", "path":ubar, "cell":"fixture", "chain":"c1", "idx":7, "L":3}),
    ];
    let mut refs = Vec::new();
    for (travail, nom) in travaux.iter().zip(["cfg_s7_q64.npy", "ubar_s7_q64.npy"]) {
        let mut ligne = travail.clone();
        ligne.as_object_mut().unwrap().remove("path");
        let mut stats = reference_fixture(&reference, nom)["brut"].clone();
        for stat in stats.as_object_mut().unwrap().values_mut() {
            stat.as_object_mut().unwrap().remove("hist_alpha_nonvides");
            verifier_cles(
                stat,
                &["N", "comptes", "sx", "sx2", "sTheta2", "sTheta4", "xmin"],
            );
        }
        ligne["stats"] = stats;
        refs.push(ligne);
    }
    let ecrire = |nom: &str, lignes: &[Value]| {
        std::fs::write(donnees.join(nom), texte_jsonl(lignes)).unwrap();
    };
    let lancer = |max: &str| {
        std::process::Command::new(env!("CARGO_BIN_EXE_kvu"))
            .args([
                "rg",
                "flux-gates",
                "--seul",
                "G-REPRO",
                "--sans-mutants",
                "--desert",
                dossier.0.to_str().unwrap(),
                "--max-configs",
                max,
                "--fils",
                "1",
            ])
            .output()
            .unwrap()
    };
    let verifier_echec = |sortie: std::process::Output, attendu: &str| {
        let texte = String::from_utf8_lossy(&sortie.stdout);
        assert_eq!(
            sortie.status.code(),
            Some(1),
            "{texte}\n{}",
            String::from_utf8_lossy(&sortie.stderr)
        );
        assert!(texte.starts_with("FAIL G-REPRO"), "{texte}");
        assert!(
            texte.contains(attendu),
            "diagnostic attendu {attendu} : {texte}"
        );
    };
    ecrire("liste_existants.jsonl", &travaux);
    ecrire("lignes_existants.jsonl", &refs);
    let positif = lancer("1");
    assert_eq!(
        positif.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&positif.stdout)
    );
    assert!(String::from_utf8_lossy(&positif.stdout).starts_with("PASS G-REPRO 2 lignes comparées"));

    // Présence de chaque ligne dans les travaux ET dans les références.
    for indice in 0..2 {
        let mut incompletes = refs.clone();
        incompletes.remove(indice);
        ecrire("lignes_existants.jsonl", &incompletes);
        verifier_echec(lancer("1"), "référence manquante");
        ecrire("lignes_existants.jsonl", &refs);
        let mut incomplets = travaux.clone();
        incomplets.remove(indice);
        ecrire("liste_existants.jsonl", &incomplets);
        verifier_echec(lancer("1"), "travail manquant");
        ecrire("liste_existants.jsonl", &travaux);
    }
    for cle in ["N", "comptes", "sx", "sx2", "sTheta2", "sTheta4", "xmin"] {
        let mut incompletes = refs.clone();
        incompletes[0]["stats"]["n1"]
            .as_object_mut()
            .unwrap()
            .remove(cle);
        ecrire("lignes_existants.jsonl", &incompletes);
        let sortie = lancer("1");
        let texte = String::from_utf8_lossy(&sortie.stdout);
        assert!(texte.contains("fixture/c1/idx=7/fin"), "{texte}");
        assert!(
            texte.contains("calcul=") && texte.contains("référence=<absente>"),
            "{texte}"
        );
        verifier_echec(sortie, &format!("stats.n1.{cle}"));
    }
    for en_plus in [true, false] {
        let mut incompletes = refs.clone();
        let stats = incompletes[0]["stats"].as_object_mut().unwrap();
        if en_plus {
            stats.insert("n99".into(), refs[0]["stats"]["n1"].clone());
        } else {
            stats.remove("n2");
        }
        ecrire("lignes_existants.jsonl", &incompletes);
        verifier_echec(lancer("1"), if en_plus { "stats.n99" } else { "stats.n2" });
    }
    let mut fausses = refs.clone();
    fausses[0]["stats"]["n1"]["sx"] = serde_json::json!(-12345.0);
    ecrire("lignes_existants.jsonl", &fausses);
    let sortie = lancer("1");
    let texte = String::from_utf8_lossy(&sortie.stdout);
    assert!(
        texte.contains("calcul=") && texte.contains("référence=-12345.0"),
        "{texte}"
    );
    verifier_echec(sortie, "fixture/c1/idx=7/fin : stats.n1.sx");

    // NaN littéral du Python permissif : JSON non valide doit rendre la gate FAIL.
    let texte = texte_jsonl(&refs);
    let texte_nan = texte.replacen(
        &format!("\"sx\":{}", refs[0]["stats"]["n1"]["sx"]),
        "\"sx\":NaN",
        1,
    );
    assert!(texte_nan.contains(":NaN"));
    std::fs::write(donnees.join("lignes_existants.jsonl"), texte_nan).unwrap();
    verifier_echec(lancer("1"), "ERREUR");
    ecrire("lignes_existants.jsonl", &refs);

    for l_invalide in [
        serde_json::json!("neuf"),
        serde_json::json!(9.5),
        serde_json::json!(-9),
        serde_json::json!(8),
    ] {
        let mut invalides = travaux.clone();
        invalides[0]["L"] = l_invalide;
        ecrire("liste_existants.jsonl", &invalides);
        verifier_echec(lancer("1"), "L");
    }
    ecrire("liste_existants.jsonl", &travaux);
    verifier_echec(lancer("0"), "0 ligne comparée");
    ecrire("liste_existants.jsonl", &[]);
    ecrire("lignes_existants.jsonl", &[]);
    verifier_echec(lancer("1"), "0 ligne comparée");

    let invalide = lancer("pas_un_entier");
    assert_eq!(invalide.status.code(), Some(2));
    let commande = kvu::rg::flux::cli::sous_commandes()
        .into_iter()
        .find(|c| c.get_name() == "flux-gates")
        .unwrap();
    let matches = commande
        .try_get_matches_from(["flux-gates", "--max-configs", "pas_un_entier"])
        .unwrap();
    let erreur = kvu::rg::flux::cli::executer("flux-gates", &matches).unwrap_err();
    assert!(matches!(erreur, Erreur::Config(_)));
}
