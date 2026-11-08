//! Bain conditionnel, référence CPU Rust (rayon) : EQC-1 (1 balayage `ref_rng` contre cb.sweep, ≤ 1e-12),
//! EQC-1b (état PCG64 identique à cb après 5 balayages), refus de β = 0, invariants Vdev / unitarité,
//! mode rapide (déterminisme, invariants), reprise de la source d'aléa.
//! Jeu commité : L = 4 (45 cas). Jeu étendu (L = 8, ≥ 100 cas) si `KVU_COND_FIXTURES` pointe sur le dossier
//! produit par `validation/cond/gen_fixtures.py --niveau complet`.
use kvu::commun::{ChampHote, Erreur, Graine};
use kvu::cond::essais::{cas_balayages, dossier_commite, dossier_complet, ecart_max, CasBalayage};
use kvu::cond::geometrie::v_paires;
use kvu::cond::moteur::{ModeCond, MoteurCond, SourceAlea};
use kvu::cond::npy_min::lire_champ_quaternions;
use kvu::cond::{GenerateurNumpy, GeometrieCond};
use std::path::Path;
use std::sync::Arc;

const TOL_BALAYAGE: f64 = 1e-12;
const TOL_CONTROLES: f64 = 1e-12;

fn champ_du_cas(c: &CasBalayage) -> ChampHote {
    let (l, q) = lire_champ_quaternions(&c.fichier_cfg).unwrap();
    assert_eq!(l, c.l);
    ChampHote { l, q }
}

fn moteur_cas(c: &CasBalayage, geos: &mut std::collections::BTreeMap<usize, Arc<GeometrieCond>>) -> MoteurCond {
    let champ = champ_du_cas(c);
    let geo = geos.entry(c.l).or_insert_with(|| Arc::new(GeometrieCond::nouvelle(c.l).unwrap())).clone();
    let vfix = v_paires(&champ, &geo);
    MoteurCond::nouveau(None, geo, &champ, vfix, c.beta, SourceAlea::depuis_graine(ModeCond::RefRng, c.graine)).unwrap()
}

/// Exécute les gates EQC-1 / EQC-1b sur un jeu de cas ; rend (n cas, écart max, Vdev max, unitarité max).
fn gates_eqc1(dossier: &Path, l: Option<usize>) -> (usize, f64, f64, f64) {
    let cas = cas_balayages(dossier, l).unwrap();
    let mut geos = Default::default();
    let (mut dmax, mut vmax, mut umax) = (0.0f64, 0.0f64, 0.0f64);
    for c in &cas {
        let mut m = moteur_cas(c, &mut geos);
        m.sweep().unwrap();
        let sortie = kvu::cond::npy_min::lire(&c.fichier_out).unwrap().f64().unwrap();
        let d = ecart_max(&m.telecharger().unwrap().q, &sortie);
        let (vdev, unit) = m.controles().unwrap();
        assert!(d <= TOL_BALAYAGE, "EQC-1 L={} {} β={} graine={} : max|ΔC| = {d:e}", c.l, c.cfg, c.beta, c.graine);
        assert!(vdev <= TOL_CONTROLES, "Vdev {vdev:e} ({} β={} s={})", c.cfg, c.beta, c.graine);
        assert!(unit <= TOL_CONTROLES, "unitarité {unit:e} ({} β={} s={})", c.cfg, c.beta, c.graine);
        // EQC-1b : l'état de numpy est celui de cb après 1 balayage, puis après 5
        match m.alea() {
            SourceAlea::Numpy(g) => assert_eq!((g.etat, g.inc), c.etat1, "EQC-1b après 1 balayage L={} {}", c.l, c.cfg),
            _ => unreachable!(),
        }
        for _ in 0..4 {
            m.sweep().unwrap();
        }
        match m.alea() {
            SourceAlea::Numpy(g) => {
                assert_eq!((g.etat, g.inc), c.etat5, "EQC-1b après 5 balayages L={} {} β={}", c.l, c.cfg, c.beta)
            }
            _ => unreachable!(),
        }
        dmax = dmax.max(d);
        vmax = vmax.max(vdev);
        umax = umax.max(unit);
    }
    (cas.len(), dmax, vmax, umax)
}

#[test]
fn eqc1_eqc1b_jeu_commite_l4() {
    let (n, d, v, u) = gates_eqc1(&dossier_commite(), Some(4));
    assert_eq!(n, 45);
    eprintln!("PASS EQC-1/EQC-1b (CPU, L=4) : {n} cas, max|ΔC| = {d:.2e}, Vdev = {v:.2e}, unitarité = {u:.2e}");
}

/// Jeu étendu (L = 4 et 8, ≥ 100 cas) : exigé par la gate EQC-1 ; ignoré sans `KVU_COND_FIXTURES`.
#[test]
fn eqc1_eqc1b_jeu_complet() {
    let Some(d) = dossier_complet() else {
        eprintln!("SKIP EQC-1 complet : KVU_COND_FIXTURES absent (ne vaut pas vert)");
        return;
    };
    let (n, dmax, v, u) = gates_eqc1(&d, None);
    assert!(n >= 100, "EQC-1 exige ≥ 100 cas, {n} trouvés");
    eprintln!(
        "PASS EQC-1/EQC-1b (CPU, jeu complet) : {n} cas, max|ΔC| = {dmax:.2e}, Vdev = {v:.2e}, unitarité = {u:.2e}"
    );
}

#[test]
fn beta_nul_refuse_en_ref_rng_et_accepte_en_rapide() {
    let champ = ChampHote::froid(4).unwrap();
    let geo = Arc::new(GeometrieCond::nouvelle(4).unwrap());
    let vfix = v_paires(&champ, &geo);
    let r = MoteurCond::nouveau(
        None,
        geo.clone(),
        &champ,
        vfix.clone(),
        0.0,
        SourceAlea::depuis_graine(ModeCond::RefRng, 1),
    );
    assert!(matches!(r, Err(Erreur::Config(_))), "β = 0 en ref_rng doit être refusé (Erreur::Config)");
    let mut m =
        MoteurCond::nouveau(None, geo, &champ, vfix, 0.0, SourceAlea::depuis_graine(ModeCond::Rapide, 1)).unwrap();
    m.sweep().unwrap();
    let (vdev, unit) = m.controles().unwrap();
    assert!(vdev <= TOL_CONTROLES && unit <= TOL_CONTROLES);
}

#[test]
fn entrees_invalides_refusees() {
    let champ = ChampHote::froid(4).unwrap();
    let geo = Arc::new(GeometrieCond::nouvelle(4).unwrap());
    let vfix = v_paires(&champ, &geo);
    let alea = || SourceAlea::depuis_graine(ModeCond::Rapide, 0);
    assert!(MoteurCond::nouveau(None, geo.clone(), &champ, vfix.clone(), -1.0, alea()).is_err());
    assert!(MoteurCond::nouveau(None, geo.clone(), &champ, vfix.clone(), f64::NAN, alea()).is_err());
    assert!(MoteurCond::nouveau(None, geo.clone(), &champ, vfix[1..].to_vec(), 2.3, alea()).is_err());
    let autre = ChampHote::froid(6).unwrap();
    assert!(MoteurCond::nouveau(None, geo, &autre, vfix, 2.3, alea()).is_err());
}

#[test]
fn rapide_invariants_sur_champ_de_production_l4() {
    let d = dossier_commite();
    for c in cas_balayages(&d, Some(4)).unwrap().iter().filter(|c| c.graine == 0 && c.cfg.starts_with("haar")) {
        let champ = champ_du_cas(c);
        let geo = Arc::new(GeometrieCond::nouvelle(4).unwrap());
        let vfix = v_paires(&champ, &geo);
        let mut m =
            MoteurCond::nouveau(None, geo, &champ, vfix, c.beta, SourceAlea::depuis_graine(ModeCond::Rapide, 7))
                .unwrap();
        for _ in 0..5 {
            m.sweep().unwrap();
        }
        let (vdev, unit) = m.controles().unwrap();
        assert!(vdev <= TOL_CONTROLES && unit <= TOL_CONTROLES, "{} β={} : Vdev {vdev:e} unit {unit:e}", c.cfg, c.beta);
        assert!(m.telecharger().unwrap().q.iter().all(|x| x.is_finite()));
        assert_eq!(m.diag_rapide().unwrap().plafond, 0);
    }
}

#[test]
fn rapide_reprise_de_la_source_d_alea_identique_a_l_octet() {
    // 2N balayages d'un coup = N balayages, source d'aléa et champ sérialisés, puis N balayages.
    let c = &cas_balayages(&dossier_commite(), Some(4)).unwrap()[10];
    for mode in [ModeCond::Rapide, ModeCond::RefRng] {
        let champ = champ_du_cas(c);
        let geo = Arc::new(GeometrieCond::nouvelle(4).unwrap());
        let vfix = v_paires(&champ, &geo);
        let mk = |ch: &ChampHote, a: SourceAlea| {
            MoteurCond::nouveau(None, geo.clone(), ch, vfix.clone(), c.beta, a).unwrap()
        };
        let mut seul = mk(&champ, SourceAlea::depuis_graine(mode, 99));
        for _ in 0..6 {
            seul.sweep().unwrap();
        }
        let mut a = mk(&champ, SourceAlea::depuis_graine(mode, 99));
        for _ in 0..3 {
            a.sweep().unwrap();
        }
        // aller-retour par JSON de la source d'aléa (le format du fichier de reprise)
        let json = serde_json::to_string(&a.alea()).unwrap();
        let alea: SourceAlea = serde_json::from_str(&json).unwrap();
        let mut b = mk(&a.telecharger().unwrap(), alea);
        for _ in 0..3 {
            b.sweep().unwrap();
        }
        let (x, y) = (seul.telecharger().unwrap().q, b.telecharger().unwrap().q);
        assert_eq!(
            x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{mode:?}"
        );
        assert_eq!(seul.alea(), b.alea());
    }
}

#[test]
fn rapide_depend_de_la_graine_et_pas_du_nombre_de_fils() {
    let c = &cas_balayages(&dossier_commite(), Some(4)).unwrap()[3];
    let champ = champ_du_cas(c);
    let geo = Arc::new(GeometrieCond::nouvelle(4).unwrap());
    let vfix = v_paires(&champ, &geo);
    let run = |graine: u64, fils: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(fils).build().unwrap();
        pool.install(|| {
            let mut m = MoteurCond::nouveau(
                None,
                geo.clone(),
                &champ,
                vfix.clone(),
                c.beta,
                SourceAlea::Philox { graine: Graine(graine), sweep_id: 0 },
            )
            .unwrap();
            for _ in 0..3 {
                m.sweep().unwrap();
            }
            m.telecharger().unwrap().q
        })
    };
    assert_eq!(run(5, 1), run(5, 4), "résultat indépendant du nombre de fils");
    assert_ne!(run(5, 2), run(6, 2));
}

#[test]
fn graine_numpy_sans_effet_de_bord_sur_le_generateur_de_l_appelant() {
    let g = GenerateurNumpy::depuis_graine(2709);
    let a = SourceAlea::Numpy(g.clone());
    assert_eq!(a.mode(), ModeCond::RefRng);
    assert_eq!(SourceAlea::depuis_graine(ModeCond::RefRng, 2709), SourceAlea::Numpy(g));
}
