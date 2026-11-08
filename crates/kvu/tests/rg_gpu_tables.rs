//! Tables et validation sans GPU ; les essais sur appareil sont exclusivement ignorés.

use kvu::commun::{ChampHote, Erreur};
use kvu::rg::geometrie::{Geometrie, Pas, Table};
use kvu::rg::gpu::{tables, valider_entree};

fn decoder(mot: u32) -> Pas {
    Pas {
        dep: std::array::from_fn(|m| ((mot >> (4 * m)) & 15) as i8 - 8),
        mu: ((mot >> 16) & 3) as u8,
        signe: if mot & (1 << 18) != 0 { 1 } else { -1 },
    }
}

fn verifier_tables(geometrie: Geometrie, ir_ref: i32, nb_pas: usize) {
    let table = Table::nouvelle(geometrie);
    let t = tables(&table);
    assert_eq!(table.chemins.len(), 633);
    assert_eq!(t.ch_ir.len(), 82);
    assert_eq!(t.ch_pas.len(), 634);
    assert_eq!(t.nb_boucles, 10_473);
    assert_eq!(t.steps.len(), nb_pas);
    assert_eq!(t.ir_ref, ir_ref);
    assert_eq!(t.poids.len(), 81);
    assert_eq!(t.ch_pas[633] as usize, t.steps.len());
    for ir in 0..81 {
        assert_eq!(t.ch_ir[ir] as usize, table.debut[ir]);
        assert_eq!(t.ch_ir[ir + 1] as usize, table.debut[ir + 1]);
        assert!(t.ch_ir[ir + 1] - t.ch_ir[ir] <= 24);
        assert_eq!(t.poids[ir].to_bits(), table.poids(ir).to_bits());
    }
    for (i, chemin) in table.chemins.iter().enumerate() {
        let debut = t.ch_pas[i] as usize;
        let fin = t.ch_pas[i + 1] as usize;
        assert_eq!(fin - debut, chemin.len());
        for (&mot, &pas) in t.steps[debut..fin].iter().zip(chemin) {
            assert_eq!(mot >> 19, 0);
            assert_eq!(decoder(mot), pas);
        }
    }
}

#[test]
fn tables_trois_centre() {
    verifier_tables(Geometrie::TroisCentre, 40, 2216);
}

#[test]
fn tables_trois_coin() {
    verifier_tables(Geometrie::TroisCoin, 0, 3324);
}

#[test]
fn compactage_aux_bornes() {
    let mut table = Table::nouvelle(Geometrie::TroisCentre);
    for mu in 0..4 {
        for signe in [-1, 1] {
            let pas = Pas {
                dep: [-8, 7, -1, 0],
                mu,
                signe,
            };
            table.chemins[0][0] = pas;
            assert_eq!(decoder(tables(&table).steps[0]), pas);
        }
    }
}

#[test]
#[should_panic(expected = "déplacement hors du compactage")]
fn refuse_un_deplacement_non_representable() {
    let mut table = Table::nouvelle(Geometrie::TroisCentre);
    table.chemins[0][0].dep[0] = 8;
    tables(&table);
}

#[test]
#[should_panic(expected = "1 à 24 chemins")]
fn refuse_plus_de_24_chemins_par_site() {
    let mut table = Table::nouvelle(Geometrie::TroisCentre);
    table.debut[1] = 25;
    tables(&table);
}

#[test]
fn l4_refuse_avant_tout_acces_gpu() {
    let champ = ChampHote {
        l: 4,
        q: vec![0.0; 16 * 4usize.pow(4)],
    };
    assert!(matches!(valider_entree(&champ), Err(Erreur::Config(_))));
}

#[test]
fn longueur_finitude_et_debordements_sans_gpu() {
    let mut champ = ChampHote {
        l: 3,
        q: vec![0.0; 16 * 81],
    };
    valider_entree(&champ).unwrap();
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        champ.q[13] = x;
        assert!(matches!(valider_entree(&champ), Err(Erreur::Config(_))));
    }
    champ.q[13] = 0.0;
    champ.q.pop();
    assert!(matches!(valider_entree(&champ), Err(Erreur::Config(_))));
    for l in [0, 1, 2, usize::MAX] {
        assert!(matches!(
            valider_entree(&ChampHote { l, q: Vec::new() }),
            Err(Erreur::Config(_))
        ));
    }
}

fn fixture_graine_7() -> ChampHote {
    kvu::stockage::npy::lire_champ(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("validation/rg/fixtures/cfg_s7_q64.npy"),
    )
    .unwrap()
}

fn appareil() -> kvu::gpu::Appareil {
    let backend = std::env::var("KEVINOTRON_BACKEND")
        .map(|s| s.parse().unwrap())
        .unwrap_or(if cfg!(feature = "cuda") {
            kvu::commun::BackendKind::Cuda
        } else {
            kvu::commun::BackendKind::Hip
        });
    kvu::gpu::ouvrir(backend, 0).unwrap()
}

#[test]
#[ignore = "GPU"]
fn gpu_contre_cpu_fixture_graine_7() {
    let app = appareil();
    let champ = fixture_graine_7();
    for geometrie in [Geometrie::TroisCentre, Geometrie::TroisCoin] {
        let pilote = kvu::rg::gpu::BlocGpu::nouveau(&app, geometrie).unwrap();
        let gpu = pilote.bloquer_hote(&champ, true).unwrap();
        let cpu = kvu::rg::bloc::bloquer(
            &champ,
            &kvu::rg::bloc::OptionsBloc {
                geometrie,
                fils: 8,
                angle_so3: false,
            },
        )
        .unwrap();
        assert_eq!(gpu.grossier.l, cpu.grossier.l);
        assert_eq!(gpu.grossier.q.len(), cpu.grossier.q.len());
        for (&a, &b) in gpu.grossier.q.iter().zip(&cpu.grossier.q) {
            assert!((a - b).abs() <= 1e-12, "max |Δq| dépassé : {a} contre {b}");
        }
        assert_eq!(gpu.stats.n, cpu.stats.n);
        assert_eq!(gpu.stats.n_hors, cpu.stats.n_hors);
        assert_eq!(gpu.stats.n_ambigues, cpu.stats.n_ambigues);
        assert_eq!(gpu.stats.histogramme, cpu.stats.histogramme);
    }
}

#[test]
#[ignore = "GPU"]
fn deux_executions_memes_octets() {
    let app = appareil();
    let champ = fixture_graine_7();
    let pilote = kvu::rg::gpu::BlocGpu::nouveau(&app, Geometrie::TroisCentre).unwrap();
    let a = pilote.bloquer_hote(&champ, true).unwrap();
    let b = pilote.bloquer_hote(&champ, true).unwrap();
    assert_eq!(a.stats, b.stats);
    assert_eq!(a.stats.theta_max.to_bits(), b.stats.theta_max.to_bits());
    assert_eq!(
        a.grossier.q.iter().map(|q| q.to_bits()).collect::<Vec<_>>(),
        b.grossier.q.iter().map(|q| q.to_bits()).collect::<Vec<_>>()
    );
    let sans_stats = pilote.bloquer_hote(&champ, false).unwrap();
    assert_eq!(sans_stats.stats, kvu::rg::bloc::StatsTheta::vide());
    assert_eq!(
        a.grossier.q.iter().map(|q| q.to_bits()).collect::<Vec<_>>(),
        sans_stats
            .grossier
            .q
            .iter()
            .map(|q| q.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "GPU"]
fn chaine_residente_decalee() {
    let app = appareil();
    let champ = fixture_graine_7();
    let mut tampon = app.alloc::<f64>(2 * champ.q.len()).unwrap();
    app.ecrire(&mut tampon, champ.q.len(), &champ.q).unwrap();
    let resident = kvu::su2::ChampGpu {
        l: champ.l,
        nchaines: 2,
        tampon: kvu::su2::TamponChamp::F64(tampon),
    };
    let pilote = kvu::rg::gpu::BlocGpu::nouveau(&app, Geometrie::TroisCentre).unwrap();
    let a = pilote.bloquer_hote(&champ, true).unwrap();
    let b = pilote.bloquer_resident(&resident, 1, true).unwrap();
    assert_eq!(a.stats, b.stats);
    assert_eq!(
        a.grossier.q.iter().map(|q| q.to_bits()).collect::<Vec<_>>(),
        b.grossier.q.iter().map(|q| q.to_bits()).collect::<Vec<_>>()
    );
    assert!(matches!(
        pilote.bloquer_resident(&resident, 2, false),
        Err(Erreur::Config(_))
    ));
    let resident_f32 = kvu::su2::ChampGpu {
        l: champ.l,
        nchaines: 1,
        tampon: kvu::su2::TamponChamp::F32(app.alloc::<f32>(champ.q.len()).unwrap()),
    };
    assert!(matches!(
        pilote.bloquer_resident(&resident_f32, 0, false),
        Err(Erreur::Config(_))
    ));
}
