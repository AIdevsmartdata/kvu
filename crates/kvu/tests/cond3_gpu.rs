//! cond3 sur GPU (HIP ou CUDA) : concordance avec la référence CPU. `#[ignore]` : à lancer dans une fenêtre GPU (`--ignored`), sous le verrou de la machine.

use kvu::cond3::geom::Geom;
use kvu::cond3::gpu::{executer_replique_gpu, FibreSu2Gpu};
use kvu::cond3::mesure::{executer_replique, ConfigMesure};
use kvu::cond3::modele::Modele;
use kvu::cond3::paires::Paires;
use kvu::cond3::su2::{champ_haar_test, FibreSu2};
use std::sync::Arc;

#[test]
#[ignore = "GPU : fenêtre GPU de la recette"]
fn concordance_cpu_gpu_sur_quelques_iterations() {
    let Some(app) = kvu::gpu::ouvrir_auto().expect("détection du GPU") else {
        eprintln!("aucun GPU : test sans objet");
        return;
    };
    let g = Arc::new(Geom::nouvelle(4, 12).unwrap());
    let mut u = champ_haar_test(g.nliens(), 91);
    // un champ chaud thermalisé quelques pas pour sortir du Haar pur
    let mut f0 = FibreSu2::nouveau(g.clone(), u.clone(), 4.0, 5);
    for t in 0..10 {
        f0.pas_libre(t, 1);
    }
    u = f0.u.clone();
    let p = Paires::nouvelles(&g, g.faces(), g.sondes_paires(), 2, 4);
    let cfg = ConfigMesure { n_dec: 0, n_mes: 4, nblk: 2, lot: 2, controle_chaque: 2, series_faces: vec![0, 5], series_sondes: vec![0, 1], ..ConfigMesure::default() };
    let mut cpu = FibreSu2::nouveau(g.clone(), u.clone(), 4.0, 11);
    let sc = executer_replique(&mut cpu, &p, &cfg).unwrap();
    let mut gpu = FibreSu2Gpu::depuis_cpu(&app, FibreSu2::nouveau(g.clone(), u, 4.0, 11), 11).unwrap();
    let sg = executer_replique_gpu(&mut gpu, &p, &cfg).unwrap();
    let rel = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max) / a.iter().fold(1e-300f64, |m, x| m.max(x.abs()));
    assert!(rel(&sc.so, &sg.so) < 1e-12, "Σ O : {}", rel(&sc.so, &sg.so));
    assert!(rel(&sc.ss, &sg.ss) < 1e-12, "Σ σ : {}", rel(&sc.ss, &sg.ss));
    assert!(rel(&sc.sos, &sg.sos) < 1e-12, "Σ O σ : {}", rel(&sc.sos, &sg.sos));
    // champ final : écart de lien à lien
    let (fc, fg) = (cpu.champ_plat(), gpu.miroir.champ_plat());
    let ecart = fc.iter().zip(&fg).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    assert!(ecart < 1e-12, "champ final : {ecart}");
    assert_eq!(sg.stats.plafond, 0);
    assert!(sg.controles.iter().all(|c| c.contrainte < 1e-12 && c.ward < 1e-10 && c.unitarite < 1e-12));
}

#[test]
#[ignore = "GPU : fenêtre GPU de la recette"]
fn rejeu_au_bit_sur_gpu() {
    let Some(app) = kvu::gpu::ouvrir_auto().expect("détection du GPU") else {
        return;
    };
    let g = Arc::new(Geom::nouvelle(4, 12).unwrap());
    let u = champ_haar_test(g.nliens(), 93);
    let p = Paires::nouvelles(&g, g.faces(), g.sondes_paires(), 2, 3);
    let cfg = ConfigMesure { n_dec: 2, n_mes: 8, nblk: 2, lot: 4, controle_chaque: 0, ..ConfigMesure::default() };
    let run = || {
        let mut m = FibreSu2Gpu::depuis_cpu(&app, FibreSu2::nouveau(g.clone(), u.clone(), 4.0, 3), 3).unwrap();
        executer_replique_gpu(&mut m, &p, &cfg).unwrap()
    };
    let (a, b) = (run(), run());
    assert_eq!(a.empreinte_finale, b.empreinte_finale);
    assert_eq!(a.sos, b.sos);
    assert_eq!(a.so, b.so);
}
