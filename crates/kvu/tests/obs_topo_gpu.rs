//! Gates WP19 GPU : à lancer uniquement par le pilote sous verrou GPU.
//! Un SKIP ne valide pas une gate. Le test de source seul s'exécute sans GPU.

use kvu::commun::ChampHote;
use kvu::gpu::{self, Appareil};
use kvu::heritage::su2_hb::{qdag, qnorm, CpuSu2, Geo, Q};
use kvu::obs::ref_cpu::{self, en_quaternions};
use kvu::obs::topo::gpu::{Cible, EspaceTopo, ModuleTopo};
use kvu::obs::topo::{self, Charges, ParamsTopo, RecordTopo, SerieCooling};

fn appareil(gate: &str) -> Option<Appareil> {
    match gpu::ouvrir_auto() {
        Ok(Some(a)) => {
            println!("GPU {} ({}, {})", a.nom(), a.arch(), a.backend().nom());
            Some(a)
        }
        Ok(None) => {
            println!("SKIP {gate} : le backend choisi est le CPU (aucun GPU)");
            None
        }
        Err(e) => {
            println!("SKIP {gate} : GPU indisponible ({e})");
            None
        }
    }
}

fn thermalise(l: usize, beta: f64, graine: u64, pas: usize) -> ChampHote {
    let mut cpu = CpuSu2::new(l, graine).unwrap();
    cpu.init(true);
    for _ in 0..pas {
        cpu.step(beta, 2);
    }
    ChampHote { l, q: cpu.flat() }
}

/// L'action S = 6V − Σ plaquettes est une différence de deux grands nombres (6V ≈ 2,5·10⁴ à L = 8) : le bruit d'arrondi
/// absolu atteint ~1e-11 même quand S ≈ 4 ; la tolérance est donc absolue (diagnostic, pas une gate).
const TOL_ACTION: f64 = 1e-10;

fn proche(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "{a:.17e} ≠ {b:.17e} : écart {:.3e} > {tol:.3e}",
        (a - b).abs()
    );
}

fn serie(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        proche(a, b, tol);
    }
}

fn bits(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

fn ecart_champ(a: &ChampHote, b: &ChampHote) -> f64 {
    assert_eq!(a.l, b.l);
    assert_eq!(a.q.len(), b.q.len());
    a.q.iter()
        .zip(&b.q)
        .map(|(&a, &b)| {
            assert!(a.is_finite() && b.is_finite());
            (a - b).abs()
        })
        .fold(0.0, f64::max)
}

fn charges_proches(a: Charges, b: Charges) {
    proche(a.plaq, b.plaq, 1e-13);
    proche(a.q_tre, b.q_tre, 1e-12);
    proche(a.q_imp, b.q_imp, 1e-12);
}

fn briques(m: &ModuleTopo, e: &EspaceTopo, cible: Cible, champ: &ChampHote) {
    let c = topo::charges(champ).unwrap();
    charges_proches(m.charges(e, cible).unwrap(), c);
    proche(m.e_plaq(e, cible).unwrap(), 24.0 * (1.0 - c.plaq), 1e-12);
    let action = topo::action_wilson(&en_quaternions(champ), &Geo::new(champ.l));
    proche(
        m.action(e, cible).unwrap(),
        action,
        TOL_ACTION,
    );
}

#[test]
#[ignore = "GPU"]
fn topo_gpu_charges() {
    let Some(app) = appareil("topo_gpu_charges") else {
        return;
    };
    let m = ModuleTopo::nouveau(&app).unwrap();
    let e = m.espace(8).unwrap();
    for champ in [
        thermalise(8, 2.5, 26102619101, 40),
        topo::champ_abelien(8),
        ChampHote::froid(8).unwrap(),
    ] {
        m.charger(&e, &champ).unwrap();
        for cible in [Cible::Source, Cible::Flot, Cible::Cooling] {
            assert_eq!(m.champ_hote(&e, cible).unwrap(), champ);
            briques(&m, &e, cible, &champ);
        }
        let c = m.charges(&e, Cible::Source).unwrap();
        if champ == topo::champ_abelien(8) {
            proche(c.q_imp, 2.001_525_336_395_644, 1e-10);
        }
        println!(
            "PASS topo_gpu_charges (plaq={:.17e}, q_tre={:.17e}, q_imp={:.17e})",
            c.plaq, c.q_tre, c.q_imp
        );
    }
}

fn cooling_gpu(m: &ModuleTopo, e: &EspaceTopo, dagger: bool) -> SerieCooling {
    let mut s = SerieCooling::default();
    for k in 0..=20 {
        if k > 0 {
            m.balayage_cooling(e, dagger).unwrap();
        }
        let c = m.charges(e, Cible::Cooling).unwrap();
        s.q_imp.push(c.q_imp);
        s.q_tre.push(c.q_tre);
        s.action.push(m.action(e, Cible::Cooling).unwrap());
    }
    s
}

fn cooling_proche(a: &SerieCooling, b: &SerieCooling) {
    serie(&a.q_imp, &b.q_imp, 1e-12);
    serie(&a.q_tre, &b.q_tre, 1e-12);
    assert_eq!(a.action.len(), b.action.len());
    for (&a, &b) in a.action.iter().zip(&b.action) {
        proche(a, b, TOL_ACTION);
    }
}

#[test]
#[ignore = "GPU"]
fn topo_gpu_flot_cooling() {
    let Some(app) = appareil("topo_gpu_flot_cooling") else {
        return;
    };
    let m = ModuleTopo::nouveau(&app).unwrap();
    let e = m.espace(8).unwrap();
    for graine in [26102619102, 26102619103, 26102619104] {
        let champ = thermalise(8, 2.5, graine, 40);
        m.charger(&e, &champ).unwrap();
        // Deux appels vérifient aussi l'avancement relatif de flot_vers.
        m.flot_vers(&e, 0.01, 100).unwrap();
        m.flot_vers(&e, 0.01, 200).unwrap();
        let reference = ref_cpu::flot_n_pas(&champ, 0.01, 300).unwrap();
        let delta = ecart_champ(&m.champ_hote(&e, Cible::Flot).unwrap(), &reference);
        assert!(delta <= 1e-12, "flot : {delta:.3e}");
        briques(&m, &e, Cible::Flot, &reference);
        assert_eq!(m.champ_hote(&e, Cible::Source).unwrap(), champ);
        assert_eq!(m.champ_hote(&e, Cible::Cooling).unwrap(), champ);
        for dagger in [true, false] {
            m.charger(&e, &champ).unwrap();
            let serie = cooling_gpu(&m, &e, dagger);
            let (reference, final_cpu) = topo::cooling(&champ, 20, dagger).unwrap();
            assert_eq!(serie.q_imp.len(), 21);
            cooling_proche(&serie, &reference);
            let delta = ecart_champ(&m.champ_hote(&e, Cible::Cooling).unwrap(), &final_cpu);
            assert!(delta <= 1e-12, "cooling dagger={dagger} : {delta:.3e}");
            briques(&m, &e, Cible::Cooling, &final_cpu);
            assert_eq!(m.champ_hote(&e, Cible::Source).unwrap(), champ);
            assert_eq!(m.champ_hote(&e, Cible::Flot).unwrap(), champ);
            println!("PASS topo_gpu_flot_cooling (graine={graine}, dagger={dagger}, max|ΔU_cool|={delta:.3e})");
        }
        println!("PASS topo_gpu_flot_cooling (graine={graine}, pas=300, max|ΔU_flot|={delta:.3e})");
    }
}

fn ecart_trace(w: &[Q], u: &[Q]) -> f64 {
    assert_eq!(w.len(), u.len());
    w.iter()
        .zip(u)
        .map(|(&w, u)| {
            let wd = qdag(w);
            let trace = 2.0 * (u[0] * wd[0] + u[1] * wd[1] + u[2] * wd[2] + u[3] * wd[3]);
            let ecart = (trace - 2.0 * qnorm(w)).abs();
            assert!(ecart.is_finite());
            ecart
        })
        .fold(0.0, f64::max)
}

#[test]
#[ignore = "GPU"]
fn topo_gpu_cool_id() {
    let Some(app) = appareil("topo_gpu_cool_id") else {
        return;
    };
    let m = ModuleTopo::nouveau(&app).unwrap();
    let w = topo::cool_id_matrices(26102619002, 1000);
    for dagger in [true, false] {
        let u = m.cool_id_maj(&w, dagger).unwrap();
        assert_eq!(u.len(), w.len());
        let mut delta: f64 = 0.0;
        for (&w, ug) in w.iter().zip(&u) {
            let reference = topo::maj_cooling(w, dagger);
            for (&a, b) in ug.iter().zip(reference) {
                proche(a, b, 1e-15);
                delta = delta.max((a - b).abs());
            }
        }
        let ecart = ecart_trace(&w, &u);
        if dagger {
            assert!(ecart <= 1e-13)
        } else {
            assert!(ecart >= 1e-2)
        }
        assert_eq!(
            m.cool_id_maj(&[[0.0; 4]], dagger).unwrap(),
            vec![ref_cpu::UN]
        );
        assert!(m.cool_id_maj(&[], dagger).unwrap().is_empty());
        println!(
            "PASS topo_gpu_cool_id (dagger={dagger}, max|ΔU|={delta:.3e}, écart_trace={ecart:.3e})"
        );
    }
}

/// Compare toutes les données numériques (hors t0 et t_eff, traités séparément).
fn donnees(r: &RecordTopo) -> Vec<f64> {
    let mut valeurs = vec![r.plaq0];
    for s in [
        &r.flot_a2.e_plaq,
        &r.flot_a2.q_tre,
        &r.flot_a2.q_imp,
        &r.cooling.q_tre,
        &r.cooling.q_imp,
        &r.cooling.action,
    ] {
        valeurs.extend(s);
    }
    if let Some(f) = &r.flot_t0 {
        valeurs.extend(&f.q_tre);
        valeurs.extend(&f.q_imp);
    }
    valeurs
}

fn records_proches(a: &RecordTopo, b: &RecordTopo) {
    assert_eq!(
        (&a.ens, a.chaine, a.cfg, &a.backend_gen, &a.fichier, a.l),
        (&b.ens, b.chaine, b.cfg, &b.backend_gen, &b.fichier, b.l)
    );
    for (a, b) in [
        (&a.flot_a2.e_plaq, &b.flot_a2.e_plaq),
        (&a.flot_a2.q_tre, &b.flot_a2.q_tre),
        (&a.flot_a2.q_imp, &b.flot_a2.q_imp),
        (&a.cooling.action, &b.cooling.action),
        (&a.cooling.q_tre, &b.cooling.q_tre),
        (&a.cooling.q_imp, &b.cooling.q_imp),
    ] {
        assert_eq!(a.len(), b.len());
    }
    serie(&donnees(a), &donnees(b), 1e-11);
    assert_eq!(a.t0.is_some(), b.t0.is_some());
    if let (Some(a), Some(b)) = (a.t0, b.t0) {
        proche(a, b, 1e-9)
    }
    assert_eq!(a.flot_t0.is_some(), b.flot_t0.is_some());
    if let (Some(a), Some(b)) = (&a.flot_t0, &b.flot_t0) {
        bits(&a.t_eff, &b.t_eff);
        serie(&a.q_tre, &b.q_tre, 1e-11);
        serie(&a.q_imp, &b.q_imp, 1e-11);
    }
}

fn records_bits(a: &RecordTopo, b: &RecordTopo) {
    assert_eq!(a, b);
    bits(&donnees(a), &donnees(b));
    if let (Some(a), Some(b)) = (a.t0, b.t0) {
        assert_eq!(a.to_bits(), b.to_bits())
    }
    if let (Some(a), Some(b)) = (&a.flot_t0, &b.flot_t0) {
        bits(&a.t_eff, &b.t_eff)
    }
}

#[test]
#[ignore = "GPU"]
fn topo_gpu_mesurer() {
    let Some(app) = appareil("topo_gpu_mesurer") else {
        return;
    };
    let m = ModuleTopo::nouveau(&app).unwrap();
    let e = m.espace(6).unwrap();
    // Témoin de croisement déjà utilisé par OBS-3 dans tests/obs_gpu.rs.
    let champ = thermalise(6, 2.3, 101, 30);
    let p = ParamsTopo::default();
    let reference = topo::mesurer_cpu(&champ, &p).unwrap();
    assert!(reference.t0.is_some(), "le champ L=6 doit croiser t²E=0,3");
    let a = m.mesurer(&e, &champ, &p).unwrap();
    assert_eq!(a.backend_mesure, app.backend().nom());
    records_proches(&a, &reference);
    assert_eq!(a.flot_a2.q_imp.len(), 4);
    assert_eq!(a.flot_t0.as_ref().unwrap().t_eff.len(), 4);
    assert_eq!(a.cooling.q_imp.len(), 21);
    records_bits(&a, &m.mesurer(&e, &champ, &p).unwrap());

    let partiel = ParamsTopo {
        flot_complet: false,
        ..p.clone()
    };
    let b = m.mesurer(&e, &champ, &partiel).unwrap();
    records_proches(&b, &topo::mesurer_cpu(&champ, &partiel).unwrap());
    assert!(b.t0.is_none() && b.flot_t0.is_none());
    bits(&b.flot_a2.e_plaq, &a.flot_a2.e_plaq[..3]);
    bits(&b.flot_a2.q_tre, &a.flot_a2.q_tre[..3]);
    bits(&b.flot_a2.q_imp, &a.flot_a2.q_imp[..3]);
    assert_eq!(b.cooling, a.cooling);
    bits(&b.cooling.action, &a.cooling.action);
    bits(&b.cooling.q_tre, &a.cooling.q_tre);
    bits(&b.cooling.q_imp, &a.cooling.q_imp);
    assert_eq!(b.plaq0.to_bits(), a.plaq0.to_bits());

    let second = thermalise(6, 2.3, 26102619106, 40);
    for params in [&p, &partiel] {
        let reutilise = m.mesurer(&e, &second, params).unwrap();
        let neuf = m.mesurer(&m.espace(6).unwrap(), &second, params).unwrap();
        records_bits(&reutilise, &neuf);
    }
    assert_eq!(m.champ_hote(&e, Cible::Source).unwrap(), second);
    println!("PASS topo_gpu_mesurer (t0={:.17e}, t_eff={:?}, déterminisme et réutilisation au bit, partiel=3 temps)", a.t0.unwrap(), a.flot_t0.unwrap().t_eff);
}

#[test]
#[ignore = "GPU"]
fn topo_gpu_inst() {
    let Some(app) = appareil("topo_gpu_inst") else {
        return;
    };
    let m = ModuleTopo::nouveau(&app).unwrap();
    for (l, qt, qi) in [
        (8, 1.993_582_728_089_921_6, 2.001_525_336_395_644),
        (12, 1.998_731_082_900_731_6, 2.000_060_618_064_416_5),
    ] {
        let e = m.espace(l).unwrap();
        let champ = topo::champ_abelien(l);
        for (champ, signe) in [
            (champ.clone(), 1.0),
            (topo::transformer_jauge(&champ, 26102619001), 1.0),
            (topo::reflechir_axe3(&champ), -1.0),
        ] {
            m.charger(&e, &champ).unwrap();
            let c = m.charges(&e, Cible::Source).unwrap();
            proche(c.q_tre, signe * qt, 1e-10);
            proche(c.q_imp, signe * qi, 1e-10);
            charges_proches(c, topo::charges(&champ).unwrap());
            println!(
                "PASS topo_gpu_inst (L={l}, signe={signe}, q_tre={:.17e}, q_imp={:.17e})",
                c.q_tre, c.q_imp
            );
        }
    }
}

#[test]
fn noyaux_obs_flot_ajout_seul() {
    let src = include_str!("../noyaux/obs_flot.cu");
    // Empreinte FNV-1a du fichier historique complet : verrouille réellement l'ajout seul,
    // y compris le texte et le corps de k_obs_densites, sans dépendance supplémentaire.
    let historique = src
        .as_bytes()
        .get(..6542)
        .expect("source historique tronquée");
    let empreinte = historique.iter().fold(0xcbf29ce484222325_u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    assert_eq!(
        empreinte, 0x58317dfa0ecbe253,
        "le préfixe historique a changé"
    );
    assert!(src[..6542].contains("extern \"C\" __global__ void k_obs_densites"));
    assert!(src[..6542].contains("obs_trefle_plan(V, g, x, mu, nu, gp[p++])"));
    for nom in [
        "k_topo_densites",
        "k_topo_plaq",
        "k_topo_cool",
        "k_topo_cool_id",
    ] {
        assert!(
            src[6542..].contains(&format!("void {nom}(")),
            "noyau manquant : {nom}"
        );
    }
    assert_eq!(kvu::noyaux::TOUS.len(), 12);
    println!(
        "PASS noyaux_obs_flot_ajout_seul (6542 octets historiques intacts, 4 noyaux, registre=12)"
    );
}
