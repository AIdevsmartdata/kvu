//! Gates GPU des observables (T7) : OBS-1 à OBS-4, HIP (acemagic) et CUDA (gamer).
//!
//! Tous les tests sont `#[ignore = "GPU"]` et se lancent SOUS LE VERROU de la machine :
//! `flock -w 7200 /tmp/kevinotron_gpu.lock bash crates/kvu/validation/obs/gates_obs.sh`
//! (gamer : `/tmp/gamer_gpu.lock`, `--cuda`). Chaque test imprime `PASS <gate>`, `FAIL <gate>` (panique)
//! ou `SKIP <gate>` (pas de GPU) : un SKIP ne vaut pas vert.
//!
//! Les champs résidents sont construits ici avec le seul runtime T2 (`alloc` + `ecrire`), sans passer
//! par `su2::champ` (T3) : ces gates jugent les noyaux d'observables, pas le générateur.

use kvu::commun::{ChampHote, Precision};
use kvu::gpu::{self, Appareil};
use kvu::heritage::su2_hb::{measure_quat, qdag, qmul, CpuSu2, Geo, Q};
use kvu::obs::enligne;
use kvu::obs::flot::{densites, flot_champ, flot_wilson, ParamsFlot};
use kvu::obs::gpu::ModuleObs;
use kvu::obs::ref_cpu;
use kvu::obs::wilson::{boucles_wilson, correlateur_polyakov, polyakov_moyen, ParamsWilson};
use kvu::commun::{Depart, Graine};
use kvu::gpu::OptionsCompil;
use kvu::su2::{construire, ChampGpu, ConfigLot, ParamChaine, TamponChamp};
use std::path::PathBuf;
use std::time::Instant;

// ------------------------------------------------------------------------------------------
// Accès au GPU et construction des champs résidents
// ------------------------------------------------------------------------------------------

/// Le GPU de la machine (détection et politique de T1), ou `None` (⇒ SKIP, qui ne vaut pas vert).
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

/// Lot résident de `hotes.len()` chaînes ; F32 : arrondi au plus proche, SANS renormalisation (le noyau promeut).
fn lot_gpu(app: &Appareil, hotes: &[&ChampHote], prec: Precision) -> ChampGpu {
    let l = hotes[0].l;
    let v = l.pow(4);
    let tampon = match prec {
        Precision::F64 => {
            let mut t = app.alloc::<f64>(16 * v * hotes.len()).expect("allocation");
            for (c, h) in hotes.iter().enumerate() {
                app.ecrire(&mut t, c * 16 * v, &h.q).expect("écriture");
            }
            TamponChamp::F64(t)
        }
        Precision::F32 => {
            let mut t = app.alloc::<f32>(16 * v * hotes.len()).expect("allocation");
            for (c, h) in hotes.iter().enumerate() {
                let f: Vec<f32> = h.q.iter().map(|&x| x as f32).collect();
                app.ecrire(&mut t, c * 16 * v, &f).expect("écriture");
            }
            TamponChamp::F32(t)
        }
    };
    ChampGpu {
        l,
        nchaines: hotes.len(),
        tampon,
    }
}

/// Le champ hôte vu par un noyau F32 : les mêmes valeurs, arrondies en float puis promues.
fn promu_f32(h: &ChampHote) -> ChampHote {
    ChampHote {
        l: h.l,
        q: h.q.iter().map(|&x| x as f32 as f64).collect(),
    }
}

fn champ_thermalise(l: usize, graine: u64, beta: f64, n_pas: usize) -> ChampHote {
    let mut c = CpuSu2::new(l, graine).unwrap();
    c.init(true);
    for _ in 0..n_pas {
        c.step(beta, 2);
    }
    ChampHote { l, q: c.flat() }
}

fn quats(c: &ChampHote) -> Vec<Q> {
    ref_cpu::en_quaternions(c)
}

/// U_μ(x) → g(x) U_μ(x) g(x+μ)†, g de Haar (graine fixe).
fn jauge_haar(c: &ChampHote, graine: u64) -> ChampHote {
    let geo = Geo::new(c.l);
    let mut etat = graine;
    let mut u64_ = || {
        etat = etat.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = etat;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut unif = || ((u64_() >> 11) as f64 + 0.5) / 9007199254740992.0;
    let g: Vec<Q> = (0..geo.v)
        .map(|_| {
            let mut n = || (-2.0 * unif().ln()).sqrt() * (2.0 * std::f64::consts::PI * unif()).cos();
            let q = [n(), n(), n(), n()];
            let m = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            [q[0] / m, q[1] / m, q[2] / m, q[3] / m]
        })
        .collect();
    let u = quats(c);
    let mut out = u.clone();
    for mu in 0..4 {
        for s in 0..geo.v {
            let x = geo.coords(s);
            let smu = geo.nb_up(s, mu, x[mu]);
            out[mu * geo.v + s] = qmul(qmul(g[s], u[mu * geo.v + s]), qdag(g[smu]));
        }
    }
    ref_cpu::en_champ(&out, c.l)
}

fn ecart_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / (x.abs().max(y.abs()).max(1e-300)))
        .fold(0.0, f64::max)
}

fn ecart_abs(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn params(r: usize, t: usize, toutes: bool) -> ParamsWilson {
    ParamsWilson {
        r_max: r,
        t_max: t,
        dir_temps: 3,
        toutes_orientations: toutes,
    }
}

// ------------------------------------------------------------------------------------------
// OBS-1 : boucles de Wilson, GPU contre CPU quaternion, critère ABSOLU max|ΔW| ≤ 1e-15
// ------------------------------------------------------------------------------------------
// Arbitrage D3 (GATES.md v1, lot WP2 ; docs/ARBITRAGES_D3.md) : l'ancien seuil « ≤ 1e-13 relatif » divergeait quand
// W → 0 (W(5,4) = −1,26e-5 : écart absolu 3,5e-18, écart relatif 2,75e-13). Les autres volets d'OBS-1 sont inchangés.

/// Seuil absolu d'OBS-1 sur toutes les tables Wilson GPU/CPU (D3, remplace « ≤ 1e-13 relatif »).
const SEUIL_OBS1_ABS: f64 = 1e-15;

/// Écart absolu maximal ; un NaN d'un côté donne +∞ (jamais ignoré, contrairement à `f64::max`).
fn ecart_abs_strict(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = (x - y).abs();
            if d.is_nan() {
                f64::INFINITY
            } else {
                d
            }
        })
        .fold(0.0, f64::max)
}

/// Juge d'OBS-1 : Ok(écart) si max|ΔW| ≤ SEUIL_OBS1_ABS, Err(écart) sinon. Employé par la gate ET par son mutant.
fn juge_obs1(gpu: &[f64], cpu: &[f64]) -> Result<f64, f64> {
    let e = ecart_abs_strict(gpu, cpu);
    if e <= SEUIL_OBS1_ABS {
        Ok(e)
    } else {
        Err(e)
    }
}

#[test]
#[ignore = "GPU"]
fn obs_1_wilson_gpu_contre_cpu() {
    let Some(app) = appareil("OBS-1") else { return };
    let h0 = champ_thermalise(8, 11, 2.3, 8);
    let h1 = champ_thermalise(8, 12, 2.8, 8);
    let lot = lot_gpu(&app, &[&h0, &h1], Precision::F64);

    // GPU contre CPU, 12 orientations puis temporelles seules, deux chaînes d'un même lot
    let mut pire: f64 = 0.0;
    for (c, h) in [&h0, &h1].into_iter().enumerate() {
        for toutes in [true, false] {
            let p = params(6, 5, toutes);
            let g = boucles_wilson(Some(&app), Some(&lot), c, None, &p).expect("GPU");
            let r = ref_cpu::table_wilson(h, &p).unwrap();
            pire = pire.max(juge_obs1(&g.w, &r.w).unwrap_or_else(|e| e));
        }
    }
    assert!(
        pire <= SEUIL_OBS1_ABS,
        "FAIL OBS-1 : Wilson GPU contre CPU : écart absolu {pire:e} > {SEUIL_OBS1_ABS:e}"
    );

    // un L non puissance de 2 et r_max = L/2 + 1 (bouclage de la périodicité)
    let h6 = champ_thermalise(6, 13, 2.3, 8);
    let lot6 = lot_gpu(&app, &[&h6], Precision::F64);
    let p = params(4, 7, true);
    let e6 = ecart_abs_strict(
        &boucles_wilson(Some(&app), Some(&lot6), 0, None, &p).unwrap().w,
        &ref_cpu::table_wilson(&h6, &p).unwrap().w,
    );
    assert!(e6 <= SEUIL_OBS1_ABS, "FAIL OBS-1 : L = 6, écart absolu {e6:e} > {SEUIL_OBS1_ABS:e}");

    // champ F32 promu : GPU F32 contre CPU sur les mêmes valeurs promues
    let lot32 = lot_gpu(&app, &[&h0], Precision::F32);
    let p = params(4, 4, true);
    let e32 = ecart_abs_strict(
        &boucles_wilson(Some(&app), Some(&lot32), 0, None, &p).unwrap().w,
        &ref_cpu::table_wilson(&promu_f32(&h0), &p).unwrap().w,
    );
    assert!(
        e32 <= SEUIL_OBS1_ABS,
        "FAIL OBS-1 : champ F32 promu, écart absolu {e32:e} > {SEUIL_OBS1_ABS:e}"
    );

    // champ froid : W = 1 exactement (L = 8, V puissance de 2)
    let froid = ChampHote::froid(8).unwrap();
    let lot_f = lot_gpu(&app, &[&froid], Precision::F64);
    let w = boucles_wilson(Some(&app), Some(&lot_f), 0, None, &params(5, 5, true)).unwrap();
    assert!(
        w.w.iter().all(|&x| x == 1.0),
        "FAIL OBS-1 : champ froid, W ≠ 1 : {:?}",
        w.w
    );

    // invariance de jauge sur GPU
    let hg = jauge_haar(&h0, 99);
    let lot_g = lot_gpu(&app, &[&hg], Precision::F64);
    let p = params(4, 4, true);
    let eg = ecart_abs(
        &boucles_wilson(Some(&app), Some(&lot), 0, None, &p).unwrap().w,
        &boucles_wilson(Some(&app), Some(&lot_g), 0, None, &p).unwrap().w,
    );
    assert!(eg <= 1e-13, "FAIL OBS-1 : invariance de jauge GPU {eg:e}");

    // déterminisme : deux lancements identiques à l'octet
    let a = boucles_wilson(Some(&app), Some(&lot), 0, None, &p).unwrap();
    let b = boucles_wilson(Some(&app), Some(&lot), 0, None, &p).unwrap();
    assert_eq!(a, b, "FAIL OBS-1 : deux lancements diffèrent");

    // r_max > limite du noyau : refus propre
    assert!(boucles_wilson(Some(&app), Some(&lot), 0, None, &params(33, 1, false)).is_err());
    // chaîne hors du lot : refus propre
    assert!(boucles_wilson(Some(&app), Some(&lot), 5, None, &params(1, 1, false)).is_err());

    println!("PASS OBS-1 (Wilson GPU contre CPU, écart absolu max {pire:e} ≤ {SEUIL_OBS1_ABS:e} ; L = 6 : {e6:e} ; F32 promu : {e32:e} ; jauge : {eg:e} ; froid exact)");
}

/// Mutant d'OBS-1 (GATES v1, règle 4) : +1e-14 ajouté à UN W de la référence CPU. Le juge d'OBS-1 doit alors rendre ROUGE
/// (écart ≥ 1e-14 > 1e-15) ; la référence non mutée doit rester verte. Un mutant vert = ECHEC_OUTIL.
/// Imprime `PASS OBS1-MUTANT` quand le mutant est bien détecté (c'est le contrôle négatif qui passe).
#[test]
#[ignore = "GPU"]
fn obs_1_mutant_rouge() {
    let Some(app) = appareil("OBS1-MUTANT") else { return };
    let h0 = champ_thermalise(8, 11, 2.3, 8);
    let lot = lot_gpu(&app, &[&h0], Precision::F64);
    let p = params(6, 5, true);
    let g = boucles_wilson(Some(&app), Some(&lot), 0, None, &p).expect("GPU");
    let r = ref_cpu::table_wilson(&h0, &p).unwrap();
    let propre = juge_obs1(&g.w, &r.w);
    assert!(
        propre.is_ok(),
        "FAIL OBS1-MUTANT : la référence non mutée est déjà rouge ({propre:?}) : le contrôle négatif est sans objet"
    );
    let mut mutant = r.w.clone();
    let k = mutant.len() / 2;
    mutant[k] += 1e-14;
    match juge_obs1(&g.w, &mutant) {
        Err(e) => println!(
            "PASS OBS1-MUTANT (+1e-14 sur W[{k}] de la référence CPU : OBS-1 ROUGE comme attendu, écart absolu {e:e} > {SEUIL_OBS1_ABS:e} ; référence propre {:e})",
            propre.unwrap_or(f64::NAN)
        ),
        Ok(e) => panic!("FAIL OBS1-MUTANT : ECHEC_OUTIL : le mutant +1e-14 passe OBS-1 (écart {e:e})"),
    }
}

/// Diagnostic hors gate (intégration) : conditionnement de l'écart relatif d'OBS-1.
///
/// W(R,T) est la moyenne de termes ½tr ∈ [−1, 1] ; aux grandes boucles elle est petite devant ses termes, et un
/// ordre de sommation différent (arbre fixe GPU contre boucle CPU) donne un écart ABSOLU ~ ε, donc un écart
/// RELATIF ~ ε/|W|. N'affirme rien : imprime, pour chaque chaîne, la case de pire écart relatif, son écart
/// absolu et le pire écart absolu de la table (lignes `INFO OBS-1-DIAG`). Le seuil gelé d'OBS-1 n'est pas touché.
#[test]
#[ignore = "GPU"]
fn obs_1_diagnostic_conditionnement() {
    let Some(app) = appareil("OBS-1-DIAG") else { return };
    let h0 = champ_thermalise(8, 11, 2.3, 8);
    let h1 = champ_thermalise(8, 12, 2.8, 8);
    let lot = lot_gpu(&app, &[&h0, &h1], Precision::F64);
    for (c, h) in [&h0, &h1].into_iter().enumerate() {
        for toutes in [true, false] {
            let p = params(6, 5, toutes);
            let g = boucles_wilson(Some(&app), Some(&lot), c, None, &p).expect("GPU");
            let r = ref_cpu::table_wilson(h, &p).unwrap();
            let (mut pire_rel, mut case, mut pire_abs) = (0.0_f64, 0, 0.0_f64);
            for (i, (x, y)) in g.w.iter().zip(&r.w).enumerate() {
                let abs = (x - y).abs();
                let rel = abs / x.abs().max(y.abs()).max(1e-300);
                pire_abs = pire_abs.max(abs);
                if rel > pire_rel {
                    (pire_rel, case) = (rel, i);
                }
            }
            let (rr, tt) = (case / p.t_max + 1, case % p.t_max + 1);
            println!(
                "INFO OBS-1-DIAG chaîne {c} toutes={toutes} : pire relatif {pire_rel:.3e} en W({rr},{tt}) = {:.6e} \
                 (GPU {:.17e}, CPU {:.17e}, |Δ| {:.3e}) ; pire |Δ| de la table {pire_abs:.3e}",
                r.w[case],
                g.w[case],
                r.w[case],
                (g.w[case] - r.w[case]).abs()
            );
        }
    }
}

// ------------------------------------------------------------------------------------------
// OBS-2 : Polyakov
// ------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn obs_2_polyakov_gpu() {
    let Some(app) = appareil("OBS-2") else { return };
    let h = champ_thermalise(8, 21, 2.3, 10);
    let lot = lot_gpu(&app, &[&h], Precision::F64);
    let geo = Geo::new(8);

    // P0..P3 contre la mesure de Ref64 (mêmes arbres fixes) : ≤ 1e-15
    let ref64 = measure_quat(&geo, &quats(&h));
    let p = polyakov_moyen(Some(&app), Some(&lot), 0, None).unwrap();
    let mut ep: f64 = 0.0;
    for mu in 0..4 {
        ep = ep.max((p[mu] - ref64[1 + mu]).abs());
    }
    assert!(ep <= 1e-15, "FAIL OBS-2 : P0..P3 GPU contre Ref64 : {ep:e}");

    // corrélateur GPU contre CPU : ≤ 1e-13
    let mut ec: f64 = 0.0;
    for dir in 0..4 {
        let g = correlateur_polyakov(Some(&app), Some(&lot), 0, None, dir).unwrap();
        let c = ref_cpu::correlateur_polyakov(&h, dir).unwrap();
        assert_eq!(g.len(), 5);
        ec = ec.max(ecart_abs(&g, &c));
    }
    assert!(ec <= 1e-13, "FAIL OBS-2 : corrélateur GPU contre CPU : {ec:e}");

    // F32 promu
    let lot32 = lot_gpu(&app, &[&h], Precision::F32);
    let hp = promu_f32(&h);
    let g32 = correlateur_polyakov(Some(&app), Some(&lot32), 0, None, 3).unwrap();
    assert!(
        ecart_abs(&g32, &ref_cpu::correlateur_polyakov(&hp, 3).unwrap()) <= 1e-13,
        "FAIL OBS-2 : F32"
    );
    let p32 = polyakov_moyen(Some(&app), Some(&lot32), 0, None).unwrap();
    assert!(
        ecart_abs(&p32, &ref_cpu::polyakov_moyen(&hp).unwrap()) <= 1e-15,
        "FAIL OBS-2 : F32, P0..P3"
    );

    // champ froid : tout vaut 1
    let froid = ChampHote::froid(8).unwrap();
    let lot_f = lot_gpu(&app, &[&froid], Precision::F64);
    let c = correlateur_polyakov(Some(&app), Some(&lot_f), 0, None, 3).unwrap();
    assert!(c.iter().all(|&x| (x - 1.0).abs() <= 1e-15));
    println!("PASS OBS-2 (P0..P3 contre Ref64 : {ep:e} ; corrélateur contre CPU : {ec:e})");
}

// ------------------------------------------------------------------------------------------
// OBS-3 : flot, 50 pas à dt = 0,01
// ------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn obs_3_flot_gpu_contre_cpu() {
    let Some(app) = appareil("OBS-3") else { return };
    let h = champ_thermalise(8, 31, 2.3, 10);
    let lot = lot_gpu(&app, &[&h], Precision::F64);

    // liens après 50 pas
    let g = flot_champ(Some(&app), Some(&lot), 0, None, 0.01, 50).unwrap();
    let r = ref_cpu::flot_n_pas(&h, 0.01, 50).unwrap();
    let elien = ecart_abs(&g.q, &r.q);
    assert!(elien <= 1e-12, "FAIL OBS-3 : liens GPU contre CPU : {elien:e}");
    assert!(g.defaut_unitarite() <= 1e-14);

    // E(t) : courbe de 50 pas, mesure à chaque pas
    let p = ParamsFlot {
        dt: 0.01,
        t_max: 0.5,
        mesure_chaque: 1,
        ..ParamsFlot::default()
    };
    let cg = flot_wilson(Some(&app), Some(&lot), 0, None, &p).unwrap();
    let cc = ref_cpu::flot(&h, &p).unwrap();
    let ee = ecart_rel(&cg.e_plaq, &cc.e_plaq).max(ecart_rel(&cg.e_trefle, &cc.e_trefle));
    assert!(ee <= 1e-12, "FAIL OBS-3 : E(t) relatif {ee:e}");
    let eq = ecart_abs(&cg.q_top, &cc.q_top);
    assert!(eq <= 1e-12, "FAIL OBS-3 : Q_top {eq:e}");
    assert_eq!(cg.t.len(), 51);

    // t0 / w0 : GPU et CPU sur un flot plus long (L = 6, β = 2,3 : t²E croise 0,3)
    let h6 = champ_thermalise(6, 101, 2.3, 30);
    let lot6 = lot_gpu(&app, &[&h6], Precision::F64);
    let p6 = ParamsFlot {
        dt: 0.05,
        t_max: 2.0,
        ..ParamsFlot::default()
    };
    let g6 = flot_wilson(Some(&app), Some(&lot6), 0, None, &p6).unwrap();
    let c6 = ref_cpu::flot(&h6, &p6).unwrap();
    let (t0g, t0c) = (g6.t0.expect("t0 GPU non encadré"), c6.t0.expect("t0 CPU non encadré"));
    assert!((t0g - t0c).abs() <= 1e-12, "FAIL OBS-3 : t0 GPU {t0g} contre CPU {t0c}");
    assert!((g6.w0.unwrap() - c6.w0.unwrap()).abs() <= 1e-12, "FAIL OBS-3 : w0");

    // jauge pure : E ≡ 0 (≤ 1e-14) le long du flot GPU
    let pur = jauge_haar(&ChampHote::froid(8).unwrap(), 7);
    let lot_p = lot_gpu(&app, &[&pur], Precision::F64);
    let pp = ParamsFlot {
        dt: 0.02,
        t_max: 0.2,
        ..ParamsFlot::default()
    };
    let cp = flot_wilson(Some(&app), Some(&lot_p), 0, None, &pp).unwrap();
    let epure = cp
        .e_plaq
        .iter()
        .chain(&cp.e_trefle)
        .chain(&cp.q_top)
        .fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(epure <= 1e-14, "FAIL OBS-3 : jauge pure, E ≠ 0 : {epure:e}");

    // le champ source n'est pas modifié (relu sur le GPU)
    let mut apres = vec![0.0; h.q.len()];
    if let TamponChamp::F64(t) = &lot.tampon {
        app.synchroniser().unwrap();
        app.lire(t, 0, &mut apres).unwrap();
    }
    assert_eq!(apres, h.q, "FAIL OBS-3 : le champ source a été modifié");

    // champ F32 : flot GPU contre CPU sur les valeurs promues
    let lot32 = lot_gpu(&app, &[&h], Precision::F32);
    let g32 = flot_champ(Some(&app), Some(&lot32), 0, None, 0.01, 50).unwrap();
    let r32 = ref_cpu::flot_n_pas(&promu_f32(&h), 0.01, 50).unwrap();
    let e32 = ecart_abs(&g32.q, &r32.q);
    assert!(e32 <= 1e-12, "FAIL OBS-3 : flot depuis un champ F32 : {e32:e}");

    // densités du champ tel quel
    let dg = densites(Some(&app), Some(&lot), 0, None).unwrap();
    let dc = ref_cpu::densites(&h).unwrap();
    assert!(
        (dg.e_plaq - dc.e_plaq).abs() / dc.e_plaq <= 1e-13 && (dg.e_trefle - dc.e_trefle).abs() / dc.e_trefle <= 1e-13
    );
    assert!((dg.q_top - dc.q_top).abs() <= 1e-13);
    println!("PASS OBS-3 (liens : {elien:e} ; E : {ee:e} ; Q : {eq:e} ; t0 GPU = CPU ({t0g:.6}) ; jauge pure : {epure:e} ; F32 : {e32:e})");
}

// ------------------------------------------------------------------------------------------
// OBS-4 : Q_top après flot. Arbitrage D3 (GATES.md v1, lot WP2) : la version L = 8 est reclassée INFO (rapportée,
// ni PASS ni FAIL) ; la gate de quantification est OBS-4-L16 (≥ 18/20 à 0,1, t = 3).
// ------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn obs_4_topologie_apres_flot_rapport() {
    let Some(app) = appareil("OBS-4") else { return };
    let (l, beta, n) = (8usize, 2.5, 20usize);
    let mut proches = 0;
    let mut valeurs = Vec::new();
    for k in 0..n {
        // décorrélation : graine différente, 150 pas de thermalisation par configuration
        let h = champ_thermalise(l, 4000 + k as u64, beta, 150);
        let lot = lot_gpu(&app, &[&h], Precision::F64);
        let p = ParamsFlot {
            dt: 0.02,
            t_max: 3.0,
            mesure_chaque: 150,
            ..ParamsFlot::default()
        };
        let c = flot_wilson(Some(&app), Some(&lot), 0, None, &p).unwrap();
        let q = *c.q_top.last().unwrap();
        valeurs.push(q);
        if (q - q.round()).abs() < 0.1 {
            proches += 1;
        }
    }
    let frac = proches as f64 / n as f64;
    println!(
        "Q_top après flot (t = 3, L = 8, β = 2,5) : {:?}",
        valeurs
            .iter()
            .map(|q| (q * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
    println!(
        "INFO OBS-4-L8 : |Q − round(Q)| < 0,1 pour {proches}/{n} configurations ({:.0} %), L = 8, β = 2,5, t = 3 (rapportée, ne verdit pas ; quantification = OBS-4-L16)",
        100.0 * frac
    );
}

/// OBS-4-L16 (D3) : quantification de la charge topologique après flot à L = 16, β = 2,5 : 20 configurations d'UNE chaîne
/// (thermalisation 2000 pas, une configuration tous les 100 pas), flot de Wilson t = 3 (dt = 0,02) ; seuil gelé
/// ≥ 18/20 avec |Q − round(Q)| < 0,1. La chaîne est produite par le générateur GPU F64 compat-prod (pas = 1 bain thermique
/// + 4 OR, comme `kvu su2 gen`) ; graine fixe. PASS/FAIL sur le seuil gelé, jamais ajusté.
#[test]
#[ignore = "GPU"]
fn obs_4_l16_quantification() {
    let Some(app) = appareil("OBS-4-L16") else { return };
    let (l, beta, n, therm, espace) = (16usize, 2.5, 20usize, 2000usize, 100usize);
    let cfg = ConfigLot {
        l,
        chaines: vec![ParamChaine {
            beta,
            graine: Graine(26_102_516_001),
            depart: Depart::Chaud,
        }],
        nor: 4,
        precision: Precision::F64,
        options: OptionsCompil::compat_production(app.backend()),
        mutant_metro: false,
    };
    let mut gen = construire(Some(&app), cfg).expect("générateur GPU");
    gen.init().expect("départ chaud");
    let pas = |gen: &mut Box<dyn kvu::su2::NoyauxSu2>| {
        gen.sweep_hb().expect("bain thermique");
        for _ in 0..4 {
            gen.sweep_or().expect("overrelaxation");
        }
    };
    let t0 = Instant::now();
    for _ in 0..therm {
        pas(&mut gen);
    }
    let mut proches = 0;
    let mut valeurs = Vec::new();
    for k in 0..n {
        if k > 0 {
            for _ in 0..espace {
                pas(&mut gen);
            }
        }
        gen.synchroniser().expect("synchronisation");
        let champ = gen.champ_gpu().expect("champ résident");
        let p = ParamsFlot {
            dt: 0.02,
            t_max: 3.0,
            mesure_chaque: 150,
            ..ParamsFlot::default()
        };
        let c = flot_wilson(Some(&app), Some(champ), 0, None, &p).expect("flot");
        let q = *c.q_top.last().unwrap();
        valeurs.push(q);
        if (q - q.round()).abs() < 0.1 {
            proches += 1;
        }
    }
    println!(
        "Q_top après flot (t = 3, L = 16, β = 2,5) : {:?} ({:.0} s)",
        valeurs.iter().map(|q| (q * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        t0.elapsed().as_secs_f64()
    );
    assert!(
        proches >= 18,
        "FAIL OBS-4-L16 : |Q − round(Q)| < 0,1 pour {proches}/{n} configurations (seuil gelé ≥ 18/{n}), L = 16, β = 2,5, t = 3"
    );
    println!("PASS OBS-4-L16 : |Q − round(Q)| < 0,1 pour {proches}/{n} configurations (seuil gelé ≥ 18/{n}), L = 16, β = 2,5, t = 3");
}

/// Diagnostic HORS GATE d'OBS-4-L16 (ajouté après le premier verdict ROUGE, pour l'arbitre ; ne verdit rien, ne change aucun
/// seuil) : régénère les mêmes 20 configurations (mêmes graine et pas que la gate), vérifie la plaquette (≈ 0,65 à β = 2,5, entre 0,602 à β = 2,3 et 0,6995 à β = 2,8) et
/// rapporte, pour des temps de flot t ∈ {3 ; 4,5 ; 6 ; 9 ; 12}, combien de charges sont à moins de 0,1 d'un entier (lignes
/// `INFO OBS-4-L16-DIAG`). Sert à dire si la non-quantification vient du temps de flot (clover non amélioré) ou du générateur.
#[test]
#[ignore = "GPU"]
fn obs_4_l16_diagnostic_flot_long() {
    let Some(app) = appareil("OBS-4-L16-DIAG") else { return };
    let cfg = ConfigLot {
        l: 16,
        chaines: vec![ParamChaine {
            beta: 2.5,
            graine: Graine(26_102_516_001),
            depart: Depart::Chaud,
        }],
        nor: 4,
        precision: Precision::F64,
        options: OptionsCompil::compat_production(app.backend()),
        mutant_metro: false,
    };
    let mut gen = construire(Some(&app), cfg).expect("générateur GPU");
    gen.init().expect("départ chaud");
    for _ in 0..2000 {
        gen.sweep_hb().unwrap();
        for _ in 0..4 {
            gen.sweep_or().unwrap();
        }
    }
    let temps = [3.0, 4.5, 6.0, 9.0, 12.0];
    let mut proches = [0usize; 5];
    let mut plaq = Vec::new();
    for k in 0..20 {
        if k > 0 {
            for _ in 0..100 {
                gen.sweep_hb().unwrap();
                for _ in 0..4 {
                    gen.sweep_or().unwrap();
                }
            }
        }
        gen.synchroniser().unwrap();
        let champ = gen.champ_gpu().expect("champ résident");
        let w = boucles_wilson(Some(&app), Some(champ), 0, None, &params(1, 1, true)).unwrap();
        plaq.push(w.w[0]);
        let p = ParamsFlot {
            dt: 0.02,
            t_max: 12.0,
            mesure_chaque: 25,
            ..ParamsFlot::default()
        };
        let c = flot_wilson(Some(&app), Some(champ), 0, None, &p).unwrap();
        let mut ligne = Vec::new();
        for (i, &t) in temps.iter().enumerate() {
            let j = c
                .t
                .iter()
                .enumerate()
                .min_by(|a, b| (a.1 - t).abs().partial_cmp(&(b.1 - t).abs()).unwrap())
                .unwrap()
                .0;
            let q = c.q_top[j];
            ligne.push((q * 1000.0).round() / 1000.0);
            if (q - q.round()).abs() < 0.1 {
                proches[i] += 1;
            }
        }
        println!("INFO OBS-4-L16-DIAG config {k} : Q(t = 3 ; 4,5 ; 6 ; 9 ; 12) = {ligne:?}");
    }
    let pm = plaq.iter().sum::<f64>() / plaq.len() as f64;
    println!("INFO OBS-4-L16-DIAG plaquette moyenne des 20 configurations (W(1,1), toutes orientations) = {pm:.5} (attendu ≈ 0,65 : 0,602 à β = 2,3, 0,6995 à β = 2,8)");
    println!("INFO OBS-4-L16-DIAG |Q − round(Q)| < 0,1 : t = 3 : {}/20 ; 4,5 : {}/20 ; 6 : {}/20 ; 9 : {}/20 ; 12 : {}/20 (hors gate)", proches[0], proches[1], proches[2], proches[3], proches[4]);
}

// ------------------------------------------------------------------------------------------
// Mesures en ligne sur GPU
// ------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn obs_enligne_gpu_egal_cpu() {
    let Some(app) = appareil("OBS-ENLIGNE") else { return };
    let h = champ_thermalise(8, 41, 2.3, 8);
    let lot = lot_gpu(&app, &[&h, &h], Precision::F64);
    let base = std::env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let (dg, dc) = (base.join("kvu_obs_enligne_gpu"), base.join("kvu_obs_enligne_cpu"));
    for d in [&dg, &dc] {
        let _ = std::fs::remove_dir_all(d);
    }
    let spec = "wilson:R3xT3:toutes:10,polyakov_corr:10,flot:0.05:0.5:10:chaque=2";
    let mut mg = enligne::analyser_avec(spec, None).unwrap();
    let mut mc = enligne::analyser_avec(spec, None).unwrap();
    for pas in [10u64, 20] {
        for (a, b) in mg.iter_mut().zip(mc.iter_mut()) {
            a.mesurer(Some(&app), Some(&lot), None, 1, pas, &dg)
                .expect("mesure GPU");
            b.mesurer(None, None, Some(&h), 0, pas, &dc).expect("mesure CPU");
        }
    }
    for f in ["wilson.csv", "polyakov_corr.csv", "flot.csv", "flot_t0.csv"] {
        let (a, b) = (
            std::fs::read_to_string(dg.join(f)).unwrap(),
            std::fs::read_to_string(dc.join(f)).unwrap(),
        );
        let (la, lb): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
        assert_eq!(la.len(), lb.len(), "FAIL OBS-ENLIGNE : {f} : nombre de lignes");
        assert_eq!(la[0], lb[0]);
        for (x, y) in la.iter().zip(&lb).skip(1) {
            let (cx, cy): (Vec<&str>, Vec<&str>) = (x.split(',').collect(), y.split(',').collect());
            assert_eq!(
                &cx[..cx.len().min(3)][..1],
                &cy[..cy.len().min(3)][..1],
                "pas différent dans {f}"
            );
            for (u, v) in cx.iter().zip(&cy).skip(1) {
                if let (Ok(u), Ok(v)) = (u.parse::<f64>(), v.parse::<f64>()) {
                    if u.is_nan() && v.is_nan() {
                        continue;
                    }
                    assert!(
                        (u - v).abs() <= 1e-12 * (1.0 + u.abs()),
                        "FAIL OBS-ENLIGNE : {f} : {u} contre {v}"
                    );
                }
            }
        }
    }
    println!("PASS OBS-ENLIGNE (wilson, polyakov_corr, flot : CSV GPU = CSV CPU à 1e-12)");
}

// ------------------------------------------------------------------------------------------
// Banc rapporté (non gaté) : débit des observables sur GPU
// ------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn obs_banc_rapporte() {
    let Some(app) = appareil("OBS-BANC") else { return };
    for l in [16usize, 24] {
        let h = ChampHote::froid(l).unwrap();
        // champ rugueux mais bon marché : jauge de Haar sur le champ froid ne change pas les temps
        let h = if l <= 16 { champ_thermalise(l, 5, 2.3, 4) } else { h };
        let lot = lot_gpu(&app, &[&h], Precision::F64);
        let m = ModuleObs::nouveau(&app).unwrap();
        let t = Instant::now();
        let w = m.boucles_wilson(&lot, 0, &params(4, 4, true)).unwrap();
        println!(
            "BANC L = {l:2} : Wilson R4xT4 (12 orientations) {:.3} s (W11 = {:.6})",
            t.elapsed().as_secs_f64(),
            w.get(1, 1)
        );
        let t = Instant::now();
        let c = m
            .flot_wilson(
                &lot,
                0,
                &ParamsFlot {
                    dt: 0.01,
                    t_max: 0.5,
                    mesure_chaque: 10,
                    ..ParamsFlot::default()
                },
            )
            .unwrap();
        println!(
            "BANC L = {l:2} : flot 50 pas + 6 mesures {:.3} s (E_final = {:.6})",
            t.elapsed().as_secs_f64(),
            c.e_plaq.last().unwrap()
        );
        let t = Instant::now();
        let _ = m.correlateur_polyakov(&lot, 0, 3).unwrap();
        println!(
            "BANC L = {l:2} : corrélateur de Polyakov {:.4} s",
            t.elapsed().as_secs_f64()
        );
    }
}
