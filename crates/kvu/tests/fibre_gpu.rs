//! Gates GPU de la mesure de fibre : FIB-1 (14 cas + bonus, L ≤ 16), FIB-2 (déterminisme), FIB-3
//! (identité à l'octet CPU = GPU, `reference_k1`), champ résident F64 / F32, banc L = 16, 24, 32.
//!
//! Tous les tests sont `#[ignore = "GPU"]` : à lancer SOUS LE VERROU de la machine, par fenêtres courtes :
//!   flock -w 7200 /tmp/kevinotron_gpu.lock cargo test -p kvu --release --test fibre_gpu \
//!       -- --ignored --nocapture --test-threads=1
//! (gamer : /tmp/gamer_gpu.lock, `--features cuda`). Sans GPU, un test imprime `SKIP …`, qui ne vaut pas vert.
//! `KVU_FIBRE_FIXTURES=<dir>` désigne le jeu complet généré par gen_fixtures.py (sinon le jeu commité).

mod fibre_commun;

use fibre_commun::*;
use kvu::commun::{BackendKind, ChampHote, Erreur};
use kvu::cond::GeometrieCond;
use kvu::fibre::{plaquettes_gradients_cpu, poids_fourier, MesureFibre, Mesureur};
use kvu::gpu::{ouvrir, ouvrir_auto, Appareil};
use kvu::su2::{ChampGpu, TamponChamp};
use std::sync::Arc;
use std::time::Instant;

/// Ouvre le GPU : choix de `machine::choisir_backend` (T1) ; tant qu'il n'est pas livré, repli sur
/// `KEVINOTRON_BACKEND` (hip|cuda) puis sur le backend compilé (cuda si la feature est active, sinon hip).
fn appareil() -> Option<Appareil> {
    let ouvert = match ouvrir_auto() {
        Err(Erreur::NonImplemente(_)) => {
            let b = match std::env::var("KEVINOTRON_BACKEND").ok().as_deref() {
                Some("cpu") => return None,
                Some(nom) => nom.parse().expect("KEVINOTRON_BACKEND : hip|cuda|cpu"),
                None if cfg!(feature = "cuda") => BackendKind::Cuda,
                None => BackendKind::Hip,
            };
            ouvrir(b, 0).map(Some)
        }
        autre => autre,
    };
    match ouvert {
        Ok(Some(a)) => {
            println!("# GPU : {} ({}, {})", a.nom(), a.arch(), a.backend().nom());
            Some(a)
        }
        Ok(None) => None,
        Err(e) => {
            println!("# GPU indisponible : {e}");
            None
        }
    }
}

fn mesureur(app: &Appareil, l: usize) -> (Mesureur, kvu::fibre::PoidsFourier) {
    let w = poids_fourier(l, 1);
    let m = Mesureur::nouveau(Some(app), Arc::new(GeometrieCond::nouvelle(l).unwrap()), &w).expect("Mesureur GPU");
    (m, w)
}

fn octets_f64(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn octets_mesure(m: &MesureFibre) -> (Vec<u64>, Vec<u64>, Vec<u32>) {
    (
        octets_f64(&m.f),
        octets_f64(&m.d),
        m.blk.iter().map(|x| x.to_bits()).collect(),
    )
}

#[test]
#[ignore = "GPU"]
fn fib1_gpu_contre_cb() {
    let Some(app) = appareil() else {
        println!("SKIP FIB-1 GPU (aucun GPU)");
        return;
    };
    let racine = dossier_fixtures();
    let noms = noms_des_cas(&racine);
    let mut bilan = Bilan::default();
    let mut mes: Vec<(usize, Mesureur, kvu::fibre::PoidsFourier)> = Vec::new();
    for nom in &noms {
        let cas = charger_cas(&racine, nom);
        if cas.l > 16 {
            continue;
        }
        if !mes.iter().any(|(l, _, _)| *l == cas.l) {
            let (m, w) = mesureur(&app, cas.l);
            mes.push((cas.l, m, w));
        }
        let (_, m, w) = mes.iter_mut().find(|(l, _, _)| *l == cas.l).unwrap();
        let t0 = Instant::now();
        let (pv, gp) = m.plaquettes(None, 0, Some(&cas.champ)).expect("plaquettes GPU");
        let mesure = m.mesurer(None, 0, Some(&cas.champ)).expect("mesurer GPU");
        let libre = m.mesurer_libre(None, 0, Some(&cas.champ)).expect("mesurer_libre GPU");
        println!(
            "  cas {nom} : D = {:.3e} ({:.0} ms)",
            norme_frob(&mesure.d),
            t0.elapsed().as_secs_f64() * 1e3
        );
        controler_fib1(&mut bilan, &cas, &Resultats { pv, gp, mesure, libre }, w, "GPU ");
    }
    bilan.imprimer();
    let n = bilan.gates.iter().map(|g| g.1).max().unwrap_or(0);
    println!(
        "{} FIB-1 GPU (≤ {n} cas, {})",
        if bilan.ok() { "PASS" } else { "FAIL" },
        app.backend().nom()
    );
    assert!(bilan.ok(), "FIB-1 GPU ROUGE : {:?}", bilan.echecs);
}

/// FIB-2 : D identique à l'octet sur 2 exécutions et pour 2 géométries de lancement (nombre de
/// tranches par lancement) ; FIB-3 : GPU = CPU à l'octet (noyaux sans contraction FMA).
#[test]
#[ignore = "GPU"]
fn fib2_fib3_determinisme_et_identite_cpu() {
    let Some(app) = appareil() else {
        println!("SKIP FIB-2 FIB-3 (aucun GPU)");
        return;
    };
    let racine = dossier_fixtures();
    let mut ok2 = true;
    let mut ok3 = true;
    let mut n = 0;
    for nom in noms_des_cas(&racine) {
        let cas = charger_cas(&racine, &nom);
        if cas.l > 16 || cas.plat_ou_pi() {
            continue;
        }
        let (mut m, w) = mesureur(&app, cas.l);
        let a = m.mesurer(None, 0, Some(&cas.champ)).unwrap();
        let b = m.mesurer(None, 0, Some(&cas.champ)).unwrap();
        let al = m.mesurer_libre(None, 0, Some(&cas.champ)).unwrap();
        // deux grilles de lancement : une tranche par lancement, puis 7 tranches par lancement
        m.fixer_lot(1).unwrap();
        let c = m.mesurer(None, 0, Some(&cas.champ)).unwrap();
        m.fixer_lot(7).unwrap();
        let d = m.mesurer(None, 0, Some(&cas.champ)).unwrap();
        let cl = m.mesurer_libre(None, 0, Some(&cas.champ)).unwrap();
        let egal2 = octets_mesure(&a) == octets_mesure(&b)
            && octets_mesure(&a) == octets_mesure(&c)
            && octets_mesure(&a) == octets_mesure(&d)
            && octets_f64(&al.1) == octets_f64(&cl.1);
        println!(
            "  {} FIB-2 {nom} : D identique sur 2 exécutions et 3 grilles",
            if egal2 { "VERT " } else { "ROUGE" }
        );
        ok2 &= egal2;
        // FIB-3 : référence CPU
        let mut cpu = Mesureur::nouveau(None, Arc::new(GeometrieCond::nouvelle(cas.l).unwrap()), &w).unwrap();
        let rc = cpu.mesurer(None, 0, Some(&cas.champ)).unwrap();
        let (pv_c, gp_c) = plaquettes_gradients_cpu(&cas.champ);
        let (pv_g, gp_g) = m.plaquettes(None, 0, Some(&cas.champ)).unwrap();
        let egal3 = octets_mesure(&rc) == octets_mesure(&a)
            && octets_f64(&pv_c) == octets_f64(&pv_g)
            && octets_f64(&gp_c) == octets_f64(&gp_g);
        println!(
            "  {} FIB-3 {nom} : GPU = CPU à l'octet (pv, gp, F, D, blk)",
            if egal3 { "VERT " } else { "ROUGE" }
        );
        ok3 &= egal3;
        n += 1;
    }
    // Tailles hors fixtures (tranches et sous-lots partiels, L non puissance de 2) : GPU = CPU à l'octet.
    for l in [6usize, 10] {
        let champ = kvu::fibre::banc::champ_pseudo_aleatoire(l, 90 + l as u64);
        let (mut m, w) = mesureur(&app, l);
        let mut cpu = Mesureur::nouveau(None, Arc::new(GeometrieCond::nouvelle(l).unwrap()), &w).unwrap();
        let egal = octets_mesure(&cpu.mesurer(None, 0, Some(&champ)).unwrap())
            == octets_mesure(&m.mesurer(None, 0, Some(&champ)).unwrap())
            && octets_f64(&cpu.mesurer_libre(None, 0, Some(&champ)).unwrap().1)
                == octets_f64(&m.mesurer_libre(None, 0, Some(&champ)).unwrap().1);
        println!(
            "  {} FIB-3 L{l} pseudo-aléatoire : GPU = CPU à l'octet (mesure et mesure libre)",
            if egal { "VERT " } else { "ROUGE" }
        );
        ok3 &= egal;
        n += 1;
    }
    println!("{} FIB-2 GPU ({n} cas)", if ok2 { "PASS" } else { "FAIL" });
    println!(
        "{} FIB-3 GPU = CPU à l'octet ({n} cas)",
        if ok3 { "PASS" } else { "FAIL" }
    );
    assert!(ok2, "FIB-2 ROUGE");
    assert!(ok3, "FIB-3 ROUGE");
}

/// Champ résident : chaîne 1 d'un lot de 2 chaînes, F64 puis F32 (promu en double dans le noyau).
#[test]
#[ignore = "GPU"]
fn champ_resident_f64_et_f32() {
    let Some(app) = appareil() else {
        println!("SKIP champ résident (aucun GPU)");
        return;
    };
    let racine = dossier_fixtures();
    let nom = noms_des_cas(&racine)
        .into_iter()
        .find(|n| n.contains("L8_haar0"))
        .expect("L8_haar0");
    let cas = charger_cas(&racine, &nom);
    let l = cas.l;
    let v = l.pow(4);
    let (mut m, w) = mesureur(&app, l);
    let froid = ChampHote::froid(l).unwrap();
    // F64 : chaîne 0 froide, chaîne 1 = cas
    let mut t64 = app.alloc::<f64>(2 * 16 * v).unwrap();
    app.ecrire(&mut t64, 0, &froid.q).unwrap();
    app.ecrire(&mut t64, 16 * v, &cas.champ.q).unwrap();
    let champ = ChampGpu {
        l,
        nchaines: 2,
        tampon: TamponChamp::F64(t64),
    };
    let depuis_hote = m.mesurer(None, 0, Some(&cas.champ)).unwrap();
    let residente = m.mesurer(Some(&champ), 1, None).unwrap();
    assert_eq!(
        octets_mesure(&depuis_hote),
        octets_mesure(&residente),
        "chaîne 1 résidente F64 ≠ hôte"
    );
    let froide = m.mesurer(Some(&champ), 0, None).unwrap();
    assert!(froide.d.iter().all(|&x| x == 0.0), "chaîne 0 froide : D = 0");
    // F32 : valeurs arrondies en f32 ; la référence CPU voit les mêmes valeurs promues
    let q32: Vec<f32> = cas.champ.q.iter().map(|&x| x as f32).collect();
    let promu = ChampHote {
        l,
        q: q32.iter().map(|&x| x as f64).collect(),
    };
    let mut t32 = app.alloc::<f32>(2 * 16 * v).unwrap();
    let froid32: Vec<f32> = froid.q.iter().map(|&x| x as f32).collect();
    app.ecrire(&mut t32, 0, &froid32).unwrap();
    app.ecrire(&mut t32, 16 * v, &q32).unwrap();
    let champ32 = ChampGpu {
        l,
        nchaines: 2,
        tampon: TamponChamp::F32(t32),
    };
    let r32 = m.mesurer(Some(&champ32), 1, None).unwrap();
    let mut cpu = Mesureur::nouveau(None, Arc::new(GeometrieCond::nouvelle(l).unwrap()), &w).unwrap();
    let rc = cpu.mesurer(None, 0, Some(&promu)).unwrap();
    assert_eq!(
        octets_mesure(&rc),
        octets_mesure(&r32),
        "champ F32 promu : GPU ≠ CPU sur les mêmes valeurs"
    );
    // mesurer_f = F de mesurer
    let f = m.mesurer_f(Some(&champ), 1, None).unwrap();
    assert_eq!(octets_f64(&f), octets_f64(&residente.f));
    // erreurs d'usage
    assert!(m.mesurer(Some(&champ), 2, None).is_err(), "chaîne hors lot");
    assert!(m.mesurer(None, 0, None).is_err(), "aucun champ");
    println!("PASS champ résident F64 (chaînes 0 et 1) et F32 promu");
}

/// Champ résident construit par T3 (`ChampGpu::allouer` + `televerser`) : tant que T3 n'est pas fusionné, SKIP.
#[test]
#[ignore = "GPU"]
fn champ_resident_de_t3() {
    use kvu::commun::Precision;
    let Some(app) = appareil() else {
        println!("SKIP champ résident T3 (aucun GPU)");
        return;
    };
    let racine = dossier_fixtures();
    let nom = noms_des_cas(&racine)
        .into_iter()
        .find(|n| n.contains("L8_haar0"))
        .expect("L8_haar0");
    let cas = charger_cas(&racine, &nom);
    let mut champ = match ChampGpu::allouer(&app, cas.l, 2, Precision::F64) {
        Ok(c) => c,
        Err(Erreur::NonImplemente(_)) => {
            println!("SKIP champ résident T3 (ChampGpu::allouer non livré)");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    champ.televerser(&app, 1, &cas.champ).expect("téléversement");
    let (mut m, _w) = mesureur(&app, cas.l);
    let depuis_hote = m.mesurer(None, 0, Some(&cas.champ)).unwrap();
    let residente = m.mesurer(Some(&champ), 1, None).unwrap();
    assert_eq!(
        octets_mesure(&depuis_hote),
        octets_mesure(&residente),
        "champ T3 résident ≠ hôte"
    );
    let relu = champ.telecharger(&app, 1).expect("téléchargement");
    assert_eq!(relu.q, cas.champ.q, "aller-retour du champ T3");
    println!("PASS champ résident de T3 (F64, chaîne 1 d'un lot de 2)");
}

/// `fibre mesurer` de bout en bout sur le GPU (verrou pris par un ancêtre, backend de la machine), avec son manifeste.
#[test]
#[ignore = "GPU"]
fn cli_gpu_de_bout_en_bout() {
    if appareil().is_none() {
        println!("SKIP CLI GPU (aucun GPU)");
        return;
    }
    let racine = dossier_fixtures();
    let nom = noms_des_cas(&racine)
        .into_iter()
        .find(|n| n.contains("L8_haar0"))
        .expect("L8_haar0");
    let cas = charger_cas(&racine, &nom);
    let l = cas.l;
    // sur le disque de la cible : la garde SSD de `fibre mesurer` refuse /tmp (tmpfs en RAM, < 100 Go)
    let base = option_env!("CARGO_TARGET_TMPDIR").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let tmp = base.join(format!("kvu_fibre_gpu_cli_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let cfg = tmp.join("cfg.npy");
    kvu::fibre::npy::ecrire_f64(&cfg, &[4, l, l, l, l, 4], &cas.champ.q).unwrap();
    let cmd = kvu::fibre::cli::commandes().remove(0);
    let backend = std::env::var("KEVINOTRON_BACKEND")
        .unwrap_or_else(|_| if cfg!(feature = "cuda") { "cuda" } else { "hip" }.into());
    for libre in [false, true] {
        let out = tmp.join(if libre { "libre" } else { "cond" });
        let mut argv = vec![
            "fibre",
            "mesurer",
            "--cfg",
            cfg.to_str().unwrap(),
            "--backend",
            &backend,
            "--out",
            out.to_str().unwrap(),
        ];
        if libre {
            argv.push("--libre");
        }
        let m = cmd.clone().try_get_matches_from(argv).unwrap();
        match kvu::fibre::cli::executer("fibre", &m) {
            Ok(rc) => assert_eq!(rc, 0),
            Err(Erreur::NonImplemente(c)) => {
                println!("SKIP CLI GPU ({c} non livré : T1)");
                std::fs::remove_dir_all(&tmp).ok();
                return;
            }
            Err(e) => panic!("{e}"),
        }
        let (_, d) = kvu::fibre::npy::lire_f64(&out.join("D.npy")).unwrap();
        let (_, f) = kvu::fibre::npy::lire_f64(&out.join("F.npy")).unwrap();
        let (dref, fref) = if libre {
            ("cb_Dlibre.npy", "cb_Flibre.npy")
        } else {
            ("cb_D.npy", "cb_F.npy")
        };
        assert!(ecart_frob(&d, &cas.f64s(dref)) <= TOL_D_REL);
        assert!(max_abs_ecart(&f, &cas.f64s(fref)) <= TOL_F_ABS);
        assert_eq!(out.join("blk.npy").exists(), !libre);
        let man: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(man["backend"], backend.as_str());
        assert!(man["modules_compiles"].as_array().is_some_and(|m| !m.is_empty()), "module fibre consigné");
        assert!(out.join("COMPLETE").exists());
    }
    std::fs::remove_dir_all(&tmp).ok();
    println!("PASS CLI GPU fibre mesurer (conditionnel et libre, backend {backend})");
}

// ------------------------------------------------------------------------------------------
// Banc
// ------------------------------------------------------------------------------------------

fn mediane(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Banc : mesure complète et mesure F seule à L = 16, 24, 32 sur GPU ; CPU (rayon) ; identité GPU = CPU.
/// Référence (README de noyau2, acemagic) : cb.mesure 124 ms à L = 16, 780 ms à L = 24, 2,04 s à L = 32.
#[test]
#[ignore = "GPU"]
fn banc_fibre() {
    let Some(app) = appareil() else {
        println!("SKIP banc (aucun GPU)");
        return;
    };
    let ls: Vec<usize> = std::env::var("KVU_FIBRE_BANC_L")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![16, 24, 32]);
    println!(
        "BANC fibre ({}, {}) : L, GPU mesurer, GPU mesurer_f, CPU (rayon) mesurer, cb.mesure (README noyau2), en ms",
        app.nom(),
        app.backend().nom()
    );
    for l in ls {
        let champ = kvu::fibre::banc::champ_pseudo_aleatoire(l, 40 + l as u64);
        let t0 = Instant::now();
        let (mut m, _w) = mesureur(&app, l);
        let t_init = t0.elapsed().as_secs_f64();
        let _ = m.mesurer(None, 0, Some(&champ)).unwrap(); // échauffement (compilation, cache)
        let b = kvu::fibre::banc::banc(Some(&app), l, Some(&champ), 5).unwrap();
        let tf: Vec<f64> = (0..5)
            .map(|_| {
                let t0 = Instant::now();
                let _ = m.mesurer_f(None, 0, Some(&champ)).unwrap();
                t0.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        // CPU (rayon) : durée, et identité à l'octet avec le GPU aux grands L (FIB-3 hors du périmètre des fixtures)
        let mut cpu = Mesureur::nouveau(
            None,
            Arc::new(GeometrieCond::nouvelle(l).unwrap()),
            &poids_fourier(l, 1),
        )
        .unwrap();
        let rc = cpu.mesurer(None, 0, Some(&champ)).unwrap();
        let t0 = Instant::now();
        let _ = cpu.mesurer(None, 0, Some(&champ)).unwrap();
        let cpu_ms = format!("{:.0}", t0.elapsed().as_secs_f64() * 1e3);
        let rg = m.mesurer(None, 0, Some(&champ)).unwrap();
        for (etape, ms) in m.dernier_profil() {
            println!("BANC L={l:2} :   étape {etape:28} {ms:8.2} ms");
        }
        let egal = octets_mesure(&rc) == octets_mesure(&rg);
        println!(
            "BANC L={l:2} : GPU = CPU à l'octet (F, D, blk) : {}",
            if egal { "oui" } else { "NON" }
        );
        assert!(egal, "L = {l} : GPU ≠ CPU");
        let cb = match l {
            16 => "124",
            24 => "780",
            32 => "2040",
            _ => "-",
        };
        println!(
            "BANC L={l:2} : GPU {:7.1} ms (min {:.1}) | champ résident {:7.1} ms | F seul {:6.1} ms | CPU {cpu_ms:>5} ms | cb {cb:>5} ms | init {:.2} s | ≈ {:.2} Go GPU",
            b.mesurer_ms,
            b.mesurer_min_ms,
            b.mesurer_residente_ms.unwrap_or(f64::NAN),
            mediane(tf),
            t_init,
            m.octets_gpu_estimes() as f64 / 1e9
        );
    }
}
