//! Gates GPU du générateur SU(2) unifié (T3). `#[ignore = "GPU"]` : à lancer SOUS le verrou GPU de la machine, par
//! fenêtres de 10 minutes au plus.
//!
//! ```text
//! acemagic : flock -w 7200 /tmp/kevinotron_gpu.lock cargo test -p kvu --release --test su2_gpu -- --ignored --nocapture --test-threads=1
//! gamer    : flock -w 7200 /tmp/gamer_gpu.lock      cargo test -p kvu --release --features cuda --test su2_gpu -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Le backend est KEVINOTRON_BACKEND s'il est posé, sinon CUDA si la feature `cuda` est compilée, sinon HIP.
//! Chaque gate imprime `PASS|FAIL|SKIP <gate> : détail` (un SKIP ne vaut PAS vert). Les références EXTERNES
//! (su2_gen historique, gpu_port, fixtures) sont lues EN PLACE ; aucun binaire de production n'est touché.
//!
//! Variables d'environnement des gates de bit-identité :
//! - KVU_REF_K1 : répertoire produit par `su2_gen --hip` historique (sous-répertoires L12 et L16) ;
//! - KVU_P2_AMD / KVU_P2_CUDA : arbres `p2` de gpu_port (défauts : chemins de l'inventaire).

use kvu::commun::{BackendKind, Depart, Erreur, Graine, Precision};
use kvu::gpu::{self, Appareil, OptionsCompil};
use kvu::heritage::su2_hb;
use kvu::su2::bord::{self, Bord};
use kvu::su2::cpu::Cpu64;
use kvu::su2::ref64::Ref64Gpu;
use kvu::su2::run::{self, ConfigBord, ConfigRun};
use kvu::su2::{construire, ConfigLot, Gel, NoyauxSu2, ParamChaine};
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------------------------
// Utilitaires
// ---------------------------------------------------------------------------------------------

fn backend_du_test() -> BackendKind {
    match std::env::var("KEVINOTRON_BACKEND").ok().and_then(|v| v.parse().ok()) {
        Some(b) if b != BackendKind::Cpu => b,
        _ if cfg!(feature = "cuda") => BackendKind::Cuda,
        _ => BackendKind::Hip,
    }
}

fn appareil_ou_skip(gate: &str) -> Option<Appareil> {
    let b = backend_du_test();
    match gpu::ouvrir(b, 0) {
        Ok(a) => Some(a),
        Err(e) => {
            verdict(&format!("SKIP {gate} : backend {} indisponible ({e})", b.nom()));
            None
        }
    }
}

/// Imprime une ligne de verdict au DÉBUT d'une ligne (libtest préfixe la première par « test nom ... »).
fn verdict(ligne: &str) {
    println!("\n{ligne}");
}

fn gate(nom: &str, corps: impl FnOnce() -> Result<String, String>) {
    match corps() {
        Ok(detail) => verdict(&format!("PASS {nom} : {detail}")),
        Err(raison) => {
            verdict(&format!("FAIL {nom} : {raison}"));
            panic!("FAIL {nom} : {raison}");
        }
    }
}

fn etiquette(base: &str) -> String {
    let suffixe = match backend_du_test() {
        BackendKind::Cuda => "CUDA",
        _ => "HIP",
    };
    format!("{base}-{suffixe}")
}

fn rep(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("su2_gpu").join(format!("{nom}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.parent().unwrap()).unwrap();
    d
}

fn lot(l: usize, chaines: &[(f64, u64, Depart)], nor: usize, options: OptionsCompil) -> ConfigLot {
    ConfigLot {
        l,
        chaines: chaines.iter().map(|&(beta, g, depart)| ParamChaine { beta, graine: Graine(g), depart }).collect(),
        nor,
        precision: Precision::F64,
        options,
        mutant_metro: false,
    }
}

#[allow(clippy::too_many_arguments)]
fn cfg_run(lot: ConfigLot, therm: u64, nmeas: u64, skip: u64, dump_every: u64, dump_initial: bool, outdir: &Path) -> ConfigRun {
    ConfigRun {
        lot,
        therm,
        nmeas,
        skip,
        dump_every,
        dump_initial,
        format_dump: "c16".into(),
        zstd: false,
        reprise_every: 0,
        outdir: outdir.to_path_buf(),
        bord: None,
    }
}

fn octets_champ(c: &kvu::commun::ChampHote) -> Vec<u8> {
    c.q.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn max_ecart(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Fichiers de `noms` identiques à l'octet entre deux répertoires.
fn memes_fichiers(a: &Path, b: &Path, noms: &[String]) -> Result<(), String> {
    for n in noms {
        let (fa, fb) = (fs::read(a.join(n)).map_err(|e| format!("{}/{n} : {e}", a.display()))?, fs::read(b.join(n)).map_err(|e| format!("{}/{n} : {e}", b.display()))?);
        if fa != fb {
            return Err(format!("{n} DIFFÈRE ({} contre {} octets)", fa.len(), fb.len()));
        }
    }
    Ok(())
}

fn noms_dumps(d: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(d)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("cfg_") && n.ends_with(".npy"))
        .collect();
    v.sort();
    v
}

fn avancer(n: &mut dyn NoyauxSu2, pas: usize, nor: usize) {
    for _ in 0..pas {
        n.sweep_hb().unwrap();
        for _ in 0..nor {
            n.sweep_or().unwrap();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// G-CPU-GPU : 1 sweep depuis la même configuration, CPU contre GPU F64 (avec et sans gel)
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn g_cpu_gpu() {
    let Some(app) = appareil_ou_skip("G-CPU-GPU") else { return };
    gate("G-CPU-GPU", || {
        let mut pire = 0.0f64;
        let mut cas = 0;
        for beta in [2.3, 2.5, 8.0] {
            for gel in [Gel::Aucun, Gel::BordX0Spatial] {
                let cfg = lot(6, &[(beta, 4242, Depart::Chaud)], 4, OptionsCompil::reference_k1());
                let mut g = Ref64Gpu::nouveau(&app, cfg.clone()).map_err(|e| e.to_string())?;
                g.init().map_err(|e| e.to_string())?;
                avancer(&mut g, 2, 4); // configuration décorrélée du départ
                let depart = g.telecharger(0).map_err(|e| e.to_string())?;
                let mut c = Cpu64::nouveau(cfg).map_err(|e| e.to_string())?;
                c.init().map_err(|e| e.to_string())?;
                c.televerser(0, &depart).map_err(|e| e.to_string())?;
                c.fixer_sweep_id(g.sweep_id());
                g.fixer_gel(gel).map_err(|e| e.to_string())?;
                c.fixer_gel(gel).map_err(|e| e.to_string())?;
                // un sweep de bain thermique : mêmes décisions (essais par lien) et mêmes liens à 1e-12
                let dg = g.sweep_hb_diag().map_err(|e| e.to_string())?;
                let dc = c.sweep_hb_diag().map_err(|e| e.to_string())?;
                let divergentes = dg.iter().zip(&dc).filter(|(a, b)| a != b).count();
                if divergentes != 0 {
                    return Err(format!("β = {beta}, gel {gel:?} : {divergentes} décisions divergentes (essais par lien)"));
                }
                let (qg, qc) = (g.telecharger(0).unwrap(), c.telecharger(0).unwrap());
                let e = max_ecart(&qg.q, &qc.q);
                pire = pire.max(e);
                if e > 1e-12 {
                    return Err(format!("β = {beta}, gel {gel:?} : max|Δq| = {e:.3e} > 1e-12 après un sweep HB"));
                }
                // un sweep d'overrelaxation
                g.sweep_or().map_err(|e| e.to_string())?;
                c.sweep_or().map_err(|e| e.to_string())?;
                let e2 = max_ecart(&g.telecharger(0).unwrap().q, &c.telecharger(0).unwrap().q);
                pire = pire.max(e2);
                if e2 > 1e-12 {
                    return Err(format!("β = {beta}, gel {gel:?} : max|Δq| = {e2:.3e} > 1e-12 après un sweep OR"));
                }
                cas += 1;
            }
        }
        Ok(format!("{cas} cas (L = 6, β ∈ {{2,3 ; 2,5 ; 8}}, avec et sans gel), 0 décision divergente, max|Δq| = {pire:.2e} ≤ 1e-12"))
    });
}

// ---------------------------------------------------------------------------------------------
// G-DET : double run identique à l'octet ; graine + 1 ⇒ différent
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn g_det() {
    let Some(app) = appareil_ou_skip("G-DET") else { return };
    gate("G-DET", || {
        let dir_a = rep("det_a");
        let dir_b = rep("det_b");
        let dir_c = rep("det_c");
        let mk = |g: u64, d: &Path| cfg_run(lot(8, &[(2.3, g, Depart::Chaud)], 4, OptionsCompil::reference_k1()), 30, 40, 2, 10, true, d);
        run::executer(Some(&app), &mk(777, &dir_a), &mut [], false).map_err(|e| e.to_string())?;
        run::executer(Some(&app), &mk(777, &dir_b), &mut [], false).map_err(|e| e.to_string())?;
        run::executer(Some(&app), &mk(778, &dir_c), &mut [], false).map_err(|e| e.to_string())?;
        let mut noms = noms_dumps(&dir_a);
        noms.extend(["therm.csv".to_string(), "measurements.csv".to_string()]);
        memes_fichiers(&dir_a, &dir_b, &noms)?;
        if memes_fichiers(&dir_a, &dir_c, &["therm.csv".to_string()]).is_ok() {
            return Err("graine + 1 donne le même therm.csv".into());
        }
        if memes_fichiers(&dir_a, &dir_c, &["cfg_000010.npy".to_string()]).is_ok() {
            return Err("graine + 1 donne le même dump".into());
        }
        Ok(format!("double run identique ({} fichiers), graine + 1 différente", noms.len()))
    });
}

// ---------------------------------------------------------------------------------------------
// LOT-1 : lot de 4 chaînes (β et graines différents) = 4 runs seuls, à l'octet
// ---------------------------------------------------------------------------------------------

fn lot1(app: &Appareil, precision: Precision) -> Result<String, String> {
    let chaines = [(2.3, 11, Depart::Chaud), (2.8, 22, Depart::Chaud), (3.5, 33, Depart::Froid), (8.0, 44, Depart::Chaud)];
    let mut cfg = lot(6, &chaines, 4, OptionsCompil::reference_k1());
    cfg.precision = precision;
    if precision == Precision::F32 {
        cfg.options = OptionsCompil::rapide();
    }
    let mut ensemble = construire(Some(app), cfg.clone()).map_err(|e| e.to_string())?;
    ensemble.init().map_err(|e| e.to_string())?;
    ensemble.allouer_historique(8).map_err(|e| e.to_string())?;
    for gel in [Gel::Aucun, Gel::BordX0Spatial] {
        ensemble.fixer_gel(gel).map_err(|e| e.to_string())?;
        for i in 0..2 {
            avancer(ensemble.as_mut(), 1, 4);
            ensemble.mesurer(if gel == Gel::Aucun { i } else { 2 + i }).map_err(|e| e.to_string())?;
        }
    }
    let h_lot = ensemble.historique().map_err(|e| e.to_string())?;
    let c_lot = ensemble.compteurs().map_err(|e| e.to_string())?;
    for (c, ch) in chaines.iter().enumerate() {
        let mut cs = cfg.clone();
        cs.chaines = vec![ParamChaine { beta: ch.0, graine: Graine(ch.1), depart: ch.2 }];
        let mut seul = construire(Some(app), cs).map_err(|e| e.to_string())?;
        seul.init().map_err(|e| e.to_string())?;
        seul.allouer_historique(8).map_err(|e| e.to_string())?;
        for gel in [Gel::Aucun, Gel::BordX0Spatial] {
            seul.fixer_gel(gel).map_err(|e| e.to_string())?;
            for i in 0..2 {
                avancer(seul.as_mut(), 1, 4);
                seul.mesurer(if gel == Gel::Aucun { i } else { 2 + i }).map_err(|e| e.to_string())?;
            }
        }
        let (a, b) = (ensemble.telecharger(c).unwrap(), seul.telecharger(0).unwrap());
        if octets_champ(&a) != octets_champ(&b) {
            return Err(format!("chaîne {c} : champ du lot ≠ champ de la chaîne seule"));
        }
        let hs = seul.historique().map_err(|e| e.to_string())?;
        let hl = &h_lot[c * 8 * 5..(c + 1) * 8 * 5];
        if hl.iter().map(|x| x.to_bits()).ne(hs.iter().map(|x| x.to_bits())) {
            return Err(format!("chaîne {c} : historique de mesures du lot ≠ chaîne seule"));
        }
        if seul.compteurs().unwrap()[0] != c_lot[c] {
            return Err(format!("chaîne {c} : compteurs du lot ≠ chaîne seule"));
        }
    }
    Ok(format!("lot de 4 chaînes (β, graines, départs différents, avec et sans gel) = 4 runs seuls à l'octet ({})", precision.nom()))
}

#[test]
#[ignore = "GPU"]
fn lot_1_f64() {
    let Some(app) = appareil_ou_skip("LOT-1-F64") else { return };
    gate("LOT-1-F64", || lot1(&app, Precision::F64));
}

#[test]
#[ignore = "GPU"]
fn lot_1_f32() {
    let Some(app) = appareil_ou_skip("LOT-1-F32") else { return };
    let mut cfg = lot(4, &[(2.3, 1, Depart::Chaud)], 4, OptionsCompil::rapide());
    cfg.precision = Precision::F32;
    if let Err(Erreur::NonImplemente(m)) = construire(Some(&app), cfg) {
        verdict(&format!("SKIP LOT-1-F32 : {m} (chemin F32 de T4 pas encore fusionné)"));
        return;
    }
    gate("LOT-1-F32", || lot1(&app, Precision::F32));
}

// ---------------------------------------------------------------------------------------------
// REPRISE-1 : 2N pas = N + reprise (NOUVEAU PROCESSUS) + N, à l'octet
// ---------------------------------------------------------------------------------------------

/// Reprend le run de KVU_ENFANT_DIR dans un nouveau processus (l'exécutable de test lui-même).
#[test]
#[ignore = "GPU"]
fn enfant_reprise() {
    let Ok(d) = std::env::var("KVU_ENFANT_DIR") else {
        println!("# enfant_reprise : KVU_ENFANT_DIR absent, rien à faire");
        return;
    };
    let app = gpu::ouvrir(backend_du_test(), 0).expect("GPU de l'enfant");
    let cfg = run::charger_config(Path::new(&d)).expect("config_run.json");
    run::executer(Some(&app), &cfg, &mut [], true).expect("reprise de l'enfant");
}

fn reprendre_dans_un_processus(dir: &Path) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let st = std::process::Command::new(exe)
        .args(["--ignored", "--exact", "enfant_reprise", "--nocapture", "--test-threads=1"])
        .env("KVU_ENFANT_DIR", dir)
        .status()
        .map_err(|e| e.to_string())?;
    if st.success() {
        Ok(())
    } else {
        Err(format!("le processus enfant a échoué : {st}"))
    }
}

fn reprise_1(app: &Appareil, bord: bool, precision: Precision) -> Result<String, String> {
    let nom = if bord { "bord" } else { "std" };
    let (da, db) = (rep(&format!("rep_{nom}_a")), rep(&format!("rep_{nom}_b")));
    let mut l = lot(6, &[(2.5, 91, Depart::Chaud), (2.8, 92, Depart::Chaud)], 4, OptionsCompil::reference_k1());
    l.precision = precision;
    if precision == Precision::F32 {
        l.options = OptionsCompil::rapide();
    }
    let mk = |d: &Path| {
        let mut c = cfg_run(l.clone(), 40, 40, 1, 10, false, d);
        if bord {
            c.dump_every = 0;
            c.bord = Some(ConfigBord { bc: "eq".into(), bc_graine: 555, jauge: Some(666), pre: 30, bloc: 10 });
        }
        c.reprise_every = 20;
        c
    };
    // A : 2N pas d'un coup ; B : arrêt à N, reprise dans un nouveau processus
    run::executer(Some(app), &mk(&da), &mut [], false).map_err(|e| e.to_string())?;
    let n_arret = if bord { 30 + 20 + 10 } else { 40 + 10 };
    let (_, fini) = run::executer_jusqua(Some(app), &mk(&db), &mut [], false, Some(n_arret)).map_err(|e| e.to_string())?;
    if fini {
        return Err("le run B devait s'arrêter avant la fin".into());
    }
    if db.join("COMPLETE").exists() {
        return Err("COMPLETE présent sur un run interrompu".into());
    }
    reprendre_dans_un_processus(&db)?;
    if !db.join("COMPLETE").exists() {
        return Err("COMPLETE absent après la reprise".into());
    }
    let mut total = 0;
    for c in 0..2 {
        let (a, b) = (da.join(format!("c{c:03}")), db.join(format!("c{c:03}")));
        let mut noms: Vec<String> = if bord {
            vec!["therm_blocs.csv".into(), "blocs.csv".into()]
        } else {
            let mut v = noms_dumps(&a);
            v.extend(["therm.csv".to_string(), "measurements.csv".to_string()]);
            v
        };
        noms.sort();
        memes_fichiers(&a, &b, &noms).map_err(|e| format!("chaîne {c} : {e}"))?;
        total += noms.len();
    }
    Ok(format!("2N pas = {n_arret} pas + reprise (nouveau processus) + reste : {total} fichiers identiques à l'octet ({})", precision.nom()))
}

#[test]
#[ignore = "GPU"]
fn reprise_1_f64() {
    let Some(app) = appareil_ou_skip("REPRISE-1-F64") else { return };
    gate("REPRISE-1-F64", || reprise_1(&app, false, Precision::F64));
}

#[test]
#[ignore = "GPU"]
fn reprise_1_bord() {
    let Some(app) = appareil_ou_skip("REPRISE-1-BORD") else { return };
    gate("REPRISE-1-BORD", || reprise_1(&app, true, Precision::F64));
}

#[test]
#[ignore = "GPU"]
fn reprise_1_f32() {
    let Some(app) = appareil_ou_skip("REPRISE-1-F32") else { return };
    let mut cfg = lot(4, &[(2.3, 1, Depart::Chaud)], 4, OptionsCompil::rapide());
    cfg.precision = Precision::F32;
    if let Err(Erreur::NonImplemente(m)) = construire(Some(&app), cfg) {
        verdict(&format!("SKIP REPRISE-1-F32 : {m} (chemin F32 de T4 pas encore fusionné)"));
        return;
    }
    // la reprise F32 exige un téléversement exact : sinon le défaut est celui de F32-REPRISE-IDENTITE
    let mut c2 = lot(6, &[(2.5, 61, Depart::Chaud), (3.5, 62, Depart::Chaud)], 4, OptionsCompil::rapide());
    c2.precision = Precision::F32;
    if let Err(e) = f32_identite(&app, &c2) {
        verdict(&format!("SKIP REPRISE-1-F32 : précondition F32-REPRISE-IDENTITE en échec ({e})"));
        return;
    }
    gate("REPRISE-1-F32", || reprise_1(&app, false, Precision::F32));
}

/// Diagnostic de la reprise F32 : `televerser(telecharger(x))` doit restituer le champ F32 AU BIT, et la
/// poursuite du calcul depuis l'état restauré doit être identique à la poursuite directe.
/// Précondition de la reprise F32 : `televerser(telecharger(x))` restitue le champ F32 AU BIT, et la poursuite
/// depuis l'état restauré est identique à la poursuite directe.
fn f32_identite(app: &Appareil, cfg: &ConfigLot) -> Result<String, String> {
    let mut a = construire(Some(app), cfg.clone()).map_err(|e| e.to_string())?;
    a.init().map_err(|e| e.to_string())?;
    avancer(a.as_mut(), 25, 4);
    let sauve: Vec<_> = (0..2).map(|c| a.telecharger(c).unwrap()).collect();
    // norme : combien de liens dépassent la tolérance de renormalisation de T4 (2^-22) ?
    let mut hors = 0usize;
    let mut pire = 0.0f64;
    for h in &sauve {
        for q in h.q.chunks_exact(4) {
            let e = (q.iter().map(|x| x * x).sum::<f64>() - 1.0).abs();
            pire = pire.max(e);
            hors += usize::from(e > 2.384_185_791_015_625e-7);
        }
    }
    let mut b = construire(Some(app), cfg.clone()).map_err(|e| e.to_string())?;
    b.init().map_err(|e| e.to_string())?;
    for (c, h) in sauve.iter().enumerate() {
        b.televerser(c, h).map_err(|e| e.to_string())?;
    }
    b.fixer_sweep_id(a.sweep_id());
    let mut diff = 0usize;
    for (c, h) in sauve.iter().enumerate() {
        let r = b.telecharger(c).map_err(|e| e.to_string())?;
        diff += h.q.iter().zip(&r.q).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    }
    if diff != 0 {
        return Err(format!("televerser(telecharger(x)) ≠ x : {diff} valeurs changées ; {hors} liens hors tolérance, pire |n²−1| = {pire:.3e}"));
    }
    avancer(a.as_mut(), 5, 4);
    avancer(b.as_mut(), 5, 4);
    for c in 0..2 {
        if octets_champ(&a.telecharger(c).unwrap()) != octets_champ(&b.telecharger(c).unwrap()) {
            return Err(format!("la poursuite depuis l'état restauré diverge (chaîne {c}) alors que le téléversement est exact"));
        }
    }
    Ok(format!("téléversement exact au bit, poursuite identique (pire |n²−1| = {pire:.2e}, {hors} liens hors tolérance)"))
}

/// Diagnostic de la reprise F32 (voir `f32_identite`).
#[test]
#[ignore = "GPU"]
fn f32_reprise_identite() {
    let Some(app) = appareil_ou_skip("F32-REPRISE-IDENTITE") else { return };
    let mut cfg = lot(6, &[(2.5, 61, Depart::Chaud), (3.5, 62, Depart::Chaud)], 4, OptionsCompil::rapide());
    cfg.precision = Precision::F32;
    if let Err(Erreur::NonImplemente(m)) = construire(Some(&app), cfg.clone()) {
        verdict(&format!("SKIP F32-REPRISE-IDENTITE : {m}"));
        return;
    }
    gate("F32-REPRISE-IDENTITE", || f32_identite(&app, &cfg));
}

/// F32 : un double run est identique à l'octet (le bruit d'ordonnancement GPU ne doit pas toucher le champ).
#[test]
#[ignore = "GPU"]
fn g_det_f32() {
    let Some(app) = appareil_ou_skip("G-DET-F32") else { return };
    let mut cfg = lot(6, &[(2.5, 71, Depart::Chaud)], 4, OptionsCompil::rapide());
    cfg.precision = Precision::F32;
    if let Err(Erreur::NonImplemente(m)) = construire(Some(&app), cfg.clone()) {
        verdict(&format!("SKIP G-DET-F32 : {m}"));
        return;
    }
    gate("G-DET-F32", || {
        let faire = |c: &ConfigLot| -> Result<Vec<u8>, String> {
            let mut n = construire(Some(&app), c.clone()).map_err(|e| e.to_string())?;
            n.init().map_err(|e| e.to_string())?;
            avancer(n.as_mut(), 30, 4);
            Ok(octets_champ(&n.telecharger(0).map_err(|e| e.to_string())?))
        };
        let (x, y) = (faire(&cfg)?, faire(&cfg)?);
        if x != y {
            return Err("deux runs F32 identiques donnent des champs différents".into());
        }
        let mut c2 = cfg.clone();
        c2.chaines[0].graine = Graine(72);
        if faire(&c2)? == x {
            return Err("graine + 1 donne le même champ F32".into());
        }
        Ok("double run F32 identique à l'octet, graine + 1 différente".into())
    });
}

// ---------------------------------------------------------------------------------------------
// BORD-C1 / BORD-C2a : gel intact, jauge exacte, noyaux device contre version hôte
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn bord_c1_c2a() {
    let Some(app) = appareil_ou_skip("BORD-C1") else { return };
    gate("BORD-C1", || {
        let d = rep("bord_c1");
        let mut c = cfg_run(lot(8, &[(2.5, 28092800002, Depart::Chaud)], 4, OptionsCompil::reference_k1()), 100, 200, 1, 0, false, &d);
        c.bord = Some(ConfigBord { bc: "hot".into(), bc_graine: 28092800003, jauge: None, pre: 100, bloc: 50 });
        run::executer(Some(&app), &c, &mut [], false).map_err(|e| e.to_string())?;
        let man: serde_json::Value = serde_json::from_str(&fs::read_to_string(d.join("manifest.json")).unwrap()).unwrap();
        let b = &man["bord"][0];
        if b["gel_intact"] != true || man["gel_intact"] != true {
            return Err(format!("gel non intact : {b}"));
        }
        if b["fnv_bord_gel_debut"] != b["fnv_bord_gel_fin"] {
            return Err(format!("FNV début ≠ fin : {b}"));
        }
        Ok(format!("empreinte FNV des liens gelés identique début/fin ({})", b["fnv_bord_gel_fin"]))
    });
    gate("BORD-C2a", || {
        let d = rep("bord_c2a");
        let mut c = cfg_run(lot(8, &[(2.5, 28092800002, Depart::Chaud)], 4, OptionsCompil::reference_k1()), 50, 100, 1, 0, false, &d);
        c.bord = Some(ConfigBord { bc: "eq".into(), bc_graine: 28092800004, jauge: Some(28092800005), pre: 100, bloc: 50 });
        run::executer(Some(&app), &c, &mut [], false).map_err(|e| e.to_string())?;
        let man: serde_json::Value = serde_json::from_str(&fs::read_to_string(d.join("manifest.json")).unwrap()).unwrap();
        let e = man["bord"][0]["jauge_ecart_max"].as_f64().ok_or("jauge_ecart_max absent")?;
        if !(0.0..=1e-14).contains(&e) {
            return Err(format!("écart de jauge des tranches {e:.3e} hors de [0, 1e-14]"));
        }
        if man["bord"][0]["fnv_bord_avant_jauge"] == man["bord"][0]["fnv_bord_gel_debut"] {
            return Err("la jauge n'a pas changé le bord".into());
        }
        Ok(format!("écart des S_t, T_t avant/après jauge = {e:.2e} ≤ 1e-14"))
    });
    // les noyaux device de pose du bord et de jauge donnent le même résultat que la version hôte (à l'ULP des
    // fonctions transcendantes près : les liens du bord chaud viennent de log/cos/sin)
    gate("BORD-DEVICE-HOTE", || {
        let cfg = lot(6, &[(2.5, 5, Depart::Froid)], 4, OptionsCompil::reference_k1());
        let mut g = Ref64Gpu::nouveau(&app, cfg.clone()).map_err(|e| e.to_string())?;
        g.init().unwrap();
        let mut c = Cpu64::nouveau(cfg).unwrap();
        c.init().unwrap();
        let (fg, eg) = bord::poser(&mut g, 0, &Bord::Chaud(Graine(31)), Some(Graine(32))).map_err(|e| e.to_string())?;
        let (fc, ec) = bord::poser(&mut c, 0, &Bord::Chaud(Graine(31)), Some(Graine(32))).map_err(|e| e.to_string())?;
        let e = max_ecart(&g.telecharger(0).unwrap().q, &c.telecharger(0).unwrap().q);
        if e > 1e-14 || eg > 1e-14 || ec > 1e-14 {
            return Err(format!("device contre hôte : max|Δq| = {e:.3e}, écarts de jauge {eg:.2e} / {ec:.2e} (FNV {fg:x} / {fc:x})"));
        }
        Ok(format!("bord chaud + jauge : device = hôte à {e:.2e} (FNV device {fg:016x}, hôte {fc:016x})"))
    });
}

// ---------------------------------------------------------------------------------------------
// ChampGpu : disposition (c·4V + li)·4 + k, aller-retour F64 et F32, isolation des chaînes d'un lot
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn champ_gpu_aller_retour() {
    use kvu::commun::ChampHote;
    use kvu::su2::ChampGpu;
    let Some(app) = appareil_ou_skip("CHAMP-GPU") else { return };
    gate("CHAMP-GPU", || {
        let l = 4;
        // trois champs hôtes distincts (Haar de trois graines) lus par la référence CPU
        let champ = |g: u64| -> ChampHote {
            let mut c = Cpu64::nouveau(lot(l, &[(2.3, g, Depart::Chaud)], 0, OptionsCompil::reference_k1())).unwrap();
            c.init().unwrap();
            c.telecharger(0).unwrap()
        };
        let (a, b, c) = (champ(1), champ(2), champ(3));
        let mut g = ChampGpu::allouer(&app, l, 3, Precision::F64).map_err(|e| e.to_string())?;
        if (g.l, g.nchaines, g.v()) != (4, 3, 256) || g.precision() != Precision::F64 {
            return Err("métadonnées du champ F64".into());
        }
        // les chaînes sont écrites dans le désordre ; chacune revient à l'octet, sans toucher aux voisines
        g.televerser(&app, 1, &b).map_err(|e| e.to_string())?;
        g.televerser(&app, 2, &c).map_err(|e| e.to_string())?;
        let zero = g.telecharger(&app, 0).map_err(|e| e.to_string())?;
        if zero.q.iter().any(|&x| x != 0.0) {
            return Err("la chaîne 0 (jamais écrite) n'est pas à zéro".into());
        }
        g.televerser(&app, 0, &a).map_err(|e| e.to_string())?;
        for (i, h) in [&a, &b, &c].into_iter().enumerate() {
            if octets_champ(&g.telecharger(&app, i).map_err(|e| e.to_string())?) != octets_champ(h) {
                return Err(format!("aller-retour F64 de la chaîne {i}"));
            }
        }
        // erreurs : chaîne hors du lot, L différent
        if g.televerser(&app, 3, &a).is_ok() || g.telecharger(&app, 3).is_ok() {
            return Err("une chaîne hors du lot doit être refusée".into());
        }
        if g.televerser(&app, 0, &ChampHote::froid(6).unwrap()).is_ok() {
            return Err("un champ de L différent doit être refusé".into());
        }
        // F32 : un champ f32-exact revient au bit ; un champ f64 quelconque est renormalisé puis arrondi
        let mut f = ChampGpu::allouer(&app, l, 2, Precision::F32).map_err(|e| e.to_string())?;
        let exact = ChampHote { l, q: a.q.iter().map(|&x| (x as f32) as f64).collect() };
        f.televerser(&app, 1, &exact).map_err(|e| e.to_string())?;
        if octets_champ(&f.telecharger(&app, 1).map_err(|e| e.to_string())?) != octets_champ(&exact) {
            return Err("aller-retour F32 d'un champ f32-exact (la reprise F32 en dépend)".into());
        }
        f.televerser(&app, 0, &a).map_err(|e| e.to_string())?;
        let r = f.telecharger(&app, 0).map_err(|e| e.to_string())?;
        if r.defaut_unitarite() > 1e-6 || max_ecart(&r.q, &a.q) > 1e-6 {
            return Err(format!("champ F64 quelconque téléversé en F32 : écart {:.2e}", max_ecart(&r.q, &a.q)));
        }
        Ok("F64 : 3 chaînes indépendantes aller-retour à l'octet ; F32 : champ f32-exact au bit, champ f64 arrondi à < 1e-6".into())
    });
}

/// Bord en F32 : pose par la version hôte (formules identiques), gel et empreinte contrôlés.
#[test]
#[ignore = "GPU"]
fn bord_f32() {
    let Some(app) = appareil_ou_skip("BORD-F32") else { return };
    let mut l = lot(6, &[(2.5, 55, Depart::Chaud)], 4, OptionsCompil::rapide());
    l.precision = Precision::F32;
    if let Err(Erreur::NonImplemente(m)) = construire(Some(&app), l.clone()) {
        verdict(&format!("SKIP BORD-F32 : {m}"));
        return;
    }
    gate("BORD-F32", || {
        let d = rep("bord_f32");
        let mut c = cfg_run(l, 20, 40, 1, 0, false, &d);
        c.bord = Some(ConfigBord { bc: "eq".into(), bc_graine: 56, jauge: Some(57), pre: 30, bloc: 20 });
        run::executer(Some(&app), &c, &mut [], false).map_err(|e| e.to_string())?;
        let man: serde_json::Value = serde_json::from_str(&fs::read_to_string(d.join("manifest.json")).unwrap()).unwrap();
        if man["gel_intact"] != true {
            return Err(format!("gel F32 non intact : {}", man["bord"]));
        }
        let e = man["bord"][0]["jauge_ecart_max"].as_f64().ok_or("jauge_ecart_max absent")?;
        // en F32 la jauge ne conserve les tranches qu'à la précision du float
        if !(0.0..=1e-5).contains(&e) {
            return Err(format!("écart de jauge F32 {e:.3e}"));
        }
        Ok(format!("bord F32 (pose hôte) : gel intact, jauge des tranches à {e:.1e} (précision float)"))
    });
}

// ---------------------------------------------------------------------------------------------
// G-H04 : tirage exact de x0 (moments), 7 α, 2·10⁷ tirages, avec contrôle négatif
// ---------------------------------------------------------------------------------------------

/// Moments exacts E[x^p], p = 1..4, de la densité sqrt(1-x²) exp(αx) (Gauss-Tchebychev 2ᵉ espèce, n = 4000).
fn moments_exacts(alpha: f64) -> [f64; 4] {
    let n = 4000;
    let (mut z, mut m) = (0.0, [0.0; 4]);
    for k in 1..=n {
        let th = k as f64 * std::f64::consts::PI / (n as f64 + 1.0);
        let x = th.cos();
        let w = std::f64::consts::PI / (n as f64 + 1.0) * th.sin().powi(2);
        let f = w * (alpha * (x - 1.0)).exp();
        z += f;
        for (p, mp) in m.iter_mut().enumerate() {
            *mp += f * x.powi(p as i32 + 1);
        }
    }
    [m[0] / z, m[1] / z, m[2] / z, m[3] / z]
}

fn z_scores(tirages: &[f64], alpha: f64) -> (f64, f64) {
    let n = tirages.len() as f64;
    let (mut m1, mut m2) = (0.0, 0.0);
    for &x in tirages {
        m1 += x;
        m2 += x * x;
    }
    let (m1, m2) = (m1 / n, m2 / n);
    let e = moments_exacts(alpha);
    let sd1 = (e[1] - e[0] * e[0]).sqrt();
    let sd2 = (e[3] - e[1] * e[1]).sqrt();
    ((m1 - e[0]) / (sd1 / n.sqrt()), (m2 - e[1]) / (sd2 / n.sqrt()))
}

#[test]
#[ignore = "GPU"]
fn g_h04() {
    let nom = etiquette("G-H04");
    let Some(app) = appareil_ou_skip(&nom) else { return };
    gate(&nom, || {
        const N: usize = 20_000_000;
        let alphas = [0.3, 1.0, 1.999, 2.0, 5.0, 20.0, 100.0];
        let cfg = lot(2, &[(1.0, 1, Depart::Froid)], 0, OptionsCompil::reference_k1());
        let g = Ref64Gpu::nouveau(&app, cfg).map_err(|e| e.to_string())?;
        let mut lignes = Vec::new();
        for (i, &a) in alphas.iter().enumerate() {
            let (t, plafond) = g.test_x0(a, N, Graine(9000 + i as u64)).map_err(|e| e.to_string())?;
            let (z1, z2) = z_scores(&t, a);
            lignes.push(format!("α={a}: z1={z1:+.2} z2={z2:+.2}"));
            if z1.abs() >= 4.0 || z2.abs() >= 4.0 || plafond != 0 {
                return Err(format!("α = {a} : |z| ≥ 4 (z1 = {z1:.2}, z2 = {z2:.2}) ou plafond = {plafond}"));
            }
        }
        // contrôle négatif : le mutant Metropolis naïf doit être ROUGE à α ≥ 5
        let mut cm = lot(2, &[(1.0, 1, Depart::Froid)], 0, OptionsCompil::reference_k1());
        cm.mutant_metro = true;
        let gm = Ref64Gpu::nouveau(&app, cm).map_err(|e| e.to_string())?;
        for a in [5.0, 20.0] {
            let (t, _) = gm.test_x0(a, 4_000_000, Graine(777)).map_err(|e| e.to_string())?;
            let (z1, z2) = z_scores(&t, a);
            if z1.abs() < 4.0 && z2.abs() < 4.0 {
                return Err(format!("le MUTANT n'est pas rouge à α = {a} (z1 = {z1:.2}, z2 = {z2:.2}) : le test n'a pas de puissance"));
            }
        }
        Ok(format!("7 α, N = 2·10⁷, |z| < 4, plafond 0 ; mutant ROUGE à α = 5 et 20 [{}]", lignes.join(" ; ")))
    });
}

// ---------------------------------------------------------------------------------------------
// Gates de bit-identité (références externes lues EN PLACE)
// ---------------------------------------------------------------------------------------------

fn racine_kvu() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Lance `cfg` dans un répertoire neuf et rend ce répertoire.
fn lancer(app: &Appareil, nom: &str, mut cfg: ConfigRun) -> Result<PathBuf, String> {
    let d = rep(nom);
    cfg.outdir = d.clone();
    run::executer(Some(app), &cfg, &mut [], false).map_err(|e| format!("{nom} : {e}"))?;
    Ok(d)
}

fn lire_champ_npy(p: &Path) -> Result<Vec<f64>, String> {
    su2_hb::read_npy_quat(p).map(|x| x.1)
}

/// G-BIT-K1 (acemagic) : contre `su2_gen --hip` historique, bâti depuis main et lancé par gates_su2.sh.
#[test]
#[ignore = "GPU"]
fn g_bit_k1() {
    let Some(app) = appareil_ou_skip("G-BIT-K1") else { return };
    let Some(refs) = std::env::var("KVU_REF_K1").ok().filter(|v| !v.is_empty()) else {
        verdict("SKIP G-BIT-K1 : KVU_REF_K1 absent (les références su2_gen sont produites par gates_su2.sh)");
        return;
    };
    gate("G-BIT-K1", || {
        let mut fichiers = 0;
        for (l, graine) in [(12usize, 926193012u64), (16, 926193016)] {
            let cfg = cfg_run(
                lot(l, &[(2.3, graine, Depart::Chaud)], 4, OptionsCompil::reference_k1()),
                300,
                1000,
                1,
                250,
                false,
                Path::new("/inutile"),
            );
            let d = lancer(&app, &format!("bitk1_L{l}"), cfg)?;
            let r = Path::new(&refs).join(format!("L{l}"));
            let mut noms = noms_dumps(&r);
            noms.extend(["therm.csv".to_string(), "measurements.csv".to_string()]);
            if noms.len() != 6 {
                return Err(format!("L{l} : la référence devrait avoir 4 NPY + 2 CSV, trouvé {noms:?}"));
            }
            memes_fichiers(&r, &d, &noms).map_err(|e| format!("L{l} : {e}"))?;
            fichiers += noms.len();
        }
        Ok(format!("{fichiers} fichiers identiques à l'octet à su2_gen --hip (4 NPY + 2 CSV, L = 12 et 16)"))
    });
}

/// Arguments de commandes_amd.txt (p2 L6/L8) : 20 pas, dump à chaque pas, dump initial.
const P2: [(usize, u64); 2] = [(6, 27092900006), (8, 27092900008)];

fn p2_run(app: &Appareil, nom: &str, l: usize, graine: u64, options: OptionsCompil) -> Result<PathBuf, String> {
    let cfg = cfg_run(lot(l, &[(2.8, graine, Depart::Chaud)], 4, options), 0, 20, 1, 1, true, Path::new("/inutile"));
    lancer(app, nom, cfg)
}

/// Compare un run p2 à la référence ; applique le REPLI GELÉ §4.4 si l'identité à l'octet échoue.
fn bit_p2(app: &Appareil, etiq: &str, ref_p2: &Path) -> Result<String, String> {
    let mut resume = Vec::new();
    for (l, g) in P2 {
        let r = ref_p2.join(format!("L{l}"));
        if !r.is_dir() {
            return Err(format!("référence absente : {}", r.display()));
        }
        let opts = OptionsCompil::compat_production(app.backend());
        let d = p2_run(app, &format!("{etiq}_L{l}"), l, g, opts.clone())?;
        let mut noms = noms_dumps(&r);
        noms.extend(["therm.csv".to_string(), "measurements.csv".to_string()]);
        match memes_fichiers(&r, &d, &noms) {
            Ok(()) => resume.push(format!("L{l} : {} fichiers identiques à l'octet", noms.len())),
            Err(echec) => {
                // REPLI GELÉ §4.4 : départ identique à l'octet ; après un pas, max|Δq| ≤ 1e-12 et 0 décision divergente
                let (a0, b0) = (fs::read(r.join("cfg_000000.npy")).unwrap(), fs::read(d.join("cfg_000000.npy")).unwrap());
                if a0 != b0 {
                    return Err(format!("L{l} : {echec} ; REPLI : cfg_000000 NON identique"));
                }
                let e1 = max_ecart(&lire_champ_npy(&r.join("cfg_000001.npy"))?, &lire_champ_npy(&d.join("cfg_000001.npy"))?);
                if e1 > 1e-12 {
                    return Err(format!("L{l} : {echec} ; REPLI : max|Δq| au pas 1 = {e1:.3e} > 1e-12"));
                }
                // 0 décision divergente : essais par lien du bain thermique, fpoff contre compat-prod, même configuration
                let div = decisions_divergentes(app, l, g)?;
                if div != 0 {
                    return Err(format!("L{l} : REPLI : {div} décisions divergentes au pas 1"));
                }
                resume.push(format!(
                    "L{l} : REPLI GELÉ §4.4 (premier désaccord : {echec}) : cfg_000000 identique, max|Δq| pas 1 = {e1:.2e} ≤ 1e-12, 0 décision divergente"
                ));
            }
        }
    }
    Ok(resume.join(" ; "))
}

/// Nombre de liens dont le nombre d'essais du tirage diffère entre `reference_k1` et `compat_production`
/// pour un sweep HB depuis la même configuration chaude.
fn decisions_divergentes(app: &Appareil, l: usize, graine: u64) -> Result<usize, String> {
    let mut diag = Vec::new();
    let mut depart = None;
    for opts in [OptionsCompil::reference_k1(), OptionsCompil::compat_production(app.backend())] {
        let mut g = Ref64Gpu::nouveau(app, lot(l, &[(2.8, graine, Depart::Chaud)], 4, opts)).map_err(|e| e.to_string())?;
        g.init().map_err(|e| e.to_string())?;
        // même configuration de départ pour les deux : celle du premier
        match &depart {
            None => depart = Some(g.telecharger(0).map_err(|e| e.to_string())?),
            Some(h) => g.televerser(0, h).map_err(|e| e.to_string())?,
        }
        diag.push(g.sweep_hb_diag().map_err(|e| e.to_string())?);
    }
    Ok(diag[0].iter().zip(&diag[1]).filter(|(a, b)| a != b).count())
}

#[test]
#[ignore = "GPU"]
fn g_bit_hip_prod() {
    let Some(app) = appareil_ou_skip("G-BIT-HIP-PROD") else { return };
    if app.backend() != BackendKind::Hip {
        verdict(&format!("SKIP G-BIT-HIP-PROD : backend {} (gate propre à HIP)", app.backend().nom()));
        return;
    }
    let p = std::env::var("KVU_P2_AMD")
        .unwrap_or_else(|_| "/chemin/vers/projet/chantiers/vortex_tests_20260927/gpu_port/ref_amd/p2".into());
    gate("G-BIT-HIP-PROD", || bit_p2(&app, "bithipprod", Path::new(&p)));
}

#[test]
#[ignore = "GPU"]
fn g_bit_cuda() {
    let Some(app) = appareil_ou_skip("G-BIT-CUDA") else { return };
    if app.backend() != BackendKind::Cuda {
        verdict(&format!("SKIP G-BIT-CUDA : backend {} (gate propre à CUDA, gamer)", app.backend().nom()));
        return;
    }
    let defaut = if Path::new("/chemin/vers/projet/chantiers/vortex_tests_20260927/gpu_port/cuda/p2").is_dir() {
        "/chemin/vers/projet/chantiers/vortex_tests_20260927/gpu_port/cuda/p2".to_string()
    } else {
        format!("{}/ym4_calcul/su2_cuda/p2", std::env::var("HOME").unwrap_or_default())
    };
    let p = std::env::var("KVU_P2_CUDA").unwrap_or(defaut);
    gate("G-BIT-CUDA", || bit_p2(&app, "bitcuda", Path::new(&p)));
    // C0 : L8 β2,5 graine 28092800001 therm 50 nmeas 200 contre reference/fixtures_bord/c0_ref
    gate("G-BIT-CUDA-C0", || {
        let cfg = cfg_run(
            lot(8, &[(2.5, 28092800001, Depart::Chaud)], 4, OptionsCompil::compat_production(BackendKind::Cuda)),
            50,
            200,
            1,
            0,
            false,
            Path::new("/inutile"),
        );
        let d = lancer(&app, "bitcuda_c0", cfg)?;
        let r = racine_kvu().join("reference/fixtures_bord/c0_ref");
        memes_fichiers(&r, &d, &["therm.csv".to_string(), "measurements.csv".to_string()])?;
        Ok("C0 : therm.csv et measurements.csv identiques à fixtures_bord/c0_ref".into())
    });
}

/// G-BIT-BORD (gamer) : arguments de tests_code.sh ; cmp de blocs.csv et therm_blocs.csv UNIQUEMENT.
#[test]
#[ignore = "GPU"]
fn g_bit_bord() {
    let Some(app) = appareil_ou_skip("G-BIT-BORD") else { return };
    if app.backend() != BackendKind::Cuda {
        verdict(&format!("SKIP G-BIT-BORD : backend {} (références produites par CUDA sur le gamer)", app.backend().nom()));
        return;
    }
    gate("G-BIT-BORD", || {
        // (nom de la fixture, bc, graine du bord, jauge)
        let cas: [(&str, &str, u64, Option<u64>); 4] = [
            ("t_cold", "cold", 28092800003, None),
            ("t_hot", "hot", 28092800003, None),
            ("t_eq", "eq", 28092800004, None),
            ("t_eqg", "eq", 28092800004, Some(28092800005)),
        ];
        let mut n = 0;
        for (nom, bc, bcs, jauge) in cas {
            let mut cfg = cfg_run(
                lot(8, &[(2.5, 28092800002, Depart::Chaud)], 4, OptionsCompil::compat_production(BackendKind::Cuda)),
                200,
                1000,
                1,
                0,
                false,
                Path::new("/inutile"),
            );
            cfg.bord = Some(ConfigBord { bc: bc.into(), bc_graine: bcs, jauge, pre: 200, bloc: 100 });
            let d = lancer(&app, &format!("bitbord_{nom}"), cfg)?;
            let r = racine_kvu().join("reference/fixtures_bord").join(nom);
            memes_fichiers(&r, &d, &["blocs.csv".to_string(), "therm_blocs.csv".to_string()]).map_err(|e| format!("{nom} : {e}"))?;
            n += 2;
        }
        Ok(format!("{n} fichiers identiques à fixtures_bord (t_cold, t_hot, t_eq, t_eqg : blocs.csv et therm_blocs.csv)"))
    });
}

// ---------------------------------------------------------------------------------------------
// Banc informatif (non gaté) : liens/s du bain thermique, un lot contre une chaîne
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn banc_info() {
    let Some(app) = appareil_ou_skip("BANC-SU2") else { return };
    for (l, nch) in [(16usize, 1usize), (16, 16), (32, 1), (8, 64)] {
        let chaines: Vec<(f64, u64, Depart)> = (0..nch).map(|i| (2.5, 100 + i as u64, Depart::Chaud)).collect();
        let mut g = Ref64Gpu::nouveau(&app, lot(l, &chaines, 4, OptionsCompil::compat_production(app.backend()))).unwrap();
        g.init().unwrap();
        avancer(&mut g, 3, 4);
        g.synchroniser().unwrap();
        let t = std::time::Instant::now();
        let pas = if l >= 32 { 10 } else { 40 };
        for _ in 0..pas {
            g.sweep_hb().unwrap();
        }
        g.synchroniser().unwrap();
        let s = t.elapsed().as_secs_f64();
        let liens = 4.0 * (l * l * l * l) as f64 * nch as f64 * pas as f64;
        println!("BANC-SU2 L={l} chaines={nch} hb_sweeps={pas} liens_par_s_hb={:.4e} s_par_sweep={:.6}", liens / s, s / pas as f64);
    }
}
