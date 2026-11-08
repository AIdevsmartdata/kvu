//! Tests CPU du générateur SU(2) unifié (T3) : boucle de production, lots, bord R2-3, reprise, manifeste.
//! Aucun GPU requis. `cargo test -p kvu --test su2_cpu`.
//!
//! Références internes : `heritage::su2_hb::CpuSu2` (K1) pour la boucle hôte de su2_hip, et les fichiers
//! `tests/donnees/su2/golden_*` (régénérables avec KVU_REGENERER_GOLDEN=1, à relire avant de les valider).

use kvu::commun::{Depart, Graine, Precision};
use kvu::gpu::OptionsCompil;
use kvu::heritage::manifest::sha256_hex;
use kvu::heritage::su2_hb::{self, g17, CpuSu2};
use kvu::su2::bord::{self, Bord};
use kvu::su2::cpu::{tranches_hote, Cpu64};
use kvu::su2::run::{self, ConfigBord, ConfigRun, MesureEnLigne};
use kvu::su2::{ConfigLot, Gel, NoyauxSu2, ParamChaine};
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------------------------
// Utilitaires
// ---------------------------------------------------------------------------------------------

fn rep(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("su2_cpu").join(format!("{nom}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.parent().unwrap()).unwrap();
    d
}

fn lot(l: usize, chaines: &[(f64, u64, Depart)], nor: usize) -> ConfigLot {
    ConfigLot {
        l,
        chaines: chaines.iter().map(|&(beta, g, depart)| ParamChaine { beta, graine: Graine(g), depart }).collect(),
        nor,
        precision: Precision::F64,
        options: OptionsCompil::reference_k1(),
        mutant_metro: false,
    }
}

fn cfg(lot: ConfigLot, therm: u64, nmeas: u64, skip: u64, dump_every: u64, dump_initial: bool, outdir: &Path) -> ConfigRun {
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

fn lire(p: &Path) -> String {
    fs::read_to_string(p).unwrap_or_else(|e| panic!("{} : {e}", p.display()))
}

fn octets(p: &Path) -> Vec<u8> {
    fs::read(p).unwrap_or_else(|e| panic!("{} : {e}", p.display()))
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

fn executer_cpu(c: &ConfigRun) -> kvu::commun::Resultat<run::RapportRun> {
    run::executer(None, c, &mut [], false)
}

/// Lignes de therm.csv et measurements.csv attendues, recalculées avec `CpuSu2` (boucle hôte de su2_hip).
#[allow(clippy::too_many_arguments)]
fn attendu_k1(l: usize, beta: f64, graine: u64, hot: bool, therm: usize, nmeas: usize, skip: usize, nor: usize) -> (String, String) {
    let mut c = CpuSu2::new(l, graine).unwrap();
    c.init(hot);
    let mut ft = String::from("step,plaquette,P0,P1,P2,P3\n");
    let mut fm = String::from("measurement_index,total_steps,phase,plaquette,P0,P1,P2,P3\n");
    let ligne = |m: [f64; 5]| m.iter().map(|x| g17(*x)).collect::<Vec<_>>().join(",");
    ft.push_str(&format!("0,{}\n", ligne(c.measure())));
    for t in 1..=therm {
        c.step(beta, nor);
        ft.push_str(&format!("{t},{}\n", ligne(c.measure())));
    }
    fm.push_str(&format!("0,{therm},initial,{}\n", ligne(c.measure())));
    for m in 1..=nmeas {
        for _ in 0..skip {
            c.step(beta, nor);
        }
        fm.push_str(&format!("{m},{},sample,{}\n", therm + m * skip, ligne(c.measure())));
    }
    (ft, fm)
}

// ---------------------------------------------------------------------------------------------
// La boucle hôte de su2_hip
// ---------------------------------------------------------------------------------------------

#[test]
fn boucle_standard_egale_la_boucle_hote_de_su2_hip() {
    for (hot, graine) in [(true, 926193012u64), (false, 5)] {
        let d = rep("std");
        let c = cfg(lot(4, &[(2.3, graine, if hot { Depart::Chaud } else { Depart::Froid })], 4), 25, 12, 3, 4, true, &d);
        let r = executer_cpu(&c).unwrap();
        assert_eq!((r.pas, r.mesures, r.dumps, r.plafond), (25 + 36, 12, 3, 0));
        let (ft, fm) = attendu_k1(4, 2.3, graine, hot, 25, 12, 3, 4);
        assert_eq!(lire(&d.join("therm.csv")), ft);
        assert_eq!(lire(&d.join("measurements.csv")), fm);
        // dumps : initial + 4, 8, 12 ; pas de dump supplémentaire à la fin, comme su2_hip
        assert_eq!(noms_dumps(&d), ["cfg_000000.npy", "cfg_000004.npy", "cfg_000008.npy", "cfg_000012.npy"]);
        // le dernier dump est le champ final de la chaîne de référence
        let mut cr = CpuSu2::new(4, graine).unwrap();
        cr.init(hot);
        for _ in 0..25 + 36 {
            cr.step(2.3, 4);
        }
        let (l, q) = su2_hb::read_npy_quat(&d.join("cfg_000012.npy")).unwrap();
        assert_eq!(l, 4);
        assert_eq!(q, cr.flat());
        // et le fichier est octet pour octet celui de write_npy_quat (en-tête compris)
        let attendu = d.join("ref.npy");
        su2_hb::write_npy_quat(&attendu, 4, &cr.flat()).unwrap();
        assert_eq!(octets(&attendu), octets(&d.join("cfg_000012.npy")));
        fs::remove_dir_all(&d).unwrap();
    }
}

#[test]
fn therm_zero_et_une_mesure() {
    // cas du p2 de gpu_port : therm 0, dump à chaque pas, dump initial
    let d = rep("p2");
    let c = cfg(lot(4, &[(2.8, 27092900006, Depart::Chaud)], 4), 0, 5, 1, 1, true, &d);
    executer_cpu(&c).unwrap();
    assert_eq!(noms_dumps(&d).len(), 6);
    let t = lire(&d.join("therm.csv"));
    assert_eq!(t.lines().count(), 2, "en-tête + indice 0");
    let m = lire(&d.join("measurements.csv"));
    assert_eq!(m.lines().count(), 7);
    assert!(m.lines().nth(1).unwrap().starts_with("0,0,initial,"));
    assert!(m.lines().nth(2).unwrap().starts_with("1,1,sample,"));
    fs::remove_dir_all(&d).unwrap();
}

/// Golden des CSV : comparaison de la STRUCTURE exacte et des valeurs à 1e-9 (libm de la machine).
fn comparer_golden(nom: &str, obtenu: &str) {
    let chemin = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/su2").join(nom);
    if std::env::var("KVU_REGENERER_GOLDEN").is_ok() {
        fs::create_dir_all(chemin.parent().unwrap()).unwrap();
        fs::write(&chemin, obtenu).unwrap();
        return;
    }
    let attendu = lire(&chemin);
    let (la, lo): (Vec<&str>, Vec<&str>) = (attendu.lines().collect(), obtenu.lines().collect());
    assert_eq!(la.len(), lo.len(), "{nom} : nombre de lignes");
    assert_eq!(la[0], lo[0], "{nom} : en-tête");
    for (i, (a, o)) in la.iter().zip(&lo).enumerate().skip(1) {
        let (ca, co): (Vec<&str>, Vec<&str>) = (a.split(',').collect(), o.split(',').collect());
        assert_eq!(ca.len(), co.len(), "{nom} ligne {i}");
        for (x, y) in ca.iter().zip(&co) {
            match (x.parse::<f64>(), y.parse::<f64>()) {
                (Ok(u), Ok(v)) => {
                    assert!((u - v).abs() <= 1e-9 * u.abs().max(1.0), "{nom} ligne {i} : {u} contre {v}");
                    // forme %.17g : jamais d'exposant sans signe, 17 chiffres significatifs au plus
                    assert!(y.len() <= 25, "{nom} ligne {i} : « {y} » trop long pour %.17g");
                }
                _ => assert_eq!(x, y, "{nom} ligne {i}"),
            }
        }
    }
}

#[test]
fn golden_des_csv_et_des_dumps() {
    let d = rep("golden");
    let c = cfg(lot(4, &[(2.3, 926193012, Depart::Chaud)], 4), 12, 8, 2, 4, true, &d);
    executer_cpu(&c).unwrap();
    comparer_golden("golden_L4_therm.csv", &lire(&d.join("therm.csv")));
    comparer_golden("golden_L4_measurements.csv", &lire(&d.join("measurements.csv")));
    // les dumps : mêmes nombres et même taille que su2_hip (c16 : 256·L⁴ octets + en-tête)
    for n in noms_dumps(&d) {
        let taille = fs::metadata(d.join(&n)).unwrap().len();
        assert_eq!(taille, (256 * 4u64.pow(4)) + su2_hb::npy_header(4).len() as u64, "{n}");
    }
    fs::remove_dir_all(&d).unwrap();
}

// ---------------------------------------------------------------------------------------------
// Lots
// ---------------------------------------------------------------------------------------------

#[test]
fn lot_de_trois_chaines_dans_des_sous_repertoires() {
    let chaines = [(2.3, 11u64, Depart::Chaud), (2.8, 22, Depart::Froid), (3.5, 33, Depart::Chaud)];
    let dl = rep("lot");
    let c = cfg(lot(4, &chaines, 2), 10, 6, 2, 3, false, &dl);
    executer_cpu(&c).unwrap();
    assert!(dl.join("COMPLETE").exists());
    for (i, ch) in chaines.iter().enumerate() {
        let ds = rep(&format!("seul{i}"));
        let cs = cfg(lot(4, &[*ch], 2), 10, 6, 2, 3, false, &ds);
        executer_cpu(&cs).unwrap();
        let sous = dl.join(format!("c{i:03}"));
        let mut noms = noms_dumps(&ds);
        noms.extend(["therm.csv".to_string(), "measurements.csv".to_string()]);
        assert_eq!(noms.len(), 4);
        for n in noms {
            assert_eq!(octets(&ds.join(&n)), octets(&sous.join(&n)), "chaîne {i} : {n}");
        }
        fs::remove_dir_all(&ds).unwrap();
    }
    fs::remove_dir_all(&dl).unwrap();
}

// ---------------------------------------------------------------------------------------------
// Manifeste, COMPLETE, refus d'écraser, échec sans COMPLETE
// ---------------------------------------------------------------------------------------------

#[test]
fn manifeste_complete_en_dernier_et_sorties_hachees() {
    let d = rep("manif");
    let c = cfg(lot(4, &[(2.3, 1, Depart::Chaud)], 4), 5, 4, 1, 2, false, &d);
    executer_cpu(&c).unwrap();
    let man: serde_json::Value = serde_json::from_str(&lire(&d.join("manifest.json"))).unwrap();
    assert_eq!(man["statut"], "termine");
    assert_eq!(man["rc"], 0);
    assert_eq!(man["backend"], "cpu");
    assert_eq!(man["precision"], "f64");
    assert_eq!(man["mode"], "su2-gen");
    assert_eq!(man["schema"], "kevinotron-unifie/1");
    assert!(man["binaire_sha256"].as_str().is_some_and(|s| s.len() == 64));
    assert!(man["noyaux_embarques"].as_array().is_some_and(|a| a.len() == 12));
    assert!(man["garde_ssd"]["seuil_go"].as_f64().is_some_and(|x| x >= 0.0), "garde SSD consignée");
    assert!(man["projection_octets"].as_u64().is_some_and(|x| x > 0));
    assert_eq!(man["options_numeriques"]["fp_contract"], "off");
    // chaque sortie a son sha256 exact
    let sorties = man["fichiers"].as_array().unwrap();
    assert!(sorties.len() >= 4);
    for s in sorties {
        let nom = s["nom"].as_str().unwrap();
        assert_eq!(s["sha256"].as_str().unwrap(), sha256_hex(&octets(&d.join(nom))), "{nom}");
        assert_eq!(s["octets"].as_u64().unwrap(), octets(&d.join(nom)).len() as u64);
    }
    // COMPLETE est le DERNIER fichier écrit
    let t_complete = fs::metadata(d.join("COMPLETE")).unwrap().modified().unwrap();
    for e in fs::read_dir(&d).unwrap().flatten() {
        let t = e.metadata().unwrap().modified().unwrap();
        assert!(t <= t_complete, "{:?} est plus récent que COMPLETE", e.file_name());
    }
    // refus d'écraser un répertoire existant
    assert!(executer_cpu(&c).is_err());
    assert!(d.join("COMPLETE").exists(), "le run refusé n'a pas touché au répertoire");
    fs::remove_dir_all(&d).unwrap();
}

/// Mesure en ligne qui échoue : simule une panne au milieu du run.
struct Panne;

impl MesureEnLigne for Panne {
    fn nom(&self) -> &str {
        "panne"
    }
    fn periode(&self) -> usize {
        2
    }
    fn mesurer(&mut self, _: Option<&kvu::gpu::Appareil>, _: Option<&kvu::su2::ChampGpu>, _: Option<&kvu::commun::ChampHote>,
               _: usize, _: u64, _: &Path) -> kvu::commun::Resultat<()> {
        Err(kvu::commun::Erreur::Gpu("panne simulée".into()))
    }
}

#[test]
fn un_echec_ne_laisse_jamais_de_complete() {
    // PROV-3 : une erreur au milieu du run (ici une mesure en ligne qui échoue) ⇒ pas de COMPLETE
    let d = rep("echec");
    let c = cfg(lot(4, &[(2.3, 1, Depart::Chaud)], 4), 3, 4, 1, 2, false, &d);
    let r = run::executer(None, &c, &mut [Box::new(Panne)], false);
    assert!(r.is_err());
    assert!(!d.join("COMPLETE").exists(), "COMPLETE présent après un échec");
    let man: serde_json::Value = serde_json::from_str(&lire(&d.join("manifest.json"))).unwrap();
    assert_eq!(man["statut"], "echec");
    assert_ne!(man["rc"], 0);
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn configuration_invalide_refusee_avant_toute_ecriture() {
    let d = rep("invalide");
    let mut c = cfg(lot(5, &[(2.3, 1, Depart::Chaud)], 4), 3, 4, 1, 0, false, &d);
    assert!(executer_cpu(&c).is_err());
    assert!(!d.exists(), "rien ne doit être créé pour une configuration invalide");
    c.lot.l = 4;
    c.skip = 0;
    assert!(executer_cpu(&c).is_err());
    assert!(!d.exists());
}

// ---------------------------------------------------------------------------------------------
// Formats de dump : q64 et zstd (module stockage)
// ---------------------------------------------------------------------------------------------

#[test]
fn dumps_q64_et_zstd_se_relisent_a_l_identique() {
    use kvu::stockage::npy;
    let mut ref_flat = None;
    for (format, zstd) in [("c16", false), ("q64", false), ("q64", true), ("c16", true)] {
        if zstd && !kvu::stockage::compression::disponible() {
            println!("# zstd absent : variante ignorée");
            continue;
        }
        let d = rep("dumps");
        let mut c = cfg(lot(4, &[(2.3, 77, Depart::Chaud)], 2), 6, 4, 1, 2, false, &d);
        c.format_dump = format.into();
        c.zstd = zstd;
        executer_cpu(&c).unwrap();
        let nom = if zstd { "cfg_000004.npy.zst" } else { "cfg_000004.npy" };
        let chemin = d.join(nom);
        assert!(chemin.exists(), "{nom} absent ({format}, zstd = {zstd})");
        assert!(!zstd || !d.join("cfg_000004.npy").exists(), "l'original est supprimé après la compression");
        let champ = npy::lire_champ(&chemin).unwrap();
        // même champ dans tous les formats
        match &ref_flat {
            None => ref_flat = Some(champ.q.clone()),
            Some(r) => assert_eq!(&champ.q, r, "{format} zstd = {zstd}"),
        }
        // le manifeste consigne le fichier compressé avec son sha256 et sa taille
        let man: serde_json::Value = serde_json::from_str(&lire(&d.join("manifest.json"))).unwrap();
        let f = man["fichiers"].as_array().unwrap().iter().find(|f| f["nom"] == nom).unwrap_or_else(|| panic!("{nom} non consigné"));
        assert_eq!(f["sha256"].as_str().unwrap(), sha256_hex(&octets(&chemin)));
        // q64 pèse la moitié de c16 (hors compression)
        if !zstd {
            let taille = fs::metadata(&chemin).unwrap().len();
            assert_eq!(taille, kvu::su2::run::octets_dump(4, format));
        }
        fs::remove_dir_all(&d).unwrap();
    }
}

#[test]
fn dump_et_reprise_avec_zstd() {
    if !kvu::stockage::compression::disponible() {
        return;
    }
    let (da, db) = (rep("rz_a"), rep("rz_b"));
    let mk = |d: &Path| {
        let mut c = cfg(lot(4, &[(2.3, 5, Depart::Chaud)], 2), 10, 8, 1, 2, true, d);
        c.format_dump = "q64".into();
        c.zstd = true;
        c.reprise_every = 6;
        c
    };
    run::executer(None, &mk(&da), &mut [], false).unwrap();
    run::executer_jusqua(None, &mk(&db), &mut [], false, Some(12)).unwrap();
    let c = run::charger_config(&db).unwrap();
    run::executer(None, &c, &mut [], true).unwrap();
    for n in ["therm.csv", "measurements.csv", "cfg_000000.npy.zst", "cfg_000002.npy.zst", "cfg_000008.npy.zst"] {
        assert_eq!(sha256_hex(&octets(&da.join(n))), sha256_hex(&octets(&db.join(n))), "{n}");
    }
    // le manifeste d'un run repris consigne TOUS les fichiers, dumps d'avant la reprise compris
    let man: serde_json::Value = serde_json::from_str(&lire(&db.join("manifest.json"))).unwrap();
    assert_eq!(man["statut"], "termine");
    assert!(man["fichiers"].as_array().unwrap().len() >= 6);
    fs::remove_dir_all(&da).unwrap();
    fs::remove_dir_all(&db).unwrap();
}

// ---------------------------------------------------------------------------------------------
// Mesures en ligne
// ---------------------------------------------------------------------------------------------

struct Compteur {
    appels: std::rc::Rc<std::cell::RefCell<Vec<(usize, u64)>>>,
    periode: usize,
}

impl MesureEnLigne for Compteur {
    fn nom(&self) -> &str {
        "compteur"
    }
    fn periode(&self) -> usize {
        self.periode
    }
    fn mesurer(&mut self, app: Option<&kvu::gpu::Appareil>, champ: Option<&kvu::su2::ChampGpu>,
               hote: Option<&kvu::commun::ChampHote>, chaine: usize, pas: u64, sortie: &Path) -> kvu::commun::Resultat<()> {
        assert!(app.is_none() && champ.is_none(), "CPU : ni appareil ni champ GPU");
        let h = hote.expect("champ hôte fourni sur CPU");
        assert_eq!(h.l, 4);
        assert!(sortie.is_dir());
        self.appels.borrow_mut().push((chaine, pas));
        Ok(())
    }
}

#[test]
fn mesures_en_ligne_appelees_a_la_bonne_periode() {
    let d = rep("enligne");
    let c = cfg(lot(4, &[(2.3, 1, Depart::Chaud), (2.5, 2, Depart::Chaud)], 2), 10, 20, 2, 0, false, &d);
    let appels = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut mesures: Vec<Box<dyn MesureEnLigne>> = vec![
        Box::new(Compteur { appels: appels.clone(), periode: 5 }),
        Box::new(Compteur { appels: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())), periode: 0 }),
    ];
    run::executer(None, &c, &mut mesures, false).unwrap();
    // m = 5, 10, 15, 20 ; pas total = therm + m·skip ; chaque chaîne
    let attendu: Vec<(usize, u64)> = [5u64, 10, 15, 20].iter().flat_map(|m| (0..2).map(move |ch| (ch, 10 + m * 2))).collect();
    assert_eq!(*appels.borrow(), attendu);
    fs::remove_dir_all(&d).unwrap();
}

// ---------------------------------------------------------------------------------------------
// Reprise (REPRISE-1) : 2N pas = N + reprise dans un NOUVEAU PROCESSUS + N, à l'octet
// ---------------------------------------------------------------------------------------------

/// Enfant : reprend le run de KVU_ENFANT_DIR (ne fait rien hors d'un appel du parent).
#[test]
fn enfant_reprise_cpu() {
    let Ok(d) = std::env::var("KVU_ENFANT_DIR") else { return };
    let c = run::charger_config(Path::new(&d)).expect("config_run.json");
    run::executer(None, &c, &mut [], true).expect("reprise de l'enfant");
}

fn reprendre_dans_un_processus(dir: &Path) {
    let st = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "enfant_reprise_cpu", "--nocapture", "--test-threads=1"])
        .env("KVU_ENFANT_DIR", dir)
        .env("KEVINOTRON_VERROU", "ignorer")
        .status()
        .unwrap();
    assert!(st.success(), "le processus enfant a échoué");
}

fn comparer_runs(a: &Path, b: &Path, nch: usize, noms: &dyn Fn(&Path) -> Vec<String>) -> usize {
    let mut n = 0;
    for c in 0..nch {
        let (da, db) = if nch == 1 { (a.to_path_buf(), b.to_path_buf()) } else { (a.join(format!("c{c:03}")), b.join(format!("c{c:03}"))) };
        for nom in noms(&da) {
            assert_eq!(octets(&da.join(&nom)), octets(&db.join(&nom)), "chaîne {c} : {nom}");
            n += 1;
        }
    }
    n
}

#[test]
fn reprise_1_standard_nouveau_processus() {
    for (nch, skip, arret) in [(1usize, 1u64, 15u64), (2, 2, 17)] {
        let chaines: Vec<(f64, u64, Depart)> = (0..nch).map(|i| (2.3 + 0.2 * i as f64, 50 + i as u64, Depart::Chaud)).collect();
        let (da, db) = (rep("rep_a"), rep("rep_b"));
        let mk = |d: &Path| {
            let mut c = cfg(lot(4, &chaines, 2), 15, 12, skip, 4, true, d);
            c.reprise_every = 7;
            c
        };
        run::executer(None, &mk(&da), &mut [], false).unwrap();
        // B : s'arrête sur une frontière d'itération ≥ `arret`, point de reprise écrit, pas de COMPLETE
        let (r, fini) = run::executer_jusqua(None, &mk(&db), &mut [], false, Some(arret)).unwrap();
        assert!(!fini && r.pas > 0);
        assert!(!db.join("COMPLETE").exists());
        assert!(db.join("reprise.json").exists() && db.join("reprise.champs").exists());
        let man: serde_json::Value = serde_json::from_str(&lire(&db.join("manifest.json"))).unwrap();
        assert_eq!(man["statut"], "en_cours", "un run interrompu n'est pas terminé");
        assert!(man["interruption"]["pas"].as_u64().is_some_and(|p| p >= arret));
        reprendre_dans_un_processus(&db);
        assert!(db.join("COMPLETE").exists());
        let n = comparer_runs(&da, &db, nch, &|d| {
            let mut v = noms_dumps(d);
            v.extend(["therm.csv".to_string(), "measurements.csv".to_string()]);
            v
        });
        assert!(n >= nch * 5, "{n} fichiers comparés");
        let man: serde_json::Value = serde_json::from_str(&lire(&db.join("manifest.json"))).unwrap();
        assert_eq!(man["statut"], "termine", "le manifeste est clos par la reprise");
        fs::remove_dir_all(&da).unwrap();
        fs::remove_dir_all(&db).unwrap();
    }
}

#[test]
fn reprise_au_milieu_de_la_thermalisation_et_tolerance_aux_lignes_en_trop() {
    let (da, db) = (rep("rt_a"), rep("rt_b"));
    let mk = |d: &Path| {
        let mut c = cfg(lot(4, &[(2.3, 9, Depart::Chaud)], 2), 30, 10, 1, 5, false, d);
        c.reprise_every = 10;
        c
    };
    run::executer(None, &mk(&da), &mut [], false).unwrap();
    run::executer_jusqua(None, &mk(&db), &mut [], false, Some(10)).unwrap();
    // simulation d'un crash APRÈS le point de reprise : des lignes en trop dans therm.csv
    let p = db.join("therm.csv");
    let mut t = lire(&p);
    t.push_str("11,0.5,0,0,0,0\n12,0.5,0,0,0,0\n");
    fs::write(&p, t).unwrap();
    let c = run::charger_config(&db).unwrap();
    run::executer(None, &c, &mut [], true).unwrap();
    for n in ["therm.csv", "measurements.csv"] {
        assert_eq!(octets(&da.join(n)), octets(&db.join(n)), "{n}");
    }
    for n in noms_dumps(&da) {
        assert_eq!(octets(&da.join(&n)), octets(&db.join(&n)), "{n}");
    }
    // un run terminé ne se reprend pas
    assert!(run::executer(None, &c, &mut [], true).is_err());
    fs::remove_dir_all(&da).unwrap();
    fs::remove_dir_all(&db).unwrap();
}

// ---------------------------------------------------------------------------------------------
// Bord R2-3 : séquence, C1, C2a, reprise
// ---------------------------------------------------------------------------------------------

fn cfg_bord(d: &Path, bc: &str, jauge: Option<u64>, nch: usize) -> ConfigRun {
    let chaines: Vec<(f64, u64, Depart)> = (0..nch).map(|i| (2.5, 700 + i as u64, Depart::Chaud)).collect();
    let mut c = cfg(lot(4, &chaines, 2), 30, 40, 1, 0, false, d);
    c.bord = Some(ConfigBord { bc: bc.into(), bc_graine: 4242, jauge, pre: 20, bloc: 10 });
    c
}

/// Séquence de su2_bord rejouée À LA MAIN avec les briques publiques : fichiers attendus (blocs de 10).
fn bord_a_la_main(bc: &str, jauge: Option<u64>) -> (String, String) {
    let l = 4;
    let mut n = Cpu64::nouveau(lot(l, &[(2.5, 700, Depart::Chaud)], 2)).unwrap();
    n.init().unwrap();
    for _ in 0..20 {
        n.sweep_hb().unwrap();
        n.sweep_or().unwrap();
        n.sweep_or().unwrap();
    }
    let b = match bc {
        "cold" => Bord::Froid,
        "hot" => Bord::Chaud(Graine(4242)),
        _ => {
            let mut eq = Cpu64::nouveau(lot(l, &[(2.5, 4242, Depart::Chaud)], 2)).unwrap();
            eq.init().unwrap();
            for _ in 0..20 {
                eq.sweep_hb().unwrap();
                eq.sweep_or().unwrap();
                eq.sweep_or().unwrap();
            }
            Bord::Copie(eq.telecharger(0).unwrap())
        }
    };
    bord::poser(&mut n, 0, &b, jauge.map(Graine)).unwrap();
    n.fixer_gel(Gel::BordX0Spatial).unwrap();
    let entete = {
        let mut s = String::from("bloc");
        for t in 0..l {
            s.push_str(&format!(",S{t}"));
        }
        for t in 0..l {
            s.push_str(&format!(",T{t}"));
        }
        s.push('\n');
        s
    };
    let mut blocs = |nsteps: usize| {
        let mut out = entete.clone();
        let mut acc = vec![0.0f64; 2 * l];
        let (mut cnt, mut nb) = (0, 0);
        for _ in 0..nsteps {
            n.sweep_hb().unwrap();
            n.sweep_or().unwrap();
            n.sweep_or().unwrap();
            let champ = n.telecharger(0).unwrap();
            let u: Vec<[f64; 4]> = (0..4 * 256).map(|i| champ.lien(i)).collect();
            let t = tranches_hote(&su2_hb::Geo::new(l), &u);
            for (a, x) in acc.iter_mut().zip(&t) {
                *a += x;
            }
            cnt += 1;
            if cnt == 10 {
                out.push_str(&format!("{nb}"));
                for a in &acc {
                    out.push_str(&format!(",{}", g17(a / 10.0)));
                }
                out.push('\n');
                acc.iter_mut().for_each(|x| *x = 0.0);
                cnt = 0;
                nb += 1;
            }
        }
        out
    };
    let th = blocs(30);
    let pr = blocs(40);
    (th, pr)
}

#[test]
fn sequence_bord_egale_la_sequence_a_la_main() {
    for (bc, jauge) in [("cold", None), ("hot", Some(31u64)), ("eq", Some(32)), ("eq", None)] {
        let d = rep("bord");
        let c = cfg_bord(&d, bc, jauge, 1);
        let r = run::executer(None, &c, &mut [], false).unwrap();
        assert_eq!((r.pas, r.mesures, r.plafond), (20 + 30 + 40, 4, 0));
        let (th, pr) = bord_a_la_main(bc, jauge);
        assert_eq!(lire(&d.join("therm_blocs.csv")), th, "{bc} jauge {jauge:?} : therm_blocs.csv");
        assert_eq!(lire(&d.join("blocs.csv")), pr, "{bc} jauge {jauge:?} : blocs.csv");
        assert!(d.join("COMPLETE").exists());
        fs::remove_dir_all(&d).unwrap();
    }
}

#[test]
fn bord_c1_gel_intact_et_c2a_jauge_exacte() {
    let d = rep("c1");
    let c = cfg_bord(&d, "eq", Some(99), 2);
    run::executer(None, &c, &mut [], false).unwrap();
    let man: serde_json::Value = serde_json::from_str(&lire(&d.join("manifest.json"))).unwrap();
    assert_eq!(man["gel_intact"], true);
    assert_eq!(man["mode"], "su2-bord");
    let infos = man["bord"].as_array().unwrap();
    assert_eq!(infos.len(), 2);
    for b in infos {
        // C1 : empreinte identique au début et à la fin du gel
        assert_eq!(b["fnv_bord_gel_debut"], b["fnv_bord_gel_fin"]);
        assert_eq!(b["gel_intact"], true);
        // C2a : jauge exacte des tranches
        let e = b["jauge_ecart_max"].as_f64().unwrap();
        assert!((0.0..=1e-14).contains(&e), "écart de jauge {e}");
        // la jauge a bien modifié le bord
        assert_ne!(b["fnv_bord_avant_jauge"], b["fnv_bord_gel_debut"]);
        assert!(b["P_bc_src"].as_f64().unwrap() > 0.0, "plaquette de la source eq");
    }
    // les deux chaînes ont des fichiers distincts
    assert_ne!(octets(&d.join("c000/blocs.csv")), octets(&d.join("c001/blocs.csv")));
    // sans jauge : −1
    let d2 = rep("c1b");
    run::executer(None, &cfg_bord(&d2, "hot", None, 1), &mut [], false).unwrap();
    let man: serde_json::Value = serde_json::from_str(&lire(&d2.join("manifest.json"))).unwrap();
    assert_eq!(man["bord"][0]["jauge_ecart_max"].as_f64().unwrap(), -1.0);
    assert_eq!(man["bord"][0]["fnv_bord_avant_jauge"], man["bord"][0]["fnv_bord_gel_debut"]);
    fs::remove_dir_all(&d).unwrap();
    fs::remove_dir_all(&d2).unwrap();
}

#[test]
fn le_bord_froid_a_l_empreinte_de_la_production() {
    // fixtures_bord/t_cold : L = 8, bord froid ⇒ FNV 6ff2d00186910383 (su2_bord, binaire de production S2)
    let d = rep("cold8");
    let mut c = cfg(lot(8, &[(2.5, 28092800002, Depart::Chaud)], 4), 10, 20, 1, 0, false, &d);
    c.bord = Some(ConfigBord { bc: "cold".into(), bc_graine: 28092800003, jauge: None, pre: 10, bloc: 10 });
    run::executer(None, &c, &mut [], false).unwrap();
    let man: serde_json::Value = serde_json::from_str(&lire(&d.join("manifest.json"))).unwrap();
    assert_eq!(man["bord"][0]["fnv_bord_gel_fin"], "6ff2d00186910383");
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn reprise_1_bord_nouveau_processus() {
    for (bc, jauge, nch, arret) in [("cold", None, 1usize, 20u64), ("eq", Some(5u64), 2, 40), ("hot", None, 1, 70)] {
        let (da, db) = (rep("rb_a"), rep("rb_b"));
        let mk = |d: &Path| {
            let mut c = cfg_bord(d, bc, jauge, nch);
            c.reprise_every = 20;
            c
        };
        run::executer(None, &mk(&da), &mut [], false).unwrap();
        let (_, fini) = run::executer_jusqua(None, &mk(&db), &mut [], false, Some(arret)).unwrap();
        assert!(!fini);
        assert!(!db.join("COMPLETE").exists());
        reprendre_dans_un_processus(&db);
        assert!(db.join("COMPLETE").exists());
        let n = comparer_runs(&da, &db, nch, &|_| vec!["therm_blocs.csv".to_string(), "blocs.csv".to_string()]);
        assert_eq!(n, 2 * nch);
        let (ma, mb): (serde_json::Value, serde_json::Value) =
            (serde_json::from_str(&lire(&da.join("manifest.json"))).unwrap(), serde_json::from_str(&lire(&db.join("manifest.json"))).unwrap());
        assert_eq!(ma["bord"], mb["bord"], "mêmes empreintes et plaquettes");
        assert_eq!(mb["gel_intact"], true);
        fs::remove_dir_all(&da).unwrap();
        fs::remove_dir_all(&db).unwrap();
    }
}

#[test]
fn bord_refuse_les_arguments_incoherents() {
    let d = rep("bordko");
    let mut c = cfg_bord(&d, "hot", None, 1);
    c.nmeas = 35; // pas multiple de bloc = 10
    assert!(executer_cpu(&c).is_err());
    c.nmeas = 40;
    c.lot.l = 2;
    assert!(executer_cpu(&c).is_err(), "bord : L ≥ 4");
    assert!(!d.exists());
}

// ---------------------------------------------------------------------------------------------
// Tirage de x0 (CPU) : même gate statistique que G-H04, avec contrôle négatif
// ---------------------------------------------------------------------------------------------

fn moments_exacts(alpha: f64) -> [f64; 4] {
    let n = 4000;
    let (mut z, mut m) = (0.0, [0.0; 4]);
    for k in 1..=n {
        let th = k as f64 * std::f64::consts::PI / (n as f64 + 1.0);
        let x = th.cos();
        let w = std::f64::consts::PI / (n as f64 + 1.0) * th.sin().powi(2) * (alpha * (x - 1.0)).exp();
        z += w;
        for (p, mp) in m.iter_mut().enumerate() {
            *mp += w * x.powi(p as i32 + 1);
        }
    }
    [m[0] / z, m[1] / z, m[2] / z, m[3] / z]
}

fn z(t: &[f64], alpha: f64) -> (f64, f64) {
    let n = t.len() as f64;
    let m1 = t.iter().sum::<f64>() / n;
    let m2 = t.iter().map(|x| x * x).sum::<f64>() / n;
    let e = moments_exacts(alpha);
    ((m1 - e[0]) / ((e[1] - e[0] * e[0]).sqrt() / n.sqrt()), (m2 - e[1]) / ((e[3] - e[1] * e[1]).sqrt() / n.sqrt()))
}

#[test]
fn tirage_de_x0_cpu_exact_et_mutant_rouge() {
    const N: usize = 2_000_000;
    for (i, a) in [0.3, 1.0, 1.999, 2.0, 5.0, 20.0, 100.0].into_iter().enumerate() {
        let (t, plafond) = su2_hb::test_x0_cpu(a, N, 9000 + i as u64, false);
        let (z1, z2) = z(&t, a);
        assert!(z1.abs() < 4.0 && z2.abs() < 4.0 && plafond == 0, "α = {a} : z1 = {z1:.2}, z2 = {z2:.2}, plafond {plafond}");
    }
    // contrôle négatif : le mutant est ROUGE à α ≥ 5
    for a in [5.0, 20.0] {
        let (t, _) = su2_hb::test_x0_cpu(a, N, 777, true);
        let (z1, z2) = z(&t, a);
        assert!(z1.abs() >= 4.0 || z2.abs() >= 4.0, "mutant non rouge à α = {a} : {z1:.2} {z2:.2}");
    }
}

#[test]
fn les_moments_exacts_sont_justes() {
    // E[x] = I2(α)/I1(α) ; à α = 5 : 0,5917… (série de Bessel)
    fn bessel_i(nu: f64, x: f64) -> f64 {
        (0..200)
            .map(|k| {
                let k = k as f64;
                ((2.0 * k + nu) * (x / 2.0).ln() - ln_gamma(k + 1.0) - ln_gamma(k + nu + 1.0)).exp()
            })
            .sum()
    }
    fn ln_gamma(x: f64) -> f64 {
        (1..(x as usize)).map(|i| (i as f64).ln()).sum()
    }
    let e = moments_exacts(5.0)[0];
    assert!((e - bessel_i(2.0, 5.0) / bessel_i(1.0, 5.0)).abs() < 1e-12, "{e}");
}

// ---------------------------------------------------------------------------------------------
// Limite du contrôle de trajectoire (intégration) : une faute qui corrompt TOUT le champ se résorbe en un pas
// ---------------------------------------------------------------------------------------------

/// Champ entièrement NaN au point de reprise (sha256 recalculé : faute silencieuse du champ résident). La garde des
/// agrafes dégénérées (copiée de su2_hip) remplace chaque lien par un tirage fini dès le premier pas : le run finit
/// avec COMPLETE, des CSV finis, mais une AUTRE trajectoire que le run sain. La sentinelle (run::sentinelle) ne voit
/// que les fautes qui atteignent une mesure ; seule une relance aux mêmes graines (déterministe) révèle celle-ci.
#[test]
fn faute_du_champ_se_resorbe_et_seule_la_relance_la_revele() {
    let mk = |d: &Path| {
        let mut c = cfg(lot(4, &[(2.3, 9, Depart::Chaud)], 2), 10, 8, 1, 0, false, d);
        c.reprise_every = 6;
        c
    };
    let (saine, fautive) = (rep("faute_saine"), rep("faute_nan"));
    for d in [&saine, &fautive] {
        let (_, fini) = run::executer_jusqua(None, &mk(d), &mut [], false, Some(6)).unwrap();
        assert!(!fini);
    }
    let champs = fautive.join(kvu::su2::reprise::NOM_CHAMPS);
    let n = octets(&champs).len() / 8;
    let nan: Vec<u8> = std::iter::repeat_n(f64::NAN.to_le_bytes(), n).flatten().collect();
    fs::write(&champs, &nan).unwrap();
    let pj = fautive.join("reprise.json");
    let mut etat: serde_json::Value = serde_json::from_str(&lire(&pj)).unwrap();
    etat["sha256_champs"] = serde_json::Value::String(sha256_hex(&nan));
    fs::write(&pj, serde_json::to_vec_pretty(&etat).unwrap()).unwrap();

    for d in [&saine, &fautive] {
        run::executer(None, &run::charger_config(d).unwrap(), &mut [], true).unwrap();
        assert!(d.join("COMPLETE").exists());
        for f in ["therm.csv", "measurements.csv"] {
            let t = lire(&d.join(f)).to_lowercase();
            assert!(!t.contains("nan") && !t.contains("inf"), "{f} contient une valeur non finie");
        }
    }
    assert_ne!(
        lire(&saine.join("measurements.csv")),
        lire(&fautive.join("measurements.csv")),
        "la faute devrait changer la trajectoire"
    );
    fs::remove_dir_all(&saine).unwrap();
    fs::remove_dir_all(&fautive).unwrap();
}
