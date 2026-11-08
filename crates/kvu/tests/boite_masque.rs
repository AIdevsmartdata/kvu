// SPDX-License-Identifier: Apache-2.0
//! Masque de liens gelés et boîtes de Dirichlet : tests CPU (aucun GPU). Lot WP-DIR, GATES-DIR.md (groupe B).
//!
//! - le masque généralise le gel R2-3 (G-DIR-MASQUE-GEL) et un masque nul égale l'absence de masque (G-DIR-MASQUE-NUL) ;
//! - les liens gelés ne bougent jamais, les liens libres si (G-DIR-GELES-INTACTS), et les mutants à masque ignoré le prouvent ;
//! - un lot égale les chaînes seules (G-DIR-LOT), le calcul est indépendant du nombre de fils (G-DIR-FILS) et déterministe (G-DIR-DET) ;
//! - exécution de bout en bout de `su2 boite` sur CPU : sorties, manifeste `kvu-regen/1` valide, rejeu par `kvu regen`.

use kvu::commun::{ChampHote, Depart, Graine, Precision};
use kvu::gpu::OptionsCompil;
use kvu::regen::schema::valider_manifeste;
use kvu::su2::boite::{champ_initial, masque_boite, masque_gel_x0_spatial, mesurer_hote, BordBoite, GeometrieBoite};
use kvu::su2::boite_run::{executer_boite, observables_fichier, ChaineBoite, ConfigBoite};
use kvu::su2::{construire, AppMasque, ConfigLot, Gel, NoyauxSu2, ParamChaine};
use std::path::{Path, PathBuf};
use std::process::Command;

fn tmp(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("boite_masque").join(format!("{nom}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.parent().unwrap()).unwrap();
    d
}

fn lot(l: usize, chaines: &[(u64, Depart)], nor: usize) -> ConfigLot {
    ConfigLot {
        l,
        chaines: chaines.iter().map(|&(g, depart)| ParamChaine { beta: 2.5, graine: Graine(g), depart }).collect(),
        nor,
        precision: Precision::F64,
        options: OptionsCompil::reference_k1(),
        mutant_metro: false,
    }
}

fn pas(n: &mut dyn NoyauxSu2, nor: usize) {
    n.sweep_hb().unwrap();
    for _ in 0..nor {
        n.sweep_or().unwrap();
    }
}

fn bits(c: &ChampHote) -> Vec<u64> {
    c.q.iter().map(|x| x.to_bits()).collect()
}

fn bords() -> Vec<BordBoite> {
    let src = {
        let (k0, k1) = Graine(26_102_610_040).cle();
        let mut s = ChampHote { l: 10, q: vec![0.0; 16 * 10usize.pow(4)] };
        for (li, q) in s.q.chunks_exact_mut(4).enumerate() {
            q.copy_from_slice(&kvu::su2::cpu::haar_q(li as u32, 0xFFFF_FFFE, kvu::commun::ETIQUETTE_BORD, k0, k1));
        }
        s
    };
    vec![
        BordBoite::Froid,
        BordBoite::Jauge(Graine(26_102_610_041)),
        BordBoite::Z2Plein,
        BordBoite::Z2Torsion01,
        BordBoite::Flux,
        BordBoite::Haar(Graine(26_102_610_042)),
        BordBoite::Therm(src),
    ]
}

/// Moteur CPU d'un lot de boîtes (une chaîne par bord), champs de départ posés, masque fixé.
fn moteur_boite(ell: usize, bords: &[BordBoite], graines: &[u64], nor: usize, app: AppMasque) -> Box<dyn NoyauxSu2> {
    let g = GeometrieBoite::nouvelle(ell).unwrap();
    let chaines: Vec<(u64, Depart)> = graines.iter().map(|&s| (s, Depart::Froid)).collect();
    let mut n = construire(None, lot(g.lc, &chaines, nor)).unwrap();
    n.init().unwrap();
    for (c, b) in bords.iter().enumerate() {
        n.televerser(c, &champ_initial(&g, b).unwrap()).unwrap();
    }
    n.fixer_masque(Some(&masque_boite(&g)), app).unwrap();
    n
}

#[test]
fn le_masque_generalise_le_gel_r2_3() {
    let l = 6;
    let mut a = construire(None, lot(l, &[(26_102_610_001, Depart::Chaud)], 4)).unwrap();
    let mut b = construire(None, lot(l, &[(26_102_610_001, Depart::Chaud)], 4)).unwrap();
    a.init().unwrap();
    b.init().unwrap();
    a.fixer_gel(Gel::BordX0Spatial).unwrap();
    b.fixer_masque(Some(&masque_gel_x0_spatial(l)), AppMasque::Tout).unwrap();
    for _ in 0..5 {
        pas(a.as_mut(), 4);
        pas(b.as_mut(), 4);
    }
    assert_eq!(bits(&a.telecharger(0).unwrap()), bits(&b.telecharger(0).unwrap()), "masque ≡ gel R2-3 : octets différents");
    // et le gel a réellement gelé quelque chose
    let mut c = construire(None, lot(l, &[(26_102_610_001, Depart::Chaud)], 4)).unwrap();
    c.init().unwrap();
    for _ in 0..5 {
        pas(c.as_mut(), 4);
    }
    assert_ne!(bits(&a.telecharger(0).unwrap()), bits(&c.telecharger(0).unwrap()));
}

#[test]
fn un_masque_nul_egale_l_absence_de_masque() {
    let l = 6;
    let mut a = construire(None, lot(l, &[(26_102_610_002, Depart::Chaud)], 4)).unwrap();
    let mut b = construire(None, lot(l, &[(26_102_610_002, Depart::Chaud)], 4)).unwrap();
    a.init().unwrap();
    b.init().unwrap();
    b.fixer_masque(Some(&vec![0u8; 4 * l.pow(4)]), AppMasque::Tout).unwrap();
    for _ in 0..5 {
        pas(a.as_mut(), 4);
        pas(b.as_mut(), 4);
    }
    assert_eq!(bits(&a.telecharger(0).unwrap()), bits(&b.telecharger(0).unwrap()));
    // un masque de mauvaise longueur est refusé (jamais ignoré)
    assert!(b.fixer_masque(Some(&[0u8; 7]), AppMasque::Tout).is_err());
    // retirer le masque est permis
    b.fixer_masque(None, AppMasque::Tout).unwrap();
}

#[test]
fn les_liens_geles_ne_bougent_jamais_et_les_libres_si() {
    for ell in [4usize, 6] {
        let g = GeometrieBoite::nouvelle(ell).unwrap();
        let masque = masque_boite(&g);
        let tous = bords();
        let graines: Vec<u64> = (0..tous.len() as u64).map(|i| 26_102_610_050 + i).collect();
        let mut n = moteur_boite(ell, &tous, &graines, 2, AppMasque::Tout);
        let avant: Vec<ChampHote> = (0..tous.len()).map(|c| n.telecharger(c).unwrap()).collect();
        for t in 0..50 {
            pas(n.as_mut(), 2);
            if t == 4 {
                for (c, a) in avant.iter().enumerate() {
                    let f = n.telecharger(c).unwrap();
                    let (mut libres, mut modifies) = (0usize, 0usize);
                    for (li, &m) in masque.iter().enumerate() {
                        if m == 0 {
                            libres += 1;
                            modifies += usize::from(a.q[4 * li..4 * li + 4] != f.q[4 * li..4 * li + 4]);
                        }
                    }
                    assert_eq!(libres, g.n_libres());
                    assert!(modifies as f64 >= 0.99 * libres as f64, "ℓ = {ell} chaîne {c} : {modifies}/{libres} liens libres modifiés");
                }
            }
        }
        for (c, a) in avant.iter().enumerate() {
            let f = n.telecharger(c).unwrap();
            for (li, &m) in masque.iter().enumerate() {
                if m != 0 {
                    let (x, y) = (&a.q[4 * li..4 * li + 4], &f.q[4 * li..4 * li + 4]);
                    assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()), "ℓ = {ell} chaîne {c} : lien gelé {li} modifié");
                }
            }
            assert!(f.defaut_unitarite() < 1e-14);
        }
    }
}

#[test]
fn les_mutants_a_masque_ignore_modifient_le_bord() {
    let ell = 4;
    let g = GeometrieBoite::nouvelle(ell).unwrap();
    let masque = masque_boite(&g);
    for app in [AppMasque::IgnoreTout, AppMasque::IgnoreBain, AppMasque::IgnoreSurrelax] {
        let mut n = moteur_boite(ell, &[BordBoite::Z2Plein], &[26_102_610_060], 2, app);
        let avant = n.telecharger(0).unwrap();
        for _ in 0..5 {
            pas(n.as_mut(), 2);
        }
        let f = n.telecharger(0).unwrap();
        let ecart = masque
            .iter()
            .enumerate()
            .filter(|(_, &m)| m != 0)
            .map(|(li, _)| (0..4).map(|k| (avant.q[4 * li + k] - f.q[4 * li + k]).abs()).fold(0.0f64, f64::max))
            .fold(0.0f64, f64::max);
        assert!(ecart > 0.1, "mutant {} : le bord n'a pas bougé (max|Δq| = {ecart:e}) : contrôle négatif aveugle", app.nom());
    }
}

#[test]
fn un_lot_egale_les_chaines_seules() {
    let ell = 4;
    let tous = bords();
    let choix = [0usize, 2, 4, 6];
    let bl: Vec<BordBoite> = choix.iter().map(|&i| tous[i].clone()).collect();
    let gr: Vec<u64> = (0..choix.len() as u64).map(|i| 26_102_610_070 + i).collect();
    let mut lot4 = moteur_boite(ell, &bl, &gr, 2, AppMasque::Tout);
    for _ in 0..30 {
        pas(lot4.as_mut(), 2);
    }
    for (c, (b, s)) in bl.iter().zip(&gr).enumerate() {
        let mut seul = moteur_boite(ell, std::slice::from_ref(b), &[*s], 2, AppMasque::Tout);
        for _ in 0..30 {
            pas(seul.as_mut(), 2);
        }
        assert_eq!(bits(&lot4.telecharger(c).unwrap()), bits(&seul.telecharger(0).unwrap()), "chaîne {c} ({})", b.nom());
    }
}

#[test]
fn le_calcul_est_independant_du_nombre_de_fils() {
    let tous = bords();
    let gr: Vec<u64> = (0..tous.len() as u64).map(|i| 26_102_610_080 + i).collect();
    let mut res: Vec<Vec<Vec<u64>>> = Vec::new();
    for fils in [1usize, 8] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(fils).build().unwrap();
        let r = pool.install(|| {
            let mut n = moteur_boite(6, &tous, &gr, 2, AppMasque::Tout);
            for _ in 0..8 {
                pas(n.as_mut(), 2);
            }
            (0..tous.len()).map(|c| bits(&n.telecharger(c).unwrap())).collect::<Vec<_>>()
        });
        res.push(r);
    }
    assert_eq!(res[0], res[1], "1 fil contre 8 fils : octets différents");
}

// ------------------------------------------------------------------------------------------------------------------------
// Exécution de bout en bout (CPU)
// ------------------------------------------------------------------------------------------------------------------------

fn config(ell: usize, chaines: &[(&str, u64)], outdir: &Path) -> ConfigBoite {
    ConfigBoite {
        ell,
        beta: 2.5,
        chaines: chaines.iter().map(|&(b, g)| ChaineBoite { bord: b.into(), graine: g }).collect(),
        therm: 3,
        nmeas: 10,
        skip: 1,
        nor: 2,
        options: "ref-k1".into(),
        profil_tous: 5,
        therm_source: 10,
        dump_final: true,
        mutant: AppMasque::Tout,
        outdir: outdir.to_path_buf(),
    }
}

const SEPT: [(&str, u64); 7] = [
    ("cold", 26_102_610_101),
    ("gauge", 26_102_610_102),
    ("z2full", 26_102_610_103),
    ("z2_01", 26_102_610_104),
    ("flux", 26_102_610_105),
    ("haar", 26_102_610_106),
    ("therm", 26_102_610_107),
];

fn lire(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{} : {e}", p.display()))
}

#[test]
fn run_cpu_de_bout_en_bout() {
    let d = tmp("bout_en_bout").join("run");
    let cfg = config(4, &SEPT, &d);
    let r = executer_boite(None, &cfg, &["kvu".to_string(), "su2".to_string(), "boite".to_string()]).expect("run CPU");
    assert!(r.bord_intact && r.chaines == 7 && r.pas == 3 + 10);
    assert!(d.join("COMPLETE").exists() && d.join("manifest.json").exists() && d.join("manifest_regen.json").exists());
    for c in 0..7 {
        let rep = d.join(format!("c{c:03}"));
        let m = lire(&rep.join("mesures.csv"));
        assert_eq!(m.lines().count(), 11, "chaîne {c} : en-tête + 10 mesures");
        assert_eq!(m.lines().next().unwrap(), "pas,pc,w2");
        assert!(m.lines().nth(1).unwrap().starts_with("4,"), "pas = therm + m·skip");
        let p = lire(&rep.join("profil.csv"));
        assert_eq!(p.lines().count(), 3, "en-tête + profil aux mesures 5 et 10");
        assert_eq!(p.lines().next().unwrap(), "pas,d0,d1");
        // la dernière mesure est celle de la configuration finale
        let obs = observables_fichier(None, &rep.join("config_finale.npy"), 4).unwrap();
        let derniere: Vec<&str> = m.lines().last().unwrap().split(',').collect();
        assert_eq!(derniere[1].parse::<f64>().unwrap().to_bits(), obs.pc.to_bits(), "chaîne {c} : P_c final");
        assert_eq!(derniere[2].parse::<f64>().unwrap().to_bits(), obs.w2.to_bits(), "chaîne {c} : W₂ final");
        let der_p: Vec<f64> = p.lines().last().unwrap().split(',').skip(1).map(|x| x.parse().unwrap()).collect();
        assert_eq!(der_p.len(), 2);
        for (a, b) in der_p.iter().zip(&obs.profil) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
    // manifeste de régénération : valide, complet, classé
    let m: serde_json::Value = serde_json::from_str(&lire(&d.join("manifest_regen.json"))).unwrap();
    assert!(valider_manifeste(&m).is_empty(), "{:?}", valider_manifeste(&m));
    assert_eq!(m["schema"], "kvu-regen/1");
    assert_eq!(m["backend"], "cpu");
    assert_eq!(m["statut"], "COMPLETE");
    assert_eq!(m["parametres"]["L"], 4);
    assert_eq!(m["parametres"]["boite"]["n_libres"], 4 * 4 * 27);
    assert_eq!(m["fichiers"].as_array().unwrap().len(), 7 * 3);
    assert!(m["commande"].as_array().unwrap().iter().any(|a| a == "--chaines"));
    assert!(!m["commande"].as_array().unwrap().iter().any(|a| a == "--outdir"));
    let std_m: serde_json::Value = serde_json::from_str(&lire(&d.join("manifest.json"))).unwrap();
    assert_eq!(std_m["controle_bord"]["intact"], true);
    assert_eq!(std_m["statut"], "termine");
    assert_eq!(std_m["sources_thermiques"].as_array().unwrap().len(), 1);
    // un répertoire existant est refusé
    assert!(executer_boite(None, &cfg, &[]).is_err());
}

#[test]
fn run_deterministe_et_lot_egal_aux_chaines_seules() {
    let base = tmp("det");
    let a = base.join("a");
    let b = base.join("b");
    let trois = [SEPT[0], SEPT[2], SEPT[6]];
    executer_boite(None, &config(4, &trois, &a), &[]).unwrap();
    executer_boite(None, &config(4, &trois, &b), &[]).unwrap();
    for c in 0..3 {
        for f in ["mesures.csv", "profil.csv", "config_finale.npy"] {
            let p = format!("c{c:03}/{f}");
            assert_eq!(std::fs::read(a.join(&p)).unwrap(), std::fs::read(b.join(&p)).unwrap(), "DET : {p}");
        }
    }
    for (c, ch) in trois.iter().enumerate() {
        let seul = base.join(format!("seul{c}"));
        executer_boite(None, &config(4, &[*ch], &seul), &[]).unwrap();
        for f in ["mesures.csv", "profil.csv", "config_finale.npy"] {
            assert_eq!(
                std::fs::read(a.join(format!("c{c:03}/{f}"))).unwrap(),
                std::fs::read(seul.join(format!("c000/{f}"))).unwrap(),
                "LOT : chaîne {c} ({}) {f}",
                ch.0
            );
        }
    }
}

#[test]
fn les_mutants_du_run_sont_signales_dans_le_manifeste() {
    let base = tmp("mutants");
    for (nom, app, intact) in [
        ("tout", AppMasque::Tout, true),
        ("ignore", AppMasque::IgnoreTout, false),
        ("bain", AppMasque::IgnoreBain, false),
        ("surrelax", AppMasque::IgnoreSurrelax, false),
    ] {
        let d = base.join(nom);
        let mut cfg = config(4, &[("z2full", 26_102_610_201), ("cold", 26_102_610_202)], &d);
        cfg.mutant = app;
        let r = executer_boite(None, &cfg, &[]).unwrap_or_else(|e| panic!("{nom} : {e}"));
        assert_eq!(r.bord_intact, intact, "{nom}");
        let m: serde_json::Value = serde_json::from_str(&lire(&d.join("manifest.json"))).unwrap();
        assert_eq!(m["controle_bord"]["intact"], intact, "{nom}");
        assert_eq!(m["controle_bord"]["mutant"], app.nom());
        if !intact {
            assert!(m["controle_bord"]["max_dq"].as_f64().unwrap() > 0.1, "{nom}");
        }
    }
}

/// G-DIR-COMPTE aux grands ℓ de GATES-DIR.md (16 et 24) : dénombrement indépendant (boucles sur les coordonnées du réseau de calcul,
/// classification écrite autrement que `lien_*`) contre la formule `4ℓ(ℓ−1)³` et contre le masque du code, octet par octet.
#[test]
fn comptes_et_masque_aux_grands_ell() {
    for ell in [16usize, 24] {
        let g = GeometrieBoite::nouvelle(ell).unwrap();
        let lc = g.lc;
        let masque = masque_boite(&g);
        assert_eq!(masque.len(), 4 * lc.pow(4));
        let (mut libres, mut fixes, mut inexistants) = (0usize, 0usize, 0usize);
        for mu in 0..4 {
            for s in 0..lc.pow(4) {
                let x = [s / lc.pow(3), (s / lc.pow(2)) % lc, (s / lc) % lc, s % lc];
                let existe = x.iter().all(|&c| c <= ell) && x[mu] < ell;
                let sur_bord = |y: [usize; 4]| y.iter().any(|&c| c == 0 || c == ell);
                let mut y = x;
                y[mu] += 1;
                let nature = if !existe { 2 } else if sur_bord(x) && sur_bord(y) { 1 } else { 0 };
                match nature {
                    0 => libres += 1,
                    1 => fixes += 1,
                    _ => inexistants += 1,
                }
                assert_eq!(masque[mu * lc.pow(4) + s], u8::from(nature != 0), "ℓ = {ell}, lien ({mu}, {x:?})");
            }
        }
        assert_eq!(libres, 4 * ell * (ell - 1).pow(3), "n_libres(ℓ = {ell})");
        assert_eq!(libres, g.n_libres());
        assert_eq!(fixes, g.n_fixes());
        assert_eq!(fixes, 4 * ell * (ell + 1).pow(3) - libres);
        assert_eq!(inexistants, g.n_inexistants());
        assert_eq!(libres + fixes + inexistants, 4 * lc.pow(4));
    }
}

#[test]
fn champ_froid_donne_des_mesures_exactes() {
    for ell in [2usize, 4, 6] {
        let g = GeometrieBoite::nouvelle(ell).unwrap();
        let o = mesurer_hote(&g, &champ_initial(&g, &BordBoite::Froid).unwrap());
        assert_eq!(o.pc, 1.0);
        assert!(o.profil.iter().all(|&p| p == 1.0));
        if ell >= 4 {
            assert_eq!(o.w2, 1.0);
        } else {
            assert!(o.w2.is_nan());
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------------
// Ligne de commande
// ------------------------------------------------------------------------------------------------------------------------

fn kvu(args: &[&str]) -> (i32, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_kvu")).args(args).env("KEVINOTRON_VERROU", "ignorer").output().unwrap();
    let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&o.stderr));
    (o.status.code().unwrap_or(-1), s)
}

#[test]
fn cli_su2_boite_cpu_et_codes_de_sortie() {
    let base = tmp("cli");
    let d = base.join("ok");
    let (rc, s) = kvu(&[
        "su2", "boite", "--backend", "cpu", "--ell", "4", "--beta", "2.5", "--bords", "cold,z2full", "--repliques", "2", "--graine", "26102610301",
        "--therm", "3", "--nmeas", "6", "--nor", "2", "--profil-tous", "3", "--dump-final", "--outdir", d.to_str().unwrap(),
    ]);
    assert_eq!(rc, 0, "{s}");
    assert!(s.contains("PERF boite ell=4 beta=2.5 K=4 backend=cpu precision=f64 pas=9 "), "{s}");
    for c in 0..4 {
        assert!(d.join(format!("c{c:03}/mesures.csv")).exists());
    }
    let m: serde_json::Value = serde_json::from_str(&lire(&d.join("manifest_regen.json"))).unwrap();
    let ch: Vec<&str> = m["graines"]["etiquettes"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    assert_eq!(ch, vec!["cold:26102610301", "cold:26102610302", "z2full:26102610303", "z2full:26102610304"]);
    // codes de sortie : usage = 2
    let neuf = |n: &str| base.join(n).to_string_lossy().into_owned();
    for (args, pourquoi) in [
        (vec!["--ell", "5", "--chaines", "cold:1"], "ℓ impair"),
        (vec!["--ell", "4", "--bords", "cold", "--chaines", "cold:1"], "--bords et --chaines"),
        (vec!["--ell", "4"], "ni --bords ni --chaines"),
        (vec!["--ell", "4", "--chaines", "tiede:1"], "bord inconnu"),
        (vec!["--ell", "4", "--chaines", "cold:1", "--precision", "f32"], "f32 refusé (G-DIR-F32-REFUS)"),
        (vec!["--ell", "4", "--chaines", "cold:1", "--mutant", "bizarre"], "mutant inconnu"),
    ] {
        let o = neuf(&format!("refus_{}", pourquoi.len()));
        let mut a = vec!["su2", "boite", "--backend", "cpu", "--beta", "2.5", "--outdir", &o];
        a.extend(args);
        let (rc, s) = kvu(&a);
        assert_eq!(rc, 2, "{pourquoi} : {s}");
    }
    // répertoire existant : refus, rien n'est écrasé
    let (rc, s) = kvu(&["su2", "boite", "--backend", "cpu", "--ell", "4", "--beta", "2.5", "--chaines", "cold:1", "--outdir", d.to_str().unwrap()]);
    assert_eq!(rc, 2, "{s}");
    assert!(s.contains("REFUS D'ÉCRASER"), "{s}");
    // l'aide documente la sous-commande
    let (rc, s) = kvu(&["su2", "boite", "--help"]);
    assert_eq!(rc, 0);
    assert!(s.contains("--chaines") && s.contains("--profil-tous") && s.contains("--outdir") && !s.contains("--mutant"), "{s}");
}

#[test]
fn kvu_regen_rejoue_la_boite_cpu_et_voit_une_graine_fausse() {
    let base = tmp("regen");
    let d = base.join("origine");
    let (rc, s) = kvu(&[
        "su2", "boite", "--backend", "cpu", "--ell", "4", "--beta", "2.5", "--chaines", "cold:26102610401,z2full:26102610402,haar:26102610403",
        "--therm", "3", "--nmeas", "8", "--nor", "2", "--profil-tous", "4", "--dump-final", "--outdir", d.to_str().unwrap(),
    ]);
    assert_eq!(rc, 0, "{s}");
    let man = d.join("manifest_regen.json");
    let (rc, s) = kvu(&["regen", man.to_str().unwrap(), "--sortie", base.join("rejeu").to_str().unwrap()]);
    assert_eq!(rc, 0, "{s}");
    assert!(s.contains("PASS REGEN 9/9 fichiers identiques"), "{s}");
    // graine modifiée dans la commande du manifeste : le rejeu diffère (rc 8)
    let mut m: serde_json::Value = serde_json::from_str(&lire(&man)).unwrap();
    let commande = m["commande"].as_array_mut().unwrap();
    let i = commande.iter().position(|a| a == "--chaines").unwrap();
    commande[i + 1] = serde_json::json!("cold:26102610401,z2full:26102610402,haar:26102610404");
    let faux = base.join("faux");
    std::fs::create_dir_all(&faux).unwrap();
    // le manifeste falsifié est placé à côté des originaux : mêmes noms de fichiers, mêmes sha attendus
    let chemin_faux = faux.join("manifest_regen.json");
    std::fs::write(&chemin_faux, serde_json::to_string_pretty(&m).unwrap()).unwrap();
    let (rc, s) = kvu(&["regen", chemin_faux.to_str().unwrap(), "--sortie", base.join("rejeu_faux").to_str().unwrap()]);
    assert_eq!(rc, 8, "{s}");
    assert!(s.contains("FAIL REGEN"), "{s}");
}
