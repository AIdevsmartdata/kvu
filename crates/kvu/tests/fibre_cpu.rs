//! FIB-1 sur la référence CPU Rust : mesure de fibre contre `cb.py` (fixtures de gen_fixtures.py).
//!
//! `cargo test -p kvu --test fibre_cpu` lit le jeu commité (cas L = 4 et 8). Avec
//! `KVU_FIBRE_FIXTURES=<dir>` (jeu complet, 14 cas jusqu'à L = 16), c'est la gate FIB-1 CPU complète
//! (à lancer en --release).

mod fibre_commun;

use fibre_commun::*;
use kvu::commun::ChampHote;
use kvu::cond::GeometrieCond;
use kvu::fibre::{plaquettes_gradients_cpu, poids_fourier, Mesureur};
use std::sync::Arc;

/// Mesureurs CPU (un par L) et résultats d'un cas.
fn calculer(cas: &Cas, mes: &mut Mesureur) -> Resultats {
    let (pv, gp) = plaquettes_gradients_cpu(&cas.champ);
    let mesure = mes.mesurer(None, 0, Some(&cas.champ)).expect("mesurer");
    let libre = mes.mesurer_libre(None, 0, Some(&cas.champ)).expect("mesurer_libre");
    Resultats { pv, gp, mesure, libre }
}

#[test]
fn fib1_cpu_contre_cb() {
    let racine = dossier_fixtures();
    let noms = noms_des_cas(&racine);
    assert!(!noms.is_empty(), "aucun cas dans {}", racine.display());
    let mut bilan = Bilan::default();
    let mut vus_l: Vec<usize> = Vec::new();
    for nom in &noms {
        let cas = charger_cas(&racine, nom);
        let w = poids_fourier(cas.l, 1);
        if !vus_l.contains(&cas.l) {
            vus_l.push(cas.l);
            comparer_poids(&mut bilan, &racine, cas.l, &w);
        }
        let geo = Arc::new(GeometrieCond::nouvelle(cas.l).unwrap());
        let mut mes = Mesureur::nouveau(None, geo, &w).unwrap();
        let r = calculer(&cas, &mut mes);
        controler_fib1(&mut bilan, &cas, &r, &w, "");
        println!("  cas {nom} : D = {:.3e}", norme_frob(&r.mesure.d));
    }
    bilan.imprimer();
    let verdict = if bilan.ok() { "PASS" } else { "FAIL" };
    println!("{verdict} FIB-1 CPU ({} cas : {})", noms.len(), noms.join(" "));
    assert!(bilan.ok(), "FIB-1 CPU ROUGE : {:?}", bilan.echecs);
}

#[test]
fn mesurer_f_egale_le_f_de_mesurer() {
    let racine = dossier_fixtures();
    let cas = charger_cas(
        &racine,
        &noms_des_cas(&racine).into_iter().find(|n| n.contains("haar0")).unwrap(),
    );
    let w = poids_fourier(cas.l, 1);
    let mut mes = Mesureur::nouveau(None, Arc::new(GeometrieCond::nouvelle(cas.l).unwrap()), &w).unwrap();
    let m = mes.mesurer(None, 0, Some(&cas.champ)).unwrap();
    let f = mes.mesurer_f(None, 0, Some(&cas.champ)).unwrap();
    assert_eq!(m.f, f);
    assert_eq!(m.f.len(), 54);
    assert_eq!(m.d.len(), 54 * 54);
    assert_eq!(m.blk.len(), 6 * (cas.l / 2).pow(4));
}

#[test]
fn deux_mesures_identiques_a_l_octet() {
    let racine = dossier_fixtures();
    let cas = charger_cas(
        &racine,
        &noms_des_cas(&racine).into_iter().find(|n| n.contains("haar0")).unwrap(),
    );
    let w = poids_fourier(cas.l, 1);
    let mut mes = Mesureur::nouveau(None, Arc::new(GeometrieCond::nouvelle(cas.l).unwrap()), &w).unwrap();
    let a = mes.mesurer(None, 0, Some(&cas.champ)).unwrap();
    let b = mes.mesurer(None, 0, Some(&cas.champ)).unwrap();
    assert_eq!(a, b);
}

#[test]
fn cpu_sans_champ_hote_refuse() {
    let w = poids_fourier(4, 1);
    let mut mes = Mesureur::nouveau(None, Arc::new(GeometrieCond::nouvelle(4).unwrap()), &w).unwrap();
    assert!(mes.mesurer(None, 0, None).is_err());
    assert!(mes.mesurer_f(None, 0, None).is_err());
    assert!(mes.mesurer_libre(None, 0, None).is_err());
    assert!(!mes.sur_gpu());
    assert_eq!(mes.l(), 4);
}

#[test]
fn champ_de_mauvais_l_refuse() {
    let w = poids_fourier(4, 1);
    let mut mes = Mesureur::nouveau(None, Arc::new(GeometrieCond::nouvelle(4).unwrap()), &w).unwrap();
    let c6 = ChampHote::froid(6).unwrap();
    assert!(mes.mesurer(None, 0, Some(&c6)).is_err());
}

#[test]
fn mesureur_refuse_une_geometrie_et_des_poids_incoherents() {
    let mut w = poids_fourier(4, 1);
    w.ptr.pop();
    assert!(Mesureur::nouveau(None, Arc::new(GeometrieCond::nouvelle(4).unwrap()), &w).is_err());
}

#[test]
fn cli_mesurer_de_bout_en_bout() {
    let racine = dossier_fixtures();
    let cas = charger_cas(
        &racine,
        &noms_des_cas(&racine).into_iter().find(|n| n.contains("haar0")).unwrap(),
    );
    // sur le disque de la cible (la garde SSD de `fibre mesurer` refuse /tmp, un tmpfs en RAM de ~59 Go < 100 Go)
    let base = option_env!("CARGO_TARGET_TMPDIR").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let tmp = base.join(format!("kvu_fibre_cpu_cli_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let l = cas.l;
    kvu::fibre::npy::ecrire_f64(&tmp.join("cfg.npy"), &[4, l, l, l, l, 4], &cas.champ.q).unwrap();
    let cmd = kvu::fibre::cli::commandes().remove(0);
    for libre in [false, true] {
        let out = tmp.join(if libre { "libre" } else { "cond" });
        let cfg = tmp.join("cfg.npy");
        let mut argv = vec![
            "fibre",
            "mesurer",
            "--cfg",
            cfg.to_str().unwrap(),
            "--backend",
            "cpu",
            "--out",
            out.to_str().unwrap(),
        ];
        if libre {
            argv.push("--libre");
        }
        let m = cmd.clone().try_get_matches_from(argv).unwrap();
        assert_eq!(kvu::fibre::cli::executer("fibre", &m).unwrap(), 0);
        let (forme, d) = kvu::fibre::npy::lire_f64(&out.join("D.npy")).unwrap();
        assert_eq!(forme, vec![54, 54]);
        let (_, f) = kvu::fibre::npy::lire_f64(&out.join("F.npy")).unwrap();
        let (dref, fref) = if libre {
            ("cb_Dlibre.npy", "cb_Flibre.npy")
        } else {
            ("cb_D.npy", "cb_F.npy")
        };
        assert!(ecart_frob(&d, &cas.f64s(dref)) <= TOL_D_REL);
        assert!(max_abs_ecart(&f, &cas.f64s(fref)) <= TOL_F_ABS);
        assert_eq!(out.join("blk.npy").exists(), !libre);
        if !libre {
            let blk = f32s(&out.join("blk.npy"));
            let refb = f32s(&cas.dir.join("cb_blk.npy"));
            assert!(blk.iter().zip(&refb).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn cli_refuse_une_commande_inconnue() {
    let m = kvu::fibre::cli::commandes()[0]
        .clone()
        .try_get_matches_from(["fibre", "mesurer", "--cfg", "x", "--out", "y", "--backend", "cpu"])
        .unwrap();
    assert!(kvu::fibre::cli::executer("autre", &m).is_err());
    // fichier absent : erreur E/S, pas de panique
    assert!(kvu::fibre::cli::executer("fibre", &m).is_err());
}

/// Banc CPU (rayon) : mesure complète à L = 16 et 24 sur une configuration pseudo-aléatoire.
/// `cargo test --release -p kvu --test fibre_cpu banc -- --ignored --nocapture`
#[test]
#[ignore = "banc"]
fn banc_cpu() {
    use std::time::Instant;
    let ls: Vec<usize> = std::env::var("KVU_FIBRE_BANC_L")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![16, 24]);
    for l in ls {
        let champ = kvu::fibre::banc::champ_pseudo_aleatoire(l, 40 + l as u64);
        let w = poids_fourier(l, 1);
        let mut mes = Mesureur::nouveau(None, Arc::new(GeometrieCond::nouvelle(l).unwrap()), &w).unwrap();
        let _ = mes.mesurer(None, 0, Some(&champ)).unwrap();
        let mut t = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = mes.mesurer(None, 0, Some(&champ)).unwrap();
            t.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("BANC CPU L={l:2} : mesurer {:.0} ms (min {:.0})", t[1], t[0]);
    }
}
