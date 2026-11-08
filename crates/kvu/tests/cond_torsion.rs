//! Lecture tordue de la saturation v3 (`cond chaine --tordu`, `fibre::torsion`) contre `torsion.py`.
//!
//! Vérificateur du 04/10 : `cond chaine` ne produisait pas Ft et Dtw ; torsion.py ne peut pas les recalculer à
//! partir de F.npy, D.npy et blk.npy (Dtw demande les gradients de plaquettes, Ft les pv par site), donc S3 v3 ne
//! pouvait pas migrer. Fixtures : `validation/cond/gen_torsion.py` (torsion.py + noyau2/cb.py, configurations Haar
//! L = 4 et 8 des fixtures de fibre).
use kvu::commun::ChampHote;
use kvu::cond::chaine::{executer_avec, ConfigChaine, Issue, OptionsChaine};
use kvu::cond::essais::dossier_commite;
use kvu::cond::moteur::ModeCond;
use kvu::cond::{npy_min, GeometrieCond};
use kvu::fibre::torsion::{MesureurTordu, NF_TW};
use kvu::fibre::{plaquettes_gradients_cpu, Mesureur, poids_fourier};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const CAS: &[&str] = &["L4_haar0", "L4_haar1", "L4_haar2", "L8_haar0", "L8_haar1"];
/// Ft : sommes de L⁴ termes O(1/L²) ; ordre de sommation différent de numpy ⇒ quelques ulp.
const TOL_FT: f64 = 1e-13;
/// Dtw : Frobenius relatif, chaque matrice.
const TOL_DTW: f64 = 1e-12;

fn racine_fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/fibre")
}

fn charger(cas: &str) -> (ChampHote, Vec<f64>, Vec<f64>) {
    let d = racine_fixtures().join(cas);
    let (forme, q) = kvu::fibre::npy::lire_f64(&d.join("q.npy")).expect("q.npy");
    let (_, ft) = kvu::fibre::npy::lire_f64(&d.join("tw_Ft.npy")).expect("tw_Ft.npy (gen_torsion.py)");
    let (fd, dtw) = kvu::fibre::npy::lire_f64(&d.join("tw_Dtw.npy")).expect("tw_Dtw.npy (gen_torsion.py)");
    assert_eq!(fd, vec![3, NF_TW, NF_TW]);
    (ChampHote { l: forme[1], q }, ft, dtw)
}

fn ecart_frob_rel(a: &[f64], b: &[f64]) -> f64 {
    let n: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt() / n.max(1e-300)
}

fn comparer(cas: &str, ft: &[f64], dtw: &[f64], ft_ref: &[f64], dtw_ref: &[f64]) {
    let e_ft = ft.iter().zip(ft_ref).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(e_ft <= TOL_FT, "{cas} : Ft écart {e_ft:e}");
    for k in 0..3 {
        let m = NF_TW * NF_TW;
        let e = ecart_frob_rel(&dtw[k * m..(k + 1) * m], &dtw_ref[k * m..(k + 1) * m]);
        assert!(e <= TOL_DTW, "{cas} : Dtw[{k}] écart relatif {e:e}");
    }
}

#[test]
fn torsion_cpu_contre_torsion_py() {
    for cas in CAS {
        let (champ, ft_ref, dtw_ref) = charger(cas);
        let geo = GeometrieCond::nouvelle(champ.l).unwrap();
        let tw = MesureurTordu::nouveau(&geo).unwrap();
        let (pv, gp) = plaquettes_gradients_cpu(&champ);
        let ft = tw.f_tw(&pv).unwrap();
        let dtw = tw.d_tw(&champ.q, &gp).unwrap();
        comparer(cas, &ft, &dtw, &ft_ref, &dtw_ref);
        // Dm et Dp diffèrent de D0 (la torsion agit) ; D0 symétrique
        assert!(ecart_frob_rel(&dtw[324..648], &dtw[..324]) > 1e-3, "{cas} : Dm = D0 ?");
        for i in 0..NF_TW {
            for j in 0..NF_TW {
                assert_eq!(dtw[i * NF_TW + j], dtw[j * NF_TW + i]);
            }
        }
    }
    println!("PASS torsion CPU = torsion.py ({} cas, Ft ≤ {TOL_FT:e}, Dtw ≤ {TOL_DTW:e})", CAS.len());
}

#[test]
fn torsion_independante_du_nombre_de_fils() {
    let (champ, _, _) = charger("L8_haar0");
    let geo = GeometrieCond::nouvelle(champ.l).unwrap();
    let tw = MesureurTordu::nouveau(&geo).unwrap();
    let (pv, gp) = plaquettes_gradients_cpu(&champ);
    let un = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let (ft1, d1) = un.install(|| (tw.f_tw(&pv).unwrap(), tw.d_tw(&champ.q, &gp).unwrap()));
    let (ftn, dn) = (tw.f_tw(&pv).unwrap(), tw.d_tw(&champ.q, &gp).unwrap());
    assert_eq!(ft1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), ftn.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    assert_eq!(d1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), dn.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
}

/// Sur le GPU : plaquettes et gradients de `k_plaq_grad`, puis la même réduction hôte.
#[test]
#[ignore = "GPU"]
fn torsion_gpu_contre_torsion_py() {
    let app = match kvu::gpu::ouvrir_auto() {
        Ok(Some(a)) => a,
        _ => {
            println!("SKIP torsion GPU (aucun GPU)");
            return;
        }
    };
    for cas in CAS {
        let (champ, ft_ref, dtw_ref) = charger(cas);
        let geo = Arc::new(GeometrieCond::nouvelle(champ.l).unwrap());
        let tw = MesureurTordu::nouveau(&geo).unwrap();
        let mut m = Mesureur::nouveau(Some(&app), geo.clone(), &poids_fourier(champ.l, 1)).unwrap();
        let (pv, gp) = m.plaquettes(None, 0, Some(&champ)).unwrap();
        let pv_seul = m.valeurs_plaquettes(None, 0, Some(&champ)).unwrap();
        assert_eq!(pv, pv_seul, "{cas} : plaquettes sans gradients différentes");
        let ft = tw.f_tw(&pv).unwrap();
        let dtw = tw.d_tw(&champ.q, &gp).unwrap();
        comparer(cas, &ft, &dtw, &ft_ref, &dtw_ref);
    }
    println!("PASS torsion GPU ({}) = torsion.py ({} cas)", app.backend().nom(), CAS.len());
}

// ------------------------------------------------------------------------------------------------
// Chaîne complète avec --tordu
// ------------------------------------------------------------------------------------------------

static SEUIL_SSD: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn cfg(out: &Path) -> ConfigChaine {
    ConfigChaine {
        cfg: dossier_commite().join("cfg_L4_haar0.npy"),
        beta: 2.3,
        graine: 2709,
        mode: ModeCond::RefRng,
        ntherm: 2,
        nsweep: 20,
        every: 2,
        deach: 5,
        ckpt: 4,
        out: out.to_path_buf(),
        tordu: true,
    }
}

fn dossier(nom: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("kvu_cond_torsion_{nom}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn opts(arret: Option<u64>) -> OptionsChaine {
    OptionsChaine { arret_apres_ckpt: arret, mesure: None, journal: false }
}

#[test]
fn chaine_tordue_formes_coherence_et_reprise_a_l_octet() {
    let _g = SEUIL_SSD.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("KEVINOTRON_SSD_MIN_GO", "0");
    let (a, b) = (dossier("seule"), dossier("reprise"));
    match executer_avec(None, &cfg(&a), false, opts(None)).unwrap() {
        Issue::Terminee(r) => assert_eq!((r.mesures_f, r.mesures_d), (10, 2)),
        autre => panic!("{autre:?}"),
    }
    let ft = npy_min::lire(&a.join("Ft.npy")).unwrap();
    let dtw = npy_min::lire(&a.join("Dtw.npy")).unwrap();
    assert_eq!(ft.forme, vec![10, NF_TW]);
    assert_eq!(dtw.forme, vec![2, 3, NF_TW, NF_TW]);
    // S1 (lignes 0..5 de Ft) = lignes 9p + 8 de F (moment nul de fourier_weights_sparse), à l'arrondi près
    let (f, ftv) = (npy_min::lire(&a.join("F.npy")).unwrap().f64().unwrap(), ft.f64().unwrap());
    for k in 0..10 {
        for p in 0..6 {
            let (x, y) = (ftv[k * NF_TW + p], f[k * 54 + 9 * p + 8]);
            assert!((x - y).abs() < 1e-13, "mesure {k}, plan {p} : Ft {x} contre F {y}");
        }
    }
    let man: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(a.join("manifest.json")).unwrap()).unwrap();
    let noms: Vec<&str> = man["fichiers"].as_array().unwrap().iter().map(|f| f["nom"].as_str().unwrap()).collect();
    assert!(noms.contains(&"Ft.npy") && noms.contains(&"Dtw.npy"), "{noms:?}");
    assert_eq!(man["config"]["tordu"], true);

    // reprise : arrêt après le 1er point de reprise, puis reprise dans un « nouveau processus » ⇒ mêmes octets
    assert_eq!(executer_avec(None, &cfg(&b), false, opts(Some(1))).unwrap(), Issue::Interrompue { ckpt: 1 });
    executer_avec(None, &cfg(&b), true, opts(None)).unwrap();
    for nom in ["F.npy", "D.npy", "blk.npy", "Ft.npy", "Dtw.npy", "controles.json", "alea.json"] {
        assert_eq!(std::fs::read(a.join(nom)).unwrap(), std::fs::read(b.join(nom)).unwrap(), "{nom} après reprise");
    }
    // un point de reprise SANS torsion ne peut pas être repris AVEC (et inversement)
    let c = dossier("melange");
    let mut sans = cfg(&c);
    sans.tordu = false;
    executer_avec(None, &sans, false, opts(Some(1))).unwrap();
    assert!(executer_avec(None, &cfg(&c), true, opts(None)).is_err(), "reprise tordue d'un point non tordu");
    std::env::remove_var("KEVINOTRON_SSD_MIN_GO");
    for d in [a, b, c] {
        let _ = std::fs::remove_dir_all(d);
    }
}
