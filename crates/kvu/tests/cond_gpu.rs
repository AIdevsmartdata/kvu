//! Gates GPU du bain conditionnel (HIP sur acemagic, CUDA sur le gamer) : EQC-1 (1 balayage `ref_rng` contre
//! cb.sweep), EQC-1b (état PCG64 après 5 balayages), EQC-3 (rapide CPU contre GPU, mêmes clés), contrôles.
//! `#[ignore = "GPU"]` : à lancer SOUS le verrou GPU de la machine, par fenêtre courte :
//!   acemagic : flock -w 7200 /tmp/kevinotron_gpu.lock cargo test -p kvu --release --test cond_gpu -- --ignored --nocapture --test-threads=1
//!   gamer    : flock -w 7200 /tmp/gamer_gpu.lock cargo test -p kvu --release --features cuda --test cond_gpu -- --ignored --nocapture --test-threads=1
//! Chaque gate imprime `PASS <nom>`, `FAIL <nom>` ou `SKIP <nom>` (SKIP ne vaut pas vert).
use kvu::commun::{BackendKind, ChampHote, Erreur, Graine};
use kvu::cond::cli::ouvrir_backend;
use kvu::cond::essais::{cas_balayages, dossier_commite, dossier_complet, ecart_max, CasBalayage};
use kvu::cond::geometrie::v_paires;
use kvu::cond::moteur::{ModeCond, MoteurCond, SourceAlea};
use kvu::cond::npy_min::lire_champ_quaternions;
use kvu::cond::GeometrieCond;
use kvu::gpu::Appareil;
use std::sync::Arc;

fn appareil() -> Option<Appareil> {
    match ouvrir_backend("auto") {
        Ok(a) => a,
        Err(e) => panic!("ouverture du GPU : {e}"),
    }
}

fn champ_du_cas(c: &CasBalayage) -> ChampHote {
    let (l, q) = lire_champ_quaternions(&c.fichier_cfg).unwrap();
    ChampHote { l, q }
}

#[test]
#[ignore = "GPU"]
fn gate_eqc1_gpu_un_balayage_ref_rng_contre_cb() {
    let Some(app) = appareil() else {
        println!("SKIP EQC-1 GPU : pas de GPU sur cette machine");
        return;
    };
    let dossier = dossier_complet().unwrap_or_else(dossier_commite);
    let cas = cas_balayages(&dossier, None).unwrap();
    let mut geos: std::collections::BTreeMap<usize, Arc<GeometrieCond>> = Default::default();
    let (mut dmax, mut vmax, mut umax) = (0.0f64, 0.0f64, 0.0f64);
    for c in &cas {
        let champ = champ_du_cas(c);
        let geo = geos.entry(c.l).or_insert_with(|| Arc::new(GeometrieCond::nouvelle(c.l).unwrap())).clone();
        let vfix = v_paires(&champ, &geo);
        let alea = SourceAlea::depuis_graine(ModeCond::RefRng, c.graine);
        let mut m = MoteurCond::nouveau(Some(&app), geo, &champ, vfix, c.beta, alea).unwrap();
        m.sweep().unwrap();
        let sortie = kvu::cond::npy_min::lire(&c.fichier_out).unwrap().f64().unwrap();
        let d = ecart_max(&m.telecharger().unwrap().q, &sortie);
        let (vdev, unit) = m.controles().unwrap();
        if d > 1e-12 || vdev > 1e-12 || unit > 1e-12 {
            println!(
                "FAIL EQC-1 GPU : L={} {} β={} graine={} : max|ΔC| = {d:e}, Vdev = {vdev:e}, unitarité = {unit:e}",
                c.l, c.cfg, c.beta, c.graine
            );
            panic!("EQC-1 GPU rouge");
        }
        match m.alea() {
            SourceAlea::Numpy(g) => assert_eq!((g.etat, g.inc), c.etat1, "état PCG64 après 1 balayage"),
            _ => unreachable!(),
        }
        // EQC-1b : 4 balayages de plus, état identique à cb
        for _ in 0..4 {
            m.sweep().unwrap();
        }
        match m.alea() {
            SourceAlea::Numpy(g) => {
                if (g.etat, g.inc) != c.etat5 {
                    println!("FAIL EQC-1b GPU : L={} {} β={} graine={}", c.l, c.cfg, c.beta, c.graine);
                    panic!("EQC-1b GPU rouge");
                }
            }
            _ => unreachable!(),
        }
        dmax = dmax.max(d);
        vmax = vmax.max(vdev);
        umax = umax.max(unit);
    }
    println!(
        "PASS EQC-1 ({}) : {} cas, max|ΔC| = {dmax:.2e}, Vdev = {vmax:.2e}, unitarité = {umax:.2e}",
        app.backend().nom(),
        cas.len()
    );
    println!(
        "PASS EQC-1b ({}) : état PCG64 identique à cb après 5 balayages sur {} cas",
        app.backend().nom(),
        cas.len()
    );
    if dossier_complet().is_none() {
        println!("NOTE : jeu commité (L = 4, 45 cas) ; la gate EQC-1 exige ≥ 100 cas (KVU_COND_FIXTURES, jeu complet)");
    }
}

#[test]
#[ignore = "GPU"]
fn gate_beta_nul_refuse_sur_gpu() {
    let Some(app) = appareil() else {
        println!("SKIP β = 0 GPU : pas de GPU");
        return;
    };
    let champ = ChampHote::froid(4).unwrap();
    let geo = Arc::new(GeometrieCond::nouvelle(4).unwrap());
    let vfix = v_paires(&champ, &geo);
    let r = MoteurCond::nouveau(Some(&app), geo, &champ, vfix, 0.0, SourceAlea::depuis_graine(ModeCond::RefRng, 1));
    assert!(matches!(r, Err(Erreur::Config(_))));
    println!("PASS β = 0 refusé en ref_rng (Erreur::Config) sur {}", app.backend().nom());
}

/// EQC-3 : rapide CPU Rust contre GPU, mêmes clés : 1 balayage ≤ 1e-12, 0 décision divergente (essais par lien).
#[test]
#[ignore = "GPU"]
fn gate_eqc3_rapide_cpu_contre_gpu() {
    let Some(app) = appareil() else {
        println!("SKIP EQC-3 : pas de GPU");
        return;
    };
    let dossier = dossier_complet().unwrap_or_else(dossier_commite);
    let cas: Vec<CasBalayage> = cas_balayages(&dossier, None).unwrap().into_iter().filter(|c| c.graine < 2).collect();
    let mut dmax = 0.0f64;
    let mut divergentes = 0usize;
    let mut liens = 0usize;
    for c in &cas {
        let champ = champ_du_cas(c);
        let geo = Arc::new(GeometrieCond::nouvelle(c.l).unwrap());
        let vfix = v_paires(&champ, &geo);
        let alea = || SourceAlea::Philox { graine: Graine(c.graine + 1000), sweep_id: 3 };
        let mut cpu = MoteurCond::nouveau(None, geo.clone(), &champ, vfix.clone(), c.beta, alea()).unwrap();
        let mut gpu = MoteurCond::nouveau(Some(&app), geo, &champ, vfix, c.beta, alea()).unwrap();
        cpu.activer_diag_essais().unwrap();
        gpu.activer_diag_essais().unwrap();
        cpu.sweep().unwrap();
        gpu.sweep().unwrap();
        let d = ecart_max(&cpu.telecharger().unwrap().q, &gpu.telecharger().unwrap().q);
        let (dc, dg) = (cpu.diag_rapide().unwrap(), gpu.diag_rapide().unwrap());
        divergentes += dc.essais.iter().zip(&dg.essais).filter(|(a, b)| a != b).count();
        liens += dc.essais.len();
        assert_eq!(dc.plafond, 0);
        assert_eq!(dg.plafond, 0);
        if d > 1e-12 {
            println!("FAIL EQC-3 : L={} {} β={} : max|ΔC| = {d:e}", c.l, c.cfg, c.beta);
            panic!("EQC-3 rouge");
        }
        dmax = dmax.max(d);
    }
    if divergentes != 0 {
        println!("FAIL EQC-3 : {divergentes} décision(s) divergente(s) sur {liens} liens");
        panic!("EQC-3 rouge");
    }
    println!(
        "PASS EQC-3 ({}) : {} cas, max|ΔC| = {dmax:.2e}, 0 décision divergente sur {liens} liens",
        app.backend().nom(),
        cas.len()
    );
}

/// Le mode rapide GPU conserve V et l'unitarité sur plusieurs balayages (dont β = 0), plafond = 0.
#[test]
#[ignore = "GPU"]
fn gate_rapide_gpu_invariants_cinq_balayages() {
    let Some(app) = appareil() else {
        println!("SKIP rapide GPU : pas de GPU");
        return;
    };
    let dossier = dossier_commite();
    for c in cas_balayages(&dossier, Some(4)).unwrap().iter().filter(|c| c.graine == 0) {
        let champ = champ_du_cas(c);
        let geo = Arc::new(GeometrieCond::nouvelle(4).unwrap());
        let vfix = v_paires(&champ, &geo);
        for beta in [c.beta, 0.0] {
            let mut m = MoteurCond::nouveau(
                Some(&app),
                geo.clone(),
                &champ,
                vfix.clone(),
                beta,
                SourceAlea::depuis_graine(ModeCond::Rapide, 7),
            )
            .unwrap();
            for _ in 0..5 {
                m.sweep().unwrap();
            }
            let (vdev, unit) = m.controles().unwrap();
            assert!(vdev <= 1e-12 && unit <= 1e-12, "{} β={beta} : Vdev {vdev:e} unit {unit:e}", c.cfg);
            assert_eq!(m.diag_rapide().unwrap().plafond, 0);
            assert!(m.telecharger().unwrap().q.iter().all(|x| x.is_finite()));
        }
    }
    println!("PASS rapide GPU ({}) : Vdev et unitarité ≤ 1e-12 sur 5 balayages, β = 0 compris", app.backend().nom());
    let _ = BackendKind::Cpu;
}

// ---------------------------------------------------------------------------------------------------------------
// EQC-4 : reprise de chaîne à l'octet, dans les deux modes, sur GPU, avec reprise dans un NOUVEAU PROCESSUS
// (les segments tournent dans des processus enfants : le binaire de test se relance lui-même).
// La mesure est le double de test `MesureProvisoire` : la gate juge la mécanique de reprise (champ, état de
// l'aléa, accumulateurs), pas la fibre ; le chemin complet avec la fibre de T6 est rejoué par gates_cond.sh.
// ---------------------------------------------------------------------------------------------------------------
use kvu::cond::chaine::{executer_avec, ConfigChaine, Issue, OptionsChaine};
use kvu::cond::mesure::MesureProvisoire;
use std::path::{Path, PathBuf};

fn config_eqc4(mode: ModeCond, out: &Path) -> ConfigChaine {
    ConfigChaine {
        cfg: dossier_commite().join("cfg_L4_haar0.npy"),
        beta: 2.3,
        graine: 2709,
        mode,
        ntherm: 5,
        nsweep: 60,
        every: 2,
        deach: 5,
        ckpt: 10,
        out: out.to_path_buf(),
        tordu: false,
    }
}

/// Segment exécuté dans un processus enfant (lancé par la gate ci-dessous, jamais seul).
#[test]
#[ignore = "enfant de la gate EQC-4"]
fn enfant_segment_eqc4() {
    let Ok(json) = std::env::var("KVU_COND_ENFANT_CFG") else {
        println!("SKIP enfant : lancé seul");
        return;
    };
    let cfg: ConfigChaine = serde_json::from_str(&json).unwrap();
    let reprendre = std::env::var("KVU_COND_ENFANT_REPRENDRE").is_ok();
    let arret = std::env::var("KVU_COND_ENFANT_ARRET").ok().and_then(|s| s.parse::<u64>().ok());
    let app = appareil().expect("GPU requis");
    let opts = OptionsChaine {
        arret_apres_ckpt: arret,
        mesure: Some(Box::new(MesureProvisoire::nouvelle(4))),
        journal: false,
    };
    match executer_avec(Some(&app), &cfg, reprendre, opts) {
        Ok(Issue::Terminee(_)) | Ok(Issue::Interrompue { .. }) => std::process::exit(0),
        Err(e) => {
            eprintln!("enfant : {e}");
            std::process::exit(e.code_sortie().max(1));
        }
    }
}

fn lancer_enfant(cfg: &ConfigChaine, reprendre: bool, arret: Option<u64>) -> bool {
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args(["--ignored", "--exact", "enfant_segment_eqc4", "--nocapture", "--test-threads=1"])
        .env("KVU_COND_ENFANT_CFG", serde_json::to_string(cfg).unwrap())
        .env("KEVINOTRON_SSD_MIN_GO", "0");
    if reprendre {
        cmd.env("KVU_COND_ENFANT_REPRENDRE", "1");
    }
    if let Some(n) = arret {
        cmd.env("KVU_COND_ENFANT_ARRET", n.to_string());
    }
    let s = cmd.output().expect("lancement de l'enfant");
    if !s.status.success() {
        eprintln!("{}{}", String::from_utf8_lossy(&s.stdout), String::from_utf8_lossy(&s.stderr));
    }
    s.status.success()
}

#[test]
#[ignore = "GPU"]
fn gate_eqc4_reprise_de_chaine_a_l_octet_gpu() {
    let Some(app) = appareil() else {
        println!("SKIP EQC-4 : pas de GPU");
        return;
    };
    let tmp = std::env::temp_dir().join(format!("kvu_cond_eqc4_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    for mode in [ModeCond::RefRng, ModeCond::Rapide] {
        let nom = format!("{mode:?}");
        let (a, b): (PathBuf, PathBuf) = (tmp.join(format!("seule_{nom}")), tmp.join(format!("reprise_{nom}")));
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(lancer_enfant(&config_eqc4(mode, &a), false, None), "chaîne ininterrompue ({nom})");
        assert!(lancer_enfant(&config_eqc4(mode, &b), false, Some(2)), "segment 1 ({nom})");
        assert!(b.join("reprise.json").exists() && !b.join("COMPLETE").exists());
        assert!(lancer_enfant(&config_eqc4(mode, &b), true, None), "segment 2 après reprise ({nom})");
        for f in ["F.npy", "D.npy", "blk.npy", "controles.json", "alea.json"] {
            let (x, y) = (std::fs::read(a.join(f)).unwrap(), std::fs::read(b.join(f)).unwrap());
            if x != y {
                println!("FAIL EQC-4 ({nom}) : {f} diffère après reprise");
                panic!("EQC-4 rouge");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    println!(
        "PASS EQC-4 ({}) : reprise de chaîne à l'octet (nouveau processus), modes ref_rng et rapide",
        app.backend().nom()
    );
}
