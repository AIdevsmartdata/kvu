// SPDX-License-Identifier: Apache-2.0
//! Gates GPU des boîtes de Dirichlet SU(2) (lot WP-DIR, GATES-DIR.md groupes A/B). `#[ignore = "GPU"]` : à lancer SOUS le verrou
//! GPU de la machine, par fenêtres de 10 minutes au plus.
//!
//! ```text
//! acemagic : flock -w 7200 /tmp/kevinotron_gpu.lock cargo test -p kvu --release --test boite_gpu -- --ignored --nocapture --test-threads=1
//! gamer    : flock -w 7200 /tmp/gamer_gpu.lock      cargo test -p kvu --release --features cuda --test boite_gpu -- --ignored --nocapture --test-threads=1
//! ```
//! Le backend est KEVINOTRON_BACKEND s'il est posé, sinon CUDA si la feature `cuda` est compilée, sinon HIP. Chaque gate imprime
//! `PASS|FAIL|SKIP|INFO <gate> : détail` (un SKIP ne vaut PAS vert).

use kvu::commun::{BackendKind, ChampHote, Depart, Graine, Precision};
use kvu::gpu::{self, Appareil, OptionsCompil};
use kvu::su2::boite::{champ_initial, masque_boite, masque_gel_x0_spatial, mesurer_hote, BordBoite, GeometrieBoite};
use kvu::su2::boite_gpu::MesureBoiteGpu;
use kvu::su2::boite_run::{executer_boite, ChaineBoite, ConfigBoite};
use kvu::su2::cpu::Cpu64;
use kvu::su2::ref64::Ref64Gpu;
use kvu::su2::{AppMasque, ConfigLot, Gel, NoyauxSu2, ParamChaine};
use std::path::{Path, PathBuf};

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

fn tmp(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("boite_gpu").join(format!("{nom}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.parent().unwrap()).unwrap();
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

fn e(x: impl std::fmt::Display) -> String {
    x.to_string()
}

fn pas(n: &mut dyn NoyauxSu2, nor: usize) -> Result<(), String> {
    n.sweep_hb().map_err(e)?;
    for _ in 0..nor {
        n.sweep_or().map_err(e)?;
    }
    Ok(())
}

fn bits(c: &ChampHote) -> Vec<u64> {
    c.q.iter().map(|x| x.to_bits()).collect()
}

fn max_ecart(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn nb_liens_differents(a: &ChampHote, b: &ChampHote) -> usize {
    a.q.chunks_exact(4).zip(b.q.chunks_exact(4)).filter(|(x, y)| x.iter().zip(y.iter()).any(|(p, q)| p.to_bits() != q.to_bits())).count()
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

// ---------------------------------------------------------------------------------------------------------------------
// G-DIR-MASQUE-GEL et G-DIR-MASQUE-NUL : le masque généralise le gel R2-3, un masque nul égale l'absence de masque
// ---------------------------------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn g_dir_masque_gel_et_nul() {
    let Some(app) = appareil_ou_skip("G-DIR-MASQUE-GEL") else { return };
    gate("G-DIR-MASQUE-GEL", || {
        let l = 6;
        let cfg = lot(l, &[(2.5, 26_102_610_001, Depart::Chaud)], 4);
        let mut a = Ref64Gpu::nouveau(&app, cfg.clone()).map_err(e)?;
        let mut b = Ref64Gpu::nouveau(&app, cfg).map_err(e)?;
        a.init().map_err(e)?;
        b.init().map_err(e)?;
        a.fixer_gel(Gel::BordX0Spatial).map_err(e)?;
        b.fixer_masque(Some(&masque_gel_x0_spatial(l)), AppMasque::Tout).map_err(e)?;
        for _ in 0..5 {
            pas(&mut a, 4)?;
            pas(&mut b, 4)?;
        }
        let (qa, qb) = (a.telecharger(0).map_err(e)?, b.telecharger(0).map_err(e)?);
        if bits(&qa) != bits(&qb) {
            return Err("masque ≡ gel R2-3 : octets différents".into());
        }
        Ok(format!("octets identiques après 5 pas (L = 6, β = 2,5, {})", app.backend().nom()))
    });
    gate("G-DIR-MASQUE-NUL", || {
        let l = 6;
        let cfg = lot(l, &[(2.5, 26_102_610_002, Depart::Chaud)], 4);
        let mut a = Ref64Gpu::nouveau(&app, cfg.clone()).map_err(e)?;
        let mut b = Ref64Gpu::nouveau(&app, cfg).map_err(e)?;
        a.init().map_err(e)?;
        b.init().map_err(e)?;
        b.fixer_masque(Some(&vec![0u8; 4 * l.pow(4)]), AppMasque::Tout).map_err(e)?;
        for _ in 0..5 {
            pas(&mut a, 4)?;
            pas(&mut b, 4)?;
        }
        if bits(&a.telecharger(0).map_err(e)?) != bits(&b.telecharger(0).map_err(e)?) {
            return Err("masque nul ≠ pas de masque".into());
        }
        if b.fixer_masque(Some(&[0u8; 5]), AppMasque::Tout).is_ok() {
            return Err("un masque de mauvaise longueur a été accepté".into());
        }
        Ok(format!("octets identiques après 5 pas ({})", app.backend().nom()))
    });
}

// ---------------------------------------------------------------------------------------------------------------------
// G-DIR-GELES-INTACTS (+ mutants M-DIR-1/2/3)
// ---------------------------------------------------------------------------------------------------------------------

fn moteur_boite_gpu(app: &Appareil, ell: usize, bords: &[BordBoite], graines: &[u64], nor: usize, mut_: AppMasque) -> Result<Ref64Gpu, String> {
    let g = GeometrieBoite::nouvelle(ell).map_err(e)?;
    let chaines: Vec<(f64, u64, Depart)> = graines.iter().map(|&s| (2.5, s, Depart::Froid)).collect();
    let mut n = Ref64Gpu::nouveau(app, lot(g.lc, &chaines, nor)).map_err(e)?;
    n.init().map_err(e)?;
    for (c, b) in bords.iter().enumerate() {
        n.televerser(c, &champ_initial(&g, b).map_err(e)?).map_err(e)?;
    }
    n.fixer_masque(Some(&masque_boite(&g)), mut_).map_err(e)?;
    Ok(n)
}

#[test]
#[ignore = "GPU"]
fn g_dir_geles_intacts() {
    let Some(app) = appareil_ou_skip("G-DIR-GELES-INTACTS") else { return };
    gate("G-DIR-GELES-INTACTS", || {
        let mut pire_libre = 1.0f64;
        for ell in [4usize, 6, 8] {
            let g = GeometrieBoite::nouvelle(ell).map_err(e)?;
            let masque = masque_boite(&g);
            let tous = bords();
            let graines: Vec<u64> = (0..tous.len() as u64).map(|i| 26_102_610_050 + i).collect();
            let mut n = moteur_boite_gpu(&app, ell, &tous, &graines, 2, AppMasque::Tout)?;
            let avant: Vec<ChampHote> = (0..tous.len()).map(|c| n.telecharger(c).unwrap()).collect();
            for t in 0..50 {
                pas(&mut n, 2)?;
                if t == 4 {
                    for (c, a) in avant.iter().enumerate() {
                        let f = n.telecharger(c).map_err(e)?;
                        let (mut libres, mut modifies) = (0usize, 0usize);
                        for (li, &m) in masque.iter().enumerate() {
                            if m == 0 {
                                libres += 1;
                                modifies += usize::from(a.q[4 * li..4 * li + 4] != f.q[4 * li..4 * li + 4]);
                            }
                        }
                        let fr = modifies as f64 / libres as f64;
                        pire_libre = pire_libre.min(fr);
                        if fr < 0.99 {
                            return Err(format!("ℓ = {ell}, chaîne {c} : seulement {modifies}/{libres} liens libres modifiés"));
                        }
                    }
                }
            }
            for (c, a) in avant.iter().enumerate() {
                let f = n.telecharger(c).map_err(e)?;
                for (li, &m) in masque.iter().enumerate() {
                    if m != 0 && a.q[4 * li..4 * li + 4].iter().zip(&f.q[4 * li..4 * li + 4]).any(|(p, q)| p.to_bits() != q.to_bits()) {
                        return Err(format!("ℓ = {ell}, chaîne {c} : le lien gelé {li} a bougé"));
                    }
                }
            }
        }
        Ok(format!("ℓ ∈ {{4,6,8}} × 7 bords × 50 pas : liens gelés identiques à l'octet ; fraction minimale de liens libres modifiés après 5 pas {pire_libre:.4} ({})", app.backend().nom()))
    });
    gate("G-DIR-GELES-INTACTS-MUTANTS", || {
        let ell = 4;
        let g = GeometrieBoite::nouvelle(ell).map_err(e)?;
        let masque = masque_boite(&g);
        let mut detail = Vec::new();
        for m in [AppMasque::IgnoreTout, AppMasque::IgnoreBain, AppMasque::IgnoreSurrelax] {
            let mut n = moteur_boite_gpu(&app, ell, &[BordBoite::Z2Plein], &[26_102_610_060], 2, m)?;
            let avant = n.telecharger(0).map_err(e)?;
            for _ in 0..5 {
                pas(&mut n, 2)?;
            }
            let f = n.telecharger(0).map_err(e)?;
            let ecart = (0..masque.len())
                .filter(|&li| masque[li] != 0)
                .map(|li| max_ecart(&avant.q[4 * li..4 * li + 4], &f.q[4 * li..4 * li + 4]))
                .fold(0.0, f64::max);
            if ecart <= 0.1 {
                return Err(format!("mutant {} : le bord n'a pas bougé (max|Δq| = {ecart:e}) : ECHEC_OUTIL", m.nom()));
            }
            detail.push(format!("{} max|Δq| = {ecart:.2}", m.nom()));
        }
        Ok(format!("les 3 mutants ROUGES : {}", detail.join(" ; ")))
    });
}

// ---------------------------------------------------------------------------------------------------------------------
// G-DIR-CPU-GPU : un bain thermique puis une surrelaxation, mêmes états de départ, CPU contre GPU
// ---------------------------------------------------------------------------------------------------------------------

struct Resultat1 {
    divergentes: usize,
    ecart_hb: f64,
    ecart_or: f64,
    gele_ok: bool,
    obs_ecart: f64,
    differents_hb: usize,
    differents_or: usize,
    agrafes_nulles: usize,
}

fn comparer(app: &Appareil, ell: usize, bord: &BordBoite, beta: f64, apres_pas: usize, app_cpu: AppMasque) -> Result<Resultat1, String> {
    let g = GeometrieBoite::nouvelle(ell).map_err(e)?;
    let masque = masque_boite(&g);
    let cfg = lot(g.lc, &[(beta, 26_102_610_020 + ell as u64, Depart::Froid)], 2);
    let mut gp = Ref64Gpu::nouveau(app, cfg.clone()).map_err(e)?;
    gp.init().map_err(e)?;
    let initial = champ_initial(&g, bord).map_err(e)?;
    gp.televerser(0, &initial).map_err(e)?;
    gp.fixer_masque(Some(&masque), AppMasque::Tout).map_err(e)?;
    for _ in 0..apres_pas {
        pas(&mut gp, 2)?;
    }
    let depart = gp.telecharger(0).map_err(e)?;
    let mut cp = Cpu64::nouveau(cfg).map_err(e)?;
    cp.init().map_err(e)?;
    cp.televerser(0, &depart).map_err(e)?;
    cp.fixer_sweep_id(gp.sweep_id());
    cp.fixer_masque(Some(&masque), app_cpu).map_err(e)?;
    let dg = gp.sweep_hb_diag().map_err(e)?;
    let dc = cp.sweep_hb_diag().map_err(e)?;
    let divergentes = dg.iter().zip(&dc).filter(|(a, b)| a != b).count();
    let agrafes_nulles = dg.iter().filter(|&&d| d == 0xFFFF_FFFF).count();
    let (qg, qc) = (gp.telecharger(0).map_err(e)?, cp.telecharger(0).map_err(e)?);
    let ecart_hb = max_ecart(&qg.q, &qc.q);
    let differents_hb = nb_liens_differents(&qg, &qc);
    gp.sweep_or().map_err(e)?;
    cp.sweep_or().map_err(e)?;
    let (rg, rc) = (gp.telecharger(0).map_err(e)?, cp.telecharger(0).map_err(e)?);
    let ecart_or = max_ecart(&rg.q, &rc.q);
    let differents_or = nb_liens_differents(&rg, &rc);
    let mut gele_ok = true;
    for (li, &m) in masque.iter().enumerate() {
        if m != 0 {
            let (a, b, c) = (&depart.q[4 * li..4 * li + 4], &rg.q[4 * li..4 * li + 4], &rc.q[4 * li..4 * li + 4]);
            gele_ok &= a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) && a.iter().zip(c).all(|(x, y)| x.to_bits() == y.to_bits());
        }
    }
    let (og, oc) = (mesurer_hote(&g, &rg), mesurer_hote(&g, &rc));
    let mut obs_ecart = (og.pc - oc.pc).abs();
    if ell >= 4 {
        obs_ecart = obs_ecart.max((og.w2 - oc.w2).abs());
    }
    for (a, b) in og.profil.iter().zip(&oc.profil) {
        obs_ecart = obs_ecart.max((a - b).abs());
    }
    Ok(Resultat1 { divergentes, ecart_hb, ecart_or, gele_ok, obs_ecart, differents_hb, differents_or, agrafes_nulles })
}

#[test]
#[ignore = "GPU"]
fn g_dir_cpu_gpu() {
    let Some(app) = appareil_ou_skip("G-DIR-CPU-GPU") else { return };
    gate("G-DIR-CPU-GPU", || {
        let mut pire = (0.0f64, 0.0f64, 0.0f64);
        let (mut cas, mut nulles_flux6, mut diff_hb, mut diff_or, mut total) = (0, 0usize, 0usize, 0usize, 0usize);
        for ell in [4usize, 6] {
            let g = GeometrieBoite::nouvelle(ell).map_err(e)?;
            for bord in [BordBoite::Froid, BordBoite::Flux, BordBoite::Haar(Graine(26_102_610_042)), BordBoite::Z2Plein] {
                for beta in [2.5, 8.0] {
                    for apres in [0usize, 3] {
                        let r = comparer(&app, ell, &bord, beta, apres, AppMasque::Tout)?;
                        let id = format!("ℓ = {ell}, bord {}, β = {beta}, après {apres} pas", bord.nom());
                        if r.divergentes != 0 {
                            return Err(format!("{id} : {} décisions divergentes", r.divergentes));
                        }
                        if r.ecart_hb > 1e-12 || r.ecart_or > 1e-12 {
                            return Err(format!("{id} : max|Δq| HB {:.3e}, OR {:.3e} > 1e-12", r.ecart_hb, r.ecart_or));
                        }
                        if !r.gele_ok {
                            return Err(format!("{id} : un lien gelé diffère (CPU, GPU ou départ)"));
                        }
                        if r.obs_ecart > 1e-12 {
                            return Err(format!("{id} : observables écart {:.3e} > 1e-12", r.obs_ecart));
                        }
                        pire = (pire.0.max(r.ecart_hb), pire.1.max(r.ecart_or), pire.2.max(r.obs_ecart));
                        if matches!(bord, BordBoite::Flux) && ell == 6 && apres == 0 {
                            nulles_flux6 += r.agrafes_nulles;
                        }
                        diff_hb += r.differents_hb;
                        diff_or += r.differents_or;
                        total += g.n_libres();
                        cas += 1;
                    }
                }
            }
        }
        if nulles_flux6 == 0 {
            verdict("SKIP G-DIR-CPU-GPU-HAAR-BRANCHE : aucune agrafe exactement nulle au départ froid de flux ℓ = 6 : la branche de Haar n'est pas testée");
        }
        verdict(&format!(
            "INFO G-DIR-CPU-GPU : agrafes exactement nulles (flux, ℓ = 6, départ froid) = {nulles_flux6} ; liens à octets différents CPU/GPU : {diff_hb} après un bain, {diff_or} après une surrelaxation (sur {total} liens libres cumulés)"
        ));
        Ok(format!(
            "{cas} cas (ℓ ∈ {{4,6}} × 4 bords × β ∈ {{2,5 ; 8}} × départ froid/après 3 pas), 0 décision divergente, max|Δq| HB {:.2e}, OR {:.2e}, observables {:.2e} (≤ 1e-12), liens gelés identiques à l'octet ({})",
            pire.0, pire.1, pire.2, app.backend().nom()
        ))
    });
    gate("G-DIR-CPU-GPU-MUTANT", || {
        // M-DIR-7 : un CPU qui ignore le masque doit être vu (décisions divergentes ou écart > 1e-3)
        let r = comparer(&app, 4, &BordBoite::Z2Plein, 2.5, 3, AppMasque::IgnoreTout)?;
        if r.divergentes == 0 && r.ecart_hb <= 1e-3 && r.gele_ok {
            return Err("le CPU qui ignore le masque n'est pas détecté : ECHEC_OUTIL".into());
        }
        Ok(format!("CPU sans masque ROUGE : {} décisions divergentes, max|Δq| HB {:.2e}, liens gelés identiques : {}", r.divergentes, r.ecart_hb, r.gele_ok))
    });
}

// ---------------------------------------------------------------------------------------------------------------------
// G-DIR-OBS : mesures GPU contre la référence hôte (mêmes ordres de sommation)
// ---------------------------------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn g_dir_obs_gpu_contre_hote() {
    let Some(app) = appareil_ou_skip("G-DIR-OBS") else { return };
    gate("G-DIR-OBS", || {
        let (mut pire, mut octets_egaux, mut total) = (0.0f64, 0usize, 0usize);
        for ell in [2usize, 4, 6] {
            let g = GeometrieBoite::nouvelle(ell).map_err(e)?;
            let mesure = MesureBoiteGpu::nouveau(&app, &g, 1, 1).map_err(e)?;
            for bord in [BordBoite::Froid, BordBoite::Haar(Graine(26_102_610_042)), BordBoite::Z2Plein, BordBoite::Flux] {
                let mut cpu = Cpu64::nouveau(lot(g.lc, &[(2.5, 26_102_610_090, Depart::Froid)], 2)).map_err(e)?;
                cpu.init().map_err(e)?;
                cpu.televerser(0, &champ_initial(&g, &bord).map_err(e)?).map_err(e)?;
                cpu.fixer_masque(Some(&masque_boite(&g)), AppMasque::Tout).map_err(e)?;
                for _ in 0..20 {
                    pas(&mut cpu, 2)?;
                }
                let h = cpu.telecharger(0).map_err(e)?;
                let mut champ = kvu::su2::ChampGpu::allouer(&app, g.lc, 1, Precision::F64).map_err(e)?;
                champ.televerser(&app, 0, &h).map_err(e)?;
                let og = mesure.observables(&champ, 0).map_err(e)?;
                let oh = mesurer_hote(&g, &h);
                let mut ecart = (og.pc - oh.pc).abs();
                total += 1;
                let mut egal = og.pc.to_bits() == oh.pc.to_bits();
                if ell >= 4 {
                    ecart = ecart.max((og.w2 - oh.w2).abs());
                    egal &= og.w2.to_bits() == oh.w2.to_bits();
                }
                for (a, b) in og.profil.iter().zip(&oh.profil) {
                    ecart = ecart.max((a - b).abs());
                    egal &= a.to_bits() == b.to_bits();
                }
                if ell < 4 && !og.w2.is_nan() {
                    return Err("ℓ < 4 : W₂ doit être NaN".into());
                }
                octets_egaux += usize::from(egal);
                pire = pire.max(ecart);
                if ecart > 1e-14 {
                    return Err(format!("ℓ = {ell}, bord {} : écart GPU/hôte {ecart:.3e} > 1e-14", bord.nom()));
                }
            }
            // champ froid : exactement 1
            let froid = champ_initial(&g, &BordBoite::Froid).map_err(e)?;
            let mut champ = kvu::su2::ChampGpu::allouer(&app, g.lc, 1, Precision::F64).map_err(e)?;
            champ.televerser(&app, 0, &froid).map_err(e)?;
            let o = mesure.observables(&champ, 0).map_err(e)?;
            if o.pc != 1.0 || o.profil.iter().any(|&p| p != 1.0) || (ell >= 4 && o.w2 != 1.0) {
                return Err(format!("ℓ = {ell} : champ froid ⇒ P_c = W₂ = profil = 1 exactement : obtenu {o:?}"));
            }
        }
        Ok(format!("{total} configurations : max écart GPU/hôte {pire:.2e} (≤ 1e-14) ; octets identiques sur {octets_egaux}/{total} ; champ froid exact ({})", app.backend().nom()))
    });
}

// ---------------------------------------------------------------------------------------------------------------------
// G-DIR-DET, G-DIR-LOT au niveau de `su2 boite` (GPU), G-DIR-F32-REFUS
// ---------------------------------------------------------------------------------------------------------------------

fn config(ell: usize, chaines: &[(&str, u64)], outdir: &Path) -> ConfigBoite {
    ConfigBoite {
        ell,
        beta: 2.5,
        chaines: chaines.iter().map(|&(b, g)| ChaineBoite { bord: b.into(), graine: g }).collect(),
        therm: 20,
        nmeas: 100,
        skip: 1,
        nor: 2,
        options: "ref-k1".into(),
        profil_tous: 10,
        therm_source: 50,
        dump_final: true,
        mutant: AppMasque::Tout,
        outdir: outdir.to_path_buf(),
    }
}

#[test]
#[ignore = "GPU"]
fn g_dir_det_et_lot_gpu() {
    let Some(app) = appareil_ou_skip("G-DIR-DET") else { return };
    let base = tmp("det");
    let sept = [("cold", 26_102_610_011), ("z2full", 26_102_610_012), ("flux", 26_102_610_013), ("haar", 26_102_610_014), ("gauge", 26_102_610_015)];
    let tous = [sept[0], sept[1], sept[2], sept[3], sept[4], ("therm", 26_102_610_016)];
    gate("G-DIR-DET", || {
        let (a, b) = (base.join("a"), base.join("b"));
        executer_boite(Some(&app), &config(6, &tous, &a), &[]).map_err(e)?;
        executer_boite(Some(&app), &config(6, &tous, &b), &[]).map_err(e)?;
        for c in 0..tous.len() {
            for f in ["mesures.csv", "profil.csv", "config_finale.npy"] {
                let p = format!("c{c:03}/{f}");
                if std::fs::read(a.join(&p)).map_err(e)? != std::fs::read(b.join(&p)).map_err(e)? {
                    return Err(format!("{p} diffère entre deux runs identiques"));
                }
            }
        }
        Ok(format!("{} chaînes × 3 fichiers identiques à l'octet entre deux runs ({})", tous.len(), app.backend().nom()))
    });
    gate("G-DIR-LOT", || {
        let a = base.join("a");
        for (c, ch) in tous.iter().enumerate() {
            let seul = base.join(format!("seul{c}"));
            executer_boite(Some(&app), &config(6, &[*ch], &seul), &[]).map_err(e)?;
            for f in ["mesures.csv", "profil.csv", "config_finale.npy"] {
                if std::fs::read(a.join(format!("c{c:03}/{f}"))).map_err(e)? != std::fs::read(seul.join(format!("c000/{f}"))).map_err(e)? {
                    return Err(format!("chaîne {c} ({}) : {f} du lot ≠ chaîne seule", ch.0));
                }
            }
        }
        Ok(format!("{} chaînes du lot identiques à la chaîne seule ({})", tous.len(), app.backend().nom()))
    });
}

#[test]
#[ignore = "GPU"]
fn g_dir_f32_refus() {
    let Some(app) = appareil_ou_skip("G-DIR-F32-REFUS") else { return };
    gate("G-DIR-F32-REFUS", || {
        let cfg = ConfigLot {
            l: 4,
            chaines: vec![ParamChaine { beta: 2.5, graine: Graine(1), depart: Depart::Froid }],
            nor: 2,
            precision: Precision::F32,
            options: OptionsCompil::rapide(),
            mutant_metro: false,
        };
        let mut f = kvu::su2::rapide32::Rapide32Gpu::nouveau(&app, cfg).map_err(e)?;
        f.fixer_masque(None, AppMasque::Tout).map_err(|x| format!("fixer_masque(None) refusé : {x}"))?;
        if f.fixer_masque(Some(&vec![0u8; 4 * 256]), AppMasque::Tout).is_ok() {
            return Err("le générateur F32 a accepté un masque sans l'appliquer".into());
        }
        Ok("le générateur F32 refuse un masque (erreur), accepte l'absence de masque".into())
    });
}
