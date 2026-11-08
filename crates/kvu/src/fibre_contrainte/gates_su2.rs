//! Gates de la route B sur le réseau SU(2)^N : FIB-CONTR, FIB-REV, FIB-ORDRE (+ mutant), FIB-CREUTZ (+ mutant) et outil
//! d'exploration (`explo_su2`, hors verdict). Lourdes : `#[ignore]`.
//! Paramètres par variables d'environnement `FIBC_*` (valeurs de verdict : défauts de PREVOL.md ; l'exploration les surcharge).

use super::alea::Flux;
use super::gates_commun::*;
use super::jacobien::Reseau;
use super::modele_su2::*;
use super::rattle::*;
use super::stat::*;
use crate::commun::ChampHote;
use std::sync::Arc;
use std::time::Instant;

const S_MOY_M1: f64 = SEUILS.moy_m1;

fn env_f(nom: &str, d: f64) -> f64 {
    std::env::var(nom).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}
fn env_u(nom: &str, d: usize) -> usize {
    std::env::var(nom).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

/// Modèle et point de départ : U₀ de Wilson (graine, 2000 pas), V = Ū(U₀), départ sur la fibre = U₀.
pub(crate) fn construire(l: usize, beta: f64, graine_u0: u64, fils: usize) -> (FibreSu2, ChampHote) {
    let reseau = Arc::new(Reseau::nouveau(l).unwrap());
    let u0 = champ_wilson(l, beta, graine_u0, 2000);
    let v = cible_depuis(&u0, &reseau);
    ligne("INFO", "EMPREINTE", &format!("U0 de Wilson L = {l}, β = {beta}, graine {graine_u0}, 2000 pas : sha256(octets) {} ; V sha256 {}", empreinte(&u0), empreinte_quaternions(&v)));
    (FibreSu2::nouveau(reseau, v, beta, fils).unwrap(), u0)
}

#[test]
#[ignore = "exploration"]
fn explo_su2() {
    let l = env_u("FIBC_L", 6);
    let beta = env_f("FIBC_BETA", 2.8);
    let dt = env_f("FIBC_DT", 0.02);
    let npas = env_usize("FIBC_NPAS");
    let ntraj = env_u("FIBC_NTRAJ", 20);
    let fils = env_u("FIBC_FILS", 8);
    let (m, u0) = construire(l, beta, 26_102_612_800 + l as u64, fils);
    let params = ParamsRattle { dt, nb_pas: npas, ..ParamsRattle::default() };
    let t0 = Instant::now();
    let ev = m.evaluer(&u0).unwrap();
    println!("INFO evaluation seule : {:.3} s ; V = {:.6}", t0.elapsed().as_secs_f64(), m.potentiel(&ev));
    let t1 = Instant::now();
    let (mut acc, mut sdh, mut nrej) = (0usize, 0.0, 0usize);
    chaine(&m, u0, &params, &Flux::nouveau(26_102_612_811), 0, ntraj, |i, _, s| {
        if s.accepte {
            acc += 1;
        }
        if s.echec.is_some() {
            nrej += 1;
        } else {
            sdh += s.delta_h;
        }
        println!("INFO traj {i} : dH = {:+.4e} accepte = {} echec = {:?} newton = {} residu = {:.2e}", s.delta_h, s.accepte, s.echec, s.nb_newton, s.residu_max);
    })
    .unwrap();
    let dt_tot = t1.elapsed().as_secs_f64();
    println!("INFO bilan : {ntraj} trajectoires de {npas} pas en {dt_tot:.1} s ({:.2} s/pas), acceptation {:.2}, <dH> = {:.3e}, échecs {nrej}", dt_tot / (ntraj * npas) as f64, acc as f64 / ntraj as f64, sdh / (ntraj - nrej).max(1) as f64);
}

fn env_usize(nom: &str) -> usize {
    env_u(nom, if nom == "FIBC_NPAS_REV" { 20 } else { 10 })
}

// ---------------------------------------------------------------------------------------------------------------- gates

fn params_prod(dt: f64, npas: usize) -> ParamsRattle {
    ParamsRattle { dt, nb_pas: npas, ..ParamsRattle::default() }
}

fn nrm_inf(a: &ChampHote, b: &ChampHote) -> f64 {
    a.q.iter().zip(&b.q).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// États équilibrés : `n_therm` propositions écartées, puis un état tous les `esp` propositions. Rend aussi le taux d'acceptation et le
/// nombre d'échecs de calcul de toute la chaîne (diagnostic d'équilibrage).
fn etats_equilibres(m: &FibreSu2, u0: ChampHote, params: &ParamsRattle, graine: u64, n_therm: usize, n_etats: usize, esp: usize) -> (Vec<ChampHote>, f64, usize) {
    let mut out = Vec::new();
    let (mut acc, mut echecs, mut total) = (0usize, 0usize, 0usize);
    chaine(m, u0, params, &Flux::nouveau(graine), 0, n_therm + n_etats * esp, |i, q, s| {
        total += 1;
        acc += usize::from(s.accepte);
        echecs += usize::from(s.echec.is_some());
        if i >= n_therm && (i + 1 - n_therm).is_multiple_of(esp) {
            out.push(q.clone());
        }
    })
    .unwrap();
    (out, acc as f64 / total as f64, echecs)
}

#[test]
#[ignore = "gate lourde FIB-CONTR"]
fn fib_contr() {
    let fils = env_u("FIBC_FILS", 8);
    // (L, β, trajectoires, dt, nb_pas, graine)
    let cas = [
        (6usize, 2.8f64, env_u("FIBC_NTRAJ", 1000), env_f("FIBC_DT", 0.01), env_u("FIBC_NPAS_CONTR", 2), 26_102_612_821u64),
        (6, 8.0, env_u("FIBC_NTRAJ", 1000), env_f("FIBC_DT8", 0.02), env_u("FIBC_NPAS8", 2), 26_102_612_822),
        (12, 8.0, env_u("FIBC_NTRAJ12", 1000), env_f("FIBC_DT12", 0.02), env_u("FIBC_NPAS12", 2), 26_102_612_823),
    ];
    let seuil = 1e-10;
    let mut fail = false;
    for (l, beta, ntraj, dt, npas, graine) in cas {
        if l == 12 && std::env::var("FIBC_SANS12").is_ok() {
            continue;
        }
        let (m, u0) = construire(l, beta, graine, fils);
        let (mut max_acc, mut max_newton, mut nrej, mut nacc) = (0.0f64, 0.0f64, 0usize, 0usize);
        let t0 = Instant::now();
        chaine(&m, u0, &params_prod(dt, npas), &Flux::nouveau(graine + 100), 0, ntraj, |i, q, s| {
            max_newton = maxn(max_newton, s.residu_max);
            if s.echec.is_some() {
                nrej += 1;
            } else if s.accepte {
                nacc += 1;
                // contrôle indépendant du pas accepté : résidu du champ rendu, par le bloc complet
                max_acc = maxn(max_acc, m.residu_max(q).unwrap());
            }
            if i % 100 == 0 {
                println!("INFO FIB-CONTR L = {l} β = {beta} : trajectoire {i}, rejets de calcul {nrej}, {:.0} s", t0.elapsed().as_secs_f64());
            }
        })
        .unwrap();
        let frac = nrej as f64 / ntraj as f64;
        ligne("INFO", "FIB-CONTR", &format!("L = {l}, β = {beta}, dt = {dt}, {npas} pas : {ntraj} trajectoires, acceptées {nacc}, rejets de calcul {nrej} ({:.2} %), max résidu de Newton {max_newton:.2e}, max résidu des états acceptés {max_acc:.2e}", 100.0 * frac));
        if frac > 0.01 || nacc == 0 {
            ligne("SKIP", "FIB-CONTR", &format!("PUISSANCE : rejets de calcul {:.2} % (> 1 %) ou aucun état accepté ({nacc}) (L = {l}, β = {beta})", 100.0 * frac));
            continue;
        }
        let ok = max_acc <= seuil && max_newton <= seuil;
        fail |= !ok;
        ligne(if ok { "PASS" } else { "FAIL" }, "FIB-CONTR", &format!("L = {l}, β = {beta} : max |ω(V⁻¹Ū)| = {:.2e} ≤ {seuil:e} sur {nacc} états acceptés (et résidus de Newton {max_newton:.2e})", maxn(max_acc, max_newton)));
    }
    assert!(!fail);
}

#[test]
#[ignore = "gate lourde FIB-REV"]
fn fib_rev() {
    let fils = env_u("FIBC_FILS", 8);
    let dt = env_f("FIBC_DT", 0.01);
    let npas = env_usize("FIBC_NPAS_REV");
    assert!(npas >= 1);
    let (m, u0) = construire(6, 2.8, 26_102_612_831, fils);
    let n_therm = env_u("FIBC_NTHERM", 40);
    let (etats, acc, ech) = etats_equilibres(&m, u0, &params_prod(dt, env_u("FIBC_NPAS", 10)), 26_102_612_832, n_therm, 30, env_u("FIBC_ESP", 5));
    ligne("INFO", "FIB-REV", &format!("chaîne d'équilibrage ({} propositions, trajectoires de {} pas, dt = {dt}) : acceptation {acc:.3}, échecs de calcul {ech}", n_therm + 30 * env_u("FIBC_ESP", 5), env_u("FIBC_NPAS", 10)));
    let params = params_prod(dt, npas);
    let (mut dq, mut dp, mut dh) = (0.0f64, 0.0f64, 0.0f64);
    let mut dhs = Vec::new();
    let mut remplaces = 0usize;
    for (k, q0) in etats.iter().enumerate() {
        if dhs.len() == 10 {
            break;
        }
        let ev0 = m.evaluer(q0).unwrap();
        let p0 = tirer_impulsion(&m, &ev0, &Flux::nouveau(26_102_612_833), 0, k as u32);
        let h0 = hamiltonien(&m, &ev0, &p0);
        // seule l'absence de solution de Newton remplace le départ (comptée) ; toute autre erreur est un échec de la gate
        let (q1, ev1, p1, _) = match integrer(&m, q0.clone(), m.evaluer(q0).unwrap(), p0.clone(), &params) {
            Ok(r) => r,
            Err(Echec::Newton) => {
                remplaces += 1;
                continue;
            }
            Err(e) => panic!("FIB-REV : erreur {e:?} autre que Newton au départ {k}"),
        };
        dhs.push(hamiltonien(&m, &ev1, &p1) - h0);
        let pf: Vec<f64> = p1.iter().map(|x| -x).collect();
        let (q2, ev2, p2, _) = integrer(&m, q1, ev1, pf, &params).expect("le retour d'une trajectoire existante doit exister");
        dq = maxn(dq, nrm_inf(&q2, q0));
        dp = maxn(dp, p2.iter().zip(&p0).map(|(a, b)| (a + b).abs()).fold(0.0, f64::max));
        dh = maxn(dh, (hamiltonien(&m, &ev2, &p2) - h0).abs());
    }
    ligne("INFO", "FIB-REV", &format!("{remplaces} départs remplacés (pas de solution de Newton sur la trajectoire aller)"));
    if dhs.len() < 10 {
        ligne("SKIP", "FIB-REV", &format!("PUISSANCE : {} départs valides sur 10 requis", dhs.len()));
        return;
    }
    ligne("INFO", "FIB-REV", &format!("ΔH aller sur {} départs : {:?}", dhs.len(), dhs.iter().map(|x| format!("{x:.3e}")).collect::<Vec<_>>()));
    let ok = dq <= 1e-10 && dp <= 1e-10 && dh <= 1e-10;
    ligne(if ok { "PASS" } else { "FAIL" }, "FIB-REV", &format!("{npas} pas, inversion, {npas} pas : max|Δq| = {dq:.2e}, max|Δp| = {dp:.2e}, |H_fin − H_début| = {dh:.2e} (≤ 1e-10 chacun)"));
    assert!(ok);
}

fn pente(dts: &[f64], vals: &[f64]) -> f64 {
    let (x, y): (Vec<f64>, Vec<f64>) = dts.iter().zip(vals).map(|(d, v)| (d.ln(), v.ln())).unzip();
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / x.iter().map(|a| (a - mx) * (a - mx)).sum::<f64>()
}

#[test]
#[ignore = "gate lourde FIB-ORDRE"]
fn fib_ordre() {
    let fils = env_u("FIBC_FILS", 8);
    let tau = env_f("FIBC_TAU", 0.08);
    let beta_o = env_f("FIBC_BETA_ORDRE", 8.0);
    let (m, u0) = construire(6, beta_o, 26_102_612_841, fils);
    let ndep = env_u("FIBC_NDEP", 50);
    let (etats, acc, ech) = etats_equilibres(&m, u0, &params_prod(env_f("FIBC_DT", 0.01), env_u("FIBC_NPAS", 10)), 26_102_612_842, env_u("FIBC_NTHERM", 40), ndep, env_u("FIBC_ESP", 3));
    ligne("INFO", "FIB-ORDRE", &format!("chaîne d'équilibrage : acceptation {acc:.3}, échecs de calcul {ech}"));
    let diviseurs = [8usize, 16, 32, 64];
    let dts: Vec<f64> = diviseurs.iter().map(|d| tau / *d as f64).collect();
    let integre = |q0: &ChampHote, k: usize, d: usize, euler: bool| -> Option<f64> {
        let ev0 = m.evaluer(q0).unwrap();
        let p0 = tirer_impulsion(&m, &ev0, &Flux::nouveau(26_102_612_843), 0, k as u32);
        let mut params = params_prod(tau / d as f64, d);
        if euler {
            params.integrateur = Integrateur::EulerSymplectique;
            params.controle_reversibilite = false;
        }
        let h0 = hamiltonien(&m, &ev0, &p0);
        match integrer(&m, q0.clone(), ev0, p0, &params) {
            Ok((_, ev1, p1, _)) => Some(hamiltonien(&m, &ev1, &p1) - h0),
            Err(Echec::Newton) => None,
            Err(e) => panic!("FIB-ORDRE : erreur {e:?} autre que Newton"),
        }
    };
    // sous-ensemble COMMUN à l'intégrateur nominal et au mutant : les départs dont toutes les trajectoires existent (Newton), comptés
    let mut table: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut ecartes = 0usize;
    for (k, q0) in etats.iter().enumerate() {
        let nom: Option<Vec<f64>> = diviseurs.iter().map(|&d| integre(q0, k, d, false)).collect();
        let mu: Option<Vec<f64>> = diviseurs.iter().map(|&d| integre(q0, k, d, true)).collect();
        match (nom, mu) {
            (Some(a), Some(b)) => table.push((a, b)),
            _ => ecartes += 1,
        }
    }
    ligne("INFO", "FIB-ORDRE", &format!("{} départs retenus (communs aux deux intégrateurs), {ecartes} écartés (pas de solution de Newton), sur {ndep}", table.len()));
    if table.len() < 40 {
        ligne("SKIP", "FIB-ORDRE", &format!("PUISSANCE : {} départs valides < 40", table.len()));
        return;
    }
    if std::env::var("FIBC_DETAIL").is_ok() {
        for (a, b) in &table {
            ligne("INFO", "FIB-ORDRE-DETAIL", &format!("Verlet {:?} ; Euler {:?}", a.iter().map(|x| format!("{x:.2e}")).collect::<Vec<_>>(), b.iter().map(|x| format!("{x:.2e}")).collect::<Vec<_>>()));
        }
    }
    let mut ok_tout = true;
    for (nom, euler) in [("Verlet contraint", false), ("mutant Euler symplectique contraint", true)] {
        let rms: Vec<f64> = (0..diviseurs.len())
            .map(|i| (table.iter().map(|l| if euler { l.1[i] } else { l.0[i] }).map(|x| x * x).sum::<f64>() / table.len() as f64).sqrt())
            .collect();
        for (d, r) in diviseurs.iter().zip(&rms) {
            ligne("INFO", "FIB-ORDRE", &format!("{nom} : dt = τ/{d} : RMS(ΔH) = {r:.4e} ({} départs)", table.len()));
        }
        let p = pente(&dts, &rms);
        if euler {
            let ok = (0.8..=1.2).contains(&p);
            ok_tout &= ok;
            ligne(if ok { "PASS" } else { "FAIL" }, "FIB-ORDRE-MUTANT", &format!("Euler symplectique contraint : pente {p:.3} ∈ [0,8 ; 1,2] (ROUGE exigé sur la gate ; sinon ECHEC_OUTIL)"));
        } else {
            let ok = (1.8..=2.2).contains(&p);
            ok_tout &= ok;
            ligne(if ok { "PASS" } else { "FAIL" }, "FIB-ORDRE", &format!("β = {beta_o}, pente de log RMS(ΔH) contre log dt à durée τ = {tau} : {p:.3} ∈ [1,8 ; 2,2]"));
        }
    }
    assert!(ok_tout);
}

#[test]
#[ignore = "gate lourde FIB-CREUTZ"]
fn fib_creutz() {
    let fils = env_u("FIBC_FILS", 8);
    let dt = env_f("FIBC_DT", 0.01);
    let npas = env_u("FIBC_NPAS", 10);
    let ntraj = env_u("FIBC_NTRAJ_CREUTZ", 4000);
    let (m, u0) = construire(6, 2.8, 26_102_612_851, fils);
    let mut ok_tout = true;
    for (nom, euler) in [("Verlet contraint", false), ("mutant Euler symplectique contraint", true)] {
        let mut params = params_prod(dt, npas);
        if euler {
            params.integrateur = Integrateur::EulerSymplectique;
            params.controle_reversibilite = false;
        }
        // thermalisation : les 20 % premières PROPOSITIONS sont écartées (indices), avant TOUT diagnostic ; ensuite série de e^{−ΔH} sur les
        // trajectoires dont ΔH est défini (une involution qui préserve la mesure, restreinte à son domaine D = Φ(D), donne E[e^{−ΔH} | D] = 1) ;
        // les échecs de calcul de l'ensemble retenu sont comptés et rapportés par cause
        let coupure = ntraj / 5;
        let mut serie = Vec::new();
        let (mut dh_somme, mut nfail) = (0.0f64, 0usize);
        let (mut e_rev, mut e_ev, mut e_new) = (0usize, 0usize, 0usize);
        let t0 = Instant::now();
        chaine(&m, u0.clone(), &params, &Flux::nouveau(26_102_612_852), 0, ntraj, |i, _, s| {
            if i >= coupure {
                match s.echec {
                    Some(e) => {
                        nfail += 1;
                        match e {
                            Echec::Newton => e_new += 1,
                            Echec::Reversibilite => e_rev += 1,
                            Echec::Evaluation => e_ev += 1,
                        }
                    }
                    None => {
                        dh_somme += s.delta_h;
                        serie.push((-s.delta_h).exp());
                    }
                }
            }
            if i % 500 == 0 {
                println!("INFO FIB-CREUTZ {nom} : {i} trajectoires, {:.0} s", t0.elapsed().as_secs_f64());
            }
        })
        .unwrap();
        let retenues = ntraj - coupure;
        let a = analyser_avec(&serie, 0.0);
        let mdh = dh_somme / serie.len().max(1) as f64;
        ligne("INFO", "FIB-CREUTZ", &format!("{nom} : ⟨e^(−ΔH)⟩ = {:.5} ± {:.2e}, τ_int = {:.2}, statut {}, ⟨ΔH⟩ = {mdh:.3}, échecs de calcul {nfail}/{retenues} (Newton {e_new}, réversibilité {e_rev}, évaluation {e_ev}) sur les {retenues} propositions retenues après 20 % de thermalisation", a.moyenne, a.erreur, a.tau_int, a.statut.texte()));
        if !domaine_campagne(&a) || (mdh > 0.5 && !euler) || nfail as f64 > 0.05 * retenues as f64 {
            ligne("SKIP", "FIB-CREUTZ", &format!("PUISSANCE : statut {}, ⟨ΔH⟩ = {mdh:.3}, échecs {nfail}/{retenues}", a.statut.texte()));
            continue;
        }
        let t = (a.moyenne - 1.0) / a.erreur;
        if euler {
            let ok = t.abs() >= S_MOY_M1;
            ok_tout &= ok;
            ligne(if ok { "PASS" } else { "FAIL" }, "FIB-CREUTZ-MUTANT", &format!("Euler (non réversible, hors involution) : |⟨e^(−ΔH)⟩ − 1|/σ = {:.2} ≥ {S_MOY_M1} (ROUGE exigé ; sinon ECHEC_OUTIL)", t.abs()));
        } else {
            let ok = t.abs() < S_MOY_M1;
            ok_tout &= ok;
            ligne(if ok { "PASS" } else { "FAIL" }, "FIB-CREUTZ", &format!("|⟨e^(−ΔH)⟩ − 1|/σ = {:.2} < {S_MOY_M1} (moyenne sur les {} trajectoires dont ΔH est défini parmi les {retenues} retenues ; {nfail} échecs de calcul)", t.abs(), serie.len()));
        }
    }
    assert!(ok_tout);
}
