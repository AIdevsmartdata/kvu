//! Gates `rg flux-gates` : lignes `PASS|FAIL|SKIP|INFO <ID> <détail>`.
//! Retourne le code 0 si toutes les lignes sont PASS/INFO (pas de FAIL ni SKIP)
//! selon `crate::groupes::gates::tout_vert`.

use super::faces::{
    angle_face, boucle_face, flux_coherent_faces, fond_abelien, grossier, stats_coherent,
    stats_faces, x_faces, Pavage, StatsCoherent, StatsFaces, VarianteFlux,
};
use super::{COH_MIN, SEUILS_COH, SEUILS_X};
use crate::commun::{code, ChampHote, Erreur, Resultat};
use crate::groupes::gates::{tout_vert, LigneGate, Verdict};
use crate::rg::operateur::Operateur;
use crate::stockage::npy::lire_champ;
use clap::ArgMatches;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Dossier du crate (crates/kvu) à la compilation : les fixtures
/// (`tests/donnees/…`, `validation/…`) y sont rangées.
fn racine_depot() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn chemin_depot(rel: &str) -> PathBuf {
    racine_depot().join(rel)
}

fn proche(a: f64, b: f64, tolerance: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance
}

/// Contrairement à une réduction par f64::max, ne masque jamais un NaN.
fn ecart_max_fini(a: &[f64], b: &[f64]) -> Resultat<f64> {
    if a.len() != b.len() {
        return Err(Erreur::Format(
            "comparaison de tableaux de tailles différentes".into(),
        ));
    }
    let mut maximum = 0.0_f64;
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let ecart = (a - b).abs();
        if !a.is_finite() || !b.is_finite() || !ecart.is_finite() {
            return Err(Erreur::Format(format!(
                "comparaison non finie à l'indice {i} : {a}, {b}"
            )));
        }
        maximum = maximum.max(ecart);
    }
    Ok(maximum)
}

/// Exécute les gates définies par `--seul`, etc.
pub fn executer_gates(m: &ArgMatches) -> Resultat<i32> {
    let desert = m.get_one::<String>("desert").cloned();
    let max_configs = m
        .get_one::<String>("max-configs")
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| Erreur::Config(format!("--max-configs « {s} » invalide")))
        })
        .transpose()?;
    let seul: Option<HashSet<String>> = m.get_one::<String>("seul").map(|s| {
        s.split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect()
    });
    let sans_mutants = m.get_flag("sans-mutants");
    let json = m.get_flag("json");
    let fils = super::cli::nombre_fils(m)?;

    let mut ids: Vec<&str> = vec![
        "RG-FLUX-ABELIEN",
        "RG-FLUX-FROID",
        "RG-FLUX-JAUGE",
        "RG-FLUX-REF-PY",
        "RG-FLUX-GROSSIER",
        "G-REPRO",
    ];
    if !sans_mutants {
        ids.push("RG-FLUX-MUT");
    }

    if let Some(seul) = &seul {
        ids.retain(|id| seul.iter().any(|s| id.starts_with(s.as_str())));
    }

    let mut lignes = Vec::new();
    for id in ids {
        let ligne = match id {
            "RG-FLUX-ABELIEN" => gate_abelien(fils),
            "RG-FLUX-FROID" => gate_froid(),
            "RG-FLUX-JAUGE" => gate_jauge(fils),
            "RG-FLUX-REF-PY" => gate_ref_py(fils),
            "RG-FLUX-GROSSIER" => gate_grossier(fils),
            "G-REPRO" => gate_repro(desert.as_deref(), max_configs, fils),
            "RG-FLUX-MUT" => gate_mut(fils),
            _ => unreachable!(),
        };
        let ligne = match ligne {
            Ok(l) => l,
            Err(e) => LigneGate::nouvelle(Verdict::Fail, id, format!("ERREUR : {e}")),
        };
        lignes.push(ligne);
    }

    if json {
        let doc: Vec<Value> = lignes
            .iter()
            .map(|l| {
                json!({
                    "verdict": l.verdict.to_string(),
                    "id": l.id,
                    "detail": l.detail,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&doc).map_err(|e| Erreur::Format(e.to_string()))?
        );
    } else {
        for ligne in &lignes {
            println!("{ligne}");
        }
    }

    Ok(if tout_vert(&lignes) {
        code::OK
    } else {
        code::ECHEC
    })
}

/// Vérifie les valeurs par face d'un fond abélien : x et Θ_atan.
fn verifier_faces_abelien(
    champ: &ChampHote,
    phi: f64,
    n: usize,
    plan_actif: (usize, usize),
    opaque: &mut Option<String>,
) -> bool {
    let Some(p_actif) = super::PLANS.iter().position(|&plan| plan == plan_actif) else {
        *opaque = Some(format!("plan actif {plan_actif:?} absent de PLANS"));
        return false;
    };
    let v = champ.v();
    let phi_n2 = (n * n) as f64 * phi;
    let repli = |t: f64| {
        if t <= 2.0 * std::f64::consts::PI {
            t
        } else {
            4.0 * std::f64::consts::PI - t
        }
    };
    let mut ok = true;
    for p in 0..6 {
        for s in 0..v {
            let w = boucle_face(champ, s, p, n);
            let x_att = if p == p_actif {
                (phi_n2 / 2.0).cos()
            } else {
                1.0
            };
            let tol = if p == p_actif { 1e-12 } else { 1e-15 };
            if !proche(w[0], x_att, tol) {
                if let Some(msg) = opaque {
                    *msg = format!(
                        "face {p}/{s} n={n} : x={} attendu {x_att} (tol {tol})",
                        w[0]
                    );
                }
                ok = false;
                break;
            }
            let angle = angle_face(w);
            let angle_att = if p == p_actif { repli(phi_n2) } else { 0.0 };
            if !proche(angle, angle_att, 1e-12) {
                if let Some(msg) = opaque {
                    *msg = format!("face {p}/{s} n={n} : Θ_atan={angle} attendu {angle_att}");
                }
                ok = false;
                break;
            }
        }
        if !ok {
            break;
        }
    }
    ok
}

fn verifier_flux_abelien(
    champ: &ChampHote,
    phi: f64,
    plan_actif: (usize, usize),
    opaque: &mut Option<String>,
) -> bool {
    let Some(p_actif) = super::PLANS.iter().position(|&plan| plan == plan_actif) else {
        *opaque = Some(format!("plan actif {plan_actif:?} absent de PLANS"));
        return false;
    };
    let v = champ.v();
    let phi_coh_att = 9.0 * phi;
    let (phi_vals, sabs_vals, _) = match flux_coherent_faces(champ, 3, VarianteFlux::Normale, 1) {
        Ok(v) => v,
        Err(e) => {
            if let Some(msg) = opaque {
                *msg = format!("erreur flux : {e}");
            }
            return false;
        }
    };
    for p in 0..6 {
        for s in 0..v {
            let idx = p * v + s;
            let att = if p == p_actif { phi_coh_att } else { 0.0 };
            let tol = if p == p_actif { 1e-11 } else { 1e-14 };
            if !proche(phi_vals[idx], att, tol) {
                if let Some(msg) = opaque {
                    *msg = format!(
                        "face {p}/{s} cohérent : Φ={} attendu {att} (tol {tol})",
                        phi_vals[idx]
                    );
                }
                return false;
            }
            if p == p_actif {
                let coh = phi_vals[idx] / sabs_vals[idx];
                if !proche(coh, 1.0, 1e-12) {
                    if let Some(msg) = opaque {
                        *msg = format!("face {p}/{s} cohérent : C={coh} attendu 1");
                    }
                    return false;
                }
            }
        }
    }
    true
}

/// RG-FLUX-ABELIEN : comptes exacts et valeurs par face des fonds abéliens (§5 de la spec).
fn gate_abelien(fils: usize) -> Resultat<LigneGate> {
    let cas_plan01 = [
        (12, 1, (0, 1)),
        (12, 2, (0, 1)),
        (12, 3, (0, 1)),
        (12, 5, (0, 1)),
        (18, 14, (0, 1)),
        (18, 16, (0, 1)),
        (18, 17, (0, 1)),
        (18, 19, (0, 1)),
    ];
    let cas_extras = [
        (12, 1, (1, 3)),
        (12, 2, (1, 3)),
        (12, 3, (1, 3)),
        (12, 5, (1, 3)),
    ];

    for (l, m, plan) in cas_plan01.iter().chain(cas_extras.iter()) {
        let (champ, phi) = fond_abelien(*l, *m, *plan)?;
        // Vérification par face pour n=1,2,3
        let mut msg = None;
        for n in 1..=3 {
            if !verifier_faces_abelien(&champ, phi, n, *plan, &mut msg) {
                return Ok(LigneGate::nouvelle(
                    Verdict::Fail,
                    "RG-FLUX-ABELIEN",
                    format!(
                        "L={l} m={m} plan={plan:?} n={n} : {}",
                        msg.unwrap_or_default()
                    ),
                ));
            }
        }
        if !verifier_flux_abelien(&champ, phi, *plan, &mut msg) {
            return Ok(LigneGate::nouvelle(
                Verdict::Fail,
                "RG-FLUX-ABELIEN",
                format!("L={l} m={m} plan={plan:?} : {}", msg.unwrap_or_default()),
            ));
        }

        // Vérification des comptes exacts
        for n in 1..=3 {
            let stats = stats_faces(&champ, n, Pavage::Tous, false, fils)?;
            let x_attendu = (n * n) as f64 * phi / 2.0;
            let faces_plan = champ.v() as u64;
            let mut comptes_att = [0u64; 5];
            for (i, (_, seuil, strict)) in SEUILS_X.iter().enumerate() {
                let cond = if *strict {
                    x_attendu.cos() < *seuil
                } else {
                    x_attendu.cos() <= *seuil
                };
                if cond {
                    comptes_att[i] = faces_plan;
                }
            }
            if stats.comptes != comptes_att {
                return Ok(LigneGate::nouvelle(
                    Verdict::Fail,
                    "RG-FLUX-ABELIEN",
                    format!(
                        "L={l} m={m} plan={plan:?} n={n}: comptes {comptes_att:?} attendus, obtenus {:?}",
                        stats.comptes
                    ),
                ));
            }
        }

        // Cohérent n=3, comptes exacts
        let stats_coh = stats_coherent(&champ, 3, Pavage::Tous, VarianteFlux::Normale, fils)?;
        let phi_attendu = 9.0 * phi;
        let faces_plan = champ.v() as u64;
        let mut comptes_coh_att = [0u64; 4];
        for (j, (_, seuil, strict)) in SEUILS_COH.iter().enumerate() {
            let cond = if *strict {
                phi_attendu > *seuil
            } else {
                phi_attendu >= *seuil
            };
            if cond {
                comptes_coh_att[j] = faces_plan;
            }
        }
        if stats_coh.comptes != comptes_coh_att {
            return Ok(LigneGate::nouvelle(
                Verdict::Fail,
                "RG-FLUX-ABELIEN",
                format!(
                    "L={l} m={m} plan={plan:?} cohérent: comptes {comptes_coh_att:?} attendus, obtenus {:?}",
                    stats_coh.comptes
                ),
            ));
        }

        // Grossier 3centre K=1, comptes exacts et valeurs par face
        if *l % 3 == 0 {
            let op = Operateur::TroisCentre;
            let champ_gr = grossier(&champ, &op, 1, fils)?;
            let p_actif_gr = super::PLANS
                .iter()
                .position(|p| p == plan)
                .ok_or_else(|| Erreur::Config(format!("plan actif {plan:?} absent de PLANS")))?;
            let v_gr = champ_gr.v();
            for p in 0..6 {
                for s in 0..v_gr {
                    let w = boucle_face(&champ_gr, s, p, 1);
                    let x_att = if p == p_actif_gr {
                        (9.0 * phi / 2.0).cos()
                    } else {
                        1.0
                    };
                    // Intégrateur (M3 r2) : r2 écrivait la même tolérance 1e-12 dans les deux branches (clippy) ; conservée telle quelle.
                    let tol = 1e-12;
                    if !proche(w[0], x_att, tol) {
                        return Ok(LigneGate::nouvelle(
                            Verdict::Fail,
                            "RG-FLUX-ABELIEN",
                            format!(
                                "L={l} m={m} grossier : face {p}/{s} x={} attendu {x_att}",
                                w[0]
                            ),
                        ));
                    }
                }
            }
            let stats_gr = stats_faces(&champ_gr, 1, Pavage::Tous, false, fils)?;
            let x_gr_attendu = (9.0 * phi / 2.0).cos();
            let faces_gr_plan = champ_gr.v() as u64;
            let mut comptes_gr_att = [0u64; 5];
            for (i, (_, seuil, strict)) in SEUILS_X.iter().enumerate() {
                let cond = if *strict {
                    x_gr_attendu < *seuil
                } else {
                    x_gr_attendu <= *seuil
                };
                if cond {
                    comptes_gr_att[i] = faces_gr_plan;
                }
            }
            if stats_gr.comptes != comptes_gr_att {
                return Ok(LigneGate::nouvelle(
                    Verdict::Fail,
                    "RG-FLUX-ABELIEN",
                    format!(
                        "L={l} m={m} grossier: comptes {comptes_gr_att:?} attendus, obtenus {:?}",
                        stats_gr.comptes
                    ),
                ));
            }
        }
    }

    Ok(LigneGate::nouvelle(
        Verdict::Pass,
        "RG-FLUX-ABELIEN",
        "tous les cas du §5 (valeurs par face, comptes exacts, grossier) conformes",
    ))
}

/// RG-FLUX-FROID : champ froid L=6, x=1 partout, comptes nuls, Φ_coh=0, C=1.
fn gate_froid() -> Resultat<LigneGate> {
    let champ = ChampHote::froid(6)?;
    let x = x_faces(&champ, 3, 1)?;
    for (i, &xi) in x.iter().enumerate() {
        if !proche(xi, 1.0, 1e-15) {
            return Ok(LigneGate::nouvelle(
                Verdict::Fail,
                "RG-FLUX-FROID",
                format!("face {i} : x={xi} attendu 1"),
            ));
        }
    }
    let stats_brut = stats_faces(&champ, 3, Pavage::Tous, false, 1)?;
    let stats_coh = stats_coherent(&champ, 3, Pavage::Tous, VarianteFlux::Normale, 1)?;
    let comptes_ok = stats_brut.comptes == [0; 5];
    let coh_ok = stats_coh.s_phi == 0.0 && stats_coh.s_coh == stats_coh.n as f64;
    if comptes_ok && coh_ok {
        Ok(LigneGate::nouvelle(
            Verdict::Pass,
            "RG-FLUX-FROID",
            "x = 1 sur chaque face, comptes nuls, Φ_coh = 0, C = 1",
        ))
    } else {
        Ok(LigneGate::nouvelle(
            Verdict::Fail,
            "RG-FLUX-FROID",
            format!("comptes_ok={comptes_ok} coh_ok={coh_ok}"),
        ))
    }
}

/// Construit un champ de Haar L×L×L×L avec une graine donnée.
fn champ_haar(l: usize, graine: u32) -> ChampHote {
    let v = l.pow(4);
    let mut q = vec![0.0; 16 * v];
    for i in 0..(4 * v) as u32 {
        let quat = crate::su2::cpu::haar_q(i, 0, 0, graine, graine.wrapping_add(0x9e37_79b9));
        let o = 4 * i as usize;
        q[o..o + 4].copy_from_slice(&quat);
    }
    ChampHote { l, q }
}

/// Applique une transformation de jauge aléatoire de Haar.
fn appliquer_jauge(champ: &ChampHote, graine: u32) -> ChampHote {
    let l = champ.l;
    let v = champ.v();
    let mut g = vec![[1.0_f64, 0.0, 0.0, 0.0]; v];
    for (s, gs) in g.iter_mut().enumerate() {
        *gs = crate::su2::cpu::haar_q(s as u32, 1, 1, graine, graine.wrapping_add(0x6a09_e667));
    }
    let mut q = champ.q.clone();
    for mu in 0..4 {
        for s in 0..v {
            let x = crate::rg::bloc::coordonnees(s, l);
            let x_mu = crate::rg::bloc::deplacer(x, mu, 1, l);
            let s_mu = crate::rg::bloc::site(x_mu, l);
            let u = crate::heritage::su2_hb::qmul(
                crate::heritage::su2_hb::qmul(g[s], champ.lien(mu * v + s)),
                crate::heritage::su2_hb::qdag(g[s_mu]),
            );
            let o = 4 * (mu * v + s);
            q[o..o + 4].copy_from_slice(&u);
        }
    }
    ChampHote { l, q }
}

/// RG-FLUX-JAUGE : invariance de jauge des comptes et du flux.
fn gate_jauge(fils: usize) -> Resultat<LigneGate> {
    let l = 6;
    let champ_base = champ_haar(l, 2611020301);
    let champ_jauge = appliquer_jauge(&champ_base, 2611020302);

    let mut max_dx = 0.0f64;
    for n in 1..=3 {
        let x_base = x_faces(&champ_base, n, fils)?;
        let x_jauge = x_faces(&champ_jauge, n, fils)?;
        max_dx = max_dx.max(ecart_max_fini(&x_base, &x_jauge)?);
        let stats_base = stats_faces(&champ_base, n, Pavage::Tous, false, fils)?;
        let stats_jauge = stats_faces(&champ_jauge, n, Pavage::Tous, false, fils)?;
        if stats_base.comptes != stats_jauge.comptes {
            return Ok(LigneGate::nouvelle(
                Verdict::Fail,
                "RG-FLUX-JAUGE",
                format!("n={n} : comptes différents"),
            ));
        }
    }

    let (phi_base, _, _) = flux_coherent_faces(&champ_base, 3, VarianteFlux::Normale, fils)?;
    let (phi_jauge, _, _) = flux_coherent_faces(&champ_jauge, 3, VarianteFlux::Normale, fils)?;
    let max_dphi = ecart_max_fini(&phi_base, &phi_jauge)?;
    let stats_coh_base = stats_coherent(&champ_base, 3, Pavage::Tous, VarianteFlux::Normale, fils)?;
    let stats_coh_jauge =
        stats_coherent(&champ_jauge, 3, Pavage::Tous, VarianteFlux::Normale, fils)?;
    if stats_coh_base.comptes != stats_coh_jauge.comptes {
        return Ok(LigneGate::nouvelle(
            Verdict::Fail,
            "RG-FLUX-JAUGE",
            "comptes cohérents différents",
        ));
    }

    if max_dx <= 1e-13 && max_dphi <= 1e-12 {
        Ok(LigneGate::nouvelle(
            Verdict::Pass,
            "RG-FLUX-JAUGE",
            format!("max|Δx| = {max_dx:.3e}, max|ΔΦ_coh| = {max_dphi:.3e}"),
        ))
    } else {
        Ok(LigneGate::nouvelle(
            Verdict::Fail,
            "RG-FLUX-JAUGE",
            format!("max|Δx| = {max_dx:.3e}, max|ΔΦ_coh| = {max_dphi:.3e}"),
        ))
    }
}

/// Charge le fichier `validation/rg_flux/reference_py.json`.
fn charger_reference_py() -> Resultat<Value> {
    let chemin = chemin_depot("validation/rg_flux/reference_py.json");
    let texte = std::fs::read_to_string(&chemin)
        .map_err(|e| Erreur::io(format!("lecture de {}", chemin.display()), e))?;
    serde_json::from_str(&texte).map_err(|e| Erreur::Format(e.to_string()))
}

/// Vérifie qu'un objet JSON contient les clés obligatoires et des nombres finis.
fn verifier_finitude_stats(stats: &Value, attendues: &[&str]) -> Resultat<()> {
    let obj = stats
        .as_object()
        .ok_or_else(|| Erreur::Format("stats non objet".into()))?;
    for cle in attendues {
        let val = obj
            .get(*cle)
            .ok_or_else(|| Erreur::Format(format!("clé {cle} absente")))?;
        if !val.as_f64().is_some_and(f64::is_finite) {
            return Err(Erreur::Format(format!(
                "clé {cle} non numérique ou non finie : {val}"
            )));
        }
    }
    Ok(())
}

/// Compare un StatsFaces calculé avec l'entrée JSON de référence (T4 strict).
fn comparer_stats_faces(calc: &StatsFaces, ref_json: &Value) -> Resultat<bool> {
    if ![calc.sx, calc.sx2, calc.s_theta2, calc.s_theta4, calc.xmin]
        .into_iter()
        .all(f64::is_finite)
    {
        return Ok(false);
    }
    let n_ref = ref_json["N"]
        .as_u64()
        .ok_or_else(|| Erreur::Format("N absent".into()))?;
    if calc.n != n_ref {
        return Ok(false);
    }
    verifier_finitude_stats(ref_json, &["N", "sx", "sx2", "sTheta2", "sTheta4", "xmin"])?;
    let comptes_ref = ref_json["comptes"]
        .as_object()
        .ok_or_else(|| Erreur::Format("comptes absent".into()))?;
    if comptes_ref.len() != SEUILS_X.len() {
        return Ok(false);
    }
    let mut ok = true;
    for (i, (nom, _, _)) in SEUILS_X.iter().enumerate() {
        let ref_val = comptes_ref
            .get(*nom)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Erreur::Format(format!("compte {nom} absent")))?;
        if calc.comptes[i] != ref_val {
            ok = false;
        }
    }
    // sommes relatives 1e-12, xmin absolu 1e-13 (arbitrage du chef de projet :
    // la tolérance reste 1e-13 à cause de l'écart résiduel de conversion c16 →
    // quaternion ; les comptes restent exigés identiques).
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-30);
    if rel(
        calc.sx,
        ref_json["sx"]
            .as_f64()
            .ok_or_else(|| Erreur::Format("sx absent".into()))?,
    ) > 1e-12
    {
        ok = false;
    }
    if rel(
        calc.sx2,
        ref_json["sx2"]
            .as_f64()
            .ok_or_else(|| Erreur::Format("sx2 absent".into()))?,
    ) > 1e-12
    {
        ok = false;
    }
    if rel(
        calc.s_theta2,
        ref_json["sTheta2"]
            .as_f64()
            .ok_or_else(|| Erreur::Format("sTheta2 absent".into()))?,
    ) > 1e-12
    {
        ok = false;
    }
    if rel(
        calc.s_theta4,
        ref_json["sTheta4"]
            .as_f64()
            .ok_or_else(|| Erreur::Format("sTheta4 absent".into()))?,
    ) > 1e-12
    {
        ok = false;
    }
    if (calc.xmin
        - ref_json["xmin"]
            .as_f64()
            .ok_or_else(|| Erreur::Format("xmin absent".into()))?)
    .abs()
        > 1e-13
    {
        ok = false;
    }
    // histogrammes non vides identiques, dans les DEUX sens
    if let Some(hist_ref) = ref_json
        .get("hist_alpha_nonvides")
        .and_then(|v| v.as_object())
    {
        if let Some(hist_calc) = &calc.hist_alpha {
            // 1) chaque classe non vide de la référence est identique
            for (i, count_ref) in hist_ref {
                let i: usize = i
                    .parse()
                    .map_err(|_| Erreur::Format("indice hist invalide".into()))?;
                let count_ref = count_ref
                    .as_u64()
                    .ok_or_else(|| Erreur::Format("compte hist non entier".into()))?;
                let count_calc = hist_calc.get(i).copied().unwrap_or(0);
                if count_calc != count_ref {
                    ok = false;
                }
            }
            // 2) aucune classe non vide en plus dans le calcul
            for (i, &count_calc) in hist_calc.iter().enumerate() {
                if count_calc > 0 {
                    let ref_count = hist_ref
                        .get(&i.to_string())
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    if ref_count != count_calc {
                        ok = false;
                    }
                }
            }
        } else {
            ok = false;
        }
    }
    Ok(ok)
}

/// Compare un StatsCoherent calculé avec l'entrée JSON de référence (T4 strict).
fn comparer_stats_coherent(calc: &StatsCoherent, ref_json: &Value) -> Resultat<bool> {
    if ![calc.s_phi, calc.s_phi2, calc.s_coh, calc.phi_max]
        .into_iter()
        .all(f64::is_finite)
    {
        return Ok(false);
    }
    let n_ref = ref_json["N"]
        .as_u64()
        .ok_or_else(|| Erreur::Format("N absent".into()))?;
    if calc.n != n_ref {
        return Ok(false);
    }
    // Intégrateur (M3 r2) : la référence Python (reference_py.json) ne produit pas `n_ambigues` ; on exige les clés qu'elle produit et l'on
    // compare `n_ambigues` seulement si elle est présente (r2 l'exigeait : RG-FLUX-REF-PY échouait en « clé absente »).
    verifier_finitude_stats(ref_json, &["N", "s_phi", "s_phi2", "s_coh", "phi_max"])?;
    if let Some(n_amb) = ref_json.get("n_ambigues") {
        let n_amb = n_amb
            .as_u64()
            .ok_or_else(|| Erreur::Format("n_ambigues non entier".into()))?;
        if calc.n_ambigues != n_amb {
            return Ok(false);
        }
    }
    let comptes_ref = ref_json["comptes"]
        .as_object()
        .ok_or_else(|| Erreur::Format("comptes absent".into()))?;
    if comptes_ref.len() != SEUILS_COH.len() {
        return Ok(false);
    }
    let mut ok = true;
    for (j, (nom, _, _)) in SEUILS_COH.iter().enumerate() {
        let ref_val = comptes_ref
            .get(*nom)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Erreur::Format(format!("compte {nom} absent")))?;
        if calc.comptes[j] != ref_val {
            ok = false;
        }
    }
    // comptes conjoints
    if let Some(joint_ref) = ref_json
        .get("comptes_coherence")
        .and_then(|v| v.as_object())
    {
        if joint_ref.len() != SEUILS_COH.len() {
            return Ok(false);
        }
        for (j, (nom, _, _)) in SEUILS_COH.iter().enumerate() {
            if let Some(par_cmin) = joint_ref.get(*nom).and_then(|v| v.as_object()) {
                if par_cmin.len() != COH_MIN.len() {
                    return Ok(false);
                }
                for (k, (cmin, _)) in COH_MIN.iter().enumerate() {
                    let ref_val = par_cmin
                        .get(*cmin)
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| Erreur::Format(format!("coherence {nom}/{cmin} absent")))?;
                    if calc.comptes_coherence[j][k] != ref_val {
                        ok = false;
                    }
                }
            } else {
                return Ok(false);
            }
        }
    } else {
        return Ok(false);
    }
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-30);
    if rel(
        calc.s_phi,
        ref_json["s_phi"]
            .as_f64()
            .ok_or_else(|| Erreur::Format("s_phi absent".into()))?,
    ) > 1e-11
    {
        ok = false;
    }
    if rel(
        calc.s_phi2,
        ref_json["s_phi2"]
            .as_f64()
            .ok_or_else(|| Erreur::Format("s_phi2 absent".into()))?,
    ) > 1e-11
    {
        ok = false;
    }
    if rel(
        calc.s_coh,
        ref_json["s_coh"]
            .as_f64()
            .ok_or_else(|| Erreur::Format("s_coh absent".into()))?,
    ) > 1e-11
    {
        ok = false;
    }
    if (calc.phi_max
        - ref_json["phi_max"]
            .as_f64()
            .ok_or_else(|| Erreur::Format("phi_max absent".into()))?)
    .abs()
        > 1e-12
    {
        ok = false;
    }
    Ok(ok)
}

/// Compare une entrée de fixture de la référence Python aux calculs locaux.
fn comparer_fixture_ref_py(fichier: &str, entree: &Value, fils: usize) -> Resultat<bool> {
    let chemin_candidats = [
        chemin_depot(&format!("tests/donnees/rg/{fichier}")),
        chemin_depot(&format!("validation/rg/fixtures/{fichier}")),
    ];
    let chemin = chemin_candidats
        .iter()
        .find(|p| p.exists())
        .ok_or_else(|| Erreur::Io {
            contexte: format!("fixture {fichier} introuvable"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "fichier manquant"),
        })?;
    let champ = lire_champ(chemin)?;

    if let Some(brut) = entree.get("brut").and_then(|v| v.as_object()) {
        for (cle, ref_stats) in brut {
            if let Some(n_str) = cle.strip_prefix('n') {
                if let Ok(n) = n_str.parse::<usize>() {
                    let stats_calc = stats_faces(&champ, n, Pavage::Tous, true, fils)?;
                    if !comparer_stats_faces(&stats_calc, ref_stats)? {
                        return Ok(false);
                    }
                }
            }
        }
    }
    if let Some(coh) = entree.get("coherent").and_then(|v| v.as_object()) {
        for (cle, ref_stats) in coh {
            if let Some(n_str) = cle.strip_prefix('n') {
                if let Ok(n) = n_str.parse::<usize>() {
                    let stats_calc =
                        stats_coherent(&champ, n, Pavage::Tous, VarianteFlux::Normale, fils)?;
                    if !comparer_stats_coherent(&stats_calc, ref_stats)? {
                        return Ok(false);
                    }
                }
            }
        }
    }
    Ok(true)
}

/// RG-FLUX-REF-PY : comparaison aux trois fixtures du §6 (T4).
fn gate_ref_py(fils: usize) -> Resultat<LigneGate> {
    let doc = charger_reference_py()?;
    let fixtures = doc["fixtures"]
        .as_array()
        .ok_or_else(|| Erreur::Format("fixtures absent".into()))?;
    let mut nb_ok = 0;
    let mut details = Vec::new();
    for entree in fixtures {
        let fichier = entree["fichier"]
            .as_str()
            .ok_or_else(|| Erreur::Format("fichier absent".into()))?;
        let res = comparer_fixture_ref_py(fichier, entree, fils);
        match res {
            Ok(true) => nb_ok += 1,
            Ok(false) => details.push(format!("{fichier} : écart")),
            Err(e) => details.push(format!("{fichier} : ERREUR {e}")),
        }
    }
    if details.is_empty() {
        Ok(LigneGate::nouvelle(
            Verdict::Pass,
            "RG-FLUX-REF-PY",
            format!("{nb_ok} fixtures conformes"),
        ))
    } else {
        Ok(LigneGate::nouvelle(
            Verdict::Fail,
            "RG-FLUX-REF-PY",
            details.join(" ; "),
        ))
    }
}

/// RG-FLUX-GROSSIER : blocage 3centre de cfg_s7_q64.npy, comparaison à ubar_s7_q64.npy.
fn gate_grossier(fils: usize) -> Resultat<LigneGate> {
    let cfg = chemin_depot("validation/rg/fixtures/cfg_s7_q64.npy");
    let ubar = chemin_depot("validation/rg/fixtures/ubar_s7_q64.npy");
    let champ_cfg = lire_champ(&cfg)?;
    let champ_ubar = lire_champ(&ubar)?;
    let op = Operateur::TroisCentre;
    let champ_gr = grossier(&champ_cfg, &op, 1, fils)?;
    let stats_gr = stats_faces(&champ_gr, 1, Pavage::Tous, false, fils)?;
    let stats_ubar = stats_faces(&champ_ubar, 1, Pavage::Tous, false, fils)?;

    let comptes_ok = stats_gr.comptes == stats_ubar.comptes;
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-30);
    let sommes_ok = rel(stats_gr.sx, stats_ubar.sx) <= 1e-12
        && rel(stats_gr.sx2, stats_ubar.sx2) <= 1e-12
        && rel(stats_gr.s_theta2, stats_ubar.s_theta2) <= 1e-12
        && rel(stats_gr.s_theta4, stats_ubar.s_theta4) <= 1e-12;
    if comptes_ok && sommes_ok {
        Ok(LigneGate::nouvelle(
            Verdict::Pass,
            "RG-FLUX-GROSSIER",
            "statistiques n=1 identiques à ubar_s7_q64.npy",
        ))
    } else {
        Ok(LigneGate::nouvelle(
            Verdict::Fail,
            "RG-FLUX-GROSSIER",
            format!("comptes_ok={comptes_ok} sommes_ok={sommes_ok}"),
        ))
    }
}

/// Un écart conserve la clé et les deux valeurs, y compris une clé absente.
fn ecart_json(cle: &str, calc: Option<&Value>, reference: Option<&Value>) -> String {
    let afficher = |v: Option<&Value>| {
        v.map(Value::to_string)
            .unwrap_or_else(|| "<absente>".into())
    };
    format!(
        "{cle} : calcul={}, référence={}",
        afficher(calc),
        afficher(reference)
    )
}

/// Premier écart entre statistiques brutes : les SEPT champs du Python réel
/// sont obligatoires. Les extensions kvu (fractions, n_ambigues) sont ignorées ;
/// l'histogramme est comparé seulement si la référence le fournit.
/// Tolérances : sommes relatives ≤ 1e-10, xmin absolu ≤ 1e-13.
fn comparer_stats_faces_json(calc: &Value, reference: &Value) -> Option<String> {
    for cle in ["N", "comptes", "sx", "sx2", "sTheta2", "sTheta4", "xmin"] {
        let a = calc.get(cle);
        let b = reference.get(cle);
        let (Some(a), Some(b)) = (a, b) else {
            return Some(ecart_json(cle, a, b));
        };
        let conforme = match cle {
            "N" => a.as_u64().is_some_and(|n| Some(n) == b.as_u64()),
            "comptes" => {
                let (Some(ca), Some(cb)) = (a.as_object(), b.as_object()) else {
                    return Some(ecart_json(cle, Some(a), Some(b)));
                };
                if ca.len() != SEUILS_X.len() || cb.len() != SEUILS_X.len() {
                    return Some(ecart_json(cle, Some(a), Some(b)));
                }
                for (nom, _, _) in SEUILS_X {
                    let va = ca.get(nom);
                    let vb = cb.get(nom);
                    if !va
                        .and_then(Value::as_u64)
                        .is_some_and(|n| Some(n) == vb.and_then(Value::as_u64))
                    {
                        return Some(ecart_json(&format!("comptes.{nom}"), va, vb));
                    }
                }
                true
            }
            _ => match (a.as_f64(), b.as_f64()) {
                (Some(a), Some(b)) if a.is_finite() && b.is_finite() => {
                    let tolerance = if cle == "xmin" {
                        1e-13
                    } else {
                        1e-10 * b.abs().max(1e-30)
                    };
                    (a - b).abs() <= tolerance
                }
                _ => false,
            },
        };
        if !conforme {
            return Some(ecart_json(cle, Some(a), Some(b)));
        }
    }
    if let Some(hist_ref) = reference.get("hist_alpha_nonvides") {
        let hist_calc = calc.get("hist_alpha_nonvides");
        if !hist_ref.is_object() || hist_calc != Some(hist_ref) {
            return Some(ecart_json("hist_alpha_nonvides", hist_calc, Some(hist_ref)));
        }
    }
    None
}

/// Les ensembles d'OBSERVABLES sont strictement identiques, indépendamment des
/// extensions autorisées à l'intérieur de chaque statistique.
fn comparer_stats_json_complet(calc: &Value, reference: &Value) -> Option<String> {
    let (Some(ca), Some(cb)) = (calc.as_object(), reference.as_object()) else {
        return Some(ecart_json("stats", Some(calc), Some(reference)));
    };
    for cle in ca.keys().chain(cb.keys()) {
        if !ca.contains_key(cle) || !cb.contains_key(cle) {
            return Some(ecart_json(
                &format!("stats.{cle}"),
                ca.get(cle),
                cb.get(cle),
            ));
        }
    }
    for (cle, valeur) in ca {
        // L'égalité des ensembles de clés vient d'être vérifiée.
        if let Some(ecart) = comparer_stats_faces_json(valeur, &cb[cle]) {
            return Some(format!("stats.{cle}.{ecart}"));
        }
    }
    None
}

/// G-REPRO : comparaison stricte du mode liste aux lignes de desert_debut_2026-11.
///
/// - chaque configuration sélectionnée doit posséder ses lignes fin ET gros1
///   dans la liste des travaux ET dans les références ; toute absence est un
///   écart nommé ;
/// - les ensembles d'observables (n1,n2,n3 pour fin ; n1 pour gros1) et les
///   sept champs obligatoires de chaque statistique sont comparés ;
/// - zéro ligne comparée ⇒ FAIL ; le détail donne le nombre de lignes comparées.
fn gate_repro(
    desert: Option<&str>,
    max_configs: Option<usize>,
    fils: usize,
) -> Resultat<LigneGate> {
    let desert = match desert {
        Some(d) => PathBuf::from(d),
        None => {
            return Ok(LigneGate::nouvelle(
                Verdict::Skip,
                "G-REPRO",
                "--desert absent",
            ));
        }
    };
    let liste = desert.join("donnees/liste_existants.jsonl");
    let refs = desert.join("donnees/lignes_existants.jsonl");
    if !liste.exists() || !refs.exists() {
        return Ok(LigneGate::nouvelle(
            Verdict::Skip,
            "G-REPRO",
            format!("fichier manquant : {liste:?} ou {refs:?}"),
        ));
    }

    // Lire les travaux et les références
    let mut travaux: Vec<Value> = Vec::new();
    for ligne in
        BufReader::new(std::fs::File::open(&liste).map_err(|e| Erreur::io("liste_existants", e))?)
            .lines()
    {
        let ligne = ligne.map_err(|e| Erreur::io("lecture ligne", e))?;
        if ligne.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&ligne).map_err(|e| Erreur::Format(e.to_string()))?;
        travaux.push(v);
    }
    let mut refs_map: HashMap<String, Value> = HashMap::new();
    for ligne in
        BufReader::new(std::fs::File::open(&refs).map_err(|e| Erreur::io("lignes_existants", e))?)
            .lines()
    {
        let ligne = ligne.map_err(|e| Erreur::io("lecture ligne", e))?;
        if ligne.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&ligne).map_err(|e| Erreur::Format(e.to_string()))?;
        let cell = v["cell"].as_str().unwrap_or("");
        let chain = v["chain"].as_str().unwrap_or("");
        let idx = v["idx"].as_i64().unwrap_or(-1);
        let kind = v["kind"].as_str().unwrap_or("");
        refs_map.insert(format!("{cell}::{chain}::{idx}::{kind}"), v);
    }

    // Grouper les travaux par configuration
    let mut configs: Vec<(String, String, i64)> = Vec::new();
    let mut travaux_par_config: HashMap<(String, String, i64), HashMap<String, Value>> =
        HashMap::new();
    for t in &travaux {
        let cell = t["cell"].as_str().unwrap_or("");
        let chain = t["chain"].as_str().unwrap_or("");
        let idx = t["idx"].as_i64().unwrap_or(-1);
        let kind = t["kind"].as_str().unwrap_or("");
        let tuple = (cell.to_string(), chain.to_string(), idx);
        if !configs.contains(&tuple) {
            configs.push(tuple.clone());
        }
        travaux_par_config
            .entry(tuple)
            .or_default()
            .insert(kind.to_string(), t.clone());
    }
    if let Some(max) = max_configs {
        configs.truncate(max);
    }

    let mut nb_lignes = 0usize;
    let mut ecarts: Vec<String> = Vec::new();

    for config in &configs {
        let cell = &config.0;
        let chain = &config.1;
        let idx = config.2;
        // Exiger les deux lignes fin et gros1 dans les travaux
        let travaux_config = travaux_par_config.get(config);
        for kind in ["fin", "gros1"] {
            let travail = travaux_config.and_then(|map| map.get(kind)).cloned();
            let reference = refs_map
                .get(&format!("{cell}::{chain}::{idx}::{kind}"))
                .cloned();
            match (travail, reference) {
                (None, reference) => {
                    ecarts.push(format!(
                        "{cell}/{chain}/idx={idx}/{kind} : travail manquant ; {}",
                        ecart_json("ligne", None, reference.as_ref())
                    ));
                }
                (Some(travail), None) => {
                    ecarts.push(format!(
                        "{cell}/{chain}/idx={idx}/{kind} : référence manquante ; {}",
                        ecart_json("ligne", Some(&travail), None)
                    ));
                }
                (Some(travail), Some(reference)) => {
                    nb_lignes += 1;
                    // Vérifier L
                    let path = travail["path"]
                        .as_str()
                        .ok_or_else(|| Erreur::Format("path absent".into()))?;
                    let champ = lire_champ(Path::new(path))?;
                    let ligne_id = format!("{cell}/{chain}/idx={idx}/{kind}");
                    if let Err(e) = super::cli::verifier_l_travail(&travail, champ.l, &ligne_id) {
                        ecarts.push(e.to_string());
                        continue;
                    }
                    // Recalculer les stats
                    let stats_calc = if kind == "fin" {
                        let mut map = Map::new();
                        for n in [1, 2, 3] {
                            let stats = stats_faces(&champ, n, Pavage::Tous, false, fils)?;
                            map.insert(format!("n{n}"), super::cli::stats_faces_json(&stats));
                        }
                        Value::Object(map)
                    } else {
                        let stats = stats_faces(&champ, 1, Pavage::Tous, false, fils)?;
                        let mut map = Map::new();
                        map.insert("n1".to_string(), super::cli::stats_faces_json(&stats));
                        Value::Object(map)
                    };
                    let ref_stats = &reference["stats"];
                    if let Some(ecart) = comparer_stats_json_complet(&stats_calc, ref_stats) {
                        ecarts.push(format!("{ligne_id} : {ecart}"));
                    }
                }
            }
        }
    }

    if nb_lignes == 0 {
        return Ok(LigneGate::nouvelle(
            Verdict::Fail,
            "G-REPRO",
            format!(
                "0 ligne comparée ; écarts : {}",
                if ecarts.is_empty() {
                    "aucun".into()
                } else {
                    ecarts.join(" ; ")
                }
            ),
        ));
    }
    if ecarts.is_empty() {
        Ok(LigneGate::nouvelle(
            Verdict::Pass,
            "G-REPRO",
            format!("{nb_lignes} lignes comparées, aucune différence"),
        ))
    } else {
        let premier = ecarts[0].clone();
        Ok(LigneGate::nouvelle(
            Verdict::Fail,
            "G-REPRO",
            format!("{nb_lignes} lignes comparées, premier écart : {premier}"),
        ))
    }
}

/// RG-FLUX-MUT : les mutants doivent casser les gates correspondantes.
fn gate_mut(fils: usize) -> Resultat<LigneGate> {
    let champ_base = champ_haar(6, 2611020301);
    let champ_jauge = appliquer_jauge(&champ_base, 2611020302);
    let (phi_base, _, _) = flux_coherent_faces(&champ_base, 3, VarianteFlux::SansTransport, fils)?;
    let (phi_jauge, _, _) =
        flux_coherent_faces(&champ_jauge, 3, VarianteFlux::SansTransport, fils)?;
    let max_dphi = ecart_max_fini(&phi_base, &phi_jauge)?;
    let rouge_jauge = max_dphi > 1e-6;

    let doc = charger_reference_py()?;
    let fixture = &doc["fixtures"][0];
    let fichier = fixture["fichier"].as_str().unwrap_or("");
    let chemin_candidats = [
        chemin_depot(&format!("tests/donnees/rg/{fichier}")),
        chemin_depot(&format!("validation/rg/fixtures/{fichier}")),
    ];
    let chemin = chemin_candidats
        .iter()
        .find(|p| p.exists())
        .ok_or_else(|| Erreur::Io {
            contexte: format!("fixture {fichier} introuvable"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "fichier manquant"),
        })?;
    let champ = lire_champ(chemin)?;
    let stats_mut = stats_coherent(&champ, 3, Pavage::Tous, VarianteFlux::PlanManquant, fils)?;
    let ref_stats = &fixture["coherent"]["n3"];
    // Isoler la sensibilité aux COMPTES : une différence de N ne suffit pas.
    let mut rouge_comptes = false;
    for (j, (nom, _, _)) in SEUILS_COH.iter().enumerate() {
        let attendu = ref_stats["comptes"][*nom]
            .as_u64()
            .ok_or_else(|| Erreur::Format(format!("compte de référence {nom} absent")))?;
        rouge_comptes |= stats_mut.comptes[j] != attendu;
    }
    // Neutraliser aussi N pour vérifier que le comparateur REF-PY rejette
    // effectivement le mutant à cause des statistiques restantes.
    let mut stats_meme_n = stats_mut;
    stats_meme_n.n = ref_stats["N"]
        .as_u64()
        .ok_or_else(|| Erreur::Format("N de référence absent".into()))?;
    let rouge_ref = rouge_comptes && !comparer_stats_coherent(&stats_meme_n, ref_stats)?;

    if rouge_jauge && rouge_ref {
        Ok(LigneGate::nouvelle(
            Verdict::Pass,
            "RG-FLUX-MUT",
            "SansTransport casse la jauge, PlanManquant casse les comptes REF-PY même à N identique",
        ))
    } else {
        Ok(LigneGate::nouvelle(
            Verdict::Fail,
            "RG-FLUX-MUT",
            format!("SansTransport rouge={rouge_jauge}, PlanManquant rouge={rouge_ref}"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ref_py_refuse_chaque_scalaire_calcule_nan_ou_infini() {
        assert!(!proche(f64::NAN, 1.0, 1e-12));
        assert!(ecart_max_fini(&[1.0, f64::NAN], &[1.0, 1.0]).is_err());
        let mut brut = StatsFaces::vide(false);
        brut.n = 1;
        brut.sx = 1.0;
        brut.sx2 = 1.0;
        brut.xmin = 1.0;
        let reference = super::super::cli::stats_faces_json(&brut);
        assert!(comparer_stats_faces(&brut, &reference).unwrap());
        for invalide in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for cle in ["sx", "sx2", "sTheta2", "sTheta4", "xmin"] {
                let mut calc = brut.clone();
                let champ = match cle {
                    "sx" => &mut calc.sx,
                    "sx2" => &mut calc.sx2,
                    "sTheta2" => &mut calc.s_theta2,
                    "sTheta4" => &mut calc.s_theta4,
                    _ => &mut calc.xmin,
                };
                *champ = invalide;
                assert!(
                    !comparer_stats_faces(&calc, &reference).unwrap(),
                    "{cle}={invalide}"
                );
            }
        }

        let doc = charger_reference_py().unwrap();
        let reference = &doc["fixtures"][0]["coherent"]["n3"];
        let coherent = StatsCoherent {
            n: reference["N"].as_u64().unwrap(),
            comptes: std::array::from_fn(|j| {
                reference["comptes"][SEUILS_COH[j].0].as_u64().unwrap()
            }),
            comptes_coherence: std::array::from_fn(|j| {
                std::array::from_fn(|k| {
                    reference["comptes_coherence"][SEUILS_COH[j].0][COH_MIN[k].0]
                        .as_u64()
                        .unwrap()
                })
            }),
            s_phi: reference["s_phi"].as_f64().unwrap(),
            s_phi2: reference["s_phi2"].as_f64().unwrap(),
            s_coh: reference["s_coh"].as_f64().unwrap(),
            phi_max: reference["phi_max"].as_f64().unwrap(),
            n_ambigues: 0, // absente de cette référence, conformément à l'arbitrage.
        };
        assert!(comparer_stats_coherent(&coherent, reference).unwrap());
        // Une erreur de compte seule doit être rouge, tous les moments et N
        // restant exactement ceux de la référence.
        let mut mauvais_compte = coherent.clone();
        mauvais_compte.comptes[0] += 1;
        assert!(!comparer_stats_coherent(&mauvais_compte, reference).unwrap());
        let mut avec_ambiguite = reference.clone();
        avec_ambiguite["n_ambigues"] = json!(0);
        assert!(comparer_stats_coherent(&coherent, &avec_ambiguite).unwrap());
        avec_ambiguite["n_ambigues"] = json!(1);
        assert!(!comparer_stats_coherent(&coherent, &avec_ambiguite).unwrap());
        for invalide in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for cle in ["s_phi", "s_phi2", "s_coh", "phi_max"] {
                let mut calc = coherent.clone();
                let champ = match cle {
                    "s_phi" => &mut calc.s_phi,
                    "s_phi2" => &mut calc.s_phi2,
                    "s_coh" => &mut calc.s_coh,
                    _ => &mut calc.phi_max,
                };
                *champ = invalide;
                assert!(
                    !comparer_stats_coherent(&calc, reference).unwrap(),
                    "{cle}={invalide}"
                );
            }
        }
    }

    #[test]
    fn repro_extensions_et_diagnostics_des_deux_valeurs() {
        let mut brut = StatsFaces::vide(false);
        brut.n = 1;
        brut.xmin = 0.0;
        let calc = super::super::cli::stats_faces_json(&brut);
        let mut reference = calc.clone();
        reference.as_object_mut().unwrap().remove("fractions");
        assert_eq!(reference.as_object().unwrap().len(), 7);
        assert!(comparer_stats_faces_json(&calc, &reference).is_none());
        for cle in ["N", "comptes", "sx", "sx2", "sTheta2", "sTheta4", "xmin"] {
            let mut invalide = reference.clone();
            invalide.as_object_mut().unwrap().remove(cle);
            let ecart = comparer_stats_faces_json(&calc, &invalide).unwrap();
            assert!(ecart.starts_with(cle));
            assert!(ecart.contains("calcul=") && ecart.contains("référence=<absente>"));
            let mut invalide = calc.clone();
            invalide.as_object_mut().unwrap().remove(cle);
            assert!(comparer_stats_faces_json(&invalide, &reference).is_some());
        }
        for cle in ["sx", "sx2", "sTheta2", "sTheta4", "xmin"] {
            let mut invalide = calc.clone();
            invalide[cle] = json!(f64::NAN); // NaN devient null en JSON, jamais accepté.
            assert!(comparer_stats_faces_json(&invalide, &reference).is_some());
        }
        let mut avec_extensions = calc.clone();
        avec_extensions["n_ambigues"] = json!(0);
        avec_extensions["hist_alpha_nonvides"] = json!({"0": 1});
        assert!(comparer_stats_faces_json(&avec_extensions, &reference).is_none());
        reference["hist_alpha_nonvides"] = json!({"0": 1});
        assert!(comparer_stats_faces_json(&avec_extensions, &reference).is_none());
        reference["hist_alpha_nonvides"] = json!({"1": 1});
        assert!(comparer_stats_faces_json(&avec_extensions, &reference).is_some());
    }
}
