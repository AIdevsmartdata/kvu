//! WP12a, re-verdict « les échecs de Newton sont un coût, pas un biais » (PREREG : `validation/fibre_contrainte/PREREG_NEWTON_2026-11.md`).
//! Exploration (hors verdict) puis contrôles à vérité connue sur le jouet CT-7 (courbe de ℝ²). Compilé en test seulement ; lourd : `#[ignore]`.

use super::alea::Flux;
use super::jouets::*;
use super::rattle::*;
use super::stat::{analyser, domaine_campagne, Analyse};
use crate::commun::Resultat;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::f64::consts::TAU;

/// Le jouet CT-7 avec x ramené dans [−π, π] à chaque avance : toutes les fonctions du jouet sont 2π-périodiques en x, donc la dynamique est
/// la même sur le cercle, mais la coordonnée ne dérive plus (sinon |x| ≈ 10³ après 10⁶ trajectoires et l'arrondi de `a sin x` — ≈ 10⁻¹³ —
/// atteint la tolérance de Newton, 10⁻¹³ : des échecs qui dépendent de |x| et non de x mod 2π). Même code de production : seul le modèle change.
pub(crate) struct CourbePeriodique(pub Courbe);

fn replier(x: f64) -> f64 {
    x - TAU * (x / TAU).round()
}

impl Variete for CourbePeriodique {
    type Etat = [f64; 2];
    type Eval = EvalCourbe;
    fn dim(&self) -> usize {
        self.0.dim()
    }
    fn nb_contraintes(&self) -> usize {
        self.0.nb_contraintes()
    }
    fn minv(&self) -> &[f64] {
        self.0.minv()
    }
    fn evaluer(&self, q: &Self::Etat) -> Resultat<Self::Eval> {
        self.0.evaluer(q)
    }
    fn potentiel(&self, e: &Self::Eval) -> f64 {
        self.0.potentiel(e)
    }
    fn grad_potentiel<'a>(&self, e: &'a Self::Eval) -> &'a [f64] {
        self.0.grad_potentiel(e)
    }
    fn jt_mul(&self, e: &Self::Eval, lambda: &[f64]) -> Vec<f64> {
        self.0.jt_mul(e, lambda)
    }
    fn j_mul(&self, e: &Self::Eval, v: &[f64]) -> Vec<f64> {
        self.0.j_mul(e, v)
    }
    fn g_resoudre(&self, e: &Self::Eval, b: &[f64]) -> Vec<f64> {
        self.0.g_resoudre(e, b)
    }
    fn residu(&self, q: &Self::Etat) -> Resultat<Vec<f64>> {
        self.0.residu(q)
    }
    fn avancer(&self, q: &Self::Etat, xi: &[f64], dt: f64) -> Self::Etat {
        [replier(q[0] + dt * xi[0]), q[1] + dt * xi[1]]
    }
    fn ecart(&self, a: &Self::Etat, b: &Self::Etat) -> f64 {
        // distance périodique en x (la différence de deux représentants repliés peut valoir ≈ 2π)
        let dx = replier(a[0] - b[0]).abs();
        dx.max((a[1] - b[1]).abs())
    }
}

fn env_u(nom: &str, d: usize) -> usize {
    std::env::var(nom).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}
fn env_f(nom: &str, d: f64) -> f64 {
    std::env::var(nom).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

/// Exploration : taux d'échec de Newton, acceptation et dépendance en x, pour (dt, nb_pas, max_newton) donnés.
#[test]
#[ignore = "exploration hors verdict"]
fn explo_newton_courbe() {
    let n = env_u("NWT_N", 400_000);
    let dt = env_f("NWT_DT", 0.15);
    let nb_pas = env_u("NWT_NPAS", 6);
    let max_newton = env_u("NWT_MAX", 50);
    let m2 = env_f("NWT_M2", 1.0);
    let m = CourbePeriodique(Courbe::nouveau(1.0, m2));
    let params = ParamsRattle { dt, nb_pas, max_newton, ..ParamsRattle::default() };
    let x0: f64 = 1.0;
    let q0 = [x0, m.0.a * x0.sin()];
    let (mut acc, mut nwt, mut rev, mut ev_) = (0usize, 0usize, 0usize, 0usize);
    let nbins = 12;
    let mut tot = vec![0usize; nbins];
    let mut fail = vec![0usize; nbins];
    let mut prec: Option<[f64; 2]> = Some(q0);
    let mut fen: Vec<String> = Vec::new();
    let (mut nf_fen, mut i_fen) = (0usize, 0usize);
    chaine(&m, q0, &params, &Flux::nouveau(26_110_100_001), 0, n, |_, q, s| {
        i_fen += 1;
        nf_fen += usize::from(s.echec.is_some());
        if i_fen == 100_000 {
            fen.push(format!("({nf_fen}, {:.4})", q[0].rem_euclid(TAU)));
            i_fen = 0;
            nf_fen = 0;
        }
        // état AVANT la proposition = état précédent ; on binne sur l'état précédent
        if let Some(p) = prec {
            let x = p[0].rem_euclid(TAU);
            let b = ((x / TAU) * nbins as f64) as usize % nbins;
            tot[b] += 1;
            if s.echec.is_some() {
                fail[b] += 1;
            }
        }
        prec = Some(*q);
        acc += usize::from(s.accepte);
        match s.echec {
            Some(Echec::Newton) => nwt += 1,
            Some(Echec::Reversibilite) => rev += 1,
            Some(Echec::Evaluation) => ev_ += 1,
            None => {}
        }
    })
    .unwrap();
    let f = |k: usize| k as f64 / n as f64;
    println!("INFO explo dt={dt} npas={nb_pas} max_newton={max_newton} M2={m2} : accept {:.4} newton {:.4} rev {:.4} eval {:.4}", f(acc), f(nwt), f(rev), f(ev_));
    println!("INFO   fenêtres de 100 000 : (échecs Newton, état final x) = {}", fen.join(" "));
    let prof: Vec<String> = (0..nbins).map(|b| format!("{:.3}", if tot[b] > 0 { fail[b] as f64 / tot[b] as f64 } else { f64::NAN })).collect();
    println!("INFO   f(x) par intervalle de x ∈ [0, 2π) : {}", prof.join(" "));
}


// ------------------------------------------------------------------------------------------------ contrôle à vérité connue

fn obs(q: &[f64; 2]) -> [f64; 2] {
    [q[0].cos(), (2.0 * q[0]).cos()]
}

fn depart(m: &CourbePeriodique) -> [f64; 2] {
    [1.0, m.0.a * 1.0_f64.sin()]
}

/// Chaîne de production (`chaine`) : série COMPLÈTE (échecs comptés comme rejets, état répété) et série « sans échecs » (les entrées
/// issues d'une proposition en échec sont retirées). Rend aussi (nombre d'échecs, nombre d'acceptées).
fn serie_production(m: &CourbePeriodique, params: &ParamsRattle, graine: u64, n: usize) -> (Vec<[f64; 2]>, Vec<[f64; 2]>, usize, usize) {
    let (mut rej, mut drop) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut nf, mut na) = (0usize, 0usize);
    chaine(m, depart(m), params, &Flux::nouveau(graine), 0, n, |_, q, s| {
        let o = obs(q);
        rej.push(o);
        if s.echec.is_some() {
            nf += 1;
        } else {
            drop.push(o);
        }
        na += usize::from(s.accepte);
    })
    .unwrap();
    (rej, drop, nf, na)
}

/// MUTANT « réessayer » : un échec de calcul est remplacé par une nouvelle proposition (nouvelle impulsion) jusqu'au succès ;
/// une entrée par proposition réussie (acceptée ou refusée par Metropolis). Compilé en test seulement.
fn serie_reessai(m: &CourbePeriodique, params: &ParamsRattle, graine: u64, n: usize) -> Vec<[f64; 2]> {
    let flux = Flux::nouveau(graine);
    let mut q = depart(m);
    let mut ev = m.evaluer(&q).unwrap();
    let mut t: u32 = 0;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        loop {
            let (prop, s) = proposition(m, &q, &ev, params, &flux, 0, t);
            t += 1;
            if s.echec.is_some() {
                continue;
            }
            if s.accepte {
                if let Some((q1, ev1)) = prop {
                    (q, ev) = (q1, ev1);
                }
            }
            break;
        }
        out.push(obs(&q));
    }
    out
}

fn resume(serie: &[[f64; 2]]) -> Value {
    let cols: Vec<Value> = (0..2)
        .map(|k| {
            let a: Analyse = analyser(&serie.iter().map(|v| v[k]).collect::<Vec<_>>());
            json!({"obs": k, "n": a.n, "moyenne": a.moyenne, "erreur": a.erreur, "tau_int": a.tau_int, "bloc": a.longueur_bloc,
                   "statut": a.statut.texte(), "domaine_campagne": domaine_campagne(&a)})
        })
        .collect();
    json!(cols)
}

/// Profil d'échec f(x) = P(échec de calcul | x) sur K nœuds de [−π, π), J propositions par nœud (impulsions de la chaîne k du flux).
fn profil_echec(m: &CourbePeriodique, params: &ParamsRattle, graine: u64, k_noeuds: usize, j: usize) -> (Vec<f64>, Vec<usize>) {
    let flux = Flux::nouveau(graine);
    let xs: Vec<f64> = (0..k_noeuds).map(|k| -std::f64::consts::PI + (k as f64 + 0.5) * TAU / k_noeuds as f64).collect();
    let nf: Vec<usize> = xs
        .par_iter()
        .enumerate()
        .map(|(k, &x)| {
            let q = [x, m.0.a * x.sin()];
            let ev = m.evaluer(&q).unwrap();
            (0..j).filter(|&i| proposition(m, &q, &ev, params, &flux, k as u32, i as u32).1.echec.is_some()).count()
        })
        .collect();
    (xs, nf)
}

/// Un retour est fautif si l'écart au départ vaut 1e-7 ou plus, ou n'est pas un nombre (NaN) : sémantique de l'ancien `!(e < 1e-7)`, qui est vrai pour NaN
/// (clippy `neg_cmp_op_on_partial_ord` refuse la forme niée ; `e.is_nan() || e >= 1e-7` est équivalente pour tout f64).
fn retour_fautif(e: f64) -> bool {
    e.is_nan() || e >= 1e-7
}

/// Invariance par renversement de l'ensemble de succès : pour `n` départs (x uniforme, impulsion tirée comme en production), une trajectoire
/// ALLER réussie doit avoir un RETOUR (depuis (q', −p')) qui réussit et revient à (q, −p).
fn invariance(m: &CourbePeriodique, params: &ParamsRattle, graine: u64, n: usize) -> Value {
    let flux = Flux::nouveau(graine);
    let (mut aller_ko, mut aller_ok, mut retour_ko, mut retour_faux) = (0usize, 0usize, 0usize, 0usize);
    let mut pire = 0.0f64;
    for i in 0..n {
        let x = -std::f64::consts::PI + TAU * flux.uniforme(0, i as u32, 0, super::alea::TAG_DEPART);
        let q = [x, m.0.a * x.sin()];
        let ev = m.evaluer(&q).unwrap();
        let p = tirer_impulsion(m, &ev, &flux, 0, i as u32);
        match trajectoire(m, q, ev, p.clone(), params) {
            Err(_) => aller_ko += 1,
            Ok((q1, ev1, p1, _)) => {
                aller_ok += 1;
                let pr: Vec<f64> = p1.iter().map(|v| -v).collect();
                match trajectoire(m, q1, ev1, pr, params) {
                    Err(_) => retour_ko += 1,
                    Ok((q2, _, p2, _)) => {
                        let e = m.ecart(&q2, &q).max(p2.iter().zip(&p).map(|(a, b)| (a + b).abs()).fold(0.0, f64::max));
                        pire = pire.max(e);
                        if retour_fautif(e) {
                            retour_faux += 1;
                        }
                    }
                }
            }
        }
    }
    json!({"n": n, "aller_echec": aller_ko, "aller_ok": aller_ok, "retour_echec": retour_ko, "retour_sans_retour_au_depart": retour_faux, "pire_ecart": pire})
}

/// Contrôle à vérité connue : écrit un JSON (création exclusive) par cellule et par estimateur ; le verdict est calculé par
/// `validation/fibre_contrainte/reverdict_newton.py`. Variables : NWT_SORTIE (fichier), NWT_N (trajectoires, défaut 2 000 000), NWT_NINV (départs d'invariance), NWT_GRAINE_BASE.
#[test]
#[ignore = "contrôle lourd WP12a re-verdict"]
fn controle_newton_courbe() {
    let n = env_u("NWT_N", 2_000_000);
    let ninv = env_u("NWT_NINV", 200_000);
    // base des graines : 26110160000 = run 1 (gelé), 26110170000 = run 2 (avenant PREREG_NEWTON_RUN2) ; toute autre base : hors verdict
    let graine_base = env_u("NWT_GRAINE_BASE", 26_110_160_000) as u64;
    let ninv_attendu = match graine_base {
        26_110_160_000 => 200_000,
        26_110_170_000 => 10_000_000,
        _ => 0,
    };
    let sortie = std::env::var("NWT_SORTIE").expect("NWT_SORTIE");
    let cellules: Vec<(f64, f64)> = vec![(1.0, 0.15), (1.0, 0.5), (1.0, 0.8), (1.0, 1.0), (4.0, 0.15), (4.0, 0.3), (4.0, 0.4), (4.0, 0.5)];
    let sorties: Vec<Value> = cellules
        .par_iter()
        .enumerate()
        .map(|(ic, &(m2, dt))| {
            let m = CourbePeriodique(Courbe::nouveau(1.0, m2));
            let params = ParamsRattle { dt, nb_pas: 6, ..ParamsRattle::default() };
            let g = graine_base + 10 * ic as u64;
            let (rej, drop, nf, na) = serie_production(&m, &params, g + 1, n);
            let retry = serie_reessai(&m, &params, g + 2, n);
            let (xs, nfp) = profil_echec(&m, &params, g + 3, 400, 20_000);
            let inv = invariance(&m, &params, g + 4, ninv);
            json!({"cellule": ic, "m2": m2, "dt": dt, "nb_pas": 6, "max_newton": params.max_newton, "n": n,
                   "tol_newton": params.tol_newton, "tol_rev": params.tol_rev, "graines": [g + 1, g + 2, g + 3, g + 4],
                   "taux_echec": nf as f64 / n as f64, "taux_acceptation": na as f64 / n as f64,
                   "production": resume(&rej), "sans_echecs": resume(&drop), "reessai": resume(&retry),
                   "profil": {"x": xs, "essais": 20_000, "echecs": nfp}, "invariance": inv})
        })
        .collect();
    let mut f = std::fs::OpenOptions::new().write(true).create_new(true).open(&sortie).unwrap();
    use std::io::Write;
    f.write_all(serde_json::to_string_pretty(&json!({"schema": "wp12a-newton-courbe/1", "hors_verdict": n != 2_000_000 || ninv != ninv_attendu, "graine_base": graine_base, "cellules": sorties})).unwrap().as_bytes()).unwrap();
}

/// Rejeu SU(2) de la reprise n2bis de FIB-CONTR (L = 6, β = 2,8, dt = 0,01, 2 pas, 1000 trajectoires, mêmes graines) : compte et classe les
/// échecs, puis rejoue chaque proposition en échec avec `max_newton` = 500 (diagnostic : y avait-il une racine que le plafond de 50 a manquée ?).
/// Sortie : JSON (création exclusive) dans NWT_SORTIE_SU2.
#[test]
#[ignore = "rejeu SU(2) lourd"]
fn rejeu_su2_n2bis() {
    use super::gates_su2::construire;
    let sortie = std::env::var("NWT_SORTIE_SU2").expect("NWT_SORTIE_SU2");
    let fils = env_u("NWT_FILS", 8);
    let graine = 26_102_612_821u64;
    let (m, u0) = construire(6, 2.8, graine, fils);
    let params = ParamsRattle { dt: 0.01, nb_pas: 2, ..ParamsRattle::default() };
    let flux = Flux::nouveau(graine + 100);
    let (mut echecs, mut acc, mut metro) = (Vec::new(), 0usize, 0usize);
    let mut etats = Vec::new();
    chaine(&m, u0, &params, &flux, 0, 1000, |i, q, s| {
        if let Some(e) = s.echec {
            echecs.push((i, format!("{e:?}")));
            etats.push((i, q.clone()));
        } else if s.accepte {
            acc += 1;
        } else {
            metro += 1;
        }
    })
    .unwrap();
    let params500 = ParamsRattle { max_newton: 500, ..params };
    let mut resolus = 0usize;
    let mut resolus_acceptes = 0usize;
    for (i, q) in &etats {
        let ev = m.evaluer(q).unwrap();
        let (_, s) = proposition(&m, q, &ev, &params500, &flux, 0, *i as u32);
        if s.echec.is_none() {
            resolus += 1;
            resolus_acceptes += usize::from(s.accepte);
        }
    }
    let fenetres: Vec<usize> = (0..10).map(|w| echecs.iter().filter(|(i, _)| i / 100 == w).count()).collect();
    let types: std::collections::BTreeMap<String, usize> = echecs.iter().fold(Default::default(), |mut a, (_, t)| {
        *a.entry(t.clone()).or_insert(0) += 1;
        a
    });
    let resume = format!(
        "1000 trajectoires : acceptées {acc}, refus Metropolis {metro}, échecs de calcul {} {:?}, par fenêtre de 100 {:?} ; rejoués avec max_newton = 500 : {resolus} sur {} aboutissent ({resolus_acceptes} acceptées)",
        echecs.len(), types, fenetres, echecs.len()
    );
    let doc = json!({"resume": resume, "echecs": echecs.len(), "acceptees": acc, "refus_metropolis": metro, "attendus_echecs": 29,
        "identique_n2bis": echecs.len() == 29 && acc == 939, "fenetres": fenetres, "resolus_cap500": resolus, "resolus_cap500_acceptes": resolus_acceptes});
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().write(true).create_new(true).open(&sortie).unwrap();
    f.write_all(serde_json::to_string_pretty(&doc).unwrap().as_bytes()).unwrap();
}

#[cfg(test)]
mod tests_predicats {
    use super::retour_fautif;

    #[test]
    fn retour_fautif_garde_la_semantique_de_la_forme_niee_y_compris_nan() {
        // équivalence exacte avec l'ancienne forme !(e < 1e-7) sur des valeurs représentatives et sur les valeurs spéciales
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let ancienne = |e: f64| !(e < 1e-7);
        let cas = [0.0, -0.0, 1e-300, 9.9999999e-8, 1e-7, 1.0000001e-7, 1e-3, 1.0, f64::MAX, f64::INFINITY, f64::NEG_INFINITY, f64::MIN_POSITIVE, -1.0];
        for e in cas {
            assert_eq!(retour_fautif(e), ancienne(e), "e = {e:e}");
        }
        // le cas NaN : fautif (un écart indéfini n'est jamais un retour au départ), comme la forme niée
        assert!(retour_fautif(f64::NAN));
        assert!(ancienne(f64::NAN));
        assert!(retour_fautif(-f64::NAN));
        // seuils
        assert!(!retour_fautif(9.9e-8));
        assert!(retour_fautif(1e-7));
    }
}
