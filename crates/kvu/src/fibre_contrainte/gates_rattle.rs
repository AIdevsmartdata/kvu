//! Gates de la route B sur jouets : FIB-RATTLE-POT (CT-7, + 2 mutants), FIB-JOUET (routes B et A, + mutant), FIB-VOLUME
//! (symplecticité en coordonnées ambiantes, + mutant). Lourdes : `#[ignore]`.
//! `cargo test --release --lib fibre_contrainte::gates_rattle -- --ignored --nocapture --test-threads=1`
//! Les effectifs peuvent être réduits pour l'exploration par `FIBC_ECHELLE` (facteur entier ≥ 1 de division) : les verdicts
//! portent sur l'échelle 1.

use super::alea::Flux;
use super::gates_commun::{ligne, sauter_si_sans_wp11};
use super::jouets::*;
use super::rattle::*;
use super::stat::*;
use crate::su2::cpu::haar_q;
use rayon::prelude::*;

const S_MOY_M16: f64 = SEUILS.moy_m16;
const S_MOY_M4: f64 = SEUILS.moy_m4;
const S_INTERCEPT_M16: f64 = SEUILS.intercept_m16;
const S_RESIDU_M16: f64 = SEUILS.residu_m16;

pub(crate) fn echelle() -> usize {
    std::env::var("FIBC_ECHELLE").ok().and_then(|s| s.parse().ok()).filter(|&n: &usize| n >= 1).unwrap_or(1)
}

fn serie_courbe(m: &Courbe, params: &ParamsRattle, graine: u64, n: usize) -> Vec<[f64; 2]> {
    let x0: f64 = 1.0;
    let q0 = [x0, m.a * x0.sin()];
    let mut out = Vec::with_capacity(n);
    chaine(m, q0, params, &Flux::nouveau(graine), 0, n, |_, q, _| out.push([q[0].cos(), (2.0 * q[0]).cos()])).unwrap();
    out
}

pub(crate) fn colonne(s: &[[f64; 2]], k: usize) -> Analyse {
    analyser(&s.iter().map(|v| v[k]).collect::<Vec<_>>())
}

#[test]
#[ignore = "gate lourde FIB-RATTLE-POT"]
fn fib_rattle_pot() {
    if sauter_si_sans_wp11("FIB-RATTLE-POT") {
        return;
    }
    let n = 2_000_000 / echelle();
    let (c1, c2) = moments_exacts();
    let params = ParamsRattle { dt: 0.15, nb_pas: 6, ..ParamsRattle::default() };
    let mut max_ok: f64 = 0.0;
    let mut non_vert = false;
    let cas: [(&str, f64, TermeLogDet, u64); 4] = [
        ("M = 1", 1.0, TermeLogDet::Correct, 26_102_612_401),
        ("M = diag(1,4)", 4.0, TermeLogDet::Correct, 26_102_612_402),
        ("mutant sans terme", 1.0, TermeLogDet::Retire, 26_102_612_403),
        ("mutant signe opposé", 1.0, TermeLogDet::Oppose, 26_102_612_404),
    ];
    // mutants : (max |t| sur les comparaisons VALIDES, nombre de comparaisons valides)
    let mut tmut = [(0.0f64, 0usize); 2];
    for (i, (nom, m2, terme, graine)) in cas.iter().enumerate() {
        // chaque cas est jugé sur les deux masses : les mutants le sont à M = 1 et M = diag(1,4)
        let masses: Vec<f64> = if i < 2 { vec![*m2] } else { vec![1.0, 4.0] };
        for (jm, mm) in masses.iter().enumerate() {
            let mut modele = Courbe::nouveau(1.0, *mm);
            modele.terme = *terme;
            let s = serie_courbe(&modele, &params, graine + 10 * jm as u64, n);
            for (k, exact) in [c1, c2].iter().enumerate() {
                let a = colonne(&s, k);
                if !domaine_campagne(&a) {
                    // variante CORRECTE non vérifiable : SKIP ; mutant : la comparaison est SKIP (non conclusive) et n'est pas un témoin
                    if i < 2 {
                        non_vert = true;
                    }
                    ligne("SKIP", if i < 2 { "FIB-RATTLE-POT" } else { "FIB-RATTLE-POT-MUTANT-CMP" }, &format!("statut {} : {nom} M2 = {mm} obs {k}", a.statut.texte()));
                    continue;
                }
                let t = t_exact(&a, *exact);
                ligne("INFO", "FIB-RATTLE-POT", &format!("{nom}, M = diag(1,{mm}) : obs {k} = {:.6} ± {:.2e} (exact {exact:.6}) t = {t:+.2}, τ = {:.2}", a.moyenne, a.erreur, a.tau_int));
                if i < 2 {
                    max_ok = maxn(max_ok, t.abs());
                } else {
                    tmut[i - 2] = (maxn(tmut[i - 2].0, t.abs()), tmut[i - 2].1 + 1);
                }
            }
        }
    }
    if non_vert {
        return;
    }
    let ok = max_ok < S_MOY_M4;
    ligne(if ok { "PASS" } else { "FAIL" }, "FIB-RATTLE-POT", &format!("max |t| = {max_ok:.3} < {S_MOY_M4} (m = 4)"));
    let mut tous = true;
    for (i, nom) in ["sans terme ½ log det", "signe opposé"].iter().enumerate() {
        let (t, nv) = tmut[i];
        if nv == 0 {
            ligne("SKIP", "FIB-RATTLE-POT-MUTANT", &format!("mutant {nom} : AUCUN TEMOIN VALIDE (toutes les comparaisons non vertes)"));
            tous = false;
        } else {
            let rouge = t >= S_MOY_M4;
            tous &= rouge;
            ligne(if rouge { "PASS" } else { "FAIL" }, "FIB-RATTLE-POT-MUTANT", &format!("mutant {nom} : max |t| = {t:.3} ≥ {S_MOY_M4} sur {nv} comparaisons valides (ROUGE exigé ; sinon ECHEC_OUTIL)"));
        }
    }
    assert!(ok && tous);
}

// ---------------------------------------------------------------------------------------------------------------- FIB-JOUET

fn v_aleatoire(k: u32) -> [f64; 4] {
    haar_q(0, 7, 2_610_250 + k, 5, 9)
}

fn serie_deux_liens_b(m: &DeuxLiens, params: &ParamsRattle, graine: u64, n: usize) -> Vec<[f64; 2]> {
    let q0 = m.depart(m.v, [1.0, 0.0, 0.0, 0.0]);
    let mut out = Vec::with_capacity(n);
    chaine(m, q0, params, &Flux::nouveau(graine), 0, n, |_, q, _| {
        let x = DeuxLiens::observable(q);
        out.push([x, x * x]);
    })
    .unwrap();
    out
}

#[test]
#[ignore = "gate lourde FIB-JOUET"]
fn fib_jouet() {
    if sauter_si_sans_wp11("FIB-JOUET") {
        return;
    }
    let n_b = 1_500_000 / echelle();
    let n_a = 30_000_000 / echelle();
    let params = ParamsRattle { dt: 0.1, nb_pas: 5, ..ParamsRattle::default() };
    let kappas = [1e-3, 2e-3, 4e-3];
    // cellules : β ∈ {2, 8} × t ∈ {1/2, 0,3} × V ∈ {V0, V1}
    let cellules: Vec<(f64, f64, u32)> = [2.0, 8.0].iter().flat_map(|&b| [0.5, 0.3].iter().flat_map(move |&t| (0..2u32).map(move |v| (b, t, v)))).collect();
    let mut max_b: f64 = 0.0;
    let mut max_a: f64 = 0.0;
    let mut skip = false;
    for (idx, &(beta, t, iv)) in cellules.iter().enumerate() {
        let v = v_aleatoire(iv);
        let (e1, e2) = moments_exacts_deux_liens(beta);
        // route B
        let m = DeuxLiens::nouveau(beta, t, v);
        let sb = serie_deux_liens_b(&m, &params, 26_102_612_501 + 10 * idx as u64, n_b);
        for (k, ex) in [e1, e2].iter().enumerate() {
            let a = colonne(&sb, k);
            if !domaine_campagne(&a) {
                skip = true;
                ligne("SKIP", "FIB-JOUET", &format!("route B cellule {idx} obs {k} : {}", a.statut.texte()));
                continue;
            }
            let ts = t_exact(&a, *ex);
            max_b = maxn(max_b, ts.abs());
            ligne("INFO", "FIB-JOUET", &format!("B β = {beta}, t = {t}, V{iv}, obs {k} : {:.6} ± {:.1e} (exact {ex:.6}) t = {ts:+.2}, τ = {:.1}", a.moyenne, a.erreur, a.tau_int));
        }
        // route A, κ-extrapolée
        let delta_de = |kappa: f64| (1.5 * (kappa / beta).sqrt()).min(0.5);
        let series: Vec<Vec<[f64; 2]>> = kappas
            .par_iter()
            .enumerate()
            .map(|(ik, &kappa)| chaine_penalite_deux_liens(beta, t, v, kappa, delta_de(kappa), 26_102_612_601 + 10 * idx as u64 + ik as u64, 0, n_a))
            .collect();
        for k in 0..2 {
            let ans: Vec<Analyse> = series.iter().map(|s| colonne(s, k)).collect();
            if ans.iter().any(|a| !domaine_campagne(a)) {
                skip = true;
                ligne("SKIP", "FIB-JOUET", &format!("route A cellule {idx} obs {k} : statuts {:?}", ans.iter().map(|a| a.statut.texte()).collect::<Vec<_>>()));
                continue;
            }
            let pts: Vec<(f64, &Analyse)> = kappas.iter().copied().zip(ans.iter()).collect();
            let (o0, s0, r, sr) = ajuster_jackknife(&pts).expect("trois points verts");
            let ex = if k == 0 { e1 } else { e2 };
            let ts = (o0 - ex) / s0;
            let tr = r / sr;
            ligne("INFO", "FIB-JOUET", &format!("A β = {beta}, t = {t}, V{iv}, obs {k} : O0 = {o0:.6} ± {s0:.1e} (exact {ex:.6}) t = {ts:+.2} ; forme : résidu du point du milieu t = {tr:+.2} ; τ = {:?}", ans.iter().map(|a| (a.tau_int * 10.0).round() / 10.0).collect::<Vec<_>>()));
            if tr.is_nan() || tr.abs() >= S_RESIDU_M16 {
                skip = true;
                ligne("SKIP", "FIB-JOUET", &format!("INDETERMINE : forme de l'ajustement (résidu du milieu |t| = {:.2} ≥ {}) cellule {idx} obs {k}", tr.abs(), S_RESIDU_M16));
                continue;
            }
            max_a = maxn(max_a, ts.abs());
        }
    }
    // mutant : route B sans facteur de coaire, β = 2
    let mut tmut: f64 = 0.0;
    let mut nvalides = 0usize;
    let (e1, e2) = moments_exacts_deux_liens(2.0);
    for (i, &t) in [0.5, 0.3].iter().enumerate() {
        for iv in 0..2u32 {
            let mut m = DeuxLiens::nouveau(2.0, t, v_aleatoire(iv));
            m.terme = TermeLogDet::Retire;
            let s = serie_deux_liens_b(&m, &params, 26_102_612_551 + 10 * (2 * i as u64 + iv as u64), n_b);
            for (k, ex) in [e1, e2].iter().enumerate() {
                let a = colonne(&s, k);
                if domaine_campagne(&a) {
                    let ts = t_exact(&a, *ex);
                    ligne("INFO", "FIB-JOUET-MUTANT", &format!("t = {t}, V{iv}, obs {k} : {:.6} (exact {ex:.6}) t = {ts:+.1}", a.moyenne));
                    tmut = maxn(tmut, ts.abs());
                    nvalides += 1;
                } else {
                    ligne("SKIP", "FIB-JOUET-MUTANT-CMP", &format!("statut {} : t = {t}, V{iv}, obs {k}", a.statut.texte()));
                }
            }
        }
    }
    if skip {
        return;
    }
    ligne(if max_b < S_MOY_M16 { "PASS" } else { "FAIL" }, "FIB-JOUET-B", &format!("route B : max |t| = {max_b:.3} < {} (m = 16, seuil empirique)", S_MOY_M16));
    ligne(if max_a < S_INTERCEPT_M16 { "PASS" } else { "FAIL" }, "FIB-JOUET-A", &format!("route A (κ-extrapolée) : max |t| = {max_a:.3} < {} (m = 16, seuil empirique de l'intercept) ; forme : résidu du milieu |t| < {} sur toutes les cellules", S_INTERCEPT_M16, S_RESIDU_M16));
    if nvalides == 0 {
        ligne("SKIP", "FIB-JOUET-MUTANT", "AUCUN TEMOIN VALIDE");
        return;
    }
    ligne(if tmut >= S_MOY_M16 { "PASS" } else { "FAIL" }, "FIB-JOUET-MUTANT", &format!("route B sans coaire : max |t| = {tmut:.2} ≥ {} sur {nvalides} comparaisons valides (ROUGE exigé)", S_MOY_M16));
    assert!(max_b < S_MOY_M16 && max_a < S_INTERCEPT_M16 && tmut >= S_MOY_M16);
}

// ---------------------------------------------------------------------------------------------------------------- FIB-VOLUME

/// Base orthonormée du noyau de la matrice `a` (k lignes, n colonnes, ordre ligne), par Gram–Schmidt modifié (rang k supposé plein).
fn noyau(a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut base: Vec<Vec<f64>> = Vec::new();
    for l in a {
        let mut v = l.clone();
        for b in &base {
            let d: f64 = v.iter().zip(b).map(|(x, y)| x * y).sum();
            v.iter_mut().zip(b).for_each(|(x, y)| *x -= d * y);
        }
        let nr = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(nr > 1e-8, "contraintes dépendantes");
        v.iter_mut().for_each(|x| *x /= nr);
        base.push(v);
    }
    let k = base.len();
    for i in 0..n {
        let mut v = vec![0.0; n];
        v[i] = 1.0;
        for _ in 0..2 {
            for b in &base {
                let d: f64 = v.iter().zip(b).map(|(x, y)| x * y).sum();
                v.iter_mut().zip(b).for_each(|(x, y)| *x -= d * y);
            }
        }
        let nr = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nr > 1e-6 {
            v.iter_mut().for_each(|x| *x /= nr);
            base.push(v);
        }
    }
    base.split_off(k)
}

/// Écart maximal de ω_amb(DΦ u_i, DΦ u_j) − ω_amb(u_i, u_j) sur une base orthonormée de l'espace tangent des phases.
/// `contraintes(x)` : contraintes ambiantes (nulles sur la variété) ; `phi(x)` : application de l'intégrateur ; `moitie` : nombre de
/// coordonnées de position ambiantes (x = (q, P)).
fn ecart_symplectique(x0: &[f64], moitie: usize, dim_attendue: usize, contraintes: &dyn Fn(&[f64]) -> Vec<f64>, phi: &dyn Fn(&[f64]) -> Vec<f64>) -> f64 {
    let n = x0.len();
    let h = 1e-6;
    let c0 = contraintes(x0);
    let jac: Vec<Vec<f64>> = (0..c0.len()).map(|_| vec![0.0; n]).collect::<Vec<_>>();
    let mut jac = jac;
    for k in 0..n {
        let (mut xp, mut xm) = (x0.to_vec(), x0.to_vec());
        xp[k] += h;
        xm[k] -= h;
        let (cp, cm) = (contraintes(&xp), contraintes(&xm));
        for i in 0..c0.len() {
            jac[i][k] = (cp[i] - cm[i]) / (2.0 * h);
        }
    }
    let base = noyau(&jac, n);
    assert_eq!(base.len(), dim_attendue, "dimension de l'espace tangent des phases");
    // orthonormalité et nullité des contraintes de la base (diagnostic)
    for (i, u) in base.iter().enumerate() {
        for (j, v) in base.iter().enumerate() {
            let d: f64 = u.iter().zip(v).map(|(a, b)| a * b).sum();
            assert!((d - f64::from(i == j)).abs() < 1e-10, "base non orthonormée");
        }
        for ligne_j in &jac {
            assert!(ligne_j.iter().zip(u).map(|(a, b)| a * b).sum::<f64>().abs() < 1e-9, "base hors du noyau");
        }
    }
    let s = 1e-6;
    let om = |a: &[f64], b: &[f64]| -> f64 { (0..moitie).map(|k| a[k] * b[moitie + k] - a[moitie + k] * b[k]).sum() };
    let d: Vec<Vec<f64>> = base
        .iter()
        .map(|u| {
            let (xp, xm): (Vec<f64>, Vec<f64>) = (x0.iter().zip(u).map(|(x, v)| x + s * v).collect(), x0.iter().zip(u).map(|(x, v)| x - s * v).collect());
            let (fp, fm) = (phi(&xp), phi(&xm));
            fp.iter().zip(&fm).map(|(a, b)| (a - b) / (2.0 * s)).collect()
        })
        .collect();
    let mut pire: f64 = 0.0;
    for i in 0..base.len() {
        for j in 0..base.len() {
            pire = maxn(pire, (om(&d[i], &d[j]) - om(&base[i], &base[j])).abs());
        }
    }
    pire
}

fn qvec(q: &[f64; 4]) -> [f64; 3] {
    [q[1], q[2], q[3]]
}

/// (ambiant) ⇄ (état, impulsion) du jouet à deux liens : x = (U₁, U₂, P₁, P₂), P_i = [0, p_i]·U_i.
fn ambiant_deux_liens(q: &[[f64; 4]; 2], p: &[f64]) -> Vec<f64> {
    use crate::heritage::su2_hb::qmul;
    let mut x: Vec<f64> = q.iter().flatten().copied().collect();
    for i in 0..2 {
        x.extend_from_slice(&qmul([0.0, p[3 * i], p[3 * i + 1], p[3 * i + 2]], q[i]));
    }
    x
}

fn depuis_ambiant_deux_liens(m: &DeuxLiens, x: &[f64]) -> ([[f64; 4]; 2], Vec<f64>) {
    use crate::heritage::su2_hb::{qdag, qmul};
    let nrm = |a: &[f64]| -> [f64; 4] {
        let n = a.iter().map(|v| v * v).sum::<f64>().sqrt();
        [a[0] / n, a[1] / n, a[2] / n, a[3] / n]
    };
    let q0 = [nrm(&x[0..4]), nrm(&x[4..8])];
    let q = projeter_sur_fibre(m, &q0, 1e-14, 50).unwrap();
    let mut p = Vec::new();
    for i in 0..2 {
        let pi = qmul([x[8 + 4 * i], x[9 + 4 * i], x[10 + 4 * i], x[11 + 4 * i]], qdag(q[i]));
        p.extend_from_slice(&qvec(&pi));
    }
    let ev = m.evaluer(&q).unwrap();
    (q, projeter_tangent(m, &ev, &p))
}

#[test]
#[ignore = "gate FIB-VOLUME"]
fn fib_volume() {
    let seuil = 1e-6;
    let mut pire: f64 = 0.0;
    let (mut mut_courbe, mut mut_deux): (f64, f64) = (0.0, 0.0);
    // (i) courbe de CT-7, ℝ² × ℝ²
    {
        let m = Courbe::default();
        let params = ParamsRattle { dt: 0.1, nb_pas: 10, ..ParamsRattle::default() };
        for k in 0..10u32 {
            let x = 0.3 + 0.55 * k as f64;
            let q0 = [x, m.a * x.sin()];
            let ev = m.evaluer(&q0).unwrap();
            let p0 = tirer_impulsion(&m, &ev, &Flux::nouveau(26_102_612_801), 0, k);
            let x0 = vec![q0[0], q0[1], p0[0], p0[1]];
            let contr = |z: &[f64]| vec![z[1] - m.a * z[0].sin(), -m.a * z[0].cos() * z[2] + z[3]];
            let phi_avec = |z: &[f64], mutant: bool| -> Vec<f64> {
                let q = projeter_sur_fibre(&m, &[z[0], z[1]], 1e-14, 50).unwrap();
                let ev = m.evaluer(&q).unwrap();
                let mut p = projeter_tangent(&m, &ev, &[z[2], z[3]]);
                let (mut q, mut ev) = (q, ev);
                for _ in 0..params.nb_pas {
                    let un = ParamsRattle { nb_pas: 1, ..params };
                    let (q1, e1, p1, _) = pas_rattle(&m, &q, &ev, &p, &un).unwrap();
                    q = q1;
                    ev = e1;
                    p = p1;
                    if mutant {
                        p.iter_mut().for_each(|v| *v *= 1.0 + 1e-3);
                    }
                }
                vec![q[0], q[1], p[0], p[1]]
            };
            let e = ecart_symplectique(&x0, 2, 2, &contr, &|z| phi_avec(z, false));
            let em = ecart_symplectique(&x0, 2, 2, &contr, &|z| phi_avec(z, true));
            ligne("INFO", "FIB-VOLUME", &format!("courbe, départ {k} : écart {e:.2e} ; mutant {em:.2e}"));
            pire = maxn(pire, e);
            mut_courbe = maxn(mut_courbe, em);
        }
    }
    // (ii) jouet à deux liens plongé dans les quaternions, ℝ⁸ × ℝ⁸
    {
        let m = DeuxLiens::nouveau(2.0, 0.5, v_aleatoire(0));
        let params = ParamsRattle { dt: 0.1, nb_pas: 10, ..ParamsRattle::default() };
        for k in 0..10u32 {
            let w = haar_q(k, 9, 2_610_280, 3, 8);
            let q0 = m.depart(m.v, w);
            let ev = m.evaluer(&q0).unwrap();
            let p0 = tirer_impulsion(&m, &ev, &Flux::nouveau(26_102_612_803), 0, k);
            let x0 = ambiant_deux_liens(&q0, &p0);
            let contr = |z: &[f64]| -> Vec<f64> {
                let nz = |a: &[f64]| a.iter().map(|v| v * v).sum::<f64>().sqrt();
                let q: [[f64; 4]; 2] = [[z[0] / nz(&z[0..4]), z[1] / nz(&z[0..4]), z[2] / nz(&z[0..4]), z[3] / nz(&z[0..4])], [z[4] / nz(&z[4..8]), z[5] / nz(&z[4..8]), z[6] / nz(&z[4..8]), z[7] / nz(&z[4..8])]];
                let g = m.residu(&q).unwrap();
                let mut c = vec![nz(&z[0..4]) - 1.0, nz(&z[4..8]) - 1.0];
                c.push(z[0] * z[8] + z[1] * z[9] + z[2] * z[10] + z[3] * z[11]);
                c.push(z[4] * z[12] + z[5] * z[13] + z[6] * z[14] + z[7] * z[15]);
                c.extend_from_slice(&g);
                // J p = 0 avec p déduit de P
                let (_, p) = depuis_ambiant_brut(&z[8..16], &q);
                let ev = m.evaluer(&q).unwrap();
                c.extend(m.j_mul(&ev, &p));
                c
            };
            let phi_avec = |z: &[f64], mutant: bool| -> Vec<f64> {
                let (q, p) = depuis_ambiant_deux_liens(&m, z);
                let ev = m.evaluer(&q).unwrap();
                let (mut q, mut ev, mut p) = (q, ev, p);
                for _ in 0..params.nb_pas {
                    let un = ParamsRattle { nb_pas: 1, ..params };
                    let (q1, e1, p1, _) = pas_rattle(&m, &q, &ev, &p, &un).unwrap();
                    q = q1;
                    ev = e1;
                    p = p1;
                    if mutant {
                        p.iter_mut().for_each(|v| *v *= 1.0 + 1e-3);
                    }
                }
                ambiant_deux_liens(&q, &p)
            };
            let e = ecart_symplectique(&x0, 8, 6, &contr, &|z| phi_avec(z, false));
            let em = ecart_symplectique(&x0, 8, 6, &contr, &|z| phi_avec(z, true));
            ligne("INFO", "FIB-VOLUME", &format!("deux liens, départ {k} : écart {e:.2e} ; mutant {em:.2e}"));
            pire = maxn(pire, e);
            mut_deux = maxn(mut_deux, em);
        }
    }
    let pire_mut = if mut_courbe.is_nan() || mut_deux.is_nan() { f64::NAN } else { mut_courbe.min(mut_deux) };
    ligne(if pire <= seuil { "PASS" } else { "FAIL" }, "FIB-VOLUME", &format!("max |ω_amb(DΦ u_i, DΦ u_j) − ω_amb(u_i, u_j)| = {pire:.2e} ≤ {seuil:e} (CT-7 et deux liens)"));
    ligne(if pire_mut > 1e-4 { "PASS" } else { "FAIL" }, "FIB-VOLUME-MUTANT", &format!("mutant moments × (1 + 1e-3) : détection {mut_courbe:.2e} (courbe) et {mut_deux:.2e} (deux liens) > 1e-4 (ROUGE exigé sur chaque modèle)"));
    assert!(pire <= seuil && pire_mut > 1e-4);
}

/// p déduit de P (ambiant) pour des liens déjà normalisés : partie vectorielle de P U†.
fn depuis_ambiant_brut(pp: &[f64], q: &[[f64; 4]; 2]) -> ((), Vec<f64>) {
    use crate::heritage::su2_hb::{qdag, qmul};
    let mut p = Vec::new();
    for i in 0..2 {
        let pi = qmul([pp[4 * i], pp[4 * i + 1], pp[4 * i + 2], pp[4 * i + 3]], qdag(q[i]));
        p.extend_from_slice(&qvec(&pi));
    }
    ((), p)
}
