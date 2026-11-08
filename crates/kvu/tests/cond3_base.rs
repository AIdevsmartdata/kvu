//! cond3 : invariants exacts de l'échantillonneur de la fibre dure de DEC (SU(2) et U(1) gaussien, d = 2 et 4) — agrafes, contrainte,
//! unitarité, identité de Ward grossière, covariance de jauge des scores, mutants (SPEC_T3 §2.2, §3.1 ; ORACLES O-0, O-6, O-10, O-11).

#![allow(clippy::needless_range_loop)]

use kvu::cond3::geom::Geom;
use kvu::cond3::modele::Modele;
use kvu::cond3::su2::{adjointe, agrafe, agrafe_de_plaquette, champ_haar_test, FibreSu2, MutantSu2};
use kvu::cond3::u1::FibreU1;
use kvu::heritage::su2_hb::{qdag, qmul, Q};
use std::sync::Arc;

fn demi_trace_plaq(u: &[Q], g: &Geom, pi: usize) -> f64 {
    let l = g.liens_plaquette(pi / g.v, pi % g.v);
    let p = qmul(qmul(u[l[0] as usize], u[l[1] as usize]), qmul(qdag(u[l[2] as usize]), qdag(u[l[3] as usize])));
    p[0]
}

#[test]
fn agrafe_somme_des_plaquettes_du_lien() {
    for d in [2usize, 4] {
        let g = Geom::nouvelle(d, 6).unwrap();
        let u = champ_haar_test(g.nliens(), 11 + d as u64);
        for &(mu, s) in &[(0usize, 0usize), (d - 1, 37 % g.v), (1, 100 % g.v), (0, g.v - 1)] {
            let li = mu * g.v + s;
            let k = agrafe(&u, &g, mu, s);
            let lhs = qmul(u[li], qdag(k))[0];
            // somme directe des plaquettes contenant le lien
            let mut somme = 0.0;
            let mut n = 0;
            for pi in 0..g.nplaq() {
                let l = g.liens_plaquette(pi / g.v, pi % g.v);
                if l.contains(&(li as u32)) {
                    somme += demi_trace_plaq(&u, &g, pi);
                    n += 1;
                }
            }
            assert_eq!(n, 2 * (d - 1));
            assert!((lhs - somme).abs() < 1e-13, "d = {d} lien {li} : {lhs} contre {somme}");
        }
    }
}

#[test]
fn agrafe_de_plaquette_somme_a_l_agrafe_du_lien() {
    let g = Geom::nouvelle(4, 6).unwrap();
    let u = champ_haar_test(g.nliens(), 5);
    let (mu, s) = (2usize, 77usize);
    let li = (mu * g.v + s) as u32;
    let k = agrafe(&u, &g, mu, s);
    let mut somme: Q = [0.0; 4];
    for pi in 0..g.nplaq() {
        let l = g.liens_plaquette(pi / g.v, pi % g.v);
        if let Some(rang) = l.iter().position(|&x| x == li) {
            let a = agrafe_de_plaquette(&u, l, rang);
            // ½ Tr U_p = ½ Tr(U_ℓ A†)
            let lhs = qmul(u[li as usize], qdag(a))[0];
            assert!((lhs - demi_trace_plaq(&u, &g, pi)).abs() < 1e-13, "rang {rang}");
            for i in 0..4 {
                somme[i] += a[i];
            }
        }
    }
    for i in 0..4 {
        assert!((somme[i] - k[i]).abs() < 1e-13);
    }
}

fn fibre_su2(d: usize, l: usize, beta: f64, graine: u64) -> FibreSu2 {
    let g = Arc::new(Geom::nouvelle(d, l).unwrap());
    let u = champ_haar_test(g.nliens(), graine);
    FibreSu2::nouveau(g, u, beta, graine + 1)
}

#[test]
fn pas_conserve_contrainte_et_unitarite() {
    for d in [2usize, 4] {
        let mut f = fibre_su2(d, 6, 2.5, 3);
        assert!(f.residu_contrainte() < 1e-15);
        for t in 0..6 {
            f.pas(t, 1);
        }
        assert!(f.residu_contrainte() < 1e-13, "d = {d} : contrainte {}", f.residu_contrainte());
        assert!(f.defaut_unitarite() < 1e-13, "unitarité {}", f.defaut_unitarite());
        assert_eq!(f.stats().plafond, 0);
        let mut s = fibre_su2(d, 6, 2.5, 3);
        for t in 0..6 {
            s.pas_sequentiel(t, 1);
        }
        assert!(s.residu_contrainte() < 1e-13);
    }
}

#[test]
fn ward_grossiere_exacte_et_mutant_rouge() {
    let mut f = fibre_su2(4, 6, 2.5, 21);
    for t in 0..3 {
        f.pas(t, 1);
    }
    let w = f.residu_ward();
    assert!(w < 1e-11, "Ward : {w}");
    // sur U⁰ aléatoire non thermalisé
    let f0 = fibre_su2(4, 6, 2.5, 22);
    assert!(f0.residu_ward() < 1e-11, "Ward sur U⁰ : {}", f0.residu_ward());
    // mutant R(V) au lieu de R(V)ᵀ : doit être très au-dessus de la tolérance
    let mut m = fibre_su2(4, 6, 2.5, 21);
    for t in 0..3 {
        m.pas(t, 1);
    }
    m.mutant = MutantSu2::WardSansTranspose;
    assert!(m.residu_ward() > 1e-3, "le mutant WardSansTranspose n'est pas détecté : {}", m.residu_ward());
}

#[test]
fn ward_u1_et_contrainte() {
    let g = Arc::new(Geom::nouvelle(4, 6).unwrap());
    let th: Vec<f64> = (0..g.nliens()).map(|i| ((i * 7919 % 1013) as f64) / 1013.0 - 0.5).collect();
    let mut f = FibreU1::nouveau(g, th, 1.0, 9);
    assert!(f.residu_ward() < 1e-12, "Ward U(1) sur U⁰ : {}", f.residu_ward());
    for t in 0..5 {
        f.pas(t, 1);
    }
    assert!(f.residu_contrainte() < 1e-12);
    assert!(f.residu_ward() < 1e-11, "Ward U(1) : {}", f.residu_ward());
    let mut s = f;
    for t in 5..8 {
        s.pas_sequentiel(t, 1);
    }
    assert!(s.residu_contrainte() < 1e-12);
}

#[test]
fn scores_covariants_de_jauge() {
    // σ^{(k)}_ē(g·U) = R(g(3X)) σ^{(k)}_ē(U) pour une jauge de Haar g (O-10a)
    let g = Arc::new(Geom::nouvelle(4, 6).unwrap());
    let u = champ_haar_test(g.nliens(), 31);
    let gauge = champ_haar_test(g.v, 77);
    // U'(x, μ) = g(x) U(x, μ) g(x + μ)†
    let mut u2 = u.clone();
    for mu in 0..4 {
        for s in 0..g.v {
            let sn = g.nbr[mu][s] as usize;
            u2[mu * g.v + s] = qmul(qmul(gauge[s], u[mu * g.v + s]), qdag(gauge[sn]));
        }
    }
    let f1 = FibreSu2::nouveau(g.clone(), u, 2.5, 1);
    let f2 = FibreSu2::nouveau(g.clone(), u2, 2.5, 1);
    let mut pire: f64 = 0.0;
    for ci in 0..g.ngros() {
        let s1 = f1.scores_ligne(ci);
        let s2 = f2.scores_ligne(ci);
        // origine de la ligne : site 3X
        let li0 = g.lignes[ci][0] as usize;
        let site0 = li0 % g.v;
        let r = adjointe(gauge[site0]);
        for k in 0..3 {
            for a in 0..3 {
                let att: f64 = (0..3).map(|b| r[a][b] * s1[k][b]).sum();
                pire = pire.max((att - s2[k][a]).abs());
            }
        }
    }
    assert!(pire < 1e-11, "covariance des scores : {pire}");
}

#[test]
fn mutant_agrafe_ne_preserve_pas_la_mesure_mais_garde_la_contrainte() {
    // le mutant garde la contrainte (U₃ asservi) : seul l'oracle statistique (O-2c, O-3a) le détecte ; ici on vérifie qu'il diffère
    let mut a = fibre_su2(4, 6, 2.5, 5);
    let mut b = fibre_su2(4, 6, 2.5, 5);
    b.mutant = MutantSu2::Agrafe;
    for t in 0..3 {
        a.pas(t, 1);
        b.pas(t, 1);
    }
    assert!(b.residu_contrainte() < 1e-13);
    assert_ne!(a.empreinte(), b.empreinte());
    // mutant U₃ non recalculé : la contrainte est violée (G-CONTRAINTE rouge)
    let mut c = fibre_su2(4, 6, 2.5, 5);
    c.mutant = MutantSu2::U3NonRecalcule;
    for t in 0..2 {
        c.pas(t, 1);
    }
    assert!(c.residu_contrainte() > 1e-3, "contrainte du mutant U3 : {}", c.residu_contrainte());
}

#[test]
fn rejeu_au_bit_meme_graine_meme_champ() {
    let mut a = fibre_su2(4, 6, 2.5, 8);
    let mut b = fibre_su2(4, 6, 2.5, 8);
    for t in 0..4 {
        a.pas(t, 1);
        b.pas(t, 1);
    }
    assert_eq!(a.empreinte(), b.empreinte());
    let mut c = fibre_su2(4, 6, 2.5, 8);
    c.set_graine(99);
    for t in 0..4 {
        c.pas(t, 1);
    }
    assert_ne!(a.empreinte(), c.empreinte());
}

// ------------------------------------------------------------------------------------------------------------
// Canaux de Stein : dérivées exactes contre différences finies de l'action et des observables (déterministe)
// ------------------------------------------------------------------------------------------------------------

use kvu::cond3::modele::Canal;
use kvu::heritage::su2_hb::qadd;

fn exp_i(a: usize, t: f64) -> Q {
    let mut q = [t.cos(), 0.0, 0.0, 0.0];
    q[1 + a] = t.sin();
    q
}

fn o_plaq(f: &FibreSu2, pi: u32) -> f64 {
    let mut o = [0.0];
    f.observables(&[pi], &mut o);
    o[0]
}

#[test]
fn canaux_de_stein_contre_differences_finies() {
    let g = Arc::new(Geom::nouvelle(4, 6).unwrap());
    let f0 = {
        let mut f = FibreSu2::nouveau(g.clone(), champ_haar_test(g.nliens(), 41), 2.5, 7);
        for t in 0..3 {
            f.pas(t, 1);
        }
        f
    };
    let h = 1e-4;
    // --- lien libre
    let mu = 1usize;
    let s = (0..g.v).find(|&s| g.pos[mu * g.v + s] == -1 && s > 50).unwrap();
    let li = (mu * g.v + s) as u32;
    assert_eq!(g.pos[li as usize], -1);
    let plaq: Vec<u32> = (0..g.nplaq() as u32)
        .filter(|&pi| g.liens_plaquette(pi as usize / g.v, pi as usize % g.v).contains(&li))
        .collect();
    assert_eq!(plaq.len(), 6);
    for a in 0..3u8 {
        for &pi in &plaq {
            let mut out = [0.0; 3];
            assert!(f0.canal(&Canal::SteinLibre { li, a, pi }, &mut out));
            let eval = |t: f64| {
                let mut f = FibreSu2::nouveau(g.clone(), f0.u.clone(), 2.5, 7);
                f.u[li as usize] = qmul(exp_i(a as usize, t), f.u[li as usize]);
                (f.action(), o_plaq(&f, pi))
            };
            let (sp, op) = eval(h);
            let (sm, om) = eval(-h);
            let (s0, _) = eval(0.0);
            let (ys, yo, y2s) = ((sp - sm) / (2.0 * h), (op - om) / (2.0 * h), (sp + sm - 2.0 * s0) / (h * h));
            assert!((out[0] - yo).abs() < 1e-6, "Y(O) libre : {} contre {}", out[0], yo);
            assert!((out[1] - ys).abs() < 1e-5 * (1.0 + ys.abs()), "Y(S) libre : {} contre {}", out[1], ys);
            assert!((out[2] - y2s).abs() < 1e-3 * (1.0 + y2s.abs()), "Y²(S) libre : {} contre {}", out[2], y2s);
        }
    }
    // --- mouvement de ligne : U1 -> U1 e^{itσ_a}, U2 -> e^{-itσ_a} U2
    let ci = 37usize;
    let [l1, l2, _l3] = g.lignes[ci];
    for &lk in &[l1, l2] {
        let plaq: Vec<u32> = (0..g.nplaq() as u32)
            .filter(|&pi| g.liens_plaquette(pi as usize / g.v, pi as usize % g.v).contains(&lk))
            .collect();
        for a in 0..3u8 {
            for &pi in &plaq {
                let mut out = [0.0; 3];
                assert!(f0.canal(&Canal::SteinLigne { ci: ci as u32, a, pi }, &mut out));
                let eval = |t: f64| {
                    let mut f = FibreSu2::nouveau(g.clone(), f0.u.clone(), 2.5, 7);
                    f.u[l1 as usize] = qmul(f.u[l1 as usize], exp_i(a as usize, t));
                    f.u[l2 as usize] = qmul(exp_i(a as usize, -t), f.u[l2 as usize]);
                    // V inchangé : DEC identique
                    let dec = f.dec();
                    for i in 0..4 {
                        assert!((dec[ci][i] - f0.vfix[ci][i]).abs() < 1e-13, "le mouvement de ligne doit être vertical");
                    }
                    (f.action(), o_plaq(&f, pi))
                };
                let (sp, op) = eval(h);
                let (sm, om) = eval(-h);
                let (s0, _) = eval(0.0);
                let (ys, yo, y2s) = ((sp - sm) / (2.0 * h), (op - om) / (2.0 * h), (sp + sm - 2.0 * s0) / (h * h));
                assert!((out[0] - yo).abs() < 1e-6, "Y(O) ligne : {} contre {}", out[0], yo);
                assert!((out[1] - ys).abs() < 1e-5 * (1.0 + ys.abs()), "Y(S) ligne : {} contre {}", out[1], ys);
                assert!((out[2] - y2s).abs() < 1e-3 * (1.0 + y2s.abs()), "Y²(S) ligne : {} contre {}", out[2], y2s);
            }
        }
    }
    let _ = qadd([0.0; 4], [0.0; 4]);
}

#[test]
fn releves_exacts_et_terme_x_contre_differences_finies() {
    // flots exacts : DEC(X_t(U)) = exp(itσ_a) DEC(U) pour k = 1, 2, 3 ; et X̄(O_p) contre la dérivée numérique
    let g = Arc::new(Geom::nouvelle(4, 6).unwrap());
    let f0 = {
        let mut f = FibreSu2::nouveau(g.clone(), champ_haar_test(g.nliens(), 43), 2.5, 9);
        for t in 0..2 {
            f.pas(t, 1);
        }
        f
    };
    let ci = 11usize;
    let t = 0.37;
    for k in 0..3usize {
        for a in 0..3usize {
            let mut f = FibreSu2::nouveau(g.clone(), f0.u.clone(), 2.5, 9);
            f.releve(k, ci, a, t);
            let dec = f.dec();
            let att = qmul(exp_i(a, t), f0.vfix[ci]);
            for i in 0..4 {
                assert!((dec[ci][i] - att[i]).abs() < 1e-13, "relèvement {k} direction {a} : DEC mal transporté");
            }
            for c in 0..g.ngros() {
                if c != ci {
                    for i in 0..4 {
                        assert!((dec[c][i] - f0.vfix[c][i]).abs() < 1e-13, "le relèvement ne doit déplacer que la ligne {ci}");
                    }
                }
            }
        }
    }
    // mutant : relèvement 2 sans rotation → le flot n'a plus la bonne image
    let mut m = FibreSu2::nouveau(g.clone(), f0.u.clone(), 2.5, 9);
    m.mutant = MutantSu2::ReleveSansRotation;
    m.releve(1, ci, 0, t);
    let att = qmul(exp_i(0, t), f0.vfix[ci]);
    let ecart: f64 = (0..4).map(|i| (m.dec()[ci][i] - att[i]).abs()).fold(0.0, f64::max);
    assert!(ecart > 1e-3, "MUT-RELEV non détecté par le flot : {ecart}");
    // terme X : dérivée numérique de O_p le long du relèvement moyen
    let [l1, l2, l3] = g.lignes[ci];
    for lk in [l1, l2, l3] {
        let plaq: Vec<u32> = (0..g.nplaq() as u32)
            .filter(|&pi| g.liens_plaquette(pi as usize / g.v, pi as usize % g.v).contains(&lk))
            .collect();
        for &pi in plaq.iter().take(3) {
            let mut x = [0.0; 3];
            f0.terme_x(&[(pi, ci as u32)], false, &mut x);
            let mut x9 = [0.0; 9];
            f0.terme_x(&[(pi, ci as u32)], true, &mut x9);
            for a in 0..3usize {
                let moyenne_num: f64 = (0..3usize)
                    .map(|k| {
                        let eval = |tt: f64| {
                            let mut f = FibreSu2::nouveau(g.clone(), f0.u.clone(), 2.5, 9);
                            f.releve(k, ci, a, tt);
                            o_plaq(&f, pi)
                        };
                        (eval(1e-5) - eval(-1e-5)) / 2e-5
                    })
                    .sum::<f64>()
                    / 3.0;
                assert!((x[a] - moyenne_num).abs() < 1e-7, "X̄(O) : {} contre {}", x[a], moyenne_num);
                let somme9: f64 = (0..3).map(|k| x9[3 * k + a]).sum::<f64>() / 3.0;
                assert!((somme9 - x[a]).abs() < 1e-14);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------
// O-0 : DEC (G0.4), lignes (G0.3), plaquettes à terme X (G0.7), O-10 : invariance de jauge de O, axes
// ------------------------------------------------------------------------------------------------------------

#[test]
fn dec_du_champ_froid_et_covariance_de_jauge() {
    for d in [2usize, 4] {
        let g = Arc::new(Geom::nouvelle(d, 6).unwrap());
        let froid = FibreSu2::nouveau(g.clone(), kvu::cond3::su2::froid(g.nliens()), 2.5, 1);
        for v in froid.dec() {
            assert!((0..4).all(|i| (v[i] - if i == 0 { 1.0 } else { 0.0 }).abs() < 1e-14));
        }
        // G0.4 : DEC de g·(froid) = g(3X) g(3X + 3 e_mu)†
        let gj = champ_haar_test(g.v, 5 + d as u64);
        let mut u = kvu::cond3::su2::froid(g.nliens());
        for mu in 0..d {
            for s in 0..g.v {
                let sn = g.nbr[mu][s] as usize;
                u[mu * g.v + s] = qmul(gj[s], qdag(gj[sn]));
            }
        }
        let f = FibreSu2::nouveau(g.clone(), u, 2.5, 1);
        let dec = f.dec();
        for mu in 0..d {
            for sc in 0..g.vc {
                let ci = mu * g.vc + sc;
                let x0 = g.lignes[ci][0] as usize % g.v;
                // site 3X + 3 e_mu : trois pas en avant dans la direction mu
                let x3 = g.nbr[mu][g.nbr[mu][g.nbr[mu][x0] as usize] as usize] as usize;
                let att = qmul(gj[x0], qdag(gj[x3]));
                assert!((0..4).all(|i| (dec[ci][i] - att[i]).abs() < 1e-14), "G0.4 d={d} ci={ci}");
            }
        }
    }
}

#[test]
fn aucune_paire_de_tranche_2_ou_plus_ne_contient_un_lien_de_la_ligne() {
    // G0.7 : X(O) = 0 pour b >= 2
    use kvu::cond3::paires::Paires;
    let g = Geom::nouvelle(4, 12).unwrap();
    let p = Paires::nouvelles(&g, g.faces(), g.sondes_paires(), 0, 12);
    for s in 0..p.sondes.len() {
        let ligne = g.lignes[p.sondes[s] as usize];
        for q in p.debut[s] as usize..p.debut[s + 1] as usize {
            let pi = p.faces[p.face_loc[q] as usize] as usize;
            let l = g.liens_plaquette(pi / g.v, pi % g.v);
            if l.iter().any(|x| ligne.contains(x)) {
                assert!(p.tranche[q] <= 1, "plaquette contenant un lien de la ligne à la tranche {}", p.tranche[q]);
            }
        }
    }
    assert!(!p.px.is_empty());
}

#[test]
fn observable_invariante_de_jauge_et_axes_permutes() {
    let g = Arc::new(Geom::nouvelle(4, 6).unwrap());
    let u = champ_haar_test(g.nliens(), 51);
    let gj = champ_haar_test(g.v, 52);
    let mut u2 = u.clone();
    for mu in 0..4 {
        for s in 0..g.v {
            u2[mu * g.v + s] = qmul(qmul(gj[s], u[mu * g.v + s]), qdag(gj[g.nbr[mu][s] as usize]));
        }
    }
    let f1 = FibreSu2::nouveau(g.clone(), u.clone(), 2.5, 1);
    let f2 = FibreSu2::nouveau(g.clone(), u2, 2.5, 1);
    let pis: Vec<u32> = (0..g.nplaq() as u32).collect();
    let (mut o1, mut o2) = (vec![0.0; pis.len()], vec![0.0; pis.len()]);
    f1.observables(&pis, &mut o1);
    f2.observables(&pis, &mut o2);
    let ecart = o1.iter().zip(&o2).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(ecart < 1e-14, "O_p non invariant de jauge : {ecart}");
    // O-10b : permutation des axes (échange 0 <-> 2) : champ permuté, plaquettes et scores permutés exactement
    let perm = [2usize, 1, 0, 3];
    let mut up = u.clone();
    for mu in 0..4 {
        for s in 0..g.v {
            let mut x = g.coords[s];
            let xs = [x[0], x[1], x[2], x[3]];
            for r in 0..4 {
                x[perm[r]] = xs[r];
            }
            let sp = g.site(&[x[0] as i64, x[1] as i64, x[2] as i64, x[3] as i64]);
            up[perm[mu] * g.v + sp] = u[mu * g.v + s];
        }
    }
    let fp = FibreSu2::nouveau(g.clone(), up, 2.5, 1);
    // scores : la ligne (X, mu) du champ permuté est la ligne (P X, perm[mu]) ; les scores sont ceux de la ligne d'origine
    let mut pire: f64 = 0.0;
    for mu in 0..4 {
        for sc in 0..g.vc {
            let mut x = g.coords_gros[sc];
            let xs = [x[0], x[1], x[2], x[3]];
            for r in 0..4 {
                x[perm[r]] = xs[r];
            }
            let scp = g.site_gros(&[x[0] as i64, x[1] as i64, x[2] as i64, x[3] as i64]);
            let s1 = f1.scores_ligne(mu * g.vc + sc);
            let s2 = fp.scores_ligne(perm[mu] * g.vc + scp);
            for k in 0..3 {
                for a in 0..3 {
                    pire = pire.max((s1[k][a] - s2[k][a]).abs());
                }
            }
        }
    }
    assert!(pire < 1e-12, "scores non permutés exactement : {pire}");
}

fn permutations4() -> Vec<[usize; 4]> {
    let mut out = Vec::new();
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let p = [a, b, c, d];
                    let mut v = p.to_vec();
                    v.sort_unstable();
                    if v == [0, 1, 2, 3] && p != [0, 1, 2, 3] {
                        out.push(p);
                    }
                }
            }
        }
    }
    out
}

#[test]
fn les_23_permutations_d_axes_permutent_exactement_scores_et_observables() {
    // O-10b : L = 6, les 23 permutations non triviales ; plaquettes (O_p) et scores permutés exactement (1e-12)
    let g = Arc::new(Geom::nouvelle(4, 6).unwrap());
    let u = champ_haar_test(g.nliens(), 61);
    let f1 = FibreSu2::nouveau(g.clone(), u.clone(), 2.5, 1);
    let perms = permutations4();
    assert_eq!(perms.len(), 23);
    let pis: Vec<u32> = (0..g.nplaq() as u32).collect();
    let mut o1 = vec![0.0; pis.len()];
    f1.observables(&pis, &mut o1);
    for perm in perms {
        let mut up = u.clone();
        for mu in 0..4 {
            for s in 0..g.v {
                let mut x = g.coords[s];
                let xs = [x[0], x[1], x[2], x[3]];
                for r in 0..4 {
                    x[perm[r]] = xs[r];
                }
                let sp = g.site(&[x[0] as i64, x[1] as i64, x[2] as i64, x[3] as i64]);
                up[perm[mu] * g.v + sp] = u[mu * g.v + s];
            }
        }
        let fp = FibreSu2::nouveau(g.clone(), up, 2.5, 1);
        let mut op = vec![0.0; pis.len()];
        fp.observables(&pis, &mut op);
        let mut pire_o: f64 = 0.0;
        for pi in 0..g.nplaq() {
            let (plan, s) = (pi / g.v, pi % g.v);
            let (mu, nu) = g.planes[plan];
            let mut x = g.coords[s];
            let xs = [x[0], x[1], x[2], x[3]];
            for r in 0..4 {
                x[perm[r]] = xs[r];
            }
            let sp = g.site(&[x[0] as i64, x[1] as i64, x[2] as i64, x[3] as i64]);
            // le coin de plus petites coordonnées du carré permuté est l'image du coin x : même ancrage, plan trié ; l'orientation éventuellement
            // inversée ne change pas ½ Tr U_p
            let (a, b) = (perm[mu].min(perm[nu]), perm[mu].max(perm[nu]));
            let plan2 = g.planes.iter().position(|&pp| pp == (a, b)).unwrap();
            pire_o = pire_o.max((o1[pi] - op[plan2 * g.v + sp]).abs());
        }
        assert!(pire_o < 1e-13, "plaquettes non permutées exactement par {perm:?} : {pire_o}");
        // scores
        let mut pire: f64 = 0.0;
        for mu in 0..4 {
            for sc in 0..g.vc {
                let mut x = g.coords_gros[sc];
                let xs = [x[0], x[1], x[2], x[3]];
                for r in 0..4 {
                    x[perm[r]] = xs[r];
                }
                let scp = g.site_gros(&[x[0] as i64, x[1] as i64, x[2] as i64, x[3] as i64]);
                let s1 = f1.scores_ligne(mu * g.vc + sc);
                let s2 = fp.scores_ligne(perm[mu] * g.vc + scp);
                for k in 0..3 {
                    for a in 0..3 {
                        pire = pire.max((s1[k][a] - s2[k][a]).abs());
                    }
                }
            }
        }
        assert!(pire < 1e-12, "scores non permutés exactement par {perm:?} : {pire}");
    }
}

#[test]
fn observable_non_invariante_de_jauge_est_detectee() {
    // O-10c : « observable » = partie scalaire d'un lien : la comparaison de jauge doit ÉCHOUER
    let g = Arc::new(Geom::nouvelle(4, 6).unwrap());
    let u = champ_haar_test(g.nliens(), 71);
    let gj = champ_haar_test(g.v, 72);
    let mut u2 = u.clone();
    for mu in 0..4 {
        for s in 0..g.v {
            u2[mu * g.v + s] = qmul(qmul(gj[s], u[mu * g.v + s]), qdag(gj[g.nbr[mu][s] as usize]));
        }
    }
    let ecart = (0..g.nliens()).map(|i| (u[i][0] - u2[i][0]).abs()).fold(0.0, f64::max);
    assert!(ecart > 1e-2, "MUT O-10c : la partie scalaire d'un lien devrait changer sous une jauge de Haar : {ecart}");
}
