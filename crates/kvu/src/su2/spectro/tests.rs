//! Portes exactes du module spectro (critères écrits dans `PREDICTIONS_SPECTRO.md` avant l'exécution).

use super::*;
use crate::obs::ref_cpu;
use crate::su2::torsion::Torsion;
use crate::su2::tube::reseau::{ChaineTube, Pile};
use std::f64::consts::PI;

fn config(d: [usize; 4], beta: f64, graine: u64, n: usize) -> (GeoA, Vec<Q>) {
    let g = GeoA::nouvelle(d).unwrap();
    let pile = Pile::depuis_torsion(&g, &Torsion::analyser("aucune").unwrap());
    let mut c = ChaineTube::nouvelle(g.clone(), graine);
    c.init(true);
    for _ in 0..n {
        c.pas(beta, 2, &pile);
    }
    (g, c.u)
}

fn param(ax: Axes, m: Mutant) -> ParamMesure {
    ParamMesure { axes: ax, niveaux: vec![0, 1, 3], alpha: 0.5, moments: vec![[0, 0], [1, 0], [1, 1]], tau_max: 2, mutant: m }
}

fn vecs(m: &Mesure) -> Vec<f64> {
    let mut v = Vec::new();
    for im in 0..m.corr.len() {
        for p in &m.corr[im] {
            v.extend(p);
        }
        for x in &m.phi_moyen[im] {
            v.extend(x);
        }
    }
    v
}

/// Invariance de jauge de tous les opérateurs (lignes lissées, φ à toutes impulsions, corrélateurs), tolérance 1e-12.
fn jauge(m: Mutant) -> bool {
    let (g, u) = config([4, 6, 6, 8], 2.3, 5, 6);
    let ax = Axes::nouveaux(0, 3).unwrap();
    let a = mesurer(&g, &u, &param(ax, m)).unwrap();
    let mut u2 = u.clone();
    transformer_jauge(&g, &mut u2, 99);
    // la transformation a bien changé les liens
    assert!(u.iter().zip(&u2).any(|(x, y)| (x[0] - y[0]).abs() > 1e-3));
    let b = mesurer(&g, &u2, &param(ax, m)).unwrap();
    vecs(&a).iter().zip(vecs(&b)).all(|(x, y)| (x - y).abs() < 1e-12)
}

/// Champ pure jauge : P = 1 pour toutes les lignes, lissées ou non (tolérance 1e-12).
fn pure_jauge(m: Mutant) -> bool {
    let g = GeoA::nouvelle([4, 6, 6, 8]).unwrap();
    let mut u = vec![UN; 4 * g.v];
    transformer_jauge(&g, &mut u, 7);
    let ax = Axes::nouveaux(0, 3).unwrap();
    [0usize, 1, 4].iter().all(|&n| {
        let l = lisser(&g, &u, 3, 0.5, n);
        lignes_points(&g, &l, &ax, m).iter().all(|p| (p - 1.0).abs() < 1e-12)
    })
}

/// −1 sur tous les liens U_w d'une tranche x_w = c : φ change de signe (tous niveaux, toutes impulsions), le reste inchangé.
#[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
fn centre(m: Mutant) -> bool {
    let (g, u) = config([4, 6, 6, 8], 2.3, 11, 6);
    let ax = Axes::nouveaux(0, 3).unwrap();
    let mut u2 = u.clone();
    for s in 0..g.v {
        if g.coords(s)[0] == 2 {
            for k in 0..4 {
                u2[s][k] = -u2[s][k];
            }
        }
    }
    let mom = [[0, 0], [1, 0], [1, 1]];
    [0usize, 2, 5].iter().all(|&n| {
        let a = phi(&g, &lisser(&g, &u, 3, 0.5, n), &ax, &mom, m);
        let b = phi(&g, &lisser(&g, &u2, 3, 0.5, n), &ax, &mom, m);
        a.iter().zip(&b).all(|(fa, fb)| fa.iter().zip(fb).all(|(x, y)| (x[0] + y[0]).abs() < 1e-12 && (x[1] + y[1]).abs() < 1e-12))
    })
}

/// Accord avec `obs polyakov_corr` (ref_cpu::correlateur_polyakov) sur un cube, sans lissage, même configuration, égalité
/// au bit, pour chaque direction d'enroulement ; puis le corrélateur en temps à impulsion nulle contre une somme brute.
fn cube(m: Mutant) -> bool {
    let (g, u) = config([6, 6, 6, 6], 2.2, 3, 6);
    let champ = ref_cpu::en_champ(&u, 6);
    let mut ok = true;
    for w in 0..4 {
        let temps = if w == 3 { 0 } else { 3 };
        let ax = Axes::nouveaux(w, temps).unwrap();
        let mien = polyakov_corr_cube(&g, &u, &ax, m);
        let ref_ = ref_cpu::correlateur_polyakov(&champ, w).unwrap();
        if mien != ref_ { eprintln!("w={w}: {mien:?} vs {ref_:?}"); }
        ok &= mien == ref_;
    }
    // corrélateur en temps, impulsion nulle, niveau 0 : somme brute indépendante sur les lignes de ref_cpu (w = 0, temps = 3)
    let ell = ref_cpu::polyakov_sites(&champ, 0).unwrap(); // j = (x1·6 + x2)·6 + x3
    let ax = Axes::nouveaux(0, 3).unwrap();
    let f = phi(&g, &u, &ax, &[[0, 0]], m);
    let c = correlateur(&[f[0].clone()], 3, m);
    for tau in 0..=3usize {
        let mut s = 0.0;
        for t in 0..6 {
            let mut a = 0.0;
            let mut b = 0.0;
            for ab in 0..36 {
                a += ell[ab * 6 + t];
                b += ell[ab * 6 + (t + tau) % 6];
            }
            s += (a / 36.0) * (b / 36.0);
        }
        if (s / 6.0 - c[0][tau]).abs() >= 1e-13 { eprintln!("tau={tau}: {} vs {}", s / 6.0, c[0][tau]); }
        ok &= (s / 6.0 - c[0][tau]).abs() < 1e-13;
    }
    ok
}

/// Oracle synthétique : φ(t) = A e^{2πi k t/T} donne C(τ) = A² cos(2π k τ/T) exactement.
fn synthetique(m: Mutant) -> bool {
    let t = 12usize;
    let k = 2.0;
    let f: Vec<[f64; 2]> = (0..t).map(|i| [1.5 * (2.0 * PI * k * i as f64 / t as f64).cos(), 1.5 * (2.0 * PI * k * i as f64 / t as f64).sin()]).collect();
    let c = correlateur(&[f], 5, m);
    (0..=5).all(|tau| (c[0][tau] - 2.25 * (2.0 * PI * k * tau as f64 / t as f64).cos()).abs() < 1e-12)
}

#[test]
fn portes_exactes_passent_sans_mutant() {
    assert!(jauge(Mutant::Aucun), "invariance de jauge");
    assert!(pure_jauge(Mutant::Aucun), "pure jauge");
    assert!(centre(Mutant::Aucun), "changement de signe par le centre");
    assert!(cube(Mutant::Aucun), "accord avec obs polyakov_corr et somme brute");
    assert!(synthetique(Mutant::Aucun), "oracle synthétique");
}

#[test]
fn mutants_mauvaise_direction() {
    let m = Mutant::MauvaiseDirection;
    assert!(!centre(m), "A3 doit échouer");
    assert!(!cube(m), "A4 doit échouer");
}

#[test]
fn mutants_lien_oublie() {
    let m = Mutant::LienOublie;
    assert!(!jauge(m), "l'invariance de jauge doit échouer");
}

#[test]
fn mutants_separation_decalee() {
    let m = Mutant::SeparationDecalee;
    assert!(!synthetique(m), "l'oracle synthétique doit échouer");
    assert!(!cube(m), "la somme brute doit échouer");
}

#[test]
fn ape_alpha_zero_identite_et_temps_intact() {
    let (g, u) = config([4, 4, 6, 8], 2.3, 2, 4);
    let a = ape_iteration(&g, &u, 3, 0.0);
    for (x, y) in a.iter().zip(&u) {
        for k in 0..4 {
            assert!((x[k] - y[k]).abs() < 1e-14);
        }
    }
    let b = lisser(&g, &u, 3, 0.5, 3);
    assert_eq!(&b[3 * g.v..], &u[3 * g.v..], "liens de temps modifiés");
    assert!(b.iter().all(|q| (qnorm(*q) - 1.0).abs() < 1e-12));
}

#[test]
fn froid_donne_un() {
    let g = GeoA::nouvelle([4, 4, 6, 8]).unwrap();
    let u = vec![UN; 4 * g.v];
    let ax = Axes::nouveaux(1, 3).unwrap();
    let m = mesurer(&g, &u, &param(ax, Mutant::Aucun)).unwrap();
    assert!((m.phi_moyen[0][0][0] - 1.0).abs() < 1e-14);
    assert!((m.corr[0][0][0] - 1.0).abs() < 1e-14);
    assert!(m.phi_moyen[1][0][0].abs() < 1e-14, "impulsion non nulle sur un champ uniforme");
}

// ---- A7 (ajout postérieur au hachage, ne relâche aucun critère) : lissage APE contre une référence indépendante en matrices 2×2 complexes
type M2 = [[(f64, f64); 2]; 2];
fn cm(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn mm(a: &M2, b: &M2) -> M2 {
    let mut r = [[(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let t = cm(a[i][k], b[k][j]);
                r[i][j] = (r[i][j].0 + t.0, r[i][j].1 + t.1);
            }
        }
    }
    r
}
fn dag(a: &M2) -> M2 {
    [[(a[0][0].0, -a[0][0].1), (a[1][0].0, -a[1][0].1)], [(a[0][1].0, -a[0][1].1), (a[1][1].0, -a[1][1].1)]]
}
fn de_q(q: Q) -> M2 {
    let m = crate::heritage::su2_hb::quat_to_mat(q);
    [[(m[0], m[1]), (m[2], m[3])], [(m[4], m[5]), (m[6], m[7])]]
}
fn plus(a: &M2, b: &M2, sa: f64, sb: f64) -> M2 {
    let mut r = [[(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            r[i][j] = (sa * a[i][j].0 + sb * b[i][j].0, sa * a[i][j].1 + sb * b[i][j].1);
        }
    }
    r
}

#[test]
fn ape_contre_reference_matrices_2x2() {
    let (g, u) = config([4, 4, 6, 6], 2.3, 8, 5);
    let alpha = 0.5;
    let got = ape_iteration(&g, &u, 3, alpha);
    let lk = |mu: usize, x: [usize; 4]| de_q(u[mu * g.v + g.site(&x)]);
    let sh = |x: [usize; 4], d: usize, k: i64| {
        let mut y = x;
        y[d] = ((x[d] as i64 + k).rem_euclid(g.d[d] as i64)) as usize;
        y
    };
    let mut pire: f64 = 0.0;
    for s in (0..g.v).step_by(7) {
        let x = g.coords(s);
        for mu in 0..3usize {
            let mut a: M2 = [[(0.0, 0.0); 2]; 2];
            for nu in 0..3usize {
                if nu == mu {
                    continue;
                }
                // haute : U_ν(x) U_μ(x+ν) U_ν(x+μ)† ; basse : U_ν(x−ν)† U_μ(x−ν) U_ν(x−ν+μ)
                let h = mm(&mm(&lk(nu, x), &lk(mu, sh(x, nu, 1))), &dag(&lk(nu, sh(x, mu, 1))));
                let xm = sh(x, nu, -1);
                let b = mm(&mm(&dag(&lk(nu, xm)), &lk(mu, xm)), &lk(nu, sh(xm, mu, 1)));
                a = plus(&a, &h, 1.0, 1.0);
                a = plus(&a, &b, 1.0, 1.0);
            }
            let v = plus(&lk(mu, x), &a, 1.0 - alpha, alpha / 4.0);
            let flat = [v[0][0].0, v[0][0].1, v[0][1].0, v[0][1].1, v[1][0].0, v[1][0].1, v[1][1].0, v[1][1].1];
            let q = crate::heritage::su2_hb::mat_to_quat(&flat);
            let n = qnorm(q);
            let qn = qscale(q, 1.0 / n);
            let r = got[mu * g.v + s];
            for k in 0..4 {
                pire = pire.max((qn[k] - r[k]).abs());
            }
        }
    }
    assert!(pire < 1e-12, "écart APE contre la référence matricielle : {pire}");
}

// =====================================================================================================================
// Secteur neutre (`neutre.rs`) : portes N1 à N7 et mutants (critères : PREDICTIONS_NEUTRE.md, partie 2)
// =====================================================================================================================
mod neutre_tests {
    use super::*;
    use crate::su2::spectro::neutre::{self, MutantN, ParamNeutre, FORMES};

    fn pn(m: MutantN) -> ParamNeutre {
        ParamNeutre { temps: 3, niveaux: vec![0, 1, 3], alpha: 0.5, formes: vec![0, 1, 2, 3, 4], tau_max: 3, mutant: m }
    }
    fn plat(m: &neutre::MesureNeutre) -> Vec<f64> {
        m.s.iter().flatten().copied().collect()
    }

    #[test]
    fn nombre_d_images() {
        let n: Vec<usize> = FORMES[..5].iter().map(|f| neutre::images(f.1).len()).collect();
        assert_eq!(n, vec![3, 6, 3, 4, 12], "images par forme (P11, R12, S22, CH1, CH2)");
    }

    /// N1 invariance de jauge, 1e-12.
    fn jauge(m: MutantN) -> bool {
        let (g, u) = config([6, 6, 6, 8], 2.3, 5, 6);
        let a = neutre::mesurer(&g, &u, &pn(m)).unwrap();
        let mut u2 = u.clone();
        transformer_jauge(&g, &mut u2, 99);
        assert!(u.iter().zip(&u2).any(|(x, y)| (x[0] - y[0]).abs() > 1e-3));
        let b = neutre::mesurer(&g, &u2, &pn(m)).unwrap();
        plat(&a).iter().zip(plat(&b)).all(|(x, y)| (x - y).abs() < 1e-12)
    }

    /// N2 pure jauge : tout vaut 1.   N3 champ froid : tout vaut 1 et C(τ) = 1.
    #[test]
    fn pure_jauge_et_froid() {
        let g = GeoA::nouvelle([6, 6, 6, 8]).unwrap();
        let mut u = vec![UN; 4 * g.v];
        let froid = neutre::mesurer(&g, &u, &pn(MutantN::Aucun)).unwrap();
        assert!(plat(&froid).iter().all(|x| (x - 1.0).abs() < 1e-14));
        assert!(froid.corr.iter().flatten().all(|x| (x - 1.0).abs() < 1e-13));
        transformer_jauge(&g, &mut u, 7);
        let pj = neutre::mesurer(&g, &u, &pn(MutantN::Aucun)).unwrap();
        assert!(plat(&pj).iter().all(|x| (x - 1.0).abs() < 1e-12));
    }

    /// N4 accord : P11 de niveau 0 = somme des plaquettes spatiales calculée avec des matrices 2×2 complexes indépendantes.
    #[test]
    fn plaquettes_contre_matrices_2x2() {
        let (g, u) = config([4, 6, 6, 6], 2.3, 8, 5);
        let p = ParamNeutre { temps: 3, niveaux: vec![0], alpha: 0.5, formes: vec![0], tau_max: 1, mutant: MutantN::Aucun };
        let m = neutre::mesurer(&g, &u, &p).unwrap();
        let lk = |mu: usize, x: [usize; 4]| de_q(u[mu * g.v + g.site(&x)]);
        let sh = |x: [usize; 4], d: usize, k: usize| {
            let mut y = x;
            y[d] = (x[d] + k) % g.d[d];
            y
        };
        for t in 0..g.d[3] {
            let (mut tot, mut n) = (0.0, 0usize);
            for s in 0..g.v {
                let x = g.coords(s);
                if x[3] != t {
                    continue;
                }
                for (a, b) in [(0usize, 1usize), (0, 2), (1, 2)] {
                    let w = mm(&mm(&lk(a, x), &lk(b, sh(x, a, 1))), &mm(&dag(&lk(a, sh(x, b, 1))), &dag(&lk(b, x))));
                    tot += 0.5 * (w[0][0].0 + w[1][1].0);
                    n += 1;
                }
            }
            let ref_ = tot / n as f64;
            assert!((ref_ - m.s[0][t]).abs() < 1e-13, "t={t}: {ref_} contre {}", m.s[0][t]);
        }
    }

    /// Permutation cyclique des trois axes spatiaux (0→1→2→0) d'une configuration : U'_{σμ}(σx) = U_μ(x).
    fn cycler(g: &GeoA, u: &[Q]) -> Vec<Q> {
        let mut o = u.to_vec();
        for mu in 0..4 {
            for s in 0..g.v {
                let x = g.coords(s);
                let y = [x[2], x[0], x[1], x[3]]; // y[σ(i)] = x[i] avec σ : 0→1, 1→2, 2→0
                let mu2 = if mu == 3 { 3 } else { (mu + 1) % 3 };
                o[mu2 * g.v + g.site(&y)] = u[mu * g.v + s];
            }
        }
        o
    }

    /// N5 invariance cubique : la rotation cyclique des axes laisse tous les opérateurs inchangés (boîte à section cubique).
    fn cubique(m: MutantN) -> bool {
        let (g, u) = config([6, 6, 6, 8], 2.3, 13, 6);
        let a = neutre::mesurer(&g, &u, &pn(m)).unwrap();
        let b = neutre::mesurer(&g, &cycler(&g, &u), &pn(m)).unwrap();
        plat(&a).iter().zip(plat(&b)).all(|(x, y)| (x - y).abs() < 1e-12)
    }

    /// N6 translation en temps : décalage de la configuration de 1 cran en temps = S(t+1).
    #[test]
    fn translation_en_temps() {
        let (g, u) = config([4, 4, 6, 8], 2.3, 17, 5);
        let mut u2 = u.clone();
        for mu in 0..4 {
            for s in 0..g.v {
                let mut x = g.coords(s);
                x[3] = (x[3] + 1) % 8;
                u2[mu * g.v + g.site(&x)] = u[mu * g.v + s];
            }
        }
        let a = neutre::mesurer(&g, &u, &pn(MutantN::Aucun)).unwrap();
        let b = neutre::mesurer(&g, &u2, &pn(MutantN::Aucun)).unwrap();
        for (sa, sb) in a.s.iter().zip(&b.s) {
            for t in 0..8 {
                assert!((sa[t] - sb[(t + 1) % 8]).abs() < 1e-12);
            }
        }
    }

    /// N7 oracle : S = c + A cos(2πkt/T) donne C(τ) = c² + (A²/2) cos(2πkτ/T) à 1e-12.
    #[test]
    #[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
    fn oracle_correlateur() {
        let t = 12usize;
        let s: Vec<f64> = (0..t).map(|i| 0.6 + 0.1 * (2.0 * PI * 2.0 * i as f64 / t as f64).cos()).collect();
        let c = neutre::correlateurs(&[s], 5);
        for tau in 0..=5 {
            let att = 0.36 + 0.005 * (2.0 * PI * 2.0 * tau as f64 / t as f64).cos();
            assert!((c[0][tau] - att).abs() < 1e-12);
        }
    }

    #[test]
    fn portes_neutres_sans_mutant() {
        assert!(jauge(MutantN::Aucun), "N1 invariance de jauge");
        assert!(cubique(MutantN::Aucun), "N5 invariance cubique");
    }

    #[test]
    fn mutant_lissage_faux() {
        assert!(!jauge(MutantN::LissageFaux), "un lissage mal fait doit casser l'invariance de jauge");
    }

    #[test]
    fn mutant_image_oubliee() {
        assert!(!cubique(MutantN::ImageOubliee), "une image oubliée doit casser l'invariance cubique");
    }

    // ---- impulsion non nulle (option, 18/11) : P1 translation, P2 somme directe indépendante, P3 champ froid, P4 jauge ----
    fn pp(niv: Vec<usize>) -> ParamNeutre {
        ParamNeutre { temps: 3, niveaux: niv, alpha: 0.5, formes: vec![0, 1, 3], tau_max: 3, mutant: MutantN::Aucun }
    }

    /// P1 : décaler la configuration d'un site le long d'un axe spatial change S(p) par une phase : C_p inchangé ; et non trivial.
    /// P4 : invariance de jauge de C_p.
    #[test]
    fn impulsion_translation_et_jauge() {
        let (g, u) = config([6, 6, 6, 8], 2.3, 21, 6);
        let a = neutre::mesurer_p(&g, &u, &pp(vec![0, 2])).unwrap();
        assert!(a.corr_p.iter().flatten().any(|x| x.abs() > 1e-6), "C_p ne doit pas être nul");
        for axe in 0..3 {
            let mut u2 = u.clone();
            for mu in 0..4 {
                for s in 0..g.v {
                    let mut x = g.coords(s);
                    x[axe] = (x[axe] + 1) % g.d[axe];
                    u2[mu * g.v + g.site(&x)] = u[mu * g.v + s];
                }
            }
            let b = neutre::mesurer_p(&g, &u2, &pp(vec![0, 2])).unwrap();
            for (pa, pb) in a.corr_p.iter().zip(&b.corr_p) {
                for (x, y) in pa.iter().zip(pb) {
                    assert!((x - y).abs() < 1e-12, "translation axe {axe}: {x} contre {y}");
                }
            }
        }
        let mut u3 = u.clone();
        transformer_jauge(&g, &mut u3, 5);
        let c = neutre::mesurer_p(&g, &u3, &pp(vec![0, 2])).unwrap();
        for (pa, pc) in a.corr_p.iter().zip(&c.corr_p) {
            for (x, y) in pa.iter().zip(pc) {
                assert!((x - y).abs() < 1e-12, "jauge: {x} contre {y}");
            }
        }
        // le corrélateur à impulsion nulle (corr) reste égal au bit à celui de `mesurer`
        let z = neutre::mesurer(&g, &u, &pp(vec![0, 2])).unwrap();
        assert!(a.corr.iter().flatten().zip(z.corr.iter().flatten()).all(|(x, y)| x == y || (x - y).abs() < 1e-13));
    }

    /// P2 : accord avec une somme directe indépendante (P11, niveau 0, axe 0 seul comparé via une boîte où les 3 axes sont décodés à la main).
    #[test]
    #[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
    fn impulsion_somme_directe() {
        let (g, u) = config([4, 4, 4, 6], 2.3, 23, 5);
        let p = ParamNeutre { temps: 3, niveaux: vec![0], alpha: 0.5, formes: vec![0], tau_max: 2, mutant: MutantN::Aucun };
        let m = neutre::mesurer_p(&g, &u, &p).unwrap();
        let tt = g.d[3];
        let lk = |mu: usize, x: [usize; 4]| de_q(u[mu * g.v + g.site(&x)]);
        let sh = |x: [usize; 4], d: usize, k: usize| {
            let mut y = x;
            y[d] = (x[d] + k) % g.d[d];
            y
        };
        // S(t;d) = Σ_x e^{2πi x_d/4} (1/3) Σ_{plans (a,b)} ½Tr plaquette(x ; a,b) / V3  (3 images de P11 : plans 01, 02, 12)
        let mut sd = vec![vec![[0.0f64; 2]; 3]; tt];
        for t in 0..tt {
            for s in 0..g.v {
                let x = g.coords(s);
                if x[3] != t {
                    continue;
                }
                for (a, b) in [(0usize, 1usize), (0, 2), (1, 2)] {
                    let w = mm(&mm(&lk(a, x), &lk(b, sh(x, a, 1))), &mm(&dag(&lk(a, sh(x, b, 1))), &dag(&lk(b, x))));
                    let q = 0.5 * (w[0][0].0 + w[1][1].0);
                    for d in 0..3 {
                        // phase au centre de la plaquette (a,b) : x_d + ½ pour d ∈ {a,b}
                        let dem = if d == a || d == b { 0.5 } else { 0.0 };
                        let th = 2.0 * PI * (x[d] as f64 + dem) / 4.0;
                        sd[t][d][0] += q * th.cos() / (3.0 * 64.0);
                        sd[t][d][1] += q * th.sin() / (3.0 * 64.0);
                    }
                }
            }
        }
        let mut s0 = vec![0.0f64; tt];
        for t in 0..tt {
            for s in 0..g.v {
                let x = g.coords(s);
                if x[3] != t {
                    continue;
                }
                for (a, b) in [(0usize, 1usize), (0, 2), (1, 2)] {
                    let w = mm(&mm(&lk(a, x), &lk(b, sh(x, a, 1))), &mm(&dag(&lk(a, sh(x, b, 1))), &dag(&lk(b, x))));
                    s0[t] += 0.5 * (w[0][0].0 + w[1][1].0) / (3.0 * 64.0);
                }
            }
            assert!((s0[t] - m.s[0][t]).abs() < 1e-13, "S0 t={t}: {} contre {}", s0[t], m.s[0][t]);
        }
        for tau in 0..=2usize {
            let mut acc = 0.0;
            for t in 0..tt {
                let t2 = (t + tau) % tt;
                for d in 0..3 {
                    let r = sd[t][d][0] * sd[t2][d][0] + sd[t][d][1] * sd[t2][d][1];
                    acc += 2.0 * r;
                }
            }
            let ref_ = acc / (2.0 * tt as f64 * 3.0);
            assert!((ref_ - m.corr_p[0][tau]).abs() < 1e-12, "tau={tau}: {ref_} contre {}", m.corr_p[0][tau]);
        }
    }

    /// P3 : champ froid : S(p ≠ 0) = 0, donc C_p = 0.
    #[test]
    fn impulsion_champ_froid() {
        let g = GeoA::nouvelle([6, 6, 6, 8]).unwrap();
        let u = vec![UN; 4 * g.v];
        let m = neutre::mesurer_p(&g, &u, &pp(vec![0, 1])).unwrap();
        assert!(m.corr_p.iter().flatten().all(|x| x.abs() < 1e-13));
    }

    /// P5 : invariance cubique de C_p (moyenne des 3 axes) : la rotation cyclique des axes le laisse inchangé (teste le centre des boucles, 5 formes).
    #[test]
    fn impulsion_cubique() {
        let (g, u) = config([6, 6, 6, 8], 2.3, 29, 6);
        let a = neutre::mesurer_p(&g, &u, &pn(MutantN::Aucun)).unwrap();
        let b = neutre::mesurer_p(&g, &cycler(&g, &u), &pn(MutantN::Aucun)).unwrap();
        assert!(a.corr_p.iter().flatten().any(|x| x.abs() > 1e-7));
        for (pa, pb) in a.corr_p.iter().zip(&b.corr_p) {
            for (x, y) in pa.iter().zip(pb) {
                assert!((x - y).abs() < 1e-12, "{x} contre {y}");
            }
        }
    }

    // ------------------------------------------------------------------
    // impulsions multiples (`--impulsions 1,2,3`). Boîte 8⁴ : ℓ spatial = 8, donc les
    // crans admis par |k| < ℓ/2 sont 1, 2, 3. I1 accord AU BIT avec l'ancienne option
    // `--impulsion`, I2 translation et jauge pour k = 2 et 3, I3 champ froid, I4 invariance
    // cubique, I5 signe C(−k) = C(k), I6 refus propres hors domaine.
    // ------------------------------------------------------------------

    /// Aplatit tous les corrélateurs à impulsion (crans, paires, τ).
    fn plat_ps(m: &neutre::MesureNeutre) -> Vec<f64> {
        m.corr_ps.iter().flat_map(|c| c.iter().flatten().copied()).collect()
    }

    /// I1 bis : le cœur de calcul, `tranches_ps` demandé sur le seul cran 1, rend les MÊMES BITS que l'implémentation
    /// d'origine de l'option `--impulsion` (conservée telle quelle comme oracle), pour les cinq formes d'opérateur : le
    /// contenu de `neutre_p.bin` est figé, et avec lui celui de `neutre.bin` (S_f(t) identique aussi).
    #[test]
    fn impulsions_cran1_contre_oracle() {
        let (g, u) = config([8, 8, 8, 8], 2.3, 43, 6);
        for forme in FORMES.iter() {
            for im in neutre::images(forme.1) {
                let (s0, p0) = neutre::tranches_p_origine(&g, &u, 3, std::slice::from_ref(&im));  // correctif du pilote (19/11)
                let (s1, p1) = neutre::tranches_ps(&g, &u, 3, &[im], &[1]);
                assert_eq!(s0.len(), s1.len(), "forme {}, séries à impulsion nulle", forme.0);
                assert_eq!(p1.len(), 1, "forme {}, un seul cran demandé", forme.0);
                assert_eq!(p0.len(), p1[0].len(), "forme {}, séries à impulsion 1", forme.0);
                for (x, y) in s0.iter().zip(&s1) {
                    assert!(x.to_bits() == y.to_bits(), "forme {}, S_f à impulsion nulle (mêmes bits) : {x} contre {y}", forme.0);
                }
                for (a, b) in p0.iter().zip(&p1[0]) {  // correctif du pilote (19/11) : p1 est indexé [cran][t][axe]
                    for (x, y) in a.iter().flatten().zip(b.iter().flatten()) {
                        assert!(x.to_bits() == y.to_bits(), "forme {}, Re/Im au cran 1 (mêmes bits) : {x} contre {y}", forme.0);
                    }
                }
            }
        }
    }

    /// I1 : le cran 1 rendu par la nouvelle option est IDENTIQUE AU BIT à celui de l'ancienne (`mesurer_p`), qu'il soit
    /// seul, en tête, au milieu de la liste ou pris avec son signe négatif ; `corr_p` rend le premier cran demandé
    /// (donc `neutre_p.bin` garde son contenu) ; et la liste de crans ne modifie en rien l'impulsion nulle.
    #[test]
    fn impulsions_k1_au_bit_pres() {
        let (g, u) = config([8, 8, 8, 8], 2.3, 31, 6);
        let p = pp(vec![0, 2]);
        let avant = neutre::mesurer_p(&g, &u, &p).unwrap();
        assert!(avant.corr_p.iter().flatten().any(|x| x.abs() > 1e-8), "C_1 ne doit pas être nul");
        for ks in [vec![1i64], vec![1, 2, 3], vec![3, 1, 2], vec![2, 3, 1], vec![-1, 3]] {
            let apres = neutre::mesurer_ps(&g, &u, &p, &ks).unwrap();
            assert_eq!(apres.corr_ps.len(), ks.len(), "un corrélateur par cran de {ks:?}");
            let i1 = ks.iter().position(|&k| k.abs() == 1).unwrap();
            let na: usize = avant.corr_p.iter().map(|v| v.len()).sum();
            let nb: usize = apres.corr_ps[i1].iter().map(|v| v.len()).sum();
            assert_eq!(na, nb, "taille du corrélateur du cran 1 de {ks:?}");
            for (x, y) in avant.corr_p.iter().flatten().zip(apres.corr_ps[i1].iter().flatten()) {
                assert!(x == y, "cran 1 (position {i1}) de {ks:?} contre --impulsion : {x} contre {y}");
            }
            // compatibilité : corr_p = corr_ps[0], le premier cran de la liste
            let nc: usize = apres.corr_p.iter().map(|v| v.len()).sum();
            let n0: usize = apres.corr_ps[0].iter().map(|v| v.len()).sum();
            assert_eq!(nc, n0, "corr_p doit être le premier cran de {ks:?}");
            for (x, y) in apres.corr_p.iter().flatten().zip(apres.corr_ps[0].iter().flatten()) {
                assert!(x == y, "corr_p contre corr_ps[0] de {ks:?} : {x} contre {y}");
            }
            // et le secteur à impulsion nulle (neutre.bin) est inchangé, lui aussi au bit près
            for (x, y) in avant.corr.iter().flatten().zip(apres.corr.iter().flatten()) {
                assert!(x == y, "corr à impulsion nulle modifié par {ks:?} : {x} contre {y}");
            }
            for (x, y) in avant.s.iter().flatten().zip(apres.s.iter().flatten()) {
                assert!(x == y, "S(t) modifié par {ks:?} : {x} contre {y}");
            }
            for (x, y) in avant.moy.iter().zip(apres.moy.iter()) {
                assert!(x == y, "⟨S_op⟩ modifié par {ks:?} : {x} contre {y}");
            }
        }
    }

    /// I2 : pour k = 2 et 3, décaler la configuration d'un site le long de CHACUN des trois axes spatiaux (S ne change que
    /// d'une phase globale, par axe) et transformer la configuration de jauge laissent C_k inchangés ; et C_k n'est pas nul.
    #[test]
    fn impulsions_2_3_translation_et_jauge() {
        let (g, u) = config([8, 8, 8, 8], 2.3, 33, 6);
        let p = pp(vec![0, 2]);
        let a = neutre::mesurer_ps(&g, &u, &p, &[2, 3]).unwrap();
        let pa = plat_ps(&a);
        // 2 crans × 21 paires (6 opérateurs = 2 niveaux × 3 formes) × 4 valeurs de τ
        assert_eq!(pa.len(), 2 * 21 * 4, "taille attendue de corr_ps");
        assert!(a.corr_ps.iter().all(|c| c.iter().flatten().any(|x| x.abs() > 1e-12)), "C_2 et C_3 non nuls");
        for axe in 0..3 {
            let mut u2 = u.clone();
            for mu in 0..4 {
                for s in 0..g.v {
                    let mut x = g.coords(s);
                    x[axe] = (x[axe] + 1) % g.d[axe];
                    u2[mu * g.v + g.site(&x)] = u[mu * g.v + s];
                }
            }
            let b = neutre::mesurer_ps(&g, &u2, &p, &[2, 3]).unwrap();
            let pb = plat_ps(&b);
            assert_eq!(pb.len(), pa.len());
            for (x, y) in pa.iter().zip(&pb) {
                assert!((x - y).abs() < 1e-12, "translation d'un site le long de l'axe {axe} : {x} contre {y}");
            }
        }
        let mut u3 = u.clone();
        transformer_jauge(&g, &mut u3, 5);
        assert!(u.iter().zip(&u3).any(|(x, y)| (x[0] - y[0]).abs() > 1e-3), "la jauge n'a rien changé");
        let c = neutre::mesurer_ps(&g, &u3, &p, &[2, 3]).unwrap();
        let pc = plat_ps(&c);
        for (x, y) in pa.iter().zip(&pc) {
            assert!((x - y).abs() < 1e-12, "transformation de jauge : {x} contre {y}");
        }
    }

    /// I3 : champ froid (toutes les liaisons à l'identité). Témoin : S = 1 et C(0) = 1. À tout cran non nul la somme de
    /// Fourier d'un champ constant est nulle (à l'epsilon machine près, par annulation terme à terme), donc |C_k| est
    /// au plus du carré de cette erreur : nul à la précision machine.
    #[test]
    fn impulsions_2_3_champ_froid() {
        let g = GeoA::nouvelle([8, 8, 8, 8]).unwrap();
        let u = vec![UN; 4 * g.v];
        let m = neutre::mesurer_ps(&g, &u, &pp(vec![0, 1]), &[1, 2, 3]).unwrap();
        assert!(m.s.iter().flatten().all(|x| (x - 1.0).abs() < 1e-14), "témoin : S = 1 à impulsion nulle");
        assert!(m.corr.iter().flatten().all(|x| (x - 1.0).abs() < 1e-13), "témoin : C(0) = 1");
        let plat = plat_ps(&m);
        assert_eq!(plat.len(), 3 * 21 * 4, "trois crans × 21 paires × 4 τ");
        let pire = plat.iter().fold(0.0f64, |acc, x| acc.max(x.abs()));
        assert!(pire < 1e-18, "champ froid : |C_k| doit être nul à la précision machine, pire écart {pire}");
    }

    /// I4 : invariance cubique des corrélateurs à impulsion (moyenne des trois axes spatiaux) pour k = 2 et 3 : la
    /// rotation cyclique des axes (0→1→2→0), qui permute aussi les crans entre axes, les laisse invariants (5 formes).
    #[test]
    fn impulsions_2_3_cubique() {
        let (g, u) = config([8, 8, 8, 8], 2.3, 37, 6);
        let p = pn(MutantN::Aucun);
        let a = neutre::mesurer_ps(&g, &u, &p, &[2, 3]).unwrap();
        let b = neutre::mesurer_ps(&g, &cycler(&g, &u), &p, &[2, 3]).unwrap();
        let pa = plat_ps(&a);
        let pb = plat_ps(&b);
        assert!(pa.iter().any(|x| x.abs() > 1e-12), "C_2 ou C_3 attendu non nul");
        assert_eq!(pb.len(), pa.len());
        for (x, y) in pa.iter().zip(&pb) {
            assert!((x - y).abs() < 1e-12, "rotation cyclique des axes : {x} contre {y}");
        }
    }

    /// I5 : C(−k) = C(k) À L'OCTET PRÈS (le champ est réel : S(−p) = conj S(p), et la corrélation symétrisée ne garde que
    /// la partie réelle), pour k = 2 et 3 ; et un cran mesuré dans une liste mélangée rend exactement le même résultat
    /// qu'une mesure menée sur ce cran seul (la sortie d'un cran ne dépend pas des autres crans demandés).
    #[test]
    fn impulsions_signe_et_melange() {
        let (g, u) = config([8, 8, 8, 8], 2.3, 41, 6);
        let p = pp(vec![0, 2]);
        let crans = [2i64, -2, 3, -3];
        let m = neutre::mesurer_ps(&g, &u, &p, &crans).unwrap();
        for (i, j) in [(0usize, 1usize), (2, 3)] {
            for (x, y) in m.corr_ps[i].iter().flatten().zip(m.corr_ps[j].iter().flatten()) {
                assert!(x == y, "C(−k) contre C(k) (crans {:?}) : {x} contre {y}", [crans[i], crans[j]]);
            }
        }
        for (i, &k) in crans.iter().enumerate() {
            let seul = neutre::mesurer_ps(&g, &u, &p, &[k]).unwrap();
            for (x, y) in m.corr_ps[i].iter().flatten().zip(seul.corr_ps[0].iter().flatten()) {
                assert!(x == y, "cran {k} mesuré avec {crans:?} contre seul : {x} contre {y}");
            }
        }
    }

    /// I6 : refus propres (erreur de configuration, pas de panique) du cran nul et des crans ≥ ℓ/2, y compris signés et
    /// au sein d'une liste par ailleurs valable.
    #[test]
    fn impulsions_refus_hors_domaine() {
        let g = GeoA::nouvelle([8, 8, 8, 8]).unwrap();
        let u = vec![UN; 4 * g.v];
        let p = pp(vec![0]);
        for &k in &[1i64, 2, 3, -1, -3] {
            assert!(neutre::mesurer_ps(&g, &u, &p, &[k]).is_ok(), "cran {k} admissible avec ℓ = 8");
        }
        for &k in &[0i64, 4, 5, 12, -4, -9] {
            match neutre::mesurer_ps(&g, &u, &p, &[k]) {
                Err(Erreur::Config(t)) => assert!(t.contains("impulsion"), "message d'erreur attendu : {t}"),
                Err(e) => panic!("erreur de configuration attendue pour le cran {k}, reçu {e:?}"),
                Ok(_) => panic!("le cran {k} (ℓ = 8) aurait dû être refusé"),
            }
        }
        // un cran hors domaine fait échouer toute la mesure, et la liste mélangée aussi
        assert!(neutre::mesurer_ps(&g, &u, &p, &[1, 2, 8]).is_err());
        assert!(neutre::mesurer_ps(&g, &u, &p, &[2, -4]).is_err());
        assert!(neutre::mesurer_ps(&g, &u, &p, &[1, 2, 3]).is_ok());
    }

    // ------------------------------------------------------------------
    // impulsions HORS AXE (`--impulsions-vecteurs 2:2:1,3:0:0`) : moyenne sur l'orbite cubique du vecteur.
    // V1 axe = ancienne option AU BIT, V2 covariance par translation, V3 invariance de la moyenne d'orbite par le groupe
    // cubique, V4 définition de l'orbite, V5 refus propres. (V6, carte contre processeur : tests/su2_spectro_gpu.rs.)
    // ------------------------------------------------------------------

    fn plat_orb(m: &neutre::MesureNeutre) -> Vec<f64> {
        m.corr_orb.iter().flat_map(|c| c.iter().flatten().copied()).collect()
    }

    /// V4 : tailles d'orbite et représentants.
    #[test]
    fn orbites_definition() {
        let o = neutre::orbite([1, 2, 2]);
        assert_eq!((o.representant, o.taille, o.vecteurs.len()), ([2, 2, 1], 24, 12));
        let o = neutre::orbite([0, -3, 0]);
        assert_eq!((o.representant, o.taille), ([3, 0, 0], 6));
        assert_eq!(o.vecteurs, vec![[3, 0, 0], [0, 3, 0], [0, 0, 3]], "ordre des axes de --impulsions");
        assert_eq!(neutre::orbite([2, 1, 0]).taille, 24);
        assert_eq!(neutre::orbite([2, 1, 1]).taille, 24);
        assert_eq!(neutre::orbite([2, 2, 0]).taille, 12);
        assert_eq!(neutre::orbite([3, 2, 1]).taille, 48);
        assert_eq!(neutre::orbite([1, 1, 1]).taille, 8);
        // les représentants sont deux à deux distincts modulo ± et couvrent l'orbite complète (avec leurs opposés)
        let o = neutre::orbite([2, 2, 1]);
        for v in &o.vecteurs {
            assert!(v.iter().map(|x| x * x).sum::<i64>() == 9);
            let m = [-v[0], -v[1], -v[2]];
            assert!(!o.vecteurs.contains(&m));
        }
    }

    /// V1 : pour k le long d'un axe, l'orbite (k,0,0) redonne EXACTEMENT (mêmes bits) les nombres de `--impulsions k`,
    /// et la sortie d'une orbite ne dépend pas du reste de la liste ; le chemin sans orbite reste celui d'avant.
    #[test]
    fn vecteurs_axe_au_bit_pres() {
        let (g, u) = config([8, 8, 8, 6], 2.3, 47, 6);
        let p = pp(vec![0, 2]);
        let ks = [1i64, 2, 3];
        let ref_ps = neutre::mesurer_ps(&g, &u, &p, &ks).unwrap();
        let orbs: Vec<_> = ks.iter().map(|&k| neutre::orbite([k, 0, 0])).collect();
        let a = neutre::mesurer_orb(&g, &u, &p, &[], &orbs).unwrap();
        assert!(plat_orb(&a).iter().any(|x| x.abs() > 1e-6));
        for (io, _) in ks.iter().enumerate() {
            for (pa, pb) in a.corr_orb[io].iter().zip(&ref_ps.corr_ps[io]) {
                for (x, y) in pa.iter().zip(pb) {
                    assert_eq!(x.to_bits(), y.to_bits(), "cran {} : {x} contre {y}", ks[io]);
                }
            }
        }
        // l'impulsion nulle et les moyennes restent celles de `mesurer` au bit près
        let z = neutre::mesurer(&g, &u, &p).unwrap();
        assert!(a.s.iter().flatten().zip(z.s.iter().flatten()).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert!(a.corr.iter().flatten().zip(z.corr.iter().flatten()).all(|(x, y)| x.to_bits() == y.to_bits()));
        // indépendance vis-à-vis du reste de la liste (une orbite hors axe en plus, et ks en plus)
        let mix = [neutre::orbite([2, 2, 1]), neutre::orbite([2, 0, 0])];
        let b = neutre::mesurer_orb(&g, &u, &p, &[1], &mix).unwrap();
        let seul = neutre::mesurer_orb(&g, &u, &p, &[], &mix[1..]).unwrap();
        assert_eq!(
            b.corr_orb[1].iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>(),
            a.corr_orb[1].iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            seul.corr_orb[0].iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>(),
            a.corr_orb[1].iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        // le premier résultat reste `mesurer_ps` pour les crans
        assert!(b.corr_ps[0].iter().flatten().zip(neutre::mesurer_ps(&g, &u, &p, &[1]).unwrap().corr_ps[0].iter().flatten()).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    /// V2 : décaler le champ d'un site le long d'un axe spatial a multiplie S(t; k) par e^{+i 2π k_a/ℓ} ; la moyenne d'orbite
    /// (corrélateur) est inchangée ; et le champ n'est pas trivial.
    #[test]
    fn vecteurs_translation_covariance() {
        let (g, u) = config([6, 6, 6, 6], 2.3, 53, 6);
        let imgs = neutre::images(FORMES[1].1);
        let vs = [[2i64, 2, 1], [2, -2, 1], [1, 0, 2], [0, 0, 1]];
        let (_, s0) = neutre::tranches_vec(&g, &u, 3, &imgs, &vs);
        assert!(s0.iter().flatten().any(|z| z[0].abs() + z[1].abs() > 1e-6));
        for axe in 0..3 {
            let mut u2 = u.clone();
            for mu in 0..4 {
                for s in 0..g.v {
                    let mut x = g.coords(s);
                    x[axe] = (x[axe] + 1) % g.d[axe];
                    u2[mu * g.v + g.site(&x)] = u[mu * g.v + s];
                }
            }
            let (_, s1) = neutre::tranches_vec(&g, &u2, 3, &imgs, &vs);
            for (t, (a, b)) in s0.iter().zip(&s1).enumerate() {
                for (iv, v) in vs.iter().enumerate() {
                    let th = 2.0 * PI * v[axe] as f64 / 6.0;
                    let (c, sn) = (th.cos(), th.sin());
                    let att = [a[iv][0] * c - a[iv][1] * sn, a[iv][0] * sn + a[iv][1] * c];
                    assert!((att[0] - b[iv][0]).abs() < 1e-12 && (att[1] - b[iv][1]).abs() < 1e-12, "axe {axe} t {t} v {v:?} : {att:?} contre {:?}", b[iv]);
                }
            }
            let p = pp(vec![0, 2]);
            let orbs = [neutre::orbite([2, 2, 1]), neutre::orbite([3, 0, 0])];
            let a = neutre::mesurer_orb(&g, &u, &p, &[], &orbs[..1]).unwrap();
            let b = neutre::mesurer_orb(&g, &u2, &p, &[], &orbs[..1]).unwrap();
            for (x, y) in plat_orb(&a).iter().zip(plat_orb(&b)) {
                assert!((x - y).abs() < 1e-12, "{x} contre {y}");
            }
        }
    }

    /// Applique l'élément (perm, signes) du groupe cubique aux trois axes spatiaux d'une configuration (réflexions comprises :
    /// un lien orienté contre l'axe s'écrit comme le dagger du lien opposé).
    fn transformer_cubique(g: &GeoA, u: &[Q], perm: [usize; 3], sg: [i64; 3]) -> Vec<Q> {
        let mut o = u.to_vec();
        for s in 0..g.v {
            let x = g.coords(s);
            let mut y = x;
            for a in 0..3 {
                let l = g.d[a] as i64;
                y[perm[a]] = if sg[a] > 0 { x[a] } else { ((l - x[a] as i64) % l) as usize };
            }
            for a in 0..3 {
                let b = perm[a];
                if sg[a] > 0 {
                    o[b * g.v + g.site(&y)] = u[a * g.v + s];
                } else {
                    let mut z = y;
                    z[b] = (y[b] + g.d[b] - 1) % g.d[b];
                    o[b * g.v + g.site(&z)] = crate::heritage::su2_hb::qdag(u[a * g.v + s]);
                }
            }
            o[3 * g.v + g.site(&y)] = u[3 * g.v + s];
        }
        o
    }

    /// V3 : la moyenne d'orbite est invariante quand on applique à la configuration un élément du groupe cubique (permutation des
    /// axes, réflexion, rotation de 90°) ; elle ne l'est PAS vecteur par vecteur (le test ne serait pas discriminant sinon).
    #[test]
    fn vecteurs_orbite_invariante_cubique() {
        let (g, u) = config([6, 6, 6, 8], 2.3, 59, 6);
        let p = pp(vec![0, 2]);
        let orbs = [neutre::orbite([2, 2, 1]), neutre::orbite([2, 1, 0]), neutre::orbite([2, 0, 0])];
        let a = neutre::mesurer_orb(&g, &u, &p, &[], &orbs).unwrap();
        assert!(plat_orb(&a).iter().any(|x| x.abs() > 1e-7));
        let elements: [([usize; 3], [i64; 3]); 5] =
            [([1, 2, 0], [1, 1, 1]), ([0, 1, 2], [-1, 1, 1]), ([1, 0, 2], [1, -1, 1]), ([2, 1, 0], [-1, -1, 1]), ([2, 0, 1], [-1, 1, -1])];
        for (perm, sg) in elements {
            let u2 = transformer_cubique(&g, &u, perm, sg);
            // la transformation est une symétrie du réseau : l'impulsion nulle (invariante par O_h) est inchangée
            let (z1, z2) = (neutre::mesurer(&g, &u, &p).unwrap(), neutre::mesurer(&g, &u2, &p).unwrap());
            assert!(z1.corr.iter().flatten().zip(z2.corr.iter().flatten()).all(|(x, y)| (x - y).abs() < 1e-12));
            let b = neutre::mesurer_orb(&g, &u2, &p, &[], &orbs).unwrap();
            for (x, y) in plat_orb(&a).iter().zip(plat_orb(&b)) {
                assert!((x - y).abs() < 1e-12, "{perm:?} {sg:?} : {x} contre {y}");
            }
        }
        // discriminant : un vecteur isolé (2,2,1) n'est pas invariant par permutation des axes
        let imgs = neutre::images(FORMES[3].1);
        let (_, s0) = neutre::tranches_vec(&g, &u, 3, &imgs, &[[2, 2, 1]]);
        let (_, s1) = neutre::tranches_vec(&g, &transformer_cubique(&g, &u, [1, 2, 0], [1, 1, 1]), 3, &imgs, &[[2, 2, 1]]);
        let d: f64 = s0.iter().zip(&s1).map(|(x, y)| (x[0][0] - y[0][0]).abs() + (x[0][1] - y[0][1]).abs()).sum();
        assert!(d > 1e-6, "le test d'invariance ne discrimine pas ({d})");
    }

    /// V5 : refus propres (vecteur nul, hors de la zone de Brillouin |k_d| < ℓ/2, côtés spatiaux inégaux, orbite en double) ;
    /// admissibles : toutes les composantes < ℓ/2, signes quelconques.
    #[test]
    fn vecteurs_refus() {
        let g = GeoA::nouvelle([8, 8, 8, 8]).unwrap();
        let u = vec![UN; 4 * g.v];
        let p = pp(vec![0]);
        for k in [[2i64, 2, 1], [3, 0, 0], [-3, 3, -3], [0, 0, 1]] {
            assert!(neutre::mesurer_orb(&g, &u, &p, &[], &[neutre::orbite(k)]).is_ok(), "{k:?} admissible (ℓ = 8)");
        }
        for k in [[0i64, 0, 0], [4, 0, 0], [0, 5, 1], [1, 1, -4], [9, 2, 1]] {
            match neutre::controler_vecteurs(&g, 3, &[k]) {
                Err(Erreur::Config(t)) => assert!(t.contains("impulsions-vecteurs"), "{t}"),
                other => panic!("{k:?} aurait dû être refusé : {other:?}"),
            }
        }
        assert!(neutre::controler_vecteurs(&g, 3, &[[2, 2, 1], [1, -2, 2]]).is_err(), "même orbite deux fois");
        assert_eq!(neutre::controler_vecteurs(&g, 3, &[[2, 2, 1], [3, 0, 0]]).unwrap().len(), 2);
        // une orbite brute hors domaine fait aussi échouer `mesurer_orb`
        let mauvais = neutre::Orbite { representant: [4, 0, 0], taille: 6, vecteurs: vec![[4, 0, 0]] };
        assert!(neutre::mesurer_orb(&g, &u, &p, &[], &[mauvais]).is_err());
        // côtés spatiaux inégaux : pas d'orbite cubique
        let g2 = GeoA::nouvelle([8, 6, 8, 8]).unwrap();
        assert!(neutre::controler_vecteurs(&g2, 3, &[[1, 0, 0]]).is_err());
        // le temps peut être n'importe lequel des quatre axes : [6,8,8,8] avec temps = 0 est valide
        let g3 = GeoA::nouvelle([6, 8, 8, 8]).unwrap();
        assert!(neutre::controler_vecteurs(&g3, 0, &[[3, 2, 1]]).is_ok());
    }


    /// V7 : avec blocage (niveaux APE puis B1), l'orbite (k,0,0) redonne `--impulsions k` (écart < 1e-12 : la phase est calculée
    /// autrement) et la moyenne d'orbite hors axe est invariante par le groupe cubique.
    #[test]
    fn vecteurs_avec_blocage() {
        use crate::su2::spectro::bloc::{mesurer_avec_blocs, mesurer_avec_blocs_orb, ParamBloc};
        let (g, u) = config([8, 8, 8, 8], 2.3, 61, 6);
        let p = pp(vec![0, 2]);
        let pb = ParamBloc { niveaux: vec![1], ape_bloc: 1, lissage_bloc: 1, p_bloc: 1.0 };
        let r = mesurer_avec_blocs(&g, &u, &p, &[1, 3], &pb).unwrap();
        let orbs = [neutre::orbite([1, 0, 0]), neutre::orbite([3, 0, 0]), neutre::orbite([2, 2, 1])];
        let a = mesurer_avec_blocs_orb(&g, &u, &p, &[], &orbs, &pb).unwrap();
        for (io, ik) in [(0usize, 0usize), (1, 1)] {
            for (x, y) in a.corr_orb[io].iter().flatten().zip(r.corr_ps[ik].iter().flatten()) {
                assert!((x - y).abs() < 1e-12, "orbite {io} : {x} contre {y}");
            }
        }
        assert!(a.corr_orb[2].iter().flatten().any(|x| x.abs() > 1e-7));
        let u2 = transformer_cubique(&g, &u, [2, 0, 1], [-1, 1, -1]);
        let b = mesurer_avec_blocs_orb(&g, &u2, &p, &[], &orbs, &pb).unwrap();
        for (x, y) in plat_orb(&a).iter().zip(plat_orb(&b)) {
            assert!((x - y).abs() < 1e-11, "{x} contre {y}");
        }
        // sans orbite : identique au bit à l'ancienne entrée
        let z = mesurer_avec_blocs_orb(&g, &u, &p, &[1, 3], &[], &pb).unwrap();
        assert!(z.corr_ps.iter().flatten().flatten().zip(r.corr_ps.iter().flatten().flatten()).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

}

/// `--ops-bruts` sur carte (porte iii de DEUX_BOULES.md §2) : translater le champ d'un pas selon un axe spatial multiplie
/// O_a(p,t) par e^{-i p·δ} (p le long de cet axe), et ne change ni p = 0 ni les impulsions des autres axes.
/// `#[ignore]` : exige une carte (`cargo test --release --features cuda -p kvu -- --ignored ops_bruts`).
#[test]
#[ignore]
fn ops_bruts_translation_carte() {
    use super::gpu::{axes_spatiaux, ParamGpu, SpectroGpu};
    use crate::commun::Precision;
    let d = [8usize, 8, 8, 16];
    let app = crate::su2::cli::ouvrir_backend("cuda").unwrap().expect("GPU requis");
    let pg = ParamGpu {
        cotes: d,
        temps: 3,
        niveaux: vec![2],
        alpha: 0.5,
        formes: vec!["S22".to_string()],
        tau_max: 2,
        ks: vec![1],
        precision: Precision::F64,
        mutant: 0,
        nor: 4,
        reps: vec![],
        betas: vec![2.39],
        graines: vec![99],
        bloc: super::bloc::ParamBloc { niveaux: vec![1], ape_bloc: 1, lissage_bloc: 1, p_bloc: 1.0 },
        ops_bruts: true,
        xi0: None,
        orbites: Vec::new(),
    };
    let mut sg = SpectroGpu::nouveau(&app, &pg).unwrap();
    sg.init().unwrap();
    sg.avancer(30).unwrap();
    let a = sg.mesurer().unwrap()[0].ops.clone();
    let g = GeoA::nouvelle(d).unwrap();
    let raw = sg.boite.telecharger(0).unwrap();
    let q: Vec<f64> = raw.clone();
    let ax = axes_spatiaux(3)[0];
    let mut out = vec![0f64; q.len()];
    for mu in 0..4 {
        for s in 0..g.v {
            let mut x = g.coords(s);
            x[ax] = (x[ax] + 1) % d[ax]; // nouveau[x + δ] = ancien[x]
            let s2 = g.site(&x);
            for k in 0..4 {
                out[(mu * g.v + s2) * 4 + k] = q[(mu * g.v + s) * 4 + k];
            }
        }
    }
    sg.boite.televerser(0, &out).unwrap();
    let b = sg.mesurer().unwrap()[0].ops.clone();
    let imps = sg.ops_impulsions();
    let nimp = imps.len();
    let nops = sg.nops();
    let sc = a.iter().fold(0f64, |m, x| m.max(x.abs()));
    let mut ecart = 0f64;
    for t in 0..d[3] {
        for (ip, p) in imps.iter().enumerate() {
            let ph = 2.0 * PI * (p[ax] as f64) / d[ax] as f64; // O'(p) = e^{-i p δ} O(p), δ = 1
            let (c, s) = (ph.cos(), -ph.sin());
            for o in 0..nops {
                let i = ((t * nimp + ip) * nops + o) * 2;
                let (re, im) = (a[i], a[i + 1]);
                let (er, ei) = (re * c - im * s, re * s + im * c);
                ecart = ecart.max((b[i] - er).abs().max((b[i + 1] - ei).abs()));
            }
        }
    }
    eprintln!("ops_bruts_translation_carte : ecart max / echelle = {:.3e} (echelle {sc:.3e})", ecart / sc);
    assert!(ecart / sc < 1e-12);
}

// ---------------------------------------------------------------------------------------------------------------------------
// PROFIL DU TUBE (--profil) : règle de somme, identité de somme sur Δ, symétrie, et cartes carte-contre-processeur.
// ---------------------------------------------------------------------------------------------------------------------------
mod profil_tests {
    use super::*;
    use crate::su2::spectro::profil::{phi_cpu, rho_cpu, Accu, Cartes};

    fn champ_plat(g: &GeoA, u: &[Q]) -> Vec<f64> {
        let _ = g;
        u.iter().flat_map(|q| q.iter().copied()).collect()
    }

    fn action_totale(d: [usize; 4], u: &[Q]) -> f64 {
        let g = GeoA::nouvelle(d).unwrap();
        let v = g.v;
        let mut tot = 0.0;
        for s in 0..v {
            let x = g.coords(s);
            for mu in 0..4 {
                for nu in mu + 1..4 {
                    let mut xm = x;
                    xm[mu] = (x[mu] + 1) % d[mu];
                    let mut xn = x;
                    xn[nu] = (x[nu] + 1) % d[nu];
                    let p = qmul(qmul(u[mu * v + s], u[nu * v + g.site(&xm)]), qmul(qdag(u[mu * v + g.site(&xn)]), qdag(u[nu * v + s])));
                    tot += 1.0 - p[0];
                }
            }
        }
        tot
    }

    #[test]
    fn regle_de_somme_cpu() {
        let d = [4usize, 4, 4, 6];
        let (g, u) = config(d, 2.39, 5, 12);
        let q = champ_plat(&g, &u);
        let rho = rho_cpu(d, &q, 0, 3, [1, 2]);
        let somme: f64 = rho.iter().sum();
        let ref_ = action_totale(d, &u);
        eprintln!("regle de somme CPU : {somme:.12} contre {ref_:.12}");
        assert!((somme - ref_).abs() < 1e-9 * ref_);
        // phi_cpu = polyakov brut moyen cohérent avec la définition (1/2 Tr) : borné par 1
        let ph = phi_cpu(d, &q, 0, 3, [1, 2]);
        assert!(ph.iter().all(|x| x.abs() <= 1.0 + 1e-12));
    }

    /// Σ_Δ N(Δ;τ) = Σ_t (Σ_x Φ(x,t)Φ(x,t+τ)) · A_ρ(t+τ/2), avec A_ρ(t) = Σ_y ρ(y,t) : identité exacte (impulsion nulle sur le puits).
    #[test]
    fn identite_somme_delta_et_symetrie() {
        let (ta, tb, tt) = (5usize, 4usize, 6usize);
        let nab = ta * tb;
        let mut graine = 12345u64;
        let mut alea = || {
            graine = graine.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((graine >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        // Séparations paires et impaires ; pour τ = 5 et t = 3, les tranches médianes sont 5 et 0 (modulo 6).
        let taus = [1usize, 2, 3, 4, 5];
        let mut acc = Accu::nouveau(ta, tb, tt, &taus);
        let mut verif = vec![0f64; taus.len() * 2];
        for _ in 0..3 {
            let c = Cartes { phi: (0..tt * nab).map(|_| alea()).collect(), rho: (0..2 * tt * nab).map(|_| alea() + 1.0).collect() };
            acc.ajouter(&c);
            for (it, &tau) in taus.iter().enumerate() {
                for k in 0..2 {
                    for t in 0..tt {
                        let sx: f64 = (0..nab).map(|i| c.phi[t * nab + i] * c.phi[((t + tau) % tt) * nab + i]).sum();
                        let a = |tp: usize| c.rho[(k * tt + tp % tt) * nab..(k * tt + tp % tt + 1) * nab].iter().sum::<f64>();
                        // oracle INDÉPENDANT du code testé : tranche basse t + ⌊τ/2⌋, tranche haute t + ⌊τ/2⌋ + (τ mod 2) (et non div_ceil)
                        verif[it * 2 + k] += sx * 0.5 * (a(t + tau / 2) + a(t + tau / 2 + tau % 2));
                    }
                }
            }
        }
        for it in 0..taus.len() {
            for k in 0..2 {
                let s: f64 = acc.n[(it * 2 + k) * nab..(it * 2 + k + 1) * nab].iter().sum();
                assert!((s - verif[it * 2 + k]).abs() < 1e-10 * verif[it * 2 + k].abs().max(1.0), "tau {} k {k}: {s} vs {}", taus[it], verif[it * 2 + k]);
            }
        }
        // Δ → −Δ : avec une source δ(x0) et un puits symétrique ρ(y) = ρ(−y) autour de x0, N(Δ) = N(−Δ)
        let mut acc2 = Accu::nouveau(ta, tb, tt, &[2]);
        let mut phi = vec![0f64; tt * nab];
        for t in 0..tt {
            phi[t * nab + tb + 2] = 1.0; // x0 = (1,2)
        }
        let mut rho = vec![0f64; 2 * tt * nab];
        for k in 0..2 {
            for t in 0..tt {
                for a in 0..ta {
                    for b in 0..tb {
                        let (da, db) = (((a + ta - 1) % ta).min((1 + ta - a) % ta), ((b + tb - 2) % tb).min((2 + tb - b) % tb));
                        rho[(k * tt + t) * nab + a * tb + b] = (-((da * da + db * db) as f64) / 2.0).exp();
                    }
                }
            }
        }
        acc2.ajouter(&Cartes { phi, rho });
        for da in 0..ta {
            for db in 0..tb {
                let (ma, mb) = ((ta - da) % ta, (tb - db) % tb);
                let (x, y) = (acc2.n[da * tb + db], acc2.n[ma * tb + mb]);
                assert!((x - y).abs() < 1e-12, "asymétrie ({da},{db}) : {x} vs {y}");
            }
        }
    }

    /// carte contre processeur : `#[ignore]`, exige une carte HIP ou CUDA compatible ; écart ≤ 1e-6 en f32.
    #[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
    fn carte_contre_cpu(backend: &str, prec: crate::commun::Precision, tol: f64) {
        use crate::su2::spectro::gpu_torelon::{ParamTorelon, TorelonGpu};
        let d = [6usize, 6, 6, 8];
        let app = crate::su2::cli::ouvrir_backend(backend).unwrap().expect("GPU requis");
        let (w, temps, tr) = (0usize, 3usize, [1usize, 2usize]);
        for niv in [0usize, 1] {
            let p = ParamTorelon {
                cotes: d,
                w,
                temps,
                niveaux: vec![0, 2],
                alpha: 0.5,
                moments: vec![[0, 0]],
                tau_max: 3,
                precision: prec,
                nor: 4,
                betas: vec![2.39, 2.39],
                graines: vec![7, 8],
                bloc: super::super::bloc::ParamBloc::vide(),
                xi0: None,
                ops_bruts: false,
                profil: vec![1, 2, 3],
                profil_niveau: Some(niv),
            };
            let mut tg = TorelonGpu::nouveau(&app, &p).unwrap();
            tg.init().unwrap();
            tg.avancer(40).unwrap();
            let res = tg.mesurer().unwrap();
            for c in 0..2 {
                let raw = tg.boite.telecharger(c).unwrap();
                let mut u: Vec<Q> = raw.chunks(4).map(|x| [x[0], x[1], x[2], x[3]]).collect();
                let g = GeoA::nouvelle(d).unwrap();
                if niv == 1 {
                    for _ in 0..2 {
                        u = ape_iteration(&g, &u, temps, 0.5);
                    }
                }
                let qs = champ_plat(&g, &u);
                let phi_ref = phi_cpu(d, &qs, w, temps, tr);
                let rho_ref = rho_cpu(d, &raw, w, temps, tr);
                let car = res[c].cartes.as_ref().unwrap();
                let e1 = phi_ref.iter().zip(&car.phi).fold(0f64, |m, (a, b)| m.max((a - b).abs()));
                let e2 = rho_ref.iter().zip(&car.rho).fold(0f64, |m, (a, b)| m.max((a - b).abs()));
                let sr = rho_ref.iter().fold(0f64, |m, x| m.max(x.abs()));
                // règle de somme : Σ ρ = 6V(1 − plaq)
                let v = d.iter().product::<usize>() as f64;
                let somme: f64 = car.rho.iter().sum();
                let attendu = 6.0 * v * (1.0 - res[c].plaq);
                eprintln!("{backend} {prec:?} niveau {niv} chaine {c} : ecart phi {e1:.3e}, ecart rho {e2:.3e} (echelle {sr:.2}), somme rho {somme:.9} vs 6V(1-plaq) {attendu:.9}");
                assert!(e1 < tol && e2 < tol);
                assert!((somme - attendu).abs() < tol * attendu);
            }
        }
    }

    #[test]
    #[ignore]
    fn carte_contre_cpu_f64_hip() {
        carte_contre_cpu("hip", crate::commun::Precision::F64, 1e-10);
    }
    #[test]
    #[ignore]
    fn carte_contre_cpu_f32_hip() {
        carte_contre_cpu("hip", crate::commun::Precision::F32, 1e-6);
    }
    #[test]
    #[ignore]
    fn carte_contre_cpu_f64_cuda() {
        carte_contre_cpu("cuda", crate::commun::Precision::F64, 1e-10);
    }
    #[test]
    #[ignore]
    fn carte_contre_cpu_f32_cuda() {
        carte_contre_cpu("cuda", crate::commun::Precision::F32, 1e-6);
    }
}
