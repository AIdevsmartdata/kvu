//! Fiche USp(4), J = [[0,I],[-I,0]], générateurs hermitiens de trace carrée 1/2.

use super::generique::{Mat, Noyau};
use super::{produit, quotient, Appartenance, Cmat, Flux, Groupe, LogU, Realite};
use crate::heritage::su2_hb::{philox4x32_10, u01};
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, PI, TAU};

type M = Mat<4>;
const C: f64 = 0.5 * FRAC_1_SQRT_2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutantSp4 {
    Aucun,
    GenerateurRetire,
    QrNaif,
}

pub struct Sp4 {
    mutant: MutantSp4,
}
impl Default for Sp4 {
    fn default() -> Self {
        Self::nouveau()
    }
}
impl Sp4 {
    pub fn nouveau() -> Self {
        Self {
            mutant: MutantSp4::Aucun,
        }
    }
    pub fn avec_mutant(mutant: MutantSp4) -> Self {
        Self { mutant }
    }
}

/// Construction directe des blocs P, Q ; aucun produit matriciel dense.
fn algebre(c: &[f64]) -> M {
    assert_eq!(c.len(), 10, "Sp(4) exige dix coefficients");
    let mut a = M::zero();
    a.re[0][0] = C * (c[0] + c[3]);
    a.re[1][1] = C * (c[0] - c[3]);
    a.re[0][1] = C * c[1];
    a.im[0][1] = -C * c[2];
    a.re[1][0] = C * c[1];
    a.im[1][0] = C * c[2];
    for i in 0..2 {
        for j in 0..2 {
            a.re[i + 2][j + 2] = -a.re[i][j];
            a.im[i + 2][j + 2] = a.im[i][j];
        }
    }
    a.re[0][2] = 0.5 * c[4];
    a.im[0][2] = 0.5 * c[5];
    a.re[1][3] = 0.5 * c[6];
    a.im[1][3] = 0.5 * c[7];
    a.re[0][3] = C * c[8];
    a.im[0][3] = C * c[9];
    a.re[1][2] = C * c[8];
    a.im[1][2] = C * c[9];
    for i in 0..2 {
        for j in 0..2 {
            a.re[i + 2][j] = a.re[i][j + 2];
            a.im[i + 2][j] = -a.im[i][j + 2];
        }
    }
    a
}

#[inline]
fn max_module(a: &M) -> f64 {
    let mut norme: f64 = 0.;
    for i in 0..4 {
        for j in 0..4 {
            norme = norme.max(a.re[i][j].hypot(a.im[i][j]));
        }
    }
    norme
}

/// J conj(M) Jᵀ (involution quaternionique).
fn conjugaison_j(a: &M) -> M {
    let mut b = M::zero();
    for i in 0..4 {
        for j in 0..4 {
            let signe = if (i < 2) == (j < 2) { 1. } else { -1. };
            b.re[i][j] = signe * a.re[(i + 2) % 4][(j + 2) % 4];
            b.im[i][j] = -signe * a.im[(i + 2) % 4][(j + 2) % 4];
        }
    }
    b
}

fn defauts(u: &M) -> (f64, f64) {
    let unitarite = max_module(&u.dag().mul(u).sub(&M::identite()));
    // Calcul explicite de Uᵀ J U, sans supposer l'unitarité de U.
    let mut structure: f64 = 0.;
    for i in 0..4 {
        for j in 0..4 {
            let mut z = (0., 0.);
            for k in 0..2 {
                let a = produit((u.re[k][i], u.im[k][i]), (u.re[k + 2][j], u.im[k + 2][j]));
                let b = produit((u.re[k + 2][i], u.im[k + 2][i]), (u.re[k][j], u.im[k][j]));
                z.0 += a.0 - b.0;
                z.1 += a.1 - b.1;
            }
            if i < 2 && j == i + 2 {
                z.0 -= 1.;
            }
            if i >= 2 && j == i - 2 {
                z.0 += 1.;
            }
            structure = structure.max(z.0.hypot(z.1));
        }
    }
    (unitarite, structure)
}

fn normaliser(m: &mut M, j: usize) {
    let mut norme = 0.;
    for i in 0..4 {
        norme += m.re[i][j] * m.re[i][j] + m.im[i][j] * m.im[i][j];
    }
    let norme = norme.sqrt();
    assert!(norme > 1e-300, "Colonne nulle pendant le QR symplectique");
    for i in 0..4 {
        m.re[i][j] /= norme;
        m.im[i][j] /= norme;
    }
}

fn partenaire(m: &mut M, j: usize) {
    for i in 0..2 {
        m.re[i][j + 2] = -m.re[i + 2][j];
        m.im[i][j + 2] = m.im[i + 2][j];
        m.re[i + 2][j + 2] = m.re[i][j];
        m.im[i + 2][j + 2] = -m.im[i][j];
    }
}

fn orthogonaliser(m: &mut M, j: usize, k: usize) {
    let mut z = (0., 0.);
    for i in 0..4 {
        z.0 += m.re[i][k] * m.re[i][j] + m.im[i][k] * m.im[i][j];
        z.1 += m.re[i][k] * m.im[i][j] - m.im[i][k] * m.re[i][j];
    }
    for i in 0..4 {
        let p = produit((m.re[i][k], m.im[i][k]), z);
        m.re[i][j] -= p.0;
        m.im[i][j] -= p.1;
    }
}

fn qr_symplectique(u: &mut M) {
    normaliser(u, 0);
    partenaire(u, 0);
    for _ in 0..2 {
        orthogonaliser(u, 1, 0);
        orthogonaliser(u, 1, 2);
    }
    normaliser(u, 1);
    partenaire(u, 1);
}

fn inverse(a: &M) -> Option<M> {
    let mut a = *a;
    let mut b = M::identite();
    for k in 0..4 {
        let p = (k..4).max_by(|&i, &j| {
            a.re[i][k]
                .hypot(a.im[i][k])
                .total_cmp(&a.re[j][k].hypot(a.im[j][k]))
        })?;
        if a.re[p][k].hypot(a.im[p][k]) < 1e-300 {
            return None;
        }
        a.re.swap(k, p);
        a.im.swap(k, p);
        b.re.swap(k, p);
        b.im.swap(k, p);
        let pivot = (a.re[k][k], a.im[k][k]);
        for j in 0..4 {
            (a.re[k][j], a.im[k][j]) = quotient((a.re[k][j], a.im[k][j]), pivot);
            (b.re[k][j], b.im[k][j]) = quotient((b.re[k][j], b.im[k][j]), pivot);
        }
        for i in 0..4 {
            if i == k {
                continue;
            }
            let z = (a.re[i][k], a.im[i][k]);
            for j in 0..4 {
                let p = produit(z, (a.re[k][j], a.im[k][j]));
                a.re[i][j] -= p.0;
                a.im[i][j] -= p.1;
                let p = produit(z, (b.re[k][j], b.im[k][j]));
                b.re[i][j] -= p.0;
                b.im[i][j] -= p.1;
            }
        }
    }
    Some(b)
}

/// Différences divisées de cos√s et sinc√s, sans annulation près d'une racine double.
fn pentes(s1: f64, s2: f64) -> (f64, f64) {
    let milieu = 0.5 * (s1 + s2);
    if milieu < 1. {
        // Σ_n a_n Σ_j s1^j s2^(n−1−j), avec récurrence des polynômes complets.
        let mut f = 0.;
        let mut g = 0.;
        let mut af = 1.;
        let mut ag = 1.;
        let mut h = 1.;
        let mut puissance = 1.;
        for n in 1..=24 {
            af /= -((2 * n - 1) * (2 * n)) as f64;
            ag /= -((2 * n) * (2 * n + 1)) as f64;
            f += af * h;
            g += ag * h;
            puissance *= s2;
            h = s1 * h + puissance;
        }
        return (f, g);
    }
    if (s1 - s2).abs() < 1e-2 * (s1 + s2).max(1e-300) {
        // Série centrée : les mêmes polynômes en (s1−m,s2−m).
        // f' = −g/2 et 2s g' = f−g donnent les coefficients de Taylor.
        let x = milieu.sqrt();
        let (sin, cos) = x.sin_cos();
        let mut af = cos;
        let mut ag = sin / x;
        let h2 = (0.5 * (s1 - s2)).powi(2);
        let mut puissance = 1.;
        let mut f = 0.;
        let mut g = 0.;
        for n in 1..=12 {
            let suivant_f = -ag / (2 * n) as f64;
            let suivant_g = (af - (2 * n - 1) as f64 * ag) / ((2 * n) as f64 * milieu);
            af = suivant_f;
            ag = suivant_g;
            if n % 2 == 1 {
                f += af * puissance;
                g += ag * puissance;
                puissance *= h2;
            }
        }
        (f, g)
    } else {
        let x = s1.sqrt();
        let y = s2.sqrt();
        let sinc = |x: f64| if x == 0. { 1. } else { x.sin() / x };
        (
            (x.cos() - y.cos()) / (s1 - s2),
            (sinc(x) - sinc(y)) / (s1 - s2),
        )
    }
}

/// Réflexion d'un seul sous-groupe ; W est recalculé depuis les lignes courantes.
fn reflechir(u: &mut M, a: &M, i: usize, j: usize) {
    let element = |r: usize, c: usize| {
        let mut z = (0., 0.);
        for k in 0..4 {
            let p = produit((u.re[r][k], u.im[r][k]), (a.re[k][c], a.im[k][c]));
            z.0 += p.0;
            z.1 += p.1;
        }
        z
    };
    let wii = element(i, i);
    let wjj = element(j, j);
    let wij = element(i, j);
    let wji = element(j, i);
    let n = [wii.0 + wjj.0, -wii.1 + wjj.1, wji.0 - wij.0, -wji.1 - wij.1];
    let kappa = n.iter().fold(0_f64, |acc, x| acc.hypot(*x));
    if kappa <= 1e-300 {
        return;
    }
    let n = n.map(|x| x / kappa);
    let alpha = (
        n[0] * n[0] - n[1] * n[1] - n[2] * n[2] - n[3] * n[3],
        2. * n[0] * n[1],
    );
    let beta = (2. * n[0] * n[2], 2. * n[0] * n[3]);
    for k in 0..4 {
        let vi = (u.re[i][k], u.im[i][k]);
        let vj = (u.re[j][k], u.im[j][k]);
        let p = produit(alpha, vi);
        let q = produit(beta, vj);
        let r = produit((-beta.0, beta.1), vi);
        let s = produit((alpha.0, -alpha.1), vj);
        u.re[i][k] = p.0 + q.0;
        u.im[i][k] = p.1 + q.1;
        u.re[j][k] = r.0 + s.0;
        u.im[j][k] = r.1 + s.1;
    }
}

impl Noyau<4> for Sp4 {
    fn nom(&self) -> &'static str {
        "sp4"
    }
    fn d_adj(&self) -> usize {
        10
    }
    fn exp_i(&self, c: &[f64]) -> M {
        let a = algebre(c);
        let a2 = a.mul(&a);
        let somme = a2.re_tr() * 0.5;
        if somme == 0. {
            return M::identite();
        }
        // B = A² − S/2 : Tr(B²) = Δ, sans soustraction de deux traces proches.
        let mut b = a2;
        for i in 0..4 {
            b.re[i][i] -= 0.5 * somme;
        }
        let delta = b.re_tr_prod(&b).max(0.).sqrt();
        let s1 = 0.5 * (somme + delta);
        let s2 = (0.5 * (somme - delta)).max(0.);
        let (f2, g3) = pentes(s1, s2);
        let x = s1.sqrt();
        let y = s2.sqrt();
        let sinc = |z: f64| if z == 0. { 1. } else { z.sin() / z };
        // Même polynôme de Cayley–Hamilton, évalué autour de S/2.
        let f_milieu = 0.5 * (x.cos() + y.cos());
        let g_milieu = 0.5 * (sinc(x) + sinc(y));
        let mut impair = b.echelle(g3);
        for i in 0..4 {
            impair.re[i][i] += g_milieu;
        }
        let impair = a.mul(&impair);
        let mut u = b.echelle(f2);
        for i in 0..4 {
            for j in 0..4 {
                u.re[i][j] -= impair.im[i][j];
                u.im[i][j] += impair.re[i][j];
            }
            u.re[i][i] += f_milieu;
        }
        u
    }
    fn reunitariser(&self, u: &mut M) {
        qr_symplectique(u);
    }
    fn defaut(&self, u: &M) -> f64 {
        let (a, b) = defauts(u);
        a.max(b)
    }
    fn haar_philox(&self, i: u32, slot: u32, k0: u32, k1: u32, etiquette: u32) -> M {
        let naif = self.mutant == MutantSp4::QrNaif;
        let mut u = M::zero();
        for j in 0..if naif { 4 } else { 2 } {
            for m in 0..4 {
                let w = philox4x32_10([i, slot, (4 * j + m) as u32, etiquette], k0, k1);
                let r = (-2. * u01(w[0], w[1]).ln()).sqrt();
                let (sin, cos) = (TAU * u01(w[2], w[3])).sin_cos();
                u.re[m][j] = r * cos;
                u.im[m][j] = r * sin;
            }
        }
        if naif {
            for j in 0..4 {
                for _ in 0..2 {
                    for k in 0..j {
                        orthogonaliser(&mut u, j, k);
                    }
                }
                normaliser(&mut u, j);
            }
        } else {
            qr_symplectique(&mut u);
        }
        u
    }
    fn a_surrelaxation(&self) -> bool {
        true
    }
    fn surrelaxer(&self, u: &mut M, a: &M) {
        reflechir(u, a, 0, 2);
        reflechir(u, a, 1, 3);
    }
}

impl Groupe for Sp4 {
    fn nom(&self) -> &'static str {
        "sp4"
    }
    fn d_f(&self) -> usize {
        4
    }
    fn d_adj(&self) -> usize {
        10
    }
    fn realite(&self) -> Realite {
        Realite::Pseudoreel
    }
    fn h_vee(&self) -> usize {
        3
    }
    fn generateurs(&self) -> Vec<Cmat> {
        (0..if self.mutant == MutantSp4::GenerateurRetire {
            9
        } else {
            10
        })
            .map(|i| {
                let mut c = [0.; 10];
                c[i] = 1.;
                algebre(&c).vers_cmat()
            })
            .collect()
    }
    fn appartient(&self, u: &Cmat) -> Appartenance {
        let m = M::depuis_cmat(u);
        let (unitarite, structure) = defauts(&m);
        let det = u.det();
        Appartenance {
            unitarite,
            structure,
            det: (det.0 - 1.).hypot(det.1),
        }
    }
    fn haar(&self, rng: &mut Flux) -> Cmat {
        let (i, slot, k0, k1, et) = rng.suivant();
        self.haar_philox(i, slot, k0, k1, et).vers_cmat()
    }
    fn projeter(&self, m: &Cmat) -> Cmat {
        let mut u = M::depuis_cmat(m);
        let det = m.det();
        if det.0.hypot(det.1) < 1e-300 {
            return M::identite().vers_cmat();
        }
        u = u.add(&conjugaison_j(&u)).echelle(0.5);
        for _ in 0..60 {
            let Some(inv) = inverse(&u.dag()) else {
                return M::identite().vers_cmat();
            };
            let suivant = u.add(&inv).echelle(0.5);
            let ecart = max_module(&suivant.sub(&u));
            u = suivant;
            if ecart < 1e-16 {
                break;
            }
        }
        u.vers_cmat()
    }
    fn exp(&self, a: &[f64]) -> Cmat {
        self.exp_i(a).vers_cmat()
    }
    fn log(&self, u: &Cmat) -> LogU {
        let mut y = M::depuis_cmat(u);
        let somme = 0.5 * y.re_tr();
        let carre = 0.5 * y.re_tr_prod(&y);
        let ecart = (carre + 2. - somme * somme).max(0.).sqrt();
        let cos_min = (0.5 * (somme - ecart)).clamp(-1., 1.);
        let theta = cos_min.acos();
        let ambigue = PI - theta < 1e-6;
        let hors_domaine = theta > FRAC_PI_2;
        if ambigue {
            return LogU {
                a: vec![0.; 10],
                theta,
                ambigue,
                hors_domaine,
            };
        }
        let id = M::identite();
        let mut racines = 0;
        while max_module(&y.sub(&id)) >= 0.2 && racines < 16 {
            let mut z = id;
            for _ in 0..60 {
                let (Some(iy), Some(iz)) = (inverse(&y), inverse(&z)) else {
                    break;
                };
                let nouveau_y = y.add(&iz).echelle(0.5);
                let nouveau_z = z.add(&iy).echelle(0.5);
                let ecart = max_module(&nouveau_y.sub(&y));
                y = nouveau_y;
                z = nouveau_z;
                if ecart < 1e-16 {
                    break;
                }
            }
            racines += 1;
        }
        let x = y.sub(&id);
        let mut puissance = x;
        let mut logarithme = M::zero();
        for n in 1..=60 {
            logarithme =
                logarithme.add(&puissance.echelle(if n % 2 == 1 { 1. } else { -1. } / n as f64));
            puissance = puissance.mul(&x);
        }
        logarithme = logarithme.echelle((1_u64 << racines) as f64);
        // Re Tr(T (−i L))/T_F = 2 Im Tr(T L).
        let a = (0..10)
            .map(|i| {
                let mut c = [0.; 10];
                c[i] = 1.;
                let t = algebre(&c);
                let mut im = 0.;
                for j in 0..4 {
                    for k in 0..4 {
                        im += t.re[j][k] * logarithme.im[k][j] + t.im[j][k] * logarithme.re[k][j];
                    }
                }
                2. * im
            })
            .collect();
        LogU {
            a,
            theta,
            ambigue,
            hors_domaine,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn haar(i: u32) -> M {
        Sp4::nouveau().haar_philox(i, 0, 16001, 26, 0x4752_4f55)
    }
    fn distance(a: &M, b: &M) -> f64 {
        max_module(&a.sub(b))
    }
    fn gaussiennes(i: u32, eps: f64) -> [f64; 10] {
        let mut c = [0.; 10];
        for j in 0..5 {
            let w = philox4x32_10([i, 0, j, 0x4558_5054], 16002, 26);
            let r = (-2. * u01(w[0], w[1]).ln()).sqrt() * eps;
            let (s, c0) = (TAU * u01(w[2], w[3])).sin_cos();
            c[2 * j as usize] = r * c0;
            c[2 * j as usize + 1] = r * s;
        }
        c
    }
    /// Référence indépendante : Taylor de iA, réduction de norme puis carrés.
    fn reference(c: &[f64]) -> M {
        let ts = Sp4::nouveau().generateurs();
        let mut a = M::zero();
        for (t, &x) in ts.iter().zip(c) {
            let t = M::depuis_cmat(t);
            for i in 0..4 {
                for j in 0..4 {
                    a.re[i][j] -= x * t.im[i][j];
                    a.im[i][j] += x * t.re[i][j];
                }
            }
        }
        let norme = (0..4)
            .map(|i| (0..4).map(|j| a.re[i][j].hypot(a.im[i][j])).sum::<f64>())
            .fold(0., f64::max);
        let n = if norme > 0.5 {
            (norme / 0.5).log2().ceil() as u32
        } else {
            0
        };
        a = a.echelle(2_f64.powi(-(n as i32)));
        let mut resultat = M::identite();
        let mut terme = resultat;
        for k in 1..=32 {
            terme = terme.mul(&a).echelle(1. / k as f64);
            resultat = resultat.add(&terme);
        }
        for _ in 0..n {
            resultat = resultat.mul(&resultat);
        }
        resultat
    }

    #[test]
    fn fiche_et_constantes_de_structure() {
        let g = Sp4::nouveau();
        let ts = g.generateurs();
        assert_eq!(Groupe::nom(&g), "sp4");
        assert_eq!(g.d_f(), 4);
        assert_eq!(Groupe::d_adj(&g), 10);
        assert_eq!(Noyau::d_adj(&g), 10);
        assert_eq!(g.realite(), Realite::Pseudoreel);
        assert_eq!(g.h_vee(), 3);
        assert!((g.t_f() - 0.5).abs() < 1e-15);
        let mut j = Cmat::zeros(4);
        for i in 0..2 {
            j.set(i, i + 2, (1., 0.));
            j.set(i + 2, i, (-1., 0.));
        }
        for (a, ta) in ts.iter().enumerate() {
            assert!(ta.max_abs_diff(&ta.dag()) < 1e-15);
            assert!(ta.trace().0.hypot(ta.trace().1) < 1e-15);
            let mut transpose = ta.dag();
            for im in &mut transpose.im {
                *im = -*im;
            }
            assert!(
                transpose
                    .mul(&j)
                    .add(&j.mul(ta))
                    .max_abs_diff(&Cmat::zeros(4))
                    < 1e-15
            );
            for (b, tb) in ts.iter().enumerate() {
                let tr = ta.mul(tb).trace();
                assert!((tr.0 - if a == b { 0.5 } else { 0. }).hypot(tr.1) < 1e-15);
            }
        }
        let f = g.f_abc();
        for a in 0..10 {
            for b in 0..10 {
                let mut somme = 0.;
                for c in 0..10 {
                    for d in 0..10 {
                        somme += f[(a * 10 + c) * 10 + d] * f[(b * 10 + c) * 10 + d];
                    }
                }
                assert!((somme - if a == b { 3. } else { 0. }).abs() < 1e-12);
            }
        }
        let mutant = Sp4::avec_mutant(MutantSp4::GenerateurRetire);
        assert_eq!(mutant.generateurs().len(), 9);
        assert_eq!(Groupe::d_adj(&mutant), 10);
    }

    #[test]
    fn haar_appartenance_et_reproductibilite() {
        let g = Sp4::nouveau();
        for i in 0..100_000 {
            let u = haar(i);
            let a = g.appartient(&u.vers_cmat());
            assert!(a.unitarite.max(a.structure).max(a.det) < 1e-14);
            if i < 100 {
                assert_eq!(u, haar(i));
            }
        }
        let mut flux = Flux::nouveau(crate::commun::Graine(26_u64 << 32 | 16001), 0x4752_4f55);
        assert_eq!(g.haar(&mut flux), haar(0).vers_cmat());
        assert_ne!(haar(0), g.haar_philox(0, 1, 16001, 26, 0x4752_4f55));
    }

    #[test]
    fn haar_moments_et_mutant_qr() {
        let n = 200_000;
        let mut sommes = [0.; 5];
        let mut carres = [0.; 5];
        for i in 0..n {
            let u = haar(i);
            let x = u.re_tr();
            let valeurs = [x, x * x, u.re_tr_prod(&u), x.powi(4), x.powi(3)];
            for k in 0..5 {
                sommes[k] += valeurs[k];
                carres[k] += valeurs[k] * valeurs[k];
            }
        }
        for k in 0..5 {
            let m = sommes[k] / n as f64;
            let sigma = ((carres[k] / n as f64 - m * m) / (n - 1) as f64).sqrt();
            assert!(
                (m - [0., 1., -1., 3., 0.][k]).abs() < 4. * sigma,
                "moment {k} : {m}, σ={sigma}"
            );
        }
        let g = Sp4::avec_mutant(MutantSp4::QrNaif);
        let mut structure: f64 = 0.;
        let mut somme = 0.;
        let mut carre = 0.;
        for i in 0..20_000 {
            let u = g.haar_philox(i, 0, 16001, 26, 0x4752_4f55);
            assert!(defauts(&u).0 < 1e-14);
            if i < 100 {
                structure = structure.max(defauts(&u).1);
            }
            let x = u.re_tr_prod(&u);
            somme += x;
            carre += x * x;
        }
        let m = somme / 20_000.;
        let sigma = ((carre / 20_000. - m * m) / 19_999.).sqrt();
        assert!(structure > 0.1);
        assert!(m.abs() < 4. * sigma);
        assert!((m + 1.).abs() > 4. * sigma);
    }

    #[test]
    fn exponentielle_reference_et_degenerescences() {
        let g = Sp4::nouveau();
        assert_eq!(g.exp_i(&[0.; 10]), M::identite());
        for i in 0..100_000 {
            let eps = [0.01, 0.3, 1., 2., 4.][i as usize % 5];
            let mut c = gaussiennes(i, eps);
            if i % 11 == 0 {
                let perturbation = [1e-2, 1e-4, 1e-8, 1e-12, 0.][(i / 11) as usize % 5];
                for x in &mut c[1..] {
                    *x *= perturbation;
                }
            }
            let u = g.exp_i(&c);
            let r = reference(&c);
            assert!(
                distance(&u, &r) <= 1e-14,
                "i={i}, ε={eps}, écart={:e}, c={c:?}",
                distance(&u, &r)
            );
            assert!(g.defaut(&u) <= 1e-14, "défaut i={i} : {:e}", g.defaut(&u));
            assert!(distance(&g.exp_i(&c.map(|x| -x)), &u.dag()) < 1e-14);
        }
        for x in [0.001, 0.3, 2., 4., 10., 20.] {
            let mut c = [0.; 10];
            c[0] = x;
            assert!(distance(&g.exp_i(&c), &reference(&c)) < 1e-14);
        }
    }

    #[test]
    fn exponentielle_logarithme_et_branches() {
        let g = Sp4::nouveau();
        for i in 0..10_000 {
            let mut c = gaussiennes(i, 0.5);
            let norme = c.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norme > 2. {
                c = c.map(|x| x * 2. / norme);
            }
            let l = g.log(&g.exp(&c));
            assert!(!l.ambigue);
            for (&x, &y) in c.iter().zip(&l.a) {
                assert!((x - y).abs() < 1e-11, "{x} != {y}");
            }
        }
        let l = g.log(&Cmat::identite(4).echelle(-1.));
        assert!(l.ambigue && l.hors_domaine);
        assert_eq!(l.a, vec![0.; 10]);
        let mut c = [0.; 10];
        c[4] = 4.;
        let l = g.log(&g.exp(&c));
        assert!((l.theta - 2.).abs() < 1e-14 && l.hors_domaine);
    }

    #[test]
    fn projections_et_reunitarisation() {
        let g = Sp4::nouveau();
        for i in 0..100 {
            let u = haar(i);
            assert!(distance(&M::depuis_cmat(&g.projeter(&u.vers_cmat())), &u) < 1e-14);
            let mut bruite = u;
            // 32 composantes de module 1e-3 / sqrt(32), norme Frobenius 1e-3.
            for r in 0..4 {
                for c in 0..4 {
                    bruite.re[r][c] += 1e-3 / 32_f64.sqrt();
                    bruite.im[r][c] -= 1e-3 / 32_f64.sqrt();
                }
            }
            let p = M::depuis_cmat(&g.projeter(&bruite.vers_cmat()));
            assert!(g.defaut(&p) < 1e-14);
            assert!(p.sub(&u).vers_cmat().norme_frobenius() < 3e-3);
            let mut bruite = u.add(&M::identite().echelle(1e-9));
            g.reunitariser(&mut bruite);
            assert!(g.defaut(&bruite) <= 1e-15, "{:e}", g.defaut(&bruite));
            assert!(distance(&bruite, &u) < 1e-8);
        }
        assert_eq!(g.projeter(&Cmat::zeros(4)), Cmat::identite(4));
    }

    #[test]
    fn surrelaxation_action_et_involution() {
        let g = Sp4::nouveau();
        for k in 0..1000 {
            let u = haar(7 * k);
            let a = (1..=6).fold(M::zero(), |a, j| a.add(&haar(7 * k + j)));
            let mut v = u;
            for (i, j) in [(0, 2), (1, 3)] {
                let avant = v;
                reflechir(&mut v, &a, i, j);
                assert!((v.re_tr_prod(&a) - avant.re_tr_prod(&a)).abs() < 1e-12);
                let mut deux = v;
                reflechir(&mut deux, &a, i, j);
                assert!(distance(&deux, &avant) < 1e-12);
            }
            g.reunitariser(&mut v);
            assert!(g.defaut(&v) < 1e-14);
            g.surrelaxer(&mut v, &a);
            assert!(distance(&v, &u) < 1e-12);
        }
    }
}
