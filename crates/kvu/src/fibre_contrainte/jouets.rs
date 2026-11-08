//! Jouets exacts CT-7 (courbe) et CT-1/CT-2 (deux liens SU(2)).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use super::alea::{Flux, TAG_ACCEPTATION, TAG_PROPOSITION};
use super::jacobien::{eg, lg, rad, M3, Q};
use super::rattle::Variete;
use crate::commun::{Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul};
use crate::rg::bloc::{exp_q, log_q};
use std::f64::consts::{PI, TAU};

/// Les variantes erronées ne sont disponibles que dans les tests.
#[derive(Clone, Copy, Debug, Default)]
pub enum TermeLogDet {
    #[default]
    Correct,
    #[cfg(test)]
    Retire,
    #[cfg(test)]
    Oppose,
}

impl TermeLogDet {
    fn coefficient(self) -> f64 {
        match self {
            Self::Correct => 1.0,
            #[cfg(test)]
            Self::Retire => 0.0,
            #[cfg(test)]
            Self::Oppose => -1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Courbe {
    pub a: f64,
    minv: [f64; 2],
    pub terme: TermeLogDet,
}

impl Courbe {
    pub fn nouveau(m1: f64, m2: f64) -> Self {
        assert!(m1.is_finite() && m1 > 0.0 && m2.is_finite() && m2 > 0.0);
        Self {
            a: 1.2,
            minv: [1.0 / m1, 1.0 / m2],
            terme: TermeLogDet::Correct,
        }
    }

    pub fn moments_exacts() -> (f64, f64) {
        moments_exacts()
    }
}

impl Default for Courbe {
    fn default() -> Self {
        Self::nouveau(1.0, 1.0)
    }
}

#[derive(Clone, Debug)]
pub struct EvalCourbe {
    pub potentiel: f64,
    pub grad: [f64; 2],
    pub j: [f64; 2],
    pub g: f64,
}

impl Variete for Courbe {
    type Etat = [f64; 2];
    type Eval = EvalCourbe;
    fn dim(&self) -> usize {
        2
    }
    fn nb_contraintes(&self) -> usize {
        1
    }
    fn minv(&self) -> &[f64] {
        &self.minv
    }
    fn evaluer(&self, q: &Self::Etat) -> Resultat<Self::Eval> {
        let (sinus, cosinus) = q[0].sin_cos();
        let j = [-self.a * cosinus, 1.0];
        let g = j[0] * j[0] * self.minv[0] + self.minv[1];
        let c = self.terme.coefficient();
        let potentiel = 0.5 * cosinus + 0.5 * q[1] * q[1] + 0.5 * c * g.ln();
        let gx = -0.5 * sinus - c * self.a * self.a * cosinus * sinus * self.minv[0] / g;
        if !potentiel.is_finite() || !gx.is_finite() {
            return Err(Erreur::Validation(
                "évaluation non finie sur la courbe".into(),
            ));
        }
        Ok(EvalCourbe {
            potentiel,
            grad: [gx, q[1]],
            j,
            g,
        })
    }
    fn potentiel(&self, e: &Self::Eval) -> f64 {
        e.potentiel
    }
    fn grad_potentiel<'a>(&self, e: &'a Self::Eval) -> &'a [f64] {
        &e.grad
    }
    fn jt_mul(&self, e: &Self::Eval, lambda: &[f64]) -> Vec<f64> {
        e.j.iter().map(|j| j * lambda[0]).collect()
    }
    fn j_mul(&self, e: &Self::Eval, v: &[f64]) -> Vec<f64> {
        vec![e.j[0] * v[0] + e.j[1] * v[1]]
    }
    fn g_resoudre(&self, e: &Self::Eval, b: &[f64]) -> Vec<f64> {
        vec![b[0] / e.g]
    }
    fn residu(&self, q: &Self::Etat) -> Resultat<Vec<f64>> {
        Ok(vec![q[1] - self.a * q[0].sin()])
    }
    fn avancer(&self, q: &Self::Etat, xi: &[f64], dt: f64) -> Self::Etat {
        [q[0] + dt * xi[0], q[1] + dt * xi[1]]
    }
    fn ecart(&self, a: &Self::Etat, b: &Self::Etat) -> f64 {
        (a[0] - b[0]).abs().max((a[1] - b[1]).abs())
    }
}

/// Quadrature périodique de CT-7, avec 200 000 points.
pub fn moments_exacts() -> (f64, f64) {
    let (mut z, mut s1, mut s2) = (0.0, 0.0, 0.0);
    for i in 0..200_000 {
        let x = TAU * i as f64 / 200_000.0;
        let poids = (-0.5 * x.cos() - 0.5 * 1.2_f64.powi(2) * x.sin().powi(2)).exp();
        z += poids;
        s1 += poids * x.cos();
        s2 += poids * (2.0 * x).cos();
    }
    (s1 / z, s2 / z)
}

fn produit(a: &M3, b: &M3) -> M3 {
    let mut r = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r[3 * i + j] += a[3 * i + k] * b[3 * k + j];
            }
        }
    }
    r
}

fn transposee(a: M3) -> M3 {
    std::array::from_fn(|k| a[3 * (k % 3) + k / 3])
}

/// Cholesky 3×3, sans réordonnancement des sommes.
fn cholesky(g: M3) -> Resultat<M3> {
    let mut l = [0.0; 9];
    for i in 0..3 {
        for j in 0..=i {
            let mut s = g[3 * i + j];
            for k in 0..j {
                s -= l[3 * i + k] * l[3 * j + k];
            }
            if i == j {
                if !s.is_finite() || s <= 0.0 {
                    return Err(Erreur::Validation(
                        "Gram singulier du jouet deux liens".into(),
                    ));
                }
                l[3 * i + j] = s.sqrt();
            } else {
                l[3 * i + j] = s / l[3 * j + j];
            }
        }
    }
    Ok(l)
}

fn resoudre(l: &M3, b: &[f64]) -> Vec<f64> {
    let mut x = b.to_vec();
    for i in 0..3 {
        for j in 0..i {
            x[i] -= l[3 * i + j] * x[j];
        }
        x[i] /= l[3 * i + i];
    }
    for i in (0..3).rev() {
        for j in i + 1..3 {
            x[i] -= l[3 * j + i] * x[j];
        }
        x[i] /= l[3 * i + i];
    }
    x
}

fn renormaliser(q: Q) -> Q {
    let norme = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    q.map(|v| v / norme)
}

#[derive(Clone, Debug)]
pub struct DeuxLiens {
    pub beta: f64,
    pub t: f64,
    pub v: Q,
    pub terme: TermeLogDet,
}

#[derive(Clone, Debug)]
pub struct EvalDeuxLiens {
    pub potentiel: f64,
    pub grad: [f64; 6],
    /// Matrice 3×6, en ordre ligne.
    pub j: [f64; 18],
    pub g: M3,
    pub chol: M3,
}

impl DeuxLiens {
    pub fn nouveau(beta: f64, t: f64, v: Q) -> Self {
        assert!(beta.is_finite() && t > 0.0 && t < 1.0);
        assert!(
            v.iter().all(|x| x.is_finite())
                && (v.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-10
        );
        Self {
            beta,
            t,
            v,
            terme: TermeLogDet::Correct,
        }
    }

    pub fn observable(q: &[Q; 2]) -> f64 {
        qmul(qdag(q[0]), q[1])[0]
    }

    /// Point exact de la fibre de v, paramétré par le quaternion relatif w.
    pub fn depart(&self, v: Q, w: Q) -> [Q; 2] {
        let omega = log_q(w).0;
        [
            qmul(v, exp_q(omega.map(|x| -self.t * x))),
            qmul(v, exp_q(omega.map(|x| (1.0 - self.t) * x))),
        ]
    }

    fn ubar(&self, q: &[Q; 2]) -> Q {
        let omega = log_q(qmul(qdag(q[0]), q[1])).0;
        qmul(q[0], exp_q(omega.map(|x| self.t * x)))
    }

    /// D_t et sa dérivée logarithmique ; série pour éviter la soustraction 2/θ.
    fn facteur_radial(&self, theta: f64) -> (f64, f64) {
        let (t, u) = (self.t, 1.0 - self.t);
        if theta < 1e-4 {
            let s2 = t.powi(2) + u.powi(2);
            let s4 = t.powi(4) + u.powi(4);
            let s6 = t.powi(6) + u.powi(6);
            let b = (s2 - s4) / 3.0;
            let c = (2.0 * s2 - 5.0 * s4 + 2.0 * s6) / 45.0;
            let d = s2 + b * theta.powi(2) + c * theta.powi(4);
            return (d, (2.0 * b * theta + 4.0 * c * theta.powi(3)) / d);
        }
        let numerateur = (t * theta).sin().powi(2) + (u * theta).sin().powi(2);
        (
            numerateur / theta.sin().powi(2),
            (t * (2.0 * t * theta).sin() + u * (2.0 * u * theta).sin()) / numerateur
                - 2.0 / theta.tan(),
        )
    }
}

impl Variete for DeuxLiens {
    type Etat = [Q; 2];
    type Eval = EvalDeuxLiens;
    fn dim(&self) -> usize {
        6
    }
    fn nb_contraintes(&self) -> usize {
        3
    }
    fn minv(&self) -> &[f64] {
        &[1.0; 6]
    }
    fn evaluer(&self, q: &Self::Etat) -> Resultat<Self::Eval> {
        if !q.iter().flatten().all(|x| x.is_finite()) {
            return Err(Erreur::Validation("quaternion non fini".into()));
        }
        let relatif = qmul(qdag(q[0]), q[1]);
        let (omega, theta, ambigu) = log_q(relatif);
        if ambigu {
            return Err(Erreur::Validation(
                "coupure du logarithme des deux liens".into(),
            ));
        }
        let r = rad(q[0]);
        let mut a = produit(
            &produit(&produit(&r, &eg(omega.map(|x| self.t * x))), &lg(omega)),
            &transposee(r),
        );
        for x in &mut a {
            *x *= self.t;
        }
        let mut j = [0.0; 18];
        for i in 0..3 {
            for k in 0..3 {
                j[6 * i + k] = f64::from(i == k) - a[3 * i + k];
                j[6 * i + 3 + k] = a[3 * i + k];
            }
        }
        let mut g = [0.0; 9];
        for i in 0..3 {
            for k in 0..3 {
                for a in 0..6 {
                    g[3 * i + k] += j[6 * i + a] * j[6 * k + a];
                }
            }
        }
        let chol = cholesky(g)?;
        let demi_logdet = chol[0].ln() + chol[4].ln() + chol[8].ln();
        let c = self.terme.coefficient();
        let potentiel = -self.beta * relatif[0] + c * demi_logdet;
        let mut grad = [0.0; 6];
        if theta >= 1e-9 {
            let derivee = self.beta * theta.sin() + c * self.facteur_radial(theta).1;
            for i in 0..3 {
                let mut composante = 0.0;
                for k in 0..3 {
                    composante += r[3 * i + k] * omega[k] / theta;
                }
                grad[i] = -derivee * composante;
                grad[3 + i] = derivee * composante;
            }
        }
        Ok(EvalDeuxLiens {
            potentiel,
            grad,
            j,
            g,
            chol,
        })
    }
    fn potentiel(&self, e: &Self::Eval) -> f64 {
        e.potentiel
    }
    fn grad_potentiel<'a>(&self, e: &'a Self::Eval) -> &'a [f64] {
        &e.grad
    }
    fn jt_mul(&self, e: &Self::Eval, lambda: &[f64]) -> Vec<f64> {
        (0..6)
            .map(|k| (0..3).map(|i| e.j[6 * i + k] * lambda[i]).sum())
            .collect()
    }
    fn j_mul(&self, e: &Self::Eval, v: &[f64]) -> Vec<f64> {
        (0..3)
            .map(|i| (0..6).map(|k| e.j[6 * i + k] * v[k]).sum())
            .collect()
    }
    fn g_resoudre(&self, e: &Self::Eval, b: &[f64]) -> Vec<f64> {
        resoudre(&e.chol, b)
    }
    fn residu(&self, q: &Self::Etat) -> Resultat<Vec<f64>> {
        let (g, _, ambigu) = log_q(qmul(self.ubar(q), qdag(self.v)));
        if ambigu || !g.iter().all(|x| x.is_finite()) {
            return Err(Erreur::Validation("résidu ambigu des deux liens".into()));
        }
        Ok(g.to_vec())
    }
    fn avancer(&self, q: &Self::Etat, xi: &[f64], dt: f64) -> Self::Etat {
        std::array::from_fn(|i| {
            renormaliser(qmul(
                exp_q([dt * xi[3 * i], dt * xi[3 * i + 1], dt * xi[3 * i + 2]]),
                q[i],
            ))
        })
    }
    fn ecart(&self, a: &Self::Etat, b: &Self::Etat) -> f64 {
        a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }
}

fn moments_radiaux(beta: f64, densite: impl Fn(f64) -> f64) -> (f64, f64) {
    assert!(beta.is_finite());
    let quadrature = |n: usize| {
        let mut sommes = [0.0; 3];
        for i in 0..=n {
            let theta = PI * i as f64 / n as f64;
            let x = theta.cos();
            let poids_simpson = if i == 0 || i == n {
                1.0
            } else if i % 2 == 0 {
                2.0
            } else {
                4.0
            };
            // L'exponentielle décalée reste finie même à β = 1000.
            let poids = poids_simpson * (beta * x - beta.abs()).exp() * densite(theta);
            sommes[0] += poids;
            sommes[1] += poids * x;
            sommes[2] += poids * x * x;
        }
        (sommes[1] / sommes[0], sommes[2] / sommes[0])
    };
    let mut n = 256;
    let mut precedent = quadrature(n);
    loop {
        n *= 2;
        let courant = quadrature(n);
        if (courant.0 - precedent.0)
            .abs()
            .max((courant.1 - precedent.1).abs())
            < 1e-13
            || n >= 1_048_576
        {
            return courant;
        }
        precedent = courant;
    }
}

pub fn moments_exacts_deux_liens(beta: f64) -> (f64, f64) {
    moments_radiaux(beta, |theta| theta.sin().powi(2))
}

pub fn moments_mutant(beta: f64, t: f64) -> (f64, f64) {
    assert!(t > 0.0 && t < 1.0);
    moments_radiaux(beta, |theta| {
        (t * theta).sin().powi(2) + ((1.0 - t) * theta).sin().powi(2)
    })
}

/// Route A : une proposition par lien et par balayage, compteurs Philox fixes.
pub fn chaine_penalite_deux_liens(
    beta: f64,
    t: f64,
    v: Q,
    kappa: f64,
    delta: f64,
    graine: u64,
    chaine: u32,
    balayages: usize,
) -> Vec<[f64; 2]> {
    assert!(beta > 0.0 && kappa.is_finite() && kappa > 0.0 && delta.is_finite() && delta >= 0.0);
    assert!(balayages <= u32::MAX as usize);
    let modele = DeuxLiens::nouveau(beta, t, v);
    let flux = Flux::nouveau(graine);
    let mut q = modele.depart(v, [1.0, 0.0, 0.0, 0.0]);
    let potentiel = |q: &[Q; 2]| {
        let g = log_q(qmul(modele.ubar(q), qdag(v))).0;
        -beta * DeuxLiens::observable(q)
            + beta * g.iter().map(|x| x * x).sum::<f64>() / (2.0 * kappa)
    };
    let mut energie = potentiel(&q);
    let mut mesures = Vec::with_capacity(balayages);
    for b in 0..balayages {
        for i in 0..2 {
            let xi = std::array::from_fn(|a| {
                delta * flux.gauss(chaine, b as u32, (3 * i + a) as u32, TAG_PROPOSITION)
            });
            let mut essai = q;
            essai[i] = renormaliser(qmul(exp_q(xi), q[i]));
            let nouvelle = potentiel(&essai);
            if flux
                .uniforme(chaine, b as u32, i as u32, TAG_ACCEPTATION)
                .ln()
                < energie - nouvelle
            {
                q = essai;
                energie = nouvelle;
            }
        }
        let x = DeuxLiens::observable(&q);
        mesures.push([x, x * x]);
    }
    mesures
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fibre_contrainte::alea::TAG_DEPART;
    use crate::fibre_contrainte::rattle::{pas_rattle, tirer_impulsion, ParamsRattle};

    fn determinant(g: &M3) -> f64 {
        g[0] * (g[4] * g[8] - g[5] * g[7]) - g[1] * (g[3] * g[8] - g[5] * g[6])
            + g[2] * (g[3] * g[7] - g[4] * g[6])
    }

    #[test]
    fn moments_courbe_ct7() {
        let (m1, m2) = moments_exacts();
        assert!((m1 + 0.2829288041).abs() < 1e-9, "{m1}");
        assert!((m2 - 0.2054655606).abs() < 1e-9, "{m2}");
    }

    #[test]
    fn moments_deux_liens_ct1_ct2() {
        for (beta, exact, mutant) in [
            (2.0, 0.433127426722312, 0.154394686533120),
            (8.0, 0.819249410355621, 0.805069521286610),
        ] {
            let (m1, m2) = moments_exacts_deux_liens(beta);
            assert!((m1 - exact).abs() < 1e-10, "{beta} : {m1}");
            assert!((m2 - (1.0 - 3.0 * m1 / beta)).abs() < 1e-12);
            assert!((moments_mutant(beta, 0.5).0 - mutant).abs() < 1e-10);
        }
        let (m1, m2) = moments_exacts_deux_liens(0.0);
        assert!(m1.abs() < 1e-12 && (m2 - 0.25).abs() < 1e-12);
        let (m1, m2) = moments_exacts_deux_liens(1000.0);
        assert!((m1 - 0.998500375375493).abs() < 1e-12);
        assert!((m2 - 0.997004498873874).abs() < 1e-12);
    }

    #[test]
    fn deux_liens_determinant_gradient_et_jacobien() {
        let flux = Flux::nouveau(774);
        for i in 0..20 {
            let t = if i % 2 == 0 { 0.3 } else { 0.5 };
            let v = exp_q([0.3, -0.2, 0.4]);
            let m = DeuxLiens::nouveau(2.0, t, v);
            let omega = std::array::from_fn(|a| 0.5 * flux.gauss(0, i, a as u32, TAG_DEPART));
            let q = m.depart(v, exp_q(omega));
            assert!(m.residu(&q).unwrap().iter().all(|x| x.abs() < 1e-14));
            let ev = m.evaluer(&q).unwrap();
            let theta = omega.iter().map(|x| x * x).sum::<f64>().sqrt();
            let det = ((1.0 - t).powi(2) + t * t) * m.facteur_radial(theta).0.powi(2);
            assert!((determinant(&ev.g) / det - 1.0).abs() < 1e-10);
            assert!(
                (ev.potentiel - (-2.0 * DeuxLiens::observable(&q) + 0.5 * determinant(&ev.g).ln()))
                    .abs()
                    < 1e-13
            );
            let h = 1e-6;
            for k in 0..6 {
                let mut direction = [0.0; 6];
                direction[k] = 1.0;
                let qp = m.avancer(&q, &direction, h);
                let qm = m.avancer(&q, &direction, -h);
                let df = (m.evaluer(&qp).unwrap().potentiel - m.evaluer(&qm).unwrap().potentiel)
                    / (2.0 * h);
                assert!(
                    (df - ev.grad[k]).abs() < 1e-6,
                    "i={i} k={k} : {df} != {}",
                    ev.grad[k]
                );
                let gp = m.residu(&qp).unwrap();
                let gm = m.residu(&qm).unwrap();
                for a in 0..3 {
                    assert!(((gp[a] - gm[a]) / (2.0 * h) - ev.j[6 * a + k]).abs() < 1e-8);
                }
            }
        }
    }

    #[test]
    fn gradients_des_variantes_et_limite_identite() {
        for terme in [
            TermeLogDet::Correct,
            TermeLogDet::Retire,
            TermeLogDet::Oppose,
        ] {
            let mut m = Courbe::nouveau(1.0, 4.0);
            m.terme = terme;
            let q = [0.4, 0.3];
            let ev = m.evaluer(&q).unwrap();
            for k in 0..2 {
                let mut d = [0.0; 2];
                d[k] = 1.0;
                let df = (m.evaluer(&m.avancer(&q, &d, 1e-6)).unwrap().potentiel
                    - m.evaluer(&m.avancer(&q, &d, -1e-6)).unwrap().potentiel)
                    / 2e-6;
                assert!((df - ev.grad[k]).abs() < 1e-8);
            }
            let mut m = DeuxLiens::nouveau(8.0, 0.3, exp_q([0.2, 0.3, -0.1]));
            m.terme = terme;
            for omega in [
                [0.0; 3],
                [1e-10, 0.0, 0.0],
                [1e-6, -2e-6, 3e-6],
                [0.5, -0.3, 0.7],
            ] {
                let q = m.depart(m.v, exp_q(omega));
                let ev = m.evaluer(&q).unwrap();
                assert!(ev.potentiel.is_finite() && ev.grad.iter().all(|x| x.is_finite()));
                for k in 0..6 {
                    let mut d = [0.0; 6];
                    d[k] = 1.0;
                    let df = (m.evaluer(&m.avancer(&q, &d, 1e-6)).unwrap().potentiel
                        - m.evaluer(&m.avancer(&q, &d, -1e-6)).unwrap().potentiel)
                        / 2e-6;
                    assert!(
                        (df - ev.grad[k]).abs() < 1e-6,
                        "{omega:?}, {k}: {df} != {}",
                        ev.grad[k]
                    );
                }
            }
        }
    }

    #[test]
    fn rattle_deux_liens() {
        let flux = Flux::nouveau(413);
        for t in [0.3, 0.5] {
            let m = DeuxLiens::nouveau(2.0, t, exp_q([0.1, 0.4, -0.2]));
            let mut q = m.depart(m.v, exp_q([0.4, -0.3, 0.2]));
            let mut ev = m.evaluer(&q).unwrap();
            let mut p = tirer_impulsion(&m, &ev, &flux, 0, 0);
            for _ in 0..20 {
                let (q1, ev1, p1, _) =
                    pas_rattle(&m, &q, &ev, &p, &ParamsRattle::default()).unwrap();
                assert!(m.residu(&q1).unwrap().iter().all(|x| x.abs() < 1e-12));
                assert!(m.j_mul(&ev1, &p1).iter().all(|x| x.abs() < 1e-12));
                (q, ev, p) = (q1, ev1, p1);
            }
        }
    }

    #[test]
    fn penalite_faible_retrouve_la_loi_libre() {
        let mesures = chaine_penalite_deux_liens(
            2.0,
            0.5,
            [1.0, 0.0, 0.0, 0.0],
            100.0,
            0.7,
            1026,
            0,
            200_000,
        );
        let moyenne = mesures.iter().skip(20_000).map(|m| m[0]).sum::<f64>() / 180_000.0;
        let exact = moments_exacts_deux_liens(2.0).0;
        assert!((moyenne / exact - 1.0).abs() < 0.03, "{moyenne} != {exact}");
        let a = chaine_penalite_deux_liens(2.0, 0.3, exp_q([0.2, -0.1, 0.4]), 10.0, 0.5, 2, 3, 100);
        let b = chaine_penalite_deux_liens(2.0, 0.3, exp_q([0.2, -0.1, 0.4]), 10.0, 0.5, 2, 3, 100);
        for (a, b) in a.iter().flatten().zip(b.iter().flatten()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
