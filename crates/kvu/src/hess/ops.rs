//! Opérateurs de S2 en base s2 (disposition `3(4x + μ) + a`) : références Q0 et Q_cov, contraintes des espaces.
//!
//! Tout est transcrit de `lat.py` (S2) : `Geo.R`, `Geo.div`, `Geo.B`, `qcov_su2`, `basis_fix`, `basis_cov`, `adj`.
//! Les opérateurs sont appliqués dans un ordre de sommation fixe (séquentiel) : résultat indépendant du nombre de fils.

use super::bases::Reseau;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul};

pub type Mat3 = [[f64; 3]; 3];

/// Adjointe R(V) d'un quaternion unitaire V = q0 + i p·σ : V (i ε·σ) V† = i (R ε)·σ,
/// R = (q0² − |p|²) 1 − 2 q0 [p×] + 2 p pᵀ (`lat.adj` : R_ab = ½ tr(σ_a V σ_b V†)).
pub fn adjointe(q: [f64; 4]) -> Mat3 {
    let [q0, p1, p2, p3] = q;
    let a = q0 * q0 - (p1 * p1 + p2 * p2 + p3 * p3);
    let p = [p1, p2, p3];
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = 2.0 * p[i] * p[j] + if i == j { a } else { 0.0 };
        }
    }
    // −2 q0 [p×] avec ([p×])_ij = −ε_ijk p_k
    let cx = [[0.0, -p3, p2], [p3, 0.0, -p1], [-p2, p1, 0.0]];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] -= 2.0 * q0 * cx[i][j];
        }
    }
    r
}

fn mat_vec(m: &Mat3, z: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[i][0] * z[0] + m[i][1] * z[1] + m[i][2] * z[2])
}
fn matt_vec(m: &Mat3, z: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[0][i] * z[0] + m[1][i] * z[1] + m[2][i] * z[2])
}

/// Quaternions des liens en disposition « s2 » (lien `4x + μ`).
pub fn liens_s2(champ: &ChampHote, r: &Reseau) -> Vec<[f64; 4]> {
    (0..r.nl).map(|l| champ.lien(r.champ_de_s2(l))).collect()
}

// ------------------------------------------------------------------------------------------
// Matrices creuses
// ------------------------------------------------------------------------------------------

/// Matrice creuse par lignes.
#[derive(Clone, Debug, Default)]
pub struct Csr {
    pub nlig: usize,
    pub ncol: usize,
    pub ptr: Vec<usize>,
    pub idx: Vec<u32>,
    pub val: Vec<f64>,
}

impl Csr {
    pub fn depuis_lignes(ncol: usize, lignes: &[Vec<(usize, f64)>]) -> Csr {
        let mut ptr = vec![0usize];
        let mut idx = Vec::new();
        let mut val = Vec::new();
        for l in lignes {
            for &(c, v) in l {
                idx.push(c as u32);
                val.push(v);
            }
            ptr.push(idx.len());
        }
        Csr { nlig: lignes.len(), ncol, ptr, idx, val }
    }

    /// y = A x (y écrasé).
    pub fn mul(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.nlig {
            let mut s = 0.0;
            for k in self.ptr[i]..self.ptr[i + 1] {
                s += self.val[k] * x[self.idx[k] as usize];
            }
            y[i] = s;
        }
    }

    /// Transposée, entrées de chaque ligne dans l'ordre croissant des lignes d'origine (déterministe).
    pub fn transposee(&self) -> Csr {
        let mut cnt = vec![0usize; self.ncol + 1];
        for &c in &self.idx {
            cnt[c as usize + 1] += 1;
        }
        for i in 0..self.ncol {
            cnt[i + 1] += cnt[i];
        }
        let mut pos = cnt.clone();
        let mut idx = vec![0u32; self.idx.len()];
        let mut val = vec![0.0; self.idx.len()];
        for i in 0..self.nlig {
            for k in self.ptr[i]..self.ptr[i + 1] {
                let c = self.idx[k] as usize;
                idx[pos[c]] = i as u32;
                val[pos[c]] = self.val[k];
                pos[c] += 1;
            }
        }
        Csr { nlig: self.ncol, ncol: self.nlig, ptr: cnt, idx, val }
    }

    /// Gram dense A Aᵀ (nlig × nlig, ligne-majeure).
    pub fn gram_dense(&self, at: &Csr) -> Vec<f64> {
        let m = self.nlig;
        let mut g = vec![0.0; m * m];
        for i in 0..m {
            for k in self.ptr[i]..self.ptr[i + 1] {
                let c = self.idx[k] as usize;
                let a = self.val[k];
                for kk in at.ptr[c]..at.ptr[c + 1] {
                    g[i * m + at.idx[kk] as usize] += a * at.val[kk];
                }
            }
        }
        g
    }

    /// Ligne i en vecteur dense.
    pub fn ligne_dense(&self, i: usize) -> Vec<f64> {
        let mut v = vec![0.0; self.ncol];
        for k in self.ptr[i]..self.ptr[i + 1] {
            v[self.idx[k] as usize] += self.val[k];
        }
        v
    }
}

// ------------------------------------------------------------------------------------------
// Contraintes (lignes) des espaces, disposition s2
// ------------------------------------------------------------------------------------------

/// div z = 0 par couleur : (div z)_x = Σ_μ z_{x,μ} − z_{x−μ,μ} ; ncol = 12 V.
pub fn contraintes_div(r: &Reseau) -> Vec<Vec<(usize, f64)>> {
    let mut lignes = Vec::with_capacity(3 * r.v);
    for x in 0..r.v {
        for c in 0..3 {
            let mut l = Vec::with_capacity(8);
            for mu in 0..4 {
                l.push((3 * (4 * x + mu) + c, 1.0));
                l.push((3 * (4 * r.arriere[mu][x] + mu) + c, -1.0));
            }
            lignes.push(l);
        }
    }
    lignes
}

/// Modes constants : Σ_x z_{x,μ,c} = 0 (12 lignes).
pub fn contraintes_constantes(r: &Reseau) -> Vec<Vec<(usize, f64)>> {
    let mut lignes = Vec::with_capacity(12);
    for mu in 0..4 {
        for c in 0..3 {
            lignes.push((0..r.v).map(|x| (3 * (4 * x + mu) + c, 1.0)).collect());
        }
    }
    lignes
}

/// Blocs de coin L = 2 (B de S2) : (B z)_{X,μ} = ⅛ Σ_{y⊥} (z_{y,μ} + z_{y+μ̂,μ}), par couleur.
pub fn contraintes_b_lineaire(r: &Reseau) -> Resultat<Vec<Vec<(usize, f64)>>> {
    if !r.l.is_multiple_of(2) {
        return Err(Erreur::Config(format!("L = {} : L pair requis pour les blocs de coin L = 2", r.l)));
    }
    let nc = r.l / 2;
    let mut lignes = Vec::new();
    for xi in 0..nc.pow(4) {
        let xb = Reseau::coordonnees_de(xi, nc);
        let x0 = xb.map(|v| 2 * v);
        for mu in 0..4 {
            let perp: Vec<usize> = (0..4).filter(|&n| n != mu).collect();
            for c in 0..3 {
                let mut l = Vec::with_capacity(16);
                for ys in 0..8usize {
                    let mut y = [0usize; 4];
                    for (k, &n) in perp.iter().enumerate() {
                        y[n] = (ys >> (2 - k)) & 1;
                    }
                    let xa = Reseau::site_de(std::array::from_fn(|m| (x0[m] + y[m]) % r.l), r.l);
                    let xbs = r.avant[mu][xa];
                    l.push((3 * (4 * xa + mu) + c, 0.125));
                    l.push((3 * (4 * xbs + mu) + c, 0.125));
                }
                lignes.push(l);
            }
        }
    }
    Ok(lignes)
}

/// Divergence covariante : (T z)_x = Σ_μ z_{x,μ} − Σ_μ R(U_{x−μ,μ})ᵀ z_{x−μ,μ} (`basis_cov`, lignes de T).
/// `seulement` : si Some(masque), ne garde que les sites x où masque[x] est vrai.
pub fn contraintes_tcov(r: &Reseau, u: &[[f64; 4]], seulement: Option<&[bool]>) -> Vec<Vec<(usize, f64)>> {
    let mut lignes = Vec::new();
    for x in 0..r.v {
        if let Some(m) = seulement {
            if !m[x] {
                continue;
            }
        }
        for i in 0..3 {
            let mut l = Vec::with_capacity(12 + 4 * 9);
            for mu in 0..4 {
                l.push((3 * (4 * x + mu) + i, 1.0));
            }
            for mu in 0..4 {
                let xm = r.arriere[mu][x];
                let lien = 4 * xm + mu;
                let rm = adjointe(u[lien]);
                // T[3x+i, 3l+j] −= R_ji
                for j in 0..3 {
                    l.push((3 * lien + j, -rm[j][i]));
                }
            }
            lignes.push(l);
        }
    }
    lignes
}

/// Blocs covariants B_U (`basis_cov`) : ligne (3 row + i), colonnes (la, j) : RW[i][j]/8 et (lb, j) : R(W U_{xa,μ})[i][j]/8.
pub fn contraintes_bcov(r: &Reseau, u: &[[f64; 4]]) -> Resultat<Vec<Vec<(usize, f64)>>> {
    if !r.l.is_multiple_of(2) {
        return Err(Erreur::Config(format!("L = {} : L pair requis pour les blocs de coin L = 2", r.l)));
    }
    let nc = r.l / 2;
    let mut lignes = Vec::new();
    for xi in 0..nc.pow(4) {
        let xb = Reseau::coordonnees_de(xi, nc);
        let x0 = xb.map(|v| 2 * v);
        for mu in 0..4 {
            let perp: Vec<usize> = (0..4).filter(|&n| n != mu).collect();
            let mut rows = [Vec::new(), Vec::new(), Vec::new()];
            for ys in 0..8usize {
                let mut y = [0usize; 4];
                for (k, &n) in perp.iter().enumerate() {
                    y[n] = (ys >> (2 - k)) & 1;
                }
                // transporteur de x0 à x0 + y le long des axes croissants
                let mut w = [1.0, 0.0, 0.0, 0.0];
                let mut x = Reseau::site_de(x0, r.l);
                for nu in 0..4 {
                    if y[nu] == 1 {
                        w = qmul(w, u[4 * x + nu]);
                        x = r.avant[nu][x];
                    }
                }
                let xa = x;
                let xbs = r.avant[mu][xa];
                let rw = adjointe(w);
                let rwu = adjointe(qmul(w, u[4 * xa + mu]));
                let (la, lb) = (4 * xa + mu, 4 * xbs + mu);
                for i in 0..3 {
                    for j in 0..3 {
                        rows[i].push((3 * la + j, rw[i][j] / 8.0));
                        rows[i].push((3 * lb + j, rwu[i][j] / 8.0));
                    }
                }
            }
            for l in rows {
                lignes.push(l);
            }
        }
    }
    Ok(lignes)
}

// ------------------------------------------------------------------------------------------
// Références
// ------------------------------------------------------------------------------------------

/// Q0 v par unité de β : ¼ Rᵀ R v ⊗ I₃ (R : incidence plaquettes-liens, +,+,−,−).
pub fn q0_appliquer(plaq: &[[usize; 4]], x: &[f64], y: &mut [f64]) {
    y.fill(0.0);
    for p in plaq {
        for c in 0..3 {
            let rot = x[3 * p[0] + c] + x[3 * p[1] + c] - x[3 * p[2] + c] - x[3 * p[3] + c];
            let t = 0.25 * rot;
            y[3 * p[0] + c] += t;
            y[3 * p[1] + c] += t;
            y[3 * p[2] + c] -= t;
            y[3 * p[3] + c] -= t;
        }
    }
}

/// Q_cov par unité de β : ¼ Σ_p J_pᵀ J_p, X_p = z1 + R(U1) z2 − R(U1U2U3†) z3 − R(U_p) z4 (`qcov_su2`).
pub struct QCov {
    plaq: Vec<[usize; 4]>,
    m: Vec<[Mat3; 3]>,
}

impl QCov {
    pub fn nouveau(r: &Reseau, u: &[[f64; 4]]) -> QCov {
        let plaq = r.plaquettes_s2();
        let m = plaq
            .iter()
            .map(|p| {
                let u1 = u[p[0]];
                let u2 = u[p[1]];
                let u3d = qdag(u[p[2]]);
                let u4d = qdag(u[p[3]]);
                let w3 = qmul(qmul(u1, u2), u3d);
                let w4 = qmul(w3, u4d);
                [adjointe(u1), adjointe(w3), adjointe(w4)]
            })
            .collect();
        QCov { plaq, m }
    }

    pub fn appliquer(&self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
        for (p, m) in self.plaq.iter().zip(&self.m) {
            let z = |l: usize| [x[3 * l], x[3 * l + 1], x[3 * l + 2]];
            let (z1, z2, z3, z4) = (z(p[0]), z(p[1]), z(p[2]), z(p[3]));
            let a = mat_vec(&m[0], z2);
            let b = mat_vec(&m[1], z3);
            let c = mat_vec(&m[2], z4);
            let xp: [f64; 3] = std::array::from_fn(|i| 0.25 * (z1[i] + a[i] - b[i] - c[i]));
            let g2 = matt_vec(&m[0], xp);
            let g3 = matt_vec(&m[1], xp);
            let g4 = matt_vec(&m[2], xp);
            for i in 0..3 {
                y[3 * p[0] + i] += xp[i];
                y[3 * p[1] + i] += g2[i];
                y[3 * p[2] + i] -= g3[i];
                y[3 * p[3] + i] -= g4[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjointe_par_definition() {
        // V (i ε·σ) V† = i (R ε)·σ : en quaternions, [0, Rε] = V [0, ε] V†
        let q = {
            let n = (0.3f64 * 0.3 + 0.5 * 0.5 + 0.2 * 0.2 + 0.7 * 0.7).sqrt();
            [0.3 / n, 0.5 / n, -0.2 / n, 0.7 / n]
        };
        let eps = [0.4, -1.1, 0.9];
        let r = adjointe(q);
        let rot = qmul(qmul(q, [0.0, eps[0], eps[1], eps[2]]), qdag(q));
        let re = mat_vec(&r, eps);
        for i in 0..3 {
            assert!((rot[1 + i] - re[i]).abs() < 1e-14, "{} {}", rot[1 + i], re[i]);
        }
        assert!(rot[0].abs() < 1e-14);
    }

    #[test]
    fn q0_froid_symetrique_et_noyau() {
        let r = Reseau::nouveau(2).unwrap();
        let plaq = r.plaquettes_s2();
        let n = r.n;
        // un gradient pur (jauge) est dans le noyau de Q0
        let lam: Vec<f64> = (0..r.v).map(|x| ((x * 7 + 3) % 5) as f64).collect();
        let mut z = vec![0.0; n];
        for x in 0..r.v {
            for mu in 0..4 {
                z[3 * (4 * x + mu) + 1] = lam[r.avant[mu][x]] - lam[x];
            }
        }
        let mut y = vec![0.0; n];
        q0_appliquer(&plaq, &z, &mut y);
        assert!(y.iter().all(|v| v.abs() < 1e-13));
    }

    #[test]
    fn csr_transposee() {
        let a = Csr::depuis_lignes(3, &[vec![(0, 1.0), (2, 2.0)], vec![(1, -1.0)]]);
        let at = a.transposee();
        let x = [1.0, 2.0];
        let mut y = [0.0; 3];
        at.mul(&x, &mut y);
        assert_eq!(y, [1.0, -2.0, 2.0]);
    }
}
