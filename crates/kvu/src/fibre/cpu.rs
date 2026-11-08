//! Référence CPU (rayon) de la mesure de fibre. Les noyaux GPU (`noyaux/fibre.cu`) suivent les MÊMES
//! expressions dans le MÊME ordre d'évaluation : compilés sans contraction FMA (`reference_k1`), ils
//! doivent donner les mêmes octets (gate FIB-3).
//!
//! Convention : U = (x, y, z, w) = x + i(y σ1 + z σ2 + w σ3), produit `qmul` de su2_commun.h ; la base
//! de l'algèbre est X_a = iσ_a/√2 et les dérivées sont invariantes à droite (U → e^{sX_a} U).

use super::plan::{ModeFibre, PlanFibre, TRANCHE};
use super::poids::PoidsFourier;
use super::tables::liens_plaquette;
use crate::commun::ChampHote;
use crate::heritage::su2_hb::{qdag, qmul, Q};
use rayon::prelude::*;

/// 1/√2 (la même valeur que le littéral 0.7071067811865476 de cb.py : voir le test `s2i_est_le_litteral_de_cb`).
pub(crate) const S2I: f64 = std::f64::consts::FRAC_1_SQRT_2;

#[inline]
fn lien(q: &[f64], e: usize) -> Q {
    [q[4 * e], q[4 * e + 1], q[4 * e + 2], q[4 * e + 3]]
}

/// Valeur ½ReTr U_p et gradients (4 liens × 3) de la plaquette `p` (= `_k_plaq` de cb.py).
/// Préfixes P1 = W0, P2 = W0 W1 W2⁺, Up = P2 W3⁺ ; Q_i = P_i⁺ Up P_i, signes + + − −.
#[inline]
pub(crate) fn plaquette(q: &[f64], e: [u32; 4]) -> (f64, [f64; 12]) {
    let w0 = lien(q, e[0] as usize);
    let w1 = lien(q, e[1] as usize);
    let w2 = qdag(lien(q, e[2] as usize));
    let w3 = qdag(lien(q, e[3] as usize));
    let p1 = w0;
    let p2 = qmul(qmul(p1, w1), w2);
    let up = qmul(p2, w3);
    let q0 = up;
    let q1 = qmul(qmul(qdag(p1), up), p1);
    let q2 = qmul(qmul(qdag(p2), up), p2);
    let q3 = qmul(qmul(qdag(up), up), up);
    let mut g = [0.0; 12];
    for (i, (qi, sg)) in [(q0, 1.0), (q1, 1.0), (q2, -1.0), (q3, -1.0)].into_iter().enumerate() {
        let c = sg * S2I;
        for a in 0..3 {
            g[3 * i + a] = -c * qi[a + 1];
        }
    }
    (up[0], g)
}

/// Plaquettes ½ReTr U_p (6V, ordre loops_plaquettes : plan majeur puis site) et gradients (6V × 4 × 3).
pub fn plaquettes_gradients(c: &ChampHote) -> (Vec<f64>, Vec<f64>) {
    let l = c.l;
    let v = c.v();
    let mut pv = vec![0.0; 6 * v];
    let mut gp = vec![0.0; 6 * v * 12];
    pv.par_iter_mut()
        .zip(gp.par_chunks_mut(12))
        .enumerate()
        .for_each(|(p, (pvp, g))| {
            let (val, grad) = plaquette(&c.q, liens_plaquette(l, p));
            *pvp = val;
            g.copy_from_slice(&grad);
        });
    (pv, gp)
}

/// Plaquettes ½ReTr U_p seules (6V), sans les gradients.
pub fn plaquettes_valeurs(c: &ChampHote) -> Vec<f64> {
    let l = c.l;
    (0..6 * c.v()).into_par_iter().map(|p| plaquette(&c.q, liens_plaquette(l, p)).0).collect()
}

/// Termes par segment de la somme de F (taille FIXE : le résultat ne dépend que d'elle et de `VOIES_F`).
pub const SEGMENT_F: usize = 4096;
/// Voies d'un segment : le terme j du segment va à la voie j mod `VOIES_F` (= fils d'un bloc de `k_F_seg`).
pub const VOIES_F: usize = 256;

/// Un pas de la somme compensée de Neumaier : (s, c) ← (s + x, c + erreur d'arrondi).
#[inline]
pub(crate) fn pas_neumaier(s: &mut f64, c: &mut f64, x: f64) {
    let t = *s + x;
    if s.abs() >= x.abs() {
        *c += (*s - t) + x;
    } else {
        *c += (x - t) + *s;
    }
    *s = t;
}

/// Premier segment de chaque ligne (nf + 1 entrées) pour des segments de `segment` termes.
pub(crate) fn pointeurs_segments(ptr: &[u64], segment: usize) -> Vec<u32> {
    let mut sp = Vec::with_capacity(ptr.len());
    sp.push(0u32);
    for i in 0..ptr.len() - 1 {
        let n = (ptr[i + 1] - ptr[i]) as usize;
        sp.push(sp[i] + n.div_ceil(segment) as u32);
    }
    sp
}

/// Somme de Neumaier d'un segment [q0, q1) de termes `val · pv[idx]` : `VOIES_F` voies (le terme q va à la voie
/// (q − q0) mod `VOIES_F`, ordre croissant dans chaque voie), puis combinaison des paires (s, c) des voies dans l'ordre,
/// par Neumaier sur la suite s₀, c₀, s₁, c₁, … Rend l'état (s, c) final. Même algorithme que `k_F_seg`.
fn somme_segment(w: &PoidsFourier, pv: &[f64], q0: usize, q1: usize) -> (f64, f64) {
    let mut s = [0.0f64; VOIES_F];
    let mut c = [0.0f64; VOIES_F];
    let mut base = q0;
    while base < q1 {
        for t in 0..VOIES_F.min(q1 - base) {
            let q = base + t;
            pas_neumaier(&mut s[t], &mut c[t], w.val[q] * pv[w.idx[q] as usize]);
        }
        base += VOIES_F;
    }
    let (mut s0, mut c0) = (0.0f64, 0.0f64);
    for t in 0..VOIES_F {
        pas_neumaier(&mut s0, &mut c0, s[t]);
        pas_neumaier(&mut s0, &mut c0, c[t]);
    }
    (s0, c0)
}

/// F = Wf · pv, sommation compensée de Neumaier par segments fixes (`SEGMENT_F` termes consécutifs d'une ligne,
/// sommés par `VOIES_F` voies), puis combinaison des paires (s, c) des segments dans l'ordre, par Neumaier sur la suite
/// s₁, c₁, s₂, c₂, … (erreur ≈ 1 ulp de F, indépendante de L). Même algorithme que `k_F_seg` / `k_F_fin`.
pub(crate) fn f_neumaier(w: &PoidsFourier, pv: &[f64]) -> Vec<f64> {
    let sp = pointeurs_segments(&w.ptr, SEGMENT_F);
    let paires: Vec<(f64, f64)> = (0..sp[w.nf] as usize)
        .into_par_iter()
        .map(|g| {
            let i = sp.partition_point(|&x| x as usize <= g) - 1;
            let q0 = w.ptr[i] as usize + (g - sp[i] as usize) * SEGMENT_F;
            somme_segment(w, pv, q0, (q0 + SEGMENT_F).min(w.ptr[i + 1] as usize))
        })
        .collect();
    (0..w.nf)
        .map(|i| {
            let (mut s, mut c) = (0.0f64, 0.0f64);
            for &(ss, cc) in &paires[sp[i] as usize..sp[i + 1] as usize] {
                pas_neumaier(&mut s, &mut c, ss);
                pas_neumaier(&mut s, &mut c, cc);
            }
            s + c
        })
        .collect()
}

/// Sommes de blocs 2⁴ des plaquettes, arrondies en float32 (= `block_sums(pv).astype(float32)`).
/// Ordre de numpy pour `sum(axis=(2,4,6,8))` : les deux sites voisins selon x3 d'abord, puis accumulation
/// dans l'ordre (d0, d1, d2) de C, à partir de 0.
pub fn somme_blocs(l: usize, pv: &[f64]) -> Vec<f32> {
    let lc = l / 2;
    let v = l * l * l * l;
    let n_bloc = lc * lc * lc * lc;
    let mut out = vec![0f32; 6 * n_bloc];
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let (k, b) = (i / n_bloc, i % n_bloc);
        let (b0, b1, b2, b3) = (b / (lc * lc * lc), (b / (lc * lc)) % lc, (b / lc) % lc, b % lc);
        let mut acc = 0.0f64;
        for d0 in 0..2 {
            for d1 in 0..2 {
                for d2 in 0..2 {
                    let s = (((2 * b0 + d0) * l + 2 * b1 + d1) * l + 2 * b2 + d2) * l + 2 * b3;
                    acc += pv[k * v + s] + pv[k * v + s + 1];
                }
            }
        }
        *o = acc as f32;
    });
    out
}

/// Accumule dans `g` (3 × ns, ligne a majeure) le gradient du lien `e` projeté sur Wf :
/// g[a][cmap(r)] += Σ_{p ∋ e} gp[p, pos, a] · Wf[r, p], plaquettes dans l'ordre de `lp`.
#[inline]
fn gradient_lien(plan: &PlanFibre, cm: &[u8], ns: usize, gp: &[f64], e: usize, g: &mut [f64]) {
    for j in 0..6 {
        let x = plan.lp[6 * e + j] as usize;
        let (p, pos) = (x >> 2, x & 3);
        let g3 = &gp[(4 * p + pos) * 3..(4 * p + pos) * 3 + 3];
        for k in plan.wt_ptr[p] as usize..plan.wt_ptr[p + 1] as usize {
            let c = cm[plan.wt_idx[k] as usize] as usize;
            let w = plan.wt_val[k];
            for a in 0..3 {
                g[a * ns + c] += g3[a] * w;
            }
        }
    }
}

/// Matrice de rotation R_ba = ½ Re Tr(σ_b U⁺ σ_a U) = composante b de U⁺ (iσ_a) U, rendue par colonne a.
#[inline]
pub(crate) fn rotation(u: Q) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for a in 0..3 {
        let mut ea = [0.0; 4];
        ea[a + 1] = 1.0;
        let t = qmul(qmul(qdag(u), ea), u);
        for b in 0..3 {
            r[b][a] = t[b + 1];
        }
    }
    r
}

/// Les trois lignes (a = 0, 1, 2) de l'élément `t` de la direction `mu`, restreintes au support, dans
/// `g1` (3 × ns). `g2` est un brouillon de même taille.
#[allow(clippy::too_many_arguments)]
fn lignes_element(
    plan: &PlanFibre,
    q: &[f64],
    gp: &[f64],
    mode: ModeFibre,
    mu: usize,
    t: usize,
    g1: &mut [f64],
    g2: &mut [f64],
) {
    let cm = &plan.cmap[mu];
    let ns = plan.sup[mu].len();
    g1.fill(0.0);
    match mode {
        ModeFibre::Libre => gradient_lien(plan, cm, ns, gp, mu * plan.v + t, g1),
        ModeFibre::Conditionnel if t < plan.nfree_mu => {
            gradient_lien(plan, cm, ns, gp, plan.lfree[mu * plan.nfree_mu + t] as usize, g1)
        }
        ModeFibre::Conditionnel => {
            let k = mu * plan.npair_mu + (t - plan.nfree_mu);
            let (e1, e2) = (plan.e1[k] as usize, plan.e2[k] as usize);
            g2.fill(0.0);
            gradient_lien(plan, cm, ns, gp, e1, g1);
            gradient_lien(plan, cm, ns, gp, e2, g2);
            let r = rotation(lien(q, e1));
            for a in 0..3 {
                for c in 0..ns {
                    let m = r[0][a] * g2[c] + r[1][a] * g2[ns + c] + r[2][a] * g2[2 * ns + c];
                    g1[a * ns + c] = (g1[a * ns + c] - m) * S2I;
                }
            }
        }
    }
}

/// Partiel de D de la tranche `tr` (nf × nf, ligne majeure, symétrique) : Σ_{éléments, a} ligne ⊗ ligne,
/// éléments puis a dans l'ordre croissant, demi-matrice supérieure recopiée à la fin.
fn partiel_tranche(plan: &PlanFibre, q: &[f64], gp: &[f64], mode: ModeFibre, tr: usize) -> Vec<f64> {
    let nf = plan.nf;
    let par_mu = plan.tranches_mu(mode);
    let (mu, i) = (tr / par_mu, tr % par_mu);
    let n_mu = plan.n_elem_mu(mode);
    let (t0, t1) = (i * TRANCHE, ((i + 1) * TRANCHE).min(n_mu));
    let sup = &plan.sup[mu];
    let ns = sup.len();
    let mut g1 = vec![0.0; 3 * ns];
    let mut g2 = vec![0.0; 3 * ns];
    let mut acc = vec![0.0; ns * ns];
    for t in t0..t1 {
        lignes_element(plan, q, gp, mode, mu, t, &mut g1, &mut g2);
        for a in 0..3 {
            let ligne = &g1[a * ns..(a + 1) * ns];
            for ii in 0..ns {
                let x = ligne[ii];
                for jj in ii..ns {
                    acc[ii * ns + jj] += x * ligne[jj];
                }
            }
        }
    }
    let mut part = vec![0.0; nf * nf];
    for ii in 0..ns {
        for jj in ii..ns {
            let (ci, cj) = (sup[ii] as usize, sup[jj] as usize);
            part[ci * nf + cj] = acc[ii * ns + jj];
            part[cj * nf + ci] = acc[ii * ns + jj];
        }
    }
    part
}

/// D = Σ_éléments Σ_a ligne ⊗ ligne (nf × nf, ligne majeure) : partiels de tranches fixes (`TRANCHE`
/// éléments) calculés en parallèle, puis sommés dans l'ordre croissant des tranches ⇒ résultat
/// indépendant du nombre de fils.
pub(crate) fn matrice_d(plan: &PlanFibre, q: &[f64], gp: &[f64], mode: ModeFibre) -> Vec<f64> {
    let nf = plan.nf;
    let partiels: Vec<Vec<f64>> = (0..plan.tranches(mode))
        .into_par_iter()
        .map(|tr| partiel_tranche(plan, q, gp, mode, tr))
        .collect();
    let mut d = vec![0.0; nf * nf];
    for p in &partiels {
        for (x, y) in d.iter_mut().zip(p) {
            *x += *y;
        }
    }
    d
}

/// Site (indice de base) de l'élément `t` de la direction `mu` en mode conditionnel : lien libre (μ, x) → x ;
/// paire (e1 = (μ, x), e2 = (μ, x + μ̂)) → x, le site de e1 (convention de torsion.py, saturation v3).
#[inline]
fn site_element(plan: &PlanFibre, mu: usize, t: usize) -> usize {
    let lien = if t < plan.nfree_mu {
        plan.lfree[mu * plan.nfree_mu + t]
    } else {
        plan.e1[mu * plan.npair_mu + (t - plan.nfree_mu)]
    };
    lien as usize % plan.v
}

/// Partiels (D0, Dm, Dp) de la tranche `tr` en mode conditionnel : Σ ligne ⊗ ligne non pondérée, pondérée par
/// e^{−2ψ(site)} et par e^{+2ψ(site)} (= `D_tw` de torsion.py). Même ordre que [`partiel_tranche`].
fn partiel_tranche_tordu(plan: &PlanFibre, q: &[f64], gp: &[f64], psi: &[f64], tr: usize) -> [Vec<f64>; 3] {
    let mode = ModeFibre::Conditionnel;
    let nf = plan.nf;
    let par_mu = plan.tranches_mu(mode);
    let (mu, i) = (tr / par_mu, tr % par_mu);
    let n_mu = plan.n_elem_mu(mode);
    let (t0, t1) = (i * TRANCHE, ((i + 1) * TRANCHE).min(n_mu));
    let sup = &plan.sup[mu];
    let ns = sup.len();
    let mut g1 = vec![0.0; 3 * ns];
    let mut g2 = vec![0.0; 3 * ns];
    let mut acc = [vec![0.0; ns * ns], vec![0.0; ns * ns], vec![0.0; ns * ns]];
    for t in t0..t1 {
        lignes_element(plan, q, gp, mode, mu, t, &mut g1, &mut g2);
        let w = (-psi[site_element(plan, mu, t)]).exp();
        let (wm, wp) = (w * w, 1.0 / (w * w));
        for a in 0..3 {
            let ligne = &g1[a * ns..(a + 1) * ns];
            for ii in 0..ns {
                let x = ligne[ii];
                for jj in ii..ns {
                    let xy = x * ligne[jj];
                    acc[0][ii * ns + jj] += xy;
                    acc[1][ii * ns + jj] += wm * xy;
                    acc[2][ii * ns + jj] += wp * xy;
                }
            }
        }
    }
    acc.map(|a| {
        let mut part = vec![0.0; nf * nf];
        for ii in 0..ns {
            for jj in ii..ns {
                let (ci, cj) = (sup[ii] as usize, sup[jj] as usize);
                part[ci * nf + cj] = a[ii * ns + jj];
                part[cj * nf + ci] = a[ii * ns + jj];
            }
        }
        part
    })
}

/// (D0, Dm, Dp) de la famille tordue (nf × nf chacune, ligne majeure), concaténées : partiels de tranches fixes
/// sommés dans l'ordre croissant ⇒ résultat indépendant du nombre de fils.
pub(crate) fn matrices_d_tordues(plan: &PlanFibre, q: &[f64], gp: &[f64], psi: &[f64]) -> Vec<f64> {
    let nf = plan.nf;
    let partiels: Vec<[Vec<f64>; 3]> = (0..plan.tranches(ModeFibre::Conditionnel))
        .into_par_iter()
        .map(|tr| partiel_tranche_tordu(plan, q, gp, psi, tr))
        .collect();
    let mut d = vec![0.0; 3 * nf * nf];
    for p in &partiels {
        for (k, m) in p.iter().enumerate() {
            for (x, y) in d[k * nf * nf..(k + 1) * nf * nf].iter_mut().zip(m) {
                *x += *y;
            }
        }
    }
    d
}

/// Sortie brute d'une mesure CPU.
pub(crate) struct SortieCpu {
    pub pv: Vec<f64>,
    pub f: Vec<f64>,
    pub d: Option<Vec<f64>>,
}

/// F seul (et pv) ; avec `avec_d`, D du mode demandé.
pub(crate) fn mesurer(plan: &PlanFibre, w: &PoidsFourier, c: &ChampHote, mode: Option<ModeFibre>) -> SortieCpu {
    let (pv, gp) = plaquettes_gradients(c);
    let f = f_neumaier(w, &pv);
    let d = mode.map(|m| matrice_d(plan, &c.q, &gp, m));
    SortieCpu { pv, f, d }
}

#[cfg(test)]
mod tests {
    use super::super::plan::PlanFibre;
    use super::super::poids::poids_fourier;
    use super::*;
    use crate::cond::GeometrieCond;

    /// Configuration pseudo-aléatoire SU(2) déterministe (xorshift + Box-Muller grossier, sans dépendance).
    pub(crate) fn champ_aleatoire(l: usize, graine: u64) -> ChampHote {
        let mut s = graine.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut u = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut c = ChampHote::froid(l).unwrap();
        for lien in c.q.chunks_exact_mut(4) {
            let mut n = 0.0;
            for x in lien.iter_mut() {
                // somme de 6 uniformes centrées : quasi-gaussienne, suffisant pour un test
                *x = (0..6).map(|_| u() - 0.5).sum::<f64>();
                n += *x * *x;
            }
            let n = n.sqrt();
            for x in lien.iter_mut() {
                *x /= n;
            }
        }
        c
    }

    #[test]
    #[allow(clippy::approx_constant)] // le littéral EST celui de cb.py
    fn s2i_est_le_litteral_de_cb() {
        assert_eq!(S2I.to_bits(), 0.7071067811865476f64.to_bits());
    }

    #[test]
    fn champ_froid_plaquettes_unite_gradients_nuls() {
        let c = ChampHote::froid(4).unwrap();
        let (pv, gp) = plaquettes_gradients(&c);
        assert!(pv.iter().all(|&x| x == 1.0));
        assert!(gp.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn gradients_contre_differences_finies() {
        // d/ds ½ReTr U_p pour U_lien → e^{s X_a} U_lien (= qmul(exp, U)), position i de la plaquette p.
        let l = 4;
        let c = champ_aleatoire(l, 7);
        let (_, gp) = plaquettes_gradients(&c);
        let h = 1e-5;
        for &p in &[0usize, 37, 300, 777, 1535] {
            let e = liens_plaquette(l, p);
            for i in 0..4 {
                for a in 0..3 {
                    let mut val = [0.0; 2];
                    for (k, sgn) in [1.0, -1.0].into_iter().enumerate() {
                        let mut cc = c.clone();
                        let th = sgn * h * S2I;
                        let mut ea = [0.0; 4];
                        ea[0] = th.cos();
                        ea[a + 1] = th.sin();
                        let u = qmul(ea, lien(&cc.q, e[i] as usize));
                        cc.q[4 * e[i] as usize..4 * e[i] as usize + 4].copy_from_slice(&u);
                        val[k] = plaquette(&cc.q, e).0;
                    }
                    let fd = (val[0] - val[1]) / (2.0 * h);
                    let g = gp[(4 * p + i) * 3 + a];
                    assert!((fd - g).abs() < 1e-8, "p = {p} i = {i} a = {a} : {fd} contre {g}");
                }
            }
        }
    }

    #[test]
    fn plaquette_unitaire() {
        let c = champ_aleatoire(4, 3);
        let (pv, _) = plaquettes_gradients(&c);
        assert!(pv.iter().all(|&x| x.abs() <= 1.0 + 1e-14));
    }

    #[test]
    fn neumaier_proche_de_somme_exacte() {
        // somme de 1e8 + 1 + (−1e8) : naïf = 0, compensé = 1
        let w = PoidsFourier {
            nf: 1,
            ptr: vec![0, 3],
            idx: vec![0, 1, 2],
            val: vec![1.0, 1.0, 1.0],
        };
        let f = f_neumaier(&w, &[1e16, 1.0, -1e16]);
        assert_eq!(f[0], 1.0);
    }

    /// Somme exacte arrondie (partiels de Shewchuk), pour juger la précision de la somme compensée.
    fn somme_exacte(v: &[f64]) -> f64 {
        let mut partiels: Vec<f64> = Vec::new();
        for &x0 in v {
            let mut x = x0;
            let mut i = 0;
            for k in 0..partiels.len() {
                let mut y = partiels[k];
                if x.abs() < y.abs() {
                    std::mem::swap(&mut x, &mut y);
                }
                let hi = x + y;
                let lo = y - (hi - x);
                if lo != 0.0 {
                    partiels[i] = lo;
                    i += 1;
                }
                x = hi;
            }
            partiels.truncate(i);
            partiels.push(x);
        }
        partiels.iter().fold(0.0, |s, p| s + p)
    }

    #[test]
    fn segments_couvrent_chaque_ligne() {
        assert_eq!(
            pointeurs_segments(&[0, 3, 3 + 2048, 3 + 2048 + 2049], 1024),
            vec![0, 1, 3, 6]
        );
        assert_eq!(pointeurs_segments(&[0, 1024, 1025], 1024), vec![0, 1, 2]);
    }

    #[test]
    fn neumaier_par_segments_proche_de_la_somme_exacte() {
        // une ligne de 5000 termes mal conditionnés (grandeurs étalées sur 12 décades, signes alternés), 5 segments
        let n = 5000;
        let val: Vec<f64> = (0..n).map(|k| if k % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let pv: Vec<f64> = (0..n)
            .map(|k| (1.0 + (k as f64) * 0.37).sin() * 10f64.powi((k % 13) as i32 - 6))
            .collect();
        let w = PoidsFourier {
            nf: 1,
            ptr: vec![0, n as u64],
            idx: (0..n as u32).collect(),
            val,
        };
        let f = f_neumaier(&w, &pv)[0];
        let termes: Vec<f64> = (0..n).map(|k| w.val[k] * pv[k]).collect();
        let exact = somme_exacte(&termes);
        let somme_abs: f64 = termes.iter().map(|x| x.abs()).sum();
        let tol = 4.0 * f64::EPSILON * exact.abs() + (n as f64) * f64::EPSILON * f64::EPSILON * somme_abs;
        assert!((f - exact).abs() <= tol, "{f} contre {exact} (tol {tol:e})");
        // la somme naïve est, elle, nettement moins précise (le test distingue bien les deux)
        let naif: f64 = termes.iter().sum();
        assert!((f - exact).abs() <= (naif - exact).abs());
    }

    #[test]
    fn f_ne_depend_pas_du_nombre_de_fils() {
        let n = 3000;
        let w = PoidsFourier {
            nf: 2,
            ptr: vec![0, 1500, 3000],
            idx: (0..n as u32).map(|k| k % 1000).collect(),
            val: (0..n).map(|k| 1.0 / (1.0 + k as f64)).collect(),
        };
        let pv: Vec<f64> = (0..1000).map(|k| ((k * 7919) % 101) as f64 - 50.0).collect();
        let r = f_neumaier(&w, &pv);
        for fils in [1usize, 3] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(fils).build().unwrap();
            assert_eq!(pool.install(|| f_neumaier(&w, &pv)), r);
        }
    }

    #[test]
    fn blocs_somme_constante() {
        let l = 4;
        let pv = vec![0.25; 6 * 256];
        let b = somme_blocs(l, &pv);
        assert_eq!(b.len(), 6 * 16);
        assert!(b.iter().all(|&x| x == 4.0));
    }

    #[test]
    fn blocs_ordre_numpy_deux_voisins_x3_d_abord() {
        // 1e16, 1, −1e16, 1 …  l'ordre (paire x3 d'abord) donne un résultat précis qui le distingue d'un ordre séquentiel.
        let l = 2;
        let mut pv = vec![0.0; 6 * 16];
        // bloc unique par plan : sites s = 0..16 ; on met des grandeurs qui rendent l'ordre visible
        let vals = [
            1e16, 1.0, -1e16, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        pv[..16].copy_from_slice(&vals);
        let b = somme_blocs(l, &pv);
        // paires : (1e16 + 1) = 1e16, (−1e16 + 1) = −1e16 + 1 (représentable ? −9999999999999999 → arrondi à −1e16) ; somme 0
        let attendu = {
            let mut acc = 0.0f64;
            for k in 0..8 {
                acc += vals[2 * k] + vals[2 * k + 1];
            }
            acc as f32
        };
        assert_eq!(b[0], attendu);
    }

    /// Construction DENSE indépendante de D = Σ ligne ligneᵀ, par élément de fibre.
    #[allow(clippy::needless_range_loop)]
    fn d_dense(plan: &PlanFibre, w: &PoidsFourier, c: &ChampHote, mode: ModeFibre) -> Vec<f64> {
        let (_, gp) = plaquettes_gradients(c);
        let l = c.l;
        let v = c.v();
        let nf = w.nf;
        // Wf dense : nf × 6V
        let mut wd = vec![0.0; nf * 6 * v];
        for r in 0..nf {
            for q in w.ptr[r] as usize..w.ptr[r + 1] as usize {
                wd[r * 6 * v + w.idx[q] as usize] = w.val[q];
            }
        }
        let grad = |e: usize, a: usize| -> Vec<f64> {
            let mut g = vec![0.0; nf];
            for j in 0..6 {
                let x = plan.lp[6 * e + j] as usize;
                let (p, pos) = (x / 4, x % 4);
                for r in 0..nf {
                    g[r] += gp[(4 * p + pos) * 3 + a] * wd[r * 6 * v + p];
                }
            }
            g
        };
        let mut d = vec![0.0; nf * nf];
        let mut ajoute = |lg: &[f64]| {
            for i in 0..nf {
                for j in 0..nf {
                    d[i * nf + j] += lg[i] * lg[j];
                }
            }
        };
        let _ = l;
        match mode {
            ModeFibre::Libre => {
                for e in 0..4 * v {
                    for a in 0..3 {
                        ajoute(&grad(e, a));
                    }
                }
            }
            ModeFibre::Conditionnel => {
                for &e in &plan.lfree {
                    for a in 0..3 {
                        ajoute(&grad(e as usize, a));
                    }
                }
                for (&e1, &e2) in plan.e1.iter().zip(&plan.e2) {
                    let r = rotation(lien(&c.q, e1 as usize));
                    for a in 0..3 {
                        let g1 = grad(e1 as usize, a);
                        let g2: Vec<Vec<f64>> = (0..3).map(|b| grad(e2 as usize, b)).collect();
                        let lg: Vec<f64> = (0..nf)
                            .map(|i| (g1[i] - (r[0][a] * g2[0][i] + r[1][a] * g2[1][i] + r[2][a] * g2[2][i])) * S2I)
                            .collect();
                        ajoute(&lg);
                    }
                }
            }
        }
        d
    }

    fn rel_frob(a: &[f64], b: &[f64]) -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
        let den: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();
        num / if den > 0.0 { den } else { 1.0 }
    }

    #[test]
    fn d_tranche_egal_construction_dense() {
        // L = 4 (une tranche partielle par direction) et L = 6 (3 tranches dont une partielle, sous-lots incomplets).
        for l in [4usize, 6] {
            let geo = GeometrieCond::nouvelle(l).unwrap();
            let w = poids_fourier(l, 1);
            let plan = PlanFibre::nouveau(&geo, &w).unwrap();
            let c = champ_aleatoire(l, 11);
            let (_, gp) = plaquettes_gradients(&c);
            for mode in [ModeFibre::Conditionnel, ModeFibre::Libre] {
                let d = matrice_d(&plan, &c.q, &gp, mode);
                let dd = d_dense(&plan, &w, &c, mode);
                assert!(rel_frob(&d, &dd) < 1e-13, "L = {l} {mode:?} : {}", rel_frob(&d, &dd));
                // symétrique exactement
                for i in 0..54 {
                    for j in 0..54 {
                        assert_eq!(d[i * 54 + j], d[j * 54 + i]);
                    }
                }
            }
        }
    }

    #[test]
    fn d_semi_definie_positive_et_non_nulle() {
        let l = 4;
        let geo = GeometrieCond::nouvelle(l).unwrap();
        let w = poids_fourier(l, 1);
        let plan = PlanFibre::nouveau(&geo, &w).unwrap();
        let c = champ_aleatoire(l, 5);
        let (_, gp) = plaquettes_gradients(&c);
        let d = matrice_d(&plan, &c.q, &gp, ModeFibre::Conditionnel);
        assert!(d.iter().any(|&x| x != 0.0));
        for k in 0..54 {
            assert!(d[k * 54 + k] >= 0.0, "diagonale négative");
        }
        // xᵀ D x ≥ 0 pour quelques vecteurs
        for graine in 0..5u64 {
            let x: Vec<f64> = (0..54)
                .map(|i| ((i as u64 * 31 + graine * 17) % 13) as f64 - 6.0)
                .collect();
            let mut q = 0.0;
            for i in 0..54 {
                for j in 0..54 {
                    q += x[i] * d[i * 54 + j] * x[j];
                }
            }
            assert!(q >= -1e-9);
        }
    }

    #[test]
    fn plate_et_pi_flux_donnent_d_nul() {
        let l = 4;
        let geo = GeometrieCond::nouvelle(l).unwrap();
        let w = poids_fourier(l, 1);
        let plan = PlanFibre::nouveau(&geo, &w).unwrap();
        // pi-flux : liens ±1 (cfg_pi de test_concordance.py), toutes plaquettes = −1
        let mut c = ChampHote::froid(l).unwrap();
        let v = c.v();
        for mu in 0..4usize {
            for s in 0..v {
                let x = [s / 64, (s / 16) % 4, (s / 4) % 4, s % 4];
                let n: usize = match mu {
                    0 => 0,
                    1 => x[0],
                    2 => x[0] + x[1],
                    _ => x[0] + x[1] + x[2],
                };
                c.q[4 * (mu * v + s)] = if n.is_multiple_of(2) { 1.0 } else { -1.0 };
            }
        }
        for champ in [ChampHote::froid(l).unwrap(), c] {
            let (pv, gp) = plaquettes_gradients(&champ);
            assert!(gp.iter().all(|&x| x == 0.0));
            assert!(pv.iter().all(|&x| x.abs() == 1.0));
            for mode in [ModeFibre::Conditionnel, ModeFibre::Libre] {
                let d = matrice_d(&plan, &champ.q, &gp, mode);
                assert!(d.iter().all(|&x| x == 0.0), "D = 0 exactement");
            }
        }
    }

    #[test]
    fn d_independant_du_nombre_de_fils() {
        let l = 4;
        let geo = GeometrieCond::nouvelle(l).unwrap();
        let w = poids_fourier(l, 1);
        let plan = PlanFibre::nouveau(&geo, &w).unwrap();
        let c = champ_aleatoire(l, 21);
        let (_, gp) = plaquettes_gradients(&c);
        let ref_d = matrice_d(&plan, &c.q, &gp, ModeFibre::Conditionnel);
        for fils in [1usize, 3] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(fils).build().unwrap();
            let d = pool.install(|| matrice_d(&plan, &c.q, &gp, ModeFibre::Conditionnel));
            assert_eq!(d, ref_d, "{fils} fils");
        }
    }
}
