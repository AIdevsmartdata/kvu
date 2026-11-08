//! Coordonnées ADDITIVES (S2, `lat.py::hess_su2_additive`) : U_l = exp(i A_l·σ/2), A_l → A_l + ζ_l.
//!
//! Relation exacte avec les coordonnées « gauche » (U → exp(i ξ·σ/2) U), en base s2 :
//! - premier ordre, exact en tout point : ξ = D(A) ζ avec `D = s I + k A Aᵀ − m [A×]`, r = |A|, s = sin r / r,
//!   k = (1 − s)/r², m = (1 − cos r)/r² (la rotation intégrée ∫₀¹ Rot(n, −s r) ds) ;
//! - gradient additif : g_add(A) = D(A)ᵀ g_gauche(U(A)) ;
//! - H_add v = Dᵀ [ H_gauche (D v) − ½ (D v) × g ] + (dD[v])ᵀ g, où dD[v] est la dérivée de D(A) dans la direction v
//!   (nombres duaux) et le terme −½ ξ × g est la correction de commutateur exp(ζ)exp(tξ) = exp(ζ + tξ − (t/2) ζ×ξ + …)
//!   (U(t) = exp(tξ)U : la dérivée du gradient « gauche » le long de la courbe n'est pas H_gauche ξ, elle est H_gauche ξ − ½ ξ×g).
//!   La formule est contrôlée contre `lat.py` (HESS-EXACT (b), 1e-10).

use super::dual::Dual;

/// Séries et formes closes de s, k, m en x = r² : renvoie ((s, ds/dx), (k, dk/dx), (m, dm/dx)).
fn fonctions(x: f64) -> [(f64, f64); 3] {
    if x < 0.25 {
        // séries en x (termes jusqu'à x⁶)
        const S: [f64; 7] = [1.0, -1.0 / 6.0, 1.0 / 120.0, -1.0 / 5040.0, 1.0 / 362_880.0, -1.0 / 39_916_800.0, 1.0 / 6_227_020_800.0];
        const K: [f64; 7] = [
            1.0 / 6.0,
            -1.0 / 120.0,
            1.0 / 5040.0,
            -1.0 / 362_880.0,
            1.0 / 39_916_800.0,
            -1.0 / 6_227_020_800.0,
            1.0 / 1_307_674_368_000.0,
        ];
        const M: [f64; 7] = [
            1.0 / 2.0,
            -1.0 / 24.0,
            1.0 / 720.0,
            -1.0 / 40_320.0,
            1.0 / 3_628_800.0,
            -1.0 / 479_001_600.0,
            1.0 / 87_178_291_200.0,
        ];
        let horner = |c: &[f64; 7]| {
            let (mut f, mut df) = (0.0, 0.0);
            for i in (0..7).rev() {
                df = df * x + f;
                f = f * x + c[i];
            }
            (f, df)
        };
        [horner(&S), horner(&K), horner(&M)]
    } else {
        let r = x.sqrt();
        let (sn, cs) = (r.sin(), r.cos());
        let s = sn / r;
        let ds = (r * cs - sn) / (2.0 * r * x);
        let k = (1.0 - s) / x;
        let dk = (-ds - k) / x;
        let m = (1.0 - cs) / x;
        let dm = (sn / (2.0 * r) - m) / x;
        [(s, ds), (k, dk), (m, dm)]
    }
}

/// D(A) et sa dérivée dans la direction `v` (matrices 3×3, ligne-majeure).
pub fn d_et_dd(a: [f64; 3], v: [f64; 3]) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    let x = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
    let dx = 2.0 * (a[0] * v[0] + a[1] * v[1] + a[2] * v[2]);
    let f = fonctions(x);
    let compo = |i: usize| Dual::<1> { v: f[i].0, d: [f[i].1 * dx] };
    let (s, k, m) = (compo(0), compo(1), compo(2));
    let ad: [Dual<1>; 3] = std::array::from_fn(|i| Dual { v: a[i], d: [v[i]] });
    let mut d = [[0.0; 3]; 3];
    let mut dd = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            // s δ_ij + k A_i A_j − m (A×)_ij ; (A×)_ij = −ε_ijk A_k
            let mut t = k.mul(ad[i]).mul(ad[j]);
            if i == j {
                t = t.add(s);
            }
            let croix = match (i, j) {
                (0, 1) => ad[2].neg(),
                (0, 2) => ad[1],
                (1, 0) => ad[2],
                (1, 2) => ad[0].neg(),
                (2, 0) => ad[1].neg(),
                (2, 1) => ad[0],
                _ => Dual::cst(0.0),
            };
            t = t.sub(m.mul(croix));
            d[i][j] = t.v;
            dd[i][j] = t.d[0];
        }
    }
    (d, dd)
}

#[inline]
pub fn mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

#[inline]
pub fn matt_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[0][i] * v[0] + m[1][i] * v[1] + m[2][i] * v[2])
}

#[inline]
pub fn croix(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raccord_serie_forme_close() {
        // continuité des valeurs et des dérivées de part et d'autre de x = 0,25
        let a = fonctions(0.25 - 1e-12);
        let b = fonctions(0.25 + 1e-12);
        for i in 0..3 {
            assert!((a[i].0 - b[i].0).abs() < 1e-11, "valeur {i}");
            assert!((a[i].1 - b[i].1).abs() < 1e-9, "dérivée {i}");
        }
    }

    #[test]
    fn dd_contre_differences_finies() {
        let a = [0.7, -0.4, 1.3];
        let v = [0.2, 0.9, -0.5];
        let (_, dd) = d_et_dd(a, v);
        let h = 1e-6;
        let ap: [f64; 3] = std::array::from_fn(|i| a[i] + h * v[i]);
        let am: [f64; 3] = std::array::from_fn(|i| a[i] - h * v[i]);
        let (dp, _) = d_et_dd(ap, v);
        let (dm, _) = d_et_dd(am, v);
        for i in 0..3 {
            for j in 0..3 {
                assert!((dd[i][j] - (dp[i][j] - dm[i][j]) / (2.0 * h)).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn petits_champs() {
        let (d, dd) = d_et_dd([0.0; 3], [1.0, 0.0, 0.0]);
        for i in 0..3 {
            for j in 0..3 {
                assert!((d[i][j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-15);
            }
        }
        // dD/dA en 0 : −m (v×) avec m = 1/2
        assert!((dd[1][2] - 0.5).abs() < 1e-15);
        assert!(dd.iter().flatten().all(|x| x.is_finite()));
    }
}
