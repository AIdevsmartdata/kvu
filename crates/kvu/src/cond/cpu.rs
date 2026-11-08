//! Référence CPU (rayon) des deux modes du bain conditionnel, en quaternions FP64. Propriétaire : T5.
//!
//! Mêmes formules, dans le même ordre d'évaluation, que `noyaux/cond.cu` (gate EQC-3 : CPU contre GPU ≤ 1e-12,
//! 0 décision divergente) et que `cb.sweep` (gate EQC-1 : ≤ 1e-12 par balayage). Un balayage = `cb.sweep` :
//! réunitarisation, 4 directions de paires, puis 4 directions × 2 parités de liens libres.
//!
//! Les mises à jour d'une direction n'écrivent que des liens que les agrafes de la MÊME étape ne lisent pas
//! (paires : les agrafes lisent des liens libres ; libres de parité p : elles lisent la parité opposée ou des
//! paires), donc le calcul en deux temps (lecture de toutes les agrafes, puis écriture) est exactement le calcul
//! en place de cb.py.

use super::geometrie::GeometrieCond;
use super::numpy_rng::GenerateurNumpy;
use crate::commun::{Erreur, Graine, Resultat, ETIQUETTE_COND};
use crate::heritage::su2_hb::{philox4x32_10, qadd, qdag, qmul, staple, u01, Geo, Q};
use rayon::prelude::*;
use std::f64::consts::PI;

/// Plafond d'essais d'un tirage de x₀ (comme K1 : 10⁶) ; au-delà le lien reste inchangé et le plafond est compté.
pub const PLAFOND_ESSAIS: u32 = 1_000_000;

/// Seuil sous lequel la norme² d'une agrafe est ramenée à 1e-300 (= `max(det, 1e-300)` de cb.py).
const NORME2_MIN: f64 = 1e-300;

/// Réunitarisation de tous les liens (= `_k_reunit` : q ← q / |q|).
pub fn reunitariser(u: &mut [Q]) {
    u.par_iter_mut().for_each(|q| {
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        *q = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
    });
}

/// Agrafe somme des 6 plaquettes du lien (mu, s) (= `_staple` de cb.py = `staple` de su2_hb).
#[inline]
pub fn agrafe(u: &[Q], g: &Geo, mu: usize, s: usize) -> Q {
    staple(u, g, mu, s, &g.coords(s))
}

/// (Vn, α) d'une agrafe effective A : k = √max(|A|², 1e-300), Vn = A / k, α = β k.
#[inline]
pub fn preparer(a: Q, beta: f64) -> (Q, f64) {
    let n2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3];
    let k = n2.max(NORME2_MIN).sqrt();
    ([a[0] / k, a[1] / k, a[2] / k, a[3] / k], beta * k)
}

/// A_eff = A₁ + A₂† V† d'une paire : agrafes aux deux liens (premier lien en s1, second en s2).
#[inline]
pub fn agrafe_effective_paire(u: &[Q], g: &Geo, mu: usize, s1: usize, s2: usize, v: Q) -> Q {
    let a1 = agrafe(u, g, mu, s1);
    let a2 = agrafe(u, g, mu, s2);
    qadd(a1, qmul(qdag(a2), qdag(v)))
}

/// U₁ = W Vn† avec W = (x0, r v0, r v1, r v2) (= `_hb_U` de cb.py).
#[inline]
pub fn lien_depuis_tirage(x0: f64, r: f64, v: [f64; 3], vn: Q) -> Q {
    qmul([x0, r * v[0], r * v[1], r * v[2]], qdag(vn))
}

/// Second lien d'une paire : U₂ = U₁† V.
#[inline]
pub fn second_lien(u1: Q, v: Q) -> Q {
    qmul(qdag(u1), v)
}

/// Balayage `ref_rng` : tirages numpy de l'hôte (`GenerateurNumpy`), agrafes et mises à jour en quaternions.
/// β = 0 refusé (la référence ne termine pas : α = 0 donne x = NaN).
pub fn balayage_ref_rng(
    u: &mut [Q],
    geo: &GeometrieCond,
    g: &Geo,
    vfix: &[Q],
    beta: f64,
    rng: &mut GenerateurNumpy,
) -> Resultat<()> {
    if beta == 0.0 {
        return Err(Erreur::Config(
            "β = 0 : le bain thermique de référence (mode ref_rng) ne termine pas (α = 0 donne x = NaN) ; utiliser le mode rapide".into(),
        ));
    }
    let np = geo.npair();
    reunitariser(u);
    for mu in 0..4 {
        let prep: Vec<(Q, f64)> = (0..np)
            .into_par_iter()
            .map(|i| {
                let a = agrafe_effective_paire(
                    u,
                    g,
                    mu,
                    geo.p1[mu][i] as usize,
                    geo.p1pm[mu][i] as usize,
                    vfix[mu * np + i],
                );
                preparer(a, beta)
            })
            .collect();
        let alpha: Vec<f64> = prep.iter().map(|p| p.1).collect();
        let (x0, r, v) = rng.hb_numpy(&alpha)?;
        let nouveaux: Vec<(Q, Q)> = (0..np)
            .into_par_iter()
            .map(|i| {
                let u1 = lien_depuis_tirage(x0[i], r[i], v[i], prep[i].0);
                (u1, second_lien(u1, vfix[mu * np + i]))
            })
            .collect();
        for (i, (u1, u2)) in nouveaux.into_iter().enumerate() {
            u[mu * geo.v + geo.p1[mu][i] as usize] = u1;
            u[mu * geo.v + geo.p1pm[mu][i] as usize] = u2;
        }
    }
    for mu in 0..4 {
        for p in 0..2 {
            let sites = &geo.fr[mu][p];
            let prep: Vec<(Q, f64)> = sites.par_iter().map(|&s| preparer(agrafe(u, g, mu, s as usize), beta)).collect();
            let alpha: Vec<f64> = prep.iter().map(|q| q.1).collect();
            let (x0, r, v) = rng.hb_numpy(&alpha)?;
            let nouveaux: Vec<Q> =
                (0..sites.len()).into_par_iter().map(|i| lien_depuis_tirage(x0[i], r[i], v[i], prep[i].0)).collect();
            for (i, q) in nouveaux.into_iter().enumerate() {
                u[mu * geo.v + sites[i] as usize] = q;
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------
// Mode rapide : Philox à compteur (li, sweep_id, tirage, ETIQUETTE_COND), li = indice LOCAL du lien
// ------------------------------------------------------------------------------------------------------------

/// Un tirage Philox : numéro de tirage `draw` incrémenté à chaque appel.
#[inline]
fn ph(li: u32, sweep_id: u32, draw: &mut u32, k0: u32, k1: u32) -> [u32; 4] {
    let r = philox4x32_10([li, sweep_id, *draw, ETIQUETTE_COND], k0, k1);
    *draw = draw.wrapping_add(1);
    r
}

/// Tirage de x₀ ~ √(1−x₀²) e^{α x₀} sur [−1, 1] (mode rapide). Rend (x₀, essais, ok).
/// - α ≥ 2 : Kennedy–Pendleton (2 appels Philox par essai, comme K1) ;
/// - 1e-12 ≤ α < 2 : proposition exponentielle de cb (`1 + log1p(−u·(−expm1(−2α)))/α`), acceptation `u₂ < √(1−x²)` ;
/// - α < 1e-12 : proposition uniforme (limite exacte à O(α) ; β = 0 admis).
pub fn x0_rapide(alpha: f64, li: u32, sweep_id: u32, draw: &mut u32, k0: u32, k1: u32) -> (f64, u32, bool) {
    let mut ntry = 0u32;
    if alpha >= 2.0 {
        while ntry < PLAFOND_ESSAIS {
            let r1 = ph(li, sweep_id, draw, k0, k1);
            let r2 = ph(li, sweep_id, draw, k0, k1);
            let (a1, a2, a3, a4) = (u01(r1[0], r1[1]), u01(r1[2], r1[3]), u01(r2[0], r2[1]), u01(r2[2], r2[3]));
            let c = (2.0 * PI * a2).cos();
            let lam2 = -(a1.ln() + c * c * a3.ln()) / (2.0 * alpha);
            ntry += 1;
            if a4 * a4 <= 1.0 - lam2 {
                return (1.0 - 2.0 * lam2, ntry, true);
            }
        }
    } else {
        let em = -(-2.0 * alpha).exp_m1();
        while ntry < PLAFOND_ESSAIS {
            let r1 = ph(li, sweep_id, draw, k0, k1);
            let (a1, a2) = (u01(r1[0], r1[1]), u01(r1[2], r1[3]));
            let x = if alpha < 1e-12 { 2.0 * a1 - 1.0 } else { 1.0 + (-a1 * em).ln_1p() / alpha };
            ntry += 1;
            if a2 < (1.0 - x * x).max(0.0).sqrt() {
                return (x, ntry, true);
            }
        }
    }
    (1.0, ntry, false)
}

/// Loi du tirage rapide sur le CPU : `n` tirages de x₀ pour UN α, mêmes compteurs que `k_cond_test_x0`.
/// Rend (x₀, nombre de tirages au plafond).
pub fn tirer_x0(alpha: f64, n: usize, graine: Graine) -> (Vec<f64>, u64) {
    let (k0, k1) = graine.cle();
    let res: Vec<(f64, bool)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut draw = 0u32;
            let (x, _, ok) = x0_rapide(alpha, i as u32, 0x7E57, &mut draw, k0, k1);
            (x, ok)
        })
        .collect();
    let plafond = res.iter().filter(|r| !r.1).count() as u64;
    (res.into_iter().map(|r| r.0).collect(), plafond)
}

/// W = (x₀, r sinθ cosφ, r sinθ sinφ, r cosθ) : direction uniforme par (cosθ, φ), un appel Philox.
#[inline]
fn w_depuis_x0(x0: f64, li: u32, sweep_id: u32, draw: &mut u32, k0: u32, k1: u32) -> Q {
    let rd = ph(li, sweep_id, draw, k0, k1);
    let ct = 2.0 * u01(rd[0], rd[1]) - 1.0;
    let phi = 2.0 * PI * u01(rd[2], rd[3]);
    let st = (1.0 - ct * ct).max(0.0).sqrt();
    let rr = (1.0 - x0 * x0).max(0.0).sqrt();
    [x0, rr * st * phi.cos(), rr * st * phi.sin(), rr * ct]
}

/// Mise à jour rapide d'un lien : (nouveau lien, essais, ok) pour l'agrafe effective `a`.
#[inline]
pub fn lien_rapide(a: Q, beta: f64, li: u32, sweep_id: u32, k0: u32, k1: u32) -> (Q, u32, bool) {
    let (vn, alpha) = preparer(a, beta);
    let mut draw = 0u32;
    let (x0, ntry, ok) = x0_rapide(alpha, li, sweep_id, &mut draw, k0, k1);
    if !ok {
        return ([1.0, 0.0, 0.0, 0.0], ntry, false);
    }
    let w = w_depuis_x0(x0, li, sweep_id, &mut draw, k0, k1);
    (qmul(w, qdag(vn)), ntry, true)
}

/// Compteurs de diagnostic d'un balayage rapide.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiagRapide {
    /// Tirages ayant atteint le plafond d'essais (attendu : 0).
    pub plafond: u64,
    /// Nombre d'essais par lien (indice mu·V + s du premier lien), si demandé ; 0 = lien non tiré.
    pub essais: Vec<u32>,
}

/// Paramètres d'un balayage rapide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamRapide {
    pub beta: f64,
    pub graine: Graine,
    pub sweep_id: u32,
}

/// Balayage rapide : tout en Philox. `diag.essais` (de taille 4V) est rempli si non vide.
pub fn balayage_rapide(u: &mut [Q], geo: &GeometrieCond, g: &Geo, vfix: &[Q], p: ParamRapide, diag: &mut DiagRapide) {
    let ParamRapide { beta, graine, sweep_id } = p;
    let (k0, k1) = graine.cle();
    let np = geo.npair();
    reunitariser(u);
    let noter = diag.essais.len() == 4 * geo.v;
    for mu in 0..4 {
        let res: Vec<(Q, Q, u32, bool)> = (0..np)
            .into_par_iter()
            .map(|i| {
                let (s1, s2) = (geo.p1[mu][i] as usize, geo.p1pm[mu][i] as usize);
                let v = vfix[mu * np + i];
                let a = agrafe_effective_paire(u, g, mu, s1, s2, v);
                let (u1, nt, ok) = lien_rapide(a, beta, (mu * geo.v + s1) as u32, sweep_id, k0, k1);
                (u1, second_lien(u1, v), nt, ok)
            })
            .collect();
        for (i, (u1, u2, nt, ok)) in res.into_iter().enumerate() {
            let s1 = geo.p1[mu][i] as usize;
            if noter {
                diag.essais[mu * geo.v + s1] = if ok { nt } else { 0 };
            }
            if ok {
                u[mu * geo.v + s1] = u1;
                u[mu * geo.v + geo.p1pm[mu][i] as usize] = u2;
            } else {
                diag.plafond += 1;
            }
        }
    }
    for mu in 0..4 {
        for p in 0..2 {
            let sites = &geo.fr[mu][p];
            let res: Vec<(Q, u32, bool)> = sites
                .par_iter()
                .map(|&s| lien_rapide(agrafe(u, g, mu, s as usize), beta, (mu * geo.v) as u32 + s, sweep_id, k0, k1))
                .collect();
            for (i, (q, nt, ok)) in res.into_iter().enumerate() {
                let li = mu * geo.v + sites[i] as usize;
                if noter {
                    diag.essais[li] = if ok { nt } else { 0 };
                }
                if ok {
                    u[li] = q;
                } else {
                    diag.plafond += 1;
                }
            }
        }
    }
}

/// (max |V(U) − Vfix|, max | |q|² − 1 |) : les gates Vdev et unitarité de run_one.py
/// (|q|² − 1 est l'écart de UU† à l'identité).
pub fn controles(u: &[Q], geo: &GeometrieCond, vfix: &[Q]) -> (f64, f64) {
    let np = geo.npair();
    let mut vdev = 0.0f64;
    for mu in 0..4 {
        for i in 0..np {
            let v = qmul(u[mu * geo.v + geo.p1[mu][i] as usize], u[mu * geo.v + geo.p1pm[mu][i] as usize]);
            let f = vfix[mu * np + i];
            for k in 0..4 {
                vdev = vdev.max((v[k] - f[k]).abs());
            }
        }
    }
    let unit =
        u.iter().map(|q| (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] - 1.0).abs()).fold(0.0, f64::max);
    (vdev, unit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commun::ChampHote;
    use crate::heritage::su2_hb::qnorm;

    /// Champ de test déterministe : liens unitaires pseudo-aléatoires (Philox), unitaires à 1e-16.
    pub(crate) fn champ_test(l: usize, graine: u32) -> Vec<Q> {
        (0..4 * l.pow(4))
            .map(|i| {
                let r = philox4x32_10([i as u32, graine, 0, 0xABCD], 1, 2);
                let q = [u01(r[0], r[1]) - 0.5, u01(r[2], r[3]) - 0.5, u01(r[1], r[2]) - 0.5, u01(r[3], r[0]) - 0.5];
                let n = qnorm(q);
                [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
            })
            .collect()
    }

    fn vfix_de(u: &[Q], geo: &GeometrieCond) -> Vec<Q> {
        let h = ChampHote { l: geo.l, q: u.iter().flatten().copied().collect() };
        super::super::geometrie::v_paires(&h, geo)
    }

    #[test]
    fn preparer_normalise_et_clamp() {
        let (vn, a) = preparer([3.0, 0.0, 4.0, 0.0], 2.0);
        assert!((qnorm(vn) - 1.0).abs() < 1e-15);
        assert!((a - 10.0).abs() < 1e-14);
        let (vn0, a0) = preparer([0.0; 4], 2.0);
        assert!(vn0.iter().all(|x| x.is_finite()) && a0.is_finite());
    }

    #[test]
    fn rapide_conserve_v_et_unitarite() {
        let geo = GeometrieCond::nouvelle(4).unwrap();
        let g = Geo::new(4);
        let mut u = champ_test(4, 1);
        let vfix = vfix_de(&u, &geo);
        let mut d = DiagRapide::default();
        for sw in 0..5 {
            balayage_rapide(
                &mut u,
                &geo,
                &g,
                &vfix,
                ParamRapide { beta: 2.3, graine: Graine(2709), sweep_id: sw },
                &mut d,
            );
        }
        let (vdev, unit) = controles(&u, &geo, &vfix);
        assert!(vdev < 1e-12, "Vdev = {vdev:e}");
        assert!(unit < 1e-12, "unitarité = {unit:e}");
        assert_eq!(d.plafond, 0);
    }

    #[test]
    fn rapide_accepte_beta_nul() {
        let geo = GeometrieCond::nouvelle(4).unwrap();
        let g = Geo::new(4);
        let mut u = champ_test(4, 2);
        let vfix = vfix_de(&u, &geo);
        let mut d = DiagRapide::default();
        balayage_rapide(&mut u, &geo, &g, &vfix, ParamRapide { beta: 0.0, graine: Graine(1), sweep_id: 0 }, &mut d);
        let (vdev, unit) = controles(&u, &geo, &vfix);
        assert!(vdev < 1e-12 && unit < 1e-12);
        assert!(u.iter().flatten().all(|x| x.is_finite()));
    }

    #[test]
    fn rapide_deterministe_et_depend_de_la_graine_et_du_sweep() {
        let geo = GeometrieCond::nouvelle(4).unwrap();
        let g = Geo::new(4);
        let u0 = champ_test(4, 3);
        let vfix = vfix_de(&u0, &geo);
        let faire = |graine: u64, sw: u32| {
            let mut u = u0.clone();
            let p = ParamRapide { beta: 2.8, graine: Graine(graine), sweep_id: sw };
            balayage_rapide(&mut u, &geo, &g, &vfix, p, &mut DiagRapide::default());
            u
        };
        assert_eq!(faire(5, 0), faire(5, 0));
        assert_ne!(faire(5, 0), faire(6, 0));
        assert_ne!(faire(5, 0), faire(5, 1));
    }

    #[test]
    fn ref_rng_refuse_beta_nul() {
        let geo = GeometrieCond::nouvelle(4).unwrap();
        let g = Geo::new(4);
        let mut u = champ_test(4, 4);
        let vfix = vfix_de(&u, &geo);
        let r = balayage_ref_rng(&mut u, &geo, &g, &vfix, 0.0, &mut GenerateurNumpy::depuis_graine(0));
        assert!(matches!(r, Err(Erreur::Config(_))));
    }

    #[test]
    fn ref_rng_conserve_v_et_unitarite() {
        let geo = GeometrieCond::nouvelle(4).unwrap();
        let g = Geo::new(4);
        let mut u = champ_test(4, 5);
        let vfix = vfix_de(&u, &geo);
        let mut rng = GenerateurNumpy::depuis_graine(7);
        for _ in 0..3 {
            balayage_ref_rng(&mut u, &geo, &g, &vfix, 2.3, &mut rng).unwrap();
        }
        let (vdev, unit) = controles(&u, &geo, &vfix);
        assert!(vdev < 1e-12 && unit < 1e-12, "vdev {vdev:e} unit {unit:e}");
    }

    #[test]
    fn plafond_d_essais_atteint_sans_blocage() {
        // α = NaN : jamais d'acceptation ; le tirage s'arrête au plafond et le signale (lien laissé inchangé)
        let mut draw = 0;
        let (x, ntry, ok) = x0_rapide(f64::NAN, 0, 0, &mut draw, 1, 2);
        assert!(!ok);
        assert_eq!(ntry, PLAFOND_ESSAIS);
        assert_eq!(x, 1.0);
        let (q, nt, ok) = lien_rapide([1.0, 0.0, 0.0, 0.0], f64::NAN, 5, 0, 1, 2);
        assert!(!ok && nt == PLAFOND_ESSAIS);
        assert_eq!(q, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn tirer_x0_cpu_deterministe_et_dans_les_bornes() {
        let (a, ca) = tirer_x0(5.0, 1000, Graine(3));
        let (b, _) = tirer_x0(5.0, 1000, Graine(3));
        let (c, _) = tirer_x0(5.0, 1000, Graine(4));
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(ca, 0);
        assert!(a.iter().all(|x| (-1.0..=1.0).contains(x)));
    }

    /// Loi du tirage rapide : E[x₀] = I₂(α)/I₁(α) (|z| < 4) pour les deux branches et α → 0 (essai interne ;
    /// la gate complète, KS compris, est dans la validation).
    #[test]
    fn loi_du_tirage_rapide_moyenne_exacte() {
        // E[x0] = I2(a)/I1(a) par quadrature de Gauss-Tchebychev de 2e espece (exacte pour ce poids)
        let moment = |a: f64| {
            let n = 4000;
            let (mut num, mut den) = (0.0, 0.0);
            for j in 1..=n {
                let th = PI * j as f64 / (n + 1) as f64;
                let x = th.cos();
                let w = th.sin().powi(2) * (a * (x - 1.0)).exp();
                num += w * x;
                den += w;
            }
            num / den
        };
        for &alpha in &[0.0, 0.5, 1.9, 2.0, 5.0, 30.0, 200.0] {
            let n = 200_000u32;
            let (mut s, mut s2) = (0.0, 0.0);
            for i in 0..n {
                let mut draw = 0;
                let (x, _, ok) = x0_rapide(alpha, i, 0x7E57, &mut draw, 17, 3);
                assert!(ok);
                s += x;
                s2 += x * x;
            }
            let m = s / n as f64;
            let var = s2 / n as f64 - m * m;
            let z = (m - moment(alpha)) / (var / n as f64).sqrt();
            assert!(z.abs() < 4.0, "alpha = {alpha} : moyenne {m}, attendue {}, z = {z}", moment(alpha));
        }
    }
}
