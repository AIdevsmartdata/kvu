//! Champs par site pour des côtés QUELCONQUES (l0, l1, l2, l3) : extension du module scanner (`obs champs`, 22/11/2026).
//!
//! Mêmes définitions que [`super::champs`] (action brute et après flot de Wilson RK3, énergie et charge au trèfle,
//! Polyakov dans les quatre directions, plaquettes à trace négative), mêmes opérations flottantes dans le même ordre ; seule la
//! géométrie change ([`GeoA`] au lieu de `Geo`). Sur une boîte cubique, ce chemin redonne au bit près celui de `champs.rs`
//! (test `cotes_cubique_identique_au_chemin_cubique`) ; la commande n'y passe pourtant que pour une boîte NON cubique.
//!
//! Disposition : site s = ((x0·l1 + x1)·l2 + x2)·l3 + x3, NPY de forme (l0, l1, l2, l3) indexé [x0, x1, x2, x3] ;
//! `polyakov` : (4, l0, l1, l2, l3).
//! Le flot (RK3) est le flot de Wilson ISOTROPE : les poids d'une configuration anisotrope ne servent pas à la définition
//! des champs (ils n'interviennent que dans la génération).

use super::champs::{ChampsSite, ParamsChamps, Porte, SEUIL};
use super::ref_cpu::{exp_su2, INV_2PI2, PLANS, UN};
use crate::commun::{Erreur, Resultat};
use crate::geom::GeoA;
use crate::heritage::su2_hb::{qadd, qdag, qmul, qnorm, qscale, Q};
use rayon::prelude::*;

const COMPLEMENT: [usize; 6] = [5, 4, 3, 2, 1, 0];
const SIGNE_EPS: [f64; 6] = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0];

/// Liens en quaternions : li = μ·V + s.
pub struct ChampG {
    pub geo: GeoA,
    pub u: Vec<Q>,
}

impl ChampG {
    pub fn nouveau(geo: GeoA, q: &[f64]) -> Resultat<ChampG> {
        if q.len() != 16 * geo.v {
            return Err(Erreur::Config(format!(
                "champ hôte incohérent : côtés {:?}, {} réels (attendu {})",
                geo.d,
                q.len(),
                16 * geo.v
            )));
        }
        Ok(ChampG {
            geo,
            u: q.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect(),
        })
    }
}

#[inline]
fn voisin(geo: &GeoA, s: usize, d: usize, pas: i32) -> usize {
    let xd = geo.coords(s)[d];
    if pas > 0 {
        geo.nb_up(s, d, xd)
    } else {
        geo.nb_dn(s, d, xd)
    }
}

#[inline]
fn lien(geo: &GeoA, u: &[Q], d: usize, s: usize) -> Q {
    u[d * geo.v + s]
}

fn produit_chemin(geo: &GeoA, u: &[Q], s: usize, pas: &[(usize, i32)]) -> Q {
    let mut p = UN;
    let mut pos = s;
    for &(d, sens) in pas {
        if sens > 0 {
            p = qmul(p, lien(geo, u, d, pos));
            pos = voisin(geo, pos, d, 1);
        } else {
            pos = voisin(geo, pos, d, -1);
            p = qmul(p, qdag(lien(geo, u, d, pos)));
        }
    }
    p
}

/// Plaquette avant de coin s dans le plan (mu, nu), même association que `ref_cpu`.
#[inline]
fn plaquette(geo: &GeoA, u: &[Q], s: usize, x: &[usize; 4], mu: usize, nu: usize) -> Q {
    let smu = geo.nb_up(s, mu, x[mu]);
    let snu = geo.nb_up(s, nu, x[nu]);
    qmul(
        qmul(lien(geo, u, mu, s), lien(geo, u, nu, smu)),
        qmul(qdag(lien(geo, u, mu, snu)), qdag(lien(geo, u, nu, s))),
    )
}

/// Σ_{μ<ν} ½ Tr P_μν(s), même ordre de sommation que `ref_cpu::plaquette_site`.
pub fn plaquette_site(geo: &GeoA, u: &[Q], s: usize) -> f64 {
    let x = geo.coords(s);
    let mut acc = 0.0;
    for mu in 0..4 {
        for nu in mu + 1..4 {
            acc += plaquette(geo, u, s, &x, mu, nu)[0];
        }
    }
    acc
}

fn traces_plaquettes(geo: &GeoA, u: &[Q], s: usize) -> [f64; 6] {
    let x = geo.coords(s);
    let mut t = [0.0; 6];
    for (p, &(mu, nu)) in PLANS.iter().enumerate() {
        t[p] = plaquette(geo, u, s, &x, mu, nu)[0];
    }
    t
}

fn action_site(geo: &GeoA, u: &[Q], s: usize) -> f64 {
    4.0 * (6.0 - plaquette_site(geo, u, s))
}

fn negatives_site(geo: &GeoA, u: &[Q], s: usize) -> f64 {
    traces_plaquettes(geo, u, s).iter().filter(|&&t| t < 0.0).count() as f64
}

fn feuilles(mu: usize, nu: usize) -> [[(usize, i32); 4]; 4] {
    [
        [(mu, 1), (nu, 1), (mu, -1), (nu, -1)],
        [(nu, 1), (mu, -1), (nu, -1), (mu, 1)],
        [(mu, -1), (nu, -1), (mu, 1), (nu, 1)],
        [(nu, -1), (mu, 1), (nu, 1), (mu, -1)],
    ]
}

fn champ_trefle(geo: &GeoA, u: &[Q], s: usize) -> [[f64; 3]; 6] {
    let mut g = [[0.0; 3]; 6];
    for (p, &(mu, nu)) in PLANS.iter().enumerate() {
        let mut somme: Q = [0.0; 4];
        for chemin in feuilles(mu, nu) {
            somme = qadd(somme, produit_chemin(geo, u, s, &chemin));
        }
        g[p] = [0.25 * somme[1], 0.25 * somme[2], 0.25 * somme[3]];
    }
    g
}

/// (E_trèfle(s), q(s)), même arithmétique que `ref_cpu::densite_trefle_site`.
fn densite_trefle_site(geo: &GeoA, u: &[Q], s: usize) -> (f64, f64) {
    let g = champ_trefle(geo, u, s);
    let n2 = |v: [f64; 3]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut e = 0.0;
    for v in g {
        e += n2(v);
    }
    let e = 2.0 * e;
    let mut q = 0.0;
    for (p, &sg) in SIGNE_EPS.iter().enumerate().take(3) {
        q += sg * dot(g[p], g[COMPLEMENT[p]]);
    }
    (e, INV_2PI2 * q)
}

/// Agrafe (même ordre que `su2_hb::staple`).
#[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
fn agrafe(geo: &GeoA, u: &[Q], mu: usize, s: usize, x: &[usize; 4]) -> Q {
    let ld = |nu: usize, site: usize| u[nu * geo.v + site];
    let mut a: Q = [0.0; 4];
    let smu = geo.nb_up(s, mu, x[mu]);
    for nu in 0..4 {
        if nu == mu {
            continue;
        }
        let snu = geo.nb_up(s, nu, x[nu]);
        let sdn = geo.nb_dn(s, nu, x[nu]);
        let smu_dn = geo.nb_dn(smu, nu, x[nu]);
        let t = qmul(qmul(ld(nu, smu), qdag(ld(mu, snu))), qdag(ld(nu, s)));
        let b = qmul(qmul(qdag(ld(nu, smu_dn)), qdag(ld(mu, sdn))), ld(nu, sdn));
        a = qadd(a, qadd(t, b));
    }
    a
}

fn champ_z(u: &[Q], geo: &GeoA) -> Vec<[f64; 3]> {
    (0..4 * geo.v)
        .into_par_iter()
        .map(|li| {
            let (mu, s) = (li / geo.v, li % geo.v);
            let x = geo.coords(s);
            let a = agrafe(geo, u, mu, s, &x);
            let m = qmul(u[li], a);
            [-m[1], -m[2], -m[3]]
        })
        .collect()
}

fn maj_u(u: &mut [Q], etape: usize, dt: f64, z0: &[[f64; 3]], z1: &[[f64; 3]], z2: &[[f64; 3]]) {
    u.par_iter_mut().enumerate().for_each(|(li, q)| {
        let mut w = [0.0; 3];
        for (k, wk) in w.iter_mut().enumerate() {
            *wk = match etape {
                0 => z0[li][k] * (dt * 0.25),
                1 => (8.0 / 9.0 * z1[li][k] + (-17.0 / 36.0) * z0[li][k]) * dt,
                _ => ((3.0 / 4.0 * z2[li][k] + 17.0 / 36.0 * z0[li][k]) * 1.0 + (-8.0 / 9.0) * z1[li][k]) * dt,
            };
        }
        let n = qmul(exp_su2(w), *q);
        *q = qscale(n, 1.0 / qnorm(n));
    });
}

fn pas_rk3(u: &mut [Q], geo: &GeoA, dt: f64) {
    let z0 = champ_z(u, geo);
    maj_u(u, 0, dt, &z0, &z0, &z0);
    let z1 = champ_z(u, geo);
    maj_u(u, 1, dt, &z0, &z1, &z1);
    let z2 = champ_z(u, geo);
    maj_u(u, 2, dt, &z0, &z1, &z2);
}

/// Flot de Wilson RK3 sur `n_pas` pas (le champ source n'est pas modifié).
pub fn flot_n_pas(c: &ChampG, dt: f64, n_pas: usize) -> Vec<Q> {
    let mut u = c.u.clone();
    for _ in 0..n_pas {
        pas_rk3(&mut u, &c.geo, dt);
    }
    u
}

/// Polyakov diffusé sur les sites : 4·V valeurs (direction d puis site), constant le long de d.
pub fn polyakov_diffuse(geo: &GeoA, u: &[Q]) -> Vec<f64> {
    let mut out = vec![0.0; 4 * geo.v];
    for d in 0..4 {
        out[d * geo.v..(d + 1) * geo.v]
            .par_iter_mut()
            .enumerate()
            .for_each(|(s, o)| {
                let mut x = geo.coords(s);
                x[d] = 0;
                let mut t = geo.site(&x);
                let mut p = UN;
                for k in 0..geo.d[d] {
                    p = qmul(p, lien(geo, u, d, t));
                    if k != geo.d[d] - 1 {
                        t += geo.st[d];
                    }
                }
                *o = p[0];
            });
    }
    out
}

/// Tous les champs d'une configuration de côtés quelconques.
pub fn calculer(c: &ChampG, p: &ParamsChamps) -> Resultat<ChampsSite> {
    if !p.dt.is_finite() || p.dt <= 0.0 {
        return Err(Erreur::Config(format!("champs : dt = {} invalide (dt > 0)", p.dt)));
    }
    let geo = &c.geo;
    let v = geo.v;
    let brut: Vec<(f64, f64)> = (0..v)
        .into_par_iter()
        .map(|s| (action_site(geo, &c.u, s), negatives_site(geo, &c.u, s)))
        .collect();
    let polyakov = polyakov_diffuse(geo, &c.u);
    let uf = flot_n_pas(c, p.dt, p.n_pas);
    let flot: Vec<(f64, f64, f64, f64)> = (0..v)
        .into_par_iter()
        .map(|s| {
            let (e, q) = densite_trefle_site(geo, &uf, s);
            (action_site(geo, &uf, s), e, q, negatives_site(geo, &uf, s))
        })
        .collect();
    Ok(ChampsSite {
        l: geo.d[0],
        cotes: geo.d,
        action_brute: brut.iter().map(|t| t.0).collect(),
        action_flot: flot.iter().map(|t| t.0).collect(),
        energie_trefle_flot: flot.iter().map(|t| t.1).collect(),
        charge_flot: flot.iter().map(|t| t.2).collect(),
        polyakov,
        plaq_negatives_brute: brut.iter().map(|t| t.1).collect(),
        plaq_negatives_flot: flot.iter().map(|t| t.3).collect(),
    })
}

/// Transformation de jauge aléatoire U_μ(x) → g(x) U_μ(x) g(x+μ)† (g de Haar, compteur Philox de kvu).
pub fn jauge_aleatoire(c: &ChampG, graine: u32) -> ChampG {
    let geo = &c.geo;
    let g: Vec<Q> = (0..geo.v)
        .map(|s| crate::su2::cpu::haar_q(s as u32, 7, graine, 0x5ca9, 0x1111))
        .collect();
    let mut u = c.u.clone();
    for mu in 0..4 {
        for s in 0..geo.v {
            let x = geo.coords(s);
            let t = geo.nb_up(s, mu, x[mu]);
            u[mu * geo.v + s] = qmul(qmul(g[s], c.u[mu * geo.v + s]), qdag(g[t]));
        }
    }
    ChampG { geo: geo.clone(), u }
}

// ------------------------------------------------------------------------------------------
// Portes (références par des chemins indépendants du calcul des champs)
// ------------------------------------------------------------------------------------------

fn ecart_rel(a: f64, b: f64, echelle: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    (a - b).abs() / echelle.max(f64::MIN_POSITIVE)
}

fn porte(id: &str, champ: f64, reference: f64, echelle: f64, detail: &str) -> Porte {
    let e = ecart_rel(champ, reference, echelle);
    Porte {
        id: id.into(),
        ok: e.is_finite() && e <= SEUIL,
        champ,
        reference,
        ecart: e,
        detail: detail.into(),
    }
}

/// Portes du chemin non cubique. Références : plaquette moyenne par `tube::reseau::plaquette` (association propre au tube,
/// boucle indépendante), boucles de Polyakov par `tube::reseau::polyakov_lignes`, plaquettes négatives par le produit de chemin
/// (1,1), recalcul local de e(x) et ℓ_d(x) en 64 sites (chemins indépendants), et somme de q(x) refaite par un second
/// passage du trèfle sur le champ flotté (non indépendante : seule la permutation des axes la contrôle, voir les tests).
pub fn portes(c: &ChampG, ch: &ChampsSite, p: &ParamsChamps) -> Vec<Porte> {
    use super::champs::somme;
    use crate::su2::tube::reseau::{plaquette as plaq_tube, polyakov_lignes};
    let geo = &c.geo;
    let vf = geo.v as f64;
    let moy = |x: &[f64]| somme(x) / vf;
    let e_depuis = |u: &[Q]| -> f64 {
        let par_site: Vec<f64> = (0..geo.v)
            .into_par_iter()
            .map(|s| {
                let x = geo.coords(s);
                let mut a = 0.0;
                for mu in 0..4 {
                    for nu in mu + 1..4 {
                        a += plaq_tube(geo, u, s, &x, mu, nu)[0];
                    }
                }
                a
            })
            .collect();
        24.0 * (1.0 - somme(&par_site) / (6.0 * vf))
    };
    let mut out = Vec::new();
    let e_b = e_depuis(&c.u);
    out.push(porte(
        "CH-SOMME-ACTION-BRUTE",
        moy(&ch.action_brute),
        e_b,
        e_b.abs(),
        "moyenne = 24 (1 - <P>), plaquettes de tube::reseau::plaquette",
    ));
    let uf = flot_n_pas(c, p.dt, p.n_pas);
    let e_f = e_depuis(&uf);
    out.push(porte(
        "CH-SOMME-ACTION-FLOT",
        moy(&ch.action_flot),
        e_f,
        e_f.abs(),
        "moyenne = 24 (1 - <P>) du champ flotté (tube::reseau::plaquette)",
    ));
    let l1q = somme(&ch.charge_flot.iter().map(|x| x.abs()).collect::<Vec<_>>());
    let q_ref = somme(&(0..geo.v).map(|s| densite_trefle_site(geo, &uf, s).1).collect::<Vec<_>>());
    out.push(porte(
        "CH-SOMME-CHARGE-FLOT",
        somme(&ch.charge_flot),
        q_ref,
        l1q,
        "somme q(x) = second passage du trèfle (non indépendant) ; échelle = somme |q|",
    ));
    for d in 0..4 {
        let tranche = &ch.polyakov[d * geo.v..(d + 1) * geo.v];
        let ell = polyakov_lignes(geo, &c.u, d);
        let m_ref = somme(&ell) / ell.len() as f64;
        let l1 = somme(&tranche.iter().map(|x| x.abs()).collect::<Vec<_>>()) / vf;
        out.push(porte(
            &format!("CH-SOMME-POLYAKOV-{d}"),
            moy(tranche),
            m_ref,
            l1,
            "moyenne = moyenne de tube::reseau::polyakov_lignes ; échelle = moyenne |l_d|",
        ));
    }
    let neg_ref = |u: &[Q]| -> f64 {
        (0..geo.v)
            .into_par_iter()
            .map(|s| {
                PLANS
                    .iter()
                    .filter(|&&(mu, nu)| produit_chemin(geo, u, s, &[(mu, 1), (nu, 1), (mu, -1), (nu, -1)])[0] < 0.0)
                    .count()
            })
            .sum::<usize>() as f64
    };
    for (id, x, r) in [
        ("CH-SOMME-NEGATIVES-BRUTE", somme(&ch.plaq_negatives_brute), neg_ref(&c.u)),
        ("CH-SOMME-NEGATIVES-FLOT", somme(&ch.plaq_negatives_flot), neg_ref(&uf)),
    ] {
        out.push(Porte {
            id: id.into(),
            ok: x == r,
            champ: x,
            reference: r,
            ecart: ecart_rel(x, r, r.abs().max(1.0)),
            detail: "égalité exacte avec le compte par produit de chemin (1,1)".into(),
        });
    }
    // localité
    let mut pire: f64 = 0.0;
    for k in 0..64u64 {
        let s = (k.wrapping_mul(2_654_435_761).wrapping_add(12_345) % geo.v as u64) as usize;
        let e_ref: f64 = 4.0
            * PLANS
                .iter()
                .map(|&(mu, nu)| 1.0 - produit_chemin(geo, &c.u, s, &[(mu, 1), (nu, 1), (mu, -1), (nu, -1)])[0])
                .sum::<f64>();
        pire = pire.max((ch.action_brute[s] - e_ref).abs());
        for d in 0..4 {
            let chemin: Vec<(usize, i32)> = vec![(d, 1); geo.d[d]];
            let ell = produit_chemin(geo, &c.u, s, &chemin)[0];
            pire = pire.max((ch.polyakov[d * geo.v + s] - ell).abs());
        }
    }
    out.push(Porte {
        id: "CH-LOCALITE".into(),
        ok: pire.is_finite() && pire <= SEUIL,
        champ: pire,
        reference: 0.0,
        ecart: pire,
        detail: "64 sites : action_brute et polyakov recalculés par produit de chemin (écart absolu max)".into(),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commun::ChampHote;

    fn aleatoire(cotes: [usize; 4]) -> Vec<f64> {
        let v: usize = cotes.iter().product();
        let mut q = Vec::with_capacity(16 * v);
        for li in 0..4 * v {
            q.extend_from_slice(&crate::su2::cpu::haar_q(li as u32, 3, 17, 0x5ca9, 0x1111));
        }
        q
    }

    #[test]
    fn cotes_cubique_identique_au_chemin_cubique() {
        let l = 4;
        let q = aleatoire([l; 4]);
        let hote = ChampHote { l, q: q.clone() };
        let p = ParamsChamps { dt: 0.02, n_pas: 5 };
        let a = super::super::champs::calculer(&hote, &p).unwrap();
        let g = ChampG::nouveau(GeoA::nouvelle([l; 4]).unwrap(), &q).unwrap();
        let b = calculer(&g, &p).unwrap();
        for (x, y) in a.tableaux().iter().zip(b.tableaux().iter()) {
            assert_eq!(x.len(), y.len());
            assert!(x.iter().zip(y.iter()).all(|(u, v)| u.to_bits() == v.to_bits()), "écart au bit");
        }
    }

    #[test]
    fn portes_vertes_boite_non_cubique() {
        let cotes = [2, 4, 6, 4];
        let q = aleatoire(cotes);
        let g = ChampG::nouveau(GeoA::nouvelle(cotes).unwrap(), &q).unwrap();
        let p = ParamsChamps { dt: 0.02, n_pas: 5 };
        let c = calculer(&g, &p).unwrap();
        assert_eq!(c.forme("polyakov"), vec![4, 2, 4, 6, 4]);
        for x in portes(&g, &c, &p) {
            assert!(x.ok, "{}", x.ligne());
        }
    }
}
