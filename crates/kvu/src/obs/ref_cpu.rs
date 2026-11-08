//! Référence CPU quaternionique (rayon) des observables : boucles de Wilson R×T, corrélateur de
//! Polyakov, flot de Wilson RK3, densités d'énergie (plaquette, trèfle) et charge topologique.
//! Propriétaire : T7.
//!
//! Ce module est la VÉRITÉ des gates OBS-* : les noyaux GPU (`noyaux/obs_wilson.cu`, `obs_flot.cu`)
//! sont jugés contre lui. Il est volontairement écrit « à la main » (chemins explicites, listes de
//! pas), sans partager d'algorithme avec les noyaux, pour que l'accord soit une vraie preuve :
//! - GPU : produits progressifs partagés (récurrences en t et en r), formules du trèfle explicites ;
//! - CPU : produit direct le long du chemin (comme `observables::wilson::wilson_loop`), trèfle par
//!   une liste générique de pas.
//!
//! Conventions (celles de `su2_hb`, de su2_hip.cpp et du noyau GPU) :
//! - lien li = mu·V + s, s = ((x0·L + x1)·L + x2)·L + x3, U = a0 + i(a1 σ1 + a2 σ2 + a3 σ3) ;
//! - ½ Tr U = a0 ; ½ Tr(A B†) = A·B (produit scalaire des 4 composantes) ;
//! - flot : Z = −Ta(U · agrafe) = −(0, m.y, m.z, m.w) avec m = U·A (partie antihermitienne sans trace),
//!   exponentielle exacte exp(0, w) = (cos|w|, sin|w|/|w| · w), puis réunitarisation ;
//! - E_plaq = (6·d/T_R)(1 − ⟨P⟩) = 24 (1 − ⟨P⟩) pour SU(2) (d = 2, T_R = ½), normalisation de
//!   `observables::wilson_flow::action_density` (correctif du 19/06) ;
//! - trèfle : G_μν = (Q_μν − Q_μν†)/8, Q_μν = somme des 4 plaquettes du plan autour de x ;
//!   E_trèfle = −(1/(4 T_R)) Σ_{μ,ν} tr G_μν² (moyenne sur les sites) ;
//!   Q_top = −(1/(32π²)) Σ_x ε_{μνρσ} tr(G_μν G_ρσ) (charge TOTALE, G antihermitien).

use super::flot::{echelles, CourbeFlot, ParamsFlot};
use super::wilson::{ParamsWilson, TableWilson};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{k_final, k_reduce, qadd, qdag, qmul, qnorm, qscale, Geo, Q};
use rayon::prelude::*;

/// Quaternion identité.
pub const UN: Q = [1.0, 0.0, 0.0, 0.0];

/// Les six plans (μ < ν), dans l'ordre des noyaux.
pub const PLANS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Plan (μ,ν) → plan complémentaire dans l'ordre de `PLANS` (01↔23, 02↔13, 03↔12) : sert à ε_{μνρσ}.
const COMPLEMENT: [usize; 6] = [5, 4, 3, 2, 1, 0];
/// Signe ε_{μνρσ} du couple (plan, complément) : +, −, +, +, −, +.
const SIGNE_EPS: [f64; 6] = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0];

/// Coefficient 6·d/T_R de l'énergie de plaquette pour SU(2) (d = 2, T_R = ½).
pub const COEF_E_PLAQ: f64 = 24.0;

/// Vue en lecture seule d'un champ hôte : géométrie et accès aux liens.
#[derive(Clone, Copy)]
pub struct Reseau<'a> {
    pub geo: Geo,
    q: &'a [f64],
}

impl<'a> Reseau<'a> {
    pub fn nouveau(champ: &'a ChampHote) -> Resultat<Reseau<'a>> {
        if champ.l < 2 || champ.q.len() != 16 * champ.v() {
            return Err(Erreur::Config(format!(
                "champ hôte incohérent : L = {}, {} réels (attendu {})",
                champ.l,
                champ.q.len(),
                16 * champ.v()
            )));
        }
        Ok(Reseau {
            geo: Geo::new(champ.l),
            q: &champ.q,
        })
    }
    /// Lien U_d(s).
    #[inline]
    pub fn u(&self, d: usize, s: usize) -> Q {
        let o = 4 * (d * self.geo.v + s);
        [self.q[o], self.q[o + 1], self.q[o + 2], self.q[o + 3]]
    }
}

/// Site voisin dans la direction `d`, de `pas` = ±1 (périodique).
#[inline]
fn voisin(geo: &Geo, s: usize, d: usize, pas: i32) -> usize {
    let xd = geo.coords(s)[d];
    if pas > 0 {
        geo.nb_up(s, d, xd)
    } else {
        geo.nb_dn(s, d, xd)
    }
}

/// Produit ordonné le long d'un chemin de pas (direction, sens) depuis le site `s`.
/// Pas (d, +1) : multiplie U_d(position) puis avance ; pas (d, −1) : recule puis multiplie U_d(position)†.
pub fn produit_chemin(res: &Reseau, s: usize, pas: &[(usize, i32)]) -> Q {
    let mut p = UN;
    let mut pos = s;
    for &(d, sens) in pas {
        if sens > 0 {
            p = qmul(p, res.u(d, pos));
            pos = voisin(&res.geo, pos, d, 1);
        } else {
            pos = voisin(&res.geo, pos, d, -1);
            p = qmul(p, qdag(res.u(d, pos)));
        }
    }
    p
}

/// Boucle de Wilson R×T du site `s` : R pas selon μ, T pas selon ν, puis retour (chemin de
/// `observables::wilson::wilson_loop`). Rend le quaternion de la boucle.
pub fn boucle(res: &Reseau, s: usize, mu: usize, nu: usize, r: usize, t: usize) -> Q {
    let mut p = UN;
    let mut pos = s;
    for _ in 0..r {
        p = qmul(p, res.u(mu, pos));
        pos = voisin(&res.geo, pos, mu, 1);
    }
    for _ in 0..t {
        p = qmul(p, res.u(nu, pos));
        pos = voisin(&res.geo, pos, nu, 1);
    }
    for _ in 0..r {
        pos = voisin(&res.geo, pos, mu, -1);
        p = qmul(p, qdag(res.u(mu, pos)));
    }
    for _ in 0..t {
        pos = voisin(&res.geo, pos, nu, -1);
        p = qmul(p, qdag(res.u(nu, pos)));
    }
    p
}

/// Orientations (μ, ν) mesurées : les 12 couples ordonnés μ ≠ ν, ou les 3 couples (μ, dir_temps).
pub fn orientations(p: &ParamsWilson) -> Vec<(usize, usize)> {
    if p.toutes_orientations {
        (0..4)
            .flat_map(|mu| (0..4).filter(move |&nu| nu != mu).map(move |nu| (mu, nu)))
            .collect()
    } else {
        (0..4)
            .filter(|&mu| mu != p.dir_temps)
            .map(|mu| (mu, p.dir_temps))
            .collect()
    }
}

/// Vérifie les paramètres de Wilson (partagé avec le chemin GPU).
pub fn verifier_params_wilson(p: &ParamsWilson) -> Resultat<()> {
    if p.r_max == 0 || p.t_max == 0 {
        return Err(Erreur::Config(format!(
            "boucles de Wilson : r_max = {} et t_max = {} doivent être ≥ 1",
            p.r_max, p.t_max
        )));
    }
    if p.dir_temps > 3 {
        return Err(Erreur::Config(format!("dir_temps = {} hors de 0..3", p.dir_temps)));
    }
    Ok(())
}

/// Moyenne sur les sites de ½ Tr W(R,T) dans l'orientation (μ, ν) : somme à arbre fixe.
pub fn wilson_orientation(champ: &ChampHote, mu: usize, nu: usize, r: usize, t: usize) -> Resultat<f64> {
    let res = Reseau::nouveau(champ)?;
    if mu > 3 || nu > 3 || mu == nu {
        return Err(Erreur::Config(format!("orientation (mu, nu) = ({mu}, {nu}) invalide")));
    }
    Ok(moyenne_orientation(&res, mu, nu, r, t))
}

fn moyenne_orientation(res: &Reseau, mu: usize, nu: usize, r: usize, t: usize) -> f64 {
    let v = res.geo.v;
    let par_site: Vec<f64> = (0..v)
        .into_par_iter()
        .map(|s| boucle(res, s, mu, nu, r, t)[0])
        .collect();
    k_final(&k_reduce(&par_site), 1.0 / v as f64)
}

/// W[(r−1)·t_max + (t−1)] = ⟨½ Tr W(r,t)⟩, moyenne arithmétique des moyennes par orientation.
pub fn table_wilson(champ: &ChampHote, p: &ParamsWilson) -> Resultat<TableWilson> {
    verifier_params_wilson(p)?;
    let res = Reseau::nouveau(champ)?;
    let ors = orientations(p);
    let mut w = vec![0.0; p.r_max * p.t_max];
    for &(mu, nu) in &ors {
        for r in 1..=p.r_max {
            for t in 1..=p.t_max {
                w[(r - 1) * p.t_max + (t - 1)] += moyenne_orientation(&res, mu, nu, r, t);
            }
        }
    }
    let n = ors.len() as f64;
    for x in &mut w {
        *x /= n;
    }
    Ok(TableWilson {
        r_max: p.r_max,
        t_max: p.t_max,
        w,
    })
}

// ------------------------------------------------------------------------------------------
// Polyakov
// ------------------------------------------------------------------------------------------

/// Les trois directions transverses à `dir`, par ordre croissant (ordre de k_polyakov).
fn transverses(dir: usize) -> [usize; 3] {
    let mut d = [0usize; 3];
    let mut k = 0;
    for x in 0..4 {
        if x != dir {
            d[k] = x;
            k += 1;
        }
    }
    d
}

/// ℓ(x⃗) = ½ Tr ∏_k U_dir(x⃗, k) pour chaque site transverse j (ordre de k_polyakov :
/// x[dirs[2]] = j mod L, x[dirs[1]] = (j/L) mod L, x[dirs[0]] = j/L²).
pub fn polyakov_sites(champ: &ChampHote, dir: usize) -> Resultat<Vec<f64>> {
    let res = Reseau::nouveau(champ)?;
    if dir > 3 {
        return Err(Erreur::Config(format!("direction de Polyakov {dir} hors de 0..3")));
    }
    let g = res.geo;
    let l = g.l;
    let l3 = g.v / l;
    let dirs = transverses(dir);
    Ok((0..l3)
        .into_par_iter()
        .map(|j| {
            let mut x = [0usize; 4];
            x[dirs[2]] = j % l;
            x[dirs[1]] = (j / l) % l;
            x[dirs[0]] = j / (l * l);
            let mut s = ((x[0] * l + x[1]) * l + x[2]) * l + x[3];
            let mut p = UN;
            for k in 0..l {
                p = qmul(p, res.u(dir, s));
                if k != l - 1 {
                    s += g.st[dir];
                }
            }
            p[0]
        })
        .collect())
}

/// ⟨ℓ⟩ dans chacune des 4 directions (P0..P3 de su2_hip), arbre fixe de k_final.
pub fn polyakov_moyen(champ: &ChampHote) -> Resultat<[f64; 4]> {
    let mut out = [0.0; 4];
    for (d, o) in out.iter_mut().enumerate() {
        let ell = polyakov_sites(champ, d)?;
        *o = k_final(&ell, 1.0 / ell.len() as f64);
    }
    Ok(out)
}

/// C(r) = ⟨ℓ(x⃗) ℓ(x⃗ + r ê_i)⟩ moyenné sur les sites et les 3 axes transverses, r = 0..L/2.
/// (ℓ réel pour SU(2) : ℓ ℓ* = ℓ ℓ.)
pub fn correlateur_polyakov(champ: &ChampHote, dir_temps: usize) -> Resultat<Vec<f64>> {
    let ell = polyakov_sites(champ, dir_temps)?;
    let l = champ.l;
    let l3 = ell.len();
    let decal = |j: usize, axe: usize, r: usize| -> usize {
        let mut c = [j / (l * l), (j / l) % l, j % l];
        c[axe] = (c[axe] + r) % l;
        (c[0] * l + c[1]) * l + c[2]
    };
    Ok((0..=l / 2)
        .map(|r| {
            let par_site: Vec<f64> = (0..l3)
                .into_par_iter()
                .map(|j| ell[j] * ell[decal(j, 0, r)] + ell[j] * ell[decal(j, 1, r)] + ell[j] * ell[decal(j, 2, r)])
                .collect();
            k_final(&k_reduce(&par_site), 1.0 / (3.0 * l3 as f64))
        })
        .collect())
}

// ------------------------------------------------------------------------------------------
// Flot de Wilson RK3 (Lüscher 2010)
// ------------------------------------------------------------------------------------------

/// exp(0, w) = (cos|w|, sin|w|/|w| · w) : exponentielle exacte de su(2) en quaternions.
#[inline]
pub fn exp_su2(w: [f64; 3]) -> Q {
    let th = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    let sinc = if th < 1e-8 { 1.0 - th * th / 6.0 } else { th.sin() / th };
    [th.cos(), sinc * w[0], sinc * w[1], sinc * w[2]]
}

/// Z(x, μ) = −Ta(U_μ(x) · A_μ(x)) pour chaque lien (3 composantes de l'algèbre, li = μ·V + s).
pub fn champ_z(u: &[Q], geo: &Geo) -> Vec<[f64; 3]> {
    (0..4 * geo.v)
        .into_par_iter()
        .map(|li| {
            let (mu, s) = (li / geo.v, li % geo.v);
            let x = geo.coords(s);
            let a = crate::heritage::su2_hb::staple(u, geo, mu, s, &x);
            let m = qmul(u[li], a);
            [-m[1], -m[2], -m[3]]
        })
        .collect()
}

/// Mise à jour V ← exp(ω_étape) V avec les combinaisons de Lüscher, mêmes expressions que le noyau :
/// - étape 0 : ω = ¼ dt Z0 ;
/// - étape 1 : ω = dt (8/9 Z1 − 17/36 Z0) ;
/// - étape 2 : ω = dt (¾ Z2 + 17/36 Z0 − 8/9 Z1).
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

/// Un pas du flot RK3 sur un champ de quaternions (modifié sur place).
pub fn pas_rk3(u: &mut [Q], geo: &Geo, dt: f64) {
    let z0 = champ_z(u, geo);
    maj_u(u, 0, dt, &z0, &z0, &z0);
    let z1 = champ_z(u, geo);
    maj_u(u, 1, dt, &z0, &z1, &z1);
    let z2 = champ_z(u, geo);
    maj_u(u, 2, dt, &z0, &z1, &z2);
}

/// Champ hôte → quaternions.
pub fn en_quaternions(champ: &ChampHote) -> Vec<Q> {
    champ.q.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}

/// Quaternions → champ hôte.
pub fn en_champ(u: &[Q], l: usize) -> ChampHote {
    ChampHote {
        l,
        q: u.iter().flat_map(|q| q.iter().copied()).collect(),
    }
}

/// Flot de `n_pas` pas de taille `dt` sur une COPIE du champ.
pub fn flot_n_pas(champ: &ChampHote, dt: f64, n_pas: usize) -> Resultat<ChampHote> {
    Reseau::nouveau(champ)?;
    let geo = Geo::new(champ.l);
    let mut u = en_quaternions(champ);
    for _ in 0..n_pas {
        pas_rk3(&mut u, &geo, dt);
    }
    Ok(en_champ(&u, champ.l))
}

// ------------------------------------------------------------------------------------------
// Densités : plaquette, trèfle, charge topologique
// ------------------------------------------------------------------------------------------

/// Observables de densité d'un champ : E plaquette, E trèfle, charge topologique TOTALE.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Densites {
    pub e_plaq: f64,
    pub e_trefle: f64,
    pub q_top: f64,
}

/// 1/(2π²) : coefficient de la densité topologique q = (1/2π²)(g01·g23 − g02·g13 + g03·g12).
pub const INV_2PI2: f64 = 1.0 / (2.0 * std::f64::consts::PI * std::f64::consts::PI);

/// Les quatre feuilles du trèfle d'un plan (μ,ν) autour du site `s`, sous forme de listes de pas.
fn feuilles(mu: usize, nu: usize) -> [[(usize, i32); 4]; 4] {
    [
        [(mu, 1), (nu, 1), (mu, -1), (nu, -1)],
        [(nu, 1), (mu, -1), (nu, -1), (mu, 1)],
        [(mu, -1), (nu, -1), (mu, 1), (nu, 1)],
        [(nu, -1), (mu, 1), (nu, 1), (mu, -1)],
    ]
}

/// Les 6 vecteurs g_μν = (Q_μν − Q_μν†)/8 (partie vectorielle de i·σ) au site `s`.
pub fn champ_trefle(res: &Reseau, s: usize) -> [[f64; 3]; 6] {
    let mut g = [[0.0; 3]; 6];
    for (p, &(mu, nu)) in PLANS.iter().enumerate() {
        let mut somme: Q = [0.0; 4];
        for chemin in feuilles(mu, nu) {
            somme = qadd(somme, produit_chemin(res, s, &chemin));
        }
        // (Q − Q†)/2 = (0, vecteur) ; G = (Q − Q†)/8 ⟹ g = vecteur / 4
        g[p] = [0.25 * somme[1], 0.25 * somme[2], 0.25 * somme[3]];
    }
    g
}

/// Plaquette sommée sur les 6 plans au site `s` (même expression que k_plaq_site).
pub fn plaquette_site(res: &Reseau, s: usize) -> f64 {
    let g = res.geo;
    let x = g.coords(s);
    let mut acc = 0.0;
    for mu in 0..4 {
        let smu = g.nb_up(s, mu, x[mu]);
        for (nu, &xnu) in x.iter().enumerate().skip(mu + 1) {
            let snu = g.nb_up(s, nu, xnu);
            let p = qmul(
                qmul(res.u(mu, s), res.u(nu, smu)),
                qmul(qdag(res.u(mu, snu)), qdag(res.u(nu, s))),
            );
            acc += p[0];
        }
    }
    acc
}

/// (E_trèfle(x), q(x)) au site `s`.
pub fn densite_trefle_site(res: &Reseau, s: usize) -> (f64, f64) {
    let g = champ_trefle(res, s);
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

/// Densités d'un champ (sommes à arbre fixe).
pub fn densites(champ: &ChampHote) -> Resultat<Densites> {
    let res = Reseau::nouveau(champ)?;
    let v = res.geo.v;
    let par_site: Vec<(f64, f64, f64)> = (0..v)
        .into_par_iter()
        .map(|s| {
            let (e, q) = densite_trefle_site(&res, s);
            (plaquette_site(&res, s), e, q)
        })
        .collect();
    let somme = |f: &dyn Fn(&(f64, f64, f64)) -> f64| -> f64 {
        let x: Vec<f64> = par_site.iter().map(f).collect();
        k_final(&k_reduce(&x), 1.0)
    };
    let sp = somme(&|t| t.0);
    let se = somme(&|t| t.1);
    let sq = somme(&|t| t.2);
    Ok(Densites {
        e_plaq: COEF_E_PLAQ * (1.0 - sp / (6.0 * v as f64)),
        e_trefle: se / v as f64,
        q_top: sq,
    })
}

/// Nombre de pas d'un flot (t_max / dt, tolérant à l'arrondi) et validation des paramètres.
pub fn nombre_de_pas(p: &ParamsFlot) -> Resultat<usize> {
    if !p.dt.is_finite() || p.dt <= 0.0 || !p.t_max.is_finite() || p.t_max <= 0.0 || p.mesure_chaque == 0 {
        return Err(Erreur::Config(format!(
            "flot : dt = {}, t_max = {}, mesure_chaque = {} invalides (dt > 0, t_max > 0, mesure_chaque ≥ 1)",
            p.dt, p.t_max, p.mesure_chaque
        )));
    }
    let n = (p.t_max / p.dt + 1e-9).floor();
    if n < 1.0 {
        return Err(Erreur::Config(format!("flot : t_max = {} < dt = {}", p.t_max, p.dt)));
    }
    Ok(n as usize)
}

/// Les indices de pas où l'on mesure : 0, m, 2m, … et toujours le dernier.
pub fn pas_mesures(n_pas: usize, chaque: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..=n_pas).step_by(chaque).collect();
    if *v.last().unwrap_or(&0) != n_pas {
        v.push(n_pas);
    }
    v
}

/// Remplit la courbe à partir des densités mesurées (t², t0, w0).
pub fn assembler_courbe(p: &ParamsFlot, pas: &[usize], dens: &[Densites]) -> CourbeFlot {
    let mut c = CourbeFlot::default();
    for (&k, d) in pas.iter().zip(dens) {
        let t = k as f64 * p.dt;
        c.t.push(t);
        c.e_plaq.push(d.e_plaq);
        c.e_trefle.push(d.e_trefle);
        c.t2e.push(t * t * d.e_plaq);
        c.q_top.push(d.q_top);
    }
    let (t0, w0) = echelles(&c.t, &c.t2e, p.cible_t2e, p.cible_w);
    c.t0 = t0;
    c.w0 = w0;
    c
}

/// Flot complet sur le CPU (copie du champ), avec mesures.
pub fn flot(champ: &ChampHote, p: &ParamsFlot) -> Resultat<CourbeFlot> {
    let n_pas = nombre_de_pas(p)?;
    Reseau::nouveau(champ)?;
    let geo = Geo::new(champ.l);
    let mut u = en_quaternions(champ);
    let mesures = pas_mesures(n_pas, p.mesure_chaque);
    let mut dens = Vec::with_capacity(mesures.len());
    let mut fait = 0usize;
    for &m in &mesures {
        while fait < m {
            pas_rk3(&mut u, &geo, p.dt);
            fait += 1;
        }
        dens.push(densites(&en_champ(&u, champ.l))?);
    }
    Ok(assembler_courbe(p, &mesures, &dens))
}
