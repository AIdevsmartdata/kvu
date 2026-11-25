//! Portes du module intrication (a)-(f) et balayages (interpolation en α, intégration thermodynamique en β).
//! Chaque porte rend des NOMBRES ; les seuils sont appliqués par les tests (`tests/intrication_portes.rs`) et par
//! `kvu-intrication portes`.

use super::*;
use crate::heritage::su2_hb::CpuSu2;

fn bits(q: &Q) -> [u64; 4] {
    [
        q[0].to_bits(),
        q[1].to_bits(),
        q[2].to_bits(),
        q[3].to_bits(),
    ]
}

/// Quaternion de Haar de compteur (i, étiquette) : sert aux tirages des portes (jauge, liens d'essai).
pub fn haar_porte(i: u32, etiquette: u32, graine: u64) -> Q {
    let (k0, k1) = ((graine & 0xFFFF_FFFF) as u32, (graine >> 32) as u32);
    let r1 = su2_hb::philox4x32_10([i, etiquette, 0, 0x504F_5254], k0, k1);
    let r2 = su2_hb::philox4x32_10([i, etiquette, 1, 0x504F_5254], k0, k1);
    haar_from(r1, r2)
}

// ------------------------------------------------------------------------------------------
// (a) A vide : deux réseaux ordinaires, bit à bit contre heritage::su2_hb::CpuSu2
// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
pub struct ResA {
    pub l: usize,
    pub pas: usize,
    pub liens_differents: usize,
    pub plaq_replique: [f64; 2],
    pub plaq_cpusu2: [f64; 2],
}

pub fn porte_a(
    l: usize,
    beta: f64,
    pas: usize,
    n_or: usize,
    graines: [u64; 2],
) -> Result<ResA, String> {
    let g = GeoRep::nouveau(l, l, 2)?;
    let vide = Region::tranche(&g, 0);
    let mut res = Reseau::pur(g, vide, &graines, Mutant::Aucun)?;
    res.beta = beta;
    res.init(true);
    let mut refs: Vec<CpuSu2> = graines
        .iter()
        .map(|&s| CpuSu2::new(l, s))
        .collect::<Result<_, _>>()?;
    for c in refs.iter_mut() {
        c.init(true);
    }
    for _ in 0..pas {
        res.pas(n_or);
        for c in refs.iter_mut() {
            c.step(beta, n_or);
        }
    }
    let mut diff = 0;
    for (r, c) in refs.iter().enumerate() {
        let mien = res.champ_replique(r);
        diff += mien
            .iter()
            .zip(&c.u)
            .filter(|(a, b)| bits(a) != bits(b))
            .count();
    }
    Ok(ResA {
        l,
        pas,
        liens_differents: diff,
        plaq_replique: [res.plaquette_replique(0), res.plaquette_replique(1)],
        plaq_cpusu2: [refs[0].measure()[0], refs[1].measure()[0]],
    })
}

// ------------------------------------------------------------------------------------------
// (b) A = toute la tranche : réseau ordinaire L³ × 2T, staples et plaquettes bit à bit
// ------------------------------------------------------------------------------------------

/// Réseau ordinaire périodique Lt × L³ (implémentation indépendante, voisins par modulo).
struct Ordinaire {
    lt: usize,
    l: usize,
    v: usize,
}

impl Ordinaire {
    fn idx(&self, x: [usize; 4]) -> usize {
        ((x[0] * self.l + x[1]) * self.l + x[2]) * self.l + x[3]
    }
    fn coords(&self, s: usize) -> [usize; 4] {
        let l = self.l;
        [s / (l * l * l), (s / (l * l)) % l, (s / l) % l, s % l]
    }
    fn haut(&self, s: usize, d: usize) -> usize {
        let mut x = self.coords(s);
        let n = if d == 0 { self.lt } else { self.l };
        x[d] = (x[d] + 1) % n;
        self.idx(x)
    }
    fn bas(&self, s: usize, d: usize) -> usize {
        let mut x = self.coords(s);
        let n = if d == 0 { self.lt } else { self.l };
        x[d] = (x[d] + n - 1) % n;
        self.idx(x)
    }
    /// Agrafe de su2_hb::staple, transcrite avec des voisins par modulo.
    fn agrafe(&self, u: &[Q], mu: usize, s: usize) -> Q {
        use su2_hb::{qadd, qdag, qmul};
        let ld = |nu: usize, site: usize| u[nu * self.v + site];
        let mut a: Q = [0.0; 4];
        let smu = self.haut(s, mu);
        for nu in 0..4 {
            if nu == mu {
                continue;
            }
            let snu = self.haut(s, nu);
            let sdn = self.bas(s, nu);
            let smu_dn = self.bas(smu, nu);
            let t = qmul(qmul(ld(nu, smu), qdag(ld(mu, snu))), qdag(ld(nu, s)));
            let b = qmul(qmul(qdag(ld(nu, smu_dn)), qdag(ld(mu, sdn))), ld(nu, sdn));
            a = qadd(a, qadd(t, b));
        }
        a
    }
    fn plaquettes(&self, u: &[Q]) -> Vec<f64> {
        use su2_hb::{qdag, qmul};
        let mut out = Vec::with_capacity(6 * self.v);
        for s in 0..self.v {
            for mu in 0..4 {
                for nu in mu + 1..4 {
                    let w = qmul(
                        qmul(
                            qmul(u[mu * self.v + s], u[nu * self.v + self.haut(s, mu)]),
                            qdag(u[mu * self.v + self.haut(s, nu)]),
                        ),
                        qdag(u[nu * self.v + s]),
                    );
                    out.push(w[0]);
                }
            }
        }
        out
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ResB {
    pub l: usize,
    pub t: usize,
    pub mutant: String,
    pub agrafes_differentes: usize,
    pub plaquettes_differentes: usize,
    pub poids_non_unitaires: usize,
    pub action_recollee: f64,
    pub action_2t: f64,
    pub ecart_action: f64,
}

pub fn porte_b(l: usize, t: usize, graine: u64, mutant: Mutant) -> Result<ResB, String> {
    let g = GeoRep::nouveau(l, t, 2)?;
    let plein = Region::tranche(&g, l);
    let mut res = Reseau::pur(g, plein, &[graine, graine ^ 0x5DEE_CE66_D1CE_4E5B], mutant)?;
    res.beta = 1.0;
    res.init(true);
    let o = Ordinaire {
        lt: 2 * t,
        l,
        v: 2 * g.v1,
    };
    // copie par COORDONNÉES : (r, t, xs) ↦ τ = r·T + t
    let mut u2 = vec![[0.0; 4]; 4 * o.v];
    for mu in 0..4 {
        for s in 0..g.v {
            let (r, tt, xs) = g.decomp(s);
            let c = g.xs_coords(xs);
            let s2 = o.idx([r * t + tt, c[0], c[1], c[2]]);
            u2[mu * o.v + s2] = res.u[g.lien(mu, s)];
        }
    }
    let mut agr_diff = 0;
    for mu in 0..4 {
        for s in 0..g.v {
            let (r, tt, xs) = g.decomp(s);
            let c = g.xs_coords(xs);
            let s2 = o.idx([r * t + tt, c[0], c[1], c[2]]);
            if bits(&res.agrafe(g.lien(mu, s))) != bits(&o.agrafe(&u2, mu, s2)) {
                agr_diff += 1;
            }
        }
    }
    let mut p_rep: Vec<u64> = Vec::new();
    let mut poids_non_unitaires = 0;
    for i in 0..res.plaqs.len() {
        if res.coef[i] != 1.0 {
            poids_non_unitaires += 1;
        }
        p_rep.push(res.valeur(i).to_bits());
    }
    let mut p_ord: Vec<u64> = o.plaquettes(&u2).iter().map(|v| v.to_bits()).collect();
    p_rep.sort_unstable();
    p_ord.sort_unstable();
    let plaq_diff = if p_rep.len() != p_ord.len() {
        p_rep.len().abs_diff(p_ord.len()) + p_rep.len().min(p_ord.len())
    } else {
        p_rep.iter().zip(&p_ord).filter(|(a, b)| a != b).count()
    };
    let somme = |v: &[u64]| v.iter().map(|b| 1.0 - f64::from_bits(*b)).sum::<f64>();
    let (sa, sb) = (somme(&p_rep), somme(&p_ord));
    Ok(ResB {
        l,
        t,
        mutant: format!("{mutant:?}"),
        agrafes_differentes: agr_diff,
        plaquettes_differentes: plaq_diff,
        poids_non_unitaires,
        action_recollee: sa,
        action_2t: sb,
        ecart_action: sa - sb,
    })
}

// ------------------------------------------------------------------------------------------
// (c) invariance de jauge sur la géométrie recollée
// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
pub struct ResC {
    pub l: usize,
    pub t: usize,
    pub largeur: usize,
    pub mutant: String,
    pub contrainte_branchement: bool,
    pub sommets_branchement: usize,
    pub ecart_max_plaquette: f64,
    pub ecart_action_relatif: f64,
}

/// Extrémité du lien (mu, s) pour la jauge : à la coupure, site de la réplique σ(r) si x est un site « tout A ».
fn extremite(g: &GeoRep, types: &[u8], mu: usize, s: usize) -> usize {
    let (r, t, xs) = g.decomp(s);
    if mu == 0 {
        if t + 1 < g.t {
            g.site(r, t + 1, xs)
        } else if types[xs] == 1 {
            g.site((r + 1) % g.s, 0, xs)
        } else {
            g.site(r, 0, xs)
        }
    } else {
        g.site(r, t, g.xs_haut(xs, mu))
    }
}

pub fn types_sites(g: &GeoRep, reg: &Region) -> Vec<u8> {
    // voisin entrant par décalage inverse
    let bas = |xs: usize, d: usize| {
        let mut c = g.xs_coords(xs);
        c[d - 1] = (c[d - 1] + g.l - 1) % g.l;
        (c[0] * g.l + c[1]) * g.l + c[2]
    };
    (0..g.l3)
        .map(|xs| {
            let (mut na, mut nb) = (0, 0);
            for d in 1..=3 {
                for f in [reg.a(xs, d), reg.a(bas(xs, d), d)] {
                    if f {
                        na += 1
                    } else {
                        nb += 1
                    }
                }
            }
            match (na, nb) {
                (_, 0) => 1,
                (0, _) => 0,
                _ => 2,
            }
        })
        .collect()
}

pub fn porte_c(
    l: usize,
    t: usize,
    largeur: usize,
    graine: u64,
    mutant: Mutant,
    contrainte: bool,
) -> Result<ResC, String> {
    let g = GeoRep::nouveau(l, t, 2)?;
    let reg = Region::tranche(&g, largeur);
    let types = types_sites(&g, &reg);
    let mut res = Reseau::pur(g, reg, &[graine, graine.rotate_left(17)], mutant)?;
    res.beta = 1.0;
    res.init(true);
    let avant: Vec<f64> = (0..res.plaqs.len()).map(|i| res.valeur(i)).collect();
    let s_avant = res.action();
    // transformation de jauge aléatoire ; aux sommets de branchement de la couture, même g dans toutes les répliques
    let gj: Vec<Q> = (0..g.v)
        .map(|s| {
            let (_, tt, xs) = g.decomp(s);
            let cle_site = if contrainte && tt == 0 && types[xs] == 2 {
                g.site(0, 0, xs)
            } else {
                s
            };
            haar_porte(cle_site as u32, 7, graine)
        })
        .collect();
    let u0 = res.u.clone();
    for mu in 0..4 {
        for s in 0..g.v {
            let li = g.lien(mu, s);
            let e = extremite(&g, &types, mu, s);
            res.u[li] = su2_hb::qmul(su2_hb::qmul(gj[s], u0[li]), su2_hb::qdag(gj[e]));
        }
    }
    let ecart = (0..res.plaqs.len())
        .map(|i| (res.valeur(i) - avant[i]).abs())
        .fold(0.0, f64::max);
    let s_apres = res.action();
    Ok(ResC {
        l,
        t,
        largeur,
        mutant: format!("{mutant:?}"),
        contrainte_branchement: contrainte,
        sommets_branchement: types.iter().filter(|&&x| x == 2).count(),
        ecart_max_plaquette: ecart,
        ecart_action_relatif: ((s_apres - s_avant) / s_avant).abs(),
    })
}

// ------------------------------------------------------------------------------------------
// (d) balance détaillée : (d1) agrafe ⟺ action, (d2) classes indépendantes, (d3) stationnarité
// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
pub struct ResD1 {
    pub liens_testes: usize,
    pub ecart_max: f64,
    pub mutant: String,
}

/// Pour chaque lien : remplace U par X (Haar) et compare S_α(X) − S_α(U) (somme directe sur les plaquettes qui
/// contiennent le lien, trouvées en parcourant la liste) à −β ((X − U)·A)[0] (agrafe du générateur).
pub fn porte_d1(
    l: usize,
    t: usize,
    largeur: usize,
    alpha: f64,
    graine: u64,
    mutant: Mutant,
) -> Result<ResD1, String> {
    let g = GeoRep::nouveau(l, t, 2)?;
    let mut res = Reseau::nouveau(
        g,
        Region::tranche(&g, largeur),
        Region::tranche(&g, largeur + 1),
        &[graine, graine + 1],
        mutant,
    )?;
    res.beta = 1.7;
    res.fixer_alpha(alpha);
    res.init(true);
    let mut contient: Vec<Vec<u32>> = vec![Vec::new(); 4 * g.v];
    for (i, p) in res.plaqs.iter().enumerate() {
        for e in p.e {
            contient[(e >> 1) as usize].push(i as u32);
        }
    }
    let mut ecart: f64 = 0.0;
    for (li, cont) in contient.iter().enumerate() {
        let x = haar_porte(li as u32, 11, graine);
        let a = res.agrafe_gen(li);
        let u = res.u[li];
        let loc = |r: &Reseau| {
            cont.iter()
                .map(|&i| r.coef[i as usize] * (1.0 - r.valeur(i as usize)))
                .sum::<f64>()
        };
        let s0 = loc(&res);
        res.u[li] = x;
        let s1 = loc(&res);
        res.u[li] = u;
        let direct = res.beta * (s1 - s0);
        let par_agrafe = -res.beta * (su2_hb::qmul(x, a)[0] - su2_hb::qmul(u, a)[0]);
        ecart = ecart.max((direct - par_agrafe).abs());
    }
    Ok(ResD1 {
        liens_testes: 4 * g.v,
        ecart_max: ecart,
        mutant: format!("{mutant:?}"),
    })
}

/// (d2) Nombre de plaquettes contenant deux liens d'une même classe de mise à jour simultanée.
pub fn porte_classes(res: &Reseau) -> usize {
    let mut classe = vec![0u8; 4 * res.geo.v];
    for (c, cl) in res.classes().iter().enumerate() {
        for &li in cl {
            classe[li as usize] = c as u8;
        }
    }
    res.plaqs
        .iter()
        .filter(|p| {
            let c: Vec<u8> = p.e.iter().map(|e| classe[(e >> 1) as usize]).collect();
            (0..4).any(|i| (i + 1..4).any(|j| c[i] == c[j]))
        })
        .count()
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ResD3 {
    pub mutant: String,
    pub liens_sensibles: usize,
    pub mesures: usize,
    pub ecart: f64,
    pub erreur: f64,
    pub z: f64,
    pub ecart_temoin: f64,
    pub z_temoin: f64,
}

/// (d3) Stationnarité : pour les liens qui touchent une plaquette de poids ≠ 1 (couture) et, en témoin, pour des liens du
/// volume, ⟨(U·A)[0]⟩ = ⟨|A| I₂(β|A|)/I₁(β|A|)⟩ (identité de Schwinger-Dyson du bain thermique, agrafe EXACTE de l'action).
#[allow(clippy::too_many_arguments)]
pub fn porte_d3(
    l: usize,
    t: usize,
    largeur: usize,
    beta: f64,
    therm: usize,
    mes: usize,
    graine: u64,
    mutant: Mutant,
) -> Result<ResD3, String> {
    let g = GeoRep::nouveau(l, t, 2)?;
    let mut res = Reseau::pur(
        g,
        Region::tranche(&g, largeur),
        &[graine, graine ^ 0xABCD],
        mutant,
    )?;
    res.beta = beta;
    res.init(true);
    let mut sensibles = vec![false; 4 * g.v];
    for (i, p) in res.plaqs.iter().enumerate() {
        if res.coef[i] != 1.0 {
            for e in p.e {
                sensibles[(e >> 1) as usize] = true;
            }
        }
    }
    let ens: Vec<usize> = (0..4 * g.v).filter(|&li| sensibles[li]).collect();
    // témoin : liens au milieu du cylindre (t = T/2)
    let temoin: Vec<usize> = (0..4 * g.v)
        .filter(|&li| g.decomp(li % g.v).1 == g.t / 2)
        .collect();
    for _ in 0..therm {
        res.pas(2);
    }
    let sd = |r: &Reseau, ens: &[usize]| {
        let v: Vec<f64> = ens
            .par_iter()
            .map(|&li| {
                let a = r.agrafe(li);
                let k = su2_hb::qnorm(a);
                su2_hb::qmul(r.u[li], a)[0] - k * rapport_i2_i1(beta * k)
            })
            .collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    let (mut y, mut yt) = (Vec::new(), Vec::new());
    for _ in 0..mes {
        res.pas(2);
        y.push(sd(&res, &ens));
        yt.push(sd(&res, &temoin));
    }
    let (m, e) = moyenne_blocs(&y, 20);
    let (mt, et) = moyenne_blocs(&yt, 20);
    Ok(ResD3 {
        mutant: format!("{mutant:?}"),
        liens_sensibles: ens.len(),
        mesures: mes,
        ecart: m,
        erreur: e,
        z: m / e,
        ecart_temoin: mt,
        z_temoin: mt / et,
    })
}

// ------------------------------------------------------------------------------------------
// Balayages : interpolation en α (F₁ − F₀) et intégration thermodynamique en β
// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
pub struct PointAlpha {
    pub alpha: f64,
    pub moyenne: f64,
    pub erreur: f64,
    pub mesures: usize,
    pub plaquette: f64,
    pub plafonds: u64,
}

/// FILET (relecture c19b, bloquante 10), indépendant du domaine de β : un point dont la chaîne a atteint le plafond d'essais
/// du bain thermique (lien gardé tel quel, tirage dégénéré), ou dont l'erreur est nulle ou non finie (série constante :
/// configurations répétées), n'est JAMAIS un résultat sain : erreur de calcul (code 8 par la ligne de commande).
pub fn sante_points(points: &[PointAlpha]) -> Result<(), String> {
    for p in points {
        if p.plafonds > 0 {
            return Err(format!(
                "point α = {} : plafond d'essais du bain thermique atteint {} fois (tirage dégénéré) : résultat refusé",
                p.alpha, p.plafonds
            ));
        }
        if !(p.erreur.is_finite() && p.erreur > 0.0) {
            return Err(format!(
                "point α = {} : erreur {} nulle ou non finie (série constante, configurations répétées) : résultat refusé",
                p.alpha, p.erreur
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Balayage {
    pub l: usize,
    pub t: usize,
    pub largeur0: usize,
    pub largeur1: usize,
    pub beta: f64,
    pub points: Vec<PointAlpha>,
    /// F₁ − F₀ = ∫ ⟨S₁ − S₀⟩_α dα (Simpson) et son erreur statistique.
    pub delta_f: f64,
    pub erreur: f64,
    /// Trapèzes sur la même grille (écart = indicateur d'erreur de quadrature).
    pub delta_f_trapezes: f64,
    pub n_plaquettes_diff: usize,
    pub secondes: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ParamsChaine {
    pub therm: usize,
    pub mes: usize,
    pub sep: usize,
    pub n_or: usize,
    pub blocs: usize,
}

/// Nombre de points d'une quadrature de Simpson (`--nalpha`, `--nbeta`) : impair et ≥ 3. Contrôlé AVANT toute chaîne
/// (relecture intrication-claude, objection 1 : `simpson` paniquait APRÈS le calcul, et `--nalpha 0` débordait).
/// Borne haute (relecture c19, bloquante 2) : `POINTS_MAX` = 32767, champ de 15 bits du flux ; les tableaux proportionnels aux
/// points restent bornés et comptés par la garde de mémoire.
pub fn controler_points(option: &str, n: usize) -> Result<(), String> {
    if n < 3 || n.is_multiple_of(2) || n > POINTS_MAX {
        return Err(format!(
            "--{option} = {n} : nombre IMPAIR de points entre 3 et {POINTS_MAX} requis (quadrature de Simpson)"
        ));
    }
    Ok(())
}

/// Domaine admis de β (relecture c19b, bloquante 10) : 10⁻³ ≤ β ≤ 10³, fini.
/// Borne basse : le tirage de Creutz (α = β|A| < 2) calcule 1 − e^(−2α) par soustraction ; pour |A| ≥ 0,1 il garde au moins
/// 12 chiffres significatifs à β = 10⁻³, alors qu'à β = 10⁻²⁰ e^(−2α) vaut 1 en flottant et TOUS les essais sont rejetés.
/// Borne haute : au-delà de 10³ le réseau est gelé (1 − ½Tr U ~ 1/β) et sans usage (β ≈ 2-3) ; β|A| ≤ 1,2·10⁴, β·Σ_p
/// ≤ 10³ × 2·6V restent finis. Un |A| exceptionnellement petit dans le domaine reste couvert par le FILET : un plafond
/// d'essais atteint, ou une erreur nulle, est une erreur de calcul (`sante_points`, code 8).
pub const BETA_MIN: f64 = 1e-3;
pub const BETA_MAX: f64 = 1e3;

/// Surrelaxations par pas : au plus 1000 (`--or` ; produit (1 + or) × liens du débit sans débordement).
pub const OR_MAX: usize = 1000;

/// Balayages de bain thermique d'une chaîne : therm + mes × sep, en arithmétique vérifiée ; doit rester < 2³² (le compteur de
/// balayage du flux ne reboucle jamais : relecture c19, majeure 5).
pub fn balayages_par_chaine(therm: usize, mes: usize, sep: usize) -> Option<u64> {
    let n = (mes as u64)
        .checked_mul(sep as u64)?
        .checked_add(therm as u64)?;
    (n < 1 << 32).then_some(n)
}

/// Paramètres d'une chaîne : `--or` ≤ 1000 et budget therm + mes × sep < 2³² balayages par chaîne. Contrôlé AVANT l'ouverture
/// du run par la ligne de commande et par `balayage_specs` / `doublement_specs`.
///
/// Relecture c19 (majeure 4) : `--sep` ≥ 1 (sinon la même configuration est mesurée plusieurs fois et l'erreur est nulle) ;
/// 2 ≤ `--blocs` ≤ `--mes` et `--mes` multiple de `--blocs` (moyenne et erreur sur les mêmes observations, erreur finie).
pub fn controler_chaine(p: &ParamsChaine) -> Result<(), String> {
    if p.sep == 0 {
        return Err("--sep = 0 : au moins un pas entre deux mesures requis".into());
    }
    if p.blocs < 2 || p.blocs > p.mes || !p.mes.is_multiple_of(p.blocs) {
        return Err(format!(
            "--blocs = {} et --mes = {} : 2 ≤ --blocs ≤ --mes et --mes multiple de --blocs requis (moyenne et erreur sur les mêmes observations)",
            p.blocs, p.mes
        ));
    }
    if p.n_or > OR_MAX {
        return Err(format!(
            "--or = {} : au plus {OR_MAX} surrelaxations par pas",
            p.n_or
        ));
    }
    if balayages_par_chaine(p.therm, p.mes, p.sep).is_none() {
        return Err(format!(
            "--therm {} + --mes {} × --sep {} : au plus 2³² − 1 balayages par chaîne (le compteur Philox de balayage ne reboucle pas)",
            p.therm, p.mes, p.sep
        ));
    }
    Ok(())
}

/// Vecteur de capacité `n` réservé sans panique (réservation faillible : relecture c19, bloquante 2).
pub fn reserver<T>(n: usize, quoi: &str) -> Result<Vec<T>, String> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|e| format!("réservation de {n} {quoi} : {e}"))?;
    Ok(v)
}

/// β fini et > 0 (relecture intrication-claude, objection 3 : β = 0 ou NaN envoyait le tirage de Creutz au plafond de 10⁶
/// essais PAR LIEN, β < 0 est hors du domaine de l'action de Wilson simulée).
pub fn controler_beta(beta: f64) -> Result<(), String> {
    if !(beta.is_finite() && (BETA_MIN..=BETA_MAX).contains(&beta)) {
        return Err(format!(
            "--beta = {beta} : β fini, > 0 et ≤ {BETA_MAX}, et ≥ {BETA_MIN} requis (domaine admis [{BETA_MIN}, {BETA_MAX}])"
        ));
    }
    Ok(())
}

/// Flux des deux répliques de chaque point α d'un balayage (route α, point i, largeur `largeur0`) : `Flux::domaine`. La
/// largeur est un CHAMP du flux (la largeur l d'une `courbe` et le balayage l → l+1 de même graine tirent les mêmes nombres).
pub fn plan_balayage(
    graine: u64,
    nalpha: usize,
    largeur0: usize,
) -> Result<Vec<[Flux; 2]>, String> {
    (0..nalpha)
        .map(|i| Flux::paire(graine, Route::Alpha, i, largeur0))
        .collect()
}

/// Flux d'une `courbe` : un balayage par largeur (largeurs distinctes, contrôlées par la ligne de commande).
pub fn plan_courbe(
    graine: u64,
    nalpha: usize,
    largeurs: &[usize],
) -> Result<Vec<[Flux; 2]>, String> {
    let mut v = Vec::new();
    for &lw in largeurs {
        v.extend(plan_balayage(graine, nalpha, lw)?);
    }
    Ok(v)
}

/// Flux de la route β d'un `doublement` : pour chaque point i (le point 0, β = 0, n'est pas simulé), géométrie vide puis pleine.
pub fn plan_beta(graine: u64, nbeta: usize) -> Result<Vec<[[Flux; 2]; 2]>, String> {
    (0..nbeta)
        .map(|i| {
            Ok([
                Flux::paire(graine, Route::BetaVide, i, 0)?,
                Flux::paire(graine, Route::BetaPleine, i, 0)?,
            ])
        })
        .collect()
}

/// Flux d'un `doublement` : route α (largeur 0) puis route β.
pub fn plan_doublement(graine: u64, nalpha: usize, nbeta: usize) -> Result<Vec<[Flux; 2]>, String> {
    let mut v = plan_balayage(graine, nalpha, 0)?;
    for [vide, pleine] in plan_beta(graine, nbeta)? {
        v.push(vide);
        v.push(pleine);
    }
    Ok(v)
}

/// Graine fixe de `debit` (aucune option `--graine`).
pub const GRAINE_DEBIT: u64 = 0;

/// Flux de `debit` (route débit, largeur l/2).
pub fn plan_debit(largeur: usize) -> Result<[Flux; 2], String> {
    Flux::paire(GRAINE_DEBIT, Route::Debit, 0, largeur)
}

/// Chaînes (réseaux) vivantes en même temps pour `n` points indépendants : au plus le nombre de fils de rayon (objection 2 :
/// la garde de mémoire multiplie l'estimation d'une chaîne par ce nombre).
pub fn chaines_simultanees(n: usize) -> usize {
    n.min(rayon::current_num_threads()).max(1)
}

/// `f(i)` pour i ∈ 0..n, en parallèle par paquets de `chaines_simultanees(n)`, résultats dans l'ordre de i. Sans paquets, le
/// vol de travail de rayon (un fil qui attend un sweep parallèle prend un autre point) pourrait garder un réseau vivant par
/// point. Chaque point ne dépend que de i (graines) : les résultats ne dépendent pas du découpage.
fn par_paquets<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    let k = chaines_simultanees(n);
    let mut out = Vec::with_capacity(n);
    let mut debut = 0;
    while debut < n {
        let fin = (debut + k).min(n);
        let mut paquet: Vec<T> = (debut..fin).into_par_iter().map(&f).collect();
        out.append(&mut paquet);
        debut = fin;
    }
    out
}

/// Balayage en α entre les régions `largeur0` et `largeur1` (tranches), chaînes indépendantes en parallèle (paquets de
/// `chaines_simultanees(nalpha)` ; la garde de mémoire de la ligne de commande est `cli::garde_memoire`).
#[allow(clippy::too_many_arguments)]
pub fn balayage(
    l: usize,
    t: usize,
    largeur0: usize,
    largeur1: usize,
    beta: f64,
    nalpha: usize,
    p: ParamsChaine,
    graine: u64,
) -> Result<Balayage, String> {
    balayage_specs(
        &spec_tranche(l, t, largeur0),
        &spec_tranche(l, t, largeur1),
        beta,
        nalpha,
        p,
        graine,
    )
}

/// Balayage en α entre deux spécifications recollées du socle `geom` (A₀ à α = 0, A₁ à α = 1) : chaque réseau est construit
/// par `Reseau::depuis_spec` à partir de CES spécifications, celles que la ligne de commande inscrit au manifeste (relecture
/// intrication-claude, objection 8 : une seule construction de la géométrie, contrôlée : largeur ≤ L, T pair, s = 2…).
pub fn balayage_specs(
    s0: &GeoSpec,
    s1: &GeoSpec,
    beta: f64,
    nalpha: usize,
    p: ParamsChaine,
    graine: u64,
) -> Result<Balayage, String> {
    controler_points("nalpha", nalpha)?;
    controler_beta(beta)?;
    controler_chaine(&p)?;
    let (g, _, largeur0, largeur1) = controler_specs(s0, s1)?;
    if largeur0 == largeur1 {
        return Err(format!(
            "--l0 = --l1 = {largeur0} : aucune plaquette ne change, ΔF = 0 par construction (erreur nulle) ; largeurs différentes requises"
        ));
    }
    let (l, t) = (g.l, g.t);
    let debut = std::time::Instant::now();
    let plan = plan_balayage(graine, nalpha, largeur0)?;
    let mut alphas: Vec<f64> = reserver(nalpha, "points α")?;
    alphas.extend((0..nalpha).map(|i| i as f64 / (nalpha - 1) as f64));
    let pts: Vec<Result<(PointAlpha, usize), String>> = par_paquets(nalpha, |i| {
        let a = alphas[i];
        let mut res = Reseau::depuis_spec_flux(s0, s1, &plan[i], Mutant::Aucun)?;
        let nd = res.diff.len();
        let c = chaine(&mut res, beta, a, p.therm, p.mes, p.sep, p.n_or, true)?;
        let (m, e) = moyenne_blocs(&c.delta, p.blocs);
        let pl = c.plaq.iter().sum::<f64>() / c.plaq.len() as f64;
        Ok((
            PointAlpha {
                alpha: a,
                moyenne: m,
                erreur: e,
                mesures: p.mes,
                plaquette: pl,
                plafonds: c.n_cap,
            },
            nd,
        ))
    });
    let mut points = reserver(nalpha, "points α")?;
    let mut nd = 0;
    for r in pts {
        let (pt, n) = r?;
        nd = n;
        points.push(pt);
    }
    let y: Vec<f64> = points.iter().map(|q| q.moyenne).collect();
    let dy: Vec<f64> = points.iter().map(|q| q.erreur).collect();
    let (v, e) = simpson(&y, &dy);
    Ok(Balayage {
        l,
        t,
        largeur0,
        largeur1,
        beta,
        points,
        delta_f: v,
        erreur: e,
        delta_f_trapezes: trapezes(&y),
        n_plaquettes_diff: nd,
        secondes: debut.elapsed().as_secs_f64(),
    })
}

/// Prédiction de couplage fort (boucles indépendantes) : ⟨S₁ − S₀⟩_α = β Σ_{p : w0≠w1} (w1 − w0)(1 − r(β c_p(α))),
/// r = I₂/I₁. Premier ordre non nul en β : −(β²/4) Σ_p (w1 − w0) c_p(α) (Σ_p (w1 − w0) = 0 tue l'ordre β), linéaire en α,
/// d'intégrale −(β²/8) Σ_p (w1² − w0²) = 0 quand les deux géométries ont les mêmes poids à translation près (l ≥ 1).
pub fn prediction_couplage_fort(
    l: usize,
    t: usize,
    largeur0: usize,
    largeur1: usize,
    beta: f64,
    alpha: f64,
) -> Result<(f64, f64), String> {
    let g = GeoRep::nouveau(l, t, 2)?;
    let mut res = Reseau::nouveau(
        g,
        Region::tranche(&g, largeur0),
        Region::tranche(&g, largeur1),
        &[1, 2],
        Mutant::Aucun,
    )?;
    res.fixer_alpha(alpha);
    let mut tot = 0.0;
    let mut ordre1 = 0.0;
    for &i in &res.diff {
        let p = &res.plaqs[i as usize];
        let c = res.coef[i as usize];
        tot += (p.w1 - p.w0) * (1.0 - rapport_i2_i1(beta * c));
        ordre1 -= (p.w1 - p.w0) * c;
    }
    Ok((beta * tot, beta * beta / 4.0 * ordre1))
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Doublement {
    pub l: usize,
    pub t: usize,
    pub beta: f64,
    /// 2 log Z(T) − log Z(2T) par interpolation géométrique (A vide → A plein).
    pub par_alpha: f64,
    pub erreur_alpha: f64,
    /// même quantité par intégration thermodynamique en β de ⟨Σ_p ½Tr U_p⟩ (deux réseaux T contre un réseau 2T).
    pub par_beta: f64,
    pub erreur_beta: f64,
    pub z: f64,
    /// Trapèzes sur les mêmes grilles (indicateurs d'erreur de quadrature).
    pub par_alpha_trapezes: f64,
    pub par_beta_trapezes: f64,
    /// Profils : intégrande en α (balayage) et en β (⟨ΣP⟩_vide − ⟨ΣP⟩_plein, erreur).
    pub profil_alpha: Vec<PointAlpha>,
    pub profil_beta: Vec<(f64, f64, f64)>,
    /// Plafonds d'essais atteints par les chaînes de la route β (le filet exige 0).
    pub plafonds_beta: u64,
}

impl Balayage {
    /// Filet sur les points du balayage (`sante_points`).
    pub fn sante(&self) -> Result<(), String> {
        sante_points(&self.points)
    }
}

impl Doublement {
    /// Filet sur les deux routes : points α (`sante_points`), plafonds de la route β, erreurs β > 0 et finies (le point
    /// β = 0 n'est pas simulé).
    pub fn sante(&self) -> Result<(), String> {
        sante_points(&self.profil_alpha)?;
        if self.plafonds_beta > 0 {
            return Err(format!(
                "route β : plafond d'essais du bain thermique atteint {} fois (tirage dégénéré) : résultat refusé",
                self.plafonds_beta
            ));
        }
        for &(b, _, e) in self.profil_beta.iter().skip(1) {
            if !(e.is_finite() && e > 0.0) {
                return Err(format!(
                    "route β, β = {b} : erreur {e} nulle ou non finie : résultat refusé"
                ));
            }
        }
        Ok(())
    }
}

/// Conséquence de la porte (b) : S₂(tranche entière) = 2 log Z(T) − log Z(2T), par deux méthodes indépendantes.
pub fn doublement(
    l: usize,
    t: usize,
    beta: f64,
    nalpha: usize,
    nbeta: usize,
    p: ParamsChaine,
    graine: u64,
) -> Result<Doublement, String> {
    doublement_specs(
        &spec_tranche(l, t, 0),
        &spec_tranche(l, t, l),
        beta,
        nalpha,
        nbeta,
        p,
        graine,
    )
}

/// `doublement` à partir des deux spécifications inscrites au manifeste (A₀ vide, A₁ = toute la tranche) ; les réseaux
/// des deux routes (interpolation en α et intégration en β) sont construits par `Reseau::depuis_spec` (objection 8).
#[allow(clippy::too_many_arguments)]
pub fn doublement_specs(
    s0: &GeoSpec,
    s1: &GeoSpec,
    beta: f64,
    nalpha: usize,
    nbeta: usize,
    p: ParamsChaine,
    graine: u64,
) -> Result<Doublement, String> {
    controler_points("nalpha", nalpha)?;
    controler_points("nbeta", nbeta)?;
    controler_beta(beta)?;
    controler_chaine(&p)?;
    let (g, axe, largeur0, largeur1) = controler_specs(s0, s1)?;
    if largeur0 != 0 || largeur1 != s1.cotes[axe] {
        return Err(format!("doublement : A₀ vide et A₁ = toute la tranche requis (largeurs lues {largeur0} et {largeur1})"));
    }
    let (l, t) = (g.l, g.t);
    let plan_b = plan_beta(graine, nbeta)?;
    let b = balayage_specs(s0, s1, beta, nalpha, p, graine)?;
    // grille par la FRACTION i/(n−1) déjà calculée (β·i/(n−1) débordait en infini pour β = 1e308 : majeure 7)
    let mut betas: Vec<f64> = reserver(nbeta, "points β")?;
    betas.extend((0..nbeta).map(|i| beta * (i as f64 / (nbeta - 1) as f64)));
    let integr: Vec<Result<(f64, f64, u64), String>> = par_paquets(nbeta, |i| {
        let bb = betas[i];
        if bb == 0.0 {
            return Ok((0.0, 0.0, 0));
        }
        let mut plafonds = 0u64;
        let mut sommes: [Vec<f64>; 2] = [reserver(p.mes, "mesures")?, reserver(p.mes, "mesures")?];
        for (geom, spec) in [(0usize, s0), (1, s1)] {
            let mut res = Reseau::depuis_spec_flux(spec, spec, &plan_b[i][geom], Mutant::Aucun)?;
            res.beta = bb;
            res.init(true);
            for _ in 0..p.therm {
                res.pas(p.n_or);
            }
            for _ in 0..p.mes {
                for _ in 0..p.sep {
                    res.pas(p.n_or);
                }
                sommes[geom].push(res.somme_plaquettes().0);
            }
            plafonds = plafonds.saturating_add(res.n_cap);
        }
        let (m0, e0) = moyenne_blocs(&sommes[0], p.blocs);
        let (m1, e1) = moyenne_blocs(&sommes[1], p.blocs);
        Ok((m0 - m1, (e0 * e0 + e1 * e1).sqrt(), plafonds))
    });
    let mut y = reserver(nbeta, "points β")?;
    let mut dy = reserver(nbeta, "points β")?;
    let mut plafonds_beta = 0u64;
    for r in integr {
        let (m, e, c) = r?;
        y.push(m);
        dy.push(e);
        plafonds_beta = plafonds_beta.saturating_add(c);
    }
    let (v, e) = simpson(&y, &dy);
    let (v, e) = (v * beta, e * beta);
    let vt = trapezes(&y) * beta;
    let profil_beta = betas
        .iter()
        .zip(y.iter().zip(&dy))
        .map(|(&b, (&m, &e))| (b, m, e))
        .collect();
    Ok(Doublement {
        plafonds_beta,
        par_alpha_trapezes: b.delta_f_trapezes,
        par_beta_trapezes: vt,
        profil_alpha: b.points.clone(),
        profil_beta,
        l,
        t,
        beta,
        par_alpha: b.delta_f,
        erreur_alpha: b.erreur,
        par_beta: v,
        erreur_beta: e,
        z: (b.delta_f - v) / (b.erreur * b.erreur + e * e).sqrt(),
    })
}

/// log c₀(x) = log ∫ dU e^{x ½TrU} = log(2 I₁(x)/x) (Haar normalisée).
pub fn log_c0(x: f64) -> f64 {
    if x < 1e-8 {
        return x * x / 8.0;
    }
    (2.0 * bessel_i(1, x) / x).ln()
}

/// Différence d'énergie libre F₁ − F₀ en couplage fort, boucles indépendantes (exacte jusqu'aux surfaces fermées, O(u⁶)) :
/// log Z = Σ_p [−β w_p + log c₀(β w_p)] ; comme Σ_p w_p ne dépend pas de la région, F₁ − F₀ = −Σ_p [log c₀(β w1) − log c₀(β w0)].
/// Premier ordre : −(β²/8) Σ_p (w1² − w0²) ; pour l = 0 → 1 (tranche) : S₂(1) = β² L²/2.
pub fn prediction_df_couplage_fort(
    l: usize,
    t: usize,
    largeur0: usize,
    largeur1: usize,
    beta: f64,
) -> Result<(f64, f64), String> {
    let g = GeoRep::nouveau(l, t, 2)?;
    let res = Reseau::nouveau(
        g,
        Region::tranche(&g, largeur0),
        Region::tranche(&g, largeur1),
        &[1, 2],
        Mutant::Aucun,
    )?;
    let mut df = 0.0;
    let mut o1 = 0.0;
    for &i in &res.diff {
        let p = &res.plaqs[i as usize];
        df -= log_c0(beta * p.w1) - log_c0(beta * p.w0);
        o1 -= beta * beta / 8.0 * (p.w1 * p.w1 - p.w0 * p.w0);
    }
    Ok((df, o1))
}
