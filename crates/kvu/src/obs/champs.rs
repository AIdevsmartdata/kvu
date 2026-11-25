//! Champs par site pour le scanner (module « scanner », 11/11/2026) : densités locales d'une configuration SU(2),
//! calculées avec les fonctions de la référence CPU [`super::ref_cpu`] (aucun nouvel algorithme de physique).
//!
//! Disposition : un champ 4D est un vecteur de V = L⁴ réels indexé par le site kvu
//! s = ((x0·L + x1)·L + x2)·L + x3, donc un NPY de forme (L, L, L, L) indexé [x0, x1, x2, x3].
//!
//! Champs (définitions exactes, reprises dans le manifeste `kvu-champs/1`) :
//! - `action_brute` : e(x) = 4 Σ_{μ<ν} (1 − ½ Tr P_μν(x)), P_μν(x) = plaquette « avant » de coin x
//!   (`ref_cpu::plaquette_site`). Moyenne sur x = E_plaq = 24 (1 − ⟨P⟩) de kvu. L'action de Wilson vaut
//!   S_W = (β/4) Σ_x e(x).
//! - `action_flot` : la même après `n_pas` pas du flot de Wilson RK3 de kvu (`ref_cpu::flot_n_pas`), t = n_pas·dt.
//! - `energie_trefle_flot` : E_trèfle(x) après flot (`ref_cpu::densite_trefle_site`, normalisation de
//!   `ref_cpu::densites` : moyenne sur x = E_trèfle de kvu).
//! - `charge_flot` : q(x) au trèfle après flot (`ref_cpu::densite_trefle_site` : Σ_x q(x) = Q_top de kvu).
//! - `polyakov` : forme (4, L, L, L, L) ; ℓ_d(x) = ½ Tr ∏_k U_d(x + k ê_d) (`ref_cpu::polyakov_sites`), constante
//!   le long de d, réelle pour SU(2) ; la NORME de la boucle est |ℓ_d(x)|. Moyenne sur x = ⟨P_d⟩ de kvu.
//! - `plaq_negatives_brute`, `plaq_negatives_flot` : nombre (0 à 6) de plaquettes avant du site dont ½ Tr P < 0.
//!   C'est le SIGNE DE LA TRACE des plaquettes, un indicateur invariant de jauge ; ce n'est PAS une projection de
//!   centre (pas de jauge maximale du centre, pas de P-vortex).

use super::flot::ParamsFlot;
use super::ref_cpu::{self, Reseau, PLANS};
use super::wilson::ParamsWilson;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{k_final, k_reduce, qdag, qmul, Geo, Q};
use rayon::prelude::*;

/// Paramètres du flot appliqué avant les champs « _flot ».
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParamsChamps {
    pub dt: f64,
    pub n_pas: usize,
}

impl ParamsChamps {
    /// t = n_pas·dt.
    pub fn t(&self) -> f64 {
        self.n_pas as f64 * self.dt
    }
    /// √(8t), le rayon de lissage usuel.
    pub fn sqrt8t(&self) -> f64 {
        (8.0 * self.t()).sqrt()
    }
}

/// Champs par site d'une configuration (FP64).
#[derive(Clone, Debug, PartialEq)]
pub struct ChampsSite {
    pub l: usize,
    /// Côtés (l0, l1, l2, l3) ; [l; 4] pour une boîte cubique.
    pub cotes: [usize; 4],
    pub action_brute: Vec<f64>,
    pub action_flot: Vec<f64>,
    pub energie_trefle_flot: Vec<f64>,
    pub charge_flot: Vec<f64>,
    /// 4·V valeurs : direction d puis site.
    pub polyakov: Vec<f64>,
    pub plaq_negatives_brute: Vec<f64>,
    pub plaq_negatives_flot: Vec<f64>,
}

/// Description d'un champ pour le manifeste.
pub struct DescChamp {
    pub nom: &'static str,
    pub definition: &'static str,
    /// Agrégat comparé à l'observable globale de kvu : « moyenne » ou « somme ».
    pub agregat: &'static str,
}

/// Les champs, dans l'ordre d'écriture.
pub const DESCRIPTIONS: [DescChamp; 7] = [
    DescChamp {
        nom: "action_brute",
        definition: "e(x) = 4 * sum_{mu<nu} (1 - 1/2 Tr P_munu(x)), P_munu(x) plaquette avant de coin x (ref_cpu::plaquette_site) ; \
                     moyenne sur x = E_plaq = 24 (1 - <P>) ; action de Wilson S_W = (beta/4) sum_x e(x)",
        agregat: "moyenne",
    },
    DescChamp {
        nom: "action_flot",
        definition: "e(x) (meme definition que action_brute) apres n_pas pas du flot de Wilson RK3 de kvu (ref_cpu::flot_n_pas), t = n_pas*dt",
        agregat: "moyenne",
    },
    DescChamp {
        nom: "energie_trefle_flot",
        definition: "E_trefle(x) = 2 * sum_{plans} |g_munu(x)|^2, G_munu = (Q_munu - Q_munu^dag)/8 (trefle de 4 plaquettes), apres flot \
                     (ref_cpu::densite_trefle_site) ; moyenne sur x = E_trefle de kvu",
        agregat: "moyenne",
    },
    DescChamp {
        nom: "charge_flot",
        definition: "q(x) = (1/(2 pi^2)) (g01.g23 - g02.g13 + g03.g12) au trefle, apres flot (ref_cpu::densite_trefle_site) ; \
                     somme sur x = Q_top de kvu",
        agregat: "somme",
    },
    DescChamp {
        nom: "polyakov",
        definition: "forme (4,L,L,L,L) : l_d(x) = 1/2 Tr prod_k U_d(x + k e_d) (ref_cpu::polyakov_sites), configuration BRUTE, \
                     constante le long de d, reelle pour SU(2) ; norme de la boucle = |l_d(x)| ; moyenne sur x = <P_d> de kvu",
        agregat: "moyenne",
    },
    DescChamp {
        nom: "plaq_negatives_brute",
        definition: "nombre (0 a 6) de plaquettes avant de coin x avec 1/2 Tr P < 0, configuration brute ; signe de la trace, \
                     invariant de jauge ; PAS une projection de centre",
        agregat: "somme",
    },
    DescChamp {
        nom: "plaq_negatives_flot",
        definition: "nombre (0 a 6) de plaquettes avant de coin x avec 1/2 Tr P < 0, apres flot ; signe de la trace ; \
                     PAS une projection de centre",
        agregat: "somme",
    },
];

impl ChampsSite {
    /// Refuse un champ non fini (NaN ou infini) : jamais exporté, même sans portes.
    pub fn verifier_finis(&self) -> Resultat<()> {
        for (desc, x) in DESCRIPTIONS.iter().zip(self.tableaux()) {
            if let Some(i) = x.iter().position(|v| !v.is_finite()) {
                return Err(Erreur::Validation(format!(
                    "champs : « {} » non fini à l'indice {i} (valeur {}) : export refusé (pas de flot trop grand ? voir --dt)",
                    desc.nom, x[i]
                )));
            }
        }
        Ok(())
    }

    /// Les champs dans l'ordre de [`DESCRIPTIONS`].
    pub fn tableaux(&self) -> [&[f64]; 7] {
        [
            &self.action_brute,
            &self.action_flot,
            &self.energie_trefle_flot,
            &self.charge_flot,
            &self.polyakov,
            &self.plaq_negatives_brute,
            &self.plaq_negatives_flot,
        ]
    }

    /// Forme NPY d'un champ.
    pub fn forme(&self, nom: &str) -> Vec<usize> {
        let c = self.cotes;
        if nom == "polyakov" {
            vec![4, c[0], c[1], c[2], c[3]]
        } else {
            vec![c[0], c[1], c[2], c[3]]
        }
    }
}

/// Traces ½ Tr P_μν(x) des six plaquettes avant du site, dans l'ordre de [`PLANS`].
pub fn traces_plaquettes(res: &Reseau, s: usize) -> [f64; 6] {
    let g = res.geo;
    let x = g.coords(s);
    let mut t = [0.0; 6];
    for (p, &(mu, nu)) in PLANS.iter().enumerate() {
        let smu = g.nb_up(s, mu, x[mu]);
        let snu = g.nb_up(s, nu, x[nu]);
        let q = qmul(
            qmul(res.u(mu, s), res.u(nu, smu)),
            qmul(qdag(res.u(mu, snu)), qdag(res.u(nu, s))),
        );
        t[p] = q[0];
    }
    t
}

/// e(x) = 4 (6 − Σ ½ Tr P) via `ref_cpu::plaquette_site`.
fn action_site(res: &Reseau, s: usize) -> f64 {
    4.0 * (6.0 - ref_cpu::plaquette_site(res, s))
}

fn negatives_site(res: &Reseau, s: usize) -> f64 {
    traces_plaquettes(res, s)
        .iter()
        .filter(|&&t| t < 0.0)
        .count() as f64
}

/// Champ de Polyakov diffusé sur les sites : 4·V valeurs.
pub fn polyakov_diffuse(champ: &ChampHote) -> Resultat<Vec<f64>> {
    let geo = Geo::new(champ.l);
    let (l, v) = (geo.l, geo.v);
    let mut out = vec![0.0; 4 * v];
    for d in 0..4 {
        let ell = ref_cpu::polyakov_sites(champ, d)?;
        // ordre de polyakov_sites : x[dirs[2]] = j mod L, x[dirs[1]] = (j/L) mod L, x[dirs[0]] = j/L²
        let dirs: Vec<usize> = (0..4).filter(|&k| k != d).collect();
        out[d * v..(d + 1) * v]
            .par_iter_mut()
            .enumerate()
            .for_each(|(s, o)| {
                let x = geo.coords(s);
                let j = (x[dirs[0]] * l + x[dirs[1]]) * l + x[dirs[2]];
                *o = ell[j];
            });
    }
    Ok(out)
}

/// (action, E_trèfle, q, plaquettes négatives), un vecteur par champ.
type Densites = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// Densités par site d'un champ (sans flot).
fn densites_locales(champ: &ChampHote) -> Resultat<Densites> {
    let res = Reseau::nouveau(champ)?;
    let v = res.geo.v;
    let par_site: Vec<(f64, f64, f64, f64)> = (0..v)
        .into_par_iter()
        .map(|s| {
            let (e, q) = ref_cpu::densite_trefle_site(&res, s);
            (action_site(&res, s), e, q, negatives_site(&res, s))
        })
        .collect();
    Ok((
        par_site.iter().map(|t| t.0).collect(),
        par_site.iter().map(|t| t.1).collect(),
        par_site.iter().map(|t| t.2).collect(),
        par_site.iter().map(|t| t.3).collect(),
    ))
}

/// Densités par site de la configuration BRUTE : (action, plaquettes négatives) seulement. Le trèfle (24 feuilles par site)
/// n'est pas calculé : aucun champ brut ne l'exporte. Mêmes fonctions de site que [`densites_locales`] : valeurs identiques.
fn densites_brutes(champ: &ChampHote) -> Resultat<(Vec<f64>, Vec<f64>)> {
    let res = Reseau::nouveau(champ)?;
    let v = res.geo.v;
    let par_site: Vec<(f64, f64)> = (0..v)
        .into_par_iter()
        .map(|s| (action_site(&res, s), negatives_site(&res, s)))
        .collect();
    Ok((
        par_site.iter().map(|t| t.0).collect(),
        par_site.iter().map(|t| t.1).collect(),
    ))
}

/// Calcule tous les champs d'une configuration (le champ source n'est pas modifié).
pub fn calculer(champ: &ChampHote, p: &ParamsChamps) -> Resultat<ChampsSite> {
    if !p.dt.is_finite() || p.dt <= 0.0 {
        return Err(Erreur::Config(format!(
            "champs : dt = {} invalide (dt > 0)",
            p.dt
        )));
    }
    let (action_brute, plaq_negatives_brute) = densites_brutes(champ)?;
    let polyakov = polyakov_diffuse(champ)?;
    let flote = ref_cpu::flot_n_pas(champ, p.dt, p.n_pas)?;
    let (action_flot, energie_trefle_flot, charge_flot, plaq_negatives_flot) =
        densites_locales(&flote)?;
    Ok(ChampsSite {
        l: champ.l,
        cotes: [champ.l; 4],
        action_brute,
        action_flot,
        energie_trefle_flot,
        charge_flot,
        polyakov,
        plaq_negatives_brute,
        plaq_negatives_flot,
    })
}

// ------------------------------------------------------------------------------------------
// Portes
// ------------------------------------------------------------------------------------------

/// Seuil des portes de somme et de jauge (écart relatif).
pub const SEUIL: f64 = 1e-10;

/// Résultat d'une porte : une ligne `PASS|FAIL <ID> <détail>`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Porte {
    pub id: String,
    pub ok: bool,
    /// Valeur tirée du champ (somme ou moyenne).
    pub champ: f64,
    /// Valeur de référence (observable globale existante de kvu).
    pub reference: f64,
    /// Écart relatif mesuré.
    pub ecart: f64,
    pub detail: String,
}

impl Porte {
    pub fn ligne(&self) -> String {
        format!(
            "{} {} champ={:.17e} reference={:.17e} ecart_rel={:.3e} {}",
            if self.ok { "PASS" } else { "FAIL" },
            self.id,
            self.champ,
            self.reference,
            self.ecart,
            self.detail
        )
    }
}

/// Somme à arbre fixe (k_reduce/k_final de kvu).
pub fn somme(x: &[f64]) -> f64 {
    k_final(&k_reduce(x), 1.0)
}

/// Écart relatif |a − b| / échelle (échelle > 0). NaN si une des trois valeurs n'est pas finie : une comparaison avec NaN
/// est fausse, donc une porte `écart ≤ seuil` reste rouge (jamais d'écart « nul » fabriqué par `f64::max`).
fn ecart(a: f64, b: f64, echelle: f64) -> f64 {
    if !(a.is_finite() && b.is_finite() && echelle.is_finite()) {
        return f64::NAN;
    }
    (a - b).abs() / echelle.max(1e-300)
}

/// Maximum qui PROPAGE les NaN (`f64::max` renvoie l'autre opérande quand l'un est NaN).
fn max_nan(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

fn porte(id: &str, champ: f64, reference: f64, echelle: f64, detail: &str) -> Porte {
    let e = ecart(champ, reference, echelle);
    Porte {
        id: id.into(),
        ok: e <= SEUIL && champ.is_finite() && reference.is_finite(),
        champ,
        reference,
        ecart: e,
        detail: detail.into(),
    }
}

/// Valeurs globales de référence, calculées par les observables EXISTANTES de kvu.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct References {
    /// `ref_cpu::densites(brut).e_plaq`.
    pub e_plaq_brut: f64,
    /// 24 (1 − W(1,1)), W(1,1) de `ref_cpu::table_wilson` (12 orientations).
    pub e_plaq_wilson11: f64,
    /// Dernier point de `ref_cpu::flot` (ParamsFlot { dt, t_max = n_pas·dt, mesure_chaque = n_pas }).
    pub e_plaq_flot: f64,
    pub e_trefle_flot: f64,
    pub q_top_flot: f64,
    pub t_flot: f64,
    /// `ref_cpu::polyakov_moyen(brut)`.
    pub polyakov_moyen: [f64; 4],
    /// Nombre de plaquettes à trace négative compté par `ref_cpu::boucle(·, 1, 1)` (chemin de code indépendant).
    pub negatives_brut: f64,
    pub negatives_flot: f64,
}

fn compter_negatives_boucles(champ: &ChampHote) -> Resultat<f64> {
    let res = Reseau::nouveau(champ)?;
    let n: usize = (0..res.geo.v)
        .into_par_iter()
        .map(|s| {
            PLANS
                .iter()
                .filter(|&&(mu, nu)| ref_cpu::boucle(&res, s, mu, nu, 1, 1)[0] < 0.0)
                .count()
        })
        .sum();
    Ok(n as f64)
}

/// Références globales (refait le flot par `ref_cpu::flot`, chemin d'entrée indépendant de [`calculer`]).
pub fn references(champ: &ChampHote, p: &ParamsChamps) -> Resultat<References> {
    let d = ref_cpu::densites(champ)?;
    let w = ref_cpu::table_wilson(
        champ,
        &ParamsWilson {
            r_max: 1,
            t_max: 1,
            dir_temps: 3,
            toutes_orientations: true,
        },
    )?;
    let (e_plaq_flot, e_trefle_flot, q_top_flot, t_flot) = if p.n_pas == 0 {
        (d.e_plaq, d.e_trefle, d.q_top, 0.0)
    } else {
        let pf = ParamsFlot {
            dt: p.dt,
            t_max: p.n_pas as f64 * p.dt,
            mesure_chaque: p.n_pas,
            ..ParamsFlot::default()
        };
        let c = ref_cpu::flot(champ, &pf)?;
        let k = c.t.len() - 1;
        (c.e_plaq[k], c.e_trefle[k], c.q_top[k], c.t[k])
    };
    let flote = ref_cpu::flot_n_pas(champ, p.dt, p.n_pas)?;
    Ok(References {
        e_plaq_brut: d.e_plaq,
        e_plaq_wilson11: 24.0 * (1.0 - w.get(1, 1)),
        e_plaq_flot,
        e_trefle_flot,
        q_top_flot,
        t_flot,
        polyakov_moyen: ref_cpu::polyakov_moyen(champ)?,
        negatives_brut: compter_negatives_boucles(champ)?,
        negatives_flot: compter_negatives_boucles(&flote)?,
    })
}

/// Portes de somme : chaque champ redonne l'observable globale de kvu (écart relatif ≤ [`SEUIL`]).
/// Échelle de l'écart : |référence| pour les grandeurs positives ; Σ|q(x)| pour la charge et V·⟨|ℓ_d|⟩ pour
/// Polyakov (grandeurs signées dont la somme peut s'annuler) ; égalité exacte pour les comptes.
pub fn portes_somme(c: &ChampsSite, r: &References, p: &ParamsChamps) -> Vec<Porte> {
    let v: usize = c.cotes.iter().product();
    let vf = v as f64;
    let moy = |x: &[f64]| somme(x) / vf;
    let mut out = Vec::new();
    let mb = moy(&c.action_brute);
    out.push(porte(
        "CH-SOMME-ACTION-BRUTE",
        mb,
        r.e_plaq_brut,
        r.e_plaq_brut.abs(),
        "moyenne = densites().e_plaq",
    ));
    out.push(porte(
        "CH-SOMME-ACTION-BRUTE-W11",
        mb,
        r.e_plaq_wilson11,
        r.e_plaq_wilson11.abs(),
        "moyenne = 24 (1 - W(1,1)), table_wilson 12 orientations",
    ));
    let t_ok = (r.t_flot - p.t()).abs() <= 1e-12 * p.t().max(1.0);
    let mut pf = porte(
        "CH-SOMME-ACTION-FLOT",
        moy(&c.action_flot),
        r.e_plaq_flot,
        r.e_plaq_flot.abs(),
        "moyenne = flot().e_plaq au dernier t",
    );
    pf.ok &= t_ok;
    pf.detail
        .push_str(&format!(" t_ref={} t_champs={}", r.t_flot, p.t()));
    out.push(pf);
    out.push(porte(
        "CH-SOMME-TREFLE-FLOT",
        moy(&c.energie_trefle_flot),
        r.e_trefle_flot,
        r.e_trefle_flot.abs(),
        "moyenne = flot().e_trefle au dernier t",
    ));
    let l1q: f64 = somme(&c.charge_flot.iter().map(|x| x.abs()).collect::<Vec<_>>());
    out.push(porte(
        "CH-SOMME-CHARGE-FLOT",
        somme(&c.charge_flot),
        r.q_top_flot,
        l1q,
        "somme = flot().q_top au dernier t ; echelle = somme |q(x)|",
    ));
    for d in 0..4 {
        let tranche = &c.polyakov[d * v..(d + 1) * v];
        let l1: f64 = somme(&tranche.iter().map(|x| x.abs()).collect::<Vec<_>>()) / vf;
        out.push(porte(
            &format!("CH-SOMME-POLYAKOV-{d}"),
            moy(tranche),
            r.polyakov_moyen[d],
            l1,
            "moyenne = polyakov_moyen()[d] ; echelle = moyenne |l_d|",
        ));
    }
    let nb = somme(&c.plaq_negatives_brute);
    let nf = somme(&c.plaq_negatives_flot);
    out.push(Porte {
        id: "CH-SOMME-NEGATIVES-BRUTE".into(),
        ok: nb == r.negatives_brut,
        champ: nb,
        reference: r.negatives_brut,
        ecart: ecart(nb, r.negatives_brut, r.negatives_brut.abs().max(1.0)),
        detail: "egalite exacte avec le compte par boucle(1,1)".into(),
    });
    out.push(Porte {
        id: "CH-SOMME-NEGATIVES-FLOT".into(),
        ok: nf == r.negatives_flot,
        champ: nf,
        reference: r.negatives_flot,
        ecart: ecart(nf, r.negatives_flot, r.negatives_flot.abs().max(1.0)),
        detail: "egalite exacte avec le compte par boucle(1,1)".into(),
    });
    out
}

/// Porte de localité : en `n_sites` sites tirés (déterministes), e(x) et ℓ_d(x) recalculés par des chemins
/// indépendants (`ref_cpu::boucle(·,1,1)` et `ref_cpu::produit_chemin` de L pas) ; détecte une erreur
/// d'indexation que les portes de somme (invariantes par permutation) ne voient pas.
pub fn porte_localite(champ: &ChampHote, c: &ChampsSite, n_sites: usize) -> Resultat<Porte> {
    let res = Reseau::nouveau(champ)?;
    let geo = res.geo;
    let v = geo.v;
    let mut pire: f64 = 0.0;
    for k in 0..n_sites {
        let s = ((k as u64).wrapping_mul(2_654_435_761).wrapping_add(12_345) % v as u64) as usize;
        let e_ref: f64 = 4.0
            * PLANS
                .iter()
                .map(|&(mu, nu)| 1.0 - ref_cpu::boucle(&res, s, mu, nu, 1, 1)[0])
                .sum::<f64>();
        pire = max_nan(pire, (c.action_brute[s] - e_ref).abs());
        for d in 0..4 {
            let chemin: Vec<(usize, i32)> = vec![(d, 1); geo.l];
            let ell = ref_cpu::produit_chemin(&res, s, &chemin)[0];
            pire = max_nan(pire, (c.polyakov[d * v + s] - ell).abs());
        }
    }
    Ok(Porte {
        id: "CH-LOCALITE".into(),
        ok: pire.is_finite() && pire <= SEUIL,
        champ: pire,
        reference: 0.0,
        ecart: pire,
        detail: format!("{n_sites} sites : action_brute vs boucle(1,1), polyakov vs produit_chemin (ecart absolu max)"),
    })
}

/// Transformation de jauge aléatoire U_μ(x) → g(x) U_μ(x) g(x+μ)†, g de Haar (compteur Philox de kvu).
pub fn jauge_aleatoire(champ: &ChampHote, graine: u32) -> ChampHote {
    let geo = Geo::new(champ.l);
    let v = geo.v;
    let g: Vec<Q> = (0..v)
        .map(|s| crate::su2::cpu::haar_q(s as u32, 7, graine, 0x5ca9, 0x1111))
        .collect();
    let mut r = champ.clone();
    for mu in 0..4 {
        for s in 0..v {
            let x = geo.coords(s);
            let t = geo.nb_up(s, mu, x[mu]);
            let u = qmul(qmul(g[s], champ.lien(mu * v + s)), qdag(g[t]));
            r.q[4 * (mu * v + s)..4 * (mu * v + s) + 4].copy_from_slice(&u);
        }
    }
    r
}

/// Plancher du dénominateur de la porte de jauge : échelle O(1) des champs exportés (action 0..48, Polyakov ≤ 1, comptes
/// 0..6). Le critère est donc `max|f − f_g| ≤ SEUIL · max(max|f|, PLANCHER_JAUGE)` : relatif pour un champ d'échelle
/// ≥ 1, absolu (≤ 1e-10) pour un champ nul ou presque nul, où les arrondis d'une transformation de jauge (~1e-14) n'ont
/// aucun sens relatif. Limite assumée : une non-invariance d'amplitude < 1e-10 sur un champ < 1 n'est pas vue.
pub const PLANCHER_JAUGE: f64 = 1.0;

/// Porte de jauge : max_x |f(x) − f_g(x)| ≤ [`SEUIL`] · max(max_x |f(x)|, [`PLANCHER_JAUGE`]) pour chaque champ.
pub fn portes_jauge(a: &ChampsSite, b: &ChampsSite) -> Vec<Porte> {
    DESCRIPTIONS
        .iter()
        .zip(a.tableaux().iter().zip(b.tableaux().iter()))
        .map(|(desc, (x, y))| {
            let echelle = x.iter().fold(0.0f64, |m, v| max_nan(m, v.abs()));
            let pire = x
                .iter()
                .zip(y.iter())
                .fold(0.0f64, |m, (p, q)| max_nan(m, (p - q).abs()));
            let e = ecart(pire, 0.0, echelle.max(PLANCHER_JAUGE));
            Porte {
                id: format!("CH-JAUGE-{}", desc.nom.to_uppercase()),
                ok: e <= SEUIL && x.len() == y.len(),
                champ: pire,
                reference: echelle,
                ecart: e,
                detail: "max|f - f_g| / max(max|f|, 1), transformation de Haar aleatoire".into(),
            }
        })
        .collect()
}
