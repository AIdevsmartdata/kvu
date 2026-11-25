//! [EXPÉRIMENTAL] Secteur neutre (0⁺⁺, A₁⁺⁺ du groupe cubique) : boucles spatiales sur liens lissés APE, impulsion nulle.
//!
//! Définitions (figées dans `PREDICTIONS_NEUTRE.md`, chantier `kvu_neutre`) :
//! - temps = axe `temps` ; les trois autres axes sont les axes spatiaux, numérotés 1, 2, 3 dans l'ordre croissant ;
//!   un chemin est une suite de pas ±1, ±2, ±3 ; le lissage APE est celui de `super::ape_iteration` (liens de temps intacts) ;
//! - une FORME est un chemin fermé ; ses IMAGES sont les classes (décalage cyclique, renversement) de ses 48 transformées
//!   par le groupe octaédral O_h (dédoublonnées) ; l'opérateur A₁⁺⁺ d'une forme est la moyenne sur les images ET sur les sites
//!   de la tranche de ½ Tr W (réel pour SU(2)) : S_f(t) = (1/(N_im · V₃)) Σ_x Σ_im ½ Tr W_im(x, t) ;
//! - opérateurs = (niveau de lissage, forme), indice op = i_niveau · N_formes + i_forme ;
//! - corrélateur symétrisé par configuration : C_{ab}(τ) = (1/2T) Σ_t [S_a(t) S_b(t+τ) + S_b(t) S_a(t+τ)], temps périodique ;
//!   ⟨S_a⟩_t est écrit à côté : la SOUSTRACTION DU VIDE (C − ⟨S_a⟩⟨S_b⟩) se fait à l'analyse, par échantillon jackknife.
//! Les formes de départ (4 des 5) viennent de la branche `integ/c03-glueball` (obs/glueball.rs : plaquette, 1×2, deux hexagones
//! gauches) ; l'impulsion nulle, l'invariance cubique et le temps périodique sont réécrits ici, sur boîtes rectangulaires ;
//! - IMPULSION NON NULLE (`--impulsions k1,k2,…`) : pour chaque cran k demandé (1 ≤ k < ℓ/2 sur chacun des trois axes spatiaux),
//!   S(t; k, d) = Σ_x e^{2πi k (x_d + c_d)/ℓ_d} (somme des images)(x, t) / (N_im · V₃) avec c le centre de la boucle, corrélation
//!   symétrisée comme à impulsion nulle puis moyenne sur les trois axes ; le champ étant réel, S(−k) = S(k)* et C(−k) = C(k) à
//!   l'octet près : la moyenne sur les deux signes de l'impulsion est donc implicite.
#![allow(clippy::doc_lazy_continuation)] // style seulement : mise en forme des listes de la documentation

use super::{ape_iteration_gen, UN};
use crate::commun::{Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul, Q};
use crate::su2::tube::reseau::GeoA;
use rayon::prelude::*;

/// (nom, chemin de départ).
pub const FORMES: [(&str, &[i8]); 11] = [
    ("P11", &[1, 2, -1, -2]),
    ("R12", &[1, 2, 2, -1, -2, -2]),
    ("S22", &[1, 1, 2, 2, -1, -1, -2, -2]),
    ("CH1", &[1, 2, 3, -1, -2, -3]),
    ("CH2", &[1, 3, 2, -3, -1, -2]),
    // base « riche » (spectro/gpu-1120) : jamais choisies par défaut ; chemins fermés, images par O_h comme les cinq premières
    ("R13", &[1, 2, 2, 2, -1, -2, -2, -2]),
    ("LL", &[1, 1, 2, -1, 2, -1, -2, -2]),
    ("CH3", &[1, 1, 2, 3, -1, -1, -2, -3]),
    ("CH4", &[1, 2, 3, -2, -1, -3]),
    ("CH5", &[1, 2, 2, 3, -1, -2, -2, -3]),
    ("CH6", &[1, 1, 2, 3, -1, -2, -1, -3]),
];

/// Contrôles négatifs (jamais en production).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutantN {
    Aucun,
    /// agrafe basse du lissage sans dagger : la covariance de jauge est perdue
    LissageFaux,
    /// la dernière image de chaque forme est oubliée : l'invariance cubique est perdue
    ImageOubliee,
}

impl MutantN {
    pub fn analyser(t: Option<&str>) -> Resultat<MutantN> {
        match t {
            None | Some("aucun") => Ok(MutantN::Aucun),
            Some("lissage-faux") => Ok(MutantN::LissageFaux),
            Some("image-oubliee") => Ok(MutantN::ImageOubliee),
            Some(a) => Err(Erreur::Config(format!("--mutant « {a} » inconnu (lissage-faux|image-oubliee)"))),
        }
    }
}

pub fn canonique(chemin: &[i8]) -> Vec<i8> {
    let n = chemin.len();
    let mut meilleur: Option<Vec<i8>> = None;
    for renverse in [false, true] {
        let seq: Vec<i8> = if renverse { chemin.iter().rev().map(|p| -p).collect() } else { chemin.to_vec() };
        for s in 0..n {
            let c: Vec<i8> = (0..n).map(|i| seq[(s + i) % n]).collect();
            if meilleur.as_ref().is_none_or(|m| c < *m) {
                meilleur = Some(c);
            }
        }
    }
    meilleur.unwrap_or_default()
}

/// Les 48 éléments de O_h : (permutation des axes, signes).
pub fn groupe() -> Vec<([usize; 3], [i8; 3])> {
    let perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    let mut out = Vec::new();
    for p in perms {
        for k in 0..8u8 {
            let s = |i: u32| if (k >> (2 - i)) & 1 == 0 { 1i8 } else { -1i8 };
            out.push((p, [s(0), s(1), s(2)]));
        }
    }
    out
}

/// Images distinctes (formes canoniques) d'un chemin sous O_h.
pub fn images(chemin: &[i8]) -> Vec<Vec<i8>> {
    let mut out: Vec<Vec<i8>> = Vec::new();
    for (p, s) in groupe() {
        let im: Vec<i8> = chemin
            .iter()
            .map(|&x| {
                let a = x.unsigned_abs() as usize - 1;
                x.signum() * s[a] * (p[a] as i8 + 1)
            })
            .collect();
        let c = canonique(&im);
        if !out.contains(&c) {
            out.push(c);
        }
    }
    out
}

/// Paramètres d'une mesure neutre.
#[derive(Clone, Debug)]
pub struct ParamNeutre {
    pub temps: usize,
    pub niveaux: Vec<usize>,
    pub alpha: f64,
    /// indices dans `FORMES`
    pub formes: Vec<usize>,
    pub tau_max: usize,
    pub mutant: MutantN,
}

impl ParamNeutre {
    pub fn nops(&self) -> usize {
        self.niveaux.len() * self.formes.len()
    }
}

/// Résultat sur une configuration.
#[derive(Clone, Debug)]
pub struct MesureNeutre {
    /// S[op][t]
    pub s: Vec<Vec<f64>>,
    /// C[paire][τ], paires (a ≤ b) dans l'ordre (0,0), (0,1) … ; τ = 0..=tau_max
    pub corr: Vec<Vec<f64>>,
    /// ⟨S_op⟩_t
    pub moy: Vec<f64>,
    /// corrélateurs à impulsion p = 2π/ℓ_d ≠ 0 le long d'UN axe spatial d, moyennés sur les 3 axes (vide si non demandés) :
    /// Cp_{ab}(τ) = (1/3) Σ_d (1/2T) Σ_t Re[S_a(t;d)* S_b(t+τ;d) + S_b(t;d)* S_a(t+τ;d)], S(t;d) = Σ_x e^{i p_d x_d} (somme des images)(x,t) / (N_im V₃)
    /// Rendu pour compatibilité : le premier cran demandé, c'est-à-dire `corr_ps[0]` (le cran 1 avec `mesurer_p`).
    pub corr_p: Vec<Vec<f64>>,
    /// corrélateurs à impulsion non nulle, UN par cran demandé, dans l'ordre de la liste de crans : `corr_ps[ik][paire][τ]`
    /// (même définition que `corr_p` avec p = 2π·|k|/ℓ_d ; vide si aucune impulsion demandée). Un cran est mesuré à l'octet
    /// près comme s'il était le seul demandé : la sortie ne dépend pas du reste de la liste.
    pub corr_ps: Vec<Vec<Vec<f64>>>,
    /// corrélateurs à impulsion HORS AXE, UN par orbite cubique demandée (`--impulsions-vecteurs`), dans l'ordre de la liste :
    /// `corr_orb[io][paire][τ]` = moyenne de C_k sur tous les vecteurs k de l'orbite (voir [`Orbite`]). Vide si non demandé.
    pub corr_orb: Vec<Vec<Vec<f64>>>,
}

fn axes_spatiaux(temps: usize) -> [usize; 3] {
    let v: Vec<usize> = (0..4).filter(|&d| d != temps).collect();
    [v[0], v[1], v[2]]
}

/// ½ Tr du produit le long du chemin fermé commençant en `s0`.
fn trace_chemin(g: &GeoA, u: &[Q], ax: &[usize; 3], pas: &[i8], s0: usize) -> f64 {
    let mut s = s0;
    let mut p = UN;
    for &st in pas {
        let a = ax[st.unsigned_abs() as usize - 1];
        let xa = (s / g.st[a]) % g.d[a];
        if st > 0 {
            p = qmul(p, u[a * g.v + s]);
            s = g.nb_up(s, a, xa);
        } else {
            s = g.nb_dn(s, a, xa);
            p = qmul(p, qdag(u[a * g.v + s]));
        }
    }
    debug_assert_eq!(s, s0, "chemin non fermé");
    p[0]
}

/// S_f(t) pour une liste d'images, t = 0..T−1 (somme séquentielle par tranche : résultat indépendant du nombre de fils).
pub fn tranches(g: &GeoA, u: &[Q], temps: usize, imgs: &[Vec<i8>]) -> Vec<f64> {
    let ax = axes_spatiaux(temps);
    let tt = g.d[temps];
    let v3 = g.v / tt;
    (0..tt)
        .into_par_iter()
        .map(|t| {
            let mut acc = 0.0;
            for s in 0..g.v {
                if (s / g.st[temps]) % g.d[temps] != t {
                    continue;
                }
                for im in imgs {
                    acc += trace_chemin(g, u, &ax, im, s);
                }
            }
            acc / (imgs.len() * v3) as f64
        })
        .collect()
}

/// Centre (moyenne des sommets visités, en unités de maille) d'un chemin fermé depuis son site de base, par axe spatial 1..3 (indices 0..2).
pub fn centre_chemin(pas: &[i8]) -> [f64; 3] {
    let mut pos = [0i64; 3];
    let mut som = [0i64; 3];
    for &st in pas {
        let a = st.unsigned_abs() as usize - 1;
        pos[a] += st.signum() as i64;
        for d in 0..3 {
            som[d] += pos[d];
        }
    }
    // le dernier sommet est le site de base (déjà compté une fois au départ, position 0) : n sommets distincts = pas.len()
    [som[0] as f64 / pas.len() as f64, som[1] as f64 / pas.len() as f64, som[2] as f64 / pas.len() as f64]
}

/// Tables (cos, sin) de e^{i k θ_x} avec θ_x = 2π x/ℓ_d, le long de chacun des trois axes spatiaux (`ax`), par coordonnée x.
/// La table est bâtie sur |k| et le signe de k est appliqué au seul sinus : e^{−iθ} = cos θ − i sin θ à L'OCTET PRÈS, sans
/// dépendre de la parité de la libm (sin(−θ) = −sin θ est imposé, pas calculé). Pour k = 1, multiplier par 1.0 est exact :
/// les opérations sont celles de la version d'origine (`pi2 * x as f64 / l as f64`), donc les mêmes bits.
fn tables_phase(g: &GeoA, ax: &[usize; 3], k: i64) -> Vec<Vec<(f64, f64)>> {
    let pi2 = 2.0 * std::f64::consts::PI;
    let ak = k.unsigned_abs() as f64;
    let sg = if k < 0 { -1.0 } else { 1.0 };
    let tab: Vec<Vec<(f64, f64)>> = (0..3)
        .map(|d| {
            let l = g.d[ax[d]];
            (0..l)
                .map(|x| {
                    let th = pi2 * x as f64 * ak / l as f64;
                    (th.cos(), sg * th.sin())
                })
                .collect()
        })
        .collect();
    tab
}

/// Phase (cos, sin) de e^{i k 2π c_d/ℓ_d} prise au CENTRE de chaque image, par axe spatial — indépendante du choix du site
/// de base de l'image. Même construction sur |k| que `tables_phase` : pour k = 1, identique au bit à la version d'origine.
fn phases_centre(imgs: &[Vec<i8>], g: &GeoA, ax: &[usize; 3], k: i64) -> Vec<[(f64, f64); 3]> {
    let pi2 = 2.0 * std::f64::consts::PI;
    let ak = k.unsigned_abs() as f64;
    let sg = if k < 0 { -1.0 } else { 1.0 };
    let ph: Vec<[(f64, f64); 3]> = imgs
        .iter()
        .map(|im| {
            let c = centre_chemin(im);
            let mut o = [(1.0, 0.0); 3];
            for d in 0..3 {
                let th = pi2 * c[d] * ak / g.d[ax[d]] as f64;
                o[d] = (th.cos(), sg * th.sin());
            }
            o
        })
        .collect();
    ph
}

/// Comme `tranches`, et en plus les sommes de Fourier à l'impulsion k·2π/ℓ_d le long de chacun des trois axes spatiaux,
/// pour CHACUN des crans demandés (`ks`, signés), la phase d'une boucle étant prise AU CENTRE de la boucle
/// (e^{ip·(x_base + c_image)}) : indépendante du choix du site de base de l'image.
/// Renvoie (S_f(t), [ik][t][d] = [Σ cos, Σ sin] normalisés comme S_f). Le premier résultat est identique au bit à `tranches`.
/// Pour un cran donné, le résultat est identique au bit à une mesure menée sur ce seul cran : les additions d'un cran donné
/// sont faites dans le même ordre (les crans ne se mélangent pas), la sortie ne dépend donc pas du reste de la liste.
#[allow(clippy::type_complexity)] // style seulement (indices explicites conservés : aucun changement numérique)
#[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
pub fn tranches_ps(g: &GeoA, u: &[Q], temps: usize, imgs: &[Vec<i8>], ks: &[i64]) -> (Vec<f64>, Vec<Vec<[[f64; 2]; 3]>>) {
    let ax = axes_spatiaux(temps);
    let tt = g.d[temps];
    let v3 = g.v / tt;
    let tab: Vec<Vec<Vec<(f64, f64)>>> = ks.iter().map(|&k| tables_phase(g, &ax, k)).collect();
    // phase du centre de chaque image, par cran et par axe : (cos, sin) de k · 2π c_d/ℓ_d
    let centres: Vec<Vec<[(f64, f64); 3]>> = ks.iter().map(|&k| phases_centre(imgs, g, &ax, k)).collect();
    let r: Vec<(f64, Vec<[[f64; 2]; 3]>)> = (0..tt)
        .into_par_iter()
        .map(|t| {
            let mut acc = 0.0;
            // par cran, par image et par axe : Σ_x tr cos(θx), Σ_x tr sin(θx)
            let mut fo = vec![vec![[[0.0f64; 2]; 3]; imgs.len()]; ks.len()];
            for s in 0..g.v {
                if (s / g.st[temps]) % g.d[temps] != t {
                    continue;
                }
                for (kk, im) in imgs.iter().enumerate() {
                    let tr = trace_chemin(g, u, &ax, im, s);
                    acc += tr;
                    for (ik, tb) in tab.iter().enumerate() {
                        for d in 0..3 {
                            let xd = (s / g.st[ax[d]]) % g.d[ax[d]];
                            fo[ik][kk][d][0] += tr * tb[d][xd].0;
                            fo[ik][kk][d][1] += tr * tb[d][xd].1;
                        }
                    }
                }
            }
            let n = (imgs.len() * v3) as f64;
            let mut tot: Vec<[[f64; 2]; 3]> = vec![[[0.0f64; 2]; 3]; ks.len()];
            for (ik, ce) in centres.iter().enumerate() {
                for (kk, cek) in ce.iter().enumerate() {
                    for d in 0..3 {
                        let (cc, cs) = cek[d];
                        let (a, b) = (fo[ik][kk][d][0], fo[ik][kk][d][1]);
                        tot[ik][d][0] += a * cc - b * cs;
                        tot[ik][d][1] += a * cs + b * cc;
                    }
                }
                for d in 0..3 {
                    tot[ik][d][0] /= n;
                    tot[ik][d][1] /= n;
                }
            }
            (acc / n, tot)
        })
        .collect();
    // correctif du pilote (19/11) : `unzip` donnait [t][cran] ; on rend [cran][t] comme le dit la documentation et comme
    // l'attendent l'appelant (`sps[ik].push(v)`) et les tests.
    let (s, pt): (Vec<f64>, Vec<Vec<[[f64; 2]; 3]>>) = r.into_iter().unzip();
    let ps: Vec<Vec<[[f64; 2]; 3]>> = (0..ks.len()).map(|ik| pt.iter().map(|v| v[ik]).collect()).collect();
    (s, ps)
}

/// ORACLE de non-régression : l'implémentation EXACTE de l'option `--impulsion` telle qu'introduite par le commit
/// `5e9e4e1` (un seul cran, 2π/ℓ_d, phase au centre de la boucle, moyenne des trois axes), reprise mot pour mot. Elle n'est
/// plus appelée en production (`tranches_ps` la remplace) mais sert de référence : le test `impulsions_cran1_contre_oracle`
/// exige que `tranches_ps(g, u, temps, imgs, &[1])` rende les MÊMES BITS qu'elle, ce qui fige le contenu de `neutre_p.bin`.
#[cfg(test)]
#[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
pub fn tranches_p_origine(g: &GeoA, u: &[Q], temps: usize, imgs: &[Vec<i8>]) -> (Vec<f64>, Vec<[[f64; 2]; 3]>) {
    let ax = axes_spatiaux(temps);
    let tt = g.d[temps];
    let v3 = g.v / tt;
    let pi2 = 2.0 * std::f64::consts::PI;
    let tab: Vec<Vec<(f64, f64)>> = (0..3)
        .map(|d| {
            let l = g.d[ax[d]];
            (0..l).map(|x| ((pi2 * x as f64 / l as f64).cos(), (pi2 * x as f64 / l as f64).sin())).collect()
        })
        .collect();
    // phase du centre de chaque image, par axe : (cos, sin) de 2π c_d/ℓ_d
    let centres: Vec<[(f64, f64); 3]> = imgs
        .iter()
        .map(|im| {
            let c = centre_chemin(im);
            let mut o = [(1.0, 0.0); 3];
            for d in 0..3 {
                let th = pi2 * c[d] / g.d[ax[d]] as f64;
                o[d] = (th.cos(), th.sin());
            }
            o
        })
        .collect();
    let r: Vec<(f64, [[f64; 2]; 3])> = (0..tt)
        .into_par_iter()
        .map(|t| {
            let mut acc = 0.0;
            // par image et par axe : Σ_x tr cos(θx), Σ_x tr sin(θx)
            let mut fo = vec![[[0.0f64; 2]; 3]; imgs.len()];
            for s in 0..g.v {
                if (s / g.st[temps]) % g.d[temps] != t {
                    continue;
                }
                for (k, im) in imgs.iter().enumerate() {
                    let tr = trace_chemin(g, u, &ax, im, s);
                    acc += tr;
                    for d in 0..3 {
                        let xd = (s / g.st[ax[d]]) % g.d[ax[d]];
                        fo[k][d][0] += tr * tab[d][xd].0;
                        fo[k][d][1] += tr * tab[d][xd].1;
                    }
                }
            }
            let n = (imgs.len() * v3) as f64;
            let mut tot = [[0.0f64; 2]; 3];
            for k in 0..imgs.len() {
                for d in 0..3 {
                    let (cc, cs) = centres[k][d];
                    let (a, b) = (fo[k][d][0], fo[k][d][1]);
                    tot[d][0] += a * cc - b * cs;
                    tot[d][1] += a * cs + b * cc;
                }
            }
            for d in 0..3 {
                tot[d][0] /= n;
                tot[d][1] /= n;
            }
            (acc / n, tot)
        })
        .collect();
    (r.iter().map(|x| x.0).collect(), r.iter().map(|x| x.1).collect())
}

/// Orbite du groupe cubique O_h (permutations des trois axes spatiaux et signes) d'un vecteur entier k = (k₁, k₂, k₃),
/// composantes en unités de 2π/ℓ le long des axes spatiaux 1, 2, 3 (ordre croissant des axes ≠ temps).
/// - `representant` : |k| rangé par ordre décroissant (forme canonique : (1,2,2) et (2,−2,1) donnent (2,2,1)) ;
/// - `taille` : nombre de vecteurs distincts de l'orbite complète (24 pour (2,2,1), 6 pour (3,0,0), 48 si trois composantes
///   distinctes non nulles) ;
/// - `vecteurs` : les représentants de l'orbite MODULO k → −k, c'est-à-dire ceux dont la première composante non nulle est > 0,
///   en ordre lexicographique DÉCROISSANT (pour k = (m,0,0) : (m,0,0), (0,m,0), (0,0,m), l'ordre des axes de `--impulsions`).
///   Le champ de liens étant réel, S(−k) = S(k)* et C(−k) = C(k) : la moyenne sur ces `taille/2` vecteurs est la moyenne sur l'orbite
///   complète.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Orbite {
    pub representant: [i64; 3],
    pub taille: usize,
    pub vecteurs: Vec<[i64; 3]>,
}

pub fn orbite(k: [i64; 3]) -> Orbite {
    let mut rep = [k[0].abs(), k[1].abs(), k[2].abs()];
    rep.sort_unstable_by(|a, b| b.cmp(a));
    let mut tous: Vec<[i64; 3]> = Vec::new();
    for (p, sg) in groupe() {
        let mut v = [0i64; 3];
        for a in 0..3 {
            v[p[a]] = sg[a] as i64 * rep[a];
        }
        if !tous.contains(&v) {
            tous.push(v);
        }
    }
    let taille = tous.len();
    let mut vecteurs: Vec<[i64; 3]> = tous
        .into_iter()
        .filter(|v| v.iter().find(|&&x| x != 0).is_some_and(|&x| x > 0))
        .collect();
    vecteurs.sort_unstable_by(|a, b| b.cmp(a));
    Orbite { representant: rep, taille, vecteurs }
}

/// Nom de fichier d'une orbite : `neutre_o<k1>_<k2>_<k3>.bin` sur le représentant (ex. `neutre_o2_2_1.bin`).
pub fn nom_fichier_orbite(o: &Orbite) -> String {
    format!("neutre_o{}_{}_{}.bin", o.representant[0], o.representant[1], o.representant[2])
}

/// Contrôle d'un vecteur d'impulsion hors axe : non nul ; |k_d| < ℓ_d/2 sur CHAQUE axe spatial (zone de Brillouin stricte :
/// au delà, k se confond avec k − ℓ) ; les trois côtés spatiaux égaux (sinon les permutations d'axes ne sont pas des symétries
/// du réseau et « l'orbite cubique » n'a pas de sens).
pub fn controler_vecteur(g: &GeoA, temps: usize, k: [i64; 3]) -> Resultat<()> {
    if k == [0, 0, 0] {
        return Err(Erreur::Config("impulsions-vecteurs : vecteur nul refusé (l'impulsion nulle est déjà mesurée, voir neutre.bin)".into()));
    }
    let ax = axes_spatiaux(temps);
    if g.d[ax[0]] != g.d[ax[1]] || g.d[ax[1]] != g.d[ax[2]] {
        return Err(Erreur::Config(format!(
            "impulsions-vecteurs : les trois côtés spatiaux doivent être égaux pour une orbite cubique (côtés {} {} {})",
            g.d[ax[0]], g.d[ax[1]], g.d[ax[2]]
        )));
    }
    for (d, &kd) in k.iter().enumerate() {
        let l = g.d[ax[d]] as i128;
        if 2 * (kd as i128).abs() >= l {
            return Err(Erreur::Config(format!(
                "impulsions-vecteurs : vecteur {}:{}:{} refusé (il faut |k_d| < ℓ/2 = {}/2 sur chaque axe : composante {} = {kd})",
                k[0], k[1], k[2], l, d + 1
            )));
        }
    }
    Ok(())
}

/// Contrôle d'une liste d'orbites demandées (chaque vecteur ; pas deux vecteurs de la même orbite).
pub fn controler_vecteurs(g: &GeoA, temps: usize, ks: &[[i64; 3]]) -> Resultat<Vec<Orbite>> {
    let mut out: Vec<Orbite> = Vec::new();
    for &k in ks {
        controler_vecteur(g, temps, k)?;
        let o = orbite(k);
        if out.iter().any(|x| x.representant == o.representant) {
            return Err(Erreur::Config(format!(
                "impulsions-vecteurs : deux vecteurs de la même orbite ({}:{}:{})",
                o.representant[0], o.representant[1], o.representant[2]
            )));
        }
        out.push(o);
    }
    Ok(out)
}

/// Table (cos, sin) de e^{i k θ_x}, θ_x = 2π x/ℓ le long d'un axe ; mêmes opérations que `tables_phase` (bâtie sur |k|, signe sur le sinus).
fn table_axe(l: usize, k: i64) -> Vec<(f64, f64)> {
    let pi2 = 2.0 * std::f64::consts::PI;
    let ak = k.unsigned_abs() as f64;
    let sg = if k < 0 { -1.0 } else { 1.0 };
    (0..l)
        .map(|x| {
            let th = pi2 * x as f64 * ak / l as f64;
            (th.cos(), sg * th.sin())
        })
        .collect()
}

#[inline]
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Comme `tranches`, et en plus les sommes de Fourier S(t; k) = Σ_x e^{i k·(x + c)} (somme sur les images)(x, t) / (N_im V₃)
/// à des vecteurs k = (k₁, k₂, k₃) QUELCONQUES (`vs`, composantes signées sur les axes spatiaux 1, 2, 3), la phase d'une boucle
/// étant prise au CENTRE c de la boucle comme dans `tranches_ps`. Renvoie (S_f(t), [t][iv] = [Re, Im]). Le premier résultat est
/// identique au bit à `tranches`. Pour un vecteur (k,0,0) (ou ses permutations) les facteurs des axes nuls valent exactement
/// (1, 0) : le résultat est identique au bit à celui de `tranches_ps` au cran k le long de cet axe.
pub fn tranches_vec(g: &GeoA, u: &[Q], temps: usize, imgs: &[Vec<i8>], vs: &[[i64; 3]]) -> (Vec<f64>, Vec<Vec<[f64; 2]>>) {
    let ax = axes_spatiaux(temps);
    let tt = g.d[temps];
    let v3 = g.v / tt;
    let l = [g.d[ax[0]], g.d[ax[1]], g.d[ax[2]]];
    let pi2 = 2.0 * std::f64::consts::PI;
    // phase de site, par vecteur, indexée par j = x₀ + ℓ₀ (x₁ + ℓ₁ x₂) (produit des trois facteurs d'axe, dans cet ordre)
    let tabs: Vec<Vec<(f64, f64)>> = vs
        .iter()
        .map(|k| {
            let t: Vec<Vec<(f64, f64)>> = (0..3).map(|d| table_axe(l[d], k[d])).collect();
            let mut o = vec![(1.0, 0.0); l[0] * l[1] * l[2]];
            for x2 in 0..l[2] {
                for x1 in 0..l[1] {
                    for x0 in 0..l[0] {
                        o[x0 + l[0] * (x1 + l[1] * x2)] = cmul(cmul(t[0][x0], t[1][x1]), t[2][x2]);
                    }
                }
            }
            o
        })
        .collect();
    // phase du centre de chaque image, par vecteur : produit des trois facteurs (cos, sin)(k_d 2π c_d/ℓ_d)
    let centres: Vec<Vec<(f64, f64)>> = vs
        .iter()
        .map(|k| {
            imgs.iter()
                .map(|im| {
                    let c = centre_chemin(im);
                    let f = |d: usize| {
                        let th = pi2 * c[d] * k[d].unsigned_abs() as f64 / l[d] as f64;
                        let sg = if k[d] < 0 { -1.0 } else { 1.0 };
                        (th.cos(), sg * th.sin())
                    };
                    cmul(cmul(f(0), f(1)), f(2))
                })
                .collect()
        })
        .collect();
    let r: Vec<(f64, Vec<[f64; 2]>)> = (0..tt)
        .into_par_iter()
        .map(|t| {
            let mut acc = 0.0;
            let mut fo = vec![vec![[0.0f64; 2]; imgs.len()]; vs.len()];
            for s in 0..g.v {
                if (s / g.st[temps]) % g.d[temps] != t {
                    continue;
                }
                let x0 = (s / g.st[ax[0]]) % l[0];
                let x1 = (s / g.st[ax[1]]) % l[1];
                let x2 = (s / g.st[ax[2]]) % l[2];
                let j = x0 + l[0] * (x1 + l[1] * x2);
                for (kk, im) in imgs.iter().enumerate() {
                    let tr = trace_chemin(g, u, &ax, im, s);
                    acc += tr;
                    for (iv, tb) in tabs.iter().enumerate() {
                        fo[iv][kk][0] += tr * tb[j].0;
                        fo[iv][kk][1] += tr * tb[j].1;
                    }
                }
            }
            let n = (imgs.len() * v3) as f64;
            let mut tot: Vec<[f64; 2]> = vec![[0.0f64; 2]; vs.len()];
            for iv in 0..vs.len() {
                for (kk, &(cc, cs)) in centres[iv].iter().enumerate() {
                    let (a, b) = (fo[iv][kk][0], fo[iv][kk][1]);
                    tot[iv][0] += a * cc - b * cs;
                    tot[iv][1] += a * cs + b * cc;
                }
                tot[iv][0] /= n;
                tot[iv][1] /= n;
            }
            (acc / n, tot)
        })
        .collect();
    let (s, p): (Vec<f64>, Vec<Vec<[f64; 2]>>) = r.into_iter().unzip();
    (s, p)
}

/// Corrélateurs d'une orbite : sp[op][t][iv] = [Re, Im] sur les vecteurs de l'orbite ; moyenne sur les vecteurs, temps périodique,
/// symétrisés. Même formule que `correlateurs_p` (qui est le cas des trois vecteurs d'axe).
#[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
pub fn correlateurs_orbite(sp: &[Vec<Vec<[f64; 2]>>], tau_max: usize) -> Vec<Vec<f64>> {
    let tt = sp[0].len();
    let nv = sp[0][0].len();
    let re = |x: [f64; 2], y: [f64; 2]| x[0] * y[0] + x[1] * y[1];
    paires(sp.len())
        .into_iter()
        .map(|(a, b)| {
            (0..=tau_max)
                .map(|tau| {
                    let mut acc = 0.0;
                    for t in 0..tt {
                        let t2 = (t + tau) % tt;
                        for iv in 0..nv {
                            acc += re(sp[a][t][iv], sp[b][t2][iv]) + re(sp[b][t][iv], sp[a][t2][iv]);
                        }
                    }
                    acc / (2.0 * tt as f64 * nv as f64)
                })
                .collect()
        })
        .collect()
}

/// Corrélateurs à impulsion de séries de Fourier sp[op][t][d] = [Re, Im] (moyenne sur les 3 axes, temps périodique, symétrisés).
#[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
pub fn correlateurs_p(sp: &[Vec<[[f64; 2]; 3]>], tau_max: usize) -> Vec<Vec<f64>> {
    let tt = sp[0].len();
    let re = |x: [f64; 2], y: [f64; 2]| x[0] * y[0] + x[1] * y[1]; // Re[conj(x) y]
    paires(sp.len())
        .into_iter()
        .map(|(a, b)| {
            (0..=tau_max)
                .map(|tau| {
                    let mut acc = 0.0;
                    for t in 0..tt {
                        let t2 = (t + tau) % tt;
                        for d in 0..3 {
                            acc += re(sp[a][t][d], sp[b][t2][d]) + re(sp[b][t][d], sp[a][t2][d]);
                        }
                    }
                    acc / (2.0 * tt as f64 * 3.0)
                })
                .collect()
        })
        .collect()
}

/// Corrélateurs par cran demandé : `correlateurs_ps(sps, τ)[ik]` = `correlateurs_p(sps[ik], τ)`.
pub fn correlateurs_ps(sps: &[Vec<Vec<[[f64; 2]; 3]>>], tau_max: usize) -> Vec<Vec<Vec<f64>>> {
    sps.iter().map(|sp| correlateurs_p(sp, tau_max)).collect()
}

pub fn paires(n: usize) -> Vec<(usize, usize)> {
    super::paires(n)
}

/// Corrélateurs symétrisés de séries S[op][t] (temps périodique).
pub fn correlateurs(s: &[Vec<f64>], tau_max: usize) -> Vec<Vec<f64>> {
    let tt = s[0].len();
    paires(s.len())
        .into_iter()
        .map(|(a, b)| {
            (0..=tau_max)
                .map(|tau| {
                    let mut acc = 0.0;
                    for t in 0..tt {
                        let t2 = (t + tau) % tt;
                        acc += s[a][t] * s[b][t2] + s[b][t] * s[a][t2];
                    }
                    acc / (2.0 * tt as f64)
                })
                .collect()
        })
        .collect()
}

/// Contrôle d'un cran d'impulsion `k` (en unités de 2π/ℓ, signé) mesurable sur cette géométrie : |k| ≥ 1 (le cran 0 est
/// l'impulsion nulle, déjà mesurée dans `corr`) et |k| < ℓ_d/2 le long de CHACUN des trois axes spatiaux — au delà une
/// impulsion se confond avec ℓ_d − k (même phase sur les sites du réseau) et la moyenne sur les axes n'a plus de sens quand
/// les côtés diffèrent. Le signe est sans effet sur le corrélateur (C(−k) = C(k)), il est juste accepté pour la symétrie.
pub fn controler_cran(g: &GeoA, temps: usize, k: i64) -> Resultat<()> {
    if k == 0 {
        return Err(Erreur::Config(
            "impulsion : cran 0 refusé (l'impulsion nulle est déjà mesurée, voir corr / neutre.bin)".into(),
        ));
    }
    let ak = (k as i128).abs();
    for d in axes_spatiaux(temps) {
        let l = g.d[d] as i128;
        if 2 * ak >= l {
            let max = (l - 1) / 2;
            let admis = if max >= 1 { format!("1 à {max}") } else { "aucun (côté spatial trop petit)".to_string() };
            return Err(Erreur::Config(format!(
                "impulsion : cran {k} refusé (il faut |k| < ℓ/2, or ℓ = {} le long de l'axe {d}) ; crans admis : {admis}{}",
                g.d[d],
                if k < 0 {
                    " — le signe ne change rien : C(−k) = C(k)"
                } else {
                    ""
                }
            )));
        }
    }
    Ok(())
}

/// Contrôle d'une liste de crans (voir [`controler_cran`]) : le premier cran hors domaine fait échouer la mesure.
pub fn controler_ks(g: &GeoA, temps: usize, ks: &[i64]) -> Resultat<()> {
    for &k in ks {
        controler_cran(g, temps, k)?;
    }
    Ok(())
}

pub fn mesurer(g: &GeoA, u: &[Q], p: &ParamNeutre) -> Resultat<MesureNeutre> {
    mesurer_gen(g, u, p, &[], &[])
}

/// Entrée historique de l'option `--impulsion` (un seul cran, 2π/ℓ_d) : `mesurer_ps(g, u, p, &[1])`, mêmes bits. Le CLI
/// passe encore par elle quand la liste des crans se réduit à `{1}`, c'est-à-dire pour `--impulsion` et `--impulsions 1`.
pub fn mesurer_p(g: &GeoA, u: &[Q], p: &ParamNeutre) -> Resultat<MesureNeutre> {
    mesurer_ps(g, u, p, &[1])
}

/// Comme `mesurer`, avec les corrélateurs à impulsion non nulle de chacun des crans demandés : `corr_ps[ik]` (et `corr_p`
/// = `corr_ps[0]`, le premier cran de la liste). Crans signés acceptés (C(−k) = C(k)) ; un cran nul ou ≥ ℓ/2 est refusé
/// ([`controler_cran`]). Un cran donné rend les mêmes bits que s'il était mesuré seul.
pub fn mesurer_ps(g: &GeoA, u: &[Q], p: &ParamNeutre, ks: &[i64]) -> Resultat<MesureNeutre> {
    controler_ks(g, p.temps, ks)?;
    mesurer_gen(g, u, p, ks, &[])
}

/// Comme `mesurer_ps`, avec en plus les corrélateurs moyennés sur chaque orbite cubique de `orbs` (impulsions hors axe, voir
/// [`Orbite`], [`controler_vecteurs`]) : `corr_orb[io]`. Sans orbite, identique à `mesurer_ps` au bit près.
pub fn mesurer_orb(g: &GeoA, u: &[Q], p: &ParamNeutre, ks: &[i64], orbs: &[Orbite]) -> Resultat<MesureNeutre> {
    controler_ks(g, p.temps, ks)?;
    for o in orbs {
        controler_vecteur(g, p.temps, o.representant)?;
    }
    mesurer_gen(g, u, p, ks, orbs)
}

fn mesurer_gen(g: &GeoA, u: &[Q], p: &ParamNeutre, ks: &[i64], orbs: &[Orbite]) -> Resultat<MesureNeutre> {
    if p.niveaux.is_empty() || p.niveaux.windows(2).any(|w| w[0] >= w[1]) || p.formes.is_empty() {
        return Err(Erreur::Config("niveaux croissants non vides et au moins une forme attendus".into()));
    }
    if p.tau_max >= g.d[p.temps] {
        return Err(Erreur::Config("tau_max doit être < ℓ_temps".into()));
    }
    let imgs: Vec<Vec<Vec<i8>>> = p
        .formes
        .iter()
        .map(|&f| {
            let mut v = images(FORMES[f].1);
            if p.mutant == MutantN::ImageOubliee && v.len() > 1 {
                v.pop();
            }
            v
        })
        .collect();
    let faux = p.mutant == MutantN::LissageFaux;
    let mut courant = u.to_vec();
    let mut fait = 0usize;
    let mut s: Vec<Vec<f64>> = Vec::new();
    let mut sps: Vec<Vec<Vec<[[f64; 2]; 3]>>> = vec![Vec::new(); ks.len()];
    // impulsions hors axe : tous les vecteurs de toutes les orbites, à plat ; sv[op][t][iv]
    let vs: Vec<[i64; 3]> = orbs.iter().flat_map(|o| o.vecteurs.iter().copied()).collect();
    let mut sv: Vec<Vec<Vec<[f64; 2]>>> = Vec::new();
    for &n in &p.niveaux {
        for _ in fait..n {
            courant = ape_iteration_gen(g, &courant, p.temps, p.alpha, faux);
        }
        fait = n;
        for im in &imgs {
            if ks.is_empty() && vs.is_empty() {
                s.push(tranches(g, &courant, p.temps, im));
            } else {
                if !ks.is_empty() {
                    let (a, b) = tranches_ps(g, &courant, p.temps, im, ks);
                    s.push(a);
                    for (ik, v) in b.into_iter().enumerate() {
                        sps[ik].push(v);
                    }
                }
                if !vs.is_empty() {
                    let (a, b) = tranches_vec(g, &courant, p.temps, im, &vs);
                    if ks.is_empty() {
                        s.push(a);
                    }
                    sv.push(b);
                }
            }
        }
    }
    let corr = correlateurs(&s, p.tau_max);
    let tt = g.d[p.temps] as f64;
    let moy = s.iter().map(|x| x.iter().sum::<f64>() / tt).collect();
    let corr_ps = correlateurs_ps(&sps, p.tau_max);
    let corr_p = corr_ps.first().cloned().unwrap_or_default();
    let mut corr_orb: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut deb = 0usize;
    for o in orbs {
        let fin = deb + o.vecteurs.len();
        let sp: Vec<Vec<Vec<[f64; 2]>>> = sv.iter().map(|op| op.iter().map(|tv| tv[deb..fin].to_vec()).collect()).collect();
        corr_orb.push(correlateurs_orbite(&sp, p.tau_max));
        deb = fin;
    }
    Ok(MesureNeutre { s, corr, moy, corr_p, corr_ps, corr_orb })
}
