//! Module « intrication » : entropie de Rényi-2 par la méthode des répliques, SU(2) Wilson 4D, CPU F64 (rayon).
//! Propriétaire : module intrication (11/11/2026). Aucun noyau GPU dans cette version.
//!
//! GÉOMÉTRIE (spécification de la note T14, §1 : `flux_tube_2026-11-11/sol/T14_astra_critere_entropique.reponse.md`) :
//! - s répliques (s = 2) d'un réseau L³ × T, direction 0 = temps (coordonnée la plus lente, comme `heritage::su2_hb`),
//!   site s = r·V₁ + ((t·L + x1)·L + x2)·L + x3, lien li = mu·V + s (V = s·V₁) ;
//! - la COUPURE est entre la tranche t = T−1 (face supérieure du cylindre r) et la tranche t = 0 (face inférieure) ;
//! - chaque LIEN SPATIAL est attribué à A ou à Ā (ici : A = liens dont les deux extrémités sont dans la tranche
//!   x1 ∈ [0, l)) ; la face supérieure de la réplique r porte les liens de A de la réplique σ(r) = r+1 mod s et les liens
//!   de Ā de la réplique r elle-même ;
//! - plaquettes temporelles de la coupure : U_0(r,T−1,x) · U_ν(π(r),0,x) · U_0(r,T−1,x+ν)† · U_ν(r,T−1,x)†, π = σ sur A ;
//! - plaquettes spatiales de la couture (t = 0) : convention de transfert SYMÉTRIQUE, poids ½ sur la face inférieure
//!   (liens de la réplique r) et ½ sur la face supérieure (liens recousus) ; quand les deux boucles coïncident (plaquette
//!   entièrement dans A ou dans Ā) elles fusionnent en une plaquette de poids 1 ;
//! - jauge : arbitraire partout, sauf aux sommets de branchement de la couture (sites portant des liens spatiaux de A et
//!   de Ā), où g(r,0,x) est le même pour toutes les répliques (identification des extrémités recousues).
//!
//! Avec ces conventions, Z₂(A)/Z² = Tr ρ_A² (note T14, §1) ; A vide : deux réseaux indépendants ; A = toute la tranche :
//! un réseau L³ × 2T exactement.
//!
//! INTERPOLATION : un `Reseau` porte DEUX géométries (régions A₀ et A₁) sur le MÊME ensemble de liens ; chaque plaquette
//! a un poids w₀ (dans S₀) et w₁ (dans S₁), et l'action simulée est S_α = (1−α)S₀ + αS₁. Alors
//! F₁ − F₀ = ∫₀¹ dα ⟨S₁ − S₀⟩_α (Buividovich–Polikarpov 2008, éq. B1-B3).
//!
//! BAIN THERMIQUE et SURRELAXATION : mêmes primitives et mêmes expressions que `heritage::su2_hb` (Kennedy-Pendleton /
//! Creutz exacts, Philox4x32-10 de compteur (lien local, sweep, tirage, étiquette), clé par réplique). Damier
//! 4 directions × 2 parités : dans une plaquette, deux liens de même direction sont toujours de parités opposées
//! (T et L pairs) ; la porte `porte_classes` le vérifie sur la liste de plaquettes elle-même.
//! L'agrafe d'un lien est la somme, dans le même ordre que `su2_hb::staple` (ν croissant, haut puis bas), des contributions
//! de ses plaquettes, pondérées par leur coefficient : A vide et T = L ⟹ bit à bit égal à `CpuSu2` (porte a).

use crate::heritage::su2_hb::{self, Q};
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;

pub mod cli;
pub mod portes;

// Encodages et capacités C2 : ceux du socle `geom` (contrat de `geom::tables`). `MAXCASE` = 4 entrées par case
// (lien, autre direction, haut/bas) : 2 boucles de couture × 2 géométries ; `NUL` = case vide.
pub use crate::geom::tables::{LIMITE_LIENS_C2, LIMITE_PLAQUETTES_C2, NUL};
use crate::geom::{Bord, Chemin, GeoSpec, RegionSpec};
pub use crate::geom::{Plaq, MAXCASE};

/// Variantes volontairement fausses (contrôles négatifs des portes). Jamais en production.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Mutant {
    #[default]
    Aucun,
    /// L'agrafe de coupure prend le lien spatial de la tranche 0 dans la MÊME réplique, même sur A (raccord manqué).
    AgrafeCoupure,
    /// Le bain thermique et la surrelaxation ignorent les poids ½ de la couture (l'action, elle, les garde).
    PoidsIgnores,
}

/// Géométrie à s répliques de L³ × T.
#[derive(Clone, Copy, Debug)]
pub struct GeoRep {
    pub l: usize,
    pub t: usize,
    pub s: usize,
    pub l3: usize,
    pub v1: usize,
    pub v: usize,
}

impl GeoRep {
    pub fn nouveau(l: usize, t: usize, s: usize) -> Result<GeoRep, String> {
        if l < 2 || !l.is_multiple_of(2) || t < 2 || !t.is_multiple_of(2) || s < 1 {
            return Err(format!(
                "L et T pairs ≥ 2, s ≥ 1 requis (L = {l}, T = {t}, s = {s})"
            ));
        }
        // relecture, objection 9 : volume calculé sans débordement (L³ débordait en silence pour L = 2²²)
        let l3 = l.checked_mul(l).and_then(|x| x.checked_mul(l));
        let v1 = l3.and_then(|x| x.checked_mul(t));
        match (l3, v1, v1.and_then(|x| x.checked_mul(s))) {
            (Some(l3), Some(v1), Some(v)) => Ok(GeoRep { l, t, s, l3, v1, v }),
            _ => Err(format!(
                "volume L³ × T × s hors limites (L = {l}, T = {t}, s = {s})"
            )),
        }
    }
    /// Variante SANS contrôle de parité (mutant de la porte des classes : T impair).
    pub fn nouveau_sans_controle(l: usize, t: usize, s: usize) -> GeoRep {
        let l3 = l * l * l;
        GeoRep {
            l,
            t,
            s,
            l3,
            v1: l3 * t,
            v: s * l3 * t,
        }
    }
    #[inline]
    pub fn site(&self, r: usize, t: usize, xs: usize) -> usize {
        r * self.v1 + t * self.l3 + xs
    }
    /// (r, t, xs) d'un site.
    #[inline]
    pub fn decomp(&self, s: usize) -> (usize, usize, usize) {
        let r = s / self.v1;
        let q = s % self.v1;
        (r, q / self.l3, q % self.l3)
    }
    #[inline]
    pub fn xs_coords(&self, xs: usize) -> [usize; 3] {
        let l = self.l;
        [xs / (l * l), (xs / l) % l, xs % l]
    }
    /// Voisin spatial (d ∈ 1..=3) vers le haut, périodique.
    #[inline]
    pub fn xs_haut(&self, xs: usize, d: usize) -> usize {
        let mut c = self.xs_coords(xs);
        c[d - 1] = (c[d - 1] + 1) % self.l;
        (c[0] * self.l + c[1]) * self.l + c[2]
    }
    #[inline]
    pub fn lien(&self, mu: usize, s: usize) -> usize {
        mu * self.v + s
    }
    /// Parité d'un site (t + x1 + x2 + x3), indépendante de la réplique.
    #[inline]
    pub fn parite(&self, s: usize) -> usize {
        let (_, t, xs) = self.decomp(s);
        let c = self.xs_coords(xs);
        (t + c[0] + c[1] + c[2]) & 1
    }
}

/// Région A : attribution de chaque lien spatial (xs, d), d ∈ 1..=3, à A (vrai) ou Ā (faux).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub dans_a: Vec<[bool; 4]>,
}

impl Region {
    /// Tranche x1 ∈ [0, largeur) (toute l'étendue de x2, x3) : un lien est dans A si ses deux extrémités y sont.
    pub fn tranche(g: &GeoRep, largeur: usize) -> Region {
        Region::tranche_axe(g, 1, largeur)
    }

    /// Tranche x_axe ∈ [0, largeur), axe ∈ 1..=3 (`geom::RegionSpec::Tranche`) ; axe 1 = `tranche`.
    pub fn tranche_axe(g: &GeoRep, axe: usize, largeur: usize) -> Region {
        assert!((1..=3).contains(&axe), "axe de tranche dans 1..=3");
        let dedans = |xs: usize| g.xs_coords(xs)[axe - 1] < largeur;
        let dans_a = (0..g.l3)
            .map(|xs| {
                let mut f = [false; 4];
                for (d, fd) in f.iter_mut().enumerate().skip(1) {
                    *fd = dedans(xs) && dedans(g.xs_haut(xs, d));
                }
                f
            })
            .collect();
        Region { dans_a }
    }
    #[inline]
    pub fn a(&self, xs: usize, d: usize) -> bool {
        self.dans_a[xs][d]
    }
}

#[inline]
fn ent(li: usize, dague: bool) -> u32 {
    ((li as u32) << 1) | (dague as u32)
}

/// Plaquettes brutes d'une géométrie (avant fusion), dans l'ordre (site, plan).
fn plaquettes_brutes(g: &GeoRep, reg: &Region, mutant: Mutant) -> Vec<([u32; 4], u8, u8, f64)> {
    let sig = |r: usize| (r + 1) % g.s;
    let mut out = Vec::with_capacity(6 * g.v + 3 * g.s * g.l3);
    for r in 0..g.s {
        for t in 0..g.t {
            for xs in 0..g.l3 {
                let s0 = g.site(r, t, xs);
                for mu in 0..4 {
                    for nu in mu + 1..4 {
                        if mu == 0 {
                            let xn = g.xs_haut(xs, nu);
                            if t + 1 < g.t {
                                let e = [
                                    ent(g.lien(0, s0), false),
                                    ent(g.lien(nu, g.site(r, t + 1, xs)), false),
                                    ent(g.lien(0, g.site(r, t, xn)), true),
                                    ent(g.lien(nu, s0), true),
                                ];
                                out.push((e, 0, nu as u8, 1.0));
                            } else {
                                // plaquette temporelle de la coupure : lien spatial de la face supérieure
                                let rr = if reg.a(xs, nu) && mutant != Mutant::AgrafeCoupure {
                                    sig(r)
                                } else {
                                    r
                                };
                                let e = [
                                    ent(g.lien(0, s0), false),
                                    ent(g.lien(nu, g.site(rr, 0, xs)), false),
                                    ent(g.lien(0, g.site(r, t, xn)), true),
                                    ent(g.lien(nu, s0), true),
                                ];
                                out.push((e, 0, nu as u8, 1.0));
                            }
                        } else {
                            let xm = g.xs_haut(xs, mu);
                            let xn = g.xs_haut(xs, nu);
                            let reg_e = |rmu: usize, rnu1: usize, rmu2: usize, rnu2: usize| {
                                [
                                    ent(g.lien(mu, g.site(rmu, t, xs)), false),
                                    ent(g.lien(nu, g.site(rnu1, t, xm)), false),
                                    ent(g.lien(mu, g.site(rmu2, t, xn)), true),
                                    ent(g.lien(nu, g.site(rnu2, t, xs)), true),
                                ]
                            };
                            if t != 0 {
                                out.push((reg_e(r, r, r, r), mu as u8, nu as u8, 1.0));
                            } else {
                                // couture : ½ face inférieure (réplique r), ½ face supérieure de la réplique r
                                out.push((reg_e(r, r, r, r), mu as u8, nu as u8, 0.5));
                                let p = |x: usize, d: usize| if reg.a(x, d) { sig(r) } else { r };
                                out.push((
                                    reg_e(p(xs, mu), p(xm, nu), p(xn, mu), p(xs, nu)),
                                    mu as u8,
                                    nu as u8,
                                    0.5,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Clé canonique d'une boucle : entrées triées.
fn cle(e: &[u32; 4]) -> [u32; 4] {
    let mut k = *e;
    k.sort_unstable();
    k
}

/// Réseau à répliques avec interpolation entre deux régions, champ et état du générateur.
pub struct Reseau {
    pub geo: GeoRep,
    pub reg0: Region,
    pub reg1: Region,
    pub plaqs: Vec<Plaq>,
    /// Plaquettes dont w0 ≠ w1 (support de S₁ − S₀).
    pub diff: Vec<u32>,
    /// Appartenances : 6 cases par lien (autre direction × haut/bas), MAXCASE entrées (id << 2 | position).
    pub cases: Vec<[u32; MAXCASE]>,
    pub coef: Vec<f64>,
    pub alpha: f64,
    pub beta: f64,
    pub u: Vec<Q>,
    /// Flux d'aléa de chaque réplique (clé et mots fixes du compteur, voir `Flux`).
    pub flux: Vec<Flux>,
    pub sweep_id: u32,
    pub n_cap: u64,
    pub mutant: Mutant,
    classes: Vec<Vec<u32>>,
}

/// Position de `o` parmi les directions ≠ d (0..3).
#[inline]
fn idx_autre(d: usize, o: usize) -> usize {
    if o < d {
        o
    } else {
        o - 1
    }
}

impl Reseau {
    /// Réseau entre les régions reg0 (α = 0) et reg1 (α = 1), flux HISTORIQUES : `graines[r]` = clé de la réplique r,
    /// compteurs de `heritage::su2_hb::CpuSu2` (réservé aux portes (a) à (d), graines fixes ; voir `Flux::historique`).
    pub fn nouveau(
        geo: GeoRep,
        reg0: Region,
        reg1: Region,
        graines: &[u64],
        mutant: Mutant,
    ) -> Result<Reseau, String> {
        let flux: Vec<Flux> = graines.iter().map(|&g| Flux::historique(g)).collect();
        Reseau::avec_flux(geo, reg0, reg1, &flux, mutant)
    }

    /// Réseau entre les régions reg0 et reg1 avec un flux d'aléa par réplique. Refuse deux répliques de même flux.
    pub fn avec_flux(
        geo: GeoRep,
        reg0: Region,
        reg1: Region,
        flux: &[Flux],
        mutant: Mutant,
    ) -> Result<Reseau, String> {
        if flux.len() != geo.s {
            return Err(format!("{} flux pour {} répliques", flux.len(), geo.s));
        }
        for (i, f) in flux.iter().enumerate() {
            if flux[..i].contains(f) {
                return Err(format!(
                    "flux d'aléa identiques pour les répliques {} et {i}",
                    flux[..i].iter().position(|g| g == f).unwrap_or(0)
                ));
            }
        }
        // Capacités C2 (geom::tables) : entrées (li << 1) | dague sur 32 bits, AVANT de les encoder.
        if geo.v.checked_mul(4).is_none_or(|n| n > LIMITE_LIENS_C2) {
            return Err(format!(
                "capacité C2 : 4·V = 4 × {} liens, au plus {LIMITE_LIENS_C2}",
                geo.v
            ));
        }
        let b0 = plaquettes_brutes(&geo, &reg0, mutant);
        let b1 = plaquettes_brutes(&geo, &reg1, mutant);
        let mut index: HashMap<[u32; 4], usize> = HashMap::with_capacity(b0.len() + b0.len() / 8);
        let mut plaqs: Vec<Plaq> = Vec::with_capacity(b0.len() + b0.len() / 8);
        for (geom, brut) in [(0, &b0), (1, &b1)] {
            for &(e, mu, nu, w) in brut.iter() {
                let k = cle(&e);
                let i = *index.entry(k).or_insert_with(|| {
                    plaqs.push(Plaq {
                        e,
                        mu,
                        nu,
                        w0: 0.0,
                        w1: 0.0,
                    });
                    plaqs.len() - 1
                });
                if plaqs[i].e != e {
                    return Err(format!("boucle {k:?} rencontrée avec deux orientations"));
                }
                if geom == 0 {
                    plaqs[i].w0 += w;
                } else {
                    plaqs[i].w1 += w;
                }
            }
        }
        // N1 (relecture du socle) : nombre RÉEL de plaquettes de l'union, avant l'encodage (id << 2) | position.
        if plaqs.len() > LIMITE_PLAQUETTES_C2 {
            return Err(format!(
                "capacité C2 : {} plaquettes, au plus {LIMITE_PLAQUETTES_C2}",
                plaqs.len()
            ));
        }
        let n_liens = 4 * geo.v;
        let mut cases = vec![[NUL; MAXCASE]; 6 * n_liens];
        for (id, p) in plaqs.iter().enumerate() {
            let (mu, nu) = (p.mu as usize, p.nu as usize);
            for pos in 0..4 {
                let li = (p.e[pos] >> 1) as usize;
                // pos 0 : lien mu, autre nu, haut ; 1 : lien nu, autre mu, bas ; 2 : lien mu, autre nu, bas ; 3 : lien nu, autre mu, haut
                let (d, o, bas) = match pos {
                    0 => (mu, nu, 0),
                    1 => (nu, mu, 1),
                    2 => (mu, nu, 1),
                    _ => (nu, mu, 0),
                };
                debug_assert_eq!(li / geo.v, d);
                let c = &mut cases[6 * li + 2 * idx_autre(d, o) + bas];
                let libre = c
                    .iter()
                    .position(|&m| m == NUL)
                    .ok_or_else(|| format!("case pleine (lien {li})"))?;
                c[libre] = ((id as u32) << 2) | pos as u32;
            }
        }
        if let Some(i) = cases.iter().position(|c| c[0] == NUL) {
            return Err(format!("case vide : lien {}, case {}", i / 6, i % 6));
        }
        let diff = plaqs
            .iter()
            .enumerate()
            .filter(|(_, p)| p.w0 != p.w1)
            .map(|(i, _)| i as u32)
            .collect();
        let mut classes = vec![Vec::new(); 8];
        for mu in 0..4 {
            for s in 0..geo.v {
                classes[2 * mu + geo.parite(s)].push(geo.lien(mu, s) as u32);
            }
        }
        let mut r = Reseau {
            geo,
            reg0,
            reg1,
            coef: vec![0.0; plaqs.len()],
            plaqs,
            diff,
            cases,
            alpha: 0.0,
            beta: 0.0,
            u: vec![[1.0, 0.0, 0.0, 0.0]; n_liens],
            flux: flux.to_vec(),
            sweep_id: 0,
            n_cap: 0,
            mutant,
            classes,
        };
        r.fixer_alpha(0.0);
        Ok(r)
    }

    /// Réseau d'interpolation entre deux `GeoSpec` recollées (A₀ à α = 0, A₁ à α = 1). Refuse tout ce qui n'est pas
    /// `Recolle` sur la direction 0 (= `dir_temps`), avec ℓ₁ = ℓ₂ = ℓ₃ pairs, T pair, mêmes côtés, même nombre de répliques
    /// et même axe de tranche dans les deux spécifications (code 2 par `Erreur::Config` côté CLI).
    pub fn depuis_spec(
        spec0: &GeoSpec,
        spec1: &GeoSpec,
        graines: &[u64],
        mutant: Mutant,
    ) -> Result<Reseau, String> {
        let (geo, r0, r1) = geo_depuis_specs(spec0, spec1)?;
        Reseau::nouveau(geo, r0, r1, graines, mutant)
    }

    /// `depuis_spec` avec des flux explicites (chaînes de production : `Flux::domaine`).
    pub fn depuis_spec_flux(
        spec0: &GeoSpec,
        spec1: &GeoSpec,
        flux: &[Flux],
        mutant: Mutant,
    ) -> Result<Reseau, String> {
        let (geo, r0, r1) = geo_depuis_specs(spec0, spec1)?;
        Reseau::avec_flux(geo, r0, r1, flux, mutant)
    }

    /// Géométrie pure (une seule région) : S_α = S pour tout α.
    pub fn pur(
        geo: GeoRep,
        reg: Region,
        graines: &[u64],
        mutant: Mutant,
    ) -> Result<Reseau, String> {
        Reseau::nouveau(geo, reg.clone(), reg, graines, mutant)
    }

    pub fn fixer_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
        for (c, p) in self.coef.iter_mut().zip(&self.plaqs) {
            *c = if p.w0 == p.w1 {
                p.w0
            } else {
                (1.0 - alpha) * p.w0 + alpha * p.w1
            };
        }
    }

    /// Classes de mise à jour simultanée (mu, parité).
    pub fn classes(&self) -> &[Vec<u32>] {
        &self.classes
    }

    /// Départ chaud (Haar, compteur [lien local, w1_init, 0|1, w3_init] du flux de la réplique) ou froid.
    pub fn init(&mut self, chaud: bool) {
        let g = self.geo;
        let flux = self.flux.clone();
        self.u.par_iter_mut().enumerate().for_each(|(li, q)| {
            if !chaud {
                *q = [1.0, 0.0, 0.0, 0.0];
                return;
            }
            let (loc, r) = local(&g, li);
            let f = flux[r];
            let r1 = philox_flux(&f, [loc, f.w1_init, 0, f.w3_init]);
            let r2 = philox_flux(&f, [loc, f.w1_init, 1, f.w3_init]);
            *q = haar_from(r1, r2);
        });
        self.sweep_id = 0;
        self.n_cap = 0;
    }

    #[inline]
    fn facteur(&self, e: u32, inverse: bool) -> Q {
        let q = self.u[(e >> 1) as usize];
        if ((e & 1) == 1) ^ inverse {
            su2_hb::qdag(q)
        } else {
            q
        }
    }

    /// Contribution d'une appartenance (id << 2 | pos) à l'agrafe, avec son coefficient.
    #[inline]
    fn contribution(&self, m: u32, avec_coef: bool) -> Q {
        let id = (m >> 2) as usize;
        let pos = (m & 3) as usize;
        let e = &self.plaqs[id].e;
        let prod = if e[pos] & 1 == 0 {
            su2_hb::qmul(
                su2_hb::qmul(
                    self.facteur(e[(pos + 1) & 3], false),
                    self.facteur(e[(pos + 2) & 3], false),
                ),
                self.facteur(e[(pos + 3) & 3], false),
            )
        } else {
            su2_hb::qmul(
                su2_hb::qmul(
                    self.facteur(e[(pos + 3) & 3], true),
                    self.facteur(e[(pos + 2) & 3], true),
                ),
                self.facteur(e[(pos + 1) & 3], true),
            )
        };
        let c = self.coef[id];
        if c == 1.0 || !avec_coef {
            prod
        } else {
            su2_hb::qscale(prod, c)
        }
    }

    #[inline]
    fn case(&self, li: usize, k: usize, avec_coef: bool) -> Q {
        let c = &self.cases[6 * li + k];
        let mut a = self.contribution(c[0], avec_coef);
        for &m in &c[1..] {
            if m == NUL {
                break;
            }
            a = su2_hb::qadd(a, self.contribution(m, avec_coef));
        }
        a
    }

    /// Agrafe pondérée A du lien li : partie locale de S_α = −β ReTr(U_li A)/2 = −β (U·A)[0].
    pub fn agrafe(&self, li: usize) -> Q {
        self.agrafe_avec(li, true)
    }

    fn agrafe_avec(&self, li: usize, avec_coef: bool) -> Q {
        let mut a: Q = [0.0; 4];
        for o in 0..3 {
            a = su2_hb::qadd(
                a,
                su2_hb::qadd(
                    self.case(li, 2 * o, avec_coef),
                    self.case(li, 2 * o + 1, avec_coef),
                ),
            );
        }
        a
    }

    /// Agrafe utilisée par le générateur (le mutant `PoidsIgnores` y oublie les coefficients).
    #[inline]
    fn agrafe_gen(&self, li: usize) -> Q {
        self.agrafe_avec(li, self.mutant != Mutant::PoidsIgnores)
    }

    /// ½ Tr de la boucle de la plaquette id : ((e0 e1) e2) e3.
    #[inline]
    pub fn valeur(&self, id: usize) -> f64 {
        let e = &self.plaqs[id].e;
        let w = su2_hb::qmul(
            su2_hb::qmul(
                su2_hb::qmul(self.facteur(e[0], false), self.facteur(e[1], false)),
                self.facteur(e[2], false),
            ),
            self.facteur(e[3], false),
        );
        w[0]
    }

    /// Action S_α = β Σ_p c_p (1 − ½Tr U_p) (somme dans l'ordre de la liste).
    pub fn action(&self) -> f64 {
        let v: Vec<f64> = (0..self.plaqs.len())
            .into_par_iter()
            .map(|i| self.coef[i] * (1.0 - self.valeur(i)))
            .collect();
        self.beta * v.iter().sum::<f64>()
    }

    /// Action d'une géométrie pure (0 ou 1) : β Σ_p w_p (1 − ½Tr U_p).
    pub fn action_geom(&self, geom: usize) -> f64 {
        let v: Vec<f64> = (0..self.plaqs.len())
            .into_par_iter()
            .map(|i| {
                let w = if geom == 0 {
                    self.plaqs[i].w0
                } else {
                    self.plaqs[i].w1
                };
                if w == 0.0 {
                    0.0
                } else {
                    w * (1.0 - self.valeur(i))
                }
            })
            .collect();
        self.beta * v.iter().sum::<f64>()
    }

    /// S₁ − S₀ = β Σ_{p : w0≠w1} (w1 − w0)(1 − ½Tr U_p) : l'intégrande de la méthode d'interpolation.
    pub fn delta_s(&self) -> f64 {
        let mut acc = 0.0;
        for &i in &self.diff {
            let p = &self.plaqs[i as usize];
            acc += (p.w1 - p.w0) * (1.0 - self.valeur(i as usize));
        }
        self.beta * acc
    }

    /// Σ_p c_p ½Tr U_p (sert à l'intégration thermodynamique en β) et Σ_p c_p.
    pub fn somme_plaquettes(&self) -> (f64, f64) {
        let v: Vec<f64> = (0..self.plaqs.len())
            .into_par_iter()
            .map(|i| self.coef[i] * self.valeur(i))
            .collect();
        (v.iter().sum(), self.coef.iter().sum())
    }

    /// Plaquette moyenne (pondérée) de la réplique r (plaquettes dont le premier lien est dans r).
    pub fn plaquette_replique(&self, r: usize) -> f64 {
        let g = self.geo;
        let (mut a, mut n) = (0.0, 0.0);
        for (i, p) in self.plaqs.iter().enumerate() {
            let li = (p.e[0] >> 1) as usize;
            if (li % g.v) / g.v1 == r {
                a += self.coef[i] * self.valeur(i);
                n += self.coef[i];
            }
        }
        a / n
    }

    fn hb_lien(&self, li: usize) -> (usize, Option<Q>, u64) {
        let (loc, r) = local(&self.geo, li);
        let f = self.flux[r];
        let sid = self.sweep_id;
        let a = self.agrafe_gen(li);
        let k = su2_hb::qnorm(a);
        let mut draw: u32 = 0;
        if k.partial_cmp(&1e-300) != Some(std::cmp::Ordering::Greater) {
            let r1 = tirage(&f, loc, sid, &mut draw);
            let r2 = tirage(&f, loc, sid, &mut draw);
            return (li, Some(haar_from(r1, r2)), 0);
        }
        let vv = su2_hb::qscale(a, 1.0 / k);
        let alpha = self.beta * k;
        let (x0, _ntry, ok) = tirer_x0(alpha, &f, loc, sid, &mut draw);
        if !ok {
            return (li, None, 1);
        }
        let rd = tirage(&f, loc, sid, &mut draw);
        let ct = 2.0 * su2_hb::u01(rd[0], rd[1]) - 1.0;
        let phi = 2.0 * PI * su2_hb::u01(rd[2], rd[3]);
        let st = fmax0(1.0 - ct * ct).sqrt();
        let rr = fmax0(1.0 - x0 * x0).sqrt();
        let w = [x0, rr * st * phi.cos(), rr * st * phi.sin(), rr * ct];
        let un = su2_hb::qmul(w, su2_hb::qdag(vv));
        (li, Some(su2_hb::qscale(un, 1.0 / su2_hb::qnorm(un))), 0)
    }

    fn or_lien(&self, li: usize) -> Option<Q> {
        let a = self.agrafe_gen(li);
        let k = su2_hb::qnorm(a);
        if k.partial_cmp(&1e-300) != Some(std::cmp::Ordering::Greater) {
            return None;
        }
        let vd = su2_hb::qdag(su2_hb::qscale(a, 1.0 / k));
        let un = su2_hb::qmul(su2_hb::qmul(vd, su2_hb::qdag(self.u[li])), vd);
        Some(su2_hb::qscale(un, 1.0 / su2_hb::qnorm(un)))
    }

    /// Un sweep de bain thermique (8 classes ; lecture puis écriture dans chaque classe).
    pub fn sweep_hb(&mut self) {
        for c in 0..8 {
            let cl = std::mem::take(&mut self.classes[c]);
            let ups: Vec<(usize, Option<Q>, u64)> = if cl.len() >= 4096 {
                cl.par_iter().map(|&li| self.hb_lien(li as usize)).collect()
            } else {
                cl.iter().map(|&li| self.hb_lien(li as usize)).collect()
            };
            self.classes[c] = cl;
            for (li, q, cap) in ups {
                self.n_cap += cap;
                if let Some(q) = q {
                    self.u[li] = q;
                }
            }
        }
        self.sweep_id = self.sweep_id.wrapping_add(1);
    }

    /// Un sweep de surrelaxation U → V† U† V†.
    pub fn sweep_or(&mut self) {
        for c in 0..8 {
            let cl = std::mem::take(&mut self.classes[c]);
            let ups: Vec<(u32, Option<Q>)> = if cl.len() >= 4096 {
                cl.par_iter()
                    .map(|&li| (li, self.or_lien(li as usize)))
                    .collect()
            } else {
                cl.iter()
                    .map(|&li| (li, self.or_lien(li as usize)))
                    .collect()
            };
            self.classes[c] = cl;
            for (li, q) in ups {
                if let Some(q) = q {
                    self.u[li as usize] = q;
                }
            }
        }
    }

    pub fn pas(&mut self, n_or: usize) {
        self.sweep_hb();
        for _ in 0..n_or {
            self.sweep_or();
        }
    }

    /// Champ de la réplique r dans la disposition de `CpuSu2` (li = mu·V₁ + s local).
    pub fn champ_replique(&self, r: usize) -> Vec<Q> {
        let g = self.geo;
        let mut out = vec![[0.0; 4]; 4 * g.v1];
        for mu in 0..4 {
            for sl in 0..g.v1 {
                out[mu * g.v1 + sl] = self.u[g.lien(mu, r * g.v1 + sl)];
            }
        }
        out
    }
}

/// Spécification d'une réplique recollée du socle `geom` : L³ × T, 2 répliques, tranche sur l'axe 1 de largeur `largeur`.
pub fn spec_tranche(l: usize, t: usize, largeur: usize) -> GeoSpec {
    GeoSpec::repliques(l, t, 2, 1, largeur)
}

/// Contrôle d'une `GeoSpec` pour l'intrication : (s répliques, axe, largeur), s = 2 seulement. Voir `Reseau::depuis_spec`.
pub fn controler_spec(spec: &GeoSpec) -> Result<(usize, usize, usize), String> {
    spec.valider().map_err(|e| e.to_string())?;
    if spec.chemin() != Chemin::C2 {
        return Err("intrication : bord recollé (chemin C2) requis".into());
    }
    let (repliques, axe, largeur) = match &spec.bords[0] {
        Bord::Recolle {
            repliques,
            region: RegionSpec::Tranche { axe, largeur },
            ..
        } => (usize::from(*repliques), usize::from(*axe), *largeur),
        _ => return Err("intrication : recollement sur la direction 0 seulement".into()),
    };
    if spec.dir_temps != 0 {
        return Err(format!(
            "intrication : le recollement est sur l'axe 0, dir_temps doit valoir 0 (lu {})",
            spec.dir_temps
        ));
    }
    // Relecture intrication-claude, objection 7 : seules 2 répliques sont couvertes par les portes (b est écrite pour 2T ;
    // avec s = 2, σ = σ⁻¹ et l'orientation de la permutation est invisible). s ≠ 2 est REFUSÉ tant qu'aucune porte ne le couvre.
    if repliques != 2 {
        return Err(format!("intrication : {repliques} répliques demandées ; seules 2 répliques sont validées (Rényi-2)"));
    }
    let c = spec.cotes;
    if c[1] != c[2] || c[2] != c[3] || !c[1].is_multiple_of(2) || !c[0].is_multiple_of(2) {
        return Err(format!(
            "intrication : côtés {c:?} : ℓ₁ = ℓ₂ = ℓ₃ pairs et T pair requis"
        ));
    }
    Ok((repliques, axe, largeur))
}

/// Contrôle de deux spécifications compatibles SANS rien allouer (relecture c19, majeure 6 : la validation précède la garde de
/// mémoire) : (GeoRep, axe, largeur A₀, largeur A₁).
pub fn controler_specs(
    spec0: &GeoSpec,
    spec1: &GeoSpec,
) -> Result<(GeoRep, usize, usize, usize), String> {
    let (s0, axe0, l0) = controler_spec(spec0)?;
    let (s1, axe1, l1) = controler_spec(spec1)?;
    if spec0.cotes != spec1.cotes || s0 != s1 || axe0 != axe1 {
        return Err("intrication : les deux géométries doivent avoir mêmes côtés, mêmes répliques et même axe de tranche".into());
    }
    let geo = GeoRep::nouveau(spec0.cotes[1], spec0.cotes[0], s0)?;
    Ok((geo, axe0, l0, l1))
}

/// (GeoRep, région A₀, région A₁) depuis deux spécifications compatibles.
pub fn geo_depuis_specs(
    spec0: &GeoSpec,
    spec1: &GeoSpec,
) -> Result<(GeoRep, Region, Region), String> {
    let (geo, axe, l0, l1) = controler_specs(spec0, spec1)?;
    Ok((
        geo,
        Region::tranche_axe(&geo, axe, l0),
        Region::tranche_axe(&geo, axe, l1),
    ))
}

/// (compteur local mu·V₁ + s_local, réplique) d'un lien.
#[inline]
fn local(g: &GeoRep, li: usize) -> (u32, usize) {
    let mu = li / g.v;
    let s = li % g.v;
    let r = s / g.v1;
    ((mu * g.v1 + (s - r * g.v1)) as u32, r)
}

/// Journal des couples (clé, compteur) RÉELLEMENT consommés par le générateur du module (relecture c19b, majeure 12) :
/// INACTIF par défaut (une lecture atomique par appel Philox) ; activé seulement par `tests/intrication_flux.rs`, seul dans
/// son processus. Il couvre le départ chaud et chaque tirage du bain thermique, rejets compris ; il ne couvre ni les tirages
/// propres aux portes (`portes::haar_porte` : jauge et liens d'essai) ni ceux de `heritage::su2_hb::CpuSu2` (porte a).
pub mod journal_flux {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    static ACTIF: AtomicBool = AtomicBool::new(false);
    static COUPLES: Mutex<Vec<[u32; 6]>> = Mutex::new(Vec::new());

    /// Vide le journal et l'active.
    pub fn activer() {
        COUPLES.lock().unwrap_or_else(|e| e.into_inner()).clear();
        ACTIF.store(true, Ordering::SeqCst);
    }

    /// Désactive le journal et rend ses couples [k0, k1, c0, c1, c2, c3], dans l'ordre d'enregistrement.
    pub fn prendre() -> Vec<[u32; 6]> {
        ACTIF.store(false, Ordering::SeqCst);
        std::mem::take(&mut *COUPLES.lock().unwrap_or_else(|e| e.into_inner()))
    }

    #[inline]
    pub(crate) fn noter(k0: u32, k1: u32, c: [u32; 4]) {
        if ACTIF.load(Ordering::Relaxed) {
            COUPLES
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push([k0, k1, c[0], c[1], c[2], c[3]]);
        }
    }
}

/// Un appel Philox du générateur du module, consigné au journal s'il est actif.
#[inline]
fn philox_flux(f: &Flux, c: [u32; 4]) -> [u32; 4] {
    journal_flux::noter(f.k0, f.k1, c);
    su2_hb::philox4x32_10(c, f.k0, f.k1)
}

/// Tirage du bain thermique : compteur [lien local, balayage, tirage, w3_hb] (mode historique : w3_hb = TAG_HB, le compteur
/// de `su2_hb`). Le nombre de tirages par (lien, balayage) est borné par le plafond de 10⁶ essais (≤ 2·10⁶ + 3 < 2³²).
#[inline]
fn tirage(f: &Flux, li: u32, sweep_id: u32, draw: &mut u32) -> [u32; 4] {
    let r = philox_flux(f, [li, sweep_id, *draw, f.w3_hb]);
    *draw = draw.wrapping_add(1);
    r
}

/// Plafond d'essais du tirage de x₀ (le même que `su2_hb`).
const CAP_ESSAIS: u32 = 1_000_000;

/// Tirage de x₀ (Kennedy-Pendleton si α ≥ 2, Creutz sinon) : MÊMES expressions que `su2_hb::sample_x0`, compteur du flux
/// (copie locale : `su2_hb` fixe TAG_HB dans le 4e mot). En mode historique, identique au bit à `su2_hb` (porte a).
fn tirer_x0(alpha: f64, f: &Flux, li: u32, sweep_id: u32, draw: &mut u32) -> (f64, u32, bool) {
    let mut x0 = 1.0;
    let mut ok = false;
    let mut ntry: u32 = 0;
    if alpha >= 2.0 {
        while ntry < CAP_ESSAIS {
            let r1 = tirage(f, li, sweep_id, draw);
            let r2 = tirage(f, li, sweep_id, draw);
            let a1 = su2_hb::u01(r1[0], r1[1]);
            let a2 = su2_hb::u01(r1[2], r1[3]);
            let a3 = su2_hb::u01(r2[0], r2[1]);
            let a4 = su2_hb::u01(r2[2], r2[3]);
            let c = (2.0 * PI * a2).cos();
            let lam2 = -(a1.ln() + c * c * a3.ln()) / (2.0 * alpha);
            ntry += 1;
            if a4 * a4 <= 1.0 - lam2 {
                x0 = 1.0 - 2.0 * lam2;
                ok = true;
                break;
            }
        }
    } else {
        let e2 = (-2.0 * alpha).exp();
        while ntry < CAP_ESSAIS {
            let r1 = tirage(f, li, sweep_id, draw);
            let a1 = su2_hb::u01(r1[0], r1[1]);
            let a2 = su2_hb::u01(r1[2], r1[3]);
            let y = 1.0 + (e2 + a1 * (1.0 - e2)).ln() / alpha;
            ntry += 1;
            if a2 * a2 <= 1.0 - y * y {
                x0 = y;
                ok = true;
                break;
            }
        }
    }
    (x0, ntry, ok)
}

// ------------------------------------------------------------------------------------------
// Flux d'aléa (relecture c19, bloquante 1) : domaines séparés PAR CONSTRUCTION
// ------------------------------------------------------------------------------------------

/// Usages (2 bits de poids fort du mot de domaine) : départ chaud et bain thermique.
pub const USAGE_INIT: u32 = 1;
pub const USAGE_HB: u32 = 2;
/// Bornes des champs du mot de domaine : point < 2¹⁵, largeur < 2¹².
pub const POINTS_MAX: usize = (1 << 15) - 1;
pub const LARGEUR_FLUX_MAX: usize = (1 << 12) - 1;

/// Route d'une chaîne (2 bits) : interpolation en α, intégration en β sur la géométrie vide ou pleine, débit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Route {
    Alpha = 0,
    BetaVide = 1,
    BetaPleine = 2,
    Debit = 3,
}

/// Flux d'aléa d'une réplique : clé Philox (k0, k1) et mots fixes du compteur. Départ chaud : [lien local, w1_init, 0|1,
/// w3_init] ; bain thermique : [lien local, balayage, tirage, w3_hb].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Flux {
    pub k0: u32,
    pub k1: u32,
    pub w1_init: u32,
    pub w3_init: u32,
    pub w3_hb: u32,
}

impl Flux {
    /// Mode HISTORIQUE (`heritage::su2_hb::CpuSu2`) : clé = graine de la réplique, compteurs à TAG_INIT / TAG_HB. Réservé aux
    /// portes (a) à (d), à graines fixes ; la porte (a) y vérifie au bit la copie locale du tirage contre `CpuSu2`.
    pub fn historique(graine: u64) -> Flux {
        Flux {
            k0: (graine & 0xFFFF_FFFF) as u32,
            k1: (graine >> 32) as u32,
            w1_init: 0xFFFF_FFFF,
            w3_init: su2_hb::TAG_INIT,
            w3_hb: su2_hb::TAG_HB,
        }
    }

    /// Mode À DOMAINES (toute chaîne de production) : clé = graine de l'utilisateur ; mot 3 = usage (bits 30-31), route
    /// (28-29), réplique (27), point (12-26), largeur (0-11).
    /// INJECTIVITÉ : la graine remplit seule la clé ; usage, route, réplique, point et largeur occupent des bits disjoints du mot
    /// 3, chacun vérifié dans sa borne avant l'empaquetage (ni addition ni XOR) ; lien (< 2³⁰), balayage (< 2³², budget vérifié)
    /// et tirage (< 2³²) remplissent les mots 0 à 2 : deux tirages d'un même run ont donc toujours des (clé, compteur) distincts.
    pub fn domaine(
        graine: u64,
        route: Route,
        replique: usize,
        point: usize,
        largeur: usize,
    ) -> Result<Flux, String> {
        if replique > 1 || point > POINTS_MAX || largeur > LARGEUR_FLUX_MAX {
            return Err(format!(
                "flux : réplique {replique} (≤ 1), point {point} (≤ {POINTS_MAX}), largeur {largeur} (≤ {LARGEUR_FLUX_MAX}) hors champ"
            ));
        }
        let dom = ((route as u32) << 28)
            | ((replique as u32) << 27)
            | ((point as u32) << 12)
            | largeur as u32;
        Ok(Flux {
            k0: (graine & 0xFFFF_FFFF) as u32,
            k1: (graine >> 32) as u32,
            w1_init: 0,
            w3_init: (USAGE_INIT << 30) | dom,
            w3_hb: (USAGE_HB << 30) | dom,
        })
    }

    /// Les deux répliques d'une chaîne à domaines.
    pub fn paire(
        graine: u64,
        route: Route,
        point: usize,
        largeur: usize,
    ) -> Result<[Flux; 2], String> {
        Ok([
            Flux::domaine(graine, route, 0, point, largeur)?,
            Flux::domaine(graine, route, 1, point, largeur)?,
        ])
    }
}

#[inline]
fn fmax0(v: f64) -> f64 {
    if v > 0.0 {
        v
    } else {
        0.0
    }
}

/// Quaternion de Haar depuis deux tirages Philox (copie de `haar_from` de su2_hb, privé là-bas).
pub fn haar_from(r1: [u32; 4], r2: [u32; 4]) -> Q {
    let u1 = su2_hb::u01(r1[0], r1[1]);
    let u2 = su2_hb::u01(r1[2], r1[3]);
    let u3 = su2_hb::u01(r2[0], r2[1]);
    let u4 = su2_hb::u01(r2[2], r2[3]);
    let ra = (-2.0 * u1.ln()).sqrt();
    let rb = (-2.0 * u3.ln()).sqrt();
    let q = [
        ra * (2.0 * PI * u2).cos(),
        ra * (2.0 * PI * u2).sin(),
        rb * (2.0 * PI * u4).cos(),
        rb * (2.0 * PI * u4).sin(),
    ];
    su2_hb::qscale(q, 1.0 / su2_hb::qnorm(q))
}

/// I_n(x) par la série (x ≤ 60).
pub fn bessel_i(n: u32, x: f64) -> f64 {
    let h = x / 2.0;
    let mut terme = h.powi(n as i32) / (1..=n).map(|k| k as f64).product::<f64>();
    let mut somme = terme;
    let h2 = h * h;
    for m in 1..400u32 {
        terme *= h2 / (m as f64 * (m + n) as f64);
        somme += terme;
        if terme < 1e-18 * somme {
            break;
        }
    }
    somme
}

/// ⟨½Tr U⟩ d'une boucle de Haar de poids e^{x ½TrU} : I₂(x)/I₁(x).
pub fn rapport_i2_i1(x: f64) -> f64 {
    if x < 1e-8 {
        return x / 4.0;
    }
    bessel_i(2, x) / bessel_i(1, x)
}

// ------------------------------------------------------------------------------------------
// Statistique : moyenne et erreur par blocs (jackknife sur blocs)
// ------------------------------------------------------------------------------------------

/// (moyenne, erreur) par blocs de taille b (erreur standard des moyennes de blocs).
///
/// Relecture c19 (majeure 4) : moyenne ET erreur portent sur les MÊMES observations, celles des `nb` blocs complets (la ligne
/// de commande exige `--mes` multiple de `--blocs` : aucune observation n'est alors écartée, et la moyenne vaut Σx/n au bit).
/// Moins de deux observations : (NaN, NaN), refusé avant toute clôture par le contrôle des résultats finis.
pub fn moyenne_blocs(x: &[f64], nblocs: usize) -> (f64, f64) {
    let nb = nblocs.min(x.len()).max(2);
    let b = x.len() / nb;
    if b == 0 {
        return (f64::NAN, f64::NAN);
    }
    let x = &x[..nb * b];
    let m = x.iter().sum::<f64>() / x.len() as f64;
    let moy: Vec<f64> = (0..nb)
        .map(|k| x[k * b..(k + 1) * b].iter().sum::<f64>() / b as f64)
        .collect();
    let mm = moy.iter().sum::<f64>() / nb as f64;
    let var =
        moy.iter().map(|v| (v - mm) * (v - mm)).sum::<f64>() / (nb as f64 * (nb as f64 - 1.0));
    (m, var.sqrt())
}

/// Intégrale de Simpson sur une grille équidistante de [0,1] (nombre impair de points) et son erreur
/// (propagation des erreurs indépendantes).
pub fn simpson(y: &[f64], dy: &[f64]) -> (f64, f64) {
    let n = y.len();
    assert!(
        n >= 3 && n % 2 == 1,
        "Simpson : nombre impair de points ≥ 3"
    );
    let h = 1.0 / (n - 1) as f64;
    let w: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 || i == n - 1 {
                h / 3.0
            } else if i % 2 == 1 {
                4.0 * h / 3.0
            } else {
                2.0 * h / 3.0
            }
        })
        .collect();
    let v = w.iter().zip(y).map(|(a, b)| a * b).sum();
    let e = w
        .iter()
        .zip(dy)
        .map(|(a, b)| (a * b) * (a * b))
        .sum::<f64>()
        .sqrt();
    (v, e)
}

/// Trapèzes sur la même grille (contrôle de l'erreur de quadrature).
pub fn trapezes(y: &[f64]) -> f64 {
    let n = y.len();
    let h = 1.0 / (n - 1) as f64;
    h * (y.iter().sum::<f64>() - 0.5 * (y[0] + y[n - 1]))
}

/// Une chaîne à α fixé : thermalisation puis `n_mes` mesures de S₁ − S₀ espacées de `sep` pas (1 HB + n_or OR).
pub struct Chaine {
    pub delta: Vec<f64>,
    pub plaq: Vec<f64>,
    pub n_cap: u64,
}

#[allow(clippy::too_many_arguments)]
pub fn chaine(
    res: &mut Reseau,
    beta: f64,
    alpha: f64,
    n_therm: usize,
    n_mes: usize,
    sep: usize,
    n_or: usize,
    chaud: bool,
) -> Result<Chaine, String> {
    // séries réservées sans panique (réservation faillible), comptées par `octets_chaine`
    let mut delta: Vec<f64> = portes::reserver(n_mes, "mesures")?;
    let mut plaq: Vec<f64> = portes::reserver(n_mes, "mesures")?;
    res.beta = beta;
    res.fixer_alpha(alpha);
    res.init(chaud);
    for _ in 0..n_therm {
        res.pas(n_or);
    }
    for _ in 0..n_mes {
        for _ in 0..sep {
            res.pas(n_or);
        }
        delta.push(res.delta_s());
        let (sp, sc) = res.somme_plaquettes();
        plaq.push(sp / sc);
    }
    Ok(Chaine {
        delta,
        plaq,
        n_cap: res.n_cap,
    })
}

/// Majorant (octets) de la mémoire de POINTE d'une chaîne : un `Reseau` d'interpolation, construction comprise, et ses
/// `n_mes` mesures (relecture intrication-claude, objection 2). Construction : deux listes brutes de 6V + 3·s·L³ boucles,
/// table de hachage et union des plaquettes (au plus 6V + 9·s·L³ ; redimensionnements comptés), 6 cases par lien ;
/// marche : champ, coefficients, classes, tampon d'un sweep ou d'une somme de plaquettes, séries de mesures. Tailles
/// d'éléments par `size_of` (aucune constante recopiée). Une chaîne de `doublement` en β (géométrie pure) est majorée par
/// la même expression.
///
/// Relecture c19 (majeure 6) : comptés en plus les deux régions du réseau et le tableau des moyennes de blocs (`n_blocs`) ;
/// calcul purement arithmétique (u128), sans aucune allocation.
/// Relecture c19b (majeure 11) : les deux séries de mesures (2 × `n_mes` réels) peuvent coexister avec la CONSTRUCTION d'un
/// réseau (la route β de `doublement` réserve ses séries avant de construire ses réseaux) : elles s'AJOUTENT au maximum de
/// la construction et de la marche, au lieu d'entrer seulement dans la marche. De même le tableau des moyennes de blocs
/// (`n_blocs` réels) et les régions (avant la construction et pendant toute la vie du réseau) : tout tampon proportionnel à
/// une entrée s'AJOUTE.
pub fn octets_chaine(geo: &GeoRep, n_mes: usize, n_blocs: usize) -> u64 {
    use std::mem::size_of;
    let o = |n: usize| n as u128;
    let (v, sl3) = (o(geo.v), o(geo.s) * o(geo.l3));
    let liens = 4 * v;
    let brutes = 6 * v + 3 * sl3;
    let union = 6 * v + 9 * sl3;
    let cap0 = brutes + brutes / 8;
    let croit = union > cap0;
    let seaux0 = (cap0 * 8).div_ceil(7).next_power_of_two();
    let index = if croit { 3 * seaux0 } else { seaux0 } * (o(size_of::<([u32; 4], usize)>()) + 1);
    let brut = 2 * brutes * o(size_of::<([u32; 4], u8, u8, f64)>());
    let plaqs_pic = if croit { 3 * cap0 } else { cap0 } * o(size_of::<Plaq>());
    let plaqs = if croit { 2 * cap0 } else { cap0 } * o(size_of::<Plaq>());
    let regions = 2 * o(geo.l3) * o(size_of::<[bool; 4]>());
    let fixe = plaqs
        + 6 * liens * o(size_of::<[u32; MAXCASE]>())
        + union * o(size_of::<f64>() + size_of::<u32>())
        + liens * o(size_of::<Q>())
        + 2 * liens * o(size_of::<u32>());
    let construction = brut + index + plaqs_pic.max(fixe);
    let marche = fixe
        + (liens.div_ceil(8) * o(size_of::<(usize, Option<Q>, u64)>()))
            .max(union * o(size_of::<f64>()));
    let series = 2 * o(n_mes) * o(size_of::<f64>());
    let blocs = o(n_blocs) * o(size_of::<f64>());
    u64::try_from(regions + series + blocs + construction.max(marche)).unwrap_or(u64::MAX)
}

/// Majorant (octets) des tableaux proportionnels au nombre de POINTS d'un run (abscisses, flux, résultats, profils, lignes
/// JSON et leurs arbres `serde_json`) : 2 Kio par point.
pub fn octets_points(n_points: usize) -> u64 {
    (n_points as u64).saturating_mul(2048)
}
