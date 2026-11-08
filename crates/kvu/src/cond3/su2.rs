//! Fibre dure de DEC pour SU(2) (quaternions FP64) : bain thermique + surrelaxation à agrafes effectives, scores des trois
//! relèvements exacts, identité de Ward grossière (SPEC_T3 §2.2, §3.1).
//!
//! Conventions (SPEC §1.2-§1.3) : U = q₀ + i q·σ, produit (ab)₀ = a₀b₀ − a·b, (ab)_v = a₀b_v + b₀a_v − a_v × b_v ; l'agrafe K_ℓ
//! du lien ℓ = (x, μ) vérifie Σ_{p ∋ ℓ} ½ Tr U_p = ½ Tr(U_ℓ K_ℓ†) ; poids exp(β ½ Tr(U K†)) = exp(β k x₀) avec K = k W,
//! X = U W†. Le bain thermique réutilise le tirage exact de kvu (`cond::cpu::lien_rapide`, Philox à compteur
//! (li, sweep, tirage, COND)) en lui passant l'agrafe A = K† de la convention de kvu.
//!
//! Ligne ē = (U₁, U₂, U₃), V = U₁U₂U₃ figé : U₁ (U₂ fixé) a l'agrafe effective K_eff = K₁ + V K₃† U₂†, U₂ (U₁ fixé)
//! K_eff = K₂ + U₁† V K₃† ; U₃ = U₂† U₁† V est asservi après chaque mise à jour.

use super::geom::Geom;
use super::modele::{Modele, StatsEchantillonneur};
use crate::cond::cpu::lien_rapide;
use crate::heritage::su2_hb::{qadd, qdag, qmul, sha256_hex, Q};
use rayon::prelude::*;
use std::sync::Arc;

/// Défauts injectés pour les tests d'outil (mutants, ORACLES O-11). `Aucun` en production.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MutantSu2 {
    #[default]
    Aucun,
    /// MUT-AGRAFE : agrafe effective de U₁ sans le terme V K₃† U₂† (et de U₂ sans U₁† V K₃†).
    Agrafe,
    /// MUT-U3 : U₃ non recalculé après la mise à jour de U₁ / U₂.
    U3NonRecalcule,
    /// Relèvement 2 des scores sans la rotation R(U₁)ᵀ (MUT-RELEV, côté score).
    ReleveSansRotation,
    /// Ward avec R(V) au lieu de R(V)ᵀ (mutant de l'oracle O-6).
    WardSansTranspose,
}

pub struct FibreSu2 {
    g: Arc<Geom>,
    pub u: Vec<Q>,
    /// V figé par lien grossier.
    pub vfix: Vec<Q>,
    beta: f64,
    k0: u32,
    k1: u32,
    stats: StatsEchantillonneur,
    pub mutant: MutantSu2,
    /// Renormaliser chaque lien mis à jour (SPEC §3.1) ; faux seulement pour le mutant MUT-DERIVE.
    pub renormaliser: bool,
    /// Mutant MUT-DERIVE-F32 : les liens mis à jour sont arrondis en simple précision (dérive réelle hors de SU(2) et de la fibre).
    pub arrondi_f32: bool,
}

// ------------------------------------------------------------------------------------------------------------
// Briques sans état
// ------------------------------------------------------------------------------------------------------------

/// Agrafe K_ℓ du lien (μ, s) : Σ_{ν≠μ} [U_ν(x) U_μ(x+e_ν) U_ν(x+e_μ)† + U_ν(x−e_ν)† U_μ(x−e_ν) U_ν(x−e_ν+e_μ)].
#[inline]
pub fn agrafe(u: &[Q], g: &Geom, mu: usize, s: usize) -> Q {
    let v = g.v;
    let smu = g.nbr[mu][s] as usize;
    let mut k: Q = [0.0; 4];
    for nu in 0..g.d {
        if nu == mu {
            continue;
        }
        let snu = g.nbr[nu][s] as usize;
        let a = qmul(qmul(u[nu * v + s], u[mu * v + snu]), qdag(u[nu * v + smu]));
        let sm = g.nbm[nu][s] as usize;
        let smm = g.nbr[mu][sm] as usize;
        let b = qmul(qmul(qdag(u[nu * v + sm]), u[mu * v + sm]), u[nu * v + smm]);
        k = qadd(k, qadd(a, b));
    }
    k
}

#[inline]
fn fin_lien(renorm_on: bool, f32_on: bool, q: Q) -> Q {
    let q = if renorm_on { renorm(q) } else { q };
    if f32_on {
        [q[0] as f32 as f64, q[1] as f32 as f64, q[2] as f32 as f64, q[3] as f32 as f64]
    } else {
        q
    }
}

#[inline]
fn renorm(q: Q) -> Q {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

/// Surrelaxation U → W U† W, W = K/|K| (inchangé si K ≈ 0).
#[inline]
fn surrelaxer(k: Q, u: Q) -> Q {
    let n2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + k[3] * k[3];
    if n2 < 1e-300 {
        return u;
    }
    let n = n2.sqrt();
    let w = [k[0] / n, k[1] / n, k[2] / n, k[3] / n];
    qmul(qmul(w, qdag(u)), w)
}

/// Partie vectorielle d'un quaternion.
#[inline]
fn vec3(q: Q) -> [f64; 3] {
    [q[1], q[2], q[3]]
}

/// Matrice d'adjointe R(U) (colonne a = partie vectorielle de U (0, e_a) U†), rangée ligne par ligne.
#[inline]
pub fn adjointe(u: Q) -> [[f64; 3]; 3] {
    let ud = qdag(u);
    let mut r = [[0.0; 3]; 3];
    for a in 0..3 {
        let mut e: Q = [0.0; 4];
        e[1 + a] = 1.0;
        let w = qmul(qmul(u, e), ud);
        for i in 0..3 {
            r[i][a] = w[1 + i];
        }
    }
    r
}

#[inline]
fn mat_vec(r: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

#[inline]
fn matt_vec(r: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[1][0] * v[1] + r[2][0] * v[2],
        r[0][1] * v[0] + r[1][1] * v[1] + r[2][1] * v[2],
        r[0][2] * v[0] + r[1][2] * v[1] + r[2][2] * v[2],
    ]
}

/// DEC d'une ligne : U₁ U₂ U₃.
#[inline]
pub fn produit_ligne(u: &[Q], l: &[u32; 3]) -> Q {
    qmul(qmul(u[l[0] as usize], u[l[1] as usize]), u[l[2] as usize])
}

/// Staple-agrafe d'une plaquette p pour le lien ℓ qu'elle contient : A_p tel que ½ Tr U_p = ½ Tr(U_ℓ A_p†) (SPEC §2.2, X(O_p)).
/// `rang` ∈ {0, 1, 2, 3} : position du lien dans (a, b, c†, d†).
pub fn agrafe_de_plaquette(u: &[Q], liens: [u32; 4], rang: usize) -> Q {
    let (a, b, c, d) = (u[liens[0] as usize], u[liens[1] as usize], u[liens[2] as usize], u[liens[3] as usize]);
    // A† tel que ½Tr(U_ℓ A†) = ½Tr U_p  (trace réelle, rotation cyclique)
    match rang {
        0 => qdag(qmul(qmul(b, qdag(c)), qdag(d))),            // U_p = a (b c† d†)         : A† = b c† d†
        1 => qdag(qmul(qmul(qdag(c), qdag(d)), a)),            // Tr(a b c† d†) = Tr(b c† d† a)
        2 => qmul(qmul(qdag(d), a), b),                        // Tr(a b c† d†) = Tr(c† d† a b) = Tr(c (d† a b)†) : A† = (d† a b)† -> A = d† a b
        3 => qmul(qmul(a, b), qdag(c)),                        // Tr(a b c† d†) = Tr(d† a b c†) = Tr(d (a b c†)†) : A = a b c†
        _ => unreachable!(),
    }
}

// ------------------------------------------------------------------------------------------------------------
// Modèle
// ------------------------------------------------------------------------------------------------------------

impl FibreSu2 {
    /// Champ de départ `u` (d·V quaternions), V = DEC(u) figé, β, graine Philox.
    pub fn nouveau(g: Arc<Geom>, u: Vec<Q>, beta: f64, graine: u64) -> FibreSu2 {
        assert_eq!(u.len(), g.nliens());
        let (k0, k1) = crate::commun::Graine(graine).cle();
        let mut f = FibreSu2 {
            g,
            u,
            vfix: Vec::new(),
            beta,
            k0,
            k1,
            stats: StatsEchantillonneur::default(),
            mutant: MutantSu2::Aucun,
            renormaliser: true,
            arrondi_f32: false,
        };
        f.figer_v();
        f
    }

    /// DEC(U) courant de toutes les lignes.
    pub fn dec(&self) -> Vec<Q> {
        self.g.lignes.iter().map(|l| produit_ligne(&self.u, l)).collect()
    }

    pub fn set_beta(&mut self, beta: f64) {
        self.beta = beta;
    }

    /// Remplace la clé Philox (nouveau flux, même champ).
    pub fn set_graine(&mut self, graine: u64) {
        let (k0, k1) = crate::commun::Graine(graine).cle();
        self.k0 = k0;
        self.k1 = k1;
    }

    fn cumuler(&mut self, res: &[(u32, bool)]) {
        for &(nt, ok) in res {
            self.stats.majs += 1;
            self.stats.essais += nt as u64;
            if !ok {
                self.stats.plafond += 1;
            }
        }
    }

    // ---- liens libres -------------------------------------------------------------------------------------------

    fn hb_libres(&mut self, mu: usize, par: usize, sweep: u32, tous: bool) {
        let g = &self.g;
        let sites: &Vec<u32> = if tous { &g.tous[mu][par] } else { &g.libres[mu][par] };
        let (u, beta, k0, k1, v) = (&self.u, self.beta, self.k0, self.k1, g.v);
        let res: Vec<(Q, u32, bool)> = sites
            .par_iter()
            .map(|&s| {
                let k = agrafe(u, g, mu, s as usize);
                let li = (mu * v) as u32 + s;
                let (un, nt, ok) = lien_rapide(qdag(k), beta, li, sweep, k0, k1);
                if ok {
                    (un, nt, true)
                } else {
                    (u[li as usize], nt, false)
                }
            })
            .collect();
        let (renorm_on, f32_on) = (self.renormaliser, self.arrondi_f32);
        let fin = |q: Q| fin_lien(renorm_on, f32_on, q);
        let mut stats = Vec::with_capacity(res.len());
        for (i, &s) in sites.iter().enumerate() {
            let li = mu * v + s as usize;
            self.u[li] = fin(res[i].0);
            stats.push((res[i].1, res[i].2));
        }
        self.cumuler(&stats);
    }

    fn or_libres(&mut self, mu: usize, par: usize, tous: bool) {
        let g = &self.g;
        let sites: &Vec<u32> = if tous { &g.tous[mu][par] } else { &g.libres[mu][par] };
        let (u, v) = (&self.u, g.v);
        let res: Vec<Q> = sites
            .par_iter()
            .map(|&s| {
                let k = agrafe(u, g, mu, s as usize);
                surrelaxer(k, u[mu * v + s as usize])
            })
            .collect();
        let (renorm_on, f32_on) = (self.renormaliser, self.arrondi_f32);
        let fin = |q: Q| fin_lien(renorm_on, f32_on, q);
        for (i, &s) in sites.iter().enumerate() {
            self.u[mu * v + s as usize] = fin(res[i]);
        }
    }

    // ---- lignes ---------------------------------------------------------------------------------------------------

    /// Agrafe effective du lien `quelle` ∈ {1, 2} (U₁ ou U₂) de la ligne `ci`, pour l'état `u`.
    #[inline]
    fn k_eff(&self, u: &[Q], ci: usize, quelle: u8) -> Q {
        let g = &self.g;
        let [l1, l2, l3] = g.lignes[ci];
        let (l1, l2, l3) = (l1 as usize, l2 as usize, l3 as usize);
        let mu = ci / g.vc;
        let v = g.v;
        let vv = self.vfix[ci];
        let k1 = agrafe(u, g, mu, l1 % v);
        let k2 = agrafe(u, g, mu, l2 % v);
        let k3 = agrafe(u, g, mu, l3 % v);
        let (u1, u2) = (u[l1], u[l2]);
        if self.mutant == MutantSu2::Agrafe {
            return if quelle == 1 { k1 } else { k2 };
        }
        if quelle == 1 {
            qadd(k1, qmul(qmul(vv, qdag(k3)), qdag(u2)))
        } else {
            qadd(k2, qmul(qmul(qdag(u1), vv), qdag(k3)))
        }
    }

    fn maj_ligne_hb(&mut self, mu: usize, quelle: u8, sweep: u32) {
        let (beta, k0, k1) = (self.beta, self.k0, self.k1);
        let res: Vec<(Q, Q, u32, bool)> = {
            let me: &FibreSu2 = &*self;
            let lignes = &me.g.lignes_dir[mu];
            lignes
                .par_iter()
                .map(|&ci| {
                    let ci = ci as usize;
                    let [l1, l2, _l3] = me.g.lignes[ci];
                    let cible = if quelle == 1 { l1 } else { l2 };
                    let ke = me.k_eff(&me.u, ci, quelle);
                    let (un, nt, ok) = lien_rapide(qdag(ke), beta, cible, sweep, k0, k1);
                    let (u1, u2) = (me.u[l1 as usize], me.u[l2 as usize]);
                    let un = if ok { un } else { me.u[cible as usize] };
                    let (n1, n2) = if quelle == 1 { (un, u2) } else { (u1, un) };
                    let vv = me.vfix[ci];
                    let u3 = qmul(qmul(qdag(n2), qdag(n1)), vv);
                    (un, u3, nt, ok)
                })
                .collect()
        };
        self.ecrire_lignes(mu, quelle, &res);
    }

    fn maj_ligne_or(&mut self, mu: usize, quelle: u8) {
        let res: Vec<(Q, Q, u32, bool)> = {
            let me: &FibreSu2 = &*self;
            me.g.lignes_dir[mu]
                .par_iter()
                .map(|&ci| {
                    let ci = ci as usize;
                    let [l1, l2, _l3] = me.g.lignes[ci];
                    let cible = if quelle == 1 { l1 } else { l2 };
                    let ke = me.k_eff(&me.u, ci, quelle);
                    let un = surrelaxer(ke, me.u[cible as usize]);
                    let (u1, u2) = (me.u[l1 as usize], me.u[l2 as usize]);
                    let (n1, n2) = if quelle == 1 { (un, u2) } else { (u1, un) };
                    let u3 = qmul(qmul(qdag(n2), qdag(n1)), me.vfix[ci]);
                    (un, u3, 0, true)
                })
                .collect()
        };
        self.ecrire_lignes(mu, quelle, &res);
    }

    fn ecrire_lignes(&mut self, mu: usize, quelle: u8, res: &[(Q, Q, u32, bool)]) {
        let (renorm_on, f32_on) = (self.renormaliser, self.arrondi_f32);
        let fin = |q: Q| fin_lien(renorm_on, f32_on, q);
        let mutu3 = self.mutant == MutantSu2::U3NonRecalcule;
        let mut stats = Vec::new();
        for (i, &ci) in self.g.lignes_dir[mu].iter().enumerate() {
            let [l1, l2, l3] = self.g.lignes[ci as usize];
            let cible = if quelle == 1 { l1 } else { l2 };
            let (un, u3, nt, ok) = res[i];
            self.u[cible as usize] = fin(un);
            if !mutu3 {
                self.u[l3 as usize] = fin(u3);
            }
            if nt > 0 || !ok {
                stats.push((nt, ok));
            }
        }
        self.cumuler(&stats);
    }

    // ---- ordonnancement ---------------------------------------------------------------------------------------------

    fn balayage_hb(&mut self, sweep: u32) {
        for mu in 0..self.g.d {
            self.hb_libres(mu, 0, sweep, false);
            self.hb_libres(mu, 1, sweep, false);
            self.maj_ligne_hb(mu, 1, sweep);
            self.maj_ligne_hb(mu, 2, sweep);
        }
    }

    fn balayage_or(&mut self) {
        for mu in 0..self.g.d {
            self.or_libres(mu, 0, false);
            self.or_libres(mu, 1, false);
            self.maj_ligne_or(mu, 1);
            self.maj_ligne_or(mu, 2);
        }
    }

    /// Mise à jour séquentielle (en place) : même loi, autre schéma de calcul (O-2b).
    fn balayage_sequentiel(&mut self, sweep: u32, hb: bool) {
        let g = self.g.clone();
        let (beta, k0, k1, v) = (self.beta, self.k0, self.k1, g.v);
        for mu in 0..g.d {
            for s in 0..g.v {
                if g.pos[mu * v + s] != -1 {
                    continue;
                }
                let k = agrafe(&self.u, &g, mu, s);
                let li = mu * v + s;
                let nouveau = if hb {
                    let (un, nt, ok) = lien_rapide(qdag(k), beta, li as u32, sweep, k0, k1);
                    self.cumuler(&[(nt, ok)]);
                    if ok { renorm(un) } else { self.u[li] }
                } else {
                    renorm(surrelaxer(k, self.u[li]))
                };
                self.u[li] = nouveau;
            }
            for &ci in &g.lignes_dir[mu] {
                let ci = ci as usize;
                let [l1, l2, l3] = g.lignes[ci];
                for quelle in [1u8, 2u8] {
                    let ke = self.k_eff(&self.u, ci, quelle);
                    let cible = if quelle == 1 { l1 } else { l2 } as usize;
                    let nouveau = if hb {
                        let (un, nt, ok) = lien_rapide(qdag(ke), beta, cible as u32, sweep, k0, k1);
                        self.cumuler(&[(nt, ok)]);
                        if ok { renorm(un) } else { self.u[cible] }
                    } else {
                        renorm(surrelaxer(ke, self.u[cible]))
                    };
                    self.u[cible] = nouveau;
                    let u3 = qmul(qmul(qdag(self.u[l2 as usize]), qdag(self.u[l1 as usize])), self.vfix[ci]);
                    self.u[l3 as usize] = renorm(u3);
                }
            }
        }
    }

    // ---- relèvements exacts (flots) -----------------------------------------------------------------------------------

    /// Flot EXACT du relèvement `k` (0, 1, 2 = relèvements 1, 2, 3 du SPEC §2.2) de paramètre `t` dans la direction `a` sur la ligne
    /// `ci` : V_ē → exp(i t σ_a) V_ē, rien d'autre ne change dans DEC. Ne modifie pas V figé.
    pub fn releve(&mut self, k: usize, ci: usize, a: usize, t: f64) {
        let [l1, l2, l3] = self.g.lignes[ci];
        let (u1, u2) = (self.u[l1 as usize], self.u[l2 as usize]);
        let (cible, eta) = match k {
            0 => (l1, {
                let mut e = [0.0; 3];
                e[a] = 1.0;
                e
            }),
            1 => {
                let mut e = [0.0; 3];
                e[a] = 1.0;
                (l2, if self.mutant == MutantSu2::ReleveSansRotation { e } else { matt_vec(&adjointe(u1), e) })
            }
            _ => {
                let mut e = [0.0; 3];
                e[a] = 1.0;
                (l3, matt_vec(&adjointe(qmul(u1, u2)), e))
            }
        };
        let n = (eta[0] * eta[0] + eta[1] * eta[1] + eta[2] * eta[2]).sqrt();
        let (c, s) = ((t * n).cos(), (t * n).sin());
        let g = [c, s * eta[0] / n, s * eta[1] / n, s * eta[2] / n];
        self.u[cible as usize] = qmul(g, self.u[cible as usize]);
    }

    /// Action de Wilson S(U) = β Σ_p (1 − ½ Tr U_p) (tous les plans, tous les sites).
    pub fn action(&self) -> f64 {
        let g = &self.g;
        let somme: f64 = (0..g.nplaq())
            .into_par_iter()
            .map(|pi| {
                let l = g.liens_plaquette(pi / g.v, pi % g.v);
                let a = qmul(self.u[l[0] as usize], self.u[l[1] as usize]);
                let b = qmul(qdag(self.u[l[2] as usize]), qdag(self.u[l[3] as usize]));
                1.0 - qmul(a, b)[0]
            })
            .sum();
        self.beta * somme
    }

    // ---- scores ---------------------------------------------------------------------------------------------------

    /// (σ¹, σ², σ³) de la ligne `ci` pour l'état courant (SPEC §2.2).
    pub fn scores_ligne(&self, ci: usize) -> [[f64; 3]; 3] {
        let g = &self.g;
        let [l1, l2, l3] = g.lignes[ci];
        let mu = ci / g.vc;
        let v = g.v;
        let (u1, u2, u3) = (self.u[l1 as usize], self.u[l2 as usize], self.u[l3 as usize]);
        let k1 = agrafe(&self.u, g, mu, l1 as usize % v);
        let k2 = agrafe(&self.u, g, mu, l2 as usize % v);
        let k3 = agrafe(&self.u, g, mu, l3 as usize % v);
        let b = self.beta;
        let sc = |x: [f64; 3]| [b * x[0], b * x[1], b * x[2]];
        let s1 = sc(vec3(qmul(u1, qdag(k1))));
        let y2 = vec3(qmul(u2, qdag(k2)));
        let y3 = vec3(qmul(u3, qdag(k3)));
        let r1 = adjointe(u1);
        let s2 = if self.mutant == MutantSu2::ReleveSansRotation { sc(y2) } else { sc(mat_vec(&r1, y2)) };
        let r12 = adjointe(qmul(u1, u2));
        let s3 = sc(mat_vec(&r12, y3));
        [s1, s2, s3]
    }
}

impl Modele for FibreSu2 {
    fn geom(&self) -> &Arc<Geom> {
        &self.g
    }
    fn beta(&self) -> f64 {
        self.beta
    }
    fn ncomp(&self) -> usize {
        3
    }
    fn groupe(&self) -> &'static str {
        "su2"
    }

    fn pas(&mut self, sweep: u32, n_or: usize) {
        self.balayage_hb(sweep);
        for _ in 0..n_or {
            self.balayage_or();
        }
    }

    fn pas_sequentiel(&mut self, sweep: u32, n_or: usize) {
        self.balayage_sequentiel(sweep, true);
        for _ in 0..n_or {
            self.balayage_sequentiel(sweep, false);
        }
    }

    fn pas_libre(&mut self, sweep: u32, n_or: usize) {
        for mu in 0..self.g.d {
            self.hb_libres(mu, 0, sweep, true);
            self.hb_libres(mu, 1, sweep, true);
        }
        for _ in 0..n_or {
            for mu in 0..self.g.d {
                self.or_libres(mu, 0, true);
                self.or_libres(mu, 1, true);
            }
        }
    }

    fn figer_v(&mut self) {
        self.vfix = self.dec();
    }

    fn observables(&self, pis: &[u32], out: &mut [f64]) {
        let g = &self.g;
        let u = &self.u;
        out.par_iter_mut().zip(pis.par_iter()).for_each(|(o, &pi)| {
            let pi = pi as usize;
            let l = g.liens_plaquette(pi / g.v, pi % g.v);
            let a = qmul(u[l[0] as usize], u[l[1] as usize]);
            let b = qmul(qdag(u[l[2] as usize]), qdag(u[l[3] as usize]));
            *o = qmul(a, b)[0];
        });
    }

    fn scores(&self, cis: &[u32], separes: bool, out: &mut [f64]) {
        let nc = if separes { 9 } else { 3 };
        assert_eq!(out.len(), cis.len() * nc);
        out.par_chunks_mut(nc).zip(cis.par_iter()).for_each(|(o, &ci)| {
            let s = self.scores_ligne(ci as usize);
            if separes {
                for k in 0..3 {
                    for a in 0..3 {
                        o[3 * k + a] = s[k][a];
                    }
                }
            } else {
                for a in 0..3 {
                    o[a] = (s[0][a] + s[1][a] + s[2][a]) / 3.0;
                }
            }
        });
    }

    fn canal(&self, c: &super::modele::Canal, out: &mut [f64]) -> bool {
        use super::modele::Canal;
        let g = &self.g;
        match *c {
            Canal::SteinLibre { li, a, pi } => {
                let a = a as usize;
                let (mu, s) = (li as usize / g.v, li as usize % g.v);
                let k = agrafe(&self.u, g, mu, s);
                let ul = self.u[li as usize];
                let ys = self.beta * vec3(qmul(ul, qdag(k)))[a];
                let l = g.liens_plaquette(pi as usize / g.v, pi as usize % g.v);
                let yo = match l.iter().position(|&z| z == li) {
                    Some(rang) => -vec3(qmul(ul, qdag(agrafe_de_plaquette(&self.u, l, rang))))[a],
                    None => 0.0,
                };
                out[0] = yo;
                out[1] = ys;
                out[2] = self.beta * qmul(ul, qdag(k))[0];
                true
            }
            Canal::SteinLigne { ci, a, pi } => {
                let a = a as usize;
                let ci = ci as usize;
                let [l1, l2, _] = g.lignes[ci];
                let mu = ci / g.vc;
                let (u1, u2) = (self.u[l1 as usize], self.u[l2 as usize]);
                let k1 = agrafe(&self.u, g, mu, l1 as usize % g.v);
                let k2 = agrafe(&self.u, g, mu, l2 as usize % g.v);
                let mut ea = [0.0; 3];
                ea[a] = 1.0;
                let eta = mat_vec(&adjointe(u1), ea);
                let m1 = vec3(qmul(u1, qdag(k1)));
                let m2 = vec3(qmul(u2, qdag(k2)));
                let ys = self.beta * (eta[0] * m1[0] + eta[1] * m1[1] + eta[2] * m1[2]) - self.beta * m2[a];
                let l = g.liens_plaquette(pi as usize / g.v, pi as usize % g.v);
                let mut yo = 0.0;
                if let Some(rang) = l.iter().position(|&z| z == l1) {
                    let y = vec3(qmul(u1, qdag(agrafe_de_plaquette(&self.u, l, rang))));
                    yo += -(eta[0] * y[0] + eta[1] * y[1] + eta[2] * y[2]);
                }
                if let Some(rang) = l.iter().position(|&z| z == l2) {
                    let y = vec3(qmul(u2, qdag(agrafe_de_plaquette(&self.u, l, rang))));
                    yo += y[a];
                }
                out[0] = yo;
                out[1] = ys;
                out[2] = self.beta * (qmul(u1, qdag(k1))[0] + qmul(u2, qdag(k2))[0]);
                true
            }
            Canal::DeltaS { ci, a, t } => {
                let ci = ci as usize;
                let l1 = g.lignes[ci][0] as usize;
                let mu = ci / g.vc;
                let k1 = agrafe(&self.u, g, mu, l1 % g.v);
                let m = qmul(self.u[l1], qdag(k1));
                out[0] = -self.beta * ((t.cos() - 1.0) * m[0] - t.sin() * m[1 + a as usize]);
                true
            }
            Canal::Plaq { pi } => {
                self.observables(&[pi], &mut out[..1]);
                true
            }
            Canal::MomentsLignes => {
                let n = g.lignes.len() as f64;
                let mut s1 = [0.0f64; 3];
                let mut s2 = [0.0f64; 3];
                for l in &g.lignes {
                    for j in 0..3 {
                        let q0 = self.u[l[j] as usize][0];
                        s1[j] += q0;
                        s2[j] += q0 * q0;
                    }
                }
                for j in 0..3 {
                    out[j] = s1[j] / n;
                    out[3 + j] = s2[j] / n;
                }
                true
            }
            Canal::GroupesPlaquettes => {
                let cls = g.classes_plaquettes();
                let ng = g.d - 1 + 3;
                let mut somme = vec![0.0f64; ng];
                let mut nn = vec![0usize; ng];
                let pis: Vec<u32> = (0..g.nplaq() as u32).collect();
                let mut o = vec![0.0f64; pis.len()];
                self.observables(&pis, &mut o);
                for (pi, v) in o.iter().enumerate() {
                    let [t, nl] = cls[pi];
                    let (a, b) = (t as usize, g.d - 1 + nl as usize);
                    if a < g.d - 1 {
                        somme[a] += v;
                        nn[a] += 1;
                    }
                    if nl <= 2 {
                        somme[b] += v;
                        nn[b] += 1;
                    }
                }
                for k in 0..ng {
                    out[k] = somme[k] / nn[k].max(1) as f64;
                }
                true
            }
        }
    }

    fn terme_x(&self, paires: &[(u32, u32)], separes: bool, out: &mut [f64]) {
        let nc = if separes { 9 } else { 3 };
        assert_eq!(out.len(), paires.len() * nc);
        let g = &self.g;
        out.par_chunks_mut(nc).zip(paires.par_iter()).for_each(|(o, &(pi, ci))| {
            let (pi, ci) = (pi as usize, ci as usize);
            let l = g.liens_plaquette(pi / g.v, pi % g.v);
            let ligne = g.lignes[ci];
            let (u1, u2) = (self.u[ligne[0] as usize], self.u[ligne[1] as usize]);
            let mut x = [[0.0f64; 3]; 3];
            for k in 0..3 {
                if let Some(rang) = l.iter().position(|&z| z == ligne[k]) {
                    let ap = agrafe_de_plaquette(&self.u, l, rang);
                    let uk = self.u[ligne[k] as usize];
                    let yq = qmul(uk, qdag(ap));
                    let y = [-yq[1], -yq[2], -yq[3]];
                    x[k] = match k {
                        0 => y,
                        1 => {
                            if self.mutant == MutantSu2::ReleveSansRotation {
                                y
                            } else {
                                mat_vec(&adjointe(u1), y)
                            }
                        }
                        _ => mat_vec(&adjointe(qmul(u1, u2)), y),
                    };
                }
            }
            if separes {
                for k in 0..3 {
                    for a in 0..3 {
                        o[3 * k + a] = x[k][a];
                    }
                }
            } else {
                for a in 0..3 {
                    o[a] = (x[0][a] + x[1][a] + x[2][a]) / 3.0;
                }
            }
        });
    }

    fn residu_contrainte(&self) -> f64 {
        self.g
            .lignes
            .par_iter()
            .zip(self.vfix.par_iter())
            .map(|(l, v)| {
                let p = produit_ligne(&self.u, l);
                (0..4).map(|i| (p[i] - v[i]).abs()).fold(0.0, f64::max)
            })
            .reduce(|| 0.0, f64::max)
    }

    fn residu_ward(&self) -> f64 {
        let g = &self.g;
        let nc = g.ngros();
        // σ¹ et σ³ de toutes les lignes
        let sc: Vec<[[f64; 3]; 3]> = (0..nc).into_par_iter().map(|ci| self.scores_ligne(ci)).collect();
        let mutant = self.mutant == MutantSu2::WardSansTranspose;
        (0..g.vc)
            .into_par_iter()
            .map(|y| {
                let mut tot = [0.0f64; 3];
                for mu in 0..g.d {
                    let s1 = sc[mu * g.vc + y][0];
                    for a in 0..3 {
                        tot[a] += s1[a];
                    }
                    // ligne entrante : (Y − e_mu, mu)
                    let mut cy = g.coords_gros[y];
                    cy[mu] = ((cy[mu] as usize + g.lc - 1) % g.lc) as u16;
                    let yi = g.site_gros(&[cy[0] as i64, cy[1] as i64, cy[2] as i64, cy[3] as i64]);
                    let ci_in = mu * g.vc + yi;
                    let s3 = sc[ci_in][2];
                    let r = adjointe(self.vfix[ci_in]);
                    let t = if mutant { mat_vec(&r, s3) } else { matt_vec(&r, s3) };
                    for a in 0..3 {
                        tot[a] -= t[a];
                    }
                }
                tot.iter().map(|x| x.abs()).fold(0.0, f64::max)
            })
            .reduce(|| 0.0, f64::max)
    }

    fn defaut_unitarite(&self) -> f64 {
        self.u
            .par_iter()
            .map(|q| ((q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt() - 1.0).abs())
            .reduce(|| 0.0, f64::max)
    }

    fn stats(&self) -> StatsEchantillonneur {
        self.stats
    }

    fn empreinte(&self) -> String {
        let mut octets = Vec::with_capacity(self.u.len() * 32);
        for q in &self.u {
            for c in q {
                octets.extend_from_slice(&c.to_le_bytes());
            }
        }
        sha256_hex(&octets)
    }

    fn champ_plat(&self) -> Vec<f64> {
        self.u.iter().flatten().copied().collect()
    }

    fn v_fige_plat(&self) -> Vec<f64> {
        self.vfix.iter().flatten().copied().collect()
    }
}

/// Champ froid (tous les liens = identité).
pub fn froid(n: usize) -> Vec<Q> {
    vec![[1.0, 0.0, 0.0, 0.0]; n]
}

/// Champ de Haar i.i.d. (quaternions) déterministe en fonction d'une graine (xorshift, hors Philox : tests seulement).
pub fn champ_haar_test(n: usize, graine: u64) -> Vec<Q> {
    let mut x = graine ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            // gaussiennes par Box–Muller
            let g = |n: &mut dyn FnMut() -> f64| {
                let (a, b) = (n(), n());
                (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
            };
            let q = [g(&mut next), g(&mut next), g(&mut next), g(&mut next)];
            renorm(q)
        })
        .collect()
}
