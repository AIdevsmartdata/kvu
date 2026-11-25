//! Géométrie anisotrope (ℓ0, ℓ1, ℓ2, ℓ3), pile de torsion en table, chaîne CPU F64 (bain thermique, OR, refroidissement).
//!
//! Indexation : site s = ((x0·ℓ1 + x1)·ℓ2 + x2)·ℓ3 + x3, lien li = μ·V + s (celle de `heritage::su2_hb::Geo` à côtés égaux).
//! La mise à jour d'un lien reproduit À L'OPÉRATION PRÈS `cpu::hb_lien` / `cpu::or_gel` (agrafe tordue
//! `torsion::agrafe_tordue`, tirages Philox de compteur (li, sweep, tirage, ETIQUETTE_HB)) : à côtés égaux et pairs, avec le
//! coloriage de parité, une chaîne `ChaineTube` est égale à l'octet à `Cpu64` (test `egal_a_cpu64_*`).
//! Côtés impairs : coloriage glouton (deux sites voisins n'ont jamais la même couleur) ; la mise à jour d'une classe en deux
//! phases (lecture puis écriture) est alors exactement la mise à jour en place.

use crate::commun::{ETIQUETTE_HB, ETIQUETTE_INIT};
use crate::heritage::su2_hb::{self, k_final, k_reduce, qadd, qdag, qmul, qnorm, qscale, Q};
use crate::su2::cpu::haar_q;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Taille de classe de couleur en dessous de laquelle un balayage est séquentiel (le parallélisme interne coûte plus qu'il ne
/// rapporte sur les petits réseaux ; résultat identique, l'ordre de collecte étant le même).
pub const SEUIL_PAR: usize = 2048;

/// Géométrie anisotrope, pile de torsion et plans : déplacés tels quels dans le socle commun `geom` (kvu v0.2).
pub use crate::geom::reseau::{indice_plan, GeoA, Pile, PLANS};

/// Une chaîne SU(2) sur réseau anisotrope (champ li = μ·V + s), mêmes compteurs que `heritage::su2_hb::CpuSu2`.
#[derive(Clone)]
pub struct ChaineTube {
    pub g: GeoA,
    pub u: Vec<Q>,
    pub k0: u32,
    pub k1: u32,
    pub sweep_id: u32,
    pub n_tries: u64,
    pub n_cap: u64,
    pub couleurs: Vec<Vec<usize>>,
}

/// Traceur des couples (clé, compteur) que ce module passe à Philox (relecture c16b, objection 14 ; test
/// `tests/su2_tube_philox.rs`). Inactif par défaut : une lecture atomique par tirage, aucun effet sur les calculs. Actif, il
/// note chaque appel direct, et les compteurs consommés dans le code partagé tels que fixés par l'appel réel :
/// `su2_hb::sample_x0` (lien, balayage, d, ETIQUETTE_HB) pour d de la valeur d'entrée à la valeur de sortie du compteur de
/// tirages, `cpu::haar_q` (lien, 0xFFFFFFFF, 0 et 1, ETIQUETTE_INIT).
pub mod trace_philox {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    /// (k0, k1, compteur).
    pub type Appel = (u32, u32, [u32; 4]);

    static ACTIF: AtomicBool = AtomicBool::new(false);
    static NOTES: Mutex<Vec<Appel>> = Mutex::new(Vec::new());

    /// Démarre une trace (vide la précédente). Pour les tests : un verrou par tirage tant qu'elle est active.
    pub fn demarrer() {
        if let Ok(mut v) = NOTES.lock() {
            v.clear();
        }
        ACTIF.store(true, Ordering::SeqCst);
    }

    /// Arrête la trace et rend les appels notés.
    pub fn arreter() -> Vec<Appel> {
        ACTIF.store(false, Ordering::SeqCst);
        NOTES
            .lock()
            .map(|mut v| std::mem::take(&mut *v))
            .unwrap_or_default()
    }

    #[inline]
    pub(super) fn actif() -> bool {
        ACTIF.load(Ordering::Relaxed)
    }

    #[inline]
    pub(super) fn noter(k0: u32, k1: u32, c: [u32; 4]) {
        if actif() {
            if let Ok(mut v) = NOTES.lock() {
                v.push((k0, k1, c));
            }
        }
    }
}

#[inline]
fn tirage(li: u32, sweep_id: u32, draw: &mut u32, k0: u32, k1: u32) -> [u32; 4] {
    trace_philox::noter(k0, k1, [li, sweep_id, *draw, ETIQUETTE_HB]);
    let r = su2_hb::philox4x32_10([li, sweep_id, *draw, ETIQUETTE_HB], k0, k1);
    *draw = draw.wrapping_add(1);
    r
}

#[inline]
fn fmax0(v: f64) -> f64 {
    if v > 0.0 {
        v
    } else {
        0.0
    }
}

/// Réseau anisotrope (S = (β/ξ₀) Σ_spatiales + β ξ₀ Σ_temporelles) : multiplie la pile (z = ±1) par le poids du plan,
/// ξ₀ s'il contient l'axe `temps`, 1/ξ₀ sinon. Pour ξ₀ = 1 la pile est rendue telle quelle (mêmes bits).
pub fn pile_aniso(mut pile: Pile, g: &GeoA, xi0: f64, temps: usize) -> Pile {
    if xi0 == 1.0 {
        return pile;
    }
    for (p, &(mu, nu)) in PLANS.iter().enumerate() {
        let w = if mu == temps || nu == temps { xi0 } else { 1.0 / xi0 };
        for s in 0..g.v {
            pile.z[p * g.v + s] *= w;
        }
    }
    pile
}

/// Plaquette U_μ(s) U_ν(s+μ) U_μ(s+ν)† U_ν(s)† (même association que `su2_hb::measure_quat`).
#[inline]
pub fn plaquette(g: &GeoA, u: &[Q], s: usize, x: &[usize; 4], mu: usize, nu: usize) -> Q {
    let ld = |d: usize, site: usize| u[d * g.v + site];
    let smu = g.nb_up(s, mu, x[mu]);
    let snu = g.nb_up(s, nu, x[nu]);
    qmul(
        qmul(ld(mu, s), ld(nu, smu)),
        qmul(qdag(ld(mu, snu)), qdag(ld(nu, s))),
    )
}

/// Agrafe tordue du lien (s, μ) : mêmes opérations que `torsion::agrafe_tordue`, z lu dans la table.
// Boucle indicée gardée telle quelle : même ordre d'opérations que `torsion::agrafe_tordue` (égalité à l'octet avec Cpu64).
#[allow(clippy::needless_range_loop)]
#[inline]
pub fn agrafe(g: &GeoA, u: &[Q], pile: &Pile, mu: usize, s: usize, x: &[usize; 4]) -> Q {
    let ld = |nu: usize, site: usize| u[nu * g.v + site];
    let mut a: Q = [0.0; 4];
    let smu = g.nb_up(s, mu, x[mu]);
    for nu in 0..4 {
        if nu == mu {
            continue;
        }
        let snu = g.nb_up(s, nu, x[nu]);
        let sdn = g.nb_dn(s, nu, x[nu]);
        let smu_dn = g.nb_dn(smu, nu, x[nu]);
        let p = indice_plan(mu, nu);
        let t_plaq = qmul(qmul(ld(nu, smu), qdag(ld(mu, snu))), qdag(ld(nu, s)));
        let zt = pile.z(p, s);
        let b_plaq = qmul(qmul(qdag(ld(nu, smu_dn)), qdag(ld(mu, sdn))), ld(nu, sdn));
        let zb = pile.z(p, sdn);
        a = qadd(a, qadd(qscale(t_plaq, zt), qscale(b_plaq, zb)));
    }
    a
}

/// Clé Philox (k0, k1) d'une chaîne de graine `graine` : ses 64 bits tels quels, sans mélange, donc deux graines distinctes
/// donnent deux clés distinctes. SEULE dérivation de clé du module (`ChaineTube::nouvelle`). Les compteurs séparent les domaines
/// par champs : (lien, balayage, tirage, ETIQUETTE_HB) pour le bain thermique, (lien, 0xFFFFFFFF, 0|1, ETIQUETTE_INIT) pour le
/// départ chaud, (lien, 0xB0B0, 0|1, ETIQUETTE_INIT) pour le bruit de `smin`.
pub fn cle_philox(graine: u64) -> (u32, u32) {
    ((graine & 0xFFFF_FFFF) as u32, (graine >> 32) as u32)
}

struct Maj {
    li: usize,
    q: Option<Q>,
    ntry: u64,
    cap: u64,
}

impl ChaineTube {
    pub fn nouvelle(g: GeoA, graine: u64) -> ChaineTube {
        let couleurs = g.couleurs();
        let v = g.v;
        ChaineTube {
            g,
            u: vec![[1.0, 0.0, 0.0, 0.0]; 4 * v],
            k0: cle_philox(graine).0,
            k1: cle_philox(graine).1,
            sweep_id: 0,
            n_tries: 0,
            n_cap: 0,
            couleurs,
        }
    }

    /// Départ chaud (Haar, compteur (li, 0xFFFFFFFF, 0|1, ETIQUETTE_INIT), comme `CpuSu2::init`) ou froid (identité).
    pub fn init(&mut self, chaud: bool) {
        let (k0, k1) = (self.k0, self.k1);
        self.u.par_iter_mut().enumerate().for_each(|(i, q)| {
            *q = if chaud {
                trace_philox::noter(k0, k1, [i as u32, 0xFFFF_FFFF, 0, ETIQUETTE_INIT]);
                trace_philox::noter(k0, k1, [i as u32, 0xFFFF_FFFF, 1, ETIQUETTE_INIT]);
                haar_q(i as u32, 0xFFFF_FFFF, ETIQUETTE_INIT, k0, k1)
            } else {
                [1.0, 0.0, 0.0, 0.0]
            };
        });
        self.sweep_id = 0;
        self.n_tries = 0;
        self.n_cap = 0;
    }

    /// Vide U*(κ0, κ1) du secteur n23 = 1 (lib_tube `vacuum`) : U2 = iσ1 sur x2 = ℓ2−1, U3 = iσ2 sur x3 = ℓ3−1,
    /// U0 = κ0 sur x0 = ℓ0−1, U1 = κ1 sur x1 = ℓ1−1, 1 ailleurs. Dans le secteur doublement tordu : vortex mince (S/β = 2ℓ2ℓ3).
    pub fn vide(&mut self, k0p: bool, k1p: bool) {
        let g = self.g.clone();
        for s in 0..g.v {
            let x = g.coords(s);
            self.u[s] = if !k0p && x[0] == g.d[0] - 1 {
                [-1.0, 0.0, 0.0, 0.0]
            } else {
                [1.0, 0.0, 0.0, 0.0]
            };
            self.u[g.v + s] = if !k1p && x[1] == g.d[1] - 1 {
                [-1.0, 0.0, 0.0, 0.0]
            } else {
                [1.0, 0.0, 0.0, 0.0]
            };
            self.u[2 * g.v + s] = if x[2] == g.d[2] - 1 {
                [0.0, 1.0, 0.0, 0.0]
            } else {
                [1.0, 0.0, 0.0, 0.0]
            };
            self.u[3 * g.v + s] = if x[3] == g.d[3] - 1 {
                [0.0, 0.0, 1.0, 0.0]
            } else {
                [1.0, 0.0, 0.0, 0.0]
            };
        }
    }

    /// Fond abélien de 't Hooft (lib_tube `abelian_thooft`) : flux σ3 uniforme π/(ℓ0ℓ1) par plaquette (0,1) et π/(ℓ2ℓ3) par
    /// plaquette (2,3) ; S/β = ℓ0ℓ1ℓ2ℓ3 [(1 − cos π/(ℓ0ℓ1)) + (1 − cos π/(ℓ2ℓ3))] dans le secteur doublement tordu.
    pub fn abelien(&mut self) {
        let g = self.g.clone();
        let u1 = |ang: f64| -> Q { [ang.cos(), 0.0, 0.0, ang.sin()] };
        let ph = PI / (g.d[0] * g.d[1]) as f64;
        let pf = PI / (g.d[2] * g.d[3]) as f64;
        for s in 0..g.v {
            let x = g.coords(s);
            let a0 = if x[0] == g.d[0] - 1 {
                -(g.d[0] as f64) * ph * x[1] as f64
            } else {
                0.0
            };
            let a1 = ph * x[0] as f64;
            let a2 = if x[2] == g.d[2] - 1 {
                -(g.d[2] as f64) * pf * x[3] as f64
            } else {
                0.0
            };
            let a3 = pf * x[2] as f64;
            for (mu, a) in [a0, a1, a2, a3].into_iter().enumerate() {
                self.u[mu * g.v + s] = u1(a);
            }
        }
    }

    /// Bruit : U ← exp(i ε a·σ/2) U, a gaussien (Philox, compteur (li, 0xB0B0, 0|1, ETIQUETTE_INIT)).
    pub fn bruiter(&mut self, eps: f64) {
        let (k0, k1) = (self.k0, self.k1);
        self.u.par_iter_mut().enumerate().for_each(|(i, q)| {
            trace_philox::noter(k0, k1, [i as u32, 0xB0B0, 0, ETIQUETTE_INIT]);
            trace_philox::noter(k0, k1, [i as u32, 0xB0B0, 1, ETIQUETTE_INIT]);
            let r1 = su2_hb::philox4x32_10([i as u32, 0xB0B0, 0, ETIQUETTE_INIT], k0, k1);
            let r2 = su2_hb::philox4x32_10([i as u32, 0xB0B0, 1, ETIQUETTE_INIT], k0, k1);
            let ra = (-2.0 * su2_hb::u01(r1[0], r1[1]).ln()).sqrt();
            let pa = 2.0 * PI * su2_hb::u01(r1[2], r1[3]);
            let rb = (-2.0 * su2_hb::u01(r2[0], r2[1]).ln()).sqrt();
            let pb = 2.0 * PI * su2_hb::u01(r2[2], r2[3]);
            let a = [ra * pa.cos(), ra * pa.sin(), rb * pb.cos()];
            let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt() * eps;
            let h = n / 2.0;
            let sh = if n > 0.0 { h.sin() / n * eps } else { 0.0 };
            let e = [h.cos(), sh * a[0], sh * a[1], sh * a[2]];
            let r = qmul(e, *q);
            *q = qscale(r, 1.0 / qnorm(r));
        });
    }

    fn hb_lien(&self, beta: f64, pile: &Pile, mu: usize, s: usize) -> Maj {
        let g = &self.g;
        let x = g.coords(s);
        let li = mu * g.v + s;
        let a = agrafe(g, &self.u, pile, mu, s, &x);
        let k = qnorm(a);
        let mut draw: u32 = 0;
        let sid = self.sweep_id;
        let (k0, k1) = (self.k0, self.k1);
        if k.partial_cmp(&1e-300) != Some(std::cmp::Ordering::Greater) {
            trace_philox::noter(k0, k1, [li as u32, sid, 0, ETIQUETTE_HB]);
            trace_philox::noter(k0, k1, [li as u32, sid, 1, ETIQUETTE_HB]);
            let r1 = su2_hb::philox4x32_10([li as u32, sid, 0, ETIQUETTE_HB], k0, k1);
            let r2 = su2_hb::philox4x32_10([li as u32, sid, 1, ETIQUETTE_HB], k0, k1);
            // haar_from(r1, r2) de su2_hb (privé) : même construction que cpu::haar_q.
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
            return Maj {
                li,
                q: Some(qscale(q, 1.0 / qnorm(q))),
                ntry: 0,
                cap: 0,
            };
        }
        let vv = qscale(a, 1.0 / k);
        let alpha = beta * k;
        let d0 = draw;
        let (x0, ntry, ok) = su2_hb::sample_x0(alpha, li as u32, sid, &mut draw, k0, k1);
        if trace_philox::actif() {
            for d in d0..draw {
                trace_philox::noter(k0, k1, [li as u32, sid, d, ETIQUETTE_HB]);
            }
        }
        if !ok {
            return Maj {
                li,
                q: None,
                ntry: ntry as u64,
                cap: 1,
            };
        }
        let rd = tirage(li as u32, sid, &mut draw, k0, k1);
        let ct = 2.0 * su2_hb::u01(rd[0], rd[1]) - 1.0;
        let phi = 2.0 * PI * su2_hb::u01(rd[2], rd[3]);
        let st = fmax0(1.0 - ct * ct).sqrt();
        let rr = fmax0(1.0 - x0 * x0).sqrt();
        let w = [x0, rr * st * phi.cos(), rr * st * phi.sin(), rr * ct];
        let un = qmul(w, qdag(vv));
        Maj {
            li,
            q: Some(qscale(un, 1.0 / qnorm(un))),
            ntry: ntry as u64,
            cap: 0,
        }
    }

    /// Un balayage de bain thermique (μ = 0..3, puis classes de couleur dans l'ordre).
    pub fn sweep_hb(&mut self, beta: f64, pile: &Pile) {
        self.sweep_hb_ordre(beta, pile, false)
    }

    /// `inverse = true` : classes de couleur en ordre inverse (MUTANT de la porte de non-régression).
    pub fn sweep_hb_ordre(&mut self, beta: f64, pile: &Pile, inverse: bool) {
        let nc = self.couleurs.len();
        for mu in 0..4 {
            for k in 0..nc {
                let kk = if inverse { nc - 1 - k } else { k };
                let cl = &self.couleurs[kk];
                let ups: Vec<Maj> = if cl.len() >= SEUIL_PAR {
                    cl.par_iter()
                        .map(|&s| self.hb_lien(beta, pile, mu, s))
                        .collect()
                } else {
                    cl.iter()
                        .map(|&s| self.hb_lien(beta, pile, mu, s))
                        .collect()
                };
                for up in ups {
                    // cumuls SATURANTS (relecture c16b, objection 11) : jamais de débordement, même plafond atteint partout
                    self.n_tries = self.n_tries.saturating_add(up.ntry);
                    self.n_cap = self.n_cap.saturating_add(up.cap);
                    if let Some(q) = up.q {
                        self.u[up.li] = q;
                    }
                }
            }
        }
        self.sweep_id = self.sweep_id.wrapping_add(1);
    }

    /// RÉFÉRENCE DE TEST (séquentielle, lente ; jamais en production) : un balayage de bain thermique EN PLACE, site par site
    /// (μ = 0..3, classes dans l'ordre, sites dans l'ordre de leur classe), chaque lien écrit dès qu'il est tiré. À compteurs
    /// Philox par lien, il est égal à l'octet à [`Self::sweep_hb`] (lecture de toute la classe, puis écriture) SI aucune
    /// classe ne contient deux sites voisins. La réciproque n'est pas universelle (un champ froid sans torsion à β très grand
    /// peut rester identique dans les deux méthodes même avec une classe invalide) : le test du coloriage
    /// (`tests/su2_tube.rs`) vérifie donc aussi l'adjacence STRUCTURELLEMENT, comme le socle `geom`.
    pub fn sweep_hb_en_place(&mut self, beta: f64, pile: &Pile) {
        for mu in 0..4 {
            for k in 0..self.couleurs.len() {
                for i in 0..self.couleurs[k].len() {
                    let up = self.hb_lien(beta, pile, mu, self.couleurs[k][i]);
                    // cumuls SATURANTS (relecture c16b, objection 11) : jamais de débordement, même plafond atteint partout
                    self.n_tries = self.n_tries.saturating_add(up.ntry);
                    self.n_cap = self.n_cap.saturating_add(up.cap);
                    if let Some(q) = up.q {
                        self.u[up.li] = q;
                    }
                }
            }
        }
        self.sweep_id = self.sweep_id.wrapping_add(1);
    }

    /// Un balayage d'overrelaxation U → V† U† V† (V = A/|A|).
    pub fn sweep_or(&mut self, pile: &Pile) {
        let g = self.g.clone();
        for mu in 0..4 {
            for k in 0..self.couleurs.len() {
                let f = |&s: &usize| {
                    let x = g.coords(s);
                    let li = mu * g.v + s;
                    let a = agrafe(&g, &self.u, pile, mu, s, &x);
                    let kn = qnorm(a);
                    if kn.partial_cmp(&1e-300) != Some(std::cmp::Ordering::Greater) {
                        return None;
                    }
                    let vd = qdag(qscale(a, 1.0 / kn));
                    let un = qmul(qmul(vd, qdag(self.u[li])), vd);
                    Some((li, qscale(un, 1.0 / qnorm(un))))
                };
                let cl = &self.couleurs[k];
                let ups: Vec<Option<(usize, Q)>> = if cl.len() >= SEUIL_PAR {
                    cl.par_iter().map(f).collect()
                } else {
                    cl.iter().map(f).collect()
                };
                for (li, q) in ups.into_iter().flatten() {
                    self.u[li] = q;
                }
            }
        }
    }

    /// Un balayage de refroidissement : U ← A†/|A| (minimum exact de l'action locale).
    pub fn sweep_refroidir(&mut self, pile: &Pile) {
        let g = self.g.clone();
        for mu in 0..4 {
            for k in 0..self.couleurs.len() {
                let f = |&s: &usize| {
                    let x = g.coords(s);
                    let a = agrafe(&g, &self.u, pile, mu, s, &x);
                    let kn = qnorm(a);
                    if kn.partial_cmp(&1e-14) != Some(std::cmp::Ordering::Greater) {
                        return None;
                    }
                    Some((mu * g.v + s, qscale(qdag(a), 1.0 / kn)))
                };
                let cl = &self.couleurs[k];
                let ups: Vec<Option<(usize, Q)>> = if cl.len() >= SEUIL_PAR {
                    cl.par_iter().map(f).collect()
                } else {
                    cl.iter().map(f).collect()
                };
                for (li, q) in ups.into_iter().flatten() {
                    self.u[li] = q;
                }
            }
        }
    }

    /// Un pas : 1 bain thermique + `nor` OR.
    pub fn pas(&mut self, beta: f64, nor: usize, pile: &Pile) {
        self.sweep_hb(beta, pile);
        for _ in 0..nor {
            self.sweep_or(pile);
        }
    }

    /// (P, P0, P1, P2, P3) : P = (1/6V) Σ_p z_p ½Tr U_p, P_μ = Polyakov brut moyen ; mêmes arbres de réduction que
    /// `su2_hb::measure_quat` / `torsion::plaquette_tordue` (égal à l'octet à côtés égaux).
    pub fn mesure5(&self, pile: &Pile) -> [f64; 5] {
        let g = &self.g;
        let u = &self.u;
        let f = |s: usize| {
            let x = g.coords(s);
            let mut acc = 0.0;
            for mu in 0..4 {
                for nu in mu + 1..4 {
                    let p = plaquette(g, u, s, &x, mu, nu);
                    acc += pile.z(indice_plan(mu, nu), s) * p[0];
                }
            }
            acc
        };
        let site: Vec<f64> = if g.v >= SEUIL_PAR {
            (0..g.v).into_par_iter().map(f).collect()
        } else {
            (0..g.v).map(f).collect()
        };
        let part = k_reduce(&site);
        let mut out = [0f64; 5];
        out[0] = k_final(&part, 1.0 / (6.0 * g.v as f64));
        for mu in 0..4 {
            let pol = polyakov_lignes(g, u, mu);
            out[1 + mu] = k_final(&pol, 1.0 / pol.len() as f64);
        }
        out
    }

    /// Plaquettes moyennes (P_ss, P_st) : ½ReTr moyen sur les 3V plaquettes sans l'axe `temps` et les 3V qui le contiennent
    /// (torsion ignorée : à utiliser sans torsion). Chemin processeur de `--xi0`.
    pub fn plaquettes_ss_st(&self, temps: usize) -> (f64, f64) {
        let g = &self.g;
        let u = &self.u;
        let f = |s: usize| {
            let x = g.coords(s);
            let (mut a, mut b) = (0.0, 0.0);
            for mu in 0..4 {
                for nu in mu + 1..4 {
                    let p = plaquette(g, u, s, &x, mu, nu)[0];
                    if mu == temps || nu == temps {
                        b += p;
                    } else {
                        a += p;
                    }
                }
            }
            (a, b)
        };
        let site: Vec<(f64, f64)> = if g.v >= SEUIL_PAR {
            (0..g.v).into_par_iter().map(f).collect()
        } else {
            (0..g.v).map(f).collect()
        };
        let sa: Vec<f64> = site.iter().map(|t| t.0).collect();
        let sb: Vec<f64> = site.iter().map(|t| t.1).collect();
        let n = 1.0 / (3.0 * g.v as f64);
        (k_final(&k_reduce(&sa), n), k_final(&k_reduce(&sb), n))
    }

    /// S/β = Σ_p (1 − z_p ½Tr U_p).
    pub fn action(&self, pile: &Pile) -> f64 {
        action(&self.g, &self.u, pile)
    }

    /// max sur les liens de |vec(U A)|/2 (= |∂S/∂a|, a coordonnées U ↦ e^{ia·σ/2}U, comme lib_tube).
    pub fn gradient_max(&self, pile: &Pile) -> f64 {
        let g = &self.g;
        (0..4 * g.v)
            .into_par_iter()
            .map(|li| {
                let mu = li / g.v;
                let s = li % g.v;
                let x = g.coords(s);
                let a = agrafe(g, &self.u, pile, mu, s, &x);
                let m = qmul(self.u[li], a);
                0.5 * m[1].abs().max(m[2].abs()).max(m[3].abs())
            })
            .reduce(|| 0.0, f64::max)
    }
}

/// ½Tr des boucles de Polyakov brutes dans la direction μ, une par site de base (x_μ = 0), dans l'ordre de
/// `su2_hb::measure_quat` (directions restantes d0 < d1 < d2, x[d2] le plus rapide).
pub fn polyakov_lignes(g: &GeoA, u: &[Q], mu: usize) -> Vec<f64> {
    let dirs: Vec<usize> = (0..4).filter(|&d| d != mu).collect();
    let n = g.v / g.d[mu];
    let f = |j: usize| {
        let mut x = [0usize; 4];
        x[dirs[2]] = j % g.d[dirs[2]];
        let t = j / g.d[dirs[2]];
        x[dirs[1]] = t % g.d[dirs[1]];
        x[dirs[0]] = t / g.d[dirs[1]];
        let mut s = g.site(&x);
        let mut p: Q = [1.0, 0.0, 0.0, 0.0];
        for k in 0..g.d[mu] {
            p = qmul(p, u[mu * g.v + s]);
            if k != g.d[mu] - 1 {
                s += g.st[mu];
            }
        }
        p[0]
    };
    if g.v >= SEUIL_PAR {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// S/β = Σ_p (1 − z_p ½Tr U_p), somme déterministe (blocs de 4096 sites).
pub fn action(g: &GeoA, u: &[Q], pile: &Pile) -> f64 {
    let f = |b: usize| {
        let mut acc = 0.0;
        for s in b * 4096..((b + 1) * 4096).min(g.v) {
            let x = g.coords(s);
            for (p, &(mu, nu)) in PLANS.iter().enumerate() {
                acc += 1.0 - pile.z(p, s) * plaquette(g, u, s, &x, mu, nu)[0];
            }
        }
        acc
    };
    let nb = g.v.div_ceil(4096);
    let parts: Vec<f64> = if nb > 1 {
        (0..nb).into_par_iter().map(f).collect()
    } else {
        (0..nb).map(f).collect()
    };
    parts.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commun::{Depart, Graine, Precision};
    use crate::gpu::OptionsCompil;
    use crate::su2::cpu::Cpu64;
    use crate::su2::torsion::Torsion;
    use crate::su2::{ConfigLot, NoyauxSu2, ParamChaine};

    fn lot(l: usize, beta: f64, graine: u64, nor: usize) -> ConfigLot {
        ConfigLot {
            l,
            chaines: vec![ParamChaine {
                beta,
                graine: Graine(graine),
                depart: Depart::Chaud,
            }],
            nor,
            precision: Precision::F64,
            options: OptionsCompil::reference_k1(),
            mutant_metro: false,
        }
    }

    fn egal_a_cpu64(torsion: &str, beta: f64) {
        let t = Torsion::analyser(torsion).unwrap();
        let mut a = Cpu64::nouveau(lot(4, beta, 17, 2)).unwrap();
        a.init().unwrap();
        a.fixer_torsion(&t).unwrap();
        a.allouer_historique(1).unwrap();
        let g = GeoA::nouvelle([4, 4, 4, 4]).unwrap();
        let pile = Pile::depuis_torsion(&g, &t);
        let mut c = ChaineTube::nouvelle(g, 17);
        c.init(true);
        for _ in 0..3 {
            a.sweep_hb().unwrap();
            a.sweep_or().unwrap();
            a.sweep_or().unwrap();
            c.pas(beta, 2, &pile);
        }
        let qa = a.telecharger(0).unwrap().q;
        let qc: Vec<f64> = c.u.iter().flat_map(|q| q.iter().copied()).collect();
        assert_eq!(qa, qc, "torsion {torsion}");
        a.mesurer(0).unwrap();
        let h = a.historique().unwrap();
        assert_eq!(&h[0..5], &c.mesure5(&pile), "mesures, torsion {torsion}");
        assert_eq!(a.compteurs().unwrap()[0].essais, c.n_tries);
    }

    #[test]
    fn egal_a_cpu64_sans_torsion() {
        egal_a_cpu64("aucune", 2.3);
        egal_a_cpu64("aucune", 0.7);
    }

    #[test]
    fn egal_a_cpu64_avec_torsion() {
        egal_a_cpu64("01,23", 2.3);
        egal_a_cpu64("23", 8.0);
    }

    #[test]
    fn vide_mince_et_abelien() {
        for (l, ff) in [(4usize, 2usize), (6, 3), (4, 4)] {
            let g = GeoA::nouvelle([l, l, ff, ff]).unwrap();
            let p23 = Pile::depuis_torsion(&g, &Torsion::analyser("23").unwrap());
            let p2 = Pile::depuis_torsion(&g, &Torsion::analyser("01,23").unwrap());
            let mut c = ChaineTube::nouvelle(g.clone(), 1);
            for (k0, k1) in [(true, true), (false, true), (true, false), (false, false)] {
                c.vide(k0, k1);
                assert!(c.action(&p23).abs() < 1e-12);
            }
            c.vide(true, true);
            assert!((c.action(&p2) - 2.0 * (ff * ff) as f64).abs() < 1e-12);
            c.abelien();
            let n = (l * l * ff * ff) as f64;
            let attendu =
                n * ((1.0 - (PI / (l * l) as f64).cos()) + (1.0 - (PI / (ff * ff) as f64).cos()));
            assert!(
                (c.action(&p2) - attendu).abs() < 1e-9,
                "{} {}",
                c.action(&p2),
                attendu
            );
        }
    }

    #[test]
    fn gradient_nul_au_vide_et_refroidissement_descend() {
        let g = GeoA::nouvelle([4, 4, 2, 2]).unwrap();
        let p2 = Pile::depuis_torsion(&g, &Torsion::analyser("01,23").unwrap());
        let mut c = ChaineTube::nouvelle(g, 3);
        c.vide(true, true);
        c.bruiter(1e-3);
        let s0 = c.action(&p2);
        for _ in 0..3000 {
            c.sweep_refroidir(&p2);
        }
        let s1 = c.action(&p2);
        assert!(s1 < s0);
        assert!((s1 - 6.740116866).abs() < 1e-6, "S_min(4,2) = {s1}");
        assert!(c.gradient_max(&p2) < 1e-6);
    }
}
