//! Référence CPU FP64 (crate::heritage::su2_hb), lots et gel. Propriétaire : T3.
//!
//! `Cpu64` porte une `CpuSu2` par chaîne du lot.
//! - SANS gel ni torsion, tout est délégué à `heritage::su2_hb::CpuSu2` : le résultat est bit-identique à la référence K1.
//! - AVEC gel (R2-3) ou torsion de 't Hooft (M4), la mise à jour d'un lien est RÉÉCRITE à partir des primitives publiques de su2_hb
//!   (`philox4x32_10`, `u01`, `staple`, `sample_x0`, `qmul`…) en reproduisant `haar_from` et `fmax0`
//!   (privés dans su2_hb) : les liens spatiaux de l'hyperplan x0 = 0 sont sautés, comme le
//!   `if (gel && mu != 0 && x[0] == 0) return;` des noyaux GPU. Le test `reecriture_sans_gel_egale_cpusu2`
//!   prouve que cette réécriture, sans gel, est égale à l'octet à `CpuSu2`.
//!   Avec torsion, `staple` est remplacé par `torsion::agrafe_tordue`.

use super::torsion::{self, Torsion};
use super::{AppMasque, Compteurs, ConfigLot, Gel, NoyauxSu2};
use crate::commun::{BackendKind, ChampHote, Erreur, Precision, Resultat, ETIQUETTE_HB};
use crate::gpu::ProvenanceCompil;
use crate::heritage::su2_hb::{self, CpuSu2, Geo, Q};
use rayon::prelude::*;
use std::f64::consts::PI;

/// Générateur CPU FP64 (rayon), implémente `NoyauxSu2`. Sans gel ni torsion : bit-identique à
/// heritage::su2_hb::CpuSu2 ; avec gel ou torsion : mêmes primitives, liens gelés sautés,
/// agrafe tordue.
pub struct Cpu64 {
    pub cfg: ConfigLot,
    chaines: Vec<CpuSu2>,
    gel: Gel,
    /// Masque de liens gelés (WP-DIR) : un octet par lien, 4V octets, partagé par les chaînes du lot.
    masque: Option<Vec<u8>>,
    app_masque: AppMasque,
    /// Torsion de 't Hooft (triviale par défaut).
    torsion: Torsion,
    hist: Vec<f64>,
    hcap: usize,
    tranches: Vec<f64>,
    tcap: usize,
}

impl Cpu64 {
    pub fn nouveau(cfg: ConfigLot) -> Resultat<Cpu64> {
        super::valider(&cfg)?;
        if cfg.precision != Precision::F64 {
            return Err(Erreur::Config("le CPU ne calcule qu'en F64 (précision f32 : GPU requis)".into()));
        }
        let mut chaines = Vec::with_capacity(cfg.chaines.len());
        for p in &cfg.chaines {
            let mut c = CpuSu2::new(cfg.l, p.graine.0).map_err(Erreur::Config)?;
            c.mutant_metro = cfg.mutant_metro;
            chaines.push(c);
        }
        Ok(Cpu64 {
            cfg,
            chaines,
            gel: Gel::Aucun,
            masque: None,
            app_masque: AppMasque::Tout,
            torsion: Torsion::triviale(),
            hist: Vec::new(),
            hcap: 0,
            tranches: Vec::new(),
            tcap: 0,
        })
    }

    /// Un sweep de bain thermique qui rend, par chaîne, les essais par lien (diagnostic `diag_ntry` du GPU) :
    /// 4V entrées par chaîne ; 0 pour un lien gelé ou un tirage au plafond, 0xFFFFFFFF pour une agrafe nulle.
    pub fn sweep_hb_diag(&mut self) -> Resultat<Vec<u32>> {
        let v4 = 4 * self.cfg.l.pow(4);
        let mut diag = vec![0u32; v4 * self.chaines.len()];
        self.hb(Some(&mut diag));
        Ok(diag)
    }

    fn hb(&mut self, diag: Option<&mut [u32]>) {
        let v4 = 4 * self.cfg.l.pow(4);
        let gel = self.gel;
        let masque = if self.app_masque.bain() { self.masque.as_deref() } else { None };
        let torsion = self.torsion;
        let mut diag = diag;
        for (c, sc) in self.chaines.iter_mut().enumerate() {
            let beta = self.cfg.chaines[c].beta;
            let d = diag.as_deref_mut().map(|d| &mut d[c * v4..(c + 1) * v4]);
            // sans gel, ni masque, ni torsion : chemin historique (octets de la référence K1) ;
            // sinon la réécriture avec liens sautés et agrafe tordue.
            if gel == Gel::Aucun && masque.is_none() && torsion.est_triviale() {
                sc.sweep_hb(beta, d);
            } else {
                hb_gel(sc, beta, gel, masque, &torsion, d);
            }
        }
    }

    /// Sweep de bain thermique par la réécriture (même sans gel) : sert à prouver qu'elle égale `CpuSu2`.
    pub fn sweep_hb_reecrit(&mut self, gel: Gel) {
        for (c, sc) in self.chaines.iter_mut().enumerate() {
            hb_gel(sc, self.cfg.chaines[c].beta, gel, None, &Torsion::triviale(), None);
        }
    }

    /// Sweep d'overrelaxation par la réécriture (même sans gel).
    pub fn sweep_or_reecrit(&mut self, gel: Gel) {
        for sc in self.chaines.iter_mut() {
            or_gel(sc, gel, None, &Torsion::triviale());
        }
    }

    /// Sweep de bain thermique réécrit avec la torsion `t` (toujours, même triviale).
    pub fn sweep_hb_tordu(&mut self, t: &Torsion) {
        for (c, sc) in self.chaines.iter_mut().enumerate() {
            hb_gel(sc, self.cfg.chaines[c].beta, Gel::Aucun, None, t, None);
        }
    }

    /// Sweep d'overrelaxation réécrit avec la torsion `t` (toujours, même triviale).
    pub fn sweep_or_tordu(&mut self, t: &Torsion) {
        for sc in self.chaines.iter_mut() {
            or_gel(sc, Gel::Aucun, None, t);
        }
    }

    /// Champ de la chaîne `c` (quaternions, li = mu·V + s).
    pub fn quaternions(&self, c: usize) -> &[Q] {
        &self.chaines[c].u
    }
}

// ------------------------------------------------------------------------------------------
// Mise à jour d'un lien, avec gel (réécriture de CpuSu2::hb_link et du sweep d'OR)
// ------------------------------------------------------------------------------------------

#[inline]
fn lien_gele(gel: Gel, mu: usize, x0: usize) -> bool {
    gel == Gel::BordX0Spatial && mu != 0 && x0 == 0
}

#[inline]
fn tirage(li: u32, sweep_id: u32, draw: &mut u32, k0: u32, k1: u32) -> [u32; 4] {
    let r = su2_hb::philox4x32_10([li, sweep_id, *draw, ETIQUETTE_HB], k0, k1);
    *draw = draw.wrapping_add(1);
    r
}

/// fmax(0.0, v) du C pour v fini.
#[inline]
fn fmax0(v: f64) -> f64 {
    if v > 0.0 {
        v
    } else {
        0.0
    }
}

/// Quaternion de Haar depuis deux tirages Philox (haar_from de su2_hb, privé là-bas).
fn haar_from(r1: [u32; 4], r2: [u32; 4]) -> Q {
    let u1 = su2_hb::u01(r1[0], r1[1]);
    let u2 = su2_hb::u01(r1[2], r1[3]);
    let u3 = su2_hb::u01(r2[0], r2[1]);
    let u4 = su2_hb::u01(r2[2], r2[3]);
    let ra = (-2.0 * u1.ln()).sqrt();
    let rb = (-2.0 * u3.ln()).sqrt();
    let q = [ra * (2.0 * PI * u2).cos(), ra * (2.0 * PI * u2).sin(), rb * (2.0 * PI * u4).cos(), rb * (2.0 * PI * u4).sin()];
    su2_hb::qscale(q, 1.0 / su2_hb::qnorm(q))
}

/// Quaternion de Haar de compteur (i, slot, 0|1, tag) : `haar_q` de su2_bord.cu côté hôte (bord chaud, jauge).
pub fn haar_q(i: u32, slot: u32, tag: u32, k0: u32, k1: u32) -> Q {
    let r1 = su2_hb::philox4x32_10([i, slot, 0, tag], k0, k1);
    let r2 = su2_hb::philox4x32_10([i, slot, 1, tag], k0, k1);
    haar_from(r1, r2)
}

/// Résultat de la mise à jour d'un lien : (indice, nouveau lien ou None, essais, plafond, diagnostic).
struct Maj {
    li: usize,
    q: Option<Q>,
    ntry: u64,
    cap: u64,
    diag: u32,
}

// M4 (intégrateur) : la torsion s'ajoute aux sept paramètres du chemin réécrit ; un regroupement changerait les appels existants.
#[allow(clippy::too_many_arguments)]
fn hb_lien(sc: &CpuSu2, beta: f64, gel: Gel, masque: Option<&[u8]>, torsion: &Torsion, mu: usize, parity: usize, t: usize) -> Option<Maj> {
    let g: &Geo = &sc.geo;
    let (s, x) = g.site_of(t, parity);
    if lien_gele(gel, mu, x[0]) {
        return None;
    }
    let li = mu * g.v + s;
    if masque.is_some_and(|m| m[li] != 0) {
        return None; // lien gelé par le masque (boîte à bord fixé)
    }
    let a = torsion::agrafe_tordue(&sc.u, g, mu, s, &x, torsion);
    let k = su2_hb::qnorm(a);
    let mut draw: u32 = 0;
    let sid = sc.sweep_id;
    let (k0, k1) = (sc.k0, sc.k1);
    if k.partial_cmp(&1e-300) != Some(std::cmp::Ordering::Greater) {
        let r1 = tirage(li as u32, sid, &mut draw, k0, k1);
        let r2 = tirage(li as u32, sid, &mut draw, k0, k1);
        return Some(Maj { li, q: Some(haar_from(r1, r2)), ntry: 0, cap: 0, diag: 0xFFFF_FFFF });
    }
    let vv = su2_hb::qscale(a, 1.0 / k);
    let alpha = beta * k;
    let (x0, ntry, ok) = if sc.mutant_metro {
        su2_hb::sample_x0_mut(alpha, su2_hb::qmul(sc.u[li], vv)[0], li as u32, sid, &mut draw, k0, k1)
    } else {
        su2_hb::sample_x0(alpha, li as u32, sid, &mut draw, k0, k1)
    };
    if !ok {
        return Some(Maj { li, q: None, ntry: ntry as u64, cap: 1, diag: 0 });
    }
    let rd = tirage(li as u32, sid, &mut draw, k0, k1);
    let ct = 2.0 * su2_hb::u01(rd[0], rd[1]) - 1.0;
    let phi = 2.0 * PI * su2_hb::u01(rd[2], rd[3]);
    let st = fmax0(1.0 - ct * ct).sqrt();
    let rr = fmax0(1.0 - x0 * x0).sqrt();
    let w = [x0, rr * st * phi.cos(), rr * st * phi.sin(), rr * ct];
    let un = su2_hb::qmul(w, su2_hb::qdag(vv));
    Some(Maj { li, q: Some(su2_hb::qscale(un, 1.0 / su2_hb::qnorm(un))), ntry: ntry as u64, cap: 0, diag: ntry })
}

/// Un sweep de bain thermique d'une chaîne, liens gelés sautés, agrafe tordue (4 mu × 2 parités, lecture puis écriture :
/// le calcul en deux phases est exactement le calcul en place du damier).
fn hb_gel(sc: &mut CpuSu2, beta: f64, gel: Gel, masque: Option<&[u8]>, torsion: &Torsion, mut diag: Option<&mut [u32]>) {
    let half = sc.geo.v / 2;
    for mu in 0..4 {
        for p in 0..2 {
            let ups: Vec<Option<Maj>> = (0..half).into_par_iter().map(|t| hb_lien(sc, beta, gel, masque, torsion, mu, p, t)).collect();
            for up in ups.into_iter().flatten() {
                sc.n_tries += up.ntry;
                sc.n_cap += up.cap;
                if let Some(d) = diag.as_deref_mut() {
                    d[up.li] = up.diag;
                }
                if let Some(q) = up.q {
                    sc.u[up.li] = q;
                }
            }
        }
    }
    sc.n_hb += 1;
    sc.sweep_id = sc.sweep_id.wrapping_add(1);
}

/// Un sweep d'overrelaxation d'une chaîne, liens gelés sautés, agrafe tordue.
fn or_gel(sc: &mut CpuSu2, gel: Gel, masque: Option<&[u8]>, torsion: &Torsion) {
    let g = sc.geo;
    let half = g.v / 2;
    for mu in 0..4 {
        for p in 0..2 {
            let ups: Vec<Option<(usize, Q)>> = (0..half)
                .into_par_iter()
                .map(|t| {
                    let (s, x) = g.site_of(t, p);
                    if lien_gele(gel, mu, x[0]) {
                        return None;
                    }
                    let li = mu * g.v + s;
                    if masque.is_some_and(|m| m[li] != 0) {
                        return None; // lien gelé par le masque (boîte à bord fixé)
                    }
                    let a = torsion::agrafe_tordue(&sc.u, &g, mu, s, &x, torsion);
                    let k = su2_hb::qnorm(a);
                    if k.partial_cmp(&1e-300) != Some(std::cmp::Ordering::Greater) {
                        return None;
                    }
                    let vd = su2_hb::qdag(su2_hb::qscale(a, 1.0 / k));
                    let un = su2_hb::qmul(su2_hb::qmul(vd, su2_hb::qdag(sc.u[li])), vd);
                    Some((li, su2_hb::qscale(un, 1.0 / su2_hb::qnorm(un))))
                })
                .collect();
            for (li, q) in ups.into_iter().flatten() {
                sc.u[li] = q;
            }
        }
    }
    sc.n_or += 1;
}

// ------------------------------------------------------------------------------------------
// Tranches R2-3 (hôte) : mêmes expressions et même arbre de réduction que k_plaq_st / k_tranche
// ------------------------------------------------------------------------------------------

/// S_0..S_{L-1}, T_0..T_{L-1} d'un champ de quaternions (2L valeurs) : moyennes par tranche x0 = t de
/// 1/2 ReTr des plaquettes spatiales (S) et temporelles (T) ; somme à arbre fixe (k_tranche).
pub fn tranches_hote(g: &Geo, u: &[Q]) -> Vec<f64> {
    let ld = |nu: usize, site: usize| u[nu * g.v + site];
    let sites: Vec<(f64, f64)> = (0..g.v)
        .into_par_iter()
        .map(|s| {
            let x = g.coords(s);
            let (mut a_s, mut a_t) = (0.0, 0.0);
            for mu in 0..4 {
                let smu = g.nb_up(s, mu, x[mu]);
                for (nu, &xn) in x.iter().enumerate().skip(mu + 1) {
                    let snu = g.nb_up(s, nu, xn);
                    let p = su2_hb::qmul(
                        su2_hb::qmul(ld(mu, s), ld(nu, smu)),
                        su2_hb::qmul(su2_hb::qdag(ld(mu, snu)), su2_hb::qdag(ld(nu, s))),
                    );
                    if mu == 0 {
                        a_t += p[0];
                    } else {
                        a_s += p[0];
                    }
                }
            }
            (a_s, a_t)
        })
        .collect();
    let l = g.l;
    let l3 = g.v / l;
    let sc = 1.0 / (3.0 * l3 as f64);
    let ss: Vec<f64> = sites.iter().map(|x| x.0).collect();
    let tt: Vec<f64> = sites.iter().map(|x| x.1).collect();
    let mut out = vec![0.0; 2 * l];
    for t in 0..l {
        out[t] = su2_hb::k_final(&ss[t * l3..(t + 1) * l3], sc);
        out[l + t] = su2_hb::k_final(&tt[t * l3..(t + 1) * l3], sc);
    }
    out
}

impl NoyauxSu2 for Cpu64 {
    fn precision(&self) -> Precision {
        Precision::F64
    }
    fn backend(&self) -> BackendKind {
        BackendKind::Cpu
    }
    fn config(&self) -> &ConfigLot {
        &self.cfg
    }
    fn init(&mut self) -> Resultat<()> {
        for (c, sc) in self.chaines.iter_mut().enumerate() {
            sc.init(self.cfg.chaines[c].depart == crate::commun::Depart::Chaud);
            sc.sweep_id = 0;
            sc.n_cap = 0;
            sc.n_tries = 0;
            sc.n_hb = 0;
            sc.n_or = 0;
        }
        Ok(())
    }
    fn fixer_gel(&mut self, gel: Gel) -> Resultat<()> {
        if gel != Gel::Aucun && !self.torsion.est_triviale() {
            return Err(Erreur::Config("gel incompatible avec une torsion de 't Hooft non triviale".into()));
        }
        self.gel = gel;
        Ok(())
    }
    fn fixer_masque(&mut self, masque: Option<&[u8]>, application: AppMasque) -> Resultat<()> {
        if masque.is_some() && !self.torsion.est_triviale() {
            return Err(Erreur::Config("masque de liens gelés incompatible avec une torsion de 't Hooft non triviale".into()));
        }
        self.app_masque = application;
        match masque {
            None => self.masque = None,
            Some(m) => {
                let attendu = 4 * self.cfg.l.pow(4);
                if m.len() != attendu {
                    return Err(Erreur::Config(format!("masque de {} octets : {attendu} attendus (4·L⁴, L = {})", m.len(), self.cfg.l)));
                }
                self.masque = Some(m.to_vec());
            }
        }
        Ok(())
    }
    fn fixer_torsion(&mut self, t: &Torsion) -> Resultat<()> {
        if !t.est_triviale() {
            if self.cfg.chaines.iter().any(|c| c.beta == 0.0) {
                return Err(Erreur::Config("torsion de 't Hooft : β = 0 interdit (bain thermique historique singulier)".into()));
            }
            if self.gel != Gel::Aucun {
                return Err(Erreur::Config("torsion de 't Hooft incompatible avec le gel (bord)".into()));
            }
            if self.masque.is_some() {
                return Err(Erreur::Config("torsion de 't Hooft incompatible avec le masque de liens gelés".into()));
            }
        }
        self.torsion = *t;
        Ok(())
    }
    fn sweep_hb(&mut self) -> Resultat<()> {
        self.hb(None);
        Ok(())
    }
    fn sweep_or(&mut self) -> Resultat<()> {
        let gel = self.gel;
        let masque = if self.app_masque.surrelax() { self.masque.as_deref() } else { None };
        let torsion = self.torsion;
        for sc in self.chaines.iter_mut() {
            if gel == Gel::Aucun && masque.is_none() && torsion.est_triviale() {
                sc.sweep_or();
            } else {
                or_gel(sc, gel, masque, &torsion);
            }
        }
        Ok(())
    }
    fn allouer_historique(&mut self, capacite: usize) -> Resultat<()> {
        self.hcap = capacite;
        self.hist = vec![0.0; self.chaines.len() * capacite * 5];
        Ok(())
    }
    fn mesurer(&mut self, idx: usize) -> Resultat<()> {
        if idx >= self.hcap {
            return Err(Erreur::Config(format!("historique plein : idx {idx} ≥ capacité {}", self.hcap)));
        }
        for (c, sc) in self.chaines.iter().enumerate() {
            let mut m = sc.measure();
            if !self.torsion.est_triviale() {
                m[0] = torsion::plaquette_tordue(&sc.geo, &sc.u, &self.torsion);
            }
            let o = (c * self.hcap + idx) * 5;
            self.hist[o..o + 5].copy_from_slice(&m);
        }
        Ok(())
    }
    fn historique(&self) -> Resultat<Vec<f64>> {
        Ok(self.hist.clone())
    }
    fn mesurer_tranches(&mut self, idx: usize, capacite: usize) -> Resultat<()> {
        let l2 = 2 * self.cfg.l;
        if self.tranches.is_empty() || capacite > self.tcap {
            let ancien = std::mem::take(&mut self.tranches);
            let ancienne_cap = self.tcap;
            self.tcap = capacite;
            self.tranches = vec![0.0; self.chaines.len() * capacite * l2];
            for c in 0..self.chaines.len() {
                // conserve les cases déjà écrites si l'historique grossit
                if ancienne_cap > 0 {
                    let n = ancienne_cap * l2;
                    self.tranches[c * capacite * l2..c * capacite * l2 + n].copy_from_slice(&ancien[c * n..(c + 1) * n]);
                }
            }
        }
        if idx >= self.tcap {
            return Err(Erreur::Config(format!("historique de tranches plein : idx {idx} ≥ capacité {}", self.tcap)));
        }
        for (c, sc) in self.chaines.iter().enumerate() {
            let m = tranches_hote(&sc.geo, &sc.u);
            let o = (c * self.tcap + idx) * l2;
            self.tranches[o..o + l2].copy_from_slice(&m);
        }
        Ok(())
    }
    fn historique_tranches(&self) -> Resultat<Vec<f64>> {
        Ok(self.tranches.clone())
    }
    fn telecharger(&self, chaine: usize) -> Resultat<ChampHote> {
        let sc = self.chaines.get(chaine).ok_or_else(|| Erreur::Config(format!("chaîne {chaine} hors du lot")))?;
        Ok(ChampHote { l: self.cfg.l, q: sc.flat() })
    }
    fn televerser(&mut self, chaine: usize, c: &ChampHote) -> Resultat<()> {
        if c.l != self.cfg.l {
            return Err(Erreur::Config(format!("champ de L = {} dans un lot de L = {}", c.l, self.cfg.l)));
        }
        let sc = self.chaines.get_mut(chaine).ok_or_else(|| Erreur::Config(format!("chaîne {chaine} hors du lot")))?;
        sc.set_flat(&c.q).map_err(Erreur::Config)
    }
    fn champ_gpu(&self) -> Option<&super::ChampGpu> {
        None
    }
    fn compteurs(&self) -> Resultat<Vec<Compteurs>> {
        Ok(self.chaines.iter().map(|c| Compteurs { plafond: c.n_cap, essais: c.n_tries }).collect())
    }
    fn sweep_id(&self) -> u32 {
        self.chaines.first().map_or(0, |c| c.sweep_id)
    }
    fn fixer_sweep_id(&mut self, s: u32) {
        for c in self.chaines.iter_mut() {
            c.sweep_id = s;
        }
    }
    fn synchroniser(&self) -> Resultat<()> {
        Ok(())
    }
    fn provenance(&self) -> Vec<ProvenanceCompil> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commun::{Depart, Graine};
    use crate::gpu::OptionsCompil;
    use crate::su2::ParamChaine;

    fn lot(l: usize, betas: &[f64], graines: &[u64]) -> ConfigLot {
        ConfigLot {
            l,
            chaines: betas
                .iter()
                .zip(graines)
                .map(|(&beta, &g)| ParamChaine { beta, graine: Graine(g), depart: Depart::Chaud })
                .collect(),
            nor: 2,
            precision: Precision::F64,
            options: OptionsCompil::reference_k1(),
            mutant_metro: false,
        }
    }

    #[test]
    fn sans_gel_egal_a_cpusu2() {
        let mut a = Cpu64::nouveau(lot(4, &[2.3], &[11])).unwrap();
        a.init().unwrap();
        let mut r = CpuSu2::new(4, 11).unwrap();
        r.init(true);
        for _ in 0..3 {
            a.sweep_hb().unwrap();
            a.sweep_or().unwrap();
            a.sweep_or().unwrap();
            r.step(2.3, 2);
        }
        assert_eq!(a.telecharger(0).unwrap().q, r.flat());
    }

    #[test]
    fn reecriture_sans_gel_egale_cpusu2() {
        // alpha < 2 (Creutz) et alpha >= 2 (Kennedy-Pendleton) sont couverts par beta = 0.3 et 2.3
        for beta in [0.3, 2.3, 8.0] {
            let mut a = Cpu64::nouveau(lot(4, &[beta], &[5])).unwrap();
            a.init().unwrap();
            let mut r = CpuSu2::new(4, 5).unwrap();
            r.init(true);
            for _ in 0..3 {
                a.sweep_hb_reecrit(Gel::Aucun);
                a.sweep_or_reecrit(Gel::Aucun);
                r.sweep_hb(beta, None);
                r.sweep_or();
            }
            assert_eq!(a.telecharger(0).unwrap().q, r.flat(), "beta = {beta}");
            assert_eq!(a.sweep_id(), r.sweep_id);
            assert_eq!(a.compteurs().unwrap()[0].essais, r.n_tries);
        }
    }

    #[test]
    fn le_gel_ne_touche_pas_au_bord_et_touche_le_reste() {
        let l = 4;
        let mut a = Cpu64::nouveau(lot(l, &[2.3], &[7])).unwrap();
        a.init().unwrap();
        let avant = a.telecharger(0).unwrap();
        a.fixer_gel(Gel::BordX0Spatial).unwrap();
        for _ in 0..2 {
            a.sweep_hb().unwrap();
            a.sweep_or().unwrap();
        }
        let apres = a.telecharger(0).unwrap();
        let v = l * l * l * l;
        let l3 = v / l;
        let mut gele_ok = true;
        let mut libre_change = 0usize;
        for mu in 0..4 {
            for s in 0..v {
                let li = mu * v + s;
                let x0 = s / l3;
                let egal = avant.lien(li) == apres.lien(li);
                if mu != 0 && x0 == 0 {
                    gele_ok &= egal;
                } else if !egal {
                    libre_change += 1;
                }
            }
        }
        assert!(gele_ok, "un lien gelé a bougé");
        assert!(libre_change > 3 * v, "les liens libres doivent évoluer ({libre_change})");
        assert!(apres.defaut_unitarite() < 1e-14);
    }

    #[test]
    fn lot_egal_aux_chaines_seules() {
        let l = 4;
        let mut lot4 = Cpu64::nouveau(lot(l, &[2.3, 2.8, 3.5, 8.0], &[1, 2, 3, 4])).unwrap();
        lot4.init().unwrap();
        for _ in 0..3 {
            lot4.sweep_hb().unwrap();
            lot4.sweep_or().unwrap();
        }
        for (c, (b, g)) in [(2.3, 1u64), (2.8, 2), (3.5, 3), (8.0, 4)].into_iter().enumerate() {
            let mut seul = Cpu64::nouveau(lot(l, &[b], &[g])).unwrap();
            seul.init().unwrap();
            for _ in 0..3 {
                seul.sweep_hb().unwrap();
                seul.sweep_or().unwrap();
            }
            assert_eq!(lot4.telecharger(c).unwrap().q, seul.telecharger(0).unwrap().q, "chaîne {c}");
        }
    }

    #[test]
    fn mesures_et_historique_par_chaine() {
        let mut a = Cpu64::nouveau(lot(4, &[2.3, 2.8], &[1, 2])).unwrap();
        a.init().unwrap();
        a.allouer_historique(3).unwrap();
        a.mesurer(0).unwrap();
        a.sweep_hb().unwrap();
        a.mesurer(1).unwrap();
        assert!(a.mesurer(3).is_err());
        let h = a.historique().unwrap();
        assert_eq!(h.len(), 2 * 3 * 5);
        // chaîne 1, idx 1 : plaquette de la chaîne 1 (début à 0 < p < 1)
        let p = h[(3 + 1) * 5];
        assert!(p > 0.0 && p < 1.0, "p = {p}");
        let m = su2_hb::measure_quat(&a.chaines[1].geo, &a.chaines[1].u);
        assert_eq!(&h[(3 + 1) * 5..(3 + 1) * 5 + 5], &m);
    }

    #[test]
    fn tranches_froid_et_gel() {
        // champ froid : toutes les plaquettes valent 1, donc S_t = T_t = 1
        let mut cfg = lot(4, &[2.3], &[1]);
        cfg.chaines[0].depart = Depart::Froid;
        let mut a = Cpu64::nouveau(cfg).unwrap();
        a.init().unwrap();
        a.mesurer_tranches(0, 2).unwrap();
        let t = a.historique_tranches().unwrap();
        assert_eq!(t.len(), 2 * 8); // capacité 2 × 2L = 8 valeurs
        assert!(t[..8].iter().all(|&x| (x - 1.0).abs() < 1e-15), "{t:?}");
    }

    #[test]
    fn le_diagnostic_signale_les_gels_et_les_agrafes() {
        let mut a = Cpu64::nouveau(lot(4, &[2.3], &[3])).unwrap();
        a.init().unwrap();
        a.fixer_gel(Gel::BordX0Spatial).unwrap();
        let d = a.sweep_hb_diag().unwrap();
        let v = 256;
        // liens gelés : diagnostic nul ; liens libres : au moins un essai
        for mu in 0..4 {
            for s in 0..v {
                let d_li = d[mu * v + s];
                if mu != 0 && s < v / 4 {
                    assert_eq!(d_li, 0);
                } else {
                    assert!(d_li >= 1 && d_li != 0xFFFF_FFFF);
                }
            }
        }
    }

    #[test]
    fn refus_f32_et_lot_vide() {
        let mut c = lot(4, &[2.3], &[1]);
        c.precision = Precision::F32;
        assert!(Cpu64::nouveau(c).is_err());
        let mut c = lot(4, &[2.3], &[1]);
        c.chaines.clear();
        assert!(Cpu64::nouveau(c).is_err());
        assert!(Cpu64::nouveau(lot(5, &[2.3], &[1])).is_err());
    }
}
