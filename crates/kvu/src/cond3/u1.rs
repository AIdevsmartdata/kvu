//! Fibre dure de DEC pour le U(1) GAUSSIEN non compact (SPEC_T3 §3.1 « interface de groupe », ORACLES O-4).
//!
//! Liens réels θ, S = (β/2) Σ_p F_p², F = rot θ (F_p = θ_μ(x) + θ_ν(x+e_μ) − θ_μ(x+e_ν) − θ_ν(x), plans μ < ν) ; carte DEC :
//! φ̄_ē = θ₁ + θ₂ + θ₃ ; observable O_p = F_p ; score d'un relèvement k : ∂S/∂θ_k (les trois relèvements décalent φ̄ de +t).
//!
//! Agrafe K_ℓ = Σ_{p ∋ ℓ} s_p r_p (F_p = s_p θ_ℓ + r_p, s_p = ±1), donc Σ_{p ∋ ℓ} s_p F_p = n_p θ_ℓ + K_ℓ, n_p = 2(d − 1).
//! Lien libre : θ ~ N(−K/n_p, 1/(β n_p)). Ligne, U₁ (θ₂ fixé, θ₃ = φ̄ − θ₁ − θ₂) : N(m, 1/(2 β n_p)),
//! m = (n_p (φ̄ − θ₂) + K₃ − K₁)/(2 n_p) ; θ₂ (θ₁ fixé) : m = (n_p (φ̄ − θ₁) + K₃ − K₂)/(2 n_p). Surrelaxation : θ → 2m − θ.

use super::geom::Geom;
use super::modele::{Modele, StatsEchantillonneur};
use crate::commun::ETIQUETTE_COND;
use crate::heritage::su2_hb::{philox4x32_10, sha256_hex, u01};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::Arc;

pub struct FibreU1 {
    g: Arc<Geom>,
    pub th: Vec<f64>,
    pub phi: Vec<f64>,
    beta: f64,
    k0: u32,
    k1: u32,
    stats: StatsEchantillonneur,
    npl: f64,
}

/// Agrafe gaussienne K_ℓ du lien (μ, s).
#[inline]
pub fn agrafe_u1(th: &[f64], g: &Geom, mu: usize, s: usize) -> f64 {
    let v = g.v;
    let smu = g.nbr[mu][s] as usize;
    let mut k = 0.0;
    for nu in 0..g.d {
        if nu == mu {
            continue;
        }
        let snu = g.nbr[nu][s] as usize;
        let sm = g.nbm[nu][s] as usize;
        let smm = g.nbr[mu][sm] as usize;
        k += th[nu * v + smu] - th[mu * v + snu] - th[nu * v + s];
        k -= th[mu * v + sm] + th[nu * v + smm] - th[nu * v + sm];
    }
    k
}

/// Gaussienne standard (Box–Muller) ; `u01` ∈ ]0, 1[ (jamais 0), donc `ln` est fini.
#[inline]
fn gauss(li: u32, sweep: u32, k0: u32, k1: u32) -> f64 {
    let r = philox4x32_10([li, sweep, 0xA11C_E000, ETIQUETTE_COND], k0, k1);
    let (a, b) = (u01(r[0], r[1]), u01(r[2], r[3]));
    (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos()
}

impl FibreU1 {
    pub fn nouveau(g: Arc<Geom>, th: Vec<f64>, beta: f64, graine: u64) -> FibreU1 {
        assert_eq!(th.len(), g.nliens());
        let (k0, k1) = crate::commun::Graine(graine).cle();
        let npl = (2 * (g.d - 1)) as f64;
        let mut f = FibreU1 { g, th, phi: Vec::new(), beta, k0, k1, stats: StatsEchantillonneur::default(), npl };
        f.figer_v();
        f
    }

    /// Q θ : somme des trois liens de chaque ligne.
    pub fn dec(&self) -> Vec<f64> {
        self.g.lignes.iter().map(|l| l.iter().map(|&li| self.th[li as usize]).sum()).collect()
    }

    pub fn set_graine(&mut self, graine: u64) {
        let (k0, k1) = crate::commun::Graine(graine).cle();
        self.k0 = k0;
        self.k1 = k1;
    }

    fn hb_libres(&mut self, mu: usize, par: usize, sweep: u32, tous: bool) {
        let g = &self.g;
        let sites: &Vec<u32> = if tous { &g.tous[mu][par] } else { &g.libres[mu][par] };
        let (th, beta, npl, k0, k1, v) = (&self.th, self.beta, self.npl, self.k0, self.k1, g.v);
        let res: Vec<f64> = sites
            .par_iter()
            .map(|&s| {
                let k = agrafe_u1(th, g, mu, s as usize);
                let li = (mu * v) as u32 + s;
                -k / npl + gauss(li, sweep, k0, k1) / (beta * npl).sqrt()
            })
            .collect();
        for (i, &s) in sites.iter().enumerate() {
            self.th[mu * v + s as usize] = res[i];
        }
        self.stats.majs += sites.len() as u64;
    }

    fn or_libres(&mut self, mu: usize, par: usize, tous: bool) {
        let g = &self.g;
        let sites: &Vec<u32> = if tous { &g.tous[mu][par] } else { &g.libres[mu][par] };
        let (th, npl, v) = (&self.th, self.npl, g.v);
        let res: Vec<f64> = sites
            .par_iter()
            .map(|&s| {
                let k = agrafe_u1(th, g, mu, s as usize);
                -2.0 * k / npl - th[mu * v + s as usize]
            })
            .collect();
        for (i, &s) in sites.iter().enumerate() {
            self.th[mu * v + s as usize] = res[i];
        }
    }

    /// Moyenne conditionnelle m du lien `quelle` (1 ou 2) de la ligne `ci`.
    #[inline]
    fn moyenne_ligne(&self, th: &[f64], ci: usize, quelle: u8) -> f64 {
        let g = &self.g;
        let [l1, l2, l3] = g.lignes[ci];
        let mu = ci / g.vc;
        let v = g.v;
        let k = |l: u32| agrafe_u1(th, g, mu, l as usize % v);
        let (t1, t2) = (th[l1 as usize], th[l2 as usize]);
        let (kc, ki) = if quelle == 1 { (self.phi[ci] - t2, k(l1)) } else { (self.phi[ci] - t1, k(l2)) };
        (self.npl * kc + k(l3) - ki) / (2.0 * self.npl)
    }

    fn maj_ligne(&mut self, mu: usize, quelle: u8, sweep: u32, hb: bool) {
        let (beta, npl, k0, k1) = (self.beta, self.npl, self.k0, self.k1);
        let res: Vec<(f64, f64)> = {
            let me: &FibreU1 = &*self;
            me.g.lignes_dir[mu]
                .par_iter()
                .map(|&ci| {
                    let ci = ci as usize;
                    let [l1, l2, _l3] = me.g.lignes[ci];
                    let cible = if quelle == 1 { l1 } else { l2 };
                    let m = me.moyenne_ligne(&me.th, ci, quelle);
                    let nouveau = if hb {
                        m + gauss(cible, sweep, k0, k1) / (2.0 * beta * npl).sqrt()
                    } else {
                        2.0 * m - me.th[cible as usize]
                    };
                    let (t1, t2) = (me.th[l1 as usize], me.th[l2 as usize]);
                    let (n1, n2) = if quelle == 1 { (nouveau, t2) } else { (t1, nouveau) };
                    (nouveau, me.phi[ci] - n1 - n2)
                })
                .collect()
        };
        for (i, &ci) in self.g.lignes_dir[mu].iter().enumerate() {
            let [l1, l2, l3] = self.g.lignes[ci as usize];
            let cible = if quelle == 1 { l1 } else { l2 };
            self.th[cible as usize] = res[i].0;
            self.th[l3 as usize] = res[i].1;
        }
        if hb {
            self.stats.majs += res.len() as u64;
        }
    }

    fn balayage_sequentiel(&mut self, sweep: u32, hb: bool) {
        let g = self.g.clone();
        let (beta, npl, k0, k1, v) = (self.beta, self.npl, self.k0, self.k1, g.v);
        for mu in 0..g.d {
            for s in 0..g.v {
                if g.pos[mu * v + s] != -1 {
                    continue;
                }
                let k = agrafe_u1(&self.th, &g, mu, s);
                let li = mu * v + s;
                self.th[li] = if hb {
                    self.stats.majs += 1;
                    -k / npl + gauss(li as u32, sweep, k0, k1) / (beta * npl).sqrt()
                } else {
                    -2.0 * k / npl - self.th[li]
                };
            }
            for &ci in &g.lignes_dir[mu] {
                let ci = ci as usize;
                let [l1, l2, l3] = g.lignes[ci];
                for quelle in [1u8, 2u8] {
                    let m = self.moyenne_ligne(&self.th, ci, quelle);
                    let cible = if quelle == 1 { l1 } else { l2 } as usize;
                    self.th[cible] = if hb {
                        self.stats.majs += 1;
                        m + gauss(cible as u32, sweep, k0, k1) / (2.0 * beta * npl).sqrt()
                    } else {
                        2.0 * m - self.th[cible]
                    };
                    self.th[l3 as usize] = self.phi[ci] - self.th[l1 as usize] - self.th[l2 as usize];
                }
            }
        }
    }

    /// (σ¹, σ², σ³) de la ligne `ci` : ∂S/∂θ_k = β (n_p θ_k + K_k).
    pub fn scores_ligne(&self, ci: usize) -> [f64; 3] {
        let g = &self.g;
        let mu = ci / g.vc;
        let v = g.v;
        let mut s = [0.0; 3];
        for j in 0..3 {
            let li = g.lignes[ci][j] as usize;
            s[j] = self.beta * (self.npl * self.th[li] + agrafe_u1(&self.th, g, mu, li % v));
        }
        s
    }
}

impl Modele for FibreU1 {
    fn geom(&self) -> &Arc<Geom> {
        &self.g
    }
    fn beta(&self) -> f64 {
        self.beta
    }
    fn ncomp(&self) -> usize {
        1
    }
    fn groupe(&self) -> &'static str {
        "u1"
    }

    fn pas(&mut self, sweep: u32, n_or: usize) {
        for mu in 0..self.g.d {
            self.hb_libres(mu, 0, sweep, false);
            self.hb_libres(mu, 1, sweep, false);
            self.maj_ligne(mu, 1, sweep, true);
            self.maj_ligne(mu, 2, sweep, true);
        }
        for _ in 0..n_or {
            for mu in 0..self.g.d {
                self.or_libres(mu, 0, false);
                self.or_libres(mu, 1, false);
                self.maj_ligne(mu, 1, sweep, false);
                self.maj_ligne(mu, 2, sweep, false);
            }
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
        self.phi = self.dec();
    }

    fn observables(&self, pis: &[u32], out: &mut [f64]) {
        let g = &self.g;
        let th = &self.th;
        out.par_iter_mut().zip(pis.par_iter()).for_each(|(o, &pi)| {
            let pi = pi as usize;
            let l = g.liens_plaquette(pi / g.v, pi % g.v);
            *o = th[l[0] as usize] + th[l[1] as usize] - th[l[2] as usize] - th[l[3] as usize];
        });
    }

    fn scores(&self, cis: &[u32], separes: bool, out: &mut [f64]) {
        let nc = if separes { 3 } else { 1 };
        assert_eq!(out.len(), cis.len() * nc);
        out.par_chunks_mut(nc).zip(cis.par_iter()).for_each(|(o, &ci)| {
            let s = self.scores_ligne(ci as usize);
            if separes {
                o.copy_from_slice(&s);
            } else {
                o[0] = (s[0] + s[1] + s[2]) / 3.0;
            }
        });
    }

    fn canal(&self, c: &super::modele::Canal, out: &mut [f64]) -> bool {
        use super::modele::Canal;
        if let Canal::Plaq { pi } = c {
            self.observables(&[*pi], &mut out[..1]);
            return true;
        }
        if let Canal::GroupesPlaquettes = c {
            let g = &self.g;
            let cls = g.classes_plaquettes();
            let ng = g.d - 1 + 3;
            let mut somme = vec![0.0f64; ng];
            let mut nn = vec![0usize; ng];
            let pis: Vec<u32> = (0..g.nplaq() as u32).collect();
            let mut o = vec![0.0f64; pis.len()];
            self.observables(&pis, &mut o);
            for (pi, v) in o.iter().enumerate() {
                let [t, nl] = cls[pi];
                if (t as usize) < g.d - 1 {
                    somme[t as usize] += v;
                    nn[t as usize] += 1;
                }
                if nl <= 2 {
                    somme[g.d - 1 + nl as usize] += v;
                    nn[g.d - 1 + nl as usize] += 1;
                }
            }
            for k in 0..ng {
                out[k] = somme[k] / nn[k].max(1) as f64;
            }
            return true;
        }
        false
    }

    fn terme_x(&self, paires: &[(u32, u32)], separes: bool, out: &mut [f64]) {
        let nc = if separes { 3 } else { 1 };
        assert_eq!(out.len(), paires.len() * nc);
        let g = &self.g;
        out.par_chunks_mut(nc).zip(paires.par_iter()).for_each(|(o, &(pi, ci))| {
            let (pi, ci) = (pi as usize, ci as usize);
            let l = g.liens_plaquette(pi / g.v, pi % g.v);
            let ligne = g.lignes[ci];
            // X^{(k)}(F_p) = ± 1 selon la position du lien dans (a, b, c, d) : F = θ_a + θ_b − θ_c − θ_d
            let mut x = [0.0f64; 3];
            for k in 0..3 {
                if let Some(rang) = l.iter().position(|&z| z == ligne[k]) {
                    x[k] = if rang < 2 { 1.0 } else { -1.0 };
                }
            }
            if separes {
                o.copy_from_slice(&x);
            } else {
                o[0] = (x[0] + x[1] + x[2]) / 3.0;
            }
        });
    }

    fn residu_contrainte(&self) -> f64 {
        self.g
            .lignes
            .par_iter()
            .zip(self.phi.par_iter())
            .map(|(l, p)| (l.iter().map(|&li| self.th[li as usize]).sum::<f64>() - p).abs())
            .reduce(|| 0.0, f64::max)
    }

    fn residu_ward(&self) -> f64 {
        let g = &self.g;
        let sc: Vec<[f64; 3]> = (0..g.ngros()).into_par_iter().map(|ci| self.scores_ligne(ci)).collect();
        (0..g.vc)
            .into_par_iter()
            .map(|y| {
                let mut tot = 0.0;
                for mu in 0..g.d {
                    tot += sc[mu * g.vc + y][0];
                    let mut cy = g.coords_gros[y];
                    cy[mu] = ((cy[mu] as usize + g.lc - 1) % g.lc) as u16;
                    let yi = g.site_gros(&[cy[0] as i64, cy[1] as i64, cy[2] as i64, cy[3] as i64]);
                    tot -= sc[mu * g.vc + yi][2];
                }
                tot.abs()
            })
            .reduce(|| 0.0, f64::max)
    }

    fn defaut_unitarite(&self) -> f64 {
        0.0
    }

    fn stats(&self) -> StatsEchantillonneur {
        self.stats
    }

    fn empreinte(&self) -> String {
        let mut octets = Vec::with_capacity(self.th.len() * 8);
        for c in &self.th {
            octets.extend_from_slice(&c.to_le_bytes());
        }
        sha256_hex(&octets)
    }

    fn champ_plat(&self) -> Vec<f64> {
        self.th.clone()
    }

    fn v_fige_plat(&self) -> Vec<f64> {
        self.phi.clone()
    }
}
