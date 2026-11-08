//! Lot K1 (PLAN_V3, 26/09/2026) : bain thermique SU(2) exact, reference CPU.
//!
//! Reference CPU FP64 du generateur HIP `src/hip/su2_sweep.hip` (port de su2_hip.cpp,
//! sha256 ed1e0fce..., chantier wilson_gpu_grandL_20260926, lecture seule).
//!
//! Meme algorithme, meme RNG, memes expressions que le code device :
//! - action S = beta * sum_p (1 - 1/2 ReTr U_p) ;
//! - U = a0 + i (a1 s1 + a2 s2 + a3 s3), quaternion (x=a0, y=a1, z=a2, w=a3) ;
//!   matrice [[a0 + i a3, a2 + i a1], [-a2 + i a1, a0 - i a3]] ;
//! - reseau L^4 periodique, s = ((x0*L + x1)*L + x2)*L + x3, lien U_mu(x) de x vers x+e_mu,
//!   direction temporelle 3 ; indice de lien li = mu*V + s ;
//! - pas = 1 sweep de bain thermique (Kennedy-Pendleton si alpha = beta|A| >= 2, Creutz
//!   sinon, les deux exacts) + nor sweeps d'overrelaxation U -> V^+ U^+ V^+, damier
//!   4 directions x 2 parites ;
//! - Philox4x32-10 a compteur (lien, sweep, tirage, etiquette), cle = (graine basse, haute).
//!
//! Dans un lancement (mu, parite), les liens mis a jour ne lisent que des liens d'une autre
//! direction ou de l'autre parite : le calcul en deux phases (lecture puis ecriture) est
//! EXACTEMENT le calcul en place, quel que soit l'ordre (rayon => resultat deterministe).
//!
//! Les seules differences attendues avec le GPU viennent des fonctions transcendantes
//! (log, cos, sin, exp : libm hote contre ocml device, ecarts de l'ordre de l'ULP). Les
//! reductions (plaquette, Polyakov) reproduisent l'arbre fixe du GPU : a configuration
//! egale, elles sont egales a l'octet.

use rayon::prelude::*;
use std::f64::consts::PI;
use std::io::Write;

// ------------------------------------------------------------------------------------------
// Philox4x32-10 (Salmon et al. 2011, Random123) et uniforme ]0,1[ sur 52 bits
// ------------------------------------------------------------------------------------------

#[inline]
pub fn philox4x32_10(mut c: [u32; 4], mut k0: u32, mut k1: u32) -> [u32; 4] {
    const M0: u32 = 0xD251_1F53;
    const M1: u32 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;
    for _ in 0..10 {
        let p0 = (M0 as u64) * (c[0] as u64);
        let p1 = (M1 as u64) * (c[2] as u64);
        let (hi0, lo0) = ((p0 >> 32) as u32, p0 as u32);
        let (hi1, lo1) = ((p1 >> 32) as u32, p1 as u32);
        c = [hi1 ^ c[1] ^ k0, lo1, hi0 ^ c[3] ^ k1, lo0];
        k0 = k0.wrapping_add(W0);
        k1 = k1.wrapping_add(W1);
    }
    c
}

/// Deux u32 -> double uniforme dans ]0,1[ ((m+0.5) 2^-52, jamais 0 ni 1).
#[inline]
pub fn u01(a: u32, b: u32) -> f64 {
    let m: u64 = (((a >> 6) as u64) << 26) | ((b >> 6) as u64);
    ((m as f64) + 0.5) * (1.0 / 4503599627370496.0)
}

pub const TAG_INIT: u32 = 0x494E_4954;
pub const TAG_HB: u32 = 0x4842_4154;

// ------------------------------------------------------------------------------------------
// Algebre quaternionique (memes expressions que le device)
// ------------------------------------------------------------------------------------------
pub type Q = [f64; 4];

#[inline]
pub fn qmul(a: Q, b: Q) -> Q {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + b[0] * a[1] - (a[2] * b[3] - a[3] * b[2]),
        a[0] * b[2] + b[0] * a[2] - (a[3] * b[1] - a[1] * b[3]),
        a[0] * b[3] + b[0] * a[3] - (a[1] * b[2] - a[2] * b[1]),
    ]
}
#[inline]
pub fn qdag(a: Q) -> Q {
    [a[0], -a[1], -a[2], -a[3]]
}
#[inline]
pub fn qadd(a: Q, b: Q) -> Q {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}
#[inline]
pub fn qnorm(a: Q) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt()
}
#[inline]
pub fn qscale(a: Q, s: f64) -> Q {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

/// Quaternion -> matrice 2x2 complexe (re,im entrelaces, ligne par ligne), convention
/// [[a0 + i a3, a2 + i a1], [-a2 + i a1, a0 - i a3]] (celle de dump_npy de su2_hip.cpp,
/// de wilson_r.haar_field et de SU2Group de kevinotron).
#[inline]
pub fn quat_to_mat(q: Q) -> [f64; 8] {
    [q[0], q[3], q[2], q[1], -q[2], q[1], q[0], -q[3]]
}

/// Matrice (re,im entrelaces) -> quaternion. Inverse exacte de `quat_to_mat` sur SU(2) ;
/// sur une matrice quelconque, projection sur les combinaisons reelles (moyennes).
#[inline]
pub fn mat_to_quat(m: &[f64]) -> Q {
    [(m[0] + m[6]) / 2., (m[3] + m[5]) / 2., (m[2] - m[4]) / 2., (m[1] - m[7]) / 2.]
}

// ------------------------------------------------------------------------------------------
// Geometrie
// ------------------------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct Geo {
    pub l: usize,
    pub v: usize,
    pub st: [usize; 4],
}

impl Geo {
    pub fn new(l: usize) -> Self {
        Geo { l, v: l * l * l * l, st: [l * l * l, l * l, l, 1] }
    }
    #[inline]
    pub fn nb_up(&self, s: usize, d: usize, xd: usize) -> usize {
        if xd == self.l - 1 { s - (self.l - 1) * self.st[d] } else { s + self.st[d] }
    }
    #[inline]
    pub fn nb_dn(&self, s: usize, d: usize, xd: usize) -> usize {
        if xd == 0 { s + (self.l - 1) * self.st[d] } else { s - self.st[d] }
    }
    #[inline]
    pub fn coords(&self, s: usize) -> [usize; 4] {
        let l = self.l;
        [s / (l * l * l), (s / (l * l)) % l, (s / l) % l, s % l]
    }
    /// indice t dans [0, V/2) -> site de parite donnee (L pair), comme site_of du device.
    #[inline]
    pub fn site_of(&self, t: usize, parity: usize) -> (usize, [usize; 4]) {
        let mut s = 2 * t;
        let mut x = self.coords(s);
        let p = (x[0] + x[1] + x[2] + x[3]) & 1;
        if p != parity {
            s += 1;
            x[3] += 1;
        }
        (s, x)
    }
}

/// Somme des 6 agrafes A : partie locale de l'action = -(beta/2) ReTr(U_mu(x) A).
#[inline]
pub fn staple(u: &[Q], g: &Geo, mu: usize, s: usize, x: &[usize; 4]) -> Q {
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
        let t = qmul(qmul(ld(nu, smu), qdag(ld(mu, snu))), qdag(ld(nu, s)));
        let b = qmul(qmul(qdag(ld(nu, smu_dn)), qdag(ld(mu, sdn))), ld(nu, sdn));
        a = qadd(a, qadd(t, b));
    }
    a
}

// ------------------------------------------------------------------------------------------
// Tirage exact de x0 ~ sqrt(1-x0^2) exp(alpha x0) sur [-1,1]
// ------------------------------------------------------------------------------------------
const CAP: u32 = 1_000_000;

#[inline]
fn ph(li: u32, sweep_id: u32, draw: &mut u32, k0: u32, k1: u32) -> [u32; 4] {
    let r = philox4x32_10([li, sweep_id, *draw, TAG_HB], k0, k1);
    *draw = draw.wrapping_add(1);
    r
}

/// Kennedy-Pendleton si alpha >= 2, Creutz sinon. Retourne (x0, essais, ok).
pub fn sample_x0(alpha: f64, li: u32, sweep_id: u32, draw: &mut u32, k0: u32, k1: u32) -> (f64, u32, bool) {
    let mut x0 = 1.0;
    let mut ok = false;
    let mut ntry: u32 = 0;
    if alpha >= 2.0 {
        while ntry < CAP {
            let r1 = ph(li, sweep_id, draw, k0, k1);
            let r2 = ph(li, sweep_id, draw, k0, k1);
            let a1 = u01(r1[0], r1[1]);
            let a2 = u01(r1[2], r1[3]);
            let a3 = u01(r2[0], r2[1]);
            let a4 = u01(r2[2], r2[3]);
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
        while ntry < CAP {
            let r1 = ph(li, sweep_id, draw, k0, k1);
            let a1 = u01(r1[0], r1[1]);
            let a2 = u01(r1[2], r1[3]);
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

/// MUTANT (controle negatif du gate 6, jamais en production) : UN pas de Metropolis naif
/// depuis x_cur, proposition uniforme sur [-1,1]. Meme definition que K1_MUTANT_METRO du device.
pub fn sample_x0_mut(alpha: f64, x_cur: f64, li: u32, sweep_id: u32, draw: &mut u32, k0: u32, k1: u32) -> (f64, u32, bool) {
    let r1 = ph(li, sweep_id, draw, k0, k1);
    let y = 2.0 * u01(r1[0], r1[1]) - 1.0;
    let pc = (1.0 - x_cur * x_cur).max(0.0).sqrt();
    let py = (1.0 - y * y).max(0.0).sqrt() * (alpha * (y - x_cur)).exp();
    let x = if u01(r1[2], r1[3]) * pc <= py { y } else { x_cur };
    (x, 1, true)
}

#[inline]
fn fmax0(v: f64) -> f64 {
    // fmax(0.0, v) du C pour v fini
    if v > 0.0 { v } else { 0.0 }
}

fn haar_from(r1: [u32; 4], r2: [u32; 4]) -> Q {
    let u1 = u01(r1[0], r1[1]);
    let u2 = u01(r1[2], r1[3]);
    let u3 = u01(r2[0], r2[1]);
    let u4 = u01(r2[2], r2[3]);
    let ra = (-2.0 * u1.ln()).sqrt();
    let rb = (-2.0 * u3.ln()).sqrt();
    let q = [ra * (2.0 * PI * u2).cos(), ra * (2.0 * PI * u2).sin(), rb * (2.0 * PI * u4).cos(), rb * (2.0 * PI * u4).sin()];
    qscale(q, 1.0 / qnorm(q))
}

/// Resultat de la mise a jour d'un lien : (nouveau lien ou None si inchange, essais, plafond, diag).
struct Upd {
    li: usize,
    q: Option<Q>,
    ntry: u64,
    cap: u64,
    diag: u32,
}

/// Reseau SU(2) L^4 sur CPU (reference).
pub struct CpuSu2 {
    pub geo: Geo,
    pub k0: u32,
    pub k1: u32,
    pub sweep_id: u32,
    pub u: Vec<Q>,
    pub n_cap: u64,
    pub n_tries: u64,
    pub n_hb: u64,
    pub n_or: u64,
    pub mutant_metro: bool,
}

impl CpuSu2 {
    pub fn new(l: usize, seed: u64) -> Result<Self, String> {
        if l < 2 || l % 2 != 0 {
            return Err("L doit etre pair >= 2".into());
        }
        let geo = Geo::new(l);
        Ok(CpuSu2 {
            geo,
            k0: (seed & 0xFFFF_FFFF) as u32,
            k1: (seed >> 32) as u32,
            sweep_id: 0,
            u: vec![[1.0, 0.0, 0.0, 0.0]; 4 * geo.v],
            n_cap: 0,
            n_tries: 0,
            n_hb: 0,
            n_or: 0,
            mutant_metro: false,
        })
    }

    /// Depart chaud (Haar, compteur (i, 0xFFFFFFFF, 0|1, TAG_INIT)) ou froid (identite).
    pub fn init(&mut self, hot: bool) {
        let (k0, k1) = (self.k0, self.k1);
        self.u.par_iter_mut().enumerate().for_each(|(i, q)| {
            if !hot {
                *q = [1.0, 0.0, 0.0, 0.0];
                return;
            }
            let r1 = philox4x32_10([i as u32, 0xFFFF_FFFF, 0, TAG_INIT], k0, k1);
            let r2 = philox4x32_10([i as u32, 0xFFFF_FFFF, 1, TAG_INIT], k0, k1);
            *q = haar_from(r1, r2);
        });
    }

    fn hb_link(&self, beta: f64, mu: usize, parity: usize, t: usize) -> Upd {
        let g = &self.geo;
        let (s, x) = g.site_of(t, parity);
        let li = mu * g.v + s;
        let a = staple(&self.u, g, mu, s, &x);
        let k = qnorm(a);
        let mut draw: u32 = 0;
        let sid = self.sweep_id;
        let (k0, k1) = (self.k0, self.k1);
        if !(k > 1e-300) {
            let r1 = ph(li as u32, sid, &mut draw, k0, k1);
            let r2 = ph(li as u32, sid, &mut draw, k0, k1);
            return Upd { li, q: Some(haar_from(r1, r2)), ntry: 0, cap: 0, diag: 0xFFFF_FFFF };
        }
        let vv = qscale(a, 1.0 / k);
        let alpha = beta * k;
        let (x0, ntry, ok) = if self.mutant_metro {
            sample_x0_mut(alpha, qmul(self.u[li], vv)[0], li as u32, sid, &mut draw, k0, k1)
        } else {
            sample_x0(alpha, li as u32, sid, &mut draw, k0, k1)
        };
        if !ok {
            return Upd { li, q: None, ntry: ntry as u64, cap: 1, diag: 0 };
        }
        let rd = ph(li as u32, sid, &mut draw, k0, k1);
        let ct = 2.0 * u01(rd[0], rd[1]) - 1.0;
        let phi = 2.0 * PI * u01(rd[2], rd[3]);
        let st = fmax0(1.0 - ct * ct).sqrt();
        let rr = fmax0(1.0 - x0 * x0).sqrt();
        let w = [x0, rr * st * phi.cos(), rr * st * phi.sin(), rr * ct];
        let un = qmul(w, qdag(vv));
        Upd { li, q: Some(qscale(un, 1.0 / qnorm(un))), ntry: ntry as u64, cap: 0, diag: ntry }
    }

    /// Un sweep de bain thermique. `diag` (4V) recoit les essais par lien comme le device.
    pub fn sweep_hb(&mut self, beta: f64, mut diag: Option<&mut [u32]>) {
        let half = self.geo.v / 2;
        for mu in 0..4 {
            for p in 0..2 {
                let ups: Vec<Upd> = (0..half).into_par_iter().map(|t| self.hb_link(beta, mu, p, t)).collect();
                for up in ups {
                    self.n_tries += up.ntry;
                    self.n_cap += up.cap;
                    if let Some(d) = diag.as_deref_mut() {
                        d[up.li] = up.diag;
                    }
                    if let Some(q) = up.q {
                        self.u[up.li] = q;
                    }
                }
            }
        }
        self.n_hb += 1;
        self.sweep_id = self.sweep_id.wrapping_add(1);
    }

    /// Un sweep d'overrelaxation U -> V^+ U^+ V^+.
    pub fn sweep_or(&mut self) {
        let g = self.geo;
        let half = g.v / 2;
        for mu in 0..4 {
            for p in 0..2 {
                let ups: Vec<(usize, Option<Q>)> = (0..half)
                    .into_par_iter()
                    .map(|t| {
                        let (s, x) = g.site_of(t, p);
                        let li = mu * g.v + s;
                        let a = staple(&self.u, &g, mu, s, &x);
                        let k = qnorm(a);
                        if !(k > 1e-300) {
                            return (li, None);
                        }
                        let vd = qdag(qscale(a, 1.0 / k));
                        let un = qmul(qmul(vd, qdag(self.u[li])), vd);
                        (li, Some(qscale(un, 1.0 / qnorm(un))))
                    })
                    .collect();
                for (li, q) in ups {
                    if let Some(q) = q {
                        self.u[li] = q;
                    }
                }
            }
        }
        self.n_or += 1;
    }

    pub fn step(&mut self, beta: f64, nor: usize) {
        self.sweep_hb(beta, None);
        for _ in 0..nor {
            self.sweep_or();
        }
    }

    /// (P, P0, P1, P2, P3) avec l'arbre de reduction du GPU (k_plaq_site, k_reduce, k_final,
    /// k_polyakov) : egal a l'octet au GPU a configuration egale.
    pub fn measure(&self) -> [f64; 5] {
        measure_quat(&self.geo, &self.u)
    }

    /// Champ en f64 plat (4 par lien), meme disposition que HipSu2::download.
    pub fn flat(&self) -> Vec<f64> {
        self.u.iter().flat_map(|q| q.iter().copied()).collect()
    }

    pub fn set_flat(&mut self, f: &[f64]) -> Result<(), String> {
        if f.len() != 16 * self.geo.v {
            return Err(format!("set_flat : {} f64 au lieu de {}", f.len(), 16 * self.geo.v));
        }
        for (i, q) in self.u.iter_mut().enumerate() {
            *q = [f[4 * i], f[4 * i + 1], f[4 * i + 2], f[4 * i + 3]];
        }
        Ok(())
    }
}

/// n tirages independants de x0 (--test-hb), compteur (i, 0x7E57, tirage, TAG_HB).
pub fn test_x0_cpu(alpha: f64, n: usize, seed: u64, mutant: bool) -> (Vec<f64>, u64) {
    let (k0, k1) = ((seed & 0xFFFF_FFFF) as u32, (seed >> 32) as u32);
    let r: Vec<(f64, bool)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut draw = 0u32;
            let (x, _, ok) = if mutant {
                sample_x0_mut(alpha, 1.0, i as u32, 0x7E57, &mut draw, k0, k1)
            } else {
                sample_x0(alpha, i as u32, 0x7E57, &mut draw, k0, k1)
            };
            (x, ok)
        })
        .collect();
    let cap = r.iter().filter(|(_, ok)| !ok).count() as u64;
    (r.into_iter().map(|(x, _)| x).collect(), cap)
}

// ------------------------------------------------------------------------------------------
// Reductions a l'identique du GPU
// ------------------------------------------------------------------------------------------

fn tree256(sh: &mut [f64; 256]) -> f64 {
    let mut w = 128;
    while w > 0 {
        for t in 0..w {
            sh[t] += sh[t + w];
        }
        w >>= 1;
    }
    sh[0]
}

/// k_reduce : blocs de 256, arbre fixe.
pub fn k_reduce(inp: &[f64]) -> Vec<f64> {
    let nb = (inp.len() + 255) / 256;
    (0..nb)
        .map(|b| {
            let mut sh = [0f64; 256];
            for t in 0..256 {
                let i = b * 256 + t;
                sh[t] = if i < inp.len() { inp[i] } else { 0.0 };
            }
            tree256(&mut sh)
        })
        .collect()
}

/// k_final : 256 sommes partielles (i = t, t+256, ...) puis arbre, fois scale.
pub fn k_final(inp: &[f64], scale: f64) -> f64 {
    let mut sh = [0f64; 256];
    for t in 0..256 {
        let mut a = 0.0;
        let mut i = t;
        while i < inp.len() {
            a += inp[i];
            i += 256;
        }
        sh[t] = a;
    }
    tree256(&mut sh) * scale
}

/// Plaquette et Polyakov d'un champ de quaternions (li = mu*V + s).
pub fn measure_quat(g: &Geo, u: &[Q]) -> [f64; 5] {
    let ld = |nu: usize, site: usize| u[nu * g.v + site];
    let site: Vec<f64> = (0..g.v)
        .into_par_iter()
        .map(|s| {
            let x = g.coords(s);
            let mut acc = 0.0;
            for mu in 0..4 {
                let smu = g.nb_up(s, mu, x[mu]);
                for nu in mu + 1..4 {
                    let snu = g.nb_up(s, nu, x[nu]);
                    let p = qmul(qmul(ld(mu, s), ld(nu, smu)), qmul(qdag(ld(mu, snu)), qdag(ld(nu, s))));
                    acc += p[0];
                }
            }
            acc
        })
        .collect();
    let part = k_reduce(&site);
    let mut out = [0f64; 5];
    out[0] = k_final(&part, 1.0 / (6.0 * g.v as f64));
    let l = g.l;
    let l3 = g.v / l;
    let pol: Vec<f64> = (0..4 * l3)
        .into_par_iter()
        .map(|i| {
            let mu = i / l3;
            let j = i % l3;
            let mut x = [0usize; 4];
            let dirs: Vec<usize> = (0..4).filter(|&d| d != mu).collect();
            x[dirs[2]] = j % l;
            let t = j / l;
            x[dirs[1]] = t % l;
            x[dirs[0]] = t / l;
            let mut s = ((x[0] * l + x[1]) * l + x[2]) * l + x[3];
            let mut p: Q = [1.0, 0.0, 0.0, 0.0];
            for k in 0..l {
                p = qmul(p, ld(mu, s));
                if k != l - 1 {
                    s += g.st[mu];
                }
            }
            p[0]
        })
        .collect();
    for mu in 0..4 {
        out[1 + mu] = k_final(&pol[mu * l3..(mu + 1) * l3], 1.0 / l3 as f64);
    }
    out
}

// ------------------------------------------------------------------------------------------
// Sorties : NPY complex128 (mu,x0..x3,i,j), CSV %.17g, sha256
// ------------------------------------------------------------------------------------------

/// En-tete NPY v1.0, identique a dump_npy de su2_hip.cpp et a io::dump_config_npy.
pub fn npy_header(l: usize) -> Vec<u8> {
    let mut hd = format!("{{'descr': '<c16', 'fortran_order': False, 'shape': (4, {l}, {l}, {l}, {l}, 2, 2), }}");
    let tot = 10 + hd.len() + 1;
    let pad = (64 - tot % 64) % 64;
    hd.push_str(&" ".repeat(pad));
    hd.push('\n');
    let mut out = vec![0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0];
    out.extend_from_slice(&(hd.len() as u16).to_le_bytes());
    out.extend_from_slice(hd.as_bytes());
    out
}

/// Ecrit le champ de quaternions (4 f64 par lien, li = mu*V + s) en NPY complex128
/// C-order (mu, x0, x1, x2, x3, i, j), via un fichier .part renomme (comme su2_hip.cpp).
pub fn write_npy_quat(path: &std::path::Path, l: usize, flat: &[f64]) -> std::io::Result<()> {
    let n = flat.len() / 4;
    let tmp = path.with_extension("npy.part");
    {
        let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp)?);
        f.write_all(&npy_header(l))?;
        for i in 0..n {
            let q = [flat[4 * i], flat[4 * i + 1], flat[4 * i + 2], flat[4 * i + 3]];
            for v in quat_to_mat(q) {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        f.flush()?;
        f.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

extern "C" {
    fn snprintf(buf: *mut std::os::raw::c_char, n: usize, fmt: *const std::os::raw::c_char, ...) -> std::os::raw::c_int;
}

/// Lit un NPY SU(2) (complex128, C-order, forme (4,L,L,L,L,2,2), fortran_order False) et
/// retourne (L, champ de quaternions plat). Refuse tout autre en-tete ; refuse une matrice
/// qui n'a pas la structure SU(2) exacte a 1e-12 (U11 = conj U00, U10 = -conj U01, |q| = 1).
pub fn read_npy_quat(path: &std::path::Path) -> Result<(usize, Vec<f64>), String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" || b[6] != 1 {
        return Err("NPY v1 attendu".into());
    }
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let hd = String::from_utf8_lossy(&b[10..10 + hl]).to_string();
    if !hd.contains("'descr': '<c16'") || !hd.contains("'fortran_order': False") {
        return Err(format!("en-tete refuse : {}", hd.trim()));
    }
    let i0 = hd.find("'shape': (").ok_or("shape absente")? + 10;
    let i1 = i0 + hd[i0..].find(')').ok_or("shape")?;
    let dims: Vec<usize> = hd[i0..i1].split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap_or(0)).collect();
    if dims.len() != 7 || dims[0] != 4 || dims[5] != 2 || dims[6] != 2 || dims[1] != dims[2] || dims[1] != dims[3] || dims[1] != dims[4] {
        return Err(format!("forme refusee : {:?}", dims));
    }
    let l = dims[1];
    let n = 4 * l * l * l * l;
    let data = &b[10 + hl..];
    if data.len() != n * 64 {
        return Err(format!("taille {} au lieu de {}", data.len(), n * 64));
    }
    let mut out = Vec::with_capacity(4 * n);
    for i in 0..n {
        let mut m = [0f64; 8];
        for k in 0..8 {
            let o = 64 * i + 8 * k;
            m[k] = f64::from_le_bytes(data[o..o + 8].try_into().unwrap());
        }
        let q = mat_to_quat(&m);
        let back = quat_to_mat(q);
        let d = m.iter().zip(back.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max).max((qnorm(q) - 1.0).abs());
        if !(d <= 1e-12) {
            return Err(format!("lien {i} hors SU(2) : defaut {d:e}"));
        }
        out.extend_from_slice(&q);
    }
    Ok((l, out))
}

/// printf("%.17g", v) de la libc : meme texte que les CSV de su2_hip.cpp.
pub fn g17(v: f64) -> String {
    let mut buf = [0u8; 64];
    let n = unsafe { snprintf(buf.as_mut_ptr() as *mut _, buf.len(), b"%.17g\0".as_ptr() as *const _, v) };
    String::from_utf8_lossy(&buf[..(n.max(0) as usize).min(63)]).into_owned()
}

/// SHA-256 (FIPS 180-4), pour la provenance (hash des sources embarquees et des sorties).
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

/// Ecart d'appartenance a SU(2) d'un champ de quaternions : max | |q| - 1 |.
pub fn max_norm_defect(flat: &[f64]) -> f64 {
    flat.chunks(4).map(|q| ((q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt() - 1.0).abs()).fold(0.0, f64::max)
}
