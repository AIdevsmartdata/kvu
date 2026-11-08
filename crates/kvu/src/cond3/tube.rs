//! Charge de tube des paires (p, ē) et classes d'environnement grossier (SPEC_T3 §1.10), dimension 4, facteur 3.
//!
//! Plaquette grossière P = (plan, Y), u_P = ½ Tr V_P, charge c_P = 1 − u_P ∈ [0, 2]. Coordonnées grossières : y_p = c_p/3, y_ē = m_ē/3,
//! y_P = Y + (e_μ + e_ν)/2 ; Δ = y_ē − y_p (image minimale, L_c). Tube de (p, ē) : les P dont le centre est à distance ⩽ 1 d'au moins un des
//! points y_p + sΔ, s ∈ {0, 1/n, …, 1}, n = ⌈2|Δ|⌉ + 1 ; f(p, ē) = moyenne de c_P sur le tube.
//!
//! La FORME du tube ne dépend que de la classe (plan de p, position 3×3 r de p dans son bloc, direction de la sonde, Y(p) − X(sonde) mod L_c) :
//! 6 × 9 × 4 × L_c⁴ classes (884 736 à L = 24). Les tubes de toutes les classes sont énumérés UNE fois, en arithmétique ENTIÈRE exacte
//! (unités 1/(6n)), puis réutilisés pour chaque configuration et chaque sonde : f d'une paire = moyenne de cP décalée sur la liste.

use super::geom::Geom;
use super::paires::Paires;
use crate::heritage::su2_hb::{qdag, qmul, Q};
use rayon::prelude::*;

pub struct TableTubes {
    pub d: usize,
    pub lc: usize,
    pub vc: usize,
    /// CSR par classe : items = plan_P·Vc + site grossier relatif (Y_P − X_ē mod L_c).
    pub debut: Vec<u32>,
    pub items: Vec<u16>,
}

#[inline]
fn red(x: i64, m: i64) -> i64 {
    // image minimale dans ]−m/2, m/2] pour un module m pair
    let r = x.rem_euclid(m);
    if r > m / 2 {
        r - m
    } else {
        r
    }
}

fn plus_petit_m_tel_que_9m2_ge(s6: i64) -> i64 {
    let mut m = ((s6 as f64).sqrt() / 3.0) as i64;
    while 9 * m * m < s6 {
        m += 1;
    }
    while m > 0 && 9 * (m - 1) * (m - 1) >= s6 {
        m -= 1;
    }
    m
}

impl TableTubes {
    /// Énumère les tubes de toutes les classes (d = 4).
    pub fn nouvelle(g: &Geom) -> TableTubes {
        assert_eq!(g.d, 4, "tubes : d = 4 seulement");
        let (lc, vc) = (g.lc as i64, g.vc);
        let planes = g.planes.clone();
        let nclasses = planes.len() * 9 * 4 * vc;
        let listes: Vec<Vec<u16>> = (0..nclasses)
            .into_par_iter()
            .map(|cls| {
                let sc = cls % vc;
                let mue = (cls / vc) % 4;
                let rcode = (cls / (vc * 4)) % 9;
                let pl = cls / (vc * 4 * 9);
                let dy = g.coords_gros[sc]; // Y − X_e (mod lc) en coordonnées grossières
                let (mu, nu) = planes[pl];
                let mut r = [0i64; 4];
                r[mu] = (rcode / 3) as i64;
                r[nu] = (rcode % 3) as i64;
                // 6·y_p' = 6ΔY + 2r + e_mu + e_nu
                let mut y6 = [0i64; 4];
                for i in 0..4 {
                    y6[i] = 6 * dy[i] as i64 + 2 * r[i];
                }
                y6[mu] += 1;
                y6[nu] += 1;
                let mut e6 = [0i64; 4];
                e6[mue] = 3;
                // 6Δ = 6 y_e' − 6 y_p', image minimale mod 6·lc
                let mut d6 = [0i64; 4];
                let mut s6 = 0i64;
                for i in 0..4 {
                    d6[i] = red(e6[i] - y6[i], 6 * lc);
                    s6 += d6[i] * d6[i];
                }
                let n = plus_petit_m_tel_que_9m2_ge(s6) + 1;
                let un = 6 * n; // unité : 1/(6n)
                let modulo = un * lc;
                let mut tube: Vec<u16> = Vec::new();
                for k in 0..=n {
                    let mut z = [0i64; 4];
                    for i in 0..4 {
                        z[i] = n * y6[i] + k * d6[i];
                    }
                    for (pl2, &(m2, n2)) in planes.iter().enumerate() {
                        let mut h = [0i64; 4];
                        h[m2] = 1;
                        h[n2] = 1;
                        // candidats par coordonnée : j tel que |un·j + (un/2)·h_i − z_i| ⩽ un (+ images)
                        let mut cand: [Vec<i64>; 4] = Default::default();
                        for i in 0..4 {
                            let centre = z[i] - (un / 2) * h[i];
                            let jmin = (centre - un).div_euclid(un) - 1;
                            let jmax = (centre + un).div_euclid(un) + 1;
                            cand[i] = (jmin..=jmax).collect();
                        }
                        for &j0 in &cand[0] {
                            for &j1 in &cand[1] {
                                for &j2 in &cand[2] {
                                    for &j3 in &cand[3] {
                                        let j = [j0, j1, j2, j3];
                                        let mut s = 0i64;
                                        for i in 0..4 {
                                            let dlt = red(un * j[i] + (un / 2) * h[i] - z[i], modulo);
                                            s += dlt * dlt;
                                        }
                                        if s <= un * un {
                                            let site = g.site_gros(&[j0, j1, j2, j3]);
                                            tube.push(u16::try_from(pl2 * vc + site).expect("tube : l'index (plan_P·Vc + site) dépasse u16 (L_c ⩽ 8, soit L ⩽ 24, seulement)"));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                tube.sort_unstable();
                tube.dedup();
                tube
            })
            .collect();
        let mut debut = Vec::with_capacity(nclasses + 1);
        let mut items = Vec::new();
        debut.push(0u32);
        for l in &listes {
            items.extend_from_slice(l);
            debut.push(items.len() as u32);
        }
        TableTubes { d: g.d, lc: g.lc, vc, debut, items }
    }

    #[inline]
    pub fn classe(&self, pl: usize, rcode: usize, mue: usize, sc_delta: usize) -> usize {
        ((pl * 9 + rcode) * 4 + mue) * self.vc + sc_delta
    }

    pub fn tube(&self, cls: usize) -> &[u16] {
        &self.items[self.debut[cls] as usize..self.debut[cls + 1] as usize]
    }
}

/// Charges des plaquettes grossières c_P = 1 − ½ Tr V_P : [plan·Vc + site grossier].
pub fn charges_grossieres(g: &Geom, vfix: &[Q]) -> Vec<f64> {
    let vc = g.vc;
    let nbr_gros = |sc: usize, mu: usize| -> usize {
        let mut y = g.coords_gros[sc];
        y[mu] = ((y[mu] as usize + 1) % g.lc) as u16;
        g.site_gros(&[y[0] as i64, y[1] as i64, y[2] as i64, y[3] as i64])
    };
    let mut c = vec![0.0f64; g.planes.len() * vc];
    for (pl, &(mu, nu)) in g.planes.iter().enumerate() {
        for sc in 0..vc {
            let a = vfix[mu * vc + sc];
            let b = vfix[nu * vc + nbr_gros(sc, mu)];
            let cc = vfix[mu * vc + nbr_gros(sc, nu)];
            let d = vfix[nu * vc + sc];
            let p = qmul(qmul(a, b), qmul(qdag(cc), qdag(d)));
            c[pl * vc + sc] = 1.0 - p[0];
        }
    }
    c
}

/// Description d'une face (pour le calcul de classe) : (plan, rcode, coordonnées grossières Y).
pub struct FaceInfo {
    pub pl: u16,
    pub rcode: u8,
    pub y: [u16; 4],
}

pub fn infos_faces(g: &Geom, faces: &[u32]) -> Vec<FaceInfo> {
    faces
        .iter()
        .map(|&pi| {
            let pi = pi as usize;
            let (pl, s) = (pi / g.v, pi % g.v);
            let (mu, nu) = g.planes[pl];
            let x = g.coords[s];
            let mut y = [0u16; 4];
            for i in 0..4 {
                y[i] = x[i] / 3;
                if i != mu && i != nu {
                    assert_eq!(x[i] % 3, 0, "face : coordonnée transverse non nulle mod 3");
                }
            }
            FaceInfo { pl: pl as u16, rcode: ((x[mu] % 3) * 3 + (x[nu] % 3)) as u8, y }
        })
        .collect()
}

/// Charge de tube f de TOUTES les paires de la fenêtre, dans l'ordre des paires de `Paires` (f32). Parallèle sur les sondes (chaque sonde écrit
/// sa plage contiguë de paires : écritures disjointes, résultat déterministe).
pub fn charges_paires(g: &Geom, p: &Paires, tt: &TableTubes, c_p: &[f64]) -> Vec<f32> {
    let infos = infos_faces(g, &p.faces);
    let vc = g.vc;
    let nplan = g.planes.len();
    let mut f = vec![0.0f32; p.npaires()];
    // découpage en tranches disjointes, une par sonde
    let mut restes: Vec<&mut [f32]> = Vec::with_capacity(p.sondes.len());
    let mut reste: &mut [f32] = &mut f;
    for s in 0..p.sondes.len() {
        let n = (p.debut[s + 1] - p.debut[s]) as usize;
        let (a, b) = reste.split_at_mut(n);
        restes.push(a);
        reste = b;
    }
    restes.into_par_iter().enumerate().for_each(|(s, out)| {
        let ci = p.sondes[s] as usize;
        let (mue, sce) = (ci / vc, ci % vc);
        let xe = g.coords_gros[sce];
        // cP décalée par X_e : cshift[plan·Vc + site(Y')] = c_P[plan·Vc + site(Y' + X_e)]
        let mut cshift = vec![0.0f64; nplan * vc];
        for pl in 0..nplan {
            for sy in 0..vc {
                let yp = g.coords_gros[sy];
                let mut z = [0i64; 4];
                for i in 0..4 {
                    z[i] = yp[i] as i64 + xe[i] as i64;
                }
                cshift[pl * vc + sy] = c_p[pl * vc + g.site_gros(&z)];
            }
        }
        let q0 = p.debut[s] as usize;
        for (j, o) in out.iter_mut().enumerate() {
            let q = q0 + j;
            let fi = &infos[p.face_loc[q] as usize];
            let mut dy = [0i64; 4];
            for i in 0..4 {
                dy[i] = fi.y[i] as i64 - xe[i] as i64;
            }
            let cls = tt.classe(fi.pl as usize, fi.rcode as usize, mue, g.site_gros(&dy));
            let tube = tt.tube(cls);
            let mut somme = 0.0f64;
            for &it in tube {
                somme += cshift[it as usize];
            }
            *o = (somme / tube.len() as f64) as f32;
        }
    });
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tube_symetrique_par_permutation_d_axes_et_taille_raisonnable() {
        // G0.8 (partiel) : sonde et face au même point (Δ = 0) → le tube est l'ensemble des P à distance ⩽ 1 du point ; taille attendue
        // d'une boule unité 4D centrée en y_p : on compte par force brute indépendante et on compare à la table.
        let g = Geom::nouvelle(4, 12).unwrap();
        let tt = TableTubes::nouvelle(&g);
        assert_eq!(tt.debut.len(), 6 * 9 * 4 * g.vc + 1);
        let moyenne = tt.items.len() as f64 / (tt.debut.len() - 1) as f64;
        assert!(moyenne > 20.0 && moyenne < 400.0, "taille moyenne des tubes : {moyenne}");
        // aucun doublon, indices valides
        for cls in (0..tt.debut.len() - 1).step_by(997) {
            let t = tt.tube(cls);
            assert!(t.windows(2).all(|w| w[0] < w[1]));
            assert!(t.iter().all(|&x| (x as usize) < 6 * g.vc));
            assert!(!t.is_empty());
        }
    }

    #[test]
    fn tubes_egalent_la_force_brute_sur_toutes_les_classes_de_l6() {
        // Force brute indépendante des fenêtres de candidats de `nouvelle` : pour CHAQUE classe de L = 6 (L_c = 2, 3456 classes), balayage de TOUS les
        // P = (plan, site relatif) et de tous les points z_k du segment [y_p, y_ē], en unités entières 1/(6n) ; le tube est l'ensemble des P à distance
        // (image minimale) ⩽ 1 d'au moins un z_k.
        let g = Geom::nouvelle(4, 6).unwrap();
        let tt = TableTubes::nouvelle(&g);
        let (lc, vc) = (g.lc as i64, g.vc);
        let planes = g.planes.clone();
        assert_eq!(tt.debut.len(), planes.len() * 9 * 4 * vc + 1);
        let mut total = 0usize;
        for cls in 0..planes.len() * 9 * 4 * vc {
            let (sc, mue, rcode, pl) = (cls % vc, (cls / vc) % 4, (cls / (vc * 4)) % 9, cls / (vc * 36));
            let dy = g.coords_gros[sc];
            let (mu, nu) = planes[pl];
            let mut y6 = [0i64; 4];
            for i in 0..4 {
                y6[i] = 6 * dy[i] as i64;
            }
            y6[mu] += 2 * (rcode / 3) as i64 + 1;
            y6[nu] += 2 * (rcode % 3) as i64 + 1;
            let mut e6 = [0i64; 4];
            e6[mue] = 3;
            let mut d6 = [0i64; 4];
            for i in 0..4 {
                d6[i] = red(e6[i] - y6[i], 6 * lc);
            }
            let s6: i64 = d6.iter().map(|x| x * x).sum();
            // n = ⌈2|Δ|⌉ + 1 avec 36|Δ|² = s6 : plus petit m tel que (3m)² ⩾ s6, plus 1 (comptage entier indépendant de f64)
            let mut m = 0i64;
            while 9 * m * m < s6 {
                m += 1;
            }
            let n = m + 1;
            let un = 6 * n;
            let modulo = un * lc;
            let mut attendu: Vec<u16> = Vec::new();
            for (pl2, &(m2, n2)) in planes.iter().enumerate() {
                for site in 0..vc {
                    let yy = g.coords_gros[site];
                    let dedans = (0..=n).any(|k| {
                        let mut s = 0i64;
                        for i in 0..4 {
                            let h = if i == m2 || i == n2 { un / 2 } else { 0 };
                            let z = n * y6[i] + k * d6[i];
                            let dl = red(un * yy[i] as i64 + h - z, modulo);
                            s += dl * dl;
                        }
                        s <= un * un
                    });
                    if dedans {
                        attendu.push((pl2 * vc + site) as u16);
                    }
                }
            }
            attendu.sort_unstable();
            assert_eq!(tt.tube(cls), &attendu[..], "classe {cls} (plan {pl}, r {rcode}, μ_ē {mue}, site {sc})");
            total += attendu.len();
        }
        assert_eq!(total, tt.items.len());
    }
}
