// SPDX-License-Identifier: Apache-2.0
//! Référence purement hôte des boîtes SU(2) de Dirichlet et de leurs mesures.

use crate::commun::{ChampHote, Erreur, Graine, Resultat, ETIQUETTE_BORD, ETIQUETTE_JAUGE};
use crate::heritage::su2_hb::{k_final, qdag, qmul, qnorm, qscale, Q};
use crate::su2::cpu::haar_q;

pub const PLANS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
const IDENTITE: Q = [1.0, 0.0, 0.0, 0.0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeometrieBoite {
    pub ell: usize,
    pub lc: usize,
}

impl GeometrieBoite {
    pub fn nouvelle(ell: usize) -> Resultat<GeometrieBoite> {
        if !(2..=126).contains(&ell) || !ell.is_multiple_of(2) {
            return Err(Erreur::Config(format!(
                "ℓ = {ell} : longueur paire ≥ 2 et côté de calcul ℓ + 2 ≤ 128 requis"
            )));
        }
        Ok(Self { ell, lc: ell + 2 })
    }

    pub fn v(&self) -> usize {
        self.lc.pow(4)
    }

    pub fn site(&self, x: [usize; 4]) -> usize {
        ((x[0] * self.lc + x[1]) * self.lc + x[2]) * self.lc + x[3]
    }

    pub fn coords(&self, mut s: usize) -> [usize; 4] {
        let mut x = [0; 4];
        for xi in x.iter_mut().rev() {
            *xi = s % self.lc;
            s /= self.lc;
        }
        x
    }

    pub fn indice_lien(&self, mu: usize, x: [usize; 4]) -> usize {
        mu * self.v() + self.site(x)
    }

    pub fn dans_boite(&self, x: [usize; 4]) -> bool {
        x.iter().all(|&xi| xi <= self.ell)
    }

    pub fn sur_bord(&self, x: [usize; 4]) -> bool {
        self.dans_boite(x) && x.iter().any(|&xi| xi == 0 || xi == self.ell)
    }

    pub fn lien_existe(&self, mu: usize, x: [usize; 4]) -> bool {
        mu < 4 && self.dans_boite(x) && x[mu] < self.ell
    }

    pub fn lien_fixe(&self, mu: usize, x: [usize; 4]) -> bool {
        if !self.lien_existe(mu, x) {
            return false;
        }
        let mut y = x;
        y[mu] += 1;
        self.sur_bord(x) && self.sur_bord(y)
    }

    pub fn lien_libre(&self, mu: usize, x: [usize; 4]) -> bool {
        self.lien_existe(mu, x) && !self.lien_fixe(mu, x)
    }

    pub fn n_libres(&self) -> usize {
        4 * self.ell * (self.ell - 1).pow(3)
    }

    pub fn n_fixes(&self) -> usize {
        4 * self.ell * (self.ell + 1).pow(3) - self.n_libres()
    }

    pub fn n_inexistants(&self) -> usize {
        4 * self.v() - 4 * self.ell * (self.ell + 1).pow(3)
    }

    pub fn centre(&self) -> [usize; 4] {
        [self.ell / 2; 4]
    }
}

/// Un octet par lien : 0 = libre, 1 = gelé (fixé ou inexistant).
pub fn masque_boite(g: &GeometrieBoite) -> Vec<u8> {
    let mut masque = vec![1; 4 * g.v()];
    for (li, octet) in masque.iter_mut().enumerate() {
        *octet = u8::from(!g.lien_libre(li / g.v(), g.coords(li % g.v())));
    }
    masque
}

/// Masque du gel spatial sur x₀ = 0 d'un réseau périodique de côté pair ≥ 2.
pub fn masque_gel_x0_spatial(l: usize) -> Vec<u8> {
    assert!(
        l >= 2 && l.is_multiple_of(2),
        "le côté doit être pair et ≥ 2"
    );
    let v = l.pow(4);
    let mut masque = vec![0; 4 * v];
    for mu in 1..4 {
        masque[mu * v..mu * v + l.pow(3)].fill(1);
    }
    masque
}

#[derive(Clone, Debug, PartialEq)]
pub enum BordBoite {
    Froid,
    Jauge(Graine),
    Z2Plein,
    Z2Torsion01,
    Flux,
    Haar(Graine),
    /// Configuration d'équilibre périodique source (côté ≥ ℓ + 1).
    Therm(ChampHote),
    /// Bord lu dans un fichier NPY (`--bord-npy`, docs/BORD_NPY.md) : liens fixés copiés au bit.
    Npy(super::boite_npy::BordNpy),
}

impl BordBoite {
    pub fn nom(&self) -> &'static str {
        match self {
            Self::Froid => "cold",
            Self::Jauge(_) => "gauge",
            Self::Z2Plein => "z2full",
            Self::Z2Torsion01 => "z2_01",
            Self::Flux => "flux",
            Self::Haar(_) => "haar",
            Self::Therm(_) => "therm",
            Self::Npy(_) => super::boite_npy::PREFIXE_NPY,
        }
    }
}

pub const NOMS_BORDS: [&str; 7] = ["cold", "gauge", "z2full", "z2_01", "flux", "haar", "therm"];

/// Initialise à l'identité, puis charge exclusivement les liens fixés.
pub fn champ_initial(g: &GeometrieBoite, bord: &BordBoite) -> Resultat<ChampHote> {
    if let BordBoite::Therm(src) = bord {
        let longueur = src.l.checked_pow(4).and_then(|v| v.checked_mul(16));
        if src.l <= g.ell || longueur != Some(src.q.len()) {
            return Err(Erreur::Config(
                "bord thermique : côté source ≥ ℓ + 1 et exactement 16·L_source⁴ composantes requis".into(),
            ));
        }
    }
    if let BordBoite::Npy(b) = bord {
        if b.l_calcul != g.lc || b.liens.len() != 4 * g.n_fixes() {
            return Err(Erreur::Config(format!(
                "bord {} : {} réels pour un réseau de côté {} ; une boîte ℓ = {} demande 4·{} réels (côté {})",
                b.nom,
                b.liens.len(),
                b.l_calcul,
                g.ell,
                g.n_fixes(),
                g.lc
            )));
        }
    }
    let mut champ = ChampHote::froid(g.lc)?;
    // rang du lien fixé courant dans l'ordre li croissant (celui des bords lus)
    let mut k = 0usize;
    for mu in 0..4 {
        for s in 0..g.v() {
            let x = g.coords(s);
            if !g.lien_fixe(mu, x) {
                continue;
            }
            let li = mu * g.v() + s;
            let q = match bord {
                BordBoite::Froid => IDENTITE,
                BordBoite::Z2Plein => {
                    let signe = if x[..mu].iter().sum::<usize>().is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    };
                    [signe, 0.0, 0.0, 0.0]
                }
                BordBoite::Z2Torsion01 => {
                    let signe = if mu == 1 && !x[0].is_multiple_of(2) {
                        -1.0
                    } else {
                        1.0
                    };
                    [signe, 0.0, 0.0, 0.0]
                }
                BordBoite::Flux => {
                    const TABLE: [(f64, f64); 4] =
                        [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
                    match mu {
                        1 => {
                            let (c, s) = TABLE[x[0] % 4];
                            [c, 0.0, 0.0, s]
                        }
                        3 => {
                            let (c, s) = TABLE[x[2] % 4];
                            [c, s, 0.0, 0.0]
                        }
                        _ => IDENTITE,
                    }
                }
                BordBoite::Jauge(graine) => {
                    let (k0, k1) = graine.cle();
                    let mut y = x;
                    y[mu] += 1;
                    let gx = haar_q(s as u32, 0xFFFF_FFFD, ETIQUETTE_JAUGE, k0, k1);
                    let gy = haar_q(g.site(y) as u32, 0xFFFF_FFFD, ETIQUETTE_JAUGE, k0, k1);
                    let produit = qmul(gx, qdag(gy));
                    qscale(produit, 1.0 / qnorm(produit))
                }
                BordBoite::Haar(graine) => {
                    let (k0, k1) = graine.cle();
                    haar_q(li as u32, 0xFFFF_FFFE, ETIQUETTE_BORD, k0, k1)
                }
                BordBoite::Therm(src) => {
                    let ss = ((x[0] * src.l + x[1]) * src.l + x[2]) * src.l + x[3];
                    src.lien(mu * src.v() + ss)
                }
                BordBoite::Npy(b) => {
                    let q = &b.liens[4 * k..4 * k + 4];
                    [q[0], q[1], q[2], q[3]]
                }
            };
            champ.q[4 * li..4 * li + 4].copy_from_slice(&q);
            k += 1;
        }
    }
    Ok(champ)
}

/// Plaquettes rangées par profondeur, puis par plan et site croissants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassesProfil {
    pub nclasses: usize,
    pub comptes: Vec<usize>,
    pub debuts: Vec<u32>,
    /// Code = 8·site + indice du plan.
    pub entrees: Vec<u32>,
}

pub fn classes_profil(g: &GeometrieBoite) -> ClassesProfil {
    let nclasses = g.ell / 2;
    let mut listes = vec![Vec::new(); nclasses];
    for (ip, &(mu, nu)) in PLANS.iter().enumerate() {
        for s in 0..g.v() {
            let x = g.coords(s);
            if !g.dans_boite(x) || x[mu] == g.ell || x[nu] == g.ell {
                continue;
            }
            // Minimum sur les coins : sur chaque axe du plan, le coin avancé
            // rapproche le bord supérieur d'une unité.
            let d = x
                .iter()
                .enumerate()
                .map(|(i, &xi)| xi.min(g.ell - xi - usize::from(i == mu || i == nu)))
                .min()
                .expect("une plaquette possède quatre coordonnées");
            debug_assert!(d < nclasses, "profondeur hors des classes du profil");
            listes[d].push((s as u32) * 8 + ip as u32);
        }
    }
    let comptes: Vec<_> = listes.iter().map(Vec::len).collect();
    let mut debuts = Vec::with_capacity(nclasses + 1);
    let mut entrees = Vec::new();
    debuts.push(0);
    for liste in listes {
        entrees.extend(liste);
        debuts.push(entrees.len() as u32);
    }
    debug_assert_eq!(entrees.len(), 6 * g.ell.pow(2) * (g.ell + 1).pow(2));
    debug_assert!(debuts
        .windows(2)
        .zip(&comptes)
        .all(|(b, &n)| (b[1] - b[0]) as usize == n));
    ClassesProfil {
        nclasses,
        comptes,
        debuts,
        entrees,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObsBoite {
    pub pc: f64,
    /// Boucle centrale 2×2, NaN pour ℓ < 4.
    pub w2: f64,
    pub profil: Vec<f64>,
}

/// Plaquette avec l'association des produits imposée par la référence GPU.
pub fn plaquette(champ: &ChampHote, g: &GeometrieBoite, ip: usize, s: usize) -> f64 {
    let (mu, nu) = PLANS[ip];
    let mut xm = g.coords(s);
    let mut xn = xm;
    xm[mu] = (xm[mu] + 1) % g.lc;
    xn[nu] = (xn[nu] + 1) % g.lc;
    qmul(
        qmul(
            champ.lien(mu * g.v() + s),
            champ.lien(g.indice_lien(nu, xm)),
        ),
        qmul(
            qdag(champ.lien(g.indice_lien(mu, xn))),
            qdag(champ.lien(nu * g.v() + s)),
        ),
    )[0]
}

/// Produit séquentiel : a pas +μ, a pas +ν, a pas −μ, a pas −ν.
pub fn boucle(champ: &ChampHote, g: &GeometrieBoite, ip: usize, s: usize, a: usize) -> f64 {
    let (mu, nu) = PLANS[ip];
    let mut y = g.coords(s);
    let mut q = IDENTITE;
    for (d, avant) in [(mu, true), (nu, true), (mu, false), (nu, false)] {
        for _ in 0..a {
            if avant {
                q = qmul(q, champ.lien(g.indice_lien(d, y)));
                y[d] = (y[d] + 1) % g.lc;
            } else {
                y[d] = (y[d] + g.lc - 1) % g.lc;
                q = qmul(q, qdag(champ.lien(g.indice_lien(d, y))));
            }
        }
    }
    q[0]
}

pub fn mesurer_hote(g: &GeometrieBoite, champ: &ChampHote) -> ObsBoite {
    let mut pc = 0.0;
    for (ip, &(mu, nu)) in PLANS.iter().enumerate() {
        for a in 0..2 {
            for b in 0..2 {
                let mut x = g.centre();
                x[mu] -= a;
                x[nu] -= b;
                pc += plaquette(champ, g, ip, g.site(x));
            }
        }
    }
    pc /= 24.0;
    let w2 = if g.ell >= 4 {
        let mut somme = 0.0;
        for (ip, &(mu, nu)) in PLANS.iter().enumerate() {
            let mut x = g.centre();
            x[mu] -= 1;
            x[nu] -= 1;
            somme += boucle(champ, g, ip, g.site(x), 2);
        }
        somme / 6.0
    } else {
        f64::NAN
    };
    let classes = classes_profil(g);
    let profil = classes
        .debuts
        .windows(2)
        .zip(&classes.comptes)
        .map(|(b, &n)| {
            let valeurs: Vec<_> = classes.entrees[b[0] as usize..b[1] as usize]
                .iter()
                .map(|&code| plaquette(champ, g, (code % 8) as usize, (code / 8) as usize))
                .collect();
            k_final(&valeurs, 1.0) / n as f64
        })
        .collect();
    ObsBoite { pc, w2, profil }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heritage::su2_hb::quat_to_mat;

    fn coordonnees(cote: usize, mut f: impl FnMut([usize; 4])) {
        for x0 in 0..cote {
            for x1 in 0..cote {
                for x2 in 0..cote {
                    for x3 in 0..cote {
                        f([x0, x1, x2, x3]);
                    }
                }
            }
        }
    }

    fn indice(l: usize, mu: usize, x: [usize; 4]) -> usize {
        mu * l.pow(4) + x[0] * l.pow(3) + x[1] * l.pow(2) + x[2] * l + x[3]
    }

    // Classification indépendante : 0 libre, 1 fixé, 2 inexistant.
    fn nature(ell: usize, mu: usize, x: [usize; 4]) -> usize {
        if x.iter().any(|&xi| xi > ell) || x[mu] >= ell {
            return 2;
        }
        let depart = x.contains(&0) || x.contains(&ell);
        let arrivee = (0..4).any(|d| {
            let z = x[d] + usize::from(d == mu);
            z == 0 || z == ell
        });
        usize::from(depart && arrivee)
    }

    fn bits(q: Q) -> [u64; 4] {
        q.map(f64::to_bits)
    }

    fn bits_champ(champ: &ChampHote) -> Vec<u64> {
        champ.q.iter().map(|x| x.to_bits()).collect()
    }

    fn source_haar(l: usize) -> ChampHote {
        let (k0, k1) = Graine(26_102_510_003).cle();
        let mut src = ChampHote {
            l,
            q: vec![0.0; 16 * l.pow(4)],
        };
        for (li, q) in src.q.chunks_exact_mut(4).enumerate() {
            q.copy_from_slice(&haar_q(li as u32, 0xFFFF_FFFE, ETIQUETTE_BORD, k0, k1));
        }
        src
    }

    fn tous_bords() -> Vec<BordBoite> {
        vec![
            BordBoite::Froid,
            BordBoite::Jauge(Graine(26_102_510_001)),
            BordBoite::Z2Plein,
            BordBoite::Z2Torsion01,
            BordBoite::Flux,
            BordBoite::Haar(Graine(26_102_510_002)),
            BordBoite::Therm(source_haar(10)),
        ]
    }

    fn plaquettes_fixees(g: &GeometrieBoite, mut f: impl FnMut(usize, [usize; 4])) {
        coordonnees(g.ell + 1, |x| {
            for (ip, &(mu, nu)) in PLANS.iter().enumerate() {
                if x[mu] == g.ell || x[nu] == g.ell {
                    continue;
                }
                let mut xm = x;
                xm[mu] += 1;
                let mut xn = x;
                xn[nu] += 1;
                if [(mu, x), (nu, xm), (mu, xn), (nu, x)]
                    .iter()
                    .all(|&(d, y)| nature(g.ell, d, y) == 1)
                {
                    f(ip, x);
                }
            }
        });
    }

    #[test]
    fn comptes_et_geometrie() {
        for ell in [2, 4, 6, 8, 10, 12] {
            let g = GeometrieBoite::nouvelle(ell).unwrap();
            let mut comptes = [0; 3];
            coordonnees(g.lc, |x| {
                let s = indice(g.lc, 0, x);
                assert_eq!(g.site(x), s);
                assert_eq!(g.coords(s), x);
                assert_eq!(g.dans_boite(x), x.iter().all(|&z| z <= ell));
                assert_eq!(
                    g.sur_bord(x),
                    x.iter().all(|&z| z <= ell) && (x.contains(&0) || x.contains(&ell))
                );
                for mu in 0..4 {
                    let n = nature(ell, mu, x);
                    comptes[n] += 1;
                    assert_eq!(g.indice_lien(mu, x), indice(g.lc, mu, x));
                    assert_eq!(g.lien_libre(mu, x), n == 0);
                    assert_eq!(g.lien_fixe(mu, x), n == 1);
                    assert_eq!(g.lien_existe(mu, x), n != 2);
                }
            });
            assert_eq!(comptes, [g.n_libres(), g.n_fixes(), g.n_inexistants()]);
            assert_eq!(comptes.iter().sum::<usize>(), 4 * g.v());
            assert_eq!(g.centre(), [ell / 2; 4]);
        }
        assert_eq!(GeometrieBoite::nouvelle(8).unwrap().n_libres(), 10_976);
    }

    #[test]
    fn masques_et_configurations_invalides() {
        for ell in [2, 4, 6] {
            let g = GeometrieBoite::nouvelle(ell).unwrap();
            let mut attendu = vec![0; 4 * g.lc.pow(4)];
            coordonnees(g.lc, |x| {
                for mu in 0..4 {
                    attendu[indice(g.lc, mu, x)] = u8::from(nature(ell, mu, x) != 0);
                }
            });
            let obtenu = masque_boite(&g);
            assert_eq!(obtenu, attendu);
            assert_eq!(obtenu.iter().filter(|&&b| b == 0).count(), g.n_libres());
        }
        for ell in [0, 1, 3, 127, 128, usize::MAX] {
            assert!(matches!(
                GeometrieBoite::nouvelle(ell),
                Err(Erreur::Config(_))
            ));
        }
        assert_eq!(GeometrieBoite::nouvelle(126).unwrap().lc, 128);
    }

    #[test]
    fn masque_du_gel_spatial() {
        for l in [4, 6] {
            let masque = masque_gel_x0_spatial(l);
            assert_eq!(masque.len(), 4 * l.pow(4));
            coordonnees(l, |x| {
                for mu in 0..4 {
                    assert_eq!(masque[indice(l, mu, x)], u8::from(mu != 0 && x[0] == 0));
                }
            });
        }
    }

    #[test]
    fn bords_exacts_et_mutant_z2() {
        for ell in [2, 4, 6] {
            let g = GeometrieBoite::nouvelle(ell).unwrap();
            let froid = champ_initial(&g, &BordBoite::Froid).unwrap();
            let plein = champ_initial(&g, &BordBoite::Z2Plein).unwrap();
            let torsion = champ_initial(&g, &BordBoite::Z2Torsion01).unwrap();
            let flux = champ_initial(&g, &BordBoite::Flux).unwrap();
            let mut mutant_detecte = false;
            let mut vus = [0; 6];
            plaquettes_fixees(&g, |ip, x| {
                vus[ip] += 1;
                let s = g.site(x);
                assert_eq!(plaquette(&froid, &g, ip, s), 1.0);
                assert_eq!(plaquette(&plein, &g, ip, s), -1.0);
                let p = plaquette(&torsion, &g, ip, s);
                assert_eq!(p, if ip == 0 { -1.0 } else { 1.0 });
                // Mutant : le chargeur z2full a pris la définition z2_01.
                mutant_detecte |= p != -1.0;
                if ip == 0 {
                    assert_eq!(plaquette(&flux, &g, ip, s), 0.0);
                } else if ip == 4 {
                    let attendu = if x[0] % 2 == 1 && x[2] % 2 == 1 {
                        -1.0
                    } else {
                        1.0
                    };
                    assert_eq!(plaquette(&flux, &g, ip, s), attendu);
                }
            });
            assert!(vus.iter().all(|&n| n > 0));
            assert!(mutant_detecte);
            for q in flux.q.chunks_exact(4) {
                assert!((qnorm([q[0], q[1], q[2], q[3]]) - 1.0).abs() <= 1e-16);
            }
        }
    }

    #[test]
    fn bord_jauge_pure() {
        let mut ecarts = Vec::new();
        for ell in [2, 4, 6] {
            let g = GeometrieBoite::nouvelle(ell).unwrap();
            let champ = champ_initial(&g, &BordBoite::Jauge(Graine(26_102_510_001))).unwrap();
            let mut ecart = 0.0_f64;
            plaquettes_fixees(&g, |ip, x| {
                ecart = ecart.max((plaquette(&champ, &g, ip, g.site(x)) - 1.0).abs());
            });
            ecarts.push((ell, ecart));
        }
        assert!(
            ecarts.iter().all(|&(_, e)| e <= 3e-15), // GATES-DIR v2 : 4e-16 (v1) ignorait l'accumulation de 8 produits de quaternions
            "écarts de jauge (ℓ, écart) : {ecarts:?}"
        );
    }

    #[test]
    fn bord_haar_normes_graines_et_compteurs() {
        let (k0, k1) = Graine(26_102_510_002).cle();
        for ell in [2, 4, 6] {
            let g = GeometrieBoite::nouvelle(ell).unwrap();
            let a = champ_initial(&g, &BordBoite::Haar(Graine(26_102_510_002))).unwrap();
            let b = champ_initial(&g, &BordBoite::Haar(Graine(26_102_510_003))).unwrap();
            let mut differents = 0;
            coordonnees(g.lc, |x| {
                for mu in 0..4 {
                    if nature(ell, mu, x) == 1 {
                        let li = indice(g.lc, mu, x);
                        let q = a.lien(li);
                        assert!((qnorm(q) - 1.0).abs() <= 4e-16);
                        assert_eq!(
                            bits(q),
                            bits(haar_q(li as u32, 0xFFFF_FFFE, ETIQUETTE_BORD, k0, k1))
                        );
                        differents += usize::from(bits(q) != bits(b.lien(li)));
                    }
                }
            });
            assert!(100 * differents >= 99 * g.n_fixes());
        }
    }

    #[test]
    fn bord_thermique_copie_exacte_et_rejets() {
        let src = source_haar(10);
        for ell in [2, 4, 6] {
            let g = GeometrieBoite::nouvelle(ell).unwrap();
            let champ = champ_initial(&g, &BordBoite::Therm(src.clone())).unwrap();
            coordonnees(g.lc, |x| {
                for mu in 0..4 {
                    if nature(ell, mu, x) == 1 {
                        assert_eq!(
                            bits(champ.lien(indice(g.lc, mu, x))),
                            bits(src.lien(indice(src.l, mu, x)))
                        );
                    }
                }
            });
            for invalide in [
                ChampHote::froid(ell).unwrap(),
                ChampHote { l: 10, q: vec![] },
                ChampHote {
                    l: usize::MAX,
                    q: vec![],
                },
            ] {
                assert!(matches!(
                    champ_initial(&g, &BordBoite::Therm(invalide)),
                    Err(Erreur::Config(_))
                ));
            }
        }
        // Une source de côté impair minimal est également admise.
        let g = GeometrieBoite::nouvelle(4).unwrap();
        assert!(champ_initial(&g, &BordBoite::Therm(source_haar(5))).is_ok());
    }

    #[test]
    fn identite_hors_liens_fixes_et_determinisme() {
        let bords = tous_bords();
        assert_eq!(
            bords.iter().map(BordBoite::nom).collect::<Vec<_>>(),
            NOMS_BORDS
        );
        for ell in [2, 4, 6] {
            let g = GeometrieBoite::nouvelle(ell).unwrap();
            for bord in &bords {
                let a = champ_initial(&g, bord).unwrap();
                let b = champ_initial(&g, bord).unwrap();
                assert_eq!(bits_champ(&a), bits_champ(&b));
                coordonnees(g.lc, |x| {
                    for mu in 0..4 {
                        if nature(ell, mu, x) != 1 {
                            assert_eq!(bits(a.lien(indice(g.lc, mu, x))), bits(IDENTITE));
                        }
                    }
                });
                let m = mesurer_hote(&g, &a);
                let n = mesurer_hote(&g, &a);
                assert_eq!(m.pc.to_bits(), n.pc.to_bits());
                assert_eq!(m.w2.to_bits(), n.w2.to_bits());
                assert_eq!(
                    m.profil.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    n.profil.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn mesures_froides() {
        for ell in [2, 4, 6, 8] {
            let g = GeometrieBoite::nouvelle(ell).unwrap();
            let champ = champ_initial(&g, &BordBoite::Froid).unwrap();
            let obs = mesurer_hote(&g, &champ);
            assert_eq!(obs.pc, 1.0);
            if ell < 4 {
                assert!(obs.w2.is_nan());
            } else {
                assert_eq!(obs.w2, 1.0);
            }
            assert_eq!(obs.profil, vec![1.0; ell / 2]);
        }
    }

    fn profondeur_coins(ell: usize, x: [usize; 4], mu: usize, nu: usize) -> usize {
        let mut d = ell;
        for a in 0..2 {
            for b in 0..2 {
                let mut y = x;
                y[mu] += a;
                y[nu] += b;
                for z in y {
                    d = d.min(z).min(ell - z);
                }
            }
        }
        d
    }

    #[test]
    fn classes_comptes_ordre_et_coins() {
        for ell in [2, 4, 6, 8] {
            let g = GeometrieBoite::nouvelle(ell).unwrap();
            let classes = classes_profil(&g);
            let mut attendues = vec![Vec::new(); ell / 2];
            let mut compte_bord = 0;
            coordonnees(ell + 1, |x| {
                for (ip, &(mu, nu)) in PLANS.iter().enumerate() {
                    if x[mu] < ell && x[nu] < ell {
                        let d = profondeur_coins(ell, x, mu, nu);
                        assert!(d < ell / 2);
                        let s = indice(g.lc, 0, x);
                        attendues[d].push((ip, s));
                        let mut touche = false;
                        for a in 0..2 {
                            for b in 0..2 {
                                let mut y = x;
                                y[mu] += a;
                                y[nu] += b;
                                touche |= y.contains(&0) || y.contains(&ell);
                            }
                        }
                        compte_bord += usize::from(touche);
                    }
                }
            });
            assert_eq!(classes.nclasses, ell / 2);
            assert_eq!(
                classes.comptes.iter().sum::<usize>(),
                6 * ell.pow(2) * (ell + 1).pow(2)
            );
            assert_eq!(classes.debuts.len(), ell / 2 + 1);
            assert_eq!(classes.debuts[0], 0);
            assert_eq!(
                *classes.debuts.last().unwrap() as usize,
                classes.entrees.len()
            );
            for (d, liste) in attendues.iter_mut().enumerate() {
                liste.sort_unstable();
                let codes: Vec<_> = liste
                    .iter()
                    .map(|&(ip, s)| s as u32 * 8 + ip as u32)
                    .collect();
                let debut = classes.debuts[d] as usize;
                let fin = classes.debuts[d + 1] as usize;
                assert_eq!(fin - debut, classes.comptes[d]);
                assert_eq!(classes.comptes[d], liste.len());
                assert_eq!(&classes.entrees[debut..fin], codes);
            }
            if ell == 8 {
                assert_eq!(classes.comptes[0], compte_bord);
            }
        }
    }

    // Référence indépendante par matrices complexes, sans produit quaternionique.
    fn produit_matriciel(a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        let mut c = [0.0; 8];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let u = 2 * (2 * i + k);
                    let v = 2 * (2 * k + j);
                    let w = 2 * (2 * i + j);
                    c[w] += a[u] * b[v] - a[u + 1] * b[v + 1];
                    c[w + 1] += a[u] * b[v + 1] + a[u + 1] * b[v];
                }
            }
        }
        c
    }

    fn boucle_matricielle(champ: &ChampHote, x: [usize; 4], mu: usize, nu: usize, a: usize) -> f64 {
        let mut y = x;
        let mut matrice = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let chemin: Vec<_> = [mu, nu, mu, nu]
            .iter()
            .enumerate()
            .flat_map(|(i, &d)| std::iter::repeat_n((d, i < 2), a))
            .collect();
        for (d, avant) in chemin {
            if !avant {
                y[d] -= 1;
            }
            let mut lien = quat_to_mat(champ.lien(indice(champ.l, d, y)));
            if avant {
                y[d] += 1;
            } else {
                lien = [
                    lien[0], -lien[1], lien[4], -lien[5], lien[2], -lien[3], lien[6], -lien[7],
                ];
            }
            matrice = produit_matriciel(matrice, lien);
        }
        assert_eq!(y, x);
        (matrice[0] + matrice[6]) / 2.0
    }

    fn champ_aleatoire(g: &GeometrieBoite) -> ChampHote {
        let mut champ = ChampHote::froid(g.lc).unwrap();
        let (k0, k1) = Graine(0x1234_5678_9ABC_DEF0).cle();
        coordonnees(g.ell + 1, |x| {
            for mu in 0..4 {
                if x[mu] < g.ell {
                    let li = indice(g.lc, mu, x);
                    let q = haar_q(li as u32, 0xFFFF_FFFE, ETIQUETTE_BORD, k0, k1);
                    champ.q[4 * li..4 * li + 4].copy_from_slice(&q);
                }
            }
        });
        champ
    }

    #[test]
    fn mesures_contre_reference_matricielle() {
        for ell in [4, 6] {
            let g = GeometrieBoite::nouvelle(ell).unwrap();
            let champ = champ_aleatoire(&g);
            let obs = mesurer_hote(&g, &champ);
            let mut pc = 0.0;
            let mut n_pc = 0;
            let mut w2 = 0.0;
            let mut profil = vec![0.0; ell / 2];
            let mut comptes = vec![0; ell / 2];
            // Sites avant plans, plans inversés : ordre différent de la production.
            coordonnees(ell + 1, |x| {
                for mu in (0..4).rev() {
                    for nu in (mu + 1..4).rev() {
                        if x[mu] >= ell || x[nu] >= ell {
                            continue;
                        }
                        let p = boucle_matricielle(&champ, x, mu, nu, 1);
                        let d = profondeur_coins(ell, x, mu, nu);
                        profil[d] += p;
                        comptes[d] += 1;
                        let c = ell / 2;
                        if (0..4).all(|i| {
                            if i == mu || i == nu {
                                x[i] == c || x[i] + 1 == c
                            } else {
                                x[i] == c
                            }
                        }) {
                            pc += p;
                            n_pc += 1;
                        }
                        if (0..4).all(|i| x[i] + usize::from(i == mu || i == nu) == c) {
                            w2 += boucle_matricielle(&champ, x, mu, nu, 2);
                        }
                    }
                }
            });
            assert_eq!(n_pc, 24);
            assert!((obs.pc - pc / 24.0).abs() <= 1e-13);
            assert!((obs.w2 - w2 / 6.0).abs() <= 1e-13);
            for d in 0..ell / 2 {
                assert!((obs.profil[d] - profil[d] / comptes[d] as f64).abs() <= 1e-13);
            }
        }
    }

    #[test]
    fn ordre_contractuel_des_sommes() {
        let g = GeometrieBoite::nouvelle(6).unwrap();
        let champ = champ_aleatoire(&g);
        let obs = mesurer_hote(&g, &champ);
        let mut pc = 0.0;
        let mut w2 = 0.0;
        let mut listes = vec![Vec::new(); g.ell / 2];
        // Lecture des liens indépendante de la géométrie de production.
        let p = |x: [usize; 4], mu: usize, nu: usize| {
            let mut xm = x;
            let mut xn = x;
            xm[mu] += 1;
            xn[nu] += 1;
            let a = champ.lien(indice(g.lc, mu, x));
            let b = champ.lien(indice(g.lc, nu, xm));
            let c = champ.lien(indice(g.lc, mu, xn));
            let d = champ.lien(indice(g.lc, nu, x));
            qmul(qmul(a, b), qmul(qdag(c), qdag(d)))[0]
        };
        for mu in 0..4 {
            for nu in mu + 1..4 {
                for a in 0..2 {
                    for b in 0..2 {
                        let mut x = [3; 4];
                        x[mu] -= a;
                        x[nu] -= b;
                        pc += p(x, mu, nu);
                    }
                }
                let mut y = [3; 4];
                y[mu] -= 1;
                y[nu] -= 1;
                let mut q = IDENTITE;
                for d in [mu, mu, nu, nu] {
                    q = qmul(q, champ.lien(indice(g.lc, d, y)));
                    y[d] += 1;
                }
                for d in [mu, mu, nu, nu] {
                    y[d] -= 1;
                    q = qmul(q, qdag(champ.lien(indice(g.lc, d, y))));
                }
                w2 += q[0];
                coordonnees(g.ell + 1, |x| {
                    if x[mu] < g.ell && x[nu] < g.ell {
                        listes[profondeur_coins(g.ell, x, mu, nu)].push(p(x, mu, nu));
                    }
                });
            }
        }
        assert_eq!(obs.pc.to_bits(), (pc / 24.0).to_bits());
        assert_eq!(obs.w2.to_bits(), (w2 / 6.0).to_bits());
        for (d, valeurs) in listes.iter().enumerate() {
            assert_eq!(
                obs.profil[d].to_bits(),
                (k_final(valeurs, 1.0) / valeurs.len() as f64).to_bits()
            );
        }
    }
}
