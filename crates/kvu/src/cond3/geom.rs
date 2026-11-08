//! Géométrie du blocage de facteur 3 en dimension d ∈ {2, 4} (T3-D, SPEC_T3 §1.1, §1.4, §1.7).
//!
//! Indices canoniques (un seul par objet) :
//! - site `s` : indice C de (x₀, …, x_{d−1}), x₀ le plus lent ; lien fin `li = mu·V + s` (de x à x + e_mu) ;
//! - plaquette `pi = plan·V + s`, plans (mu < nu) dans l'ordre lexicographique ;
//! - lien grossier `ci = mu·Vc + Sc` (de 3X à 3X + 3 e_mu) ; ligne = les trois liens (mu, 3X + j e_mu), j = 0, 1, 2.
//!
//! L'origine du blocage DEC est 3X (jamais le centre 3z + (1,1,1,1) de la carte 3centre).

use crate::commun::{Erreur, Resultat};

/// Réseau fin L^d, réseau grossier (L/3)^d et toutes les tables d'indices dérivées.
#[derive(Clone, Debug)]
pub struct Geom {
    pub d: usize,
    pub l: usize,
    pub v: usize,
    pub lc: usize,
    pub vc: usize,
    pub planes: Vec<(usize, usize)>,
    /// nbr[mu][s] : site de x + e_mu ; nbm[mu][s] : site de x − e_mu.
    pub nbr: Vec<Vec<u32>>,
    pub nbm: Vec<Vec<u32>>,
    /// Coordonnées des sites fins (les d premières composantes sont utilisées).
    pub coords: Vec<[u16; 4]>,
    /// Par lien fin : −1 (libre) ou j ∈ {0, 1, 2} (position dans sa ligne).
    pub pos: Vec<i8>,
    /// Par lien fin : lien grossier de sa ligne (u32::MAX si libre).
    pub ligne_de: Vec<u32>,
    /// Par lien grossier : les trois liens fins de la ligne.
    pub lignes: Vec<[u32; 3]>,
    /// libres[mu][parité] : sites s des liens libres (mu, s) de parité Σx mod 2, ordre croissant.
    pub libres: Vec<[Vec<u32>; 2]>,
    /// tous[mu][parité] : tous les liens (mu, s) de la parité (mode sans contrainte).
    pub tous: Vec<[Vec<u32>; 2]>,
    /// lignes_dir[mu] : liens grossiers ci des lignes de direction mu, ordre croissant.
    pub lignes_dir: Vec<Vec<u32>>,
    /// Coordonnées des sites grossiers.
    pub coords_gros: Vec<[u16; 4]>,
    /// Cache : par plaquette, (classe t, nombre de liens de ligne qu'elle contient).
    cls: std::sync::OnceLock<Vec<[u8; 2]>>,
}

impl Geom {
    /// d ∈ {2, 4}, L multiple de 6 (pair pour le damier, multiple de 3 pour le blocage).
    pub fn nouvelle(d: usize, l: usize) -> Resultat<Geom> {
        if d != 2 && d != 4 {
            return Err(Erreur::Config(format!("Geom : d = {d} (2 ou 4 requis)")));
        }
        if l < 6 || !l.is_multiple_of(6) {
            return Err(Erreur::Config(format!("Geom : L = {l} (multiple de 6 requis : damier pair et blocage de facteur 3)")));
        }
        let v = l.pow(d as u32);
        if (d * v) as u64 >= u32::MAX as u64 {
            return Err(Erreur::Config(format!("Geom : d·L^d = {} ne tient pas dans u32", d * v)));
        }
        let lc = l / 3;
        let vc = lc.pow(d as u32);
        let planes: Vec<(usize, usize)> = (0..d).flat_map(|m| (m + 1..d).map(move |n| (m, n))).collect();
        let coords: Vec<[u16; 4]> = (0..v).map(|s| coords_de(s, d, l)).collect();
        let coords_gros: Vec<[u16; 4]> = (0..vc).map(|s| coords_de(s, d, lc)).collect();
        let mut nbr = vec![vec![0u32; v]; d];
        let mut nbm = vec![vec![0u32; v]; d];
        for s in 0..v {
            for mu in 0..d {
                let mut x = coords[s];
                x[mu] = ((x[mu] as usize + 1) % l) as u16;
                nbr[mu][s] = site_de(&x, d, l) as u32;
                let mut y = coords[s];
                y[mu] = ((y[mu] as usize + l - 1) % l) as u16;
                nbm[mu][s] = site_de(&y, d, l) as u32;
            }
        }
        // lignes
        let mut lignes = vec![[0u32; 3]; d * vc];
        let mut pos = vec![-1i8; d * v];
        let mut ligne_de = vec![u32::MAX; d * v];
        let mut lignes_dir: Vec<Vec<u32>> = vec![Vec::new(); d];
        for mu in 0..d {
            for sc in 0..vc {
                let ci = mu * vc + sc;
                let mut x = coords_gros[sc];
                for c in x.iter_mut().take(d) {
                    *c *= 3;
                }
                for j in 0..3 {
                    let mut y = x;
                    y[mu] = ((x[mu] as usize + j) % l) as u16;
                    let li = mu * v + site_de(&y, d, l);
                    lignes[ci][j] = li as u32;
                    if pos[li] != -1 {
                        return Err(Erreur::Validation(format!("Geom : le lien {li} appartient à deux lignes")));
                    }
                    pos[li] = j as i8;
                    ligne_de[li] = ci as u32;
                }
                lignes_dir[mu].push(ci as u32);
            }
        }
        let mut libres: Vec<[Vec<u32>; 2]> = (0..d).map(|_| [Vec::new(), Vec::new()]).collect();
        let mut tous: Vec<[Vec<u32>; 2]> = (0..d).map(|_| [Vec::new(), Vec::new()]).collect();
        for s in 0..v {
            let par = coords[s].iter().take(d).map(|&c| c as usize).sum::<usize>() % 2;
            for mu in 0..d {
                tous[mu][par].push(s as u32);
                if pos[mu * v + s] == -1 {
                    libres[mu][par].push(s as u32);
                }
            }
        }
        Ok(Geom { d, l, v, lc, vc, planes, nbr, nbm, coords, pos, ligne_de, lignes, libres, tous, lignes_dir, coords_gros, cls: std::sync::OnceLock::new() })
    }

    pub fn nliens(&self) -> usize {
        self.d * self.v
    }
    pub fn nplans(&self) -> usize {
        self.planes.len()
    }
    pub fn nplaq(&self) -> usize {
        self.planes.len() * self.v
    }
    pub fn ngros(&self) -> usize {
        self.d * self.vc
    }

    /// Site d'indice C de coordonnées `x` (réduites modulo L).
    pub fn site(&self, x: &[i64]) -> usize {
        let mut s = 0usize;
        for &c in x.iter().take(self.d) {
            s = s * self.l + c.rem_euclid(self.l as i64) as usize;
        }
        s
    }

    /// Site grossier d'indice C de X (réduit modulo L/3).
    pub fn site_gros(&self, x: &[i64]) -> usize {
        let mut s = 0usize;
        for &c in x.iter().take(self.d) {
            s = s * self.lc + c.rem_euclid(self.lc as i64) as usize;
        }
        s
    }

    /// Les quatre liens (a, b, c, d) de la plaquette (plan, s) : U_p = U_a U_b U_c† U_d†.
    pub fn liens_plaquette(&self, plan: usize, s: usize) -> [u32; 4] {
        let (mu, nu) = self.planes[plan];
        [
            (mu * self.v + s) as u32,
            (nu * self.v) as u32 + self.nbr[mu][s],
            (mu * self.v) as u32 + self.nbr[nu][s],
            (nu * self.v + s) as u32,
        ]
    }

    /// Classe t d'une plaquette : nombre de coordonnées TRANSVERSES (hors {mu, nu}) congrues à 0 mod 3.
    pub fn classe_t(&self, pi: usize) -> usize {
        let plan = pi / self.v;
        let s = pi % self.v;
        let (mu, nu) = self.planes[plan];
        (0..self.d).filter(|&r| r != mu && r != nu && self.coords[s][r].is_multiple_of(3)).count()
    }

    /// Par plaquette : (classe t, nombre de liens de ligne contenus), calculé une fois.
    pub fn classes_plaquettes(&self) -> &Vec<[u8; 2]> {
        self.cls.get_or_init(|| {
            (0..self.nplaq())
                .map(|pi| {
                    let l = self.liens_plaquette(pi / self.v, pi % self.v);
                    let nl = l.iter().filter(|&&x| self.pos[x as usize] != -1).count();
                    [self.classe_t(pi) as u8, nl as u8]
                })
                .collect()
        })
    }

    /// Plaquettes de classe t = d − 2 (faces des plaquettes grossières ; en d = 2 : toutes), ordre croissant de `pi`.
    pub fn faces(&self) -> Vec<u32> {
        (0..self.nplaq()).filter(|&pi| self.classe_t(pi) == self.d - 2).map(|pi| pi as u32).collect()
    }

    /// Sondes : liens grossiers (X, mu) dont TOUTES les coordonnées de X sont paires, ordre croissant de `ci`.
    pub fn sondes_paires(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for mu in 0..self.d {
            for sc in 0..self.vc {
                if self.coords_gros[sc].iter().take(self.d).all(|&c| c.is_multiple_of(2)) {
                    out.push((mu * self.vc + sc) as u32);
                }
            }
        }
        out
    }

    /// Tous les liens grossiers (sondes des oracles 2D).
    pub fn tous_liens_gros(&self) -> Vec<u32> {
        (0..self.ngros() as u32).collect()
    }

    /// Centre doublé 2·c_p d'une plaquette (entiers) : 2x + e_mu + e_nu.
    pub fn centre2_plaquette(&self, pi: usize) -> [i32; 4] {
        let plan = pi / self.v;
        let s = pi % self.v;
        let (mu, nu) = self.planes[plan];
        let mut c = [0i32; 4];
        for r in 0..self.d {
            c[r] = 2 * self.coords[s][r] as i32;
        }
        c[mu] += 1;
        c[nu] += 1;
        c
    }

    /// Milieu doublé 2·m_ē d'un lien grossier : 6X + 3 e_mu.
    pub fn milieu2_gros(&self, ci: usize) -> [i32; 4] {
        let mu = ci / self.vc;
        let sc = ci % self.vc;
        let mut m = [0i32; 4];
        for r in 0..self.d {
            m[r] = 6 * self.coords_gros[sc][r] as i32;
        }
        m[mu] += 3;
        m
    }

    /// Norme carrée DOUBLÉE (entière) de la différence des centres à l'image minimale :
    /// S = Σ (Δ2_i)², Δ2 = 2c_p − 2m_ē réduit dans ]−L, L] ; r = √S / 2.
    #[inline]
    pub fn dist2_doublee(&self, cp2: &[i32; 4], me2: &[i32; 4]) -> u32 {
        let l2 = 2 * self.l as i32;
        let l = self.l as i32;
        let mut s = 0i32;
        for r in 0..self.d {
            let mut dlt = (cp2[r] - me2[r]).rem_euclid(l2);
            if dlt > l {
                dlt -= l2;
            }
            s += dlt * dlt;
        }
        s as u32
    }
}

/// Tranche b = ⌊r + 1/2⌋ à partir de S = (2r)² entier : b = ⌊(⌊√S⌋ + 1)/2⌋ (arithmétique entière exacte).
#[inline]
pub fn tranche_de_s(s: u32) -> u32 {
    let mut r = (s as f64).sqrt() as u32;
    while r * r > s {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= s {
        r += 1;
    }
    r.div_ceil(2)
}

fn coords_de(mut s: usize, d: usize, l: usize) -> [u16; 4] {
    let mut c = [0u16; 4];
    for r in (0..d).rev() {
        c[r] = (s % l) as u16;
        s /= l;
    }
    c
}

fn site_de(x: &[u16; 4], d: usize, l: usize) -> usize {
    let mut s = 0usize;
    for &c in x.iter().take(d) {
        s = s * l + c as usize;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effectifs_l12_l24() {
        for (l, liens, gros, lignes, libres, plaq, faces, sondes) in
            [(12usize, 82_944usize, 1024usize, 3072usize, 79_872usize, 124_416usize, 13_824usize, 64usize),
             (24, 1_327_104, 16_384, 49_152, 1_277_952, 1_990_656, 221_184, 1024)]
        {
            let g = Geom::nouvelle(4, l).unwrap();
            assert_eq!(g.nliens(), liens);
            assert_eq!(g.ngros(), gros);
            assert_eq!(g.lignes.len() * 3, lignes);
            assert_eq!(g.pos.iter().filter(|&&p| p == -1).count(), libres);
            assert_eq!(g.nplaq(), plaq);
            assert_eq!(g.faces().len(), faces);
            assert_eq!(g.sondes_paires().len(), sondes);
        }
    }

    #[test]
    fn chaque_lien_de_ligne_dans_une_seule_ligne() {
        for d in [2usize, 4] {
            let g = Geom::nouvelle(d, 6).unwrap();
            let mut vu = vec![0u8; g.nliens()];
            for lig in &g.lignes {
                for (j, &li) in lig.iter().enumerate() {
                    vu[li as usize] += 1;
                    assert_eq!(g.pos[li as usize], j as i8);
                }
            }
            assert!(vu.iter().all(|&c| c <= 1));
            for li in 0..g.nliens() {
                assert_eq!(g.pos[li] == -1, vu[li] == 0);
            }
            // libres : partition des liens non de ligne, par direction et parité
            for mu in 0..d {
                let n: usize = g.libres[mu][0].len() + g.libres[mu][1].len();
                assert_eq!(n, g.v - 3 * g.vc);
            }
        }
    }

    #[test]
    fn tranche_entiere_equivaut_a_floor() {
        for s in 0..4000u32 {
            let r = (s as f64).sqrt() / 2.0;
            assert_eq!(tranche_de_s(s) as f64, (r + 0.5).floor(), "S = {s}");
        }
    }

    #[test]
    fn classe_t_en_2d_est_nulle_et_faces_sont_toutes() {
        let g = Geom::nouvelle(2, 12).unwrap();
        assert!((0..g.nplaq()).all(|pi| g.classe_t(pi) == 0));
        assert_eq!(g.faces().len(), g.nplaq());
    }
}
