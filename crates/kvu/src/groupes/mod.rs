//! Algèbre matricielle complexe et registre des groupes de jauge.

pub mod batterie_generique;
pub mod cli;
pub mod gates;
pub mod gen_cli;
pub mod generique;
pub mod sp4;
pub mod su2;

/// Matrice complexe carrée, stockée ligne par ligne.
#[derive(Clone, Debug, PartialEq)]
pub struct Cmat {
    pub n: usize,
    pub re: Vec<f64>,
    pub im: Vec<f64>,
}

pub(super) fn produit(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

pub(super) fn quotient(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    // Division mise à l'échelle pour éviter de former |b|².
    let echelle = b.0.abs().max(b.1.abs());
    let (br, bi) = (b.0 / echelle, b.1 / echelle);
    let (ar, ai) = (a.0 / echelle, a.1 / echelle);
    let norme = br * br + bi * bi;
    ((ar * br + ai * bi) / norme, (ai * br - ar * bi) / norme)
}

impl Cmat {
    pub fn zeros(n: usize) -> Self {
        let taille = n.checked_mul(n).expect("Dimension matricielle trop grande");
        Self {
            n,
            re: vec![0.0; taille],
            im: vec![0.0; taille],
        }
    }

    pub fn identite(n: usize) -> Self {
        let mut m = Self::zeros(n);
        for i in 0..n {
            m.set(i, i, (1.0, 0.0));
        }
        m
    }

    fn verifier(&self) {
        let taille = self
            .n
            .checked_mul(self.n)
            .expect("Dimension matricielle trop grande");
        assert!(
            self.re.len() == taille && self.im.len() == taille,
            "Stockage incompatible avec la dimension de Cmat"
        );
    }

    fn compatibles(&self, autre: &Self) {
        self.verifier();
        autre.verifier();
        assert_eq!(self.n, autre.n, "Dimensions matricielles incompatibles");
    }

    pub fn get(&self, i: usize, j: usize) -> (f64, f64) {
        self.verifier();
        assert!(i < self.n && j < self.n, "Indices matriciels hors limites");
        let k = i * self.n + j;
        (self.re[k], self.im[k])
    }

    pub fn set(&mut self, i: usize, j: usize, z: (f64, f64)) {
        self.verifier();
        assert!(i < self.n && j < self.n, "Indices matriciels hors limites");
        let k = i * self.n + j;
        self.re[k] = z.0;
        self.im[k] = z.1;
    }

    pub fn depuis_entrelace(n: usize, valeurs: &[f64]) -> Self {
        let mut m = Self::zeros(n);
        let taille =
            m.re.len()
                .checked_mul(2)
                .expect("Stockage entrelacé trop grand");
        assert_eq!(
            valeurs.len(),
            taille,
            "Longueur incompatible avec la dimension de Cmat"
        );
        for (k, z) in valeurs.chunks_exact(2).enumerate() {
            m.re[k] = z[0];
            m.im[k] = z[1];
        }
        m
    }

    pub fn vers_entrelace(&self) -> Vec<f64> {
        self.verifier();
        self.re
            .iter()
            .zip(&self.im)
            .flat_map(|(&r, &i)| [r, i])
            .collect()
    }

    pub fn mul(&self, autre: &Self) -> Self {
        self.compatibles(autre);
        let mut m = Self::zeros(self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                let mut z = (0.0, 0.0);
                for k in 0..self.n {
                    let p = produit(self.get(i, k), autre.get(k, j));
                    z.0 += p.0;
                    z.1 += p.1;
                }
                m.set(i, j, z);
            }
        }
        m
    }

    pub fn dag(&self) -> Self {
        self.verifier();
        let mut m = Self::zeros(self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                let z = self.get(j, i);
                m.set(i, j, (z.0, -z.1));
            }
        }
        m
    }

    pub fn add(&self, autre: &Self) -> Self {
        self.compatibles(autre);
        let mut m = self.clone();
        for k in 0..m.re.len() {
            m.re[k] += autre.re[k];
            m.im[k] += autre.im[k];
        }
        m
    }

    pub fn sub(&self, autre: &Self) -> Self {
        self.compatibles(autre);
        let mut m = self.clone();
        for k in 0..m.re.len() {
            m.re[k] -= autre.re[k];
            m.im[k] -= autre.im[k];
        }
        m
    }

    pub fn echelle(&self, facteur: f64) -> Self {
        self.echelle_c((facteur, 0.0))
    }

    pub fn echelle_c(&self, facteur: (f64, f64)) -> Self {
        self.verifier();
        let mut m = self.clone();
        for k in 0..m.re.len() {
            (m.re[k], m.im[k]) = produit((self.re[k], self.im[k]), facteur);
        }
        m
    }

    pub fn trace(&self) -> (f64, f64) {
        self.verifier();
        let mut z = (0.0, 0.0);
        for i in 0..self.n {
            let d = self.get(i, i);
            z.0 += d.0;
            z.1 += d.1;
        }
        z
    }

    /// Élimination de Gauss avec pivot partiel en module complexe ; det(0×0) = 1.
    pub fn det(&self) -> (f64, f64) {
        self.verifier();
        let mut m = self.clone();
        let mut determinant = (1.0, 0.0);
        for k in 0..self.n {
            let mut pivot = k;
            let mut module = 0.0;
            for i in k..self.n {
                let z = m.get(i, k);
                let candidat = z.0.hypot(z.1);
                if candidat > module {
                    module = candidat;
                    pivot = i;
                }
            }
            if module == 0.0 {
                return (0.0, 0.0);
            }
            if pivot != k {
                for j in 0..self.n {
                    m.re.swap(k * self.n + j, pivot * self.n + j);
                    m.im.swap(k * self.n + j, pivot * self.n + j);
                }
                determinant = (-determinant.0, -determinant.1);
            }
            let diagonale = m.get(k, k);
            determinant = produit(determinant, diagonale);
            for i in k + 1..self.n {
                let facteur = quotient(m.get(i, k), diagonale);
                m.set(i, k, (0.0, 0.0));
                for j in k + 1..self.n {
                    let z = m.get(i, j);
                    let correction = produit(facteur, m.get(k, j));
                    m.set(i, j, (z.0 - correction.0, z.1 - correction.1));
                }
            }
        }
        determinant
    }

    pub fn norme_frobenius(&self) -> f64 {
        self.verifier();
        self.re
            .iter()
            .chain(&self.im)
            .fold(0.0, |norme, &x| norme.hypot(x))
    }

    pub fn commutateur(&self, autre: &Self) -> Self {
        self.mul(autre).sub(&autre.mul(self))
    }

    pub fn max_abs_diff(&self, autre: &Self) -> f64 {
        self.compatibles(autre);
        self.re
            .iter()
            .zip(&self.im)
            .zip(autre.re.iter().zip(&autre.im))
            .map(|((&ar, &ai), (&br, &bi))| (ar - br).hypot(ai - bi))
            .fold(0.0, f64::max)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Realite {
    Reel,
    Pseudoreel,
    Complexe,
}

pub const ETIQUETTE_GROUPE: u32 = 0x4752_4F55;

/// Compteur de tirages Philox, indépendant des autres usages de la graine.
pub struct Flux {
    k0: u32,
    k1: u32,
    etiquette: u32,
    n: u64,
}

impl Flux {
    pub fn nouveau(graine: crate::commun::Graine, etiquette: u32) -> Self {
        let (k0, k1) = graine.cle();
        Self {
            k0,
            k1,
            etiquette,
            n: 0,
        }
    }

    pub fn suivant(&mut self) -> (u32, u32, u32, u32, u32) {
        let n = self.n;
        self.n = n.checked_add(1).expect("Compteur du Flux épuisé");
        (n as u32, (n >> 32) as u32, self.k0, self.k1, self.etiquette)
    }

    pub fn tirages(&self) -> u64 {
        self.n
    }
}

pub struct Appartenance {
    pub unitarite: f64,
    pub det: f64,
    pub structure: f64,
}

pub struct LogU {
    pub a: Vec<f64>,
    pub theta: f64,
    pub ambigue: bool,
    pub hors_domaine: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompteBranches {
    pub total: u64,
    pub ambigues: u64,
    pub hors_domaine: u64,
}

impl CompteBranches {
    pub fn ajouter(&mut self, log: &LogU) {
        self.total += 1;
        self.ambigues += u64::from(log.ambigue);
        self.hors_domaine += u64::from(log.hors_domaine);
    }
}

/// Conventions : exp(i aᵃ Tᵃ), Tr(Tᵃ Tᵇ) = T_F δᵃᵇ.
pub trait Groupe: Send + Sync {
    fn nom(&self) -> &'static str;
    fn d_f(&self) -> usize;
    fn d_adj(&self) -> usize;
    fn realite(&self) -> Realite;
    fn h_vee(&self) -> usize;
    fn generateurs(&self) -> Vec<Cmat>;

    /// Normalisation mesurée sur les générateurs effectivement fournis.
    fn t_f(&self) -> f64 {
        let generateurs = self.generateurs();
        assert!(
            !generateurs.is_empty(),
            "Le groupe doit fournir au moins un générateur"
        );
        generateurs.iter().map(|t| t.mul(t).trace().0).sum::<f64>() / generateurs.len() as f64
    }

    /// Tableau d³, avec d = nombre réel de générateurs, y compris pour les mutants.
    fn f_abc(&self) -> Vec<f64> {
        let generateurs = self.generateurs();
        let d = generateurs.len();
        let t_f = self.t_f();
        assert!(
            t_f > 0.0,
            "La normalisation des générateurs doit être positive"
        );
        let mut f = vec![0.0; d * d * d];
        for (a, ta) in generateurs.iter().enumerate() {
            for (b, tb) in generateurs.iter().enumerate() {
                let commutateur = ta.commutateur(tb);
                for (c, tc) in generateurs.iter().enumerate() {
                    f[(a * d + b) * d + c] = commutateur.mul(tc).trace().1 / t_f;
                }
            }
        }
        f
    }

    fn appartient(&self, u: &Cmat) -> Appartenance;
    fn haar(&self, rng: &mut Flux) -> Cmat;
    fn projeter(&self, m: &Cmat) -> Cmat;
    fn exp(&self, a: &[f64]) -> Cmat;
    fn log(&self, u: &Cmat) -> LogU;
}

pub fn noms_registre() -> Vec<&'static str> {
    vec!["su2", "sp4"]
}

pub fn trouver(nom: &str) -> Option<Box<dyn Groupe>> {
    match nom {
        "su2" => Some(Box::new(su2::Su2::nouveau())),
        "sp4" => Some(Box::new(sp4::Sp4::nouveau())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commun::Graine;

    fn proche(a: (f64, f64), b: (f64, f64)) {
        assert!(
            (a.0 - b.0).hypot(a.1 - b.1) < 1e-12,
            "Valeurs différentes : {a:?}, {b:?}"
        );
    }

    #[test]
    fn algebre_deux_dimensions() {
        let a = Cmat::depuis_entrelace(2, &[1., 1., 2., 0., 0., 3., 4., -1.]);
        let b = Cmat::depuis_entrelace(2, &[0., 0., 1., 0., 1., 0., 0., 0.]);
        assert_eq!(
            a.mul(&b),
            Cmat::depuis_entrelace(2, &[2., 0., 1., 1., 4., -1., 0., 3.])
        );
        assert_eq!(
            a.dag(),
            Cmat::depuis_entrelace(2, &[1., -1., 0., -3., 2., 0., 4., 1.])
        );
        proche(a.trace(), (5., 0.));
        proche(a.det(), (5., -3.));
        proche(a.mul(&b).det(), produit(a.det(), b.det()));
        assert_eq!(
            a.commutateur(&b),
            Cmat::depuis_entrelace(2, &[2., -3., -3., 2., 3., -2., -2., 3.])
        );
        assert_eq!(a.add(&b).sub(&b), a);
        assert_eq!(a.echelle(2.).echelle(0.5), a);
        assert_eq!(
            a.echelle_c((0., 1.)),
            Cmat::depuis_entrelace(2, &[-1., 1., 0., 2., -3., 0., 1., 4.])
        );
        assert!((a.norme_frobenius() - 32_f64.sqrt()).abs() < 1e-14);
        assert_eq!(
            a.max_abs_diff(&a.add(&Cmat::identite(2).echelle_c((3., 4.)))),
            5.
        );
        assert_eq!(Cmat::depuis_entrelace(2, &a.vers_entrelace()), a);
    }

    #[test]
    fn algebre_trois_dimensions_et_pivots() {
        let a = Cmat::depuis_entrelace(
            3,
            &[
                0., 0., 1., 1., 0., 0., 2., 0., 3., 0., 0., 0., 0., 0., 0., 0., 4., -1.,
            ],
        );
        let carre = Cmat::depuis_entrelace(
            3,
            &[
                2., 2., 3., 3., 0., 0., 6., 0., 11., 2., 0., 0., 0., 0., 0., 0., 15., -8.,
            ],
        );
        assert_eq!(a.mul(&a), carre);
        assert_eq!(
            a.dag(),
            Cmat::depuis_entrelace(
                3,
                &[0., 0., 2., 0., 0., 0., 1., -1., 3., 0., 0., 0., 0., 0., 0., 0., 4., 1.]
            )
        );
        proche(a.trace(), (7., -1.));
        proche(a.det(), (-10., -6.));
        proche(carre.det(), (64., 120.));
        proche(carre.det(), produit(a.det(), a.det()));
        assert_eq!(a.dag().dag(), a);
        assert_eq!(a.commutateur(&Cmat::identite(3)), Cmat::zeros(3));
        assert_eq!(Cmat::zeros(3).det(), (0., 0.));
        assert_eq!(Cmat::zeros(0).det(), (1., 0.));
        assert_eq!(Cmat::identite(1).det(), (1., 0.));
    }

    #[test]
    #[should_panic(expected = "Dimensions matricielles incompatibles")]
    fn dimensions_incompatibles() {
        Cmat::zeros(2).mul(&Cmat::zeros(3));
    }

    #[test]
    #[should_panic(expected = "Longueur incompatible")]
    fn entrelacement_incompatible() {
        Cmat::depuis_entrelace(2, &[0.; 7]);
    }

    #[test]
    #[should_panic(expected = "Indices matriciels hors limites")]
    fn indices_incompatibles() {
        Cmat::zeros(2).get(0, 2);
    }

    #[test]
    fn flux_et_passage_au_slot_suivant() {
        let mut flux = Flux::nouveau(Graine(0x1234_5678_ABCD_EF01), ETIQUETTE_GROUPE);
        assert_eq!(flux.tirages(), 0);
        assert_eq!(
            flux.suivant(),
            (0, 0, 0xABCD_EF01, 0x1234_5678, ETIQUETTE_GROUPE)
        );
        flux.n = u32::MAX as u64;
        assert_eq!(flux.suivant().0, u32::MAX);
        assert_eq!(
            flux.suivant(),
            (0, 1, 0xABCD_EF01, 0x1234_5678, ETIQUETTE_GROUPE)
        );
        assert_eq!(flux.tirages(), (1_u64 << 32) + 1);
    }

    #[test]
    fn registre_et_branches() {
        assert_eq!(noms_registre(), vec!["su2", "sp4"]);
        let groupe = trouver("su2").unwrap();
        assert_eq!(groupe.nom(), "su2");
        for nom in ["g2", "SU2", ""] {
            assert!(trouver(nom).is_none());
        }
        let mut compte = CompteBranches::default();
        compte.ajouter(&groupe.log(&Cmat::identite(2)));
        compte.ajouter(&groupe.log(&Cmat::identite(2).echelle(-1.)));
        assert_eq!(
            compte,
            CompteBranches {
                total: 2,
                ambigues: 1,
                hors_domaine: 1
            }
        );
    }
}
