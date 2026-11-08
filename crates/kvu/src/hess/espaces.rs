//! Espaces X sur lesquels on restreint (H, Q) : `libre`, `kerB`, `landau`, `tcov`, `kerdu` (GATES v4-T2 §1.3).
//!
//! Chaque espace est défini par un PROJECTEUR EXPLICITE Π_X = 1 − Σ_k P_k, où P_k est le projecteur orthogonal sur l'espace
//! engendré par les lignes d'une famille de contraintes (les familles sont deux à deux orthogonales) :
//! - `libre`  = (ker Q0)^⊥ = {div z = 0, z ⊥ constantes} (couleur par couleur ; ker Q0 = formes fermées = jauge ⊕ constantes) ;
//! - `landau` = tranche de Landau linéarisée ∩ (ker Q0)^⊥ : MÊME sous-espace que `libre` (démontré ci-dessus ; le jeu de
//!   coordonnées additives + fixation de Landau du fond fait la différence, pas le sous-espace) ;
//! - `kerB`   = {div z = 0, B z = 0} (B z = 0 implique z ⊥ constantes) ;
//! - `tcov`   = {T z = 0 (D_U* z = 0), B_U z = 0} (`lat.basis_cov`) ;
//! - `kerdu`  = ker DŪ(U) ∩ (jauge résiduelle)^⊥, jauge résiduelle = {τ_U(λ) : λ = 0 aux centres des blocs} ⊂ ker DŪ
//!   (donc Π_X = 1 − P_{lignes de DŪ} − P_{lignes de T aux sites hors centres}).
//!
//! Deux réalisations de P_k : SVD de Jacobi (base orthonormée explicite, seuil de rang 1e-10 σ_max, valeurs écartées
//! comptées : petits réseaux) ou gradient conjugué sur A Aᵀ (sans matrice, tout L ; rang non fourni).
//! LIMITE CONNUE du gradient conjugué : si A Aᵀ a des valeurs propres presque nulles (cas de `tcov` sur un fond proche de l'identité : les rotations
//! globales de couleur sont presque dans le noyau de D_U*, σ_min ≲ 1e-2), la projection n'est précise qu'à ≈ ε·cond(A Aᵀ) et LOBPCG peut ne pas atteindre
//! tol 1e-10 (il rend alors le code 8) ; la SVD (`--projecteur svd`, L ≤ 5) reste exacte. À traiter (déflation) avant WP14 à L = 12.

use super::bases::Reseau;
use super::dense::{base_de_l_image, cholesky, pdscal};
use super::dual::jacobien_dense;
use super::lobpcg::Op;
use super::ops::{self, Csr};
use crate::commun::{ChampHote, Erreur, Resultat};
use serde_json::{json, Value};
use std::str::FromStr;
use std::sync::Mutex;

pub const SEUIL_RANG: f64 = 1e-10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Espace {
    Libre,
    KerB,
    Landau,
    Tcov,
    KerDu,
}

impl Espace {
    pub fn nom(self) -> &'static str {
        match self {
            Espace::Libre => "libre",
            Espace::KerB => "kerB",
            Espace::Landau => "landau",
            Espace::Tcov => "tcov",
            Espace::KerDu => "kerdu",
        }
    }
}

impl FromStr for Espace {
    type Err = Erreur;
    fn from_str(s: &str) -> Resultat<Self> {
        match s {
            "libre" => Ok(Espace::Libre),
            "kerB" | "kerb" => Ok(Espace::KerB),
            "landau" => Ok(Espace::Landau),
            "tcov" => Ok(Espace::Tcov),
            "kerdu" => Ok(Espace::KerDu),
            _ => Err(Erreur::Config(format!("espace « {s} » inconnu (libre | kerB | landau | tcov | kerdu)"))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeProj {
    Svd,
    Cg,
    Auto,
}

impl FromStr for ModeProj {
    type Err = Erreur;
    fn from_str(s: &str) -> Resultat<Self> {
        match s {
            "svd" => Ok(ModeProj::Svd),
            "cg" => Ok(ModeProj::Cg),
            "auto" => Ok(ModeProj::Auto),
            _ => Err(Erreur::Config(format!("mode de projecteur « {s} » inconnu (svd | cg | auto)"))),
        }
    }
}

/// Projecteur orthogonal sur l'espace des lignes d'une famille de contraintes.
pub trait ImageLignes: Sync {
    fn appliquer(&self, pool: &rayon::ThreadPool, x: &[f64], out: &mut [f64]);
    fn rang(&self) -> Option<usize>;
    fn diagnostic(&self) -> Value;
}

/// Réalisation par base orthonormée explicite (SVD de Jacobi).
pub struct ImageSvd {
    base: Vec<Vec<f64>>,
    ecartes: usize,
    sigma_max: f64,
    sigma_min_gardee: f64,
    sigma_max_ecartee: f64,
}

impl ImageSvd {
    pub fn depuis_lignes(pool: &rayon::ThreadPool, a: &Csr) -> Resultat<ImageSvd> {
        let cols: Vec<Vec<f64>> = (0..a.nlig).map(|i| a.ligne_dense(i)).collect();
        Self::depuis_vecteurs(pool, cols)
    }

    pub fn depuis_vecteurs(pool: &rayon::ThreadPool, cols: Vec<Vec<f64>>) -> Resultat<ImageSvd> {
        let (base, sigmas, ecartes) = base_de_l_image(Some(pool), cols, SEUIL_RANG).map_err(Erreur::Config)?;
        let sigma_max = sigmas.first().copied().unwrap_or(0.0);
        let r = base.len();
        let sigma_min_gardee = if r > 0 { sigmas[r - 1] } else { 0.0 };
        let sigma_max_ecartee = if ecartes > 0 { sigmas[r] } else { 0.0 };
        Ok(ImageSvd { base, ecartes, sigma_max, sigma_min_gardee, sigma_max_ecartee })
    }
}

impl ImageLignes for ImageSvd {
    fn appliquer(&self, pool: &rayon::ThreadPool, x: &[f64], out: &mut [f64]) {
        use rayon::prelude::*;
        // coefficients c_k = u_k · x (chaque produit est déterministe), puis out = Σ_k c_k u_k, k croissant, par tronçons d'éléments
        let c: Vec<f64> = pool.install(|| self.base.par_iter().map(|u| pdscal(None, u, x)).collect());
        pool.install(|| {
            out.par_chunks_mut(2048).enumerate().for_each(|(ch, t)| {
                let d = ch * 2048;
                for (i, o) in t.iter_mut().enumerate() {
                    let mut s = 0.0;
                    for (k, u) in self.base.iter().enumerate() {
                        s += c[k] * u[d + i];
                    }
                    *o = s;
                }
            })
        });
    }
    fn rang(&self) -> Option<usize> {
        Some(self.base.len())
    }
    fn diagnostic(&self) -> Value {
        json!({"mode": "svd", "rang": self.base.len(), "sv_ecartees": self.ecartes, "seuil_rel": SEUIL_RANG,
               "sigma_max": self.sigma_max, "sigma_min_gardee": self.sigma_min_gardee, "sigma_max_ecartee": self.sigma_max_ecartee})
    }
}

/// Réalisation sans matrice : gradient conjugué sur A Aᵀ y = A x, image = Aᵀ y.
pub struct ImageCg {
    a: Csr,
    at: Csr,
    tol: f64,
    iter_max: usize,
    stats: Mutex<(usize, usize, f64)>, // (appels, itérations max, résidu relatif max)
    rang_connu: Option<usize>,
    /// plus grande somme des |entrées| d'une ligne de A (échelle de ‖A‖ pour le plancher d'arrondi)
    echelle: f64,
}

impl ImageCg {
    pub fn nouveau(a: Csr) -> ImageCg {
        let at = a.transposee();
        let echelle = (0..a.nlig).map(|i| a.val[a.ptr[i]..a.ptr[i + 1]].iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max);
        ImageCg { a, at, tol: 1e-13, iter_max: 5_000, stats: Mutex::new((0, 0, 0.0)), rang_connu: None, echelle }
    }

    /// Rang établi par ailleurs (Cholesky du Gram).
    pub fn avec_rang(mut self, r: usize) -> ImageCg {
        self.rang_connu = Some(r);
        self
    }
}

impl ImageLignes for ImageCg {
    fn appliquer(&self, _pool: &rayon::ThreadPool, x: &[f64], out: &mut [f64]) {
        let m = self.a.nlig;
        let n = self.a.ncol;
        let mut b = vec![0.0; m];
        self.a.mul(x, &mut b);
        let bn = pdscal(None, &b, &b).sqrt();
        // plancher d'arrondi : si A x est au niveau du bruit de A x (1e-14 ‖A‖ ‖x‖), x est déjà dans X : on ne résout pas un système bruité
        // (le Gram a des valeurs propres ~1e-25 : résoudre le bruit l'amplifierait de 1e12).
        let plancher = 1e-14 * self.echelle * pdscal(None, x, x).sqrt();
        let cible = (self.tol * bn).max(plancher);
        let mut y = vec![0.0; m];
        let mut r = b.clone();
        let mut p = b;
        let mut rr = pdscal(None, &r, &r);
        let mut tmp = vec![0.0; n];
        let mut ap = vec![0.0; m];
        let mut it = 0;
        let (mut meilleur, mut it_meilleur) = (rr, 0usize);
        if bn > plancher {
            while it < self.iter_max && rr.sqrt() > cible {
                // stagnation au niveau de l'arrondi : 100 itérations sans amélioration d'un facteur 2
                if rr < 0.5 * meilleur {
                    meilleur = rr;
                    it_meilleur = it;
                } else if it - it_meilleur > 100 {
                    break;
                }
                self.at.mul(&p, &mut tmp);
                self.a.mul(&tmp, &mut ap);
                let pap = pdscal(None, &p, &ap);
                if pap <= 0.0 {
                    break;
                }
                let al = rr / pap;
                for i in 0..m {
                    y[i] += al * p[i];
                    r[i] -= al * ap[i];
                }
                let rr2 = pdscal(None, &r, &r);
                let be = rr2 / rr;
                for i in 0..m {
                    p[i] = r[i] + be * p[i];
                }
                rr = rr2;
                it += 1;
            }
        }
        self.at.mul(&y, out);
        let mut s = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        s.0 += 1;
        s.1 = s.1.max(it);
        if bn > plancher {
            s.2 = s.2.max(rr.sqrt() / bn);
        }
    }
    fn rang(&self) -> Option<usize> {
        self.rang_connu
    }

    fn diagnostic(&self) -> Value {
        let s = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        json!({"mode": "cg", "lignes": self.a.nlig, "appels": s.0, "iterations_max": s.1, "residu_rel_max": s.2, "tol": self.tol})
    }
}

/// Π_X = 1 − Σ_k P_k.
pub struct Projecteur {
    n: usize,
    parts: Vec<Box<dyn ImageLignes>>,
    pool: std::sync::Arc<rayon::ThreadPool>,
}

impl Projecteur {
    pub fn dim(&self) -> Option<usize> {
        let mut r = 0usize;
        for p in &self.parts {
            r += p.rang()?;
        }
        Some(self.n - r)
    }
    pub fn diagnostic(&self) -> Value {
        Value::Array(self.parts.iter().map(|p| p.diagnostic()).collect())
    }
    pub fn n(&self) -> usize {
        self.n
    }
    /// Image d'un seul bloc (diagnostic d'orthogonalité).
    pub fn partie(&self, k: usize, x: &[f64], out: &mut [f64]) {
        self.parts[k].appliquer(&self.pool, x, out);
    }
    pub fn nb_parties(&self) -> usize {
        self.parts.len()
    }
}

impl Op for Projecteur {
    fn n(&self) -> usize {
        self.n
    }
    fn appliquer(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
        let mut t = vec![0.0; self.n];
        for p in &self.parts {
            p.appliquer(&self.pool, x, &mut t);
            for i in 0..self.n {
                y[i] -= t[i];
            }
        }
    }
}

/// Espace construit : projecteur, dimension (si connue), diagnostics consignés.
pub struct EspaceConstruit {
    pub proj: Projecteur,
    pub dim: Option<usize>,
    pub info: Value,
}

fn partie(pool: &rayon::ThreadPool, lignes: &[Vec<(usize, f64)>], ncol: usize, mode: ModeProj) -> Resultat<Box<dyn ImageLignes>> {
    let a = Csr::depuis_lignes(ncol, lignes);
    let svd = match mode {
        ModeProj::Svd => true,
        ModeProj::Cg => false,
        // SVD de Jacobi : coût ~ m² n par balayage ; au-delà de ~1500 lignes ou 2e9 flops, CG
        ModeProj::Auto => a.nlig <= 1500 && (a.nlig * a.nlig).saturating_mul(a.ncol) <= 3_000_000_000,
    };
    if svd {
        Ok(Box::new(ImageSvd::depuis_lignes(pool, &a)?))
    } else {
        Ok(Box::new(ImageCg::nouveau(a)))
    }
}

/// Construit le projecteur de l'espace (disposition s2, n = 12 L⁴).
pub fn construire(
    espace: Espace,
    champ: &ChampHote,
    pool: std::sync::Arc<rayon::ThreadPool>,
    mode: ModeProj,
) -> Resultat<EspaceConstruit> {
    let r = Reseau::nouveau(champ.l)?;
    let n = r.n;
    let u = ops::liens_s2(champ, &r);
    let mut parts: Vec<Box<dyn ImageLignes>> = Vec::new();
    let mut info = json!({"espace": espace.nom()});
    match espace {
        Espace::Libre | Espace::Landau => {
            let mut l = ops::contraintes_div(&r);
            l.extend(ops::contraintes_constantes(&r));
            parts.push(partie(&pool, &l, n, mode)?);
        }
        Espace::KerB => {
            let mut l = ops::contraintes_div(&r);
            l.extend(ops::contraintes_b_lineaire(&r)?);
            parts.push(partie(&pool, &l, n, mode)?);
        }
        Espace::Tcov => {
            let mut l = ops::contraintes_tcov(&r, &u, None);
            l.extend(ops::contraintes_bcov(&r, &u)?);
            parts.push(partie(&pool, &l, n, mode)?);
        }
        Espace::KerDu => {
            if !champ.l.is_multiple_of(3) {
                return Err(Erreur::Config(format!("L = {} : multiple de 3 requis pour kerdu", champ.l)));
            }
            let j = jacobien_dense(&pool, champ)?;
            let octets = j.len() * n * 8;
            info["jacobien"] = json!({"lignes": j.len(), "octets": octets});
            parts.push(Box::new(ImageSvd::depuis_vecteurs(&pool, j)?));
            // jauge résiduelle : λ = 0 aux centres 3z + (1,1,1,1)
            let masque: Vec<bool> = (0..r.v).map(|x| Reseau::coordonnees_de(x, champ.l).iter().any(|&c| c % 3 != 1)).collect();
            let hors_centres = masque.iter().filter(|&&b| b).count();
            let lignes = ops::contraintes_tcov(&r, &u, Some(&masque));
            let a = Csr::depuis_lignes(n, &lignes);
            info["jauge_residuelle"] = json!({"sites_hors_centres": hors_centres, "lignes": a.nlig, "dim_attendue": 3 * hors_centres});
            // rang de la jauge résiduelle : Cholesky du Gram (3 hors_centres)² si raisonnable
            let mut rang_g = None;
            if a.nlig <= 4500 {
                let at = a.transposee();
                let mut g = a.gram_dense(&at);
                let m = a.nlig;
                let pivots = match cholesky(&mut g, m) {
                    Ok(()) => {
                        let mn = (0..m).map(|i| g[i * m + i]).fold(f64::INFINITY, f64::min);
                        {
                        rang_g = Some(m);
                        json!({"cholesky": "ok", "pivot_min": mn, "rang": m})
                    }
                    }
                    Err(i) => json!({"cholesky": "echec", "ligne": i}),
                };
                info["jauge_residuelle"]["gram"] = pivots;
            }
            let cg = ImageCg::nouveau(a);
            parts.push(Box::new(match rang_g {
                Some(r) => cg.avec_rang(r),
                None => cg,
            }));
        }
    }
    let proj = Projecteur { n, parts, pool };
    let dim = proj.dim();
    info["dim"] = json!(dim);
    info["parties"] = proj.diagnostic();
    Ok(EspaceConstruit { proj, dim, info })
}
