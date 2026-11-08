//! Hessienne de S (avec β) en base ym4 ou s2, coordonnées gauche ou additives, et λ_min généralisé sur un espace.

use super::additif::{croix, d_et_dd, mat_vec, matt_vec};
use super::bases::{Base, Coord, Reseau};
use super::dense::{eigh_generalise, norme2, pdscal};
use super::espaces::{construire, Espace, EspaceConstruit, ModeProj};
use super::hv::{MutantHv, OperateurGauche};
use super::lobpcg::{lobpcg, Cote, Op, Params};
use super::ops::{self, QCov};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::rg::bloc::log_q;
use serde_json::{json, Value};
use std::sync::Arc;

/// Mutants compilés (tests des gates) : M-b (terme de dérivée seconde d'un même lien retiré), facteur 2 de la base s2 omis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutantHess {
    Aucun,
    MemeLien,
    Facteur2Omis,
}

impl std::str::FromStr for MutantHess {
    type Err = Erreur;
    fn from_str(s: &str) -> Resultat<Self> {
        match s {
            "aucun" => Ok(MutantHess::Aucun),
            "M-b" | "mb" => Ok(MutantHess::MemeLien),
            "facteur2" => Ok(MutantHess::Facteur2Omis),
            _ => Err(Erreur::Config(format!("mutant « {s} » inconnu (aucun | M-b | facteur2)"))),
        }
    }
}

struct DonneesAdd {
    /// A_l = 2 log(U_l), disposition s2 (3 par lien)
    a: Vec<f64>,
    /// gradient en base s2, β inclus (3 par lien, disposition s2)
    g: Vec<f64>,
}

/// Opérateur H·v de S (β inclus) dans la base et les coordonnées demandées.
pub struct Hessienne {
    pub reseau: Reseau,
    pub base: Base,
    pub coord: Coord,
    pub beta: f64,
    pub mutant: MutantHess,
    op: OperateurGauche,
    pool: Arc<rayon::ThreadPool>,
    add: Option<DonneesAdd>,
}

impl Hessienne {
    pub fn nouveau(champ: &ChampHote, pool: Arc<rayon::ThreadPool>, base: Base, coord: Coord, beta: f64, mutant: MutantHess) -> Resultat<Self> {
        let reseau = Reseau::nouveau(champ.l)?;
        if champ.q.len() != 16 * reseau.v {
            return Err(Erreur::Config("longueur du champ invalide".into()));
        }
        let op = OperateurGauche::nouveau(&pool, &champ.q, champ.l).map_err(Erreur::Config)?;
        let add = if coord == Coord::Additive {
            if base != Base::S2 {
                return Err(Erreur::Config("les coordonnées additives sont définies en base s2 (convention de lat.py)".into()));
            }
            let mut a = vec![0.0; reseau.n];
            for l in 0..reseau.nl {
                let (w, _, _) = log_q(champ.lien(reseau.champ_de_s2(l)));
                for c in 0..3 {
                    a[3 * l + c] = 2.0 * w[c];
                }
            }
            let mut gy = vec![0.0; reseau.n];
            op.gradient(&pool, &mut gy);
            let mut g = vec![0.0; reseau.n];
            reseau.ym4_vers_s2(&gy, &mut g);
            for x in &mut g {
                *x *= beta * std::f64::consts::FRAC_1_SQRT_2;
            }
            Some(DonneesAdd { a, g })
        } else {
            None
        };
        Ok(Hessienne { reseau, base, coord, beta, mutant, op, pool, add })
    }

    pub fn n(&self) -> usize {
        self.reseau.n
    }

    /// H_gauche dans la base `base` : v et y dans la disposition de la base.
    fn hv_gauche(&self, v: &[f64], y: &mut [f64]) {
        let n = self.reseau.n;
        let mut f = self.base.facteur_hessienne() * self.beta;
        if self.mutant == MutantHess::Facteur2Omis && self.base == Base::S2 {
            f = self.beta;
        }
        let mutant = if self.mutant == MutantHess::MemeLien { MutantHv::MemeLienRetire } else { MutantHv::Aucun };
        match self.base {
            Base::Ym4 => {
                self.op.hv(&self.pool, v, y, mutant);
            }
            Base::S2 => {
                let mut vy = vec![0.0; n];
                self.reseau.s2_vers_ym4(v, &mut vy);
                let mut hy = vec![0.0; n];
                self.op.hv(&self.pool, &vy, &mut hy, mutant);
                self.reseau.ym4_vers_s2(&hy, y);
            }
        }
        // f = facteur de la base × β (la sortie de hv est en base ym4, β = 1)
        for t in y.iter_mut() {
            *t *= f;
        }
    }

    /// y = H v (y écrasé).
    pub fn hv(&self, v: &[f64], y: &mut [f64]) {
        match self.coord {
            Coord::Gauche => self.hv_gauche(v, y),
            Coord::Additive => {
                let add = self.add.as_ref().expect("données additives");
                let nl = self.reseau.nl;
                let mut u = vec![0.0; self.reseau.n];
                let mut dds = vec![[[0.0; 3]; 3]; nl];
                let mut ds = vec![[[0.0; 3]; 3]; nl];
                for l in 0..nl {
                    let a = [add.a[3 * l], add.a[3 * l + 1], add.a[3 * l + 2]];
                    let vl = [v[3 * l], v[3 * l + 1], v[3 * l + 2]];
                    let (d, dd) = d_et_dd(a, vl);
                    ds[l] = d;
                    dds[l] = dd;
                    let ul = mat_vec(&d, vl);
                    u[3 * l..3 * l + 3].copy_from_slice(&ul);
                }
                let mut w = vec![0.0; self.reseau.n];
                self.hv_gauche(&u, &mut w);
                for l in 0..nl {
                    let ul = [u[3 * l], u[3 * l + 1], u[3 * l + 2]];
                    let gl = [add.g[3 * l], add.g[3 * l + 1], add.g[3 * l + 2]];
                    let c = croix(ul, gl);
                    let wl: [f64; 3] = std::array::from_fn(|i| w[3 * l + i] - 0.5 * c[i]);
                    let r1 = matt_vec(&ds[l], wl);
                    let r2 = matt_vec(&dds[l], gl);
                    for i in 0..3 {
                        y[3 * l + i] = r1[i] + r2[i];
                    }
                }
            }
        }
    }
}

impl Op for Hessienne {
    fn n(&self) -> usize {
        self.reseau.n
    }
    fn appliquer(&self, x: &[f64], y: &mut [f64]) {
        self.hv(x, y)
    }
}

/// Référence Q (β incluse).
pub enum Reference {
    Q0,
    QCov,
}

impl std::str::FromStr for Reference {
    type Err = Erreur;
    fn from_str(s: &str) -> Resultat<Self> {
        match s {
            "q0" => Ok(Reference::Q0),
            "qcov" => Ok(Reference::QCov),
            _ => Err(Erreur::Config(format!("référence « {s} » inconnue (q0 | qcov)"))),
        }
    }
}

impl Reference {
    pub fn nom(&self) -> &'static str {
        match self {
            Reference::Q0 => "q0",
            Reference::QCov => "qcov",
        }
    }
}

pub struct OpRef {
    n: usize,
    beta: f64,
    plaq: Vec<[usize; 4]>,
    qcov: Option<QCov>,
}

impl OpRef {
    pub fn nouveau(champ: &ChampHote, reference: &Reference, beta: f64) -> Resultat<OpRef> {
        let r = Reseau::nouveau(champ.l)?;
        let qcov = match reference {
            Reference::Q0 => None,
            Reference::QCov => Some(QCov::nouveau(&r, &ops::liens_s2(champ, &r))),
        };
        Ok(OpRef { n: r.n, beta, plaq: r.plaquettes_s2(), qcov })
    }
}

impl Op for OpRef {
    fn n(&self) -> usize {
        self.n
    }
    fn appliquer(&self, x: &[f64], y: &mut [f64]) {
        match &self.qcov {
            None => ops::q0_appliquer(&self.plaq, x, y),
            Some(q) => q.appliquer(x, y),
        }
        for t in y.iter_mut() {
            *t *= self.beta;
        }
    }
}

struct Identite(usize);
impl Op for Identite {
    fn n(&self) -> usize {
        self.0
    }
    fn appliquer(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }
}

struct Compose<'a> {
    a: &'a dyn Op,
    proj: &'a dyn Op,
}
impl Op for Compose<'_> {
    fn n(&self) -> usize {
        self.a.n()
    }
    fn appliquer(&self, x: &[f64], y: &mut [f64]) {
        let mut t = vec![0.0; x.len()];
        self.a.appliquer(x, &mut t);
        self.proj.appliquer(&t, y);
    }
}

fn splitmix(etat: &mut u64) -> u64 {
    *etat = etat.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *etat;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Estimation de ‖Π A Π‖₂ (itération de la puissance, déterministe) : borne inférieure, 80 itérations.
pub fn norme_restreinte(a: &dyn Op, proj: &dyn Op, graine: u64) -> f64 {
    let n = a.n();
    let mut st = graine ^ 0xA5A5_5A5A_1234_4321;
    let x0: Vec<f64> = (0..n).map(|_| ((splitmix(&mut st) >> 11) as f64 / 9_007_199_254_740_992.0) * 2.0 - 1.0).collect();
    let mut x = vec![0.0; n];
    proj.appliquer(&x0, &mut x);
    let mut est = 0.0;
    let c = Compose { a, proj };
    for _ in 0..80 {
        let nx = norme2(None, &x);
        if nx == 0.0 {
            return 0.0;
        }
        for t in x.iter_mut() {
            *t /= nx;
        }
        let mut y = vec![0.0; n];
        c.appliquer(&x, &mut y);
        est = norme2(None, &y);
        x = y;
    }
    est
}

pub struct OptionsLmin {
    pub espace: Espace,
    pub reference: Reference,
    pub coord: Coord,
    pub beta: f64,
    pub graine: u64,
    pub tol: f64,
    pub bloc: usize,
    pub iter_max: usize,
    pub mode_proj: ModeProj,
    pub dense: bool,
    pub mutant: MutantHess,
    pub verifier_q: bool,
}

pub struct SortieLmin {
    pub lambda_min: f64,
    pub eps_prime: f64,
    pub vecteur: Vec<f64>,
    pub json: Value,
}

pub fn lmin(champ: &ChampHote, pool: Arc<rayon::ThreadPool>, o: &OptionsLmin) -> Resultat<SortieLmin> {
    let h = Hessienne::nouveau(champ, pool.clone(), Base::S2, o.coord, o.beta, o.mutant)?;
    let q = OpRef::nouveau(champ, &o.reference, o.beta)?;
    let EspaceConstruit { proj, dim, info } = construire(o.espace, champ, pool.clone(), o.mode_proj)?;
    let n = h.n();
    let echelle = norme_restreinte(&h, &proj, o.graine);
    let mut sortie_json = json!({
        "espace": o.espace.nom(), "reference": o.reference.nom(), "coord": o.coord.nom(), "beta": o.beta,
        "L": champ.l, "dim": dim, "info_espace": info, "norme_H_restreinte": echelle,
    });
    // la référence restreinte doit être définie positive : λ_min(PᵀQP) ≥ 1e-8 λ_max(PᵀQP).
    // λ_min : LOBPCG (B = identité, tol 1e-8, convergence exigée) ; majorant de λ_min vrai (valeur de Ritz). λ_max : borne RIGOUREUSE 6β :
    // Q/β = ¼ Σ_p J_pᵀ J_p avec J_p J_pᵀ = 4·1 (rotations orthogonales) ⇒ zᵀ(Q/β)z ≤ Σ_p |z_p|² = 6 |z|² (chaque lien est dans 6 plaquettes) ;
    // λ_max(Q|X) ≤ λ_max(Q) ≤ 6β. Un majorant de λ_max rend le contrôle λ_min/λ_max ≥ 1e-8 plus strict (sûr).
    if o.verifier_q {
        let id = Identite(n);
        let ech_q = norme_restreinte(&q, &proj, o.graine ^ 1);
        let params = Params { bloc: 3, tol: 1e-8, iter_max: 20_000, graine: o.graine ^ 2, cote: Cote::Min, echelle: ech_q.max(1e-300) };
        let pmin = lobpcg(&pool, &q, &id, &proj, None, &params).map_err(Erreur::Config)?;
        let qmin = pmin.valeurs[0];
        let qmax_borne = 6.0 * o.beta.abs();
        sortie_json["q_restreinte"] = json!({
            "lambda_min": qmin, "lambda_max_borne_6beta": qmax_borne, "rapport_borne": qmin / qmax_borne, "norme_puissance": ech_q,
            "residu_rel_min": pmin.residu_rel, "converge_min": pmin.converge,
        });
        if !pmin.converge {
            return Err(Erreur::Validation("positivité de la référence restreinte : LOBPCG (λ_min) non convergé".into()));
        }
        if qmin < 1e-8 * qmax_borne {
            return Err(Erreur::Validation(format!("référence restreinte non définie positive : λ_min/(6β) = {:.3e} < 1e-8", qmin / qmax_borne)));
        }
    }
    let (lam, vecteur, extra) = if o.dense {
        let sigma = 10.0 * echelle.max(1.0);
        let mut a = vec![0.0; n * n];
        let mut b = vec![0.0; n * n];
        let (mut ei, mut hi, mut qi, mut pi_) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut ph = vec![0.0; n];
        for j in 0..n {
            ei.fill(0.0);
            ei[j] = 1.0;
            proj.appliquer(&ei, &mut pi_); // Π e_j
            h.appliquer(&pi_, &mut hi);
            proj.appliquer(&hi, &mut ph); // Π H Π e_j
            q.appliquer(&pi_, &mut qi);
            proj.appliquer(&qi, &mut hi); // Π Q Π e_j (hi réutilisé)
            for i in 0..n {
                let comp = if i == j { 1.0 } else { 0.0 } - pi_[i]; // (1 − Π) e_j composante i
                a[i * n + j] = ph[i] + sigma * comp;
                b[i * n + j] = hi[i] + comp;
            }
        }
        // symétrisation (les erreurs d'arrondi des projections)
        for i in 0..n {
            for j in i + 1..n {
                let (x, y) = (0.5 * (a[i * n + j] + a[j * n + i]), 0.5 * (b[i * n + j] + b[j * n + i]));
                a[i * n + j] = x;
                a[j * n + i] = x;
                b[i * n + j] = y;
                b[j * n + i] = y;
            }
        }
        let (vals, vecs) = eigh_generalise(a, b, n, true).map_err(Erreur::Config)?;
        let vecs = vecs.ok_or_else(|| Erreur::Config("vecteurs propres absents".into()))?;
        let v0: Vec<f64> = (0..n).map(|i| vecs[i * n]).collect();
        let mut pv = vec![0.0; n];
        proj.appliquer(&v0, &mut pv);
        (vals[0], pv, json!({"methode": "dense", "valeurs_basses": vals.iter().take(4).collect::<Vec<_>>()}))
    } else {
        let s = lobpcg(
            &pool,
            &h,
            &q,
            &proj,
            None,
            &Params { bloc: o.bloc, tol: o.tol, iter_max: o.iter_max, graine: o.graine, cote: Cote::Min, echelle: echelle.max(1e-300) },
        )
        .map_err(Erreur::Config)?;
        let ex = json!({"methode": "lobpcg", "bloc": o.bloc, "tol": o.tol, "iterations": s.iterations, "converge": s.converge,
                        "valeurs_bloc": s.valeurs, "residu_abs": s.residu_abs, "residu_rel": s.residu_rel});
        (s.valeurs[0], s.vecteur, ex)
    };
    // résidu projeté du vecteur rendu (B-normalisé), indépendant de la méthode
    let mut hv_ = vec![0.0; n];
    let mut qv = vec![0.0; n];
    h.appliquer(&vecteur, &mut hv_);
    q.appliquer(&vecteur, &mut qv);
    let vqv = pdscal(None, &vecteur, &qv);
    let mut r = vec![0.0; n];
    for i in 0..n {
        r[i] = hv_[i] - lam * qv[i];
    }
    let mut pr = vec![0.0; n];
    proj.appliquer(&r, &mut pr);
    let nv = norme2(None, &vecteur);
    sortie_json["lambda_min"] = json!(lam);
    sortie_json["eps_prime"] = json!(1.0 - lam);
    sortie_json["vqv"] = json!(vqv);
    sortie_json["residu_projete_rel"] = json!(norme2(None, &pr) / (echelle.max(1e-300) * nv));
    sortie_json["solveur"] = extra;
    sortie_json["projecteur_apres"] = proj.diagnostic();
    Ok(SortieLmin { lambda_min: lam, eps_prime: 1.0 - lam, vecteur, json: sortie_json })
}
