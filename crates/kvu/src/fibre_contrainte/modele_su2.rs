//! Fibre de Bałaban sur SU(2)^N, dans le repère spatial des perturbations gauches.

use super::action::{action_wilson, grad_action};
use super::chol::CholBlocs;
use super::jacobien::{Jac, Reglages, Reseau, Q};
use super::rattle::{projeter_sur_fibre, Variete};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul};
use crate::rg::bloc::{exp_q, log_q, ubar_lien};
use rayon::prelude::*;
use std::sync::Arc;

pub struct FibreSu2 {
    reseau: Arc<Reseau>,
    v: Vec<Q>,
    beta: f64,
    pool: Arc<rayon::ThreadPool>,
    reg: Reglages,
    minv: Vec<f64>,
    #[cfg(test)]
    pub sans_coaire: bool,
    #[cfg(test)]
    pub terme_oppose: bool,
}

pub struct Eval {
    pub jac: Jac,
    pub chol: CholBlocs,
    pub potentiel: f64,
    pub grad: Vec<f64>,
}

impl FibreSu2 {
    pub fn nouveau(reseau: Arc<Reseau>, v: Vec<Q>, beta: f64, fils: usize) -> Resultat<Self> {
        if v.len() != reseau.n_c || !beta.is_finite() || fils == 0 {
            return Err(Erreur::Config(
                "fibre SU(2) : cible, β ou nombre de fils invalide".into(),
            ));
        }
        for q in &v {
            let norme = q.iter().map(|x| x * x).sum::<f64>();
            if !norme.is_finite() || (norme - 1.0).abs() > 1e-10 {
                return Err(Erreur::Config("cible SU(2) non unitaire".into()));
            }
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(fils)
            .build()
            .map_err(|e| Erreur::Config(format!("pool de la fibre : {e}")))?;
        let minv = vec![1.0; 3 * reseau.n_liens];
        Ok(Self {
            reseau,
            v,
            beta,
            pool: Arc::new(pool),
            reg: Reglages::default(),
            minv,
            #[cfg(test)]
            sans_coaire: false,
            #[cfg(test)]
            terme_oppose: false,
        })
    }

    fn verifier_champ(&self, champ: &ChampHote) -> Resultat<()> {
        if champ.l != self.reseau.l
            || champ.q.len() != 4 * self.reseau.n_liens
            || !champ.q.iter().all(|x| x.is_finite())
        {
            return Err(Erreur::Validation(
                "champ incompatible avec la fibre SU(2)".into(),
            ));
        }
        Ok(())
    }

    pub fn residu_max(&self, champ: &ChampHote) -> Resultat<f64> {
        Ok(self
            .residu(champ)?
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max))
    }
}

impl Variete for FibreSu2 {
    type Etat = ChampHote;
    type Eval = Eval;
    fn dim(&self) -> usize {
        3 * self.reseau.n_liens
    }
    fn nb_contraintes(&self) -> usize {
        3 * self.reseau.n_c
    }
    fn minv(&self) -> &[f64] {
        &self.minv
    }
    fn evaluer(&self, q: &Self::Etat) -> Resultat<Self::Eval> {
        self.verifier_champ(q)?;
        let jac = Jac::evaluer(&self.reseau, q, &self.pool, &self.reg)?;
        let chol = CholBlocs::factoriser(&jac.jjt_blocs())?;
        let ginv = chol.inverse_dense(&self.pool);
        let w = jac.w_depuis_ginv(&ginv, &self.pool);
        let gl = jac.grad_demi_logdet(&self.reseau, q, &w, &self.pool, &self.reg);
        let mut grad = grad_action(q, self.beta, &self.pool);
        #[cfg(test)]
        let coefficient = if self.sans_coaire {
            0.0
        } else if self.terme_oppose {
            -1.0
        } else {
            1.0
        };
        #[cfg(not(test))]
        let coefficient = 1.0;
        let potentiel = action_wilson(q, self.beta) + coefficient * 0.5 * chol.log_det();
        for (g, l) in grad.iter_mut().zip(gl) {
            *g += coefficient * l;
        }
        if !potentiel.is_finite() || !grad.iter().all(|x| x.is_finite()) {
            return Err(Erreur::Validation(
                "potentiel ou gradient SU(2) non fini".into(),
            ));
        }
        Ok(Eval {
            jac,
            chol,
            potentiel,
            grad,
        })
    }
    fn potentiel(&self, e: &Self::Eval) -> f64 {
        e.potentiel
    }
    fn grad_potentiel<'a>(&self, e: &'a Self::Eval) -> &'a [f64] {
        &e.grad
    }
    fn jt_mul(&self, e: &Self::Eval, lambda: &[f64]) -> Vec<f64> {
        e.jac.vjp(lambda)
    }
    fn j_mul(&self, e: &Self::Eval, v: &[f64]) -> Vec<f64> {
        e.jac.jvp(v)
    }
    fn g_resoudre(&self, e: &Self::Eval, b: &[f64]) -> Vec<f64> {
        let mut x = b.to_vec();
        e.chol.resoudre(&mut x);
        x
    }
    fn residu(&self, q: &Self::Etat) -> Resultat<Vec<f64>> {
        self.verifier_champ(q)?;
        let lignes: Resultat<Vec<[f64; 3]>> = self.pool.install(|| {
            (0..self.reseau.n_c)
                .into_par_iter()
                .map(|c| {
                    let (z, mu) = self.reseau.coords_c(c);
                    let (ubar, stats) = ubar_lien(q, &self.reseau.table, z, mu);
                    let (g, _, ambigu) = log_q(qmul(ubar, qdag(self.v[c])));
                    if ambigu || stats.n_ambigues > 0 || !g.iter().all(|x| x.is_finite()) {
                        Err(Erreur::Validation(format!(
                            "résidu SU(2) ambigu ou non fini au lien {c}"
                        )))
                    } else {
                        Ok(g)
                    }
                })
                .collect()
        });
        Ok(lignes?.into_iter().flatten().collect())
    }
    fn avancer(&self, q: &Self::Etat, xi: &[f64], dt: f64) -> Self::Etat {
        let mut suivant = q.clone();
        self.pool.install(|| {
            suivant.q.par_chunks_mut(4).enumerate().for_each(|(l, u)| {
                let e = exp_q([dt * xi[3 * l], dt * xi[3 * l + 1], dt * xi[3 * l + 2]]);
                let produit = qmul(e, [u[0], u[1], u[2], u[3]]);
                let norme = produit.iter().map(|x| x * x).sum::<f64>().sqrt();
                for (a, b) in u.iter_mut().zip(produit) {
                    *a = b / norme;
                }
            });
        });
        suivant
    }
    fn ecart(&self, a: &Self::Etat, b: &Self::Etat) -> f64 {
        a.q.iter()
            .zip(&b.q)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }
}

/// Cible dont le champ fin est exactement un point de la fibre, ordre c.
pub fn cible_depuis(champ_fin: &ChampHote, reseau: &Reseau) -> Vec<Q> {
    (0..reseau.n_c)
        .map(|c| {
            let (z, mu) = reseau.coords_c(c);
            ubar_lien(champ_fin, &reseau.table, z, mu).0
        })
        .collect()
}

/// Transport de cible par petites étapes, avec projection amortie à chaque étape.
pub fn projeter_continuation(
    reseau: Arc<Reseau>,
    champ_depart: &ChampHote,
    v_cible: &[Q],
    beta: f64,
    fils: usize,
    n_etapes: usize,
) -> Resultat<ChampHote> {
    if n_etapes == 0 {
        return Err(Erreur::Config(
            "la continuation requiert au moins une étape".into(),
        ));
    }
    let mut modele = FibreSu2::nouveau(reseau, v_cible.to_vec(), beta, fils)?;
    modele.verifier_champ(champ_depart)?;
    let origine = cible_depuis(champ_depart, &modele.reseau);
    let directions: Resultat<Vec<_>> = origine
        .iter()
        .zip(v_cible)
        .map(|(u, v)| {
            let (w, _, ambigu) = log_q(qmul(qdag(*u), *v));
            if ambigu {
                Err(Erreur::Validation(
                    "cible de continuation sur la coupure du log".into(),
                ))
            } else {
                Ok(w)
            }
        })
        .collect();
    let directions = directions?;
    let mut courant = champ_depart.clone();
    for s in 1..=n_etapes {
        if s == n_etapes {
            modele.v.copy_from_slice(v_cible);
        } else {
            let fraction = s as f64 / n_etapes as f64;
            for c in 0..modele.v.len() {
                modele.v[c] = qmul(origine[c], exp_q(directions[c].map(|x| fraction * x)));
            }
        }
        courant = projeter_sur_fibre(&modele, &courant, 1e-13, 50)?;
    }
    Ok(courant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fibre_contrainte::alea::{Flux, TAG_DEPART, TAG_OUTIL};
    use crate::fibre_contrainte::jacobien::tests::champ_amp;
    use crate::fibre_contrainte::rattle::{integrer, pas_rattle, tirer_impulsion, ParamsRattle};
    use std::sync::OnceLock;

    fn donnees() -> (Arc<Reseau>, ChampHote) {
        static RESEAU: OnceLock<Arc<Reseau>> = OnceLock::new();
        (
            RESEAU
                .get_or_init(|| Arc::new(Reseau::nouveau(6).unwrap()))
                .clone(),
            champ_amp(6, 0.8, 5),
        )
    }

    fn direction(n: usize, etiquette: u32) -> Vec<f64> {
        let flux = Flux::nouveau(513);
        (0..n)
            .map(|i| flux.gauss(0, etiquette, i as u32, TAG_OUTIL))
            .collect()
    }

    /// Évaluation indépendante de V, avec reconstruction de J et G à chaque point.
    fn potentiel_complet(m: &FibreSu2, q: &ChampHote) -> f64 {
        let jac = Jac::evaluer(&m.reseau, q, &m.pool, &m.reg).unwrap();
        let chol = CholBlocs::factoriser(&jac.jjt_blocs()).unwrap();
        action_wilson(q, m.beta) + 0.5 * chol.log_det()
    }

    #[test]
    fn gradient_potentiel_contre_richardson() {
        let (reseau, q) = donnees();
        let m = FibreSu2::nouveau(reseau.clone(), cible_depuis(&q, &reseau), 2.0, 8).unwrap();
        let ev = m.evaluer(&q).unwrap();
        assert_eq!(ev.potentiel.to_bits(), potentiel_complet(&m, &q).to_bits());
        for etiquette in 0..2 {
            let mut d = direction(m.dim(), etiquette);
            // h est une distance totale dans SU(2)^N, comme pour les tests de J.
            let norme = d.iter().map(|x| x * x).sum::<f64>().sqrt();
            d.iter_mut().for_each(|x| *x /= norme);
            let analytique = d.iter().zip(&ev.grad).map(|(v, g)| v * g).sum::<f64>();
            let df = |h| {
                (potentiel_complet(&m, &m.avancer(&q, &d, h))
                    - potentiel_complet(&m, &m.avancer(&q, &d, -h)))
                    / (2.0 * h)
            };
            let richardson = (4.0 * df(0.0005) - df(0.001)) / 3.0;
            let relatif = (richardson - analytique).abs() / analytique.abs().max(1e-12);
            assert!(
                relatif < 1e-6,
                "direction {etiquette} : {analytique} != {richardson}, relatif {relatif:e}"
            );
        }
    }

    fn point_projete() -> ChampHote {
        static POINT: OnceLock<ChampHote> = OnceLock::new();
        POINT
            .get_or_init(|| {
                let (reseau, q) = donnees();
                let cible = cible_depuis(&q, &reseau);
                let m = FibreSu2::nouveau(reseau.clone(), cible.clone(), 2.0, 8).unwrap();
                let flux = Flux::nouveau(731);
                let d: Vec<_> = (0..m.dim())
                    .map(|i| flux.gauss(0, 0, i as u32, TAG_DEPART))
                    .collect();
                let perturbe = m.avancer(&q, &d, 0.005);
                assert!(m.residu_max(&perturbe).unwrap() > 1e-5);
                drop(m);
                let projete =
                    projeter_continuation(reseau.clone(), &perturbe, &cible, 2.0, 8, 2).unwrap();
                let m = FibreSu2::nouveau(reseau, cible, 2.0, 1).unwrap();
                assert!(m.residu_max(&projete).unwrap() < 1e-13);
                projete
            })
            .clone()
    }

    #[test]
    fn pas_contraint_et_reversibilite_cinq_pas() {
        // Calculer le point avant de créer le pool du test, pour borner les fils.
        let q = point_projete();
        let (reseau, origine) = donnees();
        let m = FibreSu2::nouveau(reseau.clone(), cible_depuis(&origine, &reseau), 2.0, 8).unwrap();
        let ev = m.evaluer(&q).unwrap();
        let p = tirer_impulsion(&m, &ev, &Flux::nouveau(714), 0, 0);
        let params = ParamsRattle {
            dt: 0.01,
            nb_pas: 5,
            ..Default::default()
        };
        let (q1, ev1, p1, _) = pas_rattle(&m, &q, &ev, &p, &params).unwrap();
        assert!(m.residu_max(&q1).unwrap() < 1e-12);
        assert!(m.j_mul(&ev1, &p1).iter().all(|x| x.abs() < 1e-10));
        assert!(q1.defaut_unitarite() < 1e-14);
        let (q5, ev5, p5, h) = integrer(&m, q.clone(), ev, p.clone(), &params).unwrap();
        assert!(h.iter().all(|x| x.is_finite()));
        let opposee = p5.iter().map(|p| -p).collect();
        let (retour, _, pr, _) = integrer(&m, q5, ev5, opposee, &params).unwrap();
        let ecart_p = pr
            .iter()
            .zip(p)
            .map(|(p, p0)| (p + p0).abs())
            .fold(0.0, f64::max);
        assert!(
            m.ecart(&retour, &q) < 1e-10,
            "écart q {:e}",
            m.ecart(&retour, &q)
        );
        assert!(ecart_p < 1e-10, "écart p {ecart_p:e}");
    }

    #[test]
    fn un_pas_identique_pour_un_trois_huit_fils() {
        let (reseau, origine) = donnees();
        let q = point_projete();
        let cible = cible_depuis(&origine, &reseau);
        let mut reference = None;
        for fils in [1, 3, 8] {
            let m = FibreSu2::nouveau(reseau.clone(), cible.clone(), 2.0, fils).unwrap();
            let ev = m.evaluer(&q).unwrap();
            let p = tirer_impulsion(&m, &ev, &Flux::nouveau(714), 0, 0);
            let (q1, ev1, p1, stats) =
                pas_rattle(&m, &q, &ev, &p, &ParamsRattle::default()).unwrap();
            let octets: Vec<_> =
                q1.q.iter()
                    .chain(&ev1.grad)
                    .chain(&p1)
                    .chain(std::iter::once(&ev1.potentiel))
                    .map(|x| x.to_bits())
                    .collect();
            if let Some((ref precedent, stats_ref)) = reference {
                assert_eq!(&octets, precedent, "{fils} fils");
                assert_eq!(stats, stats_ref);
            } else {
                reference = Some((octets, stats));
            }
        }
    }

    #[test]
    fn adjoints_j_et_jt() {
        let (reseau, q) = donnees();
        let m = FibreSu2::nouveau(reseau.clone(), cible_depuis(&q, &reseau), 2.0, 8).unwrap();
        let ev = m.evaluer(&q).unwrap();
        let v = direction(m.dim(), 8);
        let u = direction(m.nb_contraintes(), 9);
        let jv = m.j_mul(&ev, &v);
        let jtu = m.jt_mul(&ev, &u);
        let a = jv.iter().zip(&u).map(|(v, u)| v * u).sum::<f64>();
        let b = v.iter().zip(jtu).map(|(v, u)| v * u).sum::<f64>();
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn constructeur_et_continuation_invalides() {
        let (reseau, q) = donnees();
        assert!(FibreSu2::nouveau(reseau.clone(), Vec::new(), 2.0, 1).is_err());
        let cible = cible_depuis(&q, &reseau);
        assert!(FibreSu2::nouveau(reseau.clone(), cible.clone(), 2.0, 0).is_err());
        assert!(projeter_continuation(reseau, &q, &cible, 2.0, 1, 0).is_err());
    }
}
