//! Verlet contraint dans le repère spatial : les perturbations sont gauches,
//! `q₁ = exp(dt ξ) q`, avec `ξ = M⁻¹ p`. Le modèle fournit **tout** le potentiel,
//! notamment `V = S + ½ log det(J M⁻¹ Jᵀ)` pour la mesure conditionnelle de coaire.
//!
//! Après la demi-force `p̃ = p − dt ∇V/2`, Newton cherche
//! `g(avancer(q, M⁻¹(p̃ − dt Jᵀλ/2), dt)) = 0`. La factorisation initiale de
//! `G = J M⁻¹ Jᵀ` reste figée, et `λ += 2 G⁻¹g/dt²` (signe positif).
//! La seconde demi-force est suivie de la projection `p − JᵀG⁻¹JM⁻¹p`.
//! Un retour depuis `(q₁, −p₁)` vérifie chaque pas avant Metropolis. Les erreurs
//! rejettent la proposition entière ; un rejet Metropolis conserve sa proposition
//! et son ΔH dans la sortie, pour permettre les diagnostics de ⟨exp(−ΔH)⟩.

use super::alea::{Flux, TAG_ACCEPTATION, TAG_IMPULSION};
use crate::commun::{Erreur, Resultat};

pub trait Variete: Sync {
    type Etat: Clone + Send + Sync;
    type Eval: Send + Sync;
    fn dim(&self) -> usize;
    fn nb_contraintes(&self) -> usize;
    fn minv(&self) -> &[f64];
    fn evaluer(&self, q: &Self::Etat) -> Resultat<Self::Eval>;
    fn potentiel(&self, e: &Self::Eval) -> f64;
    fn grad_potentiel<'a>(&self, e: &'a Self::Eval) -> &'a [f64];
    fn jt_mul(&self, e: &Self::Eval, lambda: &[f64]) -> Vec<f64>;
    fn j_mul(&self, e: &Self::Eval, v: &[f64]) -> Vec<f64>;
    fn g_resoudre(&self, e: &Self::Eval, b: &[f64]) -> Vec<f64>;
    fn residu(&self, q: &Self::Etat) -> Resultat<Vec<f64>>;
    fn avancer(&self, q: &Self::Etat, xi: &[f64], dt: f64) -> Self::Etat;
    fn ecart(&self, a: &Self::Etat, b: &Self::Etat) -> f64;
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Integrateur {
    #[default]
    Verlet,
    EulerSymplectique,
}

#[derive(Clone, Copy, Debug)]
pub struct ParamsRattle {
    pub dt: f64,
    pub nb_pas: usize,
    pub tol_newton: f64,
    pub max_newton: usize,
    pub tol_rev: f64,
    pub controle_reversibilite: bool,
    #[cfg(test)]
    pub integrateur: Integrateur,
}

impl Default for ParamsRattle {
    fn default() -> Self {
        Self {
            dt: 0.01,
            nb_pas: 20,
            tol_newton: 1e-13,
            max_newton: 50,
            tol_rev: 1e-9,
            controle_reversibilite: true,
            #[cfg(test)]
            integrateur: Integrateur::Verlet,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Echec {
    Newton,
    Reversibilite,
    Evaluation,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StatsPas {
    /// Nombre de corrections de Newton à l'aller (zéro si déjà convergé).
    pub nb_newton: usize,
    pub residu_max: f64,
    pub nb_newton_retour: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SortieProposition {
    pub accepte: bool,
    pub echec: Option<Echec>,
    /// NaN seulement si une erreur empêche de définir les deux Hamiltoniens.
    pub delta_h: f64,
    pub nb_newton: usize,
    pub residu_max: f64,
}

pub type EtatPas<M> = (
    <M as Variete>::Etat,
    <M as Variete>::Eval,
    Vec<f64>,
    StatsPas,
);
pub type Integration<M> = (
    <M as Variete>::Etat,
    <M as Variete>::Eval,
    Vec<f64>,
    Vec<f64>,
);
pub type Proposition<M> = (
    Option<(<M as Variete>::Etat, <M as Variete>::Eval)>,
    SortieProposition,
);

fn norme_max(v: &[f64]) -> Option<f64> {
    v.iter()
        .try_fold(0.0_f64, |s, x| x.is_finite().then_some(s.max(x.abs())))
}

fn evaluation_valide<M: Variete>(m: &M, ev: &M::Eval) -> bool {
    m.potentiel(ev).is_finite()
        && m.grad_potentiel(ev).len() == m.dim()
        && norme_max(m.grad_potentiel(ev)).is_some()
        && m.minv().len() == m.dim()
        && m.minv().iter().all(|x| x.is_finite() && *x > 0.0)
}

/// Projection orthogonale pour la métrique cinétique, sans changer le point.
pub fn projeter_tangent<M: Variete>(m: &M, ev: &M::Eval, p: &[f64]) -> Vec<f64> {
    let vitesse: Vec<_> = p.iter().zip(m.minv()).map(|(p, mi)| p * mi).collect();
    let correction = m.jt_mul(ev, &m.g_resoudre(ev, &m.j_mul(ev, &vitesse)));
    p.iter().zip(correction).map(|(p, c)| p - c).collect()
}

pub fn tirer_impulsion<M: Variete>(
    m: &M,
    ev: &M::Eval,
    flux: &Flux,
    chaine: u32,
    indice_traj: u32,
) -> Vec<f64> {
    let p: Vec<_> = m
        .minv()
        .iter()
        .enumerate()
        .map(|(i, mi)| flux.gauss(chaine, indice_traj, i as u32, TAG_IMPULSION) / mi.sqrt())
        .collect();
    projeter_tangent(m, ev, &p)
}

pub fn energie_cinetique(minv: &[f64], p: &[f64]) -> f64 {
    0.5 * minv.iter().zip(p).map(|(mi, p)| mi * p * p).sum::<f64>()
}

pub fn hamiltonien<M: Variete>(m: &M, ev: &M::Eval, p: &[f64]) -> f64 {
    m.potentiel(ev) + energie_cinetique(m.minv(), p)
}

/// Résout la position avec J et G figés (Newton simplifié amorti). Les tampons λ, p et ξ sont réutilisés.
///
/// La racine λ* de `g(avancer(q, minv ∘ (p_libre − facteur Jᵀ λ), dt)) = 0` ne dépend pas de la méthode de recherche : le
/// Newton simplifié (pas `G⁻¹ r/(dt·facteur)`) est essayé en pas entier ; si le résidu ne décroît pas, le pas est divisé par 2
/// (au plus 4 fois, puis le pas est pris tel quel). Le nombre maximal de corrections est `max_newton`.
fn position<M: Variete>(
    m: &M,
    q: &M::Etat,
    ev: &M::Eval,
    p_libre: &[f64],
    params: &ParamsRattle,
    facteur: f64,
) -> Result<(M::Etat, Vec<f64>, usize, f64), Echec> {
    let n = m.dim();
    let mut p = vec![0.0; n];
    let mut xi = vec![0.0; n];
    let essai = |lambda: &[f64], p: &mut Vec<f64>, xi: &mut Vec<f64>| -> Result<(M::Etat, Vec<f64>, f64), Echec> {
        let normale = m.jt_mul(ev, lambda);
        if normale.len() != n {
            return Err(Echec::Evaluation);
        }
        for i in 0..n {
            p[i] = p_libre[i] - facteur * normale[i];
            xi[i] = m.minv()[i] * p[i];
        }
        if norme_max(xi).is_none() {
            return Err(Echec::Evaluation);
        }
        let q1 = m.avancer(q, xi, params.dt);
        // résidu indéfini (logarithme ambigu, valeur non finie) : évaluation invalide, PAS une non-convergence de Newton
        let r = m.residu(&q1).map_err(|_| Echec::Evaluation)?;
        if r.len() != lambda.len() {
            return Err(Echec::Evaluation);
        }
        let maximum = norme_max(&r).ok_or(Echec::Evaluation)?;
        Ok((q1, r, maximum))
    };
    let mut lambda = vec![0.0; m.nb_contraintes()];
    let (mut q1, mut r, mut maximum) = essai(&lambda, &mut p, &mut xi)?;
    for iteration in 0..=params.max_newton {
        if maximum < params.tol_newton {
            // p doit correspondre au dernier λ accepté : recalcul sans coût d'évaluation de résidu
            let normale = m.jt_mul(ev, &lambda);
            for i in 0..n {
                p[i] = p_libre[i] - facteur * normale[i];
            }
            return Ok((q1, p, iteration, maximum));
        }
        if iteration == params.max_newton {
            break;
        }
        let correction = m.g_resoudre(ev, &r);
        if correction.len() != lambda.len() || norme_max(&correction).is_none() {
            return Err(Echec::Evaluation);
        }
        let mut pas = 1.0;
        loop {
            let candidat: Vec<f64> = lambda.iter().zip(&correction).map(|(l, c)| l + pas * c / (params.dt * facteur)).collect();
            let (q2, r2, max2) = essai(&candidat, &mut p, &mut xi)?;
            if max2 < maximum || pas <= 1.0 / 16.0 {
                lambda = candidat;
                q1 = q2;
                r = r2;
                maximum = max2;
                break;
            }
            pas *= 0.5;
        }
    }
    Err(Echec::Newton)
}

pub fn pas_rattle<M: Variete>(
    m: &M,
    q: &M::Etat,
    ev: &M::Eval,
    p: &[f64],
    params: &ParamsRattle,
) -> Result<EtatPas<M>, Echec> {
    if !evaluation_valide(m, ev) || p.len() != m.dim() || norme_max(p).is_none() {
        return Err(Echec::Evaluation);
    }
    if !params.dt.is_finite()
        || params.dt == 0.0
        || !params.tol_newton.is_finite()
        || params.tol_newton <= 0.0
        || !params.tol_rev.is_finite()
        || params.tol_rev <= 0.0
    {
        return Err(Echec::Newton);
    }
    #[cfg(test)]
    let euler = params.integrateur == Integrateur::EulerSymplectique;
    #[cfg(not(test))]
    let euler = false;
    let facteur = if euler { 1.0 } else { params.dt / 2.0 };
    let force = if euler { params.dt } else { params.dt / 2.0 };
    let p_libre: Vec<_> = p
        .iter()
        .zip(m.grad_potentiel(ev))
        .map(|(p, g)| p - force * g)
        .collect();
    let (q1, mut p_demi, nb_newton, residu_max) = position(m, q, ev, &p_libre, params, facteur)?;
    let ev1 = m.evaluer(&q1).map_err(|_| Echec::Evaluation)?;
    if !evaluation_valide(m, &ev1) {
        return Err(Echec::Evaluation);
    }
    if !euler {
        for (p, g) in p_demi.iter_mut().zip(m.grad_potentiel(&ev1)) {
            *p -= params.dt * g / 2.0;
        }
    }
    let p1 = projeter_tangent(m, &ev1, &p_demi);
    if p1.len() != m.dim() || norme_max(&p1).is_none() {
        return Err(Echec::Evaluation);
    }
    let mut stats = StatsPas {
        nb_newton,
        residu_max,
        nb_newton_retour: 0,
    };
    if params.controle_reversibilite && !euler {
        let retour: Vec<_> = p1
            .iter()
            .zip(m.grad_potentiel(&ev1))
            .map(|(p, g)| -p - params.dt * g / 2.0)
            .collect();
        let (q_retour, _, iterations, _) = position(m, &q1, &ev1, &retour, params, facteur)?;
        stats.nb_newton_retour = iterations;
        let ecart = m.ecart(&q_retour, q);
        if !ecart.is_finite() || ecart >= params.tol_rev {
            return Err(Echec::Reversibilite);
        }
    }
    Ok((q1, ev1, p1, stats))
}

fn accumuler(total: &mut StatsPas, pas: StatsPas) {
    total.nb_newton += pas.nb_newton;
    total.nb_newton_retour += pas.nb_newton_retour;
    total.residu_max = total.residu_max.max(pas.residu_max);
}

/// Une évaluation par pas ; le dernier Eval devient celui du pas suivant.
pub fn trajectoire<M: Variete>(
    m: &M,
    q: M::Etat,
    ev: M::Eval,
    p: Vec<f64>,
    params: &ParamsRattle,
) -> Result<EtatPas<M>, Echec> {
    let (mut q, mut ev, mut p) = (q, ev, p);
    let mut total = StatsPas::default();
    for _ in 0..params.nb_pas {
        let (q1, ev1, p1, stats) = pas_rattle(m, &q, &ev, &p, params)?;
        accumuler(&mut total, stats);
        (q, ev, p) = (q1, ev1, p1);
    }
    Ok((q, ev, p, total))
}

/// Intégration sans Metropolis ; H contient l'instant initial et chaque pas.
pub fn integrer<M: Variete>(
    m: &M,
    q: M::Etat,
    ev: M::Eval,
    p: Vec<f64>,
    params: &ParamsRattle,
) -> Result<Integration<M>, Echec> {
    let (mut q, mut ev, mut p) = (q, ev, p);
    let mut energies = Vec::with_capacity(params.nb_pas + 1);
    energies.push(hamiltonien(m, &ev, &p));
    for _ in 0..params.nb_pas {
        let (q1, ev1, p1, _) = pas_rattle(m, &q, &ev, &p, params)?;
        (q, ev, p) = (q1, ev1, p1);
        energies.push(hamiltonien(m, &ev, &p));
    }
    Ok((q, ev, p, energies))
}

/// L'état proposé est rendu même si Metropolis le refuse. En cas d'échec, None.
/// L'Eval initial est emprunté pour que l'appelant puisse le conserver au rejet.
pub fn proposition<M: Variete>(
    m: &M,
    q: &M::Etat,
    ev: &M::Eval,
    params: &ParamsRattle,
    flux: &Flux,
    chaine: u32,
    indice_traj: u32,
) -> Proposition<M> {
    let mut sortie = SortieProposition {
        accepte: false,
        echec: None,
        delta_h: f64::NAN,
        nb_newton: 0,
        residu_max: 0.0,
    };
    let calcul = |sortie: &mut SortieProposition| -> Result<(M::Etat, M::Eval), Echec> {
        if !evaluation_valide(m, ev) {
            return Err(Echec::Evaluation);
        }
        let p0 = tirer_impulsion(m, ev, flux, chaine, indice_traj);
        let h0 = hamiltonien(m, ev, &p0);
        if !h0.is_finite() {
            return Err(Echec::Evaluation);
        }
        let mut courant = q.clone();
        let mut evaluation = None;
        let mut p = p0;
        for _ in 0..params.nb_pas {
            let e = evaluation.as_ref().unwrap_or(ev);
            let (q1, ev1, p1, stats) = pas_rattle(m, &courant, e, &p, params)?;
            sortie.nb_newton += stats.nb_newton;
            sortie.residu_max = sortie.residu_max.max(stats.residu_max);
            courant = q1;
            evaluation = Some(ev1);
            p = p1;
        }
        sortie.delta_h = hamiltonien(m, evaluation.as_ref().unwrap_or(ev), &p) - h0;
        if !sortie.delta_h.is_finite() {
            return Err(Echec::Evaluation);
        }
        sortie.accepte =
            flux.uniforme(chaine, indice_traj, 0, TAG_ACCEPTATION).ln() < -sortie.delta_h;
        // Seule une trajectoire vide demande de reconstruire l'Eval emprunté.
        let evaluation = match evaluation {
            Some(e) => e,
            None => m.evaluer(&courant).map_err(|_| Echec::Evaluation)?,
        };
        Ok((courant, evaluation))
    };
    match calcul(&mut sortie) {
        Ok(propose) => (Some(propose), sortie),
        Err(e) => {
            sortie.accepte = false;
            sortie.echec = Some(e);
            sortie.delta_h = f64::NAN;
            (None, sortie)
        }
    }
}

/// Le rappel voit l'état courant après acceptation ou rejet, indice à partir de 0.
/// Une erreur au départ est rendue ; les erreurs des propositions sont des rejets.
pub fn chaine<M: Variete>(
    m: &M,
    q0: M::Etat,
    params: &ParamsRattle,
    flux: &Flux,
    chaine: u32,
    n_traj: usize,
    mut rappel: impl FnMut(usize, &M::Etat, &SortieProposition),
) -> Resultat<M::Etat> {
    if n_traj > u32::MAX as usize {
        return Err(Erreur::Validation("compteur de trajectoire épuisé".into()));
    }
    let mut q = q0;
    let mut ev = m.evaluer(&q)?;
    for i in 0..n_traj {
        let (propose, sortie) = proposition(m, &q, &ev, params, flux, chaine, i as u32);
        if sortie.accepte {
            if let Some((q1, ev1)) = propose {
                (q, ev) = (q1, ev1);
            }
        }
        rappel(i, &q, &sortie);
    }
    Ok(q)
}

/// Gauss–Newton amorti. Hors fibre, le modèle fournit l'approximation spatiale J.
pub fn projeter_sur_fibre<M: Variete>(
    m: &M,
    q: &M::Etat,
    tol: f64,
    max_iter: usize,
) -> Resultat<M::Etat> {
    let erreur = || Erreur::Validation("projection sur la fibre non convergée".into());
    if !tol.is_finite() || tol <= 0.0 {
        return Err(erreur());
    }
    let mut courant = q.clone();
    for iteration in 0..=max_iter {
        let r = m
            .residu(&courant)
            .map_err(|e| Erreur::Validation(format!("résidu de la projection sur fibre : {e}")))?;
        let maximum = norme_max(&r).ok_or_else(erreur)?;
        if r.len() != m.nb_contraintes() {
            return Err(erreur());
        }
        if maximum < tol {
            return Ok(courant);
        }
        if iteration == max_iter {
            break;
        }
        let ev = m.evaluer(&courant).map_err(|e| {
            Erreur::Validation(format!("évaluation de la projection sur fibre : {e}"))
        })?;
        if !evaluation_valide(m, &ev) {
            return Err(erreur());
        }
        let normale = m.jt_mul(&ev, &m.g_resoudre(&ev, &r));
        if normale.len() != m.dim() {
            return Err(erreur());
        }
        let direction: Vec<_> = normale
            .iter()
            .zip(m.minv())
            .map(|(v, mi)| -mi * v)
            .collect();
        if norme_max(&direction).is_none() {
            return Err(erreur());
        }
        let mut pas = 1.0;
        let mut suivant = None;
        for _ in 0..=10 {
            let essai = m.avancer(&courant, &direction, pas);
            // Une erreur de branche à un essai autorise un pas plus court.
            if let Ok(r) = m.residu(&essai) {
                if r.len() == m.nb_contraintes() && norme_max(&r).is_some_and(|v| v < maximum) {
                    suivant = Some(essai);
                    break;
                }
            }
            pas *= 0.5;
        }
        courant = suivant.ok_or_else(erreur)?;
    }
    Err(erreur())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fibre_contrainte::jouets::{Courbe, EvalCourbe};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn depart(m: &Courbe) -> ([f64; 2], EvalCourbe, Vec<f64>) {
        let x: f64 = 0.7;
        let q = [x, m.a * x.sin()];
        let ev = m.evaluer(&q).unwrap();
        let p = projeter_tangent(m, &ev, &[1.1, 0.3]);
        (q, ev, p)
    }

    #[test]
    fn newton_reversibilite_et_masse_diagonale() {
        for m in [Courbe::default(), Courbe::nouveau(1.0, 4.0)] {
            let (q0, mut ev, p0) = depart(&m);
            let (mut q, mut p) = (q0, p0.clone());
            let params = ParamsRattle {
                dt: 0.03,
                nb_pas: 20,
                ..Default::default()
            };
            for _ in 0..20 {
                let (q1, ev1, p1, stats) = pas_rattle(&m, &q, &ev, &p, &params).unwrap();
                assert!(m.residu(&q1).unwrap()[0].abs() <= 1e-12);
                assert!(stats.residu_max < params.tol_newton);
                let vitesse: Vec<_> = p1.iter().zip(m.minv()).map(|(p, mi)| p * mi).collect();
                assert!(m.j_mul(&ev1, &vitesse)[0].abs() < 1e-12);
                (q, ev, p) = (q1, ev1, p1);
            }
            p.iter_mut().for_each(|x| *x = -*x);
            let (q, _, p, h) = integrer(&m, q, ev, p, &params).unwrap();
            assert_eq!(h.len(), 21);
            assert!(m.ecart(&q, &q0) < 1e-10);
            for (p, p0) in p.iter().zip(p0) {
                assert!((p + p0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn ordre_deux_et_mutant_ordre_un() {
        let m = Courbe::default();
        for (integrateur, ordre) in [
            (Integrateur::Verlet, 2.0),
            (Integrateur::EulerSymplectique, 1.0),
        ] {
            let mut erreurs = Vec::new();
            for dt in [0.2, 0.1, 0.05] {
                let (q, ev, p) = depart(&m);
                let params = ParamsRattle {
                    dt,
                    nb_pas: (1.0 / dt).round() as usize,
                    integrateur,
                    ..Default::default()
                };
                let (_, _, _, h) = integrer(&m, q, ev, p, &params).unwrap();
                erreurs.push(h.iter().map(|v| (v - h[0]).abs()).fold(0.0, f64::max));
            }
            let pente = (erreurs[0] / erreurs[2]).ln() / 4.0_f64.ln();
            assert!(
                (pente - ordre).abs() < 0.2,
                "{integrateur:?} : pente {pente}, erreurs {erreurs:?}"
            );
        }
    }

    #[test]
    fn determinisme_des_propositions_et_de_la_chaine() {
        let m = Courbe::default();
        let (q, ev, _) = depart(&m);
        let params = ParamsRattle {
            dt: 0.06,
            nb_pas: 8,
            ..Default::default()
        };
        let flux = Flux::nouveau(274);
        let (a, sa) = proposition(&m, &q, &ev, &params, &flux, 3, 17);
        let (b, sb) = proposition(&m, &q, &ev, &params, &flux, 3, 17);
        assert_eq!(sa, sb);
        let (a, b) = (a.unwrap(), b.unwrap());
        for (a, b) in a.0.iter().zip(b.0) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(a.1.potentiel.to_bits(), b.1.potentiel.to_bits());
        for (a, b) in a.1.grad.iter().zip(b.1.grad) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        let collecter = || {
            let mut mesures = Vec::new();
            let fin = chaine(&m, q, &params, &flux, 3, 50, |i, q, s| {
                mesures.push((
                    i,
                    q.map(f64::to_bits),
                    s.delta_h.to_bits(),
                    s.accepte,
                    s.echec,
                ));
            })
            .unwrap();
            (fin.map(f64::to_bits), mesures)
        };
        assert_eq!(collecter(), collecter());
    }

    #[test]
    fn projection_amortie_sur_la_courbe() {
        let m = Courbe::nouveau(1.0, 4.0);
        let q = [0.7, 1.0];
        let projete = projeter_sur_fibre(&m, &q, 1e-13, 30).unwrap();
        assert!(m.residu(&projete).unwrap()[0].abs() < 1e-13);
        assert!(projeter_sur_fibre(&m, &q, 1e-13, 0).is_err());
        assert!(projeter_sur_fibre(&m, &q, f64::NAN, 30).is_err());
    }

    /// Instrumente les évaluations et les rejets sans modifier le moteur.
    struct Surveillance {
        m: Courbe,
        evaluations: AtomicUsize,
        defaut: u8,
    }
    impl Variete for Surveillance {
        type Etat = [f64; 2];
        type Eval = EvalCourbe;
        fn dim(&self) -> usize {
            self.m.dim()
        }
        fn nb_contraintes(&self) -> usize {
            1
        }
        fn minv(&self) -> &[f64] {
            self.m.minv()
        }
        fn evaluer(&self, q: &Self::Etat) -> Resultat<Self::Eval> {
            self.evaluations.fetch_add(1, Ordering::Relaxed);
            if self.defaut == 1 {
                return Err(Erreur::Validation("évaluation forcée".into()));
            }
            self.m.evaluer(q)
        }
        fn potentiel(&self, e: &Self::Eval) -> f64 {
            self.m.potentiel(e)
        }
        fn grad_potentiel<'a>(&self, e: &'a Self::Eval) -> &'a [f64] {
            &e.grad
        }
        fn jt_mul(&self, e: &Self::Eval, l: &[f64]) -> Vec<f64> {
            self.m.jt_mul(e, l)
        }
        fn j_mul(&self, e: &Self::Eval, v: &[f64]) -> Vec<f64> {
            self.m.j_mul(e, v)
        }
        fn g_resoudre(&self, e: &Self::Eval, b: &[f64]) -> Vec<f64> {
            self.m.g_resoudre(e, b)
        }
        fn residu(&self, q: &Self::Etat) -> Resultat<Vec<f64>> {
            match self.defaut {
                2 => Ok(vec![f64::NAN]),
                3 => Err(Erreur::Validation("résidu forcé".into())),
                _ => self.m.residu(q),
            }
        }
        fn avancer(&self, q: &Self::Etat, xi: &[f64], dt: f64) -> Self::Etat {
            self.m.avancer(q, xi, dt)
        }
        fn ecart(&self, a: &Self::Etat, b: &Self::Etat) -> f64 {
            if self.defaut == 4 {
                1.0
            } else {
                self.m.ecart(a, b)
            }
        }
    }

    #[test]
    fn evaluations_uniques_et_rejets_distincts() {
        let mut m = Surveillance {
            m: Courbe::default(),
            evaluations: AtomicUsize::new(0),
            defaut: 0,
        };
        let (q, ev, _) = depart(&m.m);
        let params = ParamsRattle {
            nb_pas: 7,
            ..Default::default()
        };
        let flux = Flux::nouveau(42);
        let (propose, sortie) = proposition(&m, &q, &ev, &params, &flux, 0, 0);
        assert!(propose.is_some() && sortie.echec.is_none());
        assert_eq!(m.evaluations.load(Ordering::Relaxed), 7);
        for (defaut, attendu) in [
            (1, Echec::Evaluation),
            (2, Echec::Evaluation),
            (3, Echec::Evaluation),
            (4, Echec::Reversibilite),
        ] {
            m.defaut = defaut;
            let (propose, sortie) = proposition(&m, &q, &ev, &params, &flux, 0, 0);
            assert!(propose.is_none() && !sortie.accepte && sortie.delta_h.is_nan());
            assert_eq!(sortie.echec, Some(attendu));
        }
        m.defaut = 0;
        let params = ParamsRattle {
            max_newton: 0,
            ..params
        };
        let fin = chaine(&m, q, &params, &flux, 0, 5, |_, courant, s| {
            assert_eq!(*courant, q);
            assert_eq!(s.echec, Some(Echec::Newton));
        })
        .unwrap();
        assert_eq!(fin, q);
    }

    #[test]
    fn rejet_metropolis_conserve_la_proposition_et_delta_h() {
        let m = Courbe::default();
        let (q, ev, _) = depart(&m);
        let flux = Flux::nouveau(19);
        let params = ParamsRattle {
            dt: 0.3,
            nb_pas: 5,
            ..Default::default()
        };
        let mut rejets = 0;
        for i in 0..500 {
            let (propose, s) = proposition(&m, &q, &ev, &params, &flux, 0, i);
            if s.echec.is_none() {
                assert!(s.delta_h.is_finite() && propose.is_some());
                let p = tirer_impulsion(&m, &ev, &flux, 0, i);
                let (_, ev1, p1, _) = trajectoire(&m, q, ev.clone(), p.clone(), &params).unwrap();
                assert_eq!(
                    s.delta_h.to_bits(),
                    (hamiltonien(&m, &ev1, &p1) - hamiltonien(&m, &ev, &p)).to_bits()
                );
                assert_eq!(
                    s.accepte,
                    flux.uniforme(0, i, 0, TAG_ACCEPTATION) < (-s.delta_h).exp()
                );
                if !s.accepte {
                    rejets += 1;
                }
            }
        }
        assert!(rejets > 0);
        let mut courant = q;
        chaine(&m, q, &params, &flux, 0, 100, |_, q, s| {
            if !s.accepte {
                assert_eq!(*q, courant);
            }
            courant = *q;
        })
        .unwrap();
    }
}
