//! Calcul des faces n×n, du flux cohérent et des statistiques associées.
//!
//! Toutes les fonctions publiques valident le champ d'entrée (`valider_champ` :
//! longueur 16·L⁴, composantes finies) et exigent n ≥ 1. Le parallélisme est
//! confiné à un pool local de `fils` threads ; les sommes sont calculées par
//! tranches fixes de [`super::TRANCHE`] sites (sommes partielles par tranche,
//! combinées séquentiellement dans l'ordre des tranches) : le résultat est
//! identique À L'OCTET quel que soit `fils`.

use super::{COH_MIN, NBINS_ALPHA, PLANS, SEUILS_COH, SEUILS_X, TRANCHE};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul, Q};
use crate::rg::bloc::{coordonnees, deplacer, lire, site, valider_champ};
use rayon::prelude::*;
use std::f64::consts::PI;

const IDENTITE: Q = [1.0, 0.0, 0.0, 0.0];
/// Seuil d'ambiguïté d'une plaquette : Θ_p > 2π − 2·10⁻⁶.
const EPS_AMBIGUE: f64 = 2e-6;

/// Pavage des coins de faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pavage {
    /// Tous les sites (V × 6 faces).
    Tous,
    /// Seulement les coins dont les quatre coordonnées valent `o` modulo 3.
    /// Nécessite L multiple de 3.
    Blocs(usize),
}

/// Variantes du calcul du flux cohérent (mutants pour les gates, jamais exposées
/// par la CLI).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VarianteFlux {
    /// Transport des plaquettes par les lignes S_μ^(a) et S_ν^(b).
    #[default]
    Normale,
    /// T_ab = identité : l'invariance de jauge du flux cohérent doit casser.
    SansTransport,
    /// Le plan (2,3) est omis des statistiques : les comptes de la référence
    /// Python doivent différer.
    PlanManquant,
}

/// Statistiques brutes (x = ½ Tr W) : §2.2 de la spécification.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct StatsFaces {
    pub n: u64,
    pub comptes: [u64; 5],
    pub sx: f64,
    pub sx2: f64,
    pub s_theta2: f64,
    pub s_theta4: f64,
    pub xmin: f64,
    pub hist_alpha: Option<Vec<u64>>,
}

impl StatsFaces {
    /// Accumulateur vide : sommes et comptes nuls, xmin = +∞ (élément neutre),
    /// histogramme initialisé seulement s'il est demandé.
    pub(crate) fn vide(histogramme: bool) -> Self {
        StatsFaces {
            n: 0,
            comptes: [0; 5],
            sx: 0.0,
            sx2: 0.0,
            s_theta2: 0.0,
            s_theta4: 0.0,
            xmin: f64::INFINITY,
            hist_alpha: if histogramme {
                Some(vec![0; NBINS_ALPHA])
            } else {
                None
            },
        }
    }

    /// Fusionne les statistiques partielles d'une tranche dans l'accumulateur.
    /// L'ordre des fusions est celui des tranches, imposé par l'appelant.
    pub(crate) fn fusionner(&mut self, autre: &StatsFaces) {
        if autre.n == 0 {
            return; // Une tranche vidée par le pavage ne contribue pas à xmin.
        }
        self.n += autre.n;
        for (a, b) in self.comptes.iter_mut().zip(autre.comptes) {
            *a += b;
        }
        self.sx += autre.sx;
        self.sx2 += autre.sx2;
        self.s_theta2 += autre.s_theta2;
        self.s_theta4 += autre.s_theta4;
        self.xmin = self.xmin.min(autre.xmin);
        match (&mut self.hist_alpha, &autre.hist_alpha) {
            (Some(a), Some(b)) => {
                for (ca, cb) in a.iter_mut().zip(b.iter()) {
                    *ca += cb;
                }
            }
            (None, Some(b)) => self.hist_alpha = Some(b.clone()),
            _ => {}
        }
    }
}

/// Statistiques du flux cohérent : §2.3 de la spécification.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct StatsCoherent {
    pub n: u64,
    pub comptes: [u64; 4],
    pub comptes_coherence: [[u64; 3]; 4],
    pub s_phi: f64,
    pub s_phi2: f64,
    pub s_coh: f64,
    pub phi_max: f64,
    pub n_ambigues: u64,
}

impl StatsCoherent {
    pub(crate) fn vide() -> Self {
        StatsCoherent {
            n: 0,
            comptes: [0; 4],
            comptes_coherence: [[0; 3]; 4],
            s_phi: 0.0,
            s_phi2: 0.0,
            s_coh: 0.0,
            phi_max: 0.0,
            n_ambigues: 0,
        }
    }

    pub(crate) fn fusionner(&mut self, autre: &StatsCoherent) {
        self.n += autre.n;
        for (a, b) in self.comptes.iter_mut().zip(autre.comptes) {
            *a += b;
        }
        for (ligne_a, ligne_b) in self
            .comptes_coherence
            .iter_mut()
            .zip(autre.comptes_coherence)
        {
            for (a, b) in ligne_a.iter_mut().zip(ligne_b) {
                *a += b;
            }
        }
        self.s_phi += autre.s_phi;
        self.s_phi2 += autre.s_phi2;
        self.s_coh += autre.s_coh;
        self.phi_max = self.phi_max.max(autre.phi_max);
        self.n_ambigues += autre.n_ambigues;
    }
}

/// Angle propre d'un quaternion W par la formule bien conditionnée :
/// Θ_atan = 2·atan2(|W⃗|, W[0]). Utilitaire pour les contrôles de précision
/// (§2.1) ; les statistiques utilisent `2 arccos(xc)`.
pub fn angle_face(w: Q) -> f64 {
    2.0 * (w[1].hypot(w[2]).hypot(w[3])).atan2(w[0])
}

/// Crée un pool rayon local de `fils` threads.
fn pool(fils: usize) -> Resultat<rayon::ThreadPool> {
    if fils == 0 {
        return Err(Erreur::Config("le nombre de fils doit être ≥ 1".into()));
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(fils)
        .build()
        .map_err(|e| Erreur::Config(format!("création du pool de fils impossible : {e}")))
}

/// Vérifie qu'un plan est valide.
fn plan_valide(plan: (usize, usize)) -> Resultat<()> {
    let (a, b) = plan;
    if a >= 4 || b >= 4 || a == b {
        return Err(Erreur::Config(format!(
            "plan ({a},{b}) invalide : directions distinctes dans 0..4 requises"
        )));
    }
    Ok(())
}

/// Produit des `n` liens consécutifs dans la direction `mu` à partir du site `x`.
/// S_μ^(n)(x) = U_μ(x) · U_μ(x+μ̂) · … · U_μ(x+(n−1)μ̂).
fn produit_ligne(champ: &ChampHote, x: [usize; 4], mu: usize, n: usize) -> Q {
    let mut prod = IDENTITE;
    for j in 0..n {
        let xj = deplacer(x, mu, j as isize, champ.l);
        prod = qmul(prod, lire(champ, xj, mu));
    }
    prod
}

/// Précalcule S_μ^(n)(s) pour toute direction μ et tout site s.
/// Retourne `Vec<Vec<Q>>` : `lignes[mu][s]`.
fn lignes(champ: &ChampHote, n: usize, fils: usize) -> Resultat<Vec<Vec<Q>>> {
    valider_champ(champ)?;
    if n == 0 {
        return Err(Erreur::Config("n doit être ≥ 1".into()));
    }
    let l = champ.l;
    let v = champ.v();
    let pool = pool(fils)?;
    Ok(pool.install(|| {
        (0..4)
            .into_par_iter()
            .map(|mu| {
                (0..v)
                    .into_par_iter()
                    .map(|s| produit_ligne(champ, coordonnees(s, l), mu, n))
                    .collect::<Vec<Q>>()
            })
            .collect::<Vec<Vec<Q>>>()
    }))
}

/// Précalcule les plaquettes U_p(y) pour tous les plans et sites.
/// Retourne `Vec<Vec<Q>>` : `plaquettes[plan][site]`.
fn plaquettes(champ: &ChampHote, fils: usize) -> Resultat<Vec<Vec<Q>>> {
    valider_champ(champ)?;
    let l = champ.l;
    let v = champ.v();
    let pool = pool(fils)?;
    Ok(pool.install(|| {
        PLANS
            .into_par_iter()
            .map(|(mu, nu)| {
                (0..v)
                    .into_par_iter()
                    .map(|s| {
                        let y = coordonnees(s, l);
                        let u_mu = lire(champ, y, mu);
                        let y_mu = deplacer(y, mu, 1, l);
                        let u_nu_at_mu = lire(champ, y_mu, nu);
                        let y_nu = deplacer(y, nu, 1, l);
                        let u_mu_at_nu = lire(champ, y_nu, mu);
                        let u_nu = lire(champ, y, nu);
                        // U_p(y) = (U_μ(y) · U_ν(y+μ̂)) · (U_μ(y+ν̂)† · U_ν(y)†)
                        qmul(qmul(u_mu, u_nu_at_mu), qmul(qdag(u_mu_at_nu), qdag(u_nu)))
                    })
                    .collect::<Vec<Q>>()
            })
            .collect::<Vec<Vec<Q>>>()
    }))
}

/// Calcule x = ½ Tr W pour chaque face n×n, ordre [plan][site].
pub fn x_faces(champ: &ChampHote, n: usize, fils: usize) -> Resultat<Vec<f64>> {
    valider_champ(champ)?;
    if n == 0 {
        return Err(Erreur::Config("n doit être ≥ 1".into()));
    }
    let l = champ.l;
    let v = champ.v();
    let total = 6 * v;
    let lignes_tab = lignes(champ, n, fils)?;
    let pool = pool(fils)?;
    let resultat = pool.install(|| {
        (0..total)
            .into_par_iter()
            .map(|idx| {
                let p = idx / v;
                let s = idx % v;
                let (mu, nu) = PLANS[p];
                let x = coordonnees(s, l);
                let x_nmu = deplacer(x, mu, n as isize, l);
                let x_nnu = deplacer(x, nu, n as isize, l);
                let s_mu = lignes_tab[mu][s];
                let s_nu_at_mu = lignes_tab[nu][site(x_nmu, l)];
                let s_mu_at_nu = lignes_tab[mu][site(x_nnu, l)];
                let s_nu = lignes_tab[nu][s];
                // Association EXACTE de boucles.x_boucles : ((A·B)·C†)·D† (spécification §2.1).
                // L'associativité mathématique ne garantit pas les mêmes arrondis ; cette
                // association est celle du portage de référence.
                let w = qmul(qmul(qmul(s_mu, s_nu_at_mu), qdag(s_mu_at_nu)), qdag(s_nu));
                w[0]
            })
            .collect::<Vec<f64>>()
    });
    Ok(resultat)
}

/// Quaternion W de la face (coin s, plan p, taille n) : utilitaire exact pour
/// les tests. Ne valide pas le champ (usage interne/tests).
pub fn boucle_face(champ: &ChampHote, s: usize, plan: usize, n: usize) -> Q {
    let l = champ.l;
    let x = coordonnees(s, l);
    let (mu, nu) = PLANS[plan];
    let s_mu = produit_ligne(champ, x, mu, n);
    let x_nmu = deplacer(x, mu, n as isize, l);
    let s_nu_at_mu = produit_ligne(champ, x_nmu, nu, n);
    let x_nnu = deplacer(x, nu, n as isize, l);
    let s_mu_at_nu = produit_ligne(champ, x_nnu, mu, n);
    let s_nu = produit_ligne(champ, x, nu, n);
    qmul(qmul(qmul(s_mu, s_nu_at_mu), qdag(s_mu_at_nu)), qdag(s_nu))
}

/// Statistiques d'un tableau de x = ½ Tr W (déjà filtré). Sommes séquentielles
/// déterministes sur le tableau fourni.
pub fn stats_x(x: &[f64], histogramme: bool) -> StatsFaces {
    let n = x.len() as u64;
    let mut comptes = [0u64; 5];
    let mut sx = 0.0;
    let mut sx2 = 0.0;
    let mut s_theta2 = 0.0;
    let mut s_theta4 = 0.0;
    let mut xmin = f64::INFINITY;
    let mut hist = if histogramme {
        Some(vec![0u64; NBINS_ALPHA])
    } else {
        None
    };
    for &xi in x {
        let xc = xi.clamp(-1.0, 1.0);
        for (i, (_, seuil, strict)) in SEUILS_X.iter().enumerate() {
            let cond = if *strict { xc < *seuil } else { xc <= *seuil };
            if cond {
                comptes[i] += 1;
            }
        }
        sx += xc;
        sx2 += xc * xc;
        let alpha = xc.acos();
        let theta = 2.0 * alpha;
        s_theta2 += theta * theta;
        s_theta4 += theta * theta * theta * theta;
        if xc < xmin {
            xmin = xc;
        }
        if let Some(h) = hist.as_mut() {
            let idx = ((alpha * NBINS_ALPHA as f64 / PI).floor() as usize).min(NBINS_ALPHA - 1);
            h[idx] += 1;
        }
    }
    if n == 0 {
        xmin = 0.0;
    }
    StatsFaces {
        n,
        comptes,
        sx,
        sx2,
        s_theta2,
        s_theta4,
        xmin,
        hist_alpha: hist,
    }
}

/// Tranches de sites physiques, sans jamais traverser la frontière d'un plan.
/// Le pavage est appliqué ensuite, à l'intérieur de chaque tranche.
fn tranches_sites(l: usize) -> Vec<(usize, std::ops::Range<usize>)> {
    let v = l.pow(4);
    let mut tranches = Vec::new();
    for p in 0..PLANS.len() {
        for debut in (0..v).step_by(TRANCHE) {
            tranches.push((p, debut..debut.saturating_add(TRANCHE).min(v)));
        }
    }
    tranches
}

fn coin_retenu(s: usize, l: usize, pavage: Pavage) -> bool {
    match pavage {
        Pavage::Tous => true,
        Pavage::Blocs(o) => coordonnees(s, l).iter().all(|&c| c % 3 == o),
    }
}

/// Calcule les statistiques brutes des faces n×n pour un pavage donné.
///
/// Chaque plan est découpé en tranches fixes de [`TRANCHE`] sites du réseau,
/// AVANT filtrage du pavage. Les sommes partielles sont séquentielles dans
/// chaque tranche, puis fusionnées dans l'ordre (plan, premier site).
/// Le résultat est identique à l'octet quel que soit le nombre de fils.
pub fn stats_faces(
    champ: &ChampHote,
    n: usize,
    pavage: Pavage,
    histogramme: bool,
    fils: usize,
) -> Resultat<StatsFaces> {
    valider_champ(champ)?;
    let l = champ.l;
    if let Pavage::Blocs(o) = pavage {
        if o > 2 {
            return Err(Erreur::Config(format!("pavage o = {o} hors de {{0,1,2}}")));
        }
        if !l.is_multiple_of(3) {
            return Err(Erreur::Config(
                "L multiple de 3 requis pour un pavage de blocs".into(),
            ));
        }
    }
    let x = x_faces(champ, n, fils)?;
    let v = champ.v();
    let tranches = tranches_sites(l);

    let pool = pool(fils)?;
    let partielles: Vec<StatsFaces> = pool.install(|| {
        tranches
            .par_iter()
            .map(|(p, sites)| {
                let valeurs: Vec<f64> = sites
                    .clone()
                    .filter(|&s| coin_retenu(s, l, pavage))
                    .map(|s| x[p * v + s])
                    .collect();
                stats_x(&valeurs, histogramme)
            })
            .collect()
    });

    let mut acc = StatsFaces::vide(histogramme);
    for part in &partielles {
        acc.fusionner(part);
    }
    if acc.n == 0 {
        acc.xmin = 0.0;
    }
    Ok(acc)
}

/// Calcule le flux cohérent pour toutes les faces. Retourne (phi, sabs, ambigu) :
/// phi et sabs de longueur 6·V (ordre [plan][site]), ambigu est le nombre de
/// plaquettes ambiguës par face (même ordre).
fn calculer_flux_coherent(
    champ: &ChampHote,
    n: usize,
    variante: VarianteFlux,
    fils: usize,
) -> Resultat<(Vec<f64>, Vec<f64>, Vec<u64>)> {
    valider_champ(champ)?;
    if n == 0 {
        return Err(Erreur::Config("n doit être ≥ 1".into()));
    }
    let l = champ.l;
    let v = champ.v();
    let total = 6 * v;

    let plaq = plaquettes(champ, fils)?;

    // Précalcul des lignes courtes S_μ^(a) pour a = 0..n-1.
    // `None` si variante SansTransport (T_ab = identité).
    let lignes_courtes: Option<Vec<Vec<Vec<Q>>>> = if variante != VarianteFlux::SansTransport {
        let mut lc: Vec<Vec<Vec<Q>>> = Vec::with_capacity(4);
        for mu in 0..4 {
            let mut table: Vec<Vec<Q>> = Vec::with_capacity(n);
            let ident = vec![IDENTITE; v];
            table.push(ident);
            for a in 1..n {
                let prev = &table[a - 1];
                let mut current = Vec::with_capacity(v);
                for (s, p) in prev.iter().enumerate() {
                    let x = coordonnees(s, l);
                    let x_prev = deplacer(x, mu, (a - 1) as isize, l);
                    let u = lire(champ, x_prev, mu);
                    current.push(qmul(*p, u));
                }
                table.push(current);
            }
            lc.push(table);
        }
        Some(lc)
    } else {
        None
    };

    let pool = pool(fils)?;
    let resultat = pool.install(|| {
        (0..total)
            .into_par_iter()
            .map(|idx| {
                let p = idx / v;
                let s = idx % v;
                let (mu, nu) = PLANS[p];
                if variante == VarianteFlux::PlanManquant && p == 5 {
                    return (0.0f64, 0.0f64, 0u64);
                }
                let x = coordonnees(s, l);
                let mut somme_vect = [0.0f64; 3];
                let mut somme_abs = 0.0f64;
                let mut ambigu = 0u64;
                for a in 0..n {
                    let x_a = deplacer(x, mu, a as isize, l);
                    let s_a = site(x_a, l);
                    for b in 0..n {
                        let x_ab = deplacer(x_a, nu, b as isize, l); // y
                        let s_ab = site(x_ab, l);
                        let up = plaq[p][s_ab];
                        let t_ab = match &lignes_courtes {
                            Some(lc) => qmul(lc[mu][a][s], lc[nu][b][s_a]),
                            None => IDENTITE,
                        };
                        let ut = qmul(qmul(t_ab, up), qdag(t_ab));
                        let v_vec = [ut[1], ut[2], ut[3]];
                        let norme_v = v_vec[0].hypot(v_vec[1]).hypot(v_vec[2]);
                        let theta_p = 2.0 * norme_v.atan2(ut[0]);
                        let echelle = v_vec[0].abs().max(v_vec[1].abs()).max(v_vec[2].abs());
                        if echelle > 0.0 {
                            // La direction est normalisée à l'échelle 1 : hypot(e,e)
                            // peut s'arrondir à e pour e = 2^-1074. Ni θ/|v|,
                            // ni v/|v| sur des sous-normaux ne sont fiables.
                            let direction = v_vec.map(|c| c / echelle);
                            let norme_direction =
                                direction[0].hypot(direction[1]).hypot(direction[2]);
                            for (somme, composante) in somme_vect.iter_mut().zip(direction) {
                                *somme += theta_p * (composante / norme_direction);
                            }
                        }
                        somme_abs += theta_p;
                        if theta_p > 2.0 * PI - EPS_AMBIGUE {
                            ambigu += 1;
                        }
                    }
                }
                let phi_coh = somme_vect[0].hypot(somme_vect[1]).hypot(somme_vect[2]);
                (phi_coh, somme_abs, ambigu)
            })
            .collect::<Vec<(f64, f64, u64)>>()
    });

    let mut phi = Vec::with_capacity(total);
    let mut sabs = Vec::with_capacity(total);
    let mut ambig = Vec::with_capacity(total);
    for (p, s, a) in resultat {
        phi.push(p);
        sabs.push(s);
        ambig.push(a);
    }
    Ok((phi, sabs, ambig))
}

/// (Φ_coh, S, n_ambigües) de chaque face, ordre [plan][site].
pub fn flux_coherent_faces(
    champ: &ChampHote,
    n: usize,
    variante: VarianteFlux,
    fils: usize,
) -> Resultat<(Vec<f64>, Vec<f64>, u64)> {
    let (phi, sabs, ambig) = calculer_flux_coherent(champ, n, variante, fils)?;
    let total_ambig = ambig.iter().sum();
    Ok((phi, sabs, total_ambig))
}

/// Statistiques d'une tranche d'indices pour le flux cohérent (séquentiel).
fn stats_coherent_tranche(
    phi: &[f64],
    sabs: &[f64],
    ambig: &[u64],
    indices: impl Iterator<Item = usize>,
) -> StatsCoherent {
    let mut stats = StatsCoherent::vide();
    for idx in indices {
        let phi_face = phi[idx];
        let sabs_face = sabs[idx];
        let coh = if sabs_face > 0.0 {
            phi_face / sabs_face
        } else {
            1.0
        };
        stats.n += 1;
        for (j, (_, seuil, strict)) in SEUILS_COH.iter().enumerate() {
            let cond = if *strict {
                phi_face > *seuil
            } else {
                phi_face >= *seuil
            };
            if cond {
                stats.comptes[j] += 1;
                for (k, (_, cmin)) in COH_MIN.iter().enumerate() {
                    if coh >= *cmin {
                        stats.comptes_coherence[j][k] += 1;
                    }
                }
            }
        }
        stats.s_phi += phi_face;
        stats.s_phi2 += phi_face * phi_face;
        stats.s_coh += coh;
        if phi_face > stats.phi_max {
            stats.phi_max = phi_face;
        }
        stats.n_ambigues += ambig[idx];
    }
    stats
}

/// Statistiques du flux cohérent : mêmes tranches de 4096 sites par plan que
/// pour le brut, avec filtrage du pavage à l'intérieur de chaque tranche.
pub fn stats_coherent(
    champ: &ChampHote,
    n: usize,
    pavage: Pavage,
    variante: VarianteFlux,
    fils: usize,
) -> Resultat<StatsCoherent> {
    valider_champ(champ)?;
    let l = champ.l;
    if let Pavage::Blocs(o) = pavage {
        if o > 2 {
            return Err(Erreur::Config(format!("pavage o = {o} hors de {{0,1,2}}")));
        }
        if !l.is_multiple_of(3) {
            return Err(Erreur::Config(
                "L multiple de 3 requis pour un pavage de blocs".into(),
            ));
        }
    }
    let (phi, sabs, ambig) = calculer_flux_coherent(champ, n, variante, fils)?;
    let v = champ.v();
    let mut tranches = tranches_sites(l);
    if variante == VarianteFlux::PlanManquant {
        tranches.retain(|(p, _)| *p != 5); // exclut le plan (2,3)
    }

    let pool = pool(fils)?;
    let partielles: Vec<StatsCoherent> = pool.install(|| {
        tranches
            .par_iter()
            .map(|(p, sites)| {
                let indices = sites
                    .clone()
                    .filter(|&s| coin_retenu(s, l, pavage))
                    .map(|s| p * v + s);
                stats_coherent_tranche(&phi, &sabs, &ambig, indices)
            })
            .collect()
    });

    let mut acc = StatsCoherent::vide();
    for part in &partielles {
        acc.fusionner(part);
    }
    Ok(acc)
}

/// Configuration grossière : `pas` étapes de l'opérateur `op` sur `champ`.
pub fn grossier(
    champ: &ChampHote,
    op: &crate::rg::operateur::Operateur,
    pas: usize,
    fils: usize,
) -> Resultat<ChampHote> {
    valider_champ(champ)?;
    if pas == 0 {
        return Err(Erreur::Config("pas doit être ≥ 1".into()));
    }
    let mut courant = champ.clone();
    for k in 0..pas {
        if !courant.l.is_multiple_of(3) {
            return Err(Erreur::Config(format!(
                "après {k} pas, L = {} n'est pas multiple de 3",
                courant.l
            )));
        }
        let sortie = crate::rg::lissage::bloquer_operateur(
            &courant,
            op,
            &crate::rg::lissage::OptionsOperateur::normale(fils),
        )?;
        courant = sortie.grossier;
    }
    Ok(courant)
}

/// Fond abélien uniforme F_{ab} = Φ, Φ = 4πm/L² (périodique exact), jauge de
/// Landau : portage de `fond_abelien.fond`. Tous les liens = identité sauf
/// U_b(x) = exp(i Φ x_a σ₃/2) et, pour x_a = L−1,
/// U_a(x) = exp(−i Φ L x_b σ₃/2). Retourne le champ et Φ.
pub fn fond_abelien(l: usize, m: i64, plan: (usize, usize)) -> Resultat<(ChampHote, f64)> {
    if l == 0 {
        return Err(Erreur::Config("L doit être ≥ 1".into()));
    }
    let (a, b) = plan;
    plan_valide(plan)?;
    let phi = 4.0 * PI * m as f64 / (l * l) as f64;
    let v = l.pow(4);
    let mut q = vec![0.0; 16 * v];
    // tous les liens à l'identité
    for lien in q.chunks_exact_mut(4) {
        lien[0] = 1.0;
    }
    for s in 0..v {
        let x = coordonnees(s, l);
        let xa = x[a] as f64;
        let xb = x[b] as f64;
        // U_b(x) = exp(i Φ x_a σ₃/2)
        let theta = phi * xa / 2.0;
        let idx_b = b * v + s;
        q[4 * idx_b] = theta.cos();
        q[4 * idx_b + 3] = theta.sin();
        // Si x_a = L-1, U_a(x) = exp(-i Φ L x_b σ₃/2)
        if x[a] == l - 1 {
            let theta2 = -phi * l as f64 * xb / 2.0;
            let idx_a = a * v + s;
            q[4 * idx_a] = theta2.cos();
            q[4 * idx_a + 3] = theta2.sin();
        }
    }
    let champ = ChampHote { l, q };
    valider_champ(&champ)?;
    Ok((champ, phi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_x_seuils_et_bornage() {
        let x = vec![-1.2, -0.8, -0.7, 0.0, 0.5, 1.5];
        let s = stats_x(&x, false);
        // xc borné : -1.0, -0.8, -0.7, 0.0, 0.5, 1.0
        assert_eq!(s.n, 6);
        // T1_pi : xc < 0 => les trois premiers = 3
        assert_eq!(s.comptes[0], 3);
        // T2_3pi2 : xc < -0.7071 => -1.0, -0.8 => 2
        assert_eq!(s.comptes[1], 2);
        // C32 (<= -0.995) => -1.0 => 1
        assert_eq!(s.comptes[2], 1);
        // C16 (<= -0.9808) => -1.0 => 1
        assert_eq!(s.comptes[3], 1);
        // C8 (<= -0.9239) => -1.0 => 1
        assert_eq!(s.comptes[4], 1);
        let attendu_sx: f64 = [-1.0, -0.8, -0.7, 0.0, 0.5, 1.0].iter().sum();
        assert!((s.sx - attendu_sx).abs() < 1e-15);
    }

    #[test]
    fn fond_abelien_phi_et_plaquettes() {
        let l = 6;
        let m = 1;
        let (champ, phi) = fond_abelien(l, m, (0, 1)).unwrap();
        // Φ = 4π/36 = π/9
        assert!((phi - PI / 9.0).abs() < 1e-15);
        // Vérifier une plaquette (0,1) : elle doit être exp(i Φ σ3/2), donc q[0]=cos(Φ/2)
        let x = [0, 0, 0, 0];
        let s = site(x, l);
        let p = 0; // plan (0,1)
        let w = boucle_face(&champ, s, p, 1);
        assert!((w[0] - (phi / 2.0).cos()).abs() < 1e-15);
        assert!((w[3] - (phi / 2.0).sin()).abs() < 1e-15);
    }

    #[test]
    fn association_au_seuil_t1_neuf_faces_python() {
        let mut champ = ChampHote::froid(2).unwrap();
        // Cas exact de la relecture du tour 1, construit au seuil T1.
        // boucles.py : 9 faces ; (A·B)·(C†·D†) en compterait 10.
        let liens = [
            (
                0,
                [0, 0, 0, 0],
                [
                    0.5765331040667072,
                    -0.7829653588678909,
                    -0.2332419141584607,
                    -0.013154322779198832,
                ],
            ),
            (
                1,
                [1, 0, 0, 0],
                [
                    -0.10101214447365785,
                    -0.8400468279602109,
                    -0.0005127532738829913,
                    0.5330268385240096,
                ],
            ),
            (
                0,
                [0, 1, 0, 0],
                [
                    0.40803876293722446,
                    0.2633005084602372,
                    0.7112945802121144,
                    0.5081704737055389,
                ],
            ),
            (
                1,
                [0, 0, 0, 0],
                [
                    -0.4924032155809015,
                    0.39524971521403246,
                    -0.4729213631055015,
                    0.6145422037803824,
                ],
            ),
        ];
        for (mu, x, q) in liens {
            let o = 4 * (mu * champ.v() + site(x, champ.l));
            champ.q[o..o + 4].copy_from_slice(&q);
        }
        let x = x_faces(&champ, 1, 1).unwrap();
        assert_eq!(x[0], 0.0, "la face au seuil doit rester exactement nulle");
        for p in 0..PLANS.len() {
            for s in 0..champ.v() {
                assert_eq!(x[p * champ.v() + s], boucle_face(&champ, s, p, 1)[0]);
            }
        }
        let stats = stats_faces(&champ, 1, Pavage::Tous, false, 1).unwrap();
        assert_eq!(stats.comptes, [9, 0, 0, 0, 0]);
    }

    #[test]
    fn direction_sous_normale_et_debordement_pres_de_moins_identite() {
        let e = f64::from_bits(1); // 2^-1074, plus petit sous-normal positif.
        for lien in [[-1.0, e, e, 0.0], [-1.0, 1e-310, 0.0, 0.0]] {
            let mut champ = ChampHote::froid(2).unwrap();
            champ.q[..4].copy_from_slice(&lien);
            let (phi, sabs, ambigues) =
                flux_coherent_faces(&champ, 1, VarianteFlux::Normale, 1).unwrap();
            assert_eq!(ambigues, 6); // un lien touche deux plaquettes par plan, trois plans.
            for (&p, &s) in phi.iter().zip(&sabs) {
                assert!(p.is_finite() && s.is_finite());
                assert!(p <= 2.0 * PI, "une plaquette : Φ={p}");
                let c = if s == 0.0 { 1.0 } else { p / s };
                assert!((0.0..=1.0).contains(&c), "C={c}");
            }
            let stats = stats_coherent(&champ, 1, Pavage::Tous, VarianteFlux::Normale, 1).unwrap();
            for scalaire in [stats.s_phi, stats.s_phi2, stats.s_coh, stats.phi_max] {
                assert!(scalaire.is_finite());
            }
        }
    }

    #[test]
    fn ambiguite_comptee_independamment_de_la_reference() {
        for (delta, par_plaquette) in [(0.5e-6, 6), (2.0e-6, 0)] {
            let mut champ = ChampHote::froid(2).unwrap();
            let angle = PI - delta;
            champ.q[..4].copy_from_slice(&[angle.cos(), angle.sin(), 0.0, 0.0]);
            // Les six plaquettes incidentes ont Θ=2π−2δ ; toutes les autres
            // sont froides. Chaque plaquette apparaît n² fois dans les faces.
            for n in 1..=3 {
                let attendu = par_plaquette * (n * n) as u64;
                let (_, _, ambigues) =
                    flux_coherent_faces(&champ, n, VarianteFlux::Normale, 1).unwrap();
                assert_eq!(ambigues, attendu, "delta={delta}, n={n}");
                let stats =
                    stats_coherent(&champ, n, Pavage::Tous, VarianteFlux::Normale, 1).unwrap();
                assert_eq!(stats.n_ambigues, attendu, "delta={delta}, n={n}");
            }
        }
    }

    #[test]
    fn pavages_zero_un_deux_sur_fond_non_uniforme() {
        // Intégrateur (M3 r3) : `ChampHote::froid` exige L pair ; champ froid L = 3 construit directement (même contenu).
        let mut q = vec![0.0; 16 * 81];
        for lien in q.chunks_exact_mut(4) {
            lien[0] = 1.0;
        }
        let mut champ = ChampHote { l: 3, q };
        let x_attendus: [f64; 3] = [0.5, -0.5, -0.9];
        for (o, &x) in x_attendus.iter().enumerate() {
            let debut = 4 * site([o; 4], 3); // direction 0 seulement.
            champ.q[debut..debut + 4].copy_from_slice(&[x, (1.0 - x * x).sqrt(), 0.0, 0.0]);
        }
        let comptes_attendus = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [3, 3, 0, 0, 0]];
        for (o, (&x_attendu, &comptes_attendus)) in
            x_attendus.iter().zip(&comptes_attendus).enumerate()
        {
            // Au coin [o;4] : trois faces ont x=x_attendus[o], trois ont x=1.
            let brut = stats_faces(&champ, 1, Pavage::Blocs(o), true, 1).unwrap();
            assert_eq!(brut.n, 6);
            assert_eq!(brut.comptes, comptes_attendus);
            assert!((brut.xmin - x_attendu).abs() <= 1e-15);
            assert!((brut.sx - 3.0 * (1.0 + x_attendu)).abs() <= 1e-14);
            assert_eq!(
                brut,
                stats_faces(&champ, 1, Pavage::Blocs(o), true, 4).unwrap()
            );
            let coh =
                stats_coherent(&champ, 1, Pavage::Blocs(o), VarianteFlux::Normale, 1).unwrap();
            let theta = 2.0 * x_attendu.acos();
            assert_eq!(coh.n, 6);
            assert!((coh.s_phi - 3.0 * theta).abs() <= 1e-13);
            assert!((coh.s_coh - 6.0).abs() <= 1e-14);
            assert_eq!(coh.n_ambigues, 0);
            assert_eq!(
                coh,
                stats_coherent(&champ, 1, Pavage::Blocs(o), VarianteFlux::Normale, 4).unwrap()
            );
        }
    }

    #[test]
    fn tranches_de_sites_par_plan_et_xmin_des_tranches_vides() {
        let tranches = tranches_sites(9);
        assert_eq!(tranches.len(), 12);
        for p in 0..6 {
            assert_eq!(tranches[2 * p], (p, 0..4096));
            assert_eq!(tranches[2 * p + 1], (p, 4096..6561));
        }
        let mut stats = StatsFaces::vide(false);
        stats.fusionner(&stats_x(&[], false));
        stats.fusionner(&stats_x(&[0.5, 1.0], false));
        stats.fusionner(&stats_x(&[], false));
        assert_eq!(stats.xmin, 0.5);
    }
}
