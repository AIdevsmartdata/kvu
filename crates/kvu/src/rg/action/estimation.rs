//! Estimation des couplages par identités exactes de Schwinger–Dyson : pseudo-vraisemblance (PL)
//! et système linéaire (SD), avec jackknife vectoriel.
//!
//! La mémoire de PL est contrôlée par l'appelant (CLI) : la bibliothèque stocke les matrices
//! de Gram G_ℓ (n(n+1)/2 réels par lien), jamais les w par lien.

use super::boucles::{agrafes_lien, Famille, TableFormes, TypeBoucle};
use super::TRANCHE;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::qmul;
use crate::rg::action::L_MIN_ESTIMATION;
use crate::rg::flot::bessel::rapport;
use crate::rg::flot::sd::{beta_sd, RapportBessel, StatsSd};
use crate::rg::lissage::Voisinage;
use rayon::prelude::*;
use serde_json::json;

/// Estimateur demandé.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimateur {
    Pl,
    Sd,
}

impl Estimateur {
    pub fn analyser(t: &str) -> Resultat<Estimateur> {
        match t {
            "pl" => Ok(Estimateur::Pl),
            "sd" => Ok(Estimateur::Sd),
            autre => Err(Erreur::Config(format!(
                "estimateur « {autre} » inconnu (pl|sd)"
            ))),
        }
    }

    pub fn nom(self) -> &'static str {
        match self {
            Estimateur::Pl => "pl",
            Estimateur::Sd => "sd",
        }
    }

    pub fn liste_possibles() -> &'static str {
        "pl,sd"
    }
}

/// Mode de jackknife.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeJackknife {
    Chaines,
    Blocs(usize),
}

impl ModeJackknife {
    pub fn analyser(t: &str) -> Resultat<ModeJackknife> {
        if t == "chaines" {
            return Ok(ModeJackknife::Chaines);
        }
        if let Some(rest) = t.strip_prefix("blocs:") {
            let b: usize = rest.parse().map_err(|_| {
                Erreur::Config(format!("mode jackknife « {t} » : nombre de blocs invalide"))
            })?;
            if b < 2 {
                return Err(Erreur::Config(
                    "nombre de blocs de jackknife ≥ 2 requis".into(),
                ));
            }
            return Ok(ModeJackknife::Blocs(b));
        }
        Err(Erreur::Config(format!(
            "mode jackknife « {t} » inconnu (chaines|blocs:B)"
        )))
    }

    pub fn nom(self) -> String {
        match self {
            ModeJackknife::Chaines => "chaines".to_string(),
            ModeJackknife::Blocs(b) => format!("blocs:{b}"),
        }
    }
}

/// Statistiques d'un groupe (chaîne ou bloc de configurations).
#[derive(Clone, Debug, Default)]
pub struct StatsGroupe {
    /// Somme des w_ℓ^γ sur les liens du groupe, dans l'ordre de la famille.
    pub w: Vec<f64>,
    /// Somme des M_ℓ (produits scalaires des v) sur les liens, ligne par ligne (n×n).
    pub m: Vec<f64>,
    /// Liste des G_ℓ (matrices de Gram des agrafes, triangle supérieur) pour tous les liens,
    /// dans l'ordre des configurations puis des liens.
    pub g: Vec<f64>,
    pub n_liens: usize,
    pub n_configs: usize,
}

impl StatsGroupe {
    pub fn fusionner(&mut self, autre: &StatsGroupe) {
        if self.w.is_empty() && self.m.is_empty() && self.g.is_empty() {
            self.w = vec![0.0; autre.w.len()];
            self.m = vec![0.0; autre.m.len()];
            self.g = Vec::new();
            self.n_liens = 0;
            self.n_configs = 0;
        }
        for (a, b) in self.w.iter_mut().zip(&autre.w) {
            *a += b;
        }
        for (a, b) in self.m.iter_mut().zip(&autre.m) {
            *a += b;
        }
        self.g.extend_from_slice(&autre.g);
        self.n_liens += autre.n_liens;
        self.n_configs += autre.n_configs;
    }
}

// ------------------------------------------------------------------------------------------
// Fonctions auxiliaires d'algèbre
// ------------------------------------------------------------------------------------------

/// Indice du triangle supérieur (i ≤ j) rangé ligne par ligne.
fn idx_tri(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i <= j && j < n);
    i * n + j - i * (i + 1) / 2
}

/// Reconstruit la matrice n×n symétrique à partir du triangle supérieur rangé par `idx_tri`.
fn reconstruire_gram(g_tri: &[f64], n: usize) -> Vec<f64> {
    let mut gram = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let val = g_tri[idx_tri(i, j, n)];
            gram[i * n + j] = val;
            gram[j * n + i] = val;
        }
    }
    gram
}

/// Produit matrice (n×n, ligne par ligne) par vecteur.
fn mat_vec(m: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += m[i * n + j] * v[j];
        }
        out[i] = s;
    }
    out
}

/// Produit scalaire.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Décomposition de Cholesky A = L Lᵀ (A symétrique définie positive).
/// Retourne L rangée ligne par ligne.
fn cholesky(a: &[f64], n: usize) -> Resultat<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !s.is_finite() || s <= 0.0 {
                    return Err(Erreur::Validation(
                        "matrice non définie positive (Cholesky)".into(),
                    ));
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Ok(l)
}

/// Résout L x = b (L triangulaire inférieure).
fn resoudre_inf(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * x[j];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

/// Résout Lᵀ x = b (L triangulaire supérieure, stockée ligne par ligne, on transpose).
fn resoudre_sup(lt: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    // lt est la matrice L stockée ligne par ligne ; Lᵀ est donc la transposée.
    // Nous résolvons Lᵀ x = b en remontant.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= lt[j * n + i] * x[j]; // Lᵀ[i][j] = L[j][i]
        }
        x[i] = s / lt[i * n + i];
    }
    x
}

/// Résout A x = b par Cholesky (A symétrique définie positive).
fn resoudre_cholesky(a: &[f64], b: &[f64], n: usize) -> Resultat<Vec<f64>> {
    let l = cholesky(a, n)?;
    let y = resoudre_inf(&l, b, n);
    Ok(resoudre_sup(&l, &y, n))
}

/// Jacobi cyclique pour matrices symétriques n×n (n ≤ 4). Retourne les valeurs propres
/// triées croissantes.
fn jacobi_valeurs_propres(a: &[f64], n: usize) -> Resultat<Vec<f64>> {
    if n == 0 || a.len() != n * n || a.iter().any(|x| !x.is_finite()) {
        return Err(Erreur::Config("dimension ou matrice invalide".into()));
    }
    let echelle = a.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if echelle == 0.0 {
        return Ok(vec![0.0; n]);
    }
    let mut m: Vec<f64> = a.iter().map(|x| x / echelle).collect();
    const MAX_ROT: usize = 200;
    let mut converge = false;
    for _ in 0..MAX_ROT {
        let mut max_val = 0.0;
        let (mut p, mut q) = (0, 0);
        for i in 0..n {
            for j in (i + 1)..n {
                let v = m[i * n + j].abs();
                if v > max_val {
                    max_val = v;
                    p = i;
                    q = j;
                }
            }
        }
        // La matrice est normalisée : ce seuil est relatif à l'entrée.
        if max_val <= 8.0 * f64::EPSILON {
            converge = true;
            break;
        }
        let app = m[p * n + p];
        let aqq = m[q * n + q];
        let apq = m[p * n + q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = tau.signum() / (tau.abs() + tau.hypot(1.0));
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..n {
            if k != p && k != q {
                let mkp = m[k * n + p];
                let mkq = m[k * n + q];
                m[k * n + p] = c * mkp - s * mkq;
                m[p * n + k] = m[k * n + p];
                m[k * n + q] = s * mkp + c * mkq;
                m[q * n + k] = m[k * n + q];
            }
        }
        m[p * n + p] = app - t * apq;
        m[q * n + q] = aqq + t * apq;
        m[p * n + q] = 0.0;
        m[q * n + p] = 0.0;
    }
    if !converge {
        return Err(Erreur::Validation(
            "Jacobi : convergence non atteinte".into(),
        ));
    }
    let mut vals: Vec<f64> = (0..n).map(|i| m[i * n + i] * echelle).collect();
    if vals.iter().any(|v| !v.is_finite()) {
        return Err(Erreur::Validation(
            "valeurs propres non finies (Jacobi)".into(),
        ));
    }
    vals.sort_by(f64::total_cmp);
    Ok(vals)
}

/// r(k)/k, avec sa limite propre à chaque rapport en zéro.
fn r_sur_k(k: f64, rapport_b: RapportBessel) -> f64 {
    if k < 0.05 {
        let t = k * k;
        match rapport_b {
            RapportBessel::I2SurI1 => 0.25 - t / 96.0 + t * t / 1536.0 - t * t * t / 23040.0,
            RapportBessel::I1SurI0 => {
                0.5 - t / 16.0 + t * t / 96.0 - 11.0 * t * t * t / 6144.0
                    + 19.0 * t * t * t * t / 61440.0
            }
        }
    } else {
        rapport(rapport_b, k) / k
    }
}

fn serie_coeff_hessienne(k: f64, rapport_b: RapportBessel) -> f64 {
    let k2 = k * k;
    match rapport_b {
        RapportBessel::I2SurI1 => -1.0 / 48.0 + k2 / 384.0 - k2 * k2 / 3840.0,
        RapportBessel::I1SurI0 => {
            -1.0 / 8.0 + k2 / 24.0 - 11.0 * k2 * k2 / 1024.0 + 19.0 * k2 * k2 * k2 / 7680.0
        }
    }
}

/// (r′ − r/k)/k². Pour s = I₁/I₀, s′ = 1 − s/k − s².
fn coeff_hessienne(k: f64, rapport_b: RapportBessel) -> f64 {
    if k < 0.05 {
        serie_coeff_hessienne(k, rapport_b)
    } else {
        let r = rapport(rapport_b, k);
        let facteur = match rapport_b {
            RapportBessel::I2SurI1 => 4.0,
            RapportBessel::I1SurI0 => 2.0,
        };
        (1.0 - facteur * r / k - r * r) / (k * k)
    }
}

/// Statistiques d'UNE configuration (déjà bloquée) pour la famille.
pub fn stats_config(
    champ: &ChampHote,
    table: &TableFormes,
    famille: &Famille,
    fils: usize,
) -> Resultat<StatsGroupe> {
    let mut stats = StatsGroupe::default();
    accumuler_config(champ, table, famille, fils, true, &mut stats)?;
    Ok(stats)
}

/// Accumule directement dans un groupe ; aucun G n'est alloué si `conserver_g` est faux.
/// L'appelant réserve une fois la capacité du groupe après sa garde mémoire.
pub fn accumuler_config(
    champ: &ChampHote,
    table: &TableFormes,
    famille: &Famille,
    fils: usize,
    conserver_g: bool,
    stats: &mut StatsGroupe,
) -> Resultat<()> {
    let v = champ
        .l
        .checked_pow(4)
        .filter(|_| champ.l >= L_MIN_ESTIMATION)
        .ok_or_else(|| Erreur::Config("L invalide pour l'estimation (L ≥ 4 requis)".into()))?;
    let longueur = v
        .checked_mul(16)
        .ok_or_else(|| Erreur::Config("volume du champ trop grand".into()))?;
    if champ.q.len() != longueur || champ.q.iter().any(|x| !x.is_finite()) {
        return Err(Erreur::Config(
            "longueur ou composante du champ invalide".into(),
        ));
    }
    if fils == 0 {
        return Err(Erreur::Config("le nombre de fils doit être ≥ 1".into()));
    }
    let n = famille.n();
    if stats.w.is_empty() && stats.m.is_empty() {
        stats.w = vec![0.0; n];
        stats.m = vec![0.0; n * n];
    }
    if stats.w.len() != n || stats.m.len() != n * n {
        return Err(Erreur::Config("dimensions du groupe incohérentes".into()));
    }
    let vois = Voisinage::nouveau(champ.l);
    let n_liens = 4 * v;
    let n_tri = n * (n + 1) / 2;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(fils)
        .build()
        .map_err(|e| Erreur::Config(format!("création du pool de fils impossible : {e}")))?;

    // Même découpage et même ordre de réduction avec ou sans stockage de G.
    let calculer = |tranche: usize, mut gram: Option<&mut [f64]>| {
        let debut = tranche * TRANCHE;
        let fin = (debut + TRANCHE).min(n_liens);
        let mut w = vec![0.0; n];
        let mut m = vec![0.0; n * n];
        for li in debut..fin {
            let (mu, site) = (li / v, li % v);
            let u = champ.lien(li);
            let agrafes = agrafes_lien(champ, &vois, table, famille, mu, site);
            let mut v3 = vec![[0.0; 3]; n];
            for (i, &a) in agrafes.iter().enumerate() {
                let qu = qmul(u, a);
                w[i] += qu[0];
                v3[i] = [qu[1], qu[2], qu[3]];
            }
            if let Some(g) = gram.as_deref_mut() {
                for i in 0..n {
                    for j in i..n {
                        g[(li - debut) * n_tri + idx_tri(i, j, n)] = dot(&agrafes[i], &agrafes[j]);
                    }
                }
            }
            for i in 0..n {
                for j in 0..n {
                    m[i * n + j] += dot(&v3[i], &v3[j]);
                }
            }
        }
        (w, m)
    };
    let tranches: Vec<(Vec<f64>, Vec<f64>)> = if conserver_g {
        let debut = stats.g.len();
        let ajout = n_liens
            .checked_mul(n_tri)
            .ok_or_else(|| Erreur::Config("taille des Gram trop grande".into()))?;
        stats
            .g
            .try_reserve_exact(ajout)
            .map_err(|_| Erreur::Ressources("mémoire insuffisante pour PL".into()))?;
        stats.g.resize(debut + ajout, 0.0);
        pool.install(|| {
            stats.g[debut..]
                .par_chunks_mut(TRANCHE * n_tri)
                .enumerate()
                .map(|(i, g)| calculer(i, Some(g)))
                .collect()
        })
    } else {
        pool.install(|| {
            (0..n_liens.div_ceil(TRANCHE))
                .into_par_iter()
                .map(|i| calculer(i, None))
                .collect()
        })
    };
    for (w, m) in tranches {
        for (total, x) in stats.w.iter_mut().zip(w) {
            *total += x;
        }
        for (total, x) in stats.m.iter_mut().zip(m) {
            *total += x;
        }
    }
    stats.n_liens += n_liens;
    stats.n_configs += 1;
    Ok(())
}

/// Solution de l'estimateur PL.
#[derive(Clone, Debug)]
pub struct SolutionPl {
    pub beta: Vec<f64>,
    pub iterations: usize,
    pub norme_gradient: f64,
    pub valeurs_propres_information: Vec<f64>,
}

/// Solution de l'estimateur SD.
#[derive(Clone, Debug)]
pub struct SolutionSd {
    pub beta: Vec<f64>,
    pub valeurs_propres_m: Vec<f64>,
}

/// Somme des w de tous les groupes (vecteur de longueur n).
fn somme_w(groupes: &[&StatsGroupe], n: usize) -> Vec<f64> {
    let mut s = vec![0.0; n];
    for g in groupes {
        for (si, wi) in s.iter_mut().zip(&g.w) {
            *si += wi;
        }
    }
    s
}

/// Calcule la liste des contributions (grad_neg, info) pour un lien, pour Newton PL.
fn contributions_lien(
    g_tri: &[f64],
    n: usize,
    beta: &[f64],
    rapport_b: RapportBessel,
) -> (Vec<f64>, Vec<f64>) {
    let gram = reconstruire_gram(g_tri, n);
    let g_ell = mat_vec(&gram, beta, n);
    let k2 = dot(beta, &g_ell);
    let k = k2.max(0.0).sqrt();
    let rsk = r_sur_k(k, rapport_b);
    let ch = coeff_hessienne(k, rapport_b);
    // contribution au gradient : -rsk * g_ell
    let grad_neg: Vec<f64> = g_ell.iter().map(|&x| -rsk * x).collect();
    // contribution à l'information (I = -H) : rsk * G + ch * g g^T
    let mut info = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            info[i * n + j] = rsk * gram[i * n + j] + ch * g_ell[i] * g_ell[j];
        }
    }
    (grad_neg, info)
}

/// Somme des contributions sur tous les liens de tous les groupes, parallélisée par tranches
/// de taille TRANCHE, dans l'ordre.
fn sommes_newton(
    groupes: &[&StatsGroupe],
    n: usize,
    beta: &[f64],
    rapport_b: RapportBessel,
    fils: usize,
) -> Resultat<(Vec<f64>, Vec<f64>)> {
    let n_tri = n * (n + 1) / 2;
    if n == 0 || fils == 0 {
        return Err(Erreur::Config(
            "dimension ou nombre de fils nul pour Newton".into(),
        ));
    }
    let mut taches = Vec::new();
    for (gi, g) in groupes.iter().enumerate() {
        if g.w.len() != n || g.g.len() != g.n_liens * n_tri {
            return Err(Erreur::Config(
                "StatsGroupe mal formé pour Newton (G requis)".into(),
            ));
        }
        for debut in (0..g.n_liens).step_by(TRANCHE) {
            taches.push((gi, debut, (debut + TRANCHE).min(g.n_liens)));
        }
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(fils)
        .build()
        .map_err(|e| Erreur::Config(format!("création du pool de fils impossible : {e}")))?;

    let tranches: Vec<(Vec<f64>, Vec<f64>)> = pool.install(|| {
        taches
            .par_iter()
            .map(|&(gi, debut, fin)| {
                let mut grad_local = vec![0.0; n];
                let mut info_local = vec![0.0; n * n];
                for li in debut..fin {
                    let g = groupes[gi];
                    let base = li * n_tri;
                    let g_slice = &g.g[base..base + n_tri];
                    let (gn, info) = contributions_lien(g_slice, n, beta, rapport_b);
                    for i in 0..n {
                        grad_local[i] += gn[i];
                    }
                    for i in 0..n * n {
                        info_local[i] += info[i];
                    }
                }
                (grad_local, info_local)
            })
            .collect()
    });

    let mut grad_total = somme_w(groupes, n); // terme +W
    let mut info_total = vec![0.0; n * n];
    for (gn, info) in tranches {
        for i in 0..n {
            grad_total[i] += gn[i];
        }
        for i in 0..n * n {
            info_total[i] += info[i];
        }
    }
    Ok((grad_total, info_total))
}

/// Résout le système SD M β = b par Cholesky, avec vérification du conditionnement.
pub fn resoudre_sd(groupes: &[&StatsGroupe], n: usize) -> Resultat<SolutionSd> {
    if groupes.is_empty() || n == 0 {
        return Err(Erreur::Config(
            "aucun groupe ou dimension nulle pour SD".into(),
        ));
    }
    let mut m_total = vec![0.0; n * n];
    let mut b = vec![0.0; n];
    for g in groupes {
        if g.w.len() != n || g.m.len() != n * n {
            return Err(Erreur::Config("StatsGroupe mal formé pour SD".into()));
        }
        for (mt, mi) in m_total.iter_mut().zip(&g.m) {
            *mt += mi;
        }
        for (bi, wi) in b.iter_mut().zip(&g.w) {
            *bi += 3.0 * wi;
        }
    }
    let echelle = m_total.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if echelle == 0.0 || !echelle.is_finite() || b.iter().any(|x| !x.is_finite()) {
        return Err(Erreur::Validation(
            "système SD singulier ou non fini".into(),
        ));
    }
    for x in &mut m_total {
        *x /= echelle;
    }
    for x in &mut b {
        *x /= echelle;
    }
    let vals = jacobi_valeurs_propres(&m_total, n)?;
    let (min, max) = (vals[0], vals[n - 1]);
    if min <= 0.0 || min <= 1e-12 * max {
        return Err(Erreur::Validation(format!(
            "système SD singulier ou mal conditionné (λ_min = {min:.3e}, λ_max = {max:.3e})"
        )));
    }
    let beta = resoudre_cholesky(&m_total, &b, n)?;
    Ok(SolutionSd {
        beta,
        valeurs_propres_m: vals.into_iter().map(|x| x * echelle).collect(),
    })
}

/// Résout PL par Newton protégé, avec un rapport de Bessel choisi (mutant éventuel).
pub fn resoudre_pl_avec(
    groupes: &[&StatsGroupe],
    n: usize,
    depart: &[f64],
    rapport_b: RapportBessel,
    fils: usize,
) -> Resultat<SolutionPl> {
    if groupes.is_empty() {
        return Err(Erreur::Config("aucun groupe pour PL".into()));
    }
    if n == 0 || depart.len() != n || depart.iter().any(|x| !x.is_finite()) {
        return Err(Erreur::Config("départ PL invalide".into()));
    }
    const BETA_MAX: f64 = 1e4;
    if depart.iter().any(|b| b.abs() > BETA_MAX) {
        return Err(Erreur::Validation(
            "maximum à l'infini : configurations ordonnées".into(),
        ));
    }
    let mut beta = depart.to_vec();
    let mut iterations = 0usize;
    const MAX_ITERATIONS: usize = 100;
    const SEUIL_PAS: f64 = 1e-12;
    // Intégrateur (M2 r2A) : powi n'est pas const ; 2^-40 écrit exactement.
    const T_MIN: f64 = 1.0 / 1_099_511_627_776.0;

    loop {
        iterations += 1;
        if iterations > MAX_ITERATIONS {
            return Err(Erreur::Validation(
                "Newton PL : 100 itérations dépassées".into(),
            ));
        }
        let (grad, info) = sommes_newton(groupes, n, &beta, rapport_b, fils)?;
        if !grad.iter().all(|x| x.is_finite()) || !info.iter().all(|x| x.is_finite()) {
            return Err(Erreur::Validation("Newton PL : valeur non finie".into()));
        }
        // résoudre I d = grad (I = -H) par Cholesky
        let d = match resoudre_cholesky(&info, &grad, n) {
            Ok(d) => d,
            Err(_) => return Err(Erreur::Validation("hessienne non définie positive".into())),
        };
        if d.iter().any(|x| !x.is_finite()) {
            return Err(Erreur::Validation("Newton PL : direction non finie".into()));
        }
        // recherche linéaire sur la dérivée directionnelle
        // Intégrateur (M2 r3, objection P1 d'Astra, tour 2) : un pas tronqué par la borne |β|∞ ≤ 1e4 ne peut pas conclure à la convergence
        // (sinon un β arrêté artificiellement à la borne était rendu « ok » alors que la racine est au-delà).
        let mut borne_rencontree = false;
        let mut t = 1.0;
        loop {
            let beta_essai: Vec<f64> = beta.iter().zip(&d).map(|(b, di)| b + t * di).collect();
            if beta_essai
                .iter()
                .any(|b| !b.is_finite() || b.abs() > BETA_MAX)
            {
                borne_rencontree = true;
                if t <= T_MIN {
                    return Err(Erreur::Validation(
                        "maximum à l'infini : candidat Newton excessif".into(),
                    ));
                }
                t /= 2.0;
                continue;
            }
            let (grad_essai, _) = sommes_newton(groupes, n, &beta_essai, rapport_b, fils)?;
            let derivee = dot(&grad_essai, &d);
            if !derivee.is_finite() || grad_essai.iter().any(|x| !x.is_finite()) {
                if t <= T_MIN {
                    return Err(Erreur::Validation(
                        "Newton PL : gradient candidat non fini".into(),
                    ));
                }
                t /= 2.0;
                continue;
            }
            if derivee >= 0.0 || t <= T_MIN {
                break;
            }
            t /= 2.0;
        }
        let pas: Vec<f64> = d.iter().map(|di| t * di).collect();
        let norme_pas = pas.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        for (b, p) in beta.iter_mut().zip(&pas) {
            *b += p;
        }
        let norme_beta = beta.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        if norme_beta > BETA_MAX {
            return Err(Erreur::Validation(
                "maximum à l'infini : configurations ordonnées".into(),
            ));
        }
        if !beta.iter().all(|x| x.is_finite()) {
            return Err(Erreur::Validation("Newton PL : β non fini".into()));
        }
        if norme_pas <= SEUIL_PAS * f64::max(1.0, norme_beta) {
            if borne_rencontree {
                return Err(Erreur::Validation(
                    "maximum à l'infini : la racine est au-delà de |β|∞ = 1e4 (pas tronqué par la borne)".into(),
                ));
            }
            // convergence : calculer l'information de Fisher à la solution
            // Intégrateur (M2 r2A) : r1 prenait l'information pour le gradient ((_, grad_fin) = (grad, info)).
            let (grad_fin, info_fin) = sommes_newton(groupes, n, &beta, rapport_b, fils)?;
            let vals_info = jacobi_valeurs_propres(&info_fin, n)?;
            let norme_gradient_fin = grad_fin.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
            return Ok(SolutionPl {
                beta,
                iterations,
                norme_gradient: norme_gradient_fin,
                valeurs_propres_information: vals_info,
            });
        }
    }
}

/// Résout PL avec le rapport SU(2) standard.
pub fn resoudre_pl(
    groupes: &[&StatsGroupe],
    n: usize,
    depart: &[f64],
    fils: usize,
) -> Resultat<SolutionPl> {
    resoudre_pl_avec(groupes, n, depart, RapportBessel::I2SurI1, fils)
}

/// Norme R4 : N4 = Σ_{γ ∈ F} n_lien(γ) (|γ| − 1) |β_γ|.
pub fn norme_r4(famille: &Famille, beta: &[f64]) -> f64 {
    famille
        .types()
        .iter()
        .enumerate()
        .map(|(i, ty)| ty.poids_r4() * beta[i].abs())
        .sum()
}

/// Critère P+R : c_PR = |β_P| + 5 |β_R| (β_R = 0 si R ∉ F).
pub fn critere_pr(famille: &Famille, beta: &[f64]) -> f64 {
    let mut somme = 0.0;
    if let Some(i) = famille.index(TypeBoucle::P) {
        somme += beta[i].abs();
    }
    if let Some(i) = famille.index(TypeBoucle::R) {
        somme += 5.0 * beta[i].abs();
    }
    somme
}

/// Valeur avec erreur jackknife.
#[derive(Clone, Debug, PartialEq)]
pub struct Valeur {
    pub valeur: f64,
    pub erreur: f64,
    pub replicats: Vec<f64>,
}

/// Jackknife vectoriel : même formule que `jackknife_groupes`.
///
/// `valeur` est la valeur centrale (sur tous les groupes), `replicats` contient les
/// estimations sans chaque groupe (dans l'ordre des groupes omis).
pub fn jackknife_vectoriel(valeur: &[f64], replicats: &[Vec<f64>]) -> Vec<Valeur> {
    let g = replicats.len();
    let n = valeur.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let vals: Vec<f64> = replicats.iter().map(|r| r[i]).collect();
        let moyenne_replicats = vals.iter().sum::<f64>() / g as f64;
        let somme = vals
            .iter()
            .map(|x| (x - moyenne_replicats).powi(2))
            .sum::<f64>();
        let erreur = (((g - 1) as f64 / g as f64) * somme).sqrt();
        out.push(Valeur {
            valeur: valeur[i],
            erreur,
            replicats: vals,
        });
    }
    out
}

/// Différences PL − SD sur les mêmes suppressions de groupe, puis jackknife.
pub fn differences_appariees(pl: &[Valeur], sd: &[Valeur]) -> Resultat<Vec<Valeur>> {
    if pl.is_empty() || pl.len() != sd.len() {
        return Err(Erreur::Config("couplages PL/SD incompatibles".into()));
    }
    let g = pl[0].replicats.len();
    if g < 2 || pl.iter().chain(sd).any(|v| v.replicats.len() != g) {
        return Err(Erreur::Config("réplicats PL/SD non appariables".into()));
    }
    let centrale: Vec<f64> = pl
        .iter()
        .zip(sd)
        .map(|(p, s)| p.valeur - s.valeur)
        .collect();
    let replicats: Vec<Vec<f64>> = (0..g)
        .map(|i| {
            pl.iter()
                .zip(sd)
                .map(|(p, s)| p.replicats[i] - s.replicats[i])
                .collect()
        })
        .collect();
    Ok(jackknife_vectoriel(&centrale, &replicats))
}

/// La formule delete-group impose des tailles comparables (rapport maximal 2).
pub fn verifier_tailles_groupes(tailles: &[usize]) -> Resultat<()> {
    let min = tailles.iter().copied().min().unwrap_or(0);
    let max = tailles.iter().copied().max().unwrap_or(0);
    if tailles.len() < 2 || min == 0 || max > min.saturating_mul(2) {
        return Err(Erreur::Config(
            "jackknife : groupes vides ou tailles extrêmes de rapport > 2".into(),
        ));
    }
    Ok(())
}

/// Matrice de corrélation (n×n, ligne par ligne) des couplages à partir des réplicats.
fn correlations(replicats: &[Vec<f64>], n: usize) -> Vec<f64> {
    let g = replicats.len();
    let mut corr = vec![0.0; n * n];
    if g < 2 {
        return corr;
    }
    let moyennes: Vec<f64> = (0..n)
        .map(|i| replicats.iter().map(|r| r[i]).sum::<f64>() / g as f64)
        .collect();
    for i in 0..n {
        for j in 0..n {
            let num = (0..g)
                .map(|k| (replicats[k][i] - moyennes[i]) * (replicats[k][j] - moyennes[j]))
                .sum::<f64>();
            let den_i = (0..g)
                .map(|k| (replicats[k][i] - moyennes[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            let den_j = (0..g)
                .map(|k| (replicats[k][j] - moyennes[j]).powi(2))
                .sum::<f64>()
                .sqrt();
            corr[i * n + j] = if den_i > 0.0 && den_j > 0.0 {
                num / (den_i * den_j)
            } else {
                0.0
            };
        }
    }
    corr
}

/// Résultat complet d'un estimateur.
#[derive(Clone, Debug)]
pub struct ResultatEstimateur {
    pub couplages: Vec<Valeur>,
    pub r4: Valeur,
    pub pr: Valeur,
    pub correlations: Vec<f64>,
    pub details: serde_json::Value,
}

/// Estimation complète d'une liste de groupes (valeur centrale + jackknife par groupe).
pub fn estimer(
    groupes: &[StatsGroupe],
    famille: &Famille,
    quoi: Estimateur,
    fils: usize,
) -> Resultat<ResultatEstimateur> {
    let n = famille.n();
    if n == 0 || groupes.is_empty() || groupes.len() < 2 {
        return Err(Erreur::Config(
            "estimation requiert au moins 2 groupes".into(),
        ));
    }
    if fils == 0 {
        return Err(Erreur::Config("le nombre de fils doit être ≥ 1".into()));
    }
    for g in groupes {
        if g.w.len() != n || g.m.len() != n * n {
            return Err(Erreur::Config("StatsGroupe incohérent".into()));
        }
    }

    verifier_tailles_groupes(&groupes.iter().map(|g| g.n_configs).collect::<Vec<_>>())?;

    let refs: Vec<&StatsGroupe> = groupes.iter().collect();

    // ---- Valeur centrale ----
    let (beta_central, details_central) = match quoi {
        Estimateur::Sd => {
            let sol = resoudre_sd(&refs, n)?;
            let vals = sol.valeurs_propres_m.clone();
            let cond = vals[n - 1] / vals[0];
            (
                sol.beta,
                json!({
                    "valeurs_propres_m": vals,
                    "conditionnement": cond,
                }),
            )
        }
        Estimateur::Pl => {
            // départ : SD si possible, sinon beta_sd(P) des données réduites à P
            let depart = match resoudre_sd(&refs, n) {
                Ok(sol) => sol.beta,
                Err(_) => {
                    // données réduites à P : w_P total et k_P par lien
                    let idx_p = famille.index(TypeBoucle::P).ok_or_else(|| {
                        Erreur::Config("la famille doit contenir P pour PL".into())
                    })?;
                    let mut somme_w_p = 0.0;
                    let mut ks = Vec::new();
                    let n_tri = n * (n + 1) / 2;
                    for g in groupes {
                        somme_w_p += g.w[idx_p];
                        let nb_liens = g.g.len() / n_tri;
                        for li in 0..nb_liens {
                            let base = li * n_tri;
                            let g_pp = g.g[base + idx_tri(idx_p, idx_p, n)];
                            ks.push(g_pp.sqrt());
                        }
                    }
                    let stats_p = StatsSd {
                        somme_w: somme_w_p,
                        ks,
                        n_configs: groupes.iter().map(|g| g.n_configs).sum(),
                    };
                    let beta_p = beta_sd(
                        &stats_p,
                        RapportBessel::I2SurI1,
                        crate::rg::flot::sd::ModeRapport::Exact,
                    )
                    .ok_or_else(|| {
                        Erreur::Validation(
                            "beta_sd sur données réduites à P échoue (PL impossible)".into(),
                        )
                    })?;
                    let mut depart = vec![0.0; n];
                    depart[idx_p] = beta_p;
                    depart
                }
            };
            let sol = resoudre_pl(&refs, n, &depart, fils)?;
            let vals = sol.valeurs_propres_information.clone();
            let cond = vals[n - 1] / vals[0];
            (
                sol.beta,
                json!({
                    "iterations": sol.iterations,
                    "norme_gradient": sol.norme_gradient,
                    "valeurs_propres_information": vals,
                    "conditionnement": cond,
                }),
            )
        }
    };

    // ---- Réplicats ----
    let mut betas_replicats: Vec<Vec<f64>> = Vec::with_capacity(groupes.len());
    for exclu in 0..groupes.len() {
        let sous_groupes: Vec<&StatsGroupe> = refs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != exclu)
            .map(|(_, g)| *g)
            .collect();
        let (beta_rep, _) = match quoi {
            Estimateur::Sd => {
                let sol = resoudre_sd(&sous_groupes, n)?;
                (sol.beta, serde_json::Value::Null)
            }
            Estimateur::Pl => {
                let sol = resoudre_pl(&sous_groupes, n, &beta_central, fils)?;
                (sol.beta, serde_json::Value::Null)
            }
        };
        betas_replicats.push(beta_rep);
    }

    let couplages = jackknife_vectoriel(&beta_central, &betas_replicats);

    // ---- Quantités dérivées ----
    let r4_central = norme_r4(famille, &beta_central);
    let pr_central = critere_pr(famille, &beta_central);
    let mut r4_replicats = Vec::with_capacity(groupes.len());
    let mut pr_replicats = Vec::with_capacity(groupes.len());
    for beta_rep in &betas_replicats {
        r4_replicats.push(vec![norme_r4(famille, beta_rep)]);
        pr_replicats.push(vec![critere_pr(famille, beta_rep)]);
    }
    let r4 = jackknife_vectoriel(&[r4_central], &r4_replicats)
        .into_iter()
        .next()
        .ok_or_else(|| Erreur::Validation("jackknife dérivé vide".into()))?;
    let pr = jackknife_vectoriel(&[pr_central], &pr_replicats)
        .into_iter()
        .next()
        .ok_or_else(|| Erreur::Validation("jackknife dérivé vide".into()))?;

    let correlations = correlations(&betas_replicats, n);

    Ok(ResultatEstimateur {
        couplages,
        r4,
        pr,
        correlations,
        details: details_central,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heritage::su2_hb::CpuSu2;
    use crate::rg::flot::jackknife::jackknife_groupes;

    fn champ_haar(l: usize, graine: u64) -> ChampHote {
        let mut gen = CpuSu2::new(l, graine).unwrap();
        gen.init(true);
        ChampHote { l, q: gen.flat() }
    }

    #[test]
    fn indice_triangle_superieur_couvre_tout() {
        for n in 1..=5 {
            let total = n * (n + 1) / 2;
            let mut vus = vec![false; total];
            let mut attendu = 0usize;
            for i in 0..n {
                for j in i..n {
                    let idx = idx_tri(i, j, n);
                    assert_eq!(idx, attendu, "n={n} i={i} j={j}");
                    assert!(!vus[idx], "indice {idx} dupliqué");
                    vus[idx] = true;
                    attendu += 1;
                }
            }
            assert!(vus.iter().all(|&b| b));
        }
    }

    #[test]
    fn norme_r4_et_critere_pr() {
        let famille = Famille::analyser("P,R,C,T").unwrap();
        let beta = [0.1, 0.01, 0.001, 0.002];
        assert!((norme_r4(&famille, &beta) - 3.3).abs() < 1e-15);
        assert!((critere_pr(&famille, &beta) - 0.15).abs() < 1e-15);
        let beta_neg = [-0.1, -0.01, -0.001, -0.002];
        assert!((norme_r4(&famille, &beta_neg) - 3.3).abs() < 1e-15);
        assert!((critere_pr(&famille, &beta_neg) - 0.15).abs() < 1e-15);

        let famille_pc = Famille::analyser("P,C").unwrap();
        let beta_pc = [0.3, 0.004];
        let attendu_n4 = 18.0 * 0.3 + 360.0 * 0.004;
        assert!((norme_r4(&famille_pc, &beta_pc) - attendu_n4).abs() < 1e-15);
        assert!((critere_pr(&famille_pc, &beta_pc) - 0.3).abs() < 1e-15);
    }

    #[test]
    fn jackknife_vectoriel_egale_jackknife_groupes() {
        // Exemple synthétique : 5 groupes bruts, vecteurs de dimension 3.
        let groupes: Vec<[f64; 3]> = vec![
            [5.1, -0.9, 3.2],
            [5.3, -1.1, 2.9],
            [5.0, -1.0, 3.1],
            [5.4, -1.2, 2.8],
            [5.2, -0.8, 3.0],
        ];
        // Estimateur : moyenne de la composante i sur les groupes retenus.
        let moyenne_composante =
            |i: usize, g: &[&[f64; 3]]| Some(g.iter().map(|x| x[i]).sum::<f64>() / g.len() as f64);
        // Valeur complète calculée à la main : moyenne de chaque composante sur tous les groupes.
        let valeur_complete: Vec<f64> = (0..3)
            .map(|i| groupes.iter().map(|g| g[i]).sum::<f64>() / groupes.len() as f64)
            .collect();
        // Réplicats « sans g » calculés à la main.
        let mut replicats: Vec<Vec<f64>> = Vec::with_capacity(groupes.len());
        for exclu in 0..groupes.len() {
            let retenus: Vec<&[f64; 3]> = groupes
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != exclu)
                .map(|(_, g)| g)
                .collect();
            let mut rep = Vec::with_capacity(3);
            for i in 0..3 {
                rep.push(moyenne_composante(i, &retenus).unwrap());
            }
            replicats.push(rep);
        }

        let res = jackknife_vectoriel(&valeur_complete, &replicats);
        assert_eq!(res.len(), 3);
        for i in 0..3 {
            let reference = jackknife_groupes(&groupes, |g| moyenne_composante(i, g)).unwrap();
            assert!(
                (res[i].valeur - reference.valeur).abs() <= 1e-15,
                "composante {i} : valeur ({} vs {})",
                res[i].valeur,
                reference.valeur
            );
            assert!(
                (res[i].erreur - reference.erreur).abs() <= 1e-15,
                "composante {i} : erreur"
            );
            assert_eq!(
                res[i].replicats,
                replicats.iter().map(|r| r[i]).collect::<Vec<_>>()
            );
            assert_eq!(
                res[i].replicats, reference.replicats,
                "composante {i} : réplicats"
            );
        }
    }

    #[test]
    fn newton_pl_sur_donnees_synthetiques() {
        // Un seul lien, n=1, solution connue β0 = 1.5, G = 2.0.
        let n = 1;
        let beta0 = 1.5;
        let g = 2.0_f64;
        let k0 = beta0 * g.sqrt();
        let r = rapport(RapportBessel::I2SurI1, k0);
        let w = r * g.sqrt() * beta0.signum(); // ∇F = w - r(k)/k * g*β = 0
        let stats = StatsGroupe {
            w: vec![w],
            m: vec![0.0],
            g: vec![g],
            n_liens: 1,
            n_configs: 1,
        };
        let depart = [1.0];
        let groupes = [&stats];
        let sol = resoudre_pl(&groupes, n, &depart, 1).unwrap();
        assert!(
            (sol.beta[0] - beta0).abs() <= 1e-12,
            "β = {} au lieu de {beta0}",
            sol.beta[0]
        );
    }

    #[test]
    fn stats_config_cumul_exact_sur_champ_haar() {
        let champ = champ_haar(4, 123456);
        let table = TableFormes::construire().unwrap();
        let famille = Famille::analyser("P,R,C,T").unwrap();
        let n = famille.n();
        let stats = stats_config(&champ, &table, &famille, 1).unwrap();

        // Calcul séquentiel de référence.
        let vois = Voisinage::nouveau(champ.l);
        let v = champ.v();
        let mut w_ref = vec![0.0; n];
        let mut m_ref = vec![0.0; n * n];
        let mut g_ref = Vec::new();
        let n_tri = n * (n + 1) / 2;
        for mu in 0..4 {
            for s in 0..v {
                let li = mu * v + s;
                let u = champ.lien(li);
                let agrafes = agrafes_lien(&champ, &vois, &table, &famille, mu, s);
                let mut g_tri = vec![0.0; n_tri];
                let mut v3 = Vec::with_capacity(n);
                for (i, &a) in agrafes.iter().enumerate() {
                    let qu = qmul(u, a);
                    w_ref[i] += qu[0];
                    v3.push([qu[1], qu[2], qu[3]]);
                }
                for i in 0..n {
                    for j in i..n {
                        let dot_a = agrafes[i][0] * agrafes[j][0]
                            + agrafes[i][1] * agrafes[j][1]
                            + agrafes[i][2] * agrafes[j][2]
                            + agrafes[i][3] * agrafes[j][3];
                        g_tri[idx_tri(i, j, n)] = dot_a;
                    }
                }
                for i in 0..n {
                    for j in 0..n {
                        m_ref[i * n + j] +=
                            v3[i][0] * v3[j][0] + v3[i][1] * v3[j][1] + v3[i][2] * v3[j][2];
                    }
                }
                g_ref.extend_from_slice(&g_tri);
            }
        }

        #[allow(clippy::needless_range_loop)] // intégrateur : test, indices utiles au message
        for i in 0..n {
            assert!(
                (stats.w[i] - w_ref[i]).abs() <= 1e-12,
                "w[{i}] : {} vs {}",
                stats.w[i],
                w_ref[i]
            );
        }
        #[allow(clippy::needless_range_loop)] // intégrateur : test, indices utiles au message
        for i in 0..(n * n) {
            assert!(
                (stats.m[i] - m_ref[i]).abs() <= 1e-12,
                "m[{i}] : {} vs {}",
                stats.m[i],
                m_ref[i]
            );
        }
        assert_eq!(stats.g.len(), g_ref.len());
        for (idx, (&a, &b)) in stats.g.iter().zip(&g_ref).enumerate() {
            assert!((a - b).abs() <= 1e-12, "g[{idx}] : {a} vs {b}");
        }
    }

    #[test]
    fn determinisme_stats_config_et_estimer() {
        let champ = champ_haar(4, 654321);
        let table = TableFormes::construire().unwrap();
        let famille = Famille::analyser("P,R,C,T").unwrap();

        let s1 = stats_config(&champ, &table, &famille, 1).unwrap();
        let s4 = stats_config(&champ, &table, &famille, 4).unwrap();
        assert_eq!(s1.w, s4.w);
        assert_eq!(s1.m, s4.m);
        assert_eq!(s1.g, s4.g);
        assert_eq!(s1.n_liens, s4.n_liens);
        assert_eq!(s1.n_configs, s4.n_configs);

        // Deux configurations pour l'estimateur (deux groupes).
        let champ2 = champ_haar(4, 111111);
        let s2 = stats_config(&champ2, &table, &famille, 1).unwrap();
        let groupes = vec![s1, s2];

        let res1_pl = estimer(&groupes, &famille, Estimateur::Pl, 1).unwrap();
        let res4_pl = estimer(&groupes, &famille, Estimateur::Pl, 4).unwrap();
        for i in 0..famille.n() {
            assert_eq!(res1_pl.couplages[i].valeur, res4_pl.couplages[i].valeur);
            assert!((res1_pl.couplages[i].erreur - res4_pl.couplages[i].erreur).abs() <= 1e-15);
            assert_eq!(
                res1_pl.couplages[i].replicats,
                res4_pl.couplages[i].replicats
            );
        }
        assert_eq!(res1_pl.r4.valeur, res4_pl.r4.valeur);
        assert_eq!(res1_pl.pr.valeur, res4_pl.pr.valeur);
        assert_eq!(res1_pl.correlations, res4_pl.correlations);
        assert_eq!(res1_pl.details, res4_pl.details);

        let res1_sd = estimer(&groupes, &famille, Estimateur::Sd, 1).unwrap();
        let res4_sd = estimer(&groupes, &famille, Estimateur::Sd, 4).unwrap();
        for i in 0..famille.n() {
            assert_eq!(res1_sd.couplages[i].valeur, res4_sd.couplages[i].valeur);
            assert!((res1_sd.couplages[i].erreur - res4_sd.couplages[i].erreur).abs() <= 1e-15);
        }
        assert_eq!(res1_sd.correlations, res4_sd.correlations);
        assert_eq!(res1_sd.details, res4_sd.details);
    }

    // Référence indépendante : Z = somme des Bessel (coefficients de k^(2m)),
    // (r′−r/k)/k² = 4 (Z_tt/Z − (Z_t/Z)²), t=k². Pas d'annulation en k=0.
    fn coefficient_reference_bessel(k: f64, rapport_b: RapportBessel) -> f64 {
        let t = k * k;
        let (mut z, mut z1, mut z2) = (1.0, 0.0, 0.0);
        let mut coefficient = 1.0;
        for m in 1..=40 {
            let decalage = if rapport_b == RapportBessel::I2SurI1 {
                1
            } else {
                0
            };
            coefficient /= 4.0 * f64::from(m) * f64::from(m + decalage);
            z += coefficient * t.powi(m);
            z1 += f64::from(m) * coefficient * t.powi(m - 1);
            if m >= 2 {
                z2 += f64::from(m * (m - 1)) * coefficient * t.powi(m - 2);
            }
        }
        4.0 * (z2 / z - (z1 / z).powi(2))
    }

    #[test]
    fn serie_coefficient_hessienne() {
        for rapport_b in [RapportBessel::I2SurI1, RapportBessel::I1SurI0] {
            for k in [0.0, 0.01, 0.05] {
                let reference = coefficient_reference_bessel(k, rapport_b);
                // À 0,05, on teste explicitement la SÉRIE, même si la production
                // choisit la branche directe. Les deux sont comparées à l'oracle.
                for resultat in [
                    serie_coeff_hessienne(k, rapport_b),
                    coeff_hessienne(k, rapport_b),
                ] {
                    assert!(
                        (resultat - reference).abs() / reference.abs() <= 1e-9,
                        "k={k}, {rapport_b:?} : {resultat:.15e} vs {reference:.15e}"
                    );
                }
            }
            let avant = coeff_hessienne(0.05 - 1e-12, rapport_b);
            let apres = coeff_hessienne(0.05, rapport_b);
            assert!((avant - apres).abs() <= 1e-12);
        }
        assert_eq!(r_sur_k(0.0, RapportBessel::I1SurI0), 0.5);
        assert_eq!(r_sur_k(1e-310, RapportBessel::I1SurI0), 0.5);
        assert_eq!(r_sur_k(1e-310, RapportBessel::I2SurI1), 0.25);
        assert_eq!(coeff_hessienne(0.0, RapportBessel::I1SurI0), -0.125);
    }

    #[test]
    fn mutant_bessel_information_positive_et_solution_convergee() {
        let s = rapport(RapportBessel::I1SurI0, 1.0);
        let (_, info) = contributions_lien(&[1.0], 1, &[1.0], RapportBessel::I1SurI0);
        assert!((info[0] - (1.0 - s - s * s)).abs() < 1e-14);
        assert!((info[0] - 0.354_346_032_450_356).abs() < 1e-14);
        let stats = StatsGroupe {
            w: vec![s],
            m: vec![1.0],
            g: vec![1.0],
            n_liens: 1,
            n_configs: 1,
        };
        let sol = resoudre_pl_avec(&[&stats], 1, &[0.0], RapportBessel::I1SurI0, 1).unwrap();
        assert!((sol.beta[0] - 1.0).abs() < 1e-12);
        assert!(sol.norme_gradient < 1e-12);
    }

    #[test]
    fn pl_conserve_les_couplages_signes() {
        let stats = StatsGroupe {
            w: vec![-rapport(RapportBessel::I2SurI1, 1.0)],
            m: vec![1.0],
            g: vec![1.0],
            n_liens: 1,
            n_configs: 1,
        };
        let sol = resoudre_pl(&[&stats], 1, &[0.0], 1).unwrap();
        assert!((sol.beta[0] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn newton_refuse_avant_bessel_les_departs_et_candidats_excessifs() {
        let mut stats = StatsGroupe {
            w: vec![1.0],
            m: vec![1e-12],
            g: vec![1.0],
            n_liens: 1,
            n_configs: 1,
        };
        let sd = resoudre_sd(&[&stats], 1).unwrap();
        let err = resoudre_pl(&[&stats], 1, &sd.beta, 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("maximum à l'infini"));
        for depart in [f64::INFINITY, f64::NAN, -1e5] {
            assert!(resoudre_pl(&[&stats], 1, &[depart], 1).is_err());
        }
        stats.g[0] = 1e-20; // Toutes les propositions restent excessives jusqu'à T_MIN.
        assert!(resoudre_pl(&[&stats], 1, &[0.0], 1)
            .unwrap_err()
            .to_string()
            .contains("maximum à l'infini"));
    }

    /// Intégrateur (M2 r3) : contre-exemple d'Astra (tour 2, P1) — départ SD admissible (≈ 8 044) mais racine PL à β = 10 500 > 1e4 :
    /// échec explicite « maximum à l'infini », jamais une solution « ok » arrêtée à la borne.
    #[test]
    fn newton_refuse_une_racine_au_dela_de_la_borne_atteinte_progressivement() {
        let g = 1e-8;
        let w = 1e-4 * crate::rg::flot::bessel::i2_sur_i1(1.05);
        let stats = StatsGroupe {
            w: vec![w],
            m: vec![g - w * w],
            g: vec![g],
            n_liens: 1,
            n_configs: 1,
        };
        let sd = resoudre_sd(&[&stats, &stats], 1).unwrap();
        assert!(sd.beta[0] > 7_000.0 && sd.beta[0] < 1e4, "départ SD {}", sd.beta[0]);
        let err = resoudre_pl(&[&stats, &stats], 1, &sd.beta, 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("maximum à l'infini"), "{err}");
    }

    #[test]
    fn conditionnement_sd_independant_de_echelle() {
        for echelle in [1e-16, 1.0, 1e16] {
            let stats = StatsGroupe {
                w: vec![echelle, echelle],
                m: vec![
                    echelle,
                    echelle * (1.0 - 1e-13),
                    echelle * (1.0 - 1e-13),
                    echelle,
                ],
                ..StatsGroupe::default()
            };
            assert!(resoudre_sd(&[&stats], 2)
                .unwrap_err()
                .to_string()
                .contains("mal conditionné"));
            let stats = StatsGroupe {
                w: vec![echelle, echelle],
                m: vec![2.0 * echelle, echelle, echelle, 2.0 * echelle],
                ..StatsGroupe::default()
            };
            let sol = resoudre_sd(&[&stats], 2).unwrap();
            assert!(sol.beta.iter().all(|b| (b - 1.0).abs() < 1e-14));
            assert!((sol.valeurs_propres_m[1] / sol.valeurs_propres_m[0] - 3.0).abs() < 1e-14);
        }
    }

    #[test]
    fn sd_sans_stockage_gram_et_garde_tailles() {
        let champ = champ_haar(4, 2611020118);
        let table = TableFormes::construire().unwrap();
        let famille = Famille::analyser("P,R,C,T").unwrap();
        let avec = stats_config(&champ, &table, &famille, 1).unwrap();
        let mut sans = StatsGroupe::default();
        accumuler_config(&champ, &table, &famille, 4, false, &mut sans).unwrap();
        assert_eq!(sans.w, avec.w);
        assert_eq!(sans.m, avec.m);
        assert_eq!(sans.n_liens, avec.n_liens);
        assert_eq!(sans.n_configs, avec.n_configs);
        assert!(sans.g.is_empty());
        assert_eq!(sans.g.capacity(), 0);
        assert!(verifier_tailles_groupes(&[2, 4]).is_ok());
        assert!(verifier_tailles_groupes(&[1, 3]).is_err());
    }

    #[test]
    fn ecart_apparie_annule_la_fluctuation_commune() {
        let pl = vec![Valeur {
            valeur: 3.0,
            erreur: 2.0,
            replicats: vec![1.0, 2.0, 4.0],
        }];
        let sd = vec![Valeur {
            valeur: 2.5,
            erreur: 2.0,
            replicats: vec![0.5, 1.5, 3.5],
        }];
        let ecart = differences_appariees(&pl, &sd).unwrap();
        assert_eq!(ecart[0].valeur, 0.5);
        assert_eq!(ecart[0].erreur, 0.0);
        assert_eq!(ecart[0].replicats, vec![0.5; 3]);
    }
}
