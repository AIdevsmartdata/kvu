//! Flot de Wilson (Lüscher 2010, RK3 de crate::observables::wilson_flow) en quaternions. Propriétaire : T7.
//!
//! Le flot travaille sur une COPIE du champ : le champ source n'est jamais modifié.
//! GPU si `champ` est donné (copie résidente FP64, Z0/Z1/Z2 en tampons FP64), sinon CPU sur `hote`.

use super::gpu::ModuleObs;
use super::ref_cpu::{self, Densites};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::gpu::Appareil;
use crate::su2::ChampGpu;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParamsFlot {
    pub dt: f64,
    pub t_max: f64,
    /// Mesurer E(t) tous les `mesure_chaque` pas.
    pub mesure_chaque: usize,
    /// Cible de t²E(t₀) (0,3 par défaut, convention du moteur historique ; à fixer par le PREREG pour SU(2)).
    pub cible_t2e: f64,
    /// Cible de W(w₀²) = t d/dt(t²E).
    pub cible_w: f64,
}

impl Default for ParamsFlot {
    /// dt = 0,01, t_max = 3, une mesure par pas, cibles 0,3 (convention du moteur historique).
    fn default() -> Self {
        ParamsFlot {
            dt: 0.01,
            t_max: 3.0,
            mesure_chaque: 1,
            cible_t2e: 0.3,
            cible_w: 0.3,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CourbeFlot {
    pub t: Vec<f64>,
    /// E plaquette, normalisation de crate::observables::wilson_flow::action_density (fix 2026-06-19).
    pub e_plaq: Vec<f64>,
    /// E trèfle (clover).
    pub e_trefle: Vec<f64>,
    /// t² E_plaq (la courbe dont t₀ est extrait, comme `wilson_flow_curve`).
    pub t2e: Vec<f64>,
    /// Charge topologique (trèfle), par mesure.
    pub q_top: Vec<f64>,
    /// t₀/a² : temps de flot où t²E_plaq croise `cible_t2e` (interpolation linéaire), `None` si non encadré.
    pub t0: Option<f64>,
    /// w₀/a = √t où W(t) = t d(t²E)/dt croise `cible_w` (interpolation linéaire), `None` si non encadré.
    pub w0: Option<f64>,
}

/// Interpolation linéaire du croisement de `cible` entre deux échantillons (celle de `wilson_flow_curve`).
fn croisement(t_lo: f64, f_lo: f64, t_hi: f64, f_hi: f64, cible: f64) -> f64 {
    let denom = f_hi - f_lo;
    if denom.abs() < 1e-300 {
        return t_lo;
    }
    t_lo + (cible - f_lo) / denom * (t_hi - t_lo)
}

/// t₀ et w₀ d'une courbe échantillonnée (t croissants, t²E associés), comme `wilson_flow_curve` :
/// - t₀ : premier croisement montant de `cible_t2e` par t²E, interpolé linéairement ;
/// - W(tᵢ) = tᵢ · d(t²E)/dt, différence CENTRÉE (différences unilatérales aux deux bouts) ;
/// - w₀ = √t au premier croisement montant de `cible_w` par W (si t > 0).
///
/// Pour un pas régulier, le résultat est celui de `wilson_flow_curve` à l'arrondi près.
pub fn echelles(t: &[f64], t2e: &[f64], cible_t2e: f64, cible_w: f64) -> (Option<f64>, Option<f64>) {
    let n = t.len().min(t2e.len());
    if n < 2 {
        return (None, None);
    }
    let mut t0 = None;
    for i in 1..n {
        if t2e[i - 1] < cible_t2e && t2e[i] >= cible_t2e {
            t0 = Some(croisement(t[i - 1], t2e[i - 1], t[i], t2e[i], cible_t2e));
            break;
        }
    }
    let mut w = vec![0.0; n];
    for i in 0..n {
        let (a, b) = if i == 0 {
            (0, 1)
        } else if i == n - 1 {
            (n - 2, n - 1)
        } else {
            (i - 1, i + 1)
        };
        w[i] = t[i] * ((t2e[b] - t2e[a]) / (t[b] - t[a]));
    }
    let mut w0 = None;
    for i in 1..n {
        if w[i - 1] < cible_w && w[i] >= cible_w {
            let tc = croisement(t[i - 1], w[i - 1], t[i], w[i], cible_w);
            w0 = Some(if tc > 0.0 { tc.sqrt() } else { 0.0 });
            break;
        }
    }
    (t0, w0)
}

/// Variante qui réutilise un module déjà compilé (mesures en ligne).
pub(crate) fn flot_wilson_avec(
    app: Option<&Appareil>,
    m: Option<&ModuleObs>,
    champ: Option<&ChampGpu>,
    c: usize,
    hote: Option<&ChampHote>,
    p: &ParamsFlot,
) -> Resultat<CourbeFlot> {
    match champ {
        Some(ch) => {
            let mut tmp = None;
            module(app, m, &mut tmp)?.flot_wilson(ch, c, p)
        }
        None => ref_cpu::flot(hote.ok_or_else(champ_manquant)?, p),
    }
}

fn champ_manquant() -> Erreur {
    Erreur::Config("ni champ GPU ni champ hôte fournis".into())
}

fn module<'a>(
    app: Option<&Appareil>,
    m: Option<&'a ModuleObs>,
    tmp: &'a mut Option<ModuleObs>,
) -> Resultat<&'a ModuleObs> {
    if let Some(m) = m {
        return Ok(m);
    }
    let app = app.ok_or_else(|| Erreur::Config("champ GPU fourni sans appareil".into()))?;
    Ok(tmp.insert(ModuleObs::nouveau(app)?))
}

/// Flot d'une COPIE du champ (le champ source n'est jamais modifié).
pub fn flot_wilson(
    app: Option<&Appareil>,
    champ: Option<&ChampGpu>,
    c: usize,
    hote: Option<&ChampHote>,
    p: &ParamsFlot,
) -> Resultat<CourbeFlot> {
    flot_wilson_avec(app, None, champ, c, hote, p)
}

/// Champ obtenu après `n_pas` pas de flot de taille `dt` (copie ; sert aux gates OBS-3).
pub fn flot_champ(
    app: Option<&Appareil>,
    champ: Option<&ChampGpu>,
    c: usize,
    hote: Option<&ChampHote>,
    dt: f64,
    n_pas: usize,
) -> Resultat<ChampHote> {
    match champ {
        Some(ch) => {
            let mut tmp = None;
            module(app, None, &mut tmp)?.flot_champ_final(ch, c, dt, n_pas)
        }
        None => ref_cpu::flot_n_pas(hote.ok_or_else(champ_manquant)?, dt, n_pas),
    }
}

/// E plaquette, E trèfle et Q_top du champ tel quel (sans flot).
pub fn densites(
    app: Option<&Appareil>,
    champ: Option<&ChampGpu>,
    c: usize,
    hote: Option<&ChampHote>,
) -> Resultat<Densites> {
    match champ {
        Some(ch) => {
            let mut tmp = None;
            module(app, None, &mut tmp)?.densites(ch, c)
        }
        None => ref_cpu::densites(hote.ok_or_else(champ_manquant)?),
    }
}
