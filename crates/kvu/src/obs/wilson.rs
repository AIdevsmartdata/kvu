//! Boucles de Wilson R×T et corrélateur de Polyakov. Propriétaire : T7.
//!
//! API : GPU si `champ` est donné (chaîne `c` du lot résident), sinon CPU sur `hote`
//! (référence quaternionique `ref_cpu`). Toutes les sommes sont FP64 à arbre fixe.

use super::gpu::ModuleObs;
use super::ref_cpu;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::gpu::Appareil;
use crate::su2::ChampGpu;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParamsWilson {
    pub r_max: usize,
    pub t_max: usize,
    /// Direction « temps » (3 par défaut, celle des Polyakov de su2_hip).
    pub dir_temps: usize,
    /// Vrai : moyenne sur les 12 orientations (plan, rôle de R et T) ; faux : T selon dir_temps seulement.
    pub toutes_orientations: bool,
}

impl ParamsWilson {
    /// Boucles temporelles jusqu'à R × T, T selon la direction 3.
    pub fn temporelles(r_max: usize, t_max: usize) -> Self {
        ParamsWilson {
            r_max,
            t_max,
            dir_temps: 3,
            toutes_orientations: false,
        }
    }
}

/// W[r-1][t-1] = ⟨½ Tr W(r,t)⟩ (moyenne sur sites et orientations), FP64.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TableWilson {
    pub r_max: usize,
    pub t_max: usize,
    pub w: Vec<f64>,
}

impl TableWilson {
    /// W(r, t), r ∈ 1..=r_max, t ∈ 1..=t_max.
    pub fn get(&self, r: usize, t: usize) -> f64 {
        self.w[(r - 1) * self.t_max + (t - 1)]
    }
}

/// Module GPU à utiliser : exigé si un champ résident est donné.
fn module_requis<'a>(
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

fn hote_requis(hote: Option<&ChampHote>) -> Resultat<&ChampHote> {
    hote.ok_or_else(|| Erreur::Config("ni champ GPU ni champ hôte fournis".into()))
}

/// Variante qui réutilise un module déjà compilé (mesures en ligne).
pub(crate) fn boucles_wilson_avec(
    app: Option<&Appareil>,
    m: Option<&ModuleObs>,
    champ: Option<&ChampGpu>,
    c: usize,
    hote: Option<&ChampHote>,
    p: &ParamsWilson,
) -> Resultat<TableWilson> {
    match champ {
        Some(ch) => {
            let mut tmp = None;
            module_requis(app, m, &mut tmp)?.boucles_wilson(ch, c, p)
        }
        None => ref_cpu::table_wilson(hote_requis(hote)?, p),
    }
}

pub(crate) fn correlateur_polyakov_avec(
    app: Option<&Appareil>,
    m: Option<&ModuleObs>,
    champ: Option<&ChampGpu>,
    c: usize,
    hote: Option<&ChampHote>,
    dir_temps: usize,
) -> Resultat<Vec<f64>> {
    match champ {
        Some(ch) => {
            let mut tmp = None;
            module_requis(app, m, &mut tmp)?.correlateur_polyakov(ch, c, dir_temps)
        }
        None => ref_cpu::correlateur_polyakov(hote_requis(hote)?, dir_temps),
    }
}

/// GPU si `champ` est donné (chaîne `c`), sinon CPU sur `hote`.
pub fn boucles_wilson(
    app: Option<&Appareil>,
    champ: Option<&ChampGpu>,
    c: usize,
    hote: Option<&ChampHote>,
    p: &ParamsWilson,
) -> Resultat<TableWilson> {
    boucles_wilson_avec(app, None, champ, c, hote, p)
}

/// Corrélateur ⟨P(0) P(r)†⟩ des boucles de Polyakov (direction dir_temps), r = 0..L/2, FP64.
///
/// C(r) = ⟨ℓ(x⃗) ℓ(x⃗ + r ê_i)⟩ avec ℓ = ½ Tr P (réel pour SU(2)), moyenné sur les sites et sur les
/// trois axes spatiaux ê_i ⟂ dir_temps.
pub fn correlateur_polyakov(
    app: Option<&Appareil>,
    champ: Option<&ChampGpu>,
    c: usize,
    hote: Option<&ChampHote>,
    dir_temps: usize,
) -> Resultat<Vec<f64>> {
    correlateur_polyakov_avec(app, None, champ, c, hote, dir_temps)
}

/// ⟨ℓ⟩ dans les quatre directions (P0..P3 de su2_hip), arbre fixe de `k_final`.
pub fn polyakov_moyen(
    app: Option<&Appareil>,
    champ: Option<&ChampGpu>,
    c: usize,
    hote: Option<&ChampHote>,
) -> Resultat<[f64; 4]> {
    match champ {
        Some(ch) => {
            let mut tmp = None;
            module_requis(app, None, &mut tmp)?.polyakov_moyen(ch, c)
        }
        None => ref_cpu::polyakov_moyen(hote_requis(hote)?),
    }
}
