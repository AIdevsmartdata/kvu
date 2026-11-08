//! Mesure de fibre (port GPU de cb.mesure / cb.mesure_libre / block_sums). Propriétaire : T6.
//!
//! F = Wf · pv (54 fonctionnelles de Fourier, pmax² = 1), D = Σ_lignes ligne·ligneᵀ (54×54,
//! lignes de fibre : liens libres métrique 1, paires métrique 2 avec R_ba), blk = sommes de blocs
//! 2⁴ des plaquettes (float32). GPU : réduction DÉTERMINISTE (partiels de tranches fixes, puis somme à
//! ordre fixe) ; résultat indépendant du nombre de blocs lancés.
//!
//! Organisation :
//! - [`banc`] durées de mesure (pour `bench fibre`) ; [`poids`]  Wf (CSR) ; [`tables`] plaquettes et liens ; [`plan`] tables dérivées (Wf transposée,
//!   supports, plaquettes par lien) ; [`cpu`] référence rayon ; [`gpu`] chemin HIP/CUDA ;
//!   [`npy`] E/S minimales.

pub mod banc;
pub mod cli;
pub mod cpu;
pub mod gpu;
pub mod npy;
pub mod plan;
pub mod poids;
pub mod tables;
pub mod torsion;

use crate::commun::{ChampHote, Erreur, Resultat};
use crate::cond::GeometrieCond;
use crate::gpu::Appareil;
use crate::su2::ChampGpu;
use plan::{ModeFibre, PlanFibre};
use std::sync::Arc;

pub use cpu::somme_blocs;
pub use plan::TRANCHE;
pub use poids::{poids_fourier, PoidsFourier};

/// Résultat d'une mesure (= (F, D, blk) de cb.mesure).
#[derive(Clone, Debug, PartialEq)]
pub struct MesureFibre {
    pub f: Vec<f64>,
    /// nF × nF, ligne majeure.
    pub d: Vec<f64>,
    /// 6 × (L/2)⁴, float32 (block_sums(pv).astype(float32)).
    pub blk: Vec<f32>,
}

/// Mesureur réutilisable (tables et tampons alloués une fois par L).
pub struct Mesureur {
    pub geo: Arc<GeometrieCond>,
    /// Poids pour la référence CPU (None sur GPU : les poids sont alors résidents sur l'appareil).
    w_cpu: Option<PoidsFourier>,
    plan: Arc<PlanFibre>,
    gpu: Option<gpu::MoteurGpu>,
}

impl Mesureur {
    /// `app` None ⇒ référence CPU Rust.
    pub fn nouveau(app: Option<&Appareil>, geo: Arc<GeometrieCond>, w: &PoidsFourier) -> Resultat<Mesureur> {
        let plan = Arc::new(PlanFibre::nouveau(&geo, w)?);
        let gpu = match app {
            Some(a) => Some(gpu::MoteurGpu::nouveau(a, &plan, w)?),
            None => None,
        };
        Ok(Mesureur {
            geo,
            w_cpu: if gpu.is_none() { Some(w.clone()) } else { None },
            plan,
            gpu,
        })
    }

    /// Côté du réseau.
    pub fn l(&self) -> usize {
        self.plan.l
    }

    /// Mémoire GPU approximative (octets) qu'occupe ce mesureur (tables, pv, gradients, partiels de D).
    pub fn octets_gpu_estimes(&self) -> u64 {
        gpu::octets_estimes(self.plan.v, self.plan.wt_val.len())
    }

    /// Durées par étape (ms) de la dernière mesure GPU quand `KVU_FIBRE_PROFIL=1` ; vide sinon et en CPU.
    pub fn dernier_profil(&self) -> Vec<(String, f64)> {
        self.gpu.as_ref().map(|g| g.dernier_profil()).unwrap_or_default()
    }

    /// Provenance des modules GPU compilés (vide en CPU), pour le manifeste des sorties.
    pub fn provenance(&self) -> Vec<crate::gpu::ProvenanceCompil> {
        self.gpu.as_ref().map(|g| g.provenances().to_vec()).unwrap_or_default()
    }

    /// Vrai si les mesures passent par le GPU.
    pub fn sur_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Nombre de tranches par lancement de `k_D_partiel` (GPU) : un autre découpage de la grille de
    /// lancement ne change pas un bit de D (gate FIB-2). Sans effet en CPU.
    pub fn fixer_lot(&mut self, tranches_par_lancement: usize) -> Resultat<()> {
        match self.gpu.as_mut() {
            Some(g) => g.fixer_lot(tranches_par_lancement),
            None => Ok(()),
        }
    }

    fn poids_cpu(&self) -> &PoidsFourier {
        self.w_cpu
            .as_ref()
            .expect("chemin CPU : poids conservés à la construction")
    }

    fn controler_hote(&self, h: &ChampHote) -> Resultat<()> {
        if h.l != self.plan.l || h.q.len() != 16 * self.plan.v {
            return Err(Erreur::Config(format!(
                "fibre : champ hôte L = {} ({} réels) pour un mesureur L = {}",
                h.l,
                h.q.len(),
                self.plan.l
            )));
        }
        Ok(())
    }

    fn hote_requis<'a>(&self, hote: Option<&'a ChampHote>) -> Resultat<&'a ChampHote> {
        let h = hote.ok_or_else(|| {
            Erreur::Config(
                "fibre : référence CPU : un champ hôte est requis (le champ GPU n'est lisible qu'avec un appareil)"
                    .into(),
            )
        })?;
        self.controler_hote(h)?;
        Ok(h)
    }

    /// Mesure conditionnelle (fibre de dec) sur le champ résident (chaîne `c`) ou sur un champ hôte.
    pub fn mesurer(&mut self, champ: Option<&ChampGpu>, c: usize, hote: Option<&ChampHote>) -> Resultat<MesureFibre> {
        if let Some(g) = self.gpu.as_mut() {
            let s = g.mesurer(&self.plan, champ, c, hote, Some(ModeFibre::Conditionnel), true)?;
            return Ok(MesureFibre {
                f: s.f,
                d: s.d.unwrap_or_default(),
                blk: s.blk.unwrap_or_default(),
            });
        }
        let h = self.hote_requis(hote)?;
        let s = cpu::mesurer(&self.plan, self.poids_cpu(), h, Some(ModeFibre::Conditionnel));
        Ok(MesureFibre {
            f: s.f,
            d: s.d.unwrap_or_default(),
            blk: somme_blocs(self.plan.l, &s.pv),
        })
    }

    /// Sans conditionnement (tous les liens libres, métrique 1) : témoin Haar G10.
    pub fn mesurer_libre(
        &mut self,
        champ: Option<&ChampGpu>,
        c: usize,
        hote: Option<&ChampHote>,
    ) -> Resultat<(Vec<f64>, Vec<f64>)> {
        if let Some(g) = self.gpu.as_mut() {
            let s = g.mesurer(&self.plan, champ, c, hote, Some(ModeFibre::Libre), false)?;
            return Ok((s.f, s.d.unwrap_or_default()));
        }
        let h = self.hote_requis(hote)?;
        let s = cpu::mesurer(&self.plan, self.poids_cpu(), h, Some(ModeFibre::Libre));
        Ok((s.f, s.d.unwrap_or_default()))
    }

    /// Seulement F (et pv) : mesures légères entre deux mesures D.
    pub fn mesurer_f(&mut self, champ: Option<&ChampGpu>, c: usize, hote: Option<&ChampHote>) -> Resultat<Vec<f64>> {
        if let Some(g) = self.gpu.as_mut() {
            return Ok(g.mesurer(&self.plan, champ, c, hote, None, false)?.f);
        }
        let h = self.hote_requis(hote)?;
        Ok(cpu::mesurer(&self.plan, self.poids_cpu(), h, None).f)
    }

    /// Plaquettes (6V) et gradients (6V × 4 × 3) du champ : sur le GPU, ceux calculés par `k_plaq_grad`
    /// (pour les gates FIB-1 et FIB-3) ; sinon la référence CPU.
    pub fn plaquettes(
        &mut self,
        champ: Option<&ChampGpu>,
        c: usize,
        hote: Option<&ChampHote>,
    ) -> Resultat<(Vec<f64>, Vec<f64>)> {
        if let Some(g) = self.gpu.as_mut() {
            return g.plaquettes(&self.plan, champ, c, hote);
        }
        Ok(cpu::plaquettes_gradients(self.hote_requis(hote)?))
    }
}

impl Mesureur {
    /// Plaquettes seules (6V) : par `k_plaq_grad` sans gradients sur le GPU, par la référence CPU sinon.
    pub fn valeurs_plaquettes(
        &mut self,
        champ: Option<&ChampGpu>,
        c: usize,
        hote: Option<&ChampHote>,
    ) -> Resultat<Vec<f64>> {
        if let Some(g) = self.gpu.as_mut() {
            return g.valeurs_plaquettes(&self.plan, champ, c, hote);
        }
        Ok(cpu::plaquettes_valeurs(self.hote_requis(hote)?))
    }
}

/// Plaquettes ½ReTr U_p (6V, ordre loops_plaquettes : plan (mu<nu) majeur puis site) et gradients
/// (6V × 4 liens × 3), référence CPU (= cb.plaquettes).
pub fn plaquettes_gradients_cpu(c: &ChampHote) -> (Vec<f64>, Vec<f64>) {
    cpu::plaquettes_gradients(c)
}
