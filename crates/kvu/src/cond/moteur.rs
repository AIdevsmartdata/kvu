//! Moteur du balayage conditionnel (GPU HIP/CUDA ou CPU). Propriétaire : T5.
//!
//! Un [`MoteurCond`] porte UNE chaîne : champ (résident sur le GPU si `app` est donné, sinon en mémoire
//! hôte), V figé, β et source d'aléa. Le mode se déduit de la source :
//! - [`SourceAlea::Numpy`] ⇒ `ref_rng` (tirages de cb.py rejoués par [`GenerateurNumpy`]) ;
//! - [`SourceAlea::Philox`] ⇒ `rapide` (Philox à compteur (li, sweep_id, tirage, ETIQUETTE_COND)).
//!
//! La référence CPU (rayon) des deux modes vit dans [`super::cpu`] ; elle sert de juge au GPU (EQC-3).

use super::cpu::{self, DiagRapide, ParamRapide};
use super::gpu::EtatGpu;
use super::{GenerateurNumpy, GeometrieCond};
use crate::commun::{BackendKind, ChampHote, Erreur, Graine, Resultat};
use crate::gpu::{Appareil, ProvenanceCompil};
use crate::heritage::su2_hb::{Geo, Q};
use crate::su2::ChampGpu;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModeCond {
    RefRng,
    Rapide,
}

/// Source d'aléa : numpy (RefRng, l'état avance comme le générateur de cb) ou Philox (Rapide).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SourceAlea {
    Numpy(GenerateurNumpy),
    Philox { graine: Graine, sweep_id: u32 },
}

impl SourceAlea {
    pub fn mode(&self) -> ModeCond {
        match self {
            SourceAlea::Numpy(_) => ModeCond::RefRng,
            SourceAlea::Philox { .. } => ModeCond::Rapide,
        }
    }

    /// Source initiale d'une chaîne : `ref_rng` = `np.random.default_rng(graine)`, `rapide` = Philox, balayage 0.
    pub fn depuis_graine(mode: ModeCond, graine: u64) -> SourceAlea {
        match mode {
            ModeCond::RefRng => SourceAlea::Numpy(GenerateurNumpy::depuis_graine(graine)),
            ModeCond::Rapide => SourceAlea::Philox { graine: Graine(graine), sweep_id: 0 },
        }
    }
}

struct FondCpu {
    u: Vec<Q>,
    g: Geo,
    diag: DiagRapide,
    /// Remplir `diag.essais` à chaque balayage rapide (EQC-3).
    noter_essais: bool,
}

enum Fond {
    Cpu(FondCpu),
    Gpu(Box<EtatGpu>),
}

/// Une chaîne conditionnelle : champ (résident si GPU), V figé, β, source d'aléa.
pub struct MoteurCond {
    pub geo: Arc<GeometrieCond>,
    pub beta: f64,
    pub mode: ModeCond,
    fond: Fond,
    alea: SourceAlea,
    vfix: Vec<Q>,
}

impl MoteurCond {
    /// `app` None ⇒ référence CPU Rust (rayon) des deux modes.
    ///
    /// Refus (Erreur::Config) : L du champ différent de celui de la géométrie, V de mauvaise taille, β négatif ou
    /// non fini, β = 0 en `ref_rng` (la référence ne termine pas : α = 0 donne x = NaN).
    pub fn nouveau(
        app: Option<&Appareil>,
        geo: Arc<GeometrieCond>,
        c: &ChampHote,
        vfix: Vec<[f64; 4]>,
        beta: f64,
        alea: SourceAlea,
    ) -> Resultat<MoteurCond> {
        if c.l != geo.l || c.q.len() != 16 * geo.v {
            return Err(Erreur::Config(format!(
                "champ de L = {} incompatible avec la géométrie de L = {}",
                c.l, geo.l
            )));
        }
        if vfix.len() != 4 * geo.npair() {
            return Err(Erreur::Config(format!("V figé : {} quaternions au lieu de {}", vfix.len(), 4 * geo.npair())));
        }
        if !beta.is_finite() || beta < 0.0 {
            return Err(Erreur::Config(format!("β = {beta} (β ≥ 0 fini requis)")));
        }
        let mode = alea.mode();
        if mode == ModeCond::RefRng && beta == 0.0 {
            return Err(Erreur::Config(
                "β = 0 refusé en mode ref_rng (la référence ne termine pas : α = 0 donne x = NaN) ; utiliser le mode rapide"
                    .into(),
            ));
        }
        let fond = match app {
            Some(a) => Fond::Gpu(Box::new(EtatGpu::nouveau(a, &geo, c, &vfix)?)),
            None => Fond::Cpu(FondCpu {
                u: c.q.chunks_exact(4).map(|q| [q[0], q[1], q[2], q[3]]).collect(),
                g: Geo::new(geo.l),
                diag: DiagRapide::default(),
                noter_essais: false,
            }),
        };
        Ok(MoteurCond { geo, beta, mode, fond, alea, vfix })
    }

    /// Un balayage (= cb.sweep) : réunitarisation, 4 directions de paires, puis 4 mu × 2 parités de libres.
    pub fn sweep(&mut self) -> Resultat<()> {
        let beta = self.beta;
        match (&mut self.fond, &mut self.alea) {
            (Fond::Cpu(f), SourceAlea::Numpy(rng)) => {
                cpu::balayage_ref_rng(&mut f.u, &self.geo, &f.g, &self.vfix, beta, rng)
            }
            (Fond::Cpu(f), SourceAlea::Philox { graine, sweep_id }) => {
                if f.noter_essais && f.diag.essais.len() != 4 * self.geo.v {
                    f.diag.essais = vec![0; 4 * self.geo.v];
                }
                let p = ParamRapide { beta, graine: *graine, sweep_id: *sweep_id };
                cpu::balayage_rapide(&mut f.u, &self.geo, &f.g, &self.vfix, p, &mut f.diag);
                *sweep_id = sweep_id.wrapping_add(1);
                Ok(())
            }
            (Fond::Gpu(e), SourceAlea::Numpy(rng)) => e.balayage_ref_rng(beta, rng),
            (Fond::Gpu(e), SourceAlea::Philox { graine, sweep_id }) => {
                e.balayage_rapide(beta, *graine, *sweep_id)?;
                *sweep_id = sweep_id.wrapping_add(1);
                Ok(())
            }
        }
    }

    pub fn telecharger(&self) -> Resultat<ChampHote> {
        match &self.fond {
            Fond::Cpu(f) => Ok(ChampHote { l: self.geo.l, q: f.u.iter().flatten().copied().collect() }),
            Fond::Gpu(e) => e.telecharger(),
        }
    }

    /// Remplace le champ (reprise) ; la source d'aléa est inchangée.
    pub fn charger_champ(&mut self, c: &ChampHote) -> Resultat<()> {
        if c.l != self.geo.l || c.q.len() != 16 * self.geo.v {
            return Err(Erreur::Config("champ de reprise incompatible avec la géométrie".into()));
        }
        match &mut self.fond {
            Fond::Cpu(f) => {
                f.u = c.q.chunks_exact(4).map(|q| [q[0], q[1], q[2], q[3]]).collect();
                Ok(())
            }
            Fond::Gpu(e) => e.televerser(c),
        }
    }

    /// Champ résident (lu par fibre::Mesureur sans copie) ; None en CPU.
    pub fn champ_gpu(&self) -> Option<&ChampGpu> {
        match &self.fond {
            Fond::Cpu(_) => None,
            Fond::Gpu(e) => Some(e.champ()),
        }
    }

    /// (max |V(C) − Vfix|, max |UU⁺ − 1|) : gates de run_one.py.
    pub fn controles(&self) -> Resultat<(f64, f64)> {
        match &self.fond {
            Fond::Cpu(f) => Ok(cpu::controles(&f.u, &self.geo, &self.vfix)),
            Fond::Gpu(e) => e.controles(),
        }
    }

    /// État de la source d'aléa (reprise à l'octet).
    pub fn alea(&self) -> SourceAlea {
        self.alea.clone()
    }

    /// Backend effectif (Cpu si aucun appareil).
    pub fn backend(&self) -> BackendKind {
        match &self.fond {
            Fond::Cpu(_) => BackendKind::Cpu,
            Fond::Gpu(e) => e.backend(),
        }
    }

    /// V figé (4 × npair quaternions, ordre de `v_paires`).
    pub fn vfix(&self) -> &[[f64; 4]] {
        &self.vfix
    }

    /// Provenance du module GPU compilé (manifeste), None en CPU.
    pub fn provenance(&self) -> Option<&ProvenanceCompil> {
        match &self.fond {
            Fond::Cpu(_) => None,
            Fond::Gpu(e) => Some(e.provenance()),
        }
    }

    /// Demande le nombre d'essais par lien à chaque balayage rapide (EQC-3). Sans effet en `ref_rng`.
    pub fn activer_diag_essais(&mut self) -> Resultat<()> {
        match &mut self.fond {
            Fond::Cpu(f) => {
                f.noter_essais = true;
                Ok(())
            }
            Fond::Gpu(e) => e.activer_diag(),
        }
    }

    /// Diagnostic du mode rapide : tirages au plafond cumulés (attendu 0) et essais du dernier balayage.
    pub fn diag_rapide(&self) -> Resultat<DiagRapide> {
        match &self.fond {
            Fond::Cpu(f) => Ok(f.diag.clone()),
            Fond::Gpu(e) => e.diag_rapide(),
        }
    }
}
