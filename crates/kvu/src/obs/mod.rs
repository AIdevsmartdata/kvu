//! Observables GPU validées contre le CPU : boucles de Wilson R×T, Polyakov, flot de Wilson RK3
//! (E(t) plaquette et trèfle, t₀, w₀, charge topologique). Propriétaire : T7.
//! Accumulations FP64 à arbre fixe ; le champ F32 est promu en FP64 à la lecture.
//!
//! - [`ref_cpu`]  référence quaternionique CPU (rayon), vérité des gates OBS-* ;
//! - [`wilson`]   boucles R×T, corrélateur et moyenne de Polyakov (GPU ou CPU) ;
//! - [`flot`]     flot RK3, E_plaq, E_trèfle, Q_top, t₀, w₀ (GPU ou CPU) ;
//! - [`gpu`]      pilote des noyaux `noyaux/obs_wilson.cu` et `noyaux/obs_flot.cu` ;
//! - [`enligne`]  mesures pendant la génération (`--mesures`) ;
//! - [`cli`]      sous-commande `obs wilson | polyakov | flot`.
//! - [`topo_cli`] mesures `obs topo` et validations `obs topo-gates`.
//!
//! Dépendances : T2 (runtime GPU), T3 (`ChampGpu`, `MesureEnLigne`), T8 (NPY, garde SSD) et T1 (détection de la
//! machine, verrou GPU), tous fusionnés dans `unifie/integ` (les replis `STUB-Tn` du développement sont retirés).

pub mod cli;
pub mod enligne;
pub mod flot;
pub mod gpu;
pub mod ref_cpu;
pub mod topo;
pub mod topo_cli;
pub mod topo_gates;
pub mod wilson;

pub use flot::{flot_wilson, CourbeFlot, ParamsFlot};
pub use wilson::{boucles_wilson, ParamsWilson, TableWilson};
