//! Machine : détection, politique de ressources, verrou GPU, doctor, file de jobs. Propriétaire : T1.
//!
//! - [`detection`] : hôte, CPU, RAM et GPU (sondes `sonde_hip` et `sonde_cuda` par dlopen, sans
//!   contexte) ; identification de la machine ; choix du backend (jamais de repli silencieux) ;
//! - [`politique`] : valeurs du tableau §3 de l'architecture et surcharges `KEVINOTRON_*` ;
//! - [`verrou`] : flock GPU par machine, sans interblocage avec un `flock` externe (ancêtre
//!   détecté par /proc/locks) ;
//! - [`file`] : file de jobs en fichiers, transitions par renommage atomique, jobs orphelins ;
//! - [`processus`] : préparation de toute sous-commande (affinité de la politique, surcharge de machine) ;
//! - [`admission`] : admission mémoire des runs directs (RAM disponible − besoin estimé ≥ plancher) ;
//! - [`cli`] : `doctor`, `etat`, `queue`.

pub mod admission;
pub mod cli;
pub mod compat_tests_existants;
pub mod detection;
pub mod file;
pub mod politique;
pub mod processus;
pub mod sonde_cuda;
pub mod sonde_hip;
pub mod verrou;

pub use detection::{choisir_backend, detecter, GpuDetecte, ProfilMachine};
pub use politique::{politique, Politique};
pub use verrou::{GardeVerrou, VerrouGpu};
