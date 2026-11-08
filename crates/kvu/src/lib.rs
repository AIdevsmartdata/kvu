//! kvu — le kevinotron unifié (branche unifie/*, architecture : ARCHITECTURE_UNIFIEE.md).
//!
//! Un seul moteur qui reconnaît la machine (profils de `kvu.toml` ; GPU HIP ou CUDA lu sur le périphérique ; sinon CPU)
//! et choisit le bon code à l'exécution. Les noyaux GPU sont écrits UNE fois (`noyaux/`) et compilés
//! à l'exécution par hiprtc ou NVRTC, après un prélude propre au backend.
//!
//! Carte des modules (propriétaire entre crochets ; un module n'est modifié QUE par sa tâche) :
//! - [`config`]   configuration des machines (`kvu.toml`), analyseur TOML minimal       [WP3]
//! - [`commun`]   types partagés, erreurs, codes de sortie                        [architecte, gelé]
//! - [`heritage`] modules historiques copiés à l'octet (src/heritage/, sans `#[path]`)  [architecte, gelé]
//! - [`noyaux`]   registre des sources GPU embarquées (`include_str!`)            [architecte, gelé]
//! - [`machine`]  détection, politique de ressources, verrou GPU, doctor, file    [T1]
//! - [`gpu`]      runtime GPU unifié HIP/CUDA, assemblage des sources, cache      [T2]
//! - [`su2`]      générateur SU(2) : FP64 de référence, lots, bord gelé, reprise  [T3]
//! - [`su2::rapide32`] chemin FP32 rapide et sa loi                               [T4]
//! - [`cond`]     bain thermique conditionnel « fin sachant grossier »           [T5]
//! - [`cond3`]    bain conditionnel de facteur 3 sur la fibre dure de DEC, réponse de la moyenne conditionnelle [T3-D]
//! - [`fibre`]    mesure de fibre (F, D, blk)                                     [T6]
//! - [`obs`]      observables : boucles de Wilson, flot de Wilson                [T7]
//! - [`stockage`] NPY, provenance, garde SSD, store du/gc, fetch                  [T8]
//! - [`regen`]    manifestes de régénération kvu-regen/1, `kvu regen`, classement A/B/C   [WP4]
//! - [`groupes`]  registre des groupes de jauge et gates à réponse connue            [WP5]
//! - [`rg`]       module RG par blocs : bloc de Bałaban L_b = 3, témoin Swendsen b = 2, domaine  [WP6]
//! - [`cli`]      ligne de commande unifiée, valide, bench                       [T9]

pub mod commun;
pub mod config;
pub mod heritage;
pub mod noyaux;

pub mod machine;
pub mod gpu;
pub mod su2;
pub mod cond;
pub mod cond3;
pub mod fibre;
pub mod obs;
pub mod stockage;
pub mod regen;
pub mod groupes;
pub mod rg;
pub mod hess;
pub mod fibre_contrainte;
pub mod cli;

pub use commun::{BackendKind, Depart, Erreur, Graine, Machine, Precision, Resultat};
