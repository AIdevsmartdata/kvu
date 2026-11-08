// SPDX-License-Identifier: Apache-2.0
//! Régénération des ensembles : manifestes `kvu-regen/1`, adaptateurs des manifestes existants, classement A/B/C,
//! rejeu (`kvu regen`) et simulation de classement (`kvu store classer`). Propriétaire : WP4.
//!
//! - [`schema`]       sous-ensemble de JSON Schema et validation de `kvu-regen-1.schema.json` (embarqué) ;
//! - [`manifeste`]    petits utilitaires partagés (date, hachage, parcours d'arbre) ;
//! - [`adaptateurs`]  `su2_hip/1` et `su2_gen/1` lus TELS QUELS et traduits en `kvu-regen/1` ;
//! - [`classer`]      classes A/B/C (ARCHITECTURE §7.2, règle gelée de GATES.md WP4) et contexte de preuves ;
//! - [`rejeu`]        reconstruction de la commande, rejeu sous verrou dans un scratch, comparaison sha256 fichier par fichier ;
//! - [`ensemble`]     classement d'un ensemble de chaînes (simulation, lecture seule) ;
//! - [`cli`]          sous-commande `regen` et fonctions de `store classer`.
//!
//! Règle d'or : rien de ce module n'écrit, ne déplace ni ne supprime quoi que ce soit dans un ensemble existant
//! (les ensembles sont lus ; tout ce qui est produit va dans un répertoire de sortie NEUF, jamais sous le répertoire comparé).

pub mod adaptateurs;
pub mod classer;
pub mod cli;
pub mod ensemble;
pub mod manifeste;
pub mod rejeu;
pub mod schema;
