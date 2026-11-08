// SPDX-License-Identifier: Apache-2.0
//! Aides des tests qui doivent donner le même verdict sur toute machine. Inclus par
//! `#[path = "donnees/hermetique/support.rs"] mod support;`.
//!
//! Seuil de la garde SSD : le binaire et la bibliothèque refusent d'écrire quand il resterait moins de
//! `ssd_min_go` Go libres (profil de `kvu.toml` : 50 Go sur le gamer, 100 Go sur l'acemagic ; surcharge de
//! l'environnement : `KEVINOTRON_SSD_MIN_GO`). Un test qui n'écrit que quelques Ko ne devrait pas échouer parce que
//! le disque de l'hôte est plein, et la machine qui le lance ne devrait pas avoir à poser `KEVINOTRON_SSD_MIN_GO`
//! pour toute la suite (les tests de la garde elle-même fixent leurs propres seuils).
//!
//! `KVU_TEST_SSD_MIN_GO=<Go>` règle donc ce seuil POUR LES SEULS tests qui incluent ce module, par exemple
//! `KVU_TEST_SSD_MIN_GO=0 cargo test --release` sur un disque sous le seuil du profil. Sans la variable, rien n'est
//! changé : le seuil de la machine s'applique, comme avant.

use std::sync::Once;

/// Variable lue par les tests (nombre de Go libres minimum après écriture ; la virgule décimale est acceptée).
pub const VARIABLE_SEUIL_SSD: &str = "KVU_TEST_SSD_MIN_GO";

/// Reporte `KVU_TEST_SSD_MIN_GO` dans `KEVINOTRON_SSD_MIN_GO` du processus de test, une seule fois : la garde de la
/// bibliothèque (appels dans le même processus) et celle des sous-processus lancés ensuite (ils héritent de
/// l'environnement) lisent alors le même seuil. Sans la variable, ne fait rien.
///
/// Une valeur qui n'est pas un nombre fini ≥ 0 FAIT ÉCHOUER le test : le binaire ignorerait sans rien dire une valeur
/// illisible, et le test tournerait au seuil de la machine en laissant croire le contraire.
pub fn appliquer_seuil_ssd_des_tests() {
    static UNE_FOIS: Once = Once::new();
    UNE_FOIS.call_once(|| {
        let Some(brut) = std::env::var_os(VARIABLE_SEUIL_SSD) else {
            return;
        };
        let texte = brut.to_string_lossy().trim().replace(',', ".");
        let go: f64 = texte.parse().unwrap_or_else(|_| {
            panic!("{VARIABLE_SEUIL_SSD} = {texte:?} : un nombre de Go est attendu")
        });
        assert!(
            go.is_finite() && go >= 0.0,
            "{VARIABLE_SEUIL_SSD} = {texte} : un seuil fini, positif ou nul, est attendu"
        );
        std::env::set_var("KEVINOTRON_SSD_MIN_GO", &texte);
    });
}
