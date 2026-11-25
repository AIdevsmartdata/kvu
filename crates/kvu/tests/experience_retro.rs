// SPDX-License-Identifier: Apache-2.0
//! E-RETRO : vérification facultative d'un rétro-manifeste fourni de l'extérieur.

use std::process::Command;

/// Porte E-RETRO : ignorée par défaut et ainsi VISIBLE comme telle dans la sortie de `cargo test` ;
/// la recette la lance avec `--ignored` et KVU_EXP_RETRO_DIR. Sans la variable, elle ÉCHOUE (jamais « verte à vide »).
#[test]
#[ignore = "exige KVU_EXP_RETRO_DIR (copie du cas réel) : lancer avec --ignored"]
fn e_retro_sur_copie_deja_construite() {
    let racine = std::env::var_os("KVU_EXP_RETRO_DIR")
        .expect("E-RETRO : KVU_EXP_RETRO_DIR absente, la porte n'a pas pu tourner");
    let sortie = Command::new(env!("CARGO_BIN_EXE_kvu"))
        .args(["experience", "verifier", "--racine"])
        .arg(racine)
        .env("KVU_EXPERIENCE_HORLOGE", "2001-09-09T02:00:00Z")
        .output()
        .unwrap();
    assert_eq!(sortie.status.code(), Some(0), "{sortie:?}");
}
