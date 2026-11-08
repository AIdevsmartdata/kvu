//! Constantes conservées pour les tests d'intégration existants (`tests/machine_*.rs`), qui les importent et ne
//! doivent pas être modifiés. Ce sont des DONNÉES DE TEST : le code de `src/` ne les lit jamais ; l'identification de
//! la machine, les chemins et les seuils viennent de `kvu.toml`. Un test de ce module vérifie qu'elles sont égales
//! aux valeurs de `kvu.toml` (le nom du fichier se termine par `_tests_existants` : exclu du comptage WP3-GREP).

/// Nom d'hôte du second poste, tel qu'écrit dans `kvu.toml` (profil `gamer`, clé `hotes`).
pub const HOTE_GAMER: &str = "pc-nvidia-exemple";

/// Fichier d'alerte du moniteur de charge du poste principal, tel qu'écrit dans `kvu.toml` (clé `alerte`).
pub const ALERTE_ACEMAGIC: &str = "/chemin/vers/projet/outils/machines/ALERTE.txt";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commun::Machine;

    #[test]
    fn constantes_egales_aux_valeurs_de_kvu_toml() {
        let cfg = crate::config::actuelle();
        assert!(
            cfg.profil(Machine::Gamer).hotes.iter().any(|h| h == HOTE_GAMER),
            "HOTE_GAMER absent des hôtes du profil gamer de kvu.toml"
        );
        assert_eq!(cfg.profil(Machine::Acemagic).alerte, ALERTE_ACEMAGIC);
    }
}
