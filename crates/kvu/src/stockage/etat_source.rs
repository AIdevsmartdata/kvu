//! git_head embarqué dans le binaire : décision PURE, partagée par `build.rs` (inclus par `#[path]`) et par
//! les tests du crate. Propriétaire : T8 (correctif du vérificateur, 04/10).
//!
//! Un binaire construit depuis un arbre MODIFIÉ ne doit jamais porter le commit nu : avant ce correctif, un
//! noyau modifié sur une copie rsync vers l'autre poste (ou un `cargo build` avec `KEV_GIT_HEAD` exporté à la main)
//! donnait le même `git_head` que le commit propre ; seul le `source_hash` changeait.
//!
//! Règles :
//! - arbre git : le commit est celui de `git rev-parse HEAD` (la variable déclarée est ignorée si elle diffère) ;
//!   « +sale » si un fichier de la recette `Unifiee` est modifié ou non suivi ;
//! - copie hors git : le commit DÉCLARÉ (`KEV_GIT_HEAD`) ; si `KEV_SOURCE_HASH_ATTENDU` (empreinte de la recette
//!   `Unifiee` au commit déclaré, calculée sur l'arbre propre) est fourni, il est comparé à l'empreinte calculée :
//!   égal ⇒ commit nu (vérifié), différent ⇒ « +sale » ; non fourni ⇒ « +non_verifie » ;
//! - ni git ni déclaration : aucun git_head (« non_enregistre » à l'exécution).

/// État git de l'arbre de compilation : (commit HEAD, au moins un fichier de la recette modifié ou non suivi).
pub type EtatGit<'a> = Option<(&'a str, bool)>;

/// git_head à embarquer. `declare` = KEV_GIT_HEAD de l'environnement, `attendu` = KEV_SOURCE_HASH_ATTENDU,
/// `calcule` = empreinte `Unifiee` de l'arbre compilé.
pub fn git_head_embarque(git: EtatGit, declare: Option<&str>, attendu: Option<&str>, calcule: &str) -> Option<String> {
    fn propre(s: Option<&str>) -> Option<&str> {
        s.map(str::trim).filter(|s| !s.is_empty())
    }
    match (git, propre(declare)) {
        (Some((head, sale)), _) => Some(if sale { format!("{head}+sale") } else { head.to_string() }),
        (None, Some(d)) => {
            // une déclaration « abc+sale » reste sale ; on juge le commit seul
            let commit = d.split('+').next().unwrap_or(d);
            let deja_sale = d.contains("+sale");
            Some(match propre(attendu) {
                _ if deja_sale => format!("{commit}+sale"),
                Some(a) if a.eq_ignore_ascii_case(calcule) => commit.to_string(),
                Some(_) => format!("{commit}+sale"),
                None => format!("{commit}+non_verifie"),
            })
        }
        (None, None) => None,
    }
}

/// Vrai si le git_head embarqué désigne un arbre propre ET vérifié (commit nu, sans suffixe).
pub fn git_head_propre(g: &str) -> bool {
    !g.is_empty() && g != "non_enregistre" && !g.contains('+')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbre_git_propre_ou_sale() {
        assert_eq!(git_head_embarque(Some(("c1e83d8", false)), None, None, "h").as_deref(), Some("c1e83d8"));
        assert_eq!(git_head_embarque(Some(("c1e83d8", true)), None, None, "h").as_deref(), Some("c1e83d8+sale"));
        // la déclaration ne peut pas masquer un arbre git sale ni imposer un autre commit
        assert_eq!(
            git_head_embarque(Some(("c1e83d8", true)), Some("c1e83d8"), None, "h").as_deref(),
            Some("c1e83d8+sale")
        );
        assert_eq!(git_head_embarque(Some(("aaaa", false)), Some("bbbb"), None, "h").as_deref(), Some("aaaa"));
    }

    /// Le cas du vérificateur : copie rsync, noyau modifié, `KEV_GIT_HEAD=c1e83d8` déclaré.
    #[test]
    fn copie_hors_git_noyau_modifie() {
        let propre = "a".repeat(64);
        let modifie = "b".repeat(64);
        assert_eq!(
            git_head_embarque(None, Some("c1e83d8"), Some(&propre), &modifie).as_deref(),
            Some("c1e83d8+sale")
        );
        assert_eq!(git_head_embarque(None, Some("c1e83d8"), Some(&propre), &propre).as_deref(), Some("c1e83d8"));
        assert_eq!(git_head_embarque(None, Some("c1e83d8"), None, &propre).as_deref(), Some("c1e83d8+non_verifie"));
        assert_eq!(git_head_embarque(None, Some("c1e83d8+sale"), Some(&propre), &propre).as_deref(), Some("c1e83d8+sale"));
        assert_eq!(git_head_embarque(None, Some("  "), None, &propre), None);
        assert_eq!(git_head_embarque(None, None, None, &propre), None);
    }

    #[test]
    fn propre_seulement_sans_suffixe() {
        assert!(git_head_propre("c1e83d8"));
        assert!(!git_head_propre("c1e83d8+sale"));
        assert!(!git_head_propre("c1e83d8+non_verifie"));
        assert!(!git_head_propre("non_enregistre"));
    }
}
