//! Spécification d'un opérateur de blocage de facteur 3 (WP22) : `3centre` (Bałaban, WP6), `dec`, `ape:<α>:<n>`, `flot:<t>[:<ε>]`,
//! `agr:<c>`. Définitions, famille « rayon de lissage ⩽ bloc » et supports : `docs/BLOCAGE_QUASI_LOCAL.md` (§3-§5), écrit avant ce code.

use crate::commun::{Erreur, Resultat};
use serde_json::{json, Value};

/// Pas de flot par défaut (Lüscher 1006.4518, annexe C : erreurs ≈ 1e-6 pour ε = 0,01).
pub const EPS_DEFAUT: f64 = 0.01;
/// Tolérance relative de « t multiple de ε ».
pub const TOL_MULTIPLE: f64 = 1e-9;
/// Rayon de lissage maximal (côté du bloc L_b = 3) : r² ≤ 9.
pub const RAYON2_MAX: f64 = 9.0;
/// Pas de flot maximal dans la famille (stabilité linéaire du RK3 : ε ≤ 0,157, calcul D2 ; marge).
pub const EPS_MAX_FAMILLE: f64 = 0.1;
/// α maximal dans la famille (noyau linéarisé positif : 1 − 8α/6 ≥ 0, calcul D5).
pub const ALPHA_MAX_FAMILLE: f64 = 0.75;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Operateur {
    /// Moyenne non linéaire de Bałaban (géométrie `3centre` de `rg::bloc`), inchangée.
    #[default]
    TroisCentre,
    /// Décimation pure : produit des 3 liens alignés.
    Dec,
    /// n pas APE (mise à jour simultanée, poids α), puis décimation.
    Ape { alpha: f64, n: usize },
    /// Flot de Wilson RK3 de `n` pas de taille `eps` (t = n·eps), puis décimation.
    Flot { t: f64, eps: f64, n: usize },
    /// Lien de longueur 3 + c × 6 agrafes de longueur 3, projeté.
    Agr { c: f64 },
}

fn nombre(t: &str, quoi: &str) -> Resultat<f64> {
    let v: f64 = t.trim().parse().map_err(|_| Erreur::Config(format!("opérateur : {quoi} « {t} » n'est pas un nombre")))?;
    if !v.is_finite() {
        return Err(Erreur::Config(format!("opérateur : {quoi} = {t} non fini")));
    }
    Ok(v)
}

impl Operateur {
    /// Analyse une spécification ; tout refus ici est INCONDITIONNEL (même avec `--hors-famille`).
    pub fn analyser(spec: &str) -> Resultat<Operateur> {
        let parties: Vec<&str> = spec.trim().split(':').collect();
        let formes = "3centre | dec | ape:<α>:<n> | flot:<t>[:<ε>] | agr:<c>";
        match parties.as_slice() {
            ["3centre"] => Ok(Operateur::TroisCentre),
            ["dec"] => Ok(Operateur::Dec),
            ["ape", a, n] => {
                let alpha = nombre(a, "α")?;
                if !(alpha > 0.0 && alpha < 1.0) {
                    return Err(Erreur::Config(format!("opérateur ape : α = {alpha} hors de ]0, 1[")));
                }
                let n: usize = n
                    .trim()
                    .parse()
                    .map_err(|_| Erreur::Config(format!("opérateur ape : n = « {n} » doit être un entier ≥ 0")))?;
                Ok(Operateur::Ape { alpha, n })
            }
            ["flot", t] => Self::flot(nombre(t, "t")?, EPS_DEFAUT),
            ["flot", t, e] => Self::flot(nombre(t, "t")?, nombre(e, "ε")?),
            ["agr", c] => {
                let c = nombre(c, "c")?;
                if c < 0.0 {
                    return Err(Erreur::Config(format!("opérateur agr : c = {c} < 0")));
                }
                Ok(Operateur::Agr { c })
            }
            _ => Err(Erreur::Config(format!("opérateur « {spec} » inconnu ou mal formé (formes : {formes})"))),
        }
    }

    fn flot(t: f64, eps: f64) -> Resultat<Operateur> {
        if t < 0.0 {
            return Err(Erreur::Config(format!("opérateur flot : t = {t} < 0")));
        }
        if eps <= 0.0 {
            return Err(Erreur::Config(format!("opérateur flot : ε = {eps} ≤ 0")));
        }
        let n = (t / eps).round();
        if (n * eps - t).abs() > TOL_MULTIPLE * t.max(1.0) || n > u32::MAX as f64 {
            return Err(Erreur::Config(format!("opérateur flot : t = {t} doit être un multiple de ε = {eps}")));
        }
        Ok(Operateur::Flot { t, eps, n: n as usize })
    }

    /// Forme canonique (`flot` toujours avec son ε).
    pub fn spec(&self) -> String {
        match self {
            Operateur::TroisCentre => "3centre".into(),
            Operateur::Dec => "dec".into(),
            Operateur::Ape { alpha, n } => format!("ape:{alpha}:{n}"),
            Operateur::Flot { t, eps, .. } => format!("flot:{t}:{eps}"),
            Operateur::Agr { c } => format!("agr:{c}"),
        }
    }

    pub fn nom(&self) -> &'static str {
        match self {
            Operateur::TroisCentre => "3centre",
            Operateur::Dec => "dec",
            Operateur::Ape { .. } => "ape",
            Operateur::Flot { .. } => "flot",
            Operateur::Agr { .. } => "agr",
        }
    }

    pub fn est_trois_centre(&self) -> bool {
        matches!(self, Operateur::TroisCentre)
    }

    /// Rayon quadratique moyen de lissage (doc §4) : 0, √(8t) ou √(8 t_APE) = √(4αn/3).
    pub fn rayon_lissage(&self) -> f64 {
        match *self {
            Operateur::TroisCentre | Operateur::Dec | Operateur::Agr { .. } => 0.0,
            Operateur::Flot { t, .. } => (8.0 * t).sqrt(),
            Operateur::Ape { alpha, n } => (4.0 * alpha * n as f64 / 3.0).sqrt(),
        }
    }

    /// (dans la famille ?, règle appliquée) — règles EXACTES de la doc §4.
    pub fn famille(&self) -> (bool, &'static str) {
        match *self {
            Operateur::TroisCentre | Operateur::Dec | Operateur::Agr { .. } => (true, "aucun lissage préalable (rayon 0)"),
            Operateur::Flot { t, eps, .. } => (8.0 * t <= RAYON2_MAX && eps <= EPS_MAX_FAMILLE, "8t ≤ 9 (√(8t) ≤ 3) et ε ≤ 0,1"),
            Operateur::Ape { alpha, n } => (
                4.0 * alpha * n as f64 <= 27.0 && alpha <= ALPHA_MAX_FAMILLE,
                "4αn ≤ 27 (√(4αn/3) ≤ 3) et α ≤ 3/4",
            ),
        }
    }

    /// Refus d'un opérateur hors famille, sauf `--hors-famille` explicite.
    pub fn verifier_famille(&self, hors_famille: bool) -> Resultat<()> {
        let (dans, regle) = self.famille();
        if dans || hors_famille {
            return Ok(());
        }
        Err(Erreur::Config(format!(
            "opérateur {} hors de la famille « rayon de lissage ⩽ bloc » (règle : {regle} ; rayon {:.4}) : refusé (utiliser --hors-famille pour un mutant)",
            self.spec(),
            self.rayon_lissage()
        )))
    }

    /// Rayon d_P du support (boule autour du chemin droit) : `dec` 0, `ape` n, `flot` 3N ; `None` pour `agr` et `3centre` (liste).
    pub fn rayon_support(&self) -> Option<usize> {
        match *self {
            Operateur::Dec => Some(0),
            Operateur::Ape { n, .. } => Some(n),
            Operateur::Flot { n, .. } => Some(3 * n),
            Operateur::Agr { .. } | Operateur::TroisCentre => None,
        }
    }

    /// Paramètres pour les manifestes et les documents.
    pub fn parametres_json(&self) -> Value {
        let (dans, regle) = self.famille();
        let mut v = json!({
            "spec": self.spec(),
            "nom": self.nom(),
            "rayon_lissage": self.rayon_lissage(),
            "famille": {"dans_famille": dans, "regle": regle},
        });
        match *self {
            Operateur::Ape { alpha, n } => {
                v["alpha"] = json!(alpha);
                v["n"] = json!(n);
                v["t_ape"] = json!(alpha * n as f64 / 6.0);
            }
            Operateur::Flot { t, eps, n } => {
                v["t"] = json!(t);
                v["eps"] = json!(eps);
                v["n_pas"] = json!(n);
                v["integrateur"] = json!("RK3 de Lüscher (1006.4518, éq. (C.2)-(C.3)), obs::ref_cpu::flot_n_pas");
            }
            Operateur::Agr { c } => v["c"] = json!(c),
            Operateur::TroisCentre | Operateur::Dec => {}
        }
        v
    }
}

impl std::str::FromStr for Operateur {
    type Err = Erreur;
    fn from_str(s: &str) -> Resultat<Operateur> {
        Operateur::analyser(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formes_valides() {
        assert_eq!(Operateur::analyser("3centre").unwrap(), Operateur::TroisCentre);
        assert_eq!(Operateur::analyser("dec").unwrap(), Operateur::Dec);
        assert_eq!(Operateur::analyser("ape:0.5:13").unwrap(), Operateur::Ape { alpha: 0.5, n: 13 });
        assert_eq!(Operateur::analyser("ape:0.5:0").unwrap(), Operateur::Ape { alpha: 0.5, n: 0 });
        assert_eq!(Operateur::analyser("flot:1").unwrap(), Operateur::Flot { t: 1.0, eps: 0.01, n: 100 });
        assert_eq!(Operateur::analyser("flot:0.3:0.1").unwrap(), Operateur::Flot { t: 0.3, eps: 0.1, n: 3 });
        assert_eq!(Operateur::analyser("flot:0").unwrap(), Operateur::Flot { t: 0.0, eps: 0.01, n: 0 });
        assert_eq!(Operateur::analyser("flot:1.125:0.005").unwrap(), Operateur::Flot { t: 1.125, eps: 0.005, n: 225 });
        assert_eq!(Operateur::analyser("agr:1").unwrap(), Operateur::Agr { c: 1.0 });
        assert_eq!(Operateur::analyser("agr:0").unwrap(), Operateur::Agr { c: 0.0 });
        for s in ["3centre", "dec", "ape:0.5:13", "flot:1:0.01", "flot:0.02:0.01", "agr:0.25"] {
            assert_eq!(Operateur::analyser(&Operateur::analyser(s).unwrap().spec()).unwrap(), Operateur::analyser(s).unwrap());
        }
        assert_eq!(Operateur::analyser("flot:1").unwrap().spec(), "flot:1:0.01");
        assert_eq!(Operateur::default(), Operateur::TroisCentre);
    }

    #[test]
    fn formes_invalides() {
        for s in [
            "", "3coin", "dec:1", "ape", "ape:0.5", "ape:0:3", "ape:1:3", "ape:1.5:3", "ape:-0.5:3", "ape:0.5:-1", "ape:0.5:1.5",
            "ape:nan:3", "ape:0.5:13:1", "flot", "flot:-1", "flot:1:0", "flot:1:-0.01", "flot:0.25:0.1", "flot:inf", "flot:1:x",
            "agr", "agr:-1", "agr:nan", "inconnu", "APE:0.5:6",
        ] {
            assert!(matches!(Operateur::analyser(s), Err(Erreur::Config(_))), "« {s} » aurait dû être refusé");
        }
    }

    #[test]
    fn famille_et_frontieres() {
        let f = |s: &str| Operateur::analyser(s).unwrap().famille().0;
        assert!(f("ape:0.5:13") && !f("ape:0.5:14") && !f("ape:0.5:40"));
        assert!(f("ape:0.75:9"), "4·0,75·9 = 27 : frontière incluse");
        assert!(!f("ape:0.8:1"), "α > 3/4");
        assert!(f("flot:1.125:0.005"), "8t = 9 : frontière incluse");
        assert!(!f("flot:1.13") && !f("flot:1:0.2") && f("flot:1:0.1"));
        assert!(f("dec") && f("agr:4") && f("3centre"));
        let op = Operateur::analyser("ape:0.5:40").unwrap();
        assert!(op.verifier_famille(false).is_err() && op.verifier_famille(true).is_ok());
        assert!((Operateur::analyser("ape:0.5:6").unwrap().rayon_lissage() - 2.0).abs() < 1e-15);
        assert!((Operateur::analyser("flot:1.125:0.005").unwrap().rayon_lissage() - 3.0).abs() < 1e-15);
        assert_eq!(Operateur::analyser("flot:0.02:0.01").unwrap().rayon_support(), Some(6));
        assert_eq!(Operateur::analyser("ape:0.5:13").unwrap().rayon_support(), Some(13));
        assert_eq!(Operateur::analyser("agr:1").unwrap().rayon_support(), None);
    }
}
