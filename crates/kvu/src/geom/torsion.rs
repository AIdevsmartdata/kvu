//! Torsion de 't Hooft n_{μν} ∈ ℤ₂ (M4). DÉPLACÉE TELLE QUELLE depuis `su2/torsion.rs` (socle `geom`, kvu v0.2) et
//! réexportée par `su2::torsion` : `crate::su2::torsion::Torsion` reste valide, aucun appelant n'est modifié.
//! L'agrafe et les plaquettes tordues du réseau cubique restent dans `su2::torsion`.

use crate::commun::{Erreur, Resultat};
use serde::{Deserialize, Serialize};

/// Torsion de 't Hooft n_{μν} ∈ ℤ₂, représentée par les 6 plans indépendants (μ<ν).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Torsion {
    plans: [bool; 6],
}

impl Torsion {
    /// Paires (μ,ν), μ<ν, ordre canonique.
    pub const PLANS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    /// Torsion triviale (tous les plans périodiques).
    pub fn triviale() -> Self {
        Torsion { plans: [false; 6] }
    }

    /// Analyse une spécification : "aucune" ou liste de paires "01,23".
    pub fn analyser(t: &str) -> Resultat<Self> {
        let s = t.trim();
        if s.is_empty() {
            return Err(Erreur::Config(
                "torsion vide : attendu « aucune » ou une liste de paires 01,23,...".into(),
            ));
        }
        if s == "aucune" {
            return Ok(Torsion::triviale());
        }
        let mut plans = [false; 6];
        for token in s.split(',') {
            let token = token.trim();
            if token.len() != 2 {
                return Err(Erreur::Config(format!(
                    "torsion : « {token} » n'est pas une paire de deux chiffres (ex: 01)"
                )));
            }
            let d0 = token.as_bytes()[0];
            let d1 = token.as_bytes()[1];
            if !d0.is_ascii_digit() || !d1.is_ascii_digit() {
                return Err(Erreur::Config(format!(
                    "torsion : « {token} » contient des caractères non numériques"
                )));
            }
            let mu = usize::from(d0 - b'0');
            let nu = usize::from(d1 - b'0');
            if mu > 3 || nu > 3 {
                return Err(Erreur::Config(format!(
                    "torsion : les directions doivent être entre 0 et 3, reçu {mu} et {nu}"
                )));
            }
            if mu == nu {
                return Err(Erreur::Config(format!("torsion : μ = ν = {mu} interdit")));
            }
            // normaliser
            let (a, b) = if mu < nu { (mu, nu) } else { (nu, mu) };
            let idx = Torsion::PLANS
                .iter()
                .position(|&(x, y)| x == a && y == b)
                .ok_or_else(|| Erreur::Config(format!("torsion : plan {a}{b} hors domaine")))?;
            if plans[idx] {
                return Err(Erreur::Config(format!("torsion : plan {a}{b} dupliqué")));
            }
            plans[idx] = true;
        }
        Ok(Torsion { plans })
    }

    pub fn est_triviale(&self) -> bool {
        self.plans.iter().all(|&b| !b)
    }

    pub fn tordu(&self, mu: usize, nu: usize) -> bool {
        if mu >= 4 || nu >= 4 || mu == nu {
            return false;
        }
        let (a, b) = if mu < nu { (mu, nu) } else { (nu, mu) };
        match Torsion::PLANS.iter().position(|&(x, y)| x == a && y == b) {
            Some(idx) => self.plans[idx],
            None => false,
        }
    }

    pub fn texte(&self) -> String {
        if self.est_triviale() {
            return "aucune".to_string();
        }
        let mut parties = Vec::new();
        for (i, &present) in self.plans.iter().enumerate() {
            if present {
                let (m, n) = Torsion::PLANS[i];
                parties.push(format!("{m}{n}"));
            }
        }
        parties.join(",")
    }

    pub fn json(&self) -> serde_json::Value {
        let plans: Vec<String> = self
            .plans
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| format!("{}{}", Torsion::PLANS[i].0, Torsion::PLANS[i].1))
            .collect();
        serde_json::json!({
            "plans": plans,
            "pile": "coins (L-1, L-1) de chaque plan tordu (convention kvu M4)",
            "facteur": -1
        })
    }

    /// z de la plaquette de coin y dans le plan {mu,nu}, réseau de côté l.
    pub fn z(&self, y: &[usize; 4], mu: usize, nu: usize, l: usize) -> f64 {
        if l == 0 || !self.tordu(mu, nu) {
            return 1.0;
        }
        // la pile est définie par y_mu = l-1 et y_nu = l-1
        if y[mu] == l - 1 && y[nu] == l - 1 {
            -1.0
        } else {
            1.0
        }
    }
}
