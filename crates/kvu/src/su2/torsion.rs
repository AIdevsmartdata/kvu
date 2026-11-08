//! Torsion de 't Hooft pour SU(2) (M4).

use crate::commun::{Erreur, Resultat};
use crate::heritage::su2_hb::{k_final, k_reduce, qadd, qdag, qmul, qscale, Geo, Q};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Torsion de 't Hooft n_{μν} ∈ ℤ₂, représentée par les 6 plans indépendants (μ<ν).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Torsion {
    plans: [bool; 6],
}

impl Torsion {
    /// Paires (μ,ν), μ<ν, ordre canonique.
    pub const PLANS: [(usize, usize); 6] = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)];

    /// Torsion triviale (tous les plans périodiques).
    pub fn triviale() -> Self {
        Torsion { plans: [false; 6] }
    }

    /// Analyse une spécification : "aucune" ou liste de paires "01,23".
    pub fn analyser(t: &str) -> Resultat<Self> {
        let s = t.trim();
        if s.is_empty() {
            return Err(Erreur::Config("torsion vide : attendu « aucune » ou une liste de paires 01,23,...".into()));
        }
        if s == "aucune" {
            return Ok(Torsion::triviale());
        }
        let mut plans = [false; 6];
        for token in s.split(',') {
            let token = token.trim();
            if token.len() != 2 {
                return Err(Erreur::Config(format!("torsion : « {token} » n'est pas une paire de deux chiffres (ex: 01)")));
            }
            let d0 = token.as_bytes()[0];
            let d1 = token.as_bytes()[1];
            if !d0.is_ascii_digit() || !d1.is_ascii_digit() {
                return Err(Erreur::Config(format!("torsion : « {token} » contient des caractères non numériques")));
            }
            let mu = usize::from(d0 - b'0');
            let nu = usize::from(d1 - b'0');
            if mu > 3 || nu > 3 {
                return Err(Erreur::Config(format!("torsion : les directions doivent être entre 0 et 3, reçu {mu} et {nu}")));
            }
            if mu == nu {
                return Err(Erreur::Config(format!("torsion : μ = ν = {mu} interdit")));
            }
            // normaliser
            let (a, b) = if mu < nu { (mu, nu) } else { (nu, mu) };
            let idx = Torsion::PLANS.iter().position(|&(x, y)| x == a && y == b)
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
        let plans: Vec<String> = self.plans.iter().enumerate()
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

/// Agrafe tordue pour le lien (x, μ). Égale à `su2_hb::staple` si torsion triviale.
pub fn agrafe_tordue(u: &[Q], g: &Geo, mu: usize, s: usize, x: &[usize; 4], t: &Torsion) -> Q {
    let ld = |nu: usize, site: usize| u[nu * g.v + site];
    let mut a: Q = [0.0; 4];
    let smu = g.nb_up(s, mu, x[mu]);
    for nu in 0..4 {
        if nu == mu {
            continue;
        }
        let snu = g.nb_up(s, nu, x[nu]);
        let sdn = g.nb_dn(s, nu, x[nu]);
        let smu_dn = g.nb_dn(smu, nu, x[nu]);
        // plaquette t : coin x
        let t_plaq = qmul(qmul(ld(nu, smu), qdag(ld(mu, snu))), qdag(ld(nu, s)));
        let zt = t.z(x, mu, nu, g.l);
        // plaquette b : coin x - ν̂
        let mut xb = *x;
        if xb[nu] == 0 {
            xb[nu] = g.l - 1;
        } else {
            xb[nu] -= 1;
        }
        let b_plaq = qmul(qmul(qdag(ld(nu, smu_dn)), qdag(ld(mu, sdn))), ld(nu, sdn));
        let zb = t.z(&xb, mu, nu, g.l);
        a = qadd(a, qadd(qscale(t_plaq, zt), qscale(b_plaq, zb)));
    }
    a
}

/// Plaquette moyenne tordue : P_tw = (1/6V) Σ_p z_p (1/2) Tr U_p.
pub fn plaquette_tordue(g: &Geo, u: &[Q], t: &Torsion) -> f64 {
    let ld = |nu: usize, site: usize| u[nu * g.v + site];
    let site: Vec<f64> = (0..g.v)
        .into_par_iter()
        .map(|s| {
            let x = g.coords(s);
            let mut acc = 0.0;
            for mu in 0..4 {
                let smu = g.nb_up(s, mu, x[mu]);
                for nu in mu + 1..4 {
                    let snu = g.nb_up(s, nu, x[nu]);
                    let p = qmul(qmul(ld(mu, s), ld(nu, smu)), qmul(qdag(ld(mu, snu)), qdag(ld(nu, s))));
                    let z = t.z(&x, mu, nu, g.l);
                    acc += z * p[0];
                }
            }
            acc
        })
        .collect();
    let part = k_reduce(&site);
    k_final(&part, 1.0 / (6.0 * g.v as f64))
}

/// Moyennes par plan de z_p (1/2) Tr U_p. Ordre : PLANS.
pub fn plaquettes_par_plan(g: &Geo, u: &[Q], t: &Torsion) -> [f64; 6] {
    let ld = |nu: usize, site: usize| u[nu * g.v + site];
    let contributions: Vec<[f64; 6]> = (0..g.v)
        .into_par_iter()
        .map(|s| {
            let x = g.coords(s);
            let mut sums = [0.0; 6];
            for (idx, &(mu, nu)) in Torsion::PLANS.iter().enumerate() {
                let smu = g.nb_up(s, mu, x[mu]);
                let snu = g.nb_up(s, nu, x[nu]);
                let p = qmul(qmul(ld(mu, s), ld(nu, smu)), qmul(qdag(ld(mu, snu)), qdag(ld(nu, s))));
                let z = t.z(&x, mu, nu, g.l);
                sums[idx] = z * p[0];
            }
            sums
        })
        .collect();
    let mut totals = [0.0; 6];
    for c in contributions {
        for (t, x) in totals.iter_mut().zip(c) {
            *t += x;
        }
    }
    for t in totals.iter_mut() {
        *t /= g.v as f64;
    }
    totals
}

/// Action classique du fond abélien uniforme d'un plan tordu sur L⁴ :
/// 2·(L²·sin(π/(2L²)))², forme stable sans annulation pour L grand.
/// Valeur pour L = 4 : 4,91896821677301… (identique à L⁴(1−cos(π/L²))).
pub fn action_fond_abelien_un_plan(l: usize) -> f64 {
    let lf = l as f64;
    let l2 = lf * lf;
    let s = l2 * (std::f64::consts::PI / (2.0 * l2)).sin();
    2.0 * s * s
}

pub const NOM_FICHIER_TORSION: &str = "torsion.json";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyse_et_texte() {
        let t = Torsion::analyser("01,23").unwrap();
        assert_eq!(t.texte(), "01,23");
        let t2 = Torsion::analyser("10").unwrap();
        assert_eq!(t2.texte(), "01");
        let t3 = Torsion::analyser("aucune").unwrap();
        assert!(t3.est_triviale());
        assert!(Torsion::analyser("").is_err());
        assert!(Torsion::analyser("00").is_err());
        assert!(Torsion::analyser("04").is_err());
        assert!(Torsion::analyser("01,10").is_err());
    }

    #[test]
    fn symetrie_de_tordu() {
        let t = Torsion::analyser("01").unwrap();
        assert!(t.tordu(0, 1));
        assert!(t.tordu(1, 0));
        assert!(!t.tordu(0, 2));
        assert!(!t.tordu(2, 2));
    }

    #[test]
    fn z_est_moins_un_sur_la_pile() {
        let t = Torsion::analyser("01").unwrap();
        let l = 4;
        // -1 uniquement si y0=3 et y1=3
        let y = [3usize, 3, 0, 0];
        assert_eq!(t.z(&y, 0, 1, l), -1.0);
        let y2 = [3usize, 2, 0, 0];
        assert_eq!(t.z(&y2, 0, 1, l), 1.0);
        let y3 = [2usize, 3, 0, 0];
        assert_eq!(t.z(&y3, 0, 1, l), 1.0);
        // plan non tordu : toujours 1
        assert_eq!(t.z(&y, 0, 2, l), 1.0);
    }

    #[test]
    fn json_contient_les_plans() {
        let t = Torsion::analyser("01,23").unwrap();
        let v = t.json();
        assert_eq!(v["plans"], serde_json::json!(["01", "23"]));
        assert_eq!(v["facteur"], serde_json::json!(-1));
    }

    #[test]
    fn action_fond_abelien_valeur_connue() {
        let e = action_fond_abelien_un_plan(4);
        assert!((e - 4.91896821677301).abs() < 1e-14, "e = {e}");
    }
}
