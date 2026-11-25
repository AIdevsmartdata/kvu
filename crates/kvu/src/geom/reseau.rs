//! Réseau régulier anisotrope ℓ0 × ℓ1 × ℓ2 × ℓ3 : `GeoA`, coloriage (`GeoA::couleurs`), `indice_plan`, pile de torsion en table
//! (`Pile`). DÉPLACÉ TEL QUEL depuis `su2/tube/reseau.rs` (branche `module/tube-2026-11-11`, commit 32dfd27) : le tube
//! les réexporte (`pub use crate::geom::reseau::{indice_plan, GeoA, Pile, PLANS};`).
//!
//! Indexation : site s = ((x0·ℓ1 + x1)·ℓ2 + x2)·ℓ3 + x3, lien li = μ·V + s (celle de `heritage::su2_hb::Geo` à côtés égaux).
//! Côtés impairs : coloriage à trois classes (somme mod 3 de colorations d'axes ; deux sites voisins n'ont jamais la même couleur).

use crate::commun::{Erreur, Resultat};
use crate::geom::torsion::Torsion;

/// Plans (μ<ν) dans l'ordre canonique de `Torsion::PLANS`.
pub const PLANS: [(usize, usize); 6] = Torsion::PLANS;

/// Indice du plan {μ, ν} (μ ≠ ν) dans `PLANS`.
#[inline]
pub fn indice_plan(mu: usize, nu: usize) -> usize {
    let (a, b) = if mu < nu { (mu, nu) } else { (nu, mu) };
    match (a, b) {
        (0, 1) => 0,
        (0, 2) => 1,
        (0, 3) => 2,
        (1, 2) => 3,
        (1, 3) => 4,
        _ => 5,
    }
}

/// Couture DÉCALÉE (tore oblique, M7) : en faisant le tour de l'axe `axe`, le champ revient décalé du vecteur entier `vec`
/// (réduit modulo les côtés, `vec[axe] = 0`). Le quotient est ℤ⁴/Γ, Γ = ⟨ℓ_i e_i (i ≠ axe), ℓ_axe e_axe + vec⟩ : site
/// (ℓ_axe − 1, y) a pour voisin « haut » (0, y + vec) ; Z = Tr(S^vec T^ℓ_axe) (D-1, lemme 1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Couture {
    pub axe: usize,
    pub vec: [usize; 4],
}

/// Réseau périodique anisotrope ℓ0 × ℓ1 × ℓ2 × ℓ3 (couture décalée facultative : `couture = None` par défaut, chemins inchangés).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeoA {
    pub d: [usize; 4],
    pub v: usize,
    pub st: [usize; 4],
    pub couture: Option<Couture>,
}

impl GeoA {
    pub fn nouvelle(d: [usize; 4]) -> Resultat<GeoA> {
        if d.iter().any(|&x| x < 2) {
            return Err(Erreur::Config(format!(
                "côtés {d:?} : chaque côté doit être ≥ 2"
            )));
        }
        // O3 (relecture du 11/11, défaut hérité du tube 32dfd27:reseau.rs:51) : produits VÉRIFIÉS, un débordement est
        // une erreur de configuration et non une panique ou un V faux.
        let v = d
            .iter()
            .try_fold(1usize, |a, &x| a.checked_mul(x))
            .filter(|v| v.checked_mul(4).is_some_and(|n| n <= u32::MAX as usize))
            .ok_or_else(|| {
                Erreur::Config(format!(
                    "côtés {d:?} : 4V dépasse un compteur Philox sur 32 bits"
                ))
            })?;
        Ok(GeoA {
            d,
            v,
            st: [d[1] * d[2] * d[3], d[2] * d[3], d[3], 1],
            couture: None,
        })
    }

    /// Réseau à couture décalée : le tour de l'axe `axe` ramène le champ décalé de `vec` (entiers relatifs, réduits modulo les
    /// côtés). Un décalage nul modulo les côtés rend le réseau ORDINAIRE (`couture = None`, mêmes bits que `nouvelle`).
    pub fn avec_couture(d: [usize; 4], axe: usize, vec: [i64; 4]) -> Resultat<GeoA> {
        let mut g = GeoA::nouvelle(d)?;
        if axe > 3 {
            return Err(Erreur::Config(format!("couture : axe {axe} hors 0..3")));
        }
        let mut r = [0usize; 4];
        for i in 0..4 {
            r[i] = vec[i].rem_euclid(d[i] as i64) as usize;
        }
        if r[axe] != 0 {
            return Err(Erreur::Config(format!(
                "couture : la composante du décalage selon l'axe recollé ({axe}) doit être nulle modulo {}",
                d[axe]
            )));
        }
        if r.iter().any(|&x| x != 0) {
            g.couture = Some(Couture { axe, vec: r });
        }
        Ok(g)
    }

    /// Site x + vec (composantes modulo les côtés), ou x − vec (`signe` < 0).
    fn translater(&self, s: usize, vec: &[usize; 4], signe: i32) -> usize {
        let mut x = self.coords(s);
        for i in 0..4 {
            let l = self.d[i];
            x[i] = if signe >= 0 { (x[i] + vec[i]) % l } else { (x[i] + l - vec[i] % l) % l };
        }
        self.site(&x)
    }

    /// Voisin d'un site à couture décalée (calculé depuis `s` seul : ne lit aucune coordonnée passée par l'appelant).
    fn voisin_couture(&self, c: &Couture, s: usize, d: usize, haut: bool) -> usize {
        let xd = (s / self.st[d]) % self.d[d];
        if haut {
            if xd == self.d[d] - 1 {
                let b = s - (self.d[d] - 1) * self.st[d];
                if d == c.axe { self.translater(b, &c.vec, 1) } else { b }
            } else {
                s + self.st[d]
            }
        } else if xd == 0 {
            let b = s + (self.d[d] - 1) * self.st[d];
            if d == c.axe { self.translater(b, &c.vec, -1) } else { b }
        } else {
            s - self.st[d]
        }
    }

    /// Analyse « l0,l1,l2,l3 ».
    pub fn analyser(t: &str) -> Resultat<GeoA> {
        let v: Vec<usize> = t
            .split(',')
            .map(|x| {
                x.trim()
                    .parse::<usize>()
                    .map_err(|_| Erreur::Config(format!("--cotes : « {x} » n'est pas un entier")))
            })
            .collect::<Resultat<_>>()?;
        if v.len() != 4 {
            return Err(Erreur::Config(format!(
                "--cotes : 4 côtés attendus (l0,l1,l2,l3), reçu « {t} »"
            )));
        }
        GeoA::nouvelle([v[0], v[1], v[2], v[3]])
    }

    pub fn texte(&self) -> String {
        format!("{},{},{},{}", self.d[0], self.d[1], self.d[2], self.d[3])
    }

    #[inline]
    pub fn nb_up(&self, s: usize, d: usize, xd: usize) -> usize {
        if let Some(c) = &self.couture {
            return self.voisin_couture(c, s, d, true);
        }
        if xd == self.d[d] - 1 {
            s - (self.d[d] - 1) * self.st[d]
        } else {
            s + self.st[d]
        }
    }
    #[inline]
    pub fn nb_dn(&self, s: usize, d: usize, xd: usize) -> usize {
        if let Some(c) = &self.couture {
            return self.voisin_couture(c, s, d, false);
        }
        if xd == 0 {
            s + (self.d[d] - 1) * self.st[d]
        } else {
            s - self.st[d]
        }
    }
    #[inline]
    pub fn coords(&self, s: usize) -> [usize; 4] {
        [
            s / self.st[0],
            (s / self.st[1]) % self.d[1],
            (s / self.st[2]) % self.d[2],
            s % self.d[3],
        ]
    }
    #[inline]
    pub fn site(&self, x: &[usize; 4]) -> usize {
        x[0] * self.st[0] + x[1] * self.st[1] + x[2] * self.st[2] + x[3]
    }
    /// Site x + k·μ̂ (k ≥ 0, périodique).
    #[inline]
    pub fn decaler(&self, s: usize, mu: usize, k: usize) -> usize {
        if let Some(c) = &self.couture {
            return (0..k).fold(s, |t, _| self.voisin_couture(c, t, mu, true));
        }
        let mut x = self.coords(s);
        x[mu] = (x[mu] + k) % self.d[mu];
        self.site(&x)
    }

    pub fn tous_pairs(&self) -> bool {
        self.d.iter().all(|x| x % 2 == 0)
    }

    /// Classes de couleur des sites : parité (2 classes, ordre 0 puis 1) si tous les côtés sont pairs, sinon coloriage à trois classes
    /// (au plus 3 couleurs, voir ci-dessous). Deux sites voisins (x, x ± ν̂) n'ont jamais la même couleur.
    // Code déplacé tel quel du tube (32dfd27) : seule la boucle indicée sur μ est signalée par clippy.
    #[allow(clippy::needless_range_loop)]
    pub fn couleurs(&self) -> Vec<Vec<usize>> {
        if self.couture.is_some() {
            return self.couleurs_couture();
        }
        if self.tous_pairs() {
            let mut c = vec![
                Vec::with_capacity(self.v / 2),
                Vec::with_capacity(self.v / 2),
            ];
            for s in 0..self.v {
                let x = self.coords(s);
                c[(x[0] + x[1] + x[2] + x[3]) & 1].push(s);
            }
            return c;
        }
        // Côtés impairs : coloriage à TROIS classes du tore (Sabidussi : χ(G□H) = max, somme mod 3 de colorations d'axes propres).
        // Par axe de longueur L : c(i) = i mod 2 pour i < L − 1 et c(L − 1) = 2 si L impair (cycle impair : 3 couleurs).
        let mut classes = vec![Vec::new(); 3];
        for s in 0..self.v {
            classes[self.couleur_trois(&self.coords(s))].push(s);
        }
        classes.retain(|c| !c.is_empty());
        classes
    }

    /// Couture décalée : la parité ou les trois classes si elles restent propres sur le tore oblique (`verifier_classes`), sinon
    /// coloriage glouton (premier site libre dans l'ordre des indices ; au plus 9 classes).
    fn couleurs_couture(&self) -> Vec<Vec<usize>> {
        let mut essais: Vec<Vec<Vec<usize>>> = Vec::new();
        if self.tous_pairs() {
            let mut c = vec![Vec::new(), Vec::new()];
            for s in 0..self.v {
                let x = self.coords(s);
                c[(x[0] + x[1] + x[2] + x[3]) & 1].push(s);
            }
            essais.push(c);
        }
        let mut t = vec![Vec::new(); 3];
        for s in 0..self.v {
            t[self.couleur_trois(&self.coords(s))].push(s);
        }
        t.retain(|c| !c.is_empty());
        essais.push(t);
        for c in essais {
            if self.verifier_classes(&c).is_ok() {
                return c;
            }
        }
        let mut col = vec![usize::MAX; self.v];
        let mut classes: Vec<Vec<usize>> = Vec::new();
        for s in 0..self.v {
            let mut pris = [false; 16];
            for d in 0..4 {
                for haut in [true, false] {
                    let n = self.voisin_couture(self.couture.as_ref().unwrap(), s, d, haut);
                    if col[n] != usize::MAX {
                        pris[col[n]] = true;
                    }
                }
            }
            let k = pris.iter().position(|&b| !b).unwrap_or(0);
            col[s] = k;
            if k >= classes.len() {
                classes.resize(k + 1, Vec::new());
            }
            classes[k].push(s);
        }
        classes
    }

    /// Vérifie un coloriage de sites (cahier O4 b, I1 et I2) : chaque site dans exactement UNE classe, aucune classe vide, et
    /// propreté (jamais deux sites voisins x, x + ν̂, y compris par le recouvrement périodique, dans la même classe). Retourne
    /// la description du premier défaut.
    pub fn verifier_classes(&self, classes: &[Vec<usize>]) -> Result<(), String> {
        let mut col = vec![usize::MAX; self.v];
        for (k, c) in classes.iter().enumerate() {
            if c.is_empty() {
                return Err(format!("classe {k} vide"));
            }
            for &s in c {
                if s >= self.v || col[s] != usize::MAX {
                    return Err(format!("site {s} hors réseau ou dans deux classes"));
                }
                col[s] = k;
            }
        }
        if let Some(s) = col.iter().position(|&c| c == usize::MAX) {
            return Err(format!("site {s} sans classe"));
        }
        for s in 0..self.v {
            for mu in 0..4 {
                if self.d[mu] > 1 && col[s] == col[self.decaler(s, mu, 1)] {
                    return Err(format!("sites voisins {s} et {} (axe {mu}) de même classe {}", self.decaler(s, mu, 1), col[s]));
                }
            }
        }
        Ok(())
    }

    /// Couleur (0, 1 ou 2) du site de coordonnées `x` dans le coloriage à trois classes (voir `couleurs`).
    pub fn couleur_trois(&self, x: &[usize; 4]) -> usize {
        (0..4)
            .map(|a| {
                let l = self.d[a];
                if l % 2 == 1 && x[a] == l - 1 {
                    2
                } else {
                    x[a] % 2
                }
            })
            .sum::<usize>()
            % 3
    }
}

/// Facteurs z_p ∈ {±1} de chaque plaquette (plan, coin), en table (6·V). Construite depuis une `Torsion` (pile de coin :
/// z = −1 sur les plaquettes {μ,ν} de coin x_μ = ℓ_μ − 1, x_ν = ℓ_ν − 1, convention kvu M4 = convention de lib_tube.py),
/// ou modifiée à la main (mutants).
#[derive(Clone, Debug, PartialEq)]
pub struct Pile {
    pub z: Vec<f64>,
    pub v: usize,
}

impl Pile {
    pub fn depuis_torsion(g: &GeoA, t: &Torsion) -> Pile {
        let mut z = vec![1.0; 6 * g.v];
        for (p, &(mu, nu)) in PLANS.iter().enumerate() {
            if !t.tordu(mu, nu) {
                continue;
            }
            for s in 0..g.v {
                let x = g.coords(s);
                if x[mu] == g.d[mu] - 1 && x[nu] == g.d[nu] - 1 {
                    z[p * g.v + s] = -1.0;
                }
            }
        }
        Pile { z, v: g.v }
    }
    #[inline]
    pub fn z(&self, plan: usize, s: usize) -> f64 {
        self.z[plan * self.v + s]
    }
    pub fn retourner(&mut self, plan: usize, s: usize) {
        self.z[plan * self.v + s] = -self.z[plan * self.v + s];
    }
    pub fn negatifs(&self, plan: usize) -> usize {
        self.z[plan * self.v..(plan + 1) * self.v]
            .iter()
            .filter(|&&x| x < 0.0)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debordement_refuse() {
        // O3 : 65536⁴ = 2⁶⁴ déborderait ; 256⁴ donne 4V = 2³⁴ > u32::MAX ; refus code 2 dans les deux cas.
        for d in [[65536; 4], [256; 4], [usize::MAX, 2, 2, 2]] {
            assert_eq!(GeoA::nouvelle(d).unwrap_err().code_sortie(), 2, "{d:?}");
        }
        assert_eq!(
            GeoA::analyser("65536,65536,65536,65536")
                .unwrap_err()
                .code_sortie(),
            2
        );
        assert_eq!(GeoA::nouvelle([2, 3, 4, 5]).unwrap().v, 120);
    }

    #[test]
    fn coords_site_voisins() {
        let g = GeoA::nouvelle([4, 6, 3, 2]).unwrap();
        for s in 0..g.v {
            let x = g.coords(s);
            assert_eq!(g.site(&x), s);
            for mu in 0..4 {
                let up = g.nb_up(s, mu, x[mu]);
                let mut y = x;
                y[mu] = (y[mu] + 1) % g.d[mu];
                assert_eq!(up, g.site(&y));
                assert_eq!(g.nb_dn(up, mu, y[mu]), s);
                assert_eq!(g.decaler(s, mu, 1), up);
            }
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // test déplacé tel quel du tube (32dfd27)
    fn coloriage_valide() {
        for d in [
            [4, 4, 4, 4],
            [6, 6, 2, 2],
            [4, 4, 3, 3],
            [5, 4, 3, 2],
            [8, 8, 3, 3],
            [3, 3, 3, 3],
        ] {
            let g = GeoA::nouvelle(d).unwrap();
            let c = g.couleurs();
            let mut col = vec![usize::MAX; g.v];
            for (k, cl) in c.iter().enumerate() {
                for &s in cl {
                    assert_eq!(col[s], usize::MAX);
                    col[s] = k;
                }
            }
            for s in 0..g.v {
                let x = g.coords(s);
                for mu in 0..4 {
                    let t = g.nb_up(s, mu, x[mu]);
                    assert_ne!(col[s], col[t], "{d:?} s={s} mu={mu}");
                }
            }
            if g.tous_pairs() {
                assert_eq!(c.len(), 2);
            }
        }
    }
}
