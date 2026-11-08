//! Étalonnage des observables et inversion PCHIP sans extrapolation.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observable {
    Plaquette,
    Rectangle,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PointEtalon {
    pub beta: f64,
    pub p: f64,
    pub sigma_p: f64,
    pub r: f64,
    pub sigma_r: f64,
    pub n_chaines: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Courbe {
    pub l: usize,
    pub points: Vec<PointEtalon>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Inversion {
    /// Cible encadrée : couplage, dérivée PCHIP et erreur interpolée dans l'observable.
    Dans {
        beta: f64,
        pente: f64,
        sigma_cal: f64,
        bas: f64,
        haut: f64,
    },
    /// Cible hors plage : seul le nœud extrême est fourni, sans extrapolation.
    HorsEtalonnage {
        cote_bas: bool,
        beta_extreme: f64,
        valeur_extreme: f64,
        pente_extreme: f64,
        sigma_extreme: f64,
    },
}

impl Courbe {
    pub fn nouvelle(l: usize, points: Vec<PointEtalon>) -> Result<Courbe, String> {
        let courbe = Self { l, points };
        courbe.verifier_points()?;
        Ok(courbe)
    }

    // Les champs sont publics : revérifier protège aussi les courbes modifiées.
    fn verifier_points(&self) -> Result<(), String> {
        if self.points.len() < 2 {
            return Err("Une courbe exige au moins deux points".into());
        }
        if self.points.iter().any(|p| {
            [p.beta, p.p, p.sigma_p, p.r, p.sigma_r]
                .iter()
                .any(|v| !v.is_finite())
        }) {
            return Err("Les couplages, observables et erreurs doivent être finis".into());
        }
        if self.points.windows(2).any(|p| p[0].beta >= p[1].beta) {
            return Err("Les couplages doivent être strictement croissants".into());
        }
        Ok(())
    }

    pub fn valeurs(&self, obs: Observable) -> Vec<f64> {
        self.points
            .iter()
            .map(|p| match obs {
                Observable::Plaquette => p.p,
                Observable::Rectangle => p.r,
            })
            .collect()
    }

    pub fn sigmas(&self, obs: Observable) -> Vec<f64> {
        self.points
            .iter()
            .map(|p| match obs {
                Observable::Plaquette => p.sigma_p,
                Observable::Rectangle => p.sigma_r,
            })
            .collect()
    }

    pub fn verifier_monotone(&self, obs: Observable) -> Result<(), String> {
        self.verifier_points()?;
        if self.valeurs(obs).windows(2).any(|y| y[0] >= y[1]) {
            return Err("L'observable doit être strictement croissant avec bêta".into());
        }
        Ok(())
    }

    pub fn evaluer(&self, obs: Observable, beta: f64) -> Option<(f64, f64)> {
        self.pchip(obs).ok()?.evaluer(beta)
    }

    pub fn evaluer_lineaire(&self, obs: Observable, beta: f64) -> Option<f64> {
        self.verifier_points().ok()?;
        let betas: Vec<_> = self.points.iter().map(|p| p.beta).collect();
        let y = self.valeurs(obs);
        let k = intervalle(&betas, beta)?;
        let valeur = lineaire(betas[k], betas[k + 1], y[k], y[k + 1], beta);
        valeur.is_finite().then_some(valeur)
    }

    pub fn inverser(&self, obs: Observable, cible: f64) -> Result<Inversion, String> {
        self.verifier_monotone(obs)?;
        if !cible.is_finite() {
            return Err("La cible doit être finie".into());
        }
        let courbe = self.pchip(obs)?;
        let sigmas = self.sigmas(obs);
        let dernier = courbe.y.len() - 1;
        if cible < courbe.y[0] || cible > courbe.y[dernier] {
            let cote_bas = cible < courbe.y[0];
            let k = if cote_bas { 0 } else { dernier };
            return Ok(Inversion::HorsEtalonnage {
                cote_bas,
                beta_extreme: courbe.x[k],
                valeur_extreme: courbe.y[k],
                pente_extreme: courbe.d[k],
                sigma_extreme: sigmas[k],
            });
        }
        let k = intervalle(&courbe.y, cible).ok_or("Cible impossible à encadrer")?;
        let beta = if cible == courbe.y[k] {
            courbe.x[k]
        } else if cible == courbe.y[k + 1] {
            courbe.x[k + 1]
        } else {
            let (mut bas, mut haut) = (courbe.x[k], courbe.x[k + 1]);
            for _ in 0..100 {
                let milieu = bas + (haut - bas) * 0.5;
                let (valeur, _) = courbe.evaluer(milieu).ok_or("Évaluation PCHIP non finie")?;
                if valeur < cible {
                    bas = milieu;
                } else {
                    haut = milieu;
                }
            }
            bas + (haut - bas) * 0.5
        };
        let (_, pente) = courbe.evaluer(beta).ok_or("Dérivée PCHIP non finie")?;
        let sigma_cal = lineaire(
            courbe.y[k],
            courbe.y[k + 1],
            sigmas[k],
            sigmas[k + 1],
            cible,
        );
        if !sigma_cal.is_finite() {
            return Err("Erreur d'étalonnage non finie".into());
        }
        Ok(Inversion::Dans {
            beta,
            pente,
            sigma_cal,
            bas: courbe.x[k],
            haut: courbe.x[k + 1],
        })
    }

    pub fn inverser_lineaire(&self, obs: Observable, cible: f64) -> Result<Option<f64>, String> {
        self.verifier_monotone(obs)?;
        if !cible.is_finite() {
            return Err("La cible doit être finie".into());
        }
        let y = self.valeurs(obs);
        let Some(k) = intervalle(&y, cible) else {
            return Ok(None);
        };
        let beta = lineaire(
            y[k],
            y[k + 1],
            self.points[k].beta,
            self.points[k + 1].beta,
            cible,
        );
        if !beta.is_finite() {
            return Err("Interpolation linéaire non finie".into());
        }
        Ok(Some(beta))
    }

    fn pchip(&self, obs: Observable) -> Result<Pchip, String> {
        self.verifier_points()?;
        let x: Vec<_> = self.points.iter().map(|p| p.beta).collect();
        let y = self.valeurs(obs);
        let h: Vec<_> = x.windows(2).map(|x| x[1] - x[0]).collect();
        let delta: Vec<_> = y
            .windows(2)
            .zip(&h)
            .map(|(y, h)| (y[1] - y[0]) / h)
            .collect();
        let mut d = vec![0.0; y.len()];
        if y.len() == 2 {
            d.fill(delta[0]);
        } else {
            for (k, (h, delta)) in h.windows(2).zip(delta.windows(2)).enumerate() {
                if delta[0] != 0.0 && delta[1] != 0.0 && delta[0].signum() == delta[1].signum() {
                    let w1 = 2.0 * h[1] + h[0];
                    let w2 = h[1] + 2.0 * h[0];
                    let inverse_harmonique = (w1 / delta[0] + w2 / delta[1]) / (w1 + w2);
                    d[k + 1] = 1.0 / inverse_harmonique;
                }
            }
            d[0] = derivee_bord(h[0], h[1], delta[0], delta[1]);
            let k = h.len() - 1;
            let dernier = d.len() - 1;
            d[dernier] = derivee_bord(h[k], h[k - 1], delta[k], delta[k - 1]);
        }
        if h.iter().chain(&delta).chain(&d).any(|v| !v.is_finite()) {
            return Err("Écart ou pente PCHIP non fini".into());
        }
        Ok(Pchip { x, y, d })
    }
}

fn derivee_bord(h0: f64, h1: f64, delta0: f64, delta1: f64) -> f64 {
    let d = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1);
    if d.signum() != delta0.signum() {
        0.0
    } else if delta0.signum() != delta1.signum() && d.abs() > 3.0 * delta0.abs() {
        3.0 * delta0
    } else {
        d
    }
}

fn intervalle(noeuds: &[f64], cible: f64) -> Option<usize> {
    if noeuds.len() < 2
        || !cible.is_finite()
        || cible < noeuds[0]
        || cible > noeuds[noeuds.len() - 1]
    {
        return None;
    }
    Some(
        noeuds
            .partition_point(|&x| x <= cible)
            .saturating_sub(1)
            .min(noeuds.len() - 2),
    )
}

fn lineaire(x0: f64, x1: f64, y0: f64, y1: f64, x: f64) -> f64 {
    if x == x0 {
        return y0;
    }
    if x == x1 {
        return y1;
    }
    let t = (x - x0) / (x1 - x0);
    (1.0 - t) * y0 + t * y1
}

struct Pchip {
    x: Vec<f64>,
    y: Vec<f64>,
    d: Vec<f64>,
}

impl Pchip {
    fn evaluer(&self, beta: f64) -> Option<(f64, f64)> {
        let k = intervalle(&self.x, beta)?;
        if beta == self.x[k] {
            return Some((self.y[k], self.d[k]));
        }
        if beta == self.x[k + 1] {
            return Some((self.y[k + 1], self.d[k + 1]));
        }
        let h = self.x[k + 1] - self.x[k];
        let t = (beta - self.x[k]) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let valeur = (2.0 * t3 - 3.0 * t2 + 1.0) * self.y[k]
            + (t3 - 2.0 * t2 + t) * h * self.d[k]
            + (-2.0 * t3 + 3.0 * t2) * self.y[k + 1]
            + (t3 - t2) * h * self.d[k + 1];
        let pente = (6.0 * t2 - 6.0 * t) * (self.y[k] - self.y[k + 1]) / h
            + (3.0 * t2 - 4.0 * t + 1.0) * self.d[k]
            + (3.0 * t2 - 2.0 * t) * self.d[k + 1];
        (valeur.is_finite() && pente.is_finite()).then_some((valeur, pente))
    }
}

/// Moyenne inversement pondérée par les variances, sans exclure de chaîne.
pub fn combiner_chaines(valeurs: &[(f64, f64)]) -> Option<(f64, f64)> {
    if valeurs.is_empty()
        || valeurs
            .iter()
            .any(|(m, s)| !m.is_finite() || !s.is_finite() || *s <= 0.0)
    {
        return None;
    }
    if valeurs.len() == 1 {
        return Some(valeurs[0]);
    }
    // Mise à l'échelle commune pour éviter de former 1 / sigma² à très petit sigma.
    let echelle = valeurs
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::INFINITY, f64::min);
    let somme = valeurs
        .iter()
        .map(|(_, s)| (echelle / s).powi(2))
        .sum::<f64>();
    let moyenne = valeurs
        .iter()
        .map(|(m, s)| m * ((echelle / s).powi(2) / somme))
        .sum::<f64>();
    let erreur = echelle / somme.sqrt();
    (moyenne.is_finite() && erreur.is_finite()).then_some((moyenne, erreur))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn courbe() -> Courbe {
        let points = [(0.0, 0.0), (1.0, 2.0), (3.0, 3.0), (4.0, 5.0)]
            .into_iter()
            .enumerate()
            .map(|(i, (beta, p))| PointEtalon {
                beta,
                p,
                sigma_p: 0.1 * (i + 1) as f64,
                r: 2.0 * p,
                sigma_r: 0.2,
                n_chaines: 4,
            })
            .collect();
        Courbe::nouvelle(4, points).expect("Courbe valide")
    }

    // Référence indépendante : coefficients en puissances de (bêta − bêta_k),
    // dérivées calculées à la main : [5/2, 6/7, 6/7, 5/2].
    fn reference(beta: f64) -> (f64, f64) {
        let (x, y, h, delta, a, b) = if beta <= 1.0 {
            (0.0, 0.0, 1.0, 2.0, 2.5, 6.0 / 7.0)
        } else if beta <= 3.0 {
            (1.0, 2.0, 2.0, 0.5, 6.0 / 7.0, 6.0 / 7.0)
        } else {
            (3.0, 3.0, 1.0, 2.0, 6.0 / 7.0, 2.5)
        };
        let c = (3.0 * delta - 2.0 * a - b) / h;
        let d = (a + b - 2.0 * delta) / (h * h);
        let z = beta - x;
        (
            y + z * (a + z * (c + z * d)),
            a + z * (2.0 * c + z * 3.0 * d),
        )
    }

    #[test]
    fn noeuds_monotonie_et_reference() {
        let courbe = courbe();
        for obs in [Observable::Plaquette, Observable::Rectangle] {
            for (point, y) in courbe.points.iter().zip(courbe.valeurs(obs)) {
                assert!((courbe.evaluer(obs, point.beta).expect("Nœud").0 - y).abs() < 1e-15);
            }
        }
        let mut precedent = -1.0;
        for i in 0..10_000 {
            let beta = 4.0 * i as f64 / 9999.0;
            let (y, d) = courbe
                .evaluer(Observable::Plaquette, beta)
                .expect("Dans la courbe");
            assert!(y > precedent && d >= 0.0);
            precedent = y;
            let (yr, dr) = reference(beta);
            assert!((y - yr).abs() < 4e-15, "bêta={beta}, {y} contre {yr}");
            assert!((d - dr).abs() < 1e-14);
        }
        for (beta, attendu) in [(0.5, 135.0 / 112.0), (3.5, 425.0 / 112.0)] {
            assert!(
                (courbe
                    .evaluer(Observable::Plaquette, beta)
                    .expect("Dans la courbe")
                    .0
                    - attendu)
                    .abs()
                    < 1e-15
            );
        }
    }

    #[test]
    fn derivee_et_inversion() {
        let courbe = courbe();
        let mut graine = 321_u64;
        for _ in 0..200 {
            graine = graine.wrapping_mul(6364136223846793005).wrapping_add(1);
            let beta = 0.0001 + 3.9998 * (graine >> 11) as f64 / (1_u64 << 53) as f64;
            for obs in [Observable::Plaquette, Observable::Rectangle] {
                let (y, d) = courbe.evaluer(obs, beta).expect("Dans la courbe");
                let epsilon = 1e-6;
                let difference = (courbe.evaluer(obs, beta + epsilon).expect("Voisin").0
                    - courbe.evaluer(obs, beta - epsilon).expect("Voisin").0)
                    / (2.0 * epsilon);
                assert!((d - difference).abs() < 1e-6);
                let Inversion::Dans {
                    beta: inverse,
                    pente,
                    ..
                } = courbe.inverser(obs, y).expect("Inversion")
                else {
                    panic!("Cible intérieure")
                };
                assert!((beta - inverse).abs() < 1e-12);
                assert!((d - pente).abs() < 1e-12);
            }
        }
        for point in &courbe.points {
            let Inversion::Dans {
                beta, sigma_cal, ..
            } = courbe
                .inverser(Observable::Plaquette, point.p)
                .expect("Nœud")
            else {
                panic!("Nœud intérieur")
            };
            assert_eq!(beta, point.beta);
            assert_eq!(sigma_cal, point.sigma_p);
            assert_eq!(
                courbe
                    .inverser_lineaire(Observable::Plaquette, point.p)
                    .expect("Nœud"),
                Some(point.beta)
            );
        }
        let Inversion::Dans {
            sigma_cal,
            bas,
            haut,
            ..
        } = courbe
            .inverser(Observable::Plaquette, 2.25)
            .expect("Inversion")
        else {
            panic!("Cible intérieure")
        };
        assert!((sigma_cal - 0.225).abs() < 1e-15);
        assert_eq!((bas, haut), (1.0, 3.0));
    }

    #[test]
    fn aucune_extrapolation_et_validation() {
        let courbe = courbe();
        for (cible, cote_bas, beta_extreme, valeur_extreme, sigma_extreme) in
            [(-1.0, true, 0.0, 0.0, 0.1), (6.0, false, 4.0, 5.0, 0.4)]
        {
            assert_eq!(
                courbe
                    .inverser(Observable::Plaquette, cible)
                    .expect("Hors plage"),
                Inversion::HorsEtalonnage {
                    cote_bas,
                    beta_extreme,
                    valeur_extreme,
                    pente_extreme: 2.5,
                    sigma_extreme,
                }
            );
            assert_eq!(
                courbe
                    .inverser_lineaire(Observable::Plaquette, cible)
                    .expect("Hors plage"),
                None
            );
        }
        for beta in [-0.01, 4.01, f64::NAN, f64::INFINITY] {
            assert!(courbe.evaluer(Observable::Plaquette, beta).is_none());
            assert!(courbe
                .evaluer_lineaire(Observable::Plaquette, beta)
                .is_none());
        }
        assert!(courbe.inverser(Observable::Plaquette, f64::NAN).is_err());
        assert!(Courbe::nouvelle(4, vec![]).is_err());
        assert!(Courbe::nouvelle(4, courbe.points[..1].to_vec()).is_err());
        let mut points = courbe.points.clone();
        points[1].beta = points[0].beta;
        assert!(Courbe::nouvelle(4, points).is_err());
        for champ in 0..5 {
            let mut points = courbe.points.clone();
            match champ {
                0 => points[0].beta = f64::NAN,
                1 => points[0].p = f64::INFINITY,
                2 => points[0].r = f64::NAN,
                3 => points[0].sigma_p = f64::NAN,
                _ => points[0].sigma_r = f64::INFINITY,
            }
            assert!(Courbe::nouvelle(4, points).is_err());
        }
        let mut non_monotone = courbe.clone();
        non_monotone.points[1].p = 0.0;
        assert!(non_monotone
            .verifier_monotone(Observable::Plaquette)
            .is_err());
        assert!(non_monotone.inverser(Observable::Plaquette, 1.0).is_err());
        assert!(non_monotone
            .inverser_lineaire(Observable::Plaquette, 1.0)
            .is_err());
        let invalide = Courbe {
            l: 4,
            points: vec![],
        };
        assert!(invalide.evaluer(Observable::Plaquette, 0.0).is_none());
        assert!(invalide.inverser(Observable::Plaquette, 0.0).is_err());
    }

    #[test]
    fn deux_noeuds_et_corrections_des_bords() {
        let mut courbe = courbe();
        courbe.points.truncate(2);
        assert_eq!(
            courbe.evaluer(Observable::Plaquette, 0.25),
            Some((0.5, 2.0))
        );
        assert_eq!(
            courbe.evaluer_lineaire(Observable::Plaquette, 0.25),
            Some(0.5)
        );
        assert_eq!(
            courbe
                .inverser_lineaire(Observable::Plaquette, 0.5)
                .expect("Inversion"),
            Some(0.25)
        );
        assert_eq!(derivee_bord(1.0, 1.0, 1.0, 10.0), 0.0);
        assert_eq!(derivee_bord(1.0, 1.0, 1.0, -10.0), 3.0);
        courbe.points[1].p = courbe.points[0].p;
        assert_eq!(courbe.evaluer(Observable::Plaquette, 0.5), Some((0.0, 0.0)));
        let mut non_monotone = self::courbe();
        non_monotone.points[2].p = -1.0;
        let pchip = non_monotone
            .pchip(Observable::Plaquette)
            .expect("PCHIP non monotone");
        assert_eq!((pchip.d[1], pchip.d[2]), (0.0, 0.0));
    }

    #[test]
    fn combinaison_des_chaines() {
        let (moy, err) = combiner_chaines(&[(1.0, 1.0), (3.0, 2.0)]).expect("Chaînes valides");
        assert!((moy - 1.4).abs() < 1e-15);
        assert!((err - 1.0 / 1.25_f64.sqrt()).abs() < 1e-15);
        assert_eq!(combiner_chaines(&[(2.0, 0.3)]), Some((2.0, 0.3)));
        assert!(combiner_chaines(&[]).is_none());
        for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(combiner_chaines(&[(1.0, 1.0), (2.0, sigma)]).is_none());
        }
        assert!(combiner_chaines(&[(f64::NAN, 1.0)]).is_none());
        assert!(
            combiner_chaines(&[(1.0, 1e-200), (3.0, 1e-200)])
                .expect("Petites erreurs")
                .1
                > 0.0
        );
    }
}
