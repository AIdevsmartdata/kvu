//! Temps d'autocorrélation intégré, avec fenêtre automatique de Sokal.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResTau {
    pub tau: f64,
    pub fenetre: usize,
    pub dtau: f64,
}

/// Port de `analyse_plaquette.py::tau_int`, par sommes directes paresseuses.
pub fn tau_int(x: &[f64], c: f64) -> ResTau {
    let n = x.len();
    let moy = x.iter().sum::<f64>() / n as f64;
    let centres: Vec<_> = x.iter().map(|&xi| xi - moy).collect();
    let v = centres.iter().map(|xi| xi * xi).sum::<f64>() / n as f64;
    if n < 8 || v == 0.0 {
        return ResTau {
            tau: 0.5,
            fenetre: 0,
            dtau: 0.0,
        };
    }
    let mut tau = 0.5;
    let mut fenetre = 0;
    for w in 1..n / 4 {
        let covariance = centres[..n - w]
            .iter()
            .zip(&centres[w..])
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / (n - w) as f64;
        tau += covariance / v;
        fenetre = w;
        if w as f64 >= c * tau {
            break;
        }
    }
    // Cette comparaison conserve NaN, comme max(t, 0.5) en Python.
    if tau < 0.5 {
        tau = 0.5;
    }
    ResTau {
        tau,
        fenetre,
        dtau: tau * (2.0 * (2.0 * fenetre as f64 + 1.0) / n as f64).sqrt(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StatSerie {
    pub moy: f64,
    pub err: f64,
    pub tau: f64,
    pub fenetre: usize,
    pub dtau: f64,
    pub n: usize,
    pub sd: f64,
}

/// Moyenne et erreur corrigée de l'autocorrélation ; variance à n − 1.
/// Une série vide a une moyenne NaN ; pour n < 2, sd et err valent NaN.
pub fn stat_serie(x: &[f64]) -> StatSerie {
    let n = x.len();
    let moy = x.iter().sum::<f64>() / n as f64;
    let variance = if n < 2 {
        f64::NAN
    } else {
        x.iter().map(|xi| (xi - moy).powi(2)).sum::<f64>() / (n - 1) as f64
    };
    let ResTau { tau, fenetre, dtau } = tau_int(x, 6.0);
    StatSerie {
        moy,
        err: (2.0 * tau * variance / n as f64).sqrt(),
        tau,
        fenetre,
        dtau,
        n,
        sd: variance.sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniforme(graine: &mut u64) -> f64 {
        *graine = graine
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*graine >> 11) as f64 / (1_u64 << 53) as f64
    }

    #[test]
    fn constante_et_petites_series() {
        assert_eq!(
            tau_int(&[3.0; 100], 6.0),
            ResTau {
                tau: 0.5,
                fenetre: 0,
                dtau: 0.0
            }
        );
        assert_eq!(tau_int(&[], 6.0).tau, 0.5);
        assert_eq!(tau_int(&[1.0, 2.0], 6.0).fenetre, 0);
        assert!(stat_serie(&[]).moy.is_nan());
        assert!(stat_serie(&[1.0]).err.is_nan());
        let stat = stat_serie(&[3.0; 100]);
        assert_eq!((stat.moy, stat.sd, stat.err, stat.n), (3.0, 0.0, 0.0, 100));
        let stat = stat_serie(&[1.0, 3.0]);
        assert_eq!(stat.moy, 2.0);
        assert!((stat.sd - 2.0_f64.sqrt()).abs() < 1e-15);
        assert_eq!(stat.err, 1.0);
    }

    #[test]
    fn iid_et_ar1() {
        let mut graine = 12345;
        let iid: Vec<_> = (0..200_000).map(|_| uniforme(&mut graine) - 0.5).collect();
        let res = tau_int(&iid, 6.0);
        assert!((res.tau - 0.5).abs() < 0.03, "{res:?}");
        assert!(res.fenetre <= 4);
        let mut courant = 0.0;
        let ar: Vec<_> = iid
            .iter()
            .map(|bruit| {
                courant = 0.9 * courant + bruit;
                courant
            })
            .collect();
        let res = tau_int(&ar, 6.0);
        assert!((res.tau - 9.5).abs() < 0.25 * 9.5, "{res:?}");
    }

    #[test]
    fn reference_plaquette() {
        let chemin = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/donnees/rg_flot/serie_plaquette_L12_b2.3_c1.txt");
        let contenu = match std::fs::read_to_string(chemin) {
            Ok(contenu) => contenu,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
            Err(e) => panic!("Lecture de la référence : {e}"),
        };
        let x: Vec<f64> = contenu
            .lines()
            .map(|ligne| ligne.trim().parse().expect("Valeur de référence"))
            .collect();
        assert_eq!(x.len(), 2000);
        let res = tau_int(&x, 6.0);
        assert!((res.tau - 1.6900620161191204).abs() < 1e-10, "{res:?}");
        assert_eq!(res.fenetre, 11);
        assert!((res.dtau - 0.25631059521907873).abs() < 1e-10);
    }
}
