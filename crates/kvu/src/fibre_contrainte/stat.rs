//! Analyse statistique des gates de WP12a (GATES v4-T2 partie 1 §1.0) : thermalisation de 20 %, τ_int par fenêtre de Sokal
//! (c = 6), erreurs par jackknife à K = 40 blocs de longueur ≥ 10 τ_int, stationnarité, comparaisons de Student à 39 ddl.
//!
//! Statuts non verts (jamais convertis en PASS) : `Puissance`, `NonStationnaire`, `TauNonEstimable`.

/// K : nombre de blocs du jackknife (gelé).
pub const K_BLOCS: usize = 40;
/// Constante de la fenêtre de Sokal (gelée).
pub const C_SOKAL: f64 = 6.0;
/// Quantiles bilatéraux de Student à 39 ddl, α_famille = 0,01 (gelés, `scipy.stats.t.isf`).
pub const T39_M1: f64 = 2.7079;
pub const T39_M3: f64 = 3.1269;
pub const T39_M4: f64 = 3.2322;
pub const T39_M10: f64 = 3.5581;
pub const T39_M16: f64 = 3.7207;
/// Seuils effectivement utilisés par les gates de WP12a, par statistique. Les valeurs NOMINALES (Student t₃₉) sont celles de GATES §1.0 ; les seuils
/// EMPIRIQUES (avenant WP12a-2) viennent de la calibration de l'outil (FIB-STAT-CALIB, graine de dérivation indépendante).
#[derive(Clone, Copy, Debug)]
pub struct Seuils {
    /// moyenne d'une série contre une valeur exacte, familles m = 1, 3, 4, 16
    pub moy_m1: f64,
    pub moy_m3: f64,
    pub moy_m4: f64,
    pub moy_m16: f64,
    /// différence de deux séries de même τ (m = 3)
    pub diff_m3: f64,
    /// différence de deux séries de τ très différents (m = 3), p. ex. route A contre référence
    pub diff_asym_m3: f64,
    /// intercept et résidu du point du milieu d'un ajustement à trois points (m = 16)
    pub intercept_m16: f64,
    pub residu_m16: f64,
}

pub const SEUILS_NOMINAUX: Seuils = Seuils { moy_m1: T39_M1, moy_m3: T39_M3, moy_m4: T39_M4, moy_m16: T39_M16, diff_m3: T39_M3, diff_asym_m3: T39_M3, intercept_m16: T39_M16, residu_m16: T39_M16 };

/// Seuils retenus pour les verdicts (avenant WP12a-2 : quantiles empiriques de la calibration ; 3,14 est une valeur de quantile, pas π).
#[allow(clippy::approx_constant)]
pub const SEUILS: Seuils = Seuils { moy_m1: 2.82, moy_m3: 3.23, moy_m4: 3.36, moy_m16: 3.88, diff_m3: 3.14, diff_asym_m3: 3.20, intercept_m16: 3.91, residu_m16: 3.91 };

/// Équivalent 3σ pour la stationnarité (t₃₉).
pub const T39_STATIONNAIRE: f64 = 3.2041;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Statut {
    Ok,
    /// moins de K blocs de longueur ≥ 10 τ_int
    Puissance,
    NonStationnaire,
    TauNonEstimable,
}

impl Statut {
    pub fn texte(self) -> &'static str {
        match self {
            Statut::Ok => "OK",
            Statut::Puissance => "PUISSANCE",
            Statut::NonStationnaire => "NON-STATIONNAIRE",
            Statut::TauNonEstimable => "TAU-NON-ESTIMABLE",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Analyse {
    /// valeurs retenues après thermalisation
    pub n: usize,
    pub moyenne: f64,
    /// erreur type par jackknife à K blocs
    pub erreur: f64,
    pub tau_int: f64,
    pub longueur_bloc: usize,
    pub statut: Statut,
    /// moyennes des K blocs (pour les ajustements et comparaisons jackknife)
    pub blocs: Vec<f64>,
}

/// τ_int de Sokal : τ(W) = ½ + Σ_{t=1}^{W} ρ(t), W minimal tel que W ≥ c τ(W) ; `None` si W dépasse n/10.
pub fn tau_int(x: &[f64]) -> Option<f64> {
    let n = x.len();
    if n < 20 {
        return None;
    }
    let m = x.iter().sum::<f64>() / n as f64;
    let c0 = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64;
    if c0 <= 0.0 {
        return Some(0.5);
    }
    let wmax = n / 10;
    let mut tau = 0.5;
    for w in 1..=wmax {
        let mut s = 0.0;
        for i in 0..n - w {
            s += (x[i] - m) * (x[i + w] - m);
        }
        tau += s / (n - w) as f64 / c0;
        if w as f64 >= C_SOKAL * tau {
            return Some(tau);
        }
    }
    None
}

/// Domaine de validité des verdicts de CAMPAGNE (avenant WP12a-2) : analyse verte et blocs d'au moins 40 τ_int (la calibration couvre les blocs de 15 τ à 60 τ, la
/// validité des répliques y est ≥ 98 % à 60 τ). Hors de ce domaine, le verdict est `SKIP PUISSANCE`.
pub fn domaine_campagne(a: &Analyse) -> bool {
    a.statut == Statut::Ok && a.longueur_bloc as f64 >= 40.0 * a.tau_int
}

/// Moyenne, erreur jackknife et statut d'une série brute (thermalisation de 20 % écartée en tête).
pub fn analyser(serie: &[f64]) -> Analyse {
    analyser_avec(serie, 0.2)
}

/// Comme [`analyser`], avec la fraction de thermalisation donnée (0 si l'appelant a déjà écarté la thermalisation).
pub fn analyser_avec(serie: &[f64], therm: f64) -> Analyse {
    let debut = (serie.len() as f64 * therm) as usize;
    let x = &serie[debut..];
    let n = x.len();
    let vide = |statut: Statut, tau: f64| Analyse {
        n,
        moyenne: if n > 0 { x.iter().sum::<f64>() / n as f64 } else { f64::NAN },
        erreur: f64::NAN,
        tau_int: tau,
        longueur_bloc: 0,
        statut,
        blocs: Vec::new(),
    };
    let tau = match tau_int(x) {
        Some(t) if t.is_finite() && t > 0.0 => t,
        _ => return vide(Statut::TauNonEstimable, f64::NAN),
    };
    let b = n / K_BLOCS;
    if (b as f64) < 10.0 * tau || b == 0 {
        return vide(Statut::Puissance, tau);
    }
    let blocs: Vec<f64> = (0..K_BLOCS).map(|k| x[k * b..(k + 1) * b].iter().sum::<f64>() / b as f64).collect();
    let moyenne = blocs.iter().sum::<f64>() / K_BLOCS as f64;
    let var = blocs.iter().map(|v| (v - moyenne) * (v - moyenne)).sum::<f64>() / (K_BLOCS * (K_BLOCS - 1)) as f64;
    if !(var.is_finite() && var > 0.0 && moyenne.is_finite()) {
        // variance nulle ou non finie : aucun dénominateur exploitable, jamais un PASS
        return vide(Statut::Puissance, tau);
    }
    let mut a = Analyse { n, moyenne, erreur: var.sqrt(), tau_int: tau, longueur_bloc: b, statut: Statut::Ok, blocs };
    // stationnarité : τ_int des deux moitiés dans un facteur 2, moyennes des deux moitiés à |t| < 3,2041
    let (h1, h2) = (&x[..n / 2], &x[n / 2..]);
    match (tau_int(h1), tau_int(h2)) {
        (Some(t1), Some(t2)) if t1.is_finite() && t2.is_finite() && t1 > 0.0 && t2 > 0.0 => {
            let r = t1.max(t2) / t1.min(t2);
            let m1 = &a.blocs[..K_BLOCS / 2];
            let m2 = &a.blocs[K_BLOCS / 2..];
            let (mm1, mm2) = (m1.iter().sum::<f64>() / 20.0, m2.iter().sum::<f64>() / 20.0);
            let v1 = m1.iter().map(|v| (v - mm1).powi(2)).sum::<f64>() / (20.0 * 19.0);
            let v2 = m2.iter().map(|v| (v - mm2).powi(2)).sum::<f64>() / (20.0 * 19.0);
            let t = (mm1 - mm2) / (v1 + v2).sqrt();
            // dénominateur nul ou non fini : statistique de stationnarité indéfinie, jamais un statut vert
            if !(t.is_finite()) || r > 2.0 || t.abs() >= T39_STATIONNAIRE {
                a.statut = Statut::NonStationnaire;
            }
        }
        _ => a.statut = Statut::TauNonEstimable,
    }
    a
}

/// t = (a − b)/√(σ_a² + σ_b² + σ_sys²) (variances par jackknife, séries indépendantes).
pub fn t_comparaison(a: &Analyse, b: &Analyse, sigma_sys: f64) -> f64 {
    (a.moyenne - b.moyenne) / (a.erreur * a.erreur + b.erreur * b.erreur + sigma_sys * sigma_sys).sqrt()
}

/// t = (a − valeur exacte)/σ_a.
pub fn t_exact(a: &Analyse, exact: f64) -> f64 {
    (a.moyenne - exact) / a.erreur
}

/// Maximum qui PROPAGE le NaN (jamais ignoré par `f64::max`).
pub fn maxn(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Ajustement jackknife O(κ) = O₀ + c κ sur trois points indépendants (GATES §1.0) : poids 1/σ² figés (ceux de l'analyse complète),
/// jackknife « delete-one-block » sur les K = 40 blocs (le bloc k de chaque chaîne est retiré ensemble ; chaînes indépendantes).
/// Rend (O₀, σ(O₀), résidu du point du milieu r, σ(r)) ; r = y_milieu − (droite passant par les deux autres points, évaluée en κ_milieu).
pub fn ajuster_jackknife(points: &[(f64, &Analyse)]) -> Option<(f64, f64, f64, f64)> {
    if points.len() != 3 || points.iter().any(|(_, a)| a.statut != Statut::Ok || a.blocs.len() != K_BLOCS) {
        return None;
    }
    let sig: Vec<f64> = points.iter().map(|(_, a)| a.erreur).collect();
    let fit = |y: &[f64]| -> (f64, f64) {
        let (mut s, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (i, (k, _)) in points.iter().enumerate() {
            let w = 1.0 / (sig[i] * sig[i]);
            s += w;
            sx += w * k;
            sy += w * y[i];
            sxx += w * k * k;
            sxy += w * k * y[i];
        }
        let d = s * sxx - sx * sx;
        ((sxx * sy - sx * sxy) / d, (s * sxy - sx * sy) / d)
    };
    let residu = |y: &[f64]| -> f64 {
        let (k0, k1, k2) = (points[0].0, points[1].0, points[2].0);
        let droite = y[0] + (y[2] - y[0]) * (k1 - k0) / (k2 - k0);
        y[1] - droite
    };
    let moy: Vec<f64> = points.iter().map(|(_, a)| a.moyenne).collect();
    let (o0, r) = (fit(&moy).0, residu(&moy));
    let (mut so, mut sr, mut vals_o, mut vals_r) = (0.0, 0.0, Vec::new(), Vec::new());
    for k in 0..K_BLOCS {
        let y: Vec<f64> = points.iter().map(|(_, a)| (a.moyenne * K_BLOCS as f64 - a.blocs[k]) / (K_BLOCS - 1) as f64).collect();
        vals_o.push(fit(&y).0);
        vals_r.push(residu(&y));
    }
    let (mo, mr) = (vals_o.iter().sum::<f64>() / K_BLOCS as f64, vals_r.iter().sum::<f64>() / K_BLOCS as f64);
    for k in 0..K_BLOCS {
        so += (vals_o[k] - mo).powi(2);
        sr += (vals_r[k] - mr).powi(2);
    }
    let f = (K_BLOCS - 1) as f64 / K_BLOCS as f64;
    let (eo, er) = ((f * so).sqrt(), (f * sr).sqrt());
    if !(eo.is_finite() && er.is_finite() && eo > 0.0 && er > 0.0 && o0.is_finite() && r.is_finite()) {
        return None;
    }
    Some((o0, eo, r, er))
}

/// Ajustement pondéré O(κ) = O₀ + c κ sur des points (κ_i, moyenne_i, erreur_i) avec jackknife sur les blocs communs
/// (chaque point doit fournir ses K blocs ; points indépendants). Rend (O₀, σ(O₀), c, σ(c)).
pub fn ajuster_droite(points: &[(f64, &Analyse)]) -> (f64, f64, f64, f64) {
    let fit = |vals: &[f64], sig: &[f64]| -> (f64, f64) {
        let (mut s, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (i, (k, _)) in points.iter().enumerate() {
            let w = 1.0 / (sig[i] * sig[i]);
            s += w;
            sx += w * k;
            sy += w * vals[i];
            sxx += w * k * k;
            sxy += w * k * vals[i];
        }
        let d = s * sxx - sx * sx;
        ((sxx * sy - sx * sxy) / d, (s * sxy - sx * sy) / d)
    };
    let sig: Vec<f64> = points.iter().map(|(_, a)| a.erreur).collect();
    let moy: Vec<f64> = points.iter().map(|(_, a)| a.moyenne).collect();
    let (o0, c) = fit(&moy, &sig);
    // erreurs : propagation exacte des poids (points indépendants) : σ²(O₀) = Σ_i (∂O₀/∂y_i)² σ_i²
    let (mut s, mut sx, mut sxx) = (0.0, 0.0, 0.0);
    for (i, (k, _)) in points.iter().enumerate() {
        let w = 1.0 / (sig[i] * sig[i]);
        s += w;
        sx += w * k;
        sxx += w * k * k;
    }
    let d = s * sxx - sx * sx;
    let (mut v0, mut vc) = (0.0, 0.0);
    for (i, (k, _)) in points.iter().enumerate() {
        let w = 1.0 / (sig[i] * sig[i]);
        let d0 = w * (sxx - sx * k) / d;
        let dc = w * (s * k - sx) / d;
        v0 += d0 * d0 * sig[i] * sig[i];
        vc += dc * dc * sig[i] * sig[i];
    }
    (o0, v0.sqrt(), c, vc.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fibre_contrainte::alea::{Flux, TAG_OUTIL};

    /// Série AR(1) de moyenne `mu`, τ_int = (1+φ)/(2(1−φ)), variance 1.
    pub(crate) fn ar1(n: usize, tau: f64, mu: f64, flux: &Flux, rep: u32) -> Vec<f64> {
        let phi = (2.0 * tau - 1.0) / (2.0 * tau + 1.0);
        let s = (1.0 - phi * phi).sqrt();
        let mut x = vec![0.0; n];
        let mut prev = flux.gauss(rep, 0, 0, TAG_OUTIL);
        for (i, xi) in x.iter_mut().enumerate() {
            let e = flux.gauss(rep, 1, i as u32, TAG_OUTIL);
            prev = phi * prev + s * e;
            *xi = mu + prev;
        }
        x
    }

    #[test]
    fn tau_int_de_l_ar1() {
        let f = Flux::nouveau(11);
        for tau in [1.0, 10.0] {
            let x = ar1(400_000, tau, 0.0, &f, 1);
            let t = tau_int(&x).unwrap();
            assert!((t - tau).abs() < 0.15 * tau, "τ attendu {tau}, mesuré {t}");
        }
    }

    #[test]
    fn analyse_donne_une_erreur_cohérente() {
        let f = Flux::nouveau(12);
        let x = ar1(200_000, 10.0, 3.0, &f, 2);
        let a = analyser(&x);
        assert_eq!(a.statut, Statut::Ok);
        // σ attendu ≈ sqrt(2 τ / n) pour une variance 1
        let attendu = (2.0 * 10.0 / a.n as f64).sqrt();
        assert!((a.erreur / attendu - 1.0).abs() < 0.4, "{} {}", a.erreur, attendu);
        assert!((a.moyenne - 3.0).abs() < 5.0 * a.erreur);
    }

    #[test]
    fn statuts_non_verts() {
        let f = Flux::nouveau(13);
        let court = ar1(2000, 10.0, 0.0, &f, 3);
        assert_eq!(analyser(&court).statut, Statut::Puissance);
        // dérive : moyenne qui change à mi-chaîne
        let mut x = ar1(100_000, 2.0, 0.0, &f, 4);
        for v in x.iter_mut().skip(50_000) {
            *v += 0.5;
        }
        assert_eq!(analyser(&x).statut, Statut::NonStationnaire);
    }

    #[test]
    fn ajustement_jackknife_droite_et_residu() {
        let f = Flux::nouveau(14);
        let kappas = [1e-3, 2e-3, 4e-3];
        // pente 40 : le biais d'extrapolation est nul, le point du milieu est sur la droite
        let ans: Vec<Analyse> = kappas.iter().enumerate().map(|(i, k)| analyser(&ar1(200_000, 5.0, 1.0 + 40.0 * k, &f, 10 + i as u32))).collect();
        assert!(ans.iter().all(|a| a.statut == Statut::Ok));
        let pts: Vec<(f64, &Analyse)> = kappas.iter().copied().zip(ans.iter()).collect();
        let (o0, s0, r, sr) = ajuster_jackknife(&pts).unwrap();
        assert!((o0 - 1.0).abs() < 5.0 * s0, "{o0} {s0}");
        assert!(r.abs() < 5.0 * sr, "{r} {sr}");
        // courbure : le point du milieu est décalé de 10 σ
        let mut an2 = ans.clone();
        an2[1].moyenne += 10.0 * sr;
        for b in an2[1].blocs.iter_mut() {
            *b += 10.0 * sr;
        }
        let pts2: Vec<(f64, &Analyse)> = kappas.iter().copied().zip(an2.iter()).collect();
        let (_, _, r2, sr2) = ajuster_jackknife(&pts2).unwrap();
        assert!(r2.abs() / sr2 > 8.0);
    }

    #[test]
    fn contre_exemple_a_tau_negatif_n_est_pas_vert() {
        // 8000 valeurs alternant ±100, décalées de +1 dans la seconde moitié (contre-exemple de la revue)
        let x: Vec<f64> = (0..8000).map(|i| if i % 2 == 0 { 100.0 } else { -100.0 } + if i >= 4000 { 1.0 } else { 0.0 }).collect();
        assert_ne!(analyser(&x).statut, Statut::Ok);
        // série constante : variance nulle
        assert_ne!(analyser(&vec![1.0; 8000]).statut, Statut::Ok);
    }

    #[test]
    fn maxn_propage_nan() {
        assert!(maxn(f64::NAN, 1.0).is_nan() && maxn(1.0, f64::NAN).is_nan());
        assert_eq!(maxn(1.0, 2.0), 2.0);
    }

    #[test]
    fn ajustement_exact_sur_droite() {
        let mk = |m: f64, e: f64| Analyse { n: 1, moyenne: m, erreur: e, tau_int: 1.0, longueur_bloc: 1, statut: Statut::Ok, blocs: vec![] };
        let (a, b, c) = (mk(1.0 + 2.0 * 1e-3, 0.01), mk(1.0 + 2.0 * 2e-3, 0.01), mk(1.0 + 2.0 * 4e-3, 0.01));
        let (o0, s0, cc, _) = ajuster_droite(&[(1e-3, &a), (2e-3, &b), (4e-3, &c)]);
        assert!((o0 - 1.0).abs() < 1e-12 && (cc - 2.0).abs() < 1e-9 && s0 > 0.0);
    }
}
