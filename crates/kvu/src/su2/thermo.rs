//! Analyse de la MÉTHODE INTÉGRALE (campagne « une boîte ») : quadratures linéaires en β, erreurs par jackknife sur les répliques.
//!
//! Tout ce module est du calcul hôte pur (aucun GPU). Les sommes brutes S_tot, S_feuille viennent de `su2::une_boite`.
//!   ln Z_tordu/Z_périodique (β) = ∫₀^β g,   g = ⟨Q_s⟩_tordu − ⟨Q_{+1}⟩_témoin,   Q_s = S_tot + (s − 1)·S_F ;
//!   a₃ = ln Z² … : 2 ln Z(boîte) − ln Z(cube),   ln Z(β) = ∫₀^β ⟨S_tot⟩ dβ' (Z normalisée à 1 en β = 0 ; les constantes e^{-β·6V}
//!   se compensent exactement si le volume du cube vaut deux fois celui de la boîte : exigé).
//! L'intégrande est connue en β = 0 (g = 0 pour la torsion ; ⟨S_tot⟩ = 0 pour Haar) : le nœud β = 0 est ajouté SANS simulation.
//! Les nœuds sont indépendants (graines distinctes), mais les bras ± d'un même nœud et d'une même réplique sont APPARIÉS
//! (mêmes graines) : la différence est faite par réplique avant la moyenne, la covariance est donc incluse.

use crate::commun::{Erreur, Resultat};

/// `b > a` strict, faux si NaN.
fn croit(a: f64, b: f64) -> bool {
    matches!(b.partial_cmp(&a), Some(std::cmp::Ordering::Greater))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quadrature {
    /// Trapèzes.
    Trapeze,
    /// Paraboles locales : sur chaque pas, moyenne des paraboles passant par les trois nœuds voisins disponibles (à gauche et à droite,
    /// sans dépasser le nœud d'arrivée). Exacte pour les polynômes de degré ≤ 2.
    Parabole,
}

impl Quadrature {
    pub fn analyser(t: &str) -> Resultat<Quadrature> {
        match t {
            "trapeze" => Ok(Quadrature::Trapeze),
            "parabole" => Ok(Quadrature::Parabole),
            autre => Err(Erreur::Config(format!(
                "quadrature « {autre} » inconnue (trapeze|parabole)"
            ))),
        }
    }
}

fn int_lagrange(p: [f64; 3], j: usize, lo: f64, hi: f64) -> f64 {
    let (a, b) = match j {
        0 => (p[1], p[2]),
        1 => (p[0], p[2]),
        _ => (p[0], p[1]),
    };
    let den = (p[j] - a) * (p[j] - b);
    ((hi.powi(3) - lo.powi(3)) / 3.0 - (a + b) * (hi * hi - lo * lo) / 2.0 + a * b * (hi - lo))
        / den
}

/// Poids w[j] tels que ∫_{x[0]}^{x[k]} g ≈ Σ_j w[j] g(x[j]) (x strictement croissante, x[0] = 0 en général).
pub fn poids_integrale(x: &[f64], k: usize, q: Quadrature) -> Resultat<Vec<f64>> {
    if k >= x.len() || x.windows(2).any(|w| !croit(w[0], w[1])) {
        return Err(Erreur::Config(
            "quadrature : nœuds non croissants ou indice hors grille".into(),
        ));
    }
    let mut w = vec![0.0; x.len()];
    for i in 0..k {
        let (lo, hi) = (x[i], x[i + 1]);
        match q {
            Quadrature::Trapeze => {
                w[i] += (hi - lo) / 2.0;
                w[i + 1] += (hi - lo) / 2.0;
            }
            Quadrature::Parabole => {
                let mut triples: Vec<[usize; 3]> = vec![];
                if i >= 1 {
                    triples.push([i - 1, i, i + 1]);
                }
                if i + 2 <= k {
                    triples.push([i, i + 1, i + 2]);
                }
                if triples.is_empty() {
                    // un seul pas (k = 1) : trapèze
                    w[i] += (hi - lo) / 2.0;
                    w[i + 1] += (hi - lo) / 2.0;
                    continue;
                }
                let part = 1.0 / triples.len() as f64;
                for t in triples {
                    let p = [x[t[0]], x[t[1]], x[t[2]]];
                    for (j, &idx) in t.iter().enumerate() {
                        w[idx] += part * int_lagrange(p, j, lo, hi);
                    }
                }
            }
        }
    }
    Ok(w)
}

/// Jackknife sur les répliques d'une statistique θ(moyennes par nœud) : `par_replique[nœud][réplique]`. Rend (θ, erreur).
/// Les nœuds sont INDÉPENDANTS (graines distinctes) : la variance est la somme, sur les nœuds, de la variance jackknife obtenue en
/// omettant une réplique du seul nœud considéré (méthode delta par nœud). Les bras appariés d'un nœud sont différenciés par réplique
/// AVANT cet appel, la covariance entre bras est donc incluse.
pub fn jackknife<F: Fn(&[f64]) -> f64>(
    par_replique: &[Vec<f64>],
    theta: F,
) -> Resultat<(f64, f64)> {
    let r = par_replique.first().map_or(0, |v| v.len());
    if r < 2 || par_replique.iter().any(|v| v.len() != r) {
        return Err(Erreur::Config(
            "jackknife : au moins 2 répliques, le même nombre à chaque nœud".into(),
        ));
    }
    let moyennes: Vec<f64> = par_replique
        .iter()
        .map(|v| v.iter().sum::<f64>() / r as f64)
        .collect();
    let t0 = theta(&moyennes);
    let mut var = 0.0;
    for k in 0..par_replique.len() {
        let ts: Vec<f64> = (0..r)
            .map(|i| {
                let mut m = moyennes.clone();
                m[k] = (par_replique[k].iter().sum::<f64>() - par_replique[k][i]) / (r - 1) as f64;
                theta(&m)
            })
            .collect();
        let tm = ts.iter().sum::<f64>() / r as f64;
        var += (r as f64 - 1.0) / r as f64 * ts.iter().map(|t| (t - tm).powi(2)).sum::<f64>();
    }
    Ok((t0, var.sqrt()))
}

/// Une réplique d'un nœud : moyennes des sommes brutes sur toute la série et sur chaque moitié.
#[derive(Clone, Debug, PartialEq)]
pub struct PointRep {
    pub beta: f64,
    pub rep: usize,
    pub graine: u64,
    pub depart: String,
    pub n: usize,
    /// [S_tot, S_F] moyens (série entière), première moitié, seconde moitié.
    pub tout: [f64; 2],
    pub h1: [f64; 2],
    pub h2: [f64; 2],
}

pub fn resumer(beta: f64, rep: usize, graine: u64, depart: &str, serie: &[[f64; 2]]) -> PointRep {
    let n = serie.len();
    let m = |a: usize, b: usize| {
        let (mut x, mut y) = (0.0, 0.0);
        for s in &serie[a..b] {
            x += s[0];
            y += s[1];
        }
        let d = (b - a) as f64;
        [x / d, y / d]
    };
    PointRep {
        beta,
        rep,
        graine,
        depart: depart.to_string(),
        n,
        tout: m(0, n),
        h1: m(0, n / 2),
        h2: m(n / 2, n),
    }
}

/// Résultat d'une intégration : valeur et erreur jackknife à chaque nœud simulé, écart entre moitiés.
#[derive(Clone, Debug)]
pub struct Integrale {
    pub betas: Vec<f64>,
    pub valeur: Vec<f64>,
    pub erreur: Vec<f64>,
    /// Même intégrale calculée avec les premières moitiés seules moins avec les secondes (diagnostic de thermalisation).
    pub ecart_moities: Vec<f64>,
}

/// ln R = ∫₀^β g pour la torsion. `tordu` et `temoin` : points d'un même ensemble de nœuds (β croissants), mêmes répliques
/// (appariées par `rep`). `s` = poids du bras tordu.
pub fn ln_r(tordu: &[PointRep], temoin: &[PointRep], s: f64, q: Quadrature) -> Resultat<Integrale> {
    let q_s = |p: [f64; 2], s: f64| p[0] + (s - 1.0) * p[1];
    integrer(tordu, Some(temoin), q, |a, b| {
        let b = b.ok_or_else(|| Erreur::Config("témoin manquant".into()))?;
        Ok((
            q_s(a[0], s) - q_s(b[0], 1.0),
            q_s(a[1], s) - q_s(b[1], 1.0),
            q_s(a[2], s) - q_s(b[2], 1.0),
        ))
    })
}

/// 2 ln Z(boîte) − ln Z(cube) = ∫₀^β (2⟨S_tot⟩_boîte − ⟨S_tot⟩_cube) : `boite` et `cube` sur les MÊMES nœuds et répliques.
pub fn a3(boite: &[PointRep], cube: &[PointRep], q: Quadrature) -> Resultat<Integrale> {
    integrer(boite, Some(cube), q, |a, b| {
        let b = b.ok_or_else(|| Erreur::Config("cube manquant".into()))?;
        Ok((
            2.0 * a[0][0] - b[0][0],
            2.0 * a[1][0] - b[1][0],
            2.0 * a[2][0] - b[2][0],
        ))
    })
}

fn integrer<F>(a: &[PointRep], b: Option<&[PointRep]>, q: Quadrature, f: F) -> Resultat<Integrale>
where
    F: Fn([[f64; 2]; 3], Option<[[f64; 2]; 3]>) -> Resultat<(f64, f64, f64)>,
{
    let mut betas: Vec<f64> = a.iter().map(|p| p.beta).collect();
    betas.dedup_by(|x, y| x == y);
    if betas
        .iter()
        .any(|&x| x.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater))
        || betas.windows(2).any(|w| !croit(w[0], w[1]))
    {
        return Err(Erreur::Config(
            "nœuds β > 0, strictement croissants et groupés attendus".into(),
        ));
    }
    let nn = betas.len();
    fn get(pts: &[PointRep], beta: f64) -> Vec<&PointRep> {
        pts.iter().filter(|p| p.beta == beta).collect()
    }
    let r = get(a, betas[0]).len();
    let mut g_tout = vec![vec![0.0; r]; nn];
    let mut g_h1 = vec![vec![0.0; r]; nn];
    let mut g_h2 = vec![vec![0.0; r]; nn];
    for k in 0..nn {
        let (mut pa, mut pb) = (get(a, betas[k]), b.map(|b| get(b, betas[k])));
        pa.sort_by_key(|p| p.rep);
        if let Some(v) = pb.as_mut() {
            v.sort_by_key(|p| p.rep);
            if v.len() != pa.len() || v.iter().zip(&pa).any(|(x, y)| x.rep != y.rep || x.n != y.n) {
                return Err(Erreur::Config(format!(
                    "β = {} : répliques des deux bras non appariées",
                    betas[k]
                )));
            }
        }
        if pa.len() != r {
            return Err(Erreur::Config(format!(
                "β = {} : {} répliques au lieu de {r}",
                betas[k],
                pa.len()
            )));
        }
        for i in 0..r {
            let ta = [pa[i].tout, pa[i].h1, pa[i].h2];
            let tb = pb.as_ref().map(|v| [v[i].tout, v[i].h1, v[i].h2]);
            let (x, y, z) = f(ta, tb)?;
            g_tout[k][i] = x;
            g_h1[k][i] = y;
            g_h2[k][i] = z;
        }
    }
    // grille avec le nœud β = 0 (g = 0)
    let mut x = vec![0.0];
    x.extend(&betas);
    let theta = |k: usize, g: &[f64]| -> Resultat<f64> {
        let w = poids_integrale(&x, k + 1, q)?;
        Ok(w[1..].iter().zip(g).map(|(w, g)| w * g).sum())
    };
    let (mut valeur, mut erreur, mut ecart) = (vec![], vec![], vec![]);
    for k in 0..nn {
        let w = poids_integrale(&x, k + 1, q)?;
        let (v, e) = jackknife(&g_tout, |m| w[1..].iter().zip(m).map(|(w, g)| w * g).sum())?;
        valeur.push(v);
        erreur.push(e);
        let moyenne = |gg: &Vec<Vec<f64>>| -> Vec<f64> {
            gg.iter()
                .map(|v| v.iter().sum::<f64>() / r as f64)
                .collect()
        };
        ecart.push(theta(k, &moyenne(&g_h1))? - theta(k, &moyenne(&g_h2))?);
    }
    Ok(Integrale {
        betas,
        valeur,
        erreur,
        ecart_moities: ecart,
    })
}

/// Vérifie qu'un fichier de prédictions a le sha256 annoncé (hachage AVANT la mesure) ; refus code 2 sinon.
pub fn verifier_predictions(chemin: &std::path::Path, sha256_attendu: &str) -> Resultat<String> {
    let reel = crate::heritage::manifest::sha256_fichier(chemin)
        .map_err(|e| Erreur::io(format!("lecture de {}", chemin.display()), e))?;
    if !reel.eq_ignore_ascii_case(sha256_attendu.trim()) {
        return Err(Erreur::Config(format!(
            "fichier de prédictions {} : sha256 {reel} différent de celui annoncé {sha256_attendu} (prédictions modifiées après coup ?)",
            chemin.display()
        )));
    }
    Ok(reel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t1_exactitude_des_quadratures() {
        let x = [0.0, 0.3, 0.55, 1.0, 1.2, 2.0];
        let f1 = |t: f64| 2.0 * t - 1.0;
        let f2 = |t: f64| 3.0 * t * t - t + 0.5;
        for k in 2..x.len() {
            let w = poids_integrale(&x, k, Quadrature::Parabole).unwrap();
            let i2: f64 = w.iter().zip(&x).map(|(w, &t)| w * f2(t)).sum();
            let ex2 = x[k].powi(3) - x[k] * x[k] / 2.0 + 0.5 * x[k];
            assert!(
                (i2 - ex2).abs() < 1e-13,
                "parabole k={k} : {i2} contre {ex2}"
            );
            let wt = poids_integrale(&x, k, Quadrature::Trapeze).unwrap();
            let i1: f64 = wt.iter().zip(&x).map(|(w, &t)| w * f1(t)).sum();
            assert!((i1 - (x[k] * x[k] - x[k])).abs() < 1e-13, "trapèze k={k}");
        }
        assert!(poids_integrale(&[0.0, 0.0, 1.0], 1, Quadrature::Trapeze).is_err());
    }

    /// mutant : un poids de parabole faux doit faire échouer l'exactitude
    #[test]
    fn t1_mutant_poids_faux_detecte() {
        let x = [0.0, 0.5, 1.0, 1.5];
        let mut w = poids_integrale(&x, 3, Quadrature::Parabole).unwrap();
        w[2] += 1e-3;
        let i2: f64 = w.iter().zip(&x).map(|(w, &t)| w * t * t).sum();
        assert!((i2 - 1.5f64.powi(3) / 3.0).abs() > 1e-4);
    }

    #[test]
    fn t1_jackknife_egal_propagation_pour_une_forme_lineaire() {
        let mut s = 12345u64;
        let mut u = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (1u64 << 31) as f64 - 0.5
        };
        let g: Vec<Vec<f64>> = (0..4)
            .map(|k| (0..20).map(|_| u() + k as f64).collect())
            .collect();
        let w = [0.2, 0.5, -0.3, 0.7];
        let (v, e) = jackknife(&g, |m| m.iter().zip(&w).map(|(m, w)| m * w).sum()).unwrap();
        let moy: Vec<f64> = g.iter().map(|x| x.iter().sum::<f64>() / 20.0).collect();
        let v_ex: f64 = moy.iter().zip(&w).map(|(m, w)| m * w).sum();
        let var: f64 = g
            .iter()
            .zip(&w)
            .map(|(x, w)| {
                let m = x.iter().sum::<f64>() / 20.0;
                w * w * x.iter().map(|y| (y - m).powi(2)).sum::<f64>() / 19.0 / 20.0
            })
            .sum();
        assert!((v - v_ex).abs() < 1e-13);
        assert!((e - var.sqrt()).abs() < 1e-12, "{e} contre {}", var.sqrt());
        assert!(jackknife(&[vec![1.0]], |m| m[0]).is_err());
    }

    #[test]
    fn t2_analyse_synthetique_retrouve_ln_r() {
        // g(β) = −β (1 − β/8) exact ; ln R = −β²/2 + β³/24 ; bruit gaussien 0,05 par réplique ; 64 répliques
        let mut s = 777u64;
        let mut u = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut gauss = || (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos();
        let betas: Vec<f64> = (1..=16).map(|k| 0.125 * k as f64).collect();
        let (mut tordu, mut temoin) = (vec![], vec![]);
        for &b in &betas {
            for r in 0..64 {
                // le témoin porte S_F = 0 et S_tot = t ; le tordu a S_F tel que Q_{-1} − Q_{+1} = g (S_tot identique)
                let g = -b * (1.0 - b / 8.0) + 0.05 * gauss();
                let t = 3.0 + b;
                // Q_{-1} = S_tot − 2 S_F ; témoin Q_{+1} = S_tot ⇒ g = −2 S_F
                let sf = -g / 2.0;
                let mk = |sf: f64| PointRep {
                    beta: b,
                    rep: r,
                    graine: r as u64,
                    depart: "froid".into(),
                    n: 10,
                    tout: [t, sf],
                    h1: [t, sf],
                    h2: [t, sf],
                };
                tordu.push(mk(sf));
                temoin.push(mk(0.0));
            }
        }
        for q in [Quadrature::Trapeze, Quadrature::Parabole] {
            let res = ln_r(&tordu, &temoin, -1.0, q).unwrap();
            for k in [3usize, 7, 15] {
                let b = res.betas[k];
                let ex = -b * b / 2.0 + b.powi(3) / 24.0;
                let biais_q = if q == Quadrature::Trapeze {
                    0.125f64.powi(2) / 12.0 * b * 0.5
                } else {
                    0.0
                };
                assert!(
                    (res.valeur[k] - ex).abs() <= 3.0 * res.erreur[k] + biais_q + 1e-3,
                    "{q:?} β={b} : {} ± {} contre {ex}",
                    res.valeur[k],
                    res.erreur[k]
                );
                assert!(res.ecart_moities[k].abs() < 1e-12);
            }
        }
    }

    #[test]
    fn t2_predictions_sha_refuse_si_modifie() {
        let d = std::env::temp_dir().join(format!("kvu_thermo_pred_{}", std::process::id()));
        std::fs::write(&d, b"prediction").unwrap();
        let h = crate::heritage::manifest::sha256_hex(b"prediction");
        assert_eq!(verifier_predictions(&d, &h).unwrap(), h);
        std::fs::write(&d, b"prediction modifiee").unwrap();
        let e = verifier_predictions(&d, &h).unwrap_err();
        assert_eq!(e.code_sortie(), 2);
        std::fs::remove_file(&d).ok();
    }
}
