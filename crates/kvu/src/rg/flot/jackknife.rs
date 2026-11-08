//! Jackknife avec suppression d'un groupe ou d'un bloc consécutif.

#[derive(Clone, Debug, PartialEq)]
pub struct ResJackknife {
    pub valeur: f64,
    pub moyenne_replicats: f64,
    pub erreur: f64,
    pub replicats: Vec<f64>,
}

/// Chaque réplicat omet un groupe ; l'ordre des références est conservé.
pub fn jackknife_groupes<T, F>(groupes: &[T], estimateur: F) -> Option<ResJackknife>
where
    F: Fn(&[&T]) -> Option<f64>,
{
    let c = groupes.len();
    if c < 2 {
        return None;
    }
    let tous: Vec<_> = groupes.iter().collect();
    let valeur = estimateur(&tous)?;
    if !valeur.is_finite() {
        return None;
    }
    let mut replicats = Vec::with_capacity(c);
    let mut retenus = Vec::with_capacity(c - 1);
    for exclu in 0..c {
        retenus.clear();
        retenus.extend(tous[..exclu].iter().copied());
        retenus.extend(tous[exclu + 1..].iter().copied());
        let replicat = estimateur(&retenus)?;
        if !replicat.is_finite() {
            return None;
        }
        replicats.push(replicat);
    }
    resumer(valeur, replicats)
}

fn resumer(valeur: f64, replicats: Vec<f64>) -> Option<ResJackknife> {
    let c = replicats.len();
    if c < 2 || !valeur.is_finite() || replicats.iter().any(|r| !r.is_finite()) {
        return None;
    }
    let moyenne_replicats = replicats.iter().sum::<f64>() / c as f64;
    let somme = replicats
        .iter()
        .map(|r| (r - moyenne_replicats).powi(2))
        .sum::<f64>();
    let erreur = (((c - 1) as f64 / c as f64) * somme).sqrt();
    if !moyenne_replicats.is_finite() || !erreur.is_finite() {
        return None;
    }
    Some(ResJackknife {
        valeur,
        moyenne_replicats,
        erreur,
        replicats,
    })
}

/// Jackknife de la moyenne sur les seuls blocs complets ; le reste est ignoré.
pub fn jackknife_bloc(x: &[f64], taille_bloc: usize) -> Option<ResJackknife> {
    if taille_bloc == 0 || x.len() / taille_bloc < 2 {
        return None;
    }
    let moyennes: Vec<_> = x
        .chunks_exact(taille_bloc)
        .map(|bloc| bloc.iter().sum::<f64>() / taille_bloc as f64)
        .collect();
    if moyennes.iter().any(|m| !m.is_finite()) {
        return None;
    }
    // Les blocs ont la même taille : leurs moyennes ont le même poids.
    // La soustraction de chaque bloc évite un coût quadratique en leur nombre.
    let c = moyennes.len();
    let total = moyennes.iter().sum::<f64>();
    let valeur = total / c as f64;
    let replicats = moyennes
        .iter()
        .map(|m| (total - m) / (c - 1) as f64)
        .collect();
    resumer(valeur, replicats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moyenne(groupes: &[&f64]) -> Option<f64> {
        Some(groupes.iter().map(|g| **g).sum::<f64>() / groupes.len() as f64)
    }

    #[test]
    fn erreur_de_la_moyenne() {
        let groupes = [1.0, 3.0, 4.0, 8.0, 10.0];
        let res = jackknife_groupes(&groupes, moyenne).expect("Groupes valides");
        let moy = groupes.iter().sum::<f64>() / groupes.len() as f64;
        let variance =
            groupes.iter().map(|g| (g - moy).powi(2)).sum::<f64>() / (groupes.len() - 1) as f64;
        assert!((res.erreur - (variance / groupes.len() as f64).sqrt()).abs() < 1e-14);
        assert_eq!(res.valeur, moy);
        assert!((res.moyenne_replicats - moy).abs() < 1e-14);
        assert_eq!(res.replicats, vec![6.25, 5.75, 5.5, 4.5, 4.0]);
    }

    #[test]
    fn propagation_des_echecs() {
        assert!(jackknife_groupes(&[], moyenne).is_none());
        assert!(jackknife_groupes(&[1.0], moyenne).is_none());
        assert!(jackknife_groupes(&[1.0, 2.0], |_| None).is_none());
        assert!(jackknife_groupes(&[1.0, 2.0], |g| if g.len() == 1 {
            None
        } else {
            moyenne(g)
        })
        .is_none());
        for invalide in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(jackknife_groupes(&[1.0, 2.0], |_| Some(invalide)).is_none());
            assert!(jackknife_groupes(&[1.0, 2.0], |g| if g.len() == 1 {
                Some(invalide)
            } else {
                moyenne(g)
            })
            .is_none());
        }
        assert!(jackknife_groupes(&[1, 2, 3], |g| {
            assert!(g.windows(2).all(|paire| paire[0] < paire[1]));
            Some(0.0)
        })
        .is_some());
    }

    #[test]
    fn blocs_complets_et_reste_ignore() {
        let res =
            jackknife_bloc(&[1.0, 3.0, 5.0, 7.0, 9.0, 11.0, f64::NAN], 2).expect("Trois blocs");
        let attendu = jackknife_groupes(&[2.0, 6.0, 10.0], moyenne).expect("Trois groupes");
        assert_eq!(res, attendu);
        assert!(jackknife_bloc(&[1.0, 2.0], 0).is_none());
        assert!(jackknife_bloc(&[1.0, 2.0], 2).is_none());
        assert!(jackknife_bloc(&[1.0, f64::NAN], 1).is_none());
    }
}
