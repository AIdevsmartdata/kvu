//! Bootstrap de configurations pour `kvu rg action` (`--erreurs bootstrap`).
//!
//! Le jackknife sur les chaînes (`ModeJackknife`) ne donne qu'autant de réplicats qu'il y a
//! de groupes supprimés (4 ici) et ignore la variance *intra*-chaîne. Le bootstrap ici
//! implémenté rééchantillonne les **configurations** à l'intérieur de chaque chaîne
//! (stratification par chaîne : l'effectif de chaque chaîne est préservé à chaque réplicat),
//! puis réajuste avec le même estimateur et la même famille.
//!
//! # Algorithme de tirage
//! Pour un réplicat et une chaîne d'effectif `n`, on tire `m = ceil(n / K)` points de départ
//! `s` uniformes dans `[0, n)` ; le `j`-ième bloc est `{ (s_j + i) mod n : i in 0..K }`
//! (blocs **circulaires**, donc un départ près de la fin boucle sur le début de la chaîne).
//! La concaténation des `m` blocs est tronquée à `n` : le dernier bloc peut être incomplet,
//! mais l'effectif de la chaîne est exactement préservé. `K = 1` (défaut) : tirage iid.
//! Les configurations sont indépendantes entre chaînes d'une même chaîne (méta-stables) ;
//! `K > 1` sert aux chaînes dont les configurations sont corrélées en temps.
//!
//! # Generateur pseudo-aleatoire
//! SplitMix64 : `etat += 0x9E3779B97F4A7C15`, puis trois décalages/multiplications
//! (`0xBF58476D1CE4E5B9`, `0x94D049BB133111EB`) — constantes usuelles de cette famille.
//! Une seule instance, consommée séquentiellement : réplicat par réplicat, chaîne par
//! chaîne, dans l'ordre des chaînes. Le flux ne dépend donc ni du nombre de fils, ni de
//! l'ordre des estimateurs demandés, ni du succès ou de l'échec d'un ajustement : deux
//! exécutions avec la même `--graine` et le même ensemble produisent les mêmes réplicats.
//! Les entiers bornés sont tirés par rejet (de Neumann) : on rejette `x < 2^64 mod n`, ce
//! qui donne une loi exactement uniforme sur `[0, n)`.

use super::boucles::Famille;
use super::estimation::{
    ajuster_centrale, ajuster_replicat, correlations, critere_pr, norme_r4, Estimateur, StatsGroupe,
};
use crate::commun::{Erreur, Resultat};
use serde_json::Value;

/// Mode de calcul des incertitudes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeErreurs {
    /// Jackknife delete-group (comportement historique, défaut).
    Jackknife,
    /// Bootstrap de configurations stratifié par chaîne.
    Bootstrap,
}

impl ModeErreurs {
    pub fn analyser(t: &str) -> Resultat<ModeErreurs> {
        match t {
            "jackknife" => Ok(ModeErreurs::Jackknife),
            "bootstrap" => Ok(ModeErreurs::Bootstrap),
            autre => Err(Erreur::Config(format!(
                "mode d'erreurs « {autre} » inconnu (jackknife|bootstrap)"
            ))),
        }
    }

    pub fn nom(self) -> &'static str {
        match self {
            ModeErreurs::Jackknife => "jackknife",
            ModeErreurs::Bootstrap => "bootstrap",
        }
    }
}

/// SplitMix64, voir l'en-tete du module.
#[derive(Clone, Debug)]
pub struct RngSplitMix64 {
    etat: u64,
}

impl RngSplitMix64 {
    pub fn nouveau(graine: u64) -> Self {
        Self { etat: graine }
    }

    /// Tirage 64 bits.
    pub fn tirer_u64(&mut self) -> u64 {
        self.etat = self.etat.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.etat;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Entier uniforme dans `[0, n)` par rejet (loi exactement uniforme).
    pub fn tirer_below(&mut self, n: usize) -> usize {
        let n = n as u64;
        assert!(n > 0, "tirage below(0)");
        let seuil = 0u64.wrapping_sub(n) % n; // 2^64 mod n
        loop {
            let x = self.tirer_u64();
            if x >= seuil {
                return (x % n) as usize;
            }
        }
    }
}

/// Indices d'un réplicat dans UNE chaîne d'effectif `n` : `ceil(n/K)` blocs circulaires de
/// `K` positions consécutives, tronqués à `n`.
pub fn indices_chaine(rng: &mut RngSplitMix64, n: usize, bloc: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let bloc = bloc.max(1);
    let m = n.div_ceil(bloc);
    let mut out = Vec::with_capacity(m * bloc);
    for _ in 0..m {
        let depart = rng.tirer_below(n);
        for i in 0..bloc {
            out.push((depart + i) % n);
        }
    }
    out.truncate(n);
    out
}

/// Plans des `b` réplicats, aplatis dans la concaténation des chaînes (dans l'ordre de
/// `effectifs`). Un plan = `effectifs.iter().sum()` indices, la partie de la chaîne `c`
/// occupant les positions `[offset_c, offset_c + effectif_c)`.
pub fn plans_bootstrap(effectifs: &[usize], bloc: usize, b: usize, graine: u64) -> Vec<Vec<usize>> {
    let mut rng = RngSplitMix64::nouveau(graine);
    (0..b)
        .map(|_| {
            let mut plan = Vec::with_capacity(effectifs.iter().sum());
            for &n in effectifs {
                // Indices GLOBAUX : la chaîne c occupe [offset_c, offset_c + n_c) dans la concaténation.
                let offset = plan.len();
                plan.extend(indices_chaine(&mut rng, n, bloc).into_iter().map(|i| offset + i));
            }
            plan
        })
        .collect()
}

/// Valeur résumée par bootstrap.
#[derive(Clone, Debug)]
pub struct StatBootstrap {
    /// Valeur sur l'échantillon complet (non bootstrappée).
    pub valeur: f64,
    /// Écart-type des réplicats, ddof = 1 (dénominateur B-1).
    pub erreur: f64,
    /// Intervalle percentiles [2.5 %, 97.5 %] des réplicats.
    pub ic95: [f64; 2],
    /// Tous les réplicats valides, dans l'ordre des réplicats.
    pub replicats: Vec<f64>,
}

/// Écart-type avec ddof = 1 ; NAN si moins de deux valeurs.
pub fn ecart_type(valeurs: &[f64]) -> f64 {
    if valeurs.len() < 2 {
        return f64::NAN;
    }
    let moyenne = valeurs.iter().sum::<f64>() / valeurs.len() as f64;
    let somme: f64 = valeurs.iter().map(|x| (x - moyenne).powi(2)).sum();
    (somme / (valeurs.len() - 1) as f64).sqrt()
}

/// Percentile d'un tableau DEJA TRIE, interpolation linéaire entre ordre statistiques
/// (méthode « type 7 », celle de numpy/`quantile` par défaut) : `h = p (m-1)`.
pub fn percentile_tri(trie: &[f64], p: f64) -> f64 {
    let m = trie.len();
    if m == 0 {
        return f64::NAN;
    }
    if m == 1 {
        return trie[0];
    }
    let h = (p * (m - 1) as f64).clamp(0.0, (m - 1) as f64);
    let bas = h.floor() as usize;
    let haut = (bas + 1).min(m - 1);
    trie[bas] + (h - bas as f64) * (trie[haut] - trie[bas])
}

/// Résume des réplicats bootstrap : écart-type ddof 1 + intervalle percentiles.
/// Les valeurs non finies doivent être exclues AVANT (le tirant les réplicats invalides).
pub fn resumer(valeur: f64, replicats: &[f64]) -> StatBootstrap {
    let mut tri = replicats.to_vec();
    tri.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    StatBootstrap {
        valeur,
        erreur: ecart_type(replicats),
        ic95: [
            percentile_tri(&tri, 0.025),
            percentile_tri(&tri, 0.975),
        ],
        replicats: replicats.to_vec(),
    }
}

/// Fraction de réplicats strictement sous le seuil.
pub fn p_sous_seuil(replicats: &[f64], seuil: f64) -> f64 {
    if replicats.is_empty() {
        return f64::NAN;
    }
    replicats.iter().filter(|x| **x < seuil).count() as f64 / replicats.len() as f64
}

// ---------------------------------------------------------------------------------------
// Pilotage : réajustement des réplicats par l'estimateur `rg action`
// ---------------------------------------------------------------------------------------

/// Un estimateur (`pl` ou `sd`) réajusté sur les plans bootstrap.
#[derive(Clone, Debug)]
pub struct EstimateurBootstrap {
    /// β ajusté sur l'échantillon complet.
    pub central: Vec<f64>,
    /// Détails de l'ajustement central (itérations, norme du gradient, spectre...).
    pub details: Value,
    /// β de chaque réplicat, `None` si l'ajustement a échoué ou a rendu un β non fini.
    /// L'ordre suit celui des plans ; le flux aléatoire n'en dépend pas.
    pub betas: Vec<Option<Vec<f64>>>,
}

impl EstimateurBootstrap {
    /// β des réplicats réussis, dans l'ordre des plans.
    pub fn valides(&self) -> Vec<&Vec<f64>> {
        self.betas.iter().flatten().collect()
    }

    /// Nombre de réplicats réussis.
    pub fn n_valides(&self) -> usize {
        self.betas.iter().filter(|b| b.is_some()).count()
    }

    /// Nombre de réplicats exclus (ajustement en échec, ou β non fini).
    pub fn echoues(&self) -> usize {
        self.betas.len() - self.n_valides()
    }
}

/// Réajuste les `b` réplicats décrits par `plans` (indices GLOBAUX dans `configs`, répétition
/// permise, aucune copie : un réplicat est un `Vec<&StatsGroupe>`) avec le même estimateur et
/// la même famille que la valeur centrale. L'échec d'un ajustement n'est pas fatal : le
/// réplicat est marqué `None` et compté dans [`EstimateurBootstrap::echoues`].
pub fn bootstrap_estimateur(
    configs: &[StatsGroupe],
    plans: &[Vec<usize>],
    famille: &Famille,
    quoi: Estimateur,
    fils: usize,
) -> Resultat<EstimateurBootstrap> {
    if let Some(&i) = plans.iter().flatten().find(|&&i| i >= configs.len()) {
        return Err(Erreur::Config(format!(
            "indice bootstrap {i} hors de l'échantillon ({} configurations)",
            configs.len()
        )));
    }
    let refs_tous: Vec<&StatsGroupe> = configs.iter().collect();
    let (central, details) = ajuster_centrale(&refs_tous, famille, quoi, fils)?;
    let mut betas = Vec::with_capacity(plans.len());
    for plan in plans {
        let refs: Vec<&StatsGroupe> = plan.iter().map(|&i| &configs[i]).collect();
        let rep = ajuster_replicat(&refs, famille, quoi, &central, fils)
            .ok()
            .filter(|b| b.len() == central.len() && b.iter().all(|x| x.is_finite()));
        betas.push(rep);
    }
    Ok(EstimateurBootstrap {
        central,
        details,
        betas,
    })
}

/// Résumés bootstrap d'un estimateur : un [`StatBootstrap`] par couplage, plus la norme R4 et
/// le critère P+R, et la matrice de corrélation des couplages sur les réplicats réussis.
#[derive(Clone, Debug)]
pub struct ResumeBootstrap {
    pub couplages: Vec<StatBootstrap>,
    pub r4: StatBootstrap,
    pub pr: StatBootstrap,
    pub correlations: Vec<f64>,
    pub n_valides: usize,
    pub echoues: usize,
}

/// Résume les réplicats d'un estimateur bootstrappé. Les seuils (N4 < 4, c_PR < 2/9) sont
/// appliqués par l'appelant avec [`p_sous_seuil`], sur `r4.replicats` / `pr.replicats`.
pub fn resumer_estimateur(est: &EstimateurBootstrap, famille: &Famille) -> ResumeBootstrap {
    let valides = est.valides();
    let n = famille.n();
    let couplages = (0..n)
        .map(|i| {
            resumer(
                est.central[i],
                &valides.iter().map(|b| b[i]).collect::<Vec<_>>(),
            )
        })
        .collect();
    let r4 = resumer(
        norme_r4(famille, &est.central),
        &valides
            .iter()
            .map(|b| norme_r4(famille, b))
            .collect::<Vec<_>>(),
    );
    let pr = resumer(
        critere_pr(famille, &est.central),
        &valides
            .iter()
            .map(|b| critere_pr(famille, b))
            .collect::<Vec<_>>(),
    );
    let betas_valides: Vec<Vec<f64>> = valides.iter().map(|b| (*b).clone()).collect();
    ResumeBootstrap {
        couplages,
        r4,
        pr,
        correlations: correlations(&betas_valides, n),
        n_valides: valides.len(),
        echoues: est.echoues(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_erreurs_analyse() {
        assert_eq!(ModeErreurs::analyser("jackknife").unwrap(), ModeErreurs::Jackknife);
        assert_eq!(ModeErreurs::analyser("bootstrap").unwrap(), ModeErreurs::Bootstrap);
        assert!(ModeErreurs::analyser("jack").is_err());
    }

    #[test]
    fn effectifs_preserves_par_chaine() {
        // Quel que soit (n, K), la chaîne garde son effectif et tous les indices sont valides.
        for &n in &[1usize, 2, 3, 7, 8, 13, 64] {
            for &k in &[1usize, 2, 3, 5, 8, 64] {
                let mut rng = RngSplitMix64::nouveau(12345 + n as u64 * 31 + k as u64);
                for _ in 0..20 {
                    let idx = indices_chaine(&mut rng, n, k);
                    assert_eq!(idx.len(), n, "n={n} K={k} : effectif modifié");
                    assert!(idx.iter().all(|&i| i < n), "n={n} K={k} : indice hors chaîne");
                }
            }
        }
    }

    #[test]
    fn structure_de_bloc_circulaire() {
        // n = 10, K = 3 -> 4 blocs, les trois premiers de taille 3, le dernier de taille 1.
        let n = 10usize;
        let k = 3usize;
        let mut rng = RngSplitMix64::nouveau(777);
        for _ in 0..50 {
            let idx = indices_chaine(&mut rng, n, k);
            assert_eq!(idx.len(), n);
            let m = n.div_ceil(k);
            for j in 0..m {
                let bloc = &idx[j * k..((j + 1) * k).min(n)]; // dernier bloc tronqué à n
                let depart = bloc[0];
                for (i, &v) in bloc.iter().enumerate() {
                    assert_eq!(v, (depart + i) % n, "bloc {j} non consécutif : {idx:?}");
                }
            }
        }
    }

    #[test]
    fn stratification_et_plans_aplatis() {
        let effectifs = [5usize, 8, 8, 3];
        let plans = plans_bootstrap(&effectifs, 2, 7, 4242);
        assert_eq!(plans.len(), 7);
        let offsets: Vec<usize> = effectifs
            .iter()
            .scan(0usize, |acc, &n| {
                let o = *acc;
                *acc += n;
                Some(o)
            })
            .collect();
        for plan in &plans {
            assert_eq!(plan.len(), effectifs.iter().sum::<usize>());
            for (c, &n) in effectifs.iter().enumerate() {
                let troncon = &plan[offsets[c]..offsets[c] + n];
                assert!(
                    troncon.iter().all(|&i| (offsets[c]..offsets[c] + n).contains(&i)),
                    "réplicat mélangeant les chaînes (chaîne {c})"
                );
            }
        }
    }

    #[test]
    fn déterminisme_graine_et_sensibilite() {
        let a = plans_bootstrap(&[6, 6, 6], 2, 5, 999);
        let b = plans_bootstrap(&[6, 6, 6], 2, 5, 999);
        assert_eq!(a, b, "deux tirages à graine identique different");
        let c = plans_bootstrap(&[6, 6, 6], 2, 5, 1000);
        assert_ne!(a, c, "deux graines differentes donnent le même plan");
    }

    #[test]
    fn tirage_uniforme_en_moyenne() {
        // Contrôle grossier de l'uniformité : 400 000 tirages sur n = 7, fréquences à 2 %.
        let n = 7usize;
        let mut rng = RngSplitMix64::nouveau(20261109);
        let mut compte = vec![0usize; n];
        for _ in 0..400_000 {
            compte[rng.tirer_below(n)] += 1;
        }
        let attendu = 400_000.0 / n as f64;
        for (i, &c) in compte.iter().enumerate() {
            assert!(
                (c as f64 - attendu).abs() / attendu < 0.02,
                "case {i} : {c} contre {attendu}"
            );
        }
    }

    #[test]
    fn resume_ecart_type_et_percentiles() {
        // ddof = 1 : sur [1,2,3,4] -> sqrt(5/3) ; percentiles type 7 : 1.075 et 3.925.
        let s = resumer(2.5, &[1.0, 2.0, 3.0, 4.0]);
        assert!((s.erreur - (5.0f64 / 3.0).sqrt()).abs() < 1e-15, "{}", s.erreur);
        assert!((s.ic95[0] - 1.075).abs() < 1e-12, "{}", s.ic95[0]);
        assert!((s.ic95[1] - 3.925).abs() < 1e-12, "{}", s.ic95[1]);
        assert_eq!(s.replicats, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(ecart_type(&[3.0]).is_nan());
        // l'ordre des réplicats est conservé, le tri interne ne dépend pas de l'entrée
        let s2 = resumer(2.5, &[4.0, 1.0, 3.0, 2.0]);
        assert_eq!(s2.ic95, s.ic95);
        assert_eq!(s2.replicats, vec![4.0, 1.0, 3.0, 2.0]);
    }

    #[test]
    fn p_sous_seuil_compte_les_strictements_inferieurs() {
        let r = [0.1, 0.2, 0.3, 0.4];
        assert!((p_sous_seuil(&r, 0.3) - 0.5).abs() < 1e-15);
        assert!((p_sous_seuil(&r, 0.0) - 0.0).abs() < 1e-15);
        assert!((p_sous_seuil(&r, 9.0) - 1.0).abs() < 1e-15);
        assert!(p_sous_seuil(&[], 1.0).is_nan());
    }
}
