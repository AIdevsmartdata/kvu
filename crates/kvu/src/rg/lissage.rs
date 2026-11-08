//! Opérateurs de blocage de facteur 3 quasi-locaux (WP22), CPU, F64 : `dec`, `ape:<α>:<n>`, `flot:<t>[:<ε>]`, `agr:<c>`, et l'aiguillage
//! vers le bloc de Bałaban `3centre` (inchangé). Définitions EXACTES (ordre des opérations compris) : `docs/BLOCAGE_QUASI_LOCAL.md` §3,
//! qui reproduit la référence Python BLOC2 (`operateurs.py`) à l'association près de la composante 0 du produit de quaternions.
//!
//! Lien grossier des opérateurs à décimation : V(X, μ) du site fin 3X au site fin 3X + 3μ̂ (coin du bloc), covariant comme
//! V → g(3X) V g(3X + 3μ̂)†. Déterminisme : chaque lien est calculé à partir du seul champ de l'étape précédente, collecte dans l'ordre
//! des liens, aucune réduction flottante entre liens (le minimum de |s| ne dépend pas de l'ordre) : sorties identiques au bit quel que
//! soit le nombre de fils.

use super::bloc::{bloquer, coordonnees, site, valider_champ, OptionsBloc, StatsTheta, Q};
use super::geometrie::Geometrie;
use super::operateur::Operateur;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul};
use rayon::prelude::*;

/// En dessous de cette norme, la projection SU(2) est ambiguë : refus (même règle que BLOC2 `projeter`).
pub const NORME_MIN_PROJECTION: f64 = 1e-12;

/// Variantes : `Normale` hors gates ; les autres sont les MUTANTS des gates RG-QL-* (jamais exposés par la ligne de commande).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Variante {
    #[default]
    Normale,
    /// ape/agr : agrafes FERMANTES (qdag des agrafes de x à x + μ̂) ⇒ covariance de jauge cassée.
    AgrafeMalOrientee,
    /// ape/agr : U′ = s sans division par |s| ⇒ unitarité cassée.
    SansProjection,
    /// ape : n + 1 pas réellement calculés (support déclaré B(C, n)) ⇒ support élargi.
    PasSupplementaire,
    /// ape : agrafes +ν seules, poids α/3 ⇒ covariance sous les réflexions cassée (jauge intacte).
    AgrafesPlusSeulement,
    /// dec (donc ape, flot) : V = (U(x+2μ̂) U(x+μ̂)) U(x) ⇒ covariance et identité des boucles cassées.
    DecOrdreInverse,
}

#[derive(Clone, Copy, Debug)]
pub struct OptionsOperateur {
    pub fils: usize,
    pub variante: Variante,
}

impl OptionsOperateur {
    pub fn normale(fils: usize) -> Self {
        OptionsOperateur { fils, variante: Variante::Normale }
    }
}

pub struct SortieOperateur {
    /// Configuration grossière (L/3)⁴.
    pub grossier: ChampHote,
    /// min |s| sur toutes les projections faites (ape : tous les pas ; agr) ; `None` sinon.
    pub norme_min_projection: Option<f64>,
    /// Statistiques de θ du bloc de Bałaban (`3centre` seulement).
    pub stats_theta: Option<StatsTheta>,
}

/// Sites voisins précalculés : `haut[d * v + s]` = site de x + d̂, `bas[d * v + s]` = site de x − d̂.
pub(crate) struct Voisinage {
    pub v: usize,
    haut: Vec<u32>,
    bas: Vec<u32>,
}

impl Voisinage {
    pub(crate) fn nouveau(l: usize) -> Voisinage {
        let v = l.pow(4);
        let mut haut = vec![0u32; 4 * v];
        let mut bas = vec![0u32; 4 * v];
        for s in 0..v {
            let x = coordonnees(s, l);
            for d in 0..4 {
                let mut y = x;
                y[d] = (x[d] + 1) % l;
                haut[d * v + s] = site(y, l) as u32;
                y[d] = (x[d] + l - 1) % l;
                bas[d * v + s] = site(y, l) as u32;
            }
        }
        Voisinage { v, haut, bas }
    }
    #[inline]
    pub(crate) fn haut(&self, d: usize, s: usize) -> usize {
        self.haut[d * self.v + s] as usize
    }
    #[inline]
    pub(crate) fn bas(&self, d: usize, s: usize) -> usize {
        self.bas[d * self.v + s] as usize
    }
}

fn pool(fils: usize) -> Resultat<rayon::ThreadPool> {
    if fils == 0 {
        return Err(Erreur::Config("le nombre de fils doit être ≥ 1".into()));
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(fils)
        .build()
        .map_err(|e| Erreur::Config(format!("création du pool de fils impossible : {e}")))
}

fn verifier_entree(champ: &ChampHote) -> Resultat<()> {
    if champ.l < 3 || !champ.l.is_multiple_of(3) {
        return Err(Erreur::Config(format!("L = {} : multiple de 3 ≥ 3 requis", champ.l)));
    }
    valider_champ(champ)
}

#[inline]
fn norme(s: &Q) -> f64 {
    (s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3]).sqrt()
}

/// Un pas APE (mise à jour SIMULTANÉE) : U′(x, μ) = P[(1 − α) U(x, μ) + (α/6) Σ_ν (a_ν + b_ν)] (doc §3.2). Rend (champ, min |s|).
pub fn pas_ape(champ: &ChampHote, alpha: f64, variante: Variante, pool: &rayon::ThreadPool) -> Resultat<(ChampHote, f64)> {
    let l = champ.l;
    let vois = Voisinage::nouveau(l);
    let v = vois.v;
    let u = |mu: usize, s: usize| -> Q { champ.lien(mu * v + s) };
    let plus_seulement = variante == Variante::AgrafesPlusSeulement;
    let poids = if plus_seulement { alpha / 3.0 } else { alpha / 6.0 };
    let liens: Vec<(Q, f64)> = pool.install(|| {
        (0..4 * v)
            .into_par_iter()
            .map(|li| {
                let (mu, s) = (li / v, li % v);
                let s_mu = vois.haut(mu, s);
                let mut tot = [0.0f64; 4];
                for nu in 0..4 {
                    if nu == mu {
                        continue;
                    }
                    let s_nu = vois.haut(nu, s);
                    let s_mnu = vois.bas(nu, s);
                    let s_mnu_mu = vois.haut(mu, s_mnu);
                    // a = U(x,ν) U(x+ν̂,μ) U(x+μ̂,ν)† ; b = U(x−ν̂,ν)† U(x−ν̂,μ) U(x−ν̂+μ̂,ν)
                    let mut a = qmul(qmul(u(nu, s), u(mu, s_nu)), qdag(u(nu, s_mu)));
                    let mut b = qmul(qmul(qdag(u(nu, s_mnu)), u(mu, s_mnu)), u(nu, s_mnu_mu));
                    if variante == Variante::AgrafeMalOrientee {
                        a = qdag(a);
                        b = qdag(b);
                    }
                    if plus_seulement {
                        for i in 0..4 {
                            tot[i] += a[i];
                        }
                    } else {
                        for i in 0..4 {
                            tot[i] += a[i] + b[i];
                        }
                    }
                }
                let q = u(mu, s);
                let sv: Q = std::array::from_fn(|i| (1.0 - alpha) * q[i] + poids * tot[i]);
                let n = norme(&sv);
                if variante == Variante::SansProjection {
                    (sv, n)
                } else {
                    (std::array::from_fn(|i| sv[i] / n), n)
                }
            })
            .collect()
    });
    let mut nmin = f64::INFINITY;
    let mut q = Vec::with_capacity(16 * v);
    for (li, (w, n)) in liens.into_iter().enumerate() {
        if n.is_nan() || n < NORME_MIN_PROJECTION {
            return Err(Erreur::Config(format!(
                "projection SU(2) ambiguë (|s| = {n:e}) au lien {li} (direction {}, site {})",
                li / v,
                li % v
            )));
        }
        nmin = nmin.min(n);
        q.extend_from_slice(&w);
    }
    Ok((ChampHote { l, q }, nmin))
}

/// n pas APE ; n = 0 ⇒ copie du champ, AUCUN calcul, norme minimale `None`. `PasSupplementaire` ⇒ n + 1 pas.
pub fn lisser_ape(champ: &ChampHote, alpha: f64, n: usize, opt: &OptionsOperateur) -> Resultat<(ChampHote, Option<f64>)> {
    verifier_entree(champ)?;
    let pool = pool(opt.fils)?;
    let n_pas = if opt.variante == Variante::PasSupplementaire { n + 1 } else { n };
    let mut c = champ.clone();
    let mut nmin: Option<f64> = None;
    for _ in 0..n_pas {
        let (suivant, m) = pas_ape(&c, alpha, opt.variante, &pool)?;
        nmin = Some(nmin.map_or(m, |x| x.min(m)));
        c = suivant;
    }
    Ok((c, nmin))
}

/// Flot de Wilson de `n` pas RK3 de taille `eps` : `obs::ref_cpu::flot_n_pas` (OBS-3), exécuté dans un pool de `fils` fils.
pub fn lisser_flot(champ: &ChampHote, eps: f64, n: usize, fils: usize) -> Resultat<ChampHote> {
    verifier_entree(champ)?;
    let pool = pool(fils)?;
    pool.install(|| crate::obs::ref_cpu::flot_n_pas(champ, eps, n))
}

/// Décimation : V(X, μ) = (U(x, μ) U(x+μ̂, μ)) U(x+2μ̂, μ), x = 3X (doc §3.1).
pub fn decimer(champ: &ChampHote, variante: Variante) -> Resultat<ChampHote> {
    verifier_entree(champ)?;
    let l = champ.l;
    let v = champ.v();
    let lc = l / 3;
    let vc = lc.pow(4);
    let mut q = Vec::with_capacity(16 * vc);
    for mu in 0..4 {
        for sc in 0..vc {
            let x = coordonnees(sc, lc).map(|t| 3 * t);
            let mut y = x;
            let u0 = champ.lien(mu * v + site(y, l));
            y[mu] = (x[mu] + 1) % l;
            let u1 = champ.lien(mu * v + site(y, l));
            y[mu] = (x[mu] + 2) % l;
            let u2 = champ.lien(mu * v + site(y, l));
            let w = if variante == Variante::DecOrdreInverse { qmul(qmul(u2, u1), u0) } else { qmul(qmul(u0, u1), u2) };
            q.extend_from_slice(&w);
        }
    }
    Ok(ChampHote { l: lc, q })
}

/// AGR(c) (doc §3.4) directement sur le réseau grossier : V = P[S(x) + c Σ_ν (A₊ + A₋)], x = 3X. Rend (champ grossier, min |tot|).
pub fn agr(champ: &ChampHote, c: f64, opt: &OptionsOperateur) -> Resultat<(ChampHote, f64)> {
    verifier_entree(champ)?;
    let pool = pool(opt.fils)?;
    let l = champ.l;
    let vois = Voisinage::nouveau(l);
    let v = vois.v;
    let lc = l / 3;
    let vc = lc.pow(4);
    let u = |mu: usize, s: usize| -> Q { champ.lien(mu * v + s) };
    // S(y) = (U(y,μ) U(y+μ̂,μ)) U(y+2μ̂,μ)
    let droit = |mu: usize, s: usize| -> Q {
        let s1 = vois.haut(mu, s);
        let s2 = vois.haut(mu, s1);
        qmul(qmul(u(mu, s), u(mu, s1)), u(mu, s2))
    };
    let liens: Vec<(Q, f64)> = pool.install(|| {
        (0..4 * vc)
            .into_par_iter()
            .map(|lc_i| {
                let (mu, sc) = (lc_i / vc, lc_i % vc);
                let s = site(coordonnees(sc, lc).map(|t| 3 * t), l);
                let s3mu = vois.haut(mu, vois.haut(mu, vois.haut(mu, s)));
                let mut tot = droit(mu, s);
                for nu in 0..4 {
                    if nu == mu {
                        continue;
                    }
                    let s_nu = vois.haut(nu, s);
                    let s_mnu = vois.bas(nu, s);
                    let s_mnu_3mu = vois.haut(mu, vois.haut(mu, vois.haut(mu, s_mnu)));
                    let mut a = qmul(qmul(u(nu, s), droit(mu, s_nu)), qdag(u(nu, s3mu)));
                    let mut b = qmul(qmul(qdag(u(nu, s_mnu)), droit(mu, s_mnu)), u(nu, s_mnu_3mu));
                    if opt.variante == Variante::AgrafeMalOrientee {
                        a = qdag(a);
                        b = qdag(b);
                    }
                    for i in 0..4 {
                        tot[i] += c * (a[i] + b[i]);
                    }
                }
                let n = norme(&tot);
                if opt.variante == Variante::SansProjection {
                    (tot, n)
                } else {
                    (std::array::from_fn(|i| tot[i] / n), n)
                }
            })
            .collect()
    });
    let mut nmin = f64::INFINITY;
    let mut q = Vec::with_capacity(16 * vc);
    for (li, (w, n)) in liens.into_iter().enumerate() {
        if n.is_nan() || n < NORME_MIN_PROJECTION {
            return Err(Erreur::Config(format!("projection SU(2) ambiguë (|s| = {n:e}) au lien grossier {li}")));
        }
        nmin = nmin.min(n);
        q.extend_from_slice(&w);
    }
    Ok((ChampHote { l: lc, q }, nmin))
}

/// Un pas de blocage de facteur 3 par l'opérateur `op` (doc §3).
pub fn bloquer_operateur(champ: &ChampHote, op: &Operateur, opt: &OptionsOperateur) -> Resultat<SortieOperateur> {
    verifier_entree(champ)?;
    if opt.fils == 0 {
        return Err(Erreur::Config("le nombre de fils doit être ≥ 1".into()));
    }
    match *op {
        Operateur::TroisCentre => {
            let s = bloquer(champ, &OptionsBloc { geometrie: Geometrie::TroisCentre, fils: opt.fils, angle_so3: false })?;
            Ok(SortieOperateur { grossier: s.grossier, norme_min_projection: None, stats_theta: Some(s.stats) })
        }
        Operateur::Dec => Ok(SortieOperateur { grossier: decimer(champ, opt.variante)?, norme_min_projection: None, stats_theta: None }),
        Operateur::Ape { alpha, n } => {
            let (lisse, nmin) = lisser_ape(champ, alpha, n, opt)?;
            Ok(SortieOperateur { grossier: decimer(&lisse, opt.variante)?, norme_min_projection: nmin, stats_theta: None })
        }
        Operateur::Flot { eps, n, .. } => {
            let lisse = lisser_flot(champ, eps, n, opt.fils)?;
            Ok(SortieOperateur { grossier: decimer(&lisse, opt.variante)?, norme_min_projection: None, stats_theta: None })
        }
        Operateur::Agr { c } => {
            let (g, nmin) = agr(champ, c, opt)?;
            Ok(SortieOperateur { grossier: g, norme_min_projection: Some(nmin), stats_theta: None })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rg::bloc::tests::{aleatoire, froid, jauges, transformer};
    use crate::rg::bloc::deplacer;

    fn op(s: &str) -> Operateur {
        Operateur::analyser(s).unwrap()
    }

    fn bits(a: &ChampHote) -> Vec<u64> {
        a.q.iter().map(|x| x.to_bits()).collect()
    }

    fn ecart(a: &ChampHote, b: &ChampHote) -> f64 {
        a.q.iter().zip(&b.q).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn froid_reste_froid() {
        for s in ["dec", "ape:0.5:6", "flot:0.2:0.01", "agr:1"] {
            let g = bloquer_operateur(&froid(9), &op(s), &OptionsOperateur::normale(2)).unwrap().grossier;
            assert_eq!(g.l, 3);
            assert!(ecart(&g, &froid(3)) <= 1e-15, "{s}");
        }
    }

    #[test]
    fn n_ou_t_nul_donne_dec_au_bit() {
        let c = aleatoire(9);
        let d = bloquer_operateur(&c, &op("dec"), &OptionsOperateur::normale(3)).unwrap().grossier;
        for s in ["ape:0.5:0", "flot:0", "flot:0:0.01"] {
            assert_eq!(bits(&bloquer_operateur(&c, &op(s), &OptionsOperateur::normale(3)).unwrap().grossier), bits(&d), "{s}");
        }
    }

    #[test]
    fn identite_selon_les_fils() {
        let c = aleatoire(9);
        for s in ["dec", "ape:0.5:6", "flot:0.05:0.01", "agr:1"] {
            let a = bloquer_operateur(&c, &op(s), &OptionsOperateur::normale(1)).unwrap();
            for f in [4, 7] {
                let b = bloquer_operateur(&c, &op(s), &OptionsOperateur::normale(f)).unwrap();
                assert_eq!(bits(&a.grossier), bits(&b.grossier), "{s} fils {f}");
                assert_eq!(a.norme_min_projection.map(f64::to_bits), b.norme_min_projection.map(f64::to_bits));
            }
        }
    }

    /// Développement (les gates de verdict sont dans `gates_ql`) : covariance de jauge à 1e-12, mutant d'orientation rouge.
    #[test]
    fn covariance_de_jauge_et_mutant() {
        let c = aleatoire(9);
        let g = jauges(9);
        let cg = transformer(&c, &g);
        let ecart_cov = |s: &str, variante: Variante| {
            let o = OptionsOperateur { fils: 4, variante };
            let a = bloquer_operateur(&c, &op(s), &o).unwrap().grossier;
            let b = bloquer_operateur(&cg, &op(s), &o).unwrap().grossier;
            let mut m = 0.0f64;
            for mu in 0..4 {
                for sc in 0..81 {
                    let x = coordonnees(sc, 3).map(|t| 3 * t);
                    let xp = deplacer(x, mu, 3, 9);
                    let att = qmul(qmul(g[site(x, 9)], a.lien(mu * 81 + sc)), qdag(g[site(xp, 9)]));
                    let obt = b.lien(mu * 81 + sc);
                    for i in 0..4 {
                        m = m.max((att[i] - obt[i]).abs());
                    }
                }
            }
            m
        };
        for s in ["dec", "ape:0.5:6", "flot:0.1:0.01", "agr:1"] {
            assert!(ecart_cov(s, Variante::Normale) <= 1e-12, "{s}");
        }
        assert!(ecart_cov("ape:0.5:6", Variante::AgrafeMalOrientee) > 1e-3);
        assert!(ecart_cov("agr:1", Variante::AgrafeMalOrientee) > 1e-3);
        assert!(ecart_cov("dec", Variante::DecOrdreInverse) > 1e-3);
        assert!(ecart_cov("ape:0.5:6", Variante::AgrafesPlusSeulement) <= 1e-12, "mutant de réflexion : jauge intacte");
    }

    #[test]
    fn unitarite_et_mutant() {
        let c = aleatoire(9);
        for s in ["dec", "ape:0.5:6", "flot:0.1:0.01", "agr:1"] {
            assert!(bloquer_operateur(&c, &op(s), &OptionsOperateur::normale(2)).unwrap().grossier.defaut_unitarite() <= 1e-14, "{s}");
        }
        let m = bloquer_operateur(&c, &op("ape:0.5:6"), &OptionsOperateur { fils: 2, variante: Variante::SansProjection }).unwrap();
        assert!(m.grossier.defaut_unitarite() > 1e-3);
    }

    #[test]
    fn entrees_invalides() {
        for l in [0, 1, 2, 4] {
            assert!(bloquer_operateur(&froid(l), &op("dec"), &OptionsOperateur::normale(1)).is_err());
        }
        assert!(bloquer_operateur(&froid(3), &op("dec"), &OptionsOperateur::normale(0)).is_err());
        let mut c = froid(3);
        c.q[5] = f64::NAN;
        assert!(bloquer_operateur(&c, &op("ape:0.5:1"), &OptionsOperateur::normale(1)).is_err());
        // somme nulle : champ froid sauf U(0, μ = 0) = −1 ⇒ s = 0,5·(−1) + (0,5/6)·6·1 = 0 exactement ⇒ refus (projection ambiguë)
        let mut a = froid(3);
        a.q[0] = -1.0;
        assert!(matches!(bloquer_operateur(&a, &op("ape:0.5:1"), &OptionsOperateur::normale(1)), Err(Erreur::Config(_))));
        assert!(bloquer_operateur(&a, &op("ape:0.5:0"), &OptionsOperateur::normale(1)).is_ok());
    }
}
