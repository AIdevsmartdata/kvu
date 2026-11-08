//! Estimateur Schwinger–Dyson à un lien (ARCHITECTURE §4.3, port de `n3_blocage_beta_eff.py` l. 1-13 et 24-70).
//!
//! Pour une action de Wilson de couplage b, la mesure conditionnelle d'un lien est ∝ exp((b/2) ReTr(U A)), A = somme des 6 agrafes,
//! A = k V avec V ∈ SU(2) et k = √det A ; l'espérance conditionnelle donne ⟨½ReTr(U A)⟩ = k I₂(b k)/I₁(b k) et donc, sommée sur
//! les liens :  Σ_ℓ ½ReTr(U_ℓ A_ℓ) = Σ_ℓ k_ℓ I₂(b k_ℓ)/I₁(b k_ℓ), résolue en b par bissection. Exacte si l'action effective est de
//! Wilson (**réponse connue : b = β sur un ensemble de Wilson non bloqué**) ; sinon c'est une PROJECTION de Wilson (heuristique).
//!
//! En quaternions : A est une somme de quaternions, k = |A| (norme de R⁴ = √det de la matrice 2×2) et ½ReTr(U A) = Re(U·A).
//!
//! Deux modes d'évaluation du rapport :
//! - `Exact` : fraction continue ([`super::bessel`]) ;
//! - `TableauScript` : reproduit le script Python, qui tabule le rapport sur 4000 points log-espacés de 1e-6 à 400 et interpole
//!   linéairement (`np.interp`) — d'où une erreur d'interpolation ~1e-6 sur b ; ce mode ne sert que de référence de RG-N3.
//!
//! Origine du code : premier jet de DeepSeek Flash (prompt `scratch/wp8/ds/prompt_sd.txt`), relu, corrigé et réécrit par le pilote
//! (réutilisation des utilitaires de `rg::bloc`, géomespace numpy exact, tests complétés).

use super::bessel;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul};
use crate::rg::bloc::{coordonnees, deplacer, lire, Q};
use rayon::prelude::*;
use std::sync::OnceLock;

/// Taille des tranches de la somme parallèle (le résultat ne dépend pas du nombre de fils).
const TRANCHE: usize = 4096;
/// Bornes de la bissection (celles du script).
const B_BAS: f64 = 0.0;
const B_HAUT: f64 = 50.0;
/// Pas de bissection : 80 (script) pour le tableau ; borne de sécurité de la méthode d'Illinois pour l'exact.
const PAS_TABLEAU: usize = 80;
const PAS_EXACT_MAX: usize = 200;

/// Statistiques à un lien d'une (ou plusieurs) configuration(s) : somme des w et un k par lien.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StatsSd {
    pub somme_w: f64,
    pub ks: Vec<f64>,
    pub n_configs: usize,
}

impl StatsSd {
    /// Ajoute les statistiques d'une autre configuration (ordre conservé).
    pub fn fusionner(&mut self, autre: &StatsSd) {
        self.somme_w += autre.somme_w;
        self.ks.extend_from_slice(&autre.ks);
        self.n_configs += autre.n_configs;
    }

    /// Somme (dans l'ordre) de plusieurs statistiques.
    pub fn somme<'a>(it: impl IntoIterator<Item = &'a StatsSd>) -> StatsSd {
        let mut s = StatsSd::default();
        for x in it {
            s.fusionner(x);
        }
        s
    }
}

/// Rapport de Bessel de l'équation SD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RapportBessel {
    /// SU(2) : le bon.
    I2SurI1,
    /// MUTANT : forme U(1).
    I1SurI0,
}

/// Mode d'évaluation du rapport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeRapport {
    Exact,
    TableauScript,
}

/// Somme des six agrafes du lien (mu, x) (quaternions), même ordre d'addition que le script.
pub(super) fn agrafes(champ: &ChampHote, mu: usize, x: [usize; 4]) -> Q {
    let l = champ.l;
    let mut a = [0.0; 4];
    for nu in 0..4 {
        if nu == mu {
            continue;
        }
        let x_mu = deplacer(x, mu, 1, l);
        let x_nu = deplacer(x, nu, 1, l);
        let x_mnu = deplacer(x, nu, -1, l);
        let x_mu_mnu = deplacer(x_mu, nu, -1, l);
        // sh(Cν, μ, 1) · dag(sh(Cμ, ν, 1)) · dag(Cν)
        let t1 = qmul(qmul(lire(champ, x_mu, nu), qdag(lire(champ, x_nu, mu))), qdag(lire(champ, x, nu)));
        // dag(sh(sh(Cν, μ, 1), ν, −1)) · dag(sh(Cμ, ν, −1)) · sh(Cν, ν, −1)
        let t2 = qmul(qmul(qdag(lire(champ, x_mu_mnu, nu)), qdag(lire(champ, x_mnu, mu))), lire(champ, x_mnu, nu));
        for ((ai, u), v) in a.iter_mut().zip(t1).zip(t2) {
            *ai += u;
            *ai += v;
        }
    }
    a
}

/// (somme des w, k de chaque lien) d'UNE configuration. Ordre des liens : μ = 0..3, puis site croissant (concaténation du script) ;
/// la somme de w est faite séquentiellement dans cet ordre (déterministe quel que soit le nombre de fils).
pub fn stats_sd(champ: &ChampHote) -> Resultat<StatsSd> {
    let l = champ.l;
    let v = l
        .checked_pow(4)
        .filter(|_| l >= 2)
        .ok_or_else(|| Erreur::Config(format!("L = {l} invalide (≥ 2 requis)")))?;
    if champ.q.len() != 16 * v {
        return Err(Erreur::Config(format!("champ de longueur {} au lieu de {}", champ.q.len(), 16 * v)));
    }
    let mut stats = StatsSd { somme_w: 0.0, ks: Vec::with_capacity(4 * v), n_configs: 1 };
    for mu in 0..4 {
        let kw: Vec<(f64, f64)> = (0..v)
            .into_par_iter()
            .map(|s| {
                let x = coordonnees(s, l);
                let u = champ.lien(mu * v + s);
                let a = agrafes(champ, mu, x);
                let k = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt();
                (k, qmul(u, a)[0])
            })
            .collect();
        for (k, w) in kw {
            stats.ks.push(k);
            stats.somme_w += w;
        }
    }
    Ok(stats)
}

/// Table du script : xs = [0] + geomspace(1e-6, 400, 4000), rs = rapport(xs) (x/4 si x ≤ 1e-8).
struct Table {
    xs: Vec<f64>,
    rs: Vec<f64>,
}

static TABLE_I2: OnceLock<Table> = OnceLock::new();
static TABLE_I1: OnceLock<Table> = OnceLock::new();

fn construire_table(rapport: RapportBessel) -> Table {
    const N: usize = 4000;
    // numpy.geomspace : 10**linspace(log10(a), log10(b), N), extrémités posées exactement.
    let (la, lb) = (1e-6_f64.log10(), 400.0_f64.log10());
    let pas = (lb - la) / (N - 1) as f64;
    let mut xs = Vec::with_capacity(N + 1);
    xs.push(0.0);
    for i in 0..N {
        xs.push(match i {
            0 => 1e-6,
            i if i == N - 1 => 400.0,
            i => 10f64.powf(la + i as f64 * pas),
        });
    }
    let rs = xs.iter().map(|&x| if x > 1e-8 { bessel::rapport(rapport, x) } else { x / 4.0 }).collect();
    Table { xs, rs }
}

fn table(rapport: RapportBessel) -> &'static Table {
    match rapport {
        RapportBessel::I2SurI1 => TABLE_I2.get_or_init(|| construire_table(rapport)),
        RapportBessel::I1SurI0 => TABLE_I1.get_or_init(|| construire_table(rapport)),
    }
}

/// np.interp : linéaire par morceaux, constante hors des nœuds.
fn interp(t: &Table, x: f64) -> f64 {
    let dernier = t.xs.len() - 1;
    if x <= t.xs[0] {
        return t.rs[0];
    }
    if x >= t.xs[dernier] {
        return t.rs[dernier];
    }
    // plus grand i tel que xs[i] ≤ x
    let i = t.xs.partition_point(|&v| v <= x) - 1;
    let (x0, x1) = (t.xs[i], t.xs[i + 1]);
    let (y0, y1) = (t.rs[i], t.rs[i + 1]);
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// rhs(b) = Σ_ℓ k_ℓ r(b k_ℓ) : tranches de taille fixe, combinées dans l'ordre.
fn rhs(b: f64, ks: &[f64], rapport: RapportBessel, mode: ModeRapport) -> f64 {
    let tab = match mode {
        ModeRapport::Exact => None,
        ModeRapport::TableauScript => Some(table(rapport)),
    };
    let sommes: Vec<f64> = ks
        .par_chunks(TRANCHE)
        .map(|c| {
            c.iter()
                .map(|&k| {
                    let x = b * k;
                    k * match tab {
                        Some(t) => interp(t, x),
                        None => bessel::rapport(rapport, x),
                    }
                })
                .sum::<f64>()
        })
        .collect();
    sommes.iter().sum()
}

/// Résout `somme_w = Σ_ℓ k_ℓ r(b k_ℓ)` en b, racine unique de f(b) = rhs(b) − somme_w (croissante) encadrée dans [0, 50]. `None` (le « ∞ » du
/// script) si rhs(50) < somme_w.
/// - `TableauScript` : BISSECTION pure à 80 pas, exactement celle du script (référence de RG-N3) ;
/// - `Exact` : bissection protégée par l'interpolation d'Illinois (le crochet est toujours conservé ; arrêt quand la largeur du crochet
///   ≤ 4e-16 b ou |f| ≤ 1e-15 · somme_w) : même racine à 1e-15 près, ~8 évaluations de rhs au lieu de 56 (chaque évaluation coûte
///   |liens| appels de la fraction continue).
pub fn beta_sd(stats: &StatsSd, rapport: RapportBessel, mode: ModeRapport) -> Option<f64> {
    let lhs = stats.somme_w;
    if !lhs.is_finite() || stats.ks.is_empty() {
        return None;
    }
    let f = |b: f64| rhs(b, &stats.ks, rapport, mode) - lhs;
    let (mut bas, mut haut) = (B_BAS, B_HAUT);
    let f_haut = f(haut);
    if f_haut < 0.0 {
        return None;
    }
    if mode == ModeRapport::TableauScript {
        for _ in 0..PAS_TABLEAU {
            let m = 0.5 * (bas + haut);
            if rhs(m, &stats.ks, rapport, mode) < lhs {
                bas = m;
            } else {
                haut = m;
            }
        }
        return Some(0.5 * (bas + haut));
    }
    let (mut f_bas, mut f_haut) = (-lhs, f_haut); // rhs(0) = 0
    let mut dernier_cote = 0i8;
    for _ in 0..PAS_EXACT_MAX {
        // sécante sur le crochet, ramenée dans l'intérieur ; à défaut le milieu
        let denom = f_haut - f_bas;
        let mut m = if denom > 0.0 { bas - f_bas * (haut - bas) / denom } else { 0.5 * (bas + haut) };
        if !(m > bas && m < haut) {
            m = 0.5 * (bas + haut);
        }
        let fm = f(m);
        if fm.abs() <= 1e-15 * lhs.abs() || haut - bas <= 4e-16 * haut {
            return Some(m);
        }
        if fm < 0.0 {
            bas = m;
            f_bas = fm;
            if dernier_cote == -1 {
                f_haut *= 0.5; // Illinois : évite la stagnation d'une borne
            }
            dernier_cote = -1;
        } else {
            haut = m;
            f_haut = fm;
            if dernier_cote == 1 {
                f_bas *= 0.5;
            }
            dernier_cote = 1;
        }
    }
    Some(0.5 * (bas + haut))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn champ_froid(l: usize) -> ChampHote {
        ChampHote::froid(l).unwrap()
    }

    #[test]
    fn champ_froid_k_6_w_6() {
        let s = stats_sd(&champ_froid(4)).unwrap();
        let v = 256;
        assert_eq!(s.ks.len(), 4 * v);
        assert_eq!(s.n_configs, 1);
        assert!((s.somme_w - (4 * v * 6) as f64).abs() < 1e-12);
        assert!(s.ks.iter().all(|&k| (k - 6.0).abs() < 1e-14));
    }

    #[test]
    fn longueur_invalide_refusee() {
        let mut c = champ_froid(4);
        c.q.pop();
        assert!(stats_sd(&c).is_err());
    }

    /// Cas synthétique : N liens de même k, somme_w construite avec le rapport exact à b0.
    fn synthetique(rapport: RapportBessel) -> StatsSd {
        let (n, k, b0) = (10usize, 3.0, 2.3);
        StatsSd { somme_w: n as f64 * k * bessel::rapport(rapport, b0 * k), ks: vec![k; n], n_configs: 1 }
    }

    #[test]
    fn retrouve_b0_en_mode_exact() {
        let b = beta_sd(&synthetique(RapportBessel::I2SurI1), RapportBessel::I2SurI1, ModeRapport::Exact).unwrap();
        assert!((b - 2.3).abs() < 1e-12, "b = {b}");
    }

    #[test]
    fn mode_tableau_proche_de_l_exact_mais_pas_egal() {
        let s = synthetique(RapportBessel::I2SurI1);
        let be = beta_sd(&s, RapportBessel::I2SurI1, ModeRapport::Exact).unwrap();
        let bt = beta_sd(&s, RapportBessel::I2SurI1, ModeRapport::TableauScript).unwrap();
        assert!((be - bt).abs() < 1e-3, "{be} {bt}");
        assert!((be - bt).abs() > 0.0);
    }

    #[test]
    fn illinois_egale_la_bissection_pure_a_1e_14() {
        // k variés ; somme_w = rhs(b0) avec b0 = 5,7 puis perturbée
        let ks: Vec<f64> = (0..500).map(|i| 0.5 + 5.5 * ((i * 37 % 101) as f64) / 100.0).collect();
        for b0 in [0.3, 2.3, 5.7, 20.0] {
            let w = rhs(b0, &ks, RapportBessel::I2SurI1, ModeRapport::Exact) * 1.0000001;
            let s = StatsSd { somme_w: w, ks: ks.clone(), n_configs: 1 };
            let b = beta_sd(&s, RapportBessel::I2SurI1, ModeRapport::Exact).unwrap();
            let (mut bas, mut haut) = (0.0, 50.0);
            for _ in 0..80 {
                let m = 0.5 * (bas + haut);
                if rhs(m, &ks, RapportBessel::I2SurI1, ModeRapport::Exact) < w {
                    bas = m;
                } else {
                    haut = m;
                }
            }
            assert!((b - 0.5 * (bas + haut)).abs() <= 1e-13 * b, "b0 = {b0} : {b} contre {}", 0.5 * (bas + haut));
        }
    }

    #[test]
    fn none_quand_somme_w_trop_grande() {
        let s = StatsSd { somme_w: 1e9, ks: vec![3.0; 10], n_configs: 1 };
        assert!(beta_sd(&s, RapportBessel::I2SurI1, ModeRapport::Exact).is_none());
        assert!(beta_sd(&StatsSd::default(), RapportBessel::I2SurI1, ModeRapport::Exact).is_none());
    }

    #[test]
    fn le_mutant_u1_s_ecarte() {
        let s = synthetique(RapportBessel::I2SurI1);
        let b = beta_sd(&s, RapportBessel::I1SurI0, ModeRapport::Exact).unwrap();
        assert!((b / 2.3 - 1.0).abs() > 0.01, "mutant trop proche : {b}");
    }

    #[test]
    fn fusion_et_somme() {
        let a = stats_sd(&champ_froid(2)).unwrap();
        let s = StatsSd::somme([&a, &a]);
        assert_eq!(s.n_configs, 2);
        assert_eq!(s.ks.len(), 2 * a.ks.len());
        assert_eq!(s.somme_w, 2.0 * a.somme_w);
    }

    #[test]
    fn la_somme_ne_depend_pas_du_nombre_de_fils() {
        // champ pseudo-aléatoire déterministe L = 4
        let mut c = champ_froid(4);
        let mut etat = 12345u64;
        for q in c.q.chunks_exact_mut(4) {
            let mut v = [0.0; 4];
            for x in v.iter_mut() {
                etat = etat.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *x = ((etat >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
            }
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
            for (qi, vi) in q.iter_mut().zip(v) {
                *qi = vi / n;
            }
        }
        let faire = |fils| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(fils).build().unwrap();
            pool.install(|| {
                let s = stats_sd(&c).unwrap();
                (beta_sd(&s, RapportBessel::I2SurI1, ModeRapport::Exact), s)
            })
        };
        assert_eq!(faire(1), faire(5));
    }
}
