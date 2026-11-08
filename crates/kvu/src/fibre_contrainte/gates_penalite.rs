//! Gates de la route A : FIB-A-LIBRE (+ mutant β × 1,05), FIB-UBAR-LOCAL, et FIB-STAT-CALIB (outil d'analyse de WP12a).
//! Lourdes : `#[ignore]`, `cargo test --release --lib fibre_contrainte::gates_penalite -- --ignored --nocapture --test-threads=1`.

use super::alea::{Flux, TAG_OUTIL};
use super::gates_commun::*;
use super::jacobien::Reseau;
use super::penalite::{ChainePenalite, ParamsPenalite};
use super::stat::*;
use crate::commun::ChampHote;
use crate::heritage::su2_hb::CpuSu2;
use crate::rg::bloc::{bloquer, OptionsBloc};
use crate::rg::geometrie::Geometrie;
use std::sync::Arc;

const S_DIFF_ASYM_M3: f64 = SEUILS.diff_asym_m3;

/// Plaquette moyenne après chaque balayage de Metropolis (κ = ∞), L = 4, départ froid.
fn serie_a_libre(beta: f64, delta: f64, n: usize, graine: u64) -> Vec<f64> {
    let champ = ChampHote::froid(4).unwrap();
    let params = ParamsPenalite { beta, kappa: None, delta };
    let mut ch = ChainePenalite::nouvelle(None, Vec::new(), champ, params, Flux::nouveau(graine), 0).unwrap();
    (0..n)
        .map(|i| {
            ch.balayage(i as u32);
            ch.plaquette()
        })
        .collect()
}

/// Plaquette moyenne après chaque pas du générateur de référence (bain thermique + 4 OR), L = 4.
fn serie_reference(beta: f64, n: usize, graine: u64) -> Vec<f64> {
    let mut g = CpuSu2::new(4, graine).unwrap();
    g.init(true);
    (0..n)
        .map(|_| {
            g.step(beta, 4);
            g.measure()[0]
        })
        .collect()
}

const BETAS: [(f64, f64); 3] = [(2.3, 0.4), (2.8, 0.35), (8.0, 0.12)];

#[test]
#[ignore = "gate lourde FIB-A-LIBRE"]
fn fib_a_libre() {
    let n_a = 400_000;
    let n_ref = 120_000;
    let mut pire: f64 = 0.0;
    let mut mutant_max: f64 = 0.0;
    let mut temoins = 0usize;
    let mut skip = false;
    for (k, (beta, delta)) in BETAS.iter().enumerate() {
        let r = analyser(&serie_reference(*beta, n_ref, 26_102_612_711 + k as u64));
        let a = analyser(&serie_a_libre(*beta, *delta, n_a, 26_102_612_701 + k as u64));
        for (nom, an) in [("référence", &r), ("route A", &a)] {
            ligne("INFO", "FIB-A-LIBRE", &format!("β = {beta} {nom} : ⟨P⟩ = {:.6} ± {:.2e}, τ_int = {:.2}, statut {}", an.moyenne, an.erreur, an.tau_int, an.statut.texte()));
            if !domaine_campagne(an) {
                skip = true;
            }
        }
        if domaine_campagne(&a) && domaine_campagne(&r) {
            let t = t_comparaison(&a, &r, 0.0);
            ligne("INFO", "FIB-A-LIBRE", &format!("β = {beta} : t(A − référence) = {t:+.3}"));
            pire = maxn(pire, t.abs());
        }
        // mutant β × 1,05 dans la route A
        let m = analyser(&serie_a_libre(*beta * 1.05, *delta, 100_000, 26_102_612_721 + k as u64));
        if domaine_campagne(&m) && domaine_campagne(&r) {
            let t = t_comparaison(&m, &r, 0.0);
            ligne("INFO", "FIB-A-LIBRE-MUTANT", &format!("β = {beta} : t(mutant β×1,05 − référence) = {t:+.3}"));
            mutant_max = maxn(mutant_max, t.abs());
            temoins += 1;
        } else {
            ligne("SKIP", "FIB-A-LIBRE-MUTANT-CMP", &format!("β = {beta} : statut mutant {} / référence {} (non conclusif)", m.statut.texte(), r.statut.texte()));
        }
    }
    if skip {
        ligne("SKIP", "FIB-A-LIBRE", "PUISSANCE ou statut non vert (voir INFO)");
        return;
    }
    ligne(if pire < S_DIFF_ASYM_M3 { "PASS" } else { "FAIL" }, "FIB-A-LIBRE", &format!("max |t| = {pire:.3} < {S_DIFF_ASYM_M3} (m = 3)"));
    if temoins == 0 {
        ligne("SKIP", "FIB-A-LIBRE-MUTANT", "AUCUN TEMOIN VALIDE");
        return;
    }
    ligne(if mutant_max >= S_DIFF_ASYM_M3 { "PASS" } else { "FAIL" }, "FIB-A-LIBRE-MUTANT", &format!("mutant β×1,05 : max |t| = {mutant_max:.3} ≥ {S_DIFF_ASYM_M3} sur {temoins} comparaisons valides (ROUGE exigé ; sinon ECHEC_OUTIL)"));
    assert!(pire < S_DIFF_ASYM_M3 && mutant_max >= S_DIFF_ASYM_M3);
}

#[test]
#[ignore = "gate FIB-UBAR-LOCAL"]
fn fib_ubar_local() {
    let res = Arc::new(Reseau::nouveau(6).unwrap());
    let champ = champ_wilson(6, 2.8, 26_102_612_801, 2000);
    // cible : Ū de la configuration elle-même (la pénalité ne gêne pas les propositions)
    let v = ubar_pool(&res, &champ, 8);
    ligne("INFO", "EMPREINTE", &format!("UBAR-LOCAL : U0 sha256 {} ; V sha256 {}", empreinte(&champ), empreinte_quaternions(&v)));
    let params = ParamsPenalite { beta: 2.8, kappa: Some(2e-3), delta: 0.05 };
    let mut ch = ChainePenalite::nouvelle(Some(res.clone()), v, champ, params, Flux::nouveau(26_102_612_802), 0).unwrap();
    let n_liens = 4 * 6usize.pow(4);
    let (mut acceptees, mut identiques) = (0usize, true);
    for i in 0..1000usize {
        let li = (i * 7919) % n_liens;
        acceptees += usize::from(ch.proposer_un(i as u32, li));
        // identité au bit après CHAQUE mise à jour : Ū local contre rg::bloc complet sur le champ courant
        let complet = bloquer(ch.champ(), &OptionsBloc { geometrie: Geometrie::TroisCentre, fils: 8, angle_so3: false }).unwrap().grossier;
        identiques &= ch.ubar().iter().enumerate().all(|(c, u)| (0..4).all(|k| u[k].to_bits() == complet.q[4 * c + k].to_bits()));
        if !identiques {
            ligne("INFO", "FIB-UBAR-LOCAL", &format!("première différence après la mise à jour {i}"));
            break;
        }
    }
    let ok = identiques && acceptees > 0 && acceptees < 1000;
    ligne(if ok { "PASS" } else { "FAIL" }, "FIB-UBAR-LOCAL", &format!("1000 mises à jour locales (L = 6, κ = 2e-3, δ = 0,05, {acceptees} acceptées) : Ū local = rg::bloc complet après chaque mise à jour, identité au bit sur {} liens grossiers", res.n_c));
    assert!(ok);
}

/// |t| par réplique valide : [moyenne, différence de même τ, intercept, résidu du milieu, différence asymétrique].
/// Répliques AR(1) (τ_int, variance 1, moyenne nulle) : 3 séries indépendantes de même longueur, analysées au protocole de la campagne, plus une
/// 4ᵉ série de τ_int = max(τ/10 ; 0,5) (comparaison asymétrique, comme FIB-A-LIBRE : route A lente contre référence rapide). Une réplique n'est
/// « valide » que si tous les statuts sont verts ET si toutes les erreurs et statistiques dérivées sont finies et > 0.
fn echantillon_t(tau: f64, n: usize, repl: u32, graine: u64) -> (Vec<[f64; 5]>, usize) {
    use rayon::prelude::*;
    let flux = Flux::nouveau(graine);
    let serie = |r: u32, c: u32, tau_c: f64| -> Vec<f64> {
        let phi = (2.0 * tau_c - 1.0) / (2.0 * tau_c + 1.0);
        let s = (1.0 - phi * phi).sqrt();
        let mut prev = flux.gauss(r, 3 * c, 0, TAG_OUTIL);
        (0..n)
            .map(|i| {
                prev = phi * prev + s * flux.gauss(r, 3 * c + 1, i as u32, TAG_OUTIL);
                prev
            })
            .collect()
    };
    let res: Vec<Option<[f64; 5]>> = (0..repl)
        .into_par_iter()
        .map(|r| {
            let xs = [serie(r, 0, tau), serie(r, 1, tau), serie(r, 2, tau), serie(r, 3, (tau / 10.0).max(0.5))];
            let an: Vec<Analyse> = xs.iter().map(|x| analyser(x)).collect();
            if an.iter().any(|a| a.statut != Statut::Ok) {
                return None;
            }
            let ta = (an[0].moyenne / an[0].erreur).abs();
            let tb = t_comparaison(&an[0], &an[1], 0.0).abs();
            let td = t_comparaison(&an[0], &an[3], 0.0).abs();
            let pts: Vec<(f64, &Analyse)> = vec![(1.0, &an[0]), (2.0, &an[1]), (4.0, &an[2])];
            let (o0, s0, rr, sr) = ajuster_jackknife(&pts)?;
            let t = [ta, tb, (o0 / s0).abs(), (rr / sr).abs(), td];
            t.iter().all(|x| x.is_finite()).then_some(t)
        })
        .collect();
    let valides: Vec<[f64; 5]> = res.iter().flatten().copied().collect();
    let invalides = repl as usize - valides.len();
    (valides, invalides)
}

/// Taux de fausse alarme des 8 statistiques (moyenne m = 1, 3, 4, 16 ; différence m = 3 ; intercept m = 16 ; résidu m = 16 ; différence asymétrique m = 3).
fn fausse_alarme(tau: f64, n: usize, repl: u32, graine: u64, naif: bool, sx: &Seuils) -> (Vec<f64>, usize) {
    use rayon::prelude::*;
    if naif {
        let flux = Flux::nouveau(graine);
        let res: Vec<Option<f64>> = (0..repl)
            .into_par_iter()
            .map(|r| {
                // « τ_int ignoré » : erreur naïve σ/√n, sans blocs ni τ (statistique de la moyenne seulement)
                let phi = (2.0 * tau - 1.0) / (2.0 * tau + 1.0);
                let s = (1.0 - phi * phi).sqrt();
                let mut prev = flux.gauss(r, 0, 0, TAG_OUTIL);
                let x: Vec<f64> = (0..n)
                    .map(|i| {
                        prev = phi * prev + s * flux.gauss(r, 1, i as u32, TAG_OUTIL);
                        prev
                    })
                    .collect();
                let y = &x[n / 5..];
                let m = y.iter().sum::<f64>() / y.len() as f64;
                let v = y.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / (y.len() - 1) as f64;
                let t = (m / (v / y.len() as f64).sqrt()).abs();
                t.is_finite().then_some(t)
            })
            .collect();
        let v: Vec<f64> = res.iter().flatten().copied().collect();
        let seuils = [sx.moy_m1, sx.moy_m3, sx.moy_m4, sx.moy_m16];
        let mut taux: Vec<f64> = seuils.iter().map(|q| v.iter().filter(|t| **t >= *q).count() as f64 / v.len().max(1) as f64).collect();
        taux.extend([0.0; 4]);
        return (taux, repl as usize - v.len());
    }
    let (t, invalides) = echantillon_t(tau, n, repl, graine);
    let part = |k: usize, q: f64| t.iter().filter(|v| v[k] >= q).count() as f64 / t.len().max(1) as f64;
    let taux = vec![part(0, sx.moy_m1), part(0, sx.moy_m3), part(0, sx.moy_m4), part(0, sx.moy_m16), part(1, sx.diff_m3), part(2, sx.intercept_m16), part(3, sx.residu_m16), part(4, sx.diff_asym_m3)];
    (taux, invalides)
}

const ALPHA_CALIB: [f64; 8] = [0.01, 0.01 / 3.0, 0.01 / 4.0, 0.01 / 16.0, 0.01 / 3.0, 0.01 / 16.0, 0.01 / 16.0, 0.01 / 3.0];
const NOMS_CALIB: [&str; 8] = ["moyenne m=1", "moyenne m=3", "moyenne m=4", "moyenne m=16", "différence m=3", "intercept d'ajustement m=16", "résidu du milieu m=16", "différence asymétrique (τ, τ/10) m=3"];
/// (étiquette, facteur de longueur) : blocs ≈ 15 τ (cas le plus défavorable admis par « blocs ≥ 10 τ ») et ≈ 60 τ ; les campagnes ont des blocs de
/// 160 τ à plus de 1000 τ (plus favorables).
const LONGUEURS_CALIB: [(&str, f64); 2] = [("blocs ≈ 15 τ", 1.0), ("blocs ≈ 60 τ", 4.0)];

fn longueur_calib(tau: f64, facteur: f64) -> usize {
    ((40.0 * 10.0 * tau * 1.5 / 0.8 * facteur) as usize).max(2000)
}

/// DÉRIVATION des seuils empiriques (avenant WP12a-2) : quantile empirique (1 − α) de |t| par statistique et par cellule (τ, longueur), 10⁵ répliques,
/// graine de dérivation 26102612992 (INDÉPENDANTE de celle de la vérification) ; le seuil retenu est le maximum sur les six cellules, arrondi par excès
/// à 0,01.
#[test]
#[ignore = "derivation des seuils empiriques"]
fn derivation_seuils_empiriques() {
    let repl = 100_000u32;
    let mut maxi = [0.0f64; 8];
    for (etiquette, facteur) in LONGUEURS_CALIB {
        for tau in [1.0, 10.0, 50.0] {
            let n = longueur_calib(tau, facteur);
            let (t, inval) = echantillon_t(tau, n, repl, 26_102_612_992);
            let col = |k: usize| -> Vec<f64> {
                let mut v: Vec<f64> = t.iter().map(|x| x[k]).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v
            };
            let stats = [col(0), col(1), col(2), col(3), col(4)];
            let idx = [0usize, 0, 0, 0, 1, 2, 3, 4];
            for k in 0..8 {
                let v = &stats[idx[k]];
                let rang = ((1.0 - ALPHA_CALIB[k]) * v.len() as f64).ceil() as usize;
                let q = v[rang.min(v.len() - 1)];
                maxi[k] = maxi[k].max(q);
                ligne("INFO", "QUANTILE", &format!("{etiquette}, τ = {tau}, n = {n} : {} : quantile empirique (1 − α) de |t| = {q:.4} ({} répliques valides, {inval} invalides)", NOMS_CALIB[k], v.len()));
            }
        }
    }
    let arrondi = |x: f64| (x * 100.0).ceil() / 100.0;
    ligne("INFO", "SEUILS-EMPIRIQUES", &format!("moy_m1 {:.2} moy_m3 {:.2} moy_m4 {:.2} moy_m16 {:.2} diff_m3 {:.2} intercept_m16 {:.2} residu_m16 {:.2} diff_asym_m3 {:.2}", arrondi(maxi[0]), arrondi(maxi[1]), arrondi(maxi[2]), arrondi(maxi[3]), arrondi(maxi[4]), arrondi(maxi[5]), arrondi(maxi[6]), arrondi(maxi[7])));
}

#[test]
#[ignore = "gate lourde FIB-STAT-CALIB"]
fn fib_stat_calib() {
    let repl = 10_000u32;
    let mut ok = true;
    for (etiquette, facteur) in LONGUEURS_CALIB {
        for tau in [1.0, 10.0, 50.0] {
            let n = longueur_calib(tau, facteur);
            let (taux, rejets) = fausse_alarme(tau, n, repl, 26_102_612_990, false, &SEUILS);
            let valide_frac = 1.0 - rejets as f64 / repl as f64;
            // validité exigée à blocs ≈ 60 τ (les campagnes sont à blocs ≥ 160 τ) ; à blocs ≈ 15 τ elle est rapportée
            let exigee = facteur > 1.0;
            let bon_valide = valide_frac >= 0.9;
            // effectif exploitable minimal, exigé dans chaque cellule (sinon les taux de fausse alarme n'ont pas de sens)
            let effectif_ok = repl as usize - rejets >= 5000;
            ok &= effectif_ok;
            ligne(if effectif_ok { "PASS" } else { "FAIL" }, "FIB-STAT-CALIB-EFFECTIF", &format!("{etiquette}, τ_int = {tau} : {} répliques valides ≥ 5000 exigées", repl as usize - rejets));
            ok &= bon_valide || !exigee;
            ligne(if bon_valide || !exigee { "PASS" } else { "FAIL" }, "FIB-STAT-CALIB-VALIDITE", &format!("{etiquette}, τ_int = {tau}, n = {n} : {:.1} % de répliques valides ({})", 100.0 * valide_frac, if exigee { "≥ 90 % exigé" } else { "rapporté, non exigé à blocs ≈ 15 τ" }));
            for k in 0..8 {
                let bon = taux[k] <= 2.0 * ALPHA_CALIB[k];
                ok &= bon;
                ligne(if bon { "PASS" } else { "FAIL" }, "FIB-STAT-CALIB", &format!("{etiquette}, τ_int = {tau} : {} : fausse alarme {:.2e} ≤ 2α = {:.2e}", NOMS_CALIB[k], taux[k], 2.0 * ALPHA_CALIB[k]));
            }
        }
    }
    // mutant : τ_int ignoré (erreur naïve) à τ_int = 10 : fausse alarme > 2α sur au moins un seuil de la moyenne
    let (taux, _) = fausse_alarme(10.0, 8_000, repl, 26_102_612_991, true, &SEUILS);
    let rouge = (0..4).any(|k| taux[k] > 2.0 * ALPHA_CALIB[k]);
    ok &= rouge;
    ligne(if rouge { "PASS" } else { "FAIL" }, "FIB-STAT-CALIB-MUTANT", &format!("erreur naïve à τ_int = 10 : fausses alarmes {:?} (ROUGE exigé : > 2α sur au moins un seuil)", &taux[..4]));
    assert!(ok);
}
