//! Validation CPU des gates WP19 sur des mesures synthétiques, sans appeler mesurer_cpu.
use kvu::obs::topo::{FlotA2, FlotT0, RecordTopo, SerieCooling, IDX_T3};
use kvu::obs::topo_gates::{
    analyser, clopper_pearson, extension_configs, jackknife_chaines, lire_jsonl, obs4_rejeu,
    tau_int_sokal, topo_lit, topo_q, LigneGate, Verdict, QUANTILE_T31,
};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

struct SplitMix64(u64);
impl SplitMix64 {
    fn uniforme(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniforme().ln()).sqrt() * (std::f64::consts::TAU * self.uniforme()).cos()
    }
}

fn proche(obtenu: f64, attendu: f64, tolerance: f64) {
    assert!(
        (obtenu - attendu).abs() <= tolerance,
        "obtenu={obtenu}, attendu={attendu}, tolérance={tolerance}"
    );
}

fn record(chaine: usize, cfg: usize, q: f64) -> RecordTopo {
    RecordTopo {
        ens: "synthétique".into(),
        chaine,
        cfg,
        backend_gen: if chaine <= 16 { "hip" } else { "cuda" }.into(),
        backend_mesure: "cpu".into(),
        l: 16,
        plaq0: 0.6364293 + chaine as f64 * 1e-7,
        t0: Some(0.2),
        flot_a2: FlotA2 {
            e_plaq: vec![1.0; 4],
            q_imp: vec![q; 4],
            q_tre: vec![q; 4],
        },
        flot_t0: Some(FlotT0 {
            t_eff: vec![0.2, 0.4, 0.6, 1.0],
            q_imp: vec![q; 4],
            q_tre: vec![q; 4],
        }),
        cooling: SerieCooling {
            q_imp: vec![q; 21],
            q_tre: vec![q.round() / 0.95; 21],
            action: (0..=20).map(|k| 100.0 - k as f64).collect(),
        },
        ..RecordTopo::default()
    }
}

fn jeu_q() -> Vec<RecordTopo> {
    let mut rng = SplitMix64(19);
    (0..100)
        .map(|i| {
            record(
                i / 25 + 1,
                i % 25 + 1,
                (i % 9) as f64 - 4.0 + 0.09 * (rng.uniforme() - 0.5),
            )
        })
        .collect()
}

fn jeu_lit(cible: f64, par_chaine: usize) -> Vec<RecordTopo> {
    let mut rng = SplitMix64(0x193ac);
    let mut recs = Vec::new();
    for chaine in 1..=32 {
        for cfg in 1..=par_chaine {
            // L'arrondi gaussien ajoute approximativement 1/12 à la variance.
            let q = (rng.normal() * (cible - 1.0 / 12.0).sqrt()).round();
            let mut r = record(chaine, cfg, q + 0.18 * (rng.uniforme() - 0.5));
            r.cooling
                .q_tre
                .fill(q / 0.95 + 0.01 * (rng.uniforme() - 0.5));
            recs.push(r);
        }
    }
    recs
}

fn mutant_non_rouge() -> Vec<RecordTopo> {
    // X = 8.075, petite erreur positive, chaînes toutes mobiles.
    (1..=32)
        .flat_map(|ch| {
            (0..25).map(move |i| {
                let q = if i == 24 {
                    if ch <= 28 {
                        3.0
                    } else {
                        4.0
                    }
                } else if i % 2 == 0 {
                    4.0
                } else {
                    0.0
                };
                record(ch, i + 1, q)
            })
        })
        .collect()
}

fn gate<'a>(lignes: &'a [LigneGate], id: &str) -> &'a LigneGate {
    let trouvees: Vec<_> = lignes.iter().filter(|l| l.id == id).collect();
    assert_eq!(trouvees.len(), 1, "une seule ligne pour {id} : {lignes:?}");
    trouvees[0]
}

fn verdict(lignes: &[LigneGate], id: &str, attendu: Verdict) {
    let l = gate(lignes, id);
    assert_eq!(l.verdict, attendu, "{l}");
}

fn motif(lignes: &[LigneGate], texte: &str) {
    assert!(
        lignes[0].detail.contains(texte),
        "{} ne contient pas {texte}",
        lignes[0]
    );
}

#[test]
fn clopper_pearson_valeurs_reference_et_extremes() {
    let (bas, haut) = clopper_pearson(90, 100, 0.95);
    proche(bas, 0.8238, 5e-4);
    proche(haut, 0.9510, 5e-4);
    let (bas, haut) = clopper_pearson(0, 10, 0.95);
    assert_eq!(bas, 0.0);
    proche(haut, 0.3085, 5e-4);
    let (bas, haut) = clopper_pearson(10, 10, 0.95);
    proche(bas, 0.6915, 5e-4);
    assert_eq!(haut, 1.0);
    let (bas, haut) = clopper_pearson(1, 1, 0.95);
    proche(bas, 0.025, 1e-14);
    assert_eq!(haut, 1.0);
}

#[test]
fn clopper_pearson_symetrie_et_entrees_invalides() {
    for k in 0..=20 {
        let (a, b) = clopper_pearson(k, 20, 0.95);
        let (c, d) = clopper_pearson(20 - k, 20, 0.95);
        proche(a, 1.0 - d, 1e-13);
        proche(b, 1.0 - c, 1e-13);
        assert!(a <= k as f64 / 20.0 && b >= k as f64 / 20.0);
    }
    for (k, n, niveau) in [
        (0, 0, 0.95),
        (11, 10, 0.95),
        (1, 10, 0.0),
        (1, 10, 1.0),
        (1, 10, f64::NAN),
    ] {
        let (a, b) = clopper_pearson(k, n, niveau);
        assert!(a.is_nan() && b.is_nan());
    }
}

#[test]
fn tau_sokal_bruit_blanc_et_ar1() {
    let mut rng = SplitMix64(20261026);
    let blanc: Vec<_> = (0..20_000).map(|_| rng.normal()).collect();
    proche(tau_int_sokal(&blanc, 6.0).unwrap(), 0.5, 0.1);
    let mut dernier = 0.0;
    let ar: Vec<_> = (0..21_000)
        .map(|_| {
            dernier = 0.8 * dernier + 0.6 * rng.normal();
            dernier
        })
        .skip(1000)
        .collect();
    proche(tau_int_sokal(&ar, 6.0).unwrap(), 4.5, 4.5 * 0.15);
}

#[test]
fn tau_sokal_non_estimable_et_invariance_affine() {
    assert_eq!(tau_int_sokal(&[3.0; 100], 6.0), None);
    assert_eq!(tau_int_sokal(&[0.1; 100], 0.1), None);
    // La spécification ne borne pas tau par zéro : première fenêtre W=1,
    // rho(1)=-99/100 pour ce témoin fortement anticorrélé.
    let alternee: Vec<_> = (0..100)
        .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
        .collect();
    proche(tau_int_sokal(&alternee, 6.0).unwrap(), -0.49, 1e-14);
    assert_eq!(tau_int_sokal(&[1.0, 2.0, 3.0], 6.0), None);
    assert_eq!(tau_int_sokal(&[], 6.0), None);
    assert_eq!(tau_int_sokal(&[0.0, 1.0, f64::NAN, 2.0], 6.0), None);
    let mut rng = SplitMix64(10);
    let x: Vec<_> = (0..500).map(|_| rng.normal()).collect();
    let y: Vec<_> = x.iter().map(|v| 10.0 - 2.0 * v).collect();
    proche(
        tau_int_sokal(&x, 6.0).unwrap(),
        tau_int_sokal(&y, 6.0).unwrap(),
        1e-12,
    );
    assert_eq!(tau_int_sokal(&x, 0.0), None);
    // W=1 : tau=1 ; W=2 : tau=1. La fenêtre n'est pas atteinte
    // avant N/2=3 (une fenêtre à N/2 serait déjà hors du domaine).
    assert_eq!(tau_int_sokal(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 6.0), None);
}

#[test]
fn jackknife_petit_exemple_et_replicats_explicites() {
    let (m, e) = jackknife_chaines(&[1.0, 2.0, 3.0]);
    assert_eq!(m, 2.0);
    proche(e, (1.0f64 / 3.0).sqrt(), 1e-14);
    let valeurs = [1.0, 2.0, 7.0, 10.0, -3.0];
    let n = valeurs.len() as f64;
    let somme: f64 = valeurs.iter().sum();
    let replicas: Vec<_> = valeurs.iter().map(|v| (somme - v) / (n - 1.0)).collect();
    let mr = replicas.iter().sum::<f64>() / n;
    let ej = ((n - 1.0) / n * replicas.iter().map(|v| (v - mr).powi(2)).sum::<f64>()).sqrt();
    let (m, e) = jackknife_chaines(&valeurs);
    proche(m, somme / n, 1e-14);
    proche(e, ej, 1e-14);
    assert_eq!(jackknife_chaines(&[2.0; 32]), (2.0, 0.0));
    assert!(jackknife_chaines(&[]).0.is_nan());
    assert!(jackknife_chaines(&[2.0]).1.is_nan());
}

#[test]
fn extension_arrondit_a_la_dizaine() {
    assert_eq!(extension_configs(100, 0.7, 0.35), 630);
    assert_eq!(extension_configs(100, 0.8, 1.0), 100);
    assert_eq!(extension_configs(100, 1.6, 1.0), 400);
    assert_eq!(extension_configs(101, 0.8, 1.0), 110);
    assert_eq!(extension_configs(0, 0.7, 0.35), 0);
    assert_eq!(extension_configs(100, f64::NAN, 0.35), 0);
    assert_eq!(extension_configs(100, 0.7, 0.0), 0);
}

#[test]
fn topo_q_quantifie_mutant_et_cooling_passent() {
    let lignes = topo_q(&jeu_q());
    for id in ["TOPO-Q", "TOPO-Q-MUT", "TOPO-COOL"] {
        verdict(&lignes, id, Verdict::Pass);
    }
    assert!(gate(&lignes, "TOPO-Q-MUT").detail.contains("N′=0/100"));
    assert_eq!(lignes.iter().filter(|l| l.id == "TOPO-Q-T").count(), 8);
    assert!(lignes.iter().any(|l| l
        .detail
        .contains("couverture non garantie (chaînes corrélées)")));
}

#[test]
fn topo_q_85_sur_100_echoue() {
    let mut recs = jeu_q();
    for r in &mut recs[..15] {
        r.flot_a2.q_imp[IDX_T3] = r.flot_a2.q_imp[IDX_T3].round() + 0.3;
    }
    let lignes = topo_q(&recs);
    verdict(&lignes, "TOPO-Q", Verdict::Fail);
    assert!(gate(&lignes, "TOPO-Q").detail.contains("N=85/100"));
}

#[test]
fn topo_q_seuil_90_et_tolerance_stricte_sans_arrondi_affichage() {
    let mut recs: Vec<_> = (0..100)
        .map(|i| record(i / 25 + 1, i % 25 + 1, 0.0))
        .collect();
    for r in &mut recs[..10] {
        r.flot_a2.q_imp[IDX_T3] = 0.1;
    }
    verdict(&topo_q(&recs), "TOPO-Q", Verdict::Pass);
    recs[10].flot_a2.q_imp[IDX_T3] = 0.10000001;
    verdict(&topo_q(&recs), "TOPO-Q", Verdict::Fail);
    recs[10].flot_a2.q_imp[IDX_T3] = 0.09999999;
    verdict(&topo_q(&recs), "TOPO-Q", Verdict::Pass);
}

#[test]
fn topo_q_effectif_incorrect_et_donnees_absentes() {
    let mut recs = jeu_q();
    let lignes = topo_q(&recs[..99]);
    verdict(&lignes, "TOPO-Q", Verdict::Skip);
    verdict(&lignes, "TOPO-Q-MUT", Verdict::Skip);
    recs[0].flot_a2.q_imp.clear();
    verdict(&topo_q(&recs), "TOPO-Q", Verdict::Skip);
    recs[0].flot_a2.q_imp = vec![f64::NAN; 4];
    verdict(&topo_q(&recs), "TOPO-Q", Verdict::Skip);
    verdict(&topo_q(&[]), "TOPO-COOL", Verdict::Skip);
}

#[test]
fn topo_q_mutant_analyse_decalage_demi_entier() {
    let mut recs = jeu_q();
    for r in &mut recs {
        r.flot_a2.q_imp[IDX_T3] += 0.5;
    }
    let lignes = topo_q(&recs);
    verdict(&lignes, "TOPO-Q", Verdict::Fail);
    verdict(&lignes, "TOPO-Q-MUT", Verdict::Fail);
    assert!(gate(&lignes, "TOPO-Q-MUT").detail.contains("ECHEC_OUTIL"));
}

#[test]
fn topo_cool_augmentation_tri_et_premieres_dix_seulement() {
    let mut recs = jeu_q();
    recs[0].cooling.action[10] = recs[0].cooling.action[9] * (1.0 + 1e-6);
    recs.reverse();
    verdict(&topo_q(&recs), "TOPO-COOL", Verdict::Fail);
    recs.reverse();
    recs[0] = record(1, 1, 0.0);
    recs[10].cooling.action[10] += 100.0;
    recs.reverse();
    verdict(&topo_q(&recs), "TOPO-COOL", Verdict::Pass);
    verdict(&topo_q(&recs[..9]), "TOPO-COOL", Verdict::Skip);
}

#[test]
fn topo_cool_tolerance_inclusive_et_serie_incomplete() {
    let mut recs = jeu_q();
    recs[0].cooling.action.fill(1e12);
    recs[0].cooling.action[1] += 1.0;
    verdict(&topo_q(&recs), "TOPO-COOL", Verdict::Pass);
    recs[0].cooling.action[1] += 0.001;
    verdict(&topo_q(&recs), "TOPO-COOL", Verdict::Fail);
    recs[0].cooling.action.truncate(20);
    verdict(&topo_q(&recs), "TOPO-COOL", Verdict::Skip);
}

#[test]
fn topo_q_histogrammes_bornes_et_t0_absent() {
    let mut recs = jeu_q();
    for (r, q) in recs.iter_mut().zip([-9.0, -8.0, 8.0, 9.0, -0.5, 0.5, -0.4]) {
        r.flot_a2.q_imp[IDX_T3] = q;
    }
    for r in &mut recs {
        r.flot_t0 = None;
        r.t0 = None;
    }
    let lignes = topo_q(&recs);
    let hist = lignes
        .iter()
        .find(|l| l.detail.starts_with("histogramme Q ;"))
        .unwrap();
    assert!(hist
        .detail
        .contains("sous_borne=1 hors_borne_haute=1 n=100"));
    assert!(hist.detail.contains("[7.750000,8.000000]:1"));
    let residus = lignes
        .iter()
        .find(|l| l.detail.starts_with("histogramme Q-round(Q)"))
        .unwrap();
    assert!(residus.detail.contains("[-0.400000,-0.350000):1"));
    assert!(lignes
        .iter()
        .filter(|l| l.id == "TOPO-Q-T" && l.detail.contains('₀'))
        .all(|l| l.detail.contains("amélioré=0/0 trèfle=0/0")));
}

#[test]
fn obs4_rejeu_reste_info_meme_si_historique_rouge() {
    let mut recs = jeu_q();
    let lignes = obs4_rejeu(&recs[..20]);
    verdict(&lignes, "OBS-4-L16-REJEU", Verdict::Info);
    motif(&lignes, "PASS historique à titre indicatif");
    for r in &mut recs[..3] {
        r.flot_a2.q_imp[IDX_T3] = 0.3;
    }
    let lignes = obs4_rejeu(&recs[..20]);
    verdict(&lignes, "OBS-4-L16-REJEU", Verdict::Info);
    motif(&lignes, "FAIL historique à titre indicatif");
    motif(&obs4_rejeu(&recs[..19]), "non applicable");
}

#[test]
fn topo_lit_gaussien_reference_passe_et_six_echoue() {
    let recs = jeu_lit(8.066, 100);
    let x = recs
        .iter()
        .map(|r| r.cooling.q_imp[20].round().powi(2))
        .sum::<f64>()
        / recs.len() as f64;
    proche(x, 8.066, 0.6);
    verdict(&topo_lit(&recs, false), "TOPO-LIT", Verdict::Pass);
    verdict(
        &topo_lit(&jeu_lit(6.0, 100), false),
        "TOPO-LIT",
        Verdict::Fail,
    );
    assert_eq!(QUANTILE_T31, 3.26094191889439);
}

#[test]
fn topo_lit_attribution_ambigue_en_premier() {
    let mut recs = jeu_lit(8.066, 100);
    for r in &mut recs[..1600] {
        r.cooling.q_imp[20] = r.cooling.q_imp[20].round() + 0.4;
        r.cooling.q_tre[20] = 100.0;
    }
    let lignes = topo_lit(&recs, false);
    verdict(&lignes, "TOPO-LIT", Verdict::Skip);
    motif(&lignes, "(a) attribution ambiguë");
}

#[test]
fn topo_lit_seuil_attribution_98_pourcent_inclusif() {
    let mut recs = jeu_lit(8.066, 100);
    for r in &mut recs[..64] {
        r.cooling.q_imp[20] = r.cooling.q_imp[20].round() + 0.25;
    }
    verdict(&topo_lit(&recs, false), "TOPO-LIT", Verdict::Pass);
    recs[64].cooling.q_imp[20] = recs[64].cooling.q_imp[20].round() + 0.25;
    motif(&topo_lit(&recs, false), "(a) attribution ambiguë");
}

#[test]
fn topo_lit_chaines_figees_et_erreur_nulle() {
    let mut recs = jeu_lit(8.066, 100);
    for r in &mut recs {
        r.cooling.q_imp[20] = if r.chaine % 2 == 0 { 2.0 } else { 3.0 };
    }
    let lignes = topo_lit(&recs, false);
    verdict(&lignes, "TOPO-LIT", Verdict::Skip);
    motif(&lignes, "(b)");
    for r in &mut recs {
        r.cooling.q_imp[20] = if r.cfg % 2 == 0 { 2.0 } else { 3.0 };
    }
    motif(&topo_lit(&recs, false), "(b)");
}

#[test]
fn topo_lit_moins_de_huit_chaines_mobiles() {
    let mut recs = jeu_lit(8.066, 100);
    for r in &mut recs {
        if r.chaine <= 25 {
            r.cooling.q_imp[20] = 2.0;
        }
    }
    motif(&topo_lit(&recs, false), "non constantes=7");
}

#[test]
fn topo_lit_trefle_incompatible() {
    let mut recs = jeu_lit(8.066, 100);
    let mut rng = SplitMix64(42);
    for r in &mut recs {
        r.cooling.q_tre[20] = 12.0 * (rng.uniforme() - 0.5);
    }
    let lignes = topo_lit(&recs, false);
    verdict(&lignes, "TOPO-LIT", Verdict::Skip);
    motif(&lignes, "(c) attribution dépendante de l'opérateur");
}

#[test]
fn topo_lit_seuil_desaccord_2_pourcent_inclusif() {
    let mut recs = jeu_lit(8.066, 100);
    // Changer le signe conserve X′ : seul le compte de désaccords intervient.
    let indices: Vec<_> = recs
        .iter()
        .enumerate()
        .filter(|(_, r)| r.cooling.q_imp[20].round() != 0.0)
        .take(65)
        .map(|(i, _)| i)
        .collect();
    for &i in &indices[..64] {
        recs[i].cooling.q_tre[20] *= -1.0;
    }
    verdict(&topo_lit(&recs, false), "TOPO-LIT", Verdict::Pass);
    recs[indices[64]].cooling.q_tre[20] *= -1.0;
    motif(&topo_lit(&recs, false), "(c)");
}

#[test]
fn topo_lit_raccordement_ecart_de_moyenne_compte_aussi() {
    let mut recs = jeu_lit(8.066, 100);
    // Un seul désaccord (< 2 %) mais un effet sur X′ largement supérieur à sigma.
    recs[0].cooling.q_tre[20] = 200.0;
    let lignes = topo_lit(&recs, false);
    verdict(&lignes, "TOPO-LIT", Verdict::Skip);
    motif(&lignes, "(c)");
}

#[test]
fn topo_lit_puissance_insuffisante() {
    let recs: Vec<_> = (1..=32)
        .flat_map(|ch| {
            (1..=100).map(move |cfg| {
                let q = if ch % 2 == 0 { 5.0 } else { 0.0 } + (cfg % 2) as f64;
                record(ch, cfg, q)
            })
        })
        .collect();
    let lignes = topo_lit(&recs, false);
    verdict(&lignes, "TOPO-LIT", Verdict::Skip);
    motif(&lignes, "puissance insuffisante");
    motif(&lignes, "extension proposée=");
}

#[test]
fn topo_lit_structure_et_donnees_invalides() {
    let mut recs = jeu_lit(8.066, 100);
    verdict(&topo_lit(&recs[..3100], false), "TOPO-LIT", Verdict::Skip);
    verdict(&topo_lit(&recs[..3199], false), "TOPO-LIT", Verdict::Skip);
    // Longueur constante mais différente de l'ensemble annoncé.
    let courts: Vec<_> = recs.iter().filter(|r| r.cfg <= 99).cloned().collect();
    verdict(&topo_lit(&courts, false), "TOPO-LIT", Verdict::Skip);
    recs[0].cooling.q_imp.clear();
    verdict(&topo_lit(&recs, false), "TOPO-LIT", Verdict::Skip);
    recs[0].cooling.q_imp = vec![f64::INFINITY; 21];
    verdict(&topo_lit(&recs, false), "TOPO-LIT", Verdict::Skip);
    verdict(&topo_lit(&[], false), "TOPO-LIT", Verdict::Skip);
}

#[test]
fn topo_lit_mutant_rouge_et_non_rouge() {
    verdict(
        &topo_lit(&jeu_lit(3.0, 25), true),
        "TOPO-LIT-MUT",
        Verdict::Pass,
    );
    let lignes = topo_lit(&mutant_non_rouge(), true);
    verdict(&lignes, "TOPO-LIT-MUT", Verdict::Fail);
    motif(&lignes, "ECHEC_OUTIL mutant non rouge");
}

#[test]
fn topo_lit_mutant_conditions_invalides_sont_echec_outil() {
    let mut recs = mutant_non_rouge();
    for r in &mut recs[..400] {
        r.cooling.q_imp[20] += 0.4;
    }
    let lignes = topo_lit(&recs, true);
    verdict(&lignes, "TOPO-LIT-MUT", Verdict::Fail);
    motif(&lignes, "ECHEC_OUTIL non concluant");
    let lignes = topo_lit(&recs[..775], true);
    verdict(&lignes, "TOPO-LIT-MUT", Verdict::Fail);
    motif(&lignes, "ECHEC_OUTIL non concluant");
    let recs: Vec<_> = (1..=32)
        .flat_map(|ch| (1..=25).map(move |cfg| record(ch, cfg, (ch % 2 * 5 + cfg % 2) as f64)))
        .collect();
    let lignes = topo_lit(&recs, true);
    verdict(&lignes, "TOPO-LIT-MUT", Verdict::Fail);
    motif(&lignes, "puissance insuffisante");
    motif(&lignes, "ECHEC_OUTIL non concluant");
}

#[test]
fn selection_s_exacte_092_et_departage_vers_un() {
    let mut recs = jeu_lit(8.066, 100);
    for r in &mut recs {
        r.cooling.q_tre[20] = r.cooling.q_imp[20].round() / 0.92;
    }
    let lignes = topo_lit(&recs, false);
    verdict(&lignes, "TOPO-LIT", Verdict::Pass);
    assert!(lignes.iter().any(|l| l.detail.contains("s=0.920000")));
    for r in &mut recs {
        r.cooling.q_tre[20] = 0.0;
    }
    let lignes = topo_lit(&recs, false);
    assert!(lignes.iter().any(|l| l.detail.contains("s=1.000000")));
}

#[test]
fn selection_s_egalite_symetrique_choisit_plus_petit() {
    let mut recs = jeu_lit(8.066, 100);
    // 2.5*s est entier pour s=.8 et s=1.2 : coûts exactement nuls,
    // même distance entière à 1 sur la grille ; retenir .8.
    for r in &mut recs {
        r.cooling.q_tre[20] = 2.5;
    }
    assert!(topo_lit(&recs, false)
        .iter()
        .any(|l| l.detail.contains("s=0.800000")));
}

#[test]
fn topo_lit_info_backend_tau_plaq_et_ordre_des_records() {
    let mut recs = jeu_lit(8.066, 100);
    let lignes = topo_lit(&recs, false);
    recs.reverse();
    assert_eq!(lignes, topo_lit(&recs, false));
    for texte in [
        "χ_lat=",
        "⟨Q_I⟩=",
        "⟨Q_L²⟩=",
        "procédure de lissage différente",
        "chaînes figées=",
        "τ_int(Q_I²)",
        "backend_gen=hip",
        "backend_gen=cuda",
        "s=",
    ] {
        assert!(
            lignes
                .iter()
                .any(|l| l.verdict == Verdict::Info && l.detail.contains(texte)),
            "{texte}"
        );
    }
    let plaq = gate(&lignes, "TOPO-LIT-PLAQ");
    assert_eq!(plaq.verdict, Verdict::Info);
    assert!(plaq.detail.contains("sans verdict, réseau différent"));
    assert!(plaq.detail.contains("écart en σ combinés="));
    assert_eq!(
        lignes.iter().filter(|l| l.verdict != Verdict::Info).count(),
        1
    );
}

#[test]
fn topo_lit_tau_de_q_carre_fige_et_diagnostics_manquants() {
    let mut recs = jeu_lit(8.066, 100);
    for r in &mut recs {
        let amplitude = if r.chaine % 2 == 0 { 2.0 } else { 3.0 };
        r.cooling.q_imp[20] = if r.cfg % 2 == 0 {
            amplitude
        } else {
            -amplitude
        };
        r.cooling.q_tre[20] = r.cooling.q_imp[20] / 0.95;
        r.flot_a2.q_imp.clear();
    }
    let lignes = topo_lit(&recs, false);
    assert!(lignes.iter().any(|l| l
        .detail
        .contains("médiane=non estimable, max=non estimable")));
    assert!(lignes
        .iter()
        .any(|l| l.detail.contains("chaînes figées=[]")));
    assert!(lignes
        .iter()
        .any(|l| l.detail.contains("X̂ flot=non estimable")));
    recs[0].plaq0 = f64::NAN;
    assert!(gate(&topo_lit(&recs, false), "TOPO-LIT-PLAQ")
        .detail
        .contains("non estimable"));
}

#[test]
fn format_serde_dispatch_et_verdicts_uniques() {
    let q = jeu_q();
    for (ens, recs) in [
        ("q", &q[..]),
        ("obs4", &q[..20]),
        ("lit", &[][..]),
        ("lit_mut", &[][..]),
        ("inconnu", &[][..]),
    ] {
        for l in analyser(ens, recs) {
            let texte = l.to_string();
            assert!(["PASS ", "FAIL ", "SKIP ", "INFO "]
                .iter()
                .any(|p| texte.starts_with(p)));
            let json = serde_json::to_string(&l).unwrap();
            assert_eq!(serde_json::from_str::<LigneGate>(&json).unwrap(), l);
        }
    }
    for (v, texte) in [
        (Verdict::Pass, "PASS"),
        (Verdict::Fail, "FAIL"),
        (Verdict::Skip, "SKIP"),
        (Verdict::Info, "INFO"),
    ] {
        assert_eq!(serde_json::to_value(v).unwrap(), texte);
    }
    let lignes = analyser("inconnu", &[]);
    assert_eq!(lignes.len(), 1);
    assert_eq!(lignes[0].verdict, Verdict::Skip);
    motif(&lignes, "ensemble inconnu");
    assert_eq!(analyser("q", &q), topo_q(&q));
    assert_eq!(analyser("obs4", &q[..20]), obs4_rejeu(&q[..20]));
    assert_eq!(analyser("lit", &[]), topo_lit(&[], false));
    assert_eq!(analyser("lit_mut", &[]), topo_lit(&[], true));
}

#[test]
fn jsonl_trois_records_lignes_vides_et_erreurs_contextuelles() {
    static COMPTEUR: AtomicU64 = AtomicU64::new(0);
    let chemin = std::env::temp_dir().join(format!(
        "kvu_topo_s3_{}_{}_{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        COMPTEUR.fetch_add(1, Ordering::Relaxed)
    ));
    // Valeurs binaires exactes : serde_json est compilé sans float_roundtrip dans
    // le dépôt ; la lecture JSONL ne promet pas de préserver les bits arbitraires.
    let mut recs: Vec<_> = (1..=3)
        .map(|cfg| record(1, cfg, cfg as f64 - 2.0))
        .collect();
    for r in &mut recs {
        r.plaq0 = 0.625;
        r.t0 = Some(0.25);
        r.cooling.q_tre.fill(r.cooling.q_imp[20]);
        r.flot_t0.as_mut().unwrap().t_eff.fill(0.5);
    }
    let texte = format!(
        "{}\n\n  \t\n{}\n{}\n",
        serde_json::to_string(&recs[0]).unwrap(),
        serde_json::to_string(&recs[1]).unwrap(),
        serde_json::to_string(&recs[2]).unwrap()
    );
    fs::write(&chemin, texte).unwrap();
    assert_eq!(lire_jsonl(&chemin).unwrap(), recs);
    fs::write(&chemin, "\n{invalide}\n").unwrap();
    let erreur = lire_jsonl(&chemin).unwrap_err().to_string();
    assert!(erreur.contains("ligne 2") && erreur.contains(chemin.to_str().unwrap()));
    fs::remove_file(&chemin).unwrap();
    assert!(lire_jsonl(&chemin).is_err());
}
