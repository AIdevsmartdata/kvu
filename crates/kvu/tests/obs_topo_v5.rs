//! Contrat CPU de l'avenant v5.WP19, §v5.2–v5.3 (28/10/2026).
//! Les références numériques et les oracles viennent du texte, pas des constantes
//! ni des fonctions statistiques de production. Aucune donnée ni ressource GPU.

use kvu::heritage::su2_hb::CpuSu2;
use kvu::obs::ref_cpu::en_champ;
use kvu::obs::topo::{mesurer_cpu, FlotA2, FlotT0, ParamsTopo, RecordTopo, SerieCooling};
use kvu::obs::topo_gates::{
    analyser_v5, integrite_v5, topo_lit_v5, topo_q_v5, LigneGate, RoleLitV5, Verdict,
};

/// SplitMix64 local, indépendant des graines et du Philox de production.
struct Alea(u64);

impl Alea {
    fn uniforme(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 12) as f64 + 0.5) / 4_503_599_627_370_496.0
    }
}

fn proche(obtenu: f64, attendu: f64, tolerance: f64) {
    assert!(
        (obtenu - attendu).abs() <= tolerance,
        "obtenu={obtenu:.17}, attendu={attendu:.17}, tolérance={tolerance}"
    );
}

fn record(ens: &str, chaine: usize, cfg: usize) -> RecordTopo {
    let backend = if chaine <= if ens == "q5" { 2 } else { 16 } {
        "hip"
    } else {
        "cuda"
    };
    RecordTopo {
        ens: ens.into(),
        chaine,
        cfg,
        backend_gen: backend.into(),
        backend_mesure: backend.into(),
        l: 16,
        plaq0: 0.636_429_3,
        t0: Some(0.2),
        flot_a2: FlotA2 {
            e_plaq: vec![1.0; 4],
            q_tre: vec![0.0; 4],
            q_imp: vec![0.0; 4],
        },
        flot_t0: Some(FlotT0 {
            t_eff: vec![0.2, 0.4, 0.6, 1.0],
            q_tre: vec![0.5; 4],
            q_imp: vec![0.5; 4],
        }),
        cooling: SerieCooling {
            q_imp: vec![0.0; 21],
            q_tre: vec![0.0; 21],
            action: (0..=20).map(|k| 100.0 - k as f64).collect(),
        },
        ..RecordTopo::default()
    }
}

fn jeu_q(quantifiees: usize) -> Vec<RecordTopo> {
    let mut alea = Alea(26_102_819_101);
    (0..100)
        .map(|i| {
            let mut r = record("q5", i / 25 + 1, i % 25 + 1);
            r.backend_gen = if r.chaine <= 2 { "hip" } else { "cuda" }.into();
            let entier = (i % 9) as f64 - 4.0;
            // Les autres temps/opérateurs donnent volontairement d'autres comptes.
            r.flot_t0.as_mut().unwrap().q_imp[3] = entier
                + if i < quantifiees {
                    0.04 * (alea.uniforme() - 0.5)
                } else {
                    0.5
                };
            r
        })
        .collect()
}

/// m_k = cible + pas*(k-16,5). Des paires d'écarts opposés de Q_L²
/// assurent des chaînes mobiles sans changer leur moyenne (aussi pour N=25).
fn jeu_lit(cible: f64, pas: f64, n: usize, mobiles: usize) -> Vec<RecordTopo> {
    let mut alea = Alea(26_102_819_201);
    let mut records = Vec::with_capacity(32 * n);
    for chaine in 1..=32 {
        let mk = cible + pas * (chaine as f64 - 16.5);
        let mut ecarts = Vec::with_capacity(n);
        for _ in 0..n / 2 {
            let e = if chaine <= mobiles {
                0.1 + 0.05 * alea.uniforme()
            } else {
                0.0
            };
            ecarts.extend([e, -e]);
        }
        if n % 2 == 1 {
            ecarts.push(0.0);
        }
        for (i, ecart) in ecarts.into_iter().enumerate() {
            let mut r = record(if n == 25 { "lit5_mut" } else { "lit5" }, chaine, i + 1);
            r.t0 = None;
            r.flot_t0 = None;
            r.flot_a2 = FlotA2::default();
            assert!(mk + ecart > 0.0);
            let q = (mk + ecart).sqrt() * if i % 2 == 0 { 1.0 } else { -1.0 };
            r.cooling.q_tre.fill(q);
            r.cooling.q_imp.fill(q);
            records.push(r);
        }
    }
    records
}

fn gate<'a>(lignes: &'a [LigneGate], id: &str, attendu: Verdict) -> &'a LigneGate {
    let mut trouvees = lignes.iter().filter(|l| l.id == id);
    let l = trouvees.next().unwrap_or_else(|| panic!("{id} absent"));
    assert!(trouvees.next().is_none(), "{id} dupliqué");
    assert_eq!(l.verdict, attendu, "{l}");
    l
}

fn contient(l: &LigneGate, motif: &str) {
    assert!(l.detail.contains(motif), "{l} : motif absent « {motif} »");
}

fn q_invalide(records: &[RecordTopo]) {
    let lignes = topo_q_v5(records);
    contient(gate(&lignes, "TOPO-Q-V5", Verdict::Skip), "donnée invalide");
    let l = gate(&lignes, "TOPO-Q-MUT-V5", Verdict::Skip);
    contient(l, "donnée invalide");
    assert!(
        !lignes.iter().any(|l| l.detail.contains("N=")),
        "aucun compte physique sur un ensemble invalide"
    );
}

/// Recalcul direct depuis les charges, puis réplicats explicites à 31 chaînes.
/// Retourne (X̂, σ, z), sans appeler jackknife_chaines de production.
fn statistique(records: &[RecordTopo], ameliore: bool) -> (f64, f64, f64) {
    let mut moyennes = [0.0; 32];
    let mut nombres = [0; 32];
    for r in records {
        let q = if ameliore {
            r.cooling.q_imp[20]
        } else {
            r.cooling.q_tre[20]
        };
        moyennes[r.chaine - 1] += q * q;
        nombres[r.chaine - 1] += 1;
    }
    for (m, n) in moyennes.iter_mut().zip(nombres) {
        assert!(n > 0);
        *m /= n as f64;
    }
    let x = moyennes.iter().sum::<f64>() / 32.0;
    let sigma = (moyennes.iter().map(|m| (m - x).powi(2)).sum::<f64>() / (32.0 * 31.0)).sqrt();
    let replicas: Vec<f64> = (0..32)
        .map(|omis| {
            moyennes
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != omis)
                .map(|(_, m)| m)
                .sum::<f64>()
                / 31.0
        })
        .collect();
    let centre = replicas.iter().sum::<f64>() / 32.0;
    let sigma_jk =
        (31.0 / 32.0 * replicas.iter().map(|r| (r - centre).powi(2)).sum::<f64>()).sqrt();
    proche(sigma, sigma_jk, 2e-14);
    let z = (x - 6.635) / (sigma * sigma + 0.056_f64.powi(2)).sqrt();
    (x, sigma, z)
}

/// Le rapport arrondit à six décimales ; le verdict reste testé sur l'oracle brut.
fn nombre(l: &LigneGate, marqueur: &str) -> f64 {
    l.detail
        .split_once(marqueur)
        .unwrap_or_else(|| panic!("{marqueur} absent de {l}"))
        .1
        .split([',', ';', ' '])
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

fn controle_statistique(records: &[RecordTopo], role: RoleLitV5) -> Vec<LigneGate> {
    let (x, sigma, z) = statistique(records, role == RoleLitV5::MutantOp);
    let mutant = role != RoleLitV5::Nominal;
    let id = match role {
        RoleLitV5::Nominal => "TOPO-LIT-V5",
        RoleLitV5::MutantBeta => "TOPO-LIT-MUT-V5",
        RoleLitV5::MutantOp => "TOPO-LIT-MUT-OP",
    };
    assert!(sigma > 0.0 && sigma <= 0.28);
    let accepte = if mutant {
        z.abs() > 3.260_941_918_894_39
    } else {
        z.abs() < 3.260_941_918_894_39
    };
    let lignes = topo_lit_v5(records, role);
    let l = gate(
        &lignes,
        id,
        if accepte {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
    );
    for (marqueur, attendu) in [("X̂=", x), ("σ=", sigma), ("z=", z)] {
        proche(nombre(l, marqueur), attendu, 5.1e-7);
    }
    lignes
}

#[test]
fn q_90_sur_100_passe_et_mutant_rouge() {
    let lignes = topo_q_v5(&jeu_q(90));
    contient(gate(&lignes, "TOPO-Q-V5", Verdict::Pass), "N=90/100");
    contient(gate(&lignes, "TOPO-Q-MUT-V5", Verdict::Pass), "N′=10/100");
}

#[test]
fn q_89_sur_100_echoue() {
    let lignes = topo_q_v5(&jeu_q(89));
    contient(gate(&lignes, "TOPO-Q-V5", Verdict::Fail), "N=89/100");
}

#[test]
fn q_sans_t0_non_quantifie_meme_si_flot_a2_entier() {
    let mut records = jeu_q(90);
    records[0].t0 = None;
    records[0].flot_t0 = None;
    assert!(records[0].flot_a2.q_imp.iter().all(|q| q.fract() == 0.0));
    let lignes = topo_q_v5(&records);
    contient(gate(&lignes, "TOPO-Q-V5", Verdict::Fail), "N=89/100");
}

#[test]
fn q_frontiere_01_stricte() {
    for (q, attendu, compte) in [(0.0999, Verdict::Pass, 90), (0.1, Verdict::Fail, 89)] {
        let mut records = jeu_q(89);
        // Autour de zéro : pas de soustraction 1,1 - 1 qui brouillerait la frontière.
        records[89].flot_t0.as_mut().unwrap().q_imp[3] = q;
        let lignes = topo_q_v5(&records);
        contient(
            gate(&lignes, "TOPO-Q-V5", attendu),
            &format!("N={compte}/100"),
        );
    }
}

#[test]
fn q_99_records_invalides() {
    q_invalide(&jeu_q(90)[..99]);
}

#[test]
fn q_doublon_et_cfg_hors_intervalle_invalides() {
    for cfg in [1, 0, 26] {
        let mut records = jeu_q(90);
        records[1].cfg = cfg;
        q_invalide(&records);
    }
}

#[test]
fn q_t0_et_flot_t0_dissocies_invalides() {
    let mut records = jeu_q(90);
    records[0].flot_t0 = None;
    q_invalide(&records);
    records = jeu_q(90);
    records[0].t0 = None;
    q_invalide(&records);
}

#[test]
fn q_temps_effectif_inferieur_a_5t0_invalide() {
    let mut records = jeu_q(90);
    records[0].flot_t0.as_mut().unwrap().t_eff[3] = 0.99;
    q_invalide(&records);
}

#[test]
fn q_temps_effectif_juste_sous_5t0_dans_tolerance() {
    let mut records = jeu_q(90);
    records[0].flot_t0.as_mut().unwrap().t_eff[3] = 1.0 - 5e-10;
    gate(&topo_q_v5(&records), "TOPO-Q-V5", Verdict::Pass);
}

#[test]
fn q_series_requises_finies_et_completes() {
    // Chaque membre des séries doit être contrôlé, pas seulement q_imp[3].
    for champ in 0..9 {
        for tronquer in [false, true] {
            // q_imp à 5t0 tronqué est testé séparément pour isoler une panique.
            if champ == 6 && tronquer {
                continue;
            }
            let mut records = jeu_q(90);
            let r = &mut records[99];
            let serie = match champ {
                0 => &mut r.flot_a2.e_plaq,
                1 => &mut r.flot_a2.q_imp,
                2 => &mut r.flot_a2.q_tre,
                3 => &mut r.cooling.action,
                4 => &mut r.cooling.q_imp,
                5 => &mut r.cooling.q_tre,
                6 => &mut r.flot_t0.as_mut().unwrap().q_imp,
                7 => &mut r.flot_t0.as_mut().unwrap().q_tre,
                _ => &mut r.flot_t0.as_mut().unwrap().t_eff,
            };
            if tronquer {
                serie.pop();
            } else {
                serie[0] = f64::NAN;
            }
            q_invalide(&records);
        }
    }
}

#[test]
fn q_flot_t0_q_imp_tronque_donnee_invalide() {
    let mut records = jeu_q(90);
    records[99].flot_t0.as_mut().unwrap().q_imp.pop();
    q_invalide(&records);
}

#[test]
fn cool_decroissant_passe_et_croissant_echoue() {
    let mut records = jeu_q(90);
    gate(&topo_q_v5(&records), "TOPO-COOL-V5", Verdict::Pass);
    records[4].cooling.action[12] = 100.0;
    // Vérifie que les dix premières sont choisies après tri, et non par position.
    records.reverse();
    gate(&topo_q_v5(&records), "TOPO-COOL-V5", Verdict::Fail);
}

#[test]
fn cool_limite_aux_dix_premieres_et_tolerance_relative() {
    let mut records = jeu_q(90);
    records[10].cooling.action[12] = 100.0;
    records[0].cooling.action[1] = 100.0 + 5e-11;
    records.reverse();
    gate(&topo_q_v5(&records), "TOPO-COOL-V5", Verdict::Pass);
}

#[test]
fn lit_moyenne_jackknife_et_z_recalcules() {
    let mut records = jeu_lit(6.635, 0.025, 100, 32);
    records.reverse();
    let (x, sigma, _) = statistique(&records, false);
    proche(x, 6.635, 1e-13);
    // Σ(k-16,5)²/(32*31) = 2,75, déduit indépendamment de la fabrique.
    proche(sigma, 0.025 * 2.75_f64.sqrt(), 1e-14);
    let lignes = controle_statistique(&records, RoleLitV5::Nominal);
    gate(&lignes, "TOPO-LIT-V5", Verdict::Pass);
}

#[test]
fn lit_seuil_student_des_deux_cotes() {
    let sigma = 0.025 * 2.75_f64.sqrt();
    for signe in [-1.0, 1.0] {
        for (distance, attendu) in [(-1e-6, Verdict::Pass), (1e-6, Verdict::Fail)] {
            let z_cible = signe * (3.260_941_918_894_39 + distance);
            let cible = 6.635 + z_cible * (sigma * sigma + 0.056_f64.powi(2)).sqrt();
            let records = jeu_lit(cible, 0.025, 100, 32);
            proche(statistique(&records, false).2, z_cible, 1e-12);
            let lignes = controle_statistique(&records, RoleLitV5::Nominal);
            gate(&lignes, "TOPO-LIT-V5", attendu);
        }
    }
}

#[test]
fn lit_puissance_insuffisante_definitive_sans_extension() {
    let records = jeu_lit(6.635, 0.25, 100, 32);
    let (_, sigma, _) = statistique(&records, false);
    assert!(sigma > 0.28);
    let lignes = topo_lit_v5(&records, RoleLitV5::Nominal);
    let l = gate(&lignes, "TOPO-LIT-V5", Verdict::Skip);
    contient(l, "puissance insuffisante");
    contient(
        l,
        "puissance insuffisante (définitif, aucune extension en v5)",
    );
    assert!(lignes
        .iter()
        .all(|l| !l.detail.contains("extension proposée")));
}

#[test]
fn lit_plafond_sigma_028() {
    for (sigma, attendu) in [(0.279_999, Verdict::Pass), (0.280_001, Verdict::Skip)] {
        let records = jeu_lit(6.635, sigma / 2.75_f64.sqrt(), 100, 32);
        proche(statistique(&records, false).1, sigma, 1e-13);
        gate(
            &topo_lit_v5(&records, RoleLitV5::Nominal),
            "TOPO-LIT-V5",
            attendu,
        );
    }
}

#[test]
fn lit_sept_chaines_mobiles_skip_b_avant_puissance() {
    for pas in [0.025, 0.25] {
        let records = jeu_lit(6.635, pas, 100, 7);
        let lignes = topo_lit_v5(&records, RoleLitV5::Nominal);
        let l = gate(&lignes, "TOPO-LIT-V5", Verdict::Skip);
        contient(l, "(b)");
        contient(l, "non constantes=7");
    }
}

#[test]
fn lit_huit_chaines_mobiles_suffisent() {
    let records = jeu_lit(6.635, 0.025, 100, 8);
    gate(
        &controle_statistique(&records, RoleLitV5::Nominal),
        "TOPO-LIT-V5",
        Verdict::Pass,
    );
}

#[test]
fn lit_q_changeant_de_signe_mais_carre_constant_skip_b() {
    let records = jeu_lit(6.635, 0.025, 100, 0);
    assert_ne!(records[0].cooling.q_tre[20], records[1].cooling.q_tre[20]);
    let lignes = topo_lit_v5(&records, RoleLitV5::Nominal);
    contient(gate(&lignes, "TOPO-LIT-V5", Verdict::Skip), "(b)");
}

#[test]
fn lit_sigma_nul_malgre_32_chaines_mobiles_skip_b() {
    let mut records = jeu_lit(6.635, 0.025, 100, 32);
    for r in &mut records {
        r.cooling.q_tre[20] = if r.cfg % 2 == 0 { 1.0 } else { 3.0 };
    }
    let lignes = topo_lit_v5(&records, RoleLitV5::Nominal);
    let l = gate(&lignes, "TOPO-LIT-V5", Verdict::Skip);
    contient(l, "(b)");
    assert_eq!(nombre(l, "σ="), 0.0);
}

#[test]
fn lit_31_chaines_donnee_invalide() {
    let records = jeu_lit(6.635, 0.025, 100, 32);
    let lignes = topo_lit_v5(&records[..3100], RoleLitV5::Nominal);
    contient(
        gate(&lignes, "TOPO-LIT-V5", Verdict::Skip),
        "donnée invalide",
    );
}

#[test]
fn lit_doublon_ou_charge_non_finie_donnee_invalide() {
    for doublon in [true, false] {
        let mut records = jeu_lit(6.635, 0.025, 100, 32);
        if doublon {
            records[1].cfg = 1;
        } else {
            records[3199].cooling.q_tre[20] = f64::NAN;
        }
        let lignes = topo_lit_v5(&records, RoleLitV5::Nominal);
        contient(
            gate(&lignes, "TOPO-LIT-V5", Verdict::Skip),
            "donnée invalide",
        );
    }
}

#[test]
fn lit_mut_op_115_rouge_et_identite_non_rouge() {
    let mut records = jeu_lit(6.635, 0.025, 100, 32);
    let identite = controle_statistique(&records, RoleLitV5::MutantOp);
    contient(
        gate(&identite, "TOPO-LIT-MUT-OP", Verdict::Fail),
        "ECHEC_OUTIL mutant non rouge",
    );
    for r in &mut records {
        r.cooling.q_imp[20] = 1.15 * r.cooling.q_tre[20];
    }
    let lignes = controle_statistique(&records, RoleLitV5::MutantOp);
    gate(&lignes, "TOPO-LIT-MUT-OP", Verdict::Pass);
    // La modification de l'opérateur amélioré ne change pas le verdict nominal.
    gate(
        &controle_statistique(&records, RoleLitV5::Nominal),
        "TOPO-LIT-V5",
        Verdict::Pass,
    );
}

#[test]
fn lit_mut_beta_17_rouge() {
    let records = jeu_lit(1.7, 0.025, 25, 32);
    proche(statistique(&records, false).0, 1.7, 1e-14);
    gate(
        &controle_statistique(&records, RoleLitV5::MutantBeta),
        "TOPO-LIT-MUT-V5",
        Verdict::Pass,
    );
}

#[test]
fn lit_mut_beta_66_non_rouge() {
    let records = jeu_lit(6.6, 0.025, 25, 32);
    let lignes = controle_statistique(&records, RoleLitV5::MutantBeta);
    contient(
        gate(&lignes, "TOPO-LIT-MUT-V5", Verdict::Fail),
        "ECHEC_OUTIL mutant non rouge",
    );
}

#[test]
fn lit_mut_beta_puissance_insuffisante_non_concluant() {
    let records = jeu_lit(6.6, 0.25, 25, 32);
    assert!(statistique(&records, false).1 > 0.28);
    let lignes = topo_lit_v5(&records, RoleLitV5::MutantBeta);
    let l = gate(&lignes, "TOPO-LIT-MUT-V5", Verdict::Fail);
    contient(l, "ECHEC_OUTIL non concluant");
    contient(l, "puissance insuffisante");
}

#[test]
fn lit_mut_beta_100_cfg_non_concluant_donnee_invalide() {
    let mut records = jeu_lit(1.7, 0.025, 100, 32);
    for r in &mut records {
        r.ens = "lit5_mut".into();
    }
    let lignes = topo_lit_v5(&records, RoleLitV5::MutantBeta);
    let l = gate(&lignes, "TOPO-LIT-MUT-V5", Verdict::Fail);
    contient(l, "ECHEC_OUTIL non concluant");
    contient(l, "donnée invalide");
}

#[test]
fn integrite_jeux_sains_complet_et_sans_flot() {
    for (ens, records) in [("q5", jeu_q(90)), ("lit5", jeu_lit(6.635, 0.025, 100, 32))] {
        let l = integrite_v5(ens, &records);
        assert_eq!(l.verdict, Verdict::Pass, "{l}");
        assert_eq!(l.id, format!("TOPO-V5-INTEGRITE-{ens}"));
    }
}

#[test]
fn integrite_cooling_croissant_derniere_configuration() {
    let mut records = jeu_q(90);
    records[99].cooling.action[20] = 100.0;
    // TOPO-COOL reste vert : l'intégrité doit contrôler les 100 configurations.
    gate(&topo_q_v5(&records), "TOPO-COOL-V5", Verdict::Pass);
    let l = integrite_v5("q5", &records);
    assert_eq!(l.verdict, Verdict::Fail, "{l}");
    contient(&l, "(4,25)");
    contient(&l, "action de cooling croissante");
}

#[test]
fn integrite_nan_plaquette_flot_a2_et_cooling() {
    for champ in 0..7 {
        let mut records = jeu_q(90);
        let r = &mut records[99];
        let valeur = match champ {
            0 => &mut r.plaq0,
            1 => &mut r.flot_a2.e_plaq[0],
            2 => &mut r.flot_a2.q_imp[0],
            3 => &mut r.flot_a2.q_tre[0],
            4 => &mut r.cooling.action[20],
            5 => &mut r.cooling.q_imp[20],
            _ => &mut r.cooling.q_tre[20],
        };
        *valeur = f64::NAN;
        let l = integrite_v5("q5", &records);
        assert_eq!(l.verdict, Verdict::Fail, "champ {champ} : {l}");
    }
}

#[test]
fn integrite_nan_charge_amelioree_flot_t0() {
    let mut records = jeu_q(90);
    records[99].flot_t0.as_mut().unwrap().q_imp[3] = f64::NAN;
    let l = integrite_v5("q5", &records);
    assert_eq!(l.verdict, Verdict::Fail, "{l}");
}

#[test]
fn integrite_nan_charge_trefle_flot_t0() {
    let mut records = jeu_q(90);
    records[99].flot_t0.as_mut().unwrap().q_tre[0] = f64::NAN;
    let l = integrite_v5("q5", &records);
    assert_eq!(l.verdict, Verdict::Fail, "{l}");
}

#[test]
fn integrite_plaquette_frontiere_0005_et_mediane() {
    let seuil = 0.005_f64;
    let dessous = f64::from_bits(seuil.to_bits() - 1);
    for signe in [-1.0, 1.0] {
        for (ecart, attendu) in [
            (dessous, Verdict::Pass),
            (seuil, Verdict::Fail),
            (0.006, Verdict::Fail),
        ] {
            let mut records = jeu_q(90);
            // Médiane exactement zéro : frontière représentée sans soustraction.
            for r in &mut records {
                r.plaq0 = 0.0;
            }
            records[99].plaq0 = signe * ecart;
            let l = integrite_v5("q5", &records);
            assert_eq!(l.verdict, attendu, "écart={ecart:.17} : {l}");
        }
    }
    // Distingue médiane et moyenne : moyenne décalée de 0,49*0,0049.
    let mut records = jeu_q(90);
    for (i, r) in records.iter_mut().enumerate() {
        r.plaq0 = if i < 49 { 0.0049 } else { 0.0 };
    }
    records[99].plaq0 = -0.0049;
    assert_eq!(integrite_v5("q5", &records).verdict, Verdict::Pass);
}

#[test]
fn analyse_lit_integrite_en_premier_puis_recert_infos_mutant() {
    let records = jeu_lit(6.635, 0.025, 100, 32);
    let lignes = analyser_v5("lit5", &records);
    assert_eq!(lignes[0].id, "TOPO-V5-INTEGRITE-lit5");
    assert_eq!(lignes[0].verdict, Verdict::Pass);
    assert_eq!(lignes[1].id, "TOPO-LIT-V5");
    assert_eq!(lignes[1].verdict, Verdict::Pass);
    assert_eq!(lignes[2].id, "RECERT-GEN-lit5");
    let op = lignes
        .iter()
        .position(|l| l.id == "TOPO-LIT-MUT-OP")
        .unwrap();
    assert!(op > 3);
    assert!(lignes[3..op].iter().all(|l| l.verdict == Verdict::Info));
    assert_eq!(op, lignes.len() - 2);
    assert_eq!(lignes[op + 1].id, "TOPO-COOL-V5");
}

#[test]
fn analyse_ensemble_inconnu_skip() {
    let lignes = analyser_v5("x", &[]);
    assert_eq!(lignes.len(), 1);
    gate(&lignes, "TOPO-ANALYSE-V5", Verdict::Skip);
}

#[test]
fn sans_flot_cooling_et_plaquette_identiques_au_bit_cpu_l4() {
    // Même construction que obs_topo_cpu.rs : hot, graine fixe, 24 pas HB + OR.
    // Deux fils privés suffisent sur L=4 et évitent de saturer le pool global.
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| {
            let mut cpu = CpuSu2::new(4, 26_102_619_003).unwrap();
            cpu.init(true);
            for _ in 0..24 {
                cpu.step(2.5, 2);
            }
            let champ = en_champ(&cpu.u, 4);
            let params = ParamsTopo {
                pas_max: 500,
                ..ParamsTopo::default()
            };
            let complet = mesurer_cpu(&champ, &params).unwrap();
            let sans = mesurer_cpu(
                &champ,
                &ParamsTopo {
                    sans_flot: true,
                    ..params
                },
            )
            .unwrap();
            assert_eq!(complet.flot_a2.q_imp.len(), 4);
            assert!(sans.flot_a2.e_plaq.is_empty());
            assert!(sans.flot_a2.q_imp.is_empty());
            assert!(sans.flot_a2.q_tre.is_empty());
            assert!(sans.t0.is_none());
            assert!(sans.flot_t0.is_none());
            assert_eq!(sans.plaq0.to_bits(), complet.plaq0.to_bits());
            for (a, b) in [
                (&sans.cooling.q_imp, &complet.cooling.q_imp),
                (&sans.cooling.q_tre, &complet.cooling.q_tre),
                (&sans.cooling.action, &complet.cooling.action),
            ] {
                assert_eq!((a.len(), b.len()), (21, 21));
                for (i, (x, y)) in a.iter().zip(b).enumerate() {
                    assert_eq!(x.to_bits(), y.to_bits(), "cooling[{i}]");
                }
            }
        });
}

#[test]
fn puissance_028_sans_extension_pour_les_trois_roles() {
    for (role, n, id, verdict) in [
        (RoleLitV5::Nominal, 100, "TOPO-LIT-V5", Verdict::Skip),
        (RoleLitV5::MutantOp, 100, "TOPO-LIT-MUT-OP", Verdict::Fail),
        (RoleLitV5::MutantBeta, 25, "TOPO-LIT-MUT-V5", Verdict::Fail),
    ] {
        // 0,29 était admissible avec l'ancien plafond de 0,30.
        let records = jeu_lit(6.635, 0.29 / 2.75_f64.sqrt(), n, 32);
        let lignes = topo_lit_v5(&records, role);
        let l = gate(&lignes, id, verdict);
        contient(
            l,
            "puissance insuffisante (définitif, aucune extension en v5)",
        );
        if role != RoleLitV5::Nominal {
            contient(l, "ECHEC_OUTIL non concluant");
        }
        assert!(!lignes
            .iter()
            .any(|l| l.detail.contains("extension proposée")));
    }
}

#[test]
fn integrite_echec_bloque_toutes_les_gates_dependantes() {
    for (ens, mut records, gates) in [
        (
            "q5",
            jeu_q(90),
            vec![
                ("TOPO-Q-V5", Verdict::Skip),
                ("TOPO-Q-MUT-V5", Verdict::Skip),
            ],
        ),
        (
            "lit5",
            jeu_lit(6.635, 0.025, 100, 32),
            vec![
                ("TOPO-LIT-V5", Verdict::Skip),
                ("TOPO-LIT-MUT-OP", Verdict::Fail),
            ],
        ),
        (
            "lit5_mut",
            jeu_lit(1.7, 0.025, 25, 32),
            vec![("TOPO-LIT-MUT-V5", Verdict::Fail)],
        ),
    ] {
        let dernier = records.last_mut().unwrap();
        dernier.cooling.action[20] = 100.0;
        let lignes = analyser_v5(ens, &records);
        assert_eq!(lignes[0].id, format!("TOPO-V5-INTEGRITE-{ens}"));
        assert_eq!(lignes[0].verdict, Verdict::Fail);
        for (id, verdict) in gates {
            let l = gate(&lignes, id, verdict);
            assert_eq!(
                l.detail,
                if verdict == Verdict::Fail {
                    "ECHEC_OUTIL non concluant ; intégrité en échec"
                } else {
                    "intégrité en échec"
                }
            );
        }
        assert!(lignes
            .iter()
            .any(|l| l.verdict == Verdict::Info && l.detail.contains("X̂_L=")
                || l.detail.contains("Clopper")));
        gate(&lignes, "TOPO-COOL-V5", Verdict::Pass);
        if ens != "q5" {
            gate(&lignes, &format!("RECERT-GEN-{ens}"), Verdict::Skip);
        }
    }
}

#[test]
fn integrite_t0_fini_positif_et_tous_les_vecteurs_optionnels() {
    for ens in ["q5", "lit5", "lit5_mut"] {
        for t0 in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.2] {
            let mut records = vec![record(ens, 1, 1)];
            records[0].t0 = Some(t0);
            assert_eq!(
                integrite_v5(ens, &records).verdict,
                Verdict::Fail,
                "{ens} {t0}"
            );
        }
        for champ in 0..9 {
            let mut records = vec![record(ens, 1, 1)];
            let r = &mut records[0];
            let v = match champ {
                0 => &mut r.flot_t0.as_mut().unwrap().t_eff,
                1 => &mut r.flot_t0.as_mut().unwrap().q_imp,
                2 => &mut r.flot_t0.as_mut().unwrap().q_tre,
                3 => &mut r.flot_a2.e_plaq,
                4 => &mut r.flot_a2.q_imp,
                5 => &mut r.flot_a2.q_tre,
                6 => &mut r.cooling.action,
                7 => &mut r.cooling.q_imp,
                _ => &mut r.cooling.q_tre,
            };
            v.push(f64::INFINITY); // Contrôle aussi au-delà des longueurs utiles.
            assert_eq!(
                integrite_v5(ens, &records).verdict,
                Verdict::Fail,
                "{ens} champ={champ}"
            );
        }
    }
}

#[test]
fn cooling_au_moins_21_et_fini_sur_toute_sa_longueur() {
    for champ in 0..3 {
        for longueur in [0, 20, 21, 22] {
            let mut records = jeu_q(90);
            let c = &mut records[99].cooling;
            let v = match champ {
                0 => &mut c.action,
                1 => &mut c.q_imp,
                _ => &mut c.q_tre,
            };
            v.resize(longueur, 0.0);
            let attendu = if longueur >= 21 {
                Verdict::Pass
            } else {
                Verdict::Fail
            };
            assert_eq!(integrite_v5("q5", &records).verdict, attendu);
            if longueur >= 21 {
                gate(&topo_q_v5(&records), "TOPO-Q-V5", Verdict::Pass);
            } else {
                q_invalide(&records);
            }
        }
    }
}

fn corrompre_provenance(r: &mut RecordTopo, champ: usize) -> &'static str {
    match champ {
        0 => {
            r.ens = "autre".into();
            "ens="
        }
        1 => {
            r.l = 4;
            "L="
        }
        2 => {
            r.backend_gen = if r.backend_gen == "hip" {
                "cuda"
            } else {
                "hip"
            }
            .into();
            "backend_gen="
        }
        _ => {
            r.backend_mesure = "cpu".into();
            "backend_mesure="
        }
    }
}

#[test]
fn provenance_ensemble_taille_affectation_et_backend_mesure() {
    for champ in 0..4 {
        for index in [0, 49, 50, 99] {
            let mut records = jeu_q(90);
            let motif = corrompre_provenance(&mut records[index], champ);
            let lignes = topo_q_v5(&records);
            contient(gate(&lignes, "TOPO-Q-V5", Verdict::Skip), motif);
            let l = gate(&lignes, "TOPO-Q-MUT-V5", Verdict::Fail);
            contient(l, "ECHEC_OUTIL non concluant");
            contient(l, motif);
            assert!(!lignes.iter().any(|l| l.detail.contains("N=")));
        }
        for (role, n, id) in [
            (RoleLitV5::Nominal, 100, "TOPO-LIT-V5"),
            (RoleLitV5::MutantOp, 100, "TOPO-LIT-MUT-OP"),
            (RoleLitV5::MutantBeta, 25, "TOPO-LIT-MUT-V5"),
        ] {
            for index in [0, 16 * n - 1, 16 * n, 32 * n - 1] {
                let mut records = jeu_lit(6.635, 0.025, n, 32);
                let motif = corrompre_provenance(&mut records[index], champ);
                let lignes = topo_lit_v5(&records, role);
                let l = gate(
                    &lignes,
                    id,
                    if role == RoleLitV5::Nominal {
                        Verdict::Skip
                    } else {
                        Verdict::Fail
                    },
                );
                contient(l, motif);
                if role != RoleLitV5::Nominal {
                    contient(l, "ECHEC_OUTIL non concluant");
                }
                if role != RoleLitV5::MutantOp {
                    gate(
                        &lignes,
                        &format!("RECERT-GEN-{}", if n == 25 { "lit5_mut" } else { "lit5" }),
                        Verdict::Skip,
                    );
                }
            }
        }
    }
}

fn fixer_t0(r: &mut RecordTopo, t0: f64) {
    r.t0 = Some(t0);
    r.flot_t0.as_mut().unwrap().t_eff = [1.0, 2.0, 3.0, 5.0]
        .map(|k| 0.01 * (k * t0 / 0.01 - 1e-9).ceil())
        .to_vec();
}

#[test]
fn q_grille_v431_tous_les_temps_et_tolerance_1e9() {
    for i in 0..4 {
        for delta in [-1.1e-9, -0.5e-9, 0.0, 0.5e-9, 1.1e-9] {
            let mut records = jeu_q(90);
            fixer_t0(&mut records[0], 0.203);
            records[0].flot_t0.as_mut().unwrap().t_eff[i] += delta;
            if delta.abs() <= 1e-9 {
                gate(&topo_q_v5(&records), "TOPO-Q-V5", Verdict::Pass);
            } else {
                q_invalide(&records);
            }
        }
    }
    // Valeur infinitésimalement au-dessus de la grille, avec l'epsilon prescrit avant ceil.
    let mut records = jeu_q(90);
    fixer_t0(&mut records[0], 0.2 + 1e-13);
    gate(&topo_q_v5(&records), "TOPO-Q-V5", Verdict::Pass);
}

#[test]
fn q_temps_strictement_croissants_meme_si_sur_grille() {
    let mut records = jeu_q(90);
    fixer_t0(&mut records[0], 0.001);
    assert!(records[0]
        .flot_t0
        .as_ref()
        .unwrap()
        .t_eff
        .iter()
        .all(|t| *t == 0.01));
    q_invalide(&records);
}

#[test]
fn q_hors_plafond_valide_non_quantifie_et_separe_de_sans_t0() {
    let mut records = jeu_q(90);
    fixer_t0(&mut records[0], 10.0);
    gate(&topo_q_v5(&records), "TOPO-Q-V5", Verdict::Pass);
    // Le plafond et sa tolérance sont inclusifs, la validité temporelle reste requise.
    records[0].flot_t0.as_mut().unwrap().t_eff[3] = 50.0 + 0.5e-9;
    gate(&topo_q_v5(&records), "TOPO-Q-V5", Verdict::Pass);
    fixer_t0(&mut records[0], 10.002);
    records[1].t0 = None;
    records[1].flot_t0 = None;
    let lignes = topo_q_v5(&records);
    let l = gate(&lignes, "TOPO-Q-V5", Verdict::Fail);
    contient(l, "N=88/100");
    contient(l, "sans t₀ comptées non quantifiées : 1");
    contient(l, "hors plafond comptées non quantifiées : 1");
    let info = gate(&lignes, "TOPO-Q-V5-HORS-PLAFOND", Verdict::Info);
    contient(info, "(1, 1)");
    assert_eq!(integrite_v5("q5", &records).verdict, Verdict::Pass);
    // Un record hors plafond n'est jamais compté dans le mutant non plus.
    records[0].flot_t0.as_mut().unwrap().q_imp[3] = 0.5;
    contient(
        gate(&topo_q_v5(&records), "TOPO-Q-MUT-V5", Verdict::Pass),
        "N′=10/100",
    );
}

#[test]
fn q_infos_stabilite_et_carres_arrondis_sur_population_admissible() {
    let mut records = jeu_q(90);
    for r in &mut records {
        r.flot_t0.as_mut().unwrap().q_imp = vec![1.1, -2.1, 3.1, -4.1];
    }
    fixer_t0(&mut records[0], 10.002);
    records[0].flot_t0.as_mut().unwrap().q_imp.fill(123.0);
    records[1].t0 = None;
    records[1].flot_t0 = None;
    let lignes = topo_q_v5(&records);
    contient(gate(&lignes, "TOPO-Q-V5-STABILITE", Verdict::Info), "98/98");
    let infos: Vec<_> = lignes.iter().filter(|l| l.id == "TOPO-Q-V5-Q2").collect();
    assert_eq!(infos.len(), 4);
    for (l, (k, x)) in infos.iter().zip([(1, 1.0), (2, 4.0), (3, 9.0), (5, 16.0)]) {
        assert_eq!(l.verdict, Verdict::Info);
        contient(l, &format!("k={k} ;"));
        contient(l, "N=98");
        proche(nombre(l, "⟨round(Q(k·t₀))²⟩="), x, 0.0);
    }
}

#[test]
fn lit_toutes_les_infos_invariantes_par_permutation() {
    let records = jeu_lit(6.635, 0.025, 100, 32);
    let reference = topo_lit_v5(&records, RoleLitV5::Nominal);
    let mut permutees = records.clone();
    permutees.reverse();
    assert_eq!(reference, topo_lit_v5(&permutees, RoleLitV5::Nominal));
    // Mélange déterministe : ordre de chaînes et ordre de cfg indépendamment perturbés.
    for i in 0..permutees.len() {
        let j = (i * 1741 + 79) % permutees.len();
        permutees.swap(i, j);
    }
    assert_eq!(reference, topo_lit_v5(&permutees, RoleLitV5::Nominal));
}

#[test]
fn lit_infos_10_et_15_non_disponibles_explicitement() {
    for (role, n) in [(RoleLitV5::Nominal, 100), (RoleLitV5::MutantBeta, 25)] {
        let mut records = jeu_lit(6.635, 0.025, n, 32);
        records[0].cooling.q_tre[10] = f64::NAN;
        records[0].cooling.q_tre[15] = f64::INFINITY;
        let lignes = topo_lit_v5(&records, role);
        for k in [10, 15] {
            let infos: Vec<_> = lignes
                .iter()
                .filter(|l| {
                    l.verdict == Verdict::Info && l.detail.contains(&format!("après {k} balayages"))
                })
                .collect();
            assert_eq!(infos.len(), 1);
            contient(infos[0], "non disponible");
        }
    }
    let mut records = jeu_lit(6.635, 0.025, 100, 32);
    records[0].cooling.q_tre[10] = f64::NAN;
    let lignes = analyser_v5("lit5", &records);
    assert!(lignes
        .iter()
        .any(|l| l.verdict == Verdict::Info && l.detail.starts_with("⟨Q_L²⟩ après 15 balayages=")));
}

fn jeu_recert(n: usize, decalage_plaq: f64, decalage_q2: f64) -> Vec<RecordTopo> {
    let mut records = jeu_lit(6.635, 0.025, n, 32);
    for r in &mut records {
        let local = ((r.chaine - 1) % 16 + 1) as f64;
        let cuda = f64::from(r.chaine > 16);
        r.plaq0 = 0.6364 + 1e-5 * (local - 8.5) + cuda * decalage_plaq;
        let ecart = if r.cfg == n && n % 2 == 1 {
            0.0
        } else if r.cfg % 2 == 0 {
            -0.125
        } else {
            0.125
        };
        let q = (6.635 + 0.025 * (local - 8.5) + cuda * decalage_q2 + ecart).sqrt();
        r.cooling.q_tre.fill(q);
        r.cooling.q_imp.fill(q);
    }
    records
}

#[test]
fn recert_deux_observables_et_deux_ensembles() {
    for n in [100, 25] {
        let (role, ens, principale) = if n == 100 {
            (RoleLitV5::Nominal, "lit5", "TOPO-LIT-V5")
        } else {
            (RoleLitV5::MutantBeta, "lit5_mut", "TOPO-LIT-MUT-V5")
        };
        for (dp, dq, attendu) in [
            (0.0, 0.0, Verdict::Pass),
            (0.0005, 0.0, Verdict::Fail),
            (0.0, 0.5, Verdict::Fail),
        ] {
            let records = jeu_recert(n, dp, dq);
            assert_eq!(integrite_v5(ens, &records).verdict, Verdict::Pass);
            let lignes = topo_lit_v5(&records, role);
            assert_eq!(lignes[0].id, principale);
            assert_eq!(lignes[1].id, format!("RECERT-GEN-{ens}"));
            let l = gate(&lignes, &format!("RECERT-GEN-{ens}"), attendu);
            for nom in ["plaq0 HIP=", "Q_L² HIP="] {
                contient(l, nom);
            }
            contient(l, "(16 chaînes)");
            contient(l, "|z| < 3.5");
            // Oracle analytique : mêmes 16 centres espacés, variance des moyennes = pas²*17/12.
            for (portion, decalage, pas) in [
                (l.detail.split(" ; Q_L²").next().unwrap(), dp, 1e-5),
                (l.detail.split(" ; Q_L²").nth(1).unwrap(), dq, 0.025),
            ] {
                let z_attendu = decalage / (2.0_f64 * 17.0 / 12.0).sqrt() / pas;
                let z: f64 = portion
                    .split_once("z=")
                    .unwrap()
                    .1
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                proche(z, z_attendu, 5.1e-7);
            }
        }
        let records = jeu_recert(n, 0.0, 0.0);
        gate(
            &topo_lit_v5(&records[..records.len() - n], role),
            &format!("RECERT-GEN-{ens}"),
            Verdict::Skip,
        );
    }
}

#[test]
fn recert_q5_uniquement_info_et_deux_z() {
    let lignes = analyser_v5("q5", &jeu_q(90));
    let l = gate(&lignes, "TOPO-Q-V5-RECERT-GEN-INFO", Verdict::Info);
    contient(l, "plaq0 HIP=");
    contient(l, "Q_L² HIP=");
    contient(l, "(2 chaînes)");
    assert_eq!(l.detail.matches("z=").count(), 2);
    assert!(!lignes.iter().any(|l| l.id.starts_with("RECERT-GEN-")));
}

#[test]
fn records_tronques_jamais_de_panique_pour_aucune_entree_v5() {
    for (ens, n) in [("q5", 25), ("lit5", 100), ("lit5_mut", 25)] {
        let base = if ens == "q5" {
            jeu_q(90)
        } else {
            jeu_lit(6.635, 0.025, n, 32)
        };
        for champ in 0..9 {
            for len in [0, 1, 3, 20] {
                let mut records = base.clone();
                let r = records.last_mut().unwrap();
                if r.flot_t0.is_none() {
                    r.t0 = Some(0.2);
                    r.flot_t0 = record(ens, 1, 1).flot_t0;
                }
                let v = match champ {
                    0 => &mut r.flot_a2.e_plaq,
                    1 => &mut r.flot_a2.q_imp,
                    2 => &mut r.flot_a2.q_tre,
                    3 => &mut r.flot_t0.as_mut().unwrap().t_eff,
                    4 => &mut r.flot_t0.as_mut().unwrap().q_imp,
                    5 => &mut r.flot_t0.as_mut().unwrap().q_tre,
                    6 => &mut r.cooling.action,
                    7 => &mut r.cooling.q_imp,
                    _ => &mut r.cooling.q_tre,
                };
                v.truncate(len);
                let resultat = std::panic::catch_unwind(|| {
                    integrite_v5(ens, &records);
                    topo_q_v5(&records);
                    for role in [
                        RoleLitV5::Nominal,
                        RoleLitV5::MutantOp,
                        RoleLitV5::MutantBeta,
                    ] {
                        topo_lit_v5(&records, role);
                    }
                    analyser_v5(ens, &records)
                });
                assert!(resultat.is_ok(), "{ens} champ={champ} len={len}");
            }
        }
    }
    for ens in ["q5", "lit5", "lit5_mut"] {
        assert_eq!(analyser_v5(ens, &[])[0].verdict, Verdict::Fail);
    }
}

#[test]
fn lit_egalite_exacte_au_quantile_echoue_nominal_et_mutants() {
    // Ces centres donnent z exactement ±q en f64, avant l'arrondi du rapport.
    // Aucune tolérance ne doit transformer l'égalité en PASS, dans aucun rôle.
    let q = 3.260_941_918_894_39_f64;
    for (n, cible, pas, z_attendu) in [
        (100, 6.825_792_500_561_316, 0.01022, q),
        (100, 6.443_843_515_874_17, 0.010_450_000_000_000_001, -q),
        (25, 6.825_934_046_663_902, 0.01031, q),
        (25, 6.444_502_581_036_148, 0.01003, -q),
    ] {
        let mut records = jeu_lit(cible, pas, n, 32);
        for r in &mut records {
            let mk = cible + pas * (r.chaine as f64 - 16.5);
            let ecart = if n % 2 == 1 && r.cfg == n {
                0.0
            } else if r.cfg % 2 == 0 {
                -0.125
            } else {
                0.125
            };
            let charge = (mk + ecart).sqrt();
            r.cooling.q_tre.fill(charge);
            r.cooling.q_imp.fill(charge);
        }
        let (x, sigma, _) = statistique(&records, false);
        let z = (x - 6.635) / sigma.hypot(0.056);
        assert_eq!(z, z_attendu, "oracle brut n={n}, cible={cible}");
        let roles: &[RoleLitV5] = if n == 100 {
            &[RoleLitV5::Nominal, RoleLitV5::MutantOp]
        } else {
            &[RoleLitV5::MutantBeta]
        };
        for &role in roles {
            let id = match role {
                RoleLitV5::Nominal => "TOPO-LIT-V5",
                RoleLitV5::MutantOp => "TOPO-LIT-MUT-OP",
                RoleLitV5::MutantBeta => "TOPO-LIT-MUT-V5",
            };
            let lignes = topo_lit_v5(&records, role);
            let l = gate(&lignes, id, Verdict::Fail);
            if role != RoleLitV5::Nominal {
                contient(l, "ECHEC_OUTIL mutant non rouge");
            }
        }
    }
}
