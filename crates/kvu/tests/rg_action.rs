//! Tests d'intégration du module `rg action` (M2) : T2 à T12.
//!
//! Les ensembles sont générés de façon déterministe (graines fixées) dans des dossiers
//! temporaires sous `CARGO_TARGET_TMPDIR`, supprimés à la fin. Les appels au binaire
//! utilisent `CARGO_BIN_EXE_kvu` avec sorties capturées.

use kvu::commun::ChampHote;
use kvu::heritage::su2_hb::CpuSu2;
use kvu::rg::action::boucles::{Famille, TableFormes};
use kvu::rg::action::estimation::{estimer, stats_config, Estimateur, StatsGroupe};
use kvu::rg::action::gibbs::{Gibbs, ParamGibbs};
use kvu::rg::flot::sd::{beta_sd, stats_sd, ModeRapport, RapportBessel, StatsSd};
use kvu::stockage::npy::{ecrire_champ, FormatChamp};
use kvu::stockage::GardeSsd;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

const GRAINE_T5_B23: u64 = 2611020101;
const GRAINE_T5_B15: u64 = 2611020102;
const GRAINE_T6: u64 = 2611020103;
const GRAINE_T7_GIBBS: u64 = 2611020104;
const GRAINE_T7_CPU: u64 = 2611020105;
const GRAINE_T8_JAUGE: u64 = 2611020106;
const GRAINE_T10: u64 = 2611020107;

// ------------------------------------------------------------------------------------------
// Utilitaires
// ------------------------------------------------------------------------------------------

struct Dossier(PathBuf);
impl Dossier {
    fn nouveau(nom: &str) -> Self {
        let p = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "rg_action_{}_{}",
            nom,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn chemin(&self) -> &Path {
        &self.0
    }
}
impl Drop for Dossier {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn kvu(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kvu"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()
        .unwrap()
}

fn ecrire_champ_fichier(chemin: &Path, champ: &ChampHote) {
    ecrire_champ(chemin, champ, FormatChamp::Q64, &GardeSsd::nouvelle(0.0)).unwrap();
}

/// Intégrateur (M2 r2) : nombre de fils des calculs lourds des tests (les résultats n'en dépendent pas : déterminisme testé par T9) ;
/// r2 utilisait 1 fil et T5 durait 29 min sur une machine chargée.
fn fils_tests() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(4)
}

fn ecrire_manifeste(dossier: &Path, beta: f64, l: usize) {
    std::fs::write(
        dossier.join("manifest.json"),
        format!("{{\"beta\": {beta}, \"l\": {l}}}"),
    )
    .unwrap();
}

fn champ_depuis_cpusu2(gen: &CpuSu2, l: usize) -> ChampHote {
    ChampHote { l, q: gen.flat() }
}

fn generer_wilson(
    l: usize,
    graine: u64,
    beta: f64,
    thermalisation: usize,
    n_configs: usize,
    espacement: usize,
) -> Vec<ChampHote> {
    let mut gen = CpuSu2::new(l, graine).unwrap();
    gen.init(true);
    for _ in 0..thermalisation {
        gen.step(beta, 4);
    }
    let mut configs = Vec::with_capacity(n_configs);
    for _ in 0..n_configs {
        configs.push(champ_depuis_cpusu2(&gen, l));
        for _ in 0..espacement {
            gen.step(beta, 4);
        }
    }
    configs
}

fn stats_sd_configs(configs: &[ChampHote]) -> Vec<StatsSd> {
    configs.iter().map(|c| stats_sd(c).unwrap()).collect()
}

fn stats_config_famille(
    configs: &[ChampHote],
    table: &TableFormes,
    famille: &Famille,
    fils: usize,
) -> Vec<StatsGroupe> {
    configs
        .iter()
        .map(|c| stats_config(c, table, famille, fils).unwrap())
        .collect()
}

fn former_blocs(stats: &[StatsGroupe], b: usize) -> Vec<StatsGroupe> {
    let total = stats.len();
    assert!(b >= 2 && b <= total);
    let mut groupes = Vec::with_capacity(b);
    let mut debut = 0;
    for g in 0..b {
        let taille = total / b + if g < total % b { 1 } else { 0 };
        let mut groupe = StatsGroupe::default();
        for s in &stats[debut..debut + taille] {
            groupe.fusionner(s);
        }
        groupes.push(groupe);
        debut += taille;
    }
    groupes
}

// ------------------------------------------------------------------------------------------
// Données partagées T5 (β=2.3 et β=1.5)
// ------------------------------------------------------------------------------------------

#[derive(Clone)]
struct DonneesT5 {
    configs: Vec<ChampHote>,
    stats_p: Vec<StatsGroupe>,
    stats_sd: Vec<StatsSd>,
}

static T5: OnceLock<[DonneesT5; 2]> = OnceLock::new();

fn donnees_t5() -> &'static [DonneesT5; 2] {
    T5.get_or_init(|| {
        // Générateurs indépendants : mêmes graines et même ordre des opérations
        // qu'en séquentiel, donc configurations identiques au bit.
        let configurations = std::thread::scope(|scope| {
            let b23 = scope.spawn(|| generer_wilson(8, GRAINE_T5_B23, 2.3, 200, 192, 5));
            let b15 = generer_wilson(8, GRAINE_T5_B15, 1.5, 200, 192, 5);
            [b23.join().unwrap(), b15]
        });
        let table = TableFormes::construire().unwrap();
        let famille_p = Famille::analyser("P").unwrap();
        configurations.map(|configs| {
            let stats_p = stats_config_famille(&configs, &table, &famille_p, fils_tests());
            let stats_sd = stats_sd_configs(&configs);
            DonneesT5 {
                configs,
                stats_p,
                stats_sd,
            }
        })
    })
}

fn donnees_t5_b23() -> &'static DonneesT5 {
    &donnees_t5()[0]
}
fn donnees_t5_b15() -> &'static DonneesT5 {
    &donnees_t5()[1]
}

// ------------------------------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------------------------------

#[test]
fn t2_champ_froid_cli_echec_explicite() {
    let dossier = Dossier::nouveau("t2_froid");
    let l = 4;
    let champ = ChampHote::froid(l).unwrap();
    // Intégrateur (M2 r2) : cfg_000000 est exclu par la découverte (§5.1, état initial) ; r2 écrivait les indices 0 et 1, soit UNE configuration
    // lue. On écrit les indices 1 à 4 (quatre configurations, blocs:2 valide).
    for i in 1..=4 {
        let chemin = dossier.chemin().join(format!("cfg_{:06}.npy", i));
        ecrire_champ_fichier(&chemin, &champ);
    }
    ecrire_manifeste(dossier.chemin(), 2.0, l);
    let sortie = dossier.chemin().join("sortie");
    std::fs::create_dir_all(&sortie).unwrap();

    let out = kvu(&[
        "rg",
        "action",
        "--ensemble",
        dossier.chemin().to_str().unwrap(),
        "--pas",
        "0",
        // Intégrateur (M2 r2) : une seule chaîne ⇒ « chaines » est refusé (rc 2, §5.6) ; le test vise l'échec des ESTIMATEURS (rc 8).
        "--jackknife",
        "blocs:2",
        "--sortie",
        sortie.to_str().unwrap(),
        "--fils",
        "1",
    ]);
    assert_eq!(
        out.status.code(),
        Some(8),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let json_path = sortie.join("action.json");
    assert!(json_path.exists(), "action.json absent");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
    assert_eq!(doc["pl"]["etat"], "echec");
    assert_eq!(doc["sd"]["etat"], "echec");
    assert_eq!(doc["n_configs"], 4);
    assert!(doc.get("ecart_pl_sd").is_none());
    assert!(doc["beta_sd_un_couplage"]["valeur"].is_null());
    assert!(doc["beta_sd_un_couplage"]["erreur"].is_null());
}

#[test]
fn t3_champ_haar_couplages_compatibles_zero() {
    let l = 6;
    let mut configs = Vec::with_capacity(8);
    for i in 0..8u64 {
        let mut gen = CpuSu2::new(l, 26110202 + i).unwrap();
        gen.init(true);
        configs.push(champ_depuis_cpusu2(&gen, l));
    }
    let table = TableFormes::construire().unwrap();
    let famille = Famille::analyser("P,R,C,T").unwrap();
    let stats = stats_config_famille(&configs, &table, &famille, fils_tests());
    let groupes = former_blocs(&stats, 4);

    for estimateur in [Estimateur::Pl, Estimateur::Sd] {
        let res = estimer(&groupes, &famille, estimateur, fils_tests()).unwrap();
        for (i, v) in res.couplages.iter().enumerate() {
            assert!(
                v.valeur.abs() / v.erreur.max(f64::MIN_POSITIVE) < 4.0,
                "estimateur {:?} couplage {} : |β|/σ = {:.2} ≥ 4",
                estimateur,
                famille.noms()[i],
                v.valeur.abs() / v.erreur
            );
        }
    }
}

#[test]
fn t4_non_regression_sd() {
    use kvu::rg::action::estimation::resoudre_pl;
    // Fixture
    let fixture = format!(
        "{}/tests/donnees/rg/cfg_su2_L8_b2.40_s24001.npy",
        env!("CARGO_MANIFEST_DIR")
    );
    let champ_fix = kvu::stockage::npy::lire_champ(Path::new(&fixture)).unwrap();
    let table = TableFormes::construire().unwrap();
    let famille = Famille::analyser("P").unwrap();
    let stats_fix = stats_config(&champ_fix, &table, &famille, 1).unwrap();
    let sol_fix = resoudre_pl(&[&stats_fix], 1, &[0.0], 1).unwrap();
    let sd_fix = stats_sd(&champ_fix).unwrap();
    let b_fix = beta_sd(&sd_fix, RapportBessel::I2SurI1, ModeRapport::Exact).unwrap();
    assert!(
        (sol_fix.beta[0] - b_fix).abs() / b_fix.abs().max(1e-300) <= 1e-10,
        "fixture : {:.15e} vs {:.15e}",
        sol_fix.beta[0],
        b_fix
    );

    // Ensemble T5 (β=2.3)
    let donnees = donnees_t5_b23();
    let mut stats_sd_toutes = StatsSd::default();
    for s in &donnees.stats_sd {
        stats_sd_toutes.fusionner(s);
    }
    let b_exact = beta_sd(&stats_sd_toutes, RapportBessel::I2SurI1, ModeRapport::Exact).unwrap();
    let groupes_p: Vec<StatsGroupe> = former_blocs(&donnees.stats_p, 8);
    let refs: Vec<&StatsGroupe> = groupes_p.iter().collect();
    let sol = resoudre_pl(&refs, 1, &[0.0], 1).unwrap();
    assert!(
        (sol.beta[0] - b_exact).abs() / b_exact.abs().max(1e-300) <= 1e-10,
        "T5 : {:.15e} vs {:.15e}",
        sol.beta[0],
        b_exact
    );
}

#[test]
fn t5_wilson_non_bloque_beta_connu() {
    let table = TableFormes::construire().unwrap();

    for (donnees, beta_attendu) in [(donnees_t5_b23(), 2.3), (donnees_t5_b15(), 1.5)] {
        let famille_complete = Famille::analyser("P,R,C,T").unwrap();
        let stats_complete =
            stats_config_famille(&donnees.configs, &table, &famille_complete, fils_tests());
        let groupes_complete = former_blocs(&stats_complete, 8);

        for estimateur in [Estimateur::Pl, Estimateur::Sd] {
            let res = estimer(
                &groupes_complete,
                &famille_complete,
                estimateur,
                fils_tests(),
            )
            .unwrap();
            let bp = res.couplages[0].valeur;
            let err_p = res.couplages[0].erreur;
            assert!(
                (bp / beta_attendu - 1.0).abs() < 0.01,
                "{} β_P/β−1 = {:.4e} (β={})",
                estimateur.nom(),
                (bp / beta_attendu - 1.0).abs(),
                beta_attendu
            );
            assert!(
                err_p / beta_attendu < 0.006,
                "{} σ(β_P)/β = {:.4e}",
                estimateur.nom(),
                err_p / beta_attendu
            );
            for j in [1, 2, 3] {
                let z = (res.couplages[j].valeur / res.couplages[j].erreur).abs();
                assert!(z < 3.5, "{} γ={} z={:.2e}", estimateur.nom(), j, z);
            }
        }

        // Famille {P} seule
        let famille_p = Famille::analyser("P").unwrap();
        let stats_p = stats_config_famille(&donnees.configs, &table, &famille_p, fils_tests());
        let groupes_p = former_blocs(&stats_p, 8);
        for estimateur in [Estimateur::Pl, Estimateur::Sd] {
            let res = estimer(&groupes_p, &famille_p, estimateur, fils_tests()).unwrap();
            assert!(
                (res.couplages[0].valeur / beta_attendu - 1.0).abs() < 0.01,
                "famille {{P}} {} β_P/β−1 = {:.4e}",
                estimateur.nom(),
                (res.couplages[0].valeur / beta_attendu - 1.0).abs()
            );
        }
    }
}

#[test]
fn t6_fermeture_et_troncature_gibbs() {
    let famille = Famille::analyser("P,R,C,T").unwrap();
    let beta_vrai = vec![1.5, -0.10, 0.05, 0.06];
    let params = ParamGibbs {
        l: 4,
        famille: famille.clone(),
        beta: beta_vrai.clone(),
        graine: GRAINE_T6,
        depart_chaud: true,
        sur_relaxations: 1,
    };
    let mut gibbs = Gibbs::nouveau(params).unwrap();
    for _ in 0..300 {
        gibbs.balayage();
    }
    let mut configs = Vec::with_capacity(800);
    for i in 0..800 {
        gibbs.balayage();
        configs.push(gibbs.champ());
        if i < 799 {
            for _ in 0..2 {
                gibbs.balayage();
            }
        }
    }
    assert_eq!(gibbs.plafonds(), 0, "plafonds non nuls");

    let table = TableFormes::construire().unwrap();
    let stats_complet = stats_config_famille(&configs, &table, &famille, fils_tests());
    let groupes = former_blocs(&stats_complet, 8);

    for estimateur in [Estimateur::Pl, Estimateur::Sd] {
        let res = estimer(&groupes, &famille, estimateur, fils_tests()).unwrap();
        for (i, seuil) in [(1, 0.025), (2, 0.0125), (3, 0.015)] {
            assert!(
                res.couplages[i].erreur < seuil,
                "{} γ={} : σ={} ≥ {}",
                estimateur.nom(),
                i,
                res.couplages[i].erreur,
                seuil
            );
        }
        for (i, b) in beta_vrai.iter().enumerate() {
            assert!(
                (res.couplages[i].valeur - b).abs() < 4.0 * res.couplages[i].erreur,
                "{} γ={} : {:.6}±{:.6} vs {:.6}",
                estimateur.nom(),
                i,
                res.couplages[i].valeur,
                res.couplages[i].erreur,
                b
            );
        }
    }

    // Troncature {P}
    let famille_p = Famille::analyser("P").unwrap();
    let stats_p = stats_config_famille(&configs, &table, &famille_p, fils_tests());
    let groupes_p = former_blocs(&stats_p, 8);
    let res_p = estimer(&groupes_p, &famille_p, Estimateur::Pl, fils_tests()).unwrap();
    assert!(
        (res_p.couplages[0].valeur - 1.5).abs() > 5.0 * res_p.couplages[0].erreur,
        "troncature {{P}} non détectée : {:.6}±{:.6}",
        res_p.couplages[0].valeur,
        res_p.couplages[0].erreur
    );

    // Troncature {P,R}
    let famille_pr = Famille::analyser("P,R").unwrap();
    let stats_pr = stats_config_famille(&configs, &table, &famille_pr, fils_tests());
    let groupes_pr = former_blocs(&stats_pr, 8);
    let res_pr = estimer(&groupes_pr, &famille_pr, Estimateur::Pl, fils_tests()).unwrap();
    let depasse_p = (res_pr.couplages[0].valeur - 1.5).abs() / res_pr.couplages[0].erreur;
    let depasse_r = (res_pr.couplages[1].valeur + 0.10).abs() / res_pr.couplages[1].erreur;
    assert!(
        depasse_p > 5.0 || depasse_r > 5.0,
        "troncature {{P,R}} non détectée : dépassements {:.2} et {:.2}",
        depasse_p,
        depasse_r
    );
}

#[test]
fn t7_gibbs_contre_bain_thermique_reference() {
    let famille = Famille::analyser("P").unwrap();
    let beta = 2.0;
    let l = 4;

    let params_gibbs = ParamGibbs {
        l,
        famille: famille.clone(),
        beta: vec![beta],
        graine: GRAINE_T7_GIBBS,
        depart_chaud: true,
        sur_relaxations: 1,
    };
    let mut gibbs = Gibbs::nouveau(params_gibbs).unwrap();
    let mut mesures_gibbs = Vec::with_capacity(4000);
    for _ in 0..500 {
        gibbs.balayage();
    }
    for _ in 0..4000 {
        gibbs.balayage();
        mesures_gibbs.push(kvu::rg::swendsen::plaquette_moyenne(&gibbs.champ()));
    }

    let mut cpu = CpuSu2::new(l, GRAINE_T7_CPU).unwrap();
    cpu.init(true);
    for _ in 0..500 {
        cpu.step(beta, 1);
    }
    let mut mesures_cpu = Vec::with_capacity(4000);
    for _ in 0..4000 {
        cpu.step(beta, 1);
        let flat = cpu.flat();
        let champ = ChampHote { l, q: flat };
        mesures_cpu.push(kvu::rg::swendsen::plaquette_moyenne(&champ));
    }

    let stats = |serie: &[f64]| {
        let n_blocs = serie.len() / 100;
        let moyennes: Vec<f64> = (0..n_blocs)
            .map(|b| serie[b * 100..(b + 1) * 100].iter().sum::<f64>() / 100.0)
            .collect();
        let moyenne = moyennes.iter().sum::<f64>() / n_blocs as f64;
        let variance =
            moyennes.iter().map(|m| (m - moyenne).powi(2)).sum::<f64>() / (n_blocs - 1) as f64;
        (moyenne, (variance / n_blocs as f64).sqrt())
    };
    let (moy_g, err_g) = stats(&mesures_gibbs);
    let (moy_c, err_c) = stats(&mesures_cpu);
    let ecart = (moy_g - moy_c).abs();
    let sigma_comb = (err_g.powi(2) + err_c.powi(2)).sqrt();
    assert!(
        ecart < 4.0 * sigma_comb,
        "Gibbs vs CPU : ΔP = {ecart:.5e}, 4σ = {:.5e}",
        4.0 * sigma_comb
    );
}

#[test]
fn t8_invariance_de_jauge() {
    use kvu::heritage::su2_hb::{qdag, qmul};
    let donnees = donnees_t5_b23();
    let configs_source = &donnees.configs[..4];
    let table = TableFormes::construire().unwrap();
    let famille = Famille::analyser("P,R,C,T").unwrap();

    let transformer = |champ: &ChampHote, graine: u64| {
        let l = champ.l;
        let v = champ.v();
        let (k0, k1) = ((graine & 0xFFFF_FFFF) as u32, (graine >> 32) as u32);
        let mut g = vec![[0.0; 4]; v];
        for (s, gs) in g.iter_mut().enumerate() {
            *gs =
                kvu::su2::cpu::haar_q(s as u32, 0xFFFF_FFFF, kvu::commun::ETIQUETTE_JAUGE, k0, k1);
        }
        let mut q_new = vec![0.0; 16 * v];
        for mu in 0..4 {
            for s in 0..v {
                let u = champ.lien(mu * v + s);
                let x_plus = {
                    let mut x = [0usize; 4];
                    let mut t = s;
                    for i in (0..4).rev() {
                        x[i] = t % l;
                        t /= l;
                    }
                    x[mu] = (x[mu] + 1) % l;
                    ((x[0] * l + x[1]) * l + x[2]) * l + x[3]
                };
                let ug = qmul(g[s], qmul(u, qdag(g[x_plus])));
                let norme = (ug[0] * ug[0] + ug[1] * ug[1] + ug[2] * ug[2] + ug[3] * ug[3]).sqrt();
                let o = 4 * (mu * v + s);
                for i in 0..4 {
                    q_new[o + i] = ug[i] / norme;
                }
            }
        }
        ChampHote { l, q: q_new }
    };

    let mut configs_jaugees = Vec::with_capacity(4);
    for (i, cfg) in configs_source.iter().enumerate() {
        configs_jaugees.push(transformer(cfg, GRAINE_T8_JAUGE + i as u64));
    }

    let stats_origine = stats_config_famille(configs_source, &table, &famille, fils_tests());
    let stats_jaugees = stats_config_famille(&configs_jaugees, &table, &famille, fils_tests());
    let groupes_origine = former_blocs(&stats_origine, 2); // 2 blocs de 2 configs
    let groupes_jaugees = former_blocs(&stats_jaugees, 2);

    for estimateur in [Estimateur::Pl, Estimateur::Sd] {
        let res_origine = estimer(&groupes_origine, &famille, estimateur, fils_tests()).unwrap();
        let res_jauge = estimer(&groupes_jaugees, &famille, estimateur, fils_tests()).unwrap();
        // Arbitrage du chef de projet (M2 r2) : écart rapporté à l'échelle des couplages, max(1, max_γ |β_γ|) ; l'écart RELATIF d'un
        // couplage presque nul (β_R ≈ 1e-3 sur un ensemble de Wilson) mesure l'arrondi, pas la covariance (r2 : 5,8e-10 sur β_R).
        let echelle = res_origine
            .couplages
            .iter()
            .map(|c| c.valeur.abs())
            .fold(1.0_f64, f64::max);
        for (i, nom) in famille.noms().iter().enumerate() {
            let rel =
                (res_origine.couplages[i].valeur - res_jauge.couplages[i].valeur).abs() / echelle;
            assert!(
                rel <= 1e-10,
                "{} {} : écart relatif {rel:.3e}",
                estimateur.nom(),
                nom
            );
        }
    }
}

#[test]
fn t9_determinisme_cli_fils() {
    let dossier = Dossier::nouveau("t9_determinisme");
    let l = 6;
    let base = dossier.chemin();
    let chaines = ["c1", "c2"];
    let mut graine = 2611020200u64;
    for nom in chaines.iter() {
        let dir = base.join(nom);
        std::fs::create_dir_all(&dir).unwrap();
        ecrire_manifeste(&dir, 2.4, l);
        for i in 1..=6 {
            graine += 1;
            let mut gen = CpuSu2::new(l, graine).unwrap();
            gen.init(true);
            for _ in 0..10 {
                gen.step(2.4, 4);
            }
            let champ = champ_depuis_cpusu2(&gen, l);
            let chemin = dir.join(format!("cfg_{:06}.npy", i));
            ecrire_champ_fichier(&chemin, &champ);
        }
    }
    let sortie1 = base.join("sortie1");
    let sortie4 = base.join("sortie4");
    std::fs::create_dir_all(&sortie1).unwrap();
    std::fs::create_dir_all(&sortie4).unwrap();

    let out1 = kvu(&[
        "rg",
        "action",
        "--ensemble",
        base.to_str().unwrap(),
        "--pas",
        "0",
        "--jackknife",
        "chaines",
        "--sortie",
        sortie1.to_str().unwrap(),
        "--fils",
        "1",
    ]);
    assert_eq!(
        out1.status.code(),
        Some(0),
        "fils 1 : stdout {}\nstderr {}",
        String::from_utf8_lossy(&out1.stdout),
        String::from_utf8_lossy(&out1.stderr)
    );
    let out4 = kvu(&[
        "rg",
        "action",
        "--ensemble",
        base.to_str().unwrap(),
        "--pas",
        "0",
        "--jackknife",
        "chaines",
        "--sortie",
        sortie4.to_str().unwrap(),
        "--fils",
        "4",
    ]);
    assert_eq!(
        out4.status.code(),
        Some(0),
        "fils 4 : stdout {}\nstderr {}",
        String::from_utf8_lossy(&out4.stdout),
        String::from_utf8_lossy(&out4.stderr)
    );

    let json1 = std::fs::read(sortie1.join("action.json")).unwrap();
    let json4 = std::fs::read(sortie4.join("action.json")).unwrap();
    assert_eq!(json1, json4, "JSON différents entre --fils 1 et --fils 4");
    let doc: Value = serde_json::from_slice(&json1).unwrap();
    assert_eq!(doc["n_configs"], 12);
    assert_eq!(doc["chaines"][0]["n_configs"], 6);
    assert_eq!(doc["chaines"][1]["n_configs"], 6);
}

#[test]
fn t10_pipeline_bloque() {
    let dossier = Dossier::nouveau("t10_pipeline");
    let l = 12;
    let base = dossier.chemin();
    for chaine in ["c1", "c2"] {
        let dir = base.join(chaine);
        std::fs::create_dir_all(&dir).unwrap();
        ecrire_manifeste(&dir, 2.5, l);
    }
    let mut gen = CpuSu2::new(l, GRAINE_T10).unwrap();
    gen.init(true);
    for _ in 0..20 {
        gen.step(2.5, 4);
    }
    let mut idx = 1;
    for chaine in ["c1", "c2"] {
        let dir = base.join(chaine);
        for _ in 0..3 {
            let champ = champ_depuis_cpusu2(&gen, l);
            let chemin = dir.join(format!("cfg_{:06}.npy", idx));
            ecrire_champ_fichier(&chemin, &champ);
            idx += 1;
            for _ in 0..5 {
                gen.step(2.5, 4);
            }
        }
    }

    for operateur in ["dec", "3centre"] {
        let sortie = base.join(format!("sortie_{}", operateur));
        std::fs::create_dir_all(&sortie).unwrap();
        let out = kvu(&[
            "rg",
            "action",
            "--ensemble",
            base.to_str().unwrap(),
            "--pas",
            "1",
            "--operateur",
            operateur,
            "--jackknife",
            "chaines",
            "--sortie",
            sortie.to_str().unwrap(),
            "--fils",
            "1",
        ]);
        assert_eq!(
            out.status.code(),
            Some(0),
            "op {} : stdout {}\nstderr {}",
            operateur,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let json_path = sortie.join("action.json");
        assert!(json_path.exists());
        let doc: Value =
            serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
        assert_eq!(doc["n_configs"], 6);
        assert_eq!(doc["chaines"][0]["n_configs"], 3);
        assert_eq!(doc["chaines"][1]["n_configs"], 3);
        assert_eq!(doc["l_estimation"], 4);
        assert_eq!(doc["pas"], 1);
        assert_eq!(doc["operateur"]["spec"], operateur);
        // couplages et N4 finis
        for estimateur in ["pl", "sd"] {
            assert_eq!(doc[estimateur]["etat"], "ok");
            for (_, v) in doc[estimateur]["couplages"].as_object().unwrap() {
                assert!(v["valeur"].as_f64().unwrap().is_finite());
                assert!(v["erreur"].as_f64().unwrap().is_finite());
            }
            assert!(doc[estimateur]["r4"]["norme"]["valeur"]
                .as_f64()
                .unwrap()
                .is_finite());
        }
    }
}

/// Parent de chaînes valides, chacune avec deux configurations indépendantes d'indices 1 et 2.
fn ecrire_ensemble_cli(parent: &Path, l: usize, n_chaines: usize) {
    std::fs::create_dir_all(parent).unwrap();
    for c in 1..=n_chaines {
        let dir = parent.join(format!("c{c}"));
        std::fs::create_dir_all(&dir).unwrap();
        ecrire_manifeste(&dir, 2.0, l);
        for i in 1..=2 {
            let mut gen = CpuSu2::new(l, 2611021000 + (10 * c + i) as u64).unwrap();
            gen.init(true);
            ecrire_champ_fichier(
                &dir.join(format!("cfg_{i:06}.npy")),
                &champ_depuis_cpusu2(&gen, l),
            );
        }
    }
}

#[test]
fn t12_refus_cli() {
    // Une entrée neuve par cas ; chaque option n'est passée qu'UNE fois.
    for cas in [
        "famille-vide",
        "type",
        "doublon",
        "sans-p",
        "divisibilite",
        "petit-volume",
        "hors-famille",
        "une-chaine",
        "blocs-petit",
        "blocs-grand",
        "max-zero",
        "beta",
        "cote",
        "vide",
        "ecrasement",
        "volume-reel",
        "tailles-groupes",
        "exposant",
        "conversion-exposant",
    ] {
        let dossier = Dossier::nouveau(&format!("t12_{cas}"));
        let ensemble = dossier.chemin().join("ensemble");
        let l = match cas {
            "divisibilite" => 14,
            "petit-volume" => 6,
            _ => 4,
        };
        ecrire_ensemble_cli(&ensemble, l, if cas == "une-chaine" { 1 } else { 2 });
        let sortie = dossier.chemin().join("sortie");
        let (mut famille, mut pas, mut operateur, mut jackknife) =
            ("P,R,C,T", "0", "3centre", "chaines");
        let grand_exposant = usize::MAX.to_string();
        let mut supplement = Vec::new();
        let attendu = match cas {
            "famille-vide" => {
                famille = "";
                "famille de boucles vide"
            }
            "type" => {
                famille = "X";
                "inconnu"
            }
            "doublon" => {
                famille = "P,P";
                "en double"
            }
            "sans-p" => {
                famille = "R";
                "doit contenir"
            }
            "divisibilite" => {
                pas = "1";
                operateur = "dec";
                "non divisible"
            }
            "petit-volume" => {
                pas = "1";
                operateur = "dec";
                "L/3^K"
            }
            "hors-famille" => {
                operateur = "ape:0.5:40";
                "hors"
            }
            "une-chaine" => "au moins 2 chaînes",
            "blocs-petit" => {
                jackknife = "blocs:1";
                "nombre de blocs"
            }
            "blocs-grand" => {
                jackknife = "blocs:5";
                "> nombre de configurations"
            }
            "max-zero" => {
                supplement.extend(["--max-configs", "0"]);
                "--max-configs"
            }
            "beta" => {
                // La racine est un PARENT, sans cfg ni manifeste : c2 est découvert.
                ecrire_manifeste(&ensemble.join("c2"), 2.1, l);
                "β ou L différents"
            }
            "cote" => {
                let c2 = ensemble.join("c2");
                ecrire_manifeste(&c2, 2.0, 6);
                let champ = ChampHote::froid(6).unwrap();
                for i in 1..=2 {
                    ecrire_champ_fichier(&c2.join(format!("cfg_{i:06}.npy")), &champ);
                }
                "β ou L différents"
            }
            "vide" => {
                std::fs::remove_dir_all(&ensemble).unwrap();
                std::fs::create_dir_all(&ensemble).unwrap();
                "ensemble sans configuration"
            }
            "ecrasement" => {
                std::fs::create_dir_all(&sortie).unwrap();
                std::fs::write(sortie.join("action.json"), b"contenu initial a conserver\n")
                    .unwrap();
                "refus d'écraser"
            }
            "volume-reel" => {
                ecrire_champ_fichier(
                    &ensemble.join("c1/cfg_000001.npy"),
                    &ChampHote::froid(6).unwrap(),
                );
                "côté réel"
            }
            "tailles-groupes" => {
                // 1 et 3 configurations ; aucune autre différence entre chaînes.
                std::fs::remove_file(ensemble.join("c1/cfg_000002.npy")).unwrap();
                std::fs::copy(
                    ensemble.join("c2/cfg_000002.npy"),
                    ensemble.join("c2/cfg_000003.npy"),
                )
                .unwrap();
                "rapport > 2"
            }
            "exposant" => {
                pas = "100";
                "3^K"
            }
            "conversion-exposant" => {
                pas = &grand_exposant;
                "3^K"
            }
            _ => unreachable!(),
        };
        let mut args = vec![
            "rg",
            "action",
            "--ensemble",
            ensemble.to_str().unwrap(),
            "--famille",
            famille,
            "--pas",
            pas,
            "--operateur",
            operateur,
            "--jackknife",
            jackknife,
            "--sortie",
            sortie.to_str().unwrap(),
            "--fils",
            "1",
        ];
        args.extend(supplement);
        let out = kvu(&args);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "cas {cas} : {stderr}");
        assert!(
            stderr.contains("configuration :") && stderr.contains(attendu),
            "cas {cas} : refus du module attendu ({attendu}), reçu : {stderr}"
        );
        if cas == "ecrasement" {
            assert_eq!(
                std::fs::read(sortie.join("action.json")).unwrap(),
                b"contenu initial a conserver\n"
            );
        } else {
            assert!(
                !sortie.join("action.json").exists(),
                "cas {cas} : fichier créé"
            );
        }
    }
}

#[test]
fn schema_json_selections_et_garde_memoire() {
    let dossier = Dossier::nouveau("schema_selections");
    let ensemble = dossier.chemin().join("ensemble");
    ecrire_ensemble_cli(&ensemble, 4, 2);
    for (selection, budget, rc) in [
        ("pl", "4", 0),
        ("sd", "0.000000001", 0),
        ("pl,sd", "4", 0),
        ("pl,sd", "0.000000001", 8),
        ("pl", "0.000000001", 8),
    ] {
        let sortie = dossier
            .chemin()
            .join(format!("sortie_{selection}_{budget}"));
        let out = kvu(&[
            "rg",
            "action",
            "--ensemble",
            ensemble.to_str().unwrap(),
            "--estimateurs",
            selection,
            "--memoire-max-go",
            budget,
            "--sortie",
            sortie.to_str().unwrap(),
            "--fils",
            "1",
        ]);
        assert_eq!(
            out.status.code(),
            Some(rc),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let doc: Value =
            serde_json::from_slice(&std::fs::read(sortie.join("action.json")).unwrap()).unwrap();
        assert_eq!(doc["n_configs"], 4);
        let mut deux_ok = true;
        for est in ["pl", "sd"] {
            let demande = selection.split(',').any(|s| s == est);
            assert_eq!(doc.get(est).is_some(), demande, "{selection} : {est}");
            if !demande {
                deux_ok = false;
                continue;
            }
            if est == "pl" && rc == 8 {
                assert_eq!(doc[est]["etat"], "echec");
                assert!(doc[est]["message"]
                    .as_str()
                    .unwrap()
                    .contains("mémoire insuffisante pour PL"));
                deux_ok = false;
                continue;
            }
            assert_eq!(doc[est]["etat"], "ok");
            let matrice = doc[est]["correlations"].as_array().unwrap();
            assert_eq!(matrice.len(), 4);
            assert!(matrice
                .iter()
                .all(|ligne| ligne.as_array().is_some_and(|r| r.len() == 4)));
        }
        assert_eq!(doc.get("ecart_pl_sd").is_some(), deux_ok);
        if deux_ok {
            for nom in ["P", "R", "C", "T"] {
                let pl = doc["pl"]["couplages"][nom]["replicats"].as_array().unwrap();
                let sd = doc["sd"]["couplages"][nom]["replicats"].as_array().unwrap();
                let differences: Vec<f64> = pl
                    .iter()
                    .zip(sd)
                    .map(|(p, s)| p.as_f64().unwrap() - s.as_f64().unwrap())
                    .collect();
                let moyenne = differences.iter().sum::<f64>() / differences.len() as f64;
                let erreur = ((differences.len() - 1) as f64 / differences.len() as f64
                    * differences
                        .iter()
                        .map(|d| (d - moyenne).powi(2))
                        .sum::<f64>())
                .sqrt();
                assert_eq!(doc["ecart_pl_sd"][nom]["erreur"].as_f64().unwrap(), erreur);
            }
        }
    }
}

#[test]
fn gates_json_selection_et_donnees_wilson() {
    let dossier = Dossier::nouveau("gates_json");
    let racine = dossier.chemin();
    for (mode, verdict) in [
        ("sans-option", "SKIP"),
        ("absent", "SKIP"),
        ("vide", "SKIP"),
        ("invalide", "FAIL"),
    ] {
        let grandl = racine.join(mode);
        if mode == "vide" || mode == "invalide" {
            std::fs::create_dir_all(&grandl).unwrap();
        }
        if mode == "invalide" {
            let chaine = grandl.join("L12_b2.3_c1");
            std::fs::create_dir_all(&chaine).unwrap();
            std::fs::write(chaine.join("manifest.json"), "JSON illisible").unwrap();
            ecrire_champ_fichier(
                &chaine.join("cfg_000001.npy"),
                &ChampHote::froid(12).unwrap(),
            );
        }
        let mut args = vec![
            "rg",
            "action-gates",
            "--json",
            "--seul",
            "RG-ACTION-WILSON-b2.3",
            "--fils",
            "1",
        ];
        if mode != "sans-option" {
            args.extend(["--grandl", grandl.to_str().unwrap()]);
        }
        let out = kvu(&args);
        assert_eq!(
            out.status.code(),
            Some(1),
            "{mode} : {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let doc: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(doc["tout_vert"], false);
        assert_eq!(doc["lignes"][0]["verdict"], verdict);
        assert_eq!(doc["lignes"].as_array().unwrap().len(), 1);
    }
    for seul in ["RG-ACTION-INEXISTANTE", "", "RG-ACTION-SD1,INCONNU"] {
        let out = kvu(&[
            "rg",
            "action-gates",
            "--seul",
            seul,
            "--json",
            "--fils",
            "1",
        ]);
        assert_eq!(out.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&out.stderr).contains("--seul"));
    }
    let out = kvu(&[
        "rg",
        "action-gates",
        "--seul",
        "RG-ACTION-SD1",
        "--json",
        "--fils",
        "1",
    ]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let doc: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(doc["tout_vert"], true);
    assert_eq!(doc["lignes"][0]["verdict"], "PASS");
}

/// Publication v0.1 : `rg action` lit directement la sortie d'une chaîne unique de `kvu su2 gen` (clé `betas` à une valeur
/// du manifeste ; avant, seule la clé `beta` des manifestes de production était lue et la commande rendait 4).
#[test]
fn t13_lit_un_ensemble_de_su2_gen() {
    let d = Dossier::nouveau("t13");
    let ens = d.chemin().join("ens");
    let ens_s = ens.to_str().unwrap();
    let gen = kvu(&[
        "su2", "gen", "--backend", "cpu", "--L", "4", "--beta", "2.3", "--graine", "2611080113", "--start", "cold",
        "--therm", "10", "--nmeas", "4", "--nor", "1", "--dump-every", "2", "--format", "q64", "--outdir", ens_s,
    ]);
    assert_eq!(gen.status.code(), Some(0), "{}", String::from_utf8_lossy(&gen.stderr));
    let sortie = d.chemin().join("sortie");
    let out = kvu(&["rg", "action", "--ensemble", ens_s, "--jackknife", "blocs:2", "--sortie", sortie.to_str().unwrap(), "--fils", "2"]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(sortie.join("action.json")).unwrap()).unwrap();
    assert!(doc.to_string().contains("2.3"), "β de la chaîne absent de action.json : {doc}");
}
