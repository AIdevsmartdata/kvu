//! Tests d'acceptation de la torsion de 't Hooft (M4) — révision r3.
//!
//! Les tests physiques T3, T4 et T5(b) partagent les mêmes runs (20 simulations CPU,
//! L = 4, départ froid, nor = 1, 300 pas de thermalisation, 4 000 mesures) afin de
//! rester dans la durée cible. T1, T2, T5(a), T6, T7 et T8 sont indépendants.
//!
//! Les erreurs statistiques sont calculées par blocs de 100 mesures sur la série
//! appariée ΔP(t) = P_per(t) − P_tw(t), avec contrôle par blocs de 200.
//! Ces erreurs supposent des blocs décorrélés ; σ_F utilise les poids cumulés des ΔE.

use kvu::commun::{Depart, Erreur, Graine, Precision};
use kvu::gpu::OptionsCompil;
use kvu::heritage::su2_hb::{self, CpuSu2, Geo, Q};
use kvu::su2::cpu::Cpu64;
use kvu::su2::run::{self, OptionsRun};
use kvu::su2::torsion::{self, Torsion};
use kvu::su2::{ConfigLot, NoyauxSu2, ParamChaine};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const BIN_KVU: &str = env!("CARGO_BIN_EXE_kvu");
const GRILLE_BETA: [f64; 10] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0];

// ---------------------------------------------------------------------------------------------
// Utilitaires
// ---------------------------------------------------------------------------------------------

fn rep(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("su2_torsion")
        .join(format!("{nom}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.parent().unwrap()).unwrap();
    d
}

fn lot(l: usize, beta: f64, graine: u64, depart: Depart, nor: usize) -> ConfigLot {
    ConfigLot {
        l,
        chaines: vec![ParamChaine { beta, graine: Graine(graine), depart }],
        nor,
        precision: Precision::F64,
        options: OptionsCompil::reference_k1(),
        mutant_metro: false,
    }
}

fn octets(p: &Path) -> Vec<u8> {
    fs::read(p).unwrap_or_else(|e| panic!("{} : {e}", p.display()))
}

fn noms_dumps(d: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(d)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("cfg_") && n.ends_with(".npy"))
        .collect();
    v.sort();
    v
}

fn executer_cli(args: &[&str]) -> std::process::Output {
    Command::new(BIN_KVU)
        .arg("su2")
        .args(args)
        .output()
        .expect("échec de l'exécution de kvu")
}

fn assert_rc(output: &std::process::Output, code: i32, contexte: &str) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "{} : code {} attendu, reçu {:?}, stderr: {}",
        contexte,
        code,
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_octets_egaux(a: &Path, b: &Path, contexte: &str) {
    let ba = octets(a);
    let bb = octets(b);
    assert_eq!(ba, bb, "{contexte} : fichiers différents ({a:?} vs {b:?})");
}

fn assert_bits_egaux(a: &[f64], b: &[f64], contexte: &str) {
    assert_eq!(a.len(), b.len(), "{contexte} : longueurs différentes");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{contexte} : bits différents à l'indice {i}");
    }
}

/// Inventaire récursif des noms et octets, y compris les répertoires vides.
fn instantane(dir: &Path) -> BTreeMap<PathBuf, Option<Vec<u8>>> {
    let mut fichiers = BTreeMap::new();
    let mut pile = vec![dir.to_path_buf()];
    while let Some(d) = pile.pop() {
        for entree in fs::read_dir(d).unwrap() {
            let p = entree.unwrap().path();
            let nom = p.strip_prefix(dir).unwrap().to_path_buf();
            if p.is_dir() {
                fichiers.insert(nom, None);
                pile.push(p);
            } else {
                fichiers.insert(nom, Some(octets(&p)));
            }
        }
    }
    fichiers
}

fn assert_diagnostic_torsion(message: &str, attendus: &[&str]) {
    let message = message.to_lowercase();
    assert!(message.contains("torsion"), "diagnostic sans torsion : {message}");
    for attendu in attendus {
        assert!(message.contains(attendu), "diagnostic sans « {attendu} » : {message}");
    }
}

fn assert_refus_torsion(out: &std::process::Output, contexte: &str, attendus: &[&str]) {
    assert_rc(out, 2, contexte);
    assert_diagnostic_torsion(&String::from_utf8_lossy(&out.stderr), attendus);
}

fn assert_config_torsion<T>(resultat: kvu::commun::Resultat<T>, attendus: &[&str]) {
    let erreur = match resultat {
        Err(e) => e,
        Ok(_) => panic!("refus de torsion attendu"),
    };
    assert!(matches!(&erreur, Erreur::Config(_)), "Erreur::Config attendue : {erreur}");
    assert_diagnostic_torsion(&erreur.to_string(), attendus);
}

/// Statistique d'un échantillon par blocs indépendants (moyenne et erreur standard).
fn stats_blocs(valeurs: &[f64], taille_bloc: usize) -> (f64, f64) {
    assert_eq!(valeurs.len() % taille_bloc, 0, "taille de bloc invalide");
    let n_blocs = valeurs.len() / taille_bloc;
    let moyennes: Vec<f64> = valeurs
        .chunks(taille_bloc)
        .map(|bloc| bloc.iter().sum::<f64>() / taille_bloc as f64)
        .collect();
    let moyenne = moyennes.iter().sum::<f64>() / n_blocs as f64;
    let variance = moyennes.iter().map(|m| (m - moyenne).powi(2)).sum::<f64>() / (n_blocs - 1) as f64;
    let erreur = (variance / n_blocs as f64).sqrt();
    (moyenne, erreur)
}

/// Statistique sur un vecteur de moyennes de blocs déjà calculées.
fn stats_moyennes_blocs(blocs: &[f64]) -> (f64, f64) {
    let n = blocs.len();
    assert!(n >= 2, "au moins deux blocs requis");
    let moyenne = blocs.iter().sum::<f64>() / n as f64;
    let variance = blocs.iter().map(|m| (m - moyenne).powi(2)).sum::<f64>() / (n - 1) as f64;
    (moyenne, (variance / n as f64).sqrt())
}

/// Somme directe des z_p · ½Tr U_p sur toutes les plaquettes, avec une pile
/// arbitraire dans le plan (0,1). Les autres plans ont toujours z = +1.
fn somme_plaquettes_directe(g: &Geo, u: &[Q], pile0: usize, pile1: usize) -> f64 {
    let ld = |nu: usize, site: usize| u[nu * g.v + site];
    (0..g.v)
        .map(|s| {
            let x = g.coords(s);
            let mut acc = 0.0;
            for mu in 0..4 {
                let smu = g.nb_up(s, mu, x[mu]);
                for nu in mu + 1..4 {
                    let snu = g.nb_up(s, nu, x[nu]);
                    let p = su2_hb::qmul(
                        su2_hb::qmul(ld(mu, s), ld(nu, smu)),
                        su2_hb::qmul(su2_hb::qdag(ld(mu, snu)), su2_hb::qdag(ld(nu, s))),
                    );
                    let z = if (mu, nu) == (0, 1) && x[0] == pile0 && x[1] == pile1 { -1.0 } else { 1.0 };
                    acc += z * p[0];
                }
            }
            acc
        })
        .sum()
}

/// Boucle de Polyakov brute P_mu(x) (produit des liens sur le cercle partant de x).
fn polyakov_brut(g: &Geo, u: &[Q], mu: usize, x: &[usize; 4]) -> Q {
    let mut s = ((x[0] * g.l + x[1]) * g.l + x[2]) * g.l + x[3];
    let mut longitudinal = x[mu];
    let mut p: Q = [1.0, 0.0, 0.0, 0.0];
    for _ in 0..g.l {
        p = su2_hb::qmul(p, u[mu * g.v + s]);
        s = g.nb_up(s, mu, longitudinal);
        longitudinal = (longitudinal + 1) % g.l;
    }
    p
}

// ---------------------------------------------------------------------------------------------
// T1 : torsion triviale = périodique AU BIT
// ---------------------------------------------------------------------------------------------

#[test]
fn t1_torsion_triviale_egale_periodique_au_bit() {
    // (a) agrafe_tordue = staple sur champ de Haar L=4 et L=6
    for l in [4usize, 6] {
        let g = Geo::new(l);
        let mut rng = CpuSu2::new(l, 12345).unwrap();
        rng.init(true);
        let t = Torsion::triviale();
        for mu in 0..4 {
            for s in 0..g.v {
                let x = g.coords(s);
                let a_tw = torsion::agrafe_tordue(&rng.u, &g, mu, s, &x, &t);
                let a_st = su2_hb::staple(&rng.u, &g, mu, s, &x);
                assert_bits_egaux(&a_tw, &a_st, &format!("agrafe ({mu},{s}) L={l}"));
            }
        }
    }

    // (b) Cpu64 avec sweep_hb_tordu/sweep_or_tordu (torsion triviale) contre CpuSu2, L=4 et L=6
    for l in [4usize, 6] {
        let mut cpu = Cpu64::nouveau(lot(l, 2.3, 26110401, Depart::Chaud, 2)).unwrap();
        cpu.init().unwrap();
        let mut ref_cpu = CpuSu2::new(l, 26110401).unwrap();
        ref_cpu.init(true);
        let t = Torsion::triviale();
        for _ in 0..5 {
            cpu.sweep_hb_tordu(&t);
            cpu.sweep_or_tordu(&t);
            cpu.sweep_or_tordu(&t);
            ref_cpu.step(2.3, 2);
        }
        assert_bits_egaux(&cpu.telecharger(0).unwrap().q, &ref_cpu.flat(), &format!("champs L={l}"));
        let mut standard = Cpu64::nouveau(lot(l, 2.3, 26110401, Depart::Chaud, 2)).unwrap();
        standard.init().unwrap();
        for _ in 0..5 {
            standard.sweep_hb().unwrap();
            standard.sweep_or().unwrap();
            standard.sweep_or().unwrap();
        }
        assert_bits_egaux(&standard.telecharger(0).unwrap().q, &ref_cpu.flat(), &format!("chemin historique L={l}"));
    }

    // (c) CLI : avec --torsion aucune et sans option => sorties identiques
    let dir_a = rep("t1c_a");
    let dir_b = rep("t1c_b");
    let common = [
        "gen", "--backend", "cpu", "--L", "4", "--beta", "2.3", "--graine", "26110402",
        "--therm", "20", "--nmeas", "30", "--nor", "2", "--dump-every", "10", "--format", "q64",
    ];
    let mut args_a = common.to_vec();
    args_a.extend_from_slice(&["--torsion", "aucune", "--outdir", dir_a.to_str().unwrap()]);
    let out_a = executer_cli(&args_a);
    assert_rc(&out_a, 0, "run avec --torsion aucune");

    let mut args_b = common.to_vec();
    args_b.extend_from_slice(&["--outdir", dir_b.to_str().unwrap()]);
    let out_b = executer_cli(&args_b);
    assert_rc(&out_b, 0, "run sans --torsion");

    for fichier in ["measurements.csv", "therm.csv"] {
        assert_octets_egaux(&dir_a.join(fichier), &dir_b.join(fichier), fichier);
    }
    let da = noms_dumps(&dir_a);
    let db = noms_dumps(&dir_b);
    let attendus = ["cfg_000010.npy", "cfg_000020.npy", "cfg_000030.npy"];
    assert_eq!(da, attendus, "dumps attendus avec torsion triviale");
    assert_eq!(db, attendus, "dumps attendus sans option torsion");
    for n in da {
        assert_octets_egaux(&dir_a.join(&n), &dir_b.join(&n), &n);
    }

    let _ = fs::remove_dir_all(&dir_a);
    let _ = fs::remove_dir_all(&dir_b);
}

// ---------------------------------------------------------------------------------------------
// T2 : action tordue exacte
// ---------------------------------------------------------------------------------------------

#[test]
fn t2_action_tordue_exacte() {
    let l = 4;
    let g = Geo::new(l);
    let t_01 = Torsion::analyser("01").unwrap();

    // Champ de Haar pour (a) et (b)
    let mut rng = CpuSu2::new(l, 777).unwrap();
    rng.init(true);

    // (a) somme directe z_p * 1/2 Tr U_p = 6V * plaquette_tordue
    let somme_directe = somme_plaquettes_directe(&g, &rng.u, l - 1, l - 1);
    let pt = torsion::plaquette_tordue(&g, &rng.u, &t_01);
    let rel = (somme_directe - 6.0 * g.v as f64 * pt).abs() / (6.0 * g.v as f64 * pt).abs();
    assert!(rel < 1e-12, "somme directe = {somme_directe}, 6V·P_tw = {}, rel = {rel}", 6.0 * g.v as f64 * pt);

    // (b) pour chaque lien, Re(U_ℓ · agrafe) = somme directe des z_p ½Tr U_p des 6 plaquettes
    let ld = |nu: usize, site: usize| rng.u[nu * g.v + site];
    for mu in 0..4 {
        for s in 0..g.v {
            let x = g.coords(s);
            let ag = torsion::agrafe_tordue(&rng.u, &g, mu, s, &x, &t_01);
            let u_l = rng.u[mu * g.v + s];
            // Correction r2 : Re(U_ℓ · A) = composante 0 de qmul(U, A)
            let q = su2_hb::qmul(u_l, ag);
            let valeur_agrafe = q[0];

            // Somme directe des 6 plaquettes contenant le lien (x, mu)
            let mut somme_directe_lien = 0.0;
            for nu in 0..4 {
                if nu == mu {
                    continue;
                }
                // plaquette "t" de coin x
                let smu = g.nb_up(s, mu, x[mu]);
                let snu = g.nb_up(s, nu, x[nu]);
                let p = su2_hb::qmul(
                    su2_hb::qmul(ld(mu, s), ld(nu, smu)),
                    su2_hb::qmul(su2_hb::qdag(ld(mu, snu)), su2_hb::qdag(ld(nu, s))),
                );
                // Intégrateur (M4 r2) : la torsion « 01 » porte sur le plan NON ordonné {0, 1} (r2 oubliait mu = 1, nu = 0).
                let plan01 = (mu, nu) == (0, 1) || (mu, nu) == (1, 0);
                let zt = if plan01 && x[0] == l - 1 && x[1] == l - 1 { -1.0 } else { 1.0 };
                somme_directe_lien += zt * p[0];

                // plaquette "b" de coin x - ν̂
                let sdn = g.nb_dn(s, nu, x[nu]);
                let smu_dn = g.nb_dn(smu, nu, x[nu]);
                // Intégrateur (M4 r2) : plaquette de coin x − ν̂ parcourue depuis U_μ(x) (½Tr invariant par inversion pour SU(2)) :
                // U_μ(x) U_ν(x+μ̂−ν̂)† U_μ(x−ν̂)† U_ν(x−ν̂) ; r2 écrivait un produit qui n'est pas une plaquette.
                let p2 = su2_hb::qmul(
                    su2_hb::qmul(ld(mu, s), su2_hb::qdag(ld(nu, smu_dn))),
                    su2_hb::qmul(su2_hb::qdag(ld(mu, sdn)), ld(nu, sdn)),
                );
                let mut xb = x;
                if xb[nu] == 0 {
                    xb[nu] = l - 1;
                } else {
                    xb[nu] -= 1;
                }
                let zb = if plan01 && xb[0] == l - 1 && xb[1] == l - 1 { -1.0 } else { 1.0 };
                somme_directe_lien += zb * p2[0];
            }

            assert!(
                (valeur_agrafe - somme_directe_lien).abs() < 1e-12,
                "lien ({mu},{s}) : qmul[0] = {valeur_agrafe}, somme = {somme_directe_lien}"
            );
        }
    }

    // (c) champ froid : 6V(1 - P_tw) = 2 L² par plan tordu
    let mut froid = CpuSu2::new(l, 1).unwrap();
    froid.init(false);
    let pt_froid = torsion::plaquette_tordue(&g, &froid.u, &t_01);
    let lhs = 6.0 * g.v as f64 * (1.0 - pt_froid);
    let expected = 2.0 * (l * l) as f64;
    assert!((lhs - expected).abs() < 1e-12, "champ froid : lhs = {lhs}, attendu {expected}");
    let t_01_23 = Torsion::analyser("01,23").unwrap();
    let pt_deux = torsion::plaquette_tordue(&g, &froid.u, &t_01_23);
    let action_deux = 6.0 * g.v as f64 * (1.0 - pt_deux);
    assert!((action_deux - 64.0).abs() < 1e-12, "champ froid, deux plans : {action_deux}, attendu 64");

    // Normalisation absolue par plan : 1 - 2L²/V = 0,875, sans facteur 1/6.
    for (t, attendus) in [
        (t_01, [0.875_f64, 1.0, 1.0, 1.0, 1.0, 1.0]),
        (t_01_23, [0.875_f64, 1.0, 1.0, 1.0, 1.0, 0.875]),
    ] {
        let mesures = torsion::plaquettes_par_plan(&g, &froid.u, &t);
        assert_bits_egaux(&mesures, &attendus, "normalisation des plans sur champ froid");
    }

    // (d) fond abélien uniforme de flux 2π/L² : action = action_fond_abelien_un_plan(4)
    let mut fond = vec![[1.0, 0.0, 0.0, 0.0]; 4 * g.v];
    let lf = l as f64;
    for x0 in 0..l {
        for x1 in 0..l {
            for x2 in 0..l {
                for x3 in 0..l {
                    let s = ((x0 * l + x1) * l + x2) * l + x3;
                    // U_1(x) = (cos(πx0/L²), 0, 0, sin(πx0/L²))
                    let phi = std::f64::consts::PI * x0 as f64 / (lf * lf);
                    fond[g.v + s] = [phi.cos(), 0.0, 0.0, phi.sin()];
                    // pour x0 = L-1, U_0(x) = (cos(πx1/L), 0, 0, -sin(πx1/L))
                    if x0 == l - 1 {
                        let theta = std::f64::consts::PI * x1 as f64 / lf;
                        fond[s] = [theta.cos(), 0.0, 0.0, -theta.sin()];
                    }
                }
            }
        }
    }
    let pt_fond = torsion::plaquette_tordue(&g, &fond, &t_01);
    let lhs_fond = 6.0 * g.v as f64 * (1.0 - pt_fond);
    let action_attendue = torsion::action_fond_abelien_un_plan(l);
    assert!((lhs_fond - action_attendue).abs() < 1e-10, "fond abélien : lhs = {lhs_fond}, attendu {action_attendue}");
    // L² déborderait en usize ; aucune géométrie ni allocation à cette taille.
    let grand_l = 1usize << (usize::BITS / 2);
    let limite = std::f64::consts::PI.powi(2) / 2.0;
    let action_grand_l = torsion::action_fond_abelien_un_plan(grand_l);
    assert!(action_grand_l.is_finite());
    assert!((action_grand_l - limite).abs() < 1e-14);

    // (e) configuration « mangeur de torsion » : action tordue nulle, action périodique = 2L² = 32
    let mut mangeur = vec![[1.0, 0.0, 0.0, 0.0]; 4 * g.v];
    for x0 in 0..l {
        for x1 in 0..l {
            for x2 in 0..l {
                for x3 in 0..l {
                    let s = ((x0 * l + x1) * l + x2) * l + x3;
                    if x0 == l - 1 {
                        mangeur[s] = [0.0, 1.0, 0.0, 0.0]; // iσ₁
                    }
                    if x1 == l - 1 {
                        mangeur[g.v + s] = [0.0, 0.0, 1.0, 0.0]; // iσ₂
                    }
                }
            }
        }
    }
    let pt_mangeur_tw = torsion::plaquette_tordue(&g, &mangeur, &t_01);
    let action_tw = 6.0 * g.v as f64 * (1.0 - pt_mangeur_tw);
    assert!(action_tw <= 1e-12, "action tordue du mangeur = {action_tw}, attendue nulle");

    let somme_directe_tw = somme_plaquettes_directe(&g, &mangeur, l - 1, l - 1);
    assert!((somme_directe_tw - 6.0 * g.v as f64 * pt_mangeur_tw).abs() < 1e-12, "incohérence plaquette_tordue/somme directe");

    // Action périodique (torsion triviale, donc z=+1 partout)
    let pt_mangeur_per = torsion::plaquette_tordue(&g, &mangeur, &Torsion::triviale());
    let action_per = 6.0 * g.v as f64 * (1.0 - pt_mangeur_per);
    assert!((action_per - 2.0 * (l * l) as f64).abs() < 1e-12, "action périodique du mangeur = {action_per}, attendue 32");
    let somme_directe_per = somme_plaquettes_directe(&g, &mangeur, usize::MAX, usize::MAX); // pile hors réseau => jamais -1
    assert!((somme_directe_per - 6.0 * g.v as f64 * pt_mangeur_per).abs() < 1e-12, "incohérence action périodique");
}

// ---------------------------------------------------------------------------------------------
// T3, T4, T5(b) : runs physiques partagés
// ---------------------------------------------------------------------------------------------

struct ResultatsBeta {
    beta: f64,
    delta_e: f64,
    sigma_delta_e: f64,
    // Différences et erreurs des blocs APPARIÉS, pour T3.
    delta_plans: [f64; 6],
    sigma_delta_plans: [f64; 6],
    // |P_rho| est pris après la moyenne spatiale, avant la moyenne de chaîne.
    moy_poly_tw: [f64; 4],
    moy_abs_poly_tw: [f64; 4],
    moy_abs_poly_per: [f64; 4],
    f_cumule: f64,
    sigma_f: f64,
    increment: f64,
    sigma_increment: f64,
}

/// Points β indépendants ; ΔE(0) = 0 exactement, sans variance de Haar à ajouter.
/// Les trapèzes voisins partagent ΔE_j : c_j = (h_j + h_{j+1})/2, c_k = h_k/2.
fn sigma_f_cumule(betas: &[f64], sigmas: &[f64]) -> f64 {
    assert_eq!(betas.len(), sigmas.len());
    betas.iter().zip(sigmas).enumerate().map(|(j, (&beta, &sigma))| {
        let precedent = if j == 0 { 0.0 } else { betas[j - 1] };
        let h = beta - precedent;
        let h_suivant = betas.get(j + 1).map_or(0.0, |suivant| suivant - beta);
        (((h + h_suivant) / 2.0) * sigma).powi(2)
    }).sum::<f64>().sqrt()
}

#[test]
fn t4_incertitude_trapezes_avec_points_partages() {
    assert_eq!(sigma_f_cumule(&[1.0], &[1.0]), 0.5);
    // Deux trapèzes unitaires : Var(E1 + E2/2) = 5/4, et non 3/4.
    assert!((sigma_f_cumule(&[1.0, 2.0], &[1.0, 1.0]).powi(2) - 1.25).abs() < 1e-14);
    // Grille non uniforme : poids (1,5 ; 2,5 ; 1,5).
    assert!((sigma_f_cumule(&[1.0, 3.0, 6.0], &[2.0, 3.0, 4.0]).powi(2) - 101.25).abs() < 1e-12);
}

fn run_physique(beta: f64, graine: u64) -> ResultatsBeta {
    let l = 4;
    let ntherm = 300;
    let nmeas = 4000;
    let bloc = 100;
    let n_blocs = nmeas / bloc;
    let t_01 = Torsion::analyser("01").unwrap();

    let mut per = Cpu64::nouveau(lot(l, beta, graine, Depart::Froid, 1)).unwrap();
    let mut tw = Cpu64::nouveau(lot(l, beta, graine, Depart::Froid, 1)).unwrap();
    tw.fixer_torsion(&t_01).unwrap();
    per.init().unwrap();
    tw.init().unwrap();

    per.allouer_historique(nmeas).unwrap();
    tw.allouer_historique(nmeas).unwrap();

    // Thermalisation
    for _ in 0..ntherm {
        per.sweep_hb().unwrap();
        per.sweep_or().unwrap();
        tw.sweep_hb().unwrap();
        tw.sweep_or().unwrap();
    }

    let geo = Geo::new(l);
    let mut blocs_delta_plans: Vec<[f64; 6]> = Vec::with_capacity(n_blocs);
    let mut delta_p: Vec<f64> = Vec::with_capacity(nmeas);
    let mut acc_delta_plans = [0.0f64; 6];
    let mut somme_poly_tw = [0.0f64; 4];
    let mut somme_abs_poly_tw = [0.0f64; 4];
    let mut somme_abs_poly_per = [0.0f64; 4];

    for idx in 0..nmeas {
        per.mesurer(idx).unwrap();
        tw.mesurer(idx).unwrap();

        let hp = per.historique().unwrap();
        let ht = tw.historique().unwrap();
        let o = idx * 5;
        let p_per = hp[o];
        let p_tw = ht[o];
        delta_p.push(p_per - p_tw);

        // plaquettes par plan sur l'état courant (avant sweep)
        let q_per = per.quaternions(0);
        let q_tw = tw.quaternions(0);
        let pp_per = torsion::plaquettes_par_plan(&geo, q_per, &Torsion::triviale());
        let pp_tw = torsion::plaquettes_par_plan(&geo, q_tw, &t_01);

        for (plan, acc) in acc_delta_plans.iter_mut().enumerate() {
            *acc += pp_per[plan] - pp_tw[plan];
        }
        // Chaque colonne est déjà la moyenne spatiale P_rho de la configuration.
        for (rho, somme) in somme_poly_tw.iter_mut().enumerate() {
            let p_tw = ht[o + 1 + rho];
            *somme += p_tw;
            somme_abs_poly_tw[rho] += p_tw.abs();
            somme_abs_poly_per[rho] += hp[o + 1 + rho].abs();
        }
        if (idx + 1).is_multiple_of(bloc) {
            blocs_delta_plans.push(acc_delta_plans.map(|somme| somme / bloc as f64));
            acc_delta_plans.fill(0.0);
        }

        // Étape suivante
        per.sweep_hb().unwrap();
        per.sweep_or().unwrap();
        tw.sweep_hb().unwrap();
        tw.sweep_or().unwrap();
    }
    assert_eq!(blocs_delta_plans.len(), n_blocs);

    let mut delta_plans = [0.0; 6];
    let mut sigma_delta_plans = [0.0; 6];
    for (plan, moyenne) in delta_plans.iter_mut().enumerate() {
        let blocs: Vec<f64> = blocs_delta_plans.iter().map(|b| b[plan]).collect();
        let (moy, err) = stats_moyennes_blocs(&blocs);
        *moyenne = moy;
        sigma_delta_plans[plan] = err;
    }

    // ΔE et son erreur par blocs sur la série appariée ΔP
    let (delta_p_moy, delta_p_err) = stats_blocs(&delta_p, bloc);
    let vol = 6.0 * (l as f64).powi(4);
    let delta_e = vol * delta_p_moy;
    let sigma_delta_e = vol * delta_p_err;
    let (_, erreur_200) = stats_blocs(&delta_p, 200);
    let ratio = (vol * erreur_200) / sigma_delta_e;
    assert!((0.5..=2.0).contains(&ratio),
        "T3/T4 β={beta} : blocs non décorrélés, σ_200/σ_100 = {ratio}");

    ResultatsBeta {
        beta,
        delta_e,
        sigma_delta_e,
        delta_plans,
        sigma_delta_plans,
        moy_poly_tw: somme_poly_tw.map(|somme| somme / nmeas as f64),
        moy_abs_poly_tw: somme_abs_poly_tw.map(|somme| somme / nmeas as f64),
        moy_abs_poly_per: somme_abs_poly_per.map(|somme| somme / nmeas as f64),
        f_cumule: 0.0,
        sigma_f: 0.0,
        increment: 0.0,
        sigma_increment: 0.0,
    }
}

#[test]
fn t3_t4_t5b_physique_torsion() {
    let n_runs = GRILLE_BETA.len();
    let mut resultats: Vec<ResultatsBeta> = Vec::with_capacity(n_runs);

    // Exécution des 20 runs (10 β × {périodique, tordu})
    eprintln!("Démarrage des runs T3/T4/T5(b) (20 simulations)…");
    for (k, &beta) in GRILLE_BETA.iter().enumerate() {
        let graine = 26110420u64 + k as u64;
        resultats.push(run_physique(beta, graine));
    }

    // Calcul de F par trapèzes sur {0} ∪ grille
    let mut prev_beta = 0.0;
    let mut prev_delta_e = 0.0;
    let mut prev_sigma = 0.0_f64;
    let mut f_cum = 0.0;
    let sigmas: Vec<f64> = resultats.iter().map(|r| r.sigma_delta_e).collect();
    for (idx, r) in resultats.iter_mut().enumerate() {
        let db = r.beta - prev_beta;
        let increment = (db / 2.0) * (prev_delta_e + r.delta_e);
        let sigma_inc = (db / 2.0) * (prev_sigma.powi(2) + r.sigma_delta_e.powi(2)).sqrt();
        f_cum += increment;
        r.f_cumule = f_cum;
        r.sigma_f = sigma_f_cumule(&GRILLE_BETA[..=idx], &sigmas[..=idx]);
        r.increment = increment;
        r.sigma_increment = sigma_inc;
        prev_beta = r.beta;
        prev_delta_e = r.delta_e;
        prev_sigma = r.sigma_delta_e;
    }

    // Impression de la table T4 (information)
    eprintln!("Table T4 (β, ΔE ± σ, F ± σ, β·ΔE) :");
    for r in &resultats {
        eprintln!(
            "β = {:.1} : ΔE = {:.3} ± {:.3} ; F = {:.3} ± {:.3} ; β·ΔE = {:.3}",
            r.beta, r.delta_e, r.sigma_delta_e, r.f_cumule, r.sigma_f, r.beta * r.delta_e
        );
    }

    // ---- T3 : couplage fort ----
    // β ∈ {0,5 ; 1 ; 1,5 ; 2} : |ΔE| < 3σ ; β ∈ {0,5 ; 1} : plans < 4σ apparié.
    let betas_t3 = [0.5, 1.0, 1.5, 2.0];
    for &beta in &betas_t3 {
        let r = resultats.iter().find(|r| (r.beta - beta).abs() < 1e-12).unwrap();
        assert!(
            r.delta_e.abs() < 3.0 * r.sigma_delta_e,
            "T3 β={beta} : |ΔE| = {} ≥ 3σ = {}",
            r.delta_e.abs(),
            3.0 * r.sigma_delta_e
        );
    }
    for &beta in &[0.5, 1.0] {
        let r = resultats.iter().find(|r| (r.beta - beta).abs() < 1e-12).unwrap();
        for (plan, (&difference, &sigma)) in r.delta_plans.iter().zip(&r.sigma_delta_plans).enumerate() {
            let diff = difference.abs();
            assert!(
                diff < 4.0 * sigma,
                "T3 β={beta} plan {plan} : |Δplan| = {diff:.3e} ≥ 4σ_apparié = {:.3e}",
                4.0 * sigma
            );
        }
    }

    // ---- T4 : énergie de flux ----
    for r in &resultats {
        assert!(r.delta_e > -3.0 * r.sigma_delta_e, "T4 β={} : ΔE = {} ≤ -3σ", r.beta, r.delta_e);
        assert!(r.increment > -3.0 * r.sigma_increment, "T4 incrément β={} : {} ≤ -3σ", r.beta, r.increment);
    }
    let r25 = resultats.iter().find(|r| (r.beta - 2.5).abs() < 1e-12).unwrap();
    assert!(r25.delta_e > 5.0 * r25.sigma_delta_e, "T4 β=2,5 : ΔE = {} σ={}", r25.delta_e, r25.sigma_delta_e);
    let r3 = resultats.iter().find(|r| (r.beta - 3.0).abs() < 1e-12).unwrap();
    assert!(r3.delta_e > 4.0 * r3.sigma_delta_e, "T4 β=3 : ΔE = {} σ={}", r3.delta_e, r3.sigma_delta_e);
    let r4 = resultats.iter().find(|r| (r.beta - 4.0).abs() < 1e-12).unwrap();
    assert!(r4.f_cumule > 5.0 * r4.sigma_f, "T4 F(4) = {} σ={}", r4.f_cumule, r4.sigma_f);
    let r8 = resultats.iter().find(|r| (r.beta - 8.0).abs() < 1e-12).unwrap();
    assert!(r8.f_cumule > 5.0 * r8.sigma_f, "T4 F(8) = {} σ={}", r8.f_cumule, r8.sigma_f);

    // ---- T5(b) : Polyakov tordus à β = 6 et 8 ----
    for &beta in &[6.0, 8.0] {
        let r = resultats.iter().find(|r| (r.beta - beta).abs() < 1e-12).unwrap();
        let (p0, p1) = (r.moy_poly_tw[0], r.moy_poly_tw[1]);
        assert!(
            p0.abs() < 0.05 && p1.abs() < 0.05,
            "T5(b) β={beta} : |⟨P0⟩| = {} ou |⟨P1⟩| = {} ≥ 0,05 (tordu ⇒ ⟨P0⟩ = ⟨P1⟩ = 0 en espérance)",
            p0.abs(),
            p1.abs()
        );
        for (rho, (&tw, &per)) in r.moy_abs_poly_tw[..2].iter().zip(&r.moy_abs_poly_per[..2]).enumerate() {
            assert!(tw < 0.2, "T5(b) β={beta} : ⟨|P{rho}|⟩_tw = {tw} ≥ 0,2");
            assert!(per > 0.4, "T5(b) β={beta} : ⟨|P{rho}|⟩_per = {per} ≤ 0,4");
        }
        for (rho, &tw) in r.moy_abs_poly_tw.iter().enumerate().skip(2) {
            assert!(tw > 0.5, "T5(b) β={beta} : ⟨|P{rho}|⟩_tw = {tw} ≤ 0,5");
        }
        eprintln!("T5(b) β={beta} : ⟨P⟩_tw={:?}, ⟨|P|⟩_tw={:?}, ⟨|P|⟩_per={:?}",
            r.moy_poly_tw, r.moy_abs_poly_tw, r.moy_abs_poly_per);
    }
}

// ---------------------------------------------------------------------------------------------
// T5(a) : identité exacte de déplacement de pile
// ---------------------------------------------------------------------------------------------

#[test]
fn t5a_identite_deplacement_pile() {
    let l = 4;
    let g = Geo::new(l);
    // Régression du wrap : le lien négatif est sur un AUTRE cercle (x0 = 1).
    let mut temoin = vec![[1.0, 0.0, 0.0, 0.0]; 4 * g.v];
    temoin[g.v + g.st[0]] = [-1.0, 0.0, 0.0, 0.0];
    assert_eq!(polyakov_brut(&g, &temoin, 1, &[0, 3, 0, 0]), [1.0, 0.0, 0.0, 0.0]);
    let graine = 26110499u64;
    let mut rng = CpuSu2::new(l, graine).unwrap();
    rng.init(true);
    let champ_original = rng.u.clone();

    // Action tordue avec pile en (L-1, L-1)
    let action_avant = somme_plaquettes_directe(&g, &champ_original, l - 1, l - 1);

    // Changement de variables : multiplier U_1(y) par -1 pour y0 = L-1, y1 = L-1
    let mut champ_transforme = champ_original.clone();
    for x0 in 0..l {
        for x1 in 0..l {
            for x2 in 0..l {
                for x3 in 0..l {
                    if x0 == l - 1 && x1 == l - 1 {
                        let s = ((x0 * l + x1) * l + x2) * l + x3;
                        champ_transforme[g.v + s] = su2_hb::qscale(champ_transforme[g.v + s], -1.0);
                    }
                }
            }
        }
    }

    // Action après transformation, avec pile déplacée en y0 = L-2 (spec §3 T5(a))
    let action_apres = somme_plaquettes_directe(&g, &champ_transforme, l - 2, l - 1);
    assert!(
        (action_avant - action_apres).abs() < 1e-12,
        "action avant = {action_avant}, action après = {action_apres}"
    );

    // Vérification du signe des boucles de Polyakov brutes P1(x) pour x0 = L-1
    for x0 in 0..l {
        for x1 in 0..l {
            for x2 in 0..l {
                for x3 in 0..l {
                    let x = [x0, x1, x2, x3];
                    let p_avant = polyakov_brut(&g, &champ_original, 1, &x);
                    let p_apres = polyakov_brut(&g, &champ_transforme, 1, &x);
                    if x0 == l - 1 {
                        let attendu = su2_hb::qscale(p_avant, -1.0);
                        for i in 0..4 {
                            assert!(
                                (p_apres[i] - attendu[i]).abs() < 1e-15,
                                "x0=L-1, x={x:?} : signe de P1 non inversé"
                            );
                        }
                    } else {
                        for i in 0..4 {
                            assert!(
                                (p_apres[i] - p_avant[i]).abs() < 1e-15,
                                "x0≠L-1, x={x:?} : P1 modifié à tort"
                            );
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// T6 : reprise avec torsion
// ---------------------------------------------------------------------------------------------

#[test]
fn t6_reprise_avec_torsion() {
    let dir_interrompu = rep("t6_interrompu");
    let dir_complet = rep("t6_complet");

    let args_interrompu = [
        "gen", "--backend", "cpu", "--L", "4", "--beta", "2.5", "--graine", "26110430",
        "--torsion", "01", "--therm", "20", "--nmeas", "40", "--nor", "1", "--dump-every", "10",
        "--format", "q64", "--reprise-every", "10", "--arret-apres", "34",
        "--outdir", dir_interrompu.to_str().unwrap(),
    ];
    let out = executer_cli(&args_interrompu);
    assert_rc(&out, 0, "run interrompu avec torsion");
    assert!(!dir_interrompu.join("COMPLETE").exists());
    let etat = kvu::su2::reprise::lire_etat_seul(&dir_interrompu).unwrap();
    assert_eq!(etat.pas_faits, 34, "checkpoint valide à l'arrêt demandé");

    // Reprise sans --torsion
    // --backend cpu explicite est requis pour toute torsion non triviale.
    let args_reprise = ["gen", "--backend", "cpu", "--reprendre", dir_interrompu.to_str().unwrap()];
    let out = executer_cli(&args_reprise);
    assert_rc(&out, 0, "reprise sans torsion");

    // Run ininterrompu de référence
    let args_complet = [
        "gen", "--backend", "cpu", "--L", "4", "--beta", "2.5", "--graine", "26110430",
        "--torsion", "01", "--therm", "20", "--nmeas", "40", "--nor", "1", "--dump-every", "10",
        "--format", "q64",
        "--outdir", dir_complet.to_str().unwrap(),
    ];
    let out = executer_cli(&args_complet);
    assert_rc(&out, 0, "run complet de référence");

    // Comparer CSV et dumps
    for fichier in ["measurements.csv", "therm.csv"] {
        assert_octets_egaux(&dir_interrompu.join(fichier), &dir_complet.join(fichier), fichier);
    }
    let da = noms_dumps(&dir_interrompu);
    let db = noms_dumps(&dir_complet);
    let attendus = ["cfg_000010.npy", "cfg_000020.npy", "cfg_000030.npy", "cfg_000040.npy"];
    assert_eq!(da, attendus, "quatre dumps attendus après reprise");
    assert_eq!(db, attendus, "quatre dumps attendus pour la référence");
    for n in da {
        assert_octets_egaux(&dir_interrompu.join(&n), &dir_complet.join(&n), &n);
    }

    let torsion_attendue = serde_json::json!({
        "plans": ["01"],
        "pile": "coins (L-1, L-1) de chaque plan tordu (convention kvu M4)",
        "facteur": -1
    });
    for dir in [&dir_interrompu, &dir_complet] {
        assert!(dir.join("COMPLETE").exists());
        let torsion_json: serde_json::Value = serde_json::from_slice(&octets(&dir.join("torsion.json"))).unwrap();
        assert_eq!(torsion_json, torsion_attendue, "contenu complet de torsion.json");
        let manifeste: serde_json::Value = serde_json::from_slice(&octets(&dir.join("manifest.json"))).unwrap();
        assert_eq!(manifeste["torsion"], torsion_attendue, "torsion complète du manifeste");
        assert_eq!(manifeste["action"],
            "beta sum(1 - z_p ReTr(U_p)/2), z_p = -1 sur la pile de torsion (torsion.json)");
        let polyakov = &manifeste["polyakov"];
        assert!(polyakov["valeurs"].as_str().unwrap().contains("brutes"));
        assert!(polyakov["pile"].as_str().unwrap().contains("position de la pile"));
        assert!(polyakov["symetrie"].as_str().unwrap().contains("G^L P_nu(x) = -P_nu(x)"));
    }

    let _ = fs::remove_dir_all(&dir_interrompu);
    let _ = fs::remove_dir_all(&dir_complet);
}

// ---------------------------------------------------------------------------------------------
// T7 : refus CLI (code 2, aucun dossier créé)
// ---------------------------------------------------------------------------------------------

#[test]
fn t7_refus_cli() {
    let base = rep("t7_base");
    let cas: Vec<(Vec<&str>, &str)> = vec![
        (vec!["gen", "--backend", "cpu", "--L", "4", "--torsion", "00", "--outdir"], "torsion 00"),
        (vec!["gen", "--backend", "cpu", "--L", "4", "--torsion", "04", "--outdir"], "torsion 04"),
        (vec!["gen", "--backend", "cpu", "--L", "4", "--torsion", "01,10", "--outdir"], "doublon"),
        (vec!["gen", "--backend", "cpu", "--L", "4", "--torsion", "", "--outdir"], "torsion vide"),
        (vec!["gen", "--backend", "hip", "--L", "4", "--torsion", "01", "--outdir"], "backend hip"),
        (vec!["gen", "--backend", "cuda", "--L", "4", "--torsion", "01", "--outdir"], "backend cuda"),
        (vec!["gen", "--backend", "auto", "--L", "4", "--torsion", "01", "--outdir"], "backend auto"),
        (vec!["gen", "--backend", "cpu", "--precision", "f32", "--L", "4", "--torsion", "01", "--outdir"], "précision f32"),
        (vec!["gen", "--backend", "cpu", "--L", "4", "--torsion", "01", "--bord", "--bc", "cold", "--outdir"], "bord"),
        (vec!["gen", "--backend", "cpu", "--L", "4", "--torsion", "01", "--mesures", "wilson:R2xT2:5", "--outdir"], "mesures en ligne"),
        (vec!["gen", "--backend", "cpu", "--L", "4", "--beta", "0", "--torsion", "01", "--outdir"], "beta nul"),
        (vec!["gen", "--backend", "cpu", "--L", "4", "--beta", "2.5,0", "--torsion", "01", "--outdir"], "beta nul dans un lot"),
    ];

    // Exécution des refus : chaque outdir est unique et ne doit pas exister après l'appel
    for (mut args, description) in cas {
        let dir_unique = base.join(format!("refus_{}", description.replace(' ', "_")));
        args.push(dir_unique.to_str().unwrap());
        let out = executer_cli(&args);
        assert_refus_torsion(&out, &format!("refus {description}"), &[]);
        if description.starts_with("backend") {
            assert_diagnostic_torsion(&String::from_utf8_lossy(&out.stderr), &["gpu", "cpu"]);
        }
        assert!(!dir_unique.exists(), "le dossier {} a été créé pour {description}", dir_unique.display());
    }

    // Run réellement interrompu : un refus dû à COMPLETE ne doit pas suffire.
    let dir_rep_base = base.join("reprendre_base");
    let args_creer = [
        "gen", "--backend", "cpu", "--L", "4", "--beta", "2.5", "--graine", "26110500",
        "--torsion", "01", "--therm", "1", "--nmeas", "4", "--nor", "1",
        "--reprise-every", "1", "--arret-apres", "2", "--dump-every", "1", "--format", "q64",
        "--outdir", dir_rep_base.to_str().unwrap(),
    ];
    let out = executer_cli(&args_creer);
    assert_rc(&out, 0, "création run de base pour reprise");
    assert!(!dir_rep_base.join("COMPLETE").exists());
    let etat = kvu::su2::reprise::lire_etat_seul(&dir_rep_base).unwrap();
    assert_eq!(etat.pas_faits, 2, "checkpoint valide, champs et sha256 vérifiés");
    let avant = instantane(&dir_rep_base);

    let args_reprise_diff = [
        "gen", "--backend", "cpu", "--reprendre", dir_rep_base.to_str().unwrap(), "--torsion", "02",
    ];
    let out = executer_cli(&args_reprise_diff);
    assert_refus_torsion(&out, "reprendre avec torsion différente", &["diff", "01", "02"]);
    assert_eq!(instantane(&dir_rep_base), avant, "fichiers altérés après refus de torsion différente");

    // Sans --torsion, les options CLI doivent être validées contre la torsion héritée.
    for options in [
        vec!["--backend", "cpu", "--precision", "f32"],
        vec!["--backend", "cpu", "--bord", "--bc", "cold"],
        vec!["--backend", "cpu", "--mesures", "wilson:R2xT2:5"],
        vec!["--backend", "hip"],
        vec!["--backend", "cuda"],
        vec!["--backend", "auto"],
        vec!["--backend", "cpu", "--torsion", "aucune"],
    ] {
        let mut args = vec!["gen", "--reprendre", dir_rep_base.to_str().unwrap()];
        args.extend_from_slice(&options);
        let out = executer_cli(&args);
        assert_refus_torsion(&out, &format!("reprise avec {options:?}"), &[]);
        if options[1] == "hip" || options[1] == "cuda" || options[1] == "auto" {
            assert_diagnostic_torsion(&String::from_utf8_lossy(&out.stderr), &["gpu", "cpu"]);
        }
        assert_eq!(instantane(&dir_rep_base), avant, "fichiers altérés après refus CLI {options:?}");
    }

    // Même obligation par l'API publique et les deux points d'entrée historiques.
    let cfg = run::charger_config(&dir_rep_base).unwrap();
    for texte in ["02", "aucune"] {
        let opts = OptionsRun { torsion: Torsion::analyser(texte).unwrap() };
        assert_config_torsion(run::executer_options(None, &cfg, &opts, &mut [], true, None), &["diff", "01", texte]);
        assert_eq!(instantane(&dir_rep_base), avant, "fichiers altérés après refus API {texte}");
    }
    assert_config_torsion(run::executer(None, &cfg, &mut [], true), &["diff", "01", "aucune"]);
    assert_eq!(instantane(&dir_rep_base), avant, "fichiers altérés après executer");
    assert_config_torsion(run::executer_jusqua(None, &cfg, &mut [], true, None), &["diff", "01", "aucune"]);
    assert_eq!(instantane(&dir_rep_base), avant, "fichiers altérés après executer_jusqua");

    let opts = run::charger_options(&dir_rep_base).unwrap();
    let mut cfg_f32 = cfg.clone();
    cfg_f32.lot.precision = Precision::F32;
    assert_config_torsion(run::executer_options(None, &cfg_f32, &opts, &mut [], true, None), &["f32"]);
    assert_eq!(instantane(&dir_rep_base), avant, "fichiers altérés après refus API f32");
    let mut cfg_zero = cfg.clone();
    cfg_zero.lot.chaines[0].beta = 0.0;
    assert_config_torsion(run::executer_options(None, &cfg_zero, &opts, &mut [], true, None), &["β = 0"]);
    assert_eq!(instantane(&dir_rep_base), avant, "fichiers altérés après refus API β=0");

    let mut cpu_zero = Cpu64::nouveau(lot(4, 0.0, 1, Depart::Froid, 1)).unwrap();
    assert_config_torsion(cpu_zero.fixer_torsion(&opts.torsion), &["β = 0"]);

    let _ = fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------------------------
// T8 : unitaires de Torsion
// ---------------------------------------------------------------------------------------------

#[test]
fn t8_unitaires_torsion() {
    // analyser/texte aller-retour
    let t = Torsion::analyser("01,23").unwrap();
    assert_eq!(t.texte(), "01,23");
    let t = Torsion::analyser("10").unwrap();
    assert_eq!(t.texte(), "01");
    let t = Torsion::analyser("aucune").unwrap();
    assert_eq!(t.texte(), "aucune");

    // Symétrie de tordu
    let t = Torsion::analyser("01").unwrap();
    assert!(t.tordu(0, 1));
    assert!(t.tordu(1, 0));
    assert!(!t.tordu(0, 2));
    assert!(!t.tordu(2, 2));
    for (mu, nu) in [(0, 4), (4, 0), (4, 5), (usize::MAX, 1), (1, usize::MAX)] {
        assert!(!t.tordu(mu, nu), "paire hors domaine ({mu}, {nu})");
    }
    for invalide in ["é", "0a", "01,", ",01", "0", "012", "44"] {
        assert!(matches!(Torsion::analyser(invalide), Err(Erreur::Config(_))));
    }

    // z : -1 exactement sur les L² coins de la pile, +1 ailleurs, pour chaque plan
    for &(mu, nu) in &Torsion::PLANS {
        // on active ce plan via l'analyseur (chaîne « mu nu »)
        let plan_str = format!("{}{}", mu, nu);
        let t = Torsion::analyser(&plan_str).unwrap();
        let l = 4;
        for x0 in 0..l {
            for x1 in 0..l {
                for x2 in 0..l {
                    for x3 in 0..l {
                        let y = [x0, x1, x2, x3];
                        let z = t.z(&y, mu, nu, l);
                        let attendu = if y[mu] == l - 1 && y[nu] == l - 1 { -1.0 } else { 1.0 };
                        assert_eq!(z, attendu, "z incorrect pour plan ({mu},{nu}), y={y:?}");
                    }
                }
            }
        }
        // plan non tordu
        let autre = if mu == 0 && nu == 1 { (0, 2) } else { (0, 1) };
        let y_pile = [l - 1, l - 1, 0, 0];
        let z_autre = t.z(&y_pile, autre.0, autre.1, l);
        assert_eq!(z_autre, 1.0, "z devrait être +1 pour un plan non tordu");
    }

    // Sérialisation JSON
    let t = Torsion::analyser("01,23").unwrap();
    let j = serde_json::to_value(t).unwrap();
    let t2: Torsion = serde_json::from_value(j).unwrap();
    assert_eq!(t, t2);
}
