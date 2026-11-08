//! Sous-commande `su2` (gen, test-hb, selftest) : tests d'intégration et passerelle de ligne de commande.
//!
//! Tant que le binaire `kvu` n'est pas câblé (T9), `shim_cli` fait office de `kvu su2 …` : il lit les arguments
//! dans KVU_SU2_ARGV (séparés par le caractère U+001F), exécute `su2::cli::executer` et sort avec le code rendu.
//! Les scripts de validation (validation/su2/) l'appellent ainsi :
//! `KVU_SU2_ARGV=$'gen\x1f--L\x1f8…' <binaire de test> --ignored --exact shim_cli --nocapture`.

use clap::Command;
use kvu::commun::Erreur;
use kvu::su2::cli;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as Proc;

fn executer(args: &[&str]) -> Result<i32, Erreur> {
    let mut argv = vec!["kvu", "su2"];
    argv.extend_from_slice(args);
    let m = Command::new("kvu")
        .subcommands(cli::commandes())
        .try_get_matches_from(argv)
        .map_err(|e| Erreur::Config(e.to_string()))?;
    let (nom, sm) = m.subcommand().ok_or_else(|| Erreur::Config("su2".into()))?;
    cli::executer(nom, sm)
}

/// Passerelle : exécute `su2 <KVU_SU2_ARGV>` et sort avec le code de sortie stable de `commun::code`.
#[test]
#[ignore = "passerelle CLI (utilisée par les scripts de validation)"]
fn shim_cli() {
    let Ok(v) = std::env::var("KVU_SU2_ARGV") else {
        println!("# shim_cli : KVU_SU2_ARGV absent");
        return;
    };
    let args: Vec<&str> = v.split('\u{1f}').collect();
    let code = match executer(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("erreur : {e}");
            e.code_sortie()
        }
    };
    std::process::exit(code);
}

fn shim(args: &[&str], env: &[(&str, &str)]) -> (i32, String, String) {
    let sortie = Proc::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "shim_cli", "--nocapture", "--test-threads=1"])
        .env("KVU_SU2_ARGV", args.join("\u{1f}"))
        .envs(env.iter().copied())
        .output()
        .unwrap();
    (sortie.status.code().unwrap_or(-1), String::from_utf8_lossy(&sortie.stdout).into_owned(), String::from_utf8_lossy(&sortie.stderr).into_owned())
}

fn refs(v: &[String]) -> Vec<&str> {
    v.iter().map(|s| s.as_str()).collect()
}

fn rep(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("su2_cli").join(format!("{nom}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.parent().unwrap()).unwrap();
    d
}

#[test]
fn selftest_par_la_ligne_de_commande() {
    let (rc, out, _) = shim(&["selftest"], &[]);
    assert_eq!(rc, 0, "{out}");
    assert!(out.contains("SELFTEST philox KAT 0 : PASS"));
    assert!(out.contains("SELFTEST philox KAT 2 : PASS"));
    assert!(out.contains("SELFTEST u01 bornes : PASS"));
    assert!(out.contains("SELFTEST OK"));
}

#[test]
fn gen_sur_cpu_ecrit_les_fichiers_de_su2_hip() {
    let d = rep("gen");
    let (rc, out, err) = shim(
        &[
            "gen", "--backend", "cpu", "--L", "4", "--beta", "2.3", "--seed", "926193012", "--start", "hot", "--therm", "6",
            "--nmeas", "4", "--skip", "1", "--nor", "4", "--dump-every", "2", "--dump-initial", "--outdir", d.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(rc, 0, "stdout : {out}\nstderr : {err}");
    assert!(out.contains("PERF L=4"), "{out}");
    for f in ["therm.csv", "measurements.csv", "manifest.json", "COMPLETE", "cfg_000000.npy", "cfg_000002.npy", "cfg_000004.npy"] {
        assert!(d.join(f).exists(), "{f} absent");
    }
    // même sortie que la boucle de test (golden indirect) : les 5 premières colonnes de l'indice 0
    assert!(fs::read_to_string(d.join("measurements.csv")).unwrap().lines().nth(1).unwrap().starts_with("0,6,initial,"));
    // refus d'écraser : rc de configuration
    let (rc2, _, _) = shim(&["gen", "--backend", "cpu", "--L", "4", "--outdir", d.to_str().unwrap()], &[]);
    assert_ne!(rc2, 0);
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn arret_et_reprise_par_la_ligne_de_commande() {
    let (da, db) = (rep("cli_a"), rep("cli_b"));
    let base = |d: &Path| {
        vec![
            "gen".to_string(), "--backend".into(), "cpu".into(), "--L".into(), "4".into(), "--beta".into(), "2.3,2.8".into(),
            "--graine".into(), "10,20".into(), "--therm".into(), "12".into(), "--nmeas".into(), "8".into(), "--skip".into(),
            "1".into(), "--dump-every".into(), "4".into(), "--reprise-every".into(), "5".into(), "--outdir".into(),
            d.to_str().unwrap().to_string(),
        ]
    };
    let a = base(&da);
    assert_eq!(shim(&refs(&a), &[]).0, 0);
    let mut b = base(&db);
    b.extend(["--arret-apres".to_string(), "10".to_string()]);
    let (rc, _, err) = shim(&refs(&b), &[]);
    assert_eq!(rc, 0, "{err}");
    assert!(!db.join("COMPLETE").exists());
    let (rc, _, err) = shim(&["gen", "--backend", "cpu", "--reprendre", db.to_str().unwrap()], &[]);
    assert_eq!(rc, 0, "{err}");
    assert!(db.join("COMPLETE").exists());
    for c in ["c000", "c001"] {
        for f in ["therm.csv", "measurements.csv", "cfg_000004.npy", "cfg_000008.npy"] {
            assert_eq!(fs::read(da.join(c).join(f)).unwrap(), fs::read(db.join(c).join(f)).unwrap(), "{c}/{f}");
        }
    }
    fs::remove_dir_all(&da).unwrap();
    fs::remove_dir_all(&db).unwrap();
}

#[test]
fn test_hb_sur_cpu_imprime_la_ligne_testhb() {
    let (rc, out, err) = shim(&["test-hb", "5", "200000", "--backend", "cpu", "--graine", "9004"], &[]);
    assert_eq!(rc, 0, "{err}");
    // libtest préfixe la première ligne par « test shim_cli ... » : on cherche la ligne à partir de « TESTHB »
    let l = out.lines().find_map(|l| l.find("TESTHB ").map(|i| &l[i..])).unwrap_or_else(|| panic!("pas de ligne TESTHB : {out}"));
    let kv: std::collections::HashMap<&str, &str> = l.split_whitespace().skip(1).filter_map(|t| t.split_once('=')).collect();
    assert_eq!(kv["alpha"], "5");
    assert_eq!(kv["n"], "200000");
    assert_eq!(kv["plafond"], "0");
    let (m1, m2): (f64, f64) = (kv["m1"].parse().unwrap(), kv["m2"].parse().unwrap());
    // moments exacts à α = 5 (quadrature de Gauss-Tchebychev, n = 4000) : E[x] = 0,719341, E[x²] = 0,568396
    assert!((m1 - 0.7193405813643129).abs() < 0.01 && (m2 - 0.5683956511814121).abs() < 0.01, "{l}");
    assert!(kv["min"].parse::<f64>().unwrap() >= -1.0 && kv["max"].parse::<f64>().unwrap() <= 1.0);
}

#[test]
fn erreurs_d_usage_et_backend_force_absent() {
    // usage : α négatif
    let (rc, _, _) = shim(&["test-hb", "-1", "10", "--backend", "cpu"], &[]);
    assert_ne!(rc, 0);
    // F32 sans GPU : erreur de configuration (code 2), jamais de repli silencieux
    let d = rep("f32cpu");
    let (rc, _, err) = shim(&["gen", "--backend", "cpu", "--precision", "f32", "--L", "4", "--outdir", d.to_str().unwrap()], &[]);
    assert_eq!(rc, 2, "{err}");
    assert!(!d.exists(), "rien n'est écrit pour une configuration refusée");
    // sous-commande inconnue
    assert!(executer(&["fantaisie"]).is_err());
}

#[test]
fn la_garde_ssd_refuse_avant_toute_ecriture() {
    // SSD-1 : un seuil au-dessus du libre refuse AVANT tout octet écrit, rc 7, aucun COMPLETE.
    let parent = rep("ssd");
    let libre = kvu::stockage::GardeSsd::libre_go(parent.parent().unwrap()).unwrap();
    let d = parent.join("run");
    // projection d'un run c16 L = 64 × 20 dumps (≈ 86 Go) refusée dès que libre − 86 < seuil
    let seuil = format!("{}", libre - 10.0);
    let (rc, _, err) = shim(
        &["gen", "--backend", "cpu", "--L", "64", "--therm", "0", "--nmeas", "20", "--dump-every", "1", "--outdir", d.to_str().unwrap()],
        &[("KEVINOTRON_SSD_MIN_GO", &seuil)],
    );
    assert_eq!(rc, 7, "stderr : {err}");
    assert!(err.contains("garde SSD"), "{err}");
    assert!(!d.exists(), "rien ne doit être créé");
    // un seuil absurde refuse même un petit run
    let (rc, _, _) = shim(
        &["gen", "--backend", "cpu", "--L", "4", "--therm", "1", "--nmeas", "1", "--outdir", d.to_str().unwrap()],
        &[("KEVINOTRON_SSD_MIN_GO", "1000000000")],
    );
    assert_eq!(rc, 7);
    assert!(!d.exists());
}
