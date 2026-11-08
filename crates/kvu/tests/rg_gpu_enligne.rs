//! Mesure en ligne `bloc:3` de `su2 gen` (WP7) : grammaire, configurations grossières en ligne = `rg::bloc` sur les dumps du même run (CPU, sans GPU),
//! run sans mesure inchangé à l'octet, manifeste kvu-regen/1 (`parametres.mesures`). La version GPU est `#[ignore = "GPU"]`
//! (sous le verrou : `flock -w 7200 /tmp/kevinotron_gpu.lock cargo test --release --test rg_gpu_enligne -- --ignored`).

use kvu::rg::bloc::{bloquer, OptionsBloc};
use kvu::rg::geometrie::Geometrie;
use kvu::stockage::npy::lire_champ;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn kvu(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kvu")).args(args).output().expect("lancement de kvu")
}

fn dossier(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("rg_enligne_{nom}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn gen(backend: &str, sortie: &Path, mesures: Option<&str>) -> Output {
    let mut a = vec![
        "su2", "gen", "--backend", backend, "--L", "6", "--beta", "4", "--graine", "26102607001", "--therm", "6", "--nmeas", "4", "--skip", "2", "--dump-every", "1",
        "--format", "q64", "--outdir",
    ];
    let s = sortie.to_str().unwrap().to_string();
    a.push(&s);
    if let Some(m) = mesures {
        a.push("--mesures");
        a.push(m);
    }
    kvu(&a)
}

fn octets(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{} : {e}", p.display()))
}

fn en_ligne_egal_hors_ligne(backend: &str, seuil: f64) {
    let d = dossier(backend);
    let avec = d.join("avec");
    let sans = d.join("sans");
    let o = gen(backend, &avec, Some("bloc:3:1"));
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    let o = gen(backend, &sans, None);
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    // Le run sans mesure est inchangé à l'octet : mêmes dumps, mêmes CSV.
    let mut n_dumps = 0;
    for f in std::fs::read_dir(&sans).unwrap().map(|x| x.unwrap().file_name().into_string().unwrap()) {
        if f.starts_with("cfg_") || f == "measurements.csv" || f == "therm.csv" {
            assert_eq!(octets(&avec.join(&f)), octets(&sans.join(&f)), "{f} diffère avec --mesures bloc:3");
            n_dumps += usize::from(f.starts_with("cfg_"));
        }
    }
    assert!(n_dumps >= 4, "dumps attendus : {n_dumps}");
    // Configurations grossières en ligne = rg::bloc hors ligne sur les dumps (m ↔ pas = therm + m·skip).
    let mut compares = 0;
    for m in 1..=4u64 {
        let pas = 6 + 2 * m;
        let en_ligne = lire_champ(&avec.join(format!("bloc3_cfg_{pas:08}.npy"))).unwrap();
        let fin = lire_champ(&avec.join(format!("cfg_{m:06}.npy"))).unwrap();
        let hors = bloquer(&fin, &OptionsBloc { geometrie: Geometrie::TroisCentre, fils: 2, angle_so3: false }).unwrap();
        assert_eq!(en_ligne.l, 2);
        let ecart = en_ligne.q.iter().zip(&hors.grossier.q).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(ecart <= seuil, "mesure {m} : écart {ecart:e} > {seuil:e}");
        compares += 1;
    }
    assert_eq!(compares, 4);
    let csv = String::from_utf8(octets(&avec.join("bloc3.csv"))).unwrap();
    assert_eq!(csv.lines().count(), 1 + 4, "{csv}");
    assert!(csv.starts_with("pas,l_grossier,n_boucles,fraction_hors,theta_max,n_ambigues,plaq_grossiere,sha256_cfg"));
    let hist = String::from_utf8(octets(&avec.join("bloc3_hist.csv"))).unwrap();
    assert_eq!(hist.lines().count(), 5);
    assert_eq!(hist.lines().nth(1).unwrap().split(',').count(), 65);
    // Manifeste kvu-regen/1 : valide, décrit les sorties, porte parametres.mesures.
    let j: serde_json::Value = serde_json::from_slice(&octets(&avec.join("regen_bloc3.json"))).unwrap();
    assert_eq!(j["schema"], "kvu-regen/1");
    assert_eq!(j["parametres"]["mesures"], "bloc:3:1");
    assert_eq!(j["statut"], "COMPLETE");
    let noms: Vec<&str> = j["fichiers"].as_array().unwrap().iter().map(|f| f["nom"].as_str().unwrap()).collect();
    for attendu in ["bloc3.csv", "bloc3_hist.csv", "bloc3_cfg_00000008.npy", "cfg_000001.npy"] {
        assert!(noms.contains(&attendu), "{attendu} absent de {noms:?}");
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn en_ligne_cpu_egal_hors_ligne_et_run_sans_mesure_inchange() {
    en_ligne_egal_hors_ligne("cpu", 0.0);
}

#[test]
#[ignore = "GPU"]
fn en_ligne_gpu_egal_hors_ligne_a_1e_12() {
    let backend = if cfg!(feature = "cuda") { "cuda" } else { "hip" };
    en_ligne_egal_hors_ligne(backend, 1e-12);
}

#[test]
fn grammaire_refusee() {
    let d = dossier("grammaire");
    for spec in ["bloc:2:1", "bloc:3", "bloc:x:1", "bloc:3:z", "bloc:3:1:2", "bloc:3:1,bloc:3:2"] {
        let o = gen("cpu", &d.join(spec.replace([':', ','], "_")), Some(spec));
        assert_eq!(o.status.code().map(|c| c != 0), Some(true), "{spec} aurait dû être refusé");
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn l_non_multiple_de_3_est_refuse_sans_ecrire_de_coarse() {
    let d = dossier("l4");
    let sortie = d.join("run");
    let o = kvu(&["su2", "gen", "--backend", "cpu", "--L", "8", "--beta", "4", "--graine", "1", "--therm", "2", "--nmeas", "1", "--skip", "1", "--mesures", "bloc:3:1", "--outdir", sortie.to_str().unwrap()]);
    assert_ne!(o.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&o.stderr).contains("multiple de 3"), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(!sortie.join("regen_bloc3.json").exists());
    let _ = std::fs::remove_dir_all(&d);
}
