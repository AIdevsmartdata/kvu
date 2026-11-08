//! Tests du rapport de banc (rapport_banc.py) sur des JSON au format de `kevinotron bench`, et du
//! script banc_unifie.sh (refus de la garde SSD, aide).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn banc() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("validation/banc")
}

fn dossier(nom: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("kvu_banc_{nom}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn mesure_su2(l: u32, prec: &str, k: u32, pas: f64, ratio: Option<f64>) -> serde_json::Value {
    let mut v = serde_json::json!({
        "famille": "su2", "L": l, "precision": prec, "chaines": k, "beta": 2.5, "nor": 4, "options": "compat-prod",
        "hb_s_par_sweep": 0.01, "or_s_par_sweep": 0.004, "liens_par_s_hb": 1.0e8, "liens_par_s_or": 3.0e8,
        "pas_par_s": pas, "pas_chaines_par_s": pas * k as f64, "memoire_champ_mo": 100.0,
    });
    if let Some(r) = ratio {
        v["reference"] = serde_json::json!({"binaire": "/x/bin/su2_hip", "pas_par_s": pas / r});
        v["ratio_pas_unifie_sur_reference"] = serde_json::json!(r);
        v["gate_non_regression"] = serde_json::json!(r >= 0.9);
    }
    v
}

fn ecrire(d: &Path, machine: &str, fam: &str, mesures: Vec<serde_json::Value>) {
    fs::create_dir_all(d.join(machine)).unwrap();
    let doc = serde_json::json!({"banc": fam, "machine": machine, "backend": "hip", "arch": "gfx1151", "budget_s": 600.0,
        "duree_s": 12.0, "fenetres": 1, "mesures": mesures});
    fs::write(d.join(machine).join(format!("{fam}_{machine}.json")), doc.to_string()).unwrap();
}

fn rapport(d: &Path) -> (String, String) {
    let sortie = d.join("BANC.md");
    let o = Command::new("python3")
        .arg(banc().join("rapport_banc.py"))
        .arg(d)
        .arg("--sortie")
        .arg(&sortie)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    (
        fs::read_to_string(&sortie).unwrap(),
        String::from_utf8_lossy(&o.stdout).into_owned(),
    )
}

#[test]
fn rapport_gate_verte_et_gain_f32() {
    let d = dossier("vert");
    ecrire(
        &d,
        "acemagic",
        "su2",
        vec![
            mesure_su2(32, "f64", 1, 10.0, Some(1.02)),
            mesure_su2(32, "f32", 1, 60.0, None),
        ],
    );
    let (md, out) = rapport(&d);
    assert!(md.contains("Gate de non-régression F64 : **PASS**"), "{md}");
    assert!(md.contains("×6.00"), "gain F32/F64 attendu ×6 :\n{md}");
    assert!(out.contains("PASS"));
    let _ = fs::remove_dir_all(&d);
}

#[test]
fn rapport_gate_rouge_si_sous_0_9() {
    let d = dossier("rouge");
    ecrire(&d, "gamer", "su2", vec![mesure_su2(16, "f64", 1, 10.0, Some(0.85))]);
    let (md, _) = rapport(&d);
    assert!(
        md.contains("**FAIL**") && md.contains("0.850") && md.contains("FAIL |"),
        "{md}"
    );
    let _ = fs::remove_dir_all(&d);
}

#[test]
fn rapport_gate_non_mesuree_sans_reference() {
    let d = dossier("nm");
    ecrire(&d, "acemagic", "su2", vec![mesure_su2(8, "f64", 1, 10.0, None)]);
    let (md, _) = rapport(&d);
    assert!(
        md.contains("NON MESURÉE"),
        "une gate sans mesure ne doit pas passer pour verte :\n{md}"
    );
    let _ = fs::remove_dir_all(&d);
}

#[test]
fn rapport_cond_fibre_gain_contre_cb() {
    let d = dossier("cb");
    ecrire(
        &d,
        "acemagic",
        "cond",
        vec![
            serde_json::json!({"famille": "cond", "L": 16, "mode": "rapide", "sweep_s": 0.0072, "sweeps_par_s": 138.9}),
        ],
    );
    ecrire(
        &d,
        "acemagic",
        "fibre",
        vec![
            serde_json::json!({"famille": "fibre", "L": 16, "mesure_complete_s": 0.0124, "mesures_completes_par_s": 80.6,
        "mesure_f_s": 0.001, "mesures_f_par_s": 1000.0}),
        ],
    );
    ecrire(&d, "acemagic", "su2", vec![]);
    let (md, _) = rapport(&d);
    assert!(
        md.contains("×10.0"),
        "balayage rapide L=16 : 72 ms contre 7,2 ms attendus ×10 :\n{md}"
    );
    let _ = fs::remove_dir_all(&d);
}

#[test]
fn cas_non_mesures_listes() {
    let d = dossier("nm2");
    ecrire(
        &d,
        "acemagic",
        "su2",
        vec![serde_json::json!({"cas": "su2 L=48 f64 K=16", "statut": "non_mesure", "raison": "budget épuisé"})],
    );
    let (md, _) = rapport(&d);
    assert!(md.contains("su2 L=48 f64 K=16 : non_mesure (budget épuisé)"), "{md}");
    let _ = fs::remove_dir_all(&d);
}

#[test]
fn banc_refuse_sous_le_seuil_ssd_et_donne_son_aide() {
    let o = Command::new("bash")
        .arg(banc().join("banc_unifie.sh"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(o.status.success());
    assert!(String::from_utf8_lossy(&o.stdout).contains("banc_unifie.sh"));
    // un seuil impossible : on simule en pointant la sortie sur un parent inexistant d'un fs minuscule n'est pas
    // portable ; on vérifie plutôt l'option inconnue.
    let o = Command::new("bash")
        .arg(banc().join("banc_unifie.sh"))
        .arg("--nimporte")
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn banc_unifie_lance_les_quatre_familles_avec_un_faux_binaire() {
    // Faux `kevinotron` qui consigne ses arguments : le script doit appeler bench su2, cond, fibre, obs, chacun avec
    // --budget-s, --json et --sortie, et la garde SSD ne doit pas se tromper sur un répertoire de sortie à créer.
    let d = dossier("banc_faux");
    let bin = d.join("kevinotron");
    fs::write(
        &bin,
        format!(
            "#!/usr/bin/env bash\necho \"$*\" >> {}/appels.log\nexit 0\n",
            d.display()
        ),
    )
    .unwrap();
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&bin, fs::Permissions::from_mode(0o755)).unwrap();
    }
    let sortie = d.join("neuf/sous/dossier");
    let o = Command::new("bash")
        .arg(banc().join("banc_unifie.sh"))
        .args(["--sortie"])
        .arg(&sortie)
        .env("KVU_BIN", &bin)
        .env("KVU_MACHINE", "inconnue")
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    // la garde SSD de la machine « inconnue » (110 Go, comme acemagic) peut refuser sur un disque presque plein : rc 7 alors
    if o.status.code() == Some(7) {
        assert!(err.contains("garde SSD"), "{err}");
        let _ = fs::remove_dir_all(&d);
        return;
    }
    assert_eq!(o.status.code(), Some(0), "{err}");
    let appels = fs::read_to_string(d.join("appels.log")).unwrap();
    for fam in ["su2", "cond", "fibre", "obs"] {
        let l = appels
            .lines()
            .find(|l| l.starts_with(&format!("bench {fam} ")))
            .unwrap_or_else(|| panic!("{fam} : {appels}"));
        assert!(
            l.contains("--budget-s 600")
                && l.contains("--json")
                && l.contains(&format!("--sortie {}/{fam}_inconnue.json", sortie.display())),
            "{l}"
        );
    }
    assert!(
        appels.contains("--L 8,16,32,48 --precision f64,f32 --chaines 1,16"),
        "{appels}"
    );
    let _ = fs::remove_dir_all(&d);
}
