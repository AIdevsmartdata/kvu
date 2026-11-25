//! Refus de `su2 gen` pour le chemin C1 GPU (B10, révision c09b) qui doivent tomber AVANT toute ouverture de GPU : tests
//! CPU, sans GPU, non ignorés. Sans GPU, un refus tardif rendrait le code 3 (GPU indisponible) au lieu du code 2 attendu.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN_KVU: &str = env!("CARGO_BIN_EXE_kvu");

fn rep(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("geo_gpu_cli")
        .join(format!("{nom}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.parent().expect("parent")).expect("dossier de test");
    d
}

fn su2(args: &[&str]) -> Output {
    Command::new(BIN_KVU)
        .arg("su2")
        .args(args)
        .output()
        .expect("échec de l'exécution de kvu")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_lowercase()
}

/// O5 : `--torsion 01 --backend hip|cuda` avec un L invalide pour le chemin C1 : code 2, aucun dossier créé, sans GPU.
#[test]
fn o5_validation_pure_avant_ouverture_du_gpu() {
    for (backend, l) in [("hip", "3"), ("cuda", "3"), ("hip", "182"), ("cuda", "1")] {
        let d = rep(&format!("o5_{backend}_{l}"));
        let o = su2(&[
            "gen",
            "--backend",
            backend,
            "--torsion",
            "01",
            "--L",
            l,
            "--outdir",
            d.to_str().unwrap(),
        ]);
        assert_eq!(
            o.status.code(),
            Some(2),
            "--backend {backend} --L {l} : rc {:?}, stderr {}",
            o.status.code(),
            stderr(&o)
        );
        assert!(
            stderr(&o).contains("l ="),
            "--backend {backend} --L {l} : diagnostic sans L : {}",
            stderr(&o)
        );
        assert!(
            !d.exists(),
            "--backend {backend} --L {l} : le dossier {} a été créé",
            d.display()
        );
    }
}

/// Tous les fichiers d'un dossier (chemin relatif ⇒ octets), pour vérifier qu'un refus n'écrit rien.
fn instantane(d: &Path) -> std::collections::BTreeMap<PathBuf, Vec<u8>> {
    let mut m = std::collections::BTreeMap::new();
    let mut pile = vec![d.to_path_buf()];
    while let Some(x) = pile.pop() {
        for e in std::fs::read_dir(&x).expect("lecture du dossier").flatten() {
            let p = e.path();
            if p.is_dir() {
                pile.push(p);
            } else {
                m.insert(
                    p.strip_prefix(d).expect("préfixe").to_path_buf(),
                    std::fs::read(&p).expect("lecture"),
                );
            }
        }
    }
    m
}

/// Retire le champ `invariants` de reprise.json : point de reprise tel qu'un binaire antérieur l'écrivait.
fn retirer_invariants(d: &Path) {
    let p = d.join("reprise.json");
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&p).expect("reprise.json")).expect("json");
    v.as_object_mut().expect("objet").remove("invariants");
    std::fs::write(&p, serde_json::to_vec_pretty(&v).expect("json")).expect("écriture");
}

/// O1 : un run tordu CPU interrompu consigne son moteur ; sa reprise sur GPU est refusée AVANT d'ouvrir un GPU (code 2,
/// aucun fichier modifié), y compris pour un point de reprise SANS invariants (binaire antérieur) ; sur CPU, elle aboutit.
#[test]
fn o1_reprise_tordue_cpu_jamais_sur_gpu() {
    let d = rep("o1_cpu");
    let ds = d.to_str().unwrap();
    let o = su2(&[
        "gen",
        "--backend",
        "cpu",
        "--L",
        "4",
        "--beta",
        "2.5",
        "--graine",
        "26111901",
        "--torsion",
        "01",
        "--therm",
        "1",
        "--nmeas",
        "4",
        "--nor",
        "1",
        "--reprise-every",
        "1",
        "--arret-apres",
        "2",
        "--dump-every",
        "1",
        "--format",
        "q64",
        "--outdir",
        ds,
    ]);
    assert_eq!(o.status.code(), Some(0), "création : {}", stderr(&o));
    assert!(
        !d.join("COMPLETE").exists(),
        "le run devait être interrompu"
    );
    let rj = std::fs::read_to_string(d.join("reprise.json")).expect("reprise.json");
    assert!(
        rj.contains("Cpu64+M4"),
        "reprise.json ne consigne pas le moteur tordu CPU : {rj}"
    );
    for sans_invariants in [false, true] {
        if sans_invariants {
            retirer_invariants(&d);
        }
        let avant = instantane(&d);
        for b in ["hip", "cuda"] {
            let o = su2(&["gen", "--backend", b, "--reprendre", ds]);
            assert_eq!(
                o.status.code(),
                Some(2),
                "reprise --backend {b} (sans invariants : {sans_invariants}) : {}",
                stderr(&o)
            );
            assert!(
                stderr(&o).contains("moteur"),
                "diagnostic sans « moteur » : {}",
                stderr(&o)
            );
            assert_eq!(
                instantane(&d),
                avant,
                "fichiers modifiés par le refus --backend {b}"
            );
        }
    }
    let o = su2(&["gen", "--backend", "cpu", "--reprendre", ds]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "reprise CPU du point de reprise antérieur : {}",
        stderr(&o)
    );
    assert!(
        d.join("COMPLETE").exists(),
        "la reprise CPU n'a pas terminé le run"
    );
}

/// O7 : la documentation ne contredit plus les capacités du chemin C1 (relecture c09).
#[test]
fn o7_documentation_a_jour() {
    let racine = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let torsion = std::fs::read_to_string(racine.join("docs/TORSION.md")).expect("docs/TORSION.md");
    assert!(
        !torsion.contains("`hip`, `cuda` et `auto` sont refusés"),
        "TORSION.md annonce encore le refus de hip et cuda"
    );
    assert!(
        torsion.contains("Geo64Gpu"),
        "TORSION.md ne décrit pas le chemin GPU"
    );
    let note = std::fs::read_to_string(racine.join("docs/v0.2/notes/geo_gpu.md"))
        .expect("notes/geo_gpu.md");
    assert!(
        !note.contains("C (masque + β par plan)"),
        "la note annonce encore le masque comme non porté"
    );
    assert!(
        note.contains("ÉGALITÉ BINAIRE") && note.contains("TRAJECTOIRE LIBRE"),
        "la note ne distingue pas les comparaisons"
    );
    assert!(
        note.contains("SourceBord::ParChaine"),
        "la note n'écrit pas le contrat des valeurs de bord"
    );
}

/// c09c, point 6 : un point de reprise C0 CORROMPU, repris avec un backend GPU indisponible, rend 3 (GPU indisponible) comme
/// sur 1c488c9, et non 4 (le contrôle des invariants ne lit rien pour un run sans torsion). Sans la feature `cuda`, le backend
/// CUDA est toujours indisponible : le test est déterministe.
#[cfg(not(feature = "cuda"))]
#[test]
fn c09c_p6_reprise_c0_corrompue_garde_l_ordre_des_erreurs() {
    let d = rep("p6_c0");
    let ds = d.to_str().unwrap();
    let o = su2(&[
        "gen",
        "--backend",
        "cpu",
        "--L",
        "4",
        "--beta",
        "2.3",
        "--graine",
        "26111910",
        "--therm",
        "1",
        "--nmeas",
        "4",
        "--nor",
        "1",
        "--reprise-every",
        "1",
        "--arret-apres",
        "2",
        "--outdir",
        ds,
    ]);
    assert_eq!(o.status.code(), Some(0), "création : {}", stderr(&o));
    let champs = d.join("reprise.champs");
    let n = std::fs::metadata(&champs).expect("reprise.champs").len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&champs)
        .expect("ouverture")
        .set_len(n / 2)
        .expect("troncature");
    let o = su2(&["gen", "--backend", "cuda", "--reprendre", ds]);
    assert_eq!(
        o.status.code(),
        Some(3),
        "reprise C0 corrompue, CUDA indisponible : {}",
        stderr(&o)
    );
}

/// c09c, point 1 (BLOQUANT) : un point de reprise SANS invariants d'un run tordu dont le manifeste dit « GPU » (binaire B10
/// antérieur à c09b : backend hip, module su2_geo64) n'est PAS repris sur CPU (code 2, rien d'écrit, message qui dit quoi
/// faire) ; sans invariants et de provenance CPU, la reprise CPU reste acceptée (test o1).
#[test]
fn c09c_p1_ancien_point_gpu_tordu_jamais_repris_sur_cpu() {
    let d = rep("p1_ancien_gpu");
    let ds = d.to_str().unwrap();
    let o = su2(&[
        "gen",
        "--backend",
        "cpu",
        "--L",
        "4",
        "--beta",
        "2.5",
        "--graine",
        "26111911",
        "--torsion",
        "01",
        "--therm",
        "1",
        "--nmeas",
        "4",
        "--nor",
        "1",
        "--reprise-every",
        "1",
        "--arret-apres",
        "2",
        "--format",
        "q64",
        "--outdir",
        ds,
    ]);
    assert_eq!(o.status.code(), Some(0), "création : {}", stderr(&o));
    // provenance « GPU » d'un binaire antérieur : invariants absents, manifeste backend hip avec le module su2_geo64
    retirer_invariants(&d);
    let pm = d.join("manifest.json");
    let mut m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&pm).expect("manifeste")).expect("json");
    m["backend"] = serde_json::json!("hip");
    m["modules_compiles"] = serde_json::json!([{ "nom": "su2_geo64" }]);
    std::fs::write(&pm, serde_json::to_vec_pretty(&m).expect("json")).expect("écriture");
    let avant = instantane(&d);
    for b in ["cpu", "hip"] {
        let o = su2(&["gen", "--backend", b, "--reprendre", ds]);
        assert_eq!(
            o.status.code(),
            Some(2),
            "ancien point GPU tordu repris avec --backend {b} : {}",
            stderr(&o)
        );
        assert!(
            stderr(&o).contains("relancer"),
            "le message ne dit pas quoi faire : {}",
            stderr(&o)
        );
        assert_eq!(
            instantane(&d),
            avant,
            "fichiers modifiés par le refus --backend {b}"
        );
    }
}

/// c09c, point 2 : un point de reprise de forme C1 dont le moteur est correct mais la GÉOMÉTRIE diffère (direction du temps 0
/// au lieu de 3) est refusé AVANT l'ouverture du GPU (code 2, sans GPU), et le refus ne modifie AUCUN fichier, manifeste
/// compris (sur 30a35bb, seul le moteur était comparé avant ouverture : le GPU était ouvert, code 3 sans GPU).
#[test]
fn c09c_p2_invariants_complets_avant_ouverture() {
    use kvu::su2::geo_gpu::invariants_c1;
    let d = rep("p2_geometrie");
    let ds = d.to_str().unwrap();
    let o = su2(&[
        "gen",
        "--backend",
        "cpu",
        "--L",
        "4",
        "--beta",
        "2.5",
        "--graine",
        "26111912",
        "--torsion",
        "01",
        "--therm",
        "1",
        "--nmeas",
        "4",
        "--nor",
        "1",
        "--reprise-every",
        "1",
        "--arret-apres",
        "2",
        "--format",
        "q64",
        "--outdir",
        ds,
    ]);
    assert_eq!(o.status.code(), Some(0), "création : {}", stderr(&o));
    // point de reprise C1 HIP consigné avec une géométrie différente (dir_temps = 0) ; manifeste cohérent (hip, su2_geo64)
    let t = kvu::geom::Torsion::analyser("01").expect("torsion");
    let fausse = kvu::geom::GeoSpec::cubique(4, 0).avec_torsion(t);
    let inv = invariants_c1(
        "hip",
        &fausse,
        kvu::su2::Gel::Aucun,
        None,
        kvu::su2::AppMasque::Tout,
    );
    let pj = d.join("reprise.json");
    let mut e: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&pj).expect("reprise.json")).expect("json");
    e["invariants"] = inv;
    std::fs::write(&pj, serde_json::to_vec_pretty(&e).expect("json")).expect("écriture");
    let pm = d.join("manifest.json");
    let mut m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&pm).expect("manifeste")).expect("json");
    m["backend"] = serde_json::json!("hip");
    m["modules_compiles"] = serde_json::json!([{ "nom": "su2_geo64" }]);
    std::fs::write(&pm, serde_json::to_vec_pretty(&m).expect("json")).expect("écriture");
    let avant = instantane(&d);
    let o = su2(&["gen", "--backend", "hip", "--reprendre", ds]);
    assert_eq!(
        o.status.code(),
        Some(2),
        "géométrie consignée différente : {}",
        stderr(&o)
    );
    assert!(
        stderr(&o).contains("geometrie"),
        "le diagnostic ne nomme pas la géométrie : {}",
        stderr(&o)
    );
    assert_eq!(
        instantane(&d),
        avant,
        "le refus a modifié des fichiers du run (manifeste compris)"
    );
}
