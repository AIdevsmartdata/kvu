//! Gates de provenance de la tâche T8 (ARCHITECTURE_UNIFIEE.md §6.5) :
//!
//! - PROV-1 : `binaire_sha256` = sha256 du fichier exécuté ; `source_hash` recalculé par
//!   `validation/stockage/source_hash.py` identique ; `noyaux_embarques` = sha des fichiers du dépôt ;
//!   `sha256_source` d'une source assemblée recalculé ; options présentes ;
//! - PROV-2 : modifier un fichier de `noyaux/` dans un arbre temporaire change l'empreinte `Unifiee` ;
//!   les recettes BuildSh et Complete restent INCHANGÉES ;
//! - PROV-3 : kill -9 ou rc ≠ 0 ⇒ pas de COMPLETE.
//!
//! Chaque test imprime `PASS <nom>`.

#[path = "donnees/stockage/support.rs"]
mod support;

use kvu::commun::{BackendKind, Erreur};
use kvu::gpu::{FpContract, OptionsCompil, ProvenanceCompil};
use kvu::heritage::manifest::{hash_arbre_source, sha256_hex, Recette};
use kvu::noyaux;
use kvu::stockage::garde::GardeSsd;
use kvu::stockage::provenance;
use kvu::stockage::store::verifier_repertoire;
use kvu::stockage::{npy, FormatChamp, Manifeste};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use support::*;

fn libre() -> GardeSsd {
    GardeSsd::nouvelle(0.0)
}

fn lire_json(p: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(p).unwrap()).expect("manifest.json doit être un JSON valide")
}

fn source_hash_python(racine: &Path, recette: &str) -> (String, usize) {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("validation/stockage/source_hash.py");
    let o = Command::new("python3")
        .arg(&script)
        .args(["--recette", recette, "--racine"])
        .arg(racine)
        .output()
        .expect("python3");
    assert!(
        o.status.success(),
        "source_hash.py : {}",
        String::from_utf8_lossy(&o.stderr)
    );
    let s = String::from_utf8_lossy(&o.stdout).into_owned();
    let mut it = s.split_whitespace();
    (it.next().unwrap().to_string(), it.next().unwrap().parse().unwrap())
}

/// WP1 (dépôt autonome) : recalcul Python de la recette `kvu-source/1` (remplace `source_hash.py --recette unifiee` sur le dépôt réel).
fn source_hash_kvu_python(racine: &Path) -> (String, usize) {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("validation/historique/source_hash_kvu.py");
    let o = Command::new("python3").arg("-I").arg(&script).arg(racine).output().expect("python3");
    assert!(o.status.success(), "source_hash_kvu.py : {}", String::from_utf8_lossy(&o.stderr));
    let h = String::from_utf8_lossy(&o.stdout).trim().to_string();
    let n = String::from_utf8_lossy(&o.stderr).split_whitespace().next().unwrap().parse().unwrap();
    (h, n)
}

// ------------------------------------------------------------------------------------------
// PROV-1
// ------------------------------------------------------------------------------------------

#[test]
fn prov1_manifeste_complet_et_recalcule() {
    let sc = Scratch::nouveau("prov1");
    let dir = sc.p("run");
    let argv: Vec<String> = ["kevinotron", "su2", "gen", "--L", "2"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut m = Manifeste::ouvrir_avec_garde(&dir, "su2", &argv, &libre(), 1 << 20).unwrap();
    m.champ("precision", "f64").unwrap();
    m.champ("graines", "[926193012]").unwrap();
    m.champ("options", "ref-k1").unwrap();
    m.verrou("ancetre").unwrap();

    // un fichier produit (sha256 calculé par le manifeste)
    let c = champ_hurwitz(2);
    npy::ecrire_champ(&dir.join("cfg_000000.npy"), &c, FormatChamp::C16, &libre()).unwrap();
    m.fichier(&dir.join("cfg_000000.npy")).unwrap();
    // un fichier dont le sha256 a été calculé pendant l'écriture (pas de relecture)
    let r = npy::ecrire_champ_hache(&dir.join("cfg_000001.npy"), &c, FormatChamp::Q64, &libre()).unwrap();
    m.fichier_connu(&dir.join("cfg_000001.npy"), r.sha256.as_deref().unwrap(), r.octets)
        .unwrap();

    // une source assemblée à la main (le texte et son sha256 sont recalculés plus bas par sha256sum)
    let texte = "// source d'essai\nextern \"C\" __global__ void k(double* x) { x[0] = 1.0; }\n";
    fs::write(sc.p("source.cu"), texte).unwrap();
    let pc = ProvenanceCompil {
        nom: "essai".into(),
        backend: BackendKind::Hip,
        arch: "gfx1151".into(),
        sha256_source: sha256_hex(texte.as_bytes()),
        parties: vec![
            ("prelude_hip.h".into(), "00".repeat(32)),
            ("test_runtime.cu".into(), "11".repeat(32)),
        ],
        options: OptionsCompil::reference_k1(),
        arguments: ["--offload-arch=gfx1151", "-O3", "-ffp-contract=off"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        sha256_code: "ab".repeat(32),
    };
    m.ajouter_compil(std::slice::from_ref(&pc)).unwrap();
    m.ajouter_compil(&[pc]).unwrap(); // doublon : consigné une seule fois
    m.terminer(0).unwrap();

    let j = lire_json(&dir.join("manifest.json"));
    assert_eq!(j["schema"], "kevinotron-unifie/1");
    assert_eq!(j["schema_k8"], "kevinotron-provenance/1");
    assert_eq!(j["moteur"], "su2");
    assert_eq!(j["statut"], "termine");
    assert_eq!(j["rc"], 0);
    assert_eq!(j["argv"], serde_json::json!(argv));
    assert_eq!(j["precision"], "f64");
    assert_eq!(j["graines"], serde_json::json!([926193012]));
    assert_eq!(j["verrou"], "ancetre");
    assert!(j["machine"].is_string() && j["versions"].is_object() && j["politique"].is_object());
    assert_eq!(j["garde_ssd"]["seuil_go"], 0.0);
    assert!(j["garde_ssd"]["libre_go_debut"].as_f64().unwrap() > 0.0);

    // (1) binaire_sha256 = sha256 du fichier exécuté (calculé ici par sha256sum, indépendamment)
    let exe = std::env::current_exe().unwrap();
    assert_eq!(
        j["binaire_sha256"],
        sha256sum(&exe).as_str(),
        "binaire_sha256 ≠ sha256sum du binaire en cours d'exécution"
    );
    assert_eq!(provenance::sha256_binaire().unwrap(), sha256sum(&exe));

    // (2) source_hash recalculé en Python = celui du manifeste = celui d'un recalcul Rust
    let racine = racine_depot();
    let (h_py, n_py) = source_hash_kvu_python(&racine);
    assert_eq!(
        j["source_hash"],
        h_py.as_str(),
        "source_hash du manifeste ≠ source_hash_kvu.py"
    );
    assert_eq!(j["source_hash_nfichiers"], n_py.to_string().as_str());
    let (h_rs, n_rs) = kvu::stockage::recette::hash_kvu_source_1(&racine).unwrap();
    assert_eq!((h_rs.as_str(), n_rs), (h_py.as_str(), n_py));
    assert_eq!(provenance::source_hash_unifiee(), Some(h_py.as_str()));
    assert!(j["source_hash_recette"]
        .as_str()
        .unwrap()
        .contains("crates/kvu/noyaux/**"));

    // git_head : celui embarqué (« non_enregistre » si KEV_GIT_HEAD manquait à la compilation)
    assert_eq!(j["git_head"], provenance::git_head());

    // (3) noyaux_embarques = sha256 des fichiers du dépôt
    let embarques = j["noyaux_embarques"].as_array().unwrap();
    assert_eq!(embarques.len(), noyaux::TOUS.len());
    assert_eq!(embarques.len(), 12);
    for e in embarques {
        let nom = e["nom"].as_str().unwrap();
        let fichier = racine.join("crates/kvu/noyaux").join(nom);
        assert_eq!(e["sha256"], sha256sum(&fichier).as_str(), "noyau {nom}");
    }
    let noms: Vec<&str> = embarques.iter().map(|e| e["nom"].as_str().unwrap()).collect();
    let attendus: Vec<&str> = noyaux::TOUS.iter().map(|f| f.nom).collect();
    assert_eq!(noms, attendus, "ordre du registre");

    // (4) sha256_source d'une source assemblée recalculé ; options et arguments présents
    let mc = j["modules_compiles"].as_array().unwrap();
    assert_eq!(mc.len(), 1, "le doublon ne doit être consigné qu'une fois");
    assert_eq!(mc[0]["sha256_source"], sha256sum(&sc.p("source.cu")).as_str());
    assert_eq!(mc[0]["options"]["fp_contract"], "off");
    assert_eq!(mc[0]["options"]["optimisation"], 3);
    assert!(mc[0]["options"]["extra"].is_array());
    assert_eq!(mc[0]["arguments"][2], "-ffp-contract=off");
    assert_eq!(mc[0]["sha256_code"], "ab".repeat(32).as_str());
    assert_eq!(mc[0]["parties"][0][0], "prelude_hip.h");
    assert_eq!(OptionsCompil::reference_k1().fp_contract, FpContract::Off);

    // fichiers produits : nom, sha256, octets
    let fichiers = j["fichiers"].as_array().unwrap();
    assert_eq!(fichiers.len(), 2);
    for f in fichiers {
        let p = dir.join(f["nom"].as_str().unwrap());
        assert_eq!(f["sha256"], sha256sum(&p).as_str());
        assert_eq!(f["octets"], fs::metadata(&p).unwrap().len());
    }

    // COMPLETE en dernier, lié au manifeste
    let complete = fs::read_to_string(dir.join("COMPLETE")).unwrap();
    assert_eq!(
        complete,
        format!("rc=0\nmanifest_sha256={}\n", sha256sum(&dir.join("manifest.json")))
    );
    // la vérification du store relit tout et trouve tout bon
    let r = verifier_repertoire(&dir).unwrap();
    assert!(r.ok() && r.sans_complete.is_empty());
    assert_eq!((r.unites, r.fichiers_verifies), (1, 2));
    println!("PASS PROV-1 binaire_sha256, source_hash (Python), noyaux_embarques, sha256_source, options");
}

/// Quand T2 (assembleur) sera fusionné, la source RÉELLEMENT assemblée est recalculée aussi.
#[test]
fn prov1_source_assemblee_recalculee_si_t2_fusionne() {
    let s = kvu::gpu::assembler(BackendKind::Hip, "essai", &[&noyaux::SU2_COMMUN], &[("K1_X", "1")]);
    if s.texte.is_empty() {
        println!("SKIP prov1_source_assemblee_recalculee_si_t2_fusionne (gpu::assembler = squelette de T2 : ne vaut pas vert)");
        return;
    }
    assert_eq!(
        s.sha256,
        sha256_hex(s.texte.as_bytes()),
        "sha256 de la source assemblée"
    );
    let sc = Scratch::nouveau("prov1_asm");
    fs::write(sc.p("assemble.txt"), &s.texte).unwrap();
    assert_eq!(s.sha256, sha256sum(&sc.p("assemble.txt")));
    for (nom, sha) in &s.parties {
        let f = noyaux::TOUS.iter().find(|f| f.nom == nom).expect("partie du registre");
        assert_eq!(sha, &sha256_hex(f.contenu.as_bytes()));
    }
    println!("PASS PROV-1 source assemblée par T2 : sha256 recalculé, parties = registre");
}

// ------------------------------------------------------------------------------------------
// PROV-2
// ------------------------------------------------------------------------------------------

/// Arbre factice qui a la forme du dépôt : racine (src, fichiers racine) + crates/kvu.
fn faux_depot(racine: &Path) {
    let ecrire = |rel: &str, contenu: &str| {
        let p = racine.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, contenu).unwrap();
    };
    ecrire("src/lib.rs", "pub fn a() {}\n");
    ecrire("src/hip/noyau.hip", "__global__ void a() {}\n");
    ecrire("src/sous/b.rs", "pub fn b() {}\n");
    ecrire("src/lisez_moi.txt", "texte non rust\n");
    for f in [
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "e6_generators.bin",
        "e7_generators.bin",
        "f4_generators.bin",
    ] {
        ecrire(f, &format!("racine {f}\n"));
    }
    ecrire("crates/kvu/Cargo.toml", "[package]\nname = \"kvu\"\n");
    ecrire("crates/kvu/build.rs", "fn main() {}\n");
    ecrire("crates/kvu/src/lib.rs", "pub mod x;\n");
    ecrire("crates/kvu/src/stockage/mod.rs", "// T8\n");
    ecrire("crates/kvu/noyaux/su2_ref64.cu", "__global__ void k() {}\n");
    ecrire("crates/kvu/noyaux/prelude_hip.h", "#define X 1\n");
    ecrire("crates/kvu/tests/hors_recette.rs", "// ne compte pas\n");
}

fn trois(racine: &Path) -> [String; 3] {
    [Recette::BuildSh, Recette::Complete, Recette::Unifiee].map(|r| hash_arbre_source(racine, r).unwrap().0)
}

#[test]
fn prov2_modifier_un_noyau_change_unifiee_seulement() {
    let sc = Scratch::nouveau("prov2");
    let d = sc.p("depot");
    faux_depot(&d);
    let [bs0, co0, un0] = trois(&d);

    // Les trois recettes sont distinctes, et Python reproduit chacune
    assert!(bs0 != co0 && co0 != un0 && bs0 != un0);
    for (nom, h) in [("build_sh", &bs0), ("complete", &co0), ("unifiee", &un0)] {
        assert_eq!(&source_hash_python(&d, nom).0, h, "recette {nom} : Rust ≠ Python");
    }
    // listes de fichiers : Unifiee = Complete + kvu/{src,noyaux}/** + 2 fichiers ; hors recette : tests/
    let f_co = kvu::heritage::manifest::fichiers_recette(&d, Recette::Complete).unwrap();
    let f_un = kvu::heritage::manifest::fichiers_recette(&d, Recette::Unifiee).unwrap();
    assert!(f_co.iter().all(|f| f_un.contains(f)));
    for attendu in [
        "crates/kvu/noyaux/su2_ref64.cu",
        "crates/kvu/noyaux/prelude_hip.h",
        "crates/kvu/src/lib.rs",
        "crates/kvu/src/stockage/mod.rs",
        "crates/kvu/Cargo.toml",
        "crates/kvu/build.rs",
    ] {
        assert!(
            f_un.iter().any(|f| f == attendu),
            "{attendu} absent de la recette Unifiee"
        );
        assert!(
            !f_co.iter().any(|f| f == attendu),
            "{attendu} ne doit pas être dans Complete"
        );
    }
    assert!(!f_un.iter().any(|f| f.contains("tests/")));
    assert_eq!(f_un.len(), f_co.len() + 6);
    // BuildSh ne voit que les .rs de src/ + 5 fichiers racine
    let f_bs = kvu::heritage::manifest::fichiers_recette(&d, Recette::BuildSh).unwrap();
    assert!(f_bs.iter().all(|f| f.ends_with(".rs") || !f.starts_with("src/")));
    assert!(!f_bs
        .iter()
        .any(|f| f.starts_with("crates/") || f == "build.rs" || f.ends_with(".hip")));

    // 1. un noyau de crates/kvu/noyaux/ change : Unifiee change, BuildSh et Complete INCHANGÉES
    fs::write(
        d.join("crates/kvu/noyaux/su2_ref64.cu"),
        "__global__ void k() { /* mutant */ }\n",
    )
    .unwrap();
    let [bs1, co1, un1] = trois(&d);
    assert_eq!(
        (&bs1, &co1),
        (&bs0, &co0),
        "BuildSh et Complete doivent rester INCHANGÉES"
    );
    assert_ne!(un1, un0, "Unifiee doit voir le noyau modifié");
    assert_eq!(source_hash_python(&d, "unifiee").0, un1);

    // 2. un fichier source de kvu change : idem
    fs::write(d.join("crates/kvu/src/stockage/mod.rs"), "// T8 modifié\n").unwrap();
    let [bs2, co2, un2] = trois(&d);
    assert_eq!((&bs2, &co2), (&bs0, &co0));
    assert!(un2 != un1 && un2 != un0);

    // 3. un fichier ajouté dans noyaux/ : Unifiee change
    fs::write(d.join("crates/kvu/noyaux/neuf.cu"), "// neuf\n").unwrap();
    assert_ne!(trois(&d)[2], un2);
    // 4. un fichier hors recette (tests/) : rien ne change
    let avant = trois(&d);
    fs::write(d.join("crates/kvu/tests/hors_recette.rs"), "// modifié\n").unwrap();
    assert_eq!(trois(&d), avant);

    // 5. src/ racine : un .hip change Complete et Unifiee mais pas BuildSh ; un .rs change les trois
    fs::write(d.join("src/hip/noyau.hip"), "__global__ void a() { /* x */ }\n").unwrap();
    let [bs5, co5, un5] = trois(&d);
    assert_eq!(
        bs5, bs0,
        "BuildSh est aveugle aux noyaux .hip : comportement historique conservé"
    );
    assert!(co5 != co0 && un5 != avant[2]);
    fs::write(d.join("src/sous/b.rs"), "pub fn b() { /* x */ }\n").unwrap();
    let [bs6, co6, un6] = trois(&d);
    assert!(bs6 != bs0 && co6 != co5 && un6 != un5);
    println!("PASS PROV-2 noyaux/ modifié => Unifiee change ; BuildSh et Complete inchangées");
}

#[test]
fn prov2_depot_reel_recettes_k8_reproduites_en_python() {
    // WP1 (dépôt autonome) : les recettes BuildSh/Complete/Unifiee d'origine hachent `src/**` et des fichiers racine du moteur
    // historique, absents ici (elles restent testées sur arbres jouets plus haut) ; sur le dépôt réel, c'est `kvu-source/1`.
    let racine = racine_depot();
    let (h, n) = kvu::stockage::recette::hash_kvu_source_1(&racine).unwrap();
    let (hp, np) = source_hash_kvu_python(&racine);
    assert_eq!((h, n), (hp, np), "recette kvu-source/1 sur le dépôt réel : Rust ≠ Python");
    // l'empreinte embarquée dans ce binaire est celle de l'arbre courant
    assert_eq!(
        provenance::source_hash_unifiee().unwrap(),
        kvu::stockage::recette::hash_kvu_source_1(&racine).unwrap().0
    );
    println!("PASS PROV-2 dépôt réel : kvu-source/1 identique en Rust, en Python et dans le binaire");
}

// ------------------------------------------------------------------------------------------
// PROV-3
// ------------------------------------------------------------------------------------------

/// Enfant des tests kill -9 : ouvre un manifeste puis dort. Sans la variable d'environnement, ne fait rien.
#[test]
fn prov3_enfant_long() {
    let Ok(dir) = std::env::var("KVU_PROV3_DIR") else {
        return;
    };
    let mut m = Manifeste::ouvrir_avec_garde(Path::new(&dir), "enfant", &[], &libre(), 0).unwrap();
    m.champ("pret", "true").unwrap();
    std::thread::sleep(Duration::from_secs(120));
    let _ = m.terminer(0);
}

#[test]
fn prov3_kill9_pas_de_complete() {
    let sc = Scratch::nouveau("prov3_kill");
    let dir = sc.p("run");
    let mut enfant = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "prov3_enfant_long", "--nocapture", "--test-threads=1"])
        .env("KVU_PROV3_DIR", &dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    // attendre que le manifeste « en_cours » soit complet (champ `pret`)
    let debut = Instant::now();
    loop {
        if let Ok(t) = fs::read_to_string(dir.join("manifest.json")) {
            if t.contains("\"pret\": true") {
                break;
            }
        }
        assert!(
            debut.elapsed() < Duration::from_secs(30),
            "l'enfant n'a pas ouvert son manifeste"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    enfant.kill().unwrap(); // SIGKILL
    let statut = enfant.wait().unwrap();
    assert!(!statut.success());
    assert!(
        !dir.join("COMPLETE").exists(),
        "un run tué (kill -9) ne doit JAMAIS avoir de COMPLETE"
    );
    let j = lire_json(&dir.join("manifest.json"));
    assert_eq!(j["statut"], "en_cours");
    assert_eq!(j["rc"], Value::Null);
    // le répertoire n'a pas de résidu autre que le manifeste
    let noms: Vec<String> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(noms, ["manifest.json"], "{noms:?}");
    // la vérification du store le signale : pas de COMPLETE
    let r = verifier_repertoire(&dir).unwrap();
    assert_eq!(r.sans_complete, vec![dir.clone()]);
    println!("PASS PROV-3 kill -9 : pas de COMPLETE, manifeste en_cours");
}

#[test]
fn prov3_rc_non_nul_pas_de_complete() {
    let sc = Scratch::nouveau("prov3_rc");
    for rc in [1, 2, 7, 137] {
        let dir = sc.p(&format!("run{rc}"));
        let m = Manifeste::ouvrir_avec_garde(&dir, "test", &[], &libre(), 0).unwrap();
        assert!(!dir.join("COMPLETE").exists(), "pas de COMPLETE avant terminer");
        m.terminer(rc).unwrap();
        assert!(!dir.join("COMPLETE").exists(), "rc = {rc} : pas de COMPLETE");
        let j = lire_json(&dir.join("manifest.json"));
        assert_eq!(
            (j["statut"].as_str().unwrap(), j["rc"].as_i64().unwrap()),
            ("echec", rc as i64)
        );
    }
    // clôture explicite en échec, même avec rc = 0
    let dir = sc.p("run_echouer");
    let m = Manifeste::ouvrir_avec_garde(&dir, "test", &[], &libre(), 0).unwrap();
    m.echouer(0, "panique observée").unwrap();
    assert!(!dir.join("COMPLETE").exists());
    assert_eq!(
        lire_json(&dir.join("manifest.json"))["raison_echec"],
        "panique observée"
    );
    // et un rc = 0 sain écrit COMPLETE
    let dir = sc.p("run_ok");
    Manifeste::ouvrir_avec_garde(&dir, "test", &[], &libre(), 0)
        .unwrap()
        .terminer(0)
        .unwrap();
    assert!(dir.join("COMPLETE").exists());
    println!("PASS PROV-3 rc != 0 : pas de COMPLETE ; rc = 0 : COMPLETE");
}

#[test]
fn prov3_fichier_altere_avant_la_cloture() {
    let sc = Scratch::nouveau("prov3_alt");
    let dir = sc.p("run");
    let mut m = Manifeste::ouvrir_avec_garde(&dir, "test", &[], &libre(), 0).unwrap();
    fs::write(dir.join("a.csv"), "x,y\n1,2\n").unwrap();
    m.fichier(&dir.join("a.csv")).unwrap();
    fs::write(dir.join("a.csv"), "x,y\n1,2\n3,4\n").unwrap(); // modifié après consignation
    let e = m.terminer(0).unwrap_err();
    assert!(matches!(e, Erreur::Validation(_)), "{e}");
    assert!(!dir.join("COMPLETE").exists());
    assert_eq!(lire_json(&dir.join("manifest.json"))["statut"], "echec");
    // fichier disparu
    let dir2 = sc.p("run2");
    let mut m = Manifeste::ouvrir_avec_garde(&dir2, "test", &[], &libre(), 0).unwrap();
    fs::write(dir2.join("b.csv"), "x\n").unwrap();
    m.fichier(&dir2.join("b.csv")).unwrap();
    fs::remove_file(dir2.join("b.csv")).unwrap();
    assert!(m.terminer(0).is_err());
    assert!(!dir2.join("COMPLETE").exists());
    println!("PASS PROV-3 fichier consigné altéré ou disparu : pas de COMPLETE");
}

// ------------------------------------------------------------------------------------------
// Contrat K8 : répertoire neuf, champs libres
// ------------------------------------------------------------------------------------------

#[test]
fn repertoire_existant_refuse_rc2_et_contenu_intact() {
    let sc = Scratch::nouveau("prov_existe");
    let dir = sc.p("run");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("precieux.txt"), "ne pas toucher").unwrap();
    let avant = empreinte_arbre(&dir);
    let e = Manifeste::ouvrir_avec_garde(&dir, "test", &[], &libre(), 0)
        .err()
        .expect("refus attendu");
    assert_eq!(e.code_sortie(), 2, "{e}");
    assert_eq!(empreinte_arbre(&dir), avant);
    // un fichier ou un lien à la place du répertoire : refusé aussi
    fs::write(sc.p("f"), "x").unwrap();
    assert!(Manifeste::ouvrir_avec_garde(&sc.p("f"), "test", &[], &libre(), 0).is_err());
    std::os::unix::fs::symlink(&dir, sc.p("lien")).unwrap();
    assert!(Manifeste::ouvrir_avec_garde(&sc.p("lien"), "test", &[], &libre(), 0).is_err());
    println!("PASS répertoire de run existant refusé (rc 2), contenu intact");
}

#[test]
fn champs_libres_et_cles_reservees() {
    let sc = Scratch::nouveau("prov_champs");
    let mut m = Manifeste::ouvrir_avec_garde(&sc.p("run"), "test", &[], &libre(), 0).unwrap();
    m.champ("texte", "bonjour").unwrap();
    m.champ("entier", "42").unwrap();
    m.champ("objet", "{\"a\": [1, 2.5, null], \"b\": \"é\"}").unwrap();
    m.champ("chaine_json", "\"avec guillemets\"").unwrap();
    m.champ("nul", "null").unwrap();
    for k in [
        "statut",
        "rc",
        "fin_unix",
        "schema",
        "source_hash",
        "binaire_sha256",
        "fichiers",
        "modules_compiles",
        "noyaux_embarques",
    ] {
        assert!(m.champ(k, "x").is_err(), "{k} doit être réservée");
    }
    assert!(m.verrou("n_importe_quoi").is_err());
    m.verrou("pris_ici").unwrap();
    let j = lire_json(&sc.p("run/manifest.json"));
    assert_eq!(j["texte"], "bonjour");
    assert_eq!(j["entier"], 42);
    assert_eq!(j["objet"]["a"][1], 2.5);
    assert_eq!(j["objet"]["b"], "é");
    assert_eq!(j["chaine_json"], "avec guillemets");
    assert_eq!(j["nul"], Value::Null);
    assert_eq!(j["verrou"], "pris_ici");
    assert_eq!(j["statut"], "en_cours");
    // le manifeste est un JSON valide pour Python aussi
    let o = Command::new("python3")
        .args(["-c", "import json,sys; json.load(open(sys.argv[1]))"])
        .arg(sc.p("run/manifest.json"))
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    m.terminer(0).unwrap();
    println!("PASS champs libres JSON ou texte, clés réservées, JSON valide pour Python");
}
