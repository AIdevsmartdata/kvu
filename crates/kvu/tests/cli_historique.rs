//! Tests des gates historiques G-H01..G-H05 avec un binaire `kevinotron` FACTICE (script Python) :
//! chaque gate doit passer sur un moteur correct ET devenir ROUGE sur un moteur mutant (contrôle de
//! puissance). Aucune exécution GPU. Les gates G-H06..G-H13 (lots, analyses) sont couvertes par
//! `analyse_historique.py` rejoué sur les ensembles d'origine (voir RAPPORT_T9.md) et par la suite complète.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn hist() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("validation/historique")
}

struct Banc {
    dir: PathBuf,
}

impl Banc {
    fn nouveau(nom: &str) -> Banc {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let k = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("kvu_hist_{nom}_{}_{k}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("scratch")).unwrap();
        Banc { dir }
    }

    fn ecrire_exe(&self, nom: &str, contenu: &str) -> PathBuf {
        let p = self.dir.join(nom);
        fs::write(&p, contenu).unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    fn gate(&self, script: &str, bin: &Path, env: &[(&str, &str)]) -> (String, Output) {
        let mut c = Command::new("bash");
        c.env_clear()
            .envs(std::env::vars().filter(|(k, _)| !k.starts_with("KVU_") && !k.starts_with("KEVINOTRON_")));
        c.arg(hist().join(script))
            .env("KVU_BIN", bin)
            .env("KVU_MACHINE", "acemagic")
            .env("KVU_SCRATCH", self.dir.join("scratch"))
            .env("KVU_PY", "python3")
            .env("KVU_RACINE", self.dir.join("racine"))
            // store de validation et seuil de la garde SSD propres au banc : rien n'est créé hors du banc, et le résultat
            // ne dépend pas de la place libre de la machine de test (la garde SSD des gates n'est pas l'objet de ces tests)
            .env("KVU_STORE_VALID", self.dir.join("store_validation"))
            .env("KVU_SSD_MIN_GO", "1")
            // le faux flock n'est pas installé ici : verrou sur un fichier jetable, jamais le verrou GPU
            .env(
                "PATH",
                format!("{}:{}", self.dir.join("pbin").display(), std::env::var("PATH").unwrap()),
            );
        for (k, v) in env {
            c.env(k, v);
        }
        let o = c.output().expect("lancement de la gate");
        (String::from_utf8_lossy(&o.stdout).trim().to_string(), o)
    }

    /// Un `flock` factice qui exécute la commande sans verrou (les tests ne touchent jamais au verrou GPU).
    fn sans_verrou(&self) {
        fs::create_dir_all(self.dir.join("pbin")).unwrap();
        let p = self.dir.join("pbin/flock");
        fs::write(&p, "#!/usr/bin/env bash\nwhile [ $# -gt 0 ]; do case \"$1\" in -w) shift 2;; -*) shift;; *) shift; break;; esac; done\nexec \"$@\"\n").unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
    }
}

impl Drop for Banc {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

/// Binaire factice : `su2 test-hb` et `su2 gen` (déterministe ou non), pilotés par l'environnement.
fn faux_moteur(dossier: &Path) -> String {
    format!(
        r#"#!/usr/bin/env python3
import sys, os, json, random, hashlib
sys.path.insert(0, "{hist}")
import gh04_bain as g
a = sys.argv[1:]
def opt(n, d=None):
    return a[a.index(n) + 1] if n in a else d
if a[:2] == ["su2", "test-hb"]:
    alpha, n = float(a[2]), int(a[3])
    e1, e2, e3, e4 = g.exact(alpha)
    import math
    sd1 = math.sqrt(e2 - e1 ** 2) / math.sqrt(n)
    biais = float(os.environ.get("FAUX_BIAIS_Z", "0")) * sd1
    plafond = int(os.environ.get("FAUX_PLAFOND", "0"))
    print(f"TESTHB alpha={{alpha}} n={{n}} m1={{e1 + biais:.12f}} m2={{e2:.12f}} min=-0.99 max=0.99 plafond={{plafond}} backend=hip")
elif a[:2] == ["su2", "gen"]:
    out = opt("--outdir"); os.makedirs(out)
    seed = int(opt("--graine"))
    rnd = random.Random(seed if not os.environ.get("FAUX_NONDET") else os.urandom(8))
    with open(os.path.join(out, "measurements.csv"), "w") as f:
        f.write("measurement_index,total_steps,phase,plaquette,P0,P1,P2,P3\n")
        for i in range(5):
            f.write(f"{{i}},{{i}},sample,{{rnd.random():.17g}},0,0,0,0\n")
    open(os.path.join(out, "cfg_000010.npy"), "wb").write(bytes(rnd.getrandbits(8) for _ in range(256)))
    open(os.path.join(out, "manifest.json"), "w").write(json.dumps(dict(pid=os.getpid(), t=random.random(),
        git_head=os.environ.get("FAUX_GIT_HEAD", ""), binaire_sha256=os.environ.get("FAUX_SHA_BIN", "") or hashlib.sha256(open(sys.argv[0], "rb").read()).hexdigest(),
        source_hash=os.environ.get("FAUX_SOURCE_HASH", ""))))
    open(os.path.join(out, "COMPLETE"), "w").write("ok\n")
else:
    sys.exit(2)
"#,
        hist = dossier.display()
    )
}

#[test]
fn g_h04_passe_sur_un_bain_exact_et_rougit_sur_un_biais() {
    let b = Banc::nouveau("h04");
    b.sans_verrou();
    let bin = b.ecrire_exe("kevinotron", &faux_moteur(&hist()));
    let (v, _) = b.gate("G-H04.sh", &bin, &[]);
    assert!(v.starts_with("PASS G-H04"), "moteur exact : {v}");
    // mutant : biais de 6 σ sur m1 ⇒ |z| = 6 > 4 : la gate doit être ROUGE
    let (v, _) = b.gate("G-H04.sh", &bin, &[("FAUX_BIAIS_Z", "6")]);
    assert!(v.starts_with("FAIL G-H04"), "mutant non détecté : {v}");
    // plafond ≠ 0 ⇒ ROUGE
    let (v, _) = b.gate("G-H04.sh", &bin, &[("FAUX_PLAFOND", "3")]);
    assert!(v.starts_with("FAIL G-H04") && v.contains("plafond=3"), "{v}");
    println!("PASS cli_historique_g_h04");
}

#[test]
fn g_h04_sans_binaire_est_un_fail() {
    let b = Banc::nouveau("h04b");
    b.sans_verrou();
    let (v, _) = b.gate("G-H04.sh", Path::new("/nexiste/pas"), &[]);
    assert!(v.starts_with("FAIL G-H04") && v.contains("introuvable"), "{v}");
}

#[test]
fn g_h05_passe_si_deterministe_et_rougit_sinon() {
    let b = Banc::nouveau("h05");
    b.sans_verrou();
    let bin = b.ecrire_exe("kevinotron", &faux_moteur(&hist()));
    let (v, _) = b.gate("G-H05.sh", &bin, &[]);
    assert!(v.starts_with("PASS G-H05"), "{v}");
    let (v, _) = b.gate("G-H05.sh", &bin, &[("FAUX_NONDET", "1")]);
    assert!(
        v.starts_with("FAIL G-H05") && v.contains("diffèrent"),
        "mutant non déterministe non détecté : {v}"
    );
    println!("PASS cli_historique_g_h05");
}

#[test]
fn g_h02_inventaire_intact_puis_altere() {
    let b = Banc::nouveau("h02");
    let racine = b.dir.join("racine");
    let kvu = Path::new(env!("CARGO_MANIFEST_DIR"));
    fs::create_dir_all(racine.join("crates/kvu")).unwrap();
    let cp = |de: &Path, vers: &Path| {
        let o = Command::new("cp").arg("-a").arg(de).arg(vers).output().unwrap();
        assert!(o.status.success());
    };
    cp(&kvu.join("reference"), &racine.join("crates/kvu/reference"));
    cp(
        &kvu.join("SOURCES_IMPORTEES.md"),
        &racine.join("crates/kvu/SOURCES_IMPORTEES.md"),
    );
    let bin = Path::new("/inutile");
    let (v, _) = b.gate("G-H02.sh", bin, &[]);
    assert!(v.starts_with("PASS G-H02"), "{v}");
    // altération d'un octet de su2_hip.cpp ⇒ ROUGE
    let f = racine.join("crates/kvu/reference/su2_hip/su2_hip.cpp");
    let mut t = fs::read(&f).unwrap();
    t.push(b'\n');
    fs::write(&f, t).unwrap();
    let (v, _) = b.gate("G-H02.sh", bin, &[]);
    assert!(
        v.starts_with("FAIL G-H02") && v.contains("su2_hip"),
        "altération non détectée : {v}"
    );
    println!("PASS cli_historique_g_h02");
}

/// Recette `Unifiee` refaite ici (troisième implémentation, après Rust/T8 et Python) : sha256 de
/// json.dumps({chemin: sha256}, sort_keys=True) sur les fichiers listés.
fn hash_recette(racine: &Path, fichiers: &[&str]) -> String {
    let mut v: Vec<String> = fichiers.iter().map(|s| s.to_string()).collect();
    v.sort();
    let mut blob = String::from("{");
    for (i, f) in v.iter().enumerate() {
        let h = kvu::heritage::manifest::sha256_fichier(&racine.join(f)).unwrap();
        if i > 0 {
            blob.push_str(", ");
        }
        blob.push_str(&format!("\"{f}\": \"{h}\""));
    }
    blob.push('}');
    kvu::heritage::manifest::sha256_hex(blob.as_bytes())
}

fn arbre_g_h01(b: &Banc) -> (PathBuf, String) {
    let r = b.dir.join("racine");
    for d in ["src/hip", "crates/kvu/src", "crates/kvu/noyaux"] {
        fs::create_dir_all(r.join(d)).unwrap();
    }
    fs::write(
        r.join("src/manifest.rs"),
        "pub const FICHIERS_RACINE_UNIFIEE: &[&str] = &[\n    \"Cargo.toml\", \"Cargo.lock\",\n    \"crates/kvu/Cargo.toml\",\n];\n\
         pub const DOSSIERS_KVU_UNIFIEE: &[&str] = &[\"crates/kvu/src\", \"crates/kvu/noyaux\"];\n",
    )
    .unwrap();
    for (f, c) in [
        ("src/a.rs", "fn a() {}"),
        ("src/hip/b.hip", "// noyau b"),
        ("Cargo.toml", "[package]"),
        ("Cargo.lock", "# lock"),
        ("crates/kvu/Cargo.toml", "[package]"),
        ("crates/kvu/src/x.rs", "fn x() {}"),
        ("crates/kvu/noyaux/y.cu", "// y"),
    ] {
        fs::write(r.join(f), c).unwrap();
    }
    let h = hash_recette(
        &r,
        &[
            "src/manifest.rs",
            "src/a.rs",
            "src/hip/b.hip",
            "Cargo.toml",
            "Cargo.lock",
            "crates/kvu/Cargo.toml",
            "crates/kvu/src/x.rs",
            "crates/kvu/noyaux/y.cu",
        ],
    );
    (r, h)
}

#[test]
fn g_h01_binaire_egal_source_puis_mutants() {
    let b = Banc::nouveau("h01");
    let (_, h) = arbre_g_h01(&b);
    let head = "0123456789abcdef0123456789abcdef01234567";
    let bin = b.ecrire_exe("kevinotron", &faux_moteur(&hist()));
    let ok = [
        ("FAUX_GIT_HEAD", head),
        ("FAUX_SOURCE_HASH", h.as_str()),
        ("KVU_GIT_HEAD", head),
    ];
    let (v, o) = b.gate("G-H01.sh", &bin, &ok);
    assert!(
        v.starts_with("PASS G-H01"),
        "{v}\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
    // mutant 1 : mauvais git_head
    let (v, _) = b.gate(
        "G-H01.sh",
        &bin,
        &[
            ("FAUX_GIT_HEAD", "deadbeef"),
            ("FAUX_SOURCE_HASH", &h),
            ("KVU_GIT_HEAD", head),
        ],
    );
    assert!(v.starts_with("FAIL G-H01") && v.contains("git_head"), "{v}");
    // mutant 2 : source_hash d'un autre arbre
    let (v, _) = b.gate(
        "G-H01.sh",
        &bin,
        &[
            ("FAUX_GIT_HEAD", head),
            ("FAUX_SOURCE_HASH", &"0".repeat(64)),
            ("KVU_GIT_HEAD", head),
        ],
    );
    assert!(v.starts_with("FAIL G-H01") && v.contains("source_hash"), "{v}");
    // mutant 3 : sha du binaire faux
    let (v, _) = b.gate(
        "G-H01.sh",
        &bin,
        &[
            ("FAUX_GIT_HEAD", head),
            ("FAUX_SOURCE_HASH", &h),
            ("FAUX_SHA_BIN", &"1".repeat(64)),
            ("KVU_GIT_HEAD", head),
        ],
    );
    assert!(v.starts_with("FAIL G-H01") && v.contains("binaire_sha256"), "{v}");
    // mutant 4 : une source modifiée après la compilation ⇒ le recalcul diffère du source_hash embarqué
    fs::write(b.dir.join("racine/crates/kvu/src/x.rs"), "fn x() { /* modifié */ }").unwrap();
    let (v, _) = b.gate("G-H01.sh", &bin, &ok);
    assert!(v.starts_with("FAIL G-H01") && v.contains("source_hash"), "{v}");
    println!("PASS cli_historique_g_h01");
}

#[test]
fn les_treize_scripts_existent_et_sont_syntaxiquement_valides() {
    for n in 1..=13 {
        let p = hist().join(format!("G-H{n:02}.sh"));
        assert!(p.is_file(), "{} absent", p.display());
        let o = Command::new("bash").arg("-n").arg(&p).output().unwrap();
        assert!(
            o.status.success(),
            "{} : {}",
            p.display(),
            String::from_utf8_lossy(&o.stderr)
        );
        // une gate de moins de 10 min par fichier : chaque script annonce sa gate en tête de fichier
        let t = fs::read_to_string(&p).unwrap();
        assert!(t.contains(&format!("GATE=G-H{n:02}")), "{} sans GATE=", p.display());
    }
    println!("PASS cli_historique_13_scripts");
}

fn faux_build(b: &Banc, journal: &str, rc: i32) -> PathBuf {
    let r = b.dir.join("racine");
    fs::create_dir_all(r.join("scripts")).unwrap();
    let p = r.join("scripts/build_auto.sh");
    fs::write(
        &p,
        format!(
            "#!/usr/bin/env bash\nmkdir -p \"$CARGO_TARGET_DIR\"\ncat > \"$CARGO_TARGET_DIR/build_auto.log\" <<'EOF'\n{journal}\nEOF\nexit {rc}\n"
        ),
    )
    .unwrap();
    b.dir.join("valide_test")
}

#[test]
fn g_h03_zero_avertissement_dans_kvu_sinon_rouge() {
    let b = Banc::nouveau("h03");
    let bin = Path::new("/inutile");
    // build propre côté kvu, avertissements hérités du moteur historique : PASS, et ils sont comptés
    let cible = faux_build(
        &b,
        "warning: unused variable: `x`\n  --> src/gpu/foo.rs:10:5\n\nwarning: `kevinotron` (bin) generated 1 warning",
        0,
    );
    let env = [("KVU_TARGET_VALIDE", cible.to_str().unwrap()), ("KVU_GIT_HEAD", "abc")];
    let (v, o) = b.gate("G-H03.sh", bin, &env);
    assert!(
        v.starts_with("PASS G-H03") && v.contains("1 avertissements hérités"),
        "{v}\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(!cible.exists(), "la cible jetable doit être supprimée");
    // un avertissement dans crates/kvu ⇒ ROUGE
    let cible = faux_build(
        &b,
        "warning: unused import: `Foo`\n  --> crates/kvu/src/cli/mod.rs:3:5\n",
        0,
    );
    let env = [("KVU_TARGET_VALIDE", cible.to_str().unwrap()), ("KVU_GIT_HEAD", "abc")];
    let (v, _) = b.gate("G-H03.sh", bin, &env);
    assert!(v.starts_with("FAIL G-H03") && v.contains("code unifié"), "{v}");
    // rc ≠ 0 ⇒ ROUGE
    let cible = faux_build(&b, "error: boom", 1);
    let env = [("KVU_TARGET_VALIDE", cible.to_str().unwrap()), ("KVU_GIT_HEAD", "abc")];
    let (v, _) = b.gate("G-H03.sh", bin, &env);
    assert!(v.starts_with("FAIL G-H03") && v.contains("rc=1"), "{v}");
    // une cible dont le nom ne commence pas par « valide » est refusée (elle serait supprimée)
    let (v, _) = b.gate(
        "G-H03.sh",
        bin,
        &[("KVU_TARGET_VALIDE", b.dir.join("important").to_str().unwrap())],
    );
    assert!(v.starts_with("FAIL G-H03") && v.contains("refusée"), "{v}");
    println!("PASS cli_historique_g_h03");
}

#[test]
fn le_recalcul_python_du_source_hash_egale_la_recette_unifiee_de_t8() {
    // Deux implémentations indépendantes (Rust de T8 : heritage::manifest, et Python de T9) sur l'arbre RÉEL du dépôt.
    let racine = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    // WP1 (dépôt autonome) : la recette du dépôt réel est `kvu-source/1` (Rust : stockage::recette ; Python : source_hash_kvu.py).
    let (rust, _) = kvu::stockage::recette::hash_kvu_source_1(&racine).unwrap();
    let o = Command::new("python3")
        .arg("-I")
        .arg(hist().join("source_hash_kvu.py"))
        .arg(&racine)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let py = String::from_utf8_lossy(&o.stdout).trim().to_string();
    assert_eq!(py, rust, "recalcul Python et recette Unifiee (Rust) divergent");
    println!("PASS cli_historique_source_hash_rust_eq_python");
}
