//! Mutant de la réserve 3 de T0-VERIF (lot t1-prelude) : le code retour de G-H01 suit son verdict.
//! Binaire factice et arbre jouet autonomes, sans exécution GPU.

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
            .env("KVU_RACINE", self.dir.join("racine"));
        for (k, v) in env {
            c.env(k, v);
        }
        let o = c.output().expect("lancement de la gate");
        let v = String::from_utf8_lossy(&o.stdout);
        assert_eq!(v.lines().count(), 1, "{v}");
        (v.trim().to_string(), o)
    }
}

impl Drop for Banc {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

/// Binaire factice limité à `su2 gen` : écrit le manifeste contrôlé par G-H01.
fn faux_moteur() -> &'static str {
    r#"#!/usr/bin/env python3
import sys, os, json, hashlib
a = sys.argv[1:]
def opt(n, d=None):
    return a[a.index(n) + 1] if n in a else d
if a[:2] == ["su2", "gen"]:
    if os.environ.get("FAUX_ECHEC_GEN") == "1":
        sys.exit(3)
    out = opt("--outdir"); os.makedirs(out)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(dict(
            git_head=os.environ.get("FAUX_GIT_HEAD", ""),
            binaire_sha256=hashlib.sha256(open(sys.argv[0], "rb").read()).hexdigest(),
            source_hash=os.environ.get("FAUX_SOURCE_HASH", "")), f)
else:
    sys.exit(2)
"#
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
fn g_h01_code_retour_nul_sur_pass_non_nul_sur_fail() {
    let b = Banc::nouveau("h01_rc");
    let (_, h) = arbre_g_h01(&b);
    let head = "0123456789abcdef0123456789abcdef01234567";
    let bin = b.ecrire_exe("kevinotron", faux_moteur());
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
    assert_eq!(o.status.code(), Some(0), "{v}");

    // Mutant : git_head ne correspond pas au commit attendu.
    let (v, o) = b.gate(
        "G-H01.sh",
        &bin,
        &[
            ("FAUX_GIT_HEAD", "deadbeef"),
            ("FAUX_SOURCE_HASH", &h),
            ("KVU_GIT_HEAD", head),
        ],
    );
    assert!(v.starts_with("FAIL G-H01") && v.contains("git_head"), "{v}");
    assert_eq!(o.status.code(), Some(1), "{v}");

    // Mutant : source_hash d'un autre arbre.
    let (v, o) = b.gate(
        "G-H01.sh",
        &bin,
        &[
            ("FAUX_GIT_HEAD", head),
            ("FAUX_SOURCE_HASH", &"0".repeat(64)),
            ("KVU_GIT_HEAD", head),
        ],
    );
    assert!(v.starts_with("FAIL G-H01") && v.contains("source_hash"), "{v}");
    assert_eq!(o.status.code(), Some(1), "{v}");

    // Le `exit 0` historique de binaire_ok doit également devenir un échec.
    let (v, o) = b.gate("G-H01.sh", Path::new("/nexiste/pas"), &ok);
    assert!(v.starts_with("FAIL G-H01") && v.contains("introuvable"), "{v}");
    assert_eq!(o.status.code(), Some(1), "{v}");

    // Mutant : su2 gen échoue avant toute écriture.
    let mut echec = ok.to_vec();
    echec.push(("FAUX_ECHEC_GEN", "1"));
    let (v, o) = b.gate("G-H01.sh", &bin, &echec);
    assert!(v.starts_with("FAIL G-H01"), "{v}");
    assert_eq!(o.status.code(), Some(1), "{v}");
}
