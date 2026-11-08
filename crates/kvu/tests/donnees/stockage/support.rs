//! Outils communs des tests de stockage (T8). Inclus par `#[path = "donnees/stockage/support.rs"] mod support;`.
//! Tout se passe sous `$KVU_SCRATCH` s'il est défini, sinon sous le dossier temporaire que cargo réserve aux tests
//! d'intégration (`CARGO_TARGET_TMPDIR`, dans `target/`, sur disque et non dans un /tmp éventuellement en RAM),
//! nettoyé à la destruction.
#![allow(dead_code)]

use kvu::commun::ChampHote;
use kvu::heritage::manifest::sha256_fichier;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const SCRATCH_DEFAUT: &str = concat!(env!("CARGO_TARGET_TMPDIR"), "/stockage");

/// Répertoire de travail unique, supprimé à la destruction (sauf KVU_GARDER_SCRATCH=1).
pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn nouveau(nom: &str) -> Scratch {
        static N: AtomicUsize = AtomicUsize::new(0);
        let base = std::env::var_os("KVU_SCRATCH")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(SCRATCH_DEFAUT));
        let d = base.join(format!(
            "{nom}_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).expect("création du scratch");
        Scratch(d)
    }
    pub fn p(&self, rel: &str) -> PathBuf {
        self.0.join(rel)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if std::env::var_os("KVU_GARDER_SCRATCH").is_none() {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

pub fn donnees(nom: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/donnees/stockage")
        .join(nom)
}

pub fn racine_depot() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("racine du dépôt")
}

/// Les 24 unités de Hurwitz (même ordre que gen_donnees.py) : normes exactement 1, aucun arrondi.
pub fn hurwitz() -> Vec<[f64; 4]> {
    let mut u = Vec::new();
    for slot in 0..4 {
        for s in [1.0, -1.0] {
            let mut v = [0.0; 4];
            v[slot] = s;
            u.push(v);
        }
    }
    for b in 0..16u32 {
        let mut v = [0.0; 4];
        for (k, x) in v.iter_mut().enumerate() {
            *x = if (b >> k) & 1 == 0 { 0.5 } else { -0.5 };
        }
        u.push(v);
    }
    u
}

/// Champ de Hurwitz (celui de gen_donnees.py) de côté `l`.
pub fn champ_hurwitz(l: usize) -> ChampHote {
    let u = hurwitz();
    let n = 4 * l.pow(4);
    let mut q = Vec::with_capacity(4 * n);
    for i in 0..n {
        q.extend_from_slice(&u[(i * 7 + 3) % 24]);
    }
    ChampHote { l, q }
}

/// Champ pseudo-aléatoire de quaternions unitaires (xorshift64*, Box-Muller épargné : gaussiennes par somme),
/// normalisés en double précision : |q| = 1 à quelques 1e-16.
pub fn champ_aleatoire(l: usize, graine: u64) -> ChampHote {
    let mut etat = graine.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut suivant = move || {
        etat ^= etat >> 12;
        etat ^= etat << 25;
        etat ^= etat >> 27;
        (etat.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let n = 4 * l.pow(4);
    let mut q = Vec::with_capacity(4 * n);
    for _ in 0..n {
        let mut v = [0.0f64; 4];
        for x in v.iter_mut() {
            *x = suivant() + suivant() + suivant();
        }
        let norme = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in v {
            q.push(x / norme);
        }
    }
    ChampHote { l, q }
}

/// sha256 d'un fichier par le binaire système `sha256sum` (indépendant de notre implémentation).
pub fn sha256sum(chemin: &Path) -> String {
    let s = Command::new("sha256sum").arg(chemin).output().expect("sha256sum");
    assert!(s.status.success(), "sha256sum {} : {:?}", chemin.display(), s);
    String::from_utf8_lossy(&s.stdout)
        .split_whitespace()
        .next()
        .unwrap()
        .to_string()
}

/// Empreinte de tout un arbre SANS suivre les liens : {chemin relatif: sha256 | « -> cible » pour un lien | « DIR »}.
pub fn empreinte_arbre(racine: &Path) -> BTreeMap<String, String> {
    fn rec(base: &Path, dir: &Path, sortie: &mut BTreeMap<String, String>) {
        let mut v: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        v.sort_by_key(|e| e.file_name());
        for e in v {
            let p = e.path();
            let rel = p.strip_prefix(base).unwrap().to_string_lossy().into_owned();
            let ft = e.file_type().unwrap();
            if ft.is_symlink() {
                sortie.insert(rel, format!("-> {}", fs::read_link(&p).unwrap().display()));
            } else if ft.is_dir() {
                sortie.insert(rel, "DIR".into());
                rec(base, &p, sortie);
            } else {
                sortie.insert(rel, sha256_fichier(&p).unwrap());
            }
        }
    }
    let mut m = BTreeMap::new();
    rec(racine, racine, &mut m);
    m
}

/// Vrai si python3 et numpy sont disponibles.
pub fn numpy_disponible() -> bool {
    Command::new("python3")
        .args(["-c", "import numpy"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Fixe la date de modification d'un fichier ou répertoire (il y a `age_s` secondes).
pub fn vieillir(chemin: &Path, age_s: u64) {
    let t = std::time::SystemTime::now() - std::time::Duration::from_secs(age_s);
    let f = fs::File::open(chemin).expect("ouverture pour vieillir");
    f.set_modified(t).expect("set_modified");
}

/// Vieillit récursivement tout ce qui est sous `dir` (fichiers puis répertoires), sans suivre de lien.
pub fn vieillir_arbre(dir: &Path, age_s: u64) {
    for e in fs::read_dir(dir).unwrap() {
        let e = e.unwrap();
        let ft = e.file_type().unwrap();
        if ft.is_dir() {
            vieillir_arbre(&e.path(), age_s);
        } else if ft.is_file() {
            vieillir(&e.path(), age_s);
        }
    }
    vieillir(dir, age_s);
}
