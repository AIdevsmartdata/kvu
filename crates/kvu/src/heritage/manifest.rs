//! manifest.rs — manifeste de provenance universel (lot K8, PLAN_V3 26/09).
//!
//! Module SANS dépendance hors `std` : il est compilé trois fois,
//! * dans la lib (`pub mod manifest;`) pour les autres binaires (su2_gen de
//!   K1, block_config, …) ;
//! * dans le binaire kevinotron (`mod manifest;` de main.rs), où registry.rs
//!   l'utilise pour TOUTES les invocations ;
//! * dans build.rs (`#[path = "src/manifest.rs"]`), qui y prend le SHA-256 et
//!   la recette de `source_hash` : une seule implémentation pour l'injection
//!   et pour la vérification.
//!
//! Contrat d'un répertoire de run (`RepertoireRun`) :
//! 1. le répertoire ne doit PAS exister (fichier, répertoire ou lien) : sinon
//!    refus, jamais d'écrasement ni de fusion ;
//! 2. `manifest.json` est écrit à l'ouverture (statut `en_cours`) : mode réel,
//!    argv complet, source_hash, sha256 du binaire, hôte, pid, cwd ;
//! 3. à la fin, `manifest.json` est réécrit (statut, rc, fin) PUIS, si et
//!    seulement si rc = 0, `COMPLETE` est écrit EN DERNIER (écriture atomique :
//!    fichier temporaire, fsync, rename, fsync du répertoire) ;
//! 4. un processus tué (kill -9, SIGTERM, abort) ou sorti avec rc ≠ 0 ne
//!    laisse jamais de `COMPLETE`.
#![allow(dead_code)]

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), std seulement.
// ---------------------------------------------------------------------------

const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    nbuf: usize,
    len: u64,
}

impl Sha256 {
    pub fn new() -> Self {
        Sha256 {
            h: [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19],
            buf: [0; 64], nbuf: 0, len: 0,
        }
    }

    fn bloc(h: &mut [u32; 8], b: &[u8]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let (mut a, mut bb, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K256[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & bb) ^ (a & c) ^ (bb & c);
            let t2 = s0.wrapping_add(maj);
            hh = g; g = f; f = e; e = d.wrapping_add(t1);
            d = c; c = bb; bb = a; a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a); h[1] = h[1].wrapping_add(bb);
        h[2] = h[2].wrapping_add(c); h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e); h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g); h[7] = h[7].wrapping_add(hh);
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u64);
        if self.nbuf > 0 {
            let n = (64 - self.nbuf).min(data.len());
            self.buf[self.nbuf..self.nbuf + n].copy_from_slice(&data[..n]);
            self.nbuf += n;
            data = &data[n..];
            if self.nbuf == 64 {
                let b = self.buf;
                Self::bloc(&mut self.h, &b);
                self.nbuf = 0;
            }
        }
        while data.len() >= 64 {
            Self::bloc(&mut self.h, &data[..64]);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.nbuf = data.len();
        }
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let bits = self.len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.nbuf != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());
        let mut out = [0u8; 32];
        for i in 0..8 {
            out[4 * i..4 * i + 4].copy_from_slice(&self.h[i].to_be_bytes());
        }
        out
    }
}

pub fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(2 * b.len());
    for x in b {
        s.push_str(&format!("{:02x}", x));
    }
    s
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex(&h.finalize())
}

pub fn sha256_fichier(path: &Path) -> io::Result<String> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

// ---------------------------------------------------------------------------
// source_hash : recettes (identiques à hashlib + json.dumps(sort_keys=True)).
// ---------------------------------------------------------------------------

/// Recette de l'empreinte des sources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recette {
    /// `build.sh` (Astra, K0) : src/**/*.rs + Cargo.toml + Cargo.lock +
    /// {e6,e7,f4}_generators.bin. Conservée pour la compatibilité avec les
    /// manifestes de su2_pairs ; AVEUGLE aux 49 noyaux .hip/.cu embarqués par
    /// include_str!.
    BuildSh,
    /// K8 : TOUS les fichiers réguliers de src/ (noyaux .hip/.cu compris) +
    /// Cargo.toml + Cargo.lock + build.rs + {e6,e7,f4}_generators.bin.
    Complete,
    /// Kevinotron unifié (T8) : fichiers de `Complete` + TOUS les fichiers réguliers de
    /// `crates/kvu/src/**` et `crates/kvu/noyaux/**` + `crates/kvu/Cargo.toml` +
    /// `crates/kvu/build.rs` (chemins relatifs à la racine du dépôt). Couvre donc les 11 noyaux
    /// GPU embarqués du crate `kvu`, que ni BuildSh ni Complete ne voient.
    Unifiee,
}

pub const FICHIERS_RACINE_BUILD_SH: &[&str] =
    &["Cargo.toml", "Cargo.lock", "e6_generators.bin", "e7_generators.bin", "f4_generators.bin"];
pub const FICHIERS_RACINE_COMPLETE: &[&str] =
    &["Cargo.toml", "Cargo.lock", "build.rs", "e6_generators.bin", "e7_generators.bin", "f4_generators.bin"];
/// Fichiers racine de la recette `Unifiee` : ceux de `Complete` + le manifeste et le script de
/// construction du crate `kvu` (les dossiers `crates/kvu/{src,noyaux}` sont listés à part).
pub const FICHIERS_RACINE_UNIFIEE: &[&str] = &[
    "Cargo.toml", "Cargo.lock", "build.rs", "e6_generators.bin", "e7_generators.bin", "f4_generators.bin",
    "crates/kvu/Cargo.toml", "crates/kvu/build.rs",
];
/// Dossiers de `kvu` ajoutés par la recette `Unifiee` (tous fichiers réguliers, récursif).
pub const DOSSIERS_KVU_UNIFIEE: &[&str] = &["crates/kvu/src", "crates/kvu/noyaux"];

fn lister_src(dir: &Path, rel: &str, seulement_rs: bool, out: &mut Vec<String>) -> io::Result<()> {
    for e in fs::read_dir(dir)? {
        let e = e?;
        let nom = e.file_name().to_string_lossy().into_owned();
        let r = format!("{}/{}", rel, nom);
        let ft = e.file_type()?;
        if ft.is_dir() {
            lister_src(&e.path(), &r, seulement_rs, out)?;
        } else if ft.is_file() {
            if !seulement_rs || nom.ends_with(".rs") {
                out.push(r);
            }
        } else if ft.is_symlink() {
            // Aucun lien dans src/ aujourd'hui ; en cas d'apparition on le
            // hache par son contenu pointé (comportement de rglob pour un
            // fichier) plutôt que de l'ignorer silencieusement.
            if !seulement_rs || nom.ends_with(".rs") {
                if e.path().is_file() { out.push(r); }
            }
        }
    }
    Ok(())
}

/// Liste triée des chemins relatifs à `racine` couverts par la recette.
pub fn fichiers_recette(racine: &Path, recette: Recette) -> io::Result<Vec<String>> {
    let mut v = Vec::new();
    lister_src(&racine.join("src"), "src", recette == Recette::BuildSh, &mut v)?;
    let rac = match recette {
        Recette::BuildSh => FICHIERS_RACINE_BUILD_SH,
        Recette::Complete => FICHIERS_RACINE_COMPLETE,
        Recette::Unifiee => FICHIERS_RACINE_UNIFIEE,
    };
    if recette == Recette::Unifiee {
        for d in DOSSIERS_KVU_UNIFIEE {
            lister_src(&racine.join(d), d, false, &mut v)?;
        }
    }
    for f in rac {
        v.push((*f).to_string());
    }
    v.sort();
    Ok(v)
}

/// Chaîne JSON à la Python (`json.dumps`, ensure_ascii=True).
pub fn json_str_py(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            '\u{08}' => o.push_str("\\b"),
            '\u{0c}' => o.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7f => {
                let mut b = [0u16; 2];
                for u in c.encode_utf16(&mut b) {
                    o.push_str(&format!("\\u{:04x}", u));
                }
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// `sha256(json.dumps({chemin: sha256(fichier)}, sort_keys=True))`, et le
/// nombre de fichiers. Tout fichier listé DOIT exister (sinon erreur).
pub fn hash_arbre_source(racine: &Path, recette: Recette) -> io::Result<(String, usize)> {
    let fichiers = fichiers_recette(racine, recette)?;
    let mut blob = String::from("{");
    for (i, f) in fichiers.iter().enumerate() {
        let h = sha256_fichier(&racine.join(f))
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {}", f, e)))?;
        if i > 0 { blob.push_str(", "); }
        blob.push_str(&json_str_py(f));
        blob.push_str(": ");
        blob.push_str(&json_str_py(&h));
    }
    blob.push('}');
    Ok((sha256_hex(blob.as_bytes()), fichiers.len()))
}

/// Empreinte injectée à la compilation par build.rs (recette `Complete`).
pub const SOURCE_HASH: Option<&str> = option_env!("KEVINOTRON_SOURCE_HASH");
/// Empreinte selon la recette historique de build.sh (compatibilité).
pub const SOURCE_HASH_BUILD_SH: Option<&str> = option_env!("KEVINOTRON_SOURCE_HASH_BUILD_SH");
/// Empreinte de la recette `Unifiee` (T8), injectée par build.rs (racine) et par crates/kvu/build.rs.
pub const SOURCE_HASH_UNIFIE: Option<&str> = option_env!("KEVINOTRON_SOURCE_HASH_UNIFIE");
/// Nombre de fichiers couverts par SOURCE_HASH.
pub const SOURCE_NFICHIERS: Option<&str> = option_env!("KEVINOTRON_SOURCE_NFICHIERS");

// ---------------------------------------------------------------------------
// JSON minimal (écriture seulement).
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Valeur {
    Texte(String),
    /// nombre déjà formaté (entier ou flottant fini)
    Nombre(String),
    Booleen(bool),
    Nul,
    ListeTextes(Vec<String>),
}

impl Valeur {
    pub fn texte<S: Into<String>>(s: S) -> Self { Valeur::Texte(s.into()) }
    pub fn entier<N: Into<i128>>(n: N) -> Self { Valeur::Nombre(n.into().to_string()) }
    pub fn flottant(x: f64) -> Self {
        if x.is_finite() { Valeur::Nombre(format!("{:?}", x)) } else { Valeur::Texte(format!("{}", x)) }
    }
    pub fn option_texte(o: Option<&str>) -> Self {
        match o { Some(s) => Valeur::Texte(s.to_string()), None => Valeur::Nul }
    }
    fn json(&self) -> String {
        match self {
            Valeur::Texte(s) => json_str_py(s),
            Valeur::Nombre(n) => n.clone(),
            Valeur::Booleen(b) => b.to_string(),
            Valeur::Nul => "null".to_string(),
            Valeur::ListeTextes(v) => {
                let items: Vec<String> = v.iter().map(|s| json_str_py(s)).collect();
                format!("[{}]", items.join(", "))
            }
        }
    }
}

/// Objet JSON à clés ordonnées (ordre d'insertion ; une clé réécrite garde sa place).
#[derive(Clone, Debug, Default)]
pub struct Champs(pub Vec<(String, Valeur)>);

impl Champs {
    pub fn new() -> Self { Champs(Vec::new()) }
    pub fn set(&mut self, k: &str, v: Valeur) {
        if let Some(e) = self.0.iter_mut().find(|(kk, _)| kk == k) {
            e.1 = v;
        } else {
            self.0.push((k.to_string(), v));
        }
    }
    pub fn get(&self, k: &str) -> Option<&Valeur> {
        self.0.iter().find(|(kk, _)| kk == k).map(|(_, v)| v)
    }
    pub fn json(&self) -> String {
        let mut s = String::from("{\n");
        for (i, (k, v)) in self.0.iter().enumerate() {
            s.push_str("  ");
            s.push_str(&json_str_py(k));
            s.push_str(": ");
            s.push_str(&v.json());
            if i + 1 < self.0.len() { s.push(','); }
            s.push('\n');
        }
        s.push_str("}\n");
        s
    }
}

// ---------------------------------------------------------------------------
// Écritures atomiques et répertoire neuf.
// ---------------------------------------------------------------------------

fn fsync_repertoire(dir: &Path) {
    if let Ok(f) = fs::File::open(dir) {
        let _ = f.sync_all();
    }
}

/// Écrit `path` de façon atomique : `.<nom>.tmp.<pid>`, fsync, rename, fsync(dir).
pub fn ecrire_atomique(path: &Path, contenu: &[u8]) -> io::Result<()> {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let nom = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = dir.join(format!(".{}.tmp.{}", nom, std::process::id()));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(contenu)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    fsync_repertoire(dir);
    Ok(())
}

#[derive(Debug)]
pub enum ErreurRepertoire {
    /// chemin vide
    Vide,
    /// le chemin existe déjà (répertoire, fichier ou lien) : refus d'écraser
    Existe(PathBuf),
    Io(PathBuf, io::Error),
}

impl std::fmt::Display for ErreurRepertoire {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErreurRepertoire::Vide => write!(f, "chemin de repertoire vide"),
            ErreurRepertoire::Existe(p) => write!(f,
                "REFUS D'ECRASER : {} existe deja (un repertoire de run doit etre neuf ; \
                 choisir un autre chemin ou deplacer l'ancien)", p.display()),
            ErreurRepertoire::Io(p, e) => write!(f, "{} : {}", p.display(), e),
        }
    }
}

/// Crée `path` qui ne doit PAS exister. Les parents sont créés au besoin ; le
/// dernier composant l'est par `create_dir` (exclusif : une course perdue rend
/// aussi `Existe`).
pub fn creer_repertoire_neuf(path: &Path) -> Result<(), ErreurRepertoire> {
    if path.as_os_str().is_empty() {
        return Err(ErreurRepertoire::Vide);
    }
    if fs::symlink_metadata(path).is_ok() {
        return Err(ErreurRepertoire::Existe(path.to_path_buf()));
    }
    if let Some(p) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(p).map_err(|e| ErreurRepertoire::Io(p.to_path_buf(), e))?;
    }
    match fs::create_dir(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists =>
            Err(ErreurRepertoire::Existe(path.to_path_buf())),
        Err(e) => Err(ErreurRepertoire::Io(path.to_path_buf(), e)),
    }
}

// ---------------------------------------------------------------------------
// Répertoire de run : manifest.json à l'ouverture, COMPLETE en dernier.
// ---------------------------------------------------------------------------

pub const NOM_MANIFESTE: &str = "manifest.json";
pub const NOM_COMPLETE: &str = "COMPLETE";
pub const SCHEMA: &str = "kevinotron-provenance/1";

pub fn maintenant_unix() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

pub fn nom_hote() -> String {
    fs::read_to_string("/proc/sys/kernel/hostname")
        .or_else(|_| fs::read_to_string("/etc/hostname"))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "inconnu".to_string())
}

/// Champs communs à tout manifeste : schéma, moteur, source_hash, binaire,
/// argv, hôte, pid, cwd, début, statut `en_cours`.
pub fn champs_de_base(moteur: &str, version: &str, argv: &[String]) -> Champs {
    let mut c = Champs::new();
    c.set("schema", Valeur::texte(SCHEMA));
    c.set("moteur", Valeur::texte(moteur));
    c.set("version", Valeur::texte(version));
    c.set("source_hash", Valeur::option_texte(SOURCE_HASH));
    c.set("source_hash_recette", Valeur::texte(
        "sha256(json.dumps({chemin: sha256}, sort_keys=True)) sur src/** (tous fichiers) \
         + Cargo.toml + Cargo.lock + build.rs + {e6,e7,f4}_generators.bin ; injecte par build.rs"));
    c.set("source_hash_nfichiers", Valeur::option_texte(SOURCE_NFICHIERS));
    c.set("source_hash_build_sh", Valeur::option_texte(SOURCE_HASH_BUILD_SH));
    let exe = std::env::current_exe().ok();
    c.set("binaire", Valeur::option_texte(exe.as_ref().map(|p| p.to_string_lossy()).as_deref()));
    c.set("binaire_sha256", Valeur::option_texte(
        exe.as_ref().and_then(|p| sha256_fichier(p).ok()).as_deref()));
    c.set("argv", Valeur::ListeTextes(argv.to_vec()));
    c.set("cwd", Valeur::option_texte(
        std::env::current_dir().ok().map(|p| p.to_string_lossy().into_owned()).as_deref()));
    c.set("hote", Valeur::texte(nom_hote()));
    c.set("pid", Valeur::entier(std::process::id()));
    c.set("debut_unix", Valeur::flottant(maintenant_unix()));
    c.set("statut", Valeur::texte("en_cours"));
    c.set("rc", Valeur::Nul);
    c.set("fin_unix", Valeur::Nul);
    c
}

/// Un répertoire de run neuf, avec son manifeste.
pub struct RepertoireRun {
    pub dir: PathBuf,
    pub champs: Champs,
    termine: bool,
}

impl RepertoireRun {
    /// Refuse un chemin existant ; crée le répertoire ; écrit manifest.json
    /// (statut en_cours). Aucun COMPLETE n'est écrit ici.
    pub fn ouvrir(dir: &Path, champs: Champs) -> Result<Self, ErreurRepertoire> {
        creer_repertoire_neuf(dir)?;
        let r = RepertoireRun { dir: dir.to_path_buf(), champs, termine: false };
        r.ecrire_manifeste().map_err(|e| ErreurRepertoire::Io(r.manifeste(), e))?;
        Ok(r)
    }

    pub fn manifeste(&self) -> PathBuf { self.dir.join(NOM_MANIFESTE) }
    pub fn complete(&self) -> PathBuf { self.dir.join(NOM_COMPLETE) }

    pub fn ecrire_manifeste(&self) -> io::Result<()> {
        ecrire_atomique(&self.manifeste(), self.champs.json().as_bytes())
    }

    /// Met à jour un champ et réécrit le manifeste (tant que le run n'est pas terminé).
    pub fn maj(&mut self, k: &str, v: Valeur) -> io::Result<()> {
        self.champs.set(k, v);
        if self.termine { return Ok(()); }
        self.ecrire_manifeste()
    }

    /// Clôture : manifeste final (statut, rc, fin) puis, SEULEMENT si rc = 0,
    /// COMPLETE en dernier. Idempotent.
    pub fn terminer(&mut self, rc: i32) -> io::Result<()> {
        self.terminer_interne(rc, None)
    }

    /// Clôture en échec quel que soit rc (ex. panique observée puis rc = 0) :
    /// jamais de COMPLETE.
    pub fn terminer_echec(&mut self, rc: i32, raison: &str) -> io::Result<()> {
        self.terminer_interne(rc, Some(raison))
    }

    fn terminer_interne(&mut self, rc: i32, raison: Option<&str>) -> io::Result<()> {
        if self.termine { return Ok(()); }
        self.termine = true;
        let reussi = rc == 0 && raison.is_none();
        self.champs.set("statut", Valeur::texte(if reussi { "termine" } else { "echec" }));
        self.champs.set("rc", Valeur::entier(rc));
        if let Some(r) = raison {
            self.champs.set("raison_echec", Valeur::texte(r));
        }
        self.champs.set("fin_unix", Valeur::flottant(maintenant_unix()));
        let m = self.champs.json();
        ecrire_atomique(&self.manifeste(), m.as_bytes())?;
        if reussi {
            let c = format!("rc=0\nmanifest_sha256={}\n", sha256_hex(m.as_bytes()));
            ecrire_atomique(&self.complete(), c.as_bytes())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(nom: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("k8_manifest_{}_{}", nom, std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn sha256_vecteurs_nist() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        let mut h = Sha256::new();
        let a = [b'a'; 1000];
        for _ in 0..1000 { h.update(&a); }
        assert_eq!(hex(&h.finalize()), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
        // découpage arbitraire = même empreinte
        let data: Vec<u8> = (0..10_000u32).map(|i| (i * 7 + 3) as u8).collect();
        let mut h2 = Sha256::new();
        for ch in data.chunks(37) { h2.update(ch); }
        assert_eq!(hex(&h2.finalize()), sha256_hex(&data));
        println!("PASS sha256_vecteurs_nist");
    }

    #[test]
    fn json_str_comme_python() {
        assert_eq!(json_str_py("a\"b\\c\n"), "\"a\\\"b\\\\c\\n\"");
        assert_eq!(json_str_py("é"), "\"\\u00e9\"");
        assert_eq!(json_str_py("𝄞"), "\"\\ud834\\udd1e\"");
        assert_eq!(json_str_py("\u{1}"), "\"\\u0001\"");
        println!("PASS json_str_comme_python");
    }

    #[test]
    fn repertoire_existant_refuse() {
        let d = tmp("existe");
        fs::create_dir_all(&d).unwrap();
        match RepertoireRun::ouvrir(&d, Champs::new()) {
            Err(ErreurRepertoire::Existe(_)) => {}
            _ => panic!("un repertoire existant doit etre refuse"),
        }
        assert!(!d.join(NOM_MANIFESTE).exists());
        let f = d.join("fichier");
        fs::write(&f, b"x").unwrap();
        assert!(matches!(creer_repertoire_neuf(&f), Err(ErreurRepertoire::Existe(_))));
        assert_eq!(fs::read(&f).unwrap(), b"x");
        assert!(matches!(creer_repertoire_neuf(Path::new("")), Err(ErreurRepertoire::Vide)));
        fs::remove_dir_all(&d).unwrap();
        println!("PASS repertoire_existant_refuse");
    }

    #[test]
    fn complete_seulement_apres_rc0() {
        let base = tmp("cycle");
        let ok = base.join("ok");
        let mut r = RepertoireRun::ouvrir(&ok, champs_de_base("test", "0", &["x".into()])).unwrap();
        assert!(r.manifeste().exists());
        assert!(!r.complete().exists(), "COMPLETE ne doit pas exister avant terminer()");
        r.maj("groupe", Valeur::texte("su2")).unwrap();
        assert!(!r.complete().exists());
        r.terminer(0).unwrap();
        let m = fs::read_to_string(r.manifeste()).unwrap();
        let c = fs::read_to_string(r.complete()).unwrap();
        assert!(m.contains("\"statut\": \"termine\"") && m.contains("\"rc\": 0"));
        assert_eq!(c, format!("rc=0\nmanifest_sha256={}\n", sha256_hex(m.as_bytes())));
        // COMPLETE est plus récent (ou égal) que le manifeste final
        let tm = fs::metadata(r.manifeste()).unwrap().modified().unwrap();
        let tc = fs::metadata(r.complete()).unwrap().modified().unwrap();
        assert!(tc >= tm);

        let ko = base.join("ko");
        let mut r2 = RepertoireRun::ouvrir(&ko, Champs::new()).unwrap();
        r2.terminer(3).unwrap();
        assert!(!r2.complete().exists(), "rc != 0 ne doit jamais ecrire COMPLETE");
        let m2 = fs::read_to_string(r2.manifeste()).unwrap();
        assert!(m2.contains("\"statut\": \"echec\"") && m2.contains("\"rc\": 3"));
        // aucun fichier temporaire résiduel
        for e in fs::read_dir(&ok).unwrap().chain(fs::read_dir(&ko).unwrap()) {
            let n = e.unwrap().file_name().to_string_lossy().into_owned();
            assert!(n == NOM_MANIFESTE || n == NOM_COMPLETE, "residu {}", n);
        }
        fs::remove_dir_all(&base).unwrap();
        println!("PASS complete_seulement_apres_rc0");
    }
}
