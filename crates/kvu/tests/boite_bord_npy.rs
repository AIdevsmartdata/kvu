// SPDX-License-Identifier: Apache-2.0
//! Bords de boîte lus dans un fichier NPY (`kvu su2 boite --bord-npy`, lot WP21, `docs/BORD_NPY.md`) : gates BORD-NPY-*.
//!
//! - CPU (suite normale) : `bord_npy_cpu` (EQUIV, LIBRES, MUT, MANIFESTE, REGEN) et `bord_npy_refus` (REFUS) ;
//! - GPU (`#[ignore = "GPU"]`, SOUS le verrou de la machine, fenêtre courte) : `bord_npy_gpu`, mêmes gates, suffixe = backend effectif
//!   (`… --test boite_bord_npy -- --ignored --nocapture --test-threads=1 bord_npy_gpu`).
//!
//! Chaque gate imprime UNE ligne `PASS|FAIL|SKIP <ID> : …` (SKIP n'est pas vert). Références indépendantes du code testé : la
//! géométrie « lien fixé » est recodée ici, les NPY sont écrits et relus octet par octet, les sha256 viennent de `sha256sum`.

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

const ELL: usize = 6;
const LC: usize = ELL + 2;
const V: usize = LC * LC * LC * LC;
/// 4ℓ(ℓ+1)³ − 4ℓ(ℓ−1)³ à ℓ = 6.
const N_FIXES: usize = 5232;
const BORDS: [&str; 7] = ["cold", "gauge", "z2full", "z2_01", "flux", "haar", "therm"];
const GRAINES: [u64; 7] = [26_102_721_001, 26_102_721_002, 26_102_721_003, 26_102_721_004, 26_102_721_005, 26_102_721_006, 26_102_721_007];
const PARAMS: [&str; 19] = [
    "--ell", "6", "--beta", "2.5", "--therm", "20", "--nmeas", "100", "--skip", "1", "--nor", "2", "--profil-tous", "10", "--therm-source",
    "50", "--options", "ref-k1", "--dump-final",
];
const SORTIES: [&str; 3] = ["mesures.csv", "profil.csv", "config_finale.npy"];

// ---------------------------------------------------------------------------------------------------------------------
// Outils indépendants
// ---------------------------------------------------------------------------------------------------------------------

fn tmp(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("boite_bord_npy").join(format!("{nom}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Exécute `kvu` ; `ignorer` pose KEVINOTRON_VERROU=ignorer (CPU), sinon la variable est retirée (GPU : vrai verrou).
fn kvu(ignorer: bool, args: &[String]) -> (i32, String) {
    let mut c = Command::new(env!("CARGO_BIN_EXE_kvu"));
    c.args(args);
    if ignorer {
        c.env("KEVINOTRON_VERROU", "ignorer");
    } else {
        c.env_remove("KEVINOTRON_VERROU");
    }
    let o = c.output().unwrap();
    let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&o.stderr));
    (o.status.code().unwrap_or(-1), s)
}

fn texte(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

fn lire_json(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_default()).unwrap_or(Value::Null)
}

fn sha256sum(p: &Path) -> String {
    let o = Command::new("sha256sum").arg(p).output().expect("sha256sum");
    String::from_utf8_lossy(&o.stdout).split_whitespace().next().unwrap_or("").to_string()
}

fn h8(p: &Path) -> String {
    format!("npy:{}", &sha256sum(p)[..8])
}

fn coords(s: usize) -> [usize; 4] {
    [s / (LC * LC * LC), (s / (LC * LC)) % LC, (s / LC) % LC, s % LC]
}

/// Lien fixé (recodage indépendant de docs/BORD_NPY.md §2) : x ∈ {0,…,ℓ}⁴, x_μ ≤ ℓ − 1, x et x + e_μ sur le bord du cube.
fn fixe(mu: usize, x: [usize; 4]) -> bool {
    if x.iter().any(|&c| c > ELL) || x[mu] >= ELL {
        return false;
    }
    let sur_bord = |y: [usize; 4]| y.iter().any(|&c| c == 0 || c == ELL);
    let mut y = x;
    y[mu] += 1;
    sur_bord(x) && sur_bord(y)
}

/// Indices des liens fixés, croissants.
fn liens_fixes() -> Vec<usize> {
    (0..4 * V).filter(|&li| fixe(li / V, coords(li % V))).collect()
}

/// En-tête NPY écrit à la main (v1 : dictionnaire complété d'espaces, longueur totale multiple de 64).
fn entete(descr: &str, forme: &str, fortran: bool, version: u8) -> Vec<u8> {
    let mut d = format!("{{'descr': '{descr}', 'fortran_order': {}, 'shape': {forme}, }}", if fortran { "True" } else { "False" });
    let tot = 10 + d.len() + 1;
    d.push_str(&" ".repeat((64 - tot % 64) % 64));
    d.push('\n');
    let mut o = vec![0x93, b'N', b'U', b'M', b'P', b'Y', version, 0];
    o.extend_from_slice(&(d.len() as u16).to_le_bytes());
    o.extend_from_slice(d.as_bytes());
    o
}

fn forme_q64(l: usize) -> String {
    format!("(4, {l}, {l}, {l}, {l}, 4)")
}

fn octets_f64(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ecrire_q64(p: &Path, l: usize, q: &[f64]) {
    assert_eq!(q.len(), 16 * l.pow(4));
    let mut o = entete("<f8", &forme_q64(l), false, 1);
    o.extend(octets_f64(q));
    std::fs::write(p, o).unwrap();
}

/// c16 : pour chaque lien, la matrice [[a0 + i a3, a2 + i a1], [−a2 + i a1, a0 − i a3]] (réels et imaginaires entrelacés).
fn ecrire_c16(p: &Path, q: &[f64]) {
    let mut o = entete("<c16", &format!("(4, {LC}, {LC}, {LC}, {LC}, 2, 2)"), false, 1);
    for a in q.chunks_exact(4) {
        o.extend(octets_f64(&[a[0], a[3], a[2], a[1], -a[2], a[1], a[0], -a[3]]));
    }
    std::fs::write(p, o).unwrap();
}

/// Relit un q64 de côté LC écrit en v1 (en-tête quelconque, données à la fin).
fn lire_q64(p: &Path) -> Vec<f64> {
    let b = std::fs::read(p).unwrap();
    assert_eq!(&b[..6], b"\x93NUMPY");
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let d = &b[10 + hl..];
    assert_eq!(d.len(), 16 * V * 8, "{}", p.display());
    d.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn froid(l: usize) -> Vec<f64> {
    let mut q = vec![0.0; 16 * l.pow(4)];
    for c in q.chunks_exact_mut(4) {
        c[0] = 1.0;
    }
    q
}

/// Produit de Hamilton (scalaire en premier).
fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// Générateur pseudo-aléatoire xorshift64* (graine fixe) : quaternions unitaires par rejet dans la boule.
struct Alea(u64);
impl Alea {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    fn quaternion(&mut self) -> [f64; 4] {
        loop {
            let q = [self.u(), self.u(), self.u(), self.u()];
            let n2: f64 = q.iter().map(|x| x * x).sum();
            if n2 > 1e-6 && n2 <= 1.0 {
                let n = n2.sqrt();
                return q.map(|x| x / n);
            }
        }
    }
}

/// Valeur attendue (au bit) des liens fixés des bords à forme close de docs/BOITES.md §1 ; None pour gauge, haar, therm.
fn forme_close(bord: &str, mu: usize, x: [usize; 4]) -> Option<[f64; 4]> {
    const TABLE: [(f64, f64); 4] = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
    let signe = |impair: bool| if impair { -1.0 } else { 1.0 };
    match bord {
        "cold" => Some([1.0, 0.0, 0.0, 0.0]),
        "z2full" => Some([signe(x[..mu].iter().sum::<usize>() % 2 == 1), 0.0, 0.0, 0.0]),
        "z2_01" => Some([signe(mu == 1 && x[0] % 2 == 1), 0.0, 0.0, 0.0]),
        "flux" => Some(match mu {
            1 => {
                let (c, s) = TABLE[x[0] % 4];
                [c, 0.0, 0.0, s]
            }
            3 => {
                let (c, s) = TABLE[x[2] % 4];
                [c, s, 0.0, 0.0]
            }
            _ => [1.0, 0.0, 0.0, 0.0],
        }),
        _ => None,
    }
}

fn gate(nom: &str, corps: impl FnOnce() -> Result<String, String>) {
    match corps() {
        Ok(d) if d.starts_with("SKIP") => println!("\nSKIP {nom} : {}", d.trim_start_matches("SKIP").trim()),
        Ok(d) => println!("\nPASS {nom} : {d}"),
        Err(r) => {
            println!("\nFAIL {nom} : {r}");
            panic!("FAIL {nom} : {r}");
        }
    }
}

fn identiques(x: &Path, y: &Path) -> bool {
    matches!((std::fs::read(x), std::fs::read(y)), (Ok(a), Ok(b)) if a == b)
}

// ---------------------------------------------------------------------------------------------------------------------
// Contexte d'un banc (CPU ou GPU)
// ---------------------------------------------------------------------------------------------------------------------

struct Banc {
    base: PathBuf,
    backend: &'static str,
    ignorer: bool,
}

impl Banc {
    /// `su2 boite --backend B <PARAMS> <extra> --outdir D`.
    fn boite(&self, outdir: &Path, extra: &[String]) -> (i32, String) {
        let mut a: Vec<String> = ["su2", "boite", "--backend", self.backend].iter().map(|s| s.to_string()).collect();
        a.extend(PARAMS.iter().map(|s| s.to_string()));
        a.extend(extra.iter().cloned());
        a.push("--outdir".into());
        a.push(texte(outdir));
        kvu(self.ignorer, &a)
    }

    /// Lot de 7 chaînes `npy:<h8_b>:G_b` sur les fichiers `f` (même ordre que BORDS).
    fn lot_npy(&self, nom: &str, f: &[PathBuf]) -> Result<PathBuf, String> {
        let d = self.base.join(nom);
        let mut extra = Vec::new();
        for p in f {
            extra.push("--bord-npy".to_string());
            extra.push(texte(p));
        }
        let chaines: Vec<String> = f.iter().zip(GRAINES).map(|(p, g)| format!("{}:{g}", h8(p))).collect();
        extra.push("--chaines".into());
        extra.push(chaines.join(","));
        let (rc, s) = self.boite(&d, &extra);
        if rc != 0 {
            return Err(format!("lot {nom} : rc {rc} : {}", s.lines().rev().take(3).collect::<Vec<_>>().join(" | ")));
        }
        Ok(d)
    }
}

/// Compare les sorties des chaînes : (chaîne du lot, chaîne de A).
fn comparer(lot: &Path, a: &Path, paires: &[(usize, usize)]) -> Result<usize, String> {
    let mut n = 0;
    for &(c, ca) in paires {
        for f in SORTIES {
            let (x, y) = (lot.join(format!("c{c:03}/{f}")), a.join(format!("c{ca:03}/{f}")));
            if !identiques(&x, &y) {
                return Err(format!("{} ≠ {}", x.display(), y.display()));
            }
            n += 1;
        }
    }
    Ok(n)
}

fn gates(backend: &'static str) {
    let gpu = backend != "cpu";
    let banc = Banc { base: tmp(if gpu { "gpu" } else { "cpu" }), backend, ignorer: !gpu };
    let fixes = liens_fixes();
    assert_eq!(fixes.len(), N_FIXES, "recodage de la géométrie : n_fixes(6)");

    // --- A : les 7 bords nommés, sorties de référence ; fichiers F_b = configurations finales (liens fixés = bord b, au bit)
    let a = banc.base.join("A");
    let chaines_a: Vec<String> = BORDS.iter().zip(GRAINES).map(|(b, g)| format!("{b}:{g}")).collect();
    let (rc, s) = banc.boite(&a, &["--chaines".into(), chaines_a.join(",")]);
    let man_a = lire_json(&a.join("manifest.json"));
    let effectif = man_a["backend_effectif"].as_str().unwrap_or("?").to_string();
    if gpu && rc == 0 && effectif == "cpu" {
        for id in ["EQUIV", "LIBRES", "MUT", "MANIFESTE", "REGEN"] {
            println!("\nSKIP BORD-NPY-{id}-GPU : « --backend auto » a choisi le CPU (aucun GPU utilisable)");
        }
        return;
    }
    let suffixe = match effectif.as_str() {
        "cpu" | "hip" | "cuda" => effectif.to_uppercase(),
        _ if gpu => "GPU".to_string(),
        _ => "CPU".to_string(),
    };
    let verrou = man_a["verrou"].as_str().unwrap_or("?").to_string();
    let prep = (|| -> Result<(Vec<PathBuf>, Vec<PathBuf>), String> {
        if rc != 0 {
            return Err(format!("lot A : rc {rc} : {s}"));
        }
        let intacts = man_a["chaines"].as_array().map(|c| c.iter().all(|x| x["liens_geles_intacts"] == true)).unwrap_or(false);
        if man_a["controle_bord"]["intact"] != true || !intacts {
            return Err("lot A : liens gelés non intacts".into());
        }
        let dq = banc.base.join("bords_q64");
        let dc = banc.base.join("bords_c16");
        std::fs::create_dir_all(&dq).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(&dc).map_err(|e| e.to_string())?;
        let (mut fq, mut fc) = (Vec::new(), Vec::new());
        for (k, b) in BORDS.iter().enumerate() {
            let p = dq.join(format!("{b}.npy"));
            std::fs::copy(a.join(format!("c{k:03}/config_finale.npy")), &p).map_err(|e| e.to_string())?;
            let q = lire_q64(&p);
            for &li in &fixes {
                if let Some(att) = forme_close(b, li / V, coords(li % V)) {
                    let lu = [q[4 * li], q[4 * li + 1], q[4 * li + 2], q[4 * li + 3]];
                    if lu.map(f64::to_bits) != att.map(f64::to_bits) {
                        return Err(format!("bord {b}, lien fixé {li} : {lu:?} ≠ forme close {att:?}"));
                    }
                }
            }
            let pc = dc.join(format!("{b}.npy"));
            ecrire_c16(&pc, &q);
            fq.push(p);
            fc.push(pc);
        }
        Ok((fq, fc))
    })();
    let (fq, fc) = match prep {
        Ok(x) => x,
        Err(e) => {
            println!("\nFAIL BORD-NPY-EQUIV-{suffixe} : préparation : {e}");
            panic!("préparation : {e}");
        }
    };
    let tous: Vec<(usize, usize)> = (0..7).map(|c| (c, c)).collect();

    gate(&format!("BORD-NPY-EQUIV-{suffixe}"), || {
        let mut n = 0;
        let b = banc.lot_npy("B", &fq)?;
        n += comparer(&b, &a, &tous)?;
        let b16 = banc.lot_npy("B16", &fc)?;
        n += comparer(&b16, &a, &tous)?;
        // C : lot mixte (bords nommés et lus), deux fichiers
        let c = banc.base.join("C");
        let chaines = format!("cold:{},{}:{},z2full:{},{}:{}", GRAINES[0], h8(&fq[1]), GRAINES[1], GRAINES[2], h8(&fq[5]), GRAINES[5]);
        let (rc, s) = banc.boite(&c, &["--bord-npy".into(), texte(&fq[1]), "--bord-npy".into(), texte(&fq[5]), "--chaines".into(), chaines]);
        if rc != 0 {
            return Err(format!("lot C : rc {rc} : {s}"));
        }
        n += comparer(&c, &a, &[(0, 0), (1, 1), (2, 2), (3, 5)])?;
        // D : « npy » nu (un seul fichier) avec --bords / --repliques
        let d = banc.base.join("D");
        let (rc, s) = banc.boite(
            &d,
            &[
                "--bord-npy".into(),
                texte(&fq[2]),
                "--bords".into(),
                "npy".into(),
                "--repliques".into(),
                "2".into(),
                "--graine".into(),
                GRAINES[2].to_string(),
                "--graine-pas".into(),
                "1000".into(),
            ],
        );
        if rc != 0 {
            return Err(format!("lot D : rc {rc} : {s}"));
        }
        n += comparer(&d, &a, &[(0, 2)])?;
        let nom_d = lire_json(&d.join("manifest.json"))["config_boite"]["chaines"][0]["bord"].as_str().unwrap_or("").to_string();
        if nom_d != h8(&fq[2]) {
            return Err(format!("lot D : nom résolu {nom_d} ≠ {}", h8(&fq[2])));
        }
        Ok(format!(
            "7 bords nommés en q64 et en c16, lot mixte, « npy » nu : {n} fichiers identiques à l'octet à ceux des bords nommés (backend {effectif}, verrou {verrou})"
        ))
    });

    gate(&format!("BORD-NPY-LIBRES-{suffixe}"), || {
        let mut alea = Alea(0x5EED_2610_2721_0001);
        let (dr, dg) = (banc.base.join("libres_r"), banc.base.join("libres_g"));
        std::fs::create_dir_all(&dr).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(&dg).map_err(|e| e.to_string())?;
        let rebut = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, 7.5, 1e300];
        let (mut fr, mut fg) = (Vec::new(), Vec::new());
        for (k, b) in BORDS.iter().enumerate() {
            let q = lire_q64(&fq[k]);
            let (mut qr, mut qg) = (q.clone(), q);
            let mut r = 0usize;
            for li in 0..4 * V {
                if fixe(li / V, coords(li % V)) {
                    continue;
                }
                qr[4 * li..4 * li + 4].copy_from_slice(&alea.quaternion());
                for j in 0..4 {
                    qg[4 * li + j] = rebut[r % rebut.len()];
                    r += 1;
                }
            }
            let (pr, pg) = (dr.join(format!("{b}.npy")), dg.join(format!("{b}.npy")));
            ecrire_q64(&pr, LC, &qr);
            ecrire_q64(&pg, LC, &qg);
            if sha256sum(&pr) == sha256sum(&fq[k]) || sha256sum(&pg) == sha256sum(&fq[k]) {
                return Err(format!("bord {b} : fichier modifié de même sha256 que l'original"));
            }
            fr.push(pr);
            fg.push(pg);
        }
        let lr = banc.lot_npy("L_R", &fr)?;
        let n1 = comparer(&lr, &a, &tous)?;
        let lg = banc.lot_npy("L_G", &fg)?;
        let n2 = comparer(&lg, &a, &tous)?;
        Ok(format!(
            "liens non fixés remplacés (SU(2) aléatoires ; NaN, ±Inf, 0, 7.5, 1e300) : {n1} + {n2} fichiers identiques à ceux des bords nommés"
        ))
    });

    gate(&format!("BORD-NPY-MUT-{suffixe}"), || {
        // (a) lien fixé DYNAMIQUE : U₁(0, ℓ/2, ℓ/2, ℓ/2), centre de la face x₀ = 0 (la plaquette (0,1) en x contient le lien libre U₀(x)) ;
        // (b) lien fixé INERTE : U₂(0,0,0,0), coin du bord : ses 3 plaquettes de la boîte n'ont que des liens fixés (aucune agrafe de
        //     lien libre ne le lit) ⇒ séries identiques, mais profil d0 et configuration finale différents.
        let li_dyn = V + ((ELL / 2 * LC + ELL / 2) * LC + ELL / 2);
        let li_inerte = 2 * V;
        if !fixe(1, coords(li_dyn % V)) || !fixe(2, coords(0)) || fixe(0, coords(li_dyn % V)) {
            return Err("ECHEC_OUTIL : choix des liens mutés incohérent avec la géométrie".into());
        }
        let r = [0.5f64.cos(), 0.5f64.sin(), 0.0, 0.0];
        let muter = |nom: &str, li: usize| -> Result<PathBuf, String> {
            let dm = banc.base.join(format!("mut_{nom}"));
            std::fs::create_dir_all(&dm).map_err(|e| e.to_string())?;
            let mut fm = Vec::new();
            for (k, b) in BORDS.iter().enumerate() {
                let mut q = lire_q64(&fq[k]);
                let n = qmul([q[4 * li], q[4 * li + 1], q[4 * li + 2], q[4 * li + 3]], r);
                let norme = n.iter().map(|x| x * x).sum::<f64>().sqrt();
                if (norme - 1.0).abs() > 1e-15 {
                    return Err(format!("lien muté de norme {norme}"));
                }
                q[4 * li..4 * li + 4].copy_from_slice(&n);
                let p = dm.join(format!("{b}.npy"));
                ecrire_q64(&p, LC, &q);
                fm.push(p);
            }
            banc.lot_npy(&format!("M_{nom}"), &fm)
        };
        let compte = |lot: &Path, f: &str| -> usize {
            (0..7)
                .filter(|c| {
                    let p = lot.join(format!("c{c:03}/{f}"));
                    p.exists() && !identiques(&p, &a.join(format!("c{c:03}/{f}")))
                })
                .count()
        };
        let md = muter("dyn", li_dyn)?;
        let mesures_dyn = compte(&md, "mesures.csv");
        if mesures_dyn != 7 {
            return Err(format!("ECHEC_OUTIL : lien fixé dynamique li = {li_dyn} muté : {mesures_dyn}/7 séries mesures.csv différentes de A"));
        }
        let mi = muter("inerte", li_inerte)?;
        let (m_i, p_i, c_i) = (compte(&mi, "mesures.csv"), compte(&mi, "profil.csv"), compte(&mi, "config_finale.npy"));
        if (m_i, p_i, c_i) != (0, 7, 7) {
            return Err(format!(
                "lien fixé inerte li = {li_inerte} muté : séries différentes {m_i}/7 (attendu 0), profils {p_i}/7 et configurations finales {c_i}/7 (attendu 7)"
            ));
        }
        Ok(format!(
            "lien fixé dynamique (li = {li_dyn}, U₁ au centre de la face x₀ = 0) tourné de 0,5 rad : 7/7 séries mesures.csv différentes (mutant rouge) ; \
             lien fixé inerte (li = {li_inerte}, coin) : séries identiques 7/7, profils et configurations finales différents 7/7"
        ))
    });

    gate(&format!("BORD-NPY-MANIFESTE-{suffixe}"), || {
        let b = banc.base.join("B");
        let m = lire_json(&b.join("manifest.json"));
        let r = lire_json(&b.join("manifest_regen.json"));
        let liste = m["bords_npy"].as_array().ok_or("manifest.json sans bords_npy")?;
        let entrees = r["entrees"].as_array().ok_or("manifest_regen.json sans entrees")?;
        let pbn = r["parametres"]["boite"]["bords_npy"].as_array().ok_or("parametres.boite.bords_npy absent")?;
        if liste.len() != 7 || entrees.len() != 7 || pbn.len() != 7 {
            return Err(format!("{} / {} / {} entrées au lieu de 7", liste.len(), entrees.len(), pbn.len()));
        }
        let commande: Vec<String> = r["commande"].as_array().ok_or("commande absente")?.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect();
        let i_ch = commande.iter().position(|x| x == "--chaines").ok_or("--chaines absent de la commande")?;
        for (k, p) in fq.iter().enumerate() {
            let sha = sha256sum(p);
            let canon = texte(&std::fs::canonicalize(p).map_err(|e| e.to_string())?);
            let octets = std::fs::metadata(p).map_err(|e| e.to_string())?.len();
            let e = &liste[k];
            let attendu = (format!("npy:{}", &sha[..8]), sha.clone(), canon.clone(), octets);
            let lu = (
                e["nom"].as_str().unwrap_or("").to_string(),
                e["sha256"].as_str().unwrap_or("").to_string(),
                e["chemin"].as_str().unwrap_or("").to_string(),
                e["octets"].as_u64().unwrap_or(0),
            );
            if lu != attendu || e["format"] != "q64" || e["l_calcul"] != LC || e["n_liens_fixes"] != N_FIXES {
                return Err(format!("bords_npy[{k}] = {e} ; attendu {attendu:?}, q64, {LC}, {N_FIXES}"));
            }
            let en = &entrees[k];
            if en["role"] != "bord-npy" || en["sha256"] != sha.as_str() || en["chemin"] != canon.as_str() || en["nom"] != e["nom"] || &pbn[k] != e {
                return Err(format!("entrees[{k}] = {en} incohérente avec bords_npy[{k}]"));
            }
            if m["config_boite"]["chaines"][k]["bord"] != e["nom"] {
                return Err(format!("config_boite.chaines[{k}].bord ≠ {}", e["nom"]));
            }
            let j = i_ch - 2 * (7 - k);
            if commande[j] != "--bord-npy" || commande[j + 1] != canon {
                return Err(format!("commande : paire --bord-npy {k} absente juste avant --chaines : {:?}", &commande[j..j + 2]));
            }
        }
        let erreurs = kvu::regen::schema::valider_manifeste(&r);
        if !erreurs.is_empty() {
            return Err(format!("manifest_regen.json invalide (Rust) : {erreurs:?}"));
        }
        let racine = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        // validateur Python OBLIGATOIRE : impossible à lancer ⇒ FAIL (jamais un PASS sans lui)
        let py = Command::new("python3").arg("-I").arg(racine.join("scripts/regen_valider.py")).arg(b.join("manifest_regen.json")).output();
        let note = match py {
            Ok(o) if o.status.success() => "validateur Python rc 0".to_string(),
            Ok(o) => return Err(format!("validateur Python rc {:?} : {}", o.status.code(), String::from_utf8_lossy(&o.stdout))),
            Err(e) => return Err(format!("validateur Python impossible à lancer : {e}")),
        };
        Ok(format!(
            "bords_npy (7 : nom, sha256 = sha256sum, chemin canonique, octets, q64, l_calcul 8, 5232 liens fixés), entrees role bord-npy, \
             parametres.boite.bords_npy, 7 paires --bord-npy avant --chaines ; schéma kvu-regen/1 valide (Rust), {note}"
        ))
    });

    gate(&format!("BORD-NPY-REGEN-{suffixe}"), || {
        // fichiers PROPRES au rejeu (ils seront modifiés puis supprimés)
        let dr = banc.base.join("bords_regen");
        std::fs::create_dir_all(&dr).map_err(|e| e.to_string())?;
        let mut fr = Vec::new();
        for (k, b) in BORDS.iter().enumerate() {
            let p = dr.join(format!("{b}.npy"));
            std::fs::copy(&fq[k], &p).map_err(|e| e.to_string())?;
            fr.push(p);
        }
        let bp = banc.lot_npy("B_regen", &fr)?;
        let man = texte(&bp.join("manifest_regen.json"));
        let r1 = banc.base.join("R1");
        let (rc, s) = kvu(banc.ignorer, &["regen".into(), man.clone(), "--sortie".into(), texte(&r1)]);
        let rapport = lire_json(&r1.join("rapport_regen.json"));
        if rc != 0
            || !s.contains("PASS REGEN 21/21")
            || rapport["resultat"] != "identique"
            || rapport["fichiers_total"] != 21
            || rapport["fichiers_identiques"] != 21
            || rapport["rc_enfant"] != 0
        {
            return Err(format!("rejeu : rc {rc}, rapport {rapport} : {s}"));
        }
        // mutant : un lien FIXÉ du fichier haar modifié en place ⇒ rc 8, rien n'est créé
        let mut q = lire_q64(&fr[5]);
        let li = fixes[0];
        let tourne = qmul([q[4 * li], q[4 * li + 1], q[4 * li + 2], q[4 * li + 3]], [0.0, 1.0, 0.0, 0.0]);
        q[4 * li..4 * li + 4].copy_from_slice(&tourne);
        ecrire_q64(&fr[5], LC, &q);
        let r2 = banc.base.join("R2");
        let (rc2, s2) = kvu(banc.ignorer, &["regen".into(), man.clone(), "--sortie".into(), texte(&r2)]);
        if rc2 != 8 || !s2.contains("entrée modifiée") || r2.exists() {
            return Err(format!("entrée modifiée : rc {rc2} (attendu 8), sortie créée : {} : {s2}", r2.exists()));
        }
        std::fs::remove_file(&fr[5]).map_err(|e| e.to_string())?;
        let r3 = banc.base.join("R3");
        let (rc3, s3) = kvu(banc.ignorer, &["regen".into(), man, "--sortie".into(), texte(&r3)]);
        if rc3 != 4 || r3.exists() {
            return Err(format!("entrée absente : rc {rc3} (attendu 4), sortie créée : {} : {s3}", r3.exists()));
        }
        Ok("kvu regen : 21/21 fichiers identiques (rc 0, rapport_regen.json « identique » 21/21, rc du rejeu 0) ; entrée modifiée ⇒ rc 8 sans rejeu ; entrée absente ⇒ rc 4 sans rejeu".into())
    });
    let _ = std::fs::remove_dir_all(&banc.base);
}

#[test]
fn bord_npy_cpu() {
    gates("cpu");
}

#[test]
#[ignore = "GPU"]
fn bord_npy_gpu() {
    gates("auto");
}

// ---------------------------------------------------------------------------------------------------------------------
// BORD-NPY-REFUS (CPU) : chaque refus rend le code attendu, son message contient un mot-clé, et aucun --outdir n'est créé
// ---------------------------------------------------------------------------------------------------------------------

#[test]
fn bord_npy_refus() {
    let base = tmp("refus");
    let fixes = liens_fixes();
    let li0 = fixes[0];
    let f = base.join("froid.npy");
    ecrire_q64(&f, LC, &froid(LC));
    let mut q2 = froid(LC);
    let libre = (0..4 * V).find(|&li| !fixe(li / V, coords(li % V))).unwrap();
    q2[4 * libre..4 * libre + 4].copy_from_slice(&[0.0, 1.0, 0.0, 0.0]);
    let f2 = base.join("froid_bis.npy");
    ecrire_q64(&f2, LC, &q2);
    let f_copie = base.join("froid_copie.npy");
    std::fs::copy(&f, &f_copie).unwrap();
    let ecrire_brut = |nom: &str, octets: Vec<u8>| -> PathBuf {
        let p = base.join(nom);
        std::fs::write(&p, octets).unwrap();
        p
    };
    let donnees = octets_f64(&froid(LC));
    let avec_lien = |nom: &str, lien: [f64; 4]| -> PathBuf {
        let mut q = froid(LC);
        q[4 * li0..4 * li0 + 4].copy_from_slice(&lien);
        let p = base.join(nom);
        ecrire_q64(&p, LC, &q);
        p
    };
    let texte_ = ecrire_brut("texte.npy", b"bonjour, ceci n'est pas un NPY\n".to_vec());
    let v2 = ecrire_brut("v2.npy", [entete("<f8", &forme_q64(LC), false, 2), donnees.clone()].concat());
    let fortran = ecrire_brut("fortran.npy", [entete("<f8", &forme_q64(LC), true, 1), donnees.clone()].concat());
    let f4 = ecrire_brut(
        "f4.npy",
        [entete("<f4", &forme_q64(LC), false, 1), froid(LC).iter().flat_map(|&x| (x as f32).to_le_bytes()).collect()].concat(),
    );
    let forme3 = ecrire_brut("forme3.npy", [entete("<f8", &format!("(4, {LC}, {LC}, {LC}, {LC}, 3)"), false, 1), donnees[..donnees.len() / 4 * 3].to_vec()].concat());
    let complet = std::fs::read(&f).unwrap();
    let tronque = ecrire_brut("tronque.npy", complet[..complet.len() - 32].to_vec());
    let en_trop = ecrire_brut("en_trop.npy", [complet.clone(), vec![0u8; 8]].concat());
    let cote10 = base.join("cote10.npy");
    ecrire_q64(&cote10, 10, &froid(10));
    let cote6 = base.join("cote6.npy");
    ecrire_q64(&cote6, 6, &froid(6));
    let non_unitaire = avec_lien("non_unitaire.npy", [1.0 + 1e-9, 0.0, 0.0, 0.0]);
    let nan = avec_lien("nan.npy", [f64::NAN, 0.0, 0.0, 0.0]);
    let inf = avec_lien("inf.npy", [f64::INFINITY, 0.0, 0.0, 0.0]);
    let c16 = base.join("c16_faux.npy");
    ecrire_c16(&c16, &froid(LC));
    let mut b16 = std::fs::read(&c16).unwrap();
    let hl = u16::from_le_bytes([b16[8], b16[9]]) as usize;
    let o = 10 + hl + 64 * li0 + 8 * 2; // M01, partie réelle (= a2)
    b16[o..o + 8].copy_from_slice(&1e-6f64.to_le_bytes());
    std::fs::write(&c16, b16).unwrap();
    let absent = base.join("absent.npy");
    let hf = h8(&f);
    let inconnu = if hf == "npy:00000000" { "npy:00000001" } else { "npy:00000000" };
    let s = |p: &Path| texte(p);
    let cas: Vec<(&str, i32, &str, Vec<String>)> = vec![
        ("fichier absent", 4, "bord-npy", vec!["--bord-npy".into(), s(&absent), "--chaines".into(), "npy:1".into()]),
        ("pas un NPY", 4, "NPY", vec!["--bord-npy".into(), s(&texte_), "--chaines".into(), "npy:1".into()]),
        ("NPY version 2", 4, "version", vec!["--bord-npy".into(), s(&v2), "--chaines".into(), "npy:1".into()]),
        ("fortran_order True", 4, "fortran", vec!["--bord-npy".into(), s(&fortran), "--chaines".into(), "npy:1".into()]),
        ("dtype <f4", 4, "refusé", vec!["--bord-npy".into(), s(&f4), "--chaines".into(), "npy:1".into()]),
        ("forme (…, 3)", 4, "refusé", vec!["--bord-npy".into(), s(&forme3), "--chaines".into(), "npy:1".into()]),
        ("tronqué", 4, "taille", vec!["--bord-npy".into(), s(&tronque), "--chaines".into(), "npy:1".into()]),
        ("octets en trop", 4, "taille", vec!["--bord-npy".into(), s(&en_trop), "--chaines".into(), "npy:1".into()]),
        ("côté 10 pour ℓ = 6", 2, "côté", vec!["--bord-npy".into(), s(&cote10), "--chaines".into(), "npy:1".into()]),
        ("côté 6 pour ℓ = 6", 2, "côté", vec!["--bord-npy".into(), s(&cote6), "--chaines".into(), "npy:1".into()]),
        ("lien fixé non unitaire", 4, "hors SU(2)", vec!["--bord-npy".into(), s(&non_unitaire), "--chaines".into(), "npy:1".into()]),
        ("lien fixé NaN", 4, "hors SU(2)", vec!["--bord-npy".into(), s(&nan), "--chaines".into(), "npy:1".into()]),
        ("lien fixé +Inf", 4, "hors SU(2)", vec!["--bord-npy".into(), s(&inf), "--chaines".into(), "npy:1".into()]),
        ("c16 de structure fausse", 4, "hors SU(2)", vec!["--bord-npy".into(), s(&c16), "--chaines".into(), "npy:1".into()]),
        ("npy:<h8> sans --bord-npy", 2, "aucun --bord-npy", vec!["--chaines".into(), format!("{hf}:1")]),
        ("npy nu sans --bord-npy", 2, "aucun --bord-npy", vec!["--chaines".into(), "npy:1".into()]),
        ("fichier inutilisé", 2, "inutilisé", vec!["--bord-npy".into(), s(&f), "--chaines".into(), "cold:1".into()]),
        ("npy nu, deux fichiers", 2, "ambigu", vec!["--bord-npy".into(), s(&f), "--bord-npy".into(), s(&f2), "--chaines".into(), "npy:1".into()]),
        ("npy:1234", 2, "8 premiers", vec!["--bord-npy".into(), s(&f), "--chaines".into(), format!("{hf}:1,npy:1234:2")]),
        ("npy:ZZZZZZZZ", 2, "8 premiers", vec!["--bord-npy".into(), s(&f), "--chaines".into(), format!("{hf}:1,npy:ZZZZZZZZ:2")]),
        ("empreinte inconnue", 2, "aucun --bord-npy de cette empreinte", vec!["--bord-npy".into(), s(&f), "--chaines".into(), format!("{hf}:1,{inconnu}:2")]),
        ("même fichier deux fois", 2, "même empreinte", vec!["--bord-npy".into(), s(&f), "--bord-npy".into(), s(&f), "--chaines".into(), format!("{hf}:1")]),
        ("deux chemins, même contenu", 2, "même empreinte", vec!["--bord-npy".into(), s(&f), "--bord-npy".into(), s(&f_copie), "--chaines".into(), format!("{hf}:1")]),
        ("--bords et --chaines", 2, "s'excluent", vec!["--bord-npy".into(), s(&f), "--bords".into(), "npy".into(), "--chaines".into(), format!("{hf}:1")]),
    ];
    let total = cas.len();
    gate("BORD-NPY-REFUS", || {
        let mut ok = 0;
        for (i, (quoi, code, mot, extra)) in cas.into_iter().enumerate() {
            let out = base.join(format!("sortie_{i:02}"));
            let mut a: Vec<String> = ["su2", "boite", "--backend", "cpu"].iter().map(|x| x.to_string()).collect();
            a.extend(PARAMS.iter().map(|x| x.to_string()));
            a.extend(extra);
            a.push("--outdir".into());
            a.push(texte(&out));
            let (rc, sortie) = kvu(true, &a);
            if rc != code || !sortie.contains(mot) || out.exists() {
                return Err(format!("{quoi} : rc {rc} (attendu {code}), mot « {mot} » {}, sortie créée {} : {sortie}", sortie.contains(mot), out.exists()));
            }
            ok += 1;
        }
        Ok(format!("{ok}/{total} refus au code attendu (4 : fichier absent, format, lien fixé hors SU(2)/NaN/Inf ; 2 : côté ≠ ℓ + 2, noms et combinaisons), message explicite, aucun répertoire de sortie créé"))
    });
    let _ = std::fs::remove_dir_all(&base);
}
