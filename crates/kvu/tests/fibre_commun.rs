//! Outils communs aux tests de la mesure de fibre (inclus par `fibre_cpu.rs` et `fibre_gpu.rs`) :
//! lecture des fixtures de `gen_fixtures.py`, somme exacte, bilan de gates et contrôles FIB-1.
#![allow(dead_code)]

use kvu::commun::ChampHote;
use kvu::fibre::npy::{lire, Donnees};
use kvu::fibre::{MesureFibre, PoidsFourier};
use std::path::{Path, PathBuf};

/// Tolérances gelées (ARCHITECTURE_UNIFIEE.md §6.3, FIB-1).
pub const TOL_F_ABS: f64 = 1e-13;
pub const TOL_FSUM_REL: f64 = 1e-14;
pub const TOL_D_REL: f64 = 1e-12;
pub const TOL_PLAQ: f64 = 1e-14;

/// Fixtures : `KVU_FIBRE_FIXTURES` (jeu complet généré par gen_fixtures.py) sinon le jeu commité.
pub fn dossier_fixtures() -> PathBuf {
    match std::env::var("KVU_FIBRE_FIXTURES") {
        Ok(d) if !d.is_empty() => PathBuf::from(d),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/fibre"),
    }
}

pub fn lire_json(chemin: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(chemin).unwrap_or_else(|e| panic!("{} : {e}", chemin.display())))
        .unwrap_or_else(|e| panic!("{} : {e}", chemin.display()))
}

pub fn f64s(chemin: &Path) -> Vec<f64> {
    match lire(chemin).unwrap_or_else(|e| panic!("{e}")).donnees {
        Donnees::F64(d) => d,
        autre => panic!("{} : float64 attendu, {autre:?}", chemin.display()),
    }
}

pub fn f32s(chemin: &Path) -> Vec<f32> {
    match lire(chemin).unwrap_or_else(|e| panic!("{e}")).donnees {
        Donnees::F32(d) => d,
        _ => panic!("{} : float32 attendu", chemin.display()),
    }
}

pub fn i64s(chemin: &Path) -> Vec<i64> {
    match lire(chemin).unwrap_or_else(|e| panic!("{e}")).donnees {
        Donnees::I64(d) => d,
        _ => panic!("{} : int64 attendu", chemin.display()),
    }
}

/// Un cas de test : configuration + références.
pub struct Cas {
    pub nom: String,
    pub l: usize,
    pub champ: ChampHote,
    pub dir: PathBuf,
    pub racine: PathBuf,
    pub meta: serde_json::Value,
}

impl Cas {
    pub fn existe(&self, f: &str) -> bool {
        self.dir.join(f).exists()
    }
    pub fn f64s(&self, f: &str) -> Vec<f64> {
        f64s(&self.dir.join(f))
    }
    pub fn plat_ou_pi(&self) -> bool {
        self.nom.ends_with("_plat") || self.nom.ends_with("_pi_flux")
    }
}

/// Champ π-flux (cfg_pi de test_concordance.py) : liens ±1, toutes les plaquettes valent −1.
pub fn champ_pi_flux(l: usize) -> ChampHote {
    let mut c = ChampHote::froid(l).unwrap();
    let v = c.v();
    for mu in 0..4usize {
        for s in 0..v {
            let x = [s / (l * l * l), (s / (l * l)) % l, (s / l) % l, s % l];
            let n: usize = match mu {
                0 => 0,
                1 => x[0],
                2 => x[0] + x[1],
                _ => x[0] + x[1] + x[2],
            };
            c.q[4 * (mu * v + s)] = if n.is_multiple_of(2) { 1.0 } else { -1.0 };
        }
    }
    c
}

/// Liste des cas du jeu de fixtures (index.json).
pub fn noms_des_cas(racine: &Path) -> Vec<String> {
    lire_json(&racine.join("index.json"))["cas"]
        .as_array()
        .expect("index.json : cas")
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

pub fn charger_cas(racine: &Path, nom: &str) -> Cas {
    let dir = racine.join(nom);
    let meta = lire_json(&dir.join("meta.json"));
    let l = meta["L"].as_u64().unwrap() as usize;
    let champ = if dir.join("q.npy").exists() {
        let q = f64s(&dir.join("q.npy"));
        ChampHote { l, q }
    } else if nom.ends_with("_plat") {
        ChampHote::froid(l).unwrap()
    } else if nom.ends_with("_pi_flux") {
        champ_pi_flux(l)
    } else {
        panic!("{nom} : ni q.npy ni construction connue");
    };
    Cas {
        nom: nom.to_string(),
        l,
        champ,
        dir,
        racine: racine.to_path_buf(),
        meta,
    }
}

// ------------------------------------------------------------------------------------------
// Arithmétique de référence
// ------------------------------------------------------------------------------------------

/// Somme exactement arrondie (algorithme des partiels de Shewchuk, comme `math.fsum`).
pub fn fsum(v: impl IntoIterator<Item = f64>) -> f64 {
    let mut partiels: Vec<f64> = Vec::new();
    for mut x in v {
        let mut i = 0;
        for k in 0..partiels.len() {
            let mut y = partiels[k];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partiels[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partiels.truncate(i);
        partiels.push(x);
    }
    // Les partiels sont rangés par grandeur croissante : sommation du plus petit au plus grand (sum de Python).
    partiels.iter().fold(0.0, |s, p| s + p)
}

pub fn norme_frob(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// ‖a − b‖_F / ‖b‖_F (écart ABSOLU si la référence est nulle : plat, π-flux).
pub fn ecart_frob(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let num = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let den = norme_frob(b);
    num / if den > 0.0 { den } else { 1.0 }
}

pub fn max_abs_ecart(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

// ------------------------------------------------------------------------------------------
// Bilan des gates
// ------------------------------------------------------------------------------------------

#[derive(Default)]
pub struct Bilan {
    /// (nom, n, max observé, tolérance, ok)
    pub gates: Vec<(String, usize, f64, f64, bool)>,
    pub echecs: Vec<String>,
}

impl Bilan {
    /// Gate numérique : `val ≤ tol`.
    pub fn gate(&mut self, nom: &str, val: f64, tol: f64, info: &str) {
        let ok = val <= tol;
        match self.gates.iter_mut().find(|g| g.0 == nom) {
            Some(g) => {
                g.1 += 1;
                g.2 = g.2.max(val);
                g.4 &= ok;
            }
            None => self.gates.push((nom.to_string(), 1, val, tol, ok)),
        }
        if !ok {
            self.echecs.push(format!("{nom} = {val:e} > {tol:e} ({info})"));
        }
    }

    /// Gate booléenne.
    pub fn vrai(&mut self, nom: &str, ok: bool, info: &str) {
        self.gate(nom, if ok { 0.0 } else { 1.0 }, 0.0, info);
    }

    pub fn ok(&self) -> bool {
        self.echecs.is_empty()
    }

    pub fn imprimer(&self) {
        for (nom, n, max, tol, ok) in &self.gates {
            println!(
                "{} {nom:60} n={n:3} max={max:.2e} tol={tol:.0e}",
                if *ok { "VERT " } else { "ROUGE" }
            );
        }
        for e in &self.echecs {
            println!("  ROUGE {e}");
        }
    }
}

// ------------------------------------------------------------------------------------------
// Contrôles FIB-1 partagés (CPU et GPU)
// ------------------------------------------------------------------------------------------

/// Ce qu'un moteur (CPU ou GPU) a calculé pour un cas.
pub struct Resultats {
    pub pv: Vec<f64>,
    pub gp: Vec<f64>,
    pub mesure: MesureFibre,
    pub libre: (Vec<f64>, Vec<f64>),
}

/// Contrôles FIB-1 d'un cas contre les références de `cb.py` (et de lib_Cb). `et` préfixe les noms de gates.
pub fn controler_fib1(b: &mut Bilan, cas: &Cas, r: &Resultats, w: &PoidsFourier, et: &str) {
    let g = |s: &str| format!("{et}{s}");
    let info = cas.nom.as_str();
    let nf = w.nf;
    // --- plaquettes et gradients
    let v = cas.l.pow(4);
    if cas.existe("cb_pv.npy") {
        b.gate(
            &g("plaquettes pv"),
            max_abs_ecart(&r.pv, &cas.f64s("cb_pv.npy")),
            TOL_PLAQ,
            info,
        );
    } else {
        // plat / π-flux (jeu commité) : ±1 exactement
        let attendu = if cas.nom.ends_with("_plat") { 1.0 } else { -1.0 };
        b.vrai(
            &g("plaquettes pv = ±1 exactement"),
            r.pv.iter().all(|&x| x == attendu),
            info,
        );
    }
    if cas.existe("cb_gp.npy") {
        let pas = cas.meta["gp_pas"].as_u64().unwrap() as usize;
        let gref = cas.f64s("cb_gp.npy");
        let mut maxe = 0.0f64;
        for (i, ligne) in gref.chunks_exact(12).enumerate() {
            let p = i * pas;
            for (k, x) in ligne.iter().enumerate() {
                maxe = maxe.max((r.gp[12 * p + k] - x).abs());
            }
        }
        b.gate(&g("gradients"), maxe, TOL_PLAQ, info);
    } else if cas.plat_ou_pi() {
        b.vrai(&g("gradients nuls exactement"), r.gp.iter().all(|&x| x == 0.0), info);
    }
    let _ = v;
    // --- F : Wf dense (cb, BLAS), lib_Cb, somme exacte
    b.gate(
        &g("F contre cb (Wf dense) absolu"),
        max_abs_ecart(&r.mesure.f, &cas.f64s("cb_F.npy")),
        TOL_F_ABS,
        info,
    );
    if cas.existe("ref_F.npy") {
        b.gate(
            &g("F contre lib_Cb absolu"),
            max_abs_ecart(&r.mesure.f, &cas.f64s("ref_F.npy")),
            TOL_F_ABS,
            info,
        );
    }
    let fx: Vec<f64> = (0..nf)
        .map(|i| fsum((w.ptr[i] as usize..w.ptr[i + 1] as usize).map(|q| w.val[q] * r.pv[w.idx[q] as usize])))
        .collect();
    let rel = r
        .mesure
        .f
        .iter()
        .zip(&fx)
        .map(|(a, x)| (a - x).abs() / x.abs().max(1.0))
        .fold(0.0, f64::max);
    b.gate(&g("F Neumaier contre fsum relatif"), rel, TOL_FSUM_REL, info);
    // --- D
    let d_cb = cas.f64s("cb_D.npy");
    b.gate(
        &g("D Frobenius relatif contre cb"),
        ecart_frob(&r.mesure.d, &d_cb),
        TOL_D_REL,
        info,
    );
    if cas.existe("ref_D.npy") {
        b.gate(
            &g("D Frobenius relatif contre lib_Cb"),
            ecart_frob(&r.mesure.d, &cas.f64s("ref_D.npy")),
            TOL_D_REL,
            info,
        );
    }
    if cas.plat_ou_pi() {
        b.vrai(
            &g("plat / pi_flux : D = 0 exactement"),
            r.mesure.d.iter().all(|&x| x == 0.0),
            info,
        );
        b.vrai(
            &g("plat / pi_flux : D libre = 0 exactement"),
            r.libre.1.iter().all(|&x| x == 0.0),
            info,
        );
    }
    // --- blk : bits float32
    let blk_cb = f32s(&cas.dir.join("cb_blk.npy"));
    b.vrai(
        &g("blk identique (bits float32) à cb"),
        r.mesure.blk.len() == blk_cb.len()
            && r.mesure
                .blk
                .iter()
                .zip(&blk_cb)
                .all(|(a, c)| a.to_bits() == c.to_bits()),
        info,
    );
    if cas.existe("ref_blk.npy") {
        let br = f32s(&cas.dir.join("ref_blk.npy"));
        b.vrai(
            &g("blk identique (bits float32) à lib_Cb"),
            r.mesure.blk.iter().zip(&br).all(|(a, c)| a.to_bits() == c.to_bits()),
            info,
        );
    }
    // --- mesure libre
    b.gate(
        &g("mesure_libre D Frobenius relatif contre cb"),
        ecart_frob(&r.libre.1, &cas.f64s("cb_Dlibre.npy")),
        TOL_D_REL,
        info,
    );
    b.gate(
        &g("mesure_libre F absolu contre cb"),
        max_abs_ecart(&r.libre.0, &cas.f64s("cb_Flibre.npy")),
        TOL_F_ABS,
        info,
    );
    if cas.existe("ref_Dlibre.npy") {
        b.gate(
            &g("mesure_libre D Frobenius relatif contre lib_Cb"),
            ecart_frob(&r.libre.1, &cas.f64s("ref_Dlibre.npy")),
            TOL_D_REL,
            info,
        );
    }
}

/// Compare les poids de Fourier de Rust à ceux de Python (cb.fourier_weights_sparse) : mêmes lignes
/// (ptr, idx) EXACTEMENT, valeurs au plus à `tol_ulp` ulp près ; rend le plus grand écart relatif.
pub fn comparer_poids(b: &mut Bilan, racine: &Path, l: usize, w: &PoidsFourier) {
    let meta = lire_json(&racine.join(format!("wf_L{l}.json")));
    let pas = meta["pas"].as_u64().unwrap() as usize;
    let ptr = i64s(&racine.join(format!("wf_L{l}_ptr.npy")));
    let idx = i64s(&racine.join(format!("wf_L{l}_idx.npy")));
    let val = f64s(&racine.join(format!("wf_L{l}_val.npy")));
    let nom = format!("L{l} ");
    b.vrai(
        &format!("{nom}poids : ptr identique à cb"),
        w.ptr.iter().zip(&ptr).all(|(a, c)| *a as i64 == *c) && w.ptr.len() == ptr.len(),
        "",
    );
    let mut ecart_idx = 0usize;
    let mut maxrel = 0.0f64;
    for (k, (&i, &v)) in idx.iter().zip(&val).enumerate() {
        let q = k * pas;
        if w.idx[q] as i64 != i {
            ecart_idx += 1;
        }
        maxrel = maxrel.max(((w.val[q] - v) / v).abs());
    }
    b.vrai(&format!("{nom}poids : idx identique à cb"), ecart_idx == 0, "");
    b.gate(
        &format!("{nom}poids : valeurs (écart relatif max)"),
        maxrel,
        2.5e-16,
        "",
    );
}
