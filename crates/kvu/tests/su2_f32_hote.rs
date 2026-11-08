//! Tests CPU (sans GPU) du chemin FP32 : loi du tirage de x0 en arithmétique float (miroir hôte du noyau),
//! Philox, moments exacts. Ils prouvent que l'ALGORITHME float est exact avant même de le lancer sur un GPU
//! (le noyau lui-même est jugé par `su2_f32_gpu.rs` et validation/f32/gates_f32.sh).

use kvu::heritage::su2_hb::philox4x32_10;
use kvu::su2::rapide32::{tirage_x0_hote, u01f_hote};

/// Moments exacts E[x^p], p = 1..4, de la densité sqrt(1-x²) exp(α x) sur [-1,1] : quadrature de
/// Gauss-Tchebychev de 2e espèce à `n` nœuds (comme reference/su2_cuda/test_heatbath_cuda.py).
fn moments_exacts(alpha: f64, n: usize) -> [f64; 4] {
    let (mut z, mut m) = (0.0, [0.0; 4]);
    for k in 1..=n {
        let th = k as f64 * std::f64::consts::PI / (n as f64 + 1.0);
        let x = th.cos();
        let w = std::f64::consts::PI / (n as f64 + 1.0) * th.sin().powi(2) * (alpha * (x - 1.0)).exp();
        z += w;
        for (p, mp) in m.iter_mut().enumerate() {
            *mp += w * x.powi(p as i32 + 1);
        }
    }
    m.map(|v| v / z)
}

/// Fonction de Bessel modifiée I_nu(x) par sa série (entier nu), pour contrôler la quadrature.
fn bessel_i(nu: u32, x: f64) -> f64 {
    let (mut terme, mut somme) = ((x / 2.0).powi(nu as i32) / (1..=nu).map(f64::from).product::<f64>(), 0.0);
    for k in 0..200u32 {
        somme += terme;
        terme *= (x / 2.0).powi(2) / (f64::from(k + 1) * f64::from(k + 1 + nu));
    }
    somme
}

/// CDF exacte de la densité (x = cos θ, trapèzes fins), évaluée aux points `x` triés.
fn cdf_exacte(alpha: f64, ngrille: usize) -> impl Fn(f64) -> f64 {
    let pas = std::f64::consts::PI / (ngrille as f64 - 1.0);
    let mut cumul = vec![0.0f64; ngrille];
    let f = |th: f64| th.sin().powi(2) * (alpha * (th.cos() - 1.0)).exp();
    for i in 1..ngrille {
        cumul[i] = cumul[i - 1] + 0.5 * (f((i - 1) as f64 * pas) + f(i as f64 * pas)) * pas;
    }
    let z = cumul[ngrille - 1];
    move |x: f64| {
        // F(x) = (Z - C(θ_x)) / Z, θ_x = arccos x, interpolation linéaire
        let th = x.clamp(-1.0, 1.0).acos();
        let t = th / pas;
        let i = (t.floor() as usize).min(ngrille - 2);
        let c = cumul[i] + (cumul[i + 1] - cumul[i]) * (t - i as f64);
        (z - c) / z
    }
}

/// Statistique de Kolmogorov-Smirnov D et p-valeur asymptotique (série de Kolmogorov) contre une CDF.
fn ks(mut x: Vec<f64>, cdf: impl Fn(f64) -> f64) -> (f64, f64) {
    x.sort_by(|a, b| a.partial_cmp(b).expect("pas de NaN"));
    let n = x.len() as f64;
    let d = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let f = cdf(v);
            ((i as f64 + 1.0) / n - f).abs().max((f - i as f64 / n).abs())
        })
        .fold(0.0, f64::max);
    let t = (n.sqrt() + 0.12 + 0.11 / n.sqrt()) * d;
    let p: f64 = 2.0 * (1..100).map(|j| (-1f64).powi(j - 1) * (-2.0 * f64::from(j * j) * t * t).exp()).sum::<f64>();
    (d, p.clamp(0.0, 1.0))
}

fn tirages(alpha: f32, n: usize, graine: u64) -> (Vec<f32>, u64) {
    let (k0, k1) = (graine as u32, (graine >> 32) as u32);
    let (mut v, mut plafond) = (Vec::with_capacity(n), 0u64);
    for i in 0..n as u32 {
        let mut draw = 0;
        let (x, _, ok) = tirage_x0_hote(alpha, i, 0x7E57, &mut draw, k0, k1);
        plafond += u64::from(!ok);
        v.push(x);
    }
    (v, plafond)
}

#[test]
fn quadrature_gauss_tchebychev_controlee_par_bessel() {
    // E[x0] = I_2(α) / I_1(α) (la densité sqrt(1-x²) e^{αx} est celle de I_1)
    for alpha in [0.3, 1.0, 5.0, 20.0] {
        let m1 = moments_exacts(alpha, 4000)[0];
        let b = bessel_i(2, alpha) / bessel_i(1, alpha);
        assert!((m1 - b).abs() < 1e-12, "α = {alpha} : quadrature {m1} contre Bessel {b}");
    }
}

#[test]
fn cdf_exacte_est_une_cdf() {
    for alpha in [0.3, 5.0, 100.0] {
        let f = cdf_exacte(alpha, 200_001);
        assert!(f(-1.0) < 1e-12 && (f(1.0) - 1.0).abs() < 1e-12);
        let mut prec = -1.0;
        for i in 0..=200 {
            let v = f(-1.0 + 2.0 * f64::from(i) / 200.0);
            assert!(v >= prec - 1e-15, "CDF non monotone");
            prec = v;
        }
    }
}

#[test]
fn philox_hote_reproduit_les_vecteurs_de_reference() {
    // Random123 philox4x32_10 (KAT de selftest de su2_hip.cpp)
    assert_eq!(philox4x32_10([0, 0, 0, 0], 0, 0), [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);
    assert_eq!(
        philox4x32_10([0xffff_ffff; 4], 0xffff_ffff, 0xffff_ffff),
        [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]
    );
    assert_eq!(
        philox4x32_10([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344], 0xa4093822, 0x299f31d0),
        [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]
    );
}

/// LOI-1 sur le miroir hôte : 7 α, moments (|z| < 4) et KS contre la CDF exacte (p > 1e-4), plafond 0.
/// N réduit à 3·10⁵ pour tenir en `cargo test` ; le noyau lui-même est jugé à N = 2·10⁷ par les gates GPU.
#[test]
fn loi_1_miroir_hote_sept_alphas() {
    const N: usize = 300_000;
    for (i, &alpha) in [0.3f32, 1.0, 1.999, 2.0, 5.0, 20.0, 100.0].iter().enumerate() {
        let (x, plafond) = tirages(alpha, N, 9000 + i as u64);
        assert_eq!(plafond, 0, "α = {alpha} : plafond d'essais atteint");
        let a = f64::from(alpha);
        let e = moments_exacts(a, 4000);
        let (m1, m2) = (
            x.iter().map(|&v| f64::from(v)).sum::<f64>() / N as f64,
            x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / N as f64,
        );
        let (sd1, sd2) = ((e[1] - e[0] * e[0]).sqrt(), (e[3] - e[1] * e[1]).sqrt());
        let (z1, z2) = ((m1 - e[0]) / (sd1 / (N as f64).sqrt()), (m2 - e[1]) / (sd2 / (N as f64).sqrt()));
        assert!(z1.abs() < 4.0 && z2.abs() < 4.0, "α = {alpha} : z1 = {z1:.2}, z2 = {z2:.2}");
        let (d, p) = ks(x.iter().map(|&v| f64::from(v)).collect(), cdf_exacte(a, 400_001));
        assert!(p > 1e-4, "α = {alpha} : KS D = {d:.3e}, p = {p:.3e}");
        assert!(x.iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }
}

/// Puissance du test : le Metropolis naïf depuis x = 1 (contrôle négatif, tirage uniforme) est détecté
/// sans ambiguïté par les mêmes statistiques, à α ≥ 5.
#[test]
fn le_test_de_loi_detecte_un_tirage_uniforme() {
    const N: usize = 100_000;
    for &alpha in &[5.0f64, 20.0, 100.0] {
        let x: Vec<f64> = (0..N as u32)
            .map(|i| {
                let r = philox4x32_10([i, 0x7E57, 0, 0x4842_4154], 1, 2);
                2.0 * f64::from(u01f_hote(r[0])) - 1.0
            })
            .collect();
        let e = moments_exacts(alpha, 4000);
        let m1 = x.iter().sum::<f64>() / N as f64;
        let z1 = (m1 - e[0]) / ((e[1] - e[0] * e[0]).sqrt() / (N as f64).sqrt());
        assert!(z1.abs() > 100.0, "α = {alpha} : z1 = {z1:.1} (le mutant doit être ROUGE)");
    }
}

#[test]
fn frontiere_creutz_kennedy_pendleton_continue() {
    // la moyenne varie lissement de α = 1,999 (Creutz) à 2,0 (KP) : écart bien sous l'erreur statistique
    const N: usize = 400_000;
    let m = |alpha: f32, g: u64| tirages(alpha, N, g).0.iter().map(|&v| f64::from(v)).sum::<f64>() / N as f64;
    let (a, b) = (m(1.999, 31), m(2.0, 32));
    let attendu = moments_exacts(2.0, 4000)[0] - moments_exacts(1.999, 4000)[0];
    assert!((b - a - attendu).abs() < 6.0 * 0.45 / (N as f64).sqrt(), "saut à la frontière : {} contre {attendu}", b - a);
}

#[test]
fn petit_alpha_uniforme_et_alpha_nul_admis() {
    // α < 1e-6 : proposition uniforme sous la loi sqrt(1-x²) ⇒ E[x] = 0, E[x²] = 1/4
    const N: usize = 200_000;
    let (x, plafond) = tirages(0.0, N, 5);
    assert_eq!(plafond, 0);
    let m1 = x.iter().map(|&v| f64::from(v)).sum::<f64>() / N as f64;
    let m2 = x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / N as f64;
    assert!(m1.abs() < 4.0 * 0.5 / (N as f64).sqrt(), "m1 = {m1}");
    assert!((m2 - 0.25).abs() < 4.0 * 0.25 / (N as f64).sqrt(), "m2 = {m2}");
}

// ------------------------------------------------------------------------------------------
// Assemblage et compilation hors GPU de la source assemblée (prélude + su2_commun.h + su2_f32.cu)
// ------------------------------------------------------------------------------------------

use kvu::commun::BackendKind;
use kvu::gpu::{assembler, SourceAssemblee};
use kvu::noyaux::{SU2_COMMUN, SU2_F32};
use std::path::PathBuf;
use std::process::Command;

fn assemblee(backend: BackendKind, mutant: bool) -> SourceAssemblee {
    let defines: &[(&str, &str)] = if mutant { &[("K1_MUTANT_METRO", "1")] } else { &[] };
    assembler(backend, "su2_f32", &[&SU2_COMMUN, &SU2_F32], defines)
}

#[test]
fn assemblage_ordre_empreinte_et_mutant() {
    for b in [BackendKind::Hip, BackendKind::Cuda] {
        let (n, m) = (assemblee(b, false), assemblee(b, true));
        let noms: Vec<&str> = n.parties.iter().map(|(f, _)| f.as_str()).collect();
        let prelude = if b == BackendKind::Hip { "prelude_hip.h" } else { "prelude_cuda.h" };
        assert_eq!(noms, [prelude, "su2_commun.h", "su2_f32.cu"], "ordre d'assemblage ({b:?})");
        let (i_commun, i_f32) = (n.texte.find("struct U4").expect("Philox"), n.texte.find("struct __attribute__((aligned(16))) QF").expect("QF"));
        assert!(i_commun < i_f32, "su2_commun.h doit précéder su2_f32.cu");
        assert_eq!(n.sha256, assemblee(b, false).sha256, "assemblage déterministe");
        assert_ne!(n.sha256, m.sha256, "le mutant change l'empreinte de la source");
        assert!(m.texte.contains("#define K1_MUTANT_METRO 1\n") && !n.texte.contains("#define K1_MUTANT_METRO"));
        assert!(m.texte.find("#define K1_MUTANT_METRO").expect("define") < m.texte.find("#ifdef K1_MUTANT_METRO").expect("ifdef"));
    }
}

/// Écrit la source assemblée dans un fichier temporaire et la compile avec `commande` ; SKIP si le compilateur manque.
fn compiler_hors_gpu(nom: &str, ext: &str, texte: &str, commande: &str, args: &[&str]) {
    let dir = std::env::temp_dir().join(format!("kvu_f32_{nom}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("répertoire temporaire");
    let src: PathBuf = dir.join(format!("su2_f32.{ext}"));
    std::fs::write(&src, texte).expect("écriture de la source");
    let out = Command::new(commande).args(args).arg("-o").arg(dir.join("sortie.o")).arg(&src).output();
    match out {
        Err(_) => {
            println!("SKIP {nom} : {commande} absent");
            std::fs::remove_dir_all(&dir).ok();
            return;
        }
        Ok(o) => {
            let journal = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
            std::fs::remove_dir_all(&dir).ok();
            assert!(o.status.success(), "{nom} : compilation en échec :\n{journal}");
            assert!(!journal.contains("warning"), "{nom} : avertissements du compilateur :\n{journal}");
        }
    }
    println!("PASS {nom}");
}

#[test]
fn source_assemblee_compile_avec_hipcc_sans_gpu() {
    for mutant in [false, true] {
        compiler_hors_gpu(
            &format!("hip_hors_gpu_mutant_{mutant}"),
            "hip",
            &assemblee(BackendKind::Hip, mutant).texte,
            "hipcc",
            &["--offload-arch=gfx1151", "-O1", "-Wall", "--cuda-device-only", "-c", "-x", "hip"],
        );
    }
}

#[test]
fn source_assemblee_compile_avec_nvcc_sans_gpu() {
    let nvcc = if std::path::Path::new("/usr/local/cuda-13.3/bin/nvcc").exists() { "/usr/local/cuda-13.3/bin/nvcc" } else { "nvcc" };
    for mutant in [false, true] {
        compiler_hors_gpu(
            &format!("cuda_hors_gpu_mutant_{mutant}"),
            "cu",
            &assemblee(BackendKind::Cuda, mutant).texte,
            nvcc,
            &["-arch=sm_120", "-O1", "-std=c++17", "--cubin", "-x", "cu"],
        );
    }
}
