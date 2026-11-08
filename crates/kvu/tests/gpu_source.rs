//! Tests CPU de l'assemblage des sources et des arguments de compilation (T2, gate RT-2 : « golden »).
//! Aucun GPU requis.

use kvu::commun::BackendKind;
use kvu::gpu::{assembler, essayer_assembler, FpContract, OptionsCompil};
use kvu::heritage::manifest::sha256_hex;
use kvu::noyaux::{Fichier, PRELUDE_CUDA, PRELUDE_HIP, TEST_RUNTIME};

static A: Fichier = Fichier {
    nom: "a.cu",
    contenu: "int a;\n",
};
static B: Fichier = Fichier {
    nom: "b.cu",
    contenu: "int b;\n",
};

const ARGS_GOLDEN: &str = include_str!("donnees/gpu/arguments_golden.json");
const ASM_GOLDEN: &str = include_str!("donnees/gpu/assemblage_sha256.json");

fn backend_de(nom: &str) -> BackendKind {
    nom.parse().expect("backend du golden")
}

// ---------------------------------------------------------------------------------------------
// Assemblage
// ---------------------------------------------------------------------------------------------

#[test]
fn assemblage_exact_hip() {
    let s = assembler(BackendKind::Hip, "essai", &[&A, &B], &[("N", "4"), ("DRAPEAU", "")]);
    let attendu = format!(
        "// ---- prelude_hip.h ----\n{}#define N 4\n#define DRAPEAU\n// ---- a.cu ----\nint a;\n// ---- b.cu ----\nint b;\n",
        PRELUDE_HIP.contenu
    );
    assert_eq!(s.texte, attendu);
    assert_eq!(s.nom, "essai");
    assert_eq!(s.backend, BackendKind::Hip);
    assert_eq!(s.sha256, sha256_hex(attendu.as_bytes()));
}

#[test]
fn assemblage_exact_cuda() {
    let s = assembler(BackendKind::Cuda, "essai", &[&B, &A], &[]);
    let attendu = format!(
        "// ---- prelude_cuda.h ----\n{}// ---- b.cu ----\nint b;\n// ---- a.cu ----\nint a;\n",
        PRELUDE_CUDA.contenu
    );
    assert_eq!(s.texte, attendu, "l'ordre des fichiers est celui demandé");
}

#[test]
fn assemblage_deterministe() {
    let a = assembler(BackendKind::Hip, "t", &[&A, &B], &[("X", "1")]);
    let b = assembler(BackendKind::Hip, "t", &[&A, &B], &[("X", "1")]);
    assert_eq!(a, b);
    assert_eq!(a.texte.as_bytes(), b.texte.as_bytes());
}

#[test]
fn le_texte_change_avec_define_ordre_fichier_et_backend() {
    let base = assembler(BackendKind::Hip, "t", &[&A, &B], &[("X", "1")]);
    let autre_valeur = assembler(BackendKind::Hip, "t", &[&A, &B], &[("X", "2")]);
    let autre_ordre = assembler(BackendKind::Hip, "t", &[&B, &A], &[("X", "1")]);
    let sans_b = assembler(BackendKind::Hip, "t", &[&A], &[("X", "1")]);
    let cuda = assembler(BackendKind::Cuda, "t", &[&A, &B], &[("X", "1")]);
    let hashes = [&base, &autre_valeur, &autre_ordre, &sans_b, &cuda].map(|s| s.sha256.clone());
    for i in 0..hashes.len() {
        for j in (i + 1)..hashes.len() {
            assert_ne!(hashes[i], hashes[j], "cas {i} et {j} devraient différer");
        }
    }
}

#[test]
fn le_nom_logique_ne_change_pas_le_texte() {
    let a = assembler(BackendKind::Hip, "un", &[&A], &[]);
    let b = assembler(BackendKind::Hip, "deux", &[&A], &[]);
    assert_eq!(a.texte, b.texte);
    assert_eq!(a.sha256, b.sha256);
}

#[test]
fn parties_dans_l_ordre_prelude_compris() {
    let s = assembler(BackendKind::Cuda, "t", &[&A, &B], &[]);
    let noms: Vec<&str> = s.parties.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(noms, ["prelude_cuda.h", "a.cu", "b.cu"]);
    for (n, h) in &s.parties {
        let contenu = match n.as_str() {
            "prelude_cuda.h" => PRELUDE_CUDA.contenu,
            "a.cu" => A.contenu,
            _ => B.contenu,
        };
        assert_eq!(h, &sha256_hex(contenu.as_bytes()), "sha256 de {n}");
    }
}

#[test]
fn valeurs_d_or_independantes_du_code() {
    // Calculées par un script Python indépendant (tests/donnees/gpu/assemblage_sha256.json).
    let or: serde_json::Value = serde_json::from_str(ASM_GOLDEN).expect("JSON d'or");
    for (cle, backend) in [("hip", BackendKind::Hip), ("cuda", BackendKind::Cuda)] {
        let s = assembler(
            backend,
            "test_runtime",
            &[&TEST_RUNTIME],
            &[("KV_ESSAI", "42"), ("AUTRE", "x")],
        );
        assert_eq!(s.sha256, or[cle]["sha256"].as_str().unwrap(), "sha256 assemblé {cle}");
        let attendu: Vec<(String, String)> = or[cle]["parties"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p[0].as_str().unwrap().to_string(), p[1].as_str().unwrap().to_string()))
            .collect();
        assert_eq!(s.parties, attendu, "parties {cle}");
    }
}

#[test]
fn refus_du_cpu_et_des_defines_invalides() {
    assert!(essayer_assembler(BackendKind::Cpu, "t", &[&A], &[]).is_err());
    assert!(essayer_assembler(BackendKind::Hip, "t", &[&A], &[("1X", "1")]).is_err());
    assert!(essayer_assembler(BackendKind::Hip, "t", &[&A], &[("X", "1\n#error")]).is_err());
}

#[test]
#[should_panic(expected = "assembler")]
fn assembler_panique_pour_le_cpu() {
    let _ = assembler(BackendKind::Cpu, "t", &[&A], &[]);
}

// ---------------------------------------------------------------------------------------------
// Préludes et noyaux d'essai du registre
// ---------------------------------------------------------------------------------------------

#[test]
fn preludes_conformes_a_la_specification() {
    assert!(PRELUDE_HIP.contenu.contains("#include <hip/hip_runtime.h>"));
    assert!(PRELUDE_HIP.contenu.contains("#define KV_HIP 1"));
    assert!(!PRELUDE_HIP.contenu.contains("#define KV_CUDA"));
    assert!(PRELUDE_CUDA.contenu.contains("#define KV_CUDA 1"));
    assert!(!PRELUDE_CUDA.contenu.contains("#define KV_HIP"));
    // NVRTC fournit lui-même les built-ins : aucune directive #include dans le prélude CUDA.
    assert!(!PRELUDE_CUDA
        .contenu
        .lines()
        .any(|l| l.trim_start().starts_with("#include")));
}

#[test]
fn noyaux_d_essai_presents_en_extern_c() {
    for nom in ["k_remplir", "k_saxpy_f64", "k_reduce", "k_final", "k_somme_u64"] {
        let motif = format!("extern \"C\" __global__ void {nom}(");
        assert!(
            TEST_RUNTIME.contenu.contains(&motif),
            "noyau {nom} absent ou non extern \"C\""
        );
    }
    assert!(TEST_RUNTIME.contenu.contains("atomicAdd"));
    assert!(TEST_RUNTIME.contenu.contains("__shared__ double sh[256]"));
}

// ---------------------------------------------------------------------------------------------
// Arguments de compilation
// ---------------------------------------------------------------------------------------------

#[test]
fn arguments_valeurs_d_or() {
    let or: serde_json::Value = serde_json::from_str(ARGS_GOLDEN).expect("JSON d'or");
    let cas = or["cas"].as_array().unwrap();
    assert!(cas.len() >= 10);
    for c in cas {
        let nom = c["nom"].as_str().unwrap();
        let opts: OptionsCompil = serde_json::from_value(c["options"].clone()).expect("options du golden");
        let attendu: Vec<String> = c["attendu"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let obtenu = opts.arguments(backend_de(c["backend"].as_str().unwrap()), c["arch"].as_str().unwrap());
        assert_eq!(obtenu, attendu, "cas {nom}");
    }
}

#[test]
fn constructeurs_d_options() {
    let k1 = OptionsCompil::reference_k1();
    assert_eq!((k1.fp_contract, k1.optimisation), (FpContract::Off, 3));
    assert!(k1.extra.is_empty());
    assert_eq!(
        OptionsCompil::compat_production(BackendKind::Hip).fp_contract,
        FpContract::FastHonorPragmas
    );
    assert_eq!(
        OptionsCompil::compat_production(BackendKind::Cuda).fp_contract,
        FpContract::Fast
    );
    assert_eq!(OptionsCompil::rapide().fp_contract, FpContract::Fast);
}

#[test]
fn arguments_des_constructeurs() {
    let hip = |o: OptionsCompil| o.arguments(BackendKind::Hip, "gfx1151");
    let cuda = |o: OptionsCompil| o.arguments(BackendKind::Cuda, "sm_120");
    assert_eq!(
        hip(OptionsCompil::reference_k1()),
        ["--offload-arch=gfx1151", "-O3", "-ffp-contract=off", "-std=c++17"]
    );
    assert_eq!(
        hip(OptionsCompil::compat_production(BackendKind::Hip)),
        [
            "--offload-arch=gfx1151",
            "-O3",
            "-ffp-contract=fast-honor-pragmas",
            "-std=c++17"
        ]
    );
    assert_eq!(
        cuda(OptionsCompil::reference_k1()),
        ["-arch=sm_120", "--fmad=false", "-std=c++17"]
    );
    assert_eq!(
        cuda(OptionsCompil::compat_production(BackendKind::Cuda)),
        ["-arch=sm_120", "--fmad=true", "-std=c++17"]
    );
    assert_eq!(
        cuda(OptionsCompil::rapide()),
        ["-arch=sm_120", "--fmad=true", "-std=c++17"]
    );
}

#[test]
fn cuda_fmad_faux_seulement_pour_off() {
    for fp in [FpContract::On, FpContract::Fast, FpContract::FastHonorPragmas] {
        let o = OptionsCompil {
            fp_contract: fp,
            optimisation: 3,
            extra: vec![],
        };
        assert!(
            o.arguments(BackendKind::Cuda, "sm_120")
                .contains(&"--fmad=true".to_string()),
            "{fp:?}"
        );
    }
    let o = OptionsCompil {
        fp_contract: FpContract::Off,
        optimisation: 3,
        extra: vec![],
    };
    assert!(o
        .arguments(BackendKind::Cuda, "sm_120")
        .contains(&"--fmad=false".to_string()));
}

#[test]
fn extra_ajoute_en_dernier_et_avec_extra() {
    let o = OptionsCompil::reference_k1().avec_extra("-DA=1").avec_extra("-DB=2");
    let a = o.arguments(BackendKind::Hip, "gfx1151");
    assert_eq!(&a[a.len() - 2..], ["-DA=1", "-DB=2"]);
}

#[test]
fn options_serialisables_et_stables() {
    let o = OptionsCompil::compat_production(BackendKind::Hip).avec_extra("-DX");
    let json = serde_json::to_string(&o).unwrap();
    assert!(json.contains("\"fast-honor-pragmas\""));
    assert_eq!(serde_json::from_str::<OptionsCompil>(&json).unwrap(), o);
}
