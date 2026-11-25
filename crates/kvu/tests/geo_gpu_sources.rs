//! Porte P1, sans GPU : les textes C0 restent identiques aux valeurs d'or indépendantes de `or_p1.py`.
//! Base de l'or : intégration 1c488c9 (base de la fusion de c09b ; obs_flot.cu y a été étendu par c08d, B08, pas par B10).
//! Valeurs d'ec6455d (= main 2f5f447) : b10/journaux/or_p1_ec6455d.txt ; C1 reste hors du registre gelé.

use kvu::commun::BackendKind;
use kvu::gpu::{assembler, SourceAssemblee};
use kvu::heritage::manifest::sha256_hex;
use kvu::noyaux::{
    self, Fichier, COND, OBS_FLOT, OBS_WILSON, RG_BLOC, SU2_BORD, SU2_COMMUN, SU2_F32, SU2_REF64,
};
use kvu::su2::{boite_gpu, geo_gpu, ref64};

// COND3 est privé dans cond3::gpu et absent du registre : même fichier embarqué, sans modifier son propriétaire.
static COND3: Fichier = Fichier {
    nom: "cond3.cu",
    contenu: include_str!("../noyaux/cond3.cu"),
};

const NOMS_ASSEMBLAGES: [&str; 8] = [
    "su2_ref64",
    "su2_ref64_metro",
    "su2_f32",
    "boite",
    "cond",
    "cond3",
    "obs",
    "rg_bloc",
];

const OR_HIP: [&str; 8] = [
    "2def59c9d9bd1c024088f402c877f8e2ad214234abbe14e566046ec22128a0b7",
    "f573b6ae11c56475974b090ffe0d4024f59dea4e37b06891ae417be82379e6da",
    "ba02a587525992bb9267dc80c87c2f960fd4d12142a82a696b8c2689b2eceff2",
    "72d9c3e6d1ebe7474c028ddc87b01c484919e8b7fe145e9d19b599529bdc3211",
    "271fa96b1507eae3d6aa76d4d5f0221e0e5e2b70ee6a9979425e2624c9505595",
    "47ef8a53fc3ef7c939ab2a261182a8aaba5aaae52a48eb44ce16a53a7db38daa",
    "d37ea370a002d6086edb1f371d94d122d03d6374090249926fef16d48e4d9557",
    "3c1e540c0c3596ab0ab418f2890fad0d7a3d4d187a11b123d9049c1e7f5d774b",
];

const OR_CUDA: [&str; 8] = [
    "0f42fe4ceb10bb675dba19dfe87ea227845a01aa697d4bd737610be9a156aec0",
    "606b83468e73772e380405e4520250eb26b323afec20e8fe9ea146f41cbd5c79",
    "46c1b1f0b7c7f630fd0525799aaea73dec0e2eeee01c62265732c1db7ae65598",
    "66ea02b68a85a3b8d60a9f157057092f9b3a8ec48e3530a381f259fd1e5b0c54",
    "5264bb1713557672fcbefe44e26c1a739f2d33e02aeb804e0917e9564cdaab49",
    "e6d503388158ad26daf069425c250144b063217595cf39929bcacc31d39959c1",
    "38caad272c3fb38b78210a486a53ed3f13bb7e3456eb33e7d55e4dfaf8fbb65e",
    "268cb2d856acd95d814bef7bc4d07ef5df15434336e3664d9f0f66c7c9090b9c",
];

// Les 13 fichiers du registre et de la boîte, plus cond3.cu : les 14 valeurs fournies sont vérifiées.
const OR_FICHIERS: [(&str, &str); 14] = [
    (
        "cond.cu",
        "121edf5ce6bbe95ea71d6b5bc3dd4725ac0f5258341d468f4346a4eddc122556",
    ),
    (
        "cond3.cu",
        "e538d80b8d3da80b0b3a6f73427f1461177f71926cfbb6fa1773ec96bb673543",
    ),
    (
        "fibre.cu",
        "3092b1a1cc48446488dbefed9328ab1ecd612d106c69c83520f0fb3263dd1bab",
    ),
    (
        "obs_flot.cu",
        "0f978e63286af52a8f719b24af52804a218aa72090c661de205e4aa825ad2a57",
    ),
    (
        "obs_wilson.cu",
        "1e7f25eb27de74a38f68482df939c65b47eecc9c8ca506aa7cf40dc622bd11ee",
    ),
    (
        "prelude_cuda.h",
        "0977a79c29c1f96e9a382000440581b2ced240a3dbb20cb38dfeeb816c01f52c",
    ),
    (
        "prelude_hip.h",
        "af9fd3106f6270485043c104a216c25dbc7e821f36f762b9c599e95e49dc624a",
    ),
    (
        "rg_bloc.cu",
        "32e71500b2df1ab7c397ac7ef8072223982470d19644c2010761e85266c9a888",
    ),
    (
        "su2_boite.cu",
        "204dda09b9b6743fe820c4c9d3f90c07f1865eae7f9dd6d48d54278373ee6cb0",
    ),
    (
        "su2_bord.cu",
        "716b2d38421f84665dc02bef6b5d049f4f1afabe7acf207d1d0d7affb853b834",
    ),
    (
        "su2_commun.h",
        "7c2a0b95b831e21ffc8c7fab4374657cc702096c2340d63ff5a2bac662cd2f25",
    ),
    (
        "su2_f32.cu",
        "21123c397e95a49659615fb4666e3eb9d722f0ad6ab2de58e00b5d075c8ecc7b",
    ),
    (
        "su2_ref64.cu",
        "12daefb862476e603000595e3ce52742a92c307b981dd67eabc2807b90427aea",
    ),
    (
        "test_runtime.cu",
        "6b429664d32848a60a4eab8960c968434147442c88b8c9426b17f75a831c38ad",
    ),
];

const NOMS_REGISTRE: [&str; 12] = [
    "prelude_hip.h",
    "prelude_cuda.h",
    "test_runtime.cu",
    "su2_commun.h",
    "su2_ref64.cu",
    "su2_bord.cu",
    "su2_f32.cu",
    "cond.cu",
    "fibre.cu",
    "obs_wilson.cu",
    "obs_flot.cu",
    "rg_bloc.cu",
];

fn or_assemblages(b: BackendKind) -> [&'static str; 8] {
    match b {
        BackendKind::Hip => OR_HIP,
        BackendKind::Cuda => OR_CUDA,
        BackendKind::Cpu => panic!("P1 : aucun assemblage GPU pour le backend CPU"),
    }
}

fn sources_c0(b: BackendKind) -> [SourceAssemblee; 8] {
    [
        ref64::source_ref64(b, false),
        ref64::source_ref64(b, true),
        assembler(b, "su2_f32", &[&SU2_COMMUN, &SU2_F32], &[]),
        boite_gpu::source_boite(b),
        assembler(b, "cond", &[&SU2_COMMUN, &COND], &[]),
        assembler(b, "cond3", &[&SU2_COMMUN, &COND, &COND3], &[]),
        assembler(b, "obs", &[&SU2_COMMUN, &OBS_WILSON, &OBS_FLOT], &[]),
        assembler(b, "rg_bloc", &[&SU2_COMMUN, &RG_BLOC], &[]),
    ]
}

#[test]
fn p1_textes_c0_inchanges() {
    let mut comparaisons = 0;
    for b in [BackendKind::Hip, BackendKind::Cuda] {
        for ((src, or), nom) in sources_c0(b)
            .iter()
            .zip(or_assemblages(b))
            .zip(NOMS_ASSEMBLAGES)
        {
            assert_eq!(src.sha256, or, "P1 : assemblage {nom}, backend {}", b.nom());
            comparaisons += 1;
        }
    }
    let mut fichiers = noyaux::empreintes();
    fichiers.push((
        boite_gpu::SU2_BOITE.nom,
        sha256_hex(boite_gpu::SU2_BOITE.contenu.as_bytes()),
    ));
    fichiers.push((COND3.nom, sha256_hex(COND3.contenu.as_bytes())));
    assert_eq!(
        fichiers.len(),
        OR_FICHIERS.len(),
        "P1 : nombre de fichiers hachés"
    );
    comparaisons += 1;
    for (nom, or) in OR_FICHIERS {
        let (_, sha) = fichiers
            .iter()
            .find(|(n, _)| *n == nom)
            .expect("fichier de l'or présent");
        assert_eq!(sha.as_str(), or, "P1 : contenu de {nom}");
        comparaisons += 1;
    }
    assert_eq!(noyaux::TOUS.len(), 12, "P1 : registre gelé à 12 fichiers");
    comparaisons += 1;
    let noms: Vec<&str> = noyaux::TOUS.iter().map(|f| f.nom).collect();
    assert_eq!(noms, NOMS_REGISTRE, "P1 : ordre et noms du registre gelé");
    comparaisons += NOMS_REGISTRE.len();
    for f in noyaux::TOUS {
        assert_ne!(
            f.nom, "su2_geo.cu",
            "P1 : C1 doit rester hors du registre gelé"
        );
        comparaisons += 1;
    }
    println!(
        "\nPASS P1-TEXTES-C0 : {comparaisons} comparaisons, 16 assemblages, 13 fichiers C0 et cond3.cu inchangés"
    );
}

#[test]
fn p1_mutant_fuite_c1_dans_c0() {
    let mut comparaisons = 0;
    for b in [BackendKind::Hip, BackendKind::Cuda] {
        for fichiers in [
            [&SU2_COMMUN, &SU2_REF64, &SU2_BORD, &geo_gpu::SU2_GEO],
            [&SU2_COMMUN, &SU2_REF64, &geo_gpu::SU2_GEO, &SU2_BORD],
        ] {
            let src = assembler(b, "su2_ref64", &fichiers, &[]);
            assert_ne!(
                src.sha256,
                or_assemblages(b)[0],
                "P1 : fuite C1 non détectée ({})",
                b.nom()
            );
            comparaisons += 1;
        }
    }
    println!("\nPASS P1-MUTANT-FUITE : {comparaisons} comparaisons, les deux insertions C1 rendent C0 rouge");
}

#[test]
fn p1_source_c1() {
    let mut comparaisons = 0;
    for b in [BackendKind::Hip, BackendKind::Cuda] {
        let src = geo_gpu::source_geo64(b, 0);
        assert_ne!(
            src.sha256,
            or_assemblages(b)[0],
            "P1 : empreinte C1 distincte de C0"
        );
        assert!(
            src.texte.contains("// ---- su2_geo.cu ----"),
            "P1 : séparateur C1 absent"
        );
        let prelude = if b == BackendKind::Hip {
            "prelude_hip.h"
        } else {
            "prelude_cuda.h"
        };
        let noms: Vec<&str> = src.parties.iter().map(|(nom, _)| nom.as_str()).collect();
        assert_eq!(
            noms,
            [prelude, "su2_commun.h", "su2_ref64.cu", "su2_geo.cu"]
        );
        let prefixe = assembler(b, "x", &[&SU2_COMMUN, &SU2_REF64], &[]);
        assert!(
            src.texte.starts_with(&prefixe.texte),
            "P1 : le préfixe commun C0 a changé"
        );
        comparaisons += 7; // empreinte, séparateur, quatre noms ordonnés et préfixe.
    }
    println!("\nPASS P1-SOURCE-C1 : {comparaisons} comparaisons, ordre et préfixe C0 conservés sur HIP et CUDA");
}
