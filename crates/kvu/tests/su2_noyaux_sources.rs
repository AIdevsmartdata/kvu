//! Contrat de bit-identité des sources : `su2_commun.h` est la recopie caractère pour caractère du socle de
//! `src/hip/su2_sweep.hip` (lignes 27 à 130), et les noyaux de T3 portent les marqueurs attendus.
//! Tests CPU, sans GPU.

use kvu::heritage::manifest::sha256_hex;
use std::path::PathBuf;

fn racine_depot() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn lire(rel: &str) -> String {
    let p = racine_depot().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("lecture de {} : {e}", p.display()))
}

/// Bloc recopié : de « #ifndef M_PI » (inclus) à la ligne qui précède l'en-tête « // Noyaux » (exclue).
fn bloc_source() -> String {
    let src = lire("src/hip/su2_sweep.hip");
    let debut = src.find("#ifndef M_PI").expect("M_PI absent de su2_sweep.hip");
    let fin = src.find("// ------------------------------------------------------------------------------------------\n// Noyaux").expect("en-tête Noyaux");
    src[debut..fin].trim_end_matches('\n').to_string()
}

fn bloc_commun() -> String {
    let h = lire("crates/kvu/noyaux/su2_commun.h");
    let m1 = "// DEBUT DE LA RECOPIE (src/hip/su2_sweep.hip, ligne 27)\n";
    let m2 = "\n// FIN DE LA RECOPIE (src/hip/su2_sweep.hip, ligne 130)";
    let a = h.find(m1).expect("marqueur de début") + m1.len();
    let b = h.find(m2).expect("marqueur de fin");
    h[a..b].trim_end_matches('\n').to_string()
}

#[test]
fn su2_sweep_hip_est_celui_de_l_inventaire() {
    let src = lire("src/hip/su2_sweep.hip");
    assert_eq!(
        sha256_hex(src.as_bytes()),
        "d04a876a8b5c5e493eb3b5d96eb17685d25740c19cfa395ba47c5731115fe2b2",
        "src/hip/su2_sweep.hip a changé : relancer les gates T3 (SOURCES_IMPORTEES.md)"
    );
}

#[test]
fn su2_commun_recopie_caractere_pour_caractere() {
    assert_eq!(bloc_commun(), bloc_source(), "su2_commun.h diverge de su2_sweep.hip");
}

#[test]
fn su2_commun_contient_le_socle_attendu() {
    let h = lire("crates/kvu/noyaux/su2_commun.h");
    for motif in [
        "#define M_PI 3.14159265358979323846",
        "philox4x32_10(U4 c, u32 k0, u32 k1)",
        "double u01(u32 a, u32 b)",
        "TAG_INIT = 0x494E4954u, TAG_HB = 0x48424154u",
        "struct Q { double x, y, z, w; }",
        "struct Geo { int L; long V; long st[4]; }",
        "Geo mkgeo(int L)",
        "long nb_up(", "long nb_dn(", "void coords(", "Q ld(", "Q staple(", "long site_of(",
    ] {
        assert!(h.contains(motif), "motif absent de su2_commun.h : {motif}");
    }
    assert!(
        !h.lines().any(|l| l.trim_start().starts_with("#include")),
        "l'#include est dans le prélude, pas dans su2_commun.h"
    );
}

// ---------------------------------------------------------------------------------------------
// Arithmétique inchangée : chaque ligne de calcul de su2_sweep.hip (K1) et de su2_bord.cu (R2-3) se retrouve
// VERBATIM dans su2_ref64.cu / su2_bord.cu. Seules les lignes de signature et de structure de lots diffèrent ;
// la liste exacte des écarts est gelée ici (tout autre écart fait échouer le test).
// ---------------------------------------------------------------------------------------------

fn lignes_utiles(texte: &str) -> Vec<String> {
    texte
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with("//"))
        .collect()
}

fn manquantes(source: &[String], cible: &[String]) -> Vec<String> {
    let c: std::collections::HashSet<&String> = cible.iter().collect();
    source.iter().filter(|l| !c.contains(l)).cloned().collect()
}

#[test]
fn su2_ref64_garde_l_arithmetique_de_k1() {
    let k1 = lire("src/hip/su2_sweep.hip");
    let apres = &k1[k1.find("// Noyaux").expect("en-tête Noyaux")..];
    let ref64 = lire("crates/kvu/noyaux/su2_ref64.cu");
    let manque = manquantes(&lignes_utiles(apres), &lignes_utiles(&ref64));
    let attendu: Vec<String> = [
        r#"extern "C" __global__ void k_init(Q* U, int Lg, int hot, u32 k0, u32 k1) {"#,
        "if (!hot) { Q e = {1, 0, 0, 0}; U[i] = e; return; }",
        r#"extern "C" __global__ void k_heatbath(Q* U, int Lg, int mu, int parity, double beta, u32 k0, u32 k1, u32 sweep_id,"#,
        "unsigned long long* n_cap, unsigned long long* n_tries, u32* diag_ntry) {",
        "if (!(k > 1e-300)) {   // agrafe nulle : mesure de Haar (jamais observe en pratique)",
        "double alpha = beta * k;          // densite de x0 : sqrt(1-x0^2) exp(alpha x0)",
        "atomicAdd(n_tries, (unsigned long long)ntry);",
        "if (!ok) { atomicAdd(n_cap, 1ull); return; }   // compte ; lien inchange (attendu : 0)",
        r#"extern "C" __global__ void k_overrelax(Q* U, int Lg, int mu, int parity) {"#,
        r#"extern "C" __global__ void k_final(const double* __restrict__ in, long n, double scale, double* out) {"#,
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(manque, attendu, "écart inattendu entre su2_sweep.hip et su2_ref64.cu");
}

#[test]
fn su2_bord_garde_l_arithmetique_de_su2_bord() {
    let b = lire("crates/kvu/reference/su2_bord/su2_bord.cu");
    let (i, j) = (b.find("// Ajouts R2-3").expect("Ajouts R2-3"), b.find("// Hôte").expect("Hôte"));
    let bord = lire("crates/kvu/noyaux/su2_bord.cu");
    let manque = manquantes(&lignes_utiles(&b[i..j]), &lignes_utiles(&bord));
    let attendu: Vec<String> = [
        "__global__ void k_set_bord(Q* U, Geo g, int mode, u32 k0, u32 k1, const Q* __restrict__ src) {",
        "__global__ void k_gen_jauge(Q* G, Geo g, u32 k0, u32 k1) {",
        "__global__ void k_app_jauge(Q* U, Geo g, const Q* __restrict__ G) {",
        "__global__ void k_plaq_st(const Q* __restrict__ U, Geo g, double* outS, double* outT) {",
        "__global__ void k_tranche(const double* __restrict__ in, Geo g, double scale, double* out) {",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(manque, attendu, "écart inattendu entre su2_bord.cu (référence) et noyaux/su2_bord.cu");
    // étiquettes et emplacements de compteur
    for motif in ["TAG_BORD = 0x424F5244u", "TAG_JAUGE = 0x4A415547u", "0xFFFFFFFEu, TAG_BORD", "0xFFFFFFFDu, TAG_JAUGE"] {
        assert!(bord.contains(motif), "motif absent : {motif}");
    }
}

#[test]
fn le_gel_est_place_comme_dans_su2_bord() {
    // dans k_heatbath et k_overrelax, la ligne du gel suit IMMÉDIATEMENT site_of, avant le calcul de li
    let ref64 = lire("crates/kvu/noyaux/su2_ref64.cu");
    let l = lignes_utiles(&ref64);
    let gel = "if (gel && mu != 0 && x[0] == 0) return;   // lien de bord gelé";
    let n = l.iter().filter(|x| x.as_str() == gel).count();
    assert_eq!(n, 2, "le gel doit figurer dans k_heatbath ET k_overrelax");
    for (i, x) in l.iter().enumerate() {
        if x == gel {
            assert_eq!(l[i - 1], "long s = site_of(g, t, parity, x);");
            assert!(l[i + 1].starts_with("long li = (long)mu * g.V + s;"), "li local juste après le gel : {}", l[i + 1]);
        }
    }
    // l'original
    let orig = lire("crates/kvu/reference/su2_bord/su2_bord.cu");
    assert_eq!(orig.matches("if (g.freeze && mu != 0 && x[0] == 0) return;   // lien de bord gelé").count(), 2);
}

#[test]
fn le_compteur_philox_utilise_l_indice_de_lien_local() {
    let ref64 = lire("crates/kvu/noyaux/su2_ref64.cu");
    // aucun décalage de chaîne dans les compteurs : (u32)li, li = mu·V + s
    assert!(ref64.contains("U4{{(u32)li, sweep_id, draw++, TAG_HB}}"));
    assert!(ref64.contains("// indice de lien LOCAL"));
    // la clé et β sont lus par chaîne
    assert!(ref64.contains("const double beta = betas[c];"));
    assert!(ref64.contains("const u32 k0 = k0s[c], k1 = k1s[c];"));
    assert!(ref64.contains("atomicAdd(n_tries + c, "));
    assert!(ref64.contains("atomicAdd(n_cap + c, 1ull);"));
}

#[test]
fn assemblage_hip_et_cuda_du_chemin_f64() {
    use kvu::commun::BackendKind;
    use kvu::su2::ref64::source_ref64;
    for b in [BackendKind::Hip, BackendKind::Cuda] {
        let s = source_ref64(b, false);
        let noms: Vec<&str> = s.parties.iter().map(|(n, _)| n.as_str()).collect();
        let pos = |n: &str| noms.iter().position(|x| *x == n).unwrap_or_else(|| panic!("partie {n} absente : {noms:?}"));
        assert!(pos("su2_commun.h") < pos("su2_ref64.cu") && pos("su2_ref64.cu") < pos("su2_bord.cu"), "{noms:?}");
        for k in [
            "k_init", "k_test_x0", "k_heatbath", "k_overrelax", "k_plaq_site", "k_reduce", "k_final", "k_polyakov",
            "k_set_bord", "k_gen_jauge", "k_app_jauge", "k_plaq_st", "k_tranche",
        ] {
            assert!(s.texte.contains(&format!("void {k}(")), "noyau {k} absent de la source assemblée ({})", b.nom());
        }
        assert!(!s.texte.contains("#define K1_MUTANT_METRO"));
        assert_eq!(s.sha256, sha256_hex(s.texte.as_bytes()));
        // déterminisme
        assert_eq!(source_ref64(b, false).sha256, s.sha256);
        // le mutant change la source (donc l'empreinte)
        let m = source_ref64(b, true);
        assert!(m.texte.contains("#define K1_MUTANT_METRO 1"));
        assert_ne!(m.sha256, s.sha256);
    }
}
