//! G-MUT, volet algèbre et Haar (WP5) : le générateur retiré rend G-ALG-1 ROUGE, le QR naïf (U(2)) rend G-HAAR-2 et G-HAAR-3 ROUGES,
//! et la fiche saine reste verte aux mêmes graines (un mutant rouge n'a de valeur que si l'original est vert).

use kvu::groupes::gates::{g_alg, g_haar, g_mut_algebre_haar, tout_vert, Verdict};
use kvu::groupes::su2::{MutantSu2, Su2};

#[test]
fn originale_verte_mutants_rouges() {
    assert!(tout_vert(&g_alg(&Su2::nouveau())));
    assert!(tout_vert(&g_haar(&Su2::nouveau(), 300_000)));
    let l = g_mut_algebre_haar(300_000);
    assert_eq!(l.len(), 3);
    for x in &l {
        assert_eq!(x.verdict, Verdict::Pass, "{x}");
    }
}

#[test]
fn le_qr_naif_est_unitaire_mais_pas_special() {
    let l = g_haar(&Su2::avec_mutant(MutantSu2::QrNaif), 300_000);
    assert_eq!(l[1].verdict, Verdict::Fail, "G-HAAR-2 doit être rouge : {}", l[1]);
    assert_eq!(l[2].verdict, Verdict::Fail, "G-HAAR-3 doit être rouge : {}", l[2]);
    assert!(l[2].detail.contains("max|U†U−1|"), "{}", l[2]);
}

#[test]
fn le_generateur_retire_ne_ferme_plus() {
    let l = g_alg(&Su2::avec_mutant(MutantSu2::GenerateurRetire));
    assert_eq!(l[0].verdict, Verdict::Fail, "{}", l[0]);
}
