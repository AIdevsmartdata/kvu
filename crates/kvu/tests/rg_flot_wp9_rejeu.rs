//! WP9-COURBE-REJEU (PREREG WP9 §6 R1) : rejoue sur CETTE machine trois chaînes d'une courbe d'étalonnage wp9 calculée ailleurs
//! (β = 0,25, 4 et 64, chaîne c = 1) et compare l'empreinte de leurs séries (sha256 des bits de P puis de R) à celle du JSON.
//! Test ignoré par défaut (calcul CPU de plusieurs minutes à L = 8) ; lancement explicite :
//! `KVU_REJEU_COURBE=<etal_L*_wp9.json> cargo test --release --test rg_flot_wp9_rejeu -- --ignored --nocapture`
//! Une ligne `PASS|FAIL WP9-COURBE-REJEU-L<L>-b<β> …` par chaîne est imprimée ; le test échoue si une empreinte diffère.
use kvu::heritage::manifest::{hex, Sha256};
use kvu::rg::flot::etalonner::{depuis_json, lancer_chaine};

fn empreinte(p: &[f64], r: &[f64]) -> String {
    let mut h = Sha256::new();
    for x in p.iter().chain(r) {
        h.update(&x.to_bits().to_le_bytes());
    }
    hex(&h.finalize())
}

#[test]
#[ignore = "rejeu WP9 explicite (CPU, plusieurs minutes)"]
fn rejeu_courbe_wp9() {
    let f = std::env::var("KVU_REJEU_COURBE").expect("KVU_REJEU_COURBE (chemin d'une courbe wp9) requis");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&f).expect("lecture de la courbe")).expect("JSON");
    let (par, chaines) = depuis_json(&v).expect("étalonnage");
    assert!(par.est_wp9() && !par.independante && par.facteur_beta == 1.0 && !par.est_milieux(), "courbe wp9 de production attendue");
    let cibles = [0.25_f64, 4.0, 64.0];
    let travaux: Vec<(usize, f64, String)> = cibles
        .iter()
        .map(|&b| {
            let i = par.betas.iter().position(|&x| x == b).expect("β de la grille wp9");
            let ch = chaines.iter().find(|c| c.indice_beta == i && c.c == 1).expect("chaîne c = 1");
            assert_eq!(ch.graine, par.graine(i, 1));
            (i, b, ch.empreinte.clone())
        })
        .collect();
    let resultats: Vec<(f64, String, String)> = std::thread::scope(|s| {
        let poignees: Vec<_> = travaux
            .iter()
            .map(|(i, b, attendue)| {
                let par = &par;
                s.spawn(move || {
                    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
                    let (p, r) = pool.install(|| lancer_chaine(par.l, *b, par.graine(*i, 1), par.therm, par.mesures).unwrap());
                    (*b, empreinte(&p, &r), attendue.clone())
                })
            })
            .collect();
        poignees.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut ok = true;
    for (b, obtenue, attendue) in &resultats {
        let egal = obtenue == attendue;
        ok &= egal;
        println!(
            "{} WP9-COURBE-REJEU-L{}-b{} empreinte rejouée {}… {} celle du JSON {}…",
            if egal { "PASS" } else { "FAIL" },
            par.l,
            b,
            &obtenue[..16],
            if egal { "=" } else { "≠" },
            &attendue[..16]
        );
    }
    assert!(ok, "au moins une empreinte diffère");
}
