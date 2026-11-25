//! Dette 3 de la fusion n°3 : `--secteur-flux` ne doit rien changer aux sorties existantes (`cmp`, au bit).
//! Test de carte (`#[ignore = "GPU"]`) : à lancer sous le verrou de la carte (`carte-radeon -- cargo test … -- --ignored flux_cmp`
//! ou `carte-gamer -- …` avec `--features cuda`). Une même commande `su2 spectro gpu-torelon` tourne trois fois avec le MÊME binaire :
//! sans l'option, avec `--secteur-flux 1,1,0 --flux-sans-corr`, avec `--secteur-flux 1,1,0` ; les fichiers `corr.csv`, `phi.csv`,
//! `plaq.csv`, `ops.bin` de chaque réplique doivent être identiques octet pour octet, et `ops_flux.bin` identique entre les deux
//! lancements avec l'option. Contrôle négatif fait à la fusion n°4 (Radeon) : si le code de mesure du flux écrit dans le champ vivant
//! (mutant de `mesurer_flux` : un lissage APE écrit dans `champ0`), ce test échoue sur `corr.csv`. Limite connue : un mutant de
//! `k_flux` qui écrit dans son champ d'entrée (`U[0].x = 0.5`) SURVIT, car la chaîne ne relit pas ce tampon (seul `ops_flux.bin` change) ;
//! les erreurs de valeur du noyau sont du ressort des tests `flux_carte_*` (carte contre processeur, dans `flux.rs`).
use std::path::Path;
use std::process::Command;

fn lancer(extra: &[&str], sortie: &Path) {
    let b = if cfg!(feature = "cuda") { "cuda" } else { "hip" };
    let mut a: Vec<String> = [
        "su2", "spectro", "gpu-torelon", "--backend", b, "--cotes", "8,8,8,16", "--beta", "2.4", "--repliques", "2", "--therm", "40",
        "--nmeas", "60", "--skip", "3", "--niveaux", "0,2,4", "--blocage", "1", "--ape-bloc", "1", "--lissage-bloc", "1", "--enroule", "0",
        "--tau-max", "6", "--precision", "f64", "--ops-bruts", "--graine", "20261122",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    a.extend(extra.iter().map(|s| s.to_string()));
    a.push("--sortie".into());
    a.push(sortie.display().to_string());
    let o = Command::new(env!("CARGO_BIN_EXE_kvu"))
        .args(&a)
        .env("KEVINOTRON_VERROU", "ignorer")
        .env("KEVINOTRON_SSD_MIN_GO", "0")
        .output()
        .expect("lancement de kvu");
    assert!(
        o.status.success(),
        "kvu {:?} : rc {:?}\n{}",
        extra,
        o.status.code(),
        String::from_utf8_lossy(&o.stderr)
    );
}

fn lire(d: &Path, r: &str, f: &str) -> Vec<u8> {
    std::fs::read(d.join(r).join(f)).unwrap_or_else(|e| panic!("lecture de {}/{r}/{f} : {e}", d.display()))
}

#[test]
#[ignore = "GPU"]
fn flux_cmp_option_sans_effet_sur_les_sorties_existantes() {
    let base = std::env::temp_dir().join(format!("kvu_flux_cmp_{}", std::process::id()));
    let (sans, bref, avec) = (base.join("sans"), base.join("avec_sans_corr"), base.join("avec"));
    lancer(&[], &sans);
    lancer(&["--secteur-flux", "1,1,0", "--flux-sans-corr"], &bref);
    lancer(&["--secteur-flux", "1,1,0"], &avec);
    for r in ["r0", "r1"] {
        for f in ["corr.csv", "phi.csv", "plaq.csv", "ops.bin"] {
            let x = lire(&sans, r, f);
            assert!(!x.is_empty(), "{r}/{f} vide : le test ne prouverait rien");
            assert!(x == lire(&bref, r, f), "{r}/{f} : sans l'option != avec --secteur-flux --flux-sans-corr (cmp)");
            assert!(x == lire(&avec, r, f), "{r}/{f} : sans l'option != avec --secteur-flux (cmp)");
        }
        let o1 = lire(&bref, r, "ops_flux.bin");
        assert!(!o1.is_empty(), "{r}/ops_flux.bin vide");
        assert!(o1 == lire(&avec, r, "ops_flux.bin"), "{r}/ops_flux.bin diffère entre les deux modes de l'option");
        assert!(avec.join(r).join("corr_flux.csv").is_file(), "{r}/corr_flux.csv absent");
        assert!(!bref.join(r).join("corr_flux.csv").exists(), "{r}/corr_flux.csv écrit malgré --flux-sans-corr");
    }
    let _ = std::fs::remove_dir_all(&base);
}
