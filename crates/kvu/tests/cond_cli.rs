//! Pont en ligne de commande pour les scripts de validation (`validation/cond/*.sh|py`) tant que le binaire
//! unifié (T9) n'est pas câblé : exécute `kevinotron cond …` dans CE binaire de test.
//!
//! Usage (le binaire de test est trouvé par `gates_cond.sh`) :
//!   KVU_COND_ARGV=$'chaine\x1f--cfg\x1fX.npy\x1f…' <binaire_de_test> --ignored --exact pont_cli --nocapture
//! Les arguments sont séparés par le caractère 0x1f ; le code de sortie est celui de la commande.
//! Sans `KVU_COND_ARGV`, le test ne fait rien (il vérifie seulement l'analyse des arguments).
use kvu::cond::cli::{commandes, executer};

fn separer(argv: &str) -> Vec<String> {
    argv.split('\x1f').map(str::to_string).collect()
}

#[test]
#[ignore = "pont CLI : lancé par les scripts de validation"]
fn pont_cli() {
    let Ok(argv) = std::env::var("KVU_COND_ARGV") else {
        println!("SKIP pont_cli : KVU_COND_ARGV absent");
        return;
    };
    // libtest (--test-threads=1) a déjà écrit « test pont_cli ... » sans fin de ligne : on la termine pour que la
    // sortie JSON de la commande commence en début de ligne.
    println!();
    let mut v = vec!["cond".to_string()];
    v.extend(separer(&argv));
    let m = match commandes().remove(0).try_get_matches_from(v) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(kvu::commun::code::USAGE);
        }
    };
    match executer("cond", &m) {
        Ok(rc) => std::process::exit(rc),
        Err(e) => {
            eprintln!("erreur : {e}");
            std::process::exit(e.code_sortie());
        }
    }
}

#[test]
fn separateur_d_arguments() {
    assert_eq!(separer("chaine\x1f--beta\x1f2.3"), vec!["chaine", "--beta", "2.3"]);
}

#[test]
fn le_pont_analyse_une_ligne_de_commande_valide() {
    let v =
        vec!["cond", "sweep", "--cfg", "a.npy", "--beta", "2.3", "--graine", "0", "--out", "b.npy", "--backend", "cpu"];
    assert!(commandes().remove(0).try_get_matches_from(v).is_ok());
}
