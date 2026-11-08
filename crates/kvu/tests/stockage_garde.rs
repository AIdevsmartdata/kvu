//! Gate SSD-1 (ARCHITECTURE_UNIFIEE.md §6.5) : la garde SSD refuse AVANT d'écrire le moindre octet (rc 7).
//!
//! - libre_go = df ; remontée au premier parent existant ;
//! - seuil forcé au-dessus du libre ⇒ refus de CHAQUE écriture (champ, tableaux, manifeste) sans rien créer ;
//! - projection d'un run c16 L = 64 × 20 dumps refusée sur acemagic (≈ 86 Go pour ≈ 159 Go libres, seuil 100) ;
//! - KEVINOTRON_SSD_MIN_GO surcharge le seuil de la machine.
//!
//! Chaque test imprime `PASS <nom>`.

#[path = "donnees/stockage/support.rs"]
mod support;

use kvu::commun::{code, Erreur};
use kvu::stockage::garde::{projeter, seuil_defaut, GardeSsd};
use kvu::stockage::{npy, FormatChamp, Manifeste};
use std::fs;
use std::path::Path;
use std::process::Command;
use support::*;

fn df_octets(chemin: &Path) -> f64 {
    let s = Command::new("df")
        .args(["--output=avail", "-B1"])
        .arg(chemin)
        .output()
        .expect("df");
    String::from_utf8_lossy(&s.stdout)
        .lines()
        .nth(1)
        .unwrap()
        .trim()
        .parse::<f64>()
        .unwrap()
}

#[test]
fn libre_go_concorde_avec_df() {
    let sc = Scratch::nouveau("garde_df");
    let libre = GardeSsd::libre_go(&sc.0).unwrap();
    let df = df_octets(&sc.0) / (1u64 << 30) as f64; // 1 Go = 1 Gio (df -BG)
    assert!((libre - df).abs() < 2.0, "libre_go {libre} contre df {df}");
    // un chemin inexistant remonte au premier parent existant : même disque
    let profond = GardeSsd::libre_go(&sc.p("a/b/c/d")).unwrap();
    assert!((profond - libre).abs() < 2.0);
    println!("PASS libre_go_concorde_avec_df (libre = {libre:.1} Go, df = {df:.1} Go)");
}

#[test]
fn seuils_par_machine_du_tableau() {
    use kvu::commun::Machine;
    assert_eq!(seuil_defaut(Machine::Acemagic), 100.0);
    assert_eq!(seuil_defaut(Machine::Gamer), 50.0);
    assert_eq!(seuil_defaut(Machine::Inconnue), 20.0);
    println!("PASS seuils_par_machine_du_tableau");
}

/// SSD-1 (a) : seuil forcé au-dessus du libre ⇒ refus avant tout octet, rc 7, rien créé.
#[test]
fn ssd1_refus_avant_tout_octet() {
    let sc = Scratch::nouveau("ssd1");
    let libre = GardeSsd::libre_go(&sc.0).unwrap();
    let g = GardeSsd::nouvelle(libre + 1.0);

    let err = g.verifier(&sc.0, 1).unwrap_err();
    assert!(matches!(err, Erreur::SsdPlein { .. }));
    assert_eq!(err.code_sortie(), code::SSD);
    assert_eq!(err.code_sortie(), 7);

    // champ c16 et q64
    let champ = champ_hurwitz(2);
    for (nom, f) in [("a.npy", FormatChamp::C16), ("b.npy", FormatChamp::Q64)] {
        let e = npy::ecrire_champ(&sc.p(nom), &champ, f, &g).unwrap_err();
        assert_eq!(e.code_sortie(), 7, "{e}");
    }
    // tableaux
    assert_eq!(
        npy::ecrire_f64(&sc.p("c.npy"), &[2], &[1.0, 2.0], &g)
            .unwrap_err()
            .code_sortie(),
        7
    );
    assert_eq!(
        npy::ecrire_f32(&sc.p("d.npy"), &[1], &[1.0], &g)
            .unwrap_err()
            .code_sortie(),
        7
    );
    assert_eq!(
        npy::ecrire_u32(&sc.p("e.npy"), &[1], &[1], &g)
            .unwrap_err()
            .code_sortie(),
        7
    );
    // conversion
    let ok = GardeSsd::nouvelle(0.0);
    npy::ecrire_champ(&sc.p("src.npy"), &champ, FormatChamp::C16, &ok).unwrap();
    assert_eq!(
        npy::convertir_champ(&sc.p("src.npy"), &sc.p("dst.npy"), FormatChamp::Q64, &g, true)
            .unwrap_err()
            .code_sortie(),
        7
    );
    // manifeste : le répertoire de run n'est même pas créé
    let e = Manifeste::ouvrir_avec_garde(&sc.p("run"), "test", &["x".into()], &g, 0)
        .err()
        .expect("refus attendu");
    assert_eq!(e.code_sortie(), 7);
    assert!(
        !sc.p("run").exists(),
        "le répertoire de run ne doit pas exister après un refus"
    );

    // rien n'a été écrit : seul src.npy (écrit avec la garde permissive) est là, aucun .part
    let mut noms: Vec<String> = fs::read_dir(&sc.0)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    noms.sort();
    assert_eq!(noms, ["src.npy"], "écritures refusées mais fichiers créés : {noms:?}");
    println!("PASS SSD-1 refus avant tout octet, rc 7");
}

/// SSD-1 (b) : projection c16 L = 64 × 20 dumps refusée ; la même en q64 est deux fois plus légère.
#[test]
fn ssd1_projection_c16_l64_20_dumps_refusee() {
    let sc = Scratch::nouveau("ssd1_proj");
    let p = projeter(64, 1, 20, FormatChamp::C16, 0);
    assert!((p as f64 / 1e9 - 85.9).abs() < 0.2, "projection {p}");

    // Déterministe, sur la machine de test : un seuil qui laisse 10 Go de moins que le libre ne peut pas accueillir 86 Go.
    let libre = GardeSsd::libre_go(&sc.0).unwrap();
    let g = GardeSsd::nouvelle(libre - 10.0);
    let e = g.verifier(&sc.0, p).unwrap_err();
    assert_eq!(e.code_sortie(), 7);
    // Et le manifeste de ce run est refusé à l'ouverture avec la projection : rien n'est créé.
    let e = Manifeste::ouvrir_avec_garde(&sc.p("run_l64"), "su2", &[], &g, p)
        .err()
        .expect("refus attendu");
    assert_eq!(e.code_sortie(), 7);
    assert!(!sc.p("run_l64").exists());

    // Pur, comme sur acemagic (159 Go libres, seuil 100) : refusé en c16, il resterait 100,x Go... en q64 : accepté
    let acemagic = GardeSsd::pour(kvu::commun::Machine::Acemagic);
    assert!(
        acemagic.evaluer(Path::new("/"), 159.0, p).is_err(),
        "c16 L=64 x 20 doit être refusé à 159 Go libres"
    );
    let q = projeter(64, 1, 20, FormatChamp::Q64, 0);
    assert!(
        acemagic.evaluer(Path::new("/"), 159.0, q).is_ok(),
        "q64 x 20 (43 Go) tient dans la marge de 59 Go"
    );
    println!(
        "PASS SSD-1 projection c16 L=64 x 20 dumps refusée ({:.1} Go)",
        p as f64 / 1e9
    );
}

/// Seul test de ce fichier qui touche à l'environnement.
#[test]
fn surcharge_par_environnement() {
    let sc = Scratch::nouveau("ssd_env");
    std::env::set_var("KEVINOTRON_SSD_MIN_GO", "123.5");
    assert_eq!(GardeSsd::machine().seuil_go, 123.5);
    std::env::set_var("KEVINOTRON_SSD_MIN_GO", "pas-un-nombre");
    assert!(GardeSsd::machine().seuil_go > 0.0, "une valeur illisible est ignorée");

    // seuil énorme : Manifeste::ouvrir (garde de la machine) refuse, rien n'est créé, rc 7
    std::env::set_var("KEVINOTRON_SSD_MIN_GO", "1e9");
    let e = Manifeste::ouvrir(&sc.p("run"), "test", &[])
        .err()
        .expect("refus attendu");
    assert_eq!(e.code_sortie(), 7);
    assert!(!sc.p("run").exists());
    std::env::remove_var("KEVINOTRON_SSD_MIN_GO");
    println!("PASS surcharge_par_environnement");
}
