//! Tests des exemples (crates/kvu/exemples/*.sh) : chaque exemple s'affiche en mode `dry` (rien n'est lancé) et
//! s'exécute en mode `petit` (L ≤ 8, backend cpu, répertoire jetable) avec le binaire `kvu` de cette compilation.
//! Les commandes écrites dans le README sont celles de ces scripts.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[path = "donnees/hermetique/support.rs"]
mod support;

fn exemples() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("exemples")
}

/// Store jetable des exemples : `kvu store du --projeter` (exemple 03, mode petit) exige que la racine du store existe,
/// et le store par défaut (`~/kevinotron_store`) n'existe pas sur une machine neuve ou sous un HOME isolé.
fn store_jetable() -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cli_exemples_store");
    std::fs::create_dir_all(&d).expect("store jetable des exemples");
    d
}

fn lister() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(exemples())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "sh") && p.file_name().unwrap().to_string_lossy().starts_with('0')
        })
        .collect();
    v.sort();
    v
}

fn lancer(script: &Path, mode: &str) -> Output {
    // seuil de la garde SSD des tests (KVU_TEST_SSD_MIN_GO) : hérité par le script et par les `kvu` qu'il lance
    support::appliquer_seuil_ssd_des_tests();
    Command::new("bash")
        .arg(script)
        // sorties indépendantes de la locale de l'hôte : les awk/printf des scripts écrivent « 2.15 » sous C.UTF-8 et
        // « 2,15 » sous fr_FR.UTF-8 ; les assertions ci-dessous sont celles de C.UTF-8
        .env("LC_ALL", "C.UTF-8")
        .env("EXEMPLE_MODE", mode)
        .env("KVU_BIN", env!("CARGO_BIN_EXE_kvu"))
        // jamais le verrou GPU de production dans un test
        .env("VERROU", "")
        .env("EXEMPLE_SORTIE_SANS_EFFET", "1")
        .env("KEVINOTRON_STORE", store_jetable())
        .output()
        .expect("lancement de l'exemple")
}

#[test]
fn les_quatre_exemples_existent() {
    let noms: Vec<String> = lister()
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(noms.len(), 4, "{noms:?}");
    for attendu in ["r2_3_s2", "saturation_v3", "desert", "balayage_beta"] {
        assert!(
            noms.iter().any(|n| n.contains(attendu)),
            "exemple {attendu} absent : {noms:?}"
        );
    }
}

#[test]
fn mode_dry_affiche_sans_rien_lancer() {
    for s in lister() {
        let o = lancer(&s, "dry");
        let sortie = String::from_utf8_lossy(&o.stdout);
        assert_eq!(
            o.status.code(),
            Some(0),
            "{} : {}",
            s.display(),
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(
            sortie.contains("+ ") && sortie.contains("su2"),
            "{} n'affiche aucune commande : {sortie}",
            s.display()
        );
        println!("PASS exemple_dry {}", s.file_name().unwrap().to_string_lossy());
    }
}

#[test]
fn le_desert_projette_la_place_et_les_durees() {
    let o = lancer(&exemples().join("03_desert_L48_64.sh"), "dry");
    let s = String::from_utf8_lossy(&o.stdout);
    assert!(s.contains("q64 = 2.15 Go par configuration"), "{s}");
    assert!(
        s.contains("53 min (acemagic), 22 min (gamer)"),
        "durées estimées de §2 : {s}"
    );
    assert!(s.contains("--precision f32 --format q64 --mesures"), "{s}");
}

#[test]
fn la_cellule_s2_reprend_les_graines_de_prod_sh() {
    let o = lancer(&exemples().join("01_r2_3_s2_repliques.sh"), "dry");
    let s = String::from_utf8_lossy(&o.stdout);
    // b = 2 (β = 3,0), L = 16, r = 1 : 2809280100000 + 2000 + 160 + 1
    assert!(s.contains("--graine 2809280102161 --graine-pas 1"), "{s}");
    assert!(s.contains("--bc hot --bc-graine 2809280202161"), "{s}");
    assert!(
        s.contains("--bc eq --bc-graine 2809280202162 --jauge 2809280302160"),
        "{s}"
    );
    assert!(
        s.contains("--pre 2000 --therm 4000 --nmeas 50000 --bloc 100 --nor 4"),
        "{s}"
    );
}

/// Exécution réelle à petite échelle (CPU, L ≤ 8) : su2 gen, cond chaine et store.
#[test]
fn mode_petit_execute_vraiment() {
    for s in lister() {
        let o = lancer(&s, "petit");
        assert_eq!(
            o.status.code(),
            Some(0),
            "{} :\n{}\n{}",
            s.display(),
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        );
        println!("PASS exemple_petit {}", s.file_name().unwrap().to_string_lossy());
    }
}
