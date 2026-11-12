//! Reprise des mesures en ligne de `su2 gen` après un arrêt brutal (c08c, 11/11/2026).
//! Suite autonome, applicable à la version publique 0.1.1 : Wilson, Polyakov, flot et bloc.
//! Tous les tests sont sur CPU et appartiennent à la suite normale.

use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const SPEC_WILSON: &str = "wilson:R2xT2:1";
const NOM_FICHIER_MESURES: &str = "mesures_en_ligne.json";

fn kvu(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kvu"))
        .args(args)
        .env("RAYON_NUM_THREADS", "2")
        .env("KEVINOTRON_BACKEND", backend_gpu())
        .output()
        .expect("lancement de kvu")
}

fn dossier(nom: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("su2_reprise_mesures_{nom}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("création du répertoire temporaire");
    dir
}

fn gen(
    backend: &str,
    l: usize,
    therm: u64,
    nmeas: u64,
    skip: u64,
    sortie: &Path,
    extra: &[&str],
) -> Output {
    let l = l.to_string();
    let therm = therm.to_string();
    let nmeas = nmeas.to_string();
    let skip = skip.to_string();
    let mut args = vec![
        "su2",
        "gen",
        "--backend",
        backend,
        "--L",
        &l,
        "--beta",
        "2.3",
        "--graine",
        "26111108",
        "--therm",
        &therm,
        "--nmeas",
        &nmeas,
        "--skip",
        &skip,
        "--dump-every",
        "0",
        "--format",
        "q64",
        "--outdir",
        sortie.to_str().expect("chemin UTF-8 de sortie"),
    ];
    args.extend_from_slice(extra);
    kvu(&args)
}

fn reprendre(backend: &str, dir: &Path, extra: &[&str]) -> Output {
    let mut args = vec![
        "su2",
        "gen",
        "--backend",
        backend,
        "--reprendre",
        dir.to_str().expect("chemin UTF-8 de reprise"),
    ];
    args.extend_from_slice(extra);
    kvu(&args)
}

fn reussi(sortie: &Output) {
    assert_eq!(
        sortie.status.code(),
        Some(0),
        "kvu en échec :\n{}\n{}",
        String::from_utf8_lossy(&sortie.stdout),
        String::from_utf8_lossy(&sortie.stderr),
    );
}

fn refus(sortie: &Output, motif: &str) {
    let erreur = String::from_utf8_lossy(&sortie.stderr);
    assert_eq!(
        sortie.status.code(),
        Some(2),
        "refus code 2 attendu : {erreur}"
    );
    assert!(
        erreur.contains(motif),
        "motif « {motif} » absent : {erreur}"
    );
}

fn octets(p: &Path) -> Vec<u8> {
    fs::read(p).unwrap_or_else(|e| panic!("lecture de {} : {e}", p.display()))
}

fn lire_json(p: &Path) -> Value {
    serde_json::from_slice(&octets(p)).unwrap_or_else(|e| panic!("JSON de {} : {e}", p.display()))
}

fn backend_gpu() -> &'static str {
    match std::env::var("KEVINOTRON_BACKEND").as_deref() {
        Ok("hip") => "hip",
        Ok("cuda") => "cuda",
        _ if cfg!(feature = "cuda") => "cuda",
        _ => "hip",
    }
}

fn interrompre(backend: &str, l: usize, dir: &Path, spec: &str) {
    reussi(&gen(
        backend,
        l,
        6,
        6,
        2,
        dir,
        &[
            "--reprise-every",
            "2",
            "--arret-apres",
            "10",
            "--mesures",
            spec,
        ],
    ));
    let etat = lire_json(&dir.join("reprise.json"));
    assert_eq!(
        etat["pas_faits"].as_u64(),
        Some(10),
        "point de reprise au pas 10"
    );
    assert_eq!(etat["mesures_faites"].as_u64(), Some(3)); // indice de la PROCHAINE mesure (m + 1, su2::run::point_si_du)
    assert!(!dir.join("COMPLETE").exists(), "COMPLETE avant la fin");
}

fn paire_reprise(d: &Path, backend: &str, l: usize, spec: &str) -> (PathBuf, PathBuf) {
    let a = d.join("a/chaine");
    let b = d.join("b/chaine");
    reussi(&gen(
        backend,
        l,
        6,
        6,
        2,
        &a,
        &["--reprise-every", "2", "--mesures", spec],
    ));
    assert!(a.join("COMPLETE").exists(), "COMPLETE du témoin absent");
    interrompre(backend, l, &b, spec);
    (a, b)
}

fn etat_csv(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut noms: Vec<String> = fs::read_dir(dir)
        .expect("lecture du répertoire de run")
        .map(|e| {
            e.expect("entrée du répertoire")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|n| n.ends_with(".csv"))
        .collect();
    noms.sort();
    noms.into_iter()
        .map(|nom| {
            let contenu = octets(&dir.join(&nom));
            (nom, contenu)
        })
        .collect()
}

fn fichiers_bloc(dir: &Path) -> Vec<String> {
    let mut noms: Vec<String> = fs::read_dir(dir)
        .expect("lecture du répertoire des configurations grossières")
        .map(|e| {
            e.expect("entrée du répertoire")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|n| n.starts_with("bloc3_cfg_") && n.ends_with(".npy"))
        .collect();
    noms.sort();
    noms
}

fn exiger_compagnon_consigne(dir: &Path) {
    assert!(
        dir.join(NOM_FICHIER_MESURES).exists(),
        "compagnon absent avant altération"
    );
    let man = lire_json(&dir.join("manifest.json"));
    let fichiers = man["fichiers"].as_array().expect("fichiers du manifeste");
    let entrees: Vec<&Value> = fichiers
        .iter()
        .filter(|f| f["nom"].as_str() == Some(NOM_FICHIER_MESURES))
        .collect();
    assert_eq!(entrees.len(), 1, "une seule entrée du compagnon attendue");
    let sha = entrees[0]["sha256"]
        .as_str()
        .expect("empreinte du compagnon");
    assert_eq!(sha.len(), 64, "longueur de l'empreinte du compagnon");
    assert!(
        sha.bytes().all(|b| b.is_ascii_hexdigit()),
        "empreinte non hexadécimale"
    );
}

fn arret_brutal(spec: &str, fichiers: &[&str], l: usize, porte: &str) {
    let d = dossier(porte);
    let (a, b) = paire_reprise(&d, "cpu", l, spec);
    let mut lignes_ajoutees = 0;
    for nom in fichiers {
        let texte = String::from_utf8(octets(&a.join(nom))).expect("CSV témoin UTF-8");
        let lignes: String = texte
            .split_inclusive('\n')
            .filter(|ligne| ligne.split(',').next() == Some("12"))
            .collect();
        assert!(!lignes.is_empty(), "aucune ligne au pas 12 dans {nom}");
        assert!(lignes.ends_with('\n'), "ligne témoin partielle dans {nom}");
        lignes_ajoutees += lignes.lines().count();
        let chemin = b.join(nom);
        assert!(
            octets(&chemin).ends_with(b"\n"),
            "CSV interrompu partiel : {nom}"
        );
        let mut f = fs::OpenOptions::new()
            .append(true)
            .open(&chemin)
            .expect("ouverture en ajout du CSV interrompu");
        f.write_all(lignes.as_bytes())
            .expect("injection des lignes du pas 12");
        f.write_all(b"12,0")
            .expect("injection du fragment sans fin de ligne");
        f.sync_all()
            .expect("synchronisation de l'arrêt brutal simulé");
    }
    let bloc = fichiers.contains(&"bloc3.csv");
    if bloc {
        let nom = "bloc3_cfg_00000012.npy";
        assert!(
            !b.join(nom).exists(),
            "configuration postérieure déjà présente"
        );
        fs::copy(a.join(nom), b.join(nom)).expect("copie de la configuration postérieure");
    }
    // La spécification est rechargée : ne pas repasser --mesures.
    reussi(&reprendre("cpu", &b, &[]));
    for nom in fichiers
        .iter()
        .copied()
        .chain(["measurements.csv", "therm.csv"])
    {
        assert_eq!(
            octets(&a.join(nom)),
            octets(&b.join(nom)),
            "{nom} diffère après reprise"
        );
    }
    let configs = fichiers_bloc(&a);
    assert_eq!(
        configs,
        fichiers_bloc(&b),
        "listes des configurations grossières différentes"
    );
    assert_eq!(configs.len(), if bloc { 6 } else { 0 });
    for nom in &configs {
        assert_eq!(
            octets(&a.join(nom)),
            octets(&b.join(nom)),
            "{nom} diffère après reprise"
        );
    }
    assert!(b.join("COMPLETE").exists(), "COMPLETE absent après reprise");
    let compares = fichiers.len() + configs.len() + 2;
    println!(
        "\nPASS {porte} : L={l}, {lignes_ajoutees} lignes et {} fragments injectés, \
         {compares} fichiers identiques, 0 octet différent",
        fichiers.len()
    );
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn arret_brutal_wilson() {
    arret_brutal(SPEC_WILSON, &["wilson.csv"], 4, "arret_brutal_wilson");
}

#[test]
fn arret_brutal_polyakov_corr() {
    arret_brutal(
        "polyakov_corr:1",
        &["polyakov_corr.csv"],
        4,
        "arret_brutal_polyakov_corr",
    );
}

#[test]
fn arret_brutal_flot() {
    arret_brutal(
        "flot:0.02:0.06:1",
        &["flot.csv", "flot_t0.csv"],
        4,
        "arret_brutal_flot",
    );
}

#[test]
fn arret_brutal_bloc() {
    arret_brutal(
        "bloc:3:1",
        &["bloc3.csv", "bloc3_hist.csv"],
        6,
        "arret_brutal_bloc",
    );
}

#[test]
fn calendrier_troue_refuse_avant_balayage() {
    let d = dossier("calendrier_troue");
    let b = d.join("chaine");
    interrompre("cpu", 4, &b, SPEC_WILSON);
    let chemin = b.join("wilson.csv");
    let texte = String::from_utf8(octets(&chemin)).expect("CSV Wilson UTF-8");
    let conserve: String = texte
        .split_inclusive('\n')
        .filter(|ligne| ligne.split(',').next() != Some("8"))
        .collect();
    let retirees = texte.lines().count() - conserve.lines().count();
    assert_eq!(retirees, 4, "quatre boucles Wilson attendues au pas 8");
    fs::write(&chemin, conserve).expect("suppression des lignes du pas 8");
    let avant = etat_csv(&b);
    let o = reprendre("cpu", &b, &[]);
    assert_ne!(o.status.code(), Some(0), "calendrier troué accepté");
    let erreur = String::from_utf8_lossy(&o.stderr);
    assert!(erreur.contains("calendrier"), "{erreur}");
    assert_eq!(
        etat_csv(&b),
        avant,
        "CSV modifiés avant le refus du calendrier"
    );
    assert!(
        !b.join("COMPLETE").exists(),
        "COMPLETE malgré le calendrier troué"
    );
    println!(
        "\nPASS CALENDRIER-TROUE : {retirees} lignes retirées, 1 refus, 3 CSV inchangés, 0 COMPLETE"
    );
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn compagnon_modifie_refuse() {
    let d = dossier("compagnon_modifie");
    let b = d.join("chaine");
    interrompre("cpu", 4, &b, SPEC_WILSON);
    exiger_compagnon_consigne(&b);
    let chemin = b.join(NOM_FICHIER_MESURES);
    let texte = String::from_utf8(octets(&chemin)).expect("compagnon UTF-8");
    assert_eq!(
        texte.matches(":1\"").count(),
        1,
        "période initiale unique attendue"
    );
    fs::write(&chemin, texte.replace(":1\"", ":2\""))
        .expect("modification de la période dans le compagnon");
    assert_eq!(
        lire_json(&chemin)["spec"],
        "wilson:R2xT2:2",
        "JSON modifié invalide"
    );
    let avant = etat_csv(&b);
    refus(&reprendre("cpu", &b, &[]), "modifié");
    assert_eq!(
        etat_csv(&b),
        avant,
        "CSV modifiés malgré le compagnon altéré"
    );
    assert!(
        !b.join("COMPLETE").exists(),
        "COMPLETE malgré le compagnon altéré"
    );
    println!("\nPASS COMPAGNON-MODIFIE : 1 période modifiée, 1 refus code 2, 3 CSV inchangés");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn compagnon_disparu_refuse() {
    let d = dossier("compagnon_disparu");
    let b = d.join("chaine");
    interrompre("cpu", 4, &b, SPEC_WILSON);
    exiger_compagnon_consigne(&b);
    fs::remove_file(b.join(NOM_FICHIER_MESURES)).expect("suppression du compagnon consigné");
    let avant = etat_csv(&b);
    let demandes: [&[&str]; 2] = [&[], &["--mesures", SPEC_WILSON]];
    for extra in demandes {
        refus(&reprendre("cpu", &b, extra), "disparu");
        assert_eq!(
            etat_csv(&b),
            avant,
            "CSV modifiés malgré le compagnon disparu"
        );
        assert!(
            !b.join("COMPLETE").exists(),
            "COMPLETE malgré le compagnon disparu"
        );
    }
    println!("\nPASS COMPAGNON-DISPARU : 2 refus code 2, 3 CSV inchangés, 0 COMPLETE");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn specification_equivalente_acceptee() {
    let d = dossier("specification_equivalente");
    let (a, b) = paire_reprise(&d, "cpu", 4, SPEC_WILSON);
    reussi(&reprendre("cpu", &b, &["--mesures", " wilson:R2xT2:01 "]));
    assert_eq!(octets(&a.join("wilson.csv")), octets(&b.join("wilson.csv")));
    assert!(
        b.join("COMPLETE").exists(),
        "COMPLETE après reprise équivalente absent"
    );
    println!("\nPASS SPECIFICATION-EQUIVALENTE : 1 reprise code 0, 1 CSV identique à l'octet");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

// ---------------------------------------------------------------------------------------------
// 0.1.1 : garde de reprise (PUB-01) et refus --bord (PUB-02), en version wilson
// ---------------------------------------------------------------------------------------------

const SPEC_W: &str = "wilson:R2xT2:1";

#[test]
fn pub01_reprise_sans_mesures_recharge() {
    let d = dossier("pub01_recharge");
    let (a, b) = paire_reprise(&d, "cpu", 4, SPEC_W);
    let o = reprendre("cpu", &b, &[]);
    reussi(&o);
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("rechargée"),
        "rechargement non signalé"
    );
    for f in ["wilson.csv", "measurements.csv", "therm.csv"] {
        assert_eq!(
            octets(&a.join(f)),
            octets(&b.join(f)),
            "{f} diffère du run d'une traite"
        );
    }
    assert_eq!(
        String::from_utf8(octets(&b.join("wilson.csv")))
            .unwrap()
            .lines()
            .count(),
        1 + 6 * 4
    );
    assert!(b.join("COMPLETE").exists());
    println!("\nPASS PUB01-RECHARGE : 3 fichiers identiques à l'octet, wilson.csv 25 lignes");
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn pub01_specification_differente_refusee() {
    let d = dossier("pub01_differente");
    let b = d.join("chaine");
    interrompre("cpu", 4, &b, SPEC_W);
    let avant = octets(&b.join("wilson.csv"));
    refus(
        &reprendre("cpu", &b, &["--mesures", "wilson:R2xT2:2"]),
        "diffère de la spécification du run",
    );
    refus(
        &reprendre("cpu", &b, &["--mesures", "polyakov_corr:1"]),
        "diffère de la spécification du run",
    );
    assert_eq!(
        avant,
        octets(&b.join("wilson.csv")),
        "wilson.csv modifié par un refus"
    );
    assert!(!b.join("COMPLETE").exists());
    println!("\nPASS PUB01-DIFFERENTE : 2 refus code 2, wilson.csv inchangé");
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn pub01_run_sans_mesures_repris_avec_refuse() {
    let d = dossier("pub01_ajout");
    let b = d.join("chaine");
    reussi(&gen(
        "cpu",
        4,
        6,
        6,
        2,
        &b,
        &["--reprise-every", "2", "--arret-apres", "10"],
    ));
    refus(
        &reprendre("cpu", &b, &["--mesures", SPEC_W]),
        "créé sans --mesures",
    );
    assert!(!b.join("wilson.csv").exists());
    reussi(&reprendre("cpu", &b, &[]));
    println!("\nPASS PUB01-AJOUT : 1 refus code 2, reprise sans mesure rc 0");
    fs::remove_dir_all(&d).unwrap();
}

/// Run de la 0.1.0 simulé : ni compagnon ni entrée au manifeste ; l'argv de création porte `--mesures`.
fn rendre_ancien(dir: &Path) {
    fs::remove_file(dir.join("mesures_en_ligne.json")).expect("suppression du compagnon");
    let p = dir.join("manifest.json");
    let mut m = lire_json(&p);
    if let Some(liste) = m["fichiers"].as_array_mut() {
        liste.retain(|f| f["nom"].as_str() != Some("mesures_en_ligne.json"));
    }
    fs::write(&p, serde_json::to_vec_pretty(&m).unwrap()).unwrap();
}

#[test]
fn pub01_run_ancien_argv() {
    let d = dossier("pub01_ancien");
    let (a, b) = paire_reprise(&d, "cpu", 4, SPEC_W);
    rendre_ancien(&b);
    let o = reprendre("cpu", &b, &[]);
    refus(&o, "--reprendre");
    assert!(
        String::from_utf8_lossy(&o.stderr).contains(SPEC_W),
        "la spécification de création n'est pas citée"
    );
    refus(
        &reprendre("cpu", &b, &["--mesures", "wilson:R2xT2:2"]),
        SPEC_W,
    );
    reussi(&reprendre("cpu", &b, &["--mesures", SPEC_W]));
    assert_eq!(
        octets(&a.join("wilson.csv")),
        octets(&b.join("wilson.csv")),
        "wilson.csv diffère du run d'une traite"
    );
    println!("\nPASS PUB01-ANCIEN : 2 refus code 2 (commande citée), reprise avec la même --mesures = run d'une traite");
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn pub02_bord_et_mesures_refuse() {
    let d = dossier("pub02_bord");
    let x = d.join("bord");
    let o = kvu(&[
        "su2",
        "gen",
        "--backend",
        "cpu",
        "--L",
        "4",
        "--graine",
        "1",
        "--bord",
        "--bc",
        "cold",
        "--pre",
        "10",
        "--therm",
        "10",
        "--nmeas",
        "5",
        "--bloc",
        "5",
        "--mesures",
        SPEC_W,
        "--outdir",
        x.to_str().unwrap(),
    ]);
    refus(&o, "--bord");
    assert!(!x.exists(), "répertoire créé malgré le refus");
    println!("\nPASS PUB02-BORD : refus code 2, aucun répertoire");
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn run_sans_mesures_sans_compagnon() {
    let d = dossier("sans_mesures");
    let x = d.join("chaine");
    reussi(&gen("cpu", 4, 6, 6, 2, &x, &[]));
    assert!(
        !x.join("mesures_en_ligne.json").exists(),
        "compagnon écrit pour un run sans mesure"
    );
    let m = lire_json(&x.join("manifest.json"));
    let noms: Vec<&str> = m["fichiers"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|f| f["nom"].as_str())
        .collect();
    assert!(!noms.contains(&"mesures_en_ligne.json"));
    println!("\nPASS SANS-MESURES : aucun compagnon, manifeste sans entrée");
    fs::remove_dir_all(&d).unwrap();
}

/// Relecture du diff public (P1) : `--mesures ""` (variable vide d'un script) vaut « sans mesures », à la création comme à la
/// reprise : le run se reprend en repassant `--mesures ""` et aussi sans l'option.
#[test]
fn mesures_vide_vaut_sans_mesures() {
    let d = dossier("mesures_vide");
    for (nom, extra) in [("avec_vide", vec!["--mesures", ""]), ("sans_option", vec![])] {
        let b = d.join(nom);
        interrompre("cpu", 4, &b, "");
        assert!(
            !b.join(NOM_FICHIER_MESURES).exists(),
            "compagnon écrit pour une spécification vide"
        );
        reussi(&reprendre("cpu", &b, &extra));
        assert!(b.join("COMPLETE").exists(), "COMPLETE absent ({nom})");
    }
    println!("\nPASS MESURES-VIDE : 2 reprises code 0 (avec --mesures \"\" et sans l'option)");
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

// ---------------------------------------------------------------------------------------------
// 0.1.1 (PUB-13) : format déclaré et préfixe validé AVANT toute troncature (V1, V2)
// ---------------------------------------------------------------------------------------------

/// Lignes de données du pas `pas` d'un CSV (premier champ).
fn lignes_du_pas(texte: &str, pas: u64) -> Vec<String> {
    texte
        .lines()
        .skip(1)
        .filter(|l| l.split(',').next() == Some(pas.to_string().as_str()))
        .map(str::to_string)
        .collect()
}

#[test]
fn v1_format_inconnu_refuse_sans_modifier() {
    let d = dossier("v1_format");
    let b = d.join("chaine");
    interrompre("cpu", 4, &b, "wilson:R2xT2:1");
    // contre-exemple de la troisième relecture : autre format, l'identifiant d'abord
    let mut texte = String::from("id,pas,r,t,W\n");
    for (id, pas, r, t) in [
        (8, 8, 1, 1),
        (10, 8, 1, 2),
        (12, 8, 2, 1),
        (14, 8, 2, 2),
        (16, 10, 1, 1),
        (18, 10, 1, 2),
        (20, 10, 2, 1),
        (22, 10, 2, 2),
    ] {
        texte.push_str(&format!("{id},{pas},{r},{t},0.5\n"));
    }
    fs::write(b.join("wilson.csv"), &texte).unwrap();
    refus(&reprendre("cpu", &b, &[]), "format inconnu");
    assert_eq!(
        octets(&b.join("wilson.csv")),
        texte.as_bytes(),
        "fichier modifié malgré le refus"
    );
    println!("\nPASS V1 : format id,pas,r,t,W refusé (code 2), 0 octet modifié");
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn v2_prefixe_doublonne_refuse_sans_modifier() {
    let d = dossier("v2_doublon");
    let b = d.join("chaine");
    interrompre("cpu", 4, &b, "wilson:R2xT2:1");
    // run 0.1.0 abîmé (PUB-13) puis repris proprement : le groupe du pas 10, AU point de reprise, est doublé
    let chemin = b.join("wilson.csv");
    let mut texte = String::from_utf8(octets(&chemin)).unwrap();
    let groupe = lignes_du_pas(&texte, 10);
    assert_eq!(groupe.len(), 4);
    for l in &groupe {
        texte.push_str(l);
        texte.push('\n');
    }
    fs::write(&chemin, &texte).unwrap();
    refus(&reprendre("cpu", &b, &[]), "pas 10");
    assert_eq!(
        octets(&chemin),
        texte.as_bytes(),
        "fichier modifié malgré le refus"
    );
    println!("\nPASS V2-DOUBLON : 4 lignes du pas 10 doublées, refus code 2 citant le pas 10, 0 octet modifié");
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn v2_ligne_manquante_refusee_sans_modifier() {
    let d = dossier("v2_manquante");
    let b = d.join("chaine");
    interrompre("cpu", 4, &b, "wilson:R2xT2:1");
    let chemin = b.join("wilson.csv");
    let texte = String::from_utf8(octets(&chemin)).unwrap();
    let a_retirer = lignes_du_pas(&texte, 8)[2].clone();
    let ampute: String = texte
        .lines()
        .filter(|l| *l != a_retirer)
        .map(|l| format!("{l}\n"))
        .collect();
    fs::write(&chemin, &ampute).unwrap();
    refus(&reprendre("cpu", &b, &[]), "pas 8");
    assert_eq!(
        octets(&chemin),
        ampute.as_bytes(),
        "fichier modifié malgré le refus"
    );
    println!("\nPASS V2-MANQUANTE : une ligne du pas 8 retirée, refus code 2 citant le pas 8, 0 octet modifié");
    fs::remove_dir_all(&d).unwrap();
}

// ------------------------------------------------------------------------------------------
// PUB-14 (0.1.1) : VRAI arrêt brutal (kill -9) pendant la thermalisation, puis reprise
// ------------------------------------------------------------------------------------------

/// Garde d'un processus fils : sur tout chemin (réussite ou échec du test), le fils encore vivant est tué par SON PID
/// (`Child::kill`, SIGKILL) puis récolté. Le test ne tue jamais rien d'autre que ce fils.
struct GardeFils(Option<std::process::Child>);

impl Drop for GardeFils {
    fn drop(&mut self) {
        if let Some(mut fils) = self.0.take() {
            let _ = fils.kill();
            let _ = fils.wait();
        }
    }
}

const THERM_LONGUE: u64 = 200_000;
const ENTETE_MESURES: &str = "measurement_index,total_steps,phase,plaquette,P0,P1,P2,P3\n";

/// Run CPU L = 4 à thermalisation longue (environ une minute et demie) et points de reprise fréquents.
fn args_therm_longue(sortie: &Path) -> Vec<String> {
    let mut a: Vec<String> = [
        "su2",
        "gen",
        "--backend",
        "cpu",
        "--L",
        "4",
        "--beta",
        "2.3",
        "--graine",
        "4242",
        "--therm",
        &THERM_LONGUE.to_string(),
        "--nmeas",
        "10",
        "--skip",
        "1",
        "--dump-every",
        "0",
        "--format",
        "q64",
        "--reprise-every",
        "50",
        "--outdir",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    a.push(sortie.to_str().expect("chemin UTF-8 de sortie").to_string());
    a
}

#[test]
fn pub14_kill9_en_thermalisation_reprise_acceptee() {
    use std::os::unix::process::ExitStatusExt;
    use std::process::Stdio;
    use std::time::{Duration, Instant};
    let d = dossier("pub14");
    let tue = d.join("tue");
    let journal = fs::File::create(d.join("tue.err")).expect("journal du run tué");
    let mut garde = GardeFils(Some(
        Command::new(env!("CARGO_BIN_EXE_kvu"))
            .args(args_therm_longue(&tue))
            .env("RAYON_NUM_THREADS", "2")
            .env("KEVINOTRON_BACKEND", backend_gpu())
            .stdout(Stdio::null())
            .stderr(journal)
            .spawn()
            .expect("lancement du run à tuer"),
    ));
    let fils = garde.0.as_mut().expect("fils lancé");
    let debut = Instant::now();
    loop {
        if let Some(st) = fils.try_wait().expect("état du fils") {
            panic!(
                "ARRÊT BRUTAL NON OBTENU : le run s'est terminé ({st}) avant le premier point de reprise ; journal : {}",
                fs::read_to_string(d.join("tue.err")).unwrap_or_default()
            );
        }
        if fs::metadata(tue.join("reprise.json")).is_ok_and(|m| m.len() > 0) {
            break;
        }
        assert!(
            debut.elapsed() < Duration::from_secs(60),
            "délai dépassé : aucun point de reprise en 60 s"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    fils.kill().expect("kill -9 du fils (par son PID)");
    let st = fils.wait().expect("récolte du fils tué");
    garde.0 = None;
    assert_eq!(
        st.signal(),
        Some(9),
        "ARRÊT BRUTAL NON OBTENU : le fils n'est pas mort de SIGKILL ({st})"
    );
    let etat = lire_json(&tue.join("reprise.json"));
    let pf = etat["pas_faits"].as_u64().expect("pas_faits");
    assert!(
        pf > 0 && pf < THERM_LONGUE && etat["mesures_faites"].as_u64() == Some(0),
        "l'arrêt n'est pas tombé en thermalisation : reprise.json = {etat}"
    );
    assert!(!tue.join("COMPLETE").exists(), "COMPLETE après un kill -9");
    // PUB-14 : l'en-tête de measurements.csv doit être sur le disque dès la création du run.
    assert_eq!(
        String::from_utf8_lossy(&octets(&tue.join("measurements.csv"))),
        ENTETE_MESURES,
        "measurements.csv après kill -9 en thermalisation (pas {pf}) : en-tête seul attendu (PUB-14)"
    );
    let a = (pf + 100).to_string();
    reussi(&reprendre("cpu", &tue, &["--arret-apres", &a]));
    let reference = d.join("ref");
    let mut args = args_therm_longue(&reference);
    args.extend(["--arret-apres".to_string(), a.clone()]);
    reussi(&kvu(&args.iter().map(String::as_str).collect::<Vec<_>>()));
    for nom in ["therm.csv", "measurements.csv"] {
        assert_eq!(
            octets(&reference.join(nom)),
            octets(&tue.join(nom)),
            "{nom} : run tué en thermalisation (pas {pf}) puis repris jusqu'au pas {a} ≠ run arrêté proprement au pas {a}"
        );
    }
    assert_eq!(
        lire_json(&tue.join("reprise.json"))["pas_faits"],
        lire_json(&reference.join("reprise.json"))["pas_faits"]
    );
    println!(
        "\nPASS PUB-14 : kill -9 (SIGKILL) au pas {pf} de la thermalisation, measurements.csv = en-tête seul, reprise rc 0, \
         therm.csv et measurements.csv identiques à l'octet au run arrêté proprement au pas {a}"
    );
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

// ------------------------------------------------------------------------------------------
// PUB-16 (0.1.1) : un run REPRIS passe `kvu store verifier` (COMPLETE au format d'un run neuf)
// ------------------------------------------------------------------------------------------

fn store_verifier(store: &Path, dir: &Path) -> Output {
    kvu(&[
        "store",
        "--store",
        store.to_str().expect("chemin UTF-8 du store"),
        "verifier",
        dir.to_str().expect("chemin UTF-8 du run"),
    ])
}

fn pub16_repris_verifiable(porte: &str, extra: &[&str]) {
    let d = dossier(porte);
    let b = d.join("chaine");
    let mut args = vec!["--reprise-every", "2", "--arret-apres", "10"];
    args.extend_from_slice(extra);
    reussi(&gen("cpu", 4, 6, 6, 2, &b, &args));
    assert!(!b.join("COMPLETE").exists(), "COMPLETE avant la reprise");
    reussi(&reprendre("cpu", &b, extra));
    let complete = String::from_utf8(octets(&b.join("COMPLETE"))).expect("COMPLETE UTF-8");
    let sha = kvu::heritage::manifest::sha256_hex(&octets(&b.join("manifest.json")));
    assert_eq!(
        complete,
        format!("rc=0\nmanifest_sha256={sha}\n"),
        "COMPLETE d'un run repris : format d'un run neuf attendu (PUB-16)"
    );
    let v = store_verifier(&d, &b);
    reussi(&v);
    let texte = String::from_utf8_lossy(&v.stdout);
    assert!(
        texte.contains(": OK,"),
        "verdict de store verifier : {texte}"
    );
    // la porte a des dents : un manifeste altéré après la clôture rend la vérification rouge (code 8)
    let mut f = fs::OpenOptions::new()
        .append(true)
        .open(b.join("manifest.json"))
        .expect("ouverture du manifeste");
    f.write_all(b" ").expect("altération du manifeste");
    drop(f);
    assert_eq!(
        store_verifier(&d, &b).status.code(),
        Some(8),
        "manifeste altéré non détecté"
    );
    println!(
        "\nPASS {porte} : run arrêté au pas 10 puis repris, COMPLETE = rc=0 + manifest_sha256, store verifier OK ; \
         manifeste altéré : code 8"
    );
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}

#[test]
fn pub16_run_repris_passe_store_verifier() {
    pub16_repris_verifiable("pub16_sans_mesures", &[]);
}

#[test]
fn pub16_run_repris_avec_mesures_passe_store_verifier() {
    pub16_repris_verifiable("pub16_wilson", &["--mesures", SPEC_WILSON]);
}

// ------------------------------------------------------------------------------------------
// PUB-14 (0.1.1), chemin --bord : therm_blocs.csv et blocs.csv (su2/bord.rs::ouvrir_fichiers)
// ------------------------------------------------------------------------------------------

/// Contre-exemple de la relecture pub-0.1.1-b : bord gelé, pas de pré-thermalisation, bloc de 1, un point à chaque pas.
fn args_bord_long(sortie: &Path) -> Vec<String> {
    let mut a: Vec<String> = [
        "su2",
        "gen",
        "--backend",
        "cpu",
        "--L",
        "4",
        "--beta",
        "2.3",
        "--graine",
        "4242",
        "--bord",
        "--bc",
        "cold",
        "--pre",
        "0",
        "--bloc",
        "1",
        "--therm",
        &THERM_LONGUE.to_string(),
        "--nmeas",
        "2",
        "--reprise-every",
        "1",
        "--outdir",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    a.push(sortie.to_str().expect("chemin UTF-8 de sortie").to_string());
    a
}

#[test]
fn pub14_bord_kill9_en_thermalisation_reprise_acceptee() {
    use std::os::unix::process::ExitStatusExt;
    use std::process::Stdio;
    use std::time::{Duration, Instant};
    let d = dossier("pub14_bord");
    let tue = d.join("tue");
    let journal = fs::File::create(d.join("tue.err")).expect("journal du run tué");
    let mut garde = GardeFils(Some(
        Command::new(env!("CARGO_BIN_EXE_kvu"))
            .args(args_bord_long(&tue))
            .env("RAYON_NUM_THREADS", "2")
            .env("KEVINOTRON_BACKEND", backend_gpu())
            .stdout(Stdio::null())
            .stderr(journal)
            .spawn()
            .expect("lancement du run à tuer"),
    ));
    let fils = garde.0.as_mut().expect("fils lancé");
    let debut = Instant::now();
    loop {
        if let Some(st) = fils.try_wait().expect("état du fils") {
            panic!(
                "ARRÊT BRUTAL NON OBTENU : le run --bord s'est terminé ({st}) avant le premier point de reprise ; journal : {}",
                fs::read_to_string(d.join("tue.err")).unwrap_or_default()
            );
        }
        if fs::metadata(tue.join("reprise.json")).is_ok_and(|m| m.len() > 0) {
            break;
        }
        assert!(
            debut.elapsed() < Duration::from_secs(60),
            "délai dépassé : aucun point de reprise en 60 s"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    fils.kill().expect("kill -9 du fils (par son PID)");
    let st = fils.wait().expect("récolte du fils tué");
    garde.0 = None;
    assert_eq!(
        st.signal(),
        Some(9),
        "ARRÊT BRUTAL NON OBTENU : le fils n'est pas mort de SIGKILL ({st})"
    );
    let etat = lire_json(&tue.join("reprise.json"));
    let pf = etat["pas_faits"].as_u64().expect("pas_faits");
    assert!(
        (1..THERM_LONGUE).contains(&pf),
        "l'arrêt n'est pas tombé en thermalisation : reprise.json = {etat}"
    );
    // PUB-14 (--bord) : blocs.csv (production, pas encore écrit) doit porter son en-tête sur le disque.
    let blocs = octets(&tue.join("blocs.csv"));
    assert!(
        blocs.ends_with(b"\n") && blocs.iter().filter(|&&b| b == b'\n').count() == 1,
        "blocs.csv après kill -9 en thermalisation (pas {pf}) : en-tête seul attendu (PUB-14, --bord), {} octets",
        blocs.len()
    );
    assert!(
        octets(&tue.join("therm_blocs.csv")).ends_with(b"\n"),
        "therm_blocs.csv sans en-tête complet après kill -9"
    );
    let a = (pf + 20).to_string();
    reussi(&reprendre("cpu", &tue, &["--arret-apres", &a]));
    let reference = d.join("ref");
    let mut args = args_bord_long(&reference);
    args.extend(["--arret-apres".to_string(), a.clone()]);
    reussi(&kvu(&args.iter().map(String::as_str).collect::<Vec<_>>()));
    for nom in ["therm_blocs.csv", "blocs.csv"] {
        assert_eq!(
            octets(&reference.join(nom)),
            octets(&tue.join(nom)),
            "{nom} : run --bord tué au pas {pf} puis repris jusqu'au pas {a} ≠ run arrêté proprement au pas {a}"
        );
    }
    println!(
        "\nPASS PUB-14-BORD : kill -9 (SIGKILL) au pas {pf} (--bord, thermalisation), blocs.csv = en-tête seul, reprise rc 0, \
         therm_blocs.csv et blocs.csv identiques à l'octet au run arrêté proprement au pas {a}"
    );
    fs::remove_dir_all(&d).expect("suppression du répertoire temporaire");
}
