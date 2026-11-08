//! Tests d'acceptation de l'archivage vérifiable `kvu su2 gen --archive`.

use kvu::heritage::manifest::{sha256_fichier, sha256_hex};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

const BINAIRE: &str = env!("CARGO_BIN_EXE_kvu");
static COMPTEUR: AtomicUsize = AtomicUsize::new(0);

struct Garde {
    chemin: PathBuf,
}

impl Garde {
    fn nouveau(nom: &str) -> Self {
        let n = COMPTEUR.fetch_add(1, Ordering::Relaxed);
        let chemin = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("{nom}_{}_{n}", std::process::id()));
        let _ = fs::remove_dir_all(&chemin);
        fs::create_dir_all(&chemin).unwrap();
        Garde { chemin }
    }
}

impl Drop for Garde {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.chemin);
    }
}

fn zstd_disponible() -> bool {
    Command::new("zstd")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn sauter(nom: &str) -> bool {
    if zstd_disponible() {
        false
    } else {
        eprintln!("SKIP {nom} : binaire zstd absent");
        true
    }
}

fn lancer(args: &[String]) -> Output {
    Command::new(BINAIRE).args(args).output().unwrap()
}

fn lancer_env(args: &[String], variable: &str, valeur: &str) -> Output {
    Command::new(BINAIRE)
        .args(args)
        .env(variable, valeur)
        .output()
        .unwrap()
}

fn afficher_erreur(sortie: &Output) -> String {
    format!(
        "status={} stdout={} stderr={}",
        sortie.status,
        String::from_utf8_lossy(&sortie.stdout),
        String::from_utf8_lossy(&sortie.stderr)
    )
}

fn assert_rc(sortie: &Output, attendu: i32) {
    assert_eq!(
        sortie.status.code(),
        Some(attendu),
        "code attendu {attendu}, obtenu {}",
        afficher_erreur(sortie)
    );
}

fn commande_b(outdir: &Path, graine: &str, beta: &str) -> Vec<String> {
    vec![
        "su2".to_string(),
        "gen".to_string(),
        "--backend".to_string(),
        "cpu".to_string(),
        "--L".to_string(),
        "4".to_string(),
        "--beta".to_string(),
        beta.to_string(),
        "--graine".to_string(),
        graine.to_string(),
        "--start".to_string(),
        "hot".to_string(),
        "--therm".to_string(),
        "10".to_string(),
        "--nmeas".to_string(),
        "20".to_string(),
        "--skip".to_string(),
        "2".to_string(),
        "--nor".to_string(),
        "2".to_string(),
        "--dump-every".to_string(),
        "5".to_string(),
        "--format".to_string(),
        "q64".to_string(),
        "--outdir".to_string(),
        outdir.to_string_lossy().into_owned(),
    ]
}

fn avec_extras(mut commande: Vec<String>, extras: &[&str]) -> Vec<String> {
    commande.extend(extras.iter().map(|s| (*s).to_string()));
    commande
}

fn charger_archive(dir: &Path) -> serde_json::Value {
    let texte = fs::read_to_string(dir.join("archive.json")).unwrap();
    serde_json::from_str(&texte).unwrap()
}

fn charger_json(dir: &Path, nom: &str) -> serde_json::Value {
    let texte = fs::read_to_string(dir.join(nom)).unwrap();
    serde_json::from_str(&texte).unwrap()
}

fn taille(chemin: &Path) -> u64 {
    fs::metadata(chemin).unwrap().len()
}

fn decompresser(chemin: &Path) -> Vec<u8> {
    let sortie = Command::new("zstd")
        .args(["-dc", "-q", "--"])
        .arg(chemin)
        .output()
        .unwrap();
    assert!(
        sortie.status.success(),
        "zstd -dc {} : {}",
        chemin.display(),
        String::from_utf8_lossy(&sortie.stderr)
    );
    sortie.stdout
}

fn sha_decompresse(chemin: &Path) -> String {
    sha256_hex(&decompresser(chemin))
}

fn copier_arbo(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    for entree in fs::read_dir(source).unwrap() {
        let entree = entree.unwrap();
        let chemin = entree.path();
        let cible = destination.join(entree.file_name());
        if entree.file_type().unwrap().is_dir() {
            copier_arbo(&chemin, &cible);
        } else {
            fs::copy(&chemin, &cible).unwrap();
        }
    }
}

fn extraire_cli(dir: &Path, sortie: &Path, indice: u64, format_explicite: bool) -> Output {
    let mut args = vec![
        "su2".to_string(),
        "archive".to_string(),
        "extraire".to_string(),
        dir.to_string_lossy().into_owned(),
        "--indice".to_string(),
        indice.to_string(),
        "--sortie".to_string(),
        sortie.to_string_lossy().into_owned(),
    ];
    if format_explicite {
        args.push("--format".to_string());
        args.push("q64".to_string());
    }
    lancer(&args)
}

fn ecrire_json(chemin: &Path, valeur: &serde_json::Value) {
    fs::write(chemin, serde_json::to_vec_pretty(valeur).unwrap()).unwrap();
}

fn configs(archive: &serde_json::Value) -> &Vec<serde_json::Value> {
    archive["chaines"][0]["configurations"].as_array().unwrap()
}

fn lignes_defauts(sortie: &Output) -> Vec<String> {
    String::from_utf8_lossy(&sortie.stdout)
        .lines()
        .filter(|l| l.starts_with("DEFAUT "))
        .map(|l| l.to_string())
        .collect()
}

#[test]
fn a1_index() {
    if sauter("A1 Index") {
        return;
    }
    let garde = Garde::nouveau("a1_index");
    let dir = garde.chemin.join("run");
    let commande = avec_extras(commande_b(&dir, "26110201", "2.3"), &["--archive"]);
    assert_rc(&lancer(&commande), 0);

    let archive = charger_archive(&dir);
    assert_eq!(archive["format"], "kvu-archive/1");
    assert_eq!(archive["chaines"].as_array().unwrap().len(), 1);
    let configurations = configs(&archive);
    assert_eq!(configurations.len(), 4);
    for (attendu_indice, attendu_balayages) in [(5u64, 20u64), (10, 30), (15, 40), (20, 50)] {
        let pos = configurations.iter().position(|c| c["indice"].as_u64() == Some(attendu_indice)).unwrap();
        assert_eq!(configurations[pos]["balayages"].as_u64(), Some(attendu_balayages));
    }
    for entree in configurations {
        let fichier = entree["fichier"].as_str().unwrap();
        let chemin = dir.join(fichier);
        assert_eq!(entree["octets"].as_u64(), Some(taille(&chemin)));
        assert_eq!(entree["sha256"].as_str(), Some(sha256_fichier(&chemin).unwrap().as_str()));
        assert_eq!(entree["sha256_contenu"].as_str(), Some(sha_decompresse(&chemin).as_str()));
    }
    assert_eq!(archive["complet"], true);
    let manifeste = charger_json(&dir, "manifest.json");
    // S7 : comparer les SEPT champs de provenance à manifest.json.
    for champ in ["binaire", "binaire_sha256", "argv", "source_hash", "git_head", "version", "hote"] {
        assert_eq!(
            archive["provenance"][champ],
            manifeste[champ],
            "champ de provenance {champ} différent"
        );
    }

    let verifier = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        dir.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verifier, 0);
    assert!(!String::from_utf8_lossy(&verifier.stdout).contains("DEFAUT"));

    let lister = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "lister".to_string(),
        dir.to_string_lossy().into_owned(),
    ]);
    assert_rc(&lister, 0);
    assert_eq!(String::from_utf8_lossy(&lister.stdout).lines().count(), 4);

    // S7 : extraction CLI SANS `--format` vérifie le format par défaut (celui de l'archive).
    let sortie_defaut = garde.chemin.join("defaut.npy");
    assert_rc(&extraire_cli(&dir, &sortie_defaut, 10, false), 0);
    let champ_defaut = kvu::stockage::npy::lire_champ(&sortie_defaut).unwrap();
    assert_eq!(champ_defaut.l, 4);
    assert_eq!(
        kvu::stockage::npy::inspecter_champ(&sortie_defaut).unwrap(),
        (kvu::stockage::npy::FormatChamp::Q64, 4)
    );

    let dir_initial = garde.chemin.join("initial");
    let commande_initial = avec_extras(
        commande_b(&dir_initial, "26110201", "2.3"),
        &["--archive", "--dump-initial"],
    );
    assert_rc(&lancer(&commande_initial), 0);
    let archive_initial = charger_archive(&dir_initial);
    let configurations_initial = configs(&archive_initial);
    assert_eq!(configurations_initial[0]["indice"].as_u64(), Some(0));
    assert_eq!(configurations_initial[0]["balayages"].as_u64(), Some(0));
}

#[test]
fn a2_lot() {
    if sauter("A2 Lot") {
        return;
    }
    let garde = Garde::nouveau("a2_lot");
    let dir = garde.chemin.join("lot");
    let commande = avec_extras(
        commande_b(&dir, "26110202,26110203", "2.3,2.5"),
        &["--archive"],
    );
    assert_rc(&lancer(&commande), 0);
    let archive = charger_archive(&dir);
    let chaines = archive["chaines"].as_array().unwrap();
    assert_eq!(chaines.len(), 2);
    assert_eq!(chaines[0]["dossier"], "c000");
    assert_eq!(chaines[1]["dossier"], "c001");
    assert_eq!(chaines[0]["beta"], 2.3);
    assert_eq!(chaines[1]["beta"], 2.5);
    assert_eq!(chaines[0]["graine"], 26110202u64);
    assert_eq!(chaines[1]["graine"], 26110203u64);
    assert_eq!(chaines[0]["configurations"].as_array().unwrap().len(), 4);
    assert_eq!(chaines[1]["configurations"].as_array().unwrap().len(), 4);

    // Vérifier le lot puis sélectionner explicitement la chaîne c001 à l'extraction.
    let verifier = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        dir.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verifier, 0);
    assert!(lignes_defauts(&verifier).is_empty());

    let sortie_c001 = garde.chemin.join("c001_extrait.npy");
    let extraction = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "extraire".to_string(),
        dir.to_string_lossy().into_owned(),
        "--indice".to_string(),
        "5".to_string(),
        "--chaine".to_string(),
        "1".to_string(),
        "--sortie".to_string(),
        sortie_c001.to_string_lossy().into_owned(),
    ]);
    assert_rc(&extraction, 0);
    let champ_c001 = kvu::stockage::npy::lire_champ(&sortie_c001).unwrap();
    assert_eq!(champ_c001.l, 4);
    let octets_c001 = decompresser(&dir.join("c001/cfg_000005.npy.zst"));
    assert_ne!(octets_c001, decompresser(&dir.join("c000/cfg_000005.npy.zst")));
    assert_eq!(fs::read(&sortie_c001).unwrap(), octets_c001);
}

#[test]
fn a3_aller_retour_au_bit() {
    if sauter("A3 Aller-retour au bit") {
        return;
    }
    let garde = Garde::nouveau("a3_aller_retour");
    let dir_non = garde.chemin.join("nonarchive");
    assert_rc(&lancer(&commande_b(&dir_non, "26110201", "2.3")), 0);

    let dir_archive = garde.chemin.join("archive");
    let commande_archive = avec_extras(
        commande_b(&dir_archive, "26110201", "2.3"),
        &["--archive"],
    );
    assert_rc(&lancer(&commande_archive), 0);

    let champ_fichier = kvu::stockage::npy::lire_champ(&dir_non.join("cfg_000010.npy")).unwrap();
    let champ_archive = kvu::stockage::archive::extraire(&dir_archive, 0, 10).unwrap();
    assert_eq!(champ_archive.l, 4);
    assert_eq!(champ_fichier.q.len(), champ_archive.q.len());
    for (a, b) in champ_fichier.q.iter().zip(&champ_archive.q) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    let mut cpu = kvu::heritage::su2_hb::CpuSu2::new(4, 26110201).unwrap();
    cpu.init(true);
    for _ in 0..30 {
        cpu.step(2.3, 2);
    }
    let attendu = cpu.flat();
    assert_eq!(attendu.len(), champ_archive.q.len());
    for (a, b) in attendu.iter().zip(&champ_archive.q) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    let sortie = garde.chemin.join("X.npy");
    assert_rc(&extraire_cli(&dir_archive, &sortie, 10, true), 0);
    let octets_sortie = fs::read(&sortie).unwrap();
    let octets_originaux = fs::read(dir_non.join("cfg_000010.npy")).unwrap();
    assert_eq!(octets_sortie, octets_originaux);
}

#[test]
fn a4_corruption() {
    if sauter("A4 Corruption") {
        return;
    }
    let garde = Garde::nouveau("a4_corruption");
    let base = garde.chemin.join("base");
    let commande_base = avec_extras(commande_b(&base, "26110201", "2.3"), &["--archive"]);
    assert_rc(&lancer(&commande_base), 0);

    // (a) un octet du .zst inversé : on attend `sha256` (et `contenu` si la décompression échoue).
    let cas_a = garde.chemin.join("a");
    copier_arbo(&base, &cas_a);
    let zst_a = cas_a.join("cfg_000010.npy.zst");
    let mut octets = fs::read(&zst_a).unwrap();
    let milieu = octets.len() / 2;
    octets[milieu] ^= 0x80;
    fs::write(&zst_a, octets).unwrap();
    let sortie_a = garde.chemin.join("a.npy");
    let verification_a = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        cas_a.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_a, 8);
    let defauts_a = lignes_defauts(&verification_a);
    assert!(defauts_a.iter().any(|l| l == "DEFAUT sha256 cfg_000010.npy.zst"));
    assert_rc(&extraire_cli(&cas_a, &sortie_a, 10, true), 8);
    assert!(!sortie_a.exists());

    // (b) .zst tronqué de moitié : rapport EXACT attendu : taille, sha256, contenu.
    let cas_b = garde.chemin.join("b");
    copier_arbo(&base, &cas_b);
    let zst_b = cas_b.join("cfg_000010.npy.zst");
    let octets = fs::read(&zst_b).unwrap();
    fs::write(&zst_b, &octets[..octets.len() / 2]).unwrap();
    let sortie_b = garde.chemin.join("b.npy");
    let verification_b = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        cas_b.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_b, 8);
    let mut attendus_b = vec![
        "DEFAUT contenu cfg_000010.npy.zst".to_string(),
        "DEFAUT sha256 cfg_000010.npy.zst".to_string(),
        "DEFAUT taille cfg_000010.npy.zst".to_string(),
    ];
    let mut defauts_b = lignes_defauts(&verification_b);
    defauts_b.sort();
    attendus_b.sort();
    assert_eq!(defauts_b, attendus_b, "rapport (b) inexact");
    assert_rc(&extraire_cli(&cas_b, &sortie_b, 10, true), 8);
    assert!(!sortie_b.exists());

    // (c) .zst supprimé
    let cas_c = garde.chemin.join("c");
    copier_arbo(&base, &cas_c);
    fs::remove_file(cas_c.join("cfg_000010.npy.zst")).unwrap();
    let sortie_c = garde.chemin.join("c.npy");
    let verification_c = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        cas_c.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_c, 8);
    assert!(String::from_utf8_lossy(&verification_c.stdout).contains("DEFAUT absent"));
    assert_rc(&extraire_cli(&cas_c, &sortie_c, 10, true), 8);
    assert!(!sortie_c.exists());

    // (d) fichier non indexé ajouté
    let cas_d = garde.chemin.join("d");
    copier_arbo(&base, &cas_d);
    fs::copy(
        cas_d.join("cfg_000015.npy.zst"),
        cas_d.join("cfg_000099.npy.zst"),
    )
    .unwrap();
    let verification_d = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        cas_d.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_d, 8);
    assert!(String::from_utf8_lossy(&verification_d.stdout).contains("DEFAUT non_indexe"));

    // (e) un caractère de sha256 falsifié dans archive.json :
    // on attend EXACTEMENT `DEFAUT sha256 cfg_000010.npy.zst` et l'ABSENCE de `DEFAUT contenu`.
    let cas_e = garde.chemin.join("e");
    copier_arbo(&base, &cas_e);
    let chemin_archive = cas_e.join("archive.json");
    let mut archive_e = charger_archive(&cas_e);
    let ancien = archive_e["chaines"][0]["configurations"][1]["sha256"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(archive_e["chaines"][0]["configurations"][1]["indice"], 10);
    let remplacement = if ancien.starts_with('0') { 'f' } else { '0' };
    let nouveau: String = std::iter::once(remplacement).chain(ancien.chars().skip(1)).collect();
    archive_e["chaines"][0]["configurations"][1]["sha256"] = serde_json::Value::String(nouveau);
    ecrire_json(&chemin_archive, &archive_e);
    let sortie_e = garde.chemin.join("e.npy");
    let verification_e = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        cas_e.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_e, 8);
    let defauts_e = lignes_defauts(&verification_e);
    assert_eq!(defauts_e, vec!["DEFAUT sha256 cfg_000010.npy.zst"]);
    assert_rc(&extraire_cli(&cas_e, &sortie_e, 10, true), 8);
    assert!(!sortie_e.exists());

    // (f) archive falsifiée de façon cohérente pour le .zst, mais pas pour le contenu.
    // S6 : sélectionner l'entrée par `indice == 10`, actualiser UNIQUEMENT sha256 et octets du nouveau fichier,
    // garder sha256_contenu et octets_contenu ; attendre exactement `contenu`, ni `sha256` ni `taille`.
    let cas_f = garde.chemin.join("f");
    copier_arbo(&base, &cas_f);
    let zst_f = cas_f.join("cfg_000010.npy.zst");
    fs::copy(cas_f.join("cfg_000015.npy.zst"), &zst_f).unwrap();
    let chemin_archive_f = cas_f.join("archive.json");
    let mut archive_f = charger_archive(&cas_f);
    {
        let chaine = archive_f["chaines"][0].as_object_mut().unwrap();
        let configurations = chaine.get_mut("configurations").unwrap().as_array_mut().unwrap();
        let entree_10 = configurations
            .iter_mut()
            .find(|c| c["indice"].as_u64() == Some(10))
            .expect("entrée indice 10 absente");
        entree_10["sha256"] = serde_json::Value::String(sha256_fichier(&zst_f).unwrap());
        entree_10["octets"] = serde_json::json!(taille(&zst_f));
        // Ne pas toucher à sha256_contenu ni octets_contenu.
    }
    ecrire_json(&chemin_archive_f, &archive_f);
    let sortie_f = garde.chemin.join("f.npy");
    let verification_f = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        cas_f.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_f, 8);
    assert_eq!(lignes_defauts(&verification_f), vec!["DEFAUT contenu cfg_000010.npy.zst"]);
    assert_rc(&extraire_cli(&cas_f, &sortie_f, 10, true), 8);
    assert!(!sortie_f.exists());

    // (g) deux .zst échangés
    let cas_g = garde.chemin.join("g");
    copier_arbo(&base, &cas_g);
    let a = cas_g.join("cfg_000010.npy.zst");
    let b = cas_g.join("cfg_000015.npy.zst");
    let temporaire = cas_g.join("cfg_swap_tmp.zst");
    fs::rename(&a, &temporaire).unwrap();
    fs::rename(&b, &a).unwrap();
    fs::rename(&temporaire, &b).unwrap();
    let verification_g = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        cas_g.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_g, 8);
    let mut defauts_g = lignes_defauts(&verification_g);
    defauts_g.sort();
    // Intégrateur (M1 r2) : deux fichiers échangés diffèrent de leur entrée par l'empreinte compressée ET par le contenu (natures
    // indépendantes depuis la correction O1/S4), et en général par la taille ; on exige sha256 et contenu pour chacun, taille facultative,
    // et rien d'autre.
    for f in ["cfg_000010.npy.zst", "cfg_000015.npy.zst"] {
        for nature in ["sha256", "contenu"] {
            assert!(defauts_g.contains(&format!("DEFAUT {nature} {f}")), "{nature} {f} absent : {defauts_g:?}");
        }
    }
    assert!(
        defauts_g.iter().all(|d| {
            let ok_nature = d.starts_with("DEFAUT sha256 ") || d.starts_with("DEFAUT contenu ") || d.starts_with("DEFAUT taille ");
            let ok_fichier = d.ends_with(" cfg_000010.npy.zst") || d.ends_with(" cfg_000015.npy.zst");
            ok_nature && ok_fichier
        }),
        "défaut inattendu : {defauts_g:?}"
    );
    let sortie_g = garde.chemin.join("g.npy");
    assert_rc(&extraire_cli(&cas_g, &sortie_g, 10, true), 8);
    assert!(!sortie_g.exists());

    // (h) archive.json tronqué
    let cas_h = garde.chemin.join("h");
    copier_arbo(&base, &cas_h);
    let chemin_h = cas_h.join("archive.json");
    let octets_h = fs::read(&chemin_h).unwrap();
    fs::write(&chemin_h, &octets_h[..octets_h.len() / 2]).unwrap();
    let verification_h = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        cas_h.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_h, 8);
    assert_eq!(lignes_defauts(&verification_h), vec!["DEFAUT index archive.json"]);
    let sortie_h = garde.chemin.join("h.npy");
    assert_rc(&extraire_cli(&cas_h, &sortie_h, 10, true), 8);
    assert!(!sortie_h.exists());
}

#[test]
fn a5_determinisme() {
    if sauter("A5 Déterminisme") {
        return;
    }
    let garde = Garde::nouveau("a5_determinisme");
    let dir1 = garde.chemin.join("n1");
    let dir4 = garde.chemin.join("n4");
    let commande1 = avec_extras(commande_b(&dir1, "26110201", "2.3"), &["--archive"]);
    let commande4 = avec_extras(commande_b(&dir4, "26110201", "2.3"), &["--archive"]);
    assert_rc(&lancer_env(&commande1, "RAYON_NUM_THREADS", "1"), 0);
    assert_rc(&lancer_env(&commande4, "RAYON_NUM_THREADS", "4"), 0);
    let archive1 = charger_archive(&dir1);
    let archive4 = charger_archive(&dir4);
    assert_eq!(archive1["chaines"], archive4["chaines"]);

    let avant = fs::read(dir1.join("archive.json")).unwrap();
    let rejoue = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "indexer".to_string(),
        dir1.to_string_lossy().into_owned(),
    ]);
    assert_rc(&rejoue, 0);
    let apres = fs::read(dir1.join("archive.json")).unwrap();
    assert_eq!(avant, apres);
}

#[test]
fn a6_reprise() {
    if sauter("A6 Reprise") {
        return;
    }
    let garde = Garde::nouveau("a6_reprise");
    let interruption = garde.chemin.join("interruption");
    let commande_interruption = avec_extras(
        commande_b(&interruption, "26110201", "2.3"),
        &["--archive", "--reprise-every", "10", "--arret-apres", "34"],
    );
    assert_rc(&lancer(&commande_interruption), 0);
    let archive_interruption = charger_archive(&interruption);
    assert_eq!(archive_interruption["complet"], false);
    let indices_interruption: Vec<u64> = configs(&archive_interruption)
        .iter()
        .map(|c| c["indice"].as_u64().unwrap())
        .collect();
    assert_eq!(indices_interruption, vec![5, 10]);

    let reprise = lancer(&[
        "su2".to_string(),
        "gen".to_string(),
        "--backend".to_string(),
        "cpu".to_string(),
        "--reprendre".to_string(),
        interruption.to_string_lossy().into_owned(),
    ]);
    assert_rc(&reprise, 0);
    let archive_reprise = charger_archive(&interruption);
    assert_eq!(archive_reprise["complet"], true);
    let indices_reprise: Vec<u64> = configs(&archive_reprise)
        .iter()
        .map(|c| c["indice"].as_u64().unwrap())
        .collect();
    assert_eq!(indices_reprise, vec![5, 10, 15, 20]);

    let direct = garde.chemin.join("direct");
    let commande_direct = avec_extras(commande_b(&direct, "26110201", "2.3"), &["--archive"]);
    assert_rc(&lancer(&commande_direct), 0);
    let archive_direct = charger_archive(&direct);
    assert_eq!(
        configs(&archive_reprise),
        configs(&archive_direct)
    );
}

#[test]
fn a7_refus() {
    if sauter("A7 Refus") {
        return;
    }
    let garde = Garde::nouveau("a7_refus");

    let sans_dump = garde.chemin.join("sans_dump");
    let commande_sans_dump = vec![
        "su2".to_string(),
        "gen".to_string(),
        "--backend".to_string(),
        "cpu".to_string(),
        "--L".to_string(),
        "4".to_string(),
        "--beta".to_string(),
        "2.3".to_string(),
        "--graine".to_string(),
        "26110201".to_string(),
        "--start".to_string(),
        "hot".to_string(),
        "--therm".to_string(),
        "10".to_string(),
        "--nmeas".to_string(),
        "20".to_string(),
        "--skip".to_string(),
        "2".to_string(),
        "--nor".to_string(),
        "2".to_string(),
        "--dump-every".to_string(),
        "0".to_string(),
        "--format".to_string(),
        "q64".to_string(),
        "--archive".to_string(),
        "--outdir".to_string(),
        sans_dump.to_string_lossy().into_owned(),
    ];
    assert_rc(&lancer(&commande_sans_dump), 2);

    let bord = garde.chemin.join("bord");
    let commande_bord = vec![
        "su2".to_string(),
        "gen".to_string(),
        "--backend".to_string(),
        "cpu".to_string(),
        "--L".to_string(),
        "4".to_string(),
        "--beta".to_string(),
        "2.3".to_string(),
        "--graine".to_string(),
        "26110201".to_string(),
        "--start".to_string(),
        "hot".to_string(),
        "--therm".to_string(),
        "0".to_string(),
        "--nmeas".to_string(),
        "100".to_string(),
        "--skip".to_string(),
        "1".to_string(),
        "--nor".to_string(),
        "2".to_string(),
        "--dump-every".to_string(),
        "1".to_string(),
        "--archive".to_string(),
        "--bord".to_string(),
        "--bc".to_string(),
        "hot".to_string(),
        "--outdir".to_string(),
        bord.to_string_lossy().into_owned(),
    ];
    assert_rc(&lancer(&commande_bord), 2);

    let non_run = garde.chemin.join("non_run");
    fs::create_dir_all(&non_run).unwrap();
    assert_rc(
        &lancer(&[
            "su2".to_string(),
            "archive".to_string(),
            "indexer".to_string(),
            non_run.to_string_lossy().into_owned(),
        ]),
        2,
    );
    assert_rc(
        &lancer(&[
            "su2".to_string(),
            "archive".to_string(),
            "verifier".to_string(),
            non_run.to_string_lossy().into_owned(),
        ]),
        2,
    );

    let non_compresse = garde.chemin.join("non_compresse");
    assert_rc(&lancer(&commande_b(&non_compresse, "26110201", "2.3")), 0);
    let indexage_non_compresse = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "indexer".to_string(),
        non_compresse.to_string_lossy().into_owned(),
    ]);
    assert_rc(&indexage_non_compresse, 2);
    assert!(
        String::from_utf8_lossy(&indexage_non_compresse.stderr).contains("--zstd")
            || String::from_utf8_lossy(&indexage_non_compresse.stdout).contains("--zstd")
    );

    let archivable = garde.chemin.join("archivable");
    assert_rc(
        &lancer(&avec_extras(
            commande_b(&archivable, "26110201", "2.3"),
            &["--archive"],
        )),
        0,
    );
    let sortie_absente = garde.chemin.join("absente.npy");
    assert_rc(&extraire_cli(&archivable, &sortie_absente, 3, true), 2);
    assert!(!sortie_absente.exists());
    let commande_chaine = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "extraire".to_string(),
        archivable.to_string_lossy().into_owned(),
        "--indice".to_string(),
        "10".to_string(),
        "--chaine".to_string(),
        "5".to_string(),
        "--sortie".to_string(),
        sortie_absente.to_string_lossy().into_owned(),
        "--format".to_string(),
        "q64".to_string(),
    ]);
    assert_rc(&commande_chaine, 2);
    assert!(!sortie_absente.exists());

    // .part résiduel : signalé par verifier, ignoré par une réindexation (S3, S7).
    let part = garde.chemin.join("part");
    copier_arbo(&archivable, &part);
    fs::copy(
        part.join("cfg_000010.npy.zst"),
        part.join("cfg_000099.npy.zst.part"),
    )
    .unwrap();
    let verification_part = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        part.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_part, 8);
    assert!(String::from_utf8_lossy(&verification_part.stdout).contains("DEFAUT non_indexe"));

    // Réindexation : le .part doit être ignoré, l'index reste identique à celui sans .part.
    let avant_part = fs::read(part.join("archive.json")).unwrap();
    let reindex_part = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "indexer".to_string(),
        part.to_string_lossy().into_owned(),
    ]);
    assert_rc(&reindex_part, 0);
    let apres_part = fs::read(part.join("archive.json")).unwrap();
    assert_eq!(avant_part, apres_part, "la réindexation doit ignorer le .part résiduel");

    let vide = garde.chemin.join("vide");
    let commande_vide = avec_extras(
        commande_b(&vide, "26110201", "2.3"),
        &["--archive", "--arret-apres", "0"],
    );
    assert_rc(&lancer(&commande_vide), 0);
    let archive_vide = charger_archive(&vide);
    assert_eq!(archive_vide["complet"], false);
    assert!(configs(&archive_vide).is_empty());
    let verification_vide = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        vide.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_vide, 0);
}

#[test]
fn a8_index_run_existant() {
    if sauter("A8 Index sur run existant") {
        return;
    }
    let garde = Garde::nouveau("a8_index_existant");
    let existant = garde.chemin.join("existant");
    assert_rc(
        &lancer(&avec_extras(
            commande_b(&existant, "26110201", "2.3"),
            &["--zstd"],
        )),
        0,
    );
    let indexage = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "indexer".to_string(),
        existant.to_string_lossy().into_owned(),
    ]);
    assert_rc(&indexage, 0);
    let archive_existant = charger_archive(&existant);

    let direct = garde.chemin.join("direct");
    assert_rc(
        &lancer(&avec_extras(commande_b(&direct, "26110201", "2.3"), &["--archive"])),
        0,
    );
    let archive_direct = charger_archive(&direct);
    assert_eq!(
        configs(&archive_existant),
        configs(&archive_direct)
    );
}

/// S1 : permutation des noms (et tailles/empreintes) des entrées 5 et 10 en gardant indices et balayages.
/// Un lecteur correct doit rejeter l'index (défaut `index`) et `extraire(…, 10)` doit échouer.
#[test]
fn a9_permutation_fichier_indice() {
    if sauter("A9 Permutation") {
        return;
    }
    let garde = Garde::nouveau("a9_permutation");
    let dir = garde.chemin.join("base");
    assert_rc(&lancer(&avec_extras(commande_b(&dir, "26110201", "2.3"), &["--archive"])), 0);

    let chemin_archive = dir.join("archive.json");
    let mut archive = charger_archive(&dir);
    let archive_initiale = archive.clone();
    {
        let chaine = archive["chaines"][0].as_object_mut().unwrap();
        let configurations = chaine.get_mut("configurations").unwrap().as_array_mut().unwrap();
        let pos5 = configurations.iter().position(|c| c["indice"].as_u64() == Some(5)).unwrap();
        let pos10 = configurations.iter().position(|c| c["indice"].as_u64() == Some(10)).unwrap();
        // Échanger fichier, sha256, octets, sha256_contenu, octets_contenu ; CONSERVER indice et balayages.
        let champs = ["fichier", "sha256", "octets", "sha256_contenu", "octets_contenu"];
        for champ in champs {
            let tmp = configurations[pos5][champ].clone();
            configurations[pos5][champ] = configurations[pos10][champ].clone();
            configurations[pos10][champ] = tmp;
        }
    }
    ecrire_json(&chemin_archive, &archive);

    let verification = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        dir.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification, 8);
    assert_eq!(lignes_defauts(&verification), vec!["DEFAUT index archive.json"]);

    let sortie = garde.chemin.join("s.npy");
    assert_rc(&extraire_cli(&dir, &sortie, 10, true), 8);
    assert!(!sortie.exists());

    // Une entrée INDEXÉE à sept chiffres reste hors schéma, même avec son fichier
    // présent, ses empreintes intactes et des balayages cohérents.
    let sept_chiffres = garde.chemin.join("sept_chiffres");
    copier_arbo(&dir, &sept_chiffres);
    let mut archive_sept = archive_initiale;
    archive_sept["nmeas"] = serde_json::json!(1_000_000u64);
    {
        let configurations = archive_sept["chaines"][0]["configurations"].as_array_mut().unwrap();
        let entree = configurations.iter_mut().find(|c| c["indice"].as_u64() == Some(20)).unwrap();
        fs::rename(
            sept_chiffres.join(entree["fichier"].as_str().unwrap()),
            sept_chiffres.join("cfg_1000000.npy.zst"),
        )
        .unwrap();
        entree["fichier"] = serde_json::json!("cfg_1000000.npy.zst");
        entree["indice"] = serde_json::json!(1_000_000u64);
        entree["balayages"] = serde_json::json!(2_000_010u64);
    }
    ecrire_json(&sept_chiffres.join("archive.json"), &archive_sept);
    let mut config_sept = charger_json(&sept_chiffres, "config_run.json");
    config_sept["nmeas"] = serde_json::json!(1_000_000u64);
    ecrire_json(&sept_chiffres.join("config_run.json"), &config_sept);
    assert!(sept_chiffres.join("cfg_1000000.npy.zst").is_file());
    assert!(kvu::stockage::archive::lire_index(&sept_chiffres).is_err());
    let verification_sept = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        sept_chiffres.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_sept, 8);
    assert_eq!(lignes_defauts(&verification_sept), vec!["DEFAUT index archive.json"]);
    assert!(kvu::stockage::archive::extraire(&sept_chiffres, 0, 1_000_000).is_err());
    let sortie_sept = garde.chemin.join("sept_chiffres.npy");
    assert_rc(&extraire_cli(&sept_chiffres, &sortie_sept, 1_000_000, true), 8);
    assert!(!sortie_sept.exists());
}

/// S2 : config_run.json présente mais invalide (`{}`, champ obligatoire supprimé, mauvais type)
/// doit produire un défaut `index` dans `verifier`.
#[test]
fn a10_config_invalide() {
    if sauter("A10 Config invalide") {
        return;
    }
    let garde = Garde::nouveau("a10_config_invalide");
    let base = garde.chemin.join("base");
    assert_rc(&lancer(&avec_extras(commande_b(&base, "26110201", "2.3"), &["--archive"])), 0);

    let cas_vide = garde.chemin.join("vide");
    copier_arbo(&base, &cas_vide);
    fs::write(cas_vide.join("config_run.json"), b"{}").unwrap();
    let verification_vide = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        cas_vide.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_vide, 8);
    assert_eq!(lignes_defauts(&verification_vide), vec!["DEFAUT index archive.json"]);

    let cas_champ_manquant = garde.chemin.join("champ_manquant");
    copier_arbo(&base, &cas_champ_manquant);
    let mut config_m = charger_json(&cas_champ_manquant, "config_run.json");
    config_m.as_object_mut().unwrap().remove("therm");
    ecrire_json(&cas_champ_manquant.join("config_run.json"), &config_m);
    let verification_m = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        cas_champ_manquant.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_m, 8);
    assert_eq!(lignes_defauts(&verification_m), vec!["DEFAUT index archive.json"]);

    let cas_mauvais_type = garde.chemin.join("mauvais_type");
    copier_arbo(&base, &cas_mauvais_type);
    let mut config_t = charger_json(&cas_mauvais_type, "config_run.json");
    config_t["therm"] = serde_json::Value::String("dix".to_string());
    ecrire_json(&cas_mauvais_type.join("config_run.json"), &config_t);
    let verification_t = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        cas_mauvais_type.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_t, 8);
    assert_eq!(lignes_defauts(&verification_t), vec!["DEFAUT index archive.json"]);

    // Les deux booléens sont requis : absence, chaîne, entier et null sont refusés.
    for champ in ["dump_initial", "zstd"] {
        for (nom, valeur) in [
            ("absent", None),
            ("texte", Some(serde_json::json!("invalide"))),
            ("entier", Some(serde_json::json!(1u64))),
            ("null", Some(serde_json::Value::Null)),
        ] {
            let cas = garde.chemin.join(format!("{champ}_{nom}"));
            copier_arbo(&base, &cas);
            let mut config = charger_json(&cas, "config_run.json");
            match valeur {
                Some(v) => config[champ] = v,
                None => {
                    config.as_object_mut().unwrap().remove(champ);
                }
            }
            ecrire_json(&cas.join("config_run.json"), &config);
            let verification = lancer(&[
                "su2".to_string(),
                "archive".to_string(),
                "verifier".to_string(),
                cas.to_string_lossy().into_owned(),
            ]);
            assert_rc(&verification, 8);
            assert_eq!(
                lignes_defauts(&verification),
                vec!["DEFAUT index archive.json"],
                "booléen {champ}, cas {nom}"
            );
        }
    }
}

/// S3 : détection des anomalies. `cfg_*.npy` non compressé (même non canonique) ⇒ refus d'indexer ;
/// dans `verifier`, `cfg_1000000.npy.zst` et `reste.part` ⇒ `non_indexe`.
#[test]
fn a11_anomalies() {
    if sauter("A11 Anomalies") {
        return;
    }
    let garde = Garde::nouveau("a11_anomalies");
    let base = garde.chemin.join("base");
    assert_rc(&lancer(&avec_extras(commande_b(&base, "26110201", "2.3"), &["--archive"])), 0);

    // cfg_incomplet.npy non compressé : l'indexation doit refuser.
    let non_canonique = garde.chemin.join("non_canonique");
    copier_arbo(&base, &non_canonique);
    fs::write(non_canonique.join("cfg_incomplet.npy"), b"x").unwrap();
    let indexage = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "indexer".to_string(),
        non_canonique.to_string_lossy().into_owned(),
    ]);
    assert_rc(&indexage, 2);
    assert!(
        String::from_utf8_lossy(&indexage.stderr).contains("--zstd")
            || String::from_utf8_lossy(&indexage.stdout).contains("--zstd")
    );

    // cfg_1000000.npy.zst : non canonique, doit être signalé non_indexe.
    let non_canonique_zst = garde.chemin.join("non_canonique_zst");
    copier_arbo(&base, &non_canonique_zst);
    fs::copy(
        non_canonique_zst.join("cfg_000015.npy.zst"),
        non_canonique_zst.join("cfg_1000000.npy.zst"),
    ).unwrap();
    let verification_zst = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        non_canonique_zst.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_zst, 8);
    assert!(
        lignes_defauts(&verification_zst)
            .iter()
            .any(|l| l == "DEFAUT non_indexe cfg_1000000.npy.zst")
    );

    // reste.part : tout résidu .part doit être signalé non_indexe.
    let part_residuel = garde.chemin.join("part_residuel");
    copier_arbo(&base, &part_residuel);
    fs::write(part_residuel.join("reste.part"), b"x").unwrap();
    let verification_part = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        part_residuel.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification_part, 8);
    assert!(
        lignes_defauts(&verification_part)
            .iter()
            .any(|l| l == "DEFAUT non_indexe reste.part")
    );
}

/// S5/O4 : débordements du produit et de la somme dans `balayages_pour_indice`.
/// Même avec des balayages égaux au résultat modulo, chaque index doit être refusé.
#[test]
fn a12_debordement() {
    if sauter("A12 Débordement") {
        return;
    }
    let garde = Garde::nouveau("a12_debordement");
    let base = garde.chemin.join("base");
    assert_rc(&lancer(&avec_extras(commande_b(&base, "26110201", "2.3"), &["--archive"])), 0);

    // Deux cas indépendants : produit 5*2 représentable puis somme débordante ;
    // produit 5*MAX débordant avec therm=0 pour isoler la multiplication.
    for (nom, therm, skip, attendu_modulo) in [
        ("somme", u64::MAX, 2u64, 9u64),
        ("produit", 0u64, u64::MAX, u64::MAX - 4),
    ] {
        let cas = garde.chemin.join(nom);
        copier_arbo(&base, &cas);
        let mut archive = charger_archive(&cas);
        archive["therm"] = serde_json::json!(therm);
        archive["skip"] = serde_json::json!(skip);
        archive["nmeas"] = serde_json::json!(5u64);
        let balayages_modulo = therm.wrapping_add(5u64.wrapping_mul(skip));
        assert_eq!(balayages_modulo, attendu_modulo, "cas {nom}");
        {
            let configurations = archive["chaines"][0]["configurations"].as_array_mut().unwrap();
            // Index minimal : seul le dump 5 reste présent et indexé.
            for entree in configurations.iter().filter(|c| c["indice"].as_u64() != Some(5)) {
                fs::remove_file(cas.join(entree["fichier"].as_str().unwrap())).unwrap();
            }
            configurations.retain(|c| c["indice"].as_u64() == Some(5));
            assert_eq!(configurations.len(), 1);
            configurations[0]["balayages"] = serde_json::json!(balayages_modulo);
        }
        ecrire_json(&cas.join("archive.json"), &archive);
        let mut config = charger_json(&cas, "config_run.json");
        config["therm"] = serde_json::json!(therm);
        config["skip"] = serde_json::json!(skip);
        config["nmeas"] = serde_json::json!(5u64);
        ecrire_json(&cas.join("config_run.json"), &config);

        assert!(
            kvu::stockage::archive::lire_index(&cas).is_err(),
            "débordement de {nom} accepté avec balayages modulo {balayages_modulo}"
        );
        let verification = lancer(&[
            "su2".to_string(),
            "archive".to_string(),
            "verifier".to_string(),
            cas.to_string_lossy().into_owned(),
        ]);
        assert_rc(&verification, 8);
        assert_eq!(lignes_defauts(&verification), vec!["DEFAUT index archive.json"]);
    }
}

/// Intégrateur (M1 r3) : contrôles négatifs des objections S1 et S2 d'Astra (tour 3) — entrée de manifeste LISTÉE sans empreinte valide ou
/// avec une empreinte discordante (la réindexation échoue et ne remplace pas l'index) ; `depart` hors domaine (défaut `index`, extraction
/// refusée).
#[test]
fn a13_manifeste_malforme_et_depart() {
    if sauter("A13 Manifeste malformé et départ") {
        return;
    }
    let garde = Garde::nouveau("a13_manifeste_depart");
    let base = garde.chemin.join("base");
    assert_rc(&lancer(&avec_extras(commande_b(&base, "26110201", "2.3"), &["--archive"])), 0);

    let cible = "cfg_000010.npy.zst";
    let empreintes: [(&str, Option<serde_json::Value>); 3] = [
        ("sans_sha", None),
        ("sha_mal_type", Some(serde_json::json!(12345))),
        ("sha_discordant", Some(serde_json::Value::String("0".repeat(64)))),
    ];
    for (nom, valeur) in empreintes {
        let cas = garde.chemin.join(nom);
        copier_arbo(&base, &cas);
        let mut manifeste = charger_json(&cas, "manifest.json");
        let mut trouve = false;
        for f in manifeste["fichiers"].as_array_mut().unwrap() {
            if f["nom"] == cible {
                trouve = true;
                match &valeur {
                    None => {
                        f.as_object_mut().unwrap().remove("sha256");
                    }
                    Some(v) => f["sha256"] = v.clone(),
                }
            }
        }
        assert!(trouve, "{cible} absent du manifeste");
        ecrire_json(&cas.join("manifest.json"), &manifeste);
        let avant = fs::read(cas.join("archive.json")).unwrap();
        let reindexation = lancer(&[
            "su2".to_string(),
            "archive".to_string(),
            "indexer".to_string(),
            cas.to_string_lossy().into_owned(),
        ]);
        assert_ne!(reindexation.status.code(), Some(0), "{nom} : réindexation acceptée");
        assert_eq!(fs::read(cas.join("archive.json")).unwrap(), avant, "{nom} : index remplacé");
    }

    let cas_depart = garde.chemin.join("depart_invalide");
    copier_arbo(&base, &cas_depart);
    let mut archive = charger_archive(&cas_depart);
    archive["chaines"][0]["depart"] = serde_json::Value::String("invalide".to_string());
    ecrire_json(&cas_depart.join("archive.json"), &archive);
    let verification = lancer(&[
        "su2".to_string(),
        "archive".to_string(),
        "verifier".to_string(),
        cas_depart.to_string_lossy().into_owned(),
    ]);
    assert_rc(&verification, 8);
    assert_eq!(lignes_defauts(&verification), vec!["DEFAUT index archive.json"]);
    let sortie = garde.chemin.join("depart.npy");
    assert_rc(&extraire_cli(&cas_depart, &sortie, 10, true), 8);
    assert!(!sortie.exists());
}
