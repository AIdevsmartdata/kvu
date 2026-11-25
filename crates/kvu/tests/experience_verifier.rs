// SPDX-License-Identifier: Apache-2.0
//! E-ANTIDATE, E-SHA et invariants d'ajout sur des manifestes synthétiques.

use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static COMPTEUR: AtomicU64 = AtomicU64::new(0);

struct ExperienceTest(PathBuf);

impl ExperienceTest {
    fn nouvelle() -> Self {
        let p = std::env::temp_dir().join(format!(
            "kvu-experience-verifier-{}-{}",
            std::process::id(),
            COMPTEUR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        fs::write(
            p.join("PREREG.md"),
            "# Critère synthétique\nSeuil : 1.\nFin.\n",
        )
        .unwrap();
        let e = Self(p);
        e.ok(&[
            "init",
            "--id",
            "temoin",
            "--titre",
            "Témoin",
            "--question",
            "Question",
            "--prereg",
            "PREREG.md",
            "--gele-le",
            "2001-09-09T01:46:40Z",
            "--machine-principale",
            "principale",
        ]);
        e
    }

    fn commande(&self, args: &[&str]) -> Command {
        let mut commande = Command::new(env!("CARGO_BIN_EXE_kvu"));
        commande
            .arg("experience")
            .args(args)
            .arg("--racine")
            .arg(&self.0)
            .env("KVU_EXPERIENCE_HORLOGE", "2001-09-09T02:00:00Z")
            .current_dir(&self.0);
        commande
    }

    fn lancer(&self, args: &[&str]) -> Output {
        self.commande(args).output().unwrap()
    }

    fn ok(&self, args: &[&str]) -> Output {
        let sortie = self.lancer(args);
        assert_eq!(sortie.status.code(), Some(0), "{args:?} : {sortie:?}");
        sortie
    }

    fn lire(&self) -> Value {
        serde_json::from_slice(&self.octets()).unwrap()
    }

    fn octets(&self) -> Vec<u8> {
        fs::read(self.0.join("EXPERIENCE.json")).unwrap()
    }

    fn run(&self, id: &str, graine: u64, debut: u64) {
        let manifeste = json!({
            "schema": "kevinotron-unifie/1",
            "argv": [
                "/x/kvu", "su2", "gen", "--L", "8", "--beta", "2.3", "--graine",
                graine.to_string(), "--backend", "cuda", "--therm", "10", "--nmeas",
                "5", "--skip", "2", "--outdir", "/x/o"
            ],
            "hote": "hote-synthetique", "debut_unix": debut, "fin_unix": debut + 10,
            "rc": 0, "binaire_sha256": "a".repeat(64), "source_hash": "b".repeat(64),
            "version": "test", "arch": "synthetique", "backend": "cuda",
            "versions": {"cuda": "test"}, "statut": "COMPLETE"
        });
        fs::write(
            self.0.join(format!("{id}.manifest.json")),
            serde_json::to_vec_pretty(&manifeste).unwrap(),
        )
        .unwrap();
    }

    fn rouge(&self, controle: u8) {
        let sortie = self.lancer(&["verifier"]);
        assert_eq!(sortie.status.code(), Some(8), "{sortie:?}");
        assert!(
            String::from_utf8_lossy(&sortie.stdout).contains(&format!("ROUGE V{controle}")),
            "{sortie:?}"
        );
    }
}

impl Drop for ExperienceTest {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn e_antidate_mutant_et_temoin_posterieur() {
    for (debut, rouge) in [(999_999_995, true), (1_000_000_005, false)] {
        let e = ExperienceTest::nouvelle();
        e.run("r0", 17, debut);
        e.ok(&["ajouter-run", "--machine", "principale", "r0.manifest.json"]);
        if rouge {
            let sortie = e.lancer(&["verifier"]);
            assert_eq!(sortie.status.code(), Some(8), "{sortie:?}");
            let texte = String::from_utf8_lossy(&sortie.stdout);
            assert!(
                texte.contains("ROUGE V4")
                    && texte.contains("r0")
                    && texte.contains("P0")
                    && texte.contains("5 secondes"),
                "{texte}"
            );
        } else {
            e.ok(&["verifier"]);
        }
    }
}

#[test]
fn e_sha_mutations_prereg_et_manifeste_run() {
    for fichier in ["PREREG.md", "r0.manifest.json"] {
        let e = ExperienceTest::nouvelle();
        e.run("r0", 19, 1_000_000_005);
        e.ok(&["ajouter-run", "--machine", "principale", "r0.manifest.json"]);
        e.ok(&["verifier"]);
        let p = e.0.join(fichier);
        let mut contenu = fs::read(&p).unwrap();
        let position = contenu.iter().position(|b| *b == b' ').unwrap();
        contenu[position] = b'\t';
        fs::write(p, contenu).unwrap();
        e.rouge(2);
    }
}

#[test]
fn graines_reutilisees_rouges_et_binaires_dedoublonnes() {
    let e = ExperienceTest::nouvelle();
    e.run("r0", 23, 1_000_000_005);
    e.run("r1", 23, 1_000_000_025);
    fs::write(e.0.join("autre.manifest.json"), "{\"schema\":\"autre/1\"}").unwrap();
    let sortie = e.ok(&["ajouter-run", "--machine", "principale", "."]);
    assert!(String::from_utf8_lossy(&sortie.stdout).contains("1 ignoré"));
    let v = e.lire();
    assert_eq!(v["graines"]["toutes_distinctes"], false);
    assert_eq!(v["code"]["binaires"].as_array().unwrap().len(), 1);
    e.rouge(6);
}

#[test]
fn machine_secondaire_classe_b_dans_le_manifeste_la_fiche_et_le_carnet() {
    let e = ExperienceTest::nouvelle();
    e.run("r0", 29, 1_000_000_005);
    e.ok(&["ajouter-run", "--machine", "secondaire", "r0.manifest.json"]);
    e.ok(&["verifier"]);
    let v = e.lire();
    assert_eq!(v["machines"][0]["classe"], "B");
    assert_eq!(v["verdict"]["classe"], "B");
    assert_eq!(v["runs"][0]["geometrie"]["cotes"], json!([8, 8, 8, 8]));
    assert_eq!(v["runs"][0]["parametres"]["beta"], json!(2.3));
    e.ok(&["fiche"]);
    assert!(fs::read_to_string(e.0.join("EXPERIENCE.md"))
        .unwrap()
        .contains("classe B"));
    e.ok(&["carnet", "--vers", "carnet"]);
    assert!(
        fs::read_to_string(e.0.join("carnet/experiences/EXP-temoin.md"))
            .unwrap()
            .contains("classe B")
    );
}

#[test]
fn rien_n_est_reecrit_et_verdict_archive_chaque_etat() {
    let e = ExperienceTest::nouvelle();
    let porte = [
        "porte",
        "G0",
        "--enonce",
        "Contrôle",
        "--seuil",
        "1",
        "--valeur",
        "0",
        "--statut",
        "vert",
        "--source",
        "PREREG.md",
    ];
    e.ok(&porte);
    let avant = e.octets();
    assert_eq!(e.lancer(&porte).status.code(), Some(8));
    assert_eq!(e.octets(), avant);
    let proposition = [
        "proposer",
        "resultat",
        "RES-temoin",
        "--titre",
        "Témoin",
        "--statut",
        "numerique",
        "--enonce",
        "Lecture synthétique",
        "--source",
        "PREREG.md:1-2",
    ];
    e.ok(&proposition);
    let avant = e.octets();
    assert_eq!(e.lancer(&proposition).status.code(), Some(8));
    assert_eq!(e.octets(), avant);
    e.ok(&["verdict", "--statut", "compatible", "--texte", "Premier"]);
    let premier = e.lire();
    assert_eq!(
        premier["verdict"]["historique"].as_array().unwrap().len(),
        1
    );
    assert_eq!(premier["verdict"]["historique"][0]["statut"], "en_cours");
    e.ok(&["verdict", "--statut", "indetermine", "--texte", "Second"]);
    let second = e.lire();
    assert_eq!(second["verdict"]["historique"].as_array().unwrap().len(), 2);
    assert_eq!(
        second["verdict"]["historique"][0],
        premier["verdict"]["historique"][0]
    );
    assert_eq!(second["verdict"]["historique"][1]["texte"], "Premier");
    e.ok(&["verifier"]);
}

#[test]
fn ajouter_run_identique_preserve_les_octets_meme_si_l_horloge_change() {
    let e = ExperienceTest::nouvelle();
    e.run("r0", 31, 1_000_000_005);
    let args = ["ajouter-run", "--machine", "principale", "r0.manifest.json"];
    e.ok(&args);
    let avant = e.octets();
    let sortie = e
        .commande(&args)
        .env("KVU_EXPERIENCE_HORLOGE", "2001-09-09T03:00:00Z")
        .output()
        .unwrap();
    assert_eq!(sortie.status.code(), Some(0), "{sortie:?}");
    assert_eq!(e.octets(), avant);
    e.run("r0", 32, 1_000_000_005);
    assert_eq!(e.lancer(&args).status.code(), Some(8));
    assert_eq!(e.octets(), avant);
}

#[test]
fn addendum_change_de_critere_exige_un_fichier_different() {
    let e = ExperienceTest::nouvelle();
    let avant = e.octets();
    let args = [
        "ajouter-prereg",
        "PREREG.md",
        "--change-un-critere",
        "--remplace",
        "P0",
        "--gele-le",
        "2001-09-09T01:46:41Z",
    ];
    assert_eq!(e.lancer(&args).status.code(), Some(8));
    assert_eq!(e.octets(), avant);
    fs::write(
        e.0.join("ADDENDUM.md"),
        "# Nouveau seuil\nUne obligation différente.\nFin.\n",
    )
    .unwrap();
    e.ok(&[
        "ajouter-prereg",
        "ADDENDUM.md",
        "--change-un-critere",
        "--remplace",
        "P0",
        "--gele-le",
        "2001-09-09T02:46:41+01:00",
        "--sections",
        "critere",
    ]);
    assert_eq!(e.lire()["prereg"][1]["gele_le"], "2001-09-09T01:46:41Z");
    e.ok(&["verifier"]);
}

#[test]
fn tous_les_prereg_cites_sont_controles_meme_dans_une_liste_variable() {
    let e = ExperienceTest::nouvelle();
    fs::write(e.0.join("P1.md"), "# Addendum\nCritère.\nFin.\n").unwrap();
    e.ok(&[
        "ajouter-prereg",
        "P1.md",
        "--gele-le",
        "2001-09-09T01:47:00Z",
    ]);
    e.run("r0", 37, 1_000_000_005);
    e.ok(&[
        "ajouter-run",
        "--machine",
        "principale",
        "--prereg",
        "P0",
        "P1",
        "r0.manifest.json",
    ]);
    assert_eq!(e.lire()["runs"][0]["prereg"], json!(["P0", "P1"]));
    e.rouge(4);
}

#[test]
fn gel_mtime_du_hachage_et_empreinte_de_la_consigne() {
    use std::time::{Duration, UNIX_EPOCH};

    let e = ExperienceTest::nouvelle();
    let sha = kvu::regen::manifeste::sha_fichier(&e.0.join("PREREG.md")).unwrap();
    fs::write(e.0.join("PREREG.sha256"), format!("{sha}  PREREG.md\n")).unwrap();
    let fichier = fs::OpenOptions::new()
        .write(true)
        .open(e.0.join("PREREG.sha256"))
        .unwrap();
    fichier
        .set_times(
            fs::FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(1_000_000_003)),
        )
        .unwrap();
    drop(fichier);
    fs::remove_file(e.0.join("EXPERIENCE.json")).unwrap();
    e.ok(&[
        "init",
        "--id",
        "temoin",
        "--titre",
        "Témoin",
        "--question",
        "Question",
        "--prereg",
        "PREREG.md",
        "--hachage-consigne",
        "PREREG.sha256",
    ]);
    let v = e.lire();
    assert_eq!(v["prereg"][0]["gele_le"], "2001-09-09T01:46:43Z");
    assert_eq!(v["prereg"][0]["gele_source"], "mtime");
    e.ok(&["verifier"]);
    fs::write(e.0.join("PREREG.sha256"), format!("{sha}  AUTRE.md\n")).unwrap();
    e.rouge(2);
}

#[test]
fn dates_de_run_avec_decalage_acceptees_et_reimport_identique() {
    let e = ExperienceTest::nouvelle();
    e.run("r0", 41, 1_000_000_005);
    let args = ["ajouter-run", "--machine", "principale", "r0.manifest.json"];
    e.ok(&args);
    let mut v = e.lire();
    v["runs"][0]["debut"] = json!("2001-09-09T02:46:45+01:00");
    v["runs"][0]["fin"] = json!("2001-09-09T02:46:55+01:00");
    fs::write(
        e.0.join("EXPERIENCE.json"),
        serde_json::to_vec_pretty(&v).unwrap(),
    )
    .unwrap();
    e.ok(&["verifier"]);
    let avant = e.octets();
    e.ok(&args);
    assert_eq!(e.octets(), avant);
    e.ok(&["fiche"]);
    let fiche = fs::read_to_string(e.0.join("EXPERIENCE.md")).unwrap();
    assert!(fiche.contains("2001-09-09T01:46:45Z"));
    assert!(!fiche.contains("+01:00"));
}

impl ExperienceTest {
    fn run_argv(&self, id: &str, argv: &[&str], debut: u64) {
        let mut tout = vec!["/x/kvu"];
        tout.extend_from_slice(argv);
        let manifeste = json!({
            "schema": "kevinotron-unifie/1", "argv": tout,
            "hote": "hote-synthetique", "debut_unix": debut, "fin_unix": debut + 10,
            "rc": 0, "binaire_sha256": "a".repeat(64), "source_hash": "b".repeat(64),
            "version": "test", "arch": "synthetique", "backend": "cuda",
            "versions": {"cuda": "test"}, "statut": "COMPLETE"
        });
        fs::write(
            self.0.join(format!("{id}.manifest.json")),
            serde_json::to_vec_pretty(&manifeste).unwrap(),
        )
        .unwrap();
    }

    fn sortie_texte(&self, args: &[&str]) -> (i32, String) {
        let s = self.lancer(args);
        (
            s.status.code().unwrap(),
            String::from_utf8_lossy(&s.stdout).into_owned(),
        )
    }
}

#[test]
fn o1_run_rattache_au_seul_addendum_mais_anterieur_a_la_racine_est_rouge() {
    let e = ExperienceTest::nouvelle();
    fs::write(e.0.join("ADD.md"), "# Addendum\nRien de plus.\n").unwrap();
    e.ok(&[
        "ajouter-prereg",
        "ADD.md",
        "--gele-le",
        "2001-09-09T01:46:41Z",
    ]);
    e.run("r0", 31, 999_999_995);
    e.ok(&[
        "ajouter-run",
        "--machine",
        "principale",
        "--prereg",
        "P1",
        "r0.manifest.json",
    ]);
    let (rc, texte) = e.sortie_texte(&["verifier"]);
    assert_eq!(rc, 8, "{texte}");
    assert!(
        texte.contains("ROUGE V4 run r0 avant le PREREG racine P0"),
        "{texte}"
    );
    // Témoin : le même run postérieur au gel de la racine et de l'addendum est vert.
    let t = ExperienceTest::nouvelle();
    fs::write(t.0.join("ADD.md"), "# Addendum\nRien de plus.\n").unwrap();
    t.ok(&[
        "ajouter-prereg",
        "ADD.md",
        "--gele-le",
        "2001-09-09T01:46:41Z",
    ]);
    t.run("r0", 31, 1_000_000_005);
    t.ok(&[
        "ajouter-run",
        "--machine",
        "principale",
        "--prereg",
        "P1",
        "r0.manifest.json",
    ]);
    t.ok(&["verifier"]);
}

#[test]
fn o1_addendum_gele_avant_la_racine_et_run_apres_critere_change_sans_le_citer() {
    let e = ExperienceTest::nouvelle();
    fs::write(e.0.join("ADD.md"), "# Addendum\nRien de plus.\n").unwrap();
    e.ok(&[
        "ajouter-prereg",
        "ADD.md",
        "--gele-le",
        "2001-09-09T01:00:00Z",
    ]);
    e.rouge(4);
    let e = ExperienceTest::nouvelle();
    fs::write(e.0.join("NOUVEAU.md"), "# Autre critère\nSeuil : 2.\n").unwrap();
    e.ok(&[
        "ajouter-prereg",
        "NOUVEAU.md",
        "--gele-le",
        "2001-09-09T01:50:00Z",
        "--change-un-critere",
        "--remplace",
        "P0",
    ]);
    e.run("r0", 32, 1_000_000_900);
    e.ok(&[
        "ajouter-run",
        "--machine",
        "principale",
        "--prereg",
        "P0",
        "r0.manifest.json",
    ]);
    let (rc, texte) = e.sortie_texte(&["verifier"]);
    assert_eq!(rc, 8, "{texte}");
    assert!(texte.contains("(critère changé) sans le citer"), "{texte}");
}

#[test]
fn o2_gel_declare_donne_un_avertissement_jamais_un_vert_plein() {
    let e = ExperienceTest::nouvelle();
    e.run("r0", 33, 1_000_000_005);
    e.ok(&["ajouter-run", "--machine", "principale", "r0.manifest.json"]);
    let (rc, texte) = e.sortie_texte(&["verifier"]);
    assert_eq!(rc, 0, "{texte}");
    assert!(
        texte.contains("AVERTISSEMENT V4 gel déclaré (non observé) pour P0"),
        "{texte}"
    );
    assert!(!texte.contains("OK V4"), "{texte}");
    e.ok(&["fiche"]);
    let fiche = fs::read_to_string(e.0.join("EXPERIENCE.md")).unwrap();
    assert!(
        fiche.contains("## Avertissements") && fiche.contains("gel déclaré"),
        "{fiche}"
    );
    // L'export du carnet reste possible et porte l'avertissement dans le corps de l'entrée.
    let vers = e.0.join("carnet");
    e.ok(&["carnet", "--vers", vers.to_str().unwrap()]);
    let entree = fs::read_to_string(vers.join("experiences/EXP-temoin.md")).unwrap();
    assert!(entree.contains("AVERTISSEMENT : gel déclaré"), "{entree}");
}

#[test]
fn o3_graine_effective_defaut_alias_et_chaines_de_lot() {
    // Deux runs su2 gen sans --graine : même graine par défaut (1) => rouge V6.
    let e = ExperienceTest::nouvelle();
    e.run_argv("a", &["su2", "gen", "--L", "8"], 1_000_000_005);
    e.run_argv("b", &["su2", "gen", "--L", "8"], 1_000_000_105);
    e.ok(&[
        "ajouter-run",
        "--machine",
        "principale",
        "a.manifest.json",
        "b.manifest.json",
    ]);
    assert_eq!(e.lire()["runs"][0]["graines"], json!([1]));
    e.rouge(6);
    // Autre commande sans graine lue : « graine non déclarée ».
    let e = ExperienceTest::nouvelle();
    e.run_argv("a", &["rg", "correlateur", "--L", "8"], 1_000_000_005);
    e.ok(&["ajouter-run", "--machine", "principale", "a.manifest.json"]);
    let (rc, texte) = e.sortie_texte(&["verifier"]);
    assert_eq!(rc, 8, "{texte}");
    assert!(texte.contains("graine non déclarée"), "{texte}");
    // Alias --seed lu ; chaînes d'un lot qui se recouvrent => rouge.
    let e = ExperienceTest::nouvelle();
    e.run_argv(
        "a",
        &[
            "su2",
            "gen",
            "--seed",
            "100",
            "--chaines",
            "2",
            "--graine-pas",
            "1",
        ],
        1_000_000_005,
    );
    e.run_argv(
        "b",
        &[
            "su2",
            "gen",
            "--graine",
            "101",
            "--chaines",
            "2",
            "--graine-pas",
            "1",
        ],
        1_000_000_105,
    );
    e.ok(&[
        "ajouter-run",
        "--machine",
        "principale",
        "a.manifest.json",
        "b.manifest.json",
    ]);
    assert_eq!(e.lire()["runs"][0]["graines"], json!([100, 101]));
    e.rouge(6);
}

#[test]
fn o4_ecritures_concurrentes_sans_perte() {
    let e = ExperienceTest::nouvelle();
    let enfants: Vec<_> = (0..8)
        .map(|i| {
            let mut c = e.commande(&[
                "porte",
                &format!("G{i}"),
                "--enonce",
                "e",
                "--seuil",
                "1",
                "--valeur",
                "2",
                "--statut",
                "vert",
                "--source",
                "PREREG.md",
            ]);
            std::thread::spawn(move || c.output().unwrap())
        })
        .collect();
    for h in enfants {
        let s = h.join().unwrap();
        assert_eq!(s.status.code(), Some(0), "{s:?}");
    }
    assert_eq!(e.lire()["portes"].as_array().unwrap().len(), 8);
    e.ok(&["verifier"]);
}

#[test]
fn o5_historique_du_verdict_chaine_troncature_et_edition_rouges() {
    let e = ExperienceTest::nouvelle();
    for (i, t) in ["un", "deux", "trois"].iter().enumerate() {
        let _ = i;
        e.ok(&[
            "verdict",
            "--statut",
            "non_concluant",
            "--provisoire",
            "--texte",
            t,
        ]);
    }
    assert_eq!(
        e.lire()["verdict"]["historique"].as_array().unwrap().len(),
        3
    );
    e.ok(&["verifier"]);
    let original = e.octets();
    // Troncature de la dernière entrée.
    let mut v = e.lire();
    v["verdict"]["historique"].as_array_mut().unwrap().pop();
    fs::write(
        e.0.join("EXPERIENCE.json"),
        serde_json::to_vec_pretty(&v).unwrap(),
    )
    .unwrap();
    e.rouge(11);
    // Troncature de la première entrée.
    let mut v: Value = serde_json::from_slice(&original).unwrap();
    v["verdict"]["historique"].as_array_mut().unwrap().remove(0);
    fs::write(
        e.0.join("EXPERIENCE.json"),
        serde_json::to_vec_pretty(&v).unwrap(),
    )
    .unwrap();
    e.rouge(11);
    // Édition du texte d'une entrée du milieu.
    let mut v: Value = serde_json::from_slice(&original).unwrap();
    v["verdict"]["historique"][1]["texte"] = json!("réécrit");
    fs::write(
        e.0.join("EXPERIENCE.json"),
        serde_json::to_vec_pretty(&v).unwrap(),
    )
    .unwrap();
    e.rouge(11);
}

#[test]
fn r1_p0_cherche_par_identifiant_tableau_reordonne_ou_p0_absent_rouges() {
    // P0 gelé à 10 h (1_000_000_000 + ...), P1 gelé plus tôt ; run cité seulement P1.
    let e = ExperienceTest::nouvelle();
    fs::write(e.0.join("ADD.md"), "# Addendum\nRien de plus.\n").unwrap();
    e.ok(&[
        "ajouter-prereg",
        "ADD.md",
        "--gele-le",
        "2001-09-09T01:40:00Z",
    ]);
    e.rouge(4); // P1 gelé avant la racine
    let e = ExperienceTest::nouvelle();
    fs::write(e.0.join("ADD.md"), "# Addendum\nRien de plus.\n").unwrap();
    e.ok(&[
        "ajouter-prereg",
        "ADD.md",
        "--gele-le",
        "2001-09-09T01:46:41Z",
    ]);
    e.run("r0", 41, 1_000_000_005);
    e.ok(&[
        "ajouter-run",
        "--machine",
        "principale",
        "--prereg",
        "P1",
        "r0.manifest.json",
    ]);
    e.ok(&["verifier"]);
    // Mutant : tableau prereg réordonné [P1, P0] et run antidaté cité seulement P1 (via P1 gelé tôt).
    let mut v = e.lire();
    v["prereg"].as_array_mut().unwrap().reverse();
    v["prereg"][0]["gele_le"] = json!("2001-09-09T01:40:00Z");
    fs::write(
        e.0.join("EXPERIENCE.json"),
        serde_json::to_vec_pretty(&v).unwrap(),
    )
    .unwrap();
    e.rouge(4);
    // Mutant : P0 supprimé.
    let mut v = e.lire();
    v["prereg"]
        .as_array_mut()
        .unwrap()
        .retain(|p| p["id"] != "P0");
    fs::write(
        e.0.join("EXPERIENCE.json"),
        serde_json::to_vec_pretty(&v).unwrap(),
    )
    .unwrap();
    let (rc, texte) = e.sortie_texte(&["verifier"]);
    assert_eq!(rc, 8, "{texte}");
    assert!(texte.contains("PREREG racine P0 absent"), "{texte}");
}

#[test]
fn r2_chaines_ecrites_avec_zero_initial_bornes_et_incoherences_refusees() {
    let e = ExperienceTest::nouvelle();
    e.run_argv(
        "a",
        &[
            "su2",
            "gen",
            "--graine",
            "100",
            "--chaines",
            "03",
            "--graine-pas",
            "1",
        ],
        1_000_000_005,
    );
    e.ok(&["ajouter-run", "--machine", "principale", "a.manifest.json"]);
    assert_eq!(e.lire()["runs"][0]["graines"], json!([100, 101, 102]));
    for argv in [
        vec!["su2", "gen", "--graine", "1", "--chaines", "0"],
        vec![
            "su2",
            "gen",
            "--graine",
            "1",
            "--chaines",
            "18446744073709551615",
            "--graine-pas",
            "1",
        ],
        vec!["su2", "gen", "--graine", "1,2,3", "--chaines", "2"],
        vec![
            "su2",
            "gen",
            "--beta",
            "2.3,2.5",
            "--chaines",
            "3",
            "--graine",
            "1,2,3",
        ],
    ] {
        let e = ExperienceTest::nouvelle();
        e.run_argv("a", &argv, 1_000_000_005);
        let s = e.lancer(&["ajouter-run", "--machine", "principale", "a.manifest.json"]);
        assert!(s.status.code().is_some_and(|c| c != 0), "{argv:?} : {s:?}");
        assert!(
            !String::from_utf8_lossy(&s.stderr).contains("panicked"),
            "{s:?}"
        );
    }
}

#[test]
fn r3_erreur_a_corriger_proposee_verifiee_et_exportee() {
    let e = ExperienceTest::nouvelle();
    e.ok(&[
        "proposer",
        "erreur",
        "ERR-test-a-corriger",
        "--titre",
        "Erreur ouverte",
        "--statut",
        "a_corriger",
        "--enonce",
        "Une erreur pas encore corrigée.",
        "--source",
        "PREREG.md:1",
    ]);
    e.ok(&["verifier"]);
    let vers = e.0.join("carnet");
    e.ok(&["carnet", "--vers", vers.to_str().unwrap()]);
    assert!(vers.join("erreurs/ERR-test-a-corriger.md").exists());
}

#[test]
fn r4_r5_manifeste_sans_chaine_migre_et_date_decalee_stable() {
    let e = ExperienceTest::nouvelle();
    e.ok(&[
        "verdict",
        "--statut",
        "non_concluant",
        "--provisoire",
        "--texte",
        "un",
    ]);
    // Manifeste « antérieur au chaînage » : on retire chaine et precedent.
    let mut v = e.lire();
    v["verdict"].as_object_mut().unwrap().remove("chaine");
    for h in v["verdict"]["historique"].as_array_mut().unwrap() {
        h.as_object_mut().unwrap().remove("precedent");
    }
    fs::write(
        e.0.join("EXPERIENCE.json"),
        serde_json::to_vec_pretty(&v).unwrap(),
    )
    .unwrap();
    let (rc, texte) = e.sortie_texte(&["verifier"]);
    assert_eq!(rc, 0, "{texte}");
    assert!(
        texte.contains("AVERTISSEMENT V11 historique du verdict non chaîné"),
        "{texte}"
    );
    // Une écriture ordinaire le chaîne ; le verdict reste vérifiable.
    e.ok(&[
        "porte",
        "G0",
        "--enonce",
        "e",
        "--seuil",
        "1",
        "--valeur",
        "2",
        "--statut",
        "vert",
        "--source",
        "PREREG.md",
    ]);
    assert!(e.lire()["verdict"]["chaine"].is_string());
    let (rc, texte) = e.sortie_texte(&["verifier"]);
    assert_eq!(rc, 0, "{texte}");
    assert!(
        texte.contains("AVERTISSEMENT V11 historique du verdict chaîné par migration"),
        "{texte}"
    );
    // R5 : une date à décalage dans l'historique n'est pas réécrite par une écriture ordinaire.
    let mut v = e.lire();
    let h = &mut v["verdict"]["historique"][0];
    h["remplace_le"] = json!("2001-09-09T03:00:00+01:00");
    v["verdict"]["chaine"] = json!(kvu_empreinte(h));
    fs::write(
        e.0.join("EXPERIENCE.json"),
        serde_json::to_vec_pretty(&v).unwrap(),
    )
    .unwrap();
    e.ok(&["verifier"]);
    e.ok(&[
        "porte",
        "G1",
        "--enonce",
        "e",
        "--seuil",
        "1",
        "--valeur",
        "2",
        "--statut",
        "vert",
        "--source",
        "PREREG.md",
    ]);
    e.ok(&["verifier"]);
    assert_eq!(
        e.lire()["verdict"]["historique"][0]["remplace_le"],
        "2001-09-09T03:00:00+01:00"
    );
}

/// sha256 de la forme compacte à clés triées, comme `empreinte_json` du module.
fn kvu_empreinte(v: &Value) -> String {
    kvu::regen::manifeste::sha_octets(&serde_json::to_vec(v).unwrap())
}

#[test]
fn p3_r1_chainage_partiel_est_rouge_et_jamais_reecrit_en_silence() {
    let e = ExperienceTest::nouvelle();
    for t in ["un", "deux", "trois"] {
        e.ok(&[
            "verdict",
            "--statut",
            "non_concluant",
            "--provisoire",
            "--texte",
            t,
        ]);
    }
    e.ok(&["verifier"]);
    // Corruption : texte de la 2e entrée modifié ET verdict.chaine supprimée, liens conservés.
    let mut v = e.lire();
    v["verdict"]["historique"][1]["texte"] = json!("falsifié");
    v["verdict"].as_object_mut().unwrap().remove("chaine");
    fs::write(
        e.0.join("EXPERIENCE.json"),
        serde_json::to_vec_pretty(&v).unwrap(),
    )
    .unwrap();
    e.rouge(11);
    // Une écriture ordinaire ne doit pas « réparer » : V11 reste rouge.
    e.ok(&[
        "porte",
        "G0",
        "--enonce",
        "e",
        "--seuil",
        "1",
        "--valeur",
        "2",
        "--statut",
        "vert",
        "--source",
        "PREREG.md",
    ]);
    e.rouge(11);
}

#[test]
fn p3_r2_graines_a_la_frontiere_arithmetique_de_la_production() {
    let e = ExperienceTest::nouvelle();
    e.run_argv(
        "a",
        &[
            "su2",
            "gen",
            "--graine",
            "1",
            "--chaines",
            "3",
            "--graine-pas",
            "9223372036854775809",
        ],
        1_000_000_005,
    );
    e.ok(&["ajouter-run", "--machine", "principale", "a.manifest.json"]);
    assert_eq!(
        e.lire()["runs"][0]["graines"],
        json!([1u64, 9_223_372_036_854_775_810u64, 3u64])
    );
}

#[test]
fn p3_r3_verrou_fifo_refuse_sans_bloquer() {
    let e = ExperienceTest::nouvelle();
    let _ = fs::remove_file(e.0.join(".EXPERIENCE.lock"));
    let rc = Command::new("mkfifo")
        .arg(e.0.join(".EXPERIENCE.lock"))
        .status()
        .unwrap();
    assert!(rc.success());
    let mut enfant = e
        .commande(&[
            "porte",
            "G0",
            "--enonce",
            "e",
            "--seuil",
            "1",
            "--valeur",
            "2",
            "--statut",
            "vert",
            "--source",
            "PREREG.md",
        ])
        .spawn()
        .unwrap();
    let debut = std::time::Instant::now();
    let code = loop {
        if let Some(s) = enfant.try_wait().unwrap() {
            break s.code();
        }
        if debut.elapsed() > std::time::Duration::from_secs(5) {
            let _ = enfant.kill();
            let _ = enfant.wait();
            panic!("le verrou FIFO a bloqué la commande");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    assert_eq!(code, Some(8));
}

#[test]
fn p3_r4_avertissement_de_migration_dans_la_fiche_et_le_carnet() {
    let e = ExperienceTest::nouvelle();
    e.ok(&[
        "verdict",
        "--statut",
        "non_concluant",
        "--provisoire",
        "--texte",
        "un",
    ]);
    let mut v = e.lire();
    v["verdict"].as_object_mut().unwrap().remove("chaine");
    for h in v["verdict"]["historique"].as_array_mut().unwrap() {
        h.as_object_mut().unwrap().remove("precedent");
    }
    fs::write(
        e.0.join("EXPERIENCE.json"),
        serde_json::to_vec_pretty(&v).unwrap(),
    )
    .unwrap();
    e.ok(&["fiche"]);
    let fiche = fs::read_to_string(e.0.join("EXPERIENCE.md")).unwrap();
    assert!(
        fiche.contains("AVERTISSEMENT : historique du verdict"),
        "{fiche}"
    );
    let vers = e.0.join("carnet");
    e.ok(&["carnet", "--vers", vers.to_str().unwrap()]);
    let entree = fs::read_to_string(vers.join("experiences/EXP-temoin.md")).unwrap();
    assert!(
        entree.contains("AVERTISSEMENT : historique du verdict"),
        "{entree}"
    );
}

#[test]
fn p3_r5_borne_des_chaines_deduites_des_listes() {
    let e = ExperienceTest::nouvelle();
    let betas = vec!["2.3"; 1_000_001].join(",");
    e.run_argv(
        "a",
        &[
            "su2",
            "gen",
            "--beta",
            &betas,
            "--graine",
            "1",
            "--graine-pas",
            "1",
        ],
        1_000_000_005,
    );
    let s = e.lancer(&["ajouter-run", "--machine", "principale", "a.manifest.json"]);
    assert_eq!(s.status.code(), Some(8), "{s:?}");
    assert!(!String::from_utf8_lossy(&s.stderr).contains("panicked"));
}

#[test]
fn p3_r6_liste_de_graines_avec_espaces_comme_su2_gen() {
    let e = ExperienceTest::nouvelle();
    e.run_argv(
        "a",
        &["su2", "gen", "--graine", "100, 101", "--chaines", "2"],
        1_000_000_005,
    );
    e.ok(&["ajouter-run", "--machine", "principale", "a.manifest.json"]);
    assert_eq!(e.lire()["runs"][0]["graines"], json!([100, 101]));
}

#[test]
fn p3_r7_contention_avec_echeance_et_deux_init_concurrents() {
    let e = ExperienceTest::nouvelle();
    let verrou = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(e.0.join(".EXPERIENCE.lock"))
        .unwrap();
    verrou.lock().unwrap();
    let mut enfant = e
        .commande(&[
            "porte",
            "G1",
            "--enonce",
            "e",
            "--seuil",
            "1",
            "--valeur",
            "2",
            "--statut",
            "vert",
            "--source",
            "PREREG.md",
        ])
        .spawn()
        .unwrap();
    // Tant que le verrou est tenu l'enfant ne doit jamais se terminer (échéance de 3 s, sans crochet dans le binaire).
    for _ in 0..150 {
        assert!(
            enfant.try_wait().unwrap().is_none(),
            "l'écriture a franchi un verrou détenu"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    drop(verrou);
    assert_eq!(enfant.wait().unwrap().code(), Some(0));
    assert_eq!(e.lire()["portes"].as_array().unwrap().len(), 1);
    // Deux init concurrents sur une racine neuve : exactement un réussit, le manifeste reste valide.
    let d = std::env::temp_dir().join(format!("kvu-exp-init2-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir(&d).unwrap();
    fs::write(d.join("PREREG.md"), "# P\nSeuil.\nFin.\n").unwrap();
    let lancer = |d: std::path::PathBuf| {
        std::thread::spawn(move || {
            Command::new(env!("CARGO_BIN_EXE_kvu"))
                .args([
                    "experience",
                    "init",
                    "--id",
                    "x",
                    "--titre",
                    "t",
                    "--question",
                    "q",
                    "--prereg",
                    "PREREG.md",
                    "--gele-le",
                    "2001-09-09T01:46:40Z",
                    "--racine",
                ])
                .arg(&d)
                .env("KVU_EXPERIENCE_HORLOGE", "2001-09-09T02:00:00Z")
                .current_dir(&d)
                .output()
                .unwrap()
        })
    };
    let (a, b) = (lancer(d.clone()), lancer(d.clone()));
    let (a, b) = (a.join().unwrap(), b.join().unwrap());
    let codes = [a.status.code().unwrap(), b.status.code().unwrap()];
    assert!(codes.contains(&0) && codes.contains(&8), "{codes:?}");
    let _ = fs::remove_dir_all(&d);
}

#[test]
fn p4_s1_la_variable_de_rendez_vous_ne_modifie_plus_aucun_fichier() {
    let e = ExperienceTest::nouvelle();
    let avant = e.octets();
    let cible = e.0.join("EXPERIENCE.json");
    let mut c = e.commande(&["fiche"]);
    c.env("KVU_EXPERIENCE_AVANT_VERROU", &cible);
    let s = c.output().unwrap();
    assert_eq!(s.status.code(), Some(0), "{s:?}");
    assert_eq!(
        e.octets(),
        avant,
        "le manifeste a été modifié par une variable d'environnement"
    );
}

#[test]
fn p4_s2_verdict_apres_migration_reussit_et_l_avertissement_reste() {
    let e = ExperienceTest::nouvelle();
    // Ancien manifeste : historique vide, aucun champ de chaînage.
    let mut v = e.lire();
    v["verdict"].as_object_mut().unwrap().remove("chaine");
    fs::write(
        e.0.join("EXPERIENCE.json"),
        serde_json::to_vec_pretty(&v).unwrap(),
    )
    .unwrap();
    e.ok(&[
        "verdict",
        "--statut",
        "non_concluant",
        "--provisoire",
        "--texte",
        "x",
    ]);
    let apres = e.lire();
    assert!(apres["verdict"]["migre_le"].is_string());
    assert!(apres["verdict"]["historique"][0].get("migre_le").is_none());
    let (rc, texte) = e.sortie_texte(&["verifier"]);
    assert_eq!(rc, 0, "{texte}");
    assert!(
        texte.contains("OK V1 ")
            && texte.contains("AVERTISSEMENT V11 historique du verdict chaîné par migration"),
        "{texte}"
    );
    // Un second verdict reste possible et l'avertissement subsiste.
    e.ok(&[
        "verdict",
        "--statut",
        "non_concluant",
        "--provisoire",
        "--texte",
        "y",
    ]);
    let (rc, texte) = e.sortie_texte(&["verifier"]);
    assert_eq!(rc, 0, "{texte}");
    assert!(texte.contains("AVERTISSEMENT V11"), "{texte}");
}

#[test]
fn p5_horloge_imposee_laisse_une_trace_collante_et_un_avertissement() {
    let e = ExperienceTest::nouvelle(); // écrit sous KVU_EXPERIENCE_HORLOGE
    assert_eq!(e.lire()["horodatage"]["horloge_imposee"], json!(true));
    let (rc, texte) = e.sortie_texte(&["verifier"]);
    assert_eq!(rc, 0, "{texte}");
    assert!(
        texte.contains("AVERTISSEMENT V4 dates écrites sous horloge imposée"),
        "{texte}"
    );
    e.ok(&["fiche"]);
    let fiche = fs::read_to_string(e.0.join("EXPERIENCE.md")).unwrap();
    assert!(
        fiche.contains("AVERTISSEMENT : dates écrites sous horloge imposée"),
        "{fiche}"
    );
    let vers = e.0.join("carnet");
    e.ok(&["carnet", "--vers", vers.to_str().unwrap()]);
    let entree = fs::read_to_string(vers.join("experiences/EXP-temoin.md")).unwrap();
    assert!(
        entree.contains("AVERTISSEMENT : dates écrites sous horloge imposée"),
        "{entree}"
    );
    // Écriture ultérieure SANS la variable : le champ reste.
    let s = Command::new(env!("CARGO_BIN_EXE_kvu"))
        .args([
            "experience",
            "porte",
            "G0",
            "--enonce",
            "e",
            "--seuil",
            "1",
            "--valeur",
            "2",
            "--statut",
            "vert",
            "--source",
            "PREREG.md",
            "--racine",
        ])
        .arg(&e.0)
        .env_remove("KVU_EXPERIENCE_HORLOGE")
        .current_dir(&e.0)
        .output()
        .unwrap();
    assert_eq!(s.status.code(), Some(0), "{s:?}");
    assert_eq!(e.lire()["horodatage"]["horloge_imposee"], json!(true));
}

#[test]
fn p5_sans_horloge_imposee_aucun_champ_ni_avertissement() {
    let e = ExperienceTest::nouvelle();
    let racine = e.0.join("propre");
    fs::create_dir(&racine).unwrap();
    fs::write(racine.join("PREREG.md"), "# P\nSeuil.\nFin.\n").unwrap();
    let lancer = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_kvu"))
            .arg("experience")
            .args(args)
            .arg("--racine")
            .arg(&racine)
            .env_remove("KVU_EXPERIENCE_HORLOGE")
            .current_dir(&racine)
            .output()
            .unwrap()
    };
    let s = lancer(&[
        "init",
        "--id",
        "x",
        "--titre",
        "t",
        "--question",
        "q",
        "--prereg",
        "PREREG.md",
    ]);
    assert_eq!(s.status.code(), Some(0), "{s:?}");
    let v: Value =
        serde_json::from_slice(&fs::read(racine.join("EXPERIENCE.json")).unwrap()).unwrap();
    assert!(v["horodatage"].get("horloge_imposee").is_none());
    let s = lancer(&["verifier"]);
    assert!(
        !String::from_utf8_lossy(&s.stdout).contains("horloge imposée"),
        "{s:?}"
    );
}
