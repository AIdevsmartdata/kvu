//! Arbitrages D3 de WP2 : preuves de G-H11 et statuts non bloquants de la CLI, sans GPU.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const ROUGE: &str = "FAIL G-H11 18 chaînes, 6 cellules ; C1-C4 ROUGES : L8b3.5:C2_ok";
const STATISTIQUE: &str = "L8 b3.5: moy 0.7679147(86) C1 True C2 -3.28 C3 -1.21 C4 True";

fn racine_validation() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("validation")
}

fn environnement_propre(c: &mut Command) {
    // Ne pas hériter du mode rapide, de la machine distante ni des chemins de production.
    c.env_clear().envs(
        std::env::vars().filter(|(k, _)| !k.starts_with("KVU_") && !k.starts_with("KEVINOTRON_")),
    );
}

/// Le vrai orchestrateur, des scripts factices et un store entièrement jetable.
struct Arbre {
    dir: PathBuf,
}

impl Arbre {
    fn nouveau(nom: &str) -> Self {
        static N: AtomicUsize = AtomicUsize::new(0);
        let k = N.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "kvu_wp2_arbitrages_{nom}_{}_{k}",
            std::process::id()
        ));
        fs::create_dir(&dir).expect("répertoire temporaire unique");
        let a = Self { dir };
        for rel in ["validation/faux", "scratch", "preuves", "store/gh07"] {
            fs::create_dir_all(a.dir.join(rel)).unwrap();
        }
        fs::copy(
            racine_validation().join("gates_unifie.sh"),
            a.dir.join("validation/gates_unifie.sh"),
        )
        .unwrap();
        // Racine minimale du dépôt factice, comme dans cli_valide.rs.
        fs::write(a.dir.join("Cargo.toml"), "").unwrap();
        a
    }

    fn script(&self, rel: &str, contenu: &str) {
        let p = self.dir.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, format!("#!/usr/bin/env bash\n{contenu}\n")).unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
    }

    fn chaine(&self, rel: &str) -> PathBuf {
        self.dir.join("store/gh07").join(rel)
    }

    fn store_et_preuves(&self) {
        for (lot, n) in [("L6_hot", 6), ("L6_cold", 3), ("L8_hot", 6), ("L8_cold", 3)] {
            for i in 0..n {
                let rel = format!("{lot}/c{i:03}");
                let d = self.chaine(&rel);
                fs::create_dir_all(&d).unwrap();
                // Contenus distincts pour détecter aussi une confusion entre chaînes ou fichiers.
                fs::write(d.join("measurements.csv"), format!("mesures de {rel}\n")).unwrap();
                fs::write(d.join("therm.csv"), format!("thermalisation de {rel}\n")).unwrap();
                fs::write(d.join("COMPLETE"), "COMPLET\n").unwrap();
            }
        }

        let mut bit = String::new();
        let mut therm = String::new();
        for rel in ["L8_hot/c004", "L8_hot/c005", "L8_cold/c002"] {
            let h = sha256(&self.chaine(rel).join("measurements.csv"));
            bit.push_str(&format!(
                "{h}  /chemin/vers/projet/kevinotron_store/validation_unifie/gh07/{rel}/measurements.csv\n"
            ));
            for fichier in ["therm.csv", "measurements.csv"] {
                let h = sha256(&self.chaine(rel).join(fichier));
                therm.push_str(&format!("{h}  {rel}/{fichier}\n"));
            }
        }
        // Ces chemins n'existent pas : un filtrage trop large ferait refuser le cas documenté.
        bit.push_str(&format!(
            "{}  integ/gh11_check/hip_c1/measurements.csv\n{}  integ/gh11_check/hip_c2/measurements.csv\n",
            "0".repeat(64),
            "1".repeat(64)
        ));
        fs::write(self.dir.join("preuves/gh11_bit_identite.txt"), bit).unwrap();
        fs::write(self.dir.join("preuves/gh11_therm_wp2.txt"), therm).unwrap();
    }

    fn python(&self, stdout: &str, stderr: &str) {
        // Les arguments d'analyse_historique.py sont volontairement ignorés.
        self.script(
            "python_factice",
            &format!("printf '%s\\n' '{stdout}'\nprintf '%s\\n' '{stderr}' >&2"),
        );
    }

    fn gh11(&self) -> Output {
        let mut c = Command::new("bash");
        environnement_propre(&mut c);
        c.arg(racine_validation().join("historique/G-H11.sh"))
            .env("KVU_MACHINE", "acemagic")
            .env("KVU_SCRATCH", self.dir.join("scratch"))
            .env("KVU_STORE_VALID", self.dir.join("store"))
            .env("KVU_PREUVES", self.dir.join("preuves"))
            .env("KVU_PY", self.dir.join("python_factice"))
            .env(
                "KVU_RACINE",
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."),
            );
        c.output().expect("lancement du vrai G-H11.sh")
    }

    fn valide(&self, motif: &str) -> Output {
        let mut c = Command::new(env!("CARGO_BIN_EXE_kvu"));
        environnement_propre(&mut c);
        c.args([
            "valide",
            "--gate",
            motif,
            "--machines",
            "acemagic",
            "--json",
        ])
        .env("KVU_VALIDATION_DIR", self.dir.join("validation"))
        .env("KVU_MACHINE", "acemagic")
        // Le scratch et le verrou restent dans l'arbre jetable.
        .env("KVU_SCRATCH_BASE", self.dir.join("scratch"))
        .env("KVU_VERROU", self.dir.join("gpu.lock"))
        .env("KVU_VERROU_ATTENTE_S", "2");
        c.output().expect("lancement de kvu valide")
    }
}

impl Drop for Arbre {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn sha256(p: &Path) -> String {
    let o = Command::new("sha256sum")
        .arg(p)
        .output()
        .expect("sha256sum");
    assert!(o.status.success(), "{o:?}");
    String::from_utf8(o.stdout)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_string()
}

fn scenario_gh11(nom: &str) -> Arbre {
    let a = Arbre::nouveau(nom);
    a.store_et_preuves();
    a.python(ROUGE, STATISTIQUE);
    a
}

fn verdict(o: &Output) -> String {
    // Les gates historiques rendent toujours rc 0, même lorsqu'elles émettent FAIL.
    assert_eq!(o.status.code(), Some(0), "{o:?}");
    String::from_utf8_lossy(&o.stdout)
        .lines()
        .last()
        .unwrap_or_else(|| panic!("verdict absent : {o:?}"))
        .to_string()
}

fn refus(o: &Output) -> String {
    let ligne = verdict(o);
    assert!(ligne.starts_with("FAIL G-H11 "), "{o:?}");
    assert!(ligne.contains("ARBITRAGE D3 REFUSÉ"), "{o:?}");
    ligne
}

fn json(o: &Output) -> serde_json::Value {
    serde_json::from_slice(&o.stdout).unwrap_or_else(|e| panic!("JSON invalide ({e}) : {o:?}"))
}

fn gate<'a>(j: &'a serde_json::Value, nom: &str) -> &'a serde_json::Value {
    let lignes: Vec<_> = j["gates"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|g| g["nom"] == nom && g["machine"] == "acemagic")
        .collect();
    assert_eq!(lignes.len(), 1, "une seule ligne attendue pour {nom} : {j}");
    lignes[0]
}

#[test]
fn gh11_rouge_documente_si_les_preuves_sont_identiques() {
    let a = scenario_gh11("documente");
    let o = a.gh11();
    assert!(verdict(&o).starts_with("FAIL-DOCUMENTE G-H11 "), "{o:?}");
    assert_eq!(String::from_utf8_lossy(&o.stderr).trim(), STATISTIQUE);
}

#[test]
fn gh11_refuse_un_z_different() {
    let a = scenario_gh11("z_different");
    a.python(ROUGE, &STATISTIQUE.replace("-3.28", "-3.27"));
    refus(&a.gh11());
}

#[test]
fn gh11_refuse_un_octet_de_thermalisation_modifie() {
    let a = scenario_gh11("sha_different");
    let p = a.chaine("L8_hot/c005").join("therm.csv");
    // Mutation après le calcul des empreintes, sans changer la taille du fichier.
    let mut contenu = fs::read(&p).unwrap();
    contenu[0] ^= 1;
    fs::write(p, contenu).unwrap();
    let ligne = refus(&a.gh11());
    assert!(ligne.contains("sha256 différent"), "{ligne}");
    assert!(ligne.contains("L8_hot/c005/therm.csv"), "{ligne}");
}

#[test]
fn gh11_refuse_un_autre_critere_rouge() {
    let a = scenario_gh11("autre_rouge");
    a.python(
        "FAIL G-H11 18 chaînes, 6 cellules ; C1-C4 ROUGES : L6b2.3:C1_ok, L8b3.5:C2_ok",
        STATISTIQUE,
    );
    refus(&a.gh11());
}

#[test]
fn gh11_transmet_un_pass_tel_quel() {
    let a = scenario_gh11("pass");
    let pass = "PASS G-H11 18 chaînes, 6 cellules ; C1-C4 tous verts";
    a.python(pass, "");
    assert_eq!(verdict(&a.gh11()), pass);
}

#[test]
fn gh11_refuse_moins_de_dix_huit_chaines_completes() {
    let a = scenario_gh11("incomplet");
    // Les données restent présentes, mais cette chaîne ne compte plus comme complète.
    fs::remove_file(a.chaine("L6_hot/c000").join("COMPLETE")).unwrap();
    let o = a.gh11();
    let ligne = verdict(&o);
    assert!(ligne.starts_with("FAIL G-H11 "), "{o:?}");
    assert!(ligne.contains("absent ou incomplet"), "{o:?}");
    assert!(
        !a.dir.join("scratch/gh11.out").exists(),
        "l'analyse ne doit pas être lancée"
    );
}

#[test]
fn cli_info_et_fail_documente_sont_listes_sans_bloquer() {
    let a = Arbre::nouveau("cli_statuts");
    a.script(
        "validation/faux/gates_d3.sh",
        "echo 'INFO OBS-4-L8 rapportée 15/20'\n\
         echo 'FAIL-DOCUMENTE G-H11 fait statistique'\n\
         echo 'PASS OBS-1 ok'\n\
         echo 'PASS OBS-4-L16 ok'",
    );

    let o = a.valide("OBS-4");
    assert_eq!(o.status.code(), Some(0), "{o:?}");
    let j = json(&o);
    assert_eq!(j["gates"].as_array().unwrap().len(), 2, "{j}");
    assert_eq!(gate(&j, "OBS-4-L8")["statut"], "INFO", "{j}");
    assert_eq!(gate(&j, "OBS-4-L8")["detail"], "rapportée 15/20", "{j}");
    assert_eq!(gate(&j, "OBS-4-L16")["statut"], "PASS", "{j}");
    assert_eq!(j["fail"], 0, "{j}");
    assert_eq!(j["skip"], 0, "{j}");
    assert_eq!(j["info"], 1, "{j}");
    assert_eq!(j["fail_documente"], 0, "{j}");

    let o = a.valide("G-H11");
    assert_eq!(o.status.code(), Some(0), "{o:?}");
    let j = json(&o);
    assert_eq!(j["gates"].as_array().unwrap().len(), 1, "{j}");
    assert_eq!(gate(&j, "G-H11")["statut"], "FAIL-DOCUMENTE", "{j}");
    assert_eq!(gate(&j, "G-H11")["detail"], "fait statistique", "{j}");
    assert_eq!(j["fail_documente"], 1, "{j}");
    assert_eq!(j["fail"], 0, "{j}");
    assert_eq!(j["skip"], 0, "{j}");
    assert_eq!(j["info"], 0, "{j}");
}

#[test]
fn cli_info_l8_ne_couvre_pas_la_gate_l16_manquante() {
    let a = Arbre::nouveau("cli_l16_absente");
    a.script(
        "validation/faux/gates_d3.sh",
        "echo 'INFO OBS-4-L8 rapportée 15/20'",
    );
    let o = a.valide("OBS-4");
    assert!(o.status.code().is_some_and(|rc| rc != 0), "{o:?}");
    let j = json(&o);
    assert_eq!(j["gates"].as_array().unwrap().len(), 2, "{j}");
    assert_eq!(gate(&j, "OBS-4-L8")["statut"], "INFO", "{j}");
    let manquante = gate(&j, "OBS-4-L16");
    assert_eq!(manquante["statut"], "FAIL", "{j}");
    let detail = manquante["detail"].as_str().unwrap();
    assert!(
        detail.contains("gate attendue") && detail.contains("non émise"),
        "{j}"
    );
    assert_eq!(j["fail"], 1, "{j}");
    assert_eq!(j["skip"], 0, "{j}");
    assert_eq!(j["info"], 1, "{j}");
    assert_eq!(j["fail_documente"], 0, "{j}");
}
