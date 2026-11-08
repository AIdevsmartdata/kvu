//! Tests de `kevinotron valide` / `kvu valide` avec des scripts de gates FACTICES : le contrat
//! (lignes PASS|FAIL|SKIP, rc, JSON gelé, SKIP non vert, gates attendues manquantes, verrou GPU
//! hérité d'un ancêtre, partie gamer par ssh) est vérifié sans GPU ni gamer.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::Instant;

fn kvu() -> &'static str {
    env!("CARGO_BIN_EXE_kvu")
}

fn racine_validation() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("validation")
}

/// Arbre de validation temporaire : le VRAI gates_unifie.sh + des scripts factices.
struct Arbre {
    dir: PathBuf,
}

impl Arbre {
    fn nouveau(nom: &str) -> Arbre {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let k = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("kvu_cli_valide_{nom}_{}_{k}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("validation/faux")).unwrap();
        fs::copy(
            racine_validation().join("gates_unifie.sh"),
            dir.join("validation/gates_unifie.sh"),
        )
        .unwrap();
        // racine « dépôt » minimale pour racine_depuis_script (non exigée ici : KVU_VALIDATION_DIR prime)
        fs::write(dir.join("Cargo.toml"), "").unwrap();
        Arbre { dir }
    }

    fn script(&self, rel: &str, contenu: &str) {
        let p = self.dir.join("validation").join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, format!("#!/usr/bin/env bash\n{contenu}\n")).unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
    }

    fn valide(&self, args: &[&str]) -> Output {
        self.valide_env(args, &[])
    }

    fn valide_env(&self, args: &[&str], env: &[(&str, &str)]) -> Output {
        let mut c = Command::new(kvu());
        // environnement propre : ni KVU_* ni KEVINOTRON_* hérités (la suite peut lancer ces tests elle-même, sur le gamer en
        // particulier avec KVU_DISTANT=1)
        c.env_clear()
            .envs(std::env::vars().filter(|(k, _)| !k.starts_with("KVU_") && !k.starts_with("KEVINOTRON_")));
        c.arg("valide")
            .args(args)
            .env("KVU_VALIDATION_DIR", self.dir.join("validation"))
            .env("KVU_MACHINE", "acemagic")
            // jamais le verrou GPU de production dans un test : verrou jetable, attente courte
            .env("KVU_VERROU", self.dir.join("gpu.lock"))
            .env("KVU_VERROU_ATTENTE_S", "2");
        for (k, v) in env {
            c.env(k, v);
        }
        c.output().expect("lancement de kvu valide")
    }
}

impl Drop for Arbre {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn json(o: &Output) -> serde_json::Value {
    let s = String::from_utf8_lossy(&o.stdout);
    serde_json::from_str(s.trim()).unwrap_or_else(|e| {
        panic!(
            "JSON invalide ({e}) : {s}\nstderr : {}",
            String::from_utf8_lossy(&o.stderr)
        )
    })
}

#[test]
fn gate_verte_rc_0_et_json_gele() {
    let a = Arbre::nouveau("verte");
    a.script("faux/gates_a.sh", "echo 'PASS G-H01 binaire = source'");
    let o = a.valide(&["--gate", "G-H01", "--machines", "acemagic", "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    let j = json(&o);
    assert_eq!(j["fail"], 0);
    assert_eq!(j["skip"], 0);
    assert_eq!(j["gates"][0]["nom"], "G-H01");
    assert_eq!(j["gates"][0]["machine"], "acemagic");
    assert_eq!(j["gates"][0]["statut"], "PASS");
    assert_eq!(j["gates"][0]["detail"], "binaire = source");
    println!("PASS cli_valide_gate_verte");
}

#[test]
fn un_fail_rend_8() {
    let a = Arbre::nouveau("rouge");
    a.script("faux/gates_a.sh", "echo 'FAIL G-H01 sha différent'");
    let o = a.valide(&["--gate", "G-H01", "--machines", "acemagic", "--json"]);
    assert_eq!(o.status.code(), Some(8));
    let j = json(&o);
    assert_eq!(j["fail"], 1);
    assert_eq!(j["gates"][0]["statut"], "FAIL");
}

#[test]
fn un_skip_nest_pas_vert() {
    let a = Arbre::nouveau("skip");
    a.script("faux/gates_a.sh", "echo 'SKIP G-H01 pas de GPU'");
    let o = a.valide(&["--gate", "G-H01", "--machines", "acemagic", "--json"]);
    assert_eq!(o.status.code(), Some(8), "un SKIP ne doit pas donner rc 0");
    let j = json(&o);
    assert_eq!((j["fail"].as_u64(), j["skip"].as_u64()), (Some(0), Some(1)));
}

#[test]
fn une_gate_attendue_non_emise_est_un_fail() {
    let a = Arbre::nouveau("manquante");
    a.script("faux/gates_a.sh", "echo 'PASS G-H02 ok'");
    // G-H01 est demandée mais le script n'émet que G-H02 : le motif G-H01 filtre l'émission, la gate manque.
    let o = a.valide(&["--gate", "G-H01", "--machines", "acemagic", "--json"]);
    assert_eq!(o.status.code(), Some(8));
    let j = json(&o);
    assert_eq!(j["fail"], 1);
    assert!(j["gates"][0]["detail"].as_str().unwrap().contains("non émise"), "{j}");
}

#[test]
fn script_en_echec_sans_fail_est_signale() {
    let a = Arbre::nouveau("rc");
    a.script("faux/gates_a.sh", "echo 'PASS G-H01 ok'; exit 3");
    let o = a.valide(&["--gate", "G-H01", "--machines", "acemagic", "--json"]);
    assert_eq!(o.status.code(), Some(8));
    let j = json(&o);
    assert!(
        j["gates"]
            .as_array()
            .unwrap()
            .iter()
            .any(|g| g["statut"] == "FAIL" && g["detail"].as_str().unwrap().contains("rc = 3")),
        "{j}"
    );
}

#[test]
fn lignes_libtest_et_gates_normalisees_acceptees() {
    let a = Arbre::nouveau("libtest");
    a.script(
        "faux/gates_a.sh",
        "echo 'test g1 ... PASS G-H01 depuis libtest'\nprintf 'GATE\\tPASS\\tacemagic\\tG-H02\\tdéjà normalisée\\n'\necho 'bruit sans statut'",
    );
    let o = a.valide(&["--machines", "acemagic", "--json", "--gate", "G-H0"]);
    let j = json(&o);
    let noms: Vec<_> = j["gates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["nom"].as_str().unwrap().to_string())
        .collect();
    assert!(
        noms.contains(&"G-H01".to_string()) && noms.contains(&"G-H02".to_string()),
        "{noms:?}"
    );
    assert!(
        !String::from_utf8_lossy(&o.stdout).contains("bruit sans statut"),
        "le bruit doit rester sur stderr"
    );
    assert!(String::from_utf8_lossy(&o.stderr).contains("bruit sans statut"));
}

#[test]
fn le_tableau_humain_donne_les_totaux() {
    let a = Arbre::nouveau("tableau");
    a.script("faux/gates_a.sh", "echo 'PASS G-H01 ok'");
    let o = a.valide(&["--gate", "G-H01", "--machines", "acemagic"]);
    let s = String::from_utf8_lossy(&o.stdout);
    assert!(s.contains("1 PASS, 0 FAIL, 0 SKIP"), "{s}");
    assert_eq!(o.status.code(), Some(0));
}

#[test]
fn dry_run_ne_lance_rien() {
    let a = Arbre::nouveau("dry");
    a.script(
        "faux/gates_a.sh",
        "touch \"$KVU_RACINE/NE_DOIT_PAS_EXISTER\"; echo 'PASS G-H01 ok'",
    );
    let o = a.valide(&["--rapide", "--dry-run"]);
    assert_eq!(o.status.code(), Some(0));
    let s = String::from_utf8_lossy(&o.stdout);
    assert!(s.contains("plan de validation") && s.contains("KVU_MODE=rapide"), "{s}");
    assert!(!a.dir.join("NE_DOIT_PAS_EXISTER").exists());
}

#[test]
fn options_exclusives_rc_2() {
    let a = Arbre::nouveau("usage");
    let o = a.valide(&["--rapide", "--complet"]);
    assert_eq!(o.status.code(), Some(2));
    assert_eq!(a.valide(&["--machines", "pc"]).status.code(), Some(2));
}

#[test]
fn un_flock_externe_tenu_par_un_ancetre_ne_bloque_pas() {
    // La suite est lancée SOUS flock ; le script de gates prend le même verrou : sans le faux flock,
    // il attendrait le délai (3 s) puis échouerait.
    let a = Arbre::nouveau("flock");
    let verrou = a.dir.join("gpu.lock");
    fs::write(&verrou, "").unwrap();
    a.script(
        "faux/gates_a.sh",
        &format!(
            "if flock -w 3 {0} true; then echo 'PASS G-H01 verrou hérité'; else echo 'FAIL G-H01 interblocage'; fi",
            verrou.display()
        ),
    );
    let t = Instant::now();
    let o = Command::new("flock")
        .env_remove("KVU_DISTANT")
        .env_remove("KVU_MACHINES")
        .env_remove("KVU_SANS_GPU")
        .env_remove("KVU_GROUPE")
        .env_remove("KVU_MODE")
        .args(["-w", "10"])
        .arg(&verrou)
        .arg(kvu())
        .args(["valide", "--gate", "G-H01", "--machines", "acemagic", "--json"])
        .env("KVU_VALIDATION_DIR", a.dir.join("validation"))
        .env("KVU_MACHINE", "acemagic")
        .env("KVU_VERROU", &verrou)
        .env("KVU_VERROU_ATTENTE_S", "5")
        .output()
        .expect("flock");
    let j = json(&o);
    assert_eq!(j["gates"][0]["statut"], "PASS", "{j}");
    assert!(t.elapsed().as_secs_f64() < 20.0, "attente anormale : {:?}", t.elapsed());
}

#[test]
fn un_flock_tenu_par_un_autre_processus_bloque_vraiment() {
    // Contre-épreuve : le faux flock ne doit PAS court-circuiter un verrou tenu par un étranger.
    let a = Arbre::nouveau("etranger");
    let verrou = a.dir.join("gpu.lock");
    fs::write(&verrou, "").unwrap();
    a.script("faux/gates_a.sh", "echo 'PASS G-H01 exécuté malgré le verrou étranger'");
    let mut etranger = Command::new("flock")
        .arg(&verrou)
        .args(["sleep", "6"])
        .spawn()
        .expect("flock étranger");
    std::thread::sleep(std::time::Duration::from_millis(400));
    let o = a.valide(&["--gate", "G-H01", "--machines", "acemagic", "--json"]);
    let _ = etranger.kill();
    let _ = etranger.wait();
    let j = json(&o);
    // le script n'a pas pu tourner (l'orchestrateur attend le verrou 2 s puis abandonne) ou son flock interne a échoué :
    // dans les deux cas la suite est ROUGE, jamais verte.
    assert_ne!(
        o.status.code(),
        Some(0),
        "le verrou d'un étranger doit être respecté : {j}"
    );
    assert!(j["fail"].as_u64().unwrap() >= 1, "{j}");
}

fn faux_binaires(a: &Arbre, ssh: &str) -> String {
    let bin = a.dir.join("bin");
    fs::create_dir_all(&bin).unwrap();
    for (nom, corps) in [("ssh", ssh), ("rsync", "exit 0")] {
        let p = bin.join(nom);
        fs::write(&p, format!("#!/usr/bin/env bash\n{corps}\n")).unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
    }
    format!("{}:{}", bin.display(), std::env::var("PATH").unwrap_or_default())
}

#[test]
fn la_partie_gamer_passe_par_ssh() {
    let a = Arbre::nouveau("gamer");
    let chemin = faux_binaires(
        &a,
        "case \"$*\" in *gates_unifie.sh*) printf 'GATE\\tPASS\\tgamer\\tG-H06\\t18 chaînes\\n';; esac\nexit 0",
    );
    let o = a.valide_env(
        &["--gate", "G-H06", "--machines", "gamer", "--json"],
        &[("PATH", &chemin)],
    );
    let j = json(&o);
    assert_eq!(o.status.code(), Some(0), "{j}\n{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(j["gates"][0]["machine"], "gamer");
    assert_eq!(j["gates"][0]["nom"], "G-H06");
}

#[test]
fn gamer_injoignable_est_un_fail() {
    let a = Arbre::nouveau("gamer_ko");
    let chemin = faux_binaires(&a, "exit 255");
    let o = a.valide_env(
        &["--gate", "G-H06", "--machines", "gamer", "--json"],
        &[("PATH", &chemin)],
    );
    assert_eq!(o.status.code(), Some(8));
    assert!(json(&o)["fail"].as_u64().unwrap() >= 1);
}

#[test]
fn les_gates_rapportees_report_sont_lues() {
    let a = Arbre::nouveau("report");
    a.script(
        "faux/gates_a.sh",
        "echo 'REPORT OBS-4 : |Q − round(Q)| < 0,1 pour 19/20 configurations (95 %, critère souple ≥ 90 %) : atteint'\n\
         echo 'REPORT OBS-5 : fraction 40 % : NON atteint'",
    );
    let o = a.valide(&["--gate", "OBS-", "--machines", "acemagic", "--json"]);
    let j = json(&o);
    let g = |nom: &str| {
        j["gates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["nom"] == nom)
            .cloned()
            .unwrap_or_else(|| panic!("{nom} absente : {j}"))
    };
    assert_eq!(g("OBS-4")["statut"], "PASS");
    assert!(g("OBS-4")["detail"].as_str().unwrap().starts_with("(rapportée)"));
    assert_eq!(
        g("OBS-5")["statut"],
        "FAIL",
        "« NON atteint » doit rougir la gate rapportée"
    );
}

#[test]
fn un_script_sans_declaration_tourne_sous_le_verrou_de_la_machine() {
    let a = Arbre::nouveau("sous_verrou");
    let verrou = a.dir.join("gpu.lock");
    // pendant le script, le verrou est tenu (un flock -n étranger échoue : on le teste par un processus fils indépendant)
    a.script(
        "faux/gates_a.sh",
        &format!(
            "if /usr/bin/flock -n {0} true; then echo 'FAIL G-H01 verrou non tenu pendant le script'; else echo 'PASS G-H01 verrou tenu'; fi",
            verrou.display()
        ),
    );
    let o = a.valide(&["--gate", "G-H01", "--machines", "acemagic", "--json"]);
    assert_eq!(
        json(&o)["gates"][0]["statut"],
        "PASS",
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    // après la suite, le verrou est rendu
    let r = Command::new("/usr/bin/flock")
        .args(["-n"])
        .arg(&verrou)
        .arg("true")
        .status()
        .unwrap();
    assert!(r.success(), "le verrou doit être rendu à la fin du script");
}

#[test]
fn un_script_qui_prend_lui_meme_son_verrou_nest_pas_enveloppe() {
    let a = Arbre::nouveau("autonome");
    let verrou = a.dir.join("gpu.lock");
    // déclaration textuelle : le script gère seul sa fenêtre ; ici il vérifie que le verrou n'est PAS tenu à l'entrée
    a.script(
        "faux/gates_a.sh",
        &format!(
            "# Ce script prend lui-même le verrou par fenêtre.\n\
             if /usr/bin/flock -n {0} true; then echo 'PASS G-H01 verrou libre à l entrée'; else echo 'FAIL G-H01 verrou déjà tenu'; fi",
            verrou.display()
        ),
    );
    let o = a.valide(&["--gate", "G-H01", "--machines", "acemagic", "--json"]);
    assert_eq!(
        json(&o)["gates"][0]["statut"],
        "PASS",
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
}

#[test]
fn le_mode_rapide_ne_prend_aucun_verrou_et_ne_lance_que_les_scripts_cpu() {
    let a = Arbre::nouveau("rapide_sans_verrou");
    let verrou = a.dir.join("gpu.lock");
    // script CPU de historique/ : tourne, sans verrou
    a.script(
        "historique/G-H01.sh",
        &format!(
            "if /usr/bin/flock -n {0} true; then echo 'PASS G-H01 aucun verrou'; else echo 'FAIL G-H01 verrou pris en rapide'; fi",
            verrou.display()
        ),
    );
    // script d'une tâche (suppose le verrou tenu par l'appelant) : NE DOIT PAS tourner en rapide
    a.script(
        "faux/gates_a.sh",
        "touch \"$KVU_RACINE/GPU_LANCE\"; echo 'PASS MACH-1 ne doit pas apparaître'",
    );
    let o = a.valide(&["--rapide", "--machines", "acemagic", "--json"]);
    let j = json(&o);
    let gates = j["gates"].as_array().unwrap();
    assert!(
        gates.iter().any(|g| g["nom"] == "G-H01" && g["statut"] == "PASS"),
        "{j}"
    );
    assert!(
        !a.dir.join("GPU_LANCE").exists(),
        "un script de tâche a tourné en mode rapide, sans verrou"
    );
    assert!(
        !gates.iter().any(|g| g["nom"] == "MACH-1" && g["statut"] == "PASS"),
        "{j}"
    );
}

#[test]
fn le_gamer_tourne_en_parallele_et_les_gates_dependantes_passent_apres() {
    let a = Arbre::nouveau("parallele");
    // ssh factice : 1 s de calcul distant, puis la gate du gamer
    let chemin = faux_binaires(
        &a,
        "case \"$*\" in *gates_unifie.sh*) sleep 1; printf 'GATE\\tPASS\\tgamer\\tG-H06\\tfini\\n';; esac\nexit 0",
    );
    a.script("faux/gates_a.sh", "echo 'PASS G-H07 indépendante'");
    // G-H09 est « dépendante » (analyse les données du gamer) : elle doit venir APRÈS la gate G-H06 du gamer
    a.script("historique/G-H09.sh", "echo 'PASS G-H09 dépendante'");
    let t = Instant::now();
    let o = a.valide_env(
        &["--gate", "G-H0", "--machines", "acemagic,gamer", "--json"],
        &[("PATH", &chemin)],
    );
    let j = json(&o);
    let noms: Vec<String> = j["gates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["nom"].as_str().unwrap().to_string())
        .collect();
    let pos = |n: &str| {
        noms.iter()
            .position(|x| x == n)
            .unwrap_or_else(|| panic!("{n} absente : {noms:?}"))
    };
    assert!(
        pos("G-H07") < pos("G-H06"),
        "l'indépendante tourne pendant que le gamer calcule : {noms:?}"
    );
    assert!(
        pos("G-H06") < pos("G-H09"),
        "la dépendante vient après le gamer : {noms:?}"
    );
    assert!(t.elapsed().as_secs_f64() < 15.0);
}

#[test]
fn les_gates_gpu_historiques_partagent_une_seule_fenetre_de_verrou() {
    let a = Arbre::nouveau("groupe");
    let verrou = a.dir.join("gpu.lock");
    let v = verrou.display();
    // G-H03 (hors groupe) : le verrou est libre ; G-H04 et G-H05 : tenu, et leur flock interne passe (faux flock) ;
    // G-H09 (dépendante, hors groupe) : libre de nouveau.
    a.script(
        "historique/G-H03.sh",
        &format!(
            "if /usr/bin/flock -n {v} true; then echo 'PASS G-H03 libre avant'; else echo 'FAIL G-H03 tenu avant'; fi"
        ),
    );
    for n in ["04", "05"] {
        a.script(
            &format!("historique/G-H{n}.sh"),
            &format!(
                "if /usr/bin/flock -n {v} true; then echo 'FAIL G-H{n} verrou non tenu'; \
                 elif flock -w 2 {v} true; then echo 'PASS G-H{n} fenêtre de groupe, flock interne transparent'; \
                 else echo 'FAIL G-H{n} interblocage'; fi"
            ),
        );
    }
    a.script(
        "historique/G-H09.sh",
        &format!(
            "if /usr/bin/flock -n {v} true; then echo 'PASS G-H09 libre après'; else echo 'FAIL G-H09 tenu après'; fi"
        ),
    );
    let o = a.valide(&["--gate", "G-H0", "--machines", "acemagic", "--json"]);
    let j = json(&o);
    let st = |n: &str| {
        j["gates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["nom"] == n)
            .map(|g| g["statut"].as_str().unwrap().to_string())
    };
    for n in ["G-H03", "G-H04", "G-H05", "G-H09"] {
        assert_eq!(
            st(n).as_deref(),
            Some("PASS"),
            "{n} : {j}\n{}",
            String::from_utf8_lossy(&o.stderr)
        );
    }
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("fenêtre de groupe rendue"),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
}

#[test]
fn sur_le_gamer_les_scripts_recoivent_leurs_options_cuda_et_stockage_est_ecarte() {
    let a = Arbre::nouveau("gamer_options");
    a.script("obs/gates_obs.sh", "echo \"PASS OBS-1 args=[$*]\"");
    a.script("cond/gates_cond.sh", "echo \"PASS EQC-1 features=[${KVU_FEATURES:-}]\"");
    a.script(
        "stockage/gates_stockage.sh",
        "echo 'PASS PROV-1 ne doit pas tourner sur le gamer'",
    );
    // on se place sur le gamer (suite locale du gamer : KVU_DISTANT=1, KVU_MACHINE=gamer)
    let o = a.valide_env(
        &["--gate", "-", "--machines", "gamer", "--json"],
        &[("KVU_DISTANT", "1"), ("KVU_MACHINE", "gamer")],
    );
    let j = json(&o);
    let detail = |n: &str| {
        j["gates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["nom"] == n)
            .map(|g| g["detail"].as_str().unwrap().to_string())
    };
    assert_eq!(detail("OBS-1").as_deref(), Some("args=[--cuda]"), "{j}");
    assert_eq!(detail("EQC-1").as_deref(), Some("features=[cuda]"), "{j}");
    assert!(
        detail("PROV-1").is_none()
            || j["gates"]
                .as_array()
                .unwrap()
                .iter()
                .all(|g| g["nom"] != "PROV-1" || g["statut"] == "FAIL")
    );
    assert!(
        !String::from_utf8_lossy(&o.stdout).contains("ne doit pas tourner"),
        "{j}"
    );
}
