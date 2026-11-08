//! Régressions des défauts relevés par le vérificateur de l'intégration (04/10/2026), au niveau du binaire `kvu`
//! (mêmes chemins que `kevinotron <sous-commande>`). Tout tourne sur le CPU, à L = 4, sans toucher au GPU
//! (backend `cpu` forcé ; `doctor --json` sonde les GPU sans contexte). Les tests de logique pure de chaque
//! correctif sont dans leurs modules (processus, admission, file, garde, etat_source, torsion…).
//!
//! | défaut | test |
//! |---|---|
//! | affinité 12-19 seulement par la file | `affinite_posee_par_le_binaire` |
//! | aucune admission mémoire des runs directs | `admission_memoire_refuse_avant_tout` |
//! | KEVINOTRON_MACHINE incohérente ou invalide | `surcharge_de_machine_incoherente_refusee` |
//! | garde SSD incomplète (obs flot, fibre, obs wilson vide) | `garde_ssd_refuse_sans_rien_creer` |
//! | pas de provenance pour obs, fibre, cond sweep | `provenance_des_sorties` |
//! | job orphelin après SIGKILL du worker | `job_orphelin_apres_sigkill_du_worker` |
//! | temps mur de su2 gen absent du banc | `banc_su2_publie_le_temps_mur` |
//! | banc cond sur un champ non thermalisé | `banc_cond_part_d_un_champ_thermalise` |
//! | .pyc commité | `aucun_bytecode_python_suivi_par_git` |
//! | ordre q64 mal documenté | `q64_scalaire_en_premier` |

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

const KVU: &str = env!("CARGO_BIN_EXE_kvu");

/// Répertoire de travail du test (hors /tmp, qui est en RAM sur acemagic), supprimé à la fin.
struct Scratch(PathBuf);

impl Scratch {
    fn nouveau(nom: &str) -> Scratch {
        let base = option_env!("CARGO_TARGET_TMPDIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
        let d = base.join(format!("kvu_correctifs_{nom}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
    }
    fn p(&self, rel: &str) -> PathBuf {
        self.0.join(rel)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `kvu args…` avec un environnement propre (variables KEVINOTRON_* de la session retirées) plus `env`.
fn kvu(args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut c = Command::new(KVU);
    c.args(args).stdin(Stdio::null());
    for k in [
        "KEVINOTRON_MACHINE",
        "KEVINOTRON_BACKEND",
        "KEVINOTRON_CPUS",
        "KEVINOTRON_RAM_MIN_GO",
        "KEVINOTRON_SSD_MIN_GO",
        "KEVINOTRON_FILE",
        "KEVINOTRON_STORE",
        "KEVINOTRON_VERROU",
    ] {
        c.env_remove(k);
    }
    for (k, v) in env {
        c.env(k, v);
    }
    c.output().expect("lancement de kvu")
}

fn rc(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn json_de(o: &Output) -> Value {
    let t = String::from_utf8_lossy(&o.stdout);
    t.lines()
        .find_map(|l| l.find('{').and_then(|i| serde_json::from_str::<Value>(&l[i..]).ok()))
        .unwrap_or_else(|| panic!("aucun JSON dans la sortie :\n{t}\n{}", err(o)))
}

fn cfg_l4() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/cond/cfg_L4_haar0.npy")
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Les commandes de mesure et de génération à L = 4 sur le CPU, avec leur sortie dans `d`.
fn commandes_l4(d: &Scratch) -> Vec<(&'static str, Vec<String>, PathBuf)> {
    let c = s(&cfg_l4());
    let v = |a: &[&str]| a.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    vec![
        ("obs wilson", v(&["obs", "wilson", "--cfg", &c, "--backend", "cpu", "--rmax", "2", "--tmax", "2", "--out", &s(&d.p("wilson"))]), d.p("wilson")),
        ("obs polyakov", v(&["obs", "polyakov", "--cfg", &c, "--backend", "cpu", "--out", &s(&d.p("polyakov"))]), d.p("polyakov")),
        ("obs flot", v(&["obs", "flot", "--cfg", &c, "--backend", "cpu", "--tmax", "0.05", "--out", &s(&d.p("flot"))]), d.p("flot")),
        ("fibre mesurer", v(&["fibre", "mesurer", "--cfg", &c, "--backend", "cpu", "--out", &s(&d.p("fibre"))]), d.p("fibre")),
        ("cond sweep", v(&["cond", "sweep", "--cfg", &c, "--beta", "2.3", "--graine", "1", "--backend", "cpu", "--out", &s(&d.p("sweep.npy"))]), d.p("sweep.npy")),
        ("cond tirage", v(&["cond", "tirage", "--alpha", "3", "--n", "1000", "--backend", "cpu", "--out", &s(&d.p("x0.npy"))]), d.p("x0.npy")),
        ("su2 gen", v(&["su2", "gen", "--L", "4", "--therm", "2", "--nmeas", "2", "--backend", "cpu", "--outdir", &s(&d.p("gen"))]), d.p("gen")),
        ("cond chaine", v(&["cond", "chaine", "--cfg", &c, "--beta", "2.3", "--graine", "1", "--ntherm", "1", "--nsweep", "4", "--backend", "cpu", "--sans-journal", "--out", &s(&d.p("chaine"))]), d.p("chaine")),
    ]
}

// ---------------------------------------------------------------------------------------------
// ÉLEVÉ — affinité
// ---------------------------------------------------------------------------------------------

/// Le binaire pose LUI-MÊME l'affinité de la politique (et la restriction KEVINOTRON_CPUS) : sur le gamer, un
/// appel direct tourne sur 12-19, jamais sur les cœurs P.
#[test]
fn affinite_posee_par_le_binaire() {
    let o = kvu(&["doctor", "--json"], &[]);
    let v = json_de(&o);
    let processus = v["affinite"]["processus"].as_str().expect("affinité du processus").to_string();
    if let Some(pol) = v["affinite"]["politique"].as_str() {
        assert_eq!(processus, pol, "affinité obligatoire {pol} non posée (processus : {processus})");
    }
    if v["machine"] == "gamer" {
        assert_eq!(processus, "12-19", "gamer : cœurs E seulement");
    }
    // restriction explicite à un seul CPU autorisé : appliquée au processus
    let premier = processus.split([',', '-']).next().unwrap().to_string();
    let v = json_de(&kvu(&["doctor", "--json"], &[("KEVINOTRON_CPUS", &premier)]));
    assert_eq!(v["affinite"]["processus"], premier.as_str());
    // les fils de calcul héritent : un run CPU lancé sous restriction ne déborde pas
    let d = Scratch::nouveau("affinite");
    let mut enfant = Command::new(KVU)
        .args(["cond", "tirage", "--alpha", "3", "--n", "4000000", "--backend", "cpu", "--out", &s(&d.p("x.npy"))])
        .env("KEVINOTRON_CPUS", &premier)
        .env("KEVINOTRON_SSD_MIN_GO", "0")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let pid = enfant.id();
    let mut vus = 0;
    let fin = Instant::now() + Duration::from_secs(20);
    // le fil principal pose l'affinité avant de créer le moindre fil : on observe une fois qu'elle est posée
    let pose = || {
        std::fs::read_to_string(format!("/proc/{pid}/status"))
            .ok()
            .and_then(|t| t.lines().find_map(|l| l.strip_prefix("Cpus_allowed_list:")).map(|v| v.trim() == premier))
            .unwrap_or(false)
    };
    while Instant::now() < fin {
        if !pose() {
            if enfant.try_wait().unwrap().is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }
        if let Ok(taches) = std::fs::read_dir(format!("/proc/{pid}/task")) {
            for t in taches.flatten() {
                if let Ok(st) = std::fs::read_to_string(t.path().join("status")) {
                    if let Some(l) = st.lines().find_map(|l| l.strip_prefix("Cpus_allowed_list:")) {
                        assert_eq!(l.trim(), premier, "fil {:?} hors de l'affinité", t.file_name());
                        vus += 1;
                    }
                }
            }
        }
        if enfant.try_wait().unwrap().is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let _ = enfant.wait();
    assert!(vus > 0, "aucun fil observé");
}

// ---------------------------------------------------------------------------------------------
// ÉLEVÉ — admission mémoire
// ---------------------------------------------------------------------------------------------

/// Plancher de RAM impossible ⇒ chaque run direct refuse (rc 5) AVANT de créer quoi que ce soit.
#[test]
fn admission_memoire_refuse_avant_tout() {
    let d = Scratch::nouveau("admission");
    for (nom, args, sortie) in commandes_l4(&d) {
        if nom == "obs polyakov" || nom == "obs flot" {
            continue; // même chemin que obs wilson
        }
        let a: Vec<&str> = args.iter().map(String::as_str).collect();
        let o = kvu(&a, &[("KEVINOTRON_RAM_MIN_GO", "100000"), ("KEVINOTRON_SSD_MIN_GO", "0")]);
        assert_eq!(rc(&o), 5, "{nom} : rc {} au lieu de 5 :\n{}", rc(&o), err(&o));
        assert!(err(&o).contains("ressources refusées") && err(&o).contains("plancher"), "{nom} : {}", err(&o));
        assert!(!sortie.exists(), "{nom} : {} créé malgré le refus", sortie.display());
    }
}

// ---------------------------------------------------------------------------------------------
// MOYEN-ÉLEVÉ — surcharge de machine
// ---------------------------------------------------------------------------------------------

#[test]
fn surcharge_de_machine_incoherente_refusee() {
    let reelle = json_de(&kvu(&["etat", "--json"], &[]))["machine"].as_str().unwrap().to_string();
    // valeur invalide : refus (avant le 04/10 elle était ignorée sans erreur)
    let o = kvu(&["etat", "--json"], &[("KEVINOTRON_MACHINE", "xyz")]);
    assert_eq!(rc(&o), 2, "{}", err(&o));
    assert!(err(&o).contains("xyz"));
    // l'autre machine connue : refus (le verrou ne suivrait plus le GPU réellement utilisé)
    let autre = match reelle.as_str() {
        "acemagic" => Some("gamer"),
        "gamer" => Some("acemagic"),
        _ => None,
    };
    if let Some(autre) = autre {
        for cmd in [&["etat", "--json"][..], &["doctor", "--json"][..], &["su2", "selftest"][..]] {
            let o = kvu(cmd, &[("KEVINOTRON_MACHINE", autre)]);
            assert_eq!(rc(&o), 2, "{cmd:?} sous KEVINOTRON_MACHINE={autre} : rc {}", rc(&o));
        }
        // « inconnue » reste admise, mais le verrou reste celui du matériel
        let attendu = json_de(&kvu(&["doctor", "--json"], &[]))["verrou_gpu"]["chemin"].clone();
        let v = json_de(&kvu(&["doctor", "--json"], &[("KEVINOTRON_MACHINE", "inconnue")]));
        assert_eq!(v["machine"], "inconnue");
        assert_eq!(v["verrou_gpu"]["chemin"], attendu, "le verrou doit suivre le GPU réel");
    }
}

// ---------------------------------------------------------------------------------------------
// MOYEN — garde SSD
// ---------------------------------------------------------------------------------------------

/// Disque « plein » simulé : chaque commande rend 7 et ne crée ni répertoire ni fichier.
#[test]
fn garde_ssd_refuse_sans_rien_creer() {
    let d = Scratch::nouveau("ssd");
    for (nom, args, sortie) in commandes_l4(&d) {
        let a: Vec<&str> = args.iter().map(String::as_str).collect();
        let o = kvu(&a, &[("KEVINOTRON_SSD_MIN_GO", "100000")]);
        assert_eq!(rc(&o), 7, "{nom} : rc {} au lieu de 7 :\n{}", rc(&o), err(&o));
        assert!(!sortie.exists(), "{nom} : {} créé malgré le refus", sortie.display());
    }
    assert_eq!(std::fs::read_dir(&d.0).unwrap().count(), 0, "rien ne doit avoir été écrit");
}

// ---------------------------------------------------------------------------------------------
// MOYEN — provenance
// ---------------------------------------------------------------------------------------------

fn verifier_manifeste(nom: &str, m: &Value, dir: &Path) {
    for cle in ["binaire_sha256", "source_hash", "git_head", "machine", "noyaux_embarques", "politique"] {
        assert!(!m[cle].is_null(), "{nom} : champ {cle} absent du manifeste");
    }
    assert_eq!(m["backend"], "cpu", "{nom} : backend effectif");
    assert_eq!(m["schema"], "kevinotron-unifie/1");
    assert!(m["noyaux_embarques"].as_array().is_some_and(|v| !v.is_empty()));
    let fichiers = m["fichiers"].as_array().expect("fichiers");
    assert!(!fichiers.is_empty(), "{nom} : aucun fichier consigné");
    for f in fichiers {
        let p = dir.join(f["nom"].as_str().unwrap());
        let b = std::fs::read(&p).unwrap_or_else(|_| panic!("{nom} : {} absent", p.display()));
        assert_eq!(f["sha256"].as_str().unwrap(), kvu::heritage::su2_hb::sha256_hex(&b), "{nom} : {}", p.display());
    }
}

#[test]
fn provenance_des_sorties() {
    let d = Scratch::nouveau("provenance");
    for (nom, args, sortie) in commandes_l4(&d) {
        if !matches!(nom, "obs wilson" | "obs polyakov" | "obs flot" | "fibre mesurer" | "cond sweep") {
            continue;
        }
        let a: Vec<&str> = args.iter().map(String::as_str).collect();
        let o = kvu(&a, &[("KEVINOTRON_SSD_MIN_GO", "0")]);
        assert_eq!(rc(&o), 0, "{nom} : {}", err(&o));
        if nom == "cond sweep" {
            let m: Value =
                serde_json::from_str(&std::fs::read_to_string(d.p("sweep.npy.manifest.json")).unwrap()).unwrap();
            verifier_manifeste(nom, &m, &d.0);
            assert_eq!(m["statut"], "termine");
            assert!(m["entree"]["sha256"].as_str().is_some_and(|x| x.len() == 64));
            assert!(m["parametres"]["beta"].is_number());
        } else {
            let m: Value =
                serde_json::from_str(&std::fs::read_to_string(sortie.join("manifest.json")).unwrap()).unwrap();
            verifier_manifeste(nom, &m, &sortie);
            assert!(sortie.join("COMPLETE").exists(), "{nom} : COMPLETE absent");
            assert_eq!(m["statut"], "termine");
            let sha_entree = if nom.starts_with("obs") {
                m["entrees"][0]["sha256"].clone()
            } else {
                m["mesure"]["entree"]["sha256"].clone()
            };
            assert!(sha_entree.as_str().is_some_and(|x| x.len() == 64), "{nom} : sha256 de l'entrée");
        }
    }
    // un répertoire de sortie existant est refusé (répertoire de run NEUF) : rien n'est écrasé
    let o = kvu(
        &["obs", "wilson", "--cfg", &s(&cfg_l4()), "--backend", "cpu", "--out", &s(&d.p("wilson"))],
        &[("KEVINOTRON_SSD_MIN_GO", "0")],
    );
    assert_eq!(rc(&o), 2, "{}", err(&o));
}

// ---------------------------------------------------------------------------------------------
// MOYEN — file de jobs : worker tué par SIGKILL
// ---------------------------------------------------------------------------------------------

fn etat_file(env: &[(&str, &str)]) -> Value {
    json_de(&kvu(&["etat", "--json"], env))["file"].clone()
}

fn attendre(quoi: &str, mut cond: impl FnMut() -> bool) {
    let fin = Instant::now() + Duration::from_secs(30);
    while Instant::now() < fin {
        if cond() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("délai dépassé : {quoi}");
}

#[test]
fn job_orphelin_apres_sigkill_du_worker() {
    let d = Scratch::nouveau("orphelin");
    let file = s(&d.p("file"));
    let env = [
        ("KEVINOTRON_FILE", file.as_str()),
        ("KEVINOTRON_MACHINE", "inconnue"),
        ("KEVINOTRON_SSD_MIN_GO", "0"),
        ("KEVINOTRON_RAM_MIN_GO", "0"),
    ];
    let o = kvu(&["queue", "add", "--repertoire", &s(&d.0), "--", "sleep", "60"], &env);
    assert_eq!(rc(&o), 0, "{}", err(&o));
    let id = String::from_utf8_lossy(&o.stdout).trim().to_string();
    // un worker prend le job, puis est tué par SIGKILL
    let mut worker = Command::new(KVU);
    worker.args(["queue", "run", "--une"]).stdout(Stdio::null()).stderr(Stdio::null());
    for (k, v) in env {
        worker.env(k, v);
    }
    let mut worker = worker.spawn().unwrap();
    let en_cours = d.p(&format!("file/en_cours/{id}.json"));
    let mut pid_job = 0u32;
    attendre("le job a un pid", || {
        let Ok(t) = std::fs::read_to_string(&en_cours) else { return false };
        let v: Value = serde_json::from_str(&t).unwrap_or(Value::Null);
        pid_job = v["pid"].as_u64().unwrap_or(0) as u32;
        pid_job != 0 && !v["worker_pid"].is_null() && !v["boot_id"].is_null()
    });
    worker.kill().unwrap(); // SIGKILL
    worker.wait().unwrap();
    // le processus du job (lancé par CE test) vit encore : le job est en cours, pas orphelin
    let f = etat_file(&env);
    assert_eq!((f["en_cours"].as_u64(), f["orphelins"].as_u64()), (Some(1), Some(0)), "{f}");
    // fin du processus du job : plus personne pour le terminer ⇒ orphelin, et plus « en cours »
    Command::new("kill").args(["-9", &pid_job.to_string()]).status().unwrap();
    attendre("le processus du job disparaît", || !Path::new(&format!("/proc/{pid_job}")).exists());
    let f = etat_file(&env);
    assert_eq!((f["en_cours"].as_u64(), f["orphelins"].as_u64()), (Some(0), Some(1)), "{f}");
    let liste = String::from_utf8_lossy(&kvu(&["queue", "list"], &env).stdout).into_owned();
    assert!(liste.contains("orphelin"), "{liste}");
    // un nouveau worker range l'orphelin en échec (rien n'est relancé)
    let o = kvu(&["queue", "run", "--une"], &env);
    assert_eq!(rc(&o), 0, "{}", err(&o));
    let echec = d.p(&format!("file/echec/{id}.json"));
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&echec).expect("rangé en échec")).unwrap();
    assert_eq!(v["orphelin"], true);
    assert!(v["rc"].is_null());
    let f = etat_file(&env);
    assert_eq!((f["en_cours"].as_u64(), f["orphelins"].as_u64(), f["echec"].as_u64()), (Some(0), Some(0), Some(1)));
}

// ---------------------------------------------------------------------------------------------
// MOYEN — bancs
// ---------------------------------------------------------------------------------------------

#[test]
fn banc_su2_publie_le_temps_mur() {
    let o = kvu(
        &["bench", "su2", "--backend", "cpu", "--L", "4", "--precision", "f64", "--chaines", "1", "--budget-s", "60",
          "--cible-s", "2", "--ref-bin", "aucune", "--json"],
        &[("KEVINOTRON_SSD_MIN_GO", "0")],
    );
    assert_eq!(rc(&o), 0, "{}", err(&o));
    let v = json_de(&o);
    let m = &v["mesures"][0];
    let (noyaux, mur) = (m["pas_par_s"].as_f64().unwrap(), m["mur"]["pas_par_s"].as_f64().expect("temps mur publié"));
    assert!(mur > 0.0 && mur.is_finite());
    assert!(m["mur"]["pas_mesures"].as_u64().unwrap() >= 30);
    // le temps mur ne peut pas dépasser nettement le débit des seuls noyaux (bruit de mesure toléré)
    assert!(mur <= 1.5 * noyaux, "mur {mur} pas/s contre noyaux {noyaux}");
    assert!(m["rapport_mur_sur_noyaux"].as_f64().is_some());
}

#[test]
fn banc_cond_part_d_un_champ_thermalise() {
    let o = kvu(
        &["bench", "cond", "--backend", "cpu", "--L", "4", "--mode", "rapide", "--therm", "30", "--budget-s", "60",
          "--cible-s", "1", "--json"],
        &[("KEVINOTRON_SSD_MIN_GO", "0")],
    );
    assert_eq!(rc(&o), 0, "{}", err(&o));
    let m = json_de(&o)["mesures"][0].clone();
    assert!(m["depart"].as_str().unwrap().contains("30 pas"), "{m}");
    // thermalisé à β = 2,3 : plaquette nettement au-dessus de Haar (≈ 0)
    assert!(m["plaquette_depart"].as_f64().unwrap() > 0.3, "{m}");
    // avec une configuration explicite
    let o = kvu(
        &["bench", "cond", "--backend", "cpu", "--L", "4", "--mode", "rapide", "--cfg", &s(&cfg_l4()), "--budget-s",
          "60", "--cible-s", "1", "--json"],
        &[("KEVINOTRON_SSD_MIN_GO", "0")],
    );
    assert_eq!(rc(&o), 0, "{}", err(&o));
    assert!(json_de(&o)["mesures"][0]["depart"].as_str().unwrap().contains("cfg_L4_haar0"));
}

// ---------------------------------------------------------------------------------------------
// BAS
// ---------------------------------------------------------------------------------------------

#[test]
fn aucun_bytecode_python_suivi_par_git() {
    let racine = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let Ok(o) = Command::new("git").arg("-C").arg(&racine).args(["ls-files"]).output() else {
        println!("SKIP : git absent");
        return;
    };
    if !o.status.success() {
        println!("SKIP : copie hors git");
        return;
    }
    let suivis: Vec<String> = String::from_utf8_lossy(&o.stdout)
        .lines()
        .filter(|l| l.ends_with(".pyc") || l.contains("__pycache__/"))
        .map(str::to_string)
        .collect();
    assert!(suivis.is_empty(), "bytecode Python suivi par git : {suivis:?}");
}

/// q64 = quaternions (a0, a1, a2, a3), SCALAIRE EN PREMIER : l'identité s'écrit (1, 0, 0, 0).
#[test]
fn q64_scalaire_en_premier() {
    let d = Scratch::nouveau("q64");
    let froid = kvu::commun::ChampHote::froid(2).unwrap();
    let p = d.p("froid.npy");
    kvu::stockage::npy::ecrire_champ(&p, &froid, kvu::stockage::FormatChamp::Q64, &kvu::stockage::GardeSsd::nouvelle(0.0))
        .unwrap();
    let o = std::fs::read(&p).unwrap();
    let entete = 10 + u16::from_le_bytes([o[8], o[9]]) as usize;
    let q: Vec<f64> = o[entete..entete + 32].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(q, vec![1.0, 0.0, 0.0, 0.0]);
}
