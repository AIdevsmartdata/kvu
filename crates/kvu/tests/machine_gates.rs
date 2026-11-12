//! Gates MACH-1, VERROU-1 et FILE-1 (ARCHITECTURE_UNIFIEE.md §6.5), tâche T1.
//!
//! Elles s'exécutent sur la machine réelle, SOUS son verrou GPU
//! (`flock -w 7200 /tmp/kevinotron_gpu.lock bash validation/machine/gates_machine.sh`), et
//! impriment `PASS <nom>` ou `FAIL <nom>` (`SKIP` si la machine est inconnue : ne vaut pas vert).
//! Les scénarios multi-processus relancent CE binaire de test sur un test « aide_* » (activé par
//! la variable KVU_AIDE), sans binaire séparé. Aucune des gates ne touche au verrou de production :
//! elles prennent un verrou temporaire (sauf le contrôle final de VERROU-1, en lecture seule).

use kvu::commun::Machine;
use kvu::machine::cli::{etat_json, principal_machine};
use kvu::machine::detection::{detecter_rapide, ProfilMachine};
use kvu::machine::file::*;
use kvu::machine::politique::{politique, politique_avec, Politique};
use kvu::machine::verrou::VerrouGpu;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// Aides multi-processus
// ---------------------------------------------------------------------------------------------

/// Aide : `principal_machine(KVU_AIDE_ARGS)` puis sortie avec son code (le code du « binaire »).
#[test]
fn aide_doctor() {
    let Ok(args) = std::env::var("KVU_AIDE_ARGS") else {
        return;
    };
    let argv: Vec<String> = args.split_whitespace().map(str::to_string).collect();
    std::process::exit(principal_machine(&argv));
}

/// Aide : acquiert KVU_AIDE_VERROU ; sort avec le code d'erreur typé (6 si le délai est dépassé).
#[test]
fn aide_acquerir() {
    let Ok(chemin) = std::env::var("KVU_AIDE_VERROU") else {
        return;
    };
    let attente = std::env::var("KVU_AIDE_ATTENTE_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let debut = Instant::now();
    match VerrouGpu::nouveau(Path::new(&chemin)).acquerir(Duration::from_millis(attente)) {
        Ok(g) => {
            println!("AIDE:OK mode={} ms={}", g.mode(), debut.elapsed().as_millis());
            if let Ok(tenir) = std::env::var("KVU_AIDE_TENIR_MS") {
                let t0 = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();
                std::thread::sleep(Duration::from_millis(tenir.parse().unwrap_or(0)));
                let t1 = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();
                if let Ok(trace) = std::env::var("KVU_AIDE_TRACE") {
                    use std::io::Write;
                    let mut f = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(trace)
                        .unwrap();
                    writeln!(f, "{t0} {t1}").unwrap();
                }
            }
        }
        Err(e) => {
            println!("AIDE:ERR rc={} ms={} {e}", e.code_sortie(), debut.elapsed().as_millis());
            std::process::exit(e.code_sortie());
        }
    }
}

fn politique_libre(racine: &Path) -> Politique {
    let p = ProfilMachine {
        machine: Machine::Inconnue,
        hote: "h".into(),
        cpu_modele: "c".into(),
        fils_logiques: 8,
        ram_totale_go: 1.0,
        ram_disponible_go: 1.0,
        gpus: vec![],
        versions: vec![],
        surcharge_refusee: None,
    };
    let mut pol = politique_avec(&p, &|_| None, Path::new("/tmp"));
    pol.ram_min_go = 0.0;
    pol.seuil_ssd_go = 0.0;
    pol.alerte_moniteur = None;
    pol.mem_max = None;
    pol.file = racine.to_path_buf();
    pol
}

/// Aide : un worker qui prend UN job de la file KVU_AIDE_FILE et l'exécute.
#[test]
fn aide_worker() {
    let Ok(dossier) = std::env::var("KVU_AIDE_FILE") else {
        return;
    };
    let pol = politique_libre(Path::new(&dossier));
    let f = File::ouvrir(Path::new(&dossier)).unwrap();
    // Départ synchronisé : chaque worker attend l'instant commun KVU_AIDE_DEPART (s Unix).
    if let Some(t) = std::env::var("KVU_AIDE_DEPART")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
    {
        while maintenant_unix() < t {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    match f.prendre_suivant(&pol).unwrap() {
        Some(j) => {
            let rc = f.executer(j, &pol).unwrap();
            println!("AIDE:WORKER pris=1 rc={rc}");
        }
        None => println!("AIDE:WORKER pris=0"),
    }
}

// ---------------------------------------------------------------------------------------------
// Utilitaires de gate
// ---------------------------------------------------------------------------------------------

fn gate(nom: &str, corps: impl FnOnce() -> Result<String, String>) {
    match corps() {
        Ok(detail) if detail.starts_with("SKIP") => {
            println!("SKIP {nom} : {}", detail.trim_start_matches("SKIP ").trim())
        }
        Ok(detail) => println!("PASS {nom} : {detail}"),
        Err(raison) => {
            println!("FAIL {nom} : {raison}");
            panic!("FAIL {nom} : {raison}");
        }
    }
}

fn attendre(ok: bool, msg: impl FnOnce() -> String) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(msg())
    }
}

fn dossier_temp(nom: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("kvu_gate_{nom}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Relance ce binaire de test sur `aide` avec l'environnement donné (variables KEVINOTRON_* de la
/// session retirées d'abord : la gate pose les siennes).
fn relancer(aide: &str, env: &[(&str, String)]) -> std::process::Command {
    let mut c = Command::new(std::env::current_exe().unwrap());
    c.args(["--exact", aide, "--nocapture", "--test-threads=1"])
        .stdin(Stdio::null());
    for k in [
        "KEVINOTRON_BACKEND",
        "KEVINOTRON_MACHINE",
        "KEVINOTRON_VERROU",
        "KEVINOTRON_FILE",
        "KEVINOTRON_SSD_MIN_GO",
    ] {
        c.env_remove(k);
    }
    for (k, v) in env {
        c.env(k, v);
    }
    c
}

fn ligne_aide(sortie: &[u8]) -> String {
    String::from_utf8_lossy(sortie)
        .lines()
        .find_map(|l| l.find("AIDE:").map(|i| l[i..].to_string()))
        .unwrap_or_default()
}

/// Exécute `doctor --json` dans un sous-processus : (rc, JSON).
fn doctor(env: &[(&str, String)]) -> Result<(i32, Value), String> {
    let mut e = vec![("KVU_AIDE_ARGS", "doctor --json".to_string())];
    e.extend(env.iter().map(|(k, v)| (*k, v.clone())));
    let s = relancer("aide_doctor", &e).output().map_err(|e| e.to_string())?;
    let texte = String::from_utf8_lossy(&s.stdout);
    let json = texte
        .lines()
        .find_map(|l| l.find('{').and_then(|i| serde_json::from_str::<Value>(&l[i..]).ok()))
        .ok_or_else(|| format!("aucun JSON dans la sortie de doctor :\n{texte}"))?;
    Ok((s.status.code().unwrap_or(-1), json))
}

// ---------------------------------------------------------------------------------------------
// MACH-1
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn gate_mach_1() {
    gate("MACH-1", || {
        // Machine ATTENDUE explicite : KVU_TEST_MACHINE_ATTENDUE=acemagic|gamer (plus de nom d'hôte réel dans le dépôt).
        // Absente : SKIP annoncé ; posée : la détection doit donner cette machine (sinon la gate échoue).
        let attendue = std::env::var("KVU_TEST_MACHINE_ATTENDUE").unwrap_or_default();
        let (machine, backend, arch, autre) = match attendue.as_str() {
            "acemagic" => ("acemagic", "hip", "gfx1151", "cuda"),
            "gamer" => ("gamer", "cuda", "sm_120", "hip"),
            "" => return Ok("SKIP KVU_TEST_MACHINE_ATTENDUE absente : MACH-1 ne vaut rien ici".to_string()),
            a => return Err(format!("KVU_TEST_MACHINE_ATTENDUE={a} : attendu acemagic ou gamer")),
        };
        // 1. doctor --json, aucune surcharge.
        let (rc, v) = doctor(&[])?;
        attendre(rc == 0, || format!("doctor --json rc {rc}\n{v}"))?;
        attendre(v["machine"] == machine, || {
            format!("machine {} au lieu de {machine}", v["machine"])
        })?;
        attendre(v["backend"]["choisi"] == backend, || {
            format!("backend {} au lieu de {backend}\n{v}", v["backend"])
        })?;
        let gpu = v["gpus"]
            .as_array()
            .and_then(|g| g.iter().find(|g| g["backend"] == backend))
            .cloned()
            .unwrap_or(Value::Null);
        attendre(gpu["arch"] == arch, || {
            format!("arch {} au lieu de {arch}\n{v}", gpu["arch"])
        })?;
        attendre(gpu["memoire_go"].as_f64().unwrap_or(0.0) > 1.0, || {
            format!("mémoire GPU illisible : {gpu}")
        })?;
        // 2. L'autre backend forcé : rc 3, erreur typée, jamais de repli silencieux.
        let (rc3, v3) = doctor(&[("KEVINOTRON_BACKEND", autre.to_string())])?;
        attendre(rc3 == 3, || {
            format!("KEVINOTRON_BACKEND={autre} : rc {rc3} au lieu de 3")
        })?;
        attendre(
            v3["backend"]["choisi"].is_null() && !v3["backend"]["erreur"].is_null(),
            || format!("erreur absente : {}", v3["backend"]),
        )?;
        // 3. cpu forcé : rc 0 et backend cpu.
        let (rcc, vc) = doctor(&[("KEVINOTRON_BACKEND", "cpu".to_string())])?;
        attendre(rcc == 0 && vc["backend"]["choisi"] == "cpu", || {
            format!("cpu : rc {rcc}, {}", vc["backend"])
        })?;
        // 4. Le backend propre, forcé : rc 0 (le binaire doit avoir la feature cuda sur le gamer).
        let (rcp, vp) = doctor(&[("KEVINOTRON_BACKEND", backend.to_string())])?;
        attendre(rcp == 0 && vp["backend"]["choisi"] == backend, || {
            format!("{backend} forcé : rc {rcp}, {}", vp["backend"])
        })?;
        // 5. Surcharge de machine.
        let (_, vm) = doctor(&[("KEVINOTRON_MACHINE", "inconnue".to_string())])?;
        attendre(vm["machine"] == "inconnue", || {
            format!("surcharge KEVINOTRON_MACHINE ignorée : {}", vm["machine"])
        })?;
        // 6. etat --json : contrat et rapidité (hors démarrage du processus).
        let debut = Instant::now();
        let p = detecter_rapide();
        let e = etat_json(&p, &politique(&p));
        let ms = debut.elapsed().as_millis();
        attendre(ms < 200, || format!("etat en {ms} ms (≥ 200)"))?;
        attendre(e["machine"] == machine, || format!("etat : machine {}", e["machine"]))?;
        Ok(format!(
            "{machine}/{backend}/{arch} ({}), {autre} forcé ⇒ rc 3, cpu ⇒ cpu, etat {ms} ms",
            gpu["nom"].as_str().unwrap_or("?")
        ))
    });
}

// ---------------------------------------------------------------------------------------------
// VERROU-1
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn gate_verrou_1() {
    gate("VERROU-1", || {
        let dir = dossier_temp("verrou");
        let lock = dir.join("gpu.lock");
        let exe = std::env::current_exe().unwrap();
        // (a) flock externe puis acquerir (descendant) : retour en moins de 2 s.
        let debut = Instant::now();
        let s = Command::new("flock")
            .args(["-w", "20"])
            .arg(&lock)
            .arg(&exe)
            .args(["--exact", "aide_acquerir", "--nocapture", "--test-threads=1"])
            .env("KVU_AIDE_VERROU", &lock)
            .env("KVU_AIDE_ATTENTE_MS", "15000")
            .env_remove("KEVINOTRON_VERROU")
            .output()
            .map_err(|e| e.to_string())?;
        let t_anc = debut.elapsed();
        let l = ligne_aide(&s.stdout);
        attendre(s.status.success() && l.contains("mode=ancetre"), || {
            format!("ancêtre non détecté : rc {:?} {l}", s.status.code())
        })?;
        attendre(t_anc < Duration::from_secs(2), || {
            format!("retour en {t_anc:?} (≥ 2 s) : interblocage")
        })?;
        // (b) deux processus indépendants : sérialisés.
        let trace = dir.join("trace.txt");
        let lance = || {
            relancer(
                "aide_acquerir",
                &[
                    ("KVU_AIDE_VERROU", lock.display().to_string()),
                    ("KVU_AIDE_ATTENTE_MS", "15000".to_string()),
                    ("KVU_AIDE_TENIR_MS", "600".to_string()),
                    ("KVU_AIDE_TRACE", trace.display().to_string()),
                ],
            )
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| e.to_string())
        };
        let (a, b) = (lance()?, lance()?);
        let (sa, sb) = (
            a.wait_with_output().map_err(|e| e.to_string())?,
            b.wait_with_output().map_err(|e| e.to_string())?,
        );
        attendre(sa.status.success() && sb.status.success(), || {
            "un des deux processus a échoué".to_string()
        })?;
        let mut w: Vec<(f64, f64)> = std::fs::read_to_string(&trace)
            .map_err(|e| e.to_string())?
            .lines()
            .map(|l| {
                let mut it = l.split_whitespace().map(|x| x.parse::<f64>().unwrap());
                (it.next().unwrap(), it.next().unwrap())
            })
            .collect();
        w.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        attendre(w.len() == 2 && w[1].0 >= w[0].1 - 0.001, || {
            format!("sections critiques qui se chevauchent : {w:?}")
        })?;
        // (c) délai dépassé : un détenteur indépendant (non ancêtre) ⇒ rc 6, borné.
        let mut detenteur = Command::new("flock")
            .arg(&lock)
            .args(["sleep", "4"])
            .spawn()
            .map_err(|e| e.to_string())?;
        std::thread::sleep(Duration::from_millis(400));
        let debut = Instant::now();
        let s6 = relancer(
            "aide_acquerir",
            &[
                ("KVU_AIDE_VERROU", lock.display().to_string()),
                ("KVU_AIDE_ATTENTE_MS", "1000".to_string()),
            ],
        )
        .output()
        .map_err(|e| e.to_string())?;
        let t6 = debut.elapsed();
        let _ = detenteur.wait();
        attendre(s6.status.code() == Some(6), || {
            format!(
                "délai dépassé : rc {:?} au lieu de 6 ({})",
                s6.status.code(),
                ligne_aide(&s6.stdout)
            )
        })?;
        attendre(t6 < Duration::from_secs(4), || format!("attente non bornée : {t6:?}"))?;
        // (d) verrou RÉEL de la machine, en lecture seule : si cette gate tourne sous
        // `flock … <verrou réel> bash gates_machine.sh`, l'ancêtre doit être reconnu.
        let p = detecter_rapide();
        let reel = VerrouGpu::nouveau(&politique(&p).verrou_gpu);
        let note = if reel.tenu_par_ancetre() {
            let g = reel.acquerir(Duration::from_secs(1)).map_err(|e| e.to_string())?;
            attendre(g.mode() == "ancetre" && !g.pris_ici, || {
                "verrou réel tenu par un ancêtre mais mode ≠ ancetre".to_string()
            })?;
            "verrou réel tenu par un ancêtre : reconnu"
        } else {
            "verrou réel non tenu par un ancêtre (gate lancée hors flock) : non testé"
        };
        let _ = std::fs::remove_dir_all(&dir);
        Ok(format!("flock externe puis acquerir en {} ms (ancêtre) ; 2 processus sérialisés ; délai dépassé ⇒ rc 6 en {} ms ; {note}", t_anc.as_millis(), t6.as_millis()))
    });
}

// ---------------------------------------------------------------------------------------------
// FILE-1
// ---------------------------------------------------------------------------------------------

fn pol_machine(m: Machine) -> Politique {
    let p = ProfilMachine {
        machine: m,
        hote: "h".into(),
        cpu_modele: "c".into(),
        fils_logiques: 8,
        ram_totale_go: 1.0,
        ram_disponible_go: 1.0,
        gpus: vec![],
        versions: vec![],
        surcharge_refusee: None,
    };
    politique_avec(&p, &|_| None, Path::new("/home/x"))
}

fn v(s: &[&str]) -> Vec<String> {
    s.iter().map(|x| x.to_string()).collect()
}

#[test]
#[ignore = "GPU"]
fn gate_file_1() {
    gate("FILE-1", || {
        // (a) construire_commande golden (gamer : taskset 12-19, nice 5 ; mem ⇒ systemd-run).
        let mut j = Job::nouveau(Machine::Gamer, v(&["kevinotron", "su2", "gen"]));
        j.gpu = true;
        j.mem_max = Some("8G".into());
        let attendu = v(&[
            "systemd-run",
            "--scope",
            "-p",
            "MemoryMax=8G",
            "--quiet",
            "nice",
            "-n",
            "5",
            "taskset",
            "-c",
            "12-19",
            "flock",
            "-w",
            "7200",
            "/tmp/gamer_gpu.lock",
            "kevinotron",
            "su2",
            "gen",
        ]);
        attendre(construire_commande(&j, &pol_machine(Machine::Gamer)) == attendu, || {
            "golden gamer faux".to_string()
        })?;
        let mut k = Job::nouveau(Machine::Acemagic, v(&["kevinotron", "doctor"]));
        k.gpu = true;
        let attendu = v(&[
            "systemd-run",
            "--scope",
            "-p",
            "MemoryMax=24G",
            "--quiet",
            "nice",
            "-n",
            "10",
            "flock",
            "-w",
            "7200",
            "/tmp/kevinotron_gpu.lock",
            "kevinotron",
            "doctor",
        ]);
        attendre(
            construire_commande(&k, &pol_machine(Machine::Acemagic)) == attendu,
            || "golden acemagic faux".to_string(),
        )?;
        // (b) 1 job, 2 workers (processus indépendants) ⇒ exécuté UNE fois.
        let dir = dossier_temp("file");
        let racine = dir.join("file");
        let f = File::ouvrir(&racine).map_err(|e| e.to_string())?;
        let trace = dir.join("trace.txt");
        let mut job = Job::nouveau(
            Machine::Inconnue,
            v(&["sh", "-c", &format!("echo execute >> {}; sleep 0.3", trace.display())]),
        );
        job.id = "gate-un-job".into();
        f.ajouter(job).map_err(|e| e.to_string())?;
        let depart = maintenant_unix() + 1.0;
        let lance = || {
            relancer(
                "aide_worker",
                &[
                    ("KVU_AIDE_FILE", racine.display().to_string()),
                    ("KVU_AIDE_DEPART", format!("{depart}")),
                ],
            )
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| e.to_string())
        };
        let (a, b) = (lance()?, lance()?);
        let sorties: Vec<String> = [a, b]
            .into_iter()
            .map(|c| {
                c.wait_with_output()
                    .map(|o| ligne_aide(&o.stdout))
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<_, _>>()?;
        let pris = sorties.iter().filter(|l| l.contains("pris=1")).count();
        attendre(pris == 1, || format!("{pris} workers ont pris le job : {sorties:?}"))?;
        let lignes = std::fs::read_to_string(&trace).unwrap_or_default().lines().count();
        attendre(lignes == 1, || format!("le job s'est exécuté {lignes} fois"))?;
        attendre(
            f.compter(EtatJob::Fini) == 1 && f.compter(EtatJob::Attente) == 0 && f.compter(EtatJob::EnCours) == 0,
            || "état final de la file incohérent".to_string(),
        )?;
        // (c) RAM, SSD et alerte simulés ⇒ le job reste en attente avec la raison au journal.
        let mut refus = Vec::new();
        for (cas, etiquette) in [("ram", "[ram]"), ("ssd", "[ssd]"), ("alerte", "[alerte]")] {
            let d = dossier_temp(&format!("file_{cas}"));
            let fr = File::ouvrir(&d.join("file")).map_err(|e| e.to_string())?;
            let mut job = Job::nouveau(Machine::Inconnue, v(&["true"]));
            job.id = format!("gate-{cas}");
            fr.ajouter(job).map_err(|e| e.to_string())?;
            let mut pol = politique_libre(&d.join("file"));
            match cas {
                "ram" => pol.ram_min_go = 1e9,
                "ssd" => pol.seuil_ssd_go = 1e9,
                _ => {
                    let alerte = d.join("ALERTE.txt");
                    std::fs::write(&alerte, "alerte simulée").map_err(|e| e.to_string())?;
                    pol.alerte_moniteur = Some(alerte);
                }
            }
            let pris = fr.prendre_suivant(&pol).map_err(|e| e.to_string())?;
            attendre(pris.is_none(), || {
                format!("cas {cas} : le job a été pris malgré la ressource simulée")
            })?;
            attendre(fr.compter(EtatJob::Attente) == 1, || {
                format!("cas {cas} : le job a quitté la file d'attente")
            })?;
            let journal = std::fs::read_to_string(fr.chemin_journal(&format!("gate-{cas}"))).unwrap_or_default();
            attendre(journal.contains(&format!("en attente : {etiquette}")), || {
                format!("cas {cas} : raison absente du journal :\n{journal}")
            })?;
            refus.push(cas);
            let _ = std::fs::remove_dir_all(&d);
        }
        let _ = std::fs::remove_dir_all(&dir);
        Ok(format!("golden gamer + acemagic ; 1 job, 2 workers ⇒ 1 exécution ; refus simulés {refus:?} ⇒ en attente avec raison"))
    });
}
