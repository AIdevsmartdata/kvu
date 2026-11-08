//! Tests du verrou GPU (tâche T1) : analyse de /proc/locks, chaîne des ancêtres, flock réel,
//! `flock` externe SANS interblocage, sérialisation de deux processus, délai dépassé ⇒ rc 6.
//!
//! Les scénarios multi-processus relancent CE binaire de test sur un test « aide_* » (activé par
//! la variable KVU_AIDE) : pas besoin d'un binaire séparé.

use kvu::machine::verrou::*;
use kvu::machine::VerrouGpu as Verrou;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn donnees(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/donnees/machine")
        .join(rel)
}

/// Chemin de verrou neuf dans le dossier temporaire (jamais le verrou de production).
fn verrou_temp(nom: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("kvu_verrou_{nom}_{}.lock", std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

// ---------------------------------------------------------------------------------------------
// Aides multi-processus
// ---------------------------------------------------------------------------------------------

/// Aide : acquiert le verrou KVU_AIDE_VERROU (attente KVU_AIDE_ATTENTE_MS), imprime son verdict.
/// Si KVU_AIDE_TENIR_MS est fixé, tient le verrou ce temps et note ses instants dans KVU_AIDE_TRACE.
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
    match Verrou::nouveau(Path::new(&chemin)).acquerir(Duration::from_millis(attente)) {
        Ok(g) => {
            println!(
                "AIDE:OK mode={} pris_ici={} ms={}",
                g.mode(),
                g.pris_ici,
                debut.elapsed().as_millis()
            );
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

fn aide(verrou: &Path, attente_ms: u64) -> Command {
    let mut c = Command::new(std::env::current_exe().unwrap());
    c.args(["--exact", "aide_acquerir", "--nocapture", "--test-threads=1"])
        .env("KVU_AIDE_VERROU", verrou)
        .env("KVU_AIDE_ATTENTE_MS", attente_ms.to_string())
        .env_remove("KEVINOTRON_VERROU")
        .stdin(Stdio::null());
    c
}

fn ligne_aide(sortie: &[u8]) -> String {
    // Le harnais écrit « test aide_acquerir ... » AVANT notre ligne : on coupe à « AIDE: ».
    String::from_utf8_lossy(sortie)
        .lines()
        .find_map(|l| l.find("AIDE:").map(|i| l[i..].to_string()))
        .unwrap_or_default()
}

fn flock_dispo() -> bool {
    Command::new("flock").arg("--version").output().is_ok()
}

// ---------------------------------------------------------------------------------------------
// Analyse pure
// ---------------------------------------------------------------------------------------------

#[test]
fn analyse_de_proc_locks() {
    let t = std::fs::read_to_string(donnees("locks/proc_locks_exemple")).unwrap();
    let e = analyser_locks(&t);
    // 4 FLOCK au début (1, 2, 4, 6), puis 30 : le détenteur seulement (les 3 « -> » sont des attentes).
    let pids: Vec<u32> = e.iter().map(|x| x.pid).collect();
    assert_eq!(pids, vec![2302, 1495, 2525, 64580, 2626162]);
    assert_eq!(
        e[0],
        EntreeLock {
            pid: 2302,
            majeur: 0,
            mineur: 0x3f,
            inode: 78
        }
    );
    // majeur:mineur en hexadécimal, inode en décimal.
    assert_eq!(
        e[1],
        EntreeLock {
            pid: 1495,
            majeur: 0x103,
            mineur: 2,
            inode: 14_687_182
        }
    );
    assert!(analyser_locks("").is_empty());
    assert!(analyser_locks("n'importe quoi\nFLOCK\n1: FLOCK ADVISORY WRITE x 00:3f:78 0 EOF\n").is_empty());
}

#[test]
fn majeur_mineur_encodage_linux() {
    assert_eq!(majeur_mineur(0x10302), (0x103, 2));
    assert_eq!(majeur_mineur(0), (0, 0));
    assert_eq!(majeur_mineur((8 << 8) | 17), (8, 17));
    // Mineur > 255 (encodage étendu) : makedev(259, 300).
    let dev: u64 = (300 & 0xff) | ((259 & 0xfff) << 8) | ((300 & !0xffu64) << 12);
    assert_eq!(majeur_mineur(dev), (259, 300));
}

#[test]
fn ppid_depuis_stat_gere_les_noms_piegeux() {
    assert_eq!(ppid_depuis_stat("5000 (kevinotron) S 4000 5000 4000 0"), Some(4000));
    assert_eq!(ppid_depuis_stat("6001 (a) S (b) c) S 6000 6001 6000"), Some(6000));
    assert_eq!(ppid_depuis_stat("12 (nom avec espaces) R 7 1 1"), Some(7));
    assert_eq!(ppid_depuis_stat(""), None);
    assert_eq!(ppid_depuis_stat("12 (x)"), None);
    assert_eq!(ppid_depuis_stat("12 (x) S abc"), None);
}

#[test]
fn chaine_des_ancetres_simulee() {
    let proc = donnees("proc_stat");
    assert_eq!(ancetres(&proc, 5000), vec![4000, 3000, 1]);
    assert_eq!(ancetres(&proc, 6001), vec![6000, 1]);
    assert_eq!(ancetres(&proc, 3000), vec![1]);
    // pid inconnu : chaîne vide ; cycle : termine sans boucler.
    assert!(ancetres(&proc, 9999).is_empty());
    let cyc = ancetres(&proc, 7000);
    assert_eq!(cyc, vec![7001, 7000]);
}

#[test]
fn ancetres_du_processus_courant_remontent_jusqua_1() {
    let a = ancetres(Path::new("/proc"), std::process::id());
    assert!(!a.is_empty());
    assert_eq!(*a.last().unwrap(), 1);
    assert!(a.contains(&std::os::unix::process::parent_id()));
}

// ---------------------------------------------------------------------------------------------
// flock réel dans un seul processus
// ---------------------------------------------------------------------------------------------

#[test]
fn acquerir_rendre_et_est_pris() {
    let chemin = verrou_temp("simple");
    let v = Verrou::nouveau(&chemin);
    assert!(!v.est_pris(), "le fichier n'existe pas encore");
    let g = v.acquerir(Duration::from_secs(1)).unwrap();
    assert!(g.pris_ici);
    assert_eq!(g.mode(), "pris_ici");
    assert_eq!(g.chemin, chemin);
    assert!(v.est_pris());
    assert_eq!(v.proprietaire(), Some(std::process::id()));
    drop(g);
    assert!(!v.est_pris(), "le verrou doit être rendu à la destruction");
    // Reprenable aussitôt.
    let g2 = v.acquerir(Duration::from_secs(1)).unwrap();
    assert!(g2.pris_ici);
    drop(g2);
    let _ = std::fs::remove_file(chemin);
}

#[test]
fn second_acquerir_dans_le_meme_processus_expire_en_rc6() {
    let chemin = verrou_temp("double");
    let v = Verrou::nouveau(&chemin);
    let _g = v.acquerir(Duration::from_secs(1)).unwrap();
    let debut = Instant::now();
    let e = v.acquerir(Duration::from_millis(700)).err().expect("doit échouer");
    let ecoule = debut.elapsed();
    assert_eq!(e.code_sortie(), 6);
    assert!(
        e.to_string().contains(&std::process::id().to_string()),
        "le détenteur doit être nommé : {e}"
    );
    // Attente bornée : ni immédiat (le délai est respecté), ni beaucoup plus long.
    assert!(
        ecoule >= Duration::from_millis(650) && ecoule < Duration::from_millis(1500),
        "{ecoule:?}"
    );
    let _ = std::fs::remove_file(chemin);
}

#[test]
fn attente_nulle_essaie_une_fois() {
    let chemin = verrou_temp("zero");
    let v = Verrou::nouveau(&chemin);
    let _g = v.acquerir(Duration::ZERO).unwrap();
    let debut = Instant::now();
    assert_eq!(v.acquerir(Duration::ZERO).err().unwrap().code_sortie(), 6);
    assert!(debut.elapsed() < Duration::from_millis(300));
    let _ = std::fs::remove_file(chemin);
}

#[test]
fn chemin_impossible_donne_une_erreur_dio() {
    let v = Verrou::nouveau(Path::new("/nexiste/pas/du/tout.lock"));
    let e = v.acquerir(Duration::from_millis(100)).err().unwrap();
    assert_eq!(e.code_sortie(), 4);
    assert!(!v.est_pris());
    assert_eq!(v.proprietaire(), None);
}

// ---------------------------------------------------------------------------------------------
// Multi-processus
// ---------------------------------------------------------------------------------------------

#[test]
fn flock_externe_puis_acquerir_pas_dinterblocage() {
    if !flock_dispo() {
        println!("SKIP : flock(1) absent");
        return;
    }
    let chemin = verrou_temp("ancetre");
    let debut = Instant::now();
    // `flock LOCK exe aide` : le flock tient le verrou, le test relancé en est un DESCENDANT.
    let sortie = Command::new("flock")
        .args(["-w", "20"])
        .arg(&chemin)
        .arg(std::env::current_exe().unwrap())
        .args(["--exact", "aide_acquerir", "--nocapture", "--test-threads=1"])
        .env("KVU_AIDE_VERROU", &chemin)
        .env("KVU_AIDE_ATTENTE_MS", "15000")
        .env_remove("KEVINOTRON_VERROU")
        .output()
        .unwrap();
    let ecoule = debut.elapsed();
    let l = ligne_aide(&sortie.stdout);
    assert!(sortie.status.success(), "rc {:?} : {l}", sortie.status.code());
    assert!(l.contains("mode=ancetre") && l.contains("pris_ici=false"), "{l}");
    assert!(ecoule < Duration::from_secs(2), "interblocage évité en {ecoule:?}");
    let _ = std::fs::remove_file(chemin);
}

#[test]
fn flock_externe_non_ancetre_bloque_puis_delai_rc6() {
    if !flock_dispo() {
        println!("SKIP : flock(1) absent");
        return;
    }
    let chemin = verrou_temp("frere");
    // Détenteur FRÈRE (pas un ancêtre) : `flock LOCK sleep 4` lancé par ce test.
    let mut detenteur = Command::new("flock").arg(&chemin).args(["sleep", "4"]).spawn().unwrap();
    std::thread::sleep(Duration::from_millis(400));
    assert!(Verrou::nouveau(&chemin).est_pris());
    let debut = Instant::now();
    let sortie = aide(&chemin, 1000).output().unwrap();
    let ecoule = debut.elapsed();
    let l = ligne_aide(&sortie.stdout);
    assert_eq!(sortie.status.code(), Some(6), "{l}");
    assert!(l.contains("AIDE:ERR rc=6"), "{l}");
    assert!(
        ecoule >= Duration::from_millis(900) && ecoule < Duration::from_secs(3),
        "{ecoule:?}"
    );
    detenteur.wait().unwrap();
    // Une fois le détenteur parti, le verrou se prend.
    let ok = aide(&chemin, 1000).output().unwrap();
    assert!(
        ligne_aide(&ok.stdout).contains("mode=pris_ici"),
        "{}",
        ligne_aide(&ok.stdout)
    );
    let _ = std::fs::remove_file(chemin);
}

#[test]
fn deux_processus_independants_sont_serialises() {
    let chemin = verrou_temp("serie");
    let trace = std::env::temp_dir().join(format!("kvu_verrou_trace_{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&trace);
    let lance = |n: u32| {
        let mut c = aide(&chemin, 15_000);
        c.env("KVU_AIDE_TENIR_MS", "600")
            .env("KVU_AIDE_TRACE", &trace)
            .stdout(Stdio::piped());
        let _ = n;
        c.spawn().unwrap()
    };
    let a = lance(1);
    let b = lance(2);
    let (sa, sb) = (a.wait_with_output().unwrap(), b.wait_with_output().unwrap());
    assert!(sa.status.success() && sb.status.success());
    let mut fenetres: Vec<(f64, f64)> = std::fs::read_to_string(&trace)
        .unwrap()
        .lines()
        .map(|l| {
            let mut it = l.split_whitespace().map(|x| x.parse::<f64>().unwrap());
            (it.next().unwrap(), it.next().unwrap())
        })
        .collect();
    assert_eq!(fenetres.len(), 2);
    fenetres.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    // Sections critiques disjointes : la seconde commence après la fin de la première.
    assert!(fenetres[1].0 >= fenetres[0].1 - 0.001, "chevauchement : {fenetres:?}");
    assert!(fenetres[0].1 - fenetres[0].0 >= 0.55);
    let _ = std::fs::remove_file(chemin);
    let _ = std::fs::remove_file(trace);
}

#[test]
fn verrou_ignorer_ne_prend_rien_et_se_signale() {
    let chemin = verrou_temp("ignorer");
    // Détenteur frère présent : sans « ignorer » on attendrait ; avec, retour immédiat.
    let _g = Verrou::nouveau(&chemin).acquerir(Duration::from_secs(1)).unwrap();
    let debut = Instant::now();
    let sortie = aide(&chemin, 200).env("KEVINOTRON_VERROU", "ignorer").output().unwrap();
    let l = ligne_aide(&sortie.stdout);
    assert!(sortie.status.success(), "{l}");
    assert!(l.contains("mode=ignore") && l.contains("pris_ici=false"), "{l}");
    assert!(debut.elapsed() < Duration::from_secs(2));
    let _ = std::fs::remove_file(chemin);
}

#[test]
fn tenu_par_ancetre_faux_hors_flock() {
    let chemin = verrou_temp("noanc");
    let v = Verrou::nouveau(&chemin);
    assert!(!v.tenu_par_ancetre());
    let _g = v.acquerir(Duration::from_secs(1)).unwrap();
    // Tenu par NOUS-MÊMES : pas un ancêtre (sinon deux fils d'un même processus passeraient ensemble).
    assert!(!v.tenu_par_ancetre());
    let _ = std::fs::remove_file(chemin);
}
