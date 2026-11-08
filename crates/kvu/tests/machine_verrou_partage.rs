// SPDX-License-Identifier: Apache-2.0
//! Verrou GPU partagé sous `gpu_partage` (lot WP21, `docs/MACHINES.md` §8) : tests CPU sur des verrous TEMPORAIRES seulement
//! (jamais le verrou d'une machine). Les scénarios multi-processus relancent CE binaire de test sur `aide_partage` (activé par
//! KVU_AIDE_VERROU). L'enveloppe « gpu_partage simulé » reproduit la mécanique de gpu_partage v2.1 : `flock -s` éphémère sur un
//! descripteur hérité (le PID inscrit dans /proc/locks meurt aussitôt), puis GPU_PARTAGE=1 et GPU_PARTAGE_VERROU_TENU exportés.

use kvu::machine::verrou::*;
use kvu::machine::VerrouGpu as Verrou;
use kvu::stockage::{GardeSsd, Manifeste};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn maintenant() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

fn env_u64(nom: &str, defaut: u64) -> u64 {
    std::env::var(nom).ok().and_then(|s| s.parse().ok()).unwrap_or(defaut)
}

/// Aide : attend le départ commun KVU_AIDE_DEPART (s Unix), acquiert KVU_AIDE_VERROU (`acquerir`, ou `acquerir_exclusif` si
/// KVU_AIDE_EXCLUSIF=1) en KVU_AIDE_ATTENTE_MS au plus, écrit KVU_AIDE_PRET dès la prise, tient KVU_AIDE_TENIR_MS, ajoute
/// « t0 t1 mode » à KVU_AIDE_TRACE, puis imprime son verdict. Code de sortie = code de l'erreur typée (6 : délai).
#[test]
fn aide_partage() {
    let Ok(chemin) = std::env::var("KVU_AIDE_VERROU") else {
        return;
    };
    if let Some(t) = std::env::var("KVU_AIDE_DEPART").ok().and_then(|s| s.parse::<f64>().ok()) {
        while maintenant() < t {
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    let attente = Duration::from_millis(env_u64("KVU_AIDE_ATTENTE_MS", 1000));
    let v = Verrou::nouveau(Path::new(&chemin));
    let debut = Instant::now();
    let r = if std::env::var("KVU_AIDE_EXCLUSIF").is_ok_and(|x| x == "1") { v.acquerir_exclusif(attente) } else { v.acquerir(attente) };
    match r {
        Ok(g) => {
            let t0 = maintenant();
            if let Ok(p) = std::env::var("KVU_AIDE_PRET") {
                std::fs::write(p, "").unwrap();
            }
            std::thread::sleep(Duration::from_millis(env_u64("KVU_AIDE_TENIR_MS", 0)));
            let t1 = maintenant();
            if let Ok(trace) = std::env::var("KVU_AIDE_TRACE") {
                use std::io::Write;
                // une seule écriture en mode ajout : deux aides qui finissent ensemble n'entrelacent pas leurs lignes
                let ligne = format!("{t0} {t1} {}\n", g.mode());
                let mut f = std::fs::OpenOptions::new().create(true).append(true).open(trace).unwrap();
                f.write_all(ligne.as_bytes()).unwrap();
            }
            println!("AIDE:OK mode={} pris_ici={} ms={}", g.mode(), g.pris_ici, debut.elapsed().as_millis());
        }
        Err(e) => {
            println!("AIDE:ERR rc={} ms={} {e}", e.code_sortie(), debut.elapsed().as_millis());
            std::process::exit(e.code_sortie());
        }
    }
}

/// Dossier neuf du test (dans le dossier temporaire : jamais le verrou d'une machine).
fn dossier(nom: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("kvu_verrou_partage_{nom}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn verrou(d: &Path) -> PathBuf {
    let v = d.join("gpu.lock");
    std::fs::write(&v, "").unwrap();
    assert!(v.starts_with(std::env::temp_dir()), "verrou de test hors du dossier temporaire : {}", v.display());
    v
}

fn flock_dispo() -> bool {
    Command::new("flock").arg("--version").output().is_ok()
}

/// Paramètres d'une aide.
#[derive(Clone, Default)]
struct Aide {
    attente_ms: u64,
    tenir_ms: u64,
    depart: Option<f64>,
    trace: Option<PathBuf>,
    pret: Option<PathBuf>,
    exclusif: bool,
    env: Vec<(&'static str, String)>,
}

/// Enveloppe de lancement.
enum Env<'a> {
    Aucune,
    /// gpu_partage simulé ; `sans_variable` retire GPU_PARTAGE devant l'aide (mutant : comportement d'avant WP21).
    Simulee { sans_variable: bool },
    /// /usr/local/bin/gpu_partage réel, sur le verrou temporaire (GPU_PARTAGE_VERROU).
    Reelle(&'a Path),
    /// `flock [-s] -w 20 <verrou>` ancêtre vivant, avec GPU_PARTAGE=1 et le même fichier.
    FlockAncetre { partage: bool },
}

const SIMULEE: &str = r#"exec 8>>"$1"; flock -s 8 || exit 75; V=$1; shift; export GPU_PARTAGE=1 GPU_PARTAGE_VERROU_TENU="$V"; exec "$@""#;
const SIMULEE_SANS: &str = r#"exec 8>>"$1"; flock -s 8 || exit 75; V=$1; shift; export GPU_PARTAGE=1 GPU_PARTAGE_VERROU_TENU="$V"; exec env -u GPU_PARTAGE "$@""#;

fn commande(env: &Env, v: &Path, a: &Aide) -> Command {
    let exe = std::env::current_exe().unwrap();
    let aide = [exe.to_string_lossy().into_owned(), "--exact".into(), "aide_partage".into(), "--nocapture".into(), "--test-threads=1".into()];
    let mut c = match env {
        Env::Aucune => {
            let mut c = Command::new(&aide[0]);
            c.args(&aide[1..]);
            c
        }
        Env::Simulee { sans_variable } => {
            let mut c = Command::new("bash");
            c.arg("-c").arg(if *sans_variable { SIMULEE_SANS } else { SIMULEE }).arg("_").arg(v).args(&aide);
            c
        }
        Env::Reelle(gp) => {
            let mut c = Command::new(gp);
            c.args(["--places", "2", "-w", "30", "--"]).args(&aide).env("GPU_PARTAGE_VERROU", v);
            c
        }
        Env::FlockAncetre { partage } => {
            let mut c = Command::new("flock");
            if *partage {
                c.arg("-s");
            }
            c.args(["-w", "20"]).arg(v).args(&aide).env("GPU_PARTAGE", "1").env("GPU_PARTAGE_VERROU_TENU", v);
            c
        }
    };
    // l'environnement de la session (peut-être lancée sous gpu_partage) ne doit rien décider ici
    for k in ["KEVINOTRON_VERROU", ENV_GPU_PARTAGE, ENV_GPU_PARTAGE_VERROU_TENU, "KVU_AIDE_EXCLUSIF", "KVU_AIDE_DEPART", "KVU_AIDE_TRACE", "KVU_AIDE_PRET"] {
        if !matches!(env, Env::FlockAncetre { .. }) || !k.starts_with("GPU_PARTAGE") {
            c.env_remove(k);
        }
    }
    c.env("KVU_AIDE_VERROU", v).env("KVU_AIDE_ATTENTE_MS", a.attente_ms.to_string()).env("KVU_AIDE_TENIR_MS", a.tenir_ms.to_string());
    if let Some(t) = a.depart {
        c.env("KVU_AIDE_DEPART", t.to_string());
    }
    if let Some(p) = &a.trace {
        c.env("KVU_AIDE_TRACE", p);
    }
    if let Some(p) = &a.pret {
        c.env("KVU_AIDE_PRET", p);
    }
    if a.exclusif {
        c.env("KVU_AIDE_EXCLUSIF", "1");
    }
    for (k, val) in &a.env {
        c.env(k, val);
    }
    c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    c
}

fn ligne_aide(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).lines().find_map(|l| l.find("AIDE:").map(|i| l[i..].to_string())).unwrap_or_default()
}

/// (t0, t1, mode) des aides qui ont tenu le verrou.
fn lire_trace(p: &Path) -> Vec<(f64, f64, String)> {
    std::fs::read_to_string(p)
        .unwrap_or_default()
        .lines()
        .map(|l| {
            let m: Vec<&str> = l.split_whitespace().collect();
            (m[0].parse().unwrap(), m[1].parse().unwrap(), m[2].to_string())
        })
        .collect()
}

fn recouvrement(t: &[(f64, f64, String)]) -> f64 {
    let fin = t.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
    let debut = t.iter().map(|x| x.0).fold(f64::NEG_INFINITY, f64::max);
    fin - debut
}

fn attendre_fichiers(f: &[PathBuf], delai: Duration) -> bool {
    let t = Instant::now();
    while t.elapsed() < delai {
        if f.iter().all(|p| p.exists()) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

// ---------------------------------------------------------------------------------------------------------------------
// Sous gpu_partage : partagé, recouvrement, un exclusif attend les deux ; mutant sans la variable : les deux expirent
// ---------------------------------------------------------------------------------------------------------------------

#[test]
fn deux_aides_sous_gpu_partage_simule_se_recouvrent_et_un_exclusif_attend() {
    if !flock_dispo() {
        println!("\nSKIP flock(1) absent");
        return;
    }
    let d = dossier("simule");
    let v = verrou(&d);
    let trace = d.join("trace.txt");
    let pret = [d.join("pret_a"), d.join("pret_b")];
    let depart = maintenant() + 0.8;
    let lancer = |p: &Path| {
        let a = Aide { attente_ms: 10_000, tenir_ms: 2000, depart: Some(depart), trace: Some(trace.clone()), pret: Some(p.to_path_buf()), ..Default::default() };
        commande(&Env::Simulee { sans_variable: false }, &v, &a).spawn().unwrap()
    };
    let (a, b) = (lancer(&pret[0]), lancer(&pret[1]));
    assert!(attendre_fichiers(&pret, Duration::from_secs(10)), "les deux aides n'ont pas pris le verrou");
    // demande EXCLUSIVE pendant que les deux tiennent le verrou en partagé
    let t_demande = maintenant();
    let excl_t = d.join("excl_t");
    let excl = Command::new("flock")
        .args(["-w", "15"])
        .arg(&v)
        .args(["sh", "-c", "date +%s.%N > \"$1\"", "_"])
        .arg(&excl_t)
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let (sa, sb) = (a.wait_with_output().unwrap(), b.wait_with_output().unwrap());
    let se = excl.wait_with_output().unwrap();
    for s in [&sa, &sb] {
        assert!(s.status.success() && ligne_aide(s).contains("mode=partage pris_ici=true"), "{} {}", ligne_aide(s), String::from_utf8_lossy(&s.stderr));
    }
    let t = lire_trace(&trace);
    assert_eq!(t.len(), 2);
    let r = recouvrement(&t);
    assert!(r >= 1.0, "recouvrement {r:.3} s < 1,0 s : {t:?}");
    assert!(se.status.success());
    let t_excl: f64 = std::fs::read_to_string(&excl_t).unwrap().trim().parse().unwrap();
    let fin_max = t.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
    let fin_min = t.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
    assert!(t_demande < fin_min, "demande exclusive faite trop tard ({t_demande} ≥ {fin_min})");
    assert!(t_excl >= fin_max, "l'exclusif a obtenu le verrou à {t_excl}, avant la fin des partagés ({fin_max})");
    println!("\nPASS VERROU-PARTAGE-CPU-SIMULE : recouvrement {r:.3} s ; exclusif demandé {:.3} s avant la première fin, obtenu {:.3} s après la dernière", fin_min - t_demande, t_excl - fin_max);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn mutant_sans_gpu_partage_les_deux_attendent_et_expirent() {
    if !flock_dispo() {
        println!("\nSKIP flock(1) absent");
        return;
    }
    let d = dossier("mutant");
    let v = verrou(&d);
    let trace = d.join("trace.txt");
    let depart = maintenant() + 0.8;
    let a = Aide { attente_ms: 1500, tenir_ms: 2000, depart: Some(depart), trace: Some(trace.clone()), ..Default::default() };
    let (x, y) = (
        commande(&Env::Simulee { sans_variable: true }, &v, &a).spawn().unwrap(),
        commande(&Env::Simulee { sans_variable: true }, &v, &a).spawn().unwrap(),
    );
    let (sx, sy) = (x.wait_with_output().unwrap(), y.wait_with_output().unwrap());
    for s in [&sx, &sy] {
        assert_eq!(s.status.code(), Some(6), "{}", ligne_aide(s));
        assert!(ligne_aide(s).contains("AIDE:ERR rc=6"), "{}", ligne_aide(s));
    }
    assert!(lire_trace(&trace).is_empty(), "aucune aide ne doit avoir tenu le verrou");
    println!("\nPASS VERROU-PARTAGE-CPU-MUTANT : sans GPU_PARTAGE sous l'enveloppe, les deux aides expirent (rc 6) : aucun recouvrement (gate rouge)");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn gpu_partage_reel_sur_verrou_temporaire() {
    let gp = Path::new("/usr/local/bin/gpu_partage");
    if !gp.exists() || !flock_dispo() {
        println!("\nSKIP VERROU-PARTAGE-CPU-REEL : gpu_partage absent");
        return;
    }
    let d = dossier("reel");
    let v = verrou(&d);
    let trace = d.join("trace.txt");
    let depart = maintenant() + 1.0;
    let a = Aide { attente_ms: 10_000, tenir_ms: 2000, depart: Some(depart), trace: Some(trace.clone()), ..Default::default() };
    let (x, y) = (commande(&Env::Reelle(gp), &v, &a).spawn().unwrap(), commande(&Env::Reelle(gp), &v, &a).spawn().unwrap());
    let (sx, sy) = (x.wait_with_output().unwrap(), y.wait_with_output().unwrap());
    for s in [&sx, &sy] {
        assert!(s.status.success() && ligne_aide(s).contains("mode=partage"), "{} {}", ligne_aide(s), String::from_utf8_lossy(&s.stderr));
    }
    let t = lire_trace(&trace);
    let r = recouvrement(&t);
    assert!(t.len() == 2 && r >= 1.0, "recouvrement {r:.3} s : {t:?}");
    println!("\nPASS VERROU-PARTAGE-CPU-REEL : deux aides sous le vrai gpu_partage (verrou temporaire) : partage, recouvrement {r:.3} s");
    let _ = std::fs::remove_dir_all(&d);
}

// ---------------------------------------------------------------------------------------------------------------------
// Hors correspondance : exclusif, comme avant
// ---------------------------------------------------------------------------------------------------------------------

#[test]
fn autre_fichier_ou_variable_absente_reste_exclusif() {
    let d = dossier("autre");
    let v = verrou(&d);
    let autre = d.join("autre.lock");
    std::fs::write(&autre, "").unwrap();
    // GPU_PARTAGE=1 mais le fichier tenu est un AUTRE : exclusif, sérialisé, avertissement
    let trace = d.join("trace.txt");
    let depart = maintenant() + 0.8;
    let a = Aide {
        attente_ms: 10_000,
        tenir_ms: 800,
        depart: Some(depart),
        trace: Some(trace.clone()),
        env: vec![(ENV_GPU_PARTAGE, "1".into()), (ENV_GPU_PARTAGE_VERROU_TENU, autre.to_string_lossy().into_owned())],
        ..Default::default()
    };
    let (x, y) = (commande(&Env::Aucune, &v, &a).spawn().unwrap(), commande(&Env::Aucune, &v, &a).spawn().unwrap());
    let (sx, sy) = (x.wait_with_output().unwrap(), y.wait_with_output().unwrap());
    for s in [&sx, &sy] {
        assert!(s.status.success() && ligne_aide(s).contains("mode=pris_ici"), "{}", ligne_aide(s));
        assert!(String::from_utf8_lossy(&s.stderr).contains("prise exclusive"), "{}", String::from_utf8_lossy(&s.stderr));
    }
    let mut t = lire_trace(&trace);
    t.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
    assert!(t.len() == 2 && t[1].0 >= t[0].1 - 0.001, "sections critiques qui se chevauchent : {t:?}");
    // GPU_PARTAGE=1 sans GPU_PARTAGE_VERROU_TENU : exclusif, avertissement « absent »
    let a = Aide { attente_ms: 2000, env: vec![(ENV_GPU_PARTAGE, "1".into())], ..Default::default() };
    let s = commande(&Env::Aucune, &v, &a).output().unwrap();
    assert!(ligne_aide(&s).contains("mode=pris_ici") && String::from_utf8_lossy(&s.stderr).contains("(absent)"), "{} {}", ligne_aide(&s), String::from_utf8_lossy(&s.stderr));
    // GPU_PARTAGE=0 (même fichier) : exclusif, sans avertissement ; « 1 » entouré d'espaces : partagé
    let tenu = v.to_string_lossy().into_owned();
    let a0 = Aide { attente_ms: 2000, env: vec![(ENV_GPU_PARTAGE, "0".into()), (ENV_GPU_PARTAGE_VERROU_TENU, tenu.clone())], ..Default::default() };
    let s0 = commande(&Env::Aucune, &v, &a0).output().unwrap();
    assert!(ligne_aide(&s0).contains("mode=pris_ici") && !String::from_utf8_lossy(&s0.stderr).contains("prise exclusive"), "{}", ligne_aide(&s0));
    let a1 = Aide { attente_ms: 2000, env: vec![(ENV_GPU_PARTAGE, " 1 ".into()), (ENV_GPU_PARTAGE_VERROU_TENU, tenu)], ..Default::default() };
    let s1 = commande(&Env::Aucune, &v, &a1).output().unwrap();
    assert!(ligne_aide(&s1).contains("mode=partage pris_ici=true"), "{}", ligne_aide(&s1));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn banc_exclusif_refuse_sous_gpu_partage() {
    if !flock_dispo() {
        println!("\nSKIP flock(1) absent");
        return;
    }
    let d = dossier("banc");
    let v = verrou(&d);
    let a = Aide { attente_ms: 5000, exclusif: true, ..Default::default() };
    let t = Instant::now();
    let s = commande(&Env::Simulee { sans_variable: false }, &v, &a).output().unwrap();
    assert_eq!(s.status.code(), Some(6), "{}", ligne_aide(&s));
    assert!(ligne_aide(&s).contains("exclusivité requise"), "{}", ligne_aide(&s));
    assert!(t.elapsed() < Duration::from_secs(2), "refus non immédiat : {:?}", t.elapsed());
    let s = commande(&Env::Aucune, &v, &a).output().unwrap();
    assert!(s.status.success() && ligne_aide(&s).contains("mode=pris_ici"), "{}", ligne_aide(&s));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn ignorer_reste_prioritaire() {
    let d = dossier("ignorer");
    let v = verrou(&d);
    let tenu = v.to_string_lossy().into_owned();
    let a = Aide {
        attente_ms: 1000,
        env: vec![("KEVINOTRON_VERROU", "ignorer".into()), (ENV_GPU_PARTAGE, "1".into()), (ENV_GPU_PARTAGE_VERROU_TENU, tenu)],
        ..Default::default()
    };
    let s = commande(&Env::Aucune, &v, &a).output().unwrap();
    assert!(s.status.success() && ligne_aide(&s).contains("mode=ignore pris_ici=false"), "{}", ligne_aide(&s));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn ancetre_vivant_reconnu_en_mode_partage() {
    if !flock_dispo() {
        println!("\nSKIP flock(1) absent");
        return;
    }
    let d = dossier("ancetre");
    let v = verrou(&d);
    for partage in [false, true] {
        let t = Instant::now();
        let s = commande(&Env::FlockAncetre { partage }, &v, &Aide { attente_ms: 15_000, ..Default::default() }).output().unwrap();
        assert!(s.status.success() && ligne_aide(&s).contains("mode=ancetre pris_ici=false"), "flock {} : {}", if partage { "-s" } else { "" }, ligne_aide(&s));
        assert!(t.elapsed() < Duration::from_secs(2), "interblocage : {:?}", t.elapsed());
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn detenteur_exclusif_frere_fait_attendre_le_partage() {
    if !flock_dispo() {
        println!("\nSKIP flock(1) absent");
        return;
    }
    let d = dossier("frere");
    let v = verrou(&d);
    let tenu = v.to_string_lossy().into_owned();
    let mut frere = Command::new("flock").arg(&v).args(["sleep", "3"]).spawn().unwrap();
    std::thread::sleep(Duration::from_millis(400));
    let a = Aide { attente_ms: 1000, env: vec![(ENV_GPU_PARTAGE, "1".into()), (ENV_GPU_PARTAGE_VERROU_TENU, tenu)], ..Default::default() };
    let t = Instant::now();
    let s = commande(&Env::Aucune, &v, &a).output().unwrap();
    let e = t.elapsed();
    assert_eq!(s.status.code(), Some(6), "{}", ligne_aide(&s));
    assert!(e >= Duration::from_millis(900) && e < Duration::from_secs(3), "{e:?}");
    frere.wait().unwrap();
    let s = commande(&Env::Aucune, &v, &a).output().unwrap();
    assert!(s.status.success() && ligne_aide(&s).contains("mode=partage"), "{}", ligne_aide(&s));
    let _ = std::fs::remove_dir_all(&d);
}

// ---------------------------------------------------------------------------------------------------------------------
// Règle pure et manifeste
// ---------------------------------------------------------------------------------------------------------------------

#[test]
fn regle_partage_demande_table_de_verite() {
    let (a, b) = (Some((10u64, 20u64)), Some((10u64, 21u64)));
    assert!(partage_demande(Some("1"), a, a));
    assert!(partage_demande(Some(" 1 "), a, a));
    assert!(partage_demande(Some("1\n"), a, a));
    for (g, t, v) in [
        (Some("0"), a, a),
        (Some("oui"), a, a),
        (Some(""), a, a),
        (None, a, a),
        (Some("1"), a, b),
        (Some("1"), None, a),
        (Some("1"), a, None),
        (Some("1"), None, None),
    ] {
        assert!(!partage_demande(g, t, v), "{g:?} {t:?} {v:?}");
    }
    assert_eq!(ModeVerrou::Partage.nom(), "partage");
}

#[test]
fn manifeste_accepte_partage() {
    let d = dossier("manifeste");
    let run = d.join("run");
    let argv: Vec<String> = ["kvu", "test"].iter().map(|s| s.to_string()).collect();
    let mut m = Manifeste::ouvrir_avec_garde(&run, "test", &argv, &GardeSsd::nouvelle(0.0), 1 << 20).unwrap();
    m.verrou("partage").unwrap();
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(run.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(j["verrou"], "partage");
    assert!(m.verrou("autre").is_err());
    let _ = std::fs::remove_dir_all(&d);
}
