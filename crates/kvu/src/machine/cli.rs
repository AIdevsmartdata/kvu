//! Sous-commandes `doctor`, `etat` et `queue`. Propriétaire : T1.
//!
//! - `doctor [--json] [--test]` : machine, GPU, backend choisi et pourquoi, versions, politique,
//!   SSD, store, file, cache des noyaux, recommandations ; `--test` compile et exécute un noyau
//!   d'essai sous le verrou GPU.
//! - `etat --json` : UNE ligne JSON (contrat gelé §4.11), moins de 200 ms, sans contexte GPU.
//! - `queue add|list|run|annuler` : file de jobs (voir `file.rs`).

use super::detection::{
    detecter, detecter_rapide, explication_backend, l_max_memoire, memoire_gpu_go, ram_disponible_go, ProfilMachine,
};
use super::file::{self, EtatJob, File, Job};
use super::politique::{politique, Politique};
use super::verrou::VerrouGpu;
use crate::commun::{code, BackendKind, Erreur, Machine, Resultat};
use crate::noyaux::Fichier;
use clap::{Arg, ArgAction, ArgMatches, Command};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Sous-commandes enregistrées par cette tâche (lues par cli::principal, T9).
pub fn commandes() -> Vec<clap::Command> {
    vec![
        Command::new("doctor")
            .about("Machine, GPU, backend choisi, politique, SSD, file ; --json ; --test")
            .arg(
                Arg::new("json")
                    .long("json")
                    .action(ArgAction::SetTrue)
                    .help("rapport JSON (une ligne)"),
            )
            .arg(
                Arg::new("test")
                    .long("test")
                    .action(ArgAction::SetTrue)
                    .help("compile et exécute un noyau d'essai sous le verrou GPU"),
            )
            .arg(
                Arg::new("attente-s")
                    .long("attente-s")
                    .value_name("S")
                    .default_value("30")
                    .value_parser(clap::value_parser!(u64))
                    .help("attente maximale du verrou pour --test"),
            ),
        Command::new("etat")
            .about("Une ligne JSON d'état (pour les mods et la supervision)")
            .arg(
                Arg::new("json")
                    .long("json")
                    .action(ArgAction::SetTrue)
                    .help("sortie JSON (défaut : résumé lisible)"),
            ),
        Command::new("queue")
            .about("File de jobs : add | list | run | annuler")
            .subcommand_required(true)
            .subcommand(
                Command::new("add")
                    .about("Ajoute un job : queue add [options] -- COMMANDE…")
                    .arg(
                        Arg::new("machine")
                            .long("machine")
                            .value_name("PROFIL")
                            .help("file de cette machine (défaut : locale ; l'autre passe par le câble)"),
                    )
                    .arg(
                        Arg::new("gpu")
                            .long("gpu")
                            .action(ArgAction::SetTrue)
                            .help("le job prend le verrou GPU"),
                    )
                    .arg(
                        Arg::new("mem")
                            .long("mem")
                            .value_name("8G")
                            .help("plafond mémoire (systemd-run MemoryMax)"),
                    )
                    .arg(
                        Arg::new("fils")
                            .long("fils")
                            .value_name("N")
                            .value_parser(clap::value_parser!(usize))
                            .help("fils rayon"),
                    )
                    .arg(
                        Arg::new("priorite")
                            .long("priorite")
                            .value_name("P")
                            .allow_negative_numbers(true)
                            .value_parser(clap::value_parser!(i32))
                            .default_value("0"),
                    )
                    .arg(
                        Arg::new("sortie-go")
                            .long("sortie-go")
                            .value_name("X")
                            .value_parser(clap::value_parser!(f64))
                            .default_value("0")
                            .help("sortie estimée (Go), garde SSD"),
                    )
                    .arg(
                        Arg::new("repertoire")
                            .long("repertoire")
                            .value_name("DIR")
                            .help("répertoire de travail (défaut : courant)"),
                    )
                    .arg(
                        Arg::new("cmd")
                            .num_args(1..)
                            .last(true)
                            .required(true)
                            .value_name("COMMANDE"),
                    ),
            )
            .subcommand(
                Command::new("list")
                    .about("Liste les jobs")
                    .arg(Arg::new("json").long("json").action(ArgAction::SetTrue))
                    .arg(
                        Arg::new("tout")
                            .long("tout")
                            .action(ArgAction::SetTrue)
                            .help("inclut fini, échec et annulés"),
                    ),
            )
            .subcommand(
                Command::new("run")
                    .about("Exécute les jobs admissibles")
                    .arg(
                        Arg::new("une")
                            .long("une")
                            .action(ArgAction::SetTrue)
                            .help("un seul job puis sortie"),
                    )
                    .arg(
                        Arg::new("boucle")
                            .long("boucle")
                            .action(ArgAction::SetTrue)
                            .help("worker permanent"),
                    )
                    .arg(
                        Arg::new("vider")
                            .long("vider")
                            .action(ArgAction::SetTrue)
                            .help("avec --boucle : s'arrête quand plus rien n'est admissible"),
                    )
                    .arg(
                        Arg::new("pause-s")
                            .long("pause-s")
                            .value_parser(clap::value_parser!(u64))
                            .default_value("5"),
                    ),
            )
            .subcommand(
                Command::new("annuler")
                    .about("Annule un job en attente")
                    .arg(Arg::new("id").required(true)),
            )
            .subcommand(Command::new("recuperer").about(
                "Range en échec les jobs orphelins (worker et processus disparus) ; rien n'est tué ni relancé",
            )),
    ]
}

/// Exécute la sous-commande `nom` (une de `commandes()`) ; rend le code de sortie.
pub fn executer(nom: &str, m: &clap::ArgMatches) -> Resultat<i32> {
    match nom {
        "doctor" => doctor(m),
        "etat" => etat(m),
        "queue" => queue(m),
        autre => Err(Erreur::Config(format!(
            "sous-commande inconnue « {autre} » (doctor|etat|queue)"
        ))),
    }
}

/// Point d'entrée autonome pour les tests et les gates : analyse `argv` (sans le nom du binaire)
/// avec `commandes()` et rend le code de sortie ; les erreurs sont écrites sur stderr.
pub fn principal_machine(argv: &[String]) -> i32 {
    let app = Command::new("kvu-machine")
        .subcommand_required(true)
        .subcommands(commandes());
    let m = match app.try_get_matches_from(std::iter::once("kvu".to_string()).chain(argv.iter().cloned())) {
        Ok(m) => m,
        Err(e) => {
            let _ = e.print();
            return if e.use_stderr() { code::USAGE } else { code::OK };
        }
    };
    let Some((nom, sm)) = m.subcommand() else {
        return code::USAGE;
    };
    match executer(nom, sm) {
        Ok(rc) => rc,
        Err(e) => {
            eprintln!("kvu {nom} : {e}");
            e.code_sortie()
        }
    }
}

// ------------------------------------------------------------------------------------------
// Outils
// ------------------------------------------------------------------------------------------

fn arrondi(x: f64, decimales: i32) -> f64 {
    let f = 10f64.powi(decimales);
    (x * f).round() / f
}

/// Taille cumulée d'un dossier SANS suivre les liens, bornée en entrées et en temps (le store peut
/// peser des centaines de Go). Rend (octets, complet).
pub fn taille_bornee(racine: &Path, max_entrees: usize, max_ms: u64) -> (u64, bool) {
    let debut = Instant::now();
    let mut total = 0u64;
    let mut vus = 0usize;
    let mut pile = vec![racine.to_path_buf()];
    while let Some(d) = pile.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            vus += 1;
            if vus > max_entrees || debut.elapsed() > Duration::from_millis(max_ms) {
                return (total, false);
            }
            let Ok(md) = e.path().symlink_metadata() else { continue };
            if md.is_dir() {
                pile.push(e.path());
            } else {
                total += md.len();
            }
        }
    }
    (total, true)
}

fn charge1() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|t| t.split_whitespace().next()?.parse().ok())
        .unwrap_or(0.0)
}

fn dossier_cache() -> PathBuf {
    if let Some(d) = std::env::var_os("KEVINOTRON_CACHE_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache/kevinotron/kvu")
}

/// Alerte du moniteur active : fichier présent ET modifié il y a moins de 10 min.
fn alerte_active(pol: &Politique) -> bool {
    pol.alerte_moniteur
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok()?.modified().ok())
        .and_then(|t| std::time::SystemTime::now().duration_since(t).ok())
        .is_some_and(|age| age.as_secs_f64() < file::ALERTE_RECENTE_S)
}

/// Chemin dont on mesure l'espace libre : le store (ou son premier ancêtre existant).
fn chemin_ssd(pol: &Politique) -> PathBuf {
    pol.store.clone()
}

/// Comptes de la file. `en_cours` ne compte que les jobs VIVANTS : un job dont le worker et le processus ont
/// disparu (SIGKILL, reboot) est compté dans `orphelins` (lecture seule ; le prochain worker ou
/// `queue recuperer` le range en échec). Sans cela, un job fantôme bloquait à jamais l'installation.
fn compte_file(pol: &Politique) -> Value {
    let f = File {
        racine: pol.file.clone(),
    };
    let orphelins = f.orphelins().len();
    json!({"attente": f.compter(EtatJob::Attente), "en_cours": f.compter(EtatJob::EnCours).saturating_sub(orphelins),
           "echec": f.compter(EtatJob::Echec), "orphelins": orphelins})
}

// ------------------------------------------------------------------------------------------
// etat --json (contrat gelé §4.11)
// ------------------------------------------------------------------------------------------

/// Backend ATTENDU sans sonder les GPU : KEVINOTRON_BACKEND si fixé, sinon le `backend` du profil de la machine.
pub fn backend_attendu(machine: Machine, demande: &str) -> String {
    match demande.trim().to_ascii_lowercase().as_str() {
        d @ ("hip" | "cuda" | "cpu") => d.to_string(),
        _ => crate::config::actuelle().profil(machine).backend,
    }
}

/// Valeur JSON de `etat --json`. Clés gelées : machine, backend, gpu_verrou{pris,pid}, ssd_libre_go,
/// ssd_seuil_go, ram_dispo_go, charge1, file{attente,en_cours,echec}, alerte_moniteur, horodatage.
pub fn etat_json(p: &ProfilMachine, pol: &Politique) -> Value {
    let verrou = VerrouGpu::nouveau(&pol.verrou_gpu);
    let pid = verrou.proprietaire();
    json!({
        "machine": p.machine.nom(),
        "backend": backend_attendu(p.machine, &std::env::var("KEVINOTRON_BACKEND").unwrap_or_default()),
        "gpu_verrou": {"pris": pid.is_some(), "pid": pid},
        "ssd_libre_go": file::libre_go(&chemin_ssd(pol)).map(|x| arrondi(x, 1)),
        "ssd_seuil_go": pol.seuil_ssd_go,
        "ram_dispo_go": arrondi(ram_disponible_go(), 1),
        "charge1": charge1(),
        "file": compte_file(pol),
        "alerte_moniteur": alerte_active(pol),
        "horodatage": file::horodatage_iso(file::maintenant_unix()),
    })
}

fn etat(m: &ArgMatches) -> Resultat<i32> {
    let p = detecter_rapide();
    let pol = politique(&p);
    let v = etat_json(&p, &pol);
    if m.get_flag("json") {
        println!("{v}");
    } else {
        println!(
            "{} · backend {} · verrou GPU {} · SSD {} Go libres (seuil {}) · RAM {} Go · file {}",
            v["machine"].as_str().unwrap_or("?"),
            v["backend"].as_str().unwrap_or("?"),
            if v["gpu_verrou"]["pris"] == true {
                "PRIS"
            } else {
                "libre"
            },
            v["ssd_libre_go"],
            v["ssd_seuil_go"],
            v["ram_dispo_go"],
            v["file"],
        );
    }
    Ok(code::OK)
}

// ------------------------------------------------------------------------------------------
// doctor
// ------------------------------------------------------------------------------------------

/// Noyau d'essai du doctor : y ← a·x + y en FP64 (entiers : résultat exact sur tout backend).
static ESSAI: Fichier = Fichier {
    nom: "doctor_essai.cu",
    contenu: "// Noyau d'essai de `kevinotron doctor --test` : y <- a*x + y (FP64), un fil par élément.\n\
extern \"C\" __global__ void kv_doctor_saxpy(double* y, const double* x, double a, int n) {\n\
    int i = blockIdx.x * blockDim.x + threadIdx.x;\n\
    if (i < n) y[i] = a * x[i] + y[i];\n\
}\n",
};

/// Essai GPU sous verrou : ouvre l'appareil, compile le noyau d'essai, l'exécute et compare à
/// la référence CPU. `Err` = erreur typée (rc propre) ; `Ok(json)` = rapport.
pub fn essai_gpu(p: &ProfilMachine, pol: &Politique, attente: Duration) -> Resultat<Value> {
    use crate::gpu::{self, Arg, Lancement, OptionsCompil};
    let debut = Instant::now();
    let (backend, _) = explication_backend(p)?;
    if backend == BackendKind::Cpu {
        return Ok(json!({"ok": true, "backend": "cpu", "detail": "backend CPU : aucun essai GPU", "duree_ms": 0}));
    }
    let garde = VerrouGpu::nouveau(&pol.verrou_gpu).acquerir(attente)?;
    let app = gpu::ouvrir_auto()?.ok_or_else(|| Erreur::Gpu("ouvrir_auto n'a rendu aucun appareil".to_string()))?;
    let n = 1usize << 16;
    let src = gpu::assembler(backend, "doctor_essai", &[&ESSAI], &[]);
    let module = app.compiler(&src, &OptionsCompil::rapide())?;
    let f = app.fonction(&module, "kv_doctor_saxpy")?;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y0: Vec<f64> = vec![1.0; n];
    let mut dx = app.alloc::<f64>(n)?;
    let mut dy = app.alloc::<f64>(n)?;
    app.ecrire(&mut dx, 0, &x)?;
    app.ecrire(&mut dy, 0, &y0)?;
    app.lancer(
        &f,
        &Lancement::lineaire(n, 256),
        &[dy.arg(), dx.arg(), Arg::F64(2.0), Arg::I32(n as i32)],
    )?;
    app.synchroniser()?;
    let mut y = vec![0.0f64; n];
    app.lire(&dy, 0, &mut y)?;
    let faux = y.iter().zip(&x).filter(|(yi, xi)| **yi != 2.0 * **xi + 1.0).count();
    if faux > 0 {
        return Err(Erreur::Validation(format!(
            "noyau d'essai : {faux} éléments faux sur {n}"
        )));
    }
    Ok(json!({
        "ok": true,
        "backend": backend.nom(),
        "arch": app.arch(),
        "verrou": garde.mode(),
        "elements": n,
        "duree_ms": debut.elapsed().as_millis() as u64,
    }))
}

/// Rapport complet de `doctor` : (JSON, code de sortie du choix de backend).
pub fn rapport_doctor(p: &ProfilMachine, pol: &Politique) -> (Value, i32) {
    let choix = explication_backend(p);
    let (backend_json, rc) = match &choix {
        Ok((b, raison)) => (json!({"choisi": b.nom(), "raison": raison, "erreur": null}), code::OK),
        Err(e) => (
            json!({"choisi": null, "raison": null, "erreur": e.to_string()}),
            e.code_sortie(),
        ),
    };
    let gpus: Vec<Value> = p
        .gpus
        .iter()
        .map(|g| {
            json!({"backend": g.backend.nom(), "nom": g.nom, "arch": g.arch, "memoire_go": arrondi(g.memoire_go, 2),
                   "ratio_fp32_fp64": g.ratio_fp32_fp64, "memoire_unifiee": g.memoire_unifiee})
        })
        .collect();
    let versions: serde_json::Map<String, Value> = p.versions.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
    let verrou = VerrouGpu::nouveau(&pol.verrou_gpu);
    let libre = file::libre_go(&chemin_ssd(pol));
    let (octets_store, complet_store) = taille_bornee(&pol.store, 20_000, 500);
    let cache = dossier_cache();
    let (octets_cache, complet_cache) = taille_bornee(&cache, 20_000, 300);
    // Une chaîne déjà commencée (S2 R2-3, saturation v3) garde sa loi : f64 compat-prod, ou ref_rng pour le
    // bain conditionnel. Passer en F32 exige un avenant au PREREG (ARCHITECTURE_UNIFIEE.md §10).
    let mut recos: Vec<String> = vec![format!(
        "précision conseillée pour un NOUVEL ensemble : {} (une chaîne en cours reste en f64 compat-prod ; \
         F32 seulement après avenant au PREREG)",
        pol.precision_conseillee.nom()
    )];
    if let Ok((b, _)) = &choix {
        let mem = memoire_gpu_go(p, *b);
        if mem > 0.0 {
            recos.push(format!(
                "L maximal en mémoire GPU ({mem:.0} Go, moitié utile) : {} en F32, {} en F64",
                l_max_memoire(mem, 4),
                l_max_memoire(mem, 8)
            ));
        }
    }
    if let Some(l) = libre {
        if l - pol.seuil_ssd_go < 0.0 {
            recos.push(format!(
                "SSD SOUS LE SEUIL : {l:.0} Go libres < {:.0} Go : aucune écriture lourde",
                pol.seuil_ssd_go
            ));
        } else {
            recos.push(format!("marge SSD avant le seuil : {:.0} Go", l - pol.seuil_ssd_go));
        }
    }
    if p.machine != Machine::Inconnue && p.gpus.is_empty() {
        recos.push("machine connue mais aucun GPU visible : vérifier ROCm / le pilote NVIDIA".to_string());
    }
    if p.machine == Machine::Inconnue {
        recos.push(format!(
            "profil {} : {}machine non reconnue, politique prudente (garde SSD {:.0} Go, RAM minimale {:.0} Go, \
             MemoryMax {}) ; écrire un profil dans kvu.toml (voir docs/MACHINES.md)",
            Machine::Inconnue.nom(),
            match &crate::config::actuelle().source {
                crate::config::Source::Defauts => "aucun kvu.toml trouvé, ",
                crate::config::Source::Fichier(_) => "",
            },
            pol.seuil_ssd_go,
            pol.ram_min_go,
            pol.mem_max.as_deref().unwrap_or("aucun")
        ));
    }
    let source_config = match &crate::config::actuelle().source {
        crate::config::Source::Fichier(f) => json!(f),
        crate::config::Source::Defauts => Value::Null,
    };
    let autorises = super::processus::cpus_autorises();
    if let Some(c) = &pol.cpus {
        recos.push(format!(
            "affinité obligatoire {c} (cœurs E seulement) : posée par le binaire lui-même sur chaque sous-commande \
             (processus : {})",
            autorises.as_deref().unwrap_or("?")
        ));
    }
    let v = json!({
        "machine": p.machine.nom(),
        "hote": p.hote,
        "config": {"source": source_config, "profil": p.machine.nom(), "surcharge_refusee": p.surcharge_refusee},
        "cpu": {"modele": p.cpu_modele, "fils": p.fils_logiques},
        "ram": {"totale_go": arrondi(p.ram_totale_go, 1), "disponible_go": arrondi(p.ram_disponible_go, 1)},
        "gpus": gpus,
        "versions": versions,
        "backend": backend_json,
        "verrou_gpu": {"chemin": pol.verrou_gpu, "pris": verrou.est_pris(), "pid": verrou.proprietaire(),
                       "ignore": super::verrou::verrou_ignore()},
        "politique": pol,
        "affinite": {"politique": pol.cpus, "processus": autorises},
        "ssd": {"chemin": chemin_ssd(pol), "libre_go": libre.map(|x| arrondi(x, 1)), "seuil_go": pol.seuil_ssd_go,
                "marge_go": libre.map(|x| arrondi(x - pol.seuil_ssd_go, 1))},
        "store": {"chemin": pol.store, "taille_go": arrondi(octets_store as f64 / (1u64 << 30) as f64, 2), "complet": complet_store},
        "file": {"racine": pol.file, "compte": compte_file(pol)},
        "cache": {"chemin": cache, "taille_mo": arrondi(octets_cache as f64 / (1u64 << 20) as f64, 1), "complet": complet_cache},
        "alerte_moniteur": alerte_active(pol),
        "recommandations": recos,
    });
    (v, rc)
}

fn rendre_texte(v: &Value) -> String {
    let s = |x: &Value| x.as_str().unwrap_or("?").to_string();
    let mut l: Vec<String> = Vec::new();
    l.push(format!("Machine      : {} (hôte {})", s(&v["machine"]), s(&v["hote"])));
    l.push(format!(
        "CPU          : {} ({} fils)",
        s(&v["cpu"]["modele"]),
        v["cpu"]["fils"]
    ));
    l.push(format!(
        "RAM          : {} Go dont {} Go disponibles",
        v["ram"]["totale_go"], v["ram"]["disponible_go"]
    ));
    for g in v["gpus"].as_array().into_iter().flatten() {
        l.push(format!(
            "GPU          : {} {} · {} · {} Go · FP32/FP64 = {}{}",
            s(&g["backend"]),
            s(&g["nom"]),
            s(&g["arch"]),
            g["memoire_go"],
            g["ratio_fp32_fp64"],
            if g["memoire_unifiee"] == true {
                " · mémoire unifiée"
            } else {
                ""
            }
        ));
    }
    if v["gpus"].as_array().is_some_and(Vec::is_empty) {
        l.push("GPU          : aucun".to_string());
    }
    for (k, x) in v["versions"].as_object().into_iter().flatten() {
        l.push(format!("Version      : {k} {}", s(x)));
    }
    match &v["backend"]["choisi"] {
        Value::String(b) => l.push(format!("Backend      : {b} — {}", s(&v["backend"]["raison"]))),
        _ => l.push(format!("Backend      : ERREUR — {}", s(&v["backend"]["erreur"]))),
    }
    l.push(format!(
        "Verrou GPU   : {} ({}){}",
        s(&v["verrou_gpu"]["chemin"]),
        if v["verrou_gpu"]["pris"] == true {
            format!("PRIS par le pid {}", v["verrou_gpu"]["pid"])
        } else {
            "libre".to_string()
        },
        if v["verrou_gpu"]["ignore"] == true {
            " · KEVINOTRON_VERROU=ignorer actif"
        } else {
            ""
        }
    ));
    let p = &v["politique"];
    l.push(format!(
        "Politique    : nice {} · cpus {} · fils ≤ {} · MemoryMax {} · RAM min {} Go · précision {}",
        p["nice"],
        if p["cpus"].is_null() {
            "—".to_string()
        } else {
            s(&p["cpus"])
        },
        p["fils_max"],
        if p["mem_max"].is_null() {
            "—".to_string()
        } else {
            s(&p["mem_max"])
        },
        p["ram_min_go"],
        s(&p["precision_conseillee"])
    ));
    l.push(format!(
        "SSD          : {} Go libres, seuil {} Go, marge {} Go",
        v["ssd"]["libre_go"], v["ssd"]["seuil_go"], v["ssd"]["marge_go"]
    ));
    l.push(format!(
        "Store        : {} ({} Go{})",
        s(&v["store"]["chemin"]),
        v["store"]["taille_go"],
        if v["store"]["complet"] == true {
            ""
        } else {
            ", mesure partielle"
        }
    ));
    l.push(format!(
        "File         : {} · {}",
        s(&v["file"]["racine"]),
        v["file"]["compte"]
    ));
    l.push(format!(
        "Cache noyaux : {} ({} Mo)",
        s(&v["cache"]["chemin"]),
        v["cache"]["taille_mo"]
    ));
    l.push(format!(
        "Moniteur     : {}",
        if v["alerte_moniteur"] == true {
            "EN ALERTE (les jobs attendent)"
        } else {
            "calme"
        }
    ));
    for r in v["recommandations"].as_array().into_iter().flatten() {
        l.push(format!("Conseil      : {}", s(r)));
    }
    if !v["test"].is_null() {
        l.push(format!("Essai GPU    : {}", v["test"]));
    }
    l.join("\n")
}

fn doctor(m: &ArgMatches) -> Resultat<i32> {
    let p = detecter();
    let pol = politique(&p);
    let (mut v, mut rc) = rapport_doctor(&p, &pol);
    if m.get_flag("test") {
        let attente = Duration::from_secs(*m.get_one::<u64>("attente-s").unwrap_or(&30));
        match essai_gpu(&p, &pol, attente) {
            Ok(t) => v["test"] = t,
            Err(e) => {
                v["test"] = json!({"ok": false, "erreur": e.to_string(), "code": e.code_sortie()});
                if rc == code::OK {
                    rc = e.code_sortie();
                }
            }
        }
    }
    if m.get_flag("json") {
        println!("{v}");
    } else {
        println!("{}", rendre_texte(&v));
    }
    Ok(rc)
}

// ------------------------------------------------------------------------------------------
// queue
// ------------------------------------------------------------------------------------------

fn resume(j: &Job) -> String {
    let c = j.commande.join(" ");
    if c.chars().count() > 60 {
        format!("{}…", c.chars().take(59).collect::<String>())
    } else {
        c
    }
}

fn queue(m: &ArgMatches) -> Resultat<i32> {
    let p = detecter_rapide();
    let pol = politique(&p);
    let (nom, sm) = m
        .subcommand()
        .ok_or_else(|| Erreur::Config("queue : sous-commande attendue".to_string()))?;
    match nom {
        "add" => {
            let machine = match sm.get_one::<String>("machine") {
                Some(s) => s.parse::<Machine>()?,
                None => p.machine,
            };
            let cmd: Vec<String> = sm
                .get_many::<String>("cmd")
                .map(|v| v.cloned().collect())
                .unwrap_or_default();
            let mut job = Job::nouveau(machine, cmd);
            job.gpu = sm.get_flag("gpu");
            job.mem_max = sm.get_one::<String>("mem").cloned();
            job.fils = sm.get_one::<usize>("fils").copied().unwrap_or(0);
            job.priorite = sm.get_one::<i32>("priorite").copied().unwrap_or(0);
            job.sortie_estimee_go = sm.get_one::<f64>("sortie-go").copied().unwrap_or(0.0);
            if !(job.sortie_estimee_go.is_finite() && job.sortie_estimee_go >= 0.0) {
                return Err(Erreur::Config("--sortie-go : nombre positif attendu".to_string()));
            }
            let distant = machine != p.machine && p.machine != Machine::Inconnue;
            let rep = sm.get_one::<String>("repertoire").map(PathBuf::from);
            job.repertoire = match (rep, distant) {
                (Some(r), _) => r,
                // Pour l'autre machine, le dossier courant n'existe pas : défaut distant.
                (None, true) => PathBuf::new(),
                (None, false) => std::env::current_dir().unwrap_or_default(),
            };
            if !job.repertoire.as_os_str().is_empty() && !distant && !job.repertoire.is_dir() {
                return Err(Erreur::Config(format!(
                    "répertoire {} introuvable",
                    job.repertoire.display()
                )));
            }
            let id = if distant {
                file::ajouter_distant(&job, &pol)?
            } else {
                File::ouvrir(&pol.file)?.ajouter(job)?
            };
            println!("{id}");
            Ok(code::OK)
        }
        "list" => {
            let f = File::ouvrir(&pol.file)?;
            let orphelins: Vec<String> = f.orphelins().into_iter().map(|j| j.id).collect();
            let tous = f.lister()?;
            let jobs: Vec<&Job> = tous
                .iter()
                .filter(|j| sm.get_flag("tout") || matches!(j.etat, EtatJob::Attente | EtatJob::EnCours))
                .collect();
            if sm.get_flag("json") {
                println!(
                    "{}",
                    serde_json::to_string(&jobs).map_err(|e| Erreur::Format(e.to_string()))?
                );
            } else if jobs.is_empty() {
                println!("file vide ({})", pol.file.display());
            } else {
                for j in jobs {
                    println!(
                        "{:<30} {:<9} prio {:>3} rc {:>4} {}",
                        j.id,
                        if orphelins.contains(&j.id) { "orphelin" } else { j.etat.dossier() },
                        j.priorite,
                        j.rc.map_or("-".to_string(), |r| r.to_string()),
                        resume(j)
                    );
                }
            }
            Ok(code::OK)
        }
        "run" => {
            let f = File::ouvrir(&pol.file)?;
            if sm.get_flag("boucle") {
                let pause = Duration::from_secs(*sm.get_one::<u64>("pause-s").unwrap_or(&5));
                let faits = f.boucle(&pol, sm.get_flag("vider"), pause, None)?;
                for (id, rc) in &faits {
                    println!("{id} rc={rc}");
                }
                return Ok(if faits.iter().all(|(_, rc)| *rc == 0) {
                    code::OK
                } else {
                    code::ECHEC
                });
            }
            // --une (défaut) : un seul job.
            let prise = f.prendre_suivant_detail(&pol)?;
            match prise.job {
                Some(job) => {
                    let id = job.id.clone();
                    let rc = f.executer(job, &pol)?;
                    println!("{id} rc={rc}");
                    Ok(if rc == 0 { code::OK } else { code::ECHEC })
                }
                None => {
                    if prise.refus.is_empty() {
                        println!("rien en attente");
                    }
                    for (id, raison) in &prise.refus {
                        println!("{id} reste en attente : {raison}");
                    }
                    Ok(code::OK)
                }
            }
        }
        "recuperer" => {
            let faits = File::ouvrir(&pol.file)?.recuperer_orphelins()?;
            if faits.is_empty() {
                println!("aucun job orphelin");
            }
            for id in faits {
                println!("{id} orphelin : rangé en échec");
            }
            Ok(code::OK)
        }
        "annuler" => {
            let id = sm.get_one::<String>("id").cloned().unwrap_or_default();
            File::ouvrir(&pol.file)?.annuler(&id)?;
            println!("{id} annulé");
            Ok(code::OK)
        }
        autre => Err(Erreur::Config(format!("queue : sous-commande inconnue « {autre} »"))),
    }
}
