//! Sous-commande `obs` (wilson, polyakov, flot sur une configuration ou un ensemble). Propriétaire : T7.
//!
//! ```text
//! kevinotron obs wilson   --cfg X.npy | --ensemble ID --out DIR [--rmax 4 --tmax 4 --dir-temps 3 --toutes]
//! kevinotron obs polyakov --cfg X.npy | --ensemble ID --out DIR [--dir-temps 3]
//! kevinotron obs flot     --cfg X.npy | --ensemble ID --out DIR [--dt 0.01 --tmax 3 --chaque 1
//!                                                                  --cible-t2e 0.3 --cible-w 0.3]
//! ```
//! Sorties (§5 de l'architecture) : `wilson.npy` (r_max, t_max) et `wilson.csv` (`r,t,W`) ; `polyakov_corr.csv`
//! (`r,C`) ; `flot.csv` (`t,E_plaq,E_trefle,t2E,Q`) et `flot.json` (t₀, w₀). Avec `--ensemble`, chaque
//! configuration `cfg_*.npy` de l'ensemble écrit dans son sous-répertoire de `--out`.
//!
//! Backend : `--backend auto|hip|cuda|cpu` (défaut auto : GPU de la machine pour L ≥ 16, CPU en dessous,
//! car une petite configuration se mesure plus vite sur l'hôte que le GPU ne se prend). Tout calcul GPU
//! passe sous le verrou de la machine.

use super::flot::{flot_wilson, CourbeFlot, ParamsFlot};
use super::wilson::{boucles_wilson, correlateur_polyakov, ParamsWilson, TableWilson};
use crate::commun::{BackendKind, ChampHote, Erreur, Precision, Resultat};
use crate::gpu::Appareil;
use crate::heritage::su2_hb::g17;
use crate::heritage::manifest::sha256_fichier;
use crate::machine::admission::{admettre, octets_champ, BesoinMemoire};
use crate::machine::detection::arch_memoire_unifiee;
use crate::machine::{choisir_backend, detecter, politique, GardeVerrou, VerrouGpu};
use crate::stockage::{npy, GardeSsd, Manifeste};
use crate::su2::ChampGpu;
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Seuil de L au-dessus duquel `--backend auto` utilise le GPU.
const L_MIN_GPU_AUTO: usize = 16;

fn source() -> [Arg; 3] {
    [
        Arg::new("cfg")
            .long("cfg")
            .value_name("X.npy")
            .help("configuration SU(2) (NPY c16 ou q64)"),
        Arg::new("ensemble")
            .long("ensemble")
            .value_name("ID")
            .help("ensemble du store (ou répertoire) : toutes ses cfg_*.npy"),
        Arg::new("backend")
            .long("backend")
            .default_value("auto")
            .help("auto | hip | cuda | cpu"),
    ]
}

fn commune() -> [Arg; 1] {
    [Arg::new("out")
        .long("out")
        .required(true)
        .value_name("DIR")
        .help("répertoire de sortie")]
}

fn reel(nom: &'static str, defaut: &'static str, aide: &'static str) -> Arg {
    Arg::new(nom).long(nom).default_value(defaut).help(aide)
}

pub fn commandes() -> Vec<clap::Command> {
    vec![Command::new("obs")
        .about("Observables : wilson | polyakov | flot")
        .subcommand_required(true)
        .subcommands(super::topo_cli::commandes())
        .subcommand(
            Command::new("wilson")
                .about("Boucles de Wilson R×T")
                .args(source())
                .args(commune())
                .arg(reel("rmax", "4", "R maximal"))
                .arg(reel("tmax", "4", "T maximal"))
                .arg(reel("dir-temps", "3", "direction du temps"))
                .arg(
                    Arg::new("toutes")
                        .long("toutes")
                        .action(ArgAction::SetTrue)
                        .help("moyenne sur les 12 orientations"),
                ),
        )
        .subcommand(
            Command::new("polyakov")
                .about("Corrélateur de Polyakov et ⟨P⟩")
                .args(source())
                .args(commune())
                .arg(reel("dir-temps", "3", "direction du temps")),
        )
        .subcommand(
            Command::new("flot")
                .about("Flot de Wilson RK3 : E_plaq, E_trefle, Q_top, t0, w0")
                .args(source())
                .args(commune())
                .arg(reel("dt", "0.01", "pas du flot"))
                .arg(reel("tmax", "3.0", "temps de flot maximal"))
                .arg(reel("chaque", "1", "mesurer tous les N pas"))
                .arg(reel("cible-t2e", "0.3", "cible de t²E pour t0"))
                .arg(reel("cible-w", "0.3", "cible de W pour w0")),
        )]
}

pub fn executer(nom: &str, m: &ArgMatches) -> Resultat<i32> {
    if nom != "obs" {
        return Err(Erreur::Config(format!(
            "sous-commande « {nom} » inconnue pour le module obs"
        )));
    }
    match m.subcommand() {
        Some(("topo", sm)) => super::topo_cli::mesurer(sm),
        Some(("topo-gates", sm)) => super::topo_cli::gates(sm),
        Some((sous, sm)) => lancer(sous, sm),
        None => Err(Erreur::Config("obs : wilson | polyakov | flot attendu".into())),
    }
}

fn argument<T: std::str::FromStr>(m: &ArgMatches, nom: &str) -> Resultat<T> {
    let brut = m
        .get_one::<String>(nom)
        .ok_or_else(|| Erreur::Config(format!("--{nom} manquant")))?;
    brut.parse::<T>()
        .map_err(|_| Erreur::Config(format!("--{nom} : valeur « {brut} » invalide")))
}

// ------------------------------------------------------------------------------------------
// Entrées : configurations et backend
// ------------------------------------------------------------------------------------------

/// Lit une configuration (c16 ou q64, `.zst` compris) par `stockage::npy`.
fn charger_champ(chemin: &Path) -> Resultat<ChampHote> {
    npy::lire_champ(chemin)
}

/// Racine du store : KEVINOTRON_STORE, sinon celle de la politique de la machine.
fn racine_store() -> PathBuf {
    match std::env::var_os("KEVINOTRON_STORE") {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => politique(&crate::machine::detection::detecter_rapide()).store,
    }
}

/// Fichiers `cfg_*.npy[.zst]` d'un ensemble (répertoire ou identifiant du store), triés, un niveau de
/// sous-répertoires `c<k>/` compris.
fn configurations_ensemble(id: &str) -> Resultat<Vec<PathBuf>> {
    let direct = Path::new(id);
    let racine = if direct.is_dir() {
        direct.to_path_buf()
    } else {
        racine_store().join(id)
    };
    if !racine.is_dir() {
        return Err(Erreur::Config(format!(
            "ensemble « {id} » introuvable ({})",
            racine.display()
        )));
    }
    let mut trouves = Vec::new();
    let mut a_visiter = vec![racine.clone()];
    while let Some(dossier) = a_visiter.pop() {
        let lecture = std::fs::read_dir(&dossier).map_err(|source| Erreur::Io {
            contexte: format!("lecture de {}", dossier.display()),
            source,
        })?;
        for entree in lecture.flatten() {
            let p = entree.path();
            let nom = entree.file_name().to_string_lossy().into_owned();
            let sous_niveau = dossier != racine;
            if p.is_dir() && !sous_niveau && !nom.starts_with('.') {
                a_visiter.push(p);
            } else if nom.starts_with("cfg_") && (nom.ends_with(".npy") || nom.ends_with(".npy.zst")) {
                trouves.push(p);
            }
        }
    }
    trouves.sort();
    if trouves.is_empty() {
        return Err(Erreur::Config(format!(
            "ensemble « {id} » : aucune configuration cfg_*.npy dans {}",
            racine.display()
        )));
    }
    Ok(trouves)
}

/// Liste (chemin de la configuration, sous-répertoire de sortie).
fn entrees(m: &ArgMatches, out: &Path) -> Resultat<Vec<(PathBuf, PathBuf)>> {
    match (m.get_one::<String>("cfg"), m.get_one::<String>("ensemble")) {
        (Some(c), None) => Ok(vec![(PathBuf::from(c), out.to_path_buf())]),
        (None, Some(e)) => {
            let racine = {
                let d = Path::new(e);
                if d.is_dir() {
                    d.to_path_buf()
                } else {
                    racine_store().join(e)
                }
            };
            Ok(configurations_ensemble(e)?
                .into_iter()
                .map(|p| {
                    let rel = p
                        .strip_prefix(&racine)
                        .unwrap_or(&p)
                        .to_string_lossy()
                        .replace('/', "_");
                    let nom = rel.trim_end_matches(".zst").trim_end_matches(".npy").to_string();
                    (p, out.join(nom))
                })
                .collect())
        }
        _ => Err(Erreur::Config(
            "donner exactement un de --cfg X.npy ou --ensemble ID".into(),
        )),
    }
}

/// Appareil GPU choisi (le verrou est gardé tant que `_verrou` vit).
struct Calcul {
    app: Option<Appareil>,
    _verrou: Option<GardeVerrou>,
}

fn ouvrir_gpu(backend: BackendKind) -> Resultat<Calcul> {
    let pol = politique(&crate::machine::detection::detecter_rapide());
    // Le verrou est pris AVANT d'ouvrir le GPU, et un verrou indisponible (T1 absent) interdit le GPU :
    // on ne touche jamais au GPU de production sans verrou.
    let verrou = VerrouGpu::nouveau(&pol.verrou_gpu).acquerir(Duration::from_secs(pol.attente_verrou_s))?;
    let app = crate::gpu::ouvrir(backend, 0)?;
    Ok(Calcul {
        app: Some(app),
        _verrou: Some(verrou),
    })
}

/// Backend du calcul, décidé SANS ouvrir le GPU (l'admission mémoire passe avant) : `auto` prend le GPU de la
/// machine pour L ≥ 16 et le CPU en dessous.
fn backend_voulu(demande: &str, l: usize) -> Resultat<BackendKind> {
    match demande {
        "cpu" => Ok(BackendKind::Cpu),
        "auto" if l < L_MIN_GPU_AUTO => Ok(BackendKind::Cpu),
        "auto" => choisir_backend(&detecter()),
        autre => autre.parse::<BackendKind>(),
    }
}

fn ouvrir_calcul(b: BackendKind) -> Resultat<Calcul> {
    match b {
        BackendKind::Cpu => Ok(Calcul {
            app: None,
            _verrou: None,
        }),
        gpu => ouvrir_gpu(gpu),
    }
}

/// Mémoire estimée d'une mesure `sous` à côté L : champ hôte relu (et son tampon de décodage) ; sur GPU le champ
/// résident et, pour le flot RK3, quatre champs de travail ; sur CPU ces champs de travail sont sur l'hôte.
pub fn besoin_memoire(sous: &str, l: usize, gpu: bool) -> BesoinMemoire {
    let champ = octets_champ(l, 8);
    let travail = if sous == "flot" { 5 * champ } else { champ };
    BesoinMemoire {
        hote: 2 * champ + if gpu { 0 } else { travail },
        gpu: if gpu { champ + travail } else { 0 },
    }
}

/// Octets écrits par configuration (fichiers de mesure, très petits) : réserve de 1 Mio, comptée par la garde SSD.
const OCTETS_SORTIE_PAR_CFG: u64 = 1 << 20;

/// Configuration montée sur le calcul choisi : champ résident (GPU) ou champ hôte (CPU).
struct Monte {
    gpu: Option<ChampGpu>,
    hote: ChampHote,
}

fn monter(calcul: &Calcul, hote: ChampHote) -> Resultat<Monte> {
    match &calcul.app {
        Some(app) => {
            let mut g = ChampGpu::allouer(app, hote.l, 1, Precision::F64)?;
            g.televerser(app, 0, &hote)?;
            Ok(Monte { gpu: Some(g), hote })
        }
        None => Ok(Monte { gpu: None, hote }),
    }
}

// ------------------------------------------------------------------------------------------
// Sorties
// ------------------------------------------------------------------------------------------

fn creer(dossier: &Path) -> Resultat<()> {
    std::fs::create_dir_all(dossier).map_err(|source| Erreur::Io {
        contexte: format!("création de {}", dossier.display()),
        source,
    })
}

fn ecrire_texte(chemin: &Path, texte: &str) -> Resultat<()> {
    std::fs::write(chemin, texte).map_err(|source| Erreur::Io {
        contexte: format!("écriture de {}", chemin.display()),
        source,
    })
}

/// Écrit un tableau f64 NPY par `stockage::npy` (écriture atomique, garde SSD).
fn ecrire_npy(chemin: &Path, forme: &[usize], donnees: &[f64]) -> Resultat<()> {
    npy::ecrire_f64(chemin, forme, donnees, &GardeSsd::machine())
}

fn ecrire_wilson(dossier: &Path, t: &TableWilson) -> Resultat<()> {
    ecrire_npy(&dossier.join("wilson.npy"), &[t.r_max, t.t_max], &t.w)?;
    let mut csv = String::from("r,t,W\n");
    for r in 1..=t.r_max {
        for tt in 1..=t.t_max {
            csv.push_str(&format!("{r},{tt},{}\n", g17(t.get(r, tt))));
        }
    }
    ecrire_texte(&dossier.join("wilson.csv"), &csv)
}

fn ecrire_polyakov(dossier: &Path, c: &[f64], moyen: Option<[f64; 4]>) -> Resultat<()> {
    let mut csv = String::from("r,C\n");
    for (r, v) in c.iter().enumerate() {
        csv.push_str(&format!("{r},{}\n", g17(*v)));
    }
    ecrire_texte(&dossier.join("polyakov_corr.csv"), &csv)?;
    if let Some(p) = moyen {
        let json = serde_json::json!({ "P0": p[0], "P1": p[1], "P2": p[2], "P3": p[3] });
        ecrire_texte(&dossier.join("polyakov_moyen.json"), &(json.to_string() + "\n"))?;
    }
    Ok(())
}

fn ecrire_flot(dossier: &Path, c: &CourbeFlot, p: &ParamsFlot) -> Resultat<()> {
    let mut csv = String::from("t,E_plaq,E_trefle,t2E,Q\n");
    for i in 0..c.t.len() {
        csv.push_str(&format!(
            "{},{},{},{},{}\n",
            g17(c.t[i]),
            g17(c.e_plaq[i]),
            g17(c.e_trefle[i]),
            g17(c.t2e[i]),
            g17(c.q_top[i])
        ));
    }
    ecrire_texte(&dossier.join("flot.csv"), &csv)?;
    let json = serde_json::json!({
        "t0": c.t0, "w0": c.w0, "dt": p.dt, "t_max": p.t_max, "mesure_chaque": p.mesure_chaque,
        "cible_t2e": p.cible_t2e, "cible_w": p.cible_w,
        "Q_final": c.q_top.last(), "E_plaq_final": c.e_plaq.last(), "E_trefle_final": c.e_trefle.last(),
    });
    ecrire_texte(
        &dossier.join("flot.json"),
        &(serde_json::to_string_pretty(&json).unwrap_or_default() + "\n"),
    )
}

// ------------------------------------------------------------------------------------------
// Les trois commandes
// ------------------------------------------------------------------------------------------

fn lancer(sous: &str, m: &ArgMatches) -> Resultat<i32> {
    let out = PathBuf::from(argument::<String>(m, "out")?);
    let demande = argument::<String>(m, "backend")?;
    let liste = entrees(m, &out)?;
    // Paramètres lus et validés avant tout calcul.
    let pw = if sous == "wilson" {
        let r_max: usize = argument(m, "rmax")?;
        let t_max: usize = argument(m, "tmax")?;
        Some(ParamsWilson {
            r_max,
            t_max,
            dir_temps: argument(m, "dir-temps")?,
            toutes_orientations: m.get_flag("toutes"),
        })
    } else {
        None
    };
    let dir_temps: usize = if sous == "polyakov" {
        argument(m, "dir-temps")?
    } else {
        3
    };
    let pf = if sous == "flot" {
        Some(ParamsFlot {
            dt: argument(m, "dt")?,
            t_max: argument(m, "tmax")?,
            mesure_chaque: argument(m, "chaque")?,
            cible_t2e: argument(m, "cible-t2e")?,
            cible_w: argument(m, "cible-w")?,
        })
    } else {
        None
    };
    if !matches!(sous, "wilson" | "polyakov" | "flot") {
        return Err(Erreur::Config(format!("obs {sous} : wilson | polyakov | flot attendu")));
    }

    // Avant tout octet écrit et avant d'ouvrir le GPU : L de la première configuration (en-tête NPY), admission
    // mémoire, puis manifeste (répertoire NEUF, garde SSD avec la projection : refus rc 7 sans rien créer).
    let (_, l) = npy::inspecter_champ(&liste[0].0)?;
    let backend = backend_voulu(&demande, l)?;
    let gpu_unifie = backend == BackendKind::Hip
        && detecter().gpus.iter().any(|g| g.backend == BackendKind::Hip && arch_memoire_unifiee(&g.arch));
    admettre(
        &format!("obs {sous} L={l} ({})", backend.nom()),
        besoin_memoire(sous, l, backend != BackendKind::Cpu),
        gpu_unifie,
    )?;
    let argv: Vec<String> = std::env::args().collect();
    let projection = OCTETS_SORTIE_PAR_CFG.saturating_mul(liste.len() as u64);
    let mut man = Manifeste::ouvrir_avec_garde(&out, &format!("kvu obs {sous}"), &argv, &GardeSsd::machine(), projection)?;
    match mesurer_tout(sous, &liste, &out, backend, pw.as_ref(), dir_temps, pf.as_ref(), &mut man) {
        Ok(()) => {
            man.terminer(0)?;
            Ok(crate::commun::code::OK)
        }
        Err(e) => {
            let _ = man.echouer(e.code_sortie(), &e.to_string());
            Err(e)
        }
    }
}

/// Mesure chaque configuration et consigne entrées, sorties, backend, verrou et modules au manifeste.
#[allow(clippy::too_many_arguments)]
fn mesurer_tout(
    sous: &str,
    liste: &[(PathBuf, PathBuf)],
    out: &Path,
    backend: BackendKind,
    pw: Option<&ParamsWilson>,
    dir_temps: usize,
    pf: Option<&ParamsFlot>,
    man: &mut Manifeste,
) -> Resultat<()> {
    let calc = ouvrir_calcul(backend)?;
    man.champ("backend", &serde_json::json!(backend.nom()).to_string())?;
    // CPU : aucun verrou GPU (le champ reste « non_determine », ou « ignore » sous KEVINOTRON_VERROU=ignorer).
    if let (Some(app), Some(v)) = (&calc.app, &calc._verrou) {
        man.verrou(v.mode())?;
        man.champ("arch", &serde_json::json!(app.arch()).to_string())?;
        man.ajouter_compil(&[super::gpu::ModuleObs::nouveau(app)?.provenance()])?;
    }
    let parametres = serde_json::json!({
        "sous_commande": sous,
        "wilson": pw.map(|p| serde_json::json!({"r_max": p.r_max, "t_max": p.t_max, "dir_temps": p.dir_temps,
                                                 "toutes_orientations": p.toutes_orientations})),
        "polyakov_dir_temps": (sous == "polyakov").then_some(dir_temps),
        "flot": pf.map(|p| serde_json::json!({"dt": p.dt, "t_max": p.t_max, "mesure_chaque": p.mesure_chaque,
                                               "cible_t2e": p.cible_t2e, "cible_w": p.cible_w})),
    });
    man.champ("parametres", &parametres.to_string())?;
    let mut entrees = Vec::new();
    for (cfg, dossier) in liste {
        let hote = charger_champ(cfg)?;
        let octets = std::fs::metadata(cfg).map(|m| m.len()).unwrap_or(0);
        let sha = sha256_fichier(cfg).map_err(|e| Erreur::io(format!("sha256 {}", cfg.display()), e))?;
        entrees.push(serde_json::json!({"chemin": cfg, "sha256": sha, "octets": octets, "L": hote.l}));
        let mont = monter(&calc, hote)?;
        let (app, champ, h) = (calc.app.as_ref(), mont.gpu.as_ref(), Some(&mont.hote));
        if dossier != out {
            creer(dossier)?;
        }
        let produits: Vec<&str> = match sous {
            "wilson" => {
                let p = pw.ok_or_else(|| Erreur::Config("paramètres wilson absents".into()))?;
                let t = boucles_wilson(app, champ, 0, h, p)?;
                ecrire_wilson(dossier, &t)?;
                println!("obs wilson {} : W(1,1) = {}", cfg.display(), g17(t.get(1, 1)));
                vec!["wilson.npy", "wilson.csv"]
            }
            "polyakov" => {
                let c = correlateur_polyakov(app, champ, 0, h, dir_temps)?;
                let moyen = super::wilson::polyakov_moyen(app, champ, 0, h)?;
                ecrire_polyakov(dossier, &c, Some(moyen))?;
                println!("obs polyakov {} : C(0) = {}", cfg.display(), g17(c[0]));
                vec!["polyakov_corr.csv", "polyakov_moyen.json"]
            }
            _ => {
                let p = pf.ok_or_else(|| Erreur::Config("paramètres flot absents".into()))?;
                let c = flot_wilson(app, champ, 0, h, p)?;
                ecrire_flot(dossier, &c, p)?;
                println!(
                    "obs flot {} : t0 = {:?}, w0 = {:?}, Q = {:?}",
                    cfg.display(),
                    c.t0,
                    c.w0,
                    c.q_top.last()
                );
                vec!["flot.csv", "flot.json"]
            }
        };
        for nom in produits {
            man.fichier(&dossier.join(nom))?;
        }
    }
    man.champ("entrees", &serde_json::Value::Array(entrees).to_string())
}

