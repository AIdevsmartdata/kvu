//! `kevinotron bench` : bancs su2 (f64/f32, L), cond, fibre, obs ; fenêtres ≤ 10 min sous verrou ;
//! sortie JSON. Propriétaire : T9.
//!
//! Règles de bon voisinage du GPU (production S2/S3 en cours sur les deux machines) :
//! - le verrou GPU de la machine est pris PAR FENÊTRE (≤ 600 s) puis rendu : un chantier qui
//!   attend le verrou passe entre deux fenêtres ;
//! - un budget (`--budget-s`, 600 par défaut) borne le temps passé SOUS VERROU ; l'attente du verrou
//!   (production en cours) n'est pas décomptée ; les cas non mesurés faute de budget sont listés comme
//!   tels, jamais omis en silence ;
//! - aucune écriture lourde : un seul petit JSON (`--sortie`), après la garde SSD.
//!
//! Gate de non-régression (ARCHITECTURE_UNIFIEE.md §6.7) : le débit F64 unifié (pas/s à une
//! chaîne, mêmes options que la production) doit valoir au moins 0,9 × celui du binaire de référence du profil
//! (clé `reference_bench` de `kvu.toml`), mesuré dans la MÊME fenêtre, au même L.

use super::usage;
use crate::commun::{BackendKind, ChampHote, Depart, Erreur, Graine, Machine, Precision, Resultat};
use crate::gpu::{Appareil, OptionsCompil};
use crate::machine::{self, GardeVerrou, Politique, VerrouGpu};
use crate::su2::{self, ConfigLot, NoyauxSu2, ParamChaine};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command as Proc;
use std::time::{Duration, Instant};

/// Seuil de la gate de non-régression : unifié ≥ 0,9 × référence.
pub const SEUIL_NON_REGRESSION: f64 = 0.9;
/// Fenêtre maximale d'un banc sous verrou (s).
pub const FENETRE_MAX_S: f64 = 600.0;
/// Surcoût estimé (construction, compilation, référence) ajouté au temps de mesure d'un cas (s).
const SURCOUT_CAS_S: f64 = 10.0;

// ------------------------------------------------------------------------------------------
// Ligne de commande
// ------------------------------------------------------------------------------------------

pub fn commandes() -> Vec<clap::Command> {
    use clap::{Arg, ArgAction, Command};
    let commun = |c: Command, defaut_l: &'static str| {
        c.arg(
            Arg::new("L")
                .long("L")
                .value_name("LISTE")
                .default_value(defaut_l)
                .help("tailles L (liste, L pair)"),
        )
        .arg(
            Arg::new("budget-s")
                .long("budget-s")
                .value_name("S")
                .default_value("600")
                .help("budget de temps sous verrou GPU (s) ; l'attente du verrou n'est pas comptée"),
        )
        .arg(
            Arg::new("cible-s")
                .long("cible-s")
                .value_name("S")
                .help("durée de mesure par cas (défaut : budget/cas, 1..30)"),
        )
        .arg(
            Arg::new("backend")
                .long("backend")
                .default_value("auto")
                .help("auto|hip|cuda|cpu"),
        )
        .arg(
            Arg::new("json")
                .long("json")
                .action(ArgAction::SetTrue)
                .help("JSON sur la sortie standard"),
        )
        .arg(
            Arg::new("sortie")
                .long("sortie")
                .value_name("FICHIER")
                .help("écrit aussi le JSON dans ce fichier"),
        )
        .arg(
            Arg::new("dry-run")
                .long("dry-run")
                .action(ArgAction::SetTrue)
                .help("affiche le plan sans toucher au GPU"),
        )
    };
    let su2 = commun(
        Command::new("su2").about("Générateur SU(2) : liens/s du bain thermique et de l'OR, pas/s"),
        "8,16,32",
    )
    .arg(
        Arg::new("precision")
            .long("precision")
            .default_value("f64,f32")
            .help("f64,f32"),
    )
    .arg(
        Arg::new("chaines")
            .long("chaines")
            .default_value("1,16")
            .help("tailles de lot"),
    )
    .arg(Arg::new("beta").long("beta").default_value("2.5"))
    .arg(Arg::new("nor").long("nor").default_value("4"))
    .arg(
        Arg::new("options")
            .long("options")
            .default_value("compat-prod")
            .help("compat-prod|ref-k1"),
    )
    .arg(
        Arg::new("ref-bin")
            .long("ref-bin")
            .value_name("BIN|auto|aucune")
            .default_value("auto")
            .help("su2_hip / su2_cuda de référence (gate de non-régression F64)"),
    )
    .arg(
        Arg::new("sans-mur")
            .long("sans-mur")
            .action(ArgAction::SetTrue)
            .help("ne mesure pas le temps mur de su2 gen (seulement les noyaux HB/OR)"),
    );
    let cond = commun(
        Command::new("cond").about("Bain conditionnel : balayages/s"),
        "16,24,32",
    )
    .arg(
        Arg::new("mode")
            .long("mode")
            .default_value("rapide,ref_rng")
            .help("rapide,ref_rng"),
    )
    .arg(
        Arg::new("therm")
            .long("therm")
            .value_name("N")
            .default_value("300")
            .help("pas de thermalisation (HB + 4 OR, β = 2,3) du champ de départ : le coût du bain dépend de l'état"),
    )
    .arg(
        Arg::new("cfg")
            .long("cfg")
            .value_name("X.npy")
            .help("configuration de départ (ex. une configuration de production) pour les cas de même L"),
    );
    let fibre = commun(
        Command::new("fibre").about("Mesure de fibre : F, D, blk par seconde"),
        "16,24,32",
    );
    let obs = commun(
        Command::new("obs").about("Observables : boucles de Wilson et flot"),
        "16,24,32",
    );
    vec![Command::new("bench")
        .about("Bancs : su2 | cond | fibre | obs ; --budget-s 600 ; --json")
        .long_about("Bancs GPU sous verrou, par fenêtres de 10 minutes au plus, GPU rendu entre deux fenêtres.")
        .subcommand_required(true)
        .subcommand(su2)
        .subcommand(cond)
        .subcommand(fibre)
        .subcommand(obs)]
}

// ------------------------------------------------------------------------------------------
// Fonctions pures (testées)
// ------------------------------------------------------------------------------------------

/// Liste séparée par des virgules, analysée par `f`.
pub fn liste<T>(s: &str, mut f: impl FnMut(&str) -> Resultat<T>) -> Resultat<Vec<T>> {
    let v: Vec<T> = s
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(&mut f)
        .collect::<Resultat<_>>()?;
    if v.is_empty() {
        return Err(usage(format!("liste vide « {s} »")));
    }
    Ok(v)
}

pub fn liste_usize(s: &str) -> Resultat<Vec<usize>> {
    liste(s, |x| {
        x.parse::<usize>()
            .map_err(|_| usage(format!("entier attendu, reçu « {x} »")))
    })
}

pub fn valider_l(l: usize) -> Resultat<usize> {
    if l < 4 || !l.is_multiple_of(2) {
        return Err(usage(format!("L = {l} : L pair ≥ 4 requis")));
    }
    Ok(l)
}

/// Décision du fenêtrage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    Continuer,
    /// Rendre le GPU puis reprendre le verrou : la fenêtre courante est pleine.
    NouvelleFenetre,
    /// Le budget total ne permet plus ce cas.
    BudgetEpuise,
}

/// Temps imputé au budget : la somme des fenêtres tenues sous verrou, SANS l'attente du verrou. Le GPU est
/// partagé avec la production (fenêtres de ~10 min) : attendre son tour ne doit pas épuiser le budget
/// (04/10 : 566 s d'un budget de 600 s perdues en attente, 12 cas sur 16 non mesurés).
#[derive(Debug, Default)]
pub struct Chrono {
    clos_s: f64,
    ouverte: Option<Instant>,
}

impl Chrono {
    /// Ouvre une fenêtre (à appeler APRÈS la prise du verrou) ; ferme la précédente si besoin.
    pub fn ouvrir(&mut self) {
        self.fermer();
        self.ouverte = Some(Instant::now());
    }

    /// Ferme la fenêtre courante (à appeler AVANT de rendre le verrou).
    pub fn fermer(&mut self) {
        if let Some(t) = self.ouverte.take() {
            self.clos_s += t.elapsed().as_secs_f64();
        }
    }

    /// Temps consommé : fenêtres closes + fenêtre ouverte.
    pub fn consomme_s(&self) -> f64 {
        self.clos_s + self.ouverte.map_or(0.0, |t| t.elapsed().as_secs_f64())
    }
}

/// Règle de fenêtrage : budget total et fenêtre maximale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fenetrage {
    pub budget_s: f64,
    pub fenetre_s: f64,
}

impl Fenetrage {
    pub fn nouveau(budget_s: f64, fenetre_politique_s: f64) -> Fenetrage {
        Fenetrage {
            budget_s,
            fenetre_s: fenetre_politique_s.min(FENETRE_MAX_S),
        }
    }

    /// `total_s` : temps déjà consommé ; `fenetre_ecoulee_s` : depuis la prise du verrou ;
    /// `cout_s` : coût estimé du prochain cas.
    pub fn decider(&self, total_s: f64, fenetre_ecoulee_s: f64, cout_s: f64) -> Decision {
        if total_s + cout_s > self.budget_s {
            Decision::BudgetEpuise
        } else if fenetre_ecoulee_s > 0.0 && fenetre_ecoulee_s + cout_s > self.fenetre_s {
            Decision::NouvelleFenetre
        } else {
            Decision::Continuer
        }
    }
}

/// Durée de mesure par cas : celle demandée, sinon budget/cas bornée à [1, 30] s.
pub fn cible_par_cas(budget_s: f64, ncas: usize, demandee: Option<f64>) -> f64 {
    match demandee {
        Some(c) => c.max(0.01),
        None => (budget_s / ncas.max(1) as f64 - SURCOUT_CAS_S).clamp(1.0, 30.0),
    }
}

/// Nombre d'itérations du prochain lot pour viser `reste_s`, à partir du temps par itération mesuré.
pub fn taille_lot(t_par_iter_s: f64, reste_s: f64) -> u64 {
    if t_par_iter_s.is_nan() || t_par_iter_s <= 0.0 || reste_s <= 0.0 {
        return 1;
    }
    ((reste_s / t_par_iter_s).floor() as u64).clamp(1, 2000)
}

/// Mesure `f` pendant ~`cible_s` secondes ; `synchro` ferme chaque lot. Rend (itérations, secondes).
/// Un lot d'échauffement (non compté) précède la mesure.
pub fn chronometrer(
    cible_s: f64,
    f: &mut dyn FnMut() -> Resultat<()>,
    synchro: &dyn Fn() -> Resultat<()>,
) -> Resultat<(u64, f64)> {
    f()?;
    synchro()?;
    let debut = Instant::now();
    let (mut n, mut t_lot) = (0u64, 1u64);
    loop {
        for _ in 0..t_lot {
            f()?;
        }
        synchro()?;
        n += t_lot;
        let ecoule = debut.elapsed().as_secs_f64();
        if ecoule >= cible_s {
            return Ok((n, ecoule));
        }
        t_lot = taille_lot(ecoule / n as f64, cible_s - ecoule);
    }
}

/// Liens mis à jour par sweep : 4·V par chaîne.
pub fn liens_par_sweep(l: usize, chaines: usize) -> f64 {
    4.0 * (l as f64).powi(4) * chaines as f64
}

/// Octets du champ résident d'un lot : chaînes × 4V liens × 4 composantes.
pub fn octets_champ(l: usize, chaines: usize, p: Precision) -> u64 {
    let b = match p {
        Precision::F64 => 8,
        Precision::F32 => 4,
    };
    chaines as u64 * 4 * (l as u64).pow(4) * 4 * b
}

/// Résultat d'un banc de référence su2_hip / su2_cuda (ligne `BENCH …`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefBench {
    pub hb_s: f64,
    pub or_s: f64,
}

impl RefBench {
    pub fn liens_hb(&self, l: usize) -> f64 {
        liens_par_sweep(l, 1) / self.hb_s
    }
    pub fn liens_or(&self, l: usize) -> f64 {
        liens_par_sweep(l, 1) / self.or_s
    }
    /// Secondes par pas = 1 HB + nor OR.
    pub fn s_par_pas(&self, nor: usize) -> f64 {
        self.hb_s + nor as f64 * self.or_s
    }
}

/// Analyse `BENCH L=32 … hb_s_par_sweep=4.08e-02 or_s_par_sweep=1.33e-02 …`.
pub fn analyser_ligne_bench(texte: &str) -> Option<RefBench> {
    let ligne = texte.lines().find(|l| l.starts_with("BENCH"))?;
    let val = |cle: &str| -> Option<f64> {
        ligne
            .split_whitespace()
            .find_map(|t| t.strip_prefix(cle)?.strip_prefix('=')?.parse::<f64>().ok())
    };
    let (hb_s, or_s) = (val("hb_s_par_sweep")?, val("or_s_par_sweep")?);
    (hb_s > 0.0 && or_s > 0.0).then_some(RefBench { hb_s, or_s })
}

/// Rapport pas/s unifié sur référence, et verdict de la gate.
pub fn ratio_non_regression(unifie_s_par_pas: f64, ref_s_par_pas: f64) -> (f64, bool) {
    let r = ref_s_par_pas / unifie_s_par_pas;
    (r, r >= SEUIL_NON_REGRESSION)
}

/// Binaire de référence par défaut selon la machine : clé `reference_bench` du profil de `kvu.toml` (absente ⇒ None).
pub fn reference_par_defaut(m: Machine, home: &Path) -> Option<PathBuf> {
    let p = crate::config::actuelle().profil(m);
    crate::config::ProfilConfig::chemin_optionnel(&p.reference_bench, home)
}

/// En-tête d'un document de banc.
pub struct Entete<'a> {
    pub famille: &'a str,
    pub machine: &'a str,
    pub backend: BackendKind,
    pub arch: &'a str,
    pub budget_s: f64,
    pub duree_s: f64,
    /// Temps passé sous verrou (imputé au budget) ; `duree_s - sous_verrou_s` ≈ attente du verrou.
    pub sous_verrou_s: f64,
    pub fenetres: u32,
}

/// Document JSON final.
pub fn document(e: &Entete<'_>, mesures: Vec<Value>) -> Value {
    let gate = mesures
        .iter()
        .filter_map(|m| m.get("gate_non_regression").and_then(Value::as_bool))
        .collect::<Vec<_>>();
    json!({
        "schema": "kvu-banc/2",
        "banc": e.famille,
        "machine": e.machine,
        "backend": e.backend.nom(),
        "arch": e.arch,
        "budget_s": e.budget_s,
        "duree_s": e.duree_s,
        "sous_verrou_s": e.sous_verrou_s,
        "fenetres": e.fenetres,
        "gate_non_regression": if gate.is_empty() { Value::Null } else { Value::Bool(gate.iter().all(|&g| g)) },
        "mesures": mesures,
    })
}

/// Écrit le JSON (aucune garde ici : l'appelant l'a faite).
pub fn ecrire_json(chemin: &Path, doc: &Value) -> Resultat<()> {
    let texte = serde_json::to_string_pretty(doc).map_err(|e| Erreur::Format(format!("JSON : {e}")))?;
    std::fs::write(chemin, texte + "\n").map_err(|e| Erreur::io(format!("écriture de {}", chemin.display()), e))
}

// ------------------------------------------------------------------------------------------
// Exécution
// ------------------------------------------------------------------------------------------

/// Un cas de banc planifié.
#[derive(Clone, Debug, PartialEq)]
pub struct Cas {
    pub l: usize,
    pub precision: Precision,
    pub chaines: usize,
    /// Pour cond : mode ; vide sinon.
    pub mode: String,
}

impl Cas {
    pub fn etiquette(&self, famille: &str) -> String {
        match famille {
            "su2" => format!("su2 L={} {} K={}", self.l, self.precision.nom(), self.chaines),
            "cond" => format!("cond L={} {}", self.l, self.mode),
            f => format!("{f} L={}", self.l),
        }
    }
}

/// Contexte d'exécution : appareil, politique, verrou et fenêtre courante.
struct Contexte {
    backend: BackendKind,
    app: Option<Appareil>,
    pol: Politique,
    machine: Machine,
    garde: Option<GardeVerrou>,
    debut_fenetre: Instant,
    chrono: Chrono,
    fenetres: u32,
}

impl Contexte {
    fn prendre_verrou(&mut self) -> Resultat<()> {
        if self.backend != BackendKind::Cpu && self.garde.is_none() {
            let v = VerrouGpu::nouveau(&self.pol.verrou_gpu);
            // un banc exige l'exclusivité : refus immédiat sous gpu_partage (docs/MACHINES.md §8)
            self.garde = Some(v.acquerir_exclusif(Duration::from_secs(self.pol.attente_verrou_s))?);
        }
        self.debut_fenetre = Instant::now();
        self.chrono.ouvrir();
        self.fenetres += 1;
        Ok(())
    }

    /// Rend le GPU (le verrou est libéré par la destruction de la garde) et laisse la main aux autres.
    fn rendre_verrou(&mut self) {
        self.chrono.fermer();
        self.garde = None;
        std::thread::sleep(Duration::from_secs(2));
    }
}

fn ouvrir_contexte(m: &clap::ArgMatches) -> Resultat<Contexte> {
    let profil = machine::detecter();
    let pol = machine::politique(&profil);
    let demande = m.get_one::<String>("backend").map(String::as_str).unwrap_or("auto");
    let backend = if demande == "auto" {
        machine::choisir_backend(&profil)?
    } else {
        demande.parse::<BackendKind>()?
    };
    let app = if backend == BackendKind::Cpu {
        None
    } else {
        Some(crate::gpu::ouvrir(backend, 0)?)
    };
    Ok(Contexte {
        backend,
        app,
        pol,
        machine: profil.machine,
        garde: None,
        debut_fenetre: Instant::now(),
        chrono: Chrono::default(),
        fenetres: 0,
    })
}

fn flottant(m: &clap::ArgMatches, nom: &str) -> Resultat<f64> {
    let s = m
        .get_one::<String>(nom)
        .ok_or_else(|| usage(format!("--{nom} manquant")))?;
    s.parse::<f64>()
        .ok()
        .filter(|x| x.is_finite())
        .ok_or_else(|| usage(format!("--{nom} : nombre attendu, reçu « {s} »")))
}

/// Planifie les cas d'une famille depuis les options.
pub fn planifier(famille: &str, m: &clap::ArgMatches) -> Resultat<Vec<Cas>> {
    let ls = liste_usize(m.get_one::<String>("L").map(String::as_str).unwrap_or(""))?;
    let ls = ls.into_iter().map(valider_l).collect::<Resultat<Vec<_>>>()?;
    let mut cas = Vec::new();
    match famille {
        "su2" => {
            let ps = liste(
                m.get_one::<String>("precision").map(String::as_str).unwrap_or(""),
                |x| x.parse::<Precision>(),
            )?;
            let ks = liste_usize(m.get_one::<String>("chaines").map(String::as_str).unwrap_or(""))?;
            for &l in &ls {
                for &p in &ps {
                    for &k in &ks {
                        if k == 0 {
                            return Err(usage("--chaines : 0 interdit"));
                        }
                        cas.push(Cas {
                            l,
                            precision: p,
                            chaines: k,
                            mode: String::new(),
                        });
                    }
                }
            }
        }
        "cond" => {
            let modes = liste(
                m.get_one::<String>("mode").map(String::as_str).unwrap_or(""),
                |x| match x {
                    "rapide" | "ref_rng" => Ok(x.to_string()),
                    autre => Err(usage(format!("mode « {autre} » inconnu (rapide|ref_rng)"))),
                },
            )?;
            for &l in &ls {
                for md in &modes {
                    cas.push(Cas {
                        l,
                        precision: Precision::F64,
                        chaines: 1,
                        mode: md.clone(),
                    });
                }
            }
        }
        _ => {
            for &l in &ls {
                cas.push(Cas {
                    l,
                    precision: Precision::F64,
                    chaines: 1,
                    mode: String::new(),
                });
            }
        }
    }
    Ok(cas)
}

fn config_lot(l: usize, k: usize, beta: f64, nor: usize, p: Precision, options: OptionsCompil) -> ConfigLot {
    ConfigLot {
        l,
        chaines: (0..k)
            .map(|c| ParamChaine {
                beta,
                graine: Graine(20_261_004_000 + c as u64),
                depart: Depart::Chaud,
            })
            .collect(),
        nor,
        precision: p,
        options,
        mutant_metro: false,
    }
}

fn options_su2(nom: &str, p: Precision, backend: BackendKind) -> Resultat<OptionsCompil> {
    Ok(match (p, nom) {
        (Precision::F32, _) => OptionsCompil::rapide(),
        (Precision::F64, "compat-prod") => OptionsCompil::compat_production(backend),
        (Precision::F64, "ref-k1") => OptionsCompil::reference_k1(),
        (_, autre) => return Err(usage(format!("--options « {autre} » inconnu (compat-prod|ref-k1)"))),
    })
}

/// Banc de référence externe (su2_hip / su2_cuda), exécuté dans la fenêtre courante sous le même verrou.
fn mesurer_reference(bin: &Path, l: usize, beta: f64, cible_s: f64) -> Resultat<RefBench> {
    let lancer = |n: u64| -> Resultat<(RefBench, f64)> {
        let t = Instant::now();
        let sortie = Proc::new(bin)
            .args([
                "--bench",
                "--L",
                &l.to_string(),
                "--beta",
                &beta.to_string(),
                "--nsweeps",
                &n.to_string(),
            ])
            .output()
            .map_err(|e| Erreur::io(format!("lancement de {}", bin.display()), e))?;
        let txt = String::from_utf8_lossy(&sortie.stdout).into_owned() + &String::from_utf8_lossy(&sortie.stderr);
        let r = analyser_ligne_bench(&txt)
            .ok_or_else(|| Erreur::Format(format!("{} : ligne BENCH absente ou illisible", bin.display())))?;
        Ok((r, t.elapsed().as_secs_f64()))
    };
    let (r0, t0) = lancer(10)?;
    // La seconde passe vise `cible_s` ; la première suffit si elle dure déjà assez.
    let n = taille_lot(t0 / 10.0, cible_s).max(10);
    if t0 >= cible_s || n <= 10 {
        return Ok(r0);
    }
    Ok(lancer(n)?.0)
}

fn mesure_su2(ctx: &Contexte, cas: &Cas, m: &clap::ArgMatches, cible_s: f64) -> Resultat<Value> {
    let beta = flottant(m, "beta")?;
    let nor = m
        .get_one::<String>("nor")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| usage("--nor : entier attendu"))?;
    let opt_nom = m
        .get_one::<String>("options")
        .map(String::as_str)
        .unwrap_or("compat-prod");
    let options = options_su2(opt_nom, cas.precision, ctx.backend)?;
    let libre_avant = ctx.app.as_ref().and_then(|a| a.memoire_libre_octets().ok());
    let mut noy: Box<dyn NoyauxSu2> = su2::construire(
        ctx.app.as_ref(),
        config_lot(cas.l, cas.chaines, beta, nor, cas.precision, options.clone()),
    )?;
    noy.init()?;
    let libre_apres = ctx.app.as_ref().and_then(|a| a.memoire_libre_octets().ok());
    // Thermalisation légère : 5 pas avant de chronométrer (le débit ne dépend pas de l'état, la convergence de l'OR si).
    for _ in 0..5 {
        noy.sweep_hb()?;
        for _ in 0..nor {
            noy.sweep_or()?;
        }
    }
    noy.synchroniser()?;
    let (n_hb, t_hb) = {
        let cellule = std::cell::RefCell::new(&mut noy);
        chronometrer(cible_s / 2.0, &mut || cellule.borrow_mut().sweep_hb(), &|| {
            cellule.borrow().synchroniser()
        })?
    };
    let (n_or, t_or) = {
        let cellule = std::cell::RefCell::new(&mut noy);
        chronometrer(cible_s / 2.0, &mut || cellule.borrow_mut().sweep_or(), &|| {
            cellule.borrow().synchroniser()
        })?
    };
    let (hb_s, or_s) = (t_hb / n_hb as f64, t_or / n_or as f64);
    // Le lot est rendu avant la mesure au temps mur, qui en construit un autre (mémoire GPU d'un seul lot à la fois).
    drop(noy);
    let liens = liens_par_sweep(cas.l, cas.chaines);
    let s_pas = hb_s + nor as f64 * or_s;
    let mut v = json!({
        "schema_mesure": "kvu-banc-su2/2", "mode_mesure": "micro_seul",
        "famille": "su2", "L": cas.l, "precision": cas.precision.nom(), "chaines": cas.chaines, "beta": beta, "nor": nor,
        "options": opt_nom,
        "hb_s_par_sweep": hb_s, "or_s_par_sweep": or_s,
        "liens_par_s_hb": liens / hb_s, "liens_par_s_or": liens / or_s,
        "pas_par_s": 1.0 / s_pas, "pas_chaines_par_s": cas.chaines as f64 / s_pas,
        "memoire_champ_mo": octets_champ(cas.l, cas.chaines, cas.precision) as f64 / 1e6,
        "memoire_gpu_delta_mo": match (libre_avant, libre_apres) {
            (Some(a), Some(b)) if a >= b => json!((a - b) as f64 / 1e6),
            _ => Value::Null,
        },
    });
    // Les microbancs restent les seules entrées de la comparaison à la référence externe.
    v["microbench"] = json!({
        "hb_s_par_sweep": hb_s, "or_s_par_sweep": or_s,
        "liens_par_s_hb": liens / hb_s, "liens_par_s_or": liens / or_s,
        "s_par_pas": s_pas, "pas_par_s": 1.0 / s_pas,
        "pas_chaines_par_s": cas.chaines as f64 / s_pas,
    });
    // Le couple apparié mesure deux intervalles imbriqués du même run instrumenté.
    // S/D contrôle la cohérence de ce profilage, pas sa performance contre le microbanc.
    if !m.get_flag("sans-mur") {
        let profil = temps_apparie_su2(ctx, cas, beta, nor, &options, s_pas, cible_s / 2.0)?;
        let n = profil.pas_executes as f64;
        let d = profil.secondes_totales();
        let s = profil.secondes_pas();
        v["mode_mesure"] = json!("apparie");
        v["pas_par_s"] = json!(n / s);
        v["pas_chaines_par_s"] = json!(cas.chaines as f64 * n / s);
        v["profilage"] = serde_json::to_value(&profil).map_err(|e| Erreur::Format(e.to_string()))?;
        v["profilage"]["schema"] = json!("kvu-profil-run/1");
        v["profilage"]["unite_durees"] = json!("nanoseconde monotone écoulée");
        v["profilage"]["perimetre_D"] = json!("appel complet exécuté sous verrou déjà acquis : admission, construction, initialisation, pas, mesures/CSV, contrôle, clôture et destructeurs ; hors attente du verrou, JSON de banc et nettoyage");
        v["profilage"]["perimetre_S"] = json!("somme des intervalles disjoints HB + nor OR réellement exécutés, drain préalable hors S, synchronisation finale dans S ; horloges et synchronisations non soustraites");
        v["mur"] = json!({
            "s_par_pas": d / n, "pas_par_s": n / d, "pas_chaines_par_s": cas.chaines as f64 * n / d,
            "pas_mesures": profil.pas_executes,
            "estimation": "run_instrumente_apparie",
            "methode": "N/D du run instrumenté complet ; aucune soustraction de runs courts ni correction des valeurs ; ne représente pas le débit exact de production non instrumentée",
        });
        v["rapport_mur_sur_noyaux"] = json!(s / d);
    }
    // Gate de non-régression : F64, une chaîne, référence disponible.
    if cas.precision == Precision::F64 && cas.chaines == 1 {
        let spec = m.get_one::<String>("ref-bin").map(String::as_str).unwrap_or("auto");
        let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
        let bin = match spec {
            "aucune" => None,
            "auto" => reference_par_defaut(ctx.machine, &home),
            autre => Some(PathBuf::from(autre)),
        };
        if let Some(bin) = bin.filter(|b| b.is_file()) {
            let r = mesurer_reference(&bin, cas.l, beta, cible_s / 2.0)?;
            let (ratio, ok) = ratio_non_regression(s_pas, r.s_par_pas(nor));
            v["reference"] = json!({
                "binaire": bin.display().to_string(),
                "hb_s_par_sweep": r.hb_s, "or_s_par_sweep": r.or_s,
                "liens_par_s_hb": r.liens_hb(cas.l), "liens_par_s_or": r.liens_or(cas.l),
                "pas_par_s": 1.0 / r.s_par_pas(nor),
            });
            v["ratio_pas_unifie_sur_reference"] = json!(ratio);
            v["gate_non_regression"] = json!(ok);
        }
    }
    Ok(v)
}

// Ancien estimateur conservé uniquement pour ses tests historiques et son contre-exemple.
// Il ne gouverne plus la publication : des chronométrages séparés ne sont pas imbriqués.
#[cfg(test)]
const TOLERANCE_MUR_NOYAUX: f64 = 1.5;

#[cfg(test)]
struct EstimationMur {
    s_par_pas: f64,
    pas_mesures: u64,
    estimation: &'static str,
}

/// Les pics de coûts fixes sont un bruit positif : le minimum est l'estimateur robuste.
#[cfg(test)]
fn estimer_mur(t_courts: &[f64], t_longs: &[f64], n0: u64, n1: u64, s_pas_noyaux: f64) -> EstimationMur {
    let t0 = t_courts.iter().copied().fold(f64::INFINITY, f64::min);
    let t1 = t_longs.iter().copied().fold(f64::INFINITY, f64::min);
    let d = (t1 - t0) / (n1 - n0) as f64;
    if d.is_finite() && d >= s_pas_noyaux / TOLERANCE_MUR_NOYAUX {
        EstimationMur {
            s_par_pas: d,
            pas_mesures: n1 - n0,
            estimation: "difference",
        }
    } else {
        EstimationMur {
            s_par_pas: t1 / n1 as f64,
            pas_mesures: n1,
            estimation: "borne_haute_t_sur_n",
        }
    }
}

/// Un seul run instrumenté ; sa longueur dépend du microbanc, jamais du ratio final.
fn temps_apparie_su2(
    ctx: &Contexte,
    cas: &Cas,
    beta: f64,
    nor: usize,
    options: &OptionsCompil,
    s_pas_noyaux: f64,
    cible_s: f64,
) -> Resultat<crate::su2::run::ProfilRun> {
    use crate::su2::run::{self, ConfigRun};
    // Sur le disque (pas /tmp, un tmpfs en RAM refusé par la garde SSD) : petits runs sans dump, supprimés.
    let base = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(format!(".cache/kevinotron/bench_mur_{}", std::process::id()));
    let n = ((cible_s / s_pas_noyaux.max(1e-6)) as u64).clamp(40, 20_000);
    let outdir = base.join("apparie");
    // Le dossier appartient à ce processus ; une présence inattendue n'est pas réécrite.
    if outdir.exists() {
        return Err(Erreur::Config(format!("banc profilé : dossier existant {}", outdir.display())));
    }
    let cfg = ConfigRun {
        lot: config_lot(cas.l, cas.chaines, beta, nor, cas.precision, options.clone()),
        therm: 0,
        nmeas: n,
        skip: 1,
        dump_every: 0,
        dump_initial: false,
        format_dump: "c16".into(),
        zstd: false,
        reprise_every: 0,
        outdir: outdir.clone(),
        bord: None,
    };
    let (_, termine, profil) = run::executer_profile(ctx.app.as_ref(), &cfg, &mut [], None, ctx.garde.as_ref())?;
    profil.valider(n, cas.chaines, nor, true)?;
    if !termine { return Err(Erreur::Config("banc profilé interrompu".into())); }
    let _ = std::fs::remove_dir_all(&base);
    Ok(profil)
}

/// Champ de départ de L donné : généré sur place (GPU si disponible), quelques pas pour sortir de Haar.
fn champ_de_depart(ctx: &Contexte, l: usize) -> Resultat<(Box<dyn NoyauxSu2>, ChampHote)> {
    let p = Precision::F64;
    let mut noy = su2::construire(
        ctx.app.as_ref(),
        config_lot(l, 1, 2.3, 4, p, OptionsCompil::compat_production(ctx.backend)),
    )?;
    noy.init()?;
    for _ in 0..20 {
        noy.sweep_hb()?;
        for _ in 0..4 {
            noy.sweep_or()?;
        }
    }
    noy.synchroniser()?;
    let hote = noy.telecharger(0)?;
    Ok((noy, hote))
}

/// Champ de départ THERMALISÉ du banc cond (vérificateur du 04/10 : avec 20 pas depuis Haar, `ref_rng` coûtait ×2
/// à ×2,5 ce qu'il coûte sur des configurations thermalisées : le nombre de tirages numpy rejoués dépend de
/// l'état). `therm` pas HB + 4 OR à β = 2,3, en F32 sur le GPU (rapide), renormalisés en f64 ; ou la
/// configuration `cfg` si son L est celui du cas. Rend (champ, description, plaquette moyenne du départ).
fn champ_thermalise(ctx: &Contexte, l: usize, therm: u64, cfg: Option<&Path>) -> Resultat<(ChampHote, String, f64)> {
    let plaquette = |c: &ChampHote| -> f64 {
        let pv = crate::fibre::cpu::plaquettes_valeurs(c);
        pv.iter().sum::<f64>() / pv.len() as f64
    };
    if let Some(p) = cfg {
        let (_, lc) = crate::stockage::npy::inspecter_champ(p)?;
        if lc == l {
            let c = crate::stockage::npy::lire_champ(p)?;
            let pl = plaquette(&c);
            return Ok((c, format!("configuration {}", p.display()), pl));
        }
    }
    let (p, options) = match ctx.app {
        Some(_) => (Precision::F32, OptionsCompil::rapide()),
        None => (Precision::F64, OptionsCompil::compat_production(ctx.backend)),
    };
    let mut noy = su2::construire(ctx.app.as_ref(), config_lot(l, 1, 2.3, 4, p, options))?;
    noy.init()?;
    for _ in 0..therm {
        noy.sweep_hb()?;
        for _ in 0..4 {
            noy.sweep_or()?;
        }
    }
    noy.synchroniser()?;
    let mut c = noy.telecharger(0)?;
    crate::stockage::npy::renormaliser(&mut c)?;
    let pl = plaquette(&c);
    Ok((c, format!("thermalisé : {therm} pas HB + 4 OR, β = 2,3, {}", p.nom()), pl))
}

fn mesure_cond(ctx: &Contexte, cas: &Cas, m: &clap::ArgMatches, cible_s: f64) -> Resultat<Value> {
    use crate::cond::{geometrie::v_paires, GenerateurNumpy, GeometrieCond, ModeCond, MoteurCond, SourceAlea};
    use std::sync::Arc;
    let therm: u64 = m
        .get_one::<String>("therm")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| usage("--therm : entier attendu"))?;
    let cfg = m.get_one::<String>("cfg").map(PathBuf::from);
    let (hote, depart, plaquette) = champ_thermalise(ctx, cas.l, therm, cfg.as_deref())?;
    let geo = Arc::new(GeometrieCond::nouvelle(cas.l)?);
    let vfix = v_paires(&hote, &geo);
    let (mode, alea) = match cas.mode.as_str() {
        "ref_rng" => (
            ModeCond::RefRng,
            SourceAlea::Numpy(GenerateurNumpy::depuis_graine(2709)),
        ),
        _ => (
            ModeCond::Rapide,
            SourceAlea::Philox {
                graine: Graine(2709),
                sweep_id: 0,
            },
        ),
    };
    let mut moteur = MoteurCond::nouveau(ctx.app.as_ref(), geo, &hote, vfix, 2.3, alea)?;
    let app = ctx.app.clone();
    let synchro = move || match &app {
        Some(a) => a.synchroniser(),
        None => Ok(()),
    };
    let (n, t) = chronometrer(cible_s, &mut || moteur.sweep(), &synchro)?;
    Ok(json!({
        "famille": "cond", "L": cas.l, "mode": cas.mode, "beta": 2.3,
        "sweep_s": t / n as f64, "sweeps_par_s": n as f64 / t, "liens_par_s": 4.0 * (cas.l as f64).powi(4) * n as f64 / t,
        "mode_enum": format!("{mode:?}"),
        "depart": depart, "plaquette_depart": plaquette,
    }))
}

fn mesure_fibre(ctx: &Contexte, cas: &Cas, cible_s: f64) -> Resultat<Value> {
    use crate::cond::GeometrieCond;
    use crate::fibre::{poids_fourier, Mesureur};
    use std::sync::Arc;
    let (noy, hote) = champ_de_depart(ctx, cas.l)?;
    let geo = Arc::new(GeometrieCond::nouvelle(cas.l)?);
    let w = poids_fourier(cas.l, 1);
    let mut mes = Mesureur::nouveau(ctx.app.as_ref(), Arc::clone(&geo), &w)?;
    let champ = noy.champ_gpu();
    let app = ctx.app.clone();
    let synchro = move || match &app {
        Some(a) => a.synchroniser(),
        None => Ok(()),
    };
    let (n, t) = chronometrer(
        cible_s / 2.0,
        &mut || mes.mesurer(champ, 0, Some(&hote)).map(|_| ()),
        &synchro,
    )?;
    let (nf, tf) = chronometrer(
        cible_s / 2.0,
        &mut || mes.mesurer_f(champ, 0, Some(&hote)).map(|_| ()),
        &synchro,
    )?;
    Ok(json!({
        "famille": "fibre", "L": cas.l,
        "mesure_complete_s": t / n as f64, "mesures_completes_par_s": n as f64 / t,
        "mesure_f_s": tf / nf as f64, "mesures_f_par_s": nf as f64 / tf,
    }))
}

fn mesure_obs(ctx: &Contexte, cas: &Cas, cible_s: f64) -> Resultat<Value> {
    use crate::obs::{boucles_wilson, flot_wilson, ParamsFlot, ParamsWilson};
    let (noy, hote) = champ_de_depart(ctx, cas.l)?;
    let champ = noy.champ_gpu();
    let app = ctx.app.clone();
    let synchro = move || match &app {
        Some(a) => a.synchroniser(),
        None => Ok(()),
    };
    let pw = ParamsWilson {
        r_max: (cas.l / 2).min(8),
        t_max: (cas.l / 2).min(8),
        dir_temps: 3,
        toutes_orientations: false,
    };
    let (n, t) = chronometrer(
        cible_s / 2.0,
        &mut || boucles_wilson(ctx.app.as_ref(), champ, 0, Some(&hote), &pw).map(|_| ()),
        &synchro,
    )?;
    let pf = ParamsFlot {
        dt: 0.01,
        t_max: 0.05,
        mesure_chaque: 1,
        cible_t2e: 0.3,
        cible_w: 0.3,
    };
    let (nf, tf) = chronometrer(
        cible_s / 2.0,
        &mut || flot_wilson(ctx.app.as_ref(), champ, 0, Some(&hote), &pf).map(|_| ()),
        &synchro,
    )?;
    Ok(json!({
        "famille": "obs", "L": cas.l,
        "wilson_s": t / n as f64, "wilsons_par_s": n as f64 / t,
        "flot_5pas_s": tf / nf as f64, "pas_de_flot_par_s": 5.0 * nf as f64 / tf,
    }))
}

pub fn executer(nom: &str, m: &clap::ArgMatches) -> Resultat<i32> {
    debug_assert_eq!(nom, "bench");
    let (famille, sm) = m
        .subcommand()
        .ok_or_else(|| usage("bench : famille attendue (su2|cond|fibre|obs)"))?;
    let budget_s = flottant(sm, "budget-s")?;
    if budget_s <= 0.0 {
        return Err(usage("--budget-s doit être > 0"));
    }
    let cas = planifier(famille, sm)?;
    let demandee = match sm.get_one::<String>("cible-s") {
        Some(s) => Some(
            s.parse::<f64>()
                .map_err(|_| usage(format!("--cible-s : nombre attendu, reçu « {s} »")))?,
        ),
        None => None,
    };
    let cible_s = cible_par_cas(budget_s, cas.len(), demandee);
    if sm.get_flag("dry-run") {
        println!("plan du banc {famille} (rien n'est lancé) : budget {budget_s:.0} s, {cible_s:.1} s par cas, fenêtres ≤ {FENETRE_MAX_S:.0} s");
        for c in &cas {
            println!("  - {}", c.etiquette(famille));
        }
        return Ok(crate::commun::code::OK);
    }
    let mut ctx = ouvrir_contexte(sm)?;
    let regle = Fenetrage::nouveau(budget_s, ctx.pol.fenetre_banc_s as f64);
    let debut = Instant::now();
    ctx.prendre_verrou()?;
    let mut mesures: Vec<Value> = Vec::new();
    for c in &cas {
        let cout = cible_s + SURCOUT_CAS_S;
        let total = ctx.chrono.consomme_s();
        match regle.decider(total, ctx.debut_fenetre.elapsed().as_secs_f64(), cout) {
            Decision::BudgetEpuise => {
                mesures.push(json!({"cas": c.etiquette(famille), "statut": "non_mesure", "raison": "budget épuisé"}));
                continue;
            }
            Decision::NouvelleFenetre => {
                ctx.rendre_verrou();
                ctx.prendre_verrou()?;
            }
            Decision::Continuer => {}
        }
        let r = match famille {
            "su2" => mesure_su2(&ctx, c, sm, cible_s),
            "cond" => mesure_cond(&ctx, c, sm, cible_s),
            "fibre" => mesure_fibre(&ctx, c, cible_s),
            _ => mesure_obs(&ctx, c, cible_s),
        };
        match r {
            Ok(v) => {
                eprintln!("[bench] {} : mesuré", c.etiquette(famille));
                mesures.push(v);
            }
            // Un cas qui ne tient pas (mémoire, GPU) est consigné et le banc continue ; le code de sortie le dit.
            Err(e) => {
                eprintln!("[bench] {} : ÉCHEC : {e}", c.etiquette(famille));
                mesures.push(json!({"cas": c.etiquette(famille), "statut": "echec", "erreur": e.to_string()}));
            }
        }
    }
    ctx.rendre_verrou();
    let profil = machine::detecter();
    let arch = ctx.app.as_ref().map(|a| a.arch()).unwrap_or_default();
    let echecs = mesures
        .iter()
        .filter(|v| v.get("statut").and_then(Value::as_str) == Some("echec"))
        .count();
    let doc = document(
        &Entete {
            famille,
            machine: profil.machine.nom(),
            backend: ctx.backend,
            arch: &arch,
            budget_s,
            duree_s: debut.elapsed().as_secs_f64(),
            sous_verrou_s: ctx.chrono.consomme_s(),
            fenetres: ctx.fenetres,
        },
        mesures,
    );

    if let Some(chemin) = sm.get_one::<String>("sortie") {
        let chemin = Path::new(chemin);
        crate::stockage::GardeSsd::machine().verifier(chemin, 1 << 20)?;
        ecrire_json(chemin, &doc)?;
    }
    if sm.get_flag("json") {
        println!("{doc}");
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(&doc).map_err(|e| Erreur::Format(e.to_string()))?
        );
    }
    let regression = doc["gate_non_regression"] == Value::Bool(false);
    if regression {
        eprintln!("[bench] GATE DE NON-RÉGRESSION ROUGE : F64 unifié < 0,9 × référence");
    }
    Ok(if echecs > 0 || regression {
        crate::commun::code::VALIDATION
    } else {
        crate::commun::code::OK
    })
}

#[cfg(test)]
mod regression_repli_r6;

#[cfg(test)]
mod tests {
    use super::*;

    fn matches(args: &str) -> clap::ArgMatches {
        let cmd = commandes().remove(0);
        cmd.try_get_matches_from(args.split_whitespace()).unwrap()
    }

    #[test]
    fn listes_analysees_et_validees() {
        assert_eq!(liste_usize("8, 16,32").unwrap(), vec![8, 16, 32]);
        assert!(liste_usize("").is_err());
        assert!(liste_usize("8,x").is_err());
        assert!(valider_l(7).is_err());
        assert!(valider_l(2).is_err());
        assert_eq!(valider_l(48).unwrap(), 48);
    }

    #[test]
    fn fenetrage_borne_a_10_minutes_et_au_budget() {
        let f = Fenetrage::nouveau(1800.0, 7200.0);
        assert_eq!(
            f.fenetre_s, 600.0,
            "une fenêtre de politique plus longue est ramenée à 600 s"
        );
        assert_eq!(f.decider(0.0, 0.0, 40.0), Decision::Continuer);
        assert_eq!(f.decider(100.0, 580.0, 40.0), Decision::NouvelleFenetre);
        assert_eq!(f.decider(1790.0, 10.0, 40.0), Decision::BudgetEpuise);
        // une fenêtre vide accepte toujours un cas (jamais de boucle infinie de fenêtres)
        assert_eq!(f.decider(0.0, 0.0, 900.0), Decision::Continuer);
        assert_eq!(Fenetrage::nouveau(600.0, 300.0).fenetre_s, 300.0);
    }

    #[test]
    fn cible_par_cas_bornee() {
        assert_eq!(cible_par_cas(600.0, 4, None), 30.0);
        assert_eq!(cible_par_cas(60.0, 20, None), 1.0);
        assert_eq!(cible_par_cas(600.0, 40, Some(5.0)), 5.0);
        assert!(cible_par_cas(10.0, 0, None) >= 1.0);
    }

    #[test]
    fn taille_de_lot() {
        assert_eq!(taille_lot(0.0, 10.0), 1);
        assert_eq!(taille_lot(0.01, 1.0), 100);
        assert_eq!(taille_lot(1e-9, 10.0), 2000);
        assert_eq!(taille_lot(5.0, 1.0), 1);
        assert_eq!(taille_lot(0.1, -1.0), 1);
    }

    #[test]
    fn chronometrage_compte_les_iterations_et_synchronise() {
        let n = std::cell::Cell::new(0u64);
        let syn = std::cell::Cell::new(0u64);
        let (it, t) = chronometrer(
            0.05,
            &mut || {
                n.set(n.get() + 1);
                std::thread::sleep(Duration::from_millis(2));
                Ok(())
            },
            &|| {
                syn.set(syn.get() + 1);
                Ok(())
            },
        )
        .unwrap();
        assert!((0.05..1.0).contains(&t), "t = {t}");
        assert_eq!(n.get(), it + 1, "un tour d'échauffement non compté");
        assert!(syn.get() >= 2);
        // une erreur remonte
        let e = chronometrer(0.01, &mut || Err(Erreur::Gpu("x".into())), &|| Ok(()));
        assert!(matches!(e, Err(Erreur::Gpu(_))));
    }

    #[test]
    fn metrique_liens_et_memoire() {
        assert_eq!(liens_par_sweep(32, 1), 4.0 * 1_048_576.0);
        assert_eq!(liens_par_sweep(8, 16), 4.0 * 4096.0 * 16.0);
        // L = 64 FP32 : 1,07 Go (ARCHITECTURE_UNIFIEE.md §2)
        assert_eq!(octets_champ(64, 1, Precision::F32), 1_073_741_824);
        assert_eq!(octets_champ(64, 1, Precision::F64), 2_147_483_648);
    }

    #[test]
    fn ligne_bench_de_su2_hip_et_su2_cuda() {
        let hip = "# su2_hip : 1 GPU, Radeon 8060S Graphics gfx1151\nBENCH L=32 beta=2.5 V=1048576 hb_sweeps=100 hb_s_par_sweep=4.082823e-02 or_sweeps=100 or_s_par_sweep=1.334481e-02 liens_par_s_hb=1.0273e+08 liens_par_s_or=3.1430e+08 wall=5.493 P=0.6517536158\n";
        let r = analyser_ligne_bench(hip).unwrap();
        assert_eq!(r.hb_s, 4.082823e-02);
        assert_eq!(r.or_s, 1.334481e-02);
        // 4·V / hb_s = 1,0273e8 liens/s (ligne d'origine)
        assert!((r.liens_hb(32) / 1.0273e8 - 1.0).abs() < 1e-3, "{}", r.liens_hb(32));
        assert!((r.liens_or(32) / 3.1430e8 - 1.0).abs() < 1e-3);
        let cuda = "BENCH L=32 beta=2.5 V=1048576 hb_sweeps=100 hb_s_par_sweep=1.615371e-02 or_sweeps=100 or_s_par_sweep=7.963931e-03 liens_par_s_hb=2.5965e+08\n";
        assert!((analyser_ligne_bench(cuda).unwrap().liens_hb(32) / 2.5965e8 - 1.0).abs() < 1e-3);
        assert!(analyser_ligne_bench("rien").is_none());
        assert!(analyser_ligne_bench("BENCH L=8 hb_s_par_sweep=0 or_s_par_sweep=1").is_none());
        assert!((r.s_par_pas(4) - (4.082823e-02 + 4.0 * 1.334481e-02)).abs() < 1e-15);
    }

    #[test]
    fn gate_de_non_regression() {
        // unifié aussi rapide : ratio 1 ; 10 % plus lent : 0,909 ≥ 0,9 ; 12 % plus lent : rouge.
        assert!(ratio_non_regression(1.0, 1.0).1);
        let (r, ok) = ratio_non_regression(1.1, 1.0);
        assert!(ok && (r - 1.0 / 1.1).abs() < 1e-15);
        assert!(!ratio_non_regression(1.12, 1.0).1);
        assert!(ratio_non_regression(0.5, 1.0).1, "plus rapide que la référence");
    }

    #[test]
    fn reference_selon_la_machine() {
        let h = Path::new("/home/x");
        assert!(reference_par_defaut(Machine::Acemagic, h)
            .unwrap()
            .ends_with("bin/su2_hip"));
        assert_eq!(
            reference_par_defaut(Machine::Gamer, h).unwrap(),
            PathBuf::from("/home/x/ym4_calcul/su2_cuda/bin/su2_cuda")
        );
        assert!(reference_par_defaut(Machine::Inconnue, h).is_none());
    }

    #[test]
    fn planification_su2_produit_le_produit_cartesien() {
        let m = matches("bench su2 --L 8,16 --precision f64,f32 --chaines 1,16");
        let (_, sm) = m.subcommand().unwrap();
        let cas = planifier("su2", sm).unwrap();
        assert_eq!(cas.len(), 8);
        assert_eq!(
            cas[0],
            Cas {
                l: 8,
                precision: Precision::F64,
                chaines: 1,
                mode: String::new()
            }
        );
        assert_eq!(cas[0].etiquette("su2"), "su2 L=8 f64 K=1");
        let m = matches("bench su2 --L 7");
        assert!(planifier("su2", m.subcommand().unwrap().1).is_err());
        let m = matches("bench su2 --chaines 0");
        assert!(planifier("su2", m.subcommand().unwrap().1).is_err());
    }

    #[test]
    fn planification_cond_et_defauts_des_autres() {
        let m = matches("bench cond");
        let cas = planifier("cond", m.subcommand().unwrap().1).unwrap();
        assert_eq!(cas.len(), 6, "3 L × 2 modes");
        let m = matches("bench cond --mode quantique");
        assert!(planifier("cond", m.subcommand().unwrap().1).is_err());
        let m = matches("bench fibre");
        assert_eq!(planifier("fibre", m.subcommand().unwrap().1).unwrap().len(), 3);
        let m = matches("bench obs --L 24");
        assert_eq!(
            planifier("obs", m.subcommand().unwrap().1).unwrap()[0].etiquette("obs"),
            "obs L=24"
        );
    }

    #[test]
    fn defauts_de_budget_et_de_fenetre() {
        let m = matches("bench su2");
        let (_, sm) = m.subcommand().unwrap();
        assert_eq!(sm.get_one::<String>("budget-s").unwrap(), "600");
        assert_eq!(sm.get_one::<String>("L").unwrap(), "8,16,32");
        assert!(
            commandes().remove(0).try_get_matches_from(["bench"]).is_err(),
            "une famille est exigée"
        );
    }

    #[test]
    fn options_su2_par_precision() {
        let o = options_su2("compat-prod", Precision::F64, BackendKind::Hip).unwrap();
        assert_eq!(o, OptionsCompil::compat_production(BackendKind::Hip));
        assert_eq!(
            options_su2("ref-k1", Precision::F64, BackendKind::Cuda).unwrap(),
            OptionsCompil::reference_k1()
        );
        assert_eq!(
            options_su2("n-importe", Precision::F32, BackendKind::Hip).unwrap(),
            OptionsCompil::rapide()
        );
        assert!(options_su2("n-importe", Precision::F64, BackendKind::Hip).is_err());
    }

    #[test]
    fn document_json_et_gate_agregee() {
        let ent = |famille, machine, backend, arch, duree_s| Entete {
            famille,
            machine,
            backend,
            arch,
            budget_s: 600.0,
            duree_s,
            sous_verrou_s: duree_s / 2.0,
            fenetres: 1,
        };
        let ok = json!({"gate_non_regression": true});
        let ko = json!({"gate_non_regression": false});
        let e = ent("su2", "acemagic", BackendKind::Hip, "gfx1151", 12.0);
        assert_eq!(document(&e, vec![ok.clone(), ok.clone()])["gate_non_regression"], true);
        assert_eq!(document(&e, vec![ok, ko])["gate_non_regression"], false);
        let d = document(&ent("cond", "gamer", BackendKind::Cuda, "sm_120", 1.0), vec![]);
        assert!(d["gate_non_regression"].is_null());
        assert_eq!(d["backend"], "cuda");
        assert_eq!(d["sous_verrou_s"], 0.5);
    }

    #[test]
    fn le_budget_ne_compte_pas_l_attente_du_verrou() {
        let t0 = Instant::now();
        let mut c = Chrono::default();
        assert_eq!(c.consomme_s(), 0.0);
        c.ouvrir();
        std::thread::sleep(Duration::from_millis(30));
        c.fermer();
        let apres_premiere = c.consomme_s();
        std::thread::sleep(Duration::from_millis(120)); // attente du verrou : non comptée
        assert_eq!(c.consomme_s(), apres_premiere, "fenêtre fermée : rien ne court");
        c.ouvrir();
        std::thread::sleep(Duration::from_millis(30));
        let consomme = c.consomme_s();
        let mur = t0.elapsed().as_secs_f64();
        assert!(consomme >= 0.06, "{consomme}");
        assert!(mur - consomme >= 0.12, "l'attente a été imputée : mur {mur}, consommé {consomme}");
        c.ouvrir(); // rouvrir ferme la fenêtre courante sans la perdre
        assert!(c.consomme_s() >= consomme);
    }

    #[test]
    fn ecriture_du_json() {
        let p = std::env::temp_dir().join(format!("kvu_bench_{}.json", std::process::id()));
        ecrire_json(&p, &json!({"a": 1})).unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert_eq!(v["a"], 1);
        let _ = std::fs::remove_file(&p);
        assert!(ecrire_json(Path::new("/proc/inexistant/x.json"), &json!({})).is_err());
    }

    #[test]
    fn estimation_mur_par_difference_nominale() {
        let mur = estimer_mur(&[0.05, 0.06, 0.05], &[1.05], 10, 1010, 0.001);
        assert_eq!(mur.estimation, "difference");
        assert_eq!(mur.pas_mesures, 1000);
        assert!((mur.s_par_pas - 1.0e-3).abs() < 1e-12);
    }

    #[test]
    fn estimation_mur_borne_haute_si_le_run_court_est_plus_lent() {
        // L'ancienne formule sur ces données donnerait un débit > 1e9 pas/s.
        let mur = estimer_mur(&[1.2, 1.3, 1.25], &[1.0], 10, 924, 1.0 / 924.0);
        assert_eq!(mur.estimation, "borne_haute_t_sur_n");
        assert_eq!(mur.pas_mesures, 924);
        assert_eq!(mur.s_par_pas, 1.0 / 924.0);
        assert!(1.0 / mur.s_par_pas <= TOLERANCE_MUR_NOYAUX * 924.0);
    }

    #[test]
    fn estimation_mur_absorbe_un_pic_sur_un_run_court() {
        let mur = estimer_mur(&[2.0, 0.05, 0.05], &[1.05], 10, 1010, 0.001);
        assert_eq!(mur.estimation, "difference");
        assert_eq!(mur.pas_mesures, 1000);
        assert!((mur.s_par_pas - 1.0e-3).abs() < 1e-12);
    }
}
