//! Sous-commande `rg` : bloc de Bałaban L_b = 3, témoin de Swendsen b = 2, rapport de domaine, gates. Propriétaire : WP6
//! (partagée ensuite avec WP7 : option `--backend`, et WP8 : `rg flot`, une ligne chacun).
//!
//! ```text
//! kvu rg bloc --entree X.npy --sortie Y.npy [--fils N] [--geometrie 3centre|3coin] [--backend cpu|hip|cuda|auto]
//!             [--operateur 3centre|dec|ape:<α>:<n>|flot:<t>[:<ε>]|agr:<c>] [--hors-famille]                                (WP22)
//! kvu rg swendsen --entree X.npy --sortie Y.npy [--c 0.5] [--pas 1]
//! kvu rg domaine --entree X.npy [--rapport R.json] [--fils N] [--backend cpu|hip|cuda|auto]
//! kvu rg gates --fixtures D --donnees D [--grandl D] [--rapports D] [--fils N] [--seul ID,…] [--sans-mutants] [--json]
//!              [--certifier [FICHIER]]
//! kvu rg gates-gpu --backend hip|cuda [--fixtures D] [--grandl D] [--sortie D] [--fils N] [--repetitions N] [--sans-mutants] [--json]   (WP7 : RG-GPU, RG-DOMAINE-GPU, G-DET)
//! kvu rg comparer --a DIR --b DIR [--min N] [--id ID] [--sans-theta]                                                                                          (WP7 : RG-HIP-CUDA)
//! ```
//! `bloc` écrit la configuration grossière (même format que l'entrée, côté L/3) et son manifeste `<sortie>.manifeste.json`
//! (`kvu-rg-bloc/1` : pas_de_bloc, géométrie, distribution de θ, fraction hors domaine, sha256 de l'entrée et de la sortie).
//! Le manifeste ne contient ni durée ni nombre de fils : sortie ET manifeste sont identiques à l'octet quel que soit `--fils` (RG-FILS).

use super::bloc::{bloquer, OptionsBloc};
use super::lissage::{bloquer_operateur, OptionsOperateur};
use super::operateur::Operateur;
use super::gates_gpu::{self, ContexteGpu};
use super::gpu::BlocGpu;
use super::domaine;
use super::gates::{self, Contexte};
use super::geometrie::{Geometrie, Table, PAS};
use super::swendsen::{bloquer_b2, plaquette_moyenne};
use crate::commun::{code, Erreur, Resultat};
use crate::groupes::gates::{certifiable, tout_vert, LigneGate, Verdict};
use crate::heritage::manifest::sha256_fichier;
use crate::stockage::garde::GardeSsd;
use crate::stockage::npy::{ecrire_champ_hache, inspecter_champ, lire_champ};
use clap::{Arg, ArgAction, ArgMatches, Command};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const SCHEMA_MANIFESTE_BLOC: &str = "kvu-rg-bloc/1";
pub const SCHEMA_CERT: &str = "kvu-rg-cert/1";
pub const CERT_DEFAUT: &str = "validation/rg/rg_bloc.json";

pub fn commandes() -> Vec<Command> {
    let entree = || Arg::new("entree").long("entree").required(true).num_args(1).help("configuration d'entrée (.npy c16 ou q64, .npy.zst accepté)");
    let sortie = || Arg::new("sortie").long("sortie").required(true).num_args(1).help("configuration de sortie (.npy, même format que l'entrée)");
    let fils = || Arg::new("fils").long("fils").num_args(1).help("nombre de fils (défaut : RAYON_NUM_THREADS ou 4) ; la sortie n'en dépend pas");
    let backend = || Arg::new("backend").long("backend").num_args(1).default_value("cpu").help("cpu | hip | cuda | auto (GPU : sous le verrou GPU de la machine ; la sortie GPU est déterministe, à 1e-12 de celle du CPU)");
    vec![Command::new("rg")
        .about("Groupe de renormalisation par blocs : bloc de Bałaban L_b = 3, témoin de Swendsen b = 2, domaine, gates")
        .subcommand_required(true)
        .subcommand(
            Command::new("bloc")
                .about("Configuration grossière (L/3)⁴ par la moyenne non linéaire de Bałaban (SU(2), CPU)")
                .arg(entree())
                .arg(sortie())
                .arg(fils())
                .arg(Arg::new("geometrie").long("geometrie").num_args(1).default_value("3centre").help("3centre (définition S1) | 3coin (MUTANT, non invariant par réflexion)"))
                .arg(backend())
                .arg(Arg::new("operateur").long("operateur").num_args(1).default_value("3centre").help("3centre (Bałaban, défaut, inchangé) | dec | ape:<α>:<n> | flot:<t>[:<ε>] (ε défaut 0,01) | agr:<c> : docs/BLOCAGE_QUASI_LOCAL.md"))
                .arg(Arg::new("hors-famille").long("hors-famille").action(ArgAction::SetTrue).help("accepte un opérateur dont le rayon de lissage dépasse le bloc (mutant ; écrit au manifeste)")),
        )
        .subcommand(
            Command::new("swendsen")
                .about("Blocage de Swendsen b = 2 (témoin de schéma ; port de block_config)")
                .arg(entree())
                .arg(sortie())
                .arg(Arg::new("c").long("c").num_args(1).default_value("0.5").help("coefficient des 6 chemins en U"))
                .arg(Arg::new("pas").long("pas").num_args(1).default_value("1").help("nombre de pas")),
        )
        .subcommand(
            Command::new("domaine")
                .about("Rapport de domaine (RG-DOMAINE) d'une configuration : distribution de θ, fraction hors domaine, branches ambiguës")
                .arg(entree())
                .arg(fils())
                .arg(Arg::new("rapport").long("rapport").num_args(1).help("écrit le rapport JSON ici (sinon sur la sortie standard)"))
                .arg(backend()),
        )
        .subcommand(
            Command::new("gates")
                .about("Gates RG-* : une ligne PASS|FAIL|SKIP par gate")
                .arg(Arg::new("fixtures").long("fixtures").num_args(1).default_value("validation/rg/fixtures"))
                .arg(Arg::new("donnees").long("donnees").num_args(1).default_value("tests/donnees/rg"))
                .arg(Arg::new("grandl").long("grandl").num_args(1).help("racine des ensembles grandL (…/ensembles/production) ; absent ⇒ SKIP"))
                .arg(Arg::new("rapports").long("rapports").num_args(1).help("dossier des rapports JSON de RG-DOMAINE"))
                .arg(fils())
                .arg(Arg::new("seul").long("seul").num_args(1).help("préfixes d'ID à exécuter (liste à virgules)"))
                .arg(Arg::new("sans-mutants").long("sans-mutants").action(ArgAction::SetTrue))
                .arg(Arg::new("json").long("json").action(ArgAction::SetTrue))
                .arg(Arg::new("certifier").long("certifier").num_args(0..=1).default_missing_value(CERT_DEFAUT).help("écrit le certificat (défaut validation/rg/rg_bloc.json) ; refusé si une ligne est FAIL ou SKIP")),
        )
        .subcommand(
            Command::new("reponse")
                .about("Instrument de réponse covariante des liens grossiers (CPU, WP22)")
                .arg(entree()).arg(sortie()).arg(fils())
                .arg(Arg::new("operateur").long("operateur").required(true).num_args(1))
                .arg(Arg::new("hors-famille").long("hors-famille").action(ArgAction::SetTrue))
                .arg(Arg::new("echantillon").long("echantillon").default_value("324").num_args(1))
                .arg(Arg::new("pas").long("pas").default_value("0.001").num_args(1))
                .arg(Arg::new("detail").long("detail").action(ArgAction::SetTrue)),
        )
        .subcommand(
            Command::new("reponse-gates")
                .about("Sept contrôles de l'instrument de réponse (précondition : certificat A entièrement PASS)")
                .arg(Arg::new("fixtures").long("fixtures").num_args(1).default_value("validation/rg/fixtures"))
                .arg(fils())
                .arg(Arg::new("json").long("json").action(ArgAction::SetTrue)),
        )
        .subcommand(
            Command::new("ql-gates")
                .about("Gates RG-QL-* du blocage quasi-local (WP22)")
                .arg(Arg::new("fixtures").long("fixtures").num_args(1).default_value("validation/rg/fixtures"))
                .arg(Arg::new("references").long("references").num_args(1).default_value("validation/rg_ql/references"))
                .arg(Arg::new("grandl").long("grandl").num_args(1))
                .arg(fils())
                .arg(Arg::new("seul").long("seul").num_args(1).help("préfixes d'identifiants, séparés par des virgules"))
                .arg(Arg::new("sans-mutants").long("sans-mutants").action(ArgAction::SetTrue))
                .arg(Arg::new("json").long("json").action(ArgAction::SetTrue))
                .arg(Arg::new("certifier").long("certifier").num_args(0..=1).default_missing_value(super::gates_ql::CERT_DEFAUT)),
        )
        .subcommand(
            Command::new("gates-gpu")
                .about("Gates GPU du bloc (RG-GPU, RG-DOMAINE-GPU, G-DET, mutant) sur un backend GPU, sous le verrou GPU")
                .arg(Arg::new("backend").long("backend").num_args(1).required(true).help("hip | cuda | auto"))
                .arg(Arg::new("fixtures").long("fixtures").num_args(1).default_value("validation/rg/fixtures"))
                .arg(Arg::new("grandl").long("grandl").num_args(1).help("racine des ensembles grandL (…/ensembles/production) : L24_b8_c1/cfg_002000.npy ; absent ⇒ SKIP"))
                .arg(Arg::new("sortie").long("sortie").num_args(1).help("écrit les configurations grossières et statistiques dans DIR/<backend>/ (entrée de `rg comparer`)"))
                .arg(fils())
                .arg(Arg::new("repetitions").long("repetitions").num_args(1).default_value("3").help("répétitions du banc de temps (meilleur retenu)"))
                .arg(Arg::new("sans-mutants").long("sans-mutants").action(ArgAction::SetTrue))
                .arg(Arg::new("json").long("json").action(ArgAction::SetTrue)),
        )
        .subcommand(
            Command::new("comparer")
                .about("RG-HIP-CUDA : compare deux dossiers de sorties de `gates-gpu --sortie` (≤ 1e-12 sur les quaternions et les rapports de domaine)")
                .arg(Arg::new("a").long("a").num_args(1).required(true))
                .arg(Arg::new("b").long("b").num_args(1).required(true))
                .arg(Arg::new("min").long("min").num_args(1).default_value("4").help("nombre minimal d'entrées communes"))
                .arg(Arg::new("id").long("id").num_args(1).default_value("RG-HIP-CUDA").help("identifiant de la ligne de verdict (ex. ENLIGNE-1)"))
                .arg(Arg::new("sans-theta").long("sans-theta").action(ArgAction::SetTrue).help("compare les quaternions seulement (pas de <nom>.theta.json)")),
        )
        .subcommands(super::flot::cli::sous_commandes())
        .subcommands(super::action::cli::sous_commandes())
        .subcommands(super::flux::cli::sous_commandes())]
}

fn texte(m: &ArgMatches, nom: &str) -> Resultat<String> {
    m.get_one::<String>(nom).cloned().ok_or_else(|| Erreur::Config(format!("--{nom} requis")))
}

fn nombre_fils(m: &ArgMatches) -> Resultat<usize> {
    let n = match m.get_one::<String>("fils") {
        Some(t) => t.parse::<usize>().map_err(|_| Erreur::Config(format!("--fils « {t} » invalide")))?,
        None => std::env::var("RAYON_NUM_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(4),
    };
    if n == 0 {
        return Err(Erreur::Config("--fils doit être ≥ 1".into()));
    }
    Ok(n)
}

pub fn executer(nom: &str, m: &ArgMatches) -> Resultat<i32> {
    if nom != "rg" {
        return Err(Erreur::Config(format!("sous-commande « {nom} » inconnue du module rg")));
    }
    match m.subcommand() {
        Some(("bloc", s)) => {
            let op = Operateur::analyser(&texte(s, "operateur")?)?;
            if !op.est_trois_centre() {
                return bloc_ql_cmd(s, &op);
            }
            let g: Geometrie = texte(s, "geometrie")?.parse()?;
            let calcul = ouvrir_calcul(s)?;
            let r = bloquer_fichier_avec(Path::new(&texte(s, "entree")?), Path::new(&texte(s, "sortie")?), g, nombre_fils(s)?, calcul.bloc(g)?.as_ref())?;
            println!("INFO RG-BLOC {} → {} (L = {} → {}), géométrie {}, f = {:.6e}, θ max = {:.4}, {} ambiguës, {}", r.entree, r.sortie, r.l_entree, r.l_sortie, g.nom(), r.fraction_hors, r.theta_max, r.n_ambigues, r.etiquette);
            Ok(code::OK)
        }
        Some(("swendsen", s)) => swendsen_cmd(s),
        Some(("domaine", s)) => domaine_cmd(s),
        Some(("gates", s)) => gates_cmd(s),
        Some(("ql-gates", s)) => gates_ql_cmd(s),
        Some(("reponse", s)) => reponse_cmd(s),
        Some(("reponse-gates", s)) => reponse_gates_cmd(s),
        Some(("gates-gpu", s)) => gates_gpu_cmd(s),
        Some(("comparer", s)) => comparer_cmd(s),
        Some((n @ ("flot" | "etalonner" | "flot-gates" | "etalonnage-gates"), s)) => super::flot::cli::executer(n, s),
        Some((n @ ("action" | "action-gates"), s)) => super::action::cli::executer(n, s),
        Some((n @ ("flux" | "flux-gates"), s)) => super::flux::cli::executer(n, s),
        _ => Err(Erreur::Config("sous-commande de `rg` manquante".into())),
    }
}

// ------------------------------------------------------------------------------------------
// bloc
// ------------------------------------------------------------------------------------------

/// Chemin du manifeste d'une sortie : `<sortie>.manifeste.json`.
pub fn chemin_manifeste(sortie: &Path) -> PathBuf {
    let mut s = sortie.as_os_str().to_os_string();
    s.push(".manifeste.json");
    PathBuf::from(s)
}

/// Synthèse rendue par [`bloquer_fichier`].
pub struct ResumeBloc {
    pub entree: String,
    pub sortie: String,
    pub l_entree: usize,
    pub l_sortie: usize,
    pub fraction_hors: f64,
    pub theta_max: f64,
    pub n_ambigues: u64,
    pub etiquette: &'static str,
}

/// Lit, bloque, écrit la sortie et son manifeste. Même format de fichier que l'entrée.
pub fn bloquer_fichier(entree: &Path, sortie: &Path, geometrie: Geometrie, fils: usize) -> Resultat<ResumeBloc> {
    bloquer_fichier_avec(entree, sortie, geometrie, fils, None)
}

/// Comme [`bloquer_fichier`], avec un calcul GPU si `gpu` est donné. Le manifeste d'un calcul CPU est inchangé (octets de WP6) ;
/// un calcul GPU y ajoute le bloc `calcul` (backend, arch, sha256 du code objet).
pub fn bloquer_fichier_avec(entree: &Path, sortie: &Path, geometrie: Geometrie, fils: usize, gpu: Option<&BlocGpu>) -> Resultat<ResumeBloc> {
    let (format, _) = inspecter_champ(entree)?;
    let sha_entree = sha256_fichier(entree).map_err(|e| Erreur::io(format!("hachage {}", entree.display()), e))?;
    let champ = lire_champ(entree)?;
    let s = match gpu {
        Some(g) => g.bloquer_hote(&champ, true)?,
        None => bloquer(&champ, &OptionsBloc { geometrie, fils, angle_so3: false })?,
    };
    let ecrit = ecrire_champ_hache(sortie, &s.grossier, format, &GardeSsd::machine())?;
    let table = Table::nouvelle(geometrie);
    let mut manifeste = json!({
        "format": SCHEMA_MANIFESTE_BLOC,
        "pas_de_bloc": PAS,
        "geometrie": geometrie.nom(),
        "format_fichier": format.nom(),
        "l_entree": champ.l,
        "l_sortie": s.grossier.l,
        "n_liens_grossiers": 4 * s.grossier.v(),
        "n_boucles_par_lien": table.nb_boucles(),
        "sha256_entree": sha_entree,
        "sha256_sortie": ecrit.sha256,
        "theta": domaine::theta_json(&s.stats),
        "version": crate::cli::version_texte().replace('\n', " | "),
    });
    if let Some(g) = gpu {
        let p = g.provenance();
        manifeste["calcul"] = json!({"backend": p.backend.nom(), "arch": p.arch, "source_sha256": p.sha256_source, "code_objet_sha256": p.sha256_code, "arguments": p.arguments});
    }
    domaine::ecrire_json(&chemin_manifeste(sortie), &manifeste)?;
    Ok(ResumeBloc {
        entree: entree.display().to_string(),
        sortie: sortie.display().to_string(),
        l_entree: champ.l,
        l_sortie: s.grossier.l,
        fraction_hors: s.stats.fraction_hors(),
        theta_max: s.stats.theta_max,
        n_ambigues: s.stats.n_ambigues,
        etiquette: domaine::etiquette(&s.stats),
    })
}

// ------------------------------------------------------------------------------------------
// bloc par un opérateur quasi-local (WP22)
// ------------------------------------------------------------------------------------------

pub const SCHEMA_MANIFESTE_BLOC_QL: &str = "kvu-rg-bloc-ql/1";

/// `rg bloc --operateur <dec|ape|flot|agr>` : refus avant toute lecture si l'opérateur est hors famille (sans `--hors-famille`), si
/// `--geometrie` est donnée explicitement ou si le backend n'est pas le CPU ; manifeste `kvu-rg-bloc-ql/1` (ni durée ni fils).
fn bloc_ql_cmd(s: &ArgMatches, op: &Operateur) -> Resultat<i32> {
    if s.value_source("geometrie") == Some(clap::parser::ValueSource::CommandLine) {
        return Err(Erreur::Config("--geometrie ne s'applique qu'à l'opérateur 3centre".into()));
    }
    let backend = texte(s, "backend")?;
    if !backend.trim().eq_ignore_ascii_case("cpu") {
        return Err(Erreur::Config(format!("opérateur {} : seul le backend cpu est disponible (reçu « {backend} »)", op.spec())));
    }
    let hors_famille = s.get_flag("hors-famille");
    op.verifier_famille(hors_famille)?;
    let fils = nombre_fils(s)?;
    let entree = PathBuf::from(texte(s, "entree")?);
    let sortie = PathBuf::from(texte(s, "sortie")?);
    let (format, _) = inspecter_champ(&entree)?;
    let sha_entree = sha256_fichier(&entree).map_err(|e| Erreur::io(format!("hachage {}", entree.display()), e))?;
    let champ = lire_champ(&entree)?;
    let r = bloquer_operateur(&champ, op, &OptionsOperateur::normale(fils))?;
    let n_support = super::support::support(op, champ.l, [0; 4], 0).len();
    let unitarite = r.grossier.defaut_unitarite();
    let ecrit = ecrire_champ_hache(&sortie, &r.grossier, format, &GardeSsd::machine())?;
    let manifeste = json!({
        "format": SCHEMA_MANIFESTE_BLOC_QL,
        "pas_de_bloc": PAS,
        "operateur": op.parametres_json(),
        "hors_famille": hors_famille,
        "support": {
            "definition": super::support::definition(op),
            "rayon_dp": op.rayon_support(),
            "lien_reference": "V(X = 0, μ = 0)",
            "n_liens_fins": n_support,
            "n_liens_tore": 4 * champ.v(),
        },
        "format_fichier": format.nom(),
        "l_entree": champ.l,
        "l_sortie": r.grossier.l,
        "n_liens_grossiers": 4 * r.grossier.v(),
        "sha256_entree": sha_entree,
        "sha256_sortie": ecrit.sha256,
        "unitarite_max": unitarite,
        "norme_min_projection": r.norme_min_projection,
        "version": crate::cli::version_texte().replace('\n', " | "),
    });
    domaine::ecrire_json(&chemin_manifeste(&sortie), &manifeste)?;
    let (dans, _) = op.famille();
    println!(
        "INFO RG-BLOC-QL {} → {} (L = {} → {}), opérateur {}, rayon de lissage {:.4}, famille {}, support {} liens fins, unitarité {:.2e}",
        entree.display(),
        sortie.display(),
        champ.l,
        r.grossier.l,
        op.spec(),
        op.rayon_lissage(),
        if dans { "oui" } else { "non (hors famille demandé)" },
        n_support,
        unitarite
    );
    Ok(code::OK)
}

// ------------------------------------------------------------------------------------------
// swendsen
// ------------------------------------------------------------------------------------------

fn swendsen_cmd(m: &ArgMatches) -> Resultat<i32> {
    let entree = PathBuf::from(texte(m, "entree")?);
    let sortie = PathBuf::from(texte(m, "sortie")?);
    let c: f64 = texte(m, "c")?.parse().map_err(|_| Erreur::Config("--c invalide".into()))?;
    let pas: usize = texte(m, "pas")?.parse().map_err(|_| Erreur::Config("--pas invalide".into()))?;
    if pas == 0 {
        return Err(Erreur::Config("--pas doit être ≥ 1".into()));
    }
    let (format, _) = inspecter_champ(&entree)?;
    let sha_entree = sha256_fichier(&entree).map_err(|e| Erreur::io("hachage", e))?;
    let mut champ = lire_champ(&entree)?;
    let p0 = plaquette_moyenne(&champ);
    for k in 1..=pas {
        champ = bloquer_b2(&champ, c)?;
        println!("INFO RG-SWENDSEN pas {k} → L = {}, plaquette {:.12}", champ.l, plaquette_moyenne(&champ));
    }
    let ecrit = ecrire_champ_hache(&sortie, &champ, format, &GardeSsd::machine())?;
    let manifeste = json!({
        "format": "kvu-rg-swendsen/1", "pas_de_bloc": 2, "c": c, "pas": pas, "l_sortie": champ.l,
        "plaquette_entree": p0, "plaquette_sortie": plaquette_moyenne(&champ),
        "sha256_entree": sha_entree, "sha256_sortie": ecrit.sha256,
    });
    domaine::ecrire_json(&chemin_manifeste(&sortie), &manifeste)?;
    Ok(code::OK)
}

// ------------------------------------------------------------------------------------------
// domaine
// ------------------------------------------------------------------------------------------

fn domaine_cmd(m: &ArgMatches) -> Resultat<i32> {
    let entree = PathBuf::from(texte(m, "entree")?);
    let sha = sha256_fichier(&entree).map_err(|e| Erreur::io("hachage", e))?;
    let champ = lire_champ(&entree)?;
    let calcul = ouvrir_calcul(m)?;
    let s = match calcul.bloc(Geometrie::TroisCentre)? {
        Some(g) => g.bloquer_hote(&champ, true)?.stats,
        None => bloquer(&champ, &OptionsBloc { geometrie: Geometrie::TroisCentre, fils: nombre_fils(m)?, angle_so3: false })?.stats,
    };
    let nom = entree.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let rap = domaine::rapport(&nom, &sha, champ.l, Geometrie::TroisCentre, &s);
    match m.get_one::<String>("rapport") {
        Some(p) => domaine::ecrire_json(Path::new(p), &rap)?,
        None => println!("{}", serde_json::to_string_pretty(&rap).map_err(|e| Erreur::Format(e.to_string()))?),
    }
    println!("INFO RG-DOMAINE {nom} : f = {:.6e}, θ max = {:.4}, {} ambiguës, {}", s.fraction_hors(), s.theta_max, s.n_ambigues, domaine::etiquette(&s));
    Ok(code::OK)
}

// ------------------------------------------------------------------------------------------
// gates et certificat
// ------------------------------------------------------------------------------------------

fn lignes_json(l: &[LigneGate]) -> Vec<Value> {
    l.iter().map(|x| json!({"verdict": x.verdict.to_string(), "id": x.id, "detail": x.detail})).collect()
}

fn gates_cmd(m: &ArgMatches) -> Resultat<i32> {
    let seul: Vec<String> = m.get_one::<String>("seul").map(|s| s.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or_default();
    let mutants = !m.get_flag("sans-mutants");
    if m.get_one::<String>("certifier").is_some() && (!seul.is_empty() || !mutants) {
        return Err(Erreur::Config("--certifier exige la batterie complète (ni --seul, ni --sans-mutants)".into()));
    }
    let ctx = Contexte {
        fixtures: PathBuf::from(texte(m, "fixtures")?),
        donnees: PathBuf::from(texte(m, "donnees")?),
        grandl: m.get_one::<String>("grandl").map(PathBuf::from),
        rapports: m.get_one::<String>("rapports").map(PathBuf::from),
        fils: nombre_fils(m)?,
        mutants,
        seul,
    };
    let t0 = std::time::Instant::now();
    let lignes = gates::batterie(&ctx)?;
    for l in &lignes {
        println!("{l}");
    }
    let c = |v: Verdict| lignes.iter().filter(|x| x.verdict == v).count();
    let (np, nf, ns) = (c(Verdict::Pass), c(Verdict::Fail), c(Verdict::Skip));
    println!("INFO RG-BILAN : {np} PASS, {nf} FAIL, {ns} SKIP, {:.1} s, {} fils", t0.elapsed().as_secs_f64(), ctx.fils);
    if m.get_flag("json") {
        let doc = json!({"tout_vert": tout_vert(&lignes), "pass": np, "fail": nf, "skip": ns, "lignes": lignes});
        println!("{}", serde_json::to_string(&doc).map_err(|e| Erreur::Format(e.to_string()))?);
    }
    if let Some(f) = m.get_one::<String>("certifier") {
        // Aucun SKIP prescrit pour ce module : tout SKIP est non vert.
        if certifiable(&lignes, "cpu").is_none() {
            return Err(Erreur::Validation("certificat refusé : au moins une ligne FAIL ou SKIP".into()));
        }
        let sha_bin = std::env::current_exe().ok().and_then(|p| sha256_fichier(&p).ok()).unwrap_or_else(|| "inconnu".into());
        let doc = json!({
            "schema": SCHEMA_CERT,
            "horodatage": crate::regen::manifeste::maintenant_iso(),
            "binaire_sha256": sha_bin,
            "version": crate::cli::version_texte().replace('\n', " | "),
            "backend": "cpu",
            "tout_vert": true,
            "verdicts": lignes_json(&lignes),
        });
        let p = PathBuf::from(f);
        if let Some(d) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(d).map_err(|e| Erreur::io("création du dossier du certificat", e))?;
        }
        domaine::ecrire_json(&p, &doc)?;
        println!("INFO RG-CERT certificat écrit : {}", p.display());
    }
    Ok(if nf == 0 && ns == 0 { code::OK } else { code::VALIDATION })
}

fn gates_ql_cmd(m: &ArgMatches) -> Resultat<i32> {
    use super::gates_ql;
    if m.get_one::<String>("certifier").is_some()
        && (m.get_one::<String>("seul").is_some() || m.get_flag("sans-mutants")) {
        return Err(Erreur::Config("--certifier exige la batterie complète (ni --seul, ni --sans-mutants)".into()));
    }
    let ctx = gates_ql::Contexte {
        fixtures: PathBuf::from(texte(m, "fixtures")?),
        references: PathBuf::from(texte(m, "references")?),
        grandl: m.get_one::<String>("grandl").map(PathBuf::from),
        fils: nombre_fils(m)?,
        seul: m.get_one::<String>("seul").map(|s| s.split(',').map(|p| p.trim().to_string()).collect()).unwrap_or_default(),
        mutants: !m.get_flag("sans-mutants"),
    };
    let t0 = std::time::Instant::now();
    let lignes = gates_ql::batterie(&ctx)?;
    for ligne in &lignes { println!("{ligne}"); }
    let compte = |v| lignes.iter().filter(|l| l.verdict == v).count();
    let (np, nf, ns) = (compte(Verdict::Pass), compte(Verdict::Fail), compte(Verdict::Skip));
    println!("INFO RG-QL-BILAN : {np} PASS, {nf} FAIL, {ns} SKIP, {:.1} s, {} fils", t0.elapsed().as_secs_f64(), ctx.fils);
    if m.get_flag("json") {
        println!("{}", json!({"tout_vert": tout_vert(&lignes), "pass": np, "fail": nf, "skip": ns, "lignes": lignes}));
    }
    if let Some(f) = m.get_one::<String>("certifier") {
        gates_ql::verifier_certificat(&lignes)?;
        let exe = std::env::current_exe().map_err(|e| Erreur::io("binaire courant", e))?;
        let sha = sha256_fichier(&exe).map_err(|e| Erreur::io("hachage du binaire courant", e))?;
        let doc = json!({
            "schema": gates_ql::SCHEMA_CERT,
            "horodatage": crate::regen::manifeste::maintenant_iso(),
            "binaire_sha256": sha,
            "version": crate::cli::version_texte().replace('\n', " | "),
            "backend": "cpu", "tout_vert": true, "verdicts": lignes_json(&lignes),
        });
        let p = PathBuf::from(f);
        if let Some(d) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(d).map_err(|e| Erreur::io("création du dossier du certificat", e))?;
        }
        domaine::ecrire_json(&p, &doc)?;
    }
    Ok(if nf == 0 && ns == 0 { code::OK } else { code::VALIDATION })
}

// ------------------------------------------------------------------------------------------
// backend GPU, gates-gpu, comparer (WP7)
// ------------------------------------------------------------------------------------------

/// Calcul choisi par `--backend` : appareil GPU et verrou GPU de la machine, tenu jusqu'à la destruction.
pub(crate) struct Calcul {
    app: Option<crate::gpu::Appareil>,
    _verrou: Option<crate::machine::GardeVerrou>,
}

impl Calcul {
    /// Le pilote du bloc pour la géométrie demandée (`None` sur CPU).
    pub(crate) fn bloc(&self, g: Geometrie) -> Resultat<Option<BlocGpu>> {
        match &self.app {
            Some(a) => BlocGpu::nouveau(a, g).map(Some),
            None => Ok(None),
        }
    }
}

fn ouvrir_calcul(m: &ArgMatches) -> Resultat<Calcul> {
    let demande = m.get_one::<String>("backend").map(|s| s.as_str()).unwrap_or("cpu");
    ouvrir_calcul_nom(demande)
}

pub(crate) fn ouvrir_calcul_nom(demande: &str) -> Resultat<Calcul> {
    use crate::commun::BackendKind;
    let b = match demande.trim().to_ascii_lowercase().as_str() {
        "cpu" => BackendKind::Cpu,
        "auto" => crate::machine::choisir_backend(&crate::machine::detecter())?,
        autre => autre.parse::<BackendKind>()?,
    };
    if b == BackendKind::Cpu {
        return Ok(Calcul { app: None, _verrou: None });
    }
    // Le verrou est pris AVANT d'ouvrir le GPU (règle 0.7 de GATES.md) ; sous un `flock` externe, il est reconnu comme tenu par un ancêtre.
    let pol = crate::machine::politique(&crate::machine::detection::detecter_rapide());
    let verrou = crate::machine::VerrouGpu::nouveau(&pol.verrou_gpu).acquerir(std::time::Duration::from_secs(pol.attente_verrou_s))?;
    let app = crate::gpu::ouvrir(b, 0)?;
    Ok(Calcul { app: Some(app), _verrou: Some(verrou) })
}

fn gates_gpu_cmd(m: &ArgMatches) -> Resultat<i32> {
    let demande = texte(m, "backend")?;
    let calcul = ouvrir_calcul_nom(&demande)?;
    let Some(app) = calcul.app.as_ref() else {
        return Err(Erreur::Config("gates-gpu exige un backend GPU (hip | cuda | auto résolu en GPU)".into()));
    };
    let ctx = ContexteGpu {
        app,
        fixtures: PathBuf::from(texte(m, "fixtures")?),
        grandl: m.get_one::<String>("grandl").map(PathBuf::from),
        sortie: m.get_one::<String>("sortie").map(PathBuf::from),
        fils: nombre_fils(m)?,
        mutants: !m.get_flag("sans-mutants"),
        repetitions: texte(m, "repetitions")?.parse().map_err(|_| Erreur::Config("--repetitions invalide".into()))?,
    };
    let t0 = std::time::Instant::now();
    let lignes = gates_gpu::batterie_gpu(&ctx)?;
    for l in &lignes {
        println!("{l}");
    }
    let c = |v: Verdict| lignes.iter().filter(|x| x.verdict == v).count();
    let (np, nf, ns) = (c(Verdict::Pass), c(Verdict::Fail), c(Verdict::Skip));
    println!("INFO RG-GPU-BILAN backend {} : {np} PASS, {nf} FAIL, {ns} SKIP, {:.1} s", app.backend().nom(), t0.elapsed().as_secs_f64());
    if m.get_flag("json") {
        let doc = json!({"backend": app.backend().nom(), "tout_vert": tout_vert(&lignes), "pass": np, "fail": nf, "skip": ns, "lignes": gates_gpu::lignes_json(&lignes)});
        println!("{}", serde_json::to_string(&doc).map_err(|e| Erreur::Format(e.to_string()))?);
    }
    Ok(if nf == 0 && ns == 0 { code::OK } else { code::VALIDATION })
}

fn comparer_cmd(m: &ArgMatches) -> Resultat<i32> {
    let min: usize = texte(m, "min")?.parse().map_err(|_| Erreur::Config("--min invalide".into()))?;
    let lignes = gates_gpu::comparer_dossiers(Path::new(&texte(m, "a")?), Path::new(&texte(m, "b")?), min, &texte(m, "id")?, !m.get_flag("sans-theta"))?;
    for l in &lignes {
        println!("{l}");
    }
    Ok(if tout_vert(&lignes) { code::OK } else { code::VALIDATION })
}

// WP22-B : aucun changement des chemins de blocage existants.
fn reponse_cmd(m: &ArgMatches) -> Resultat<i32> {
    let op = Operateur::analyser(&texte(m, "operateur")?)?;
    let hors = m.get_flag("hors-famille");
    op.verifier_famille(hors)?;
    let k: usize = texte(m, "echantillon")?.parse().map_err(|_| Erreur::Config("--echantillon invalide".into()))?;
    if !(1..=super::reponse::LIENS_BLOC).contains(&k) {
        return Err(Erreur::Config("--echantillon doit être dans [1, 324]".into()));
    }
    let h: f64 = texte(m, "pas")?.parse().map_err(|_| Erreur::Config("--pas invalide".into()))?;
    super::reponse::valider_pas(h)?;
    let fils = nombre_fils(m)?;
    let sortie = PathBuf::from(texte(m, "sortie")?);
    if sortie.symlink_metadata().is_ok() {
        return Err(Erreur::Config("--sortie existe déjà : écrasement refusé".into()));
    }
    let entree = PathBuf::from(texte(m, "entree")?);
    let champ = lire_champ(&entree)?;
    let sha = sha256_fichier(&entree).map_err(|e| Erreur::io("hachage de l'entrée réponse", e))?;
    let doc = super::reponse::document(&champ, &op, &super::reponse::OptionsReponse { hors_famille: hors, echantillon: k, pas: h, fils, detail: m.get_flag("detail") }, &sha)?;
    let mut octets = serde_json::to_vec_pretty(&doc).map_err(|e| Erreur::Format(e.to_string()))?;
    octets.push(b'\n');
    // create_new protège également de l'apparition de la sortie pendant le calcul.
    use std::io::Write;
    let mut fichier = std::fs::OpenOptions::new().write(true).create_new(true).open(&sortie)
        .map_err(|e| if e.kind() == std::io::ErrorKind::AlreadyExists { Erreur::Config("--sortie existe déjà : écrasement refusé".into()) } else { Erreur::io("création de la sortie réponse", e) })?;
    fichier.write_all(&octets).map_err(|e| Erreur::io("écriture de la sortie réponse", e))?;
    println!("INFO RG-REPONSE opérateur {}, L = {}, {} liens fins", op.spec(), champ.l, k);
    Ok(code::OK)
}
fn reponse_gates_cmd(m: &ArgMatches) -> Resultat<i32> {
    let debut = std::time::Instant::now();
    let lignes = super::gates_reponse::batterie(Path::new(&texte(m, "fixtures")?), nombre_fils(m)?)?;
    for ligne in &lignes { println!("{ligne}"); }
    let np = lignes.iter().filter(|l| l.verdict == Verdict::Pass).count();
    let ok = np == super::gates_reponse::IDS.len() && lignes.len() == super::gates_reponse::IDS.len();
    println!("INFO RG-QL-REP-BILAN : {np} PASS, {} FAIL, 0 SKIP, {:.1} s", lignes.len()-np, debut.elapsed().as_secs_f64());
    if m.get_flag("json") { println!("{}", json!({"tout_vert": ok, "pass": np, "fail": lignes.len()-np, "skip": 0, "lignes": lignes})); }
    Ok(if ok { code::OK } else { code::VALIDATION })
}
