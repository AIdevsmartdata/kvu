//! Sous-commande `hess` : H·v sans matrice, DŪ·v par nombres duaux, λ_min généralisé sur un espace. Propriétaire : WP13.
//!
//! ```text
//! kvu hess hv   --config X.npy --vecteurs V.npy --sortie Y.npy [--base ym4|s2] [--coord gauche|additive] [--beta B] [--fils N] [--mutant aucun|M-b|facteur2]
//! kvu hess jvp  --config X.npy --vecteurs V.npy --sortie Y.npy [--ubar-sortie U.npy] [--fils N]
//! kvu hess espace --config X.npy --espace libre|kerB|landau|tcov|kerdu [--projecteur svd|cg|auto] [--projeter V.npy --sortie Y.npy] [--rapport R.json] [--fils N]
//! kvu hess landau --config X.npy --sortie Y.npy [--tol 1e-24] [--iter-max 5000] [--omega 1.7] [--fils N]
//! kvu hess lmin --config X.npy --espace libre|kerB|landau|tcov|kerdu [--reference q0|qcov] [--coord gauche|additive] [--beta B]
//!               [--graine G] [--tol 1e-10] [--bloc 4] [--iter-max N] [--projecteur svd|cg|auto] [--dense] [--sans-verif-q]
//!               [--vecteur-sortie V.npy] [--sortie R.json] [--fils N] [--mutant aucun|M-b|facteur2]
//! ```
//! `hv` : V est (k, n) ou (n,) en disposition de la base (`ym4` : 3(μV + x) + a ; `s2` : 3(4x + μ) + a) ; Y a la même forme ; H est celle de S
//! AVEC β (défaut 1). `jvp` : V en disposition s2 fine, Y en disposition s2 grossière (3 · 4 (L/3)⁴), coordonnées `a` (ζ → ζ' = 2 vec(dŪ Ū†)).
//! `lmin` : ε′ = 1 − λ_min(H|_X, Q|_X) ; coordonnées par défaut : `additive` pour libre/kerB/landau (S2), `gauche` pour tcov/kerdu ;
//! référence par défaut : `qcov` pour tcov, `q0` sinon. Toutes les sorties (NPY, JSON) sont indépendantes de `--fils`.

use super::bases::{Base, Coord};
use super::dual::jvp;
use super::espaces::{construire, Espace, ModeProj};
use super::lobpcg::Op;
use super::hamilt::{lmin, Hessienne, MutantHess, OptionsLmin, Reference};
use crate::commun::{code, Erreur, Resultat};
use crate::stockage::garde::GardeSsd;
use crate::stockage::npy::{ecrire_champ, ecrire_f64, lire_champ, lire_f64, FormatChamp};
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::path::Path;
use std::sync::Arc;

pub fn commandes() -> Vec<Command> {
    let config = || Arg::new("config").long("config").required(true).num_args(1).help("configuration (.npy c16 ou q64)");
    let fils = || Arg::new("fils").long("fils").num_args(1).help("nombre de fils (défaut : RAYON_NUM_THREADS ou 4) ; les sorties n'en dépendent pas");
    let beta = || Arg::new("beta").long("beta").num_args(1).default_value("1").help("β (H et Q sont rendues avec β)");
    let mutant = || Arg::new("mutant").long("mutant").num_args(1).default_value("aucun").help("INSTRUMENT DE TEST des gates : aucun | M-b | facteur2");
    vec![Command::new("hess")
        .about("Hessienne de l'action de Wilson SU(2) sur fonds réels : H·v, DŪ·v, λ_min généralisé")
        .subcommand_required(true)
        .subcommand(
            Command::new("hv")
                .about("H·v sans matrice (plaquettes, β inclus)")
                .arg(config())
                .arg(Arg::new("vecteurs").long("vecteurs").required(true).num_args(1).help("NPY float64 (k, n) ou (n,)"))
                .arg(Arg::new("sortie").long("sortie").required(true).num_args(1))
                .arg(Arg::new("base").long("base").num_args(1).default_value("ym4"))
                .arg(Arg::new("coord").long("coord").num_args(1).default_value("gauche"))
                .arg(beta())
                .arg(fils())
                .arg(mutant()),
        )
        .subcommand(
            Command::new("jvp")
                .about("DŪ·v par nombres duaux (carte de Bałaban 3centre)")
                .arg(config())
                .arg(Arg::new("vecteurs").long("vecteurs").required(true).num_args(1))
                .arg(Arg::new("sortie").long("sortie").required(true).num_args(1))
                .arg(Arg::new("ubar-sortie").long("ubar-sortie").num_args(1).help("écrit aussi Ū (quaternions, format q64) pour contrôle contre `rg bloc`"))
                .arg(fils()),
        )
        .subcommand(
            Command::new("landau")
                .about("Fixation de Landau réseau (port de lat.py::landau_fix) ; écrit la configuration fixée (même format que l'entrée) et rapporte θ")
                .arg(config())
                .arg(Arg::new("sortie").long("sortie").required(true).num_args(1))
                .arg(Arg::new("tol").long("tol").num_args(1).default_value("1e-24"))
                .arg(Arg::new("iter-max").long("iter-max").num_args(1).default_value("5000"))
                .arg(Arg::new("omega").long("omega").num_args(1).default_value("1.7"))
                .arg(fils()),
        )
        .subcommand(
            Command::new("espace")
                .about("Construit l'espace (dimension, rangs, valeurs singulières écartées) ; projette éventuellement des vecteurs sur X")
                .arg(config())
                .arg(Arg::new("espace").long("espace").required(true).num_args(1))
                .arg(Arg::new("projecteur").long("projecteur").num_args(1).default_value("auto"))
                .arg(Arg::new("projeter").long("projeter").num_args(1).help("NPY float64 (k, n) ou (n,) : vecteurs projetés par Π_X"))
                .arg(Arg::new("sortie").long("sortie").num_args(1).help("vecteurs projetés (avec --projeter)"))
                .arg(Arg::new("rapport").long("rapport").num_args(1).help("rapport JSON (sinon sortie standard)"))
                .arg(fils()),
        )
        .subcommand(
            Command::new("lmin")
                .about("ε′ = 1 − λ_min(H|_X, Q|_X) par LOBPCG généralisé F64")
                .arg(config())
                .arg(Arg::new("espace").long("espace").required(true).num_args(1))
                .arg(Arg::new("reference").long("reference").num_args(1))
                .arg(Arg::new("coord").long("coord").num_args(1))
                .arg(beta())
                .arg(Arg::new("graine").long("graine").num_args(1).default_value("1"))
                .arg(Arg::new("tol").long("tol").num_args(1).default_value("1e-10"))
                .arg(Arg::new("bloc").long("bloc").num_args(1).default_value("4"))
                .arg(Arg::new("iter-max").long("iter-max").num_args(1).default_value("3000"))
                .arg(Arg::new("projecteur").long("projecteur").num_args(1).default_value("auto").help("svd (exact, rang compté, L ≤ 5) | cg (sans matrice ; imprécis pour tcov près de l'identité) | auto"))
                .arg(Arg::new("dense").long("dense").action(ArgAction::SetTrue).help("résolution dense (petits réseaux) au lieu de LOBPCG"))
                .arg(Arg::new("sans-verif-q").long("sans-verif-q").action(ArgAction::SetTrue).help("ne vérifie pas λ_min(Q|X) ≥ 1e-8 λ_max(Q|X)"))
                .arg(Arg::new("vecteur-sortie").long("vecteur-sortie").num_args(1))
                .arg(Arg::new("sortie").long("sortie").num_args(1).help("écrit le rapport JSON ici (sinon sur la sortie standard)"))
                .arg(fils())
                .arg(mutant()),
        )]
}

fn texte(m: &ArgMatches, nom: &str) -> Resultat<String> {
    m.get_one::<String>(nom).cloned().ok_or_else(|| Erreur::Config(format!("--{nom} requis")))
}

fn nombre<T: std::str::FromStr>(m: &ArgMatches, nom: &str) -> Resultat<T> {
    let t = texte(m, nom)?;
    t.parse::<T>().map_err(|_| Erreur::Config(format!("--{nom} « {t} » invalide")))
}

fn fils(m: &ArgMatches) -> Resultat<usize> {
    let n = match m.get_one::<String>("fils") {
        Some(t) => t.parse::<usize>().map_err(|_| Erreur::Config(format!("--fils « {t} » invalide")))?,
        None => std::env::var("RAYON_NUM_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(4),
    };
    if n == 0 {
        return Err(Erreur::Config("--fils doit être ≥ 1".into()));
    }
    Ok(n)
}

fn pool(n: usize) -> Resultat<Arc<rayon::ThreadPool>> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .map(Arc::new)
        .map_err(|e| Erreur::Config(format!("création du pool de fils impossible : {e}")))
}

pub fn executer(nom: &str, m: &ArgMatches) -> Resultat<i32> {
    if nom != "hess" {
        return Err(Erreur::Config(format!("sous-commande « {nom} » inconnue du module hess")));
    }
    match m.subcommand() {
        Some(("hv", s)) => hv_cmd(s),
        Some(("jvp", s)) => jvp_cmd(s),
        Some(("lmin", s)) => lmin_cmd(s),
        Some(("espace", s)) => espace_cmd(s),
        Some(("landau", s)) => landau_cmd(s),
        _ => Err(Erreur::Config("sous-commande de `hess` manquante".into())),
    }
}

fn lire_vecteurs(chemin: &Path, n: usize) -> Resultat<(Vec<usize>, Vec<f64>, usize)> {
    let (forme, d) = lire_f64(chemin)?;
    let k = match forme.as_slice() {
        [m] if *m == n => 1,
        [k, m] if *m == n => *k,
        _ => return Err(Erreur::Config(format!("{} : forme {:?}, attendu (k, {n}) ou ({n},)", chemin.display(), forme))),
    };
    Ok((forme, d, k))
}

fn hv_cmd(m: &ArgMatches) -> Resultat<i32> {
    let champ = lire_champ(Path::new(&texte(m, "config")?))?;
    let base: Base = texte(m, "base")?.parse()?;
    let coord: Coord = texte(m, "coord")?.parse()?;
    let beta: f64 = nombre(m, "beta")?;
    let mutant: MutantHess = texte(m, "mutant")?.parse()?;
    let p = pool(fils(m)?)?;
    let h = Hessienne::nouveau(&champ, p, base, coord, beta, mutant)?;
    let n = h.n();
    let (forme, d, k) = lire_vecteurs(Path::new(&texte(m, "vecteurs")?), n)?;
    let mut out = vec![0.0; d.len()];
    for i in 0..k {
        h.hv(&d[i * n..(i + 1) * n], &mut out[i * n..(i + 1) * n]);
    }
    ecrire_f64(Path::new(&texte(m, "sortie")?), &forme, &out, &GardeSsd::machine())?;
    println!("INFO HESS-HV L = {}, base {}, coord {}, β = {beta}, {k} vecteur(s)", champ.l, base.nom(), coord.nom());
    Ok(code::OK)
}

fn jvp_cmd(m: &ArgMatches) -> Resultat<i32> {
    let champ = lire_champ(Path::new(&texte(m, "config")?))?;
    let p = pool(fils(m)?)?;
    let n = 12 * champ.v();
    let (forme, d, k) = lire_vecteurs(Path::new(&texte(m, "vecteurs")?), n)?;
    let nc = 12 * (champ.l / 3).pow(4);
    let mut out = Vec::with_capacity(k * nc);
    let mut ubar = None;
    for i in 0..k {
        let s = jvp(&p, &champ, &d[i * n..(i + 1) * n])?;
        out.extend_from_slice(&s.dubar_v);
        ubar = Some(s.ubar);
    }
    let forme_sortie: Vec<usize> = if forme.len() == 1 { vec![nc] } else { vec![k, nc] };
    ecrire_f64(Path::new(&texte(m, "sortie")?), &forme_sortie, &out, &GardeSsd::machine())?;
    if let (Some(chemin), Some(u)) = (m.get_one::<String>("ubar-sortie"), ubar) {
        ecrire_champ(Path::new(chemin), &u, FormatChamp::Q64, &GardeSsd::machine())?;
    }
    println!("INFO HESS-JVP L = {} → {}, {k} direction(s)", champ.l, champ.l / 3);
    Ok(code::OK)
}

fn landau_cmd(m: &ArgMatches) -> Resultat<i32> {
    let chemin = texte(m, "config")?;
    let (format, _) = crate::stockage::npy::inspecter_champ(Path::new(&chemin))?;
    let champ = lire_champ(Path::new(&chemin))?;
    let p = pool(fils(m)?)?;
    let s = super::landau::fixer(&p, &champ, nombre(m, "tol")?, nombre(m, "iter-max")?, nombre(m, "omega")?)?;
    ecrire_champ(Path::new(&texte(m, "sortie")?), &s.champ, format, &GardeSsd::machine())?;
    println!("INFO HESS-LANDAU L = {} : θ = {:.3e} après {} itérations, convergé = {}", champ.l, s.theta, s.iterations, s.converge);
    Ok(if s.converge { code::OK } else { code::ECHEC })
}

fn espace_cmd(m: &ArgMatches) -> Resultat<i32> {
    let champ = lire_champ(Path::new(&texte(m, "config")?))?;
    let espace: Espace = texte(m, "espace")?.parse()?;
    let p = pool(fils(m)?)?;
    let e = construire(espace, &champ, p, texte(m, "projecteur")?.parse::<ModeProj>()?)?;
    if let Some(chemin) = m.get_one::<String>("projeter") {
        let n = e.proj.n();
        let (forme, d, k) = lire_vecteurs(Path::new(chemin), n)?;
        let mut out = vec![0.0; d.len()];
        for i in 0..k {
            e.proj.appliquer(&d[i * n..(i + 1) * n], &mut out[i * n..(i + 1) * n]);
        }
        ecrire_f64(Path::new(&texte(m, "sortie")?), &forme, &out, &GardeSsd::machine())?;
    }
    let t = serde_json::to_string_pretty(&e.info).map_err(|er| Erreur::Format(er.to_string()))?;
    match m.get_one::<String>("rapport") {
        Some(c) => std::fs::write(c, t.as_bytes()).map_err(|er| Erreur::io(format!("écriture {c}"), er))?,
        None => println!("{t}"),
    }
    println!("INFO HESS-ESPACE {} L = {} : dim = {:?}", espace.nom(), champ.l, e.dim);
    Ok(code::OK)
}

fn lmin_cmd(m: &ArgMatches) -> Resultat<i32> {
    let champ = lire_champ(Path::new(&texte(m, "config")?))?;
    let espace: Espace = texte(m, "espace")?.parse()?;
    let reference: Reference = match m.get_one::<String>("reference") {
        Some(t) => t.parse()?,
        None => {
            if espace == Espace::Tcov {
                Reference::QCov
            } else {
                Reference::Q0
            }
        }
    };
    let coord: Coord = match m.get_one::<String>("coord") {
        Some(t) => t.parse()?,
        None => {
            if matches!(espace, Espace::Tcov | Espace::KerDu) {
                Coord::Gauche
            } else {
                Coord::Additive
            }
        }
    };
    let opts = OptionsLmin {
        espace,
        reference,
        coord,
        beta: nombre(m, "beta")?,
        graine: nombre(m, "graine")?,
        tol: nombre(m, "tol")?,
        bloc: nombre(m, "bloc")?,
        iter_max: nombre(m, "iter-max")?,
        mode_proj: texte(m, "projecteur")?.parse::<ModeProj>()?,
        dense: m.get_flag("dense"),
        mutant: texte(m, "mutant")?.parse()?,
        verifier_q: !m.get_flag("sans-verif-q"),
    };
    let p = pool(fils(m)?)?;
    let s = lmin(&champ, p, &opts)?;
    let texte_json = serde_json::to_string_pretty(&s.json).map_err(|e| Erreur::Format(e.to_string()))?;
    match m.get_one::<String>("sortie") {
        Some(c) => std::fs::write(c, texte_json.as_bytes()).map_err(|e| Erreur::io(format!("écriture {c}"), e))?,
        None => println!("{texte_json}"),
    }
    if let Some(c) = m.get_one::<String>("vecteur-sortie") {
        ecrire_f64(Path::new(c), &[s.vecteur.len()], &s.vecteur, &GardeSsd::machine())?;
    }
    println!("INFO HESS-LMIN espace {} L = {} : λ_min = {:.15e}, ε′ = {:.6e}", espace.nom(), champ.l, s.lambda_min, s.eps_prime);
    if s.json["solveur"]["converge"] == serde_json::Value::Bool(false) {
        eprintln!("kvu hess : LOBPCG non convergé (iter_max atteint) : le résultat n'est PAS certifié (code 8)");
        return Ok(code::VALIDATION);
    }
    Ok(code::OK)
}
