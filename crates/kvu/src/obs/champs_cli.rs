//! Sous-commande `kvu obs champs` (module « scanner », 11/11/2026) : export des champs par site d'une
//! configuration, pour la visualisation. Enregistrement séparé (comme `topo_cli`) : les commandes historiques du
//! module obs ne changent pas.
//!
//! ```text
//! kvu obs champs --cfg X.npy --out DIR [--t-flot 0.5 | --n-pas N] [--dt 0.01] [--beta B] [--dir-temps 3] [--f64]
//!                [--porte-jauge] [--graine-jauge 1] [--sans-portes]
//! ```
//! Sorties : `DIR/<champ>.npy` (f32 par défaut, f64 avec `--f64`) et le manifeste COMMUN `DIR/manifest.json`
//! (`stockage::provenance::Manifeste` : répertoire neuf, provenance unifiée, `fichiers` avec sha256, COMPLETE si rc = 0),
//! avec deux champs propres : `geometrie` (dimensions, conditions aux bords, torsion, direction du temps) et `kvu_champs`
//! (format `kvu-champs/1` : β, flot, source, définition exacte, sommes FP64 et références de chaque champ, portes).
//! Code retour 1 si une porte est rouge (pas de COMPLETE).
//!
//! REFUS : une configuration tordue ou non périodique (torsion de 't Hooft, boîte à bord, masque) déclarée par un
//! `torsion.json` ou un `manifest.json` / `config_run.json` voisin est refusée AVANT tout calcul : les densités
//! exportées ignorent les facteurs z_p = −1 de la pile de torsion et supposent la périodicité. Une configuration sans
//! métadonnée voisine est supposée périodique ; c'est consigné (`geometrie.declaration`).

use super::champs::{
    calculer, jauge_aleatoire, porte_localite, portes_jauge, portes_somme, references, somme,
    ChampsSite, ParamsChamps, Porte, DESCRIPTIONS,
};
use crate::commun::{Erreur, Resultat};
use crate::heritage::manifest::sha256_fichier;
use crate::stockage::{npy, GardeSsd, Manifeste};
use clap::{Arg, ArgAction, ArgMatches, Command};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Mention portée par chaque export (et reprise sur chaque image tirée de ces champs).
pub const MENTION: &str =
    "observation d'une configuration ; aucune image ne prouve un trou de masse";

/// Format du champ `kvu_champs` du manifeste.
pub const FORMAT: &str = "kvu-champs/1";

pub fn commandes() -> Vec<Command> {
    vec![Command::new("champs")
        .about("Champs par site (action brute et lissée, énergie et charge au trèfle, Polyakov, plaquettes à trace négative) en NPY + manifest.json")
        .arg(Arg::new("cfg").long("cfg").required(true).value_name("X.npy").help("configuration SU(2) PÉRIODIQUE sans torsion (NPY c16 ou q64)"))
        .arg(Arg::new("out").long("out").required(true).value_name("DIR").help("répertoire de sortie NEUF"))
        .arg(Arg::new("t-flot").long("t-flot").value_name("T").help("temps de flot t/a² (défaut 0,5 : √8t = 2)"))
        .arg(Arg::new("n-pas").long("n-pas").value_name("N").help("nombre de pas de flot (exclusif de --t-flot)"))
        .arg(Arg::new("dt").long("dt").default_value("0.01").help("pas du flot RK3"))
        .arg(Arg::new("beta").long("beta").value_name("B").help("β de la configuration (consigné au manifeste)"))
        .arg(
            Arg::new("dir-temps")
                .long("dir-temps")
                .default_value("3")
                .help("direction du temps consignée dans geometrie (aucun champ exporté n'en dépend)"),
        )
        .arg(Arg::new("f64").long("f64").action(ArgAction::SetTrue).help("écrire les champs en f64 (f32 par défaut)"))
        .arg(
            Arg::new("porte-jauge")
                .long("porte-jauge")
                .action(ArgAction::SetTrue)
                .help("porte d'invariance de jauge (transformation de Haar aléatoire ; double le coût)"),
        )
        .arg(Arg::new("graine-jauge").long("graine-jauge").default_value("1").help("graine de la transformation de jauge"))
        .arg(
            Arg::new("sans-portes")
                .long("sans-portes")
                .action(ArgAction::SetTrue)
                .help("ne pas recalculer les références globales (portes de somme non jouées)"),
        )]
}

fn arg<T: std::str::FromStr>(m: &ArgMatches, nom: &str) -> Resultat<Option<T>> {
    match m.get_one::<String>(nom) {
        None => Ok(None),
        Some(b) => b
            .parse::<T>()
            .map(Some)
            .map_err(|_| Erreur::Config(format!("--{nom} : valeur « {b} » invalide"))),
    }
}

/// Nombre de pas au-delà duquel `t/dt` n'est plus convertible sans perte (2^53).
const PAS_MAX: f64 = 9_007_199_254_740_992.0;

/// Paramètres du flot à partir de --t-flot / --n-pas / --dt.
pub fn params(m: &ArgMatches) -> Resultat<ParamsChamps> {
    let dt: f64 = arg(m, "dt")?.unwrap_or(0.01);
    if !dt.is_finite() || dt <= 0.0 {
        return Err(Erreur::Config(format!("--dt = {dt} invalide")));
    }
    let n_pas = match (arg::<f64>(m, "t-flot")?, arg::<usize>(m, "n-pas")?) {
        (Some(_), Some(_)) => {
            return Err(Erreur::Config(
                "donner --t-flot OU --n-pas, pas les deux".into(),
            ))
        }
        (None, Some(n)) => n,
        (t, None) => {
            let t = t.unwrap_or(0.5);
            if !t.is_finite() || t < 0.0 {
                return Err(Erreur::Config(format!("--t-flot = {t} invalide")));
            }
            let n = (t / dt).round();
            // Un quotient non représentable (au-delà de 2^53 les entiers ne sont plus tous exacts) ne se convertit PAS :
            // `as usize` le saturerait en silence (--t-flot 1 --dt 1e-30 donnerait usize::MAX pas).
            if !n.is_finite() || n >= PAS_MAX {
                return Err(Erreur::Config(format!(
                    "--t-flot = {t} avec --dt = {dt} : nombre de pas t/dt = {n:e} non représentable (limite {PAS_MAX:e})"
                )));
            }
            if ((n * dt) - t).abs() > 1e-9 * t.max(1.0) {
                return Err(Erreur::Config(format!(
                    "--t-flot = {t} n'est pas un multiple de dt = {dt}"
                )));
            }
            n as usize
        }
    };
    let t = n_pas as f64 * dt;
    if !t.is_finite() {
        return Err(Erreur::Config(format!(
            "temps de flot n_pas·dt = {n_pas}·{dt} non fini"
        )));
    }
    Ok(ParamsChamps { dt, n_pas })
}

fn io(contexte: String) -> impl FnOnce(std::io::Error) -> Erreur {
    move |source| Erreur::Io { contexte, source }
}

// ------------------------------------------------------------------------------------------
// Géométrie : refus des configurations tordues ou non périodiques
// ------------------------------------------------------------------------------------------

/// Dossiers où chercher les métadonnées d'une configuration : son dossier et, pour `c<k>/`, le parent. Le chemin est
/// d'abord rendu ABSOLU (le répertoire courant est consulté) : `c000/cfg.npy` depuis la racine d'un lot, `cfg.npy` depuis
/// `c000/` et `../c000/cfg.npy` cherchent aux mêmes endroits. Si le chemin passe par un lien symbolique, les dossiers de la
/// cible RÉSOLUE s'ajoutent à ceux du chemin donné. Un chemin qu'on ne sait pas rendre absolu est une erreur (jamais ignorée).
fn dossiers_meta(cfg: &Path) -> Resultat<Vec<PathBuf>> {
    let absolu =
        std::path::absolute(cfg).map_err(io(format!("chemin absolu de {}", cfg.display())))?;
    let resolu =
        std::fs::canonicalize(cfg).map_err(io(format!("résolution de {}", cfg.display())))?;
    let mut v: Vec<PathBuf> = Vec::new();
    for chemin in [absolu, resolu] {
        let mut d = chemin.parent().map(Path::to_path_buf);
        for _ in 0..2 {
            let Some(cur) = d else { break };
            if !v.contains(&cur) {
                v.push(cur.clone());
            }
            d = cur.parent().map(Path::to_path_buf);
        }
    }
    Ok(v)
}

/// Vrai si `v` déclare une torsion non triviale : objet avec `plans` non vide, ou texte autre que « aucune ».
fn torsion_non_triviale(v: &Value) -> bool {
    match v {
        Value::Null | Value::Bool(false) => false,
        Value::Object(o) => match o.get("plans") {
            Some(Value::Array(a)) => !a.is_empty(),
            Some(Value::String(s)) => !s.is_empty() && s != "aucune",
            Some(Value::Null) | None => false,
            Some(_) => true,
        },
        Value::String(s) => !s.is_empty() && s != "aucune",
        Value::Array(a) => !a.is_empty(),
        _ => true,
    }
}

/// Vrai si `v` déclare un bord ou un masque (configuration non périodique).
fn bord_declare(v: &Value) -> bool {
    match v {
        Value::Null | Value::Bool(false) => false,
        Value::Array(a) => !a.is_empty(),
        Value::String(s) => !s.is_empty(),
        _ => true,
    }
}

fn lire_json(p: &Path) -> Resultat<Value> {
    let t = std::fs::read_to_string(p).map_err(io(format!("lecture de {}", p.display())))?;
    serde_json::from_str(&t)
        .map_err(|e| Erreur::Format(format!("{} : JSON invalide ({e})", p.display())))
}

/// Contrôle la géométrie déclarée à côté de la configuration et rend le bloc `geometrie` du manifeste.
/// Refus (`Erreur::Config`) si une torsion non triviale ou un bord/masque est déclaré.
pub fn verifier_geometrie(cfg: &Path, l: usize, dir_temps: usize) -> Resultat<Value> {
    if dir_temps > 3 {
        return Err(Erreur::Config(format!(
            "--dir-temps = {dir_temps} hors de 0..3"
        )));
    }
    let declaration = declaration_voisine(cfg)?;
    Ok(json!({
        "l": l,
        "dimensions": [l, l, l, l],
        "axes": ["x0", "x1", "x2", "x3"],
        "conditions_aux_bords": "périodiques dans les 4 directions",
        "torsion": "aucune (z_p = +1 pour toutes les plaquettes)",
        "direction_temps": dir_temps,
        "note_direction_temps": "consignée seulement : les champs exportés sont définis sans direction privilégiée \
                                 (Polyakov dans les 4 directions)",
        "declaration": declaration,
    }))
}

/// Refus d'une torsion ou d'un bord déclarés à côté de la configuration ; rend la déclaration consignée (`geometrie.declaration`).
fn declaration_voisine(cfg: &Path) -> Resultat<Value> {
    let mut lus = Vec::new();
    for d in dossiers_meta(cfg)? {
        let t = d.join(crate::su2::torsion::NOM_FICHIER_TORSION);
        if t.is_file() {
            let v = lire_json(&t)?;
            if torsion_non_triviale(&v) {
                return Err(Erreur::Config(format!(
                    "obs champs : configuration TORDUE ({} : {}) refusée : les densités exportées ignorent les z_p = −1 de la \
                     pile de torsion",
                    t.display(),
                    v.get("plans").unwrap_or(&v)
                )));
            }
            lus.push(t.display().to_string());
        }
        for nom in ["manifest.json", "config_run.json"] {
            let p = d.join(nom);
            if !p.is_file() {
                continue;
            }
            let v = lire_json(&p)?;
            let cr = v.get("config_run").cloned().unwrap_or(Value::Null);
            let tordu = [v.get("torsion"), cr.get("torsion")]
                .into_iter()
                .flatten()
                .any(torsion_non_triviale);
            if tordu {
                return Err(Erreur::Config(format!(
                    "obs champs : configuration TORDUE (champ « torsion » de {}) refusée : les densités exportées ignorent \
                     les z_p = −1 de la pile de torsion",
                    p.display()
                )));
            }
            let bord = [
                v.get("bord"),
                v.get("masque"),
                cr.get("bord"),
                cr.get("masque"),
            ]
            .into_iter()
            .flatten()
            .any(bord_declare)
                || v.get("mode")
                    .and_then(Value::as_str)
                    .is_some_and(|m| m.contains("bord") || m.contains("boite"));
            if bord {
                return Err(Erreur::Config(format!(
                    "obs champs : configuration NON PÉRIODIQUE (bord, masque ou mode su2-bord/su2-boite déclaré dans {}) refusée",
                    p.display()
                )));
            }
            lus.push(p.display().to_string());
        }
    }
    let declaration = if lus.is_empty() {
        json!("aucune métadonnée voisine (torsion.json, manifest.json, config_run.json) : configuration SUPPOSÉE périodique sans torsion")
    } else {
        json!({ "fichiers_lus": lus, "verdict": "ni torsion ni bord déclarés" })
    };
    Ok(declaration)
}

// ------------------------------------------------------------------------------------------
// Écriture
// ------------------------------------------------------------------------------------------

fn ecrire_champs(
    dossier: &Path,
    c: &ChampsSite,
    f64_sortie: bool,
    man: &mut Manifeste,
) -> Resultat<serde_json::Map<String, Value>> {
    let garde = GardeSsd::machine();
    let mut champs = serde_json::Map::new();
    for (desc, x) in DESCRIPTIONS.iter().zip(c.tableaux()) {
        let fichier = format!("{}.npy", desc.nom);
        let chemin = dossier.join(&fichier);
        let forme = c.forme(desc.nom);
        if f64_sortie {
            npy::ecrire_f64(&chemin, &forme, x, &garde)?;
        } else {
            let y: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            npy::ecrire_f32(&chemin, &forme, &y, &garde)?;
        }
        let sha = sha256_fichier(&chemin).map_err(io(format!("sha256 de {}", chemin.display())))?;
        let octets = std::fs::metadata(&chemin)
            .map_err(io(format!("stat de {}", chemin.display())))?
            .len();
        man.fichier_connu(&chemin, &sha, octets)?;
        let s = somme(x);
        let (mn, mx) = x
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        champs.insert(
            desc.nom.to_string(),
            json!({
                "fichier": fichier,
                "forme": forme,
                "dtype": if f64_sortie { "<f8" } else { "<f4" },
                "sha256": sha,
                "definition": desc.definition,
                "agregat_compare": desc.agregat,
                "somme_f64": s,
                "moyenne_f64": s / x.len() as f64,
                "min": mn,
                "max": mx,
            }),
        );
    }
    Ok(champs)
}

pub fn executer(m: &ArgMatches) -> Resultat<i32> {
    let cfg = Path::new(m.get_one::<String>("cfg").expect("--cfg requis"));
    let out = Path::new(m.get_one::<String>("out").expect("--out requis"));
    let p = params(m)?;
    let beta: Option<f64> = arg(m, "beta")?;
    let graine: u32 = arg(m, "graine-jauge")?.unwrap_or(1);
    let dir_temps: usize = arg(m, "dir-temps")?.unwrap_or(3);
    let debut = Instant::now();
    // Côtés non égaux (l0,l1,l2,l3) : chemin séparé (`champs_cotes`) ; une boîte cubique suit le chemin historique, inchangé.
    if let Some(cotes) = cotes_non_cubiques(cfg)? {
        return executer_cotes(m, cfg, out, &p, beta, graine, dir_temps, cotes, debut);
    }
    let champ = npy::lire_champ(cfg)?;
    // Refus AVANT toute création de répertoire.
    let geometrie = verifier_geometrie(cfg, champ.l, dir_temps)?;
    let sha_src = sha256_fichier(cfg).map_err(io(format!("sha256 de {}", cfg.display())))?;
    let argv: Vec<String> = std::env::args().collect();
    let n_champs = DESCRIPTIONS.len() as u64 + 3; // polyakov compte pour 4
    let octets_val = if m.get_flag("f64") { 8 } else { 4 };
    let projection = n_champs * (champ.l as u64).pow(4) * octets_val;
    let mut man = Manifeste::ouvrir_avec_garde(
        out,
        "kvu obs champs",
        &argv,
        &GardeSsd::machine(),
        projection,
    )?;
    // Le calcul est TOUJOURS CPU FP64 (ref_cpu) : on remplace le backend que la détection de la machine a inscrit à
    // l'ouverture commune (comme les commandes `obs` historiques, `obs/cli.rs`) et on retire l'architecture GPU.
    man.champ("backend", &json!("cpu").to_string())?;
    man.champ("arch", "null")?;
    man.champ("precision_calcul", &json!("FP64").to_string())?;
    man.champ("geometrie", &geometrie.to_string())?;
    match exporter(
        &champ, cfg, &sha_src, out, &p, beta, graine, m, &mut man, debut,
    ) {
        Ok(tout_ok) => {
            man.terminer(if tout_ok { 0 } else { 1 })?;
            Ok(if tout_ok { 0 } else { 1 })
        }
        Err(e) => {
            let _ = man.echouer(e.code_sortie(), &e.to_string());
            Err(e)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn exporter(
    champ: &crate::commun::ChampHote,
    cfg: &Path,
    sha_src: &str,
    out: &Path,
    p: &ParamsChamps,
    beta: Option<f64>,
    graine: u32,
    m: &ArgMatches,
    man: &mut Manifeste,
    debut: Instant,
) -> Resultat<bool> {
    let c = calculer(champ, p)?;
    c.verifier_finis()?;
    let t_calcul = debut.elapsed().as_secs_f64();
    let mut portes: Vec<Porte> = Vec::new();
    let mut refs = Value::Null;
    if !m.get_flag("sans-portes") {
        let r = references(champ, p)?;
        portes.extend(portes_somme(&c, &r, p));
        portes.push(porte_localite(champ, &c, 64)?);
        refs = serde_json::to_value(&r)
            .map_err(|e| Erreur::Config(format!("JSON des références : {e}")))?;
    }
    let jauge = if m.get_flag("porte-jauge") {
        let cg = calculer(&jauge_aleatoire(champ, graine), p)?;
        cg.verifier_finis()?;
        portes.extend(portes_jauge(&c, &cg));
        json!({ "graine": graine, "transformation": "g(x) de Haar, su2::cpu::haar_q(s, 7, graine, 0x5ca9, 0x1111)" })
    } else {
        Value::Null
    };
    let champs = ecrire_champs(out, &c, m.get_flag("f64"), man)?;
    let tout_ok = portes.iter().all(|x| x.ok);
    let sha_bin = crate::stockage::provenance::sha256_binaire().unwrap_or_default();
    let bloc = json!({
        "format": FORMAT,
        "mention": MENTION,
        "l": c.l,
        "beta": beta,
        "disposition": "champ 4D : NPY (L,L,L,L) ordre C indexe [x0,x1,x2,x3], site kvu s = ((x0*L + x1)*L + x2)*L + x3 ; \
                        polyakov : (4,L,L,L,L) indexe [d,x0,x1,x2,x3] ; plaquette avant de coin x dans le plan (mu,nu), plans 01 02 03 12 13 23",
        "flot": {
            "methode": "flot de Wilson RK3 (Luscher 2010) de kvu, ref_cpu::flot_n_pas (CPU, FP64)",
            "dt": p.dt, "n_pas": p.n_pas, "t": p.t(), "sqrt8t": p.sqrt8t(),
        },
        "source": { "chemin": cfg.display().to_string(), "sha256": sha_src },
        "kvu": { "version": env!("CARGO_PKG_VERSION"), "sha256_binaire": sha_bin },
        "champs": champs,
        "references_kvu": refs,
        "porte_jauge": jauge,
        "portes": portes,
        "portes_ok": tout_ok,
        "duree_calcul_s": t_calcul,
        "duree_totale_s": debut.elapsed().as_secs_f64(),
    });
    man.champ("kvu_champs", &bloc.to_string())?;
    for x in &portes {
        println!("{}", x.ligne());
    }
    println!(
        "champs L={} t={} (sqrt8t={:.4}) -> {} ; portes {}/{} vertes ; {:.2} s",
        c.l,
        p.t(),
        p.sqrt8t(),
        out.display(),
        portes.iter().filter(|x| x.ok).count(),
        portes.len(),
        debut.elapsed().as_secs_f64()
    );
    Ok(tout_ok)
}

// ------------------------------------------------------------------------------------------
// Côtés quelconques (l0, l1, l2, l3)
// ------------------------------------------------------------------------------------------

/// Côtés du NPY si l'en-tête est celui d'un champ régulier dont les quatre côtés NE sont PAS égaux ; `None` sinon (le chemin
/// cubique, inchangé, lit et juge ce fichier). Une erreur de lecture d'en-tête n'est pas ici : le chemin cubique la rapporte.
fn cotes_non_cubiques(cfg: &Path) -> Resultat<Option<[usize; 4]>> {
    let Ok(f) = std::fs::File::open(cfg) else {
        return Ok(None);
    };
    let mut r = std::io::BufReader::new(f);
    let Ok(e) = npy::lire_entete(&mut r) else {
        return Ok(None);
    };
    match npy::disposition_champ_geo(&e) {
        Ok((_, c)) if !c.iter().all(|&x| x == c[0]) => Ok(Some(c)),
        _ => Ok(None),
    }
}

#[allow(clippy::too_many_arguments)]
fn executer_cotes(
    m: &ArgMatches,
    cfg: &Path,
    out: &Path,
    p: &ParamsChamps,
    beta: Option<f64>,
    graine: u32,
    dir_temps: usize,
    cotes: [usize; 4],
    debut: Instant,
) -> Resultat<i32> {
    use super::champs_cotes as ct;
    use crate::geom::{Bord, GeoA};
    if dir_temps > 3 {
        return Err(Erreur::Config(format!("--dir-temps = {dir_temps} hors de 0..3")));
    }
    let cg = npy::lire_champ_geo(cfg, dir_temps as u8)?;
    let spec = &cg.geo;
    // Refus avant tout calcul : seule une boîte périodique, sans torsion, sans masque ni réplique est admise.
    if !spec.torsion.est_triviale() {
        return Err(Erreur::Config(format!(
            "obs champs : configuration TORDUE ({}) refusée : les densités exportées ignorent les z_p = −1 de la pile de torsion",
            spec.torsion.texte()
        )));
    }
    if spec.bords.iter().any(|b| !matches!(b, Bord::Periodique)) || spec.masque.is_some() || spec.repliques_nb() != 1 {
        return Err(Erreur::Config(
            "obs champs : configuration NON PÉRIODIQUE (bord fixe, recollement, masque ou répliques) refusée".into(),
        ));
    }
    let declaration = declaration_voisine(cfg)?;
    let geo = GeoA::nouvelle(cotes)?;
    let champ = ct::ChampG::nouveau(geo.clone(), &cg.q)?;
    let sha_src = sha256_fichier(cfg).map_err(io(format!("sha256 de {}", cfg.display())))?;
    let argv: Vec<String> = std::env::args().collect();
    let n_champs = DESCRIPTIONS.len() as u64 + 3;
    let octets_val = if m.get_flag("f64") { 8 } else { 4 };
    let projection = n_champs * (geo.v as u64) * octets_val;
    let mut man = Manifeste::ouvrir_avec_garde(out, "kvu obs champs", &argv, &GardeSsd::machine(), projection)?;
    man.champ("backend", &json!("cpu").to_string())?;
    man.champ("arch", "null")?;
    man.champ("precision_calcul", &json!("FP64").to_string())?;
    let xi0 = spec.beta_plans.map(|b| b.to_vec());
    let geometrie = json!({
        "cotes": cotes,
        "dimensions": cotes,
        "axes": ["x0", "x1", "x2", "x3"],
        "conditions_aux_bords": "périodiques dans les 4 directions",
        "torsion": "aucune (z_p = +1 pour toutes les plaquettes)",
        "direction_temps": dir_temps,
        "note_direction_temps": "consignée seulement : les champs exportés sont définis sans direction privilégiée \
                                 (Polyakov dans les 4 directions)",
        "poids_de_plans_de_la_generation": xi0,
        "note_poids": "poids des plans de la génération (réseau anisotrope) consigné ; les champs et le flot sont ceux de l'action de Wilson isotrope",
        "declaration": declaration,
    });
    man.champ("geometrie", &geometrie.to_string())?;
    let res = (|| -> Resultat<bool> {
        let c = ct::calculer(&champ, p)?;
        c.verifier_finis()?;
        let t_calcul = debut.elapsed().as_secs_f64();
        let mut portes: Vec<Porte> = Vec::new();
        if !m.get_flag("sans-portes") {
            portes.extend(ct::portes(&champ, &c, p));
        }
        let jauge = if m.get_flag("porte-jauge") {
            let cg2 = ct::calculer(&ct::jauge_aleatoire(&champ, graine), p)?;
            cg2.verifier_finis()?;
            portes.extend(portes_jauge(&c, &cg2));
            json!({ "graine": graine, "transformation": "g(x) de Haar, su2::cpu::haar_q(s, 7, graine, 0x5ca9, 0x1111)" })
        } else {
            Value::Null
        };
        let champs = ecrire_champs(out, &c, m.get_flag("f64"), &mut man)?;
        let tout_ok = portes.iter().all(|x| x.ok);
        let sha_bin = crate::stockage::provenance::sha256_binaire().unwrap_or_default();
        let bloc = json!({
            "format": FORMAT,
            "mention": MENTION,
            "l": Value::Null,
            "cotes": cotes,
            "beta": beta,
            "disposition": "champ 4D : NPY (l0,l1,l2,l3) ordre C indexe [x0,x1,x2,x3], site kvu s = ((x0*l1 + x1)*l2 + x2)*l3 + x3 ; \
                            polyakov : (4,l0,l1,l2,l3) indexe [d,x0,x1,x2,x3] ; plaquette avant de coin x dans le plan (mu,nu), plans 01 02 03 12 13 23",
            "flot": {
                "methode": "flot de Wilson RK3 (Luscher 2010) de kvu, obs::champs_cotes::flot_n_pas (CPU, FP64, mêmes opérations que ref_cpu)",
                "dt": p.dt, "n_pas": p.n_pas, "t": p.t(), "sqrt8t": p.sqrt8t(),
            },
            "source": { "chemin": cfg.display().to_string(), "sha256": sha_src },
            "kvu": { "version": env!("CARGO_PKG_VERSION"), "sha256_binaire": sha_bin },
            "champs": champs,
            "references_kvu": Value::Null,
            "porte_jauge": jauge,
            "portes": portes,
            "portes_ok": tout_ok,
            "duree_calcul_s": t_calcul,
            "duree_totale_s": debut.elapsed().as_secs_f64(),
        });
        man.champ("kvu_champs", &bloc.to_string())?;
        for x in &portes {
            println!("{}", x.ligne());
        }
        println!(
            "champs cotes={:?} t={} (sqrt8t={:.4}) -> {} ; portes {}/{} vertes ; {:.2} s",
            cotes,
            p.t(),
            p.sqrt8t(),
            out.display(),
            portes.iter().filter(|x| x.ok).count(),
            portes.len(),
            debut.elapsed().as_secs_f64()
        );
        Ok(tout_ok)
    })();
    match res {
        Ok(tout_ok) => {
            man.terminer(if tout_ok { 0 } else { 1 })?;
            Ok(if tout_ok { 0 } else { 1 })
        }
        Err(e) => {
            let _ = man.echouer(e.code_sortie(), &e.to_string());
            Err(e)
        }
    }
}

