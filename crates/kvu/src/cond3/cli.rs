//! Sous-commande `cond facteur3` (T3-D) : `repliques`, `effectifs`.
//!
//! ```text
//! kvu cond facteur3 repliques --groupe su2|u1 --d 2|4 --l L --beta B
//!     (--cfg X.npy | --wilson [--therm 400 --espacement 100 --index 0 --graine-w S])
//!     --graine-a SA [--graine-b SB] [--faces faces|toutes] [--sondes paires|toutes] [--bmin 2 --bmax 10]
//!     [--n-dec 200 --n-mes 10000 --n-or 1 --nblk 2 --lot 64] [--separes] [--ord simultane|sequentiel|libre]
//!     [--controle 50] [--series-faces 64 --series-sondes 0 --chunks 40] [--mutant M] [--sans-renorm] [--acc]
//!     --sortie DIR [--fils N]
//! kvu cond facteur3 effectifs --d 2|4 --l L [--bmax 10]
//! ```

use super::geom::Geom;
use super::mesure::ConfigMesure;
use super::modele::Ordonnancement;
use super::paires::Paires;
use super::run::{executer_cellule, Depart, FacesChoix, Groupe, ParamsCellule, SondesChoix};
use crate::commun::{Erreur, Resultat};
use crate::stockage::GardeSsd;
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::path::PathBuf;

fn u(nom: &'static str, defaut: Option<&'static str>, aide: &'static str) -> Arg {
    let a = Arg::new(nom).long(nom).help(aide).value_parser(clap::value_parser!(u64));
    match defaut {
        Some(d) => a.default_value(d),
        None => a,
    }
}

pub fn commande() -> Command {
    let repliques = Command::new("repliques")
        .about("Cellule T3-D : départ (fichier ou Wilson), V = DEC figé, répliques A et B, sommes globales Σ O, Σ σ, Σ O σ par bloc de temps")
        .arg(Arg::new("groupe").long("groupe").default_value("su2").value_parser(["su2", "u1"]))
        .arg(u("d", Some("4"), "dimension 2 ou 4"))
        .arg(u("l", None, "côté du réseau fin (multiple de 6)").required(true))
        .arg(Arg::new("beta").long("beta").required(true).value_parser(clap::value_parser!(f64)))
        .arg(Arg::new("cfg").long("cfg").help("configuration SU(2) d = 4 (NPY c16 ou q64)"))
        .arg(Arg::new("wilson").long("wilson").action(ArgAction::SetTrue).help("génère la configuration par l'échantillonneur sans contrainte"))
        .arg(u("therm", Some("400"), "pas de thermalisation sans contrainte (--wilson)"))
        .arg(u("espacement", Some("100"), "pas entre deux configurations (--wilson)"))
        .arg(u("index", Some("0"), "indice de la configuration (--wilson)"))
        .arg(u("graine-w", Some("0"), "graine de la génération de Wilson"))
        .arg(u("graine-a", None, "graine de la réplique A").required(true))
        .arg(u("graine-b", None, "graine de la réplique B (absente : une seule réplique)"))
        .arg(Arg::new("faces").long("faces").default_value("faces").value_parser(["faces", "toutes"]))
        .arg(Arg::new("sondes").long("sondes").default_value("paires").value_parser(["paires", "toutes"]))
        .arg(u("bmin", Some("2"), "tranche minimale"))
        .arg(u("bmax", Some("10"), "tranche maximale"))
        .arg(u("n-dec", Some("200"), "itérations écartées"))
        .arg(u("n-mes", Some("10000"), "itérations mesurées"))
        .arg(u("n-or", Some("1"), "balayages de surrelaxation par itération"))
        .arg(u("nblk", Some("2"), "blocs de temps"))
        .arg(u("lot", Some("64"), "taille des lots d'accumulation"))
        .arg(Arg::new("separes").long("separes").action(ArgAction::SetTrue).help("scores des trois relèvements séparés"))
        .arg(Arg::new("ord").long("ord").default_value("simultane").value_parser(["simultane", "sequentiel", "libre"]))
        .arg(u("controle", Some("50"), "contrôles tous les N balayages (0 : aucun)"))
        .arg(u("series-faces", Some("0"), "séries temporelles des N premières plaquettes de P_obs"))
        .arg(u("series-sondes", Some("0"), "séries temporelles des scores des N premières sondes"))
        .arg(u("chunks", Some("0"), "chunks de temps pour la corrélation croisée A/B (0 : aucune)"))
        .arg(Arg::new("mutant").long("mutant").help("défaut injecté (tests d'outil) : agrafe | u3 | releve-sans-rotation"))
        .arg(Arg::new("sans-renorm").long("sans-renorm").action(ArgAction::SetTrue).help("pas de renormalisation (mutant de dérive)"))
        .arg(Arg::new("reduire").long("reduire").action(ArgAction::SetTrue).help("réduction de production : sommes par tranche et histogrammes de charge de tube (exige 2 répliques, nblk = 2, bmin ⩾ 2)"))
        .arg(Arg::new("tube-edges").long("tube-edges").help("bords des classes de charge de tube (NPY f64 [tranches, nbin − 1]) pour --reduire"))
        .arg(Arg::new("acc").long("acc").action(ArgAction::SetTrue).help("écrit les accumulateurs complets (NPY)"))
        .arg(Arg::new("classes-t").long("classes-t").action(ArgAction::SetTrue).help("moyennes de plaquette par classe t aux contrôles"))
        .arg(u("jauge", None, "applique une jauge fine de Haar (graine) à U⁰ avant le figeage de V (oracle O-10a)"))
        .arg(Arg::new("canaux").long("canaux").help("fichier JSON de canaux d'oracle (Stein, ΔS, moments, groupes)"))
        .arg(Arg::new("champ-final").long("champ-final").action(ArgAction::SetTrue).help("écrit le champ final de chaque réplique (NPY)"))
        .arg(Arg::new("backend").long("backend").default_value("cpu").value_parser(["auto", "hip", "cuda", "cpu"]).help("calcul : cpu (référence), hip, cuda ou auto (SU(2) d = 4)"))
        .arg(Arg::new("sortie").long("sortie").required(true).help("répertoire de sortie (créé, jamais écrasé)"))
        .arg(Arg::new("fils").long("fils").value_parser(clap::value_parser!(usize)));
    let effectifs = Command::new("effectifs")
        .about("Effectifs de la géométrie et des paires par tranche (oracle O-0)")
        .arg(u("d", Some("4"), "dimension"))
        .arg(u("l", None, "côté").required(true))
        .arg(u("bmax", Some("10"), "tranche maximale de la fenêtre"));
    let tube_classe = Command::new("tube-classe")
        .about("Tube d'une classe (oracle G0.8) : plaquettes grossières relatives (plan, Y − X_e) du tube de (plan de p, r, direction de la sonde, Y − X_e)")
        .arg(u("l", None, "côté").required(true))
        .arg(u("pl", None, "indice de plan de p").required(true))
        .arg(u("rcode", None, "3·r_mu + r_nu (0..8)").required(true))
        .arg(u("mue", None, "direction de la sonde").required(true))
        .arg(Arg::new("dy").long("dy").required(true).help("Y − X_e mod L/3 : a,b,c,d"));
    let tube_f = Command::new("tube-f")
        .about("Charge de tube f de toutes les paires d'une configuration (oracle G0.8, comparaison à une référence indépendante) : NPY f64 par paire")
        .arg(u("l", None, "côté").required(true))
        .arg(Arg::new("cfg").long("cfg").required(true))
        .arg(u("bmin", Some("2"), "tranche minimale"))
        .arg(u("bmax", Some("10"), "tranche maximale"))
        .arg(Arg::new("sortie").long("sortie").required(true))
        .arg(Arg::new("fils").long("fils").value_parser(clap::value_parser!(usize)));
    let tube_hist = Command::new("tube-hist")
        .about("Histogramme fin (65536 classes sur [0, 2]) de la charge de tube f des paires d'une configuration, par tranche (passe 1 des classes de charge)")
        .arg(u("l", None, "côté").required(true))
        .arg(Arg::new("cfg").long("cfg").required(true))
        .arg(u("bmin", Some("2"), "tranche minimale"))
        .arg(u("bmax", Some("10"), "tranche maximale"))
        .arg(Arg::new("sortie").long("sortie").required(true))
        .arg(Arg::new("fils").long("fils").value_parser(clap::value_parser!(usize)));
    Command::new("facteur3")
        .about("Fibre dure de DEC (facteur 3) : bain thermique conditionnel et noyau de réponse (T3-D) : repliques | effectifs | tube-classe | tube-f | tube-hist")
        .subcommand_required(true)
        .subcommand(repliques)
        .subcommand(effectifs)
        .subcommand(tube_classe)
        .subcommand(tube_f)
        .subcommand(tube_hist)
}

fn fixer_fils(m: &ArgMatches) {
    if let Some(n) = m.get_one::<usize>("fils") {
        let _ = rayon::ThreadPoolBuilder::new().num_threads((*n).max(1)).build_global();
    }
}

fn req<T: Clone + Send + Sync + 'static>(m: &ArgMatches, nom: &str) -> Resultat<T> {
    m.get_one::<T>(nom).cloned().ok_or_else(|| Erreur::Config(format!("--{nom} est requis")))
}

/// Option entière tenant dans un u32 (refus plutôt que troncature silencieuse : `--bmax 4294967359` ne doit pas valoir 63).
fn req_u32(m: &ArgMatches, nom: &str) -> Resultat<u32> {
    let v = req::<u64>(m, nom)?;
    u32::try_from(v).map_err(|_| Erreur::Config(format!("--{nom} = {v} : dépasse u32")))
}

pub fn executer(m: &ArgMatches) -> Resultat<i32> {
    match m.subcommand() {
        Some(("repliques", s)) => cmd_repliques(s),
        Some(("effectifs", s)) => cmd_effectifs(s),
        Some(("tube-classe", s)) => cmd_tube_classe(s),
        Some(("tube-f", s)) => cmd_tube_f(s),
        Some(("tube-hist", s)) => cmd_tube_hist(s),
        _ => Err(Erreur::Config("cond facteur3 : sous-commande attendue (repliques | effectifs)".into())),
    }
}

fn cmd_effectifs(m: &ArgMatches) -> Resultat<i32> {
    let d = req::<u64>(m, "d")? as usize;
    let l = req::<u64>(m, "l")? as usize;
    let bmax = req_u32(m, "bmax")?;
    if bmax as usize >= super::paires::B_COMPTE {
        return Err(Erreur::Config(format!("bmax = {bmax} : ⩽ {} attendu", super::paires::B_COMPTE - 1)));
    }
    let g = Geom::nouvelle(d, l)?;
    let p = Paires::nouvelles(&g, g.faces(), g.sondes_paires(), 0, bmax);
    let libres = g.pos.iter().filter(|&&x| x == -1).count();
    println!(
        "{}",
        serde_json::json!({
            "d": d, "l": l, "liens": g.nliens(), "liens_grossiers": g.ngros(), "liens_de_ligne": 3 * g.lignes.len(),
            "liens_libres": libres, "plaquettes": g.nplaq(), "faces": g.faces().len(), "sondes": g.sondes_paires().len(),
            "paires_par_tranche": p.effectifs[..=(bmax as usize + 1)].to_vec(),
        })
    );
    Ok(crate::commun::code::OK)
}

fn cmd_repliques(m: &ArgMatches) -> Resultat<i32> {
    fixer_fils(m);
    let groupe = if m.get_one::<String>("groupe").map(String::as_str) == Some("u1") { Groupe::U1 } else { Groupe::Su2 };
    let d = req::<u64>(m, "d")? as usize;
    let l = req::<u64>(m, "l")? as usize;
    let depart = if m.get_flag("wilson") {
        Depart::Wilson {
            therm: req::<u64>(m, "therm")? as usize,
            espacement: req::<u64>(m, "espacement")? as usize,
            index: req::<u64>(m, "index")? as usize,
            graine: req::<u64>(m, "graine-w")?,
        }
    } else {
        Depart::Fichier(PathBuf::from(m.get_one::<String>("cfg").ok_or_else(|| Erreur::Config("--cfg ou --wilson est requis".into()))?))
    };
    let ga = req::<u64>(m, "graine-a")?;
    let gb = m.get_one::<u64>("graine-b").copied();
    let ord = match m.get_one::<String>("ord").map(String::as_str) {
        Some("sequentiel") => Ordonnancement::Sequentiel,
        Some("libre") => Ordonnancement::Libre,
        _ => Ordonnancement::Simultane,
    };
    let mesure = ConfigMesure {
        n_dec: req::<u64>(m, "n-dec")? as usize,
        n_mes: req::<u64>(m, "n-mes")? as usize,
        n_or: req::<u64>(m, "n-or")? as usize,
        nblk: req::<u64>(m, "nblk")? as usize,
        lot: req::<u64>(m, "lot")? as usize,
        separes: m.get_flag("separes"),
        ord,
        controle_chaque: req::<u64>(m, "controle")? as usize,
        classes_t: m.get_flag("classes-t"),
        ..ConfigMesure::default()
    };
    let prm = ParamsCellule {
        groupe,
        d,
        l,
        beta: req::<f64>(m, "beta")?,
        depart,
        graines: [ga, gb.unwrap_or(ga.wrapping_add(1))],
        nrep: if gb.is_some() { 2 } else { 1 },
        faces: if m.get_one::<String>("faces").map(String::as_str) == Some("toutes") { FacesChoix::Toutes } else { FacesChoix::Faces },
        sondes: if m.get_one::<String>("sondes").map(String::as_str) == Some("toutes") { SondesChoix::Toutes } else { SondesChoix::Paires },
        bmin: req_u32(m, "bmin")?,
        bmax: req_u32(m, "bmax")?,
        mesure,
        n_series_faces: req::<u64>(m, "series-faces")? as usize,
        n_series_sondes: req::<u64>(m, "series-sondes")? as usize,
        nchunk_croise: req::<u64>(m, "chunks")? as usize,
        mutant: m.get_one::<String>("mutant").cloned(),
        sans_renorm: m.get_flag("sans-renorm"),
        jauge: m.get_one::<u64>("jauge").copied(),
        reduire: m.get_flag("reduire"),
        tube_edges: m.get_one::<String>("tube-edges").map(PathBuf::from),
        backend: m.get_one::<String>("backend").cloned().unwrap_or_else(|| "cpu".into()),
        champ_final: m.get_flag("champ-final"),
    };
    let mut prm = prm;
    if let Some(f) = m.get_one::<String>("canaux") {
        prm.mesure.canaux = lire_canaux(std::path::Path::new(f))?;
    }
    super::run::valider_parametres(&prm)?;
    let dir = PathBuf::from(req::<String>(m, "sortie")?);
    // admission mémoire (rc 5) : accumulateurs σ O des répliques + tampons
    let g = std::sync::Arc::new(Geom::nouvelle(prm.d, prm.l)?);
    let p = super::run::selection_paires(&prm, &g);
    {
        let p = (p.npaires(), p.faces.len());
        let nc = (if prm.mesure.separes { 3 } else { 1 }) * (if prm.groupe == Groupe::Su2 { 3 } else { 1 });
        let hote = (p.0 as u64 * prm.nrep as u64 * prm.mesure.nblk as u64 * nc as u64 * 8)
            + (p.1 as u64 * prm.mesure.lot as u64 * 8)
            + (p.0 as u64 * 5)
            + (g.nliens() as u64 * 32 * prm.nrep as u64)
            + (1 << 28);
        crate::cond::ressources::admettre(
            &format!("cond facteur3 repliques d={} L={}", prm.d, prm.l),
            crate::machine::admission::BesoinMemoire { hote, gpu: 0 },
            None,
        )?;
    }
    let garde = GardeSsd::machine();
    let meta = executer_cellule(&prm, &g, &p, &dir, &garde, m.get_flag("acc"))?;
    println!("{}", serde_json::json!({"sortie": dir.display().to_string(), "npaires": meta["npaires"], "repliques": meta["repliques"]}));
    Ok(crate::commun::code::OK)
}

/// Lit une liste de canaux au format JSON : [{"type": "stein_libre", "li": 12, "a": 1, "pi": 34}, {"type": "stein_ligne", "ci": 5, "a": 0, "pi": 9},
/// {"type": "delta_s", "ci": 5, "a": 2, "t": 0.1}, {"type": "moments_lignes"}, {"type": "groupes_plaquettes"}].
pub fn lire_canaux(p: &std::path::Path) -> Resultat<Vec<super::modele::Canal>> {
    use super::modele::Canal;
    let t = std::fs::read_to_string(p).map_err(|e| Erreur::io(format!("lecture {}", p.display()), e))?;
    let v: serde_json::Value = serde_json::from_str(&t).map_err(|e| Erreur::Format(format!("{} : {e}", p.display())))?;
    let liste = v.as_array().ok_or_else(|| Erreur::Format("canaux : une liste JSON est attendue".into()))?;
    let get = |c: &serde_json::Value, k: &str| -> Resultat<u64> {
        c[k].as_u64().ok_or_else(|| Erreur::Format(format!("canal sans entier « {k} » : {c}")))
    };
    let mut out = Vec::new();
    for c in liste {
        let typ = c["type"].as_str().unwrap_or("");
        out.push(match typ {
            "stein_libre" => Canal::SteinLibre { li: get(c, "li")? as u32, a: get(c, "a")? as u8, pi: get(c, "pi")? as u32 },
            "stein_ligne" => Canal::SteinLigne { ci: get(c, "ci")? as u32, a: get(c, "a")? as u8, pi: get(c, "pi")? as u32 },
            "delta_s" => Canal::DeltaS {
                ci: get(c, "ci")? as u32,
                a: get(c, "a")? as u8,
                t: c["t"].as_f64().ok_or_else(|| Erreur::Format(format!("canal sans « t » : {c}")))?,
            },
            "moments_lignes" => Canal::MomentsLignes,
            "groupes_plaquettes" => Canal::GroupesPlaquettes,
            "plaq" => Canal::Plaq { pi: get(c, "pi")? as u32 },
            autre => return Err(Erreur::Format(format!("type de canal inconnu : « {autre} »"))),
        });
    }
    Ok(out)
}

fn cmd_tube_classe(m: &ArgMatches) -> Resultat<i32> {
    let l = req::<u64>(m, "l")? as usize;
    let g = Geom::nouvelle(4, l)?;
    let tt = super::tube::TableTubes::nouvelle(&g);
    let dy: Vec<i64> = req::<String>(m, "dy")?.split(',').map(|x| x.trim().parse::<i64>().map_err(|e| Erreur::Config(format!("--dy : {e}")))).collect::<Resultat<_>>()?;
    if dy.len() != 4 {
        return Err(Erreur::Config("--dy attend 4 entiers".into()));
    }
    let sc = g.site_gros(&dy);
    let cls = tt.classe(req::<u64>(m, "pl")? as usize, req::<u64>(m, "rcode")? as usize, req::<u64>(m, "mue")? as usize, sc);
    let items: Vec<serde_json::Value> = tt
        .tube(cls)
        .iter()
        .map(|&it| {
            let (pl2, site) = (it as usize / g.vc, it as usize % g.vc);
            let c = g.coords_gros[site];
            serde_json::json!([pl2, c[0], c[1], c[2], c[3]])
        })
        .collect();
    println!("{}", serde_json::json!({"classe": cls, "taille": items.len(), "tube": items}));
    Ok(crate::commun::code::OK)
}

fn charges_config(l: usize, cfg: &str, bmin: u32, bmax: u32) -> Resultat<(Geom, Paires, Vec<f32>)> {
    let g = Geom::nouvelle(4, l)?;
    let c = crate::stockage::npy::lire_champ(std::path::Path::new(cfg))?;
    if c.l != l {
        return Err(Erreur::Config(format!("{cfg} : L = {} au lieu de {l}", c.l)));
    }
    let u: Vec<crate::heritage::su2_hb::Q> = c.q.chunks_exact(4).map(|q| [q[0], q[1], q[2], q[3]]).collect();
    let v: Vec<crate::heritage::su2_hb::Q> = g.lignes.iter().map(|l3| super::su2::produit_ligne(&u, l3)).collect();
    let p = Paires::nouvelles(&g, g.faces(), g.sondes_paires(), bmin, bmax);
    let tt = super::tube::TableTubes::nouvelle(&g);
    let cp = super::tube::charges_grossieres(&g, &v);
    let f = super::tube::charges_paires(&g, &p, &tt, &cp);
    Ok((g, p, f))
}

fn cmd_tube_f(m: &ArgMatches) -> Resultat<i32> {
    if let Some(n) = m.get_one::<usize>("fils") {
        let _ = rayon::ThreadPoolBuilder::new().num_threads((*n).max(1)).build_global();
    }
    let (_g, p, f) = charges_config(req::<u64>(m, "l")? as usize, &req::<String>(m, "cfg")?, req_u32(m, "bmin")?, req_u32(m, "bmax")?)?;
    let f64s: Vec<f64> = f.iter().map(|&x| x as f64).collect();
    crate::stockage::npy::ecrire_f64(std::path::Path::new(&req::<String>(m, "sortie")?), &[f64s.len()], &f64s, &GardeSsd::machine())?;
    println!("{}", serde_json::json!({"npaires": p.npaires()}));
    Ok(crate::commun::code::OK)
}

/// Nombre de classes fines de l'histogramme de charge de tube de la passe 1 (par tranche), sur [0, 2].
pub const NB_HIST_F: usize = 1 << 16;

fn cmd_tube_hist(m: &ArgMatches) -> Resultat<i32> {
    if let Some(n) = m.get_one::<usize>("fils") {
        let _ = rayon::ThreadPoolBuilder::new().num_threads((*n).max(1)).build_global();
    }
    let (_g, p, f) = charges_config(req::<u64>(m, "l")? as usize, &req::<String>(m, "cfg")?, req_u32(m, "bmin")?, req_u32(m, "bmax")?)?;
    let nb = p.bmax as usize + 1;
    let mut h = vec![0u32; nb * NB_HIST_F];
    for q in 0..p.npaires() {
        let b = p.tranche[q] as usize;
        let k = ((f[q] as f64 / 2.0) * NB_HIST_F as f64) as usize;
        h[b * NB_HIST_F + k.min(NB_HIST_F - 1)] += 1;
    }
    crate::stockage::npy::ecrire_u32(std::path::Path::new(&req::<String>(m, "sortie")?), &[nb, NB_HIST_F], &h, &GardeSsd::machine())?;
    let fmin = f.iter().cloned().fold(f32::INFINITY, f32::min);
    let fmax = f.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("{}", serde_json::json!({"npaires": p.npaires(), "f_min": fmin, "f_max": fmax, "nbins": NB_HIST_F}));
    Ok(crate::commun::code::OK)
}
