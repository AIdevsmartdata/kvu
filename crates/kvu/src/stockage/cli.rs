//! Sous-commandes `store` (du, gc, verifier, lister, exporter) et `fetch`. Propriétaire : T8.
//!
//! ```text
//! kevinotron store [--store DIR] du [--chemin X…] [--projeter L:CHAINES:DUMPS:FORMAT[:REPRISES]] [--json]
//! kevinotron store [--store DIR] gc [--appliquer] [--age-h 24] [--garder-chaque K] [--convertir-q64] [--zstd]
//!                                   [--garder-reprises] [--json]
//! kevinotron store [--store DIR] verifier ID
//! kevinotron store [--store DIR] lister [--json]
//! kevinotron store [--store DIR] exporter ID DEST [--format c16|q64]
//! kevinotron fetch <machine>:<chemin|store/ID> [DEST] [--cable HOTE]
//! ```
//!
//! `gc` simule par défaut ; sans `--appliquer`, RIEN n'est modifié. Codes de sortie : ceux de `commun::code`
//! (SSD plein = 7, vérification rouge = 8…), rendus par `cli::principal` à partir de l'erreur.

use super::fetch;
use super::garde::{projeter, GardeSsd, OCTETS_PAR_GO};
use super::npy::FormatChamp;
use super::store::{
    verifier_repertoire, LigneDu, LigneListe, PolitiqueEclaircissement, RapportGc, Store, AGE_ORPHELIN_DEFAUT_S,
};
use crate::commun::{Erreur, Resultat};
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::path::{Path, PathBuf};

pub fn commandes() -> Vec<clap::Command> {
    vec![commande_store(), commande_fetch()]
}

fn commande_store() -> Command {
    Command::new("store")
        .about("Store d'ensembles : du | gc | verifier | lister | exporter | classer (simulation A/B/C)")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .arg(
            Arg::new("store")
                .long("store")
                .global(true)
                .value_name("DIR")
                .help("racine du store (défaut : KEVINOTRON_STORE, sinon la politique de la machine)"),
        )
        .subcommand(
            Command::new("du")
                .about("occupation par ensemble (décroissante), total, Go libres, marge au seuil, projection")
                .arg(
                    Arg::new("chemin")
                        .long("chemin")
                        .value_name("X")
                        .action(ArgAction::Append)
                        .help("répertoire externe, en LECTURE seule"),
                )
                .arg(
                    Arg::new("projeter")
                        .long("projeter")
                        .value_name("L:CHAINES:DUMPS:FORMAT[:REPRISES]")
                        .help("projette la taille d'un run (ex. 64:1:20:c16) et dit si la garde SSD l'accepterait"),
                )
                .arg(Arg::new("json").long("json").action(ArgAction::SetTrue)),
        )
        .subcommand(
            Command::new("gc")
                .about("éclaircit le store : SIMULATION par défaut, --appliquer pour agir (jamais hors de la racine)")
                .arg(
                    Arg::new("appliquer")
                        .long("appliquer")
                        .action(ArgAction::SetTrue)
                        .help("agit réellement"),
                )
                .arg(
                    Arg::new("age-h")
                        .long("age-h")
                        .value_name("H")
                        .default_value("24")
                        .help("âge minimal d'un run orphelin (heures)"),
                )
                .arg(
                    Arg::new("garder-chaque")
                        .long("garder-chaque")
                        .value_name("K")
                        .default_value("1")
                        .help("un dump sur K (premier et dernier gardés) ; 1 = tout garder"),
                )
                .arg(
                    Arg::new("convertir-q64")
                        .long("convertir-q64")
                        .action(ArgAction::SetTrue)
                        .help("c16 -> q64, aller-retour vérifié"),
                )
                .arg(
                    Arg::new("zstd")
                        .long("zstd")
                        .action(ArgAction::SetTrue)
                        .help("compresse les dumps conservés (zstd -3)"),
                )
                .arg(
                    Arg::new("garder-reprises")
                        .long("garder-reprises")
                        .action(ArgAction::SetTrue)
                        .help("ne purge pas les reprises des runs COMPLETE"),
                )
                .arg(Arg::new("json").long("json").action(ArgAction::SetTrue)),
        )
        .subcommand(
            Command::new("verifier")
                .about("revérifie les sha256 du manifeste d'un ensemble (ID du store, ou chemin d'un répertoire)")
                .arg(Arg::new("id").required(true).value_name("ID|CHEMIN")),
        )
        .subcommand(
            Command::new("lister")
                .about("liste les ensembles")
                .arg(Arg::new("json").long("json").action(ArgAction::SetTrue)),
        )
        .subcommand(crate::regen::cli::commande_classer())
        .subcommand(
            Command::new("exporter")
                .about("exporte un ensemble vers un répertoire neuf (dumps convertis en flux, c16 par défaut)")
                .arg(Arg::new("id").required(true))
                .arg(Arg::new("dest").required(true))
                .arg(
                    Arg::new("format")
                        .long("format")
                        .value_name("c16|q64")
                        .default_value("c16"),
                ),
        )
}

fn commande_fetch() -> Command {
    Command::new("fetch")
        .about("Rapatrie par le câble du profil (clé `cable` de kvu.toml), sha256 vérifiés, garde SSD appliquée sur la taille distante")
        .arg(Arg::new("source").required(true).value_name("MACHINE:<chemin|store/ID>"))
        .arg(
            Arg::new("dest")
                .value_name("DEST")
                .help("défaut : <store>/<ID> pour store/ID, sinon ./<nom>"),
        )
        .arg(
            Arg::new("cable")
                .long("cable")
                .value_name("HOTE")
                .help("hôte du câble direct (défaut : clé `cable` du profil de kvu.toml)"),
        )
        .arg(
            Arg::new("store")
                .long("store")
                .value_name("DIR")
                .help("racine du store local pour la destination par défaut"),
        )
}

pub fn executer(nom: &str, m: &ArgMatches) -> Resultat<i32> {
    match nom {
        "store" => executer_store(m),
        "fetch" => executer_fetch(m),
        autre => Err(Erreur::Config(format!(
            "sous-commande de stockage inconnue « {autre} »"
        ))),
    }
}

// ------------------------------------------------------------------------------------------
// Mise en forme (pures, testées)
// ------------------------------------------------------------------------------------------

/// Taille lisible : « 12.3 Go », « 4.1 Mo », « 850 Ko », « 12 o », en unités binaires (1 Go = 2³⁰ octets,
/// comme la garde SSD et `df -BG`).
pub fn taille_lisible(octets: u64) -> String {
    let o = octets as f64;
    let (go, mo, ko) = (OCTETS_PAR_GO, (1u64 << 20) as f64, 1024.0);
    if o >= go {
        format!("{:.2} Go", o / go)
    } else if o >= mo {
        format!("{:.1} Mo", o / mo)
    } else if o >= ko {
        format!("{:.0} Ko", o / ko)
    } else {
        format!("{octets} o")
    }
}

/// Durée lisible : « 12 s », « 5 min », « 3.2 h », « 4.5 j ».
pub fn duree_lisible(s: f64) -> String {
    if s < 90.0 {
        format!("{s:.0} s")
    } else if s < 5400.0 {
        format!("{:.0} min", s / 60.0)
    } else if s < 129_600.0 {
        format!("{:.1} h", s / 3600.0)
    } else {
        format!("{:.1} j", s / 86_400.0)
    }
}

/// Projection demandée par `--projeter`.
#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
    pub l: usize,
    pub chaines: usize,
    pub dumps: u64,
    pub format: FormatChamp,
    pub reprises: u64,
}

impl Projection {
    /// « L:CHAINES:DUMPS:FORMAT[:REPRISES] », ex. « 64:1:20:c16 ».
    pub fn analyser(s: &str) -> Resultat<Projection> {
        let v: Vec<&str> = s.split(':').collect();
        if v.len() < 4 || v.len() > 5 {
            return Err(Erreur::Config(format!(
                "projection « {s} » : forme L:CHAINES:DUMPS:FORMAT[:REPRISES]"
            )));
        }
        let nombre = |t: &str, nom: &str| {
            t.trim()
                .parse::<u64>()
                .map_err(|_| Erreur::Config(format!("projection : {nom} « {t} » n'est pas un entier")))
        };
        Ok(Projection {
            l: nombre(v[0], "L")? as usize,
            chaines: nombre(v[1], "CHAINES")? as usize,
            dumps: nombre(v[2], "DUMPS")?,
            format: v[3].parse()?,
            reprises: if v.len() == 5 { nombre(v[4], "REPRISES")? } else { 0 },
        })
    }

    pub fn octets(&self) -> u64 {
        projeter(self.l, self.chaines, self.dumps, self.format, self.reprises)
    }
}

/// Tableau de `store du`.
pub fn rendre_du(
    racine: &Path,
    lignes: &[LigneDu],
    libre_go: f64,
    garde: &GardeSsd,
    projection: Option<&Projection>,
) -> String {
    let mut s = String::new();
    s.push_str(&format!("store {}\n", racine.display()));
    s.push_str(&format!(
        "{:>10}  {:>8}  {:>8}  {:>9}  chemin\n",
        "taille", "fichiers", "COMPLETE", "âge"
    ));
    let mut total_store = 0u64;
    let mut total_externes = 0u64;
    for l in lignes {
        let interne = l.chemin.starts_with(racine);
        if interne {
            total_store += l.octets;
        } else {
            total_externes += l.octets;
        }
        s.push_str(&format!(
            "{:>10}  {:>8}  {:>8}  {:>9}  {}{}\n",
            taille_lisible(l.octets),
            l.fichiers,
            if l.complete { "oui" } else { "non" },
            duree_lisible(l.age_s),
            l.chemin.display(),
            if interne { "" } else { "  (externe, lecture seule)" }
        ));
    }
    s.push_str(&format!("total du store : {}\n", taille_lisible(total_store)));
    if total_externes > 0 {
        s.push_str(&format!("total externe  : {}\n", taille_lisible(total_externes)));
    }
    let marge = libre_go - garde.seuil_go;
    s.push_str(&format!(
        "libre : {libre_go:.1} Go ; seuil de la garde : {:.0} Go ; marge : {marge:.1} Go{}\n",
        garde.seuil_go,
        if marge < 0.0 {
            "  (SOUS LE SEUIL : toute écriture sera refusée)"
        } else {
            ""
        }
    ));
    if let Some(p) = projection {
        let octets = p.octets();
        let verdict = match garde.evaluer(racine, libre_go, octets) {
            Ok(()) => "ACCEPTÉ".to_string(),
            Err(_) => format!("REFUSÉ (il manque {:.1} Go)", octets as f64 / OCTETS_PAR_GO - marge),
        };
        s.push_str(&format!(
            "projection L={} x {} chaîne(s) x {} dump(s) {} + {} reprise(s) : {} -> {verdict}\n",
            p.l,
            p.chaines,
            p.dumps,
            p.format.nom(),
            p.reprises,
            taille_lisible(octets)
        ));
    }
    s
}

/// Rapport de `store gc`.
pub fn rendre_gc(r: &RapportGc) -> String {
    let mut s = String::new();
    s.push_str(if r.applique {
        "gc APPLIQUÉ\n"
    } else {
        "gc en SIMULATION : rien n'est modifié (relancer avec --appliquer pour agir)\n"
    });
    for (chemin, octets, raison) in &r.actions {
        s.push_str(&format!(
            "{:>10}  {:<28}  {}\n",
            taille_lisible(*octets),
            raison,
            chemin.display()
        ));
    }
    for (chemin, raison) in &r.ignores {
        s.push_str(&format!("{:>10}  {:<28}  {}\n", "-", raison, chemin.display()));
    }
    s.push_str(&format!(
        "{} {}\n",
        if r.applique { "libéré :" } else { "libérable :" },
        taille_lisible(r.octets_liberes)
    ));
    s
}

/// Tableau de `store lister`.
pub fn rendre_liste(l: &[LigneListe]) -> String {
    let mut s = format!(
        "{:<32} {:>10} {:>8} {:>9} {:>3}  {:<9} {}\n",
        "ensemble", "taille", "COMPLETE", "âge", "rc", "machine", "moteur"
    );
    for e in l {
        s.push_str(&format!(
            "{:<32} {:>10} {:>8} {:>9} {:>3}  {:<9} {}{}\n",
            e.id,
            taille_lisible(e.octets),
            if e.complete { "oui" } else { "non" },
            duree_lisible(e.age_s),
            e.rc.map(|r| r.to_string()).unwrap_or_else(|| "-".into()),
            e.machine.as_deref().unwrap_or("-"),
            e.moteur.as_deref().unwrap_or("-"),
            if e.garder { "  [.garder]" } else { "" }
        ));
    }
    s
}

// ------------------------------------------------------------------------------------------
// Exécution
// ------------------------------------------------------------------------------------------

fn racine_store(m: &ArgMatches) -> PathBuf {
    if let Some(r) = m.try_get_one::<String>("store").ok().flatten() {
        return PathBuf::from(r);
    }
    if let Ok(r) = std::env::var("KEVINOTRON_STORE") {
        if !r.trim().is_empty() {
            return PathBuf::from(r.trim());
        }
    }
    crate::machine::politique(&crate::machine::detection::detecter_rapide()).store
}

fn json_sortie<T: serde::Serialize>(v: &T) -> Resultat<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(v).map_err(|e| Erreur::Format(format!("sérialisation JSON : {e}")))?
    );
    Ok(())
}

fn executer_store(m: &ArgMatches) -> Resultat<i32> {
    let racine = racine_store(m);
    let (sous, a) = m
        .subcommand()
        .ok_or_else(|| Erreur::Config("store : sous-commande requise (du|gc|verifier|lister|exporter)".into()))?;
    if sous == "classer" {
        // Simulation de classement A/B/C (WP4) : ne touche à aucun store, aucune écriture.
        return crate::regen::cli::executer_classer(a);
    }
    let store = Store::ouvrir(&racine)?;
    match sous {
        "du" => {
            let autres: Vec<PathBuf> = a
                .get_many::<String>("chemin")
                .map(|v| v.map(PathBuf::from).collect())
                .unwrap_or_default();
            let lignes = store.du(&autres)?;
            let garde = GardeSsd::machine();
            let libre = GardeSsd::libre_go(&racine)?;
            let projection = a
                .get_one::<String>("projeter")
                .map(|s| Projection::analyser(s))
                .transpose()?;
            if a.get_flag("json") {
                json_sortie(&serde_json::json!({
                    "racine": racine, "lignes": lignes, "libre_go": libre, "seuil_go": garde.seuil_go, "marge_go": libre - garde.seuil_go,
                    "projection_octets": projection.as_ref().map(Projection::octets),
                }))?;
            } else {
                print!("{}", rendre_du(&racine, &lignes, libre, &garde, projection.as_ref()));
            }
            Ok(0)
        }
        "gc" => {
            let age_h: f64 = a
                .get_one::<String>("age-h")
                .map(|s| s.parse())
                .transpose()
                .map_err(|_| Erreur::Config("--age-h : nombre attendu".into()))?
                .unwrap_or(AGE_ORPHELIN_DEFAUT_S / 3600.0);
            let k: u64 = a
                .get_one::<String>("garder-chaque")
                .map(|s| s.parse())
                .transpose()
                .map_err(|_| Erreur::Config("--garder-chaque : entier attendu".into()))?
                .unwrap_or(1);
            let p = PolitiqueEclaircissement {
                garder_chaque: k,
                convertir_q64: a.get_flag("convertir-q64"),
                compresser_zstd: a.get_flag("zstd"),
                purger_reprises: !a.get_flag("garder-reprises"),
            };
            let r = store.gc(&p, age_h * 3600.0, a.get_flag("appliquer"))?;
            if a.get_flag("json") {
                json_sortie(&r)?;
            } else {
                print!("{}", rendre_gc(&r));
            }
            Ok(0)
        }
        "verifier" => {
            let id = a.get_one::<String>("id").expect("requis");
            // Un chemin (absolu, relatif avec « / » ou « . », ou répertoire existant hors du store) est vérifié tel
            // quel : un fetch refusé se contrôle là où il a été déposé.
            let chemin = Path::new(id);
            let est_chemin = chemin.is_absolute() || id.starts_with('.') || (chemin.is_dir() && store.chemin_ensemble(id).is_err());
            let dir = if est_chemin { chemin.to_path_buf() } else { store.chemin_ensemble(id)? };
            let r = verifier_repertoire(&dir)?;
            if dir.join(super::fetch::NOM_FETCH_REFUSE).exists() {
                return Err(Erreur::Validation(format!(
                    "{id} : fetch REFUSÉ ({} présent) : ensemble partiel",
                    super::fetch::NOM_FETCH_REFUSE
                )));
            }
            r.en_resultat(id)?;
            println!(
                "ensemble {id} : OK, {} unité(s), {} fichier(s) vérifié(s), {}",
                r.unites,
                r.fichiers_verifies,
                taille_lisible(r.octets_verifies)
            );
            Ok(0)
        }
        "lister" => {
            let l = store.lister()?;
            if a.get_flag("json") {
                json_sortie(&l)?;
            } else {
                print!("{}", rendre_liste(&l));
            }
            Ok(0)
        }
        "exporter" => {
            let id = a.get_one::<String>("id").expect("requis");
            let dest = PathBuf::from(a.get_one::<String>("dest").expect("requis"));
            let format: FormatChamp = a
                .get_one::<String>("format")
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(FormatChamp::C16);
            let r = store.exporter(id, &dest, format, &GardeSsd::machine())?;
            println!(
                "exporté : {} fichier(s), dont {} converti(s) en {}, {} vers {}",
                r.fichiers,
                r.convertis,
                format.nom(),
                taille_lisible(r.octets),
                r.destination.display()
            );
            Ok(0)
        }
        autre => Err(Erreur::Config(format!("store : sous-commande inconnue « {autre} »"))),
    }
}

/// Destination par défaut d'un fetch.
pub fn destination_par_defaut(source: &str, racine_store_local: &Path) -> Resultat<PathBuf> {
    let s = fetch::analyser_source(source)?;
    let nom = s
        .chemin
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("fetch")
        .to_string();
    Ok(
        if source.split_once(':').is_some_and(|(_, r)| r.starts_with("store/")) {
            racine_store_local.join(nom)
        } else {
            PathBuf::from(".").join(nom)
        },
    )
}

fn executer_fetch(m: &ArgMatches) -> Resultat<i32> {
    let source = m.get_one::<String>("source").expect("requis");
    let dest = match m.get_one::<String>("dest") {
        Some(d) => PathBuf::from(d),
        None => destination_par_defaut(source, &racine_store(m))?,
    };
    let hote = m
        .get_one::<String>("cable")
        .cloned()
        .unwrap_or_else(fetch::hote_cable_defaut);
    if hote.trim().is_empty() {
        return Err(Erreur::Config(
            "aucun câble : --cable HOTE, KEVINOTRON_CABLE ou la clé `cable` d'un profil de kvu.toml".to_string(),
        ));
    }
    let r = fetch::recuperer(source, &dest, &GardeSsd::machine(), &hote)?;
    println!(
        "fetch {} -> {} : {} en {:.1} s ({:.0} Mo/s), {} sha256 vérifié(s)",
        r.source,
        r.destination.display(),
        taille_lisible(r.octets),
        r.secondes,
        r.octets as f64 / 1e6 / r.secondes.max(1e-3),
        r.sha_verifies
    );
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commandes_coherentes() {
        for c in commandes() {
            c.debug_assert();
        }
        let noms: Vec<String> = commandes().iter().map(|c| c.get_name().to_string()).collect();
        assert_eq!(noms, ["store", "fetch"]);
    }

    #[test]
    fn analyse_des_arguments() {
        let m = commande_store()
            .try_get_matches_from(["store", "--store", "/x", "gc", "--appliquer", "--garder-chaque", "5"])
            .unwrap();
        assert_eq!(m.get_one::<String>("store").unwrap(), "/x");
        let (nom, a) = m.subcommand().unwrap();
        assert_eq!(nom, "gc");
        assert!(a.get_flag("appliquer"));
        // simulation par défaut
        let m = commande_store().try_get_matches_from(["store", "gc"]).unwrap();
        assert!(!m.subcommand().unwrap().1.get_flag("appliquer"));
        assert!(commande_store().try_get_matches_from(["store"]).is_err());
        let m = commande_fetch()
            .try_get_matches_from(["fetch", "gamer:store/r1", "--cable", "192.0.2.2"])
            .unwrap();
        assert_eq!(m.get_one::<String>("cable").unwrap(), "192.0.2.2");
    }

    #[test]
    fn projection_analysee() {
        let p = Projection::analyser("64:1:20:c16").unwrap();
        assert_eq!(
            p,
            Projection {
                l: 64,
                chaines: 1,
                dumps: 20,
                format: FormatChamp::C16,
                reprises: 0
            }
        );
        assert!(p.octets() > 85_000_000_000);
        assert_eq!(Projection::analyser("8:4:10:q64:2").unwrap().reprises, 2);
        assert!(Projection::analyser("64:1:20").is_err());
        assert!(Projection::analyser("x:1:2:c16").is_err());
        assert!(Projection::analyser("8:1:2:f32").is_err());
    }

    #[test]
    fn lisibles() {
        assert_eq!(taille_lisible(4_300_000_000), "4.00 Go"); // 1 Go = 2³⁰ octets
        assert_eq!(taille_lisible(3 << 20), "3.0 Mo");
        assert_eq!(taille_lisible(12), "12 o");
        assert_eq!(duree_lisible(30.0), "30 s");
        assert_eq!(duree_lisible(7200.0), "2.0 h");
        assert_eq!(duree_lisible(200_000.0), "2.3 j");
    }

    #[test]
    fn du_rendu_refuse_la_projection_c16_l64() {
        let g = GardeSsd::nouvelle(100.0);
        let p = Projection::analyser("64:1:20:c16").unwrap();
        let s = rendre_du(Path::new("/store"), &[], 159.0, &g, Some(&p));
        assert!(s.contains("REFUSÉ"), "{s}");
        let ok = rendre_du(Path::new("/store"), &[], 400.0, &g, Some(&p));
        assert!(ok.contains("ACCEPTÉ"), "{ok}");
    }

    #[test]
    fn destination_par_defaut_du_fetch() {
        assert_eq!(
            destination_par_defaut("gamer:store/run7", Path::new("/data/store")).unwrap(),
            PathBuf::from("/data/store/run7")
        );
        assert_eq!(
            destination_par_defaut("gamer:/home/k/x.npy", Path::new("/data/store")).unwrap(),
            PathBuf::from("./x.npy")
        );
        assert!(destination_par_defaut("gamer:../x", Path::new("/s")).is_err());
    }
}
