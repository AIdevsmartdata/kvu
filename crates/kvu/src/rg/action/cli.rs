//! Sous-commandes `rg action` et `rg action-gates`.
//!
//! ```text
//! kvu rg action --ensemble DIR [--filtre MOTIF] [--operateur 3centre|dec|ape:<α>:<n>|flot:<t>[:<ε>]|agr:<c>] [--hors-famille] [--pas K]
//!               [--famille P,R,C,T] [--estimateurs pl,sd] [--jackknife chaines|blocs:B] [--max-configs N] [--sortie DIR]
//!               [--fils N] [--memoire-max-go X]
//! kvu rg action-gates [--grandl DIR] [--seul ID,…] [--sans-mutants] [--json] [--fils N]
//! ```
//!
//! `action` écrit `<sortie>/action.json` (refus d'écraser, écriture atomique). Son contenu ne dépend
//! ni de la durée ni du nombre de fils.

use super::boucles::{Famille, TableFormes};
use super::estimation::{
    accumuler_config, differences_appariees, estimer, verifier_tailles_groupes, Estimateur,
    ModeJackknife, StatsGroupe,
};
use super::gates::{batterie, Contexte};
use super::{L_MIN_ESTIMATION, SCHEMA_ACTION, SEUIL_PR, SEUIL_R4};
use crate::commun::{code, Erreur, Resultat};
use crate::groupes::gates::{tout_vert, Verdict};
use crate::rg::flot::cellule::correspond;
use crate::rg::flot::jackknife::jackknife_groupes;
use crate::rg::flot::sd::{beta_sd, stats_sd, ModeRapport, RapportBessel, StatsSd};
use crate::rg::lissage::{bloquer_operateur, OptionsOperateur};
use crate::rg::operateur::Operateur;
use crate::stockage::npy::lire_champ;
use clap::{Arg, ArgAction, ArgMatches, Command};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Chaîne découverte par la CLI (dossier contenant manifest.json et des configurations).
#[derive(Clone, Debug)]
pub(crate) struct ChaineTrouvee {
    pub(crate) nom: String,
    #[allow(dead_code)] // utilisé par d'éventuels diagnostics ; conservé pour l'API interne
    pub(crate) dossier: PathBuf,
    pub(crate) beta: f64,
    pub(crate) l: usize,
    pub(crate) configs: Vec<PathBuf>,
}

pub fn sous_commandes() -> Vec<Command> {
    let fils = || {
        Arg::new("fils").long("fils").num_args(1).help(
            "nombre de fils (défaut : RAYON_NUM_THREADS ou 4) ; les sorties n'en dépendent pas",
        )
    };
    vec![
        Command::new("action")
            .about("Couplages d'une action effective tronquée (plaquette, rectangle, chaise, torsade) sur des configurations bloquées")
            .arg(Arg::new("ensemble").long("ensemble").required(true).num_args(1).help("dossier d'une chaîne (cfg_*.npy + manifest.json) ou parent de chaînes"))
            .arg(Arg::new("filtre").long("filtre").num_args(1).help("motif (jokers *) des dossiers de chaînes"))
            .arg(Arg::new("operateur").long("operateur").num_args(1).default_value("3centre").help("opérateur de blocage : 3centre | dec | ape:<α>:<n> | flot:<t>[:<ε>] | agr:<c>"))
            .arg(Arg::new("hors-famille").long("hors-famille").action(ArgAction::SetTrue).help("accepte un opérateur hors de la famille quasi-locale (mutant)"))
            .arg(Arg::new("pas").long("pas").num_args(1).default_value("0").help("nombre de pas de blocage (défaut : aucun)"))
            .arg(Arg::new("famille").long("famille").num_args(1).default_value("P,R,C,T").help("famille de boucles : sous-ensemble de P,R,C,T (P obligatoire)"))
            .arg(Arg::new("estimateurs").long("estimateurs").num_args(1).default_value("pl,sd").help("estimateurs demandés : pl,sd"))
            .arg(Arg::new("jackknife").long("jackknife").num_args(1).default_value("chaines").help("mode jackknife : chaines | blocs:B"))
            .arg(Arg::new("max-configs").long("max-configs").num_args(1).help("nombre maximal de configurations par chaîne"))
            .arg(Arg::new("sortie").long("sortie").num_args(1).default_value(".").help("dossier de sortie (action.json)"))
            .arg(Arg::new("memoire-max-go").long("memoire-max-go").num_args(1).default_value("4").help("mémoire maximale pour PL (Go)"))
            .arg(fils()),
        Command::new("action-gates")
            .about("Gates RG-ACTION-* : une ligne PASS|FAIL|SKIP|INFO par gate, rc 0 ssi tout est vert")
            .arg(Arg::new("grandl").long("grandl").num_args(1).help("racine des ensembles grandL ; absent ⇒ les gates WILSON sont SKIP"))
            .arg(Arg::new("seul").long("seul").num_args(1).help("préfixes d'ID à exécuter (liste à virgules)"))
            .arg(Arg::new("sans-mutants").long("sans-mutants").action(ArgAction::SetTrue))
            .arg(Arg::new("json").long("json").action(ArgAction::SetTrue))
            .arg(fils()),
    ]
}

fn nombre_fils(m: &ArgMatches) -> Resultat<usize> {
    let n = match m.get_one::<String>("fils") {
        Some(t) => t
            .parse::<usize>()
            .map_err(|_| Erreur::Config(format!("--fils « {t} » invalide")))?,
        None => std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(4),
    };
    if n == 0 {
        return Err(Erreur::Config("--fils doit être ≥ 1".into()));
    }
    Ok(n)
}

/// Découverte des chaînes : `ens` est une chaîne (contient manifest.json et des `cfg_*.npy` /
/// `cfg_*.npy.zst`) ou un parent de chaînes (sous-dossiers triés, filtrés par `--filtre`).
/// `cfg_000000.*` est exclu ; un même indice sous les deux extensions est refusé.
pub(crate) fn decouvrir_chaines(ens: &Path, filtre: Option<&str>) -> Resultat<Vec<ChaineTrouvee>> {
    fn lire_une(dossier: &Path) -> Resultat<Option<ChaineTrouvee>> {
        let mut configs: Vec<PathBuf> = std::fs::read_dir(dossier)
            .map_err(|e| Erreur::io(format!("lecture de {}", dossier.display()), e))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                    let base = n.strip_suffix(".zst").unwrap_or(n);
                    base.starts_with("cfg_") && base.ends_with(".npy") && base != "cfg_000000.npy"
                })
            })
            .collect();
        if configs.is_empty() && !dossier.join("manifest.json").exists() {
            return Ok(None);
        }
        configs.sort();
        // refus si un même indice existe sous les deux extensions
        let mut indices = std::collections::HashSet::new();
        for c in &configs {
            let nom = c.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let indice = nom
                .trim_start_matches("cfg_")
                .trim_end_matches(".npy.zst")
                .trim_end_matches(".npy");
            if !indices.insert(indice.to_string()) {
                return Err(Erreur::Config(format!(
                    "indice de configuration « {indice} » en double dans {}",
                    dossier.display()
                )));
            }
        }
        let manifeste_path = dossier.join("manifest.json");
        let texte = std::fs::read_to_string(&manifeste_path)
            .map_err(|e| Erreur::io(format!("manifest.json de {}", dossier.display()), e))?;
        let man: Value = serde_json::from_str(&texte)
            .map_err(|e| Erreur::Format(format!("manifest.json de {} : {e}", dossier.display())))?;
        // `beta` (manifestes de production su2_hip/1) ou `betas` (manifeste de `kvu su2 gen`, une valeur par chaîne du lot) :
        // un dossier de chaîne de `su2 gen` n'en porte qu'une ; un lot de plusieurs β dans un même dossier est refusé.
        let beta = match (man["beta"].as_f64(), man["betas"].as_array().map(|v| v.as_slice())) {
            (Some(b), _) => b,
            (None, Some([b])) => b
                .as_f64()
                .ok_or_else(|| Erreur::Format(format!("betas non numérique dans {}", dossier.display())))?,
            (None, Some(v)) if v.len() > 1 => {
                return Err(Erreur::Format(format!(
                    "{} : manifeste d'un lot de {} chaînes (betas) : rg action lit une chaîne, donc un seul β, à la fois",
                    dossier.display(),
                    v.len()
                )))
            }
            _ => return Err(Erreur::Format(format!("beta absent de {}", dossier.display()))),
        };
        let l = man["l"]
            .as_u64()
            .and_then(|l| usize::try_from(l).ok())
            .filter(|&l| l > 0)
            .ok_or_else(|| {
                Erreur::Format(format!("l absent ou invalide de {}", dossier.display()))
            })?;
        if !beta.is_finite() {
            return Err(Erreur::Format(format!(
                "beta non fini de {}",
                dossier.display()
            )));
        }
        if configs.is_empty() {
            return Ok(None);
        }
        Ok(Some(ChaineTrouvee {
            nom: dossier
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            dossier: dossier.to_path_buf(),
            beta,
            l,
            configs,
        }))
    }

    if let Some(c) = lire_une(ens)? {
        return Ok(vec![c]);
    }
    let mut dossiers: Vec<PathBuf> = std::fs::read_dir(ens)
        .map_err(|e| Erreur::io(format!("lecture de {}", ens.display()), e))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter(|p| {
            filtre.is_none_or(|m| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| correspond(m, n))
            })
        })
        .collect();
    dossiers.sort();
    let mut chaines = Vec::new();
    for d in dossiers {
        if let Some(c) = lire_une(&d)? {
            chaines.push(c);
        }
    }
    Ok(chaines)
}

fn texte<'a>(m: &'a ArgMatches, nom: &str) -> Resultat<&'a str> {
    m.get_one::<String>(nom)
        .map(String::as_str)
        .ok_or_else(|| Erreur::Config(format!("--{nom} requis")))
}

/// Le témoin historique ne cherche que dans [0, 50]. Un résidu non nul invalide sa sortie,
/// notamment quand les données demanderaient un couplage négatif.
fn beta_sd_verifie(stats: &StatsSd) -> Option<f64> {
    if stats.somme_w < 0.0 {
        return None;
    }
    let beta = beta_sd(stats, RapportBessel::I2SurI1, ModeRapport::Exact)?;
    if !(0.0..=50.0).contains(&beta) {
        return None;
    }
    let rhs: f64 = stats
        .ks
        .iter()
        .map(|&k| k * crate::rg::flot::bessel::i2_sur_i1(beta * k))
        .sum();
    let echelle = stats.somme_w.abs().max(1.0);
    ((rhs - stats.somme_w).abs() <= 1e-10 * echelle).then_some(beta)
}

fn action_cmd(m: &ArgMatches) -> Resultat<i32> {
    let ensemble = m
        .get_one::<String>("ensemble")
        .cloned()
        .ok_or_else(|| Erreur::Config("--ensemble requis".into()))?;
    let filtre = m.get_one::<String>("filtre").cloned();
    let sortie = m
        .get_one::<String>("sortie")
        .cloned()
        .ok_or_else(|| Erreur::Config("--sortie requis".into()))?;
    let operateur_spec = m
        .get_one::<String>("operateur")
        .cloned()
        .ok_or_else(|| Erreur::Config("--operateur requis".into()))?;
    let hors_famille = m.get_flag("hors-famille");
    let k_pas: usize = texte(m, "pas")?
        .parse()
        .map_err(|_| Erreur::Config("--pas invalide".into()))?;
    let famille = Famille::analyser(texte(m, "famille")?)?;
    let estimateurs_txt = texte(m, "estimateurs")?;
    let mut estimateurs_demandes: Vec<Estimateur> = estimateurs_txt
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(Estimateur::analyser)
        .collect::<Resultat<_>>()?;
    if estimateurs_demandes.is_empty() {
        return Err(Erreur::Config("aucun estimateur demandé".into()));
    }
    estimateurs_demandes.sort_by_key(|e| e.nom());
    estimateurs_demandes.dedup();
    let mode_jackknife = ModeJackknife::analyser(texte(m, "jackknife")?)?;
    let max_configs = match m.get_one::<String>("max-configs") {
        Some(t) => Some(
            t.parse::<usize>()
                .map_err(|_| Erreur::Config("--max-configs invalide".into()))?,
        ),
        None => None,
    };
    let memoire_max_go: f64 = texte(m, "memoire-max-go")?
        .parse()
        .map_err(|_| Erreur::Config("--memoire-max-go invalide".into()))?;
    let fils = nombre_fils(m)?;

    if !memoire_max_go.is_finite() || memoire_max_go <= 0.0 {
        return Err(Erreur::Config(
            "--memoire-max-go doit être fini et > 0".into(),
        ));
    }
    if max_configs == Some(0) {
        return Err(Erreur::Config("--max-configs doit être ≥ 1".into()));
    }
    // L'opérateur et 3^K sont validés avant toute découverte/lecture de manifeste.
    let operateur = Operateur::analyser(&operateur_spec)?;
    operateur.verifier_famille(hors_famille)?;
    let exposant =
        u32::try_from(k_pas).map_err(|_| Erreur::Config("--pas trop grand pour 3^K".into()))?;
    let facteur = 3usize
        .checked_pow(exposant)
        .ok_or_else(|| Erreur::Config("3^K déborde : --pas trop grand".into()))?;
    let chemin_sortie = Path::new(&sortie).join("action.json");
    if chemin_sortie.exists() {
        return Err(Erreur::Config(format!(
            "{} existe déjà : refus d'écraser",
            chemin_sortie.display()
        )));
    }

    let chaines = decouvrir_chaines(Path::new(&ensemble), filtre.as_deref())?;
    if chaines.is_empty() {
        return Err(Erreur::Config("ensemble sans configuration".into()));
    }
    if matches!(mode_jackknife, ModeJackknife::Chaines) && chaines.len() < 2 {
        return Err(Erreur::Config(
            "jackknife chaines requiert au moins 2 chaînes".into(),
        ));
    }
    let (beta_ref, l_ref) = (chaines[0].beta, chaines[0].l);
    for c in chaines.iter().skip(1) {
        if (c.beta - beta_ref).abs() > 1e-9 || c.l != l_ref {
            return Err(Erreur::Config("β ou L différents entre chaînes".into()));
        }
    }

    if !l_ref.is_multiple_of(facteur) {
        return Err(Erreur::Config(format!(
            "L = {l_ref} non divisible par 3^{k_pas}"
        )));
    }
    let l_estimation = l_ref / facteur;
    if l_estimation < L_MIN_ESTIMATION {
        return Err(Erreur::Config(format!(
            "L/3^K = {l_estimation} < {L_MIN_ESTIMATION}"
        )));
    }

    let table = TableFormes::construire()?;
    let n = famille.n();
    let n_tri = n * (n + 1) / 2;

    let effectifs: Vec<usize> = chaines
        .iter()
        .map(|c| c.configs.len().min(max_configs.unwrap_or(usize::MAX)))
        .collect();
    let n_configs_total = effectifs
        .iter()
        .try_fold(0usize, |a, &b| a.checked_add(b))
        .ok_or_else(|| Erreur::Config("trop de configurations".into()))?;
    if n_configs_total == 0 {
        return Err(Erreur::Config("aucune configuration à lire".into()));
    }
    let tailles = match mode_jackknife {
        ModeJackknife::Chaines => effectifs.clone(),
        ModeJackknife::Blocs(b) => {
            if b > n_configs_total {
                return Err(Erreur::Config(format!(
                    "blocs:{b} > nombre de configurations {n_configs_total}"
                )));
            }
            (0..b)
                .map(|g| n_configs_total / b + usize::from(g < n_configs_total % b))
                .collect()
        }
    };
    verifier_tailles_groupes(&tailles)?;
    let liens_par_config = l_estimation
        .checked_pow(4)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| Erreur::Config("volume trop grand".into()))?;
    let n_liens_total = n_configs_total
        .checked_mul(liens_par_config)
        .ok_or_else(|| Erreur::Config("nombre de liens trop grand".into()))?;
    // Décision AVANT toute allocation de G, y compris pour une seule configuration.
    let octets = n_liens_total
        .checked_mul(n_tri)
        .and_then(|v| v.checked_mul(std::mem::size_of::<f64>()));
    let demande_pl = estimateurs_demandes.contains(&Estimateur::Pl);
    let mut pl_memoire_insuffisante =
        demande_pl && octets.is_none_or(|v| v as f64 > memoire_max_go * 1024.0_f64.powi(3));
    let mut groupes: Vec<StatsGroupe> = tailles.iter().map(|_| StatsGroupe::default()).collect();
    if demande_pl && !pl_memoire_insuffisante {
        for (g, &taille) in groupes.iter_mut().zip(&tailles) {
            let len = taille * liens_par_config * n_tri; // borné par octets, vérifié ci-dessus
            if g.g.try_reserve_exact(len).is_err() {
                pl_memoire_insuffisante = true;
                break;
            }
        }
    }
    if pl_memoire_insuffisante {
        // Libère aussi les réservations précédant un éventuel refus de l'allocateur.
        for g in &mut groupes {
            g.g = Vec::new();
        }
    }
    let conserver_g = demande_pl && !pl_memoire_insuffisante;
    let mut groupes_sd = vec![StatsSd::default(); tailles.len()];
    let mut info_chaines = Vec::with_capacity(chaines.len());
    let (mut groupe_idx, mut dans_groupe) = (0, 0);
    for (chaine, &effectif) in chaines.iter().zip(&effectifs) {
        let configs = &chaine.configs[..effectif];
        let mut p_moy_chaine = 0.0;
        for chemin in configs {
            let mut champ = lire_champ(chemin)?;
            if champ.l != l_ref {
                return Err(Erreur::Config(format!(
                    "{} : côté réel {} différent du manifeste L={l_ref}",
                    chemin.display(),
                    champ.l
                )));
            }
            let mut attendu = l_ref;
            for _ in 0..k_pas {
                champ = bloquer_operateur(&champ, &operateur, &OptionsOperateur::normale(fils))?
                    .grossier;
                attendu /= 3;
                if champ.l != attendu {
                    return Err(Erreur::Config(format!(
                        "côté après blocage {} au lieu de {attendu}",
                        champ.l
                    )));
                }
            }
            if champ.l != l_estimation {
                return Err(Erreur::Config("côté d'estimation incohérent".into()));
            }
            p_moy_chaine += crate::rg::swendsen::plaquette_moyenne(&champ);
            accumuler_config(
                &champ,
                &table,
                &famille,
                fils,
                conserver_g,
                &mut groupes[groupe_idx],
            )?;
            // Les k du témoin sont déplacés dans le groupe, jamais conservés par configuration.
            let mut sd_cfg = stats_sd(&champ)?;
            let sd = &mut groupes_sd[groupe_idx];
            sd.somme_w += sd_cfg.somme_w;
            sd.ks.append(&mut sd_cfg.ks);
            sd.n_configs += 1;
            dans_groupe += 1;
            if dans_groupe == tailles[groupe_idx] {
                groupe_idx += 1;
                dans_groupe = 0;
            }
        }
        info_chaines.push(json!({
            "nom": chaine.nom, "n_configs": effectif,
            "premiere": configs.first().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            "derniere": configs.last().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            "p_moyenne": p_moy_chaine / effectif as f64,
        }));
    }
    debug_assert_eq!(groupe_idx, groupes.len());
    let stats_sd_toutes = StatsSd::somme(groupes_sd.iter());
    let beta_sd_un_valeur = beta_sd_verifie(&stats_sd_toutes);
    let residu_temoin = beta_sd_un_valeur.map(|b| {
        stats_sd_toutes
            .ks
            .iter()
            .map(|&k| k * crate::rg::flot::bessel::i2_sur_i1(b * k))
            .sum::<f64>()
            - stats_sd_toutes.somme_w
    });
    drop(stats_sd_toutes);
    let beta_sd_un_res = jackknife_groupes(&groupes_sd, |g| {
        beta_sd_verifie(&StatsSd::somme(g.iter().copied()))
    });
    let beta_sd_un_erreur = match (beta_sd_un_valeur, beta_sd_un_res) {
        (Some(_), Some(r)) => Some(r.erreur),
        _ => None,
    };

    // Estimation des estimateurs demandés
    let mut resultats: serde_json::Map<String, Value> = serde_json::Map::new();
    let mut tous_ok = true;
    let mut couplages_pl = None;
    let mut couplages_sd = None;

    if estimateurs_demandes.contains(&Estimateur::Pl) {
        if pl_memoire_insuffisante {
            resultats.insert(
                "pl".to_string(),
                json!({"etat": "echec", "message": "mémoire insuffisante pour PL"}),
            );
            tous_ok = false;
        } else {
            match estimer(&groupes, &famille, Estimateur::Pl, fils) {
                Ok(res) => {
                    resultats.insert("pl".to_string(), json_resultat_pl(&res, &famille));
                    couplages_pl = Some(res.couplages);
                }
                Err(e) => {
                    resultats.insert(
                        "pl".to_string(),
                        json!({"etat": "echec", "message": e.to_string()}),
                    );
                    tous_ok = false;
                }
            }
        }
    }

    if estimateurs_demandes.contains(&Estimateur::Sd) {
        match estimer(&groupes, &famille, Estimateur::Sd, fils) {
            Ok(res) => {
                resultats.insert("sd".to_string(), json_resultat_sd(&res, &famille));
                couplages_sd = Some(res.couplages);
            }
            Err(e) => {
                resultats.insert(
                    "sd".to_string(),
                    json!({"etat": "echec", "message": e.to_string()}),
                );
                tous_ok = false;
            }
        }
    }

    if let (Some(pl), Some(sd)) = (&couplages_pl, &couplages_sd) {
        let valeurs = differences_appariees(pl, sd)?;
        let ecart: serde_json::Map<String, Value> = famille
            .noms()
            .iter()
            .zip(valeurs)
            .map(|(nom, v)| {
                (
                    (*nom).to_string(),
                    json!({"valeur": v.valeur, "erreur": v.erreur, "replicats": v.replicats}),
                )
            })
            .collect();
        resultats.insert("ecart_pl_sd".into(), Value::Object(ecart));
    }

    let mut doc = json!({
        "format": SCHEMA_ACTION,
        "version": crate::cli::version_texte().replace('\n', " | "),
        "ensemble": ensemble,
        "filtre": filtre,
        "pas": k_pas,
        "operateur": if k_pas >= 1 { operateur.parametres_json() } else { json!({"nom": "aucun"}) },
        "hors_famille": hors_famille,
        "l_entree": l_ref,
        "l_estimation": l_estimation,
        "famille": famille.noms(),
        "estimateurs": estimateurs_demandes.iter().map(|e| e.nom()).collect::<Vec<_>>(),
        "jackknife": {"mode": match mode_jackknife { ModeJackknife::Chaines => "chaines", ModeJackknife::Blocs(_) => "blocs" }, "groupes": groupes.len()},
        "n_chaines": chaines.len(),
        "n_configs": n_configs_total,
        "n_liens": n_liens_total,
        "chaines": info_chaines,
        "boucles": boucles_json(&famille),
        "beta_sd_un_couplage": {"valeur": beta_sd_un_valeur, "erreur": beta_sd_un_erreur,
            "residu": residu_temoin, "domaine": [0, 50],
            "note": "témoin de Wilson limité à [0,50] ; null si racine ou résidu invalide"},
        "note": "mesure de schéma : couplages d'une action tronquée par identités exactes ; aucune lecture physique",
    });

    doc.as_object_mut()
        .ok_or_else(|| Erreur::Format("objet JSON action invalide".into()))?
        .extend(resultats.clone());

    // Écriture atomique, refus d'écraser
    if chemin_sortie.exists() {
        return Err(Erreur::Config(format!(
            "{} existe déjà : refus d'écraser",
            chemin_sortie.display()
        )));
    }
    if let Some(parent) = chemin_sortie.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Erreur::io("création du dossier de sortie", e))?;
        }
    }
    crate::rg::domaine::ecrire_json(&chemin_sortie, &doc)?;

    // Ligne INFO stdout
    let mut ligne = format!(
        "INFO RG-ACTION L={} K={} op={} famille={}",
        l_ref,
        k_pas,
        if k_pas >= 1 {
            operateur.spec()
        } else {
            "aucun".to_string()
        },
        famille.noms().join(",")
    );
    for est in &estimateurs_demandes {
        if let Some(obj) = resultats.get(est.nom()) {
            if obj["etat"] == "ok" {
                ligne.push_str(&format!(" {} β=", est.nom()));
                for nom in famille.noms() {
                    let val = &obj["couplages"][nom];
                    ligne.push_str(&format!(
                        "{}: {:.5}±{:.5} ",
                        nom,
                        val["valeur"].as_f64().unwrap_or(f64::NAN),
                        val["erreur"].as_f64().unwrap_or(f64::NAN)
                    ));
                }
                let r4 = &obj["r4"]["norme"];
                ligne.push_str(&format!(
                    "N4: {:.5}±{:.5}",
                    r4["valeur"].as_f64().unwrap_or(f64::NAN),
                    r4["erreur"].as_f64().unwrap_or(f64::NAN)
                ));
            }
        }
    }
    println!("{ligne}");

    if tous_ok {
        Ok(0)
    } else {
        Err(Erreur::Validation(
            "estimateur(s) en échec (voir action.json)".into(),
        ))
    }
}

fn action_gates_cmd(m: &ArgMatches) -> Resultat<i32> {
    let ctx = Contexte {
        grandl: m.get_one::<String>("grandl").cloned(),
        seul: m
            .get_one::<String>("seul")
            .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
            .unwrap_or_default(),
        sans_mutants: m.get_flag("sans-mutants"),
        json: m.get_flag("json"),
        fils: nombre_fils(m)?,
    };
    let lignes = batterie(&ctx)?;
    let vert = tout_vert(&lignes);
    if ctx.json {
        let pass = lignes.iter().filter(|l| l.verdict == Verdict::Pass).count();
        let fail = lignes.iter().filter(|l| l.verdict == Verdict::Fail).count();
        let skip = lignes.iter().filter(|l| l.verdict == Verdict::Skip).count();
        let doc =
            json!({"tout_vert": vert, "pass": pass, "fail": fail, "skip": skip, "lignes": lignes});
        println!(
            "{}",
            serde_json::to_string(&doc).map_err(|e| Erreur::Format(e.to_string()))?
        );
    } else {
        for ligne in &lignes {
            println!("{ligne}");
        }
    }
    Ok(if vert { code::OK } else { code::ECHEC })
}

fn boucles_json(famille: &Famille) -> Value {
    let mut obj = serde_json::Map::new();
    for &ty in famille.types() {
        obj.insert(
            ty.nom().to_string(),
            json!({
                "longueur": ty.longueur(),
                "par_lien": ty.par_lien(),
                "par_site": ty.par_site(),
                "poids_r4": ty.poids_r4(),
            }),
        );
    }
    Value::Object(obj)
}

fn json_resultat_pl(res: &super::estimation::ResultatEstimateur, famille: &Famille) -> Value {
    let mut couplages = serde_json::Map::new();
    for (i, nom) in famille.noms().iter().enumerate() {
        couplages.insert(
            (*nom).to_string(),
            json!({
                "valeur": res.couplages[i].valeur,
                "erreur": res.couplages[i].erreur,
                "replicats": res.couplages[i].replicats,
            }),
        );
    }
    let details = res.details.clone();
    json!({
        "etat": "ok",
        "message": Value::Null,
        "couplages": Value::Object(couplages),
        "r4": {
            "norme": {"valeur": res.r4.valeur, "erreur": res.r4.erreur, "replicats": res.r4.replicats},
            "alpha4": res.r4.valeur / 4.0,
            "inferieur_a_4": res.r4.valeur < SEUIL_R4,
            "note": "mesure de schéma (trou de type Dobrushin), pas une preuve",
        },
        "critere_pr": {
            "valeur": {"valeur": res.pr.valeur, "erreur": res.pr.erreur, "replicats": res.pr.replicats},
            "seuil": SEUIL_PR,
            "inferieur_au_seuil": res.pr.valeur < SEUIL_PR,
            "note": "ignore C et T",
        },
        "correlations": res.correlations.chunks(famille.n()).collect::<Vec<_>>(),
        "iterations": details["iterations"],
        "norme_gradient": details["norme_gradient"],
        "valeurs_propres_information": details["valeurs_propres_information"],
        "conditionnement": details["conditionnement"],
    })
}

fn json_resultat_sd(res: &super::estimation::ResultatEstimateur, famille: &Famille) -> Value {
    let mut couplages = serde_json::Map::new();
    for (i, nom) in famille.noms().iter().enumerate() {
        couplages.insert(
            (*nom).to_string(),
            json!({
                "valeur": res.couplages[i].valeur,
                "erreur": res.couplages[i].erreur,
                "replicats": res.couplages[i].replicats,
            }),
        );
    }
    let details = res.details.clone();
    json!({
        "etat": "ok",
        "message": Value::Null,
        "couplages": Value::Object(couplages),
        "r4": {
            "norme": {"valeur": res.r4.valeur, "erreur": res.r4.erreur, "replicats": res.r4.replicats},
            "alpha4": res.r4.valeur / 4.0,
            "inferieur_a_4": res.r4.valeur < SEUIL_R4,
            "note": "mesure de schéma (trou de type Dobrushin), pas une preuve",
        },
        "critere_pr": {
            "valeur": {"valeur": res.pr.valeur, "erreur": res.pr.erreur, "replicats": res.pr.replicats},
            "seuil": SEUIL_PR,
            "inferieur_au_seuil": res.pr.valeur < SEUIL_PR,
            "note": "ignore C et T",
        },
        "correlations": res.correlations.chunks(famille.n()).collect::<Vec<_>>(),
        "valeurs_propres_m": details["valeurs_propres_m"],
        "conditionnement": details["conditionnement"],
    })
}

pub fn executer(nom: &str, m: &ArgMatches) -> Resultat<i32> {
    match nom {
        "action" => action_cmd(m),
        "action-gates" => action_gates_cmd(m),
        _ => Err(Erreur::Config(format!(
            "sous-commande « {nom} » inconnue du module action"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temoin_wilson_verifie_le_domaine_et_le_residu() {
        let r = crate::rg::flot::bessel::i2_sur_i1(1.0);
        let mut stats = StatsSd {
            somme_w: r,
            ks: vec![1.0],
            n_configs: 1,
        };
        assert!((beta_sd_verifie(&stats).unwrap() - 1.0).abs() < 1e-12);
        stats.somme_w = -r;
        assert!(beta_sd_verifie(&stats).is_none());
        stats.somme_w = 1.0;
        assert!(beta_sd_verifie(&stats).is_none());
    }

    #[test]
    fn operateur_valide_avant_decouverte() {
        let commande = sous_commandes().remove(0);
        let m = commande
            .try_get_matches_from([
                "action",
                "--ensemble",
                "/entree-inexistante-m2",
                "--operateur",
                "inconnu",
            ])
            .unwrap();
        let err = action_cmd(&m).unwrap_err().to_string();
        assert!(err.contains("opérateur") && err.contains("inconnu"));
    }
}
