//! Sous-commandes `rg flux` et `rg flux-gates`.
//!
//! `flux` produit les statistiques d'une configuration ou d'une liste JSONL ;
//! `flux-gates` exécute les gates de validation (voir `super::gates`).
//! Les sorties JSON ne contiennent ni durée ni nombre de fils : elles sont
//! identiques à l'octet quel que soit `--fils`.

use super::faces::{
    grossier, stats_coherent, stats_faces, Pavage, StatsCoherent, StatsFaces, VarianteFlux,
};
use super::{SCHEMA_FLUX, SEUILS_COH, SEUILS_X};
use crate::commun::{code, Erreur, Resultat};
use crate::heritage::manifest::sha256_fichier;
use crate::rg::operateur::Operateur;
use crate::stockage::npy::lire_champ;
use clap::{parser::ValueSource, Arg, ArgAction, ArgMatches, Command};
use serde_json::{json, Map, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Construit les définitions clap des sous-commandes `flux` et `flux-gates`.
pub fn sous_commandes() -> Vec<Command> {
    let fils = || {
        Arg::new("fils").long("fils").num_args(1).help(
            "nombre de fils (défaut : RAYON_NUM_THREADS ou 4) ; les sorties n'en dépendent pas",
        )
    };
    vec![
        Command::new("flux")
            .about("Statistiques des boucles n×n et flux cohérent des faces de bloc")
            .arg(
                Arg::new("entree")
                    .long("entree")
                    .num_args(1)
                    .help("configuration fine (.npy ou .npy.zst, c16 ou q64)"),
            )
            .arg(
                Arg::new("n")
                    .long("n")
                    .num_args(1)
                    .default_value("3")
                    .help("tailles des faces (liste à virgules, ex. 1,2,3)"),
            )
            .arg(
                Arg::new("coherent")
                    .long("coherent")
                    .num_args(1)
                    .help("tailles pour le flux cohérent (liste à virgules, ex. 3)"),
            )
            .arg(
                Arg::new("pavage")
                    .long("pavage")
                    .num_args(1)
                    .default_value("tous")
                    .help("tous | 0 | 1 | 2 : coins des blocs 3^4 (L multiple de 3)"),
            )
            .arg(
                Arg::new("grossier")
                    .long("grossier")
                    .action(ArgAction::SetTrue)
                    .help("calcule la plaquette du réseau grossier via l'opérateur (défaut 3centre, 1 pas)"),
            )
            .arg(
                Arg::new("grossier-operateur")
                    .long("grossier-operateur")
                    .num_args(1)
                    .default_value("3centre")
                    .help("opérateur de blocage pour --grossier : 3centre | dec | ape:<α>:<n> | flot:<t>[:<ε>] | agr:<c>"),
            )
            .arg(
                Arg::new("grossier-pas")
                    .long("grossier-pas")
                    .num_args(1)
                    .default_value("1")
                    .help("nombre de pas de blocage (défaut 1)"),
            )
            .arg(
                Arg::new("hors-famille")
                    .long("hors-famille")
                    .action(ArgAction::SetTrue)
                    .help("accepte un opérateur dont le rayon de lissage dépasse le bloc (mutant)"),
            )
            .arg(
                Arg::new("grossier-fichier")
                    .long("grossier-fichier")
                    .num_args(1)
                    .conflicts_with("grossier")
                    .help("fichier de configuration grossière (.npy) pour la plaquette"),
            )
            .arg(
                Arg::new("histogramme")
                    .long("histogramme")
                    .action(ArgAction::SetTrue)
                    .help("ajoute l'histogramme de alpha aux statistiques brutes"),
            )
            .arg(
                Arg::new("sortie")
                    .long("sortie")
                    .num_args(1)
                    .help("fichier JSON de sortie (refus d'écraser)"),
            )
            .arg(
                Arg::new("liste")
                    .long("liste")
                    .num_args(1)
                    .help("fichier JSONL de travaux (mode liste, nécessite --sortie)"),
            )
            .arg(fils()),
        Command::new("flux-gates")
            .about("Gates RG-FLUX-* : validation du module flux")
            .arg(
                Arg::new("desert")
                    .long("desert")
                    .num_args(1)
                    .help("dossier de desert_debut_2026-11 pour G-REPRO (absent ⇒ SKIP)"),
            )
            .arg(
                Arg::new("max-configs")
                    .long("max-configs")
                    .num_args(1)
                    .help("nombre maximal de configurations à comparer dans G-REPRO"),
            )
            .arg(
                Arg::new("seul")
                    .long("seul")
                    .num_args(1)
                    .help("préfixes d'ID à exécuter (liste à virgules)"),
            )
            .arg(
                Arg::new("sans-mutants")
                    .long("sans-mutants")
                    .action(ArgAction::SetTrue)
                    .help("n'exécute pas les gates de mutants (RG-FLUX-MUT)"),
            )
            .arg(Arg::new("json").long("json").action(ArgAction::SetTrue))
            .arg(fils()),
    ]
}

/// Point d'entrée des sous-commandes `flux` et `flux-gates`.
pub fn executer(nom: &str, m: &ArgMatches) -> Resultat<i32> {
    match nom {
        "flux" => flux_cmd(m),
        "flux-gates" => super::gates::executer_gates(m),
        _ => Err(Erreur::Config(format!(
            "sous-commande « {nom} » inconnue du module rg flux"
        ))),
    }
}

pub(crate) fn nombre_fils(m: &ArgMatches) -> Resultat<usize> {
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

/// Parse une liste d'entiers séparés par des virgules, sans doublons et non vide.
fn parser_liste_entiers(s: &str) -> Resultat<Vec<usize>> {
    let mut v = Vec::new();
    for t in s.split(',') {
        let t = t.trim();
        if t.is_empty() {
            return Err(Erreur::Config("liste d'entiers vide".into()));
        }
        let n: usize = t
            .parse()
            .map_err(|_| Erreur::Config(format!("« {t} » n'est pas un entier")))?;
        if n == 0 {
            return Err(Erreur::Config("les entiers doivent être ≥ 1".into()));
        }
        if v.contains(&n) {
            return Err(Erreur::Config(format!("doublon {n} dans la liste")));
        }
        v.push(n);
    }
    Ok(v)
}

/// Parse `--pavage` : "tous" ou un entier 0..=2.
fn parser_pavage(s: &str) -> Resultat<Pavage> {
    match s.trim() {
        "tous" => Ok(Pavage::Tous),
        autre => {
            let o: usize = autre
                .parse()
                .map_err(|_| Erreur::Config(format!("pavage « {autre} » invalide (tous|0|1|2)")))?;
            if o > 2 {
                return Err(Erreur::Config(format!("pavage o = {o} hors de {{0,1,2}}")));
            }
            Ok(Pavage::Blocs(o))
        }
    }
}

/// Refuse les options du mode configuration dans le mode liste (`--liste`).
fn verifier_absentes_liste(m: &ArgMatches) -> Resultat<()> {
    for nom in [
        "n",
        "pavage",
        "grossier-operateur",
        "grossier-pas",
        "grossier-fichier",
    ] {
        if m.value_source(nom)
            .map(|s| s != ValueSource::DefaultValue)
            .unwrap_or(false)
        {
            return Err(Erreur::Config(format!(
                "--{nom} n'est pas autorisé avec --liste"
            )));
        }
    }
    if m.get_flag("grossier") || m.get_flag("hors-famille") || m.get_flag("histogramme") {
        return Err(Erreur::Config(
            "--grossier, --hors-famille et --histogramme ne sont pas autorisés avec --liste".into(),
        ));
    }
    Ok(())
}

/// Écrit un contenu de façon atomique SANS écraser une destination existante.
///
/// Garanties :
/// - le temporaire `<sortie>.<pid>.part` est créé avec `create_new` (échec si
///   un fichier du même nom existe déjà) ;
/// - l'écriture est suivie de `sync_all` ;
/// - la publication se fait par `hard_link` (échec `AlreadyExists` si la
///   destination apparaît entre-temps) ;
/// - seul le temporaire créé par cet appel est ensuite supprimé.
fn ecrire_atomique_sans_ecraser(chemin: &Path, contenu: &[u8]) -> Resultat<()> {
    let pid = std::process::id();
    let mut part = chemin.as_os_str().to_os_string();
    part.push(format!(".{pid}.part"));
    let part_path = PathBuf::from(part);

    // Sortir avant le nettoyage si create_new échoue : le fichier préexistant
    // appartient à un autre appel et ne doit jamais être supprimé ici.
    let mut fichier = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&part_path)
        .map_err(|e| Erreur::io(format!("création {}", part_path.display()), e))?;
    let resultat = (|| -> Resultat<()> {
        fichier
            .write_all(contenu)
            .map_err(|e| Erreur::io(format!("écriture {}", part_path.display()), e))?;
        fichier
            .sync_all()
            .map_err(|e| Erreur::io(format!("synchronisation {}", part_path.display()), e))?;
        drop(fichier);
        match std::fs::hard_link(&part_path, chemin) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(Erreur::Config(format!(
                    "{} existe déjà : refus d'écraser (supprimer à la main si voulu)",
                    chemin.display()
                )))
            }
            Err(e) => Err(Erreur::io(format!("publication {}", chemin.display()), e)),
        }
    })();

    let _ = std::fs::remove_file(&part_path);
    resultat
}

/// Écrit un document JSON (pretty) atomiquement, refuse d'écraser.
fn ecrire_json_neuf(chemin: &Path, doc: &Value) -> Resultat<()> {
    let texte =
        serde_json::to_string_pretty(doc).map_err(|e| Erreur::Format(e.to_string()))? + "\n";
    ecrire_atomique_sans_ecraser(chemin, texte.as_bytes())
}

/// Écrit une liste de lignes JSON (JSONL) atomiquement, refuse d'écraser.
fn ecrire_jsonl_atomique(chemin: &Path, lignes: &[Value]) -> Resultat<()> {
    let mut tampon = String::new();
    for ligne in lignes {
        let texte = serde_json::to_string(ligne).map_err(|e| Erreur::Format(e.to_string()))?;
        tampon.push_str(&texte);
        tampon.push('\n');
    }
    ecrire_atomique_sans_ecraser(chemin, tampon.as_bytes())
}

/// Convertit `StatsFaces` en objet JSON du schéma (noms identiques à `boucles.stats_x`).
pub(crate) fn stats_faces_json(s: &StatsFaces) -> Value {
    let mut comptes = Map::new();
    let mut fractions = Map::new();
    for (i, (nom, _, _)) in SEUILS_X.iter().enumerate() {
        comptes.insert((*nom).to_string(), json!(s.comptes[i]));
        fractions.insert((*nom).to_string(), json!(s.comptes[i] as f64 / s.n as f64));
    }
    let mut obj = Map::new();
    obj.insert("N".into(), json!(s.n));
    obj.insert("comptes".into(), Value::Object(comptes));
    obj.insert("fractions".into(), Value::Object(fractions));
    obj.insert("sx".into(), json!(s.sx));
    obj.insert("sx2".into(), json!(s.sx2));
    obj.insert("sTheta2".into(), json!(s.s_theta2));
    obj.insert("sTheta4".into(), json!(s.s_theta4));
    obj.insert("xmin".into(), json!(s.xmin));
    if let Some(hist) = &s.hist_alpha {
        let mut nonvides = Map::new();
        for (i, &count) in hist.iter().enumerate() {
            if count > 0 {
                nonvides.insert(i.to_string(), json!(count));
            }
        }
        obj.insert("hist_alpha_nonvides".into(), Value::Object(nonvides));
    }
    Value::Object(obj)
}

/// Convertit `StatsCoherent` en objet JSON du schéma.
fn stats_coherent_json(s: &StatsCoherent) -> Value {
    let mut comptes = Map::new();
    let mut fractions = Map::new();
    let mut comptes_coherence = Map::new();
    for (j, (nom, _, _)) in SEUILS_COH.iter().enumerate() {
        comptes.insert((*nom).to_string(), json!(s.comptes[j]));
        fractions.insert((*nom).to_string(), json!(s.comptes[j] as f64 / s.n as f64));
        let mut par_cmin = Map::new();
        for (k, (cmin, _)) in super::COH_MIN.iter().enumerate() {
            par_cmin.insert((*cmin).to_string(), json!(s.comptes_coherence[j][k]));
        }
        comptes_coherence.insert((*nom).to_string(), Value::Object(par_cmin));
    }
    let mut obj = Map::new();
    obj.insert("N".into(), json!(s.n));
    obj.insert("comptes".into(), Value::Object(comptes));
    obj.insert("comptes_coherence".into(), Value::Object(comptes_coherence));
    obj.insert("fractions".into(), Value::Object(fractions));
    obj.insert("s_phi".into(), json!(s.s_phi));
    obj.insert("s_phi2".into(), json!(s.s_phi2));
    obj.insert("s_coh".into(), json!(s.s_coh));
    obj.insert("phi_max".into(), json!(s.phi_max));
    obj.insert("n_ambigues".into(), json!(s.n_ambigues));
    Value::Object(obj)
}

/// Mode configuration unique.
fn flux_cmd(m: &ArgMatches) -> Resultat<i32> {
    if m.get_one::<String>("liste").is_some() {
        verifier_absentes_liste(m)?;
        if m.get_one::<String>("entree").is_some() {
            return Err(Erreur::Config("--liste et --entree sont exclusifs".into()));
        }
        return traiter_liste(m);
    }

    let entree = m
        .get_one::<String>("entree")
        .ok_or_else(|| Erreur::Config("--entree requis (sauf mode --liste)".into()))?;
    let fils = nombre_fils(m)?;
    let champ = lire_champ(Path::new(entree))?;
    let l = champ.l;

    let liste_n =
        parser_liste_entiers(m.get_one::<String>("n").map(String::as_str).unwrap_or("3"))?;
    let liste_coh = match m.get_one::<String>("coherent") {
        Some(s) => parser_liste_entiers(s)?,
        None => Vec::new(),
    };
    let pavage = parser_pavage(
        m.get_one::<String>("pavage")
            .map(String::as_str)
            .unwrap_or("tous"),
    )?;
    if let Pavage::Blocs(o) = pavage {
        if o > 2 {
            return Err(Erreur::Config(format!("pavage o = {o} hors de {{0,1,2}}")));
        }
        if l % 3 != 0 {
            return Err(Erreur::Config(
                "L multiple de 3 requis pour un pavage de blocs".into(),
            ));
        }
    }

    let histogramme = m.get_flag("histogramme");

    // Validations AVANT tout calcul (opérateur, famille, divisibilité).
    // Intégrateur (M3 r2) : condition englobante redondante retirée (clippy collapsible_if) ; même comportement.
    {
        if m.get_flag("grossier") {
            let spec_op = m
                .get_one::<String>("grossier-operateur")
                .map(String::as_str)
                .unwrap_or("3centre");
            let op = Operateur::analyser(spec_op)?;
            op.verifier_famille(m.get_flag("hors-famille"))?;
            let pas = match m.get_one::<String>("grossier-pas") {
                Some(t) => t
                    .parse::<usize>()
                    .map_err(|_| Erreur::Config(format!("--grossier-pas « {t} » invalide")))?,
                None => 1,
            };
            if pas == 0 {
                return Err(Erreur::Config("--grossier-pas doit être ≥ 1".into()));
            }
            let mut tmp = l;
            for _ in 0..pas {
                if tmp % 3 != 0 {
                    return Err(Erreur::Config(format!(
                        "L = {l} non divisible par 3^{pas} pour le grossier"
                    )));
                }
                tmp /= 3;
            }
        }
    }

    // Statistiques brutes
    let mut brut = Map::new();
    for n in &liste_n {
        let stats = stats_faces(&champ, *n, pavage, histogramme, fils)?;
        brut.insert(format!("n{n}"), stats_faces_json(&stats));
    }
    let brut = Value::Object(brut);

    // Statistiques cohérentes
    let mut coherent = Map::new();
    for n in &liste_coh {
        let stats = stats_coherent(&champ, *n, pavage, VarianteFlux::Normale, fils)?;
        coherent.insert(format!("n{n}"), stats_coherent_json(&stats));
    }
    let coherent = Value::Object(coherent);

    // Partie grossier
    let mut source_grossier: Option<String> = None;
    let mut operateur_json: Option<Value> = None;
    let mut pas_grossier: Option<usize> = None;
    let mut fichier_grossier: Option<String> = None;
    let mut sha256_grossier: Option<String> = None;
    let mut l_grossier: Option<usize> = None;
    let mut grossier_stats: Option<Value> = None;

    if m.get_flag("grossier") {
        let spec_op = m
            .get_one::<String>("grossier-operateur")
            .map(String::as_str)
            .unwrap_or("3centre");
        let op = Operateur::analyser(spec_op)?; // déjà validé, mais on ré-analyse proprement
        op.verifier_famille(m.get_flag("hors-famille"))?;
        let pas = match m.get_one::<String>("grossier-pas") {
            Some(t) => t
                .parse::<usize>()
                .map_err(|_| Erreur::Config(format!("--grossier-pas « {t} » invalide")))?,
            None => 1,
        };
        let champ_grossier = grossier(&champ, &op, pas, fils)?;
        let stats = stats_faces(&champ_grossier, 1, Pavage::Tous, histogramme, fils)?;
        grossier_stats = Some(stats_faces_json(&stats));
        source_grossier = Some("calcul".into());
        operateur_json = Some(op.parametres_json());
        pas_grossier = Some(pas);
        l_grossier = Some(champ_grossier.l);
    }

    if let Some(fichier) = m.get_one::<String>("grossier-fichier") {
        let champ_gr = lire_champ(Path::new(fichier))?;
        let stats = stats_faces(&champ_gr, 1, Pavage::Tous, histogramme, fils)?;
        grossier_stats = Some(stats_faces_json(&stats));
        source_grossier = Some("fichier".into());
        fichier_grossier = Some(fichier.clone());
        sha256_grossier = Some(
            sha256_fichier(Path::new(fichier))
                .map_err(|e| Erreur::io(format!("sha256 de {fichier}"), e))?,
        );
        l_grossier = Some(champ_gr.l);
        operateur_json = None;
        pas_grossier = None;
    }

    let mut doc = Map::new();
    doc.insert("format".into(), json!(SCHEMA_FLUX));
    doc.insert(
        "version".into(),
        json!(crate::cli::version_texte().replace('\n', " | ")),
    );
    doc.insert("entree".into(), json!(entree));
    doc.insert(
        "sha256_entree".into(),
        json!(sha256_fichier(Path::new(entree))
            .map_err(|e| Erreur::io(format!("sha256 de {entree}"), e))?),
    );
    doc.insert("L".into(), json!(l));
    doc.insert(
        "pavage".into(),
        json!(match pavage {
            Pavage::Tous => "tous".to_string(),
            Pavage::Blocs(o) => o.to_string(),
        }),
    );
    doc.insert("brut".into(), brut);
    doc.insert("coherent".into(), coherent);

    if let Some(src) = source_grossier {
        let mut gr = Map::new();
        gr.insert("source".into(), json!(src));
        gr.insert("operateur".into(), operateur_json.unwrap_or(Value::Null));
        gr.insert("pas".into(), json!(pas_grossier));
        gr.insert("fichier".into(), json!(fichier_grossier));
        gr.insert("sha256".into(), json!(sha256_grossier));
        gr.insert("L".into(), json!(l_grossier));
        gr.insert("n1".into(), grossier_stats.unwrap_or(Value::Null));
        doc.insert("grossier".into(), Value::Object(gr));
    }

    doc.insert(
        "seuils".into(),
        json!({
            "x": SEUILS_X.iter().map(|(n, s, strict)| json!({"nom": n, "seuil": s, "strict": strict})).collect::<Vec<_>>(),
            "coherent": SEUILS_COH.iter().map(|(n, s, strict)| json!({"nom": n, "seuil": s, "strict": strict})).collect::<Vec<_>>(),
        }),
    );
    doc.insert(
        "note".into(),
        json!("Angle propre Θ = 2 arccos(½ Tr W) replié ; flux cohérent Φ_coh non replié, dépendant des chemins de transport (chaque choix fixé est une observable invariante de jauge)."),
    );

    let doc = Value::Object(doc);

    // Écriture ou affichage
    if let Some(sortie) = m.get_one::<String>("sortie") {
        ecrire_json_neuf(Path::new(sortie), &doc)?;
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(&doc).map_err(|e| Erreur::Format(e.to_string()))?
        );
    }

    // Ligne INFO sur stderr
    let mut info = format!("INFO RG-FLUX L={l}");
    if let Value::Object(brut_obj) = &doc["brut"] {
        for n in &liste_n {
            let key = format!("n{n}");
            if let Some(stat) = brut_obj.get(&key) {
                let f_t1 = stat["fractions"]["T1_pi"].as_f64().unwrap_or(0.0);
                info.push_str(&format!(" brut_n{n}_f_T1={f_t1:.6}"));
            }
        }
    }
    if let Value::Object(coh_obj) = &doc["coherent"] {
        for n in &liste_coh {
            let key = format!("n{n}");
            if let Some(stat) = coh_obj.get(&key) {
                let f_q1 = stat["fractions"]["Q1"].as_f64().unwrap_or(0.0);
                info.push_str(&format!(" coh_n{n}_f_Q1={f_q1:.6}"));
            }
        }
    }
    if let Some(gr) = doc.get("grossier") {
        let f_t1 = gr["n1"]["fractions"]["T1_pi"].as_f64().unwrap_or(0.0);
        info.push_str(&format!(" grossier_n1_f_T1={f_t1:.6}"));
    }
    eprintln!("{info}");

    Ok(code::OK)
}

/// Validation commune du L optionnel des travaux du mode liste et de G-REPRO.
pub(crate) fn verifier_l_travail(job: &Value, l_fichier: usize, contexte: &str) -> Resultat<()> {
    if let Some(valeur) = job.get("L") {
        let l_attendu = valeur
            .as_u64()
            .and_then(|l| usize::try_from(l).ok())
            .ok_or_else(|| {
                Erreur::Config(format!(
                "{contexte} : L présent mais non entier valide : calcul={l_fichier}, référence={valeur}"
                ))
            })?;
        if l_attendu != l_fichier {
            return Err(Erreur::Config(format!(
                "{contexte} : L calculé={l_fichier}, référence={l_attendu}"
            )));
        }
    }
    Ok(())
}

/// Mode liste (portage de `traiter.py`).
fn traiter_liste(m: &ArgMatches) -> Resultat<i32> {
    let liste_path = m
        .get_one::<String>("liste")
        .ok_or_else(|| Erreur::Config("--liste requis en mode liste".into()))?;
    let sortie = m
        .get_one::<String>("sortie")
        .ok_or_else(|| Erreur::Config("--sortie requis avec --liste".into()))?;
    let fils = nombre_fils(m)?;
    let liste_coh = match m.get_one::<String>("coherent") {
        Some(s) => parser_liste_entiers(s)?,
        None => Vec::new(),
    };

    let fichier_liste = std::fs::File::open(liste_path)
        .map_err(|e| Erreur::io(format!("ouverture de {liste_path}"), e))?;
    let lecteur = BufReader::new(fichier_liste);

    let mut lignes_sortie = Vec::new();
    for (num_ligne, ligne) in lecteur.lines().enumerate() {
        let ligne = ligne.map_err(|e| Erreur::io("lecture de la liste", e))?;
        if ligne.trim().is_empty() {
            continue;
        }
        let job: Value = serde_json::from_str(&ligne).map_err(|e| {
            Erreur::Format(format!("ligne {} : JSON invalide : {e}", num_ligne + 1))
        })?;
        let kind = job
            .get("kind")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Erreur::Format(format!("ligne {} : clé kind absente", num_ligne + 1)))?
            .to_string();
        let path = job
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Erreur::Format(format!("ligne {} : clé path absente", num_ligne + 1)))?
            .to_string();

        let champ = lire_champ(Path::new(&path))?;
        verifier_l_travail(&job, champ.l, &format!("ligne {} ({path})", num_ligne + 1))?;

        let mut stats_obj = Map::new();
        match kind.as_str() {
            "fin" => {
                for n in [1, 2, 3] {
                    let stats = stats_faces(&champ, n, Pavage::Tous, false, fils)?;
                    stats_obj.insert(format!("n{n}"), stats_faces_json(&stats));
                }
                for n in &liste_coh {
                    let stats =
                        stats_coherent(&champ, *n, Pavage::Tous, VarianteFlux::Normale, fils)?;
                    stats_obj.insert(format!("coherent_n{n}"), stats_coherent_json(&stats));
                }
            }
            "gros1" | "gros2" => {
                let stats = stats_faces(&champ, 1, Pavage::Tous, false, fils)?;
                stats_obj.insert("n1".to_string(), stats_faces_json(&stats));
            }
            autre => {
                return Err(Erreur::Config(format!(
                    "ligne {} : kind « {autre} » inconnu (fin|gros1|gros2)",
                    num_ligne + 1
                )));
            }
        }

        // Recopier les clés optionnelles
        let mut ligne_sortie = Map::new();
        for cle in ["kind", "cell", "chain", "idx", "L", "beta", "src"] {
            if let Some(v) = job.get(cle) {
                ligne_sortie.insert(cle.to_string(), v.clone());
            }
        }
        ligne_sortie.insert("stats".to_string(), Value::Object(stats_obj));
        lignes_sortie.push(Value::Object(ligne_sortie));
    }

    ecrire_jsonl_atomique(Path::new(sortie), &lignes_sortie)?;
    eprintln!(
        "INFO RG-FLUX liste : {} configurations traitées",
        lignes_sortie.len()
    );
    Ok(code::OK)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecriture_atomique_collisions_preservent_les_fichiers() {
        let dossier = std::env::temp_dir().join(format!("rg_flux_atomique_{}", std::process::id()));
        // create_dir, sans nettoyage préalable : ne toucher qu'à nos fichiers.
        std::fs::create_dir(&dossier).unwrap();
        let sortie = dossier.join("sortie.json");
        let part = dossier.join(format!("sortie.json.{}.part", std::process::id()));
        std::fs::write(&part, b"SENTINELLE TEMPORAIRE").unwrap();
        assert!(ecrire_atomique_sans_ecraser(&sortie, b"NOUVEAU").is_err());
        assert_eq!(std::fs::read(&part).unwrap(), b"SENTINELLE TEMPORAIRE");
        assert!(!sortie.exists());
        std::fs::remove_file(&part).unwrap();

        // Appel direct : la collision a bien lieu à la publication hard_link,
        // après création et écriture du temporaire (aucun test exists en amont).
        std::fs::write(&sortie, b"SENTINELLE SORTIE").unwrap();
        let erreur = ecrire_atomique_sans_ecraser(&sortie, b"NOUVEAU").unwrap_err();
        assert!(matches!(&erreur, Erreur::Config(_)));
        assert_eq!(erreur.code_sortie(), 2);
        assert_eq!(std::fs::read(&sortie).unwrap(), b"SENTINELLE SORTIE");
        assert!(!part.exists());
        std::fs::remove_dir_all(dossier).unwrap();
    }
}
