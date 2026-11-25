//! `kvu su2 spectro run` : génère une chaîne SU(2) (bain thermique + OR, CPU F64, `su2::tube`, sans torsion) sur une boîte
//! rectangulaire et écrit, par configuration mesurée, les corrélateurs de torelon lissés (voir le module parent).
//!
//! Sorties du dossier `--sortie` : `corr.csv` (une ligne par cfg et par impulsion ; colonnes `c<a><b>_<τ>`, a ≤ b indices de
//! niveau), `phi.csv` (⟨φ_s⟩_t), `plaq.csv` (plaquette moyenne, Polyakov brut dans la direction d'enroulement), `manifeste.json`,
//! `etat.bin` (point de reprise : liens et numéro de balayage ; `--reprendre` continue à l'identique car les tirages Philox
//! dépendent seulement de (lien, balayage, tirage)).

use super::{mesurer, paires, Axes, Mutant, ParamMesure};
use super::neutre;
use crate::commun::{Erreur, Resultat};
use crate::su2::torsion::Torsion;
use crate::su2::tube::reseau::{pile_aniso, ChaineTube, GeoA, Pile};
use clap::{Arg, ArgMatches, Command};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

fn io(ctx: &str, p: &Path, e: std::io::Error) -> Erreur {
    Erreur::Io { contexte: format!("{ctx} {}", p.display()), source: e }
}

fn arg(m: &ArgMatches, n: &str) -> String {
    m.get_one::<String>(n).cloned().unwrap_or_default()
}

fn num<T: std::str::FromStr>(m: &ArgMatches, n: &str) -> Resultat<T> {
    arg(m, n).parse::<T>().map_err(|_| Erreur::Config(format!("--{n} : « {} » invalide", arg(m, n))))
}

/// Paramètres de blocage des options `--blocage`, `--ape-bloc`, `--lissage-bloc`, `--p-bloc`.
/// `--xi0` : None (absent) ou Some(ξ₀ > 0 fini).
fn param_xi0(m: &ArgMatches) -> Resultat<Option<f64>> {
    match m.get_one::<String>("xi0") {
        None => Ok(None),
        Some(t) => {
            let x: f64 = t.trim().parse().map_err(|_| Erreur::Config(format!("--xi0 : « {t} » illisible")))?;
            if !(x.is_finite() && x > 0.0) {
                return Err(Erreur::Config("--xi0 : réel fini > 0 attendu".into()));
            }
            Ok(Some(x))
        }
    }
}

fn param_bloc(m: &ArgMatches) -> Resultat<super::bloc::ParamBloc> {
    let t = arg(m, "blocage");
    let niveaux: Vec<usize> = if t.trim().is_empty() || t.trim() == "aucun" { Vec::new() } else { liste(&t, "--blocage")? };
    Ok(super::bloc::ParamBloc { niveaux, ape_bloc: num(m, "ape-bloc")?, lissage_bloc: num(m, "lissage-bloc")?, p_bloc: num(m, "p-bloc")? })
}

/// Liste `niveaux_ape` du manifeste : les itérations APE (entiers) puis, si le blocage est actif, les étiquettes « B<b>[s<n>] ».
fn niveaux_manifeste(niveaux: &[usize], pb: &super::bloc::ParamBloc) -> serde_json::Value {
    let mut v: Vec<serde_json::Value> = niveaux.iter().map(|&n| serde_json::json!(n)).collect();
    for &b in &pb.niveaux {
        v.push(serde_json::json!(pb.etiquette(b)));
    }
    serde_json::Value::Array(v)
}

/// Champ `blocage` du manifeste (ajouté seulement si le blocage est actif : manifeste d'origine inchangé sinon).
fn ajouter_bloc_manifeste(man: &mut serde_json::Value, pb: &super::bloc::ParamBloc) {
    if pb.actif() {
        man["blocage"] = serde_json::json!({
            "niveaux": pb.niveaux, "ape_bloc": pb.ape_bloc, "lissage_bloc": pb.lissage_bloc, "p_bloc": pb.p_bloc,
            "longueurs_superlien": pb.niveaux.iter().map(|&b| 1usize << b).collect::<Vec<_>>(),
            "definition": "su2::spectro::bloc (LTW04 eqn_oldblock, SU(2), V/|V|) ; niveaux mesurés après les niveaux APE, op = i_niveau * n_formes + i_forme sur la liste concaténée",
        });
    }
}

pub fn commande() -> Command {
    let a = |n: &'static str, h: &'static str| Arg::new(n).long(n).num_args(1).help(h);
    Command::new("spectro")
        .about("[EXPÉRIMENTAL] Spectroscopie de torelons sur boîte rectangulaire (CPU F64) : run")
        .subcommand_required(true)
        .subcommand(
            Command::new("neutre")
                .about("chaîne + opérateurs scalaires neutres (boucles spatiales lissées, impulsion nulle) : corrélateurs par configuration")
                .arg(a("cotes", "l0,l1,l2,l3").required(true))
                .arg(a("beta", "β").required(true))
                .arg(a("graine", "graine Philox (u64)").default_value("1"))
                .arg(a("therm", "balayages de thermalisation").default_value("500"))
                .arg(a("nmeas", "nombre de configurations mesurées").default_value("100"))
                .arg(a("skip", "balayages entre deux mesures").default_value("5"))
                .arg(a("nor", "OR par balayage").default_value("4"))
                .arg(a("temps", "axe du temps (0..3)").default_value("3"))
                .arg(a("niveaux", "itérations APE croissantes").default_value("2,4,8"))
                .arg(a("alpha", "paramètre APE").default_value("0.5"))
                .arg(a("formes", "formes de boucles (P11,R12,S22,CH1,CH2)").default_value("P11,R12,S22,CH1,CH2"))
                .arg(a("tau-max", "τ maximal stocké").default_value("6"))
                .arg(a("sortie", "dossier de sortie").required(true))
                .arg(Arg::new("reprendre").long("reprendre").num_args(0).help("reprend depuis etat.bin"))
                .arg(a("xi0", "réseau anisotrope : ξ₀ > 0 (plaquettes temporelles pondérées ξ₀, spatiales 1/ξ₀) ; défaut absent = isotrope, sorties identiques au bit ; écrit en plus plaq_ss_st.csv (cfg,plaq_ss,plaq_st)"))
                .arg(Arg::new("impulsion").long("impulsion").num_args(0).help("écrit aussi neutre_p.bin : corrélateurs à impulsion 2π/ℓ (moyenne des 3 axes)"))
                .arg(a("impulsions", "crans d'impulsion mesurés, séparés par des virgules (1 ≤ k < ℓ/2) : un fichier neutre_p<k>.bin par cran ; équivalent de --impulsion : --impulsions 1"))
                .arg(a("impulsions-vecteurs", "impulsions HORS AXE : vecteurs entiers k1:k2:k3 (unités 2π/ℓ, axes spatiaux 1,2,3) séparés par des virgules, ex. 2:2:1,3:0:0 ; pour chacun, corrélateur moyenné sur TOUTE l'orbite du groupe cubique (permutations et signes) dans neutre_o<k1>_<k2>_<k3>.bin (représentant = |k| décroissant) ; chaque |k_d| < ℓ/2, côtés spatiaux égaux ; éteint par défaut, sorties inchangées au bit"))
                .arg(Arg::new("mutant").long("mutant").num_args(1).hide(true)),
        )
        .subcommand(
            Command::new("gpu")
                .about("chaînes + opérateurs scalaires neutres SUR CARTE (génération et mesure GPU) : un dossier r<i> par réplique, même format que `neutre`")
                .arg(a("cotes", "l0,l1,l2,l3").required(true))
                .arg(a("beta", "β").required(true))
                .arg(a("graine", "graine de la réplique 0 (la réplique i prend graine + i)").default_value("1"))
                .arg(a("repliques", "nombre de répliques (chaînes indépendantes) mesurées ensemble").default_value("8"))
                .arg(a("therm", "balayages de thermalisation").default_value("500"))
                .arg(a("nmeas", "nombre de configurations mesurées PAR réplique").default_value("100"))
                .arg(a("skip", "balayages entre deux mesures").default_value("5"))
                .arg(a("nor", "OR par balayage").default_value("4"))
                .arg(a("temps", "axe du temps (0..3)").default_value("3"))
                .arg(a("niveaux", "itérations APE croissantes").default_value("2,4,8"))
                .arg(a("alpha", "paramètre APE").default_value("0.5"))
                .arg(a("formes", "formes (P11,R12,S22,CH1,CH2 ; riches : R13,LL,CH3,CH4,CH5,CH6)").default_value("P11,R12,S22,CH1,CH2"))
                .arg(Arg::new("base-riche").long("base-riche").num_args(0).help("ajoute les formes riches R13,LL,CH3,CH4,CH5,CH6 à --formes"))
                .arg(a("tau-max", "τ maximal stocké").default_value("6"))
                .arg(a("sortie", "dossier de sortie (r0, r1, … créés dedans)").required(true))
                .arg(a("precision", "f32 (rapide) | f64").default_value("f32"))
                .arg(a("backend", "cuda | hip").default_value("cuda"))
                .arg(Arg::new("impulsion").long("impulsion").num_args(0).help("écrit neutre_p.bin (cran 1)"))
                .arg(a("impulsions", "crans d'impulsion (1 ≤ k < ℓ/2), séparés par des virgules : neutre_p<k>.bin"))
                .arg(a("impulsions-vecteurs", "impulsions HORS AXE : vecteurs entiers k1:k2:k3 (unités 2π/ℓ, axes spatiaux 1,2,3) séparés par des virgules, ex. 2:2:1,3:0:0 ; pour chacun, corrélateur moyenné sur TOUTE l'orbite du groupe cubique (permutations et signes) dans neutre_o<k1>_<k2>_<k3>.bin (représentant = |k| décroissant) ; chaque |k_d| < ℓ/2, côtés spatiaux égaux ; éteint par défaut, sorties inchangées au bit"))
                .arg(a("reps", "représentations cubiques supplémentaires à impulsion nulle, ex. A2+,E+,T2+,T1- (dossier r<i>/reps/<rep>/ : neutre.bin et manifeste.json de même format)"))
                .arg(a("blocage", "niveaux de blocage de Teper mesurés APRÈS les niveaux APE (superliens de longueur 2^b, b ≥ 1), ex. 1,2,3 ; « aucun » = comportement d'origine").default_value("aucun"))
                .arg(a("ape-bloc", "itérations APE ordinaires du champ avant le premier blocage").default_value("0"))
                .arg(a("lissage-bloc", "itérations de lissage APE des superliens (agrafes de superliens) après chaque blocage").default_value("0"))
                .arg(a("p-bloc", "poids des agrafes allongées du blocage (p_a de Lucini-Teper-Wenger 2004, éq. eqn_oldblock)").default_value("1.0"))
                .arg(a("controle-cpu", "toutes les N configurations, remesure la réplique 0 sur le processeur (F64) et rapporte l'écart relatif maximal (0 = jamais)").default_value("0"))
                .arg(a("xi0", "réseau anisotrope : ξ₀ > 0 (plaquettes temporelles pondérées ξ₀, spatiales 1/ξ₀ ; défaut absent = isotrope, sorties identiques au bit) ; écrit en plus plaq_ss_st.csv (cfg,plaq_ss,plaq_st) si ξ₀ ≠ 1"))
                .arg(Arg::new("ops-bruts").long("ops-bruts").num_args(0).help("écrit aussi ops.bin (par réplique) : sommes brutes O_a(p,t) par configuration, f64, [config][t][p][a][re,im]"))
                .arg(a("deux-niveaux", "algorithme à deux niveaux (Lüscher–Weisz) D,N1[,S1] (S1 = balayages entre échantillons internes, défaut 1) : liens spatiaux des tranches de temps t ≡ 0 mod D gelés, N1 mises à jour internes, opérateurs moyennés par tranche épaisse ; écrit deux_niveaux.bin (même format que neutre.bin, impulsion nulle) en plus des fichiers ordinaires ; D = 0 : sans gel, N1 = 1 (test) ; défaut absent = éteint, sorties identiques au bit"))
                .arg(Arg::new("dn-hote").long("dn-hote").num_args(0).help("deux niveaux : ancienne voie (cumul sur l'hôte, opérateurs relus à chaque échantillon interne) au lieu du cumul sur la carte ; sorties identiques (contrôle)"))
                .arg(Arg::new("mutant").long("mutant").num_args(1).hide(true)),
        )
        .subcommand(
            Command::new("gpu-torelon")
                .about("torelons (lignes de Polyakov lissées) SUR CARTE, équivalent de `run` : un dossier r<i> par réplique, mêmes fichiers que `run`")
                .arg(a("xi0", "réseau anisotrope : ξ₀ > 0 (plaquettes temporelles pondérées ξ₀, spatiales 1/ξ₀ ; défaut absent = isotrope, sorties identiques au bit) ; écrit en plus plaq_ss_st.csv (cfg,plaq_ss,plaq_st) si ξ₀ ≠ 1"))
                .arg(Arg::new("ops-bruts").long("ops-bruts").num_args(0).help("écrit aussi ops.bin (par réplique) : sommes brutes Φ_a(p⊥,t) par configuration, f64, [config][t][p][a][re,im]"))
                .arg(a("secteur-flux", "charges de centre sur les axes spatiaux croissants (1,1,0 et permutations : lacets en L, escaliers et torelons d'axe ; 1,1,1 : escalier dans les trois directions) ; accepte aussi quatre charges avec 0 sur le temps ; côtés de flux égaux, impulsion nulle, mêmes niveaux de lissage ; écrit ops_flux.bin, corr_flux.csv, flux_moy.csv et l'entrée « secteur_flux » du manifeste"))
                .arg(Arg::new("flux-sans-corr").long("flux-sans-corr").num_args(0).help("avec --secteur-flux : écrit seulement ops_flux.bin, à corréler hors ligne"))
                .arg(a("profil", "profil du tube de flux (IRM) : séparations τ séparées par des virgules (1 ≤ τ ≤ tau-max) ; écrit profil.bin (f64, par paquet de configurations) et l'entrée « profil » du manifeste ; défaut absent = éteint, sorties identiques au bit"))
                .arg(a("profil-paquet", "configurations par paquet du profil (unité du jackknife)").default_value("25"))
                .arg(a("profil-niveau", "indice de l'opérateur (niveaux APE puis blocs) dont la carte Φ résolue en x⊥ sert de source/puits ; défaut = le dernier (le plus lissé)"))
                .arg(a("cotes", "l0,l1,l2,l3").required(true))
                .arg(a("beta", "β").required(true))
                .arg(a("graine", "graine de la réplique 0 (la réplique i prend graine + i)").default_value("1"))
                .arg(a("repliques", "nombre de répliques").default_value("8"))
                .arg(a("therm", "balayages de thermalisation").default_value("500"))
                .arg(a("nmeas", "configurations mesurées PAR réplique").default_value("100"))
                .arg(a("skip", "balayages entre deux mesures").default_value("5"))
                .arg(a("nor", "OR par balayage").default_value("4"))
                .arg(a("enroule", "direction d'enroulement w (0..3)").default_value("0"))
                .arg(a("temps", "axe du temps (0..3)").default_value("3"))
                .arg(a("niveaux", "itérations APE, croissantes").default_value("0,2,4,8,16"))
                .arg(a("alpha", "paramètre APE").default_value("0.5"))
                .arg(a("moments", "impulsions transverses na:nb séparées par des virgules").default_value("0:0"))
                .arg(a("tau-max", "τ maximal stocké").default_value("8"))
                .arg(a("sortie", "dossier de sortie").required(true))
                .arg(a("precision", "f32 | f64").default_value("f32"))
                .arg(a("blocage", "niveaux de blocage de Teper mesurés APRÈS les niveaux APE (superliens de longueur 2^b, b ≥ 1), ex. 1,2,3 ; « aucun » = comportement d'origine").default_value("aucun"))
                .arg(a("ape-bloc", "itérations APE ordinaires du champ avant le premier blocage").default_value("0"))
                .arg(a("lissage-bloc", "itérations de lissage APE des superliens (agrafes de superliens) après chaque blocage").default_value("0"))
                .arg(a("p-bloc", "poids des agrafes allongées du blocage (p_a de Lucini-Teper-Wenger 2004, éq. eqn_oldblock)").default_value("1.0"))
                .arg(a("deux-niveaux", "algorithme à deux niveaux (Lüscher–Weisz) D,N1[,S1] (S1 = balayages entre échantillons internes, défaut 1) : liens spatiaux des tranches de temps t ≡ 0 mod D gelés, N1 mises à jour internes ; écrit corr_deux_niveaux.csv (même format que corr.csv) en plus des fichiers ordinaires ; D = 0 : sans gel, N1 = 1 (test) ; défaut absent = éteint, sorties identiques au bit"))
                .arg(a("backend", "cuda | hip").default_value("cuda")),
        )
        .subcommand(
            Command::new("run")
                .about("chaîne + corrélateurs de torelon lissés (APE) par configuration")
                .arg(a("cotes", "l0,l1,l2,l3").required(true))
                .arg(a("beta", "β").required(true))
                .arg(a("graine", "graine Philox (u64), enregistrée dans le manifeste").default_value("1"))
                .arg(a("therm", "balayages de thermalisation").default_value("500"))
                .arg(a("nmeas", "nombre de configurations mesurées").default_value("100"))
                .arg(a("skip", "balayages (1 HB + nor OR) entre deux mesures").default_value("5"))
                .arg(a("nor", "OR par balayage").default_value("4"))
                .arg(a("enroule", "direction d'enroulement w (0..3)").default_value("0"))
                .arg(a("temps", "axe du temps (0..3)").default_value("3"))
                .arg(a("niveaux", "itérations APE des opérateurs, croissantes (ex. 0,2,4,8,16)").default_value("0,2,4,8,16"))
                .arg(a("alpha", "paramètre APE").default_value("0.5"))
                .arg(a("moments", "impulsions transverses na:nb séparées par des virgules").default_value("0:0"))
                .arg(a("tau-max", "τ maximal stocké").default_value("8"))
                .arg(a("depart", "chaud | froid").default_value("chaud"))
                .arg(a("sortie", "dossier de sortie").required(true))
                .arg(a("xi0", "réseau anisotrope : ξ₀ > 0 (plaquettes temporelles pondérées ξ₀, spatiales 1/ξ₀) ; défaut absent = isotrope, sorties identiques au bit ; écrit en plus plaq_ss_st.csv (cfg,plaq_ss,plaq_st)"))
                .arg(Arg::new("ponctuel").long("ponctuel").num_args(0).help("écrit aussi ponc.csv (corrélateur ponctuel, test de couplage fort)"))
                .arg(Arg::new("reprendre").long("reprendre").num_args(0).help("reprend depuis etat.bin du dossier"))
                .arg(Arg::new("mutant").long("mutant").num_args(1).hide(true)),
        )
}

pub fn executer(m: &ArgMatches) -> Resultat<i32> {
    match m.subcommand() {
        Some(("run", g)) => executer_run(g),
        Some(("neutre", g)) => executer_neutre(g),
        Some(("gpu", g)) => executer_gpu(g),
        Some(("gpu-torelon", g)) => executer_gpu_torelon(g),
        _ => Err(Erreur::Config("su2 spectro : sous-commande attendue (run)".into())),
    }
}

fn liste<T: std::str::FromStr>(t: &str, quoi: &str) -> Resultat<Vec<T>> {
    t.split(',')
        .map(|x| x.trim().parse::<T>().map_err(|_| Erreur::Config(format!("{quoi} : « {x} » invalide"))))
        .collect()
}

/// `--impulsions-vecteurs 2:2:1,3:0:0` -> orbites (contrôlées : non nul, |k_d| < ℓ/2, côtés spatiaux égaux, pas d'orbite en double).
fn vecteurs_orbites(m: &ArgMatches, g: &GeoA, temps: usize) -> Resultat<Vec<neutre::Orbite>> {
    let Some(t) = m.get_one::<String>("impulsions-vecteurs") else { return Ok(Vec::new()) };
    let mut ks: Vec<[i64; 3]> = Vec::new();
    for x in t.split(',') {
        let v: Vec<&str> = x.trim().split(':').collect();
        let n: Vec<i64> = v.iter().filter_map(|c| c.trim().parse::<i64>().ok()).collect();
        if v.len() != 3 || n.len() != 3 {
            return Err(Erreur::Config(format!("--impulsions-vecteurs : « {x} » invalide (attendu k1:k2:k3, entiers)")));
        }
        ks.push([n[0], n[1], n[2]]);
    }
    neutre::controler_vecteurs(g, temps, &ks)
}

/// Champs de manifeste d'une liste d'orbites.
fn manifeste_orbites(orbs: &[neutre::Orbite]) -> serde_json::Value {
    serde_json::json!({
        "impulsions_vecteurs": orbs.iter().map(|o| serde_json::json!({
            "representant": o.representant, "k2": o.representant.iter().map(|x| x * x).sum::<i64>(),
            "taille_orbite": o.taille, "vecteurs_mesures": o.vecteurs.len(), "vecteurs": o.vecteurs,
            "fichier": neutre::nom_fichier_orbite(o),
        })).collect::<Vec<_>>(),
        "definition_orbite": "orbite du groupe cubique O_h (permutations des trois axes spatiaux et signes de chaque composante) du representant = |k| par ordre decroissant, composantes en unites de 2pi/l le long des axes spatiaux 1,2,3 (ordre croissant des axes != temps) ; vecteurs = orbite modulo k -> -k (premiere composante non nulle > 0, ordre lexicographique decroissant) ; taille_orbite = nombre de vecteurs de l'orbite complete = 2 x vecteurs_mesures",
        "enregistrement_o": "neutre_o<k1>_<k2>_<k3>.bin : paires (a<=b) x (tau_max+1) f64 ; C_orb(tau) = (1/(2T nv)) sum_t sum_k Re[S_a(t;k)* S_b(t+tau;k) + S_b(t;k)* S_a(t+tau;k)], S(t;k) = sum_x e^{i k.(x+c)} (somme des images)(x,t)/(N_im V3), nv = vecteurs_mesures ; C(-k) = C(k) (champ reel) donc la moyenne sur les vecteurs mesures est la moyenne sur l'orbite complete ; memes operateurs, niveaux de lissage et bases que neutre.bin ; pour (k,0,0) : identique au bit a neutre_p<k>.bin",
    })
}

fn moments(t: &str) -> Resultat<Vec<[i32; 2]>> {
    t.split(',')
        .map(|x| {
            let v: Vec<&str> = x.trim().split(':').collect();
            if v.len() != 2 {
                return Err(Erreur::Config(format!("--moments : « {x} » (attendu na:nb)")));
            }
            let p = |s: &str| s.parse::<i32>().map_err(|_| Erreur::Config(format!("--moments : « {x} »")));
            Ok([p(v[0])?, p(v[1])?])
        })
        .collect()
}

fn ecrire_etat(dossier: &Path, c: &ChaineTube) -> Resultat<()> {
    let tmp = dossier.join("etat.bin.tmp");
    let fin = dossier.join("etat.bin");
    {
        let mut f = BufWriter::new(File::create(&tmp).map_err(|e| io("création", &tmp, e))?);
        let mut w = |b: &[u8]| f.write_all(b).map_err(|e| io("écriture", &tmp, e));
        w(b"SPECTRO1")?;
        w(&(c.sweep_id as u64).to_le_bytes())?;
        w(&(c.u.len() as u64).to_le_bytes())?;
        for q in &c.u {
            for x in q {
                w(&x.to_le_bytes())?;
            }
        }
        f.flush().map_err(|e| io("flush", &tmp, e))?;
    }
    fs::rename(&tmp, &fin).map_err(|e| io("renommage", &fin, e))
}

#[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
fn lire_etat(dossier: &Path, c: &mut ChaineTube) -> Resultat<()> {
    let p = dossier.join("etat.bin");
    let mut b = Vec::new();
    File::open(&p).and_then(|mut f| f.read_to_end(&mut b)).map_err(|e| io("lecture", &p, e))?;
    if b.len() < 24 || &b[0..8] != b"SPECTRO1" {
        return Err(Erreur::Format("etat.bin : en-tête inattendu".into()));
    }
    let sid = u64::from_le_bytes(b[8..16].try_into().unwrap());
    // B2-18 : identifiant de balayage lu sur 64 bits ; refus (avant toute modification de la chaîne) s'il ne tient pas sur 32 bits
    let sid32 = u32::try_from(sid).map_err(|_| Erreur::Format("etat.bin : identifiant de balayage supérieur à u32::MAX".into()))?;
    let n = usize::try_from(u64::from_le_bytes(b[16..24].try_into().unwrap()))
        .map_err(|_| Erreur::Format("etat.bin : nombre de liens hors de portée sur cette cible".into()))?;
    if n != c.u.len() || n.checked_mul(32).and_then(|x| x.checked_add(24)) != Some(b.len()) {
        return Err(Erreur::Format("etat.bin : taille incompatible avec les côtés".into()));
    }
    for (i, q) in c.u.iter_mut().enumerate() {
        for k in 0..4 {
            let o = 24 + 32 * i + 8 * k;
            q[k] = f64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        }
    }
    c.sweep_id = sid32;
    Ok(())
}

fn executer_run(m: &ArgMatches) -> Resultat<i32> {
    let g = GeoA::analyser(&arg(m, "cotes"))?;
    let beta: f64 = num(m, "beta")?;
    let graine: u64 = num(m, "graine")?;
    let therm: u64 = num(m, "therm")?;
    let nmeas: u64 = num(m, "nmeas")?;
    let skip: u64 = num(m, "skip")?;
    let nor: usize = num(m, "nor")?;
    let w: usize = num(m, "enroule")?;
    let temps: usize = num(m, "temps")?;
    let alpha: f64 = num(m, "alpha")?;
    let tau_max: usize = num(m, "tau-max")?;
    let niveaux: Vec<usize> = liste(&arg(m, "niveaux"), "--niveaux")?;
    let mom = moments(&arg(m, "moments"))?;
    let mutant = Mutant::analyser(m.get_one::<String>("mutant").map(|s| s.as_str()))?;
    let chaud = match arg(m, "depart").as_str() {
        "chaud" => true,
        "froid" => false,
        a => return Err(Erreur::Config(format!("--depart « {a} » (chaud|froid)"))),
    };
    if !(beta.is_finite() && beta >= 0.0) || skip == 0 {
        return Err(Erreur::Config("beta ≥ 0 fini et skip ≥ 1 requis".into()));
    }
    let axes = Axes::nouveaux(w, temps)?;
    let param = ParamMesure { axes, niveaux: niveaux.clone(), alpha, moments: mom.clone(), tau_max, mutant };
    let dossier = Path::new(&arg(m, "sortie")).to_path_buf();
    let reprise = m.get_flag("reprendre");
    if dossier.join("corr.csv").exists() && !reprise {
        return Err(Erreur::Config(format!("{} existe déjà (utiliser --reprendre ou un dossier neuf)", dossier.display())));
    }
    fs::create_dir_all(&dossier).map_err(|e| io("création", &dossier, e))?;

    let xi0 = param_xi0(m)?.filter(|&x| x != 1.0);
    let pile = Pile::depuis_torsion(&g, &Torsion::analyser("aucune")?);
    let pile_s = match xi0 {
        Some(x) => pile_aniso(Pile::depuis_torsion(&g, &Torsion::analyser("aucune")?), &g, x, temps),
        None => Pile::depuis_torsion(&g, &Torsion::analyser("aucune")?),
    };
    let mut c = ChaineTube::nouvelle(g.clone(), graine);
    c.init(chaud);
    let mut cfg0 = 0u64;
    if reprise {
        lire_etat(&dossier, &mut c)?;
        // nombre de lignes déjà écrites dans corr.csv (hors en-tête, une ligne par impulsion)
        let txt = fs::read_to_string(dossier.join("corr.csv")).map_err(|e| io("lecture", &dossier, e))?;
        cfg0 = (txt.lines().count().saturating_sub(1) / mom.len()) as u64;
    } else {
        for _ in 0..therm {
            c.pas(beta, nor, &pile_s);
        }
    }
    let ouvrir = |nom: &str, entete: &str| -> Resultat<BufWriter<File>> {
        let p = dossier.join(nom);
        let nouveau = !p.exists();
        let f = OpenOptions::new().create(true).append(true).open(&p).map_err(|e| io("ouverture", &p, e))?;
        let mut b = BufWriter::new(f);
        if nouveau {
            writeln!(b, "{entete}").map_err(|e| io("écriture", &p, e))?;
        }
        Ok(b)
    };
    let mut ent = String::from("cfg,px,py");
    for (ia, ib) in paires(niveaux.len()) {
        for tau in 0..=tau_max {
            ent.push_str(&format!(",c{ia}{ib}_{tau}"));
        }
    }
    let mut fcorr = ouvrir("corr.csv", &ent)?;
    let mut fphi = ouvrir("phi.csv", "cfg,niv,px,py,re,im")?;
    let mut fpl = ouvrir("plaq.csv", "cfg,plaq,poly_w")?;
    let ponc = m.get_flag("ponctuel");
    let mut fponc = if ponc { Some(ouvrir("ponc.csv", "cfg,niv,tau,val")?) } else { None };
    if !reprise {
        let man = serde_json::json!({
            "schema": "kvu-spectro-run/1", "statut": "experimental",
            "cotes": g.d, "beta": beta, "graine": graine, "therm": therm, "nmeas": nmeas, "skip": skip, "nor": nor,
            "enroule": w, "temps": temps, "transverses": axes.tr, "niveaux_ape": niveaux, "alpha": alpha, "moments": mom,
            "tau_max": tau_max, "depart": arg(m, "depart"), "mutant": format!("{mutant:?}"),
            "torsion": "aucune", "precision": "F64 CPU (su2::tube::ChaineTube)"
        });
        let mut man = man;
        if let Some(x) = xi0 {
            man["xi0"] = serde_json::json!({"xi0": x, "poids": "plaquettes contenant l'axe du temps xi0, spatiales 1/xi0"});
        }
        fs::write(dossier.join("manifeste.json"), serde_json::to_string_pretty(&man).unwrap() + "\n")
            .map_err(|e| io("écriture", &dossier, e))?;
    }
    let mut fss = if xi0.is_some() { Some(ouvrir("plaq_ss_st.csv", "cfg,plaq_ss,plaq_st")?) } else { None };
    let t0 = std::time::Instant::now();
    for cfg in cfg0..nmeas {
        for _ in 0..skip {
            c.pas(beta, nor, &pile_s);
        }
        let r = mesurer(&g, &c.u, &param)?;
        let me = c.mesure5(&pile);
        for (i, mm) in mom.iter().enumerate() {
            let mut l = format!("{cfg},{},{}", mm[0], mm[1]);
            for pr in &r.corr[i] {
                for v in pr {
                    l.push_str(&format!(",{v:.17e}"));
                }
            }
            writeln!(fcorr, "{l}").map_err(|e| io("écriture", &dossier, e))?;
            for (n, v) in r.phi_moyen[i].iter().enumerate() {
                writeln!(fphi, "{cfg},{n},{},{},{:.17e},{:.17e}", mm[0], mm[1], v[0], v[1]).map_err(|e| io("écriture", &dossier, e))?;
            }
        }
        if let Some(f) = fponc.as_mut() {
            for (n, v) in r.ponctuel.iter().enumerate() {
                for (tau, x) in v.iter().enumerate() {
                    writeln!(f, "{cfg},{n},{tau},{x:.17e}").map_err(|e| io("écriture", &dossier, e))?;
                }
            }
        }
        writeln!(fpl, "{cfg},{:.17e},{:.17e}", me[0], me[1 + w]).map_err(|e| io("écriture", &dossier, e))?;
        if let Some(f) = fss.as_mut() {
            let (pss, pst) = c.plaquettes_ss_st(temps);
            writeln!(f, "{cfg},{pss:.17e},{pst:.17e}").map_err(|e| io("écriture", &dossier, e))?;
        }
        if (cfg + 1) % 25 == 0 || cfg + 1 == nmeas {
            fcorr.flush().ok();
            fphi.flush().ok();
            fpl.flush().ok();
            if let Some(f) = fss.as_mut() {
                f.flush().ok();
            }
            if let Some(f) = fponc.as_mut() {
                f.flush().ok();
            }
            ecrire_etat(&dossier, &c)?;
            eprintln!("spectro: cfg {}/{} ({:.1}s)", cfg + 1, nmeas, t0.elapsed().as_secs_f64());
        }
    }
    fcorr.flush().ok();
    fphi.flush().ok();
    fpl.flush().ok();
    if let Some(f) = fss.as_mut() {
        f.flush().ok();
    }
    Ok(0)
}

/// `su2 spectro neutre` : sorties `neutre.bin` (f64 LE ; par configuration : corrélateurs [paire][τ], puis ⟨S_op⟩ [op]),
/// `plaq.csv` (cfg, plaquette), `manifeste.json`, `etat.bin` ; et, si des crans d'impulsion sont demandés, un
/// `neutre_p<k>.bin` par cran (f64 LE, [paire][τ] ; `neutre_p.bin` pour le cran 1 demandé par l'option `--impulsion`).
fn executer_neutre(m: &ArgMatches) -> Resultat<i32> {
    use super::neutre::{self, MutantN, ParamNeutre, FORMES};
    use std::io::Write as _;
    let g = GeoA::analyser(&arg(m, "cotes"))?;
    let beta: f64 = num(m, "beta")?;
    let graine: u64 = num(m, "graine")?;
    let therm: u64 = num(m, "therm")?;
    let nmeas: u64 = num(m, "nmeas")?;
    let skip: u64 = num(m, "skip")?;
    let nor: usize = num(m, "nor")?;
    let temps: usize = num(m, "temps")?;
    let alpha: f64 = num(m, "alpha")?;
    let tau_max: usize = num(m, "tau-max")?;
    let niveaux: Vec<usize> = liste(&arg(m, "niveaux"), "--niveaux")?;
    let noms: Vec<String> = arg(m, "formes").split(',').map(|x| x.trim().to_string()).collect();
    let formes: Vec<usize> = noms
        .iter()
        .map(|n| FORMES.iter().position(|f| f.0 == n).ok_or_else(|| Erreur::Config(format!("--formes : « {n} » inconnue"))))
        .collect::<Resultat<_>>()?;
    let mutant = MutantN::analyser(m.get_one::<String>("mutant").map(|s| s.as_str()))?;
    if temps > 3 || skip == 0 || !(beta.is_finite() && beta >= 0.0) {
        return Err(Erreur::Config("temps 0..3, skip ≥ 1, beta ≥ 0 fini requis".into()));
    }
    // crans d'impulsion mesurés : `--impulsions 1,2,3` (triés, dédoublonnés). L'ancienne option `--impulsion` demande le
    // cran 1 et garde son nom de fichier, `neutre_p.bin` (la campagne en cours lit ce nom) ; la nouvelle écrit un
    // `neutre_p<k>.bin` par cran. Aucune des deux : rien que l'impulsion nulle (comportement d'origine).
    let legacy = m.get_flag("impulsion");
    let demande = m.get_one::<String>("impulsions").cloned();
    let mut ks: Vec<i64> = match &demande {
        Some(t) => liste(t, "--impulsions")?,
        None => Vec::new(),
    };
    for &k in &ks {
        if k < 1 {
            return Err(Erreur::Config(format!(
                "--impulsions : cran {k} refusé (des crans entiers ≥ 1 attendus ; le signe n'a pas d'effet, C(−k) = C(k))"
            )));
        }
    }
    if legacy && !ks.contains(&1) {
        ks.push(1);
    }
    ks.sort_unstable();
    ks.dedup();
    neutre::controler_ks(&g, temps, &ks)?;
    let orbs = vecteurs_orbites(m, &g, temps)?;
    let param = ParamNeutre { temps, niveaux: niveaux.clone(), alpha, formes, tau_max, mutant };
    let dossier = Path::new(&arg(m, "sortie")).to_path_buf();
    let reprise = m.get_flag("reprendre");
    let binp = dossier.join("neutre.bin");
    if binp.exists() && !reprise {
        return Err(Erreur::Config(format!("{} existe déjà (--reprendre ou dossier neuf)", dossier.display())));
    }
    fs::create_dir_all(&dossier).map_err(|e| io("création", &dossier, e))?;
    let nops = param.nops();
    let reclen = (nops * (nops + 1) / 2 * (tau_max + 1) + nops) * 8;
    let reclen_p = nops * (nops + 1) / 2 * (tau_max + 1) * 8;
    // un fichier par cran mesuré : (rang dans `ks`, cran, chemin). Le cran 1 demandé par `--impulsion` s'écrit dans
    // `neutre_p.bin` ; si les deux options sont données ensemble, il a AUSSI son fichier numéroté `neutre_p1.bin`.
    let mut fichiers: Vec<(usize, i64, PathBuf)> = Vec::new();
    for (ik, &k) in ks.iter().enumerate() {
        if k == 1 && legacy {
            fichiers.push((ik, k, dossier.join("neutre_p.bin")));
            if demande.is_some() {
                fichiers.push((ik, k, dossier.join("neutre_p1.bin")));
            }
        } else {
            fichiers.push((ik, k, dossier.join(format!("neutre_p{k}.bin"))));
        }
    }
    let fichiers_orb: Vec<PathBuf> = orbs.iter().map(|o| dossier.join(neutre::nom_fichier_orbite(o))).collect();
    let xi0 = param_xi0(m)?.filter(|&x| x != 1.0);
    let pile = Pile::depuis_torsion(&g, &Torsion::analyser("aucune")?);
    let pile_s = match xi0 {
        Some(x) => pile_aniso(Pile::depuis_torsion(&g, &Torsion::analyser("aucune")?), &g, x, temps),
        None => Pile::depuis_torsion(&g, &Torsion::analyser("aucune")?),
    };
    let mut c = ChaineTube::nouvelle(g.clone(), graine);
    c.init(true);
    let mut cfg0 = 0u64;
    if reprise {
        lire_etat(&dossier, &mut c)?;
        let taille = fs::metadata(&binp).map(|x| x.len() as usize).unwrap_or(0);
        cfg0 = (taille / reclen) as u64;
        let f = OpenOptions::new().write(true).open(&binp).map_err(|e| io("ouverture", &binp, e))?;
        f.set_len((cfg0 as usize * reclen) as u64).map_err(|e| io("troncature", &binp, e))?;
        for (_, k, p) in &fichiers {
            let tp = fs::metadata(p).map(|x| x.len() as usize).unwrap_or(0);
            if tp < cfg0 as usize * reclen_p {
                return Err(Erreur::Config(format!(
                    "{} plus court que neutre.bin : reprise impossible (cran {k})",
                    p.display()
                )));
            }
            let f = OpenOptions::new().write(true).open(p).map_err(|e| io("ouverture", p, e))?;
            f.set_len((cfg0 as usize * reclen_p) as u64).map_err(|e| io("troncature", p, e))?;
        }
        for p in &fichiers_orb {
            let tp = fs::metadata(p).map(|x| x.len() as usize).unwrap_or(0);
            if tp < cfg0 as usize * reclen_p {
                return Err(Erreur::Config(format!("{} plus court que neutre.bin : reprise impossible", p.display())));
            }
            let f = OpenOptions::new().write(true).open(p).map_err(|e| io("ouverture", p, e))?;
            f.set_len((cfg0 as usize * reclen_p) as u64).map_err(|e| io("troncature", p, e))?;
        }
        // plaq.csv : on le recoupe au même nombre de lignes
        let pp = dossier.join("plaq.csv");
        let t = fs::read_to_string(&pp).unwrap_or_default();
        let mut l: Vec<&str> = t.lines().collect();
        l.truncate(1 + cfg0 as usize);
        fs::write(&pp, l.join("\n") + "\n").map_err(|e| io("écriture", &pp, e))?;
        if xi0.is_some() {
            let p2 = dossier.join("plaq_ss_st.csv");
            let t = fs::read_to_string(&p2).unwrap_or_default();
            let mut l: Vec<&str> = t.lines().collect();
            l.truncate(1 + cfg0 as usize);
            fs::write(&p2, l.join("\n") + "\n").map_err(|e| io("écriture", &p2, e))?;
        }
    } else {
        for _ in 0..therm {
            c.pas(beta, nor, &pile_s);
        }
        let fp: Vec<String> = fichiers
            .iter()
            .map(|(_, k, p)| format!("{k}={}", p.file_name().and_then(|x| x.to_str()).unwrap_or("?")))
            .collect();
        let man = serde_json::json!({
            "schema": "kvu-spectro-neutre/1", "statut": "experimental", "cotes": g.d, "beta": beta, "graine": graine,
            "therm": therm, "nmeas": nmeas, "skip": skip, "nor": nor, "temps": temps, "niveaux_ape": niveaux, "alpha": alpha,
            "formes": noms, "tau_max": tau_max, "nops": nops, "mutant": format!("{mutant:?}"),
            "ordre_ops": "op = i_niveau * n_formes + i_forme", "enregistrement": "paires (a<=b) x (tau_max+1) f64, puis nops moyennes f64", "impulsion": legacy, "impulsions": ks.clone(), "fichiers_impulsion": fp,
            "enregistrement_p": "neutre_p<k>.bin par cran k (neutre_p.bin pour le cran 1 demandé par --impulsion) : paires (a<=b) x (tau_max+1) f64 (impulsion k.2pi/l, moyenne des 3 axes, C(-k) = C(k))",
            "precision": "F64 CPU (su2::tube::ChaineTube)", "torsion": "aucune"
        });
        let mut man = man;
        if !orbs.is_empty() {
            for (k, v) in manifeste_orbites(&orbs).as_object().unwrap() {
                man[k] = v.clone();
            }
        }
        if let Some(x) = xi0 {
            man["xi0"] = serde_json::json!({"xi0": x, "poids": "plaquettes contenant l'axe du temps xi0, spatiales 1/xi0"});
            fs::write(dossier.join("plaq_ss_st.csv"), "cfg,plaq_ss,plaq_st\n").map_err(|e| io("écriture", &dossier, e))?;
        }
        fs::write(dossier.join("manifeste.json"), serde_json::to_string_pretty(&man).unwrap() + "\n")
            .map_err(|e| io("écriture", &dossier, e))?;
        fs::write(dossier.join("plaq.csv"), "cfg,plaq\n").map_err(|e| io("écriture", &dossier, e))?;
    }
    let mut fb = BufWriter::new(OpenOptions::new().create(true).append(true).open(&binp).map_err(|e| io("ouverture", &binp, e))?);
    // un rédacteur par fichier d'impulsion (le même cran peut être écrit deux fois : neutre_p.bin et neutre_p1.bin)
    let mut ecr: Vec<(usize, PathBuf, BufWriter<File>)> = fichiers
        .iter()
        .map(|(ik, _, p)| -> Resultat<(usize, PathBuf, BufWriter<File>)> {
            let f = OpenOptions::new().create(true).append(true).open(p).map_err(|e| io("ouverture", p, e))?;
            Ok((*ik, p.clone(), BufWriter::new(f)))
        })
        .collect::<Resultat<_>>()?;
    let mut ecr_orb: Vec<(PathBuf, BufWriter<File>)> = fichiers_orb
        .iter()
        .map(|p| -> Resultat<(PathBuf, BufWriter<File>)> {
            let f = OpenOptions::new().create(true).append(true).open(p).map_err(|e| io("ouverture", p, e))?;
            Ok((p.clone(), BufWriter::new(f)))
        })
        .collect::<Resultat<_>>()?;
    let pp = dossier.join("plaq.csv");
    let mut fp = BufWriter::new(OpenOptions::new().append(true).open(&pp).map_err(|e| io("ouverture", &pp, e))?);
    let mut fss = if xi0.is_some() {
        let p2 = dossier.join("plaq_ss_st.csv");
        Some(BufWriter::new(OpenOptions::new().append(true).open(&p2).map_err(|e| io("ouverture", &p2, e))?))
    } else {
        None
    };
    let t0 = std::time::Instant::now();
    for cfg in cfg0..nmeas {
        for _ in 0..skip {
            c.pas(beta, nor, &pile_s);
        }
        // le cas historique (la liste réduite au cran 1) repasse par l'entrée d'origine de l'option `--impulsion` : mêmes
        // bits pour la campagne en cours ; sans cran demandé, on reste sur `mesurer` (le comportement d'origine)
        let r = if !orbs.is_empty() {
            neutre::mesurer_orb(&g, &c.u, &param, &ks, &orbs)?
        } else {
            match ks.as_slice() {
                [] => neutre::mesurer(&g, &c.u, &param)?,
                [1] => neutre::mesurer_p(&g, &c.u, &param)?,
                _ => neutre::mesurer_ps(&g, &c.u, &param, &ks)?,
            }
        };
        let me = c.mesure5(&pile);
        let mut buf = Vec::with_capacity(reclen);
        for pr in &r.corr {
            for v in pr {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in &r.moy {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        fb.write_all(&buf).map_err(|e| io("écriture", &binp, e))?;
        for (ik, p, w) in ecr.iter_mut() {
            let mut bp = Vec::with_capacity(reclen_p);
            for pr in &r.corr_ps[*ik] {
                for v in pr {
                    bp.extend_from_slice(&v.to_le_bytes());
                }
            }
            w.write_all(&bp).map_err(|e| io("écriture", p.as_path(), e))?;
        }
        for (io_, (p, w)) in ecr_orb.iter_mut().enumerate() {
            let mut bp = Vec::with_capacity(reclen_p);
            for pr in &r.corr_orb[io_] {
                for v in pr {
                    bp.extend_from_slice(&v.to_le_bytes());
                }
            }
            w.write_all(&bp).map_err(|e| io("écriture", p.as_path(), e))?;
        }
        writeln!(fp, "{cfg},{:.17e}", me[0]).map_err(|e| io("écriture", &pp, e))?;
        if let Some(f) = fss.as_mut() {
            let (pss, pst) = c.plaquettes_ss_st(temps);
            writeln!(f, "{cfg},{pss:.17e},{pst:.17e}").map_err(|e| io("écriture", &dossier, e))?;
        }
        if (cfg + 1) % 25 == 0 || cfg + 1 == nmeas {
            fb.flush().ok();
            for (_, _, w) in ecr.iter_mut() {
                w.flush().ok();
            }
            for (_, w) in ecr_orb.iter_mut() {
                w.flush().ok();
            }
            fp.flush().ok();
            if let Some(f) = fss.as_mut() {
                f.flush().ok();
            }
            ecrire_etat(&dossier, &c)?;
            eprintln!("neutre: cfg {}/{} ({:.1}s)", cfg + 1, nmeas, t0.elapsed().as_secs_f64());
        }
    }
    fb.flush().ok();
    for (_, _, w) in ecr.iter_mut() {
        w.flush().ok();
    }
    fp.flush().ok();
    if let Some(f) = fss.as_mut() {
        f.flush().ok();
    }
    Ok(0)
}

/// `su2 spectro gpu` : voir `super::gpu`. Pas de point de reprise (une série interrompue se refait) ; sorties au format de `neutre`.
fn manifeste_deux_niveaux(dn: super::deux_niveaux::ParamDeuxNiveaux, tt: usize, tau_max: usize) -> serde_json::Value {
    serde_json::json!({
        "schema": "kvu-spectro-deux-niveaux/1", "statut": "experimental", "d": dn.d, "n1": dn.n1, "s1": dn.s1,
        "fichier": "deux_niveaux.bin : même enregistrement que neutre.bin (paires a<=b x (tau_max+1) f64, puis nops moyennes), impulsion nulle seulement",
        "estimateur": "temps gelés : liens spatiaux des tranches t = 0 mod d ; N1 mises à jour internes (mesure après chaque) ; paire (t,t+tau) dans deux tranches épaisses différentes (ou un temps gelé) : produit des moyennes internes ; même tranche : produit ordinaire moyenné sur les N1 échantillons ; d = 0 : sans gel, N1 = 1",
        "paires_valides_par_tau": super::deux_niveaux::paires_valides(tt, dn.d, tau_max),
        "temps_geles": (0..tt).filter(|t| dn.d > 0 && t % dn.d == 0).collect::<Vec<_>>(),
    })
}

fn executer_gpu(m: &ArgMatches) -> Resultat<i32> {
    use super::gpu::{axes_spatiaux, toutes_formes, ParamGpu, SpectroGpu};
    use super::neutre::{self, MutantN, ParamNeutre};
    use crate::commun::Precision;
    use crate::heritage::su2_hb::Q;
    use std::io::Write as _;
    let cotes = GeoA::analyser(&arg(m, "cotes"))?.d;
    let beta: f64 = num(m, "beta")?;
    let graine: u64 = num(m, "graine")?;
    let nrep: usize = num(m, "repliques")?;
    let therm: usize = num(m, "therm")?;
    let nmeas: usize = num(m, "nmeas")?;
    let skip: usize = num(m, "skip")?;
    let nor: usize = num(m, "nor")?;
    let temps: usize = num(m, "temps")?;
    let alpha: f64 = num(m, "alpha")?;
    let tau_max: usize = num(m, "tau-max")?;
    let ctl: usize = num(m, "controle-cpu")?;
    let niveaux: Vec<usize> = liste(&arg(m, "niveaux"), "--niveaux")?;
    let mut formes: Vec<String> = arg(m, "formes").split(',').map(|x| x.trim().to_string()).collect();
    if m.get_flag("base-riche") {
        for (n, _) in super::neutre::FORMES[5..].iter() {
            if !formes.iter().any(|f| f == n) {
                formes.push(n.to_string());
            }
        }
    }
    let precision: Precision = arg(m, "precision").parse()?;
    let mutant: u8 = match m.get_one::<String>("mutant").map(|s| s.as_str()) {
        None | Some("aucun") => 0,
        Some("agrafe-sans-dagger") => 1,
        Some("lissage-sans-projection") => 2,
        Some("trace-sans-demi") => 3,
        Some("somme-tronquee") => 4,
        Some("image-oubliee") => 5,
        Some(a) => return Err(Erreur::Config(format!("--mutant « {a} » inconnu"))),
    };
    if temps > 3 || skip == 0 || nrep == 0 || !(beta.is_finite() && beta >= 0.0) {
        return Err(Erreur::Config("temps 0..3, skip ≥ 1, repliques ≥ 1, beta ≥ 0 fini requis".into()));
    }
    let legacy = m.get_flag("impulsion");
    let demande = m.get_one::<String>("impulsions").cloned();
    let mut ks: Vec<i64> = match &demande {
        Some(t) => liste(t, "--impulsions")?,
        None => Vec::new(),
    };
    if ks.iter().any(|&k| k < 1) {
        return Err(Erreur::Config("--impulsions : crans entiers ≥ 1 attendus".into()));
    }
    if legacy && !ks.contains(&1) {
        ks.push(1);
    }
    ks.sort_unstable();
    ks.dedup();
    let deux: Option<super::deux_niveaux::ParamDeuxNiveaux> = match m.get_one::<String>("deux-niveaux") {
        Some(t) => Some(super::deux_niveaux::ParamDeuxNiveaux::analyser(t, cotes[temps]).map_err(Erreur::Config)?),
        None => None,
    };
    let g = GeoA::analyser(&arg(m, "cotes"))?;
    neutre::controler_ks(&g, temps, &ks)?;
    let orbs = vecteurs_orbites(m, &g, temps)?;
    if !orbs.is_empty() && (deux.is_some() || !ks.is_empty() || m.get_flag("ops-bruts")) {
        return Err(Erreur::Config("--impulsions-vecteurs : incompatible avec --impulsions, --impulsion, --deux-niveaux et --ops-bruts (une seule famille d'impulsions par essai ; le blocage est accepté)".into()));
    }
    let _ = axes_spatiaux(temps);
    let app = super::super::cli::ouvrir_backend(&arg(m, "backend"))?
        .ok_or_else(|| Erreur::Config("spectro gpu exige un GPU (--backend cuda|hip)".into()))?;
    let pg = ParamGpu {
        cotes,
        temps,
        niveaux: niveaux.clone(),
        alpha,
        formes: formes.clone(),
        tau_max,
        ks: ks.clone(),
        precision,
        mutant,
        nor,
        reps: m.get_one::<String>("reps").map(|t| t.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or_default(),
        betas: vec![beta; nrep],
        graines: (0..nrep as u64).map(|i| graine + i).collect(),
        bloc: param_bloc(m)?,
        ops_bruts: m.get_flag("ops-bruts"),
        xi0: param_xi0(m)?,
        orbites: orbs.clone(),
    };
    let aniso = pg.xi0.filter(|&x| x != 1.0);
    let ops_bruts = pg.ops_bruts;
    let pb = pg.bloc.clone();
    let niv_man = niveaux_manifeste(&niveaux, &pb);
    let mut sg = SpectroGpu::nouveau(&app, &pg)?;
    let nops = sg.nops();
    let dossier = Path::new(&arg(m, "sortie")).to_path_buf();
    // fichiers par cran (même règle de noms que `neutre`)
    let mut noms_p: Vec<(usize, i64, String)> = Vec::new();
    for (ik, &k) in ks.iter().enumerate() {
        if k == 1 && legacy {
            noms_p.push((ik, k, "neutre_p.bin".into()));
            if demande.is_some() {
                noms_p.push((ik, k, "neutre_p1.bin".into()));
            }
        } else {
            noms_p.push((ik, k, format!("neutre_p{k}.bin")));
        }
    }
    let mut fb = Vec::new();
    let mut fd: Vec<BufWriter<File>> = Vec::new();
    let mut fdp: Vec<Vec<(i64, BufWriter<File>)>> = Vec::new();
    let mut fo: Vec<BufWriter<File>> = Vec::new();
    let mut fp = Vec::new();
    let mut fpo: Vec<Vec<BufWriter<File>>> = Vec::new();
    let mut fpl = Vec::new();
    let mut fpa: Vec<BufWriter<File>> = Vec::new();
    let mut fr: Vec<Vec<BufWriter<File>>> = Vec::new();
    let mut frd: Vec<Vec<BufWriter<File>>> = Vec::new();
    for r in 0..nrep {
        let d = dossier.join(format!("r{r}"));
        if d.join("neutre.bin").exists() {
            return Err(Erreur::Config(format!("{} existe déjà (dossier neuf requis)", d.display())));
        }
        fs::create_dir_all(&d).map_err(|e| io("création", &d, e))?;
        let mut man = serde_json::json!({
            "schema": "kvu-spectro-neutre/1", "statut": "experimental", "cotes": g.d, "beta": beta, "graine": graine + r as u64,
            "therm": therm, "nmeas": nmeas, "skip": skip, "nor": nor, "temps": temps, "niveaux_ape": niv_man, "alpha": alpha,
            "formes": formes, "tau_max": tau_max, "nops": nops, "mutant": format!("gpu-mutant-{mutant}"),
            "ordre_ops": "op = i_niveau * n_formes + i_forme", "enregistrement": "paires (a<=b) x (tau_max+1) f64, puis nops moyennes f64",
            "impulsion": legacy, "impulsions": ks.clone(),
            "fichiers_impulsion": noms_p.iter().map(|(_, k, n)| format!("{k}={n}")).collect::<Vec<_>>(),
            "enregistrement_p": "neutre_p<k>.bin par cran k (neutre_p.bin pour le cran 1 demandé par --impulsion) : paires (a<=b) x (tau_max+1) f64 (impulsion k.2pi/l, moyenne des 3 axes, C(-k) = C(k))",
            "precision": format!("GPU {} (champ et produits de liens ; sommes F64 à arbre fixe), su2::spectro::gpu", precision.nom()),
            "torsion": "aucune", "repliques_gpu": nrep, "graine_replique": graine + r as u64,
        });
        ajouter_bloc_manifeste(&mut man, &pb);
        if !orbs.is_empty() {
            for (k, v) in manifeste_orbites(&orbs).as_object().unwrap() {
                man[k] = v.clone();
            }
        }
        if let Some(dn) = deux {
            man["deux_niveaux"] = manifeste_deux_niveaux(dn, cotes[temps], tau_max);
            if !ks.is_empty() {
                man["deux_niveaux"]["fichiers_impulsion"] = serde_json::json!(ks.iter().map(|k| format!("{k}=deux_niveaux_p{k}.bin")).collect::<Vec<_>>());
            }
        }
        if let Some(x) = aniso {
            man["xi0"] = serde_json::json!({"xi0": x, "poids": "plaquettes temporelles (direction temps) xi0, spatiales 1/xi0 ; S = (beta/xi0) sum_s + beta xi0 sum_t", "fichier": "plaq_ss_st.csv"});
        }
        if ops_bruts {
            man["ops_bruts"] = serde_json::json!({
                "fichier": "ops.bin", "format": "f64 little-endian, [configuration][t][p][a][re, im]; O_a(p,t) = somme sur la tranche t de exp(-i p.x) O_a(x), phase au centre de chaque image, poids 1/(N_images*V3) comme neutre.bin",
                "temps": g.d[temps], "n_impulsions": 1 + 6 * ks.len(), "n_operateurs": nops,
                "impulsions": sg.ops_impulsions(), "octets_par_configuration": g.d[temps] * (1 + 6 * ks.len()) * nops * 16,
                "ordre_ops": "a = i_niveau * n_formes + i_forme", "formes": formes, "niveaux_ape": niv_man,
            });
        }
        fs::write(d.join("manifeste.json"), serde_json::to_string_pretty(&man).unwrap() + "\n").map_err(|e| io("écriture", &d, e))?;
        fs::write(d.join("plaq.csv"), "cfg,plaq\n").map_err(|e| io("écriture", &d, e))?;
        let ouv = |n: &str| -> Resultat<BufWriter<File>> {
            let p = d.join(n);
            Ok(BufWriter::new(OpenOptions::new().create(true).append(true).open(&p).map_err(|e| io("ouverture", &p, e))?))
        };
        fb.push(ouv("neutre.bin")?);
        if deux.is_some() {
            fd.push(ouv("deux_niveaux.bin")?);
            let mut v = Vec::new();
            for (_, k, _) in &noms_p {
                let n = format!("deux_niveaux_p{k}.bin");
                if !v.iter().any(|(kk, _): &(i64, BufWriter<File>)| *kk == *k) {
                    v.push((*k, ouv(&n)?));
                }
            }
            fdp.push(v);
        }
        if ops_bruts {
            fo.push(ouv("ops.bin")?);
        }
        fpl.push(ouv("plaq.csv")?);
        if aniso.is_some() {
            let mut b = ouv("plaq_ss_st.csv")?;
            writeln!(b, "cfg,plaq_ss,plaq_st").map_err(|e| io("écriture", &d, e))?;
            fpa.push(b);
        }
        let mut v = Vec::new();
        for (_, _, n) in &noms_p {
            v.push(ouv(n)?);
        }
        fp.push(v);
        let mut vo = Vec::new();
        for o in &orbs {
            vo.push(ouv(&neutre::nom_fichier_orbite(o))?);
        }
        fpo.push(vo);
        let mut vr = Vec::new();
        let mut vrd: Vec<BufWriter<File>> = Vec::new();
        for rp in &sg.reps {
            let dr = d.join("reps").join(rp.nom.replace('+', "p").replace('-', "m"));
            fs::create_dir_all(&dr).map_err(|e| io("création", &dr, e))?;
            let mut manr = serde_json::json!({
                "schema": "kvu-spectro-neutre/1", "statut": "experimental", "cotes": g.d, "beta": beta, "graine": graine + r as u64,
                "therm": therm, "nmeas": nmeas, "skip": skip, "nor": nor, "temps": temps, "niveaux_ape": niv_man, "alpha": alpha,
                "formes": rp.formes, "tau_max": tau_max, "nops": rp.formes.len() * (niveaux.len() + pb.niveaux.len()), "representation": rp.nom,
                "ordre_ops": "op = i_niveau * n_formes_retenues + i_forme (formes retenues = opérateurs non nuls de la représentation)",
                "enregistrement": "paires (a<=b) x (tau_max+1) f64, puis nops moyennes f64 ; corrélateur moyenné sur les lignes de la représentation",
                "impulsions": [], "precision": format!("GPU {}", precision.nom()), "torsion": "aucune",
            });
            ajouter_bloc_manifeste(&mut manr, &pb);
            if let Some(dn) = deux {
                manr["deux_niveaux"] = manifeste_deux_niveaux(dn, cotes[temps], tau_max);
            }
            fs::write(dr.join("manifeste.json"), serde_json::to_string_pretty(&manr).unwrap() + "\n").map_err(|e| io("écriture", &dr, e))?;
            let pth = dr.join("neutre.bin");
            vr.push(BufWriter::new(OpenOptions::new().create(true).append(true).open(&pth).map_err(|e| io("ouverture", &pth, e))?));
            if deux.is_some() {
                let pd = dr.join("deux_niveaux.bin");
                vrd.push(BufWriter::new(OpenOptions::new().create(true).append(true).open(&pd).map_err(|e| io("ouverture", &pd, e))?));
            }
        }
        fr.push(vr);
        frd.push(vrd);
    }
    let t0 = std::time::Instant::now();
    sg.init()?;
    sg.avancer(therm)?;
    eprintln!("gpu: thermalisation {therm} balayages : {:.1}s", t0.elapsed().as_secs_f64());
    let t1 = std::time::Instant::now();
    let pn = ParamNeutre { temps, niveaux: niveaux.clone(), alpha, formes: Vec::new(), tau_max, mutant: MutantN::Aucun };
    let formes_cpu: Vec<usize> = formes
        .iter()
        .map(|n| neutre::FORMES.iter().position(|f| f.0 == n))
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default();
    let pn = ParamNeutre { formes: formes_cpu.clone(), ..pn };
    let mut ecart_rel_max = 0.0f64;
    let mut ecart_norm_max = 0.0f64;
    let mut nctl = 0usize;
    let _ = toutes_formes();
    let mut ecart_hote = 0.0f64;
    let (mut chrono_mises_a_jour, mut chrono_operateurs, mut chrono_hote) = (0.0f64, 0.0f64, 0.0f64);
    let mut controle_gel: Option<serde_json::Value> = None;
    let masque: Vec<u8> = match deux {
        Some(dn) if dn.d > 0 => {
            let gg = GeoA::analyser(&arg(m, "cotes"))?;
            super::deux_niveaux::masque_gel(gg.v, temps, dn.d, |s| gg.coords(s)[temps])
        }
        _ => Vec::new(),
    };
    for cfg in 0..nmeas {
        sg.avancer(skip)?;
        let res = sg.mesurer()?;
        if let Some(dn) = deux {
            let (tt, nops) = (cotes[temps], sg.nops());
            // contrôle hôte : le même estimateur ordinaire recalculé sur l'hôte doit retrouver le noyau k_corr
            if cfg < 2 {
                let (main, _) = sg.operateurs_hote()?;
                for (c, mc) in res.iter().enumerate() {
                    let h = super::deux_niveaux::corr_ordinaire(&main[c * nops * tt..(c + 1) * nops * tt], nops, tt, tau_max);
                    let sc = mc.corr.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
                    for (x, y) in h.iter().zip(&mc.corr) {
                        ecart_hote = ecart_hote.max((x - y).abs() / sc);
                    }
                }
            }
            let nlev_t = sg.nlev_tot();
            let nf_t = nlev_t.max(1);
            let _ = nf_t;
            let nq_t = 1 + 6 * ks.len();
            let nf_f = nops / nlev_t;
            let npair_t = nops * (nops + 1) / 2;
            let ntau_t = tau_max + 1;
            let voie_hote = m.get_flag("dn-hote");
            let avant = if cfg == 0 && dn.d > 0 { Some(sg.boite.telecharger(0)?) } else { None };
            if dn.d > 0 {
                sg.fixer_masque(Some(&masque))?;
            }
            let mut cumuls: Vec<super::deux_niveaux::Cumul> = Vec::new();
            let mut cumuls_reps: Vec<Vec<Vec<super::deux_niveaux::Cumul>>> = Vec::new();
            // corrélateurs aux impulsions : [chaîne][cran] -> [paire][τ]
            let mut corr_p: Vec<Vec<Vec<f64>>> = Vec::new();
            if voie_hote {
                cumuls = (0..nrep).map(|_| super::deux_niveaux::Cumul::nouveau(nops, tt, tau_max, dn.d)).collect();
                cumuls_reps = (0..nrep)
                    .map(|_| sg.reps.iter().map(|rp| (0..rp.dim).map(|_| super::deux_niveaux::Cumul::nouveau(rp.nops(), tt, tau_max, dn.d)).collect()).collect())
                    .collect();
                let mut brut: Vec<super::deux_niveaux::CumulBrut> =
                    (0..nrep).map(|_| super::deux_niveaux::CumulBrut::nouveau(nops, tt, nq_t, tau_max, dn.d)).collect();
                for _j in 0..dn.n1 {
                    let tp = std::time::Instant::now();
                    if dn.d > 0 {
                        sg.avancer(dn.s1)?;
                    }
                    chrono_mises_a_jour += tp.elapsed().as_secs_f64();
                    let tp = std::time::Instant::now();
                    let (main, reps_h) = sg.operateurs_hote()?;
                    let sgv = if ks.is_empty() { Vec::new() } else { sg.sg_hote()? };
                    chrono_operateurs += tp.elapsed().as_secs_f64();
                    let tp = std::time::Instant::now();
                    for (c, cu) in cumuls.iter_mut().enumerate() {
                        cu.ajouter(&main[c * nops * tt..(c + 1) * nops * tt]);
                    }
                    if !ks.is_empty() {
                        let lc = nops * tt * nq_t;
                        for (c, bc) in brut.iter_mut().enumerate() {
                            bc.ajouter(&sgv[c * lc..(c + 1) * lc]);
                        }
                    }
                    for (i, rows) in reps_h.iter().enumerate() {
                        let n2 = sg.reps[i].nops();
                        for (row, v) in rows.iter().enumerate() {
                            for c in 0..nrep {
                                cumuls_reps[c][i][row].ajouter(&v[c * n2 * tt..(c + 1) * n2 * tt]);
                            }
                        }
                    }
                    chrono_hote += tp.elapsed().as_secs_f64();
                }
                for bc in brut.iter() {
                    if ks.is_empty() {
                        corr_p.push(Vec::new());
                    } else {
                        corr_p.push((0..ks.len()).map(|ik| super::deux_niveaux::corr_somme(&bc.somme, &bc.ordinaire, bc.n, nops, tt, nq_t, tau_max, dn.d, ik as i32)).collect());
                    }
                }
            } else {
                sg.cumul_init(dn.d)?;
                let verif = std::env::var("KVU_DN_VERIF").is_ok();
                let mut verif_brut: Vec<super::deux_niveaux::CumulBrut> = if verif { (0..nrep).map(|_| super::deux_niveaux::CumulBrut::nouveau(nops, tt, nq_t, tau_max, dn.d)).collect() } else { Vec::new() };
                for _j in 0..dn.n1 {
                    let tp = std::time::Instant::now();
                    if dn.d > 0 {
                        sg.avancer(dn.s1)?;
                    }
                    chrono_mises_a_jour += tp.elapsed().as_secs_f64();
                    let tp = std::time::Instant::now();
                    sg.cumul_ajouter()?;
                    chrono_operateurs += tp.elapsed().as_secs_f64();
                    if verif {
                        let sgv = sg.sg_hote()?;
                        let lc = nops * tt * nq_t;
                        for (c, bc) in verif_brut.iter_mut().enumerate() {
                            bc.ajouter(&sgv[c * lc..(c + 1) * lc]);
                        }
                    }
                }
                let tp = std::time::Instant::now();
                let sc = sg.cumul_lire()?;
                if verif {
                    let (mut nd, mut nt, mut ns) = (0usize, 0usize, 0usize);
                    for (c, bc) in verif_brut.iter().enumerate() {
                        let base = c * (1 + ks.len()) * npair_t * ntau_t;
                        for (i, x) in bc.ordinaire.iter().enumerate() {
                            nt += 1;
                            if *x != sc.o[base + i] {
                                nd += 1;
                            }
                        }
                        let lc = nops * tt * nq_t;
                        for (i, x) in bc.somme.iter().enumerate() {
                            if *x != sc.s[c * lc + i] {
                                ns += 1;
                            }
                        }
                    }
                    eprintln!("dn-verif cfg {cfg} : produits differents {nd}/{nt}, sommes differentes {ns}");
                }
                chrono_hote += tp.elapsed().as_secs_f64();
                let lc = nops * tt * nq_t;
                for c in 0..nrep {
                    let mut somme = vec![0f64; nops * tt];
                    for a in 0..nops {
                        for t in 0..tt {
                            somme[a * tt + t] = sc.s[c * lc + (a * tt + t) * nq_t];
                        }
                    }
                    let base = c * (1 + ks.len()) * npair_t * ntau_t;
                    cumuls.push(super::deux_niveaux::Cumul::depuis_sommes(nops, tt, 1, tau_max, dn.d, sc.n, somme, sc.o[base..base + npair_t * ntau_t].to_vec()));
                    corr_p.push(
                        (0..ks.len())
                            .map(|ik| {
                                super::deux_niveaux::corr_somme(&sc.s[c * lc..(c + 1) * lc], &sc.o[base..base + (1 + ks.len()) * npair_t * ntau_t], sc.n, nops, tt, nq_t, tau_max, dn.d, ik as i32)
                            })
                            .collect(),
                    );
                }
                let _ = nf_f;
                for c in 0..nrep {
                    let mut per_rep = Vec::new();
                    for (i, rows) in sc.reps.iter().enumerate() {
                        let n2 = sg.reps[i].nops();
                        let np2 = n2 * (n2 + 1) / 2;
                        let mut v = Vec::new();
                        for (sm, ord) in rows.iter() {
                            v.push(super::deux_niveaux::Cumul::depuis_sommes(n2, tt, 1, tau_max, dn.d, sc.n, sm[c * n2 * tt..(c + 1) * n2 * tt].to_vec(), ord[c * np2 * ntau_t..(c + 1) * np2 * ntau_t].to_vec()));
                        }
                        per_rep.push(v);
                    }
                    cumuls_reps.push(per_rep);
                }
            }
            if dn.d > 0 {
                sg.fixer_masque(None)?;
            }
            if let Some(av) = avant {
                let ap = sg.boite.telecharger(0)?;
                let (mut gm, mut gn, mut lm, mut ln) = (0usize, 0usize, 0usize, 0usize);
                for li in 0..masque.len() {
                    let ch = av[4 * li..4 * li + 4] != ap[4 * li..4 * li + 4];
                    match (masque[li] != 0, ch) {
                        (true, true) => gm += 1,
                        (true, false) => gn += 1,
                        (false, true) => lm += 1,
                        (false, false) => ln += 1,
                    }
                }
                controle_gel = Some(serde_json::json!({"liens_geles": gm + gn, "geles_modifies": gm, "libres": lm + ln, "libres_inchanges": ln}));
            }
            for (r, cu) in cumuls.iter().enumerate() {
                let (c2, m2) = cu.corr();
                let mut buf = Vec::with_capacity((c2.len() + m2.len()) * 8);
                for v in c2.iter().chain(m2.iter()) {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                fd[r].write_all(&buf).map_err(|e| io("écriture", &dossier, e))?;
                for (k, f) in fdp[r].iter_mut() {
                    let ik = ks.iter().position(|x| x == k).unwrap();
                    let mut bp = Vec::with_capacity(corr_p[r][ik].len() * 8);
                    for v in &corr_p[r][ik] {
                        bp.extend_from_slice(&v.to_le_bytes());
                    }
                    f.write_all(&bp).map_err(|e| io("écriture", &dossier, e))?;
                }
                for (i, rows) in cumuls_reps[r].iter().enumerate() {
                    let dim = rows.len() as f64;
                    let mut cc: Vec<f64> = Vec::new();
                    let mut mm: Vec<f64> = Vec::new();
                    for cr in rows {
                        let (c3, m3) = cr.corr();
                        if cc.is_empty() {
                            cc = vec![0.0; c3.len()];
                            mm = vec![0.0; m3.len()];
                        }
                        for (x, y) in cc.iter_mut().zip(&c3) {
                            *x += y / dim;
                        }
                        for (x, y) in mm.iter_mut().zip(&m3) {
                            *x += y / dim;
                        }
                    }
                    let mut bb = Vec::new();
                    for v in cc.iter().chain(mm.iter()) {
                        bb.extend_from_slice(&v.to_le_bytes());
                    }
                    frd[r][i].write_all(&bb).map_err(|e| io("écriture", &dossier, e))?;
                }
            }
        }
        for (r, mc) in res.iter().enumerate() {
            let mut buf = Vec::with_capacity((mc.corr.len() + mc.moy.len()) * 8);
            for v in mc.corr.iter().chain(mc.moy.iter()) {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            fb[r].write_all(&buf).map_err(|e| io("écriture", &dossier, e))?;
            if ops_bruts {
                let mut bo = Vec::with_capacity(mc.ops.len() * 8);
                for v in &mc.ops {
                    bo.extend_from_slice(&v.to_le_bytes());
                }
                fo[r].write_all(&bo).map_err(|e| io("écriture", &dossier, e))?;
            }
            for (j, (ik, _, _)) in noms_p.iter().enumerate() {
                let mut bp = Vec::with_capacity(mc.corr_ps[*ik].len() * 8);
                for v in &mc.corr_ps[*ik] {
                    bp.extend_from_slice(&v.to_le_bytes());
                }
                fp[r][j].write_all(&bp).map_err(|e| io("écriture", &dossier, e))?;
            }
            for (io_, w) in fpo[r].iter_mut().enumerate() {
                let mut bp = Vec::with_capacity(mc.corr_orb[io_].len() * 8);
                for v in &mc.corr_orb[io_] {
                    bp.extend_from_slice(&v.to_le_bytes());
                }
                w.write_all(&bp).map_err(|e| io("écriture", &dossier, e))?;
            }
            for (i, (c, mo)) in mc.reps.iter().enumerate() {
                let mut bb = Vec::with_capacity((c.len() + mo.len()) * 8);
                for v in c.iter().chain(mo.iter()) {
                    bb.extend_from_slice(&v.to_le_bytes());
                }
                fr[r][i].write_all(&bb).map_err(|e| io("écriture", &dossier, e))?;
            }
            writeln!(fpl[r], "{cfg},{:.17e}", mc.plaq).map_err(|e| io("écriture", &dossier, e))?;
            if let Some((ps, pt)) = mc.plaq_st {
                writeln!(fpa[r], "{cfg},{ps:.17e},{pt:.17e}").map_err(|e| io("écriture", &dossier, e))?;
            }
        }
        if ctl > 0 && (cfg % ctl == 0) && formes_cpu.len() == formes.len() {
            // remesure de la réplique 0 sur le processeur (F64), mêmes configurations
            let raw = sg.boite.telecharger(0)?;
            let u: Vec<Q> = raw.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
            let r = if pb.actif() { super::bloc::mesurer_avec_blocs_orb(&g, &u, &pn, &ks, &orbs, &pb)? } else { neutre::mesurer_orb(&g, &u, &pn, &ks, &orbs)? };
            let flat: Vec<f64> = r.corr.iter().flatten().copied().collect();
            let (mut e, mut n) = (0.0f64, 0.0f64);
            let mut cmp = |a: &[f64], b: &[f64]| {
                let sc = b.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                for (x, y) in a.iter().zip(b) {
                    if y.abs() > 1e-6 * sc {
                        e = e.max((x - y).abs() / y.abs());
                    }
                    n = n.max((x - y).abs() / sc);
                }
            };
            cmp(&res[0].corr, &flat);
            for (ik, _) in ks.iter().enumerate() {
                let fl: Vec<f64> = r.corr_ps[ik].iter().flatten().copied().collect();
                cmp(&res[0].corr_ps[ik], &fl);
            }
            for (io_, _) in orbs.iter().enumerate() {
                let fl: Vec<f64> = r.corr_orb[io_].iter().flatten().copied().collect();
                cmp(&res[0].corr_orb[io_], &fl);
            }
            ecart_rel_max = ecart_rel_max.max(e);
            ecart_norm_max = ecart_norm_max.max(n);
            nctl += 1;
        }
        if (cfg + 1) % 250 == 0 || cfg + 1 == nmeas {
            eprintln!("gpu: cfg {}/{} ({:.1}s, {:.1} cfg/s toutes répliques)", cfg + 1, nmeas, t1.elapsed().as_secs_f64(), ((cfg + 1) * nrep) as f64 / t1.elapsed().as_secs_f64());
        }
    }
    for w in fb.iter_mut().chain(fpl.iter_mut()).chain(fo.iter_mut()).chain(fpa.iter_mut()).chain(fd.iter_mut()) {
        w.flush().ok();
    }
    for v in fp.iter_mut().chain(fpo.iter_mut()).chain(fr.iter_mut()).chain(frd.iter_mut()) {
        for w in v.iter_mut() {
            w.flush().ok();
        }
    }
    let dt = t1.elapsed().as_secs_f64();
    let bilan = serde_json::json!({"cfg_par_replique": nmeas, "repliques": nrep, "secondes_mesure": dt,
        "cfg_par_seconde_total": (nmeas * nrep) as f64 / dt, "controle_cpu": {"n": nctl, "ecart_relatif_max": ecart_rel_max, "ecart_normalise_max": ecart_norm_max},
        "deux_niveaux": deux.map(|dn| serde_json::json!({"d": dn.d, "n1": dn.n1, "s1": dn.s1, "voie": if m.get_flag("dn-hote") { "hote" } else { "carte" }, "ecart_hote_vs_noyau_normalise_max": ecart_hote, "controle_gel": controle_gel,
            "chrono_s": {"mises_a_jour_internes": chrono_mises_a_jour, "operateurs_carte_et_remontee": chrono_operateurs, "cumul_hote": chrono_hote}}))});
    fs::write(dossier.join("bilan_gpu.json"), serde_json::to_string_pretty(&bilan).unwrap() + "\n").map_err(|e| io("écriture", &dossier, e))?;
    eprintln!("gpu: fini : {}", bilan);
    Ok(0)
}


/// `su2 spectro gpu-torelon` : voir `super::gpu_torelon`. Fichiers de `run` : corr.csv, phi.csv, plaq.csv, manifeste.json.
#[allow(clippy::type_complexity)] // style seulement (indices explicites conservés : aucun changement numérique)
#[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
fn executer_gpu_torelon(m: &ArgMatches) -> Resultat<i32> {
    use super::gpu_torelon::{ParamTorelon, TorelonGpu};
    use crate::commun::Precision;
    use std::io::Write as _;
    let g = GeoA::analyser(&arg(m, "cotes"))?;
    let beta: f64 = num(m, "beta")?;
    let graine: u64 = num(m, "graine")?;
    let nrep: usize = num(m, "repliques")?;
    let therm: usize = num(m, "therm")?;
    let nmeas: usize = num(m, "nmeas")?;
    let skip: usize = num(m, "skip")?;
    let nor: usize = num(m, "nor")?;
    let w: usize = num(m, "enroule")?;
    let temps: usize = num(m, "temps")?;
    let alpha: f64 = num(m, "alpha")?;
    let tau_max: usize = num(m, "tau-max")?;
    let niveaux: Vec<usize> = liste(&arg(m, "niveaux"), "--niveaux")?;
    let mom = moments(&arg(m, "moments"))?;
    let precision: Precision = arg(m, "precision").parse()?;
    if skip == 0 || nrep == 0 || !(beta.is_finite() && beta >= 0.0) {
        return Err(Erreur::Config("skip ≥ 1, repliques ≥ 1, beta ≥ 0 fini requis".into()));
    }
    let axes = Axes::nouveaux(w, temps)?;
    let pbt = param_bloc(m)?;
    let niv_man = niveaux_manifeste(&niveaux, &pbt);
    let deux: Option<super::deux_niveaux::ParamDeuxNiveaux> = match m.get_one::<String>("deux-niveaux") {
        Some(t) => Some(super::deux_niveaux::ParamDeuxNiveaux::analyser(t, g.d[temps]).map_err(Erreur::Config)?),
        None => None,
    };
    let xi0 = param_xi0(m)?;
    let aniso_t = xi0.filter(|&x| x != 1.0);
    let flux_sans_corr = m.get_flag("flux-sans-corr");
    let flux: Option<super::flux::SecteurCentre> = match m.get_one::<String>("secteur-flux") {
        Some(t) => {
            let f = super::flux::SecteurCentre::analyser(t, &g.d, temps)?;
            f.controler_niveaux(&g.d, temps, &pbt)?;
            Some(f)
        }
        None => None,
    };
    let ops_bruts = m.get_flag("ops-bruts");
    let profil_taus: Vec<usize> = match m.get_one::<String>("profil") {
        Some(t) => liste(t, "--profil")?,
        None => Vec::new(),
    };
    let profil_paquet: usize = num(m, "profil-paquet")?;
    let profil_niveau: Option<usize> = match m.get_one::<String>("profil-niveau") {
        Some(t) => Some(t.trim().parse().map_err(|_| Erreur::Config("--profil-niveau : entier attendu".into()))?),
        None => None,
    };
    if profil_paquet == 0 {
        return Err(Erreur::Config("--profil-paquet ≥ 1 requis".into()));
    }
    let app = super::super::cli::ouvrir_backend(&arg(m, "backend"))?
        .ok_or_else(|| Erreur::Config("spectro gpu-torelon exige un GPU (--backend cuda|hip)".into()))?;
    let p = ParamTorelon {
        cotes: g.d,
        w,
        temps,
        niveaux: niveaux.clone(),
        alpha,
        moments: mom.clone(),
        tau_max,
        precision,
        nor,
        betas: vec![beta; nrep],
        graines: (0..nrep as u64).map(|i| graine + i).collect(),
        bloc: pbt.clone(),
        xi0,
        ops_bruts,
        profil: profil_taus.clone(),
        profil_niveau,
    };
    let mut tg = TorelonGpu::nouveau(&app, &p)?;
    let dossier = Path::new(&arg(m, "sortie")).to_path_buf();
    let mut ent = String::from("cfg,px,py");
    for (ia, ib) in paires(niveaux.len() + pbt.niveaux.len()) {
        for tau in 0..=tau_max {
            ent.push_str(&format!(",c{ia}{ib}_{tau}"));
        }
    }
    let mut fc = Vec::new();
    let mut fcd: Vec<BufWriter<File>> = Vec::new();
    let mut fph = Vec::new();
    let mut fpl = Vec::new();
    let mut fpa: Vec<BufWriter<File>> = Vec::new();
    let mut fo: Vec<BufWriter<File>> = Vec::new();
    let (mut ffo, mut fflc, mut fflm): (Vec<BufWriter<File>>, Vec<BufWriter<File>>, Vec<BufWriter<File>>) = (Vec::new(), Vec::new(), Vec::new());
    let mut chrono_flux = 0.0f64;
    let imps_ops = tg.ops_impulsions();
    let (pta, ptb) = (g.d[axes.tr[0]], g.d[axes.tr[1]]);
    let mut fprof: Vec<BufWriter<File>> = Vec::new();
    let mut accus: Vec<super::profil::Accu> = Vec::new();
    for r in 0..nrep {
        let d = dossier.join(format!("r{r}"));
        if d.join("corr.csv").exists() {
            return Err(Erreur::Config(format!("{} existe déjà (dossier neuf requis)", d.display())));
        }
        fs::create_dir_all(&d).map_err(|e| io("création", &d, e))?;
        let mut man = serde_json::json!({
            "schema": "kvu-spectro-run/1", "statut": "experimental",
            "cotes": g.d, "beta": beta, "graine": graine + r as u64, "therm": therm, "nmeas": nmeas, "skip": skip, "nor": nor,
            "enroule": w, "temps": temps, "transverses": axes.tr, "niveaux_ape": niv_man, "alpha": alpha, "moments": mom,
            "tau_max": tau_max, "depart": "chaud", "mutant": "Aucun", "torsion": "aucune",
            "precision": format!("GPU {} (champ et produits de liens ; sommes F64), su2::spectro::gpu_torelon", precision.nom()),
        });
        ajouter_bloc_manifeste(&mut man, &pbt);
        if let Some(dn) = deux {
            let mut mj = manifeste_deux_niveaux(dn, g.d[temps], tau_max);
            mj["fichier"] = serde_json::json!("corr_deux_niveaux.csv : même format que corr.csv (une ligne par cfg et par impulsion, colonnes c<a><b>_<tau>)");
            man["deux_niveaux"] = mj;
        }
        if let Some(x) = aniso_t {
            man["xi0"] = serde_json::json!({"xi0": x, "poids": "plaquettes temporelles (direction temps) xi0, spatiales 1/xi0 ; S = (beta/xi0) sum_s + beta xi0 sum_t", "fichier": "plaq_ss_st.csv"});
        }
        if let Some(f) = &flux {
            for nom in ["ops_flux.bin", "corr_flux.csv", "flux_moy.csv"] {
                let chemin = d.join(nom);
                if chemin.exists() {
                    return Err(Erreur::Config(format!("{} existe déjà (dossier neuf requis)", chemin.display())));
                }
            }
            man["secteur_flux"] = super::flux::manifeste_centre(f, &p, &niv_man);
            if flux_sans_corr {
                man["secteur_flux"]["fichiers"] = serde_json::json!({"ops_flux.bin": man["secteur_flux"]["fichiers"]["ops_flux.bin"].clone(), "note": "--flux-sans-corr : ni corr_flux.csv ni flux_moy.csv"});
            }
        }
        if ops_bruts {
            let nlev_t = niveaux.len() + pbt.niveaux.len();
            let imp4: Vec<[i64; 4]> = imps_ops
                .iter()
                .map(|m| {
                    let mut n = [0i64; 4];
                    n[axes.tr[0]] = m[0] as i64;
                    n[axes.tr[1]] = m[1] as i64;
                    n
                })
                .collect();
            man["ops_bruts"] = serde_json::json!({
                "fichier": "ops.bin", "format": "f64 little-endian, [configuration][t][p][a][re, im]; Phi_a(p,t) = (1/N_perp) somme_{x_perp} exp(-i p.x_perp) (1/2)Tr prod U_w lisse, a = i_niveau (APE puis bloques), p en unites de 2pi/L (axes transverses)",
                "temps": g.d[temps], "n_impulsions": imps_ops.len(), "n_operateurs": nlev_t,
                "impulsions": imp4, "octets_par_configuration": g.d[temps] * imps_ops.len() * nlev_t * 16,
                "ordre_ops": "a = i_niveau", "niveaux_ape": niv_man, "enroule": w, "transverses": axes.tr,
            });
        }
        if !profil_taus.is_empty() {
            let niv_p = profil_niveau.unwrap_or(niveaux.len() + pbt.niveaux.len() - 1);
            let acc = super::profil::Accu::nouveau(pta, ptb, g.d[temps], &profil_taus);
            man["profil"] = serde_json::json!({
                "fichier": "profil.bin", "format": "f64 little-endian ; un enregistrement par paquet de configurations (le dernier peut etre plus court)",
                "enregistrement": "[ncfg, R_e, R_m, puis pour chaque tau (ordre de 'taus') : D, N_e[da*tb+db], N_m[da*tb+db]]",
                "longueur_enregistrement": acc.longueur_record(), "taus": profil_taus, "paquet": profil_paquet,
                "ta": pta, "tb": ptb, "tt": g.d[temps], "axes_transverses": axes.tr, "enroule": w, "temps": temps, "niveau_phi": niv_p,
                "definition": "C_k(Delta;tau) = sum_cfg N_k[Delta] / sum_cfg D - sum_cfg R_k / (ncfg_total * ta*tb*tt) ; D = sum_{x,t} Phi(x,t)Phi(x,t+tau), N_k = sum_{x,t} Phi(x,t)Phi(x,t+tau) rho_k(x+Delta, t+tau/2) (tau impair : moyenne des deux tranches du milieu) ; R_k = sum_{x,t} rho_k",
                "rho": "sum_{x_w} (1 - 1/2 Tr P) ; k=0 electrique (plans contenant le temps), k=1 magnetique ; chaque plaquette repartie a parts egales sur ses coins non-w (carte centree sur le reseau, symetrique sous reflexion) ; non ponderee par xi0",
            });
            accus.push(acc);
        }
        fs::write(d.join("manifeste.json"), serde_json::to_string_pretty(&man).unwrap() + "\n").map_err(|e| io("écriture", &d, e))?;
        let ouv = |n: &str, entete: &str| -> Resultat<BufWriter<File>> {
            let p = d.join(n);
            let mut b = BufWriter::new(OpenOptions::new().create(true).append(true).open(&p).map_err(|e| io("ouverture", &p, e))?);
            writeln!(b, "{entete}").map_err(|e| io("écriture", &p, e))?;
            Ok(b)
        };
        if !profil_taus.is_empty() {
            let p = d.join("profil.bin");
            fprof.push(BufWriter::new(OpenOptions::new().create(true).append(true).open(&p).map_err(|e| io("ouverture", &p, e))?));
        }
        fc.push(ouv("corr.csv", &ent)?);
        if let Some(f) = &flux {
            if !flux_sans_corr {
                let nb = (niveaux.len() + pbt.niveaux.len()) * f.n_base();
                let mut e = String::from("cfg,px,py");
                for (ia, ib) in paires(nb) {
                    for tau in 0..=tau_max {
                        e.push_str(&format!(",c{ia}_{ib}_{tau}"));
                    }
                }
                fflc.push(ouv("corr_flux.csv", &e)?);
                fflm.push(ouv("flux_moy.csv", "cfg,operateur,moyenne_t")?);
            }
            let p = d.join("ops_flux.bin");
            ffo.push(BufWriter::new(OpenOptions::new().create_new(true).write(true).open(&p).map_err(|e| io("ouverture", &p, e))?));
        }
        if deux.is_some() {
            fcd.push(ouv("corr_deux_niveaux.csv", &ent)?);
        }
        fph.push(ouv("phi.csv", "cfg,niv,px,py,re,im")?);
        fpl.push(ouv("plaq.csv", "cfg,plaq,poly_w")?);
        if aniso_t.is_some() {
            fpa.push(ouv("plaq_ss_st.csv", "cfg,plaq_ss,plaq_st")?);
        }
        if ops_bruts {
            let p = d.join("ops.bin");
            fo.push(BufWriter::new(OpenOptions::new().create(true).append(true).open(&p).map_err(|e| io("ouverture", &p, e))?));
        }
    }
    let t0 = std::time::Instant::now();
    tg.init()?;
    tg.avancer(therm)?;
    eprintln!("gpu-torelon: thermalisation {therm} : {:.1}s", t0.elapsed().as_secs_f64());
    let t1 = std::time::Instant::now();
    let tt = g.d[temps];
    let nlev = niveaux.len() + pbt.niveaux.len();
    let masque: Vec<u8> = match deux {
        Some(dn) if dn.d > 0 => super::deux_niveaux::masque_gel(g.v, temps, dn.d, |s| g.coords(s)[temps]),
        _ => Vec::new(),
    };
    let mut ecart_hote = 0.0f64;
    let (mut chrono_mises_a_jour, mut chrono_operateurs, mut chrono_hote) = (0.0f64, 0.0f64, 0.0f64);
    let mut controle_gel: Option<serde_json::Value> = None;
    for cfg in 0..nmeas {
        tg.avancer(skip)?;
        let res = tg.mesurer()?;
        if let Some(fl) = &flux {
            let tf = std::time::Instant::now();
            let rf = tg.mesurer_flux_centre(fl)?;
            chrono_flux += tf.elapsed().as_secs_f64();
            for (r, ops) in rf.iter().enumerate() {
                let mut bo = Vec::with_capacity(ops.len() * 8);
                for v in ops {
                    bo.extend_from_slice(&v.to_le_bytes());
                }
                ffo[r].write_all(&bo).map_err(|e| io("écriture", &dossier, e))?;
                if flux_sans_corr {
                    continue;
                }
                let mut l = format!("{cfg},0,0");
                for pr in super::flux::corr_flux_centre(fl, ops, nlev, tt, tau_max) {
                    for v in pr {
                        l.push_str(&format!(",{v:.17e}"));
                    }
                }
                writeln!(fflc[r], "{l}").map_err(|e| io("écriture", &dossier, e))?;
                for (i, v) in super::flux::moyennes_centre(fl, ops, nlev, tt).iter().enumerate() {
                    writeln!(fflm[r], "{cfg},{i},{v:.17e}").map_err(|e| io("écriture", &dossier, e))?;
                }
            }
        }
        if let Some(dn) = deux {
            let nmom = mom.len();
            let extraire = |phi: &[f64], c: usize, m: usize| -> Vec<f64> {
                let mut o = vec![0f64; nlev * tt * 2];
                for lev in 0..nlev {
                    for t in 0..tt {
                        let src = ((((c * nlev + lev) * nmom + m) * tt) + t) * 2;
                        let dst = (lev * tt + t) * 2;
                        o[dst] = phi[src];
                        o[dst + 1] = phi[src + 1];
                    }
                }
                o
            };
            if cfg < 2 {
                let phi = tg.phi_hote()?;
                for (c, mc) in res.iter().enumerate() {
                    for m in 0..nmom {
                        let mut cu = super::deux_niveaux::Cumul::nouveau_comp(nlev, tt, 2, tau_max, 0);
                        cu.ajouter(&extraire(&phi, c, m));
                        let h = cu.corr().0;
                        let gpu: Vec<f64> = mc.corr[m].iter().flatten().copied().collect();
                        let sc = gpu.iter().fold(0.0f64, |a, x| a.max(x.abs())).max(1e-300);
                        for (x, y) in h.iter().zip(&gpu) {
                            ecart_hote = ecart_hote.max((x - y).abs() / sc);
                        }
                    }
                }
            }
            let mut cumuls: Vec<Vec<super::deux_niveaux::Cumul>> = (0..nrep)
                .map(|_| (0..nmom).map(|_| super::deux_niveaux::Cumul::nouveau_comp(nlev, tt, 2, tau_max, dn.d)).collect())
                .collect();
            let avant = if cfg == 0 && dn.d > 0 { Some(tg.boite.telecharger(0)?) } else { None };
            if dn.d > 0 {
                tg.fixer_masque(Some(&masque))?;
            }
            for _j in 0..dn.n1 {
                let tp = std::time::Instant::now();
                if dn.d > 0 {
                    tg.avancer(dn.s1)?;
                }
                chrono_mises_a_jour += tp.elapsed().as_secs_f64();
                let tp = std::time::Instant::now();
                let phi = tg.phi_hote()?;
                chrono_operateurs += tp.elapsed().as_secs_f64();
                let tp = std::time::Instant::now();
                for c in 0..nrep {
                    for m in 0..nmom {
                        cumuls[c][m].ajouter(&extraire(&phi, c, m));
                    }
                }
                chrono_hote += tp.elapsed().as_secs_f64();
            }
            if dn.d > 0 {
                tg.fixer_masque(None)?;
            }
            if let Some(av) = avant {
                let ap = tg.boite.telecharger(0)?;
                let (mut gm, mut gn, mut lm, mut ln) = (0usize, 0usize, 0usize, 0usize);
                for li in 0..masque.len() {
                    let ch = av[4 * li..4 * li + 4] != ap[4 * li..4 * li + 4];
                    match (masque[li] != 0, ch) {
                        (true, true) => gm += 1,
                        (true, false) => gn += 1,
                        (false, true) => lm += 1,
                        (false, false) => ln += 1,
                    }
                }
                controle_gel = Some(serde_json::json!({"liens_geles": gm + gn, "geles_modifies": gm, "libres": lm + ln, "libres_inchanges": ln}));
            }
            for (r, cm) in cumuls.iter().enumerate() {
                for (i, mm) in mom.iter().enumerate() {
                    let mut l = format!("{cfg},{},{}", mm[0], mm[1]);
                    for v in cm[i].corr().0 {
                        l.push_str(&format!(",{v:.17e}"));
                    }
                    writeln!(fcd[r], "{l}").map_err(|e| io("écriture", &dossier, e))?;
                }
            }
        }
        for (r, mc) in res.iter().enumerate() {
            for (i, mm) in mom.iter().enumerate() {
                let mut l = format!("{cfg},{},{}", mm[0], mm[1]);
                for pr in &mc.corr[i] {
                    for v in pr {
                        l.push_str(&format!(",{v:.17e}"));
                    }
                }
                writeln!(fc[r], "{l}").map_err(|e| io("écriture", &dossier, e))?;
                for (n, v) in mc.phi_moyen[i].iter().enumerate() {
                    writeln!(fph[r], "{cfg},{n},{},{},{:.17e},{:.17e}", mm[0], mm[1], v[0], v[1]).map_err(|e| io("écriture", &dossier, e))?;
                }
            }
            writeln!(fpl[r], "{cfg},{:.17e},{:.17e}", mc.plaq, mc.poly_w).map_err(|e| io("écriture", &dossier, e))?;
            if let Some((ps, pt)) = mc.plaq_st {
                writeln!(fpa[r], "{cfg},{ps:.17e},{pt:.17e}").map_err(|e| io("écriture", &dossier, e))?;
            }
            if let Some(cartes) = &mc.cartes {
                accus[r].ajouter(cartes);
                if accus[r].ncfg == profil_paquet || cfg + 1 == nmeas {
                    let mut bo = Vec::new();
                    for v in accus[r].record() {
                        bo.extend_from_slice(&v.to_le_bytes());
                    }
                    fprof[r].write_all(&bo).map_err(|e| io("écriture", &dossier, e))?;
                    accus[r].raz();
                }
            }
            if ops_bruts {
                let mut bo = Vec::with_capacity(mc.ops.len() * 8);
                for v in &mc.ops {
                    bo.extend_from_slice(&v.to_le_bytes());
                }
                fo[r].write_all(&bo).map_err(|e| io("écriture", &dossier, e))?;
            }
        }
        if (cfg + 1) % 250 == 0 || cfg + 1 == nmeas {
            eprintln!("gpu-torelon: cfg {}/{} ({:.1}s, {:.2} cfg/s toutes répliques)", cfg + 1, nmeas, t1.elapsed().as_secs_f64(), ((cfg + 1) * nrep) as f64 / t1.elapsed().as_secs_f64());
        }
    }
    for wr in fc.iter_mut().chain(fph.iter_mut()).chain(fpl.iter_mut()).chain(fpa.iter_mut()).chain(fo.iter_mut()).chain(fprof.iter_mut()).chain(fcd.iter_mut()).chain(ffo.iter_mut()).chain(fflc.iter_mut()).chain(fflm.iter_mut()) {
        wr.flush().map_err(|e| io("écriture", &dossier, e))?;
    }
    let dt = t1.elapsed().as_secs_f64();
    if let Some(dn) = deux {
        let b = serde_json::json!({"d": dn.d, "n1": dn.n1, "s1": dn.s1, "secondes_mesure": dt, "ecart_hote_vs_noyau_normalise_max": ecart_hote, "controle_gel": controle_gel,
            "chrono_s": {"mises_a_jour_internes": chrono_mises_a_jour, "operateurs_carte_et_remontee": chrono_operateurs, "cumul_hote": chrono_hote}});
        fs::write(dossier.join("bilan_deux_niveaux.json"), serde_json::to_string_pretty(&b).unwrap() + "\n").map_err(|e| io("écriture", &dossier, e))?;
    }
    if flux.is_some() {
        eprintln!("gpu-torelon: secteur de flux : {chrono_flux:.1}s sur {dt:.1}s de mesure ({:.1} %)", 100.0 * chrono_flux / dt);
    }
    eprintln!("gpu-torelon: fini : {} cfg/réplique, {} répliques, {:.1}s, {:.2} cfg/s au total", nmeas, nrep, dt, (nmeas * nrep) as f64 / dt);
    Ok(0)
}
