//! Flot d'une cellule (β, L) : découverte des chaînes, blocs, estimateurs `sd`, `plaq`, `mcrg2`, jackknife par chaîne, τ_int, domaine.
//!
//! Définitions GELÉES au journal (« WP8 GEL des protocoles », points 1, 6, 7, 8) :
//! - une cellule = toutes les chaînes d'un même β ; l'estimateur d'une cellule porte sur les sommes POOLÉES de toutes leurs configurations ;
//! - jackknife = suppression d'une chaîne, σ² = (C−1)/C Σ(θ_c − θ̄)², estimateur non linéaire recalculé à chaque réplicat ;
//! - `plaq` : β′ tel que ⟨P⟩_Wilson(β′ ; L′) = ⟨P̄⟩ sur la courbe d'étalonnage de côté L′ (PCHIP, jamais d'extrapolation) ;
//! - `mcrg2` : même chose pour P (1×1) et R (1×2) ; tête = moyenne des deux β′, écart |β′_P − β′_R| = systématique de schéma ;
//! - a_j = β′ − β − 0,4081 (SU(2), L_b = 3, schéma de Bałaban, un pas) : quantité cible rapportée, JAMAIS une gate, aucune lecture physique.

use super::ape::{bloquer_script, SchemaScript};
use super::etalonner::{courbe_depuis, depuis_json};
use super::extensions::{ajouter_options, ConfigFlot, OptionsFlot};
use super::jackknife::{jackknife_groupes, ResJackknife};
use super::mcrg2::plaquette_et_rectangle;
use super::plaq::{Courbe, Inversion, Observable};
use super::sd::{beta_sd, stats_sd, ModeRapport, RapportBessel, StatsSd};
use super::tau::{stat_serie, tau_int};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::rg::bloc::{bloquer, OptionsBloc, StatsTheta};
use crate::rg::domaine;
use crate::rg::geometrie::Geometrie;
use crate::rg::lissage::{bloquer_operateur, OptionsOperateur};
use crate::rg::operateur::Operateur;
use crate::rg::swendsen::bloquer_b2;
use crate::stockage::npy::lire_champ;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const SCHEMA_FLOT: &str = "kvu-rg-flot/1";
/// δβ₁(L_b = 3) = 11 C_A² ln 3 /(12 π²) pour SU(2) (ARCHITECTURE §4.3 ; S1/VERDICT.md:132-134).
pub const DELTA_BETA_1: f64 = 0.4081;
pub const C_SWENDSEN: f64 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schema {
    Balaban,
    Swendsen,
    /// Blocage `ape` (c = 0,5) du script n3 : référence de RG-N3, pas le témoin de Swendsen.
    Ape,
    /// Décimation du script n3.
    Dec,
}

impl Schema {
    pub fn nom(self) -> &'static str {
        match self {
            Schema::Balaban => "balaban",
            Schema::Swendsen => "swendsen",
            Schema::Ape => "ape",
            Schema::Dec => "dec",
        }
    }

    pub fn analyser(t: &str) -> Resultat<Schema> {
        match t {
            "balaban" => Ok(Schema::Balaban),
            "swendsen" => Ok(Schema::Swendsen),
            "ape" => Ok(Schema::Ape),
            "dec" => Ok(Schema::Dec),
            _ => Err(Erreur::Config(format!("schéma « {t} » inconnu (balaban|swendsen|ape|dec)"))),
        }
    }

    fn facteur(self) -> usize {
        match self {
            Schema::Balaban => 3,
            Schema::Swendsen | Schema::Ape | Schema::Dec => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Estimateurs {
    pub sd: bool,
    pub plaq: bool,
    pub mcrg2: bool,
}

impl Estimateurs {
    pub fn analyser(t: &str) -> Resultat<Estimateurs> {
        let mut e = Estimateurs { sd: false, plaq: false, mcrg2: false };
        for m in t.split(',').map(str::trim).filter(|m| !m.is_empty()) {
            match m {
                "sd" => e.sd = true,
                "plaq" => e.plaq = true,
                "mcrg2" => e.mcrg2 = true,
                _ => return Err(Erreur::Config(format!("estimateur « {m} » inconnu (sd|plaq|mcrg2)"))),
            }
        }
        if !(e.sd || e.plaq || e.mcrg2) {
            return Err(Erreur::Config("aucun estimateur demandé".into()));
        }
        Ok(e)
    }
}

// ------------------------------------------------------------------------------------------
// Chaînes
// ------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Chaine {
    pub nom: String,
    pub dossier: PathBuf,
    pub beta: f64,
    pub l: usize,
    pub configs: Vec<PathBuf>,
    pub dump_every: Option<u64>,
}

/// Motif avec jokers `*` (aucun autre caractère spécial).
pub fn correspond(motif: &str, texte: &str) -> bool {
    let parts: Vec<&str> = motif.split('*').collect();
    if parts.len() == 1 {
        return motif == texte;
    }
    let (premier, dernier) = (parts[0], parts[parts.len() - 1]);
    if !texte.starts_with(premier) || !texte.ends_with(dernier) || texte.len() < premier.len() + dernier.len() {
        return false;
    }
    let mut reste = &texte[premier.len()..texte.len() - dernier.len()];
    for p in &parts[1..parts.len() - 1] {
        match reste.find(p) {
            Some(i) => reste = &reste[i + p.len()..],
            None => return false,
        }
    }
    true
}

fn configs_de(dossier: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dossier)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("cfg_") && n.ends_with(".npy") && n != "cfg_000000.npy")
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn lire_chaine(dossier: &Path) -> Resultat<Option<Chaine>> {
    let configs = configs_de(dossier);
    if configs.is_empty() {
        return Ok(None);
    }
    let man: Value = serde_json::from_str(
        &std::fs::read_to_string(dossier.join("manifest.json"))
            .map_err(|e| Erreur::io(format!("manifest.json de {}", dossier.display()), e))?,
    )
    .map_err(|e| Erreur::Format(format!("manifest.json de {} : {e}", dossier.display())))?;
    let beta = man["beta"].as_f64().ok_or_else(|| Erreur::Format(format!("{} : beta absent du manifeste", dossier.display())))?;
    let l = man["l"].as_u64().ok_or_else(|| Erreur::Format(format!("{} : l absent du manifeste", dossier.display())))? as usize;
    Ok(Some(Chaine {
        nom: dossier.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        dossier: dossier.to_path_buf(),
        beta,
        l,
        configs,
        dump_every: man["dump_every"].as_u64(),
    }))
}

/// Chaînes d'un ensemble : `ens` lui-même s'il contient des `cfg_*.npy`, sinon ses sous-dossiers (filtrés par motif), triés par nom.
pub fn decouvrir(ens: &Path, filtre: Option<&str>) -> Resultat<Vec<Chaine>> {
    if let Some(c) = lire_chaine(ens)? {
        return Ok(vec![c]);
    }
    let mut dossiers: Vec<PathBuf> = std::fs::read_dir(ens)
        .map_err(|e| Erreur::io(format!("lecture de {}", ens.display()), e))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter(|p| filtre.is_none_or(|m| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| correspond(m, n))))
        .collect();
    dossiers.sort();
    let mut v = Vec::new();
    for d in dossiers {
        if let Some(c) = lire_chaine(&d)? {
            v.push(c);
        }
    }
    if v.is_empty() {
        return Err(Erreur::Config(format!("aucune chaîne (dossier avec cfg_*.npy et manifest.json) dans {}", ens.display())));
    }
    Ok(v)
}

/// Groupe les chaînes par β (clé = β arrondi à 1e-9), ordre de β croissant.
pub fn grouper_par_beta(chaines: Vec<Chaine>) -> Vec<(f64, Vec<Chaine>)> {
    let mut groupes: Vec<(i64, f64, Vec<Chaine>)> = Vec::new();
    for c in chaines {
        let cle = (c.beta * 1e9).round() as i64;
        match groupes.iter_mut().find(|g| g.0 == cle) {
            Some(g) => g.2.push(c),
            None => groupes.push((cle, c.beta, vec![c])),
        }
    }
    groupes.sort_by_key(|g| g.0);
    groupes.into_iter().map(|g| (g.1, g.2)).collect()
}

/// Série de plaquette (phase `sample`) de `measurements.csv` d'une chaîne, si le fichier existe.
pub fn serie_plaquette(dossier: &Path) -> Option<Vec<f64>> {
    let texte = std::fs::read_to_string(dossier.join("measurements.csv")).ok()?;
    let mut lignes = texte.lines();
    let en_tete: Vec<&str> = lignes.next()?.split(',').collect();
    let (ip, ic) = (en_tete.iter().position(|c| *c == "phase")?, en_tete.iter().position(|c| *c == "plaquette")?);
    let mut v = Vec::new();
    for l in lignes {
        let c: Vec<&str> = l.split(',').collect();
        if c.get(ip) == Some(&"sample") {
            v.push(c.get(ic)?.parse().ok()?);
        }
    }
    Some(v)
}

// ------------------------------------------------------------------------------------------
// Données d'une chaîne
// ------------------------------------------------------------------------------------------

/// Résultats par configuration d'une chaîne, au niveau final du flot.
#[derive(Clone, Debug)]
pub struct DonneesChaine {
    pub nom: String,
    pub l_fin: usize,
    pub sd: StatsSd,
    /// Plaquette et rectangle de chaque configuration (ordre des fichiers).
    pub p: Vec<f64>,
    pub r: Vec<f64>,
    /// Domaine de chaque pas de bloc (Bałaban), pooled sur les configurations de la chaîne.
    pub theta: Vec<StatsTheta>,
}

/// Bloque `pas` fois et rend (champ final, stats θ de chaque pas).
pub fn bloquer_champ(champ: ChampHote, schema: Schema, pas: usize, fils: usize) -> Resultat<(ChampHote, Vec<StatsTheta>)> {
    bloquer_champ_avec(champ, schema, pas, fils, None, &Operateur::TroisCentre)
}

fn bloquer_champ_avec(champ: ChampHote, schema: Schema, pas: usize, fils: usize, gpu: Option<&crate::rg::gpu::BlocGpu>, op: &Operateur) -> Resultat<(ChampHote, Vec<StatsTheta>)> {
    let mut c = champ;
    let mut thetas = Vec::new();
    for _ in 0..pas {
        match schema {
            // WP22 : opérateur quasi-local (CPU) ; pas de logarithmes, donc pas de statistiques θ.
            Schema::Balaban if !op.est_trois_centre() => {
                c = bloquer_operateur(&c, op, &OptionsOperateur::normale(fils))?.grossier;
            }
            Schema::Balaban => {
                let s = match gpu {
                    Some(g) => g.bloquer_hote(&c, true)?,
                    None => bloquer(&c, &OptionsBloc { geometrie: Geometrie::TroisCentre, fils, angle_so3: false })?,
                };
                thetas.push(s.stats);
                c = s.grossier;
            }
            Schema::Swendsen => c = bloquer_b2(&c, C_SWENDSEN)?,
            Schema::Ape => c = bloquer_script(&c, SchemaScript::Ape, C_SWENDSEN)?,
            Schema::Dec => c = bloquer_script(&c, SchemaScript::Dec, 0.0)?,
        }
    }
    Ok((c, thetas))
}

pub fn traiter_chaine(ch: &Chaine, schema: Schema, pas: usize, fils: usize) -> Resultat<DonneesChaine> {
    Ok(traiter_chaine_avec(ch, schema, pas, fils, None, false, &Operateur::TroisCentre)?.0)
}

fn traiter_chaine_avec(ch: &Chaine, schema: Schema, pas: usize, fils: usize, gpu: Option<&crate::rg::gpu::BlocGpu>, garder: bool, op: &Operateur) -> Resultat<(DonneesChaine, Vec<ConfigFlot>)> {
    let mut configs = Vec::new();
    let mut d = DonneesChaine { nom: ch.nom.clone(), l_fin: 0, sd: StatsSd::default(), p: vec![], r: vec![], theta: vec![] };
    for f in &ch.configs {
        let champ = lire_champ(f)?;
        if champ.l != ch.l {
            return Err(Erreur::Format(format!("{} : L = {} ≠ L du manifeste {}", f.display(), champ.l, ch.l)));
        }
        let (fin, th) = bloquer_champ_avec(champ, schema, pas, fils, gpu, op)?;
        d.l_fin = fin.l;
        let sd = stats_sd(&fin)?;
        d.sd.fusionner(&sd);
        let o = plaquette_et_rectangle(&fin)?;
        d.p.push(o.plaquette);
        d.r.push(o.rectangle);
        if garder {
            configs.push(ConfigFlot { fichier: f.file_name().unwrap_or_default().to_string_lossy().into_owned(), sd, theta: th.first().cloned() });
        }
        if d.theta.is_empty() {
            d.theta = th;
        } else {
            for (a, b) in d.theta.iter_mut().zip(&th) {
                a.fusionner(b);
            }
        }
    }
    Ok((d, configs))
}

// ------------------------------------------------------------------------------------------
// Estimateurs d'une cellule (somme poolée sur un sous-ensemble de chaînes)
// ------------------------------------------------------------------------------------------

pub fn beta_sd_cellule(ds: &[&DonneesChaine], rapport: RapportBessel, mode: ModeRapport) -> Option<f64> {
    let s = StatsSd::somme(ds.iter().map(|d| &d.sd));
    beta_sd(&s, rapport, mode)
}

pub fn moyenne_observable(ds: &[&DonneesChaine], obs: Observable) -> f64 {
    let (mut somme, mut n) = (0.0, 0usize);
    for d in ds {
        let v = if obs == Observable::Plaquette { &d.p } else { &d.r };
        for x in v {
            somme += x;
        }
        n += v.len();
    }
    somme / n as f64
}

/// β′ par inversion de la courbe (None si hors étalonnage).
pub fn beta_courbe(courbe: &Courbe, obs: Observable, cible: f64) -> Option<(f64, f64, f64)> {
    match courbe.inverser(obs, cible) {
        Ok(Inversion::Dans { beta, pente, sigma_cal, .. }) => Some((beta, pente, sigma_cal)),
        _ => None,
    }
}

fn json_jk(jk: &Option<ResJackknife>) -> Value {
    match jk {
        Some(j) => json!({"erreur": j.erreur, "moyenne_replicats": j.moyenne_replicats, "replicats": j.replicats}),
        None => Value::Null,
    }
}

fn hors_json(courbe: &Courbe, obs: Observable, cible: f64) -> Value {
    match courbe.inverser(obs, cible) {
        Ok(Inversion::HorsEtalonnage { cote_bas, beta_extreme, valeur_extreme, .. }) => json!({
            "etat": "hors_etalonnage", "cote": if cote_bas { "bas" } else { "haut" },
            "beta_extreme": beta_extreme, "valeur_extreme": valeur_extreme, "cible": cible,
            "note": "aucune extrapolation : aucune valeur de β′ n'est rendue"
        }),
        Ok(Inversion::Dans { .. }) => json!({"etat": "dans_la_plage"}),
        Err(e) => json!({"etat": "erreur", "message": e}),
    }
}

/// Bloc JSON d'un estimateur à partir d'une valeur, d'un jackknife et de l'erreur d'étalonnage propagée.
fn bloc_estimateur(valeur: f64, jk: &Option<ResJackknife>, sigma_cal_beta: Option<f64>) -> Value {
    let err_jk = jk.as_ref().map(|j| j.erreur);
    let totale = match (err_jk, sigma_cal_beta) {
        (Some(a), Some(b)) => Some(a.hypot(b)),
        (Some(a), None) => Some(a),
        _ => None,
    };
    json!({
        "etat": "ok", "valeur": valeur, "jackknife": json_jk(jk),
        "erreur_jackknife": err_jk, "sigma_etalonnage": sigma_cal_beta, "erreur_totale": totale,
    })
}

fn a_j(valeur: f64, beta: f64, erreur: Option<f64>) -> Value {
    json!({"a_j": valeur - beta - DELTA_BETA_1, "erreur": erreur, "delta_beta_1": DELTA_BETA_1,
           "note": "quantité cible rapportée, jamais une gate ; aucune lecture physique dans ce lot (WP9 et chantier vision)"})
}

pub struct ParamFlot {
    pub schema: Schema,
    pub pas: usize,
    pub estimateurs: Estimateurs,
    pub rapport: RapportBessel,
    /// Courbes d'étalonnage disponibles (une par côté L′).
    pub courbes: Vec<Courbe>,
    pub fils: usize,
}

/// Document JSON d'une cellule : tous les estimateurs demandés, τ_int, domaine, a_j.
pub fn cellule_json(beta: f64, chaines: &[Chaine], donnees: &[DonneesChaine], par: &ParamFlot) -> Resultat<Value> {
    cellule_json_op(beta, chaines, donnees, par, &Operateur::TroisCentre)
}

/// Comme [`cellule_json`] pour un opérateur de bloc donné (WP22) : le domaine (logarithmes) et `a_j` (schéma de Bałaban) ne sont
/// écrits que pour `3centre`.
pub fn cellule_json_op(beta: f64, chaines: &[Chaine], donnees: &[DonneesChaine], par: &ParamFlot, op: &Operateur) -> Resultat<Value> {
    let balaban = par.schema == Schema::Balaban && op.est_trois_centre();
    let refs: Vec<&DonneesChaine> = donnees.iter().collect();
    let l_fin = donnees[0].l_fin;
    let n_configs: usize = donnees.iter().map(|d| d.p.len()).sum();
    let courbe = par.courbes.iter().find(|c| c.l == l_fin);
    let mut cel = serde_json::Map::new();
    cel.insert("beta".into(), json!(beta));
    cel.insert("l_entree".into(), json!(chaines[0].l));
    cel.insert("l_fin".into(), json!(l_fin));
    cel.insert("n_chaines".into(), json!(chaines.len()));
    cel.insert("n_configs".into(), json!(n_configs));
    cel.insert("chaines".into(), json!(chaines.iter().zip(donnees).map(|(c, d)| json!({
        "nom": c.nom, "n_configs": d.p.len(), "premiere": c.configs.first().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()),
        "derniere": c.configs.last().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()),
        "p_moyenne": d.p.iter().sum::<f64>() / d.p.len() as f64,
        "r_moyenne": d.r.iter().sum::<f64>() / d.r.len() as f64,
    })).collect::<Vec<_>>()));
    cel.insert("p_moyenne".into(), json!(moyenne_observable(&refs, Observable::Plaquette)));
    cel.insert("r_moyenne".into(), json!(moyenne_observable(&refs, Observable::Rectangle)));
    // domaine (Bałaban seulement : le témoin de Swendsen n'a pas de logarithmes)
    if balaban {
        cel.insert("domaine".into(), json!((0..par.pas).map(|k| {
            let mut s = StatsTheta::vide();
            for d in donnees {
                s.fusionner(&d.theta[k]);
            }
            json!({"pas": k + 1, "fraction_hors": s.fraction_hors(), "theta_max": s.theta_max, "n_ambigues": s.n_ambigues,
                   "n_boucles": s.n, "etiquette": domaine::etiquette(&s)})
        }).collect::<Vec<_>>()));
    }
    // τ_int de la série de plaquette de production (pas) et des plaquettes bloquées (indicatif : n petit)
    let taus: Vec<Value> = chaines.iter().zip(donnees).map(|(c, d)| {
        let prod = serie_plaquette(&c.dossier).map(|s| (stat_serie(&s), s.len()));
        let blq = tau_int(&d.p, 6.0);
        json!({
            "chaine": c.nom,
            "plaquette_production": prod.map(|(s, n)| json!({"tau": s.tau, "fenetre": s.fenetre, "dtau": s.dtau, "n": n, "err": s.err})),
            "plaquette_bloquee_indicatif": {"tau": blq.tau, "fenetre": blq.fenetre, "dtau": blq.dtau, "n": d.p.len(),
                                            "note": "série courte (une valeur par configuration gardée) : indicatif"},
            "espacement_pas": c.dump_every,
        })
    }).collect();
    cel.insert("tau_int".into(), json!({"fenetre_sokal_c": 6.0, "chaines": taus}));
    let mut est = serde_json::Map::new();
    let est_cfg = par.estimateurs;
    if est_cfg.sd {
        let valeur = beta_sd_cellule(&refs, par.rapport, ModeRapport::Exact);
        let jk = jackknife_groupes(donnees, |g| beta_sd_cellule(g, par.rapport, ModeRapport::Exact));
        est.insert("sd".into(), match valeur {
            Some(v) => {
                let mut b = bloc_estimateur(v, &jk, None);
                if balaban && par.pas == 1 {
                    b["a_j"] = a_j(v, beta, jk.as_ref().map(|j| j.erreur));
                }
                b
            }
            None => json!({"etat": "indefini", "note": "rhs(50) < somme des w : « ∞ » du script"}),
        });
    }
    let absent = |raison: String| json!({"etat": "hors_etalonnage", "note": raison});
    if est_cfg.plaq || est_cfg.mcrg2 {
        match courbe {
            None => {
                let r = format!("pas de courbe d'étalonnage de côté L = {l_fin} fournie (aucune extrapolation, aucune courbe inventée)");
                if est_cfg.plaq {
                    est.insert("plaq".into(), absent(r.clone()));
                }
                if est_cfg.mcrg2 {
                    est.insert("mcrg2".into(), absent(r));
                }
            }
            Some(courbe) => {
                let pbar = moyenne_observable(&refs, Observable::Plaquette);
                let rbar = moyenne_observable(&refs, Observable::Rectangle);
                let bp = |g: &[&DonneesChaine]| beta_courbe(courbe, Observable::Plaquette, moyenne_observable(g, Observable::Plaquette)).map(|x| x.0);
                let br = |g: &[&DonneesChaine]| beta_courbe(courbe, Observable::Rectangle, moyenne_observable(g, Observable::Rectangle)).map(|x| x.0);
                let inv_p = beta_courbe(courbe, Observable::Plaquette, pbar);
                let inv_r = beta_courbe(courbe, Observable::Rectangle, rbar);
                if est_cfg.plaq {
                    est.insert("plaq".into(), match inv_p {
                        Some((v, pente, scal)) => {
                            let jk = jackknife_groupes(donnees, bp);
                            let mut b = bloc_estimateur(v, &jk, Some(scal / pente.abs()));
                            b["p_cible"] = json!(pbar);
                            b["pente"] = json!(pente);
                            b["interpolation_lineaire_info"] = json!(courbe.inverser_lineaire(Observable::Plaquette, pbar).ok().flatten());
                            if balaban && par.pas == 1 {
                                b["a_j"] = a_j(v, beta, b["erreur_totale"].as_f64());
                            }
                            b
                        }
                        None => hors_json(courbe, Observable::Plaquette, pbar),
                    });
                }
                if est_cfg.mcrg2 {
                    est.insert("mcrg2".into(), match (inv_p, inv_r) {
                        (Some((vp, pp, sp)), Some((vr, pr, sr))) => {
                            let moyenne = 0.5 * (vp + vr);
                            let jk_m = jackknife_groupes(donnees, |g| Some(0.5 * (bp(g)? + br(g)?)));
                            let jk_e = jackknife_groupes(donnees, |g| Some(br(g)? - bp(g)?));
                            let sig_cal = 0.5 * ((sp / pp.abs()).powi(2) + (sr / pr.abs()).powi(2)).sqrt();
                            let mut b = bloc_estimateur(moyenne, &jk_m, Some(sig_cal));
                            b["beta_p"] = json!(vp);
                            b["beta_r"] = json!(vr);
                            b["ecart_schema"] = json!({"valeur": (vr - vp).abs(), "signe_r_moins_p": vr - vp, "jackknife": json_jk(&jk_e),
                                "note": "systématique de schéma rapportée, jamais un seuil"});
                            if balaban && par.pas == 1 {
                                b["a_j"] = a_j(moyenne, beta, b["erreur_totale"].as_f64());
                                b["a_j_p"] = a_j(vp, beta, None);
                                b["a_j_r"] = a_j(vr, beta, None);
                            }
                            b
                        }
                        _ => json!({"etat": "hors_etalonnage", "plaquette": hors_json(courbe, Observable::Plaquette, pbar),
                                    "rectangle": hors_json(courbe, Observable::Rectangle, rbar)}),
                    });
                }
            }
        }
    }
    cel.insert("estimateurs".into(), Value::Object(est));
    Ok(Value::Object(cel))
}

/// Flot complet d'un ensemble : un document `kvu-rg-flot/1` (sans durée ni nombre de fils : identique à l'octet quel que soit `fils`).
pub fn flot(ens: &Path, filtre: Option<&str>, par: &ParamFlot, fichiers_courbes: &[FichierCourbe]) -> Resultat<Value> {
    flot_options(ens, filtre, par, fichiers_courbes, &OptionsFlot::default())
}

/// Extensions facultatives WP9 ; le chemin historique conserve ses sommes et son JSON.
pub fn flot_options(ens: &Path, filtre: Option<&str>, par: &ParamFlot, fichiers_courbes: &[FichierCourbe], options: &OptionsFlot) -> Resultat<Value> {
    options.verifier(par.schema)?;
    if par.pas == 0 {
        return Err(Erreur::Config("--pas doit être ≥ 1".into()));
    }
    let calcul = crate::rg::cli::ouvrir_calcul_nom(&options.backend)?;
    let gpu = calcul.bloc(Geometrie::TroisCentre)?;
    let chaines = decouvrir(ens, filtre)?;
    let groupes = grouper_par_beta(chaines);
    let mut cellules = Vec::new();
    for (beta, chs) in &groupes {
        let mut donnees = Vec::new();
        let mut configs = Vec::new();
        for c in chs {
            let (d, cs) = traiter_chaine_avec(c, par.schema, par.pas, par.fils, gpu.as_ref(), !options.tailles_bins().is_empty() || options.detail_configs, &options.operateur)?;
            donnees.push(d);
            configs.push(cs);
        }
        let mut cellule = cellule_json_op(*beta, chs, &donnees, par, &options.operateur)?;
        ajouter_options(&mut cellule, &donnees, &configs, par, options);
        cellules.push(cellule);
    }
    let mut doc = json!({
        "format": SCHEMA_FLOT,
        "ensemble": ens.display().to_string(),
        "filtre": filtre,
        "schema": par.schema.nom(),
        "facteur_de_bloc": par.schema.facteur(),
        "pas": par.pas,
        "estimateurs": {"sd": par.estimateurs.sd, "plaq": par.estimateurs.plaq, "mcrg2": par.estimateurs.mcrg2},
        "sd": {"rapport": match par.rapport { RapportBessel::I2SurI1 => "I2/I1 (SU(2))", RapportBessel::I1SurI0 => "I1/I0 (MUTANT U(1))" },
               "mode": "exact (fraction continue)", "bissection": "sur [0, 50]"},
        "courbes": par.courbes.iter().zip(fichiers_courbes).map(|(c, (f, sha))| json!({
            "l": c.l, "fichier": f.display().to_string(), "sha256": sha, "points": c.points.len(),
            "beta_min": c.points[0].beta, "beta_max": c.points[c.points.len() - 1].beta,
            "interpolation": "PCHIP monotone (Fritsch-Carlson), aucune extrapolation"})).collect::<Vec<_>>(),
        "jackknife": "suppression d'une chaîne ; σ² = (C−1)/C Σ(θ_c − θ̄)² ; estimateurs recalculés à chaque réplicat",
        "a_j_definition": "a_j = β′ − β − 0,4081 (SU(2), L_b = 3, Bałaban, un pas) ; quantité cible, jamais une gate",
        "cellules": cellules,
        "version": crate::cli::version_texte().replace('\n', " | "),
    });
    if let Some(g) = gpu {
        let p = g.provenance();
        doc["calcul"] = json!({"backend": p.backend.nom(), "arch": p.arch,
            "source_sha256": p.sha256_source, "code_objet_sha256": p.sha256_code});
    }
    if !options.operateur.est_trois_centre() {
        let mut o = options.operateur.parametres_json();
        o["hors_famille"] = json!(options.hors_famille);
        o["note"] = json!("WP22 : opérateur de chaque pas de bloc ; ni domaine ni a_j (définis pour 3centre seulement)");
        doc["operateur"] = o;
    }
    Ok(doc)
}

/// Lit des courbes d'étalonnage (JSON `kvu-rg-etalonnage/1`) : (courbe, (chemin, sha256)).
/// Fichier de courbe lu : (chemin, sha256).
pub type FichierCourbe = (PathBuf, String);

pub fn lire_courbes(fichiers: &[PathBuf]) -> Resultat<(Vec<Courbe>, Vec<FichierCourbe>)> {
    let (mut cs, mut hs) = (Vec::new(), Vec::new());
    for f in fichiers {
        let texte = std::fs::read_to_string(f).map_err(|e| Erreur::io(format!("lecture {}", f.display()), e))?;
        let v: Value = serde_json::from_str(&texte).map_err(|e| Erreur::Format(format!("{} : {e}", f.display())))?;
        let (par, chaines) = depuis_json(&v)?;
        if par.est_milieux() || v.get("milieux").and_then(Value::as_bool) == Some(true) {
            return Err(Erreur::Config(format!("{} : étalonnage milieux, inutilisable comme courbe", f.display())));
        }
        if par.independante || par.facteur_beta != 1.0 {
            return Err(Erreur::Config(format!("{} : étalonnage indépendant ou mutant, inutilisable comme courbe", f.display())));
        }
        cs.push(courbe_depuis(par.l, &chaines)?);
        hs.push((f.clone(), crate::heritage::manifest::sha256_hex(texte.as_bytes())));
    }
    Ok((cs, hs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motifs() {
        assert!(correspond("L12_b2.3_c*", "L12_b2.3_c1"));
        assert!(!correspond("L12_b2.3_c*", "L12_b2.5_c1"));
        assert!(correspond("*", "x"));
        assert!(correspond("L12_*_c1", "L12_b8_c1"));
        assert!(!correspond("L12_*_c1", "L12_b8_c2"));
        assert!(correspond("exact", "exact") && !correspond("exact", "exacte"));
    }

    #[test]
    fn analyse_des_estimateurs() {
        let e = Estimateurs::analyser("sd, plaq").unwrap();
        assert!(e.sd && e.plaq && !e.mcrg2);
        assert!(Estimateurs::analyser("sd,xx").is_err());
        assert!(Estimateurs::analyser("").is_err());
        assert!(Schema::analyser("autre").is_err());
    }

    #[test]
    fn regroupement_par_beta() {
        let mk = |n: &str, b: f64| Chaine { nom: n.into(), dossier: PathBuf::new(), beta: b, l: 12, configs: vec![], dump_every: None };
        let g = grouper_par_beta(vec![mk("a", 8.0), mk("b", 2.3), mk("c", 8.0), mk("d", 2.3)]);
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].0, 2.3);
        assert_eq!(g[1].1.len(), 2);
    }
}
