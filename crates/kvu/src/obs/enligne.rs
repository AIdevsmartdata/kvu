//! Mesures en ligne pendant la génération (implémentations de su2::run::MesureEnLigne) et
//! analyse de la spécification « --mesures ». Propriétaire : T7.
//!
//! Grammaire : éléments séparés par des virgules, champs séparés par `:`.
//!
//! | élément | effet |
//! |---|---|
//! | `wilson:R<r>xT<t>[:toutes]:<période>` | ajoute à `<sortie>/wilson.csv` (`pas,r,t,W`), r ≤ R, t ≤ T |
//! | `polyakov_corr:<période>` | ajoute à `<sortie>/polyakov_corr.csv` (`pas,r,C`), r = 0..L/2 |
//! | `flot:<dt>:<tmax>:<période>[:cible=0.3][:chaque=N]` | ajoute à `<sortie>/flot.csv` (`pas,t,E_plaq,E_trefle,t2E,Q`) et `flot_t0.csv` (`pas,t0,w0`) |
//! | `bloc:3:<période>` | bloc de Bałaban L_b = 3 (WP7, `rg::en_ligne`) : écrit `bloc3_cfg_<pas>.npy` (q64) et ajoute à `bloc3.csv`, `bloc3_hist.csv` |
//!
//! `<période>` est en pas de production (0 = jamais). `toutes` moyenne sur les 12 orientations
//! (sinon T selon la direction 3). `cible=` fixe à la fois la cible de t²E (t₀) et celle de W (w₀) ;
//! `cible_t2e=` et `cible_w=` les fixent séparément. `chaque=N` mesure E(t) tous les N pas du flot.
//!
//! `sortie` (argument de `MesureEnLigne::mesurer`) est le répertoire de la chaîne, `<outdir>/c<k>`.
//! Les écritures se font en ajout : l'en-tête n'est écrit qu'à la création du fichier, et la garde SSD
//! est consultée AVANT chaque ajout.

use super::flot::{flot_wilson_avec, CourbeFlot, ParamsFlot};
use super::gpu::ModuleObs;
use super::wilson::{boucles_wilson_avec, correlateur_polyakov_avec, ParamsWilson, TableWilson};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::gpu::Appareil;
use crate::heritage::su2_hb::g17;
use crate::stockage::GardeSsd;
use crate::su2::run::MesureEnLigne;
use crate::su2::ChampGpu;
use std::io::Write;
use std::path::Path;

/// Un élément de la spécification, avant construction des mesures.
#[derive(Clone, Debug, PartialEq)]
pub enum ElementMesure {
    Wilson { params: ParamsWilson, periode: usize },
    PolyakovCorr { dir_temps: usize, periode: usize },
    Flot { params: ParamsFlot, periode: usize },
    Bloc { periode: usize },
}

impl ElementMesure {
    /// Nom de la mesure (celui du type dans la grammaire).
    pub fn nom(&self) -> &'static str {
        match self {
            ElementMesure::Wilson { .. } => "wilson",
            ElementMesure::PolyakovCorr { .. } => "polyakov_corr",
            ElementMesure::Flot { .. } => "flot",
            ElementMesure::Bloc { .. } => "bloc",
        }
    }
}

fn erreur_spec(element: &str, raison: &str) -> Erreur {
    Erreur::Config(format!(
        "--mesures : élément « {element} » invalide ({raison}). Grammaire : wilson:R<r>xT<t>[:toutes]:<période> | \
         polyakov_corr:<période> | flot:<dt>:<tmax>:<période>[:cible=0.3][:chaque=N] | bloc:3:<période>"
    ))
}

fn periode(champ: &str, element: &str) -> Resultat<usize> {
    champ
        .parse::<usize>()
        .map_err(|_| erreur_spec(element, &format!("période « {champ} » : entier ≥ 0 attendu")))
}

fn reel(champ: &str, element: &str, nom: &str) -> Resultat<f64> {
    match champ.parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err(erreur_spec(element, &format!("{nom} « {champ} » : réel fini attendu"))),
    }
}

/// « R4xT6 » → (4, 6).
fn rxt(champ: &str, element: &str) -> Resultat<(usize, usize)> {
    let corps = champ.strip_prefix('R').or_else(|| champ.strip_prefix('r'));
    let (r, t) = corps
        .and_then(|c| c.split_once(['x', 'X']))
        .and_then(|(r, t)| {
            Some((
                r.parse::<usize>().ok()?,
                t.strip_prefix(['T', 't'])?.parse::<usize>().ok()?,
            ))
        })
        .ok_or_else(|| erreur_spec(element, &format!("« {champ} » : forme R<r>xT<t> attendue")))?;
    if r == 0 || t == 0 {
        return Err(erreur_spec(element, "R et T doivent être ≥ 1"));
    }
    Ok((r, t))
}

fn analyser_element(element: &str) -> Resultat<ElementMesure> {
    let champs: Vec<&str> = element.split(':').map(str::trim).collect();
    match champs[0] {
        "wilson" => {
            let (r_max, t_max) = rxt(champs.get(1).copied().unwrap_or(""), element)?;
            let (toutes, reste) = match champs.get(2) {
                Some(&"toutes") => (true, &champs[3..]),
                _ => (false, &champs[2..]),
            };
            if reste.len() != 1 {
                return Err(erreur_spec(element, "période manquante ou champs en trop"));
            }
            Ok(ElementMesure::Wilson {
                params: ParamsWilson {
                    r_max,
                    t_max,
                    dir_temps: 3,
                    toutes_orientations: toutes,
                },
                periode: periode(reste[0], element)?,
            })
        }
        "polyakov_corr" => {
            if champs.len() != 2 {
                return Err(erreur_spec(element, "forme polyakov_corr:<période>"));
            }
            Ok(ElementMesure::PolyakovCorr {
                dir_temps: 3,
                periode: periode(champs[1], element)?,
            })
        }
        "flot" => {
            if champs.len() < 4 {
                return Err(erreur_spec(element, "forme flot:<dt>:<tmax>:<période>[:cible=...]"));
            }
            let mut params = ParamsFlot {
                dt: reel(champs[1], element, "dt")?,
                t_max: reel(champs[2], element, "tmax")?,
                ..ParamsFlot::default()
            };
            if params.dt <= 0.0 || params.t_max < params.dt {
                return Err(erreur_spec(element, "il faut 0 < dt ≤ tmax"));
            }
            for option in &champs[4..] {
                let (cle, val) = option
                    .split_once('=')
                    .ok_or_else(|| erreur_spec(element, &format!("option « {option} »")))?;
                match cle {
                    "cible" => {
                        let v = reel(val, element, "cible")?;
                        params.cible_t2e = v;
                        params.cible_w = v;
                    }
                    "cible_t2e" => params.cible_t2e = reel(val, element, "cible_t2e")?,
                    "cible_w" => params.cible_w = reel(val, element, "cible_w")?,
                    "chaque" => {
                        params.mesure_chaque = val
                            .parse::<usize>()
                            .ok()
                            .filter(|&n| n >= 1)
                            .ok_or_else(|| erreur_spec(element, "chaque : entier ≥ 1 attendu"))?
                    }
                    autre => return Err(erreur_spec(element, &format!("option inconnue « {autre} »"))),
                }
            }
            Ok(ElementMesure::Flot {
                params,
                periode: periode(champs[3], element)?,
            })
        }
        "bloc" => Ok(ElementMesure::Bloc {
            periode: crate::rg::en_ligne::analyser_element(&champs, element)?,
        }),
        autre => Err(erreur_spec(element, &format!("type « {autre} » inconnu"))),
    }
}

/// Analyse la spécification en éléments typés. Un type ne peut apparaître qu'une fois (il écrit des
/// fichiers à nom fixe : deux paramétrages se mélangeraient).
pub fn decrire(spec: &str) -> Resultat<Vec<ElementMesure>> {
    let mut out: Vec<ElementMesure> = Vec::new();
    for element in spec.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        let e = analyser_element(element)?;
        if out.iter().any(|o| o.nom() == e.nom()) {
            return Err(erreur_spec(
                element,
                &format!("« {} » déjà présent : un seul élément par type", e.nom()),
            ));
        }
        out.push(e);
    }
    Ok(out)
}

/// « wilson:R4xT4:10,polyakov_corr:10,flot:0.01:3.0:20 » → mesures (type:paramètres:période),
/// protégées par la garde SSD de la machine.
pub fn analyser(spec: &str) -> Resultat<Vec<Box<dyn MesureEnLigne>>> {
    analyser_avec(spec, Some(GardeSsd::machine()))
}

/// Comme `analyser`, avec une garde SSD explicite (`None` : aucune garde, pour les tests).
pub fn analyser_avec(spec: &str, garde: Option<GardeSsd>) -> Resultat<Vec<Box<dyn MesureEnLigne>>> {
    Ok(decrire(spec)?
        .into_iter()
        .map(|e| -> Box<dyn MesureEnLigne> {
            let sortie = Ecrivain { garde: garde.clone() };
            match e {
                ElementMesure::Wilson { params, periode } => Box::new(MesureWilson {
                    params,
                    periode,
                    sortie,
                    module: None,
                }),
                ElementMesure::PolyakovCorr { dir_temps, periode } => Box::new(MesurePolyakovCorr {
                    dir_temps,
                    periode,
                    sortie,
                    module: None,
                }),
                ElementMesure::Flot { params, periode } => Box::new(MesureFlot {
                    params,
                    periode,
                    sortie,
                    module: None,
                }),
                ElementMesure::Bloc { periode } => Box::new(crate::rg::en_ligne::MesureBloc::nouvelle(periode, garde.clone())),
            }
        })
        .collect())
}

// ------------------------------------------------------------------------------------------
// Écriture en ajout, protégée par la garde SSD
// ------------------------------------------------------------------------------------------

pub(crate) struct Ecrivain {
    garde: Option<GardeSsd>,
}

impl Ecrivain {
    pub(crate) fn nouveau(garde: Option<GardeSsd>) -> Self {
        Ecrivain { garde }
    }

    /// Ajoute `lignes` (terminées par '\n') à `chemin`, en écrivant `entete` si le fichier est neuf ou vide.
    pub(crate) fn ajouter(&self, chemin: &Path, entete: &str, lignes: &str) -> Resultat<()> {
        let io = |contexte: &str| {
            let c = format!("{contexte} {}", chemin.display());
            move |source| Erreur::Io { contexte: c, source }
        };
        if let Some(parent) = chemin.parent() {
            std::fs::create_dir_all(parent).map_err(io("création du répertoire de"))?;
        }
        let neuf = std::fs::metadata(chemin).map(|m| m.len() == 0).unwrap_or(true);
        let mut texte = String::new();
        if neuf {
            texte.push_str(entete);
            texte.push('\n');
        }
        texte.push_str(lignes);
        if let Some(g) = &self.garde {
            g.verifier(chemin, texte.len() as u64)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(chemin)
            .map_err(io("ouverture de"))?;
        f.write_all(texte.as_bytes()).map_err(io("écriture dans"))
    }
}

/// Module GPU compilé une fois (au premier champ résident rencontré) puis réutilisé.
fn module_paresseux<'a>(
    cache: &'a mut Option<ModuleObs>,
    app: Option<&Appareil>,
    champ: Option<&ChampGpu>,
) -> Resultat<Option<&'a ModuleObs>> {
    if champ.is_none() {
        return Ok(None);
    }
    if cache.is_none() {
        let app = app.ok_or_else(|| Erreur::Config("champ GPU fourni sans appareil".into()))?;
        *cache = Some(ModuleObs::nouveau(app)?);
    }
    Ok(cache.as_ref())
}

// ------------------------------------------------------------------------------------------
// Les trois mesures
// ------------------------------------------------------------------------------------------

/// Lignes CSV `pas,r,t,W` d'une table de Wilson.
fn lignes_wilson(pas: u64, t: &TableWilson) -> String {
    let mut s = String::new();
    for r in 1..=t.r_max {
        for tt in 1..=t.t_max {
            s.push_str(&format!("{pas},{r},{tt},{}\n", g17(t.get(r, tt))));
        }
    }
    s
}

/// Lignes CSV `pas,r,C`.
fn lignes_polyakov(pas: u64, c: &[f64]) -> String {
    c.iter()
        .enumerate()
        .map(|(r, v)| format!("{pas},{r},{}\n", g17(*v)))
        .collect()
}

/// Lignes de `flot.csv` (une par échantillon) et de `flot_t0.csv` (une seule).
fn lignes_flot(pas: u64, c: &CourbeFlot) -> (String, String) {
    let mut flot = String::new();
    for i in 0..c.t.len() {
        flot.push_str(&format!(
            "{pas},{},{},{},{},{}\n",
            g17(c.t[i]),
            g17(c.e_plaq[i]),
            g17(c.e_trefle[i]),
            g17(c.t2e[i]),
            g17(c.q_top[i])
        ));
    }
    let opt = |x: Option<f64>| x.map(g17).unwrap_or_else(|| "nan".to_string());
    (flot, format!("{pas},{},{}\n", opt(c.t0), opt(c.w0)))
}

struct MesureWilson {
    params: ParamsWilson,
    periode: usize,
    sortie: Ecrivain,
    module: Option<ModuleObs>,
}

impl MesureEnLigne for MesureWilson {
    fn nom(&self) -> &str {
        "wilson"
    }
    fn periode(&self) -> usize {
        self.periode
    }
    fn mesurer(
        &mut self,
        app: Option<&Appareil>,
        champ: Option<&ChampGpu>,
        hote: Option<&ChampHote>,
        chaine: usize,
        pas: u64,
        sortie: &Path,
    ) -> Resultat<()> {
        let m = module_paresseux(&mut self.module, app, champ)?;
        let t = boucles_wilson_avec(app, m, champ, chaine, hote, &self.params)?;
        self.sortie
            .ajouter(&sortie.join("wilson.csv"), "pas,r,t,W", &lignes_wilson(pas, &t))
    }
}

struct MesurePolyakovCorr {
    dir_temps: usize,
    periode: usize,
    sortie: Ecrivain,
    module: Option<ModuleObs>,
}

impl MesureEnLigne for MesurePolyakovCorr {
    fn nom(&self) -> &str {
        "polyakov_corr"
    }
    fn periode(&self) -> usize {
        self.periode
    }
    fn mesurer(
        &mut self,
        app: Option<&Appareil>,
        champ: Option<&ChampGpu>,
        hote: Option<&ChampHote>,
        chaine: usize,
        pas: u64,
        sortie: &Path,
    ) -> Resultat<()> {
        let m = module_paresseux(&mut self.module, app, champ)?;
        let c = correlateur_polyakov_avec(app, m, champ, chaine, hote, self.dir_temps)?;
        self.sortie
            .ajouter(&sortie.join("polyakov_corr.csv"), "pas,r,C", &lignes_polyakov(pas, &c))
    }
}

struct MesureFlot {
    params: ParamsFlot,
    periode: usize,
    sortie: Ecrivain,
    module: Option<ModuleObs>,
}

impl MesureEnLigne for MesureFlot {
    fn nom(&self) -> &str {
        "flot"
    }
    fn periode(&self) -> usize {
        self.periode
    }
    fn mesurer(
        &mut self,
        app: Option<&Appareil>,
        champ: Option<&ChampGpu>,
        hote: Option<&ChampHote>,
        chaine: usize,
        pas: u64,
        sortie: &Path,
    ) -> Resultat<()> {
        let m = module_paresseux(&mut self.module, app, champ)?;
        let c = flot_wilson_avec(app, m, champ, chaine, hote, &self.params)?;
        let (flot, t0) = lignes_flot(pas, &c);
        self.sortie
            .ajouter(&sortie.join("flot.csv"), "pas,t,E_plaq,E_trefle,t2E,Q", &flot)?;
        self.sortie.ajouter(&sortie.join("flot_t0.csv"), "pas,t0,w0", &t0)
    }
}
