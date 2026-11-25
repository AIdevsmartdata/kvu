//! CLI du module tube : `su2 gen --cotes l0,l1,l2,l3 …` (chemin CPU F64 anisotrope), `su2 tube smin` (action minimale par
//! refroidissement), `su2 tube apparie` (secteurs appariés avec et sans torsion dans le plan long : ⟨V⟩₁ − ⟨V⟩₀).

use super::domaine;
use super::obs::{self, moyenne_erreur, ParamObs};
use super::reseau::{indice_plan, pile_aniso, ChaineTube, GeoA, Pile};
use crate::commun::{Erreur, Resultat};
use crate::geom::{self, ChampGeo, GeoSpec};
use crate::heritage::su2_hb::{g17, Q};
use crate::stockage::npy::{ecrire_champ_geo, FormatChamp};
use crate::stockage::provenance::{ecrire_manifeste_fichier, Manifeste};
use crate::stockage::GardeSsd;
use crate::su2::torsion::Torsion;
use clap::parser::ValueSource;
use clap::{Arg, ArgMatches, Command};
use rayon::prelude::*;
use serde_json::json;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

fn texte<'a>(m: &'a ArgMatches, nom: &str) -> Option<&'a str> {
    m.try_get_one::<String>(nom)
        .ok()
        .flatten()
        .map(|s| s.as_str())
}

fn io(ctx: &str, p: &Path, e: std::io::Error) -> Erreur {
    Erreur::Io {
        contexte: format!("{ctx} {}", p.display()),
        source: e,
    }
}

/// Mutants (contrôles négatifs des portes, jamais en production).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutant {
    Aucun,
    /// classes de couleur du bain thermique en ordre inverse (porte de non-régression)
    OrdreCouleurs,
    /// pile du plan long privée d'une position de fibre (z = +1 en (ℓ0−1, ℓ1−1, 0, 0))
    PileIncomplete,
    /// signe du plan long appliqué au plan (0,2) au lieu de (0,1)
    MauvaisPlan,
}

impl Mutant {
    pub fn analyser(t: Option<&str>) -> Resultat<Mutant> {
        match t {
            None | Some("aucun") => Ok(Mutant::Aucun),
            Some("ordre-couleurs") => Ok(Mutant::OrdreCouleurs),
            Some("pile-incomplete") => Ok(Mutant::PileIncomplete),
            Some("mauvais-plan") => Ok(Mutant::MauvaisPlan),
            Some(a) => Err(Erreur::Config(format!(
                "--mutant « {a} » inconnu (ordre-couleurs|pile-incomplete|mauvais-plan)"
            ))),
        }
    }
}

/// Pile de la torsion `t`, modifiée par le mutant.
pub fn pile_mutee(g: &GeoA, t: &Torsion, m: Mutant) -> Resultat<Pile> {
    match m {
        Mutant::Aucun | Mutant::OrdreCouleurs => Ok(Pile::depuis_torsion(g, t)),
        Mutant::PileIncomplete => {
            if !t.tordu(0, 1) {
                return Err(Erreur::Config(
                    "mutant pile-incomplete : le plan 01 doit être tordu".into(),
                ));
            }
            let mut p = Pile::depuis_torsion(g, t);
            p.retourner(indice_plan(0, 1), g.site(&[g.d[0] - 1, g.d[1] - 1, 0, 0]));
            Ok(p)
        }
        Mutant::MauvaisPlan => {
            if !t.tordu(0, 1) || t.tordu(0, 2) {
                return Err(Erreur::Config(
                    "mutant mauvais-plan : plan 01 tordu et 02 non tordu requis".into(),
                ));
            }
            let txt: Vec<String> = t
                .texte()
                .split(',')
                .map(|p| {
                    if p == "01" {
                        "02".to_string()
                    } else {
                        p.to_string()
                    }
                })
                .collect();
            Ok(Pile::depuis_torsion(g, &Torsion::analyser(&txt.join(","))?))
        }
    }
}

/// Torsion consignée pour un mutant (pile incomplète : la torsion nominale, le mutant étant consigné à part).
fn pile_torsion(t: &Torsion, m: Mutant) -> Torsion {
    if m == Mutant::MauvaisPlan {
        let txt: Vec<String> = t
            .texte()
            .split(',')
            .map(|p| {
                if p == "01" {
                    "02".to_string()
                } else {
                    p.to_string()
                }
            })
            .collect();
        Torsion::analyser(&txt.join(",")).unwrap_or(*t)
    } else {
        *t
    }
}

// ------------------------------------------------------------------------------------------
// Fichiers
// ------------------------------------------------------------------------------------------

/// NPY c16 de forme (4, ℓ0, ℓ1, ℓ2, ℓ3, 2, 2) par `stockage::npy::ecrire_champ_geo` (socle `geom` : en-tête et octets de
/// `su2_hb::write_npy_quat` à côtés égaux ; écriture atomique, garde SSD, sha256).
pub fn ecrire_npy(chemin: &Path, spec: &GeoSpec, u: &[Q]) -> Resultat<()> {
    let champ = ChampGeo {
        geo: spec.clone(),
        q: u.iter().flat_map(|q| q.iter().copied()).collect(),
    };
    ecrire_champ_geo(chemin, &champ, FormatChamp::C16, &GardeSsd::machine())?;
    Ok(())
}

fn creer(p: &Path) -> Resultat<BufWriter<File>> {
    Ok(BufWriter::new(
        File::create(p).map_err(|e| io("création", p, e))?,
    ))
}

fn ecrire(f: &mut BufWriter<File>, p: &Path, l: &str) -> Resultat<()> {
    writeln!(f, "{l}").map_err(|e| io("écriture", p, e))
}

// ------------------------------------------------------------------------------------------
// su2 gen --cotes
// ------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DepartT {
    Chaud,
    Froid,
    Vide,
}

struct ParamGen {
    geo: GeoA,
    therm: u64,
    nmeas: u64,
    skip: u64,
    nor: usize,
    dump_every: u64,
    dump_initial: bool,
    obs: Option<ParamObs>,
    obs_every: u64,
    mutant: Mutant,
    /// `--xi0` (réseau anisotrope, direction du temps 3) ; None : isotrope, sorties inchangées au bit.
    xi0: Option<f64>,
}

/// Spécification géométrique du tube (socle `geom`, schéma kvu-geometrie/1) : côtés `g.d`, bords périodiques, torsion `t`,
/// direction du temps TOUJOURS explicite : 3, convention historique de `su2 gen` (disposition « temporal = 3 ») ; le tube ne
/// distingue pas de temps (directions longues 0, 1 ; fibre 2, 3).
pub fn spec_tube(g: &GeoA, t: &Torsion) -> GeoSpec {
    GeoSpec::anisotrope(g.d, 3).avec_torsion(*t)
}

/// Ce que rend le refroidissement (`smin`, départ « minimum » d'`apparie`) : un CANDIDAT minimum.
pub const NATURE_CANDIDAT: &str = "candidat minimum : configuration refroidie (U ← A†/|A|) jusqu'à max|∇S/β| < gtol si la \
     convergence est atteinte ; un gradient nul ne certifie pas la minimalité (point stationnaire : minimum local ou selle)";

/// Tolérance du refroidissement du départ « minimum » d'`apparie` (secteur 1).
const GTOL_APPARIE: f64 = 1e-10;

/// Moteur consigné dans les manifestes du module.
pub const MOTEUR: &str = "kvu su2 tube (tore anisotrope, CPU F64)";

/// Moteur consigné par `su2 gen --cotes --backend hip|cuda` (moteur de la campagne « une boîte », `su2::une_boite`).
pub const MOTEUR_GPU: &str = "kvu su2 tube (tore anisotrope, carte F64, su2::une_boite)";

/// Étiquette EXPÉRIMENTALE (mêmes mots que le module intrication) : ligne écrite sur stderr au lancement de
/// `su2 gen --cotes`, `su2 tube smin` et `su2 tube apparie`.
pub const AVERTISSEMENT_EXPERIMENTAL: &str =
    "module expérimental : interface et formats susceptibles de changer ; résultats à confirmer indépendamment";

/// `parametres.statut` des manifestes du module (manifeste de run et manifeste de fichier de `smin --cfg`).
pub const STATUT: &str = "experimental";

fn annoncer() {
    eprintln!("kvu su2 tube : {AVERTISSEMENT_EXPERIMENTAL}");
}

fn parametres() -> serde_json::Value {
    json!({ "statut": STATUT })
}

/// Exécute `corps` sous le manifeste ouvert `man` : succès ⇒ `terminer(0)` (COMPLETE en dernier) ; TOUTE erreur ⇒
/// `echouer(code, raison)` (statut « echec », code et raison consignés, jamais de COMPLETE), puis l'erreur est rendue.
pub fn sous_manifeste<T>(
    mut man: Manifeste,
    corps: impl FnOnce(&mut Manifeste) -> Resultat<T>,
) -> Resultat<T> {
    match corps(&mut man) {
        Ok(v) => {
            man.terminer(0)?;
            Ok(v)
        }
        Err(e) => {
            let _ = man.echouer(e.code_sortie(), &e.to_string());
            Err(e)
        }
    }
}

/// Consigne dans le manifeste tous les fichiers réguliers de `dir` et de ses sous-répertoires (hors manifest.json, COMPLETE).
fn consigner_fichiers(man: &mut Manifeste, dir: &Path) -> Resultat<()> {
    let mut a_voir = vec![dir.to_path_buf()];
    while let Some(d) = a_voir.pop() {
        let mut entrees: Vec<PathBuf> = fs::read_dir(&d)
            .map_err(|e| io("lecture", &d, e))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        entrees.sort();
        for p in entrees {
            if p.is_dir() {
                a_voir.push(p);
            } else if p
                .file_name()
                .is_some_and(|n| n != "manifest.json" && n != "COMPLETE")
            {
                man.fichier(&p)?;
            }
        }
    }
    Ok(())
}

/// Seule option ajoutée à `su2 gen` : `--cotes`. Les observables du tube passent par `--mesures tube:ab:R:T[:N]`, les mutants
/// (contrôles négatifs) par `--mutant`.
pub fn args_gen(c: Command) -> Command {
    args_xi0(c).arg(Arg::new("cotes").long("cotes").num_args(1).help(
        "[EXPÉRIMENTAL] tube : quatre côtés l0,l1,l2,l3 (F64 anisotrope, --backend cpu|hip|cuda explicite ; --start hot|cold|vide ; \
         --mesures tube:ab:RMAX:TMAX[:N] ; therm + nmeas·skip ≤ 2³² − 1 balayages par chaîne ; au plus 4096 chaînes de graines \
         distinctes ; --L, --options et les options du bord refusées ; non reprenable ; domaine admis de chaque option : \
         kvu su2 tube --help)",
    ))
}

/// `--xi0` de `su2 gen --cotes` : réseau anisotrope (direction du temps 3), éteint par défaut.
pub fn args_xi0(c: Command) -> Command {
    c.arg(Arg::new("xi0").long("xi0").num_args(1).help(
        "[avec --cotes] réseau anisotrope : plaquettes contenant l'axe 3 pondérées ξ₀, les autres 1/ξ₀ (S = (β/ξ₀) Σ_spatiales + β ξ₀ Σ_temporelles) ; \
         écrit ξ₀ dans le manifeste et les poids de plans dans geometrie.json (beta_plans) ; plaq_ss_st.csv par chaîne ; \
         sans torsion ni mutant ; absent ou 1 : isotrope, sorties inchangées au bit",
    ))
}

/// `--mesures tube:ab:RMAX:TMAX[:N]` (N = période en indices de mesure, 1 par défaut).
fn analyser_mesures(spec: &str, g: &GeoA, nmeas: u64) -> Resultat<(ParamObs, u64)> {
    let r = spec.strip_prefix("tube:").ok_or_else(|| {
        Erreur::Config(format!(
            "--cotes : --mesures « {spec} » non pris en charge (seul tube:ab:RMAX:TMAX[:N])"
        ))
    })?;
    let p: Vec<&str> = r.split(':').collect();
    if p.len() != 3 && p.len() != 4 {
        return Err(Erreur::Config(format!(
            "--mesures « {spec} » : attendu tube:ab:RMAX:TMAX[:N]"
        )));
    }
    // domaine admis : R, T et période N, puis contraintes croisées (R ≤ ℓ_a/2, T ≤ ℓ_b/2 ; N ≤ nmeas, ⌊nmeas/N⌋ ≥ 2)
    let rmax = domaine::entier("gen", "--mesures tube:ab:RMAX", p[1])?;
    let tmax = domaine::entier("gen", "--mesures tube:ab:RMAX:TMAX", p[2])?;
    let n = match p.get(3) {
        Some(x) => domaine::entier("gen", "--mesures tube:…:N (période)", x)?,
        None => 1,
    };
    if n > nmeas || nmeas / n < 2 {
        return Err(Erreur::Config(format!(
            "--mesures tube:…:{n} : période {n} pour --nmeas {nmeas} ⇒ {} mesure(s) d'observables ; au moins 2 requises \
             (moyenne et erreur de blocs)",
            nmeas / n
        )));
    }
    Ok((
        ParamObs::analyser(&format!("{}:{rmax}:{tmax}", p[0]), g)?,
        n,
    ))
}

/// `su2 gen --reprendre DIR` sur un run de ce module (manifest.json de mode « su2-gen-cotes » ou « su2-tube-apparie ») : refus
/// clair, code 2 (ces runs n'écrivent pas de point de reprise). Tout autre répertoire (pas de manifeste lisible, autre mode) :
/// rien, le chemin de reprise existant s'applique tel quel.
pub fn refuser_reprise(dir: &Path) -> Resultat<()> {
    let Ok(txt) = fs::read_to_string(dir.join("manifest.json")) else {
        return Ok(());
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
        return Ok(());
    };
    match v.get("mode").and_then(|x| x.as_str()) {
        Some(mode @ ("su2-gen-cotes" | "su2-tube-apparie")) => Err(Erreur::Config(format!(
            "--reprendre {} : run du module tube (mode {mode}), non reprenable (pas de point de reprise) : relancer dans un \
             répertoire neuf",
            dir.display()
        ))),
        _ => Ok(()),
    }
}

// ------------------------------------------------------------------------------------------
// Bornes des tailles et des totaux (relecture c16) : arithmétique VÉRIFIÉE, refus avant toute écriture
// ------------------------------------------------------------------------------------------

/// Balayages de bain thermique par chaîne, au plus : le numéro de balayage du compteur Philox (`sweep_id`) tient sur 32 bits ;
/// avec au plus 2³² − 1 balayages, deux balayages d'une même chaîne n'emploient jamais le même compteur.
pub const BALAYAGES_MAX: u64 = u32::MAX as u64;

/// Chaînes d'un lot `su2 gen --cotes`, au plus.
pub const CHAINES_MAX: usize = 4096;

/// Nombre total de balayages de bain thermique d'une chaîne, therm + nmeas·skip, calculé en arithmétique vérifiée et borné
/// par [`BALAYAGES_MAX`] (refus d'usage, code 2).
pub fn balayages_par_chaine(therm: u64, nmeas: u64, skip: u64) -> Resultat<u64> {
    nmeas
        .checked_mul(skip)
        .and_then(|n| n.checked_add(therm))
        .filter(|&n| n <= BALAYAGES_MAX)
        .ok_or_else(|| {
            Erreur::Config(format!(
                "therm + nmeas·skip = {therm} + {nmeas}·{skip} : au plus 2³² − 1 balayages par chaîne (le compteur Philox des \
                 balayages tient sur 32 bits)"
            ))
        })
}

/// Erreur d'usage d'une estimation de taille non représentable sur 64 bits.
fn non_representable(quoi: &str) -> Erreur {
    Erreur::Config(format!(
        "{quoi} : taille non représentable sur 64 bits (débordement), refusé"
    ))
}

/// Somme vérifiée d'estimations d'octets.
fn somme(termes: &[Option<u64>]) -> Option<u64> {
    termes.iter().try_fold(0u64, |a, t| a.checked_add((*t)?))
}

/// Octets de travail d'une chaîne : champ 4V quaternions (128 o/site), classes de couleur et tableau du coloriage glouton
/// (16 o/site), tampon d'un balayage (≤ 64 o/site), mesure5 (8 o/site).
fn octets_chaine(g: &GeoA) -> Option<u64> {
    (g.v as u64).checked_mul(216)
}

/// Table z de la pile (6V réels).
fn octets_pile(g: &GeoA) -> Option<u64> {
    (g.v as u64).checked_mul(48)
}

/// Transitoires d'une mesure du tube : lignes de liens (R + T)·V quaternions, sommes préfixes (≤ 25 o/site), fraction de
/// vortex et Polyakov (≤ 20 o/site), et (relecture c16c, objection 21) les sommes partielles SIMULTANÉES des boucles de
/// Wilson (`obs::somme_sites` : ⌈V/BLOC⌉ vecteurs de 2·R·T réels, tous vivants avant leur réduction) : 16·R·T·⌈V/BLOC⌉ o.
fn octets_mesure_obs(g: &GeoA, o: &ParamObs) -> Option<u64> {
    let par_site = ((o.rmax + o.tmax) as u64)
        .checked_mul(32)?
        .checked_add(48)?
        .checked_mul(g.v as u64)?;
    let partielles = (o.rmax as u64)
        .checked_mul(o.tmax as u64)?
        .checked_mul(16)?
        .checked_mul((g.v as u64).div_ceil(obs::BLOC as u64))?;
    par_site.checked_add(partielles)
}

/// Séries gardées en mémoire par chaîne (gen) : par mesure d'observables, R·T boucles + V, fraction, 6 plaquettes,
/// 12 Polyakov, en-tête de vecteur et copie finale d'une composante (moyennes).
fn octets_series_gen(nobs: u64, o: &ParamObs) -> Option<u64> {
    // MAJORANT (vecteurs réservés à leur taille exacte, nobs) : par mesure V, fraction, 6 plaquettes, 12 Polyakov (160 o),
    // en-tête du vecteur des boucles (24 o), ses 8·R·T o et 16 o d'allocation ; puis une copie d'une composante (8 o)
    nobs.checked_mul(
        ((o.rmax * o.tmax) as u64)
            .checked_mul(8)?
            .checked_add(160 + 24 + 16 + 8)?,
    )
}

/// Catalogue d'un répertoire de run (MAJORANT) : liste des fichiers consignés au manifeste, ses copies JSON et chemins de
/// `consigner_fichiers`, 2 Kio par fichier.
fn octets_catalogue(nfichiers: u64) -> Option<u64> {
    nfichiers.checked_mul(2048)
}

/// Écriture d'une configuration : ChampGeo (16V réels) et octets NPY c16 (256 o/site).
fn octets_dump_memoire(g: &GeoA) -> Option<u64> {
    (g.v as u64).checked_mul(128 + 256)
}

/// Octets d'un fichier NPY c16 (4V × 4 complexes de 16 octets) et de son en-tête.
fn octets_npy(g: &GeoA) -> Option<u64> {
    (g.v as u64).checked_mul(256)?.checked_add(256)
}

/// Admission mémoire de kvu (`machine::admission`, rc 5) pour `quoi`, besoin `octets` (estimation vérifiée).
fn admettre_memoire(quoi: &str, octets: Option<u64>) -> Resultat<()> {
    let hote = octets.ok_or_else(|| non_representable(&format!("{quoi} : mémoire")))?;
    crate::machine::admission::admettre(
        quoi,
        crate::machine::admission::BesoinMemoire { hote, gpu: 0 },
        false,
    )
}

/// Mémoire et disque de `su2 gen --cotes` : (mémoire, projection d'écriture), par estimations vérifiées.
fn besoins_gen(p: &ParamGen, nch: usize) -> Resultat<(Option<u64>, u64)> {
    let g = &p.geo;
    let nobs = match (&p.obs, p.obs_every) {
        (Some(_), n) if n > 0 => p.nmeas / n,
        _ => 0,
    };
    let ndumps = p.nmeas.checked_div(p.dump_every).unwrap_or(0) + u64::from(p.dump_initial);
    let par_chaine_mem = somme(&[
        octets_chaine(g),
        p.obs.as_ref().map_or(Some(0), |o| octets_mesure_obs(g, o)),
        p.obs
            .as_ref()
            .map_or(Some(0), |o| octets_series_gen(nobs, o)),
        if ndumps > 0 {
            octets_dump_memoire(g)
        } else {
            Some(0)
        },
        // catalogue : configurations écrites + au plus 8 autres fichiers par chaîne
        ndumps.checked_add(8).and_then(octets_catalogue),
    ]);
    let memoire = somme(&[
        octets_pile(g),
        par_chaine_mem.and_then(|x| x.checked_mul(nch as u64)),
    ]);
    // disque, par chaîne : therm.csv (≤ 140 o/ligne), measurements.csv (≤ 160 o/ligne), tube.csv (≤ 26 o par nombre,
    // 2RT + 21 nombres, + 32), dumps, geometrie.json et resume.json (≤ 8 Kio)
    let tube_ligne = p.obs.as_ref().map_or(Some(0), |o| {
        ((2 * o.rmax * o.tmax + 21) as u64)
            .checked_mul(26)?
            .checked_add(32)
    });
    let disque_chaine = somme(&[
        p.therm.checked_add(1).and_then(|n| n.checked_mul(140)),
        p.nmeas.checked_add(1).and_then(|n| n.checked_mul(160)),
        tube_ligne.and_then(|l| l.checked_mul(nobs)),
        octets_npy(g).and_then(|o| o.checked_mul(ndumps)),
        Some(8192),
    ]);
    let disque = disque_chaine
        .and_then(|d| d.checked_mul(nch as u64))
        .ok_or_else(|| non_representable("su2 gen --cotes : écriture projetée"))?;
    Ok((memoire, disque))
}

/// Mémoire et disque de `su2 tube apparie` : (mémoire, projection d'écriture), par estimations vérifiées. Toutes les tâches
/// (β × réplique × secteur) sont comptées comme simultanées (borne haute).
fn besoins_apparie(
    g: &GeoA,
    nbeta: usize,
    p: &ParamApparie,
    minimum: bool,
) -> Resultat<(Option<u64>, u64)> {
    let ntaches = (nbeta as u64)
        .checked_mul(p.nrep as u64)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| non_representable("--repliques × --beta : nombre de tâches"))?;
    let serie = if p.series {
        p.nmeas.checked_mul(8)
    } else {
        Some(0)
    };
    // par tâche : chaîne, résultat (blocs, série à capacité exacte nmeas), entrée de la liste des tâches, catalogue
    // (fichier de série consigné)
    let par_tache = somme(&[
        octets_chaine(g),
        Some(24 + 256 + 64 + 32),
        serie,
        octets_catalogue(1),
    ]);
    let memoire = somme(&[
        octets_pile(g).and_then(|x| x.checked_mul(2)),
        if minimum { octets_chaine(g) } else { Some(0) },
        par_tache.and_then(|x| x.checked_mul(ntaches)),
    ]);
    // disque : séries (≤ 26 o par valeur), apparie.json (≤ 256 o par tâche, ≤ 1 Kio par β, ≤ 64 Kio d'en-tête)
    let disque = somme(&[
        if p.series {
            p.nmeas.checked_mul(26).and_then(|x| x.checked_mul(ntaches))
        } else {
            Some(0)
        },
        ntaches.checked_mul(256),
        (nbeta as u64).checked_mul(1024),
        Some(65536),
    ])
    .ok_or_else(|| non_representable("su2 tube apparie : écriture projetée"))?;
    Ok((memoire, disque))
}

/// FILET FINAL (relecture c16b) : tout nombre écrit dans un résultat doit être fini ; sinon erreur de validation (code 8) :
/// le manifeste est clos en échec, jamais de COMPLETE.
fn fini(x: f64, quoi: &str) -> Resultat<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(Erreur::Validation(format!(
            "{quoi} non fini ({x}) : résultat refusé"
        )))
    }
}

/// Côtés de `--cotes` confrontés au domaine admis (chaque côté, puis le volume en arithmétique vérifiée) ; aucune table.
fn geo_domaine(commande: &str, cotes: &str) -> Resultat<GeoA> {
    let v: Vec<&str> = cotes.split(',').collect();
    if v.len() != 4 {
        return Err(Erreur::Config(format!(
            "--cotes : 4 côtés attendus (l0,l1,l2,l3), reçu « {cotes} »"
        )));
    }
    let mut d = [0usize; 4];
    for (x, t) in d.iter_mut().zip(&v) {
        *x = domaine::entier(commande, "--cotes", t)? as usize;
    }
    let vol = d
        .iter()
        .try_fold(1u64, |a, &x| a.checked_mul(x as u64))
        .unwrap_or(u64::MAX);
    domaine::borne(commande, "volume", vol as f64)?;
    GeoA::nouvelle(d)
}

/// Liste de réels (`--beta`) confrontée, valeur par valeur, au domaine admis.
fn liste_reels(commande: &str, option: &str, texte: &str) -> Resultat<Vec<f64>> {
    texte
        .split(',')
        .map(|x| domaine::reel(commande, option, x))
        .collect()
}

/// Liste d'entiers (`--graine`) confrontée, valeur par valeur, au domaine admis.
fn liste_entiers(commande: &str, option: &str, texte: &str) -> Resultat<Vec<u64>> {
    texte
        .split(',')
        .map(|x| domaine::entier(commande, option, x))
        .collect()
}

/// Compatibilité d'un mutant (contrôle négatif) avec la torsion, sans construire la pile.
fn verifier_mutant(t: &Torsion, m: Mutant) -> Resultat<()> {
    match m {
        Mutant::PileIncomplete if !t.tordu(0, 1) => Err(Erreur::Config(
            "mutant pile-incomplete : le plan 01 doit être tordu".into(),
        )),
        Mutant::MauvaisPlan if !t.tordu(0, 1) || t.tordu(0, 2) => Err(Erreur::Config(
            "mutant mauvais-plan : plan 01 tordu et 02 non tordu requis".into(),
        )),
        _ => Ok(()),
    }
}

/// Nombre de chaînes d'un lot (règles de `su2 gen` : listes de longueur 1 ou n, une graine seule exige --graine-pas), au
/// plus 4096, de graines DEUX À DEUX DISTINCTES, vérifié sans construire le lot : liste explicite triée ; graine + i·pas
/// (i < n) distinctes ssi n ≤ 2^(64 − v₂(pas)) (pas = 0 : une seule graine).
fn nombre_de_chaines(
    betas: &[f64],
    graines: &[u64],
    k: Option<usize>,
    pas: Option<u64>,
) -> Resultat<usize> {
    let n = k.unwrap_or_else(|| betas.len().max(graines.len()));
    domaine::borne("gen", "--chaines", n as f64)?;
    if betas.len() != 1 && betas.len() != n {
        return Err(Erreur::Config(format!(
            "{} valeurs de β pour {n} chaînes (1 ou {n} attendues)",
            betas.len()
        )));
    }
    if graines.len() != 1 && graines.len() != n {
        return Err(Erreur::Config(format!(
            "{} graines pour {n} chaînes (1 ou {n} attendues)",
            graines.len()
        )));
    }
    if graines.len() == 1 && n > 1 && pas.is_none() {
        return Err(Erreur::Config(format!(
            "{n} chaînes avec une seule graine : donner {n} graines ou --graine-pas PAS (graine + i·PAS)"
        )));
    }
    let distinctes = if n <= 1 {
        true
    } else if graines.len() == n {
        let mut g = graines.to_vec();
        g.sort_unstable();
        g.windows(2).all(|w| w[0] != w[1])
    } else {
        let periode_log2 = 64 - pas.unwrap_or(0).trailing_zeros();
        periode_log2 >= 63 || (n as u64) <= (1u64 << periode_log2)
    };
    if !distinctes {
        return Err(Erreur::Config(
            "--cotes : graines des chaînes non distinctes : deux chaînes auraient la même clé Philox ; donner des graines \
             distinctes (ou --graine-pas qui ne retombe pas sur une graine déjà prise)"
                .into(),
        ));
    }
    Ok(n)
}

/// Résultat de l'analyse de `su2 gen --cotes`.
struct AnalyseGen {
    backend: String,
    p: ParamGen,
    torsion: Torsion,
    depart: DepartT,
    betas: Vec<f64>,
    graines: Vec<u64>,
    k: Option<usize>,
    pas: Option<u64>,
    nch: usize,
    outdir: PathBuf,
}

/// Analyse de `su2 gen --cotes` : TOUTE la ligne de commande lue et confrontée au domaine admis (`domaine`) et aux
/// contraintes croisées, sans allocation de champ ni écriture (les petites listes d'arguments sont construites avant leur refus).
fn analyser_gen(m: &ArgMatches, cotes: &str) -> Resultat<AnalyseGen> {
    let backend = texte(m, "backend").unwrap_or("auto").trim().to_ascii_lowercase();
    if !matches!(backend.as_str(), "cpu" | "hip" | "cuda") {
        return Err(Erreur::Config(
            "--cotes : --backend explicite requis : cpu, hip ou cuda (carte : F64, côtés quelconques, sans torsion ni mutant ni --mesures tube, départ hot|cold)".into(),
        ));
    }
    if texte(m, "precision").unwrap_or("f64") != "f64" {
        return Err(Erreur::Config("--cotes : précision f64 seulement".into()));
    }
    for nom in ["bord", "zstd", "archive"] {
        if m.get_flag(nom) {
            return Err(Erreur::Config(format!(
                "--cotes : --{nom} non pris en charge"
            )));
        }
    }
    if texte(m, "reprendre").is_some() {
        return Err(Erreur::Config(
            "--cotes : --reprendre non pris en charge (un run --cotes est non reprenable)".into(),
        ));
    }
    for nom in ["arret-apres", "fenetre-s"] {
        if texte(m, nom).is_some() {
            return Err(Erreur::Config(format!(
                "--cotes : --{nom} non pris en charge"
            )));
        }
    }
    // options de `su2 gen` sans effet sur ce chemin : refusées si elles sont DONNÉES (leurs valeurs par défaut ne comptent pas)
    for nom in [
        "L",
        "options",
        "bc",
        "bc-graine",
        "jauge",
        "pre",
        "bloc",
        "source-hash",
    ] {
        if m.value_source(nom)
            .is_some_and(|v| v != ValueSource::DefaultValue)
        {
            return Err(Erreur::Config(format!("--cotes : --{nom} sans effet sur le tube, refusé (tore donné par --cotes, CPU F64)")));
        }
    }
    if texte(m, "reprise-every").unwrap_or("0").trim() != "0" {
        return Err(Erreur::Config(
            "--cotes : points de reprise non pris en charge (--reprise-every 0)".into(),
        ));
    }
    if texte(m, "format").unwrap_or("c16") != "c16" {
        return Err(Erreur::Config(
            "--cotes : dumps au format c16 seulement".into(),
        ));
    }
    let geo = geo_domaine("gen", cotes)?;
    let geo = match texte(m, "decalage") {
        None => geo,
        Some(t) => {
            let v: Vec<&str> = t.split(',').collect();
            if v.len() != 4 {
                return Err(Erreur::Config(format!("--decalage : 4 entiers attendus (v0,v1,v2,v3), reçu « {t} »")));
            }
            let mut vec = [0i64; 4];
            for (x, c) in vec.iter_mut().zip(&v) {
                *x = c.trim().parse::<i64>().map_err(|_| Erreur::Config(format!("--decalage : « {c} » n'est pas un entier")))?;
            }
            let axe: usize = texte(m, "decalage-axe")
                .unwrap_or("0")
                .trim()
                .parse()
                .map_err(|_| Erreur::Config("--decalage-axe : entier 0..3 attendu".into()))?;
            GeoA::avec_couture(geo.d, axe, vec)?
        }
    };
    let depart = match texte(m, "start").unwrap_or("hot") {
        "hot" => DepartT::Chaud,
        "cold" => DepartT::Froid,
        "vide" => DepartT::Vide,
        a => {
            return Err(Erreur::Config(format!(
                "--start « {a} » inconnu (hot|cold|vide)"
            )))
        }
    };
    let torsion = match texte(m, "torsion") {
        Some(t) => Torsion::analyser(t)?,
        None => Torsion::triviale(),
    };
    let betas = liste_reels("gen", "--beta", texte(m, "beta").unwrap_or("2.3"))?;
    let graines = liste_entiers("gen", "--graine", texte(m, "graine").unwrap_or("1"))?;
    let k = texte(m, "chaines")
        .map(|t| domaine::entier("gen", "--chaines", t))
        .transpose()?
        .map(|k| k as usize);
    let pas = texte(m, "graine-pas")
        .map(|t| domaine::entier("gen", "--graine-pas", t))
        .transpose()?;
    let nch = nombre_de_chaines(&betas, &graines, k, pas)?;
    let therm = domaine::entier("gen", "--therm", texte(m, "therm").unwrap_or("1000"))?;
    let nmeas = domaine::entier("gen", "--nmeas", texte(m, "nmeas").unwrap_or("100"))?;
    let skip = domaine::entier("gen", "--skip", texte(m, "skip").unwrap_or("1"))?;
    let nor = domaine::entier("gen", "--nor", texte(m, "nor").unwrap_or("4"))? as usize;
    let dump_every = domaine::entier("gen", "--dump-every", texte(m, "dump-every").unwrap_or("0"))?;
    balayages_par_chaine(therm, nmeas, skip)?;
    if dump_every > nmeas {
        return Err(Erreur::Config(format!(
            "--dump-every {dump_every} : 0 (jamais) ou au plus --nmeas {nmeas}"
        )));
    }
    let (obs, obs_every) = match texte(m, "mesures") {
        Some(t) => {
            let (o, n) = analyser_mesures(t, &geo, nmeas)?;
            (Some(o), n)
        }
        None => (None, 0),
    };
    let mutant = match texte(m, "mutant") {
        Some("metro") => {
            return Err(Erreur::Config(
                "--cotes : mutant metro non pris en charge".into(),
            ))
        }
        autre => Mutant::analyser(autre)?,
    };
    verifier_mutant(&torsion, mutant)?;
    let xi0 = match texte(m, "xi0") {
        None => None,
        Some(t) => {
            let x: f64 = t
                .trim()
                .parse()
                .map_err(|_| Erreur::Config(format!("--xi0 : « {t} » illisible")))?;
            if !(x.is_finite() && x > 0.0) {
                return Err(Erreur::Config("--xi0 : réel fini > 0 attendu".into()));
            }
            Some(x).filter(|&x| x != 1.0)
        }
    };
    if xi0.is_some() && (!torsion.est_triviale() || mutant != Mutant::Aucun) {
        return Err(Erreur::Config(
            "--cotes : --xi0 incompatible avec --torsion et --mutant".into(),
        ));
    }
    if geo.couture.is_some() && (backend != "cpu" || !torsion.est_triviale() || mutant != Mutant::Aucun || obs.is_some() || xi0.is_some() || depart == DepartT::Vide) {
        return Err(Erreur::Config(
            "--decalage : seulement --backend cpu, sans --torsion, --mutant, --mesures tube:…, --xi0 ni --start vide (première version : bain thermique, surrelaxation, plaquettes)".into(),
        ));
    }
    if backend != "cpu" {
        if !torsion.est_triviale() || mutant != Mutant::Aucun || obs.is_some() || depart == DepartT::Vide {
            return Err(Erreur::Config(format!(
                "--cotes --backend {backend} : sans --torsion, --mutant, --mesures tube:… ni --start vide (chemin carte F64 : réseau périodique, départ hot|cold)"
            )));
        }
        if nor_trop_grand(nor) {
            return Err(Erreur::Config("--nor trop grand".into()));
        }
    }
    let outdir =
        PathBuf::from(texte(m, "outdir").ok_or_else(|| Erreur::Config("--outdir requis".into()))?);
    Ok(AnalyseGen {
        backend,
        p: ParamGen {
            geo,
            therm,
            nmeas,
            skip,
            nor,
            dump_every,
            dump_initial: m.get_flag("dump-initial"),
            obs,
            obs_every,
            mutant,
            xi0,
        },
        torsion,
        depart,
        betas,
        graines,
        k,
        pas,
        nch,
        outdir,
    })
}

/// Analyse seule (sans calcul, sans allocation de champ, sans écriture) d'une ligne de commande du module
/// (`su2 gen --cotes …`, `su2 tube smin …`, `su2 tube apparie …`) contre le domaine admis : la MÊME analyse que celle qui
/// précède chaque exécution. Sert au test paramétré du domaine.
pub fn analyser_seulement(argv: &[&str]) -> Resultat<()> {
    let cmd = crate::su2::cli::commandes()
        .into_iter()
        .next()
        .ok_or_else(|| Erreur::Config("commande su2 absente".into()))?;
    let m = cmd
        .try_get_matches_from(argv)
        .map_err(|e| Erreur::Config(e.to_string()))?;
    match m.subcommand() {
        Some(("gen", g)) => {
            let c = texte(g, "cotes").ok_or_else(|| Erreur::Config("--cotes requis".into()))?;
            analyser_gen(g, c).map(|_| ())
        }
        Some(("tube", t)) => match t.subcommand() {
            Some(("smin", s)) => analyser_smin(s).map(|_| ()),
            Some(("apparie", a)) => analyser_apparie(a).map(|_| ()),
            _ => Err(Erreur::Config("su2 tube : smin | apparie attendu".into())),
        },
        _ => Err(Erreur::Config("su2 gen --cotes | su2 tube attendu".into())),
    }
}

/// Majorant de la mémoire de `su2 gen --cotes` (octets) pour ces paramètres (`periode` = 0 : pas d'observables) : celui que
/// l'admission emploie (`besoins_gen`).
#[allow(clippy::too_many_arguments)]
pub fn majorant_memoire_gen(
    cotes: [usize; 4],
    nch: usize,
    nmeas: u64,
    periode: u64,
    rmax: usize,
    tmax: usize,
    dump_every: u64,
    dump_initial: bool,
) -> Option<u64> {
    let geo = GeoA::nouvelle(cotes).ok()?;
    let p = ParamGen {
        geo,
        therm: 0,
        nmeas,
        skip: 1,
        nor: 0,
        dump_every,
        dump_initial,
        obs: (periode > 0).then_some(ParamObs {
            plan: (0, 1),
            rmax,
            tmax,
        }),
        obs_every: periode,
        mutant: Mutant::Aucun,
        xi0: None,
    };
    besoins_gen(&p, nch).ok()?.0
}

pub fn executer_gen_cotes(m: &ArgMatches, cotes: &str) -> Resultat<i32> {
    annoncer();
    // 1. domaine admis (une seule analyse, ni champ alloué ni écriture ; seules les petites listes d'arguments existent) ; 2. admission mémoire sur des MAJORANTS ; 3. seulement
    // ensuite, allocations (lot, pile, chaînes) et écritures (manifeste ouvert avec la projection disque)
    let AnalyseGen {
        backend,
        p,
        torsion,
        depart,
        betas,
        graines,
        k,
        pas,
        nch,
        outdir,
    } = analyser_gen(m, cotes)?;
    let (memoire, projection) = besoins_gen(&p, nch)?;
    admettre_memoire("su2 gen --cotes", memoire)?;
    let chaines = chaines_tube(&betas, &graines, k, pas)?;
    let geo = p.geo.clone();
    let config = json!({
        "cotes": geo.d, "v": geo.v,
        "pile": "z_p = -1 sur les plaquettes {mu,nu} tordues de coin x_mu = l_mu - 1, x_nu = l_nu - 1 (= Torsion::z de kvu M4 = lib_tube.py)",
        "action": "beta sum_p (1 - z_p ReTr(U_p)/2)",
        "betas": chaines.iter().map(|c| c.beta).collect::<Vec<_>>(),
        "graines": chaines.iter().map(|c| c.graine.0).collect::<Vec<_>>(),
        "depart": format!("{depart:?}"), "xi0": p.xi0, "backend": backend,
        "therm": p.therm, "nmeas": p.nmeas, "skip": p.skip, "nor": p.nor,
        "dump_every": p.dump_every, "dump_initial": p.dump_initial,
        "tube_obs": p.obs.as_ref().map(|o| json!({"plan": [o.plan.0, o.plan.1], "rmax": o.rmax, "tmax": o.tmax, "every": p.obs_every})),
        "mutant": format!("{:?}", p.mutant),
        "couture": geo.couture.as_ref().map(|c| json!({"axe": c.axe, "decalage": c.vec, "sens": "voisin haut du site (l_axe-1, y) = (0, y + decalage) ; Z = Tr(S^decalage T^l_axe)"})),
        "wilson": "Wt = (prod z_p interieurs, plan (a,b)) ReTr W(C)/2 ; W = ReTr W(C)/2 brut ; moyennes sur les V coins",
        "fichiers_par_chaine": "therm.csv, measurements.csv (format su2 gen), tube.csv (si --mesures tube:...), cfg_NNNNNN.npy, geometrie.json",
    });
    // Manifeste commun (répertoire NEUF, « en_cours » à l'ouverture, COMPLETE en dernier), ouvert avec la projection
    // d'écriture (garde SSD, rc 7) ; toute erreur après son ouverture le clôt en échec (`sous_manifeste`).
    let argv: Vec<String> = std::env::args().collect();
    let gpu = backend != "cpu";
    let moteur = if gpu { MOTEUR_GPU } else { MOTEUR };
    // La carte n'est ouverte qu'après l'admission et avant le manifeste : un backend absent est une erreur de configuration
    // sans écriture.
    let app = if gpu {
        Some(
            crate::su2::cli::ouvrir_backend(&backend)?
                .ok_or_else(|| Erreur::Config(format!("--backend {backend} : carte absente")))?,
        )
    } else {
        None
    };
    let man = Manifeste::ouvrir_avec_garde(&outdir, moteur, &argv, &GardeSsd::machine(), projection)?;
    let mut spec = spec_tube(&geo, &torsion);
    if let Some(x) = p.xi0 {
        spec = spec.avec_beta_plans(crate::su2::une_boite::poids_plans_aniso(x, 3));
    }
    let pas_tot = (p.therm + p.nmeas * p.skip) as f64 * nch as f64;
    let secondes = sous_manifeste(man, |man| {
        man.champ("mode", "\"su2-gen-cotes\"")?;
        man.champ("parametres", &parametres().to_string())?;
        man.champ("backend", &json!(backend).to_string())?;
        man.champ("precision", "\"f64\"")?;
        if let Some(x) = p.xi0 {
            man.champ("xi0", &json!({"xi0": x, "direction_temps": 3, "poids": "plaquettes contenant l'axe 3 : xi0 ; autres : 1/xi0 ; S = (beta/xi0) sum_spatiales + beta xi0 sum_temporelles"}).to_string())?;
        }
        spec.inscrire(man, "geometrie")?;
        man.champ("tube", &config.to_string())?;
        let mut pile = pile_mutee(&geo, &torsion, p.mutant)?;
        if let Some(x) = p.xi0 {
            pile = pile_aniso(pile, &geo, x, 3);
        }
        let dirs: Vec<PathBuf> = (0..nch)
            .map(|c| crate::su2::run::repertoire_chaine(&outdir, nch, c))
            .collect();
        for d in &dirs {
            fs::create_dir_all(d).map_err(|e| io("création", d, e))?;
            geom::ecrire_geometrie(d, &spec)?;
        }
        if !torsion.est_triviale() {
            let tj = serde_json::to_vec_pretty(&torsion.json())
                .map_err(|e| Erreur::Format(e.to_string()))?;
            fs::write(outdir.join("torsion.json"), tj).map_err(|e| io("écriture", &outdir, e))?;
        }
        let t0 = std::time::Instant::now();
        let resumes: Vec<Resultat<serde_json::Value>> = match &app {
            Some(app) => executer_lot_gpu(app, &p, &spec, &chaines, depart, &dirs)?
                .into_iter()
                .map(Ok)
                .collect(),
            None => chaines
                .par_iter()
                .enumerate()
                .map(|(c, pc)| {
                    executer_chaine(&p, &pile, &spec, pc.beta, pc.graine.0, depart, &dirs[c])
                })
                .collect(),
        };
        // première erreur dans l'ordre des chaînes
        let out = resumes.into_iter().collect::<Resultat<Vec<_>>>()?;
        let secondes = t0.elapsed().as_secs_f64();
        let resume = json!({
            "chaines": out, "secondes": fini(secondes, "durée")?,
            "pas_par_s": fini(pas_tot / secondes.max(1e-9), "débit")?,
            "fils_rayon": rayon::current_num_threads(),
        });
        let rj = serde_json::to_vec_pretty(&resume).map_err(|e| Erreur::Format(e.to_string()))?;
        fs::write(outdir.join("resume.json"), rj).map_err(|e| io("écriture", &outdir, e))?;
        man.champ("wall_s", &format!("{secondes}"))?;
        consigner_fichiers(man, &outdir)?;
        Ok(secondes)
    })?;
    println!(
        "PERF tube cotes={} chaines={} pas={} secondes={:.2} pas_par_s={:.2} fils={}",
        geo.texte(),
        nch,
        p.therm + p.nmeas * p.skip,
        secondes,
        pas_tot / secondes.max(1e-9),
        rayon::current_num_threads()
    );
    Ok(0)
}

fn executer_chaine(
    p: &ParamGen,
    pile: &Pile,
    spec: &GeoSpec,
    beta: f64,
    graine: u64,
    depart: DepartT,
    dir: &Path,
) -> Resultat<serde_json::Value> {
    let mut c = ChaineTube::nouvelle(p.geo.clone(), graine);
    match depart {
        DepartT::Chaud => c.init(true),
        DepartT::Froid => c.init(false),
        DepartT::Vide => c.vide(true, true),
    }
    let inverse = p.mutant == Mutant::OrdreCouleurs;
    // un pas ; ARRÊT dès qu'un lien a épuisé les essais du bain thermique (relecture c16b, objection 11) : la chaîne est
    // perdue, inutile de continuer (le run est clos en échec, code 8)
    let pas = |c: &mut ChaineTube| -> Resultat<()> {
        c.sweep_hb_ordre(beta, pile, inverse);
        if c.n_cap != 0 {
            return Err(Erreur::Validation(format!(
                "plafond d'essais du bain thermique atteint {} fois (balayage {}) : chaîne arrêtée",
                c.n_cap, c.sweep_id
            )));
        }
        for _ in 0..p.nor {
            c.sweep_or(pile);
        }
        Ok(())
    };
    let (pt, pm) = (dir.join("therm.csv"), dir.join("measurements.csv"));
    let mut ft = creer(&pt)?;
    let mut fm = creer(&pm)?;
    ecrire(&mut ft, &pt, "step,plaquette,P0,P1,P2,P3")?;
    ecrire(
        &mut fm,
        &pm,
        "measurement_index,total_steps,phase,plaquette,P0,P1,P2,P3",
    )?;
    let l5 = |r: [f64; 5]| {
        format!(
            "{},{},{},{},{}",
            g17(r[0]),
            g17(r[1]),
            g17(r[2]),
            g17(r[3]),
            g17(r[4])
        )
    };
    let verifier = |r: &[f64; 5], i: u64| -> Resultat<()> {
        if r.iter().any(|x| !x.is_finite() || x.abs() > 1.0 + 1e-9) {
            return Err(Erreur::Validation(format!(
                "contrôle de trajectoire : indice {i}, {r:?}"
            )));
        }
        Ok(())
    };
    let r = c.mesure5(pile);
    verifier(&r, 0)?;
    ecrire(&mut ft, &pt, &format!("0,{}", l5(r)))?;
    if p.dump_initial {
        ecrire_npy(&dir.join("cfg_000000.npy"), spec, &c.u)?;
    }
    for t in 1..=p.therm {
        pas(&mut c)?;
        let r = c.mesure5(pile);
        verifier(&r, t)?;
        ecrire(&mut ft, &pt, &format!("{t},{}", l5(r)))?;
    }
    ft.flush().map_err(|e| io("écriture", &pt, e))?;
    let r = c.mesure5(pile);
    ecrire(&mut fm, &pm, &format!("0,{},initial,{}", p.therm, l5(r)))?;
    // `--xi0` : plaquettes moyennes spatiale et temporelle (non pondérées) à chaque mesure ; la colonne `plaquette` de
    // measurements.csv est alors la moyenne PONDÉRÉE par les poids de plans (z·½ReTr).
    let pss = dir.join("plaq_ss_st.csv");
    let mut fss = if p.xi0.is_some() {
        let mut f = creer(&pss)?;
        ecrire(&mut f, &pss, "measurement_index,plaq_ss,plaq_st")?;
        let (a, b) = c.plaquettes_ss_st(3);
        ecrire(&mut f, &pss, &format!("0,{},{}", g17(a), g17(b)))?;
        Some(f)
    } else {
        None
    };
    let (po, mut fo) = match &p.obs {
        Some(o) => {
            let po = dir.join("tube.csv");
            let mut f = creer(&po)?;
            ecrire(
                &mut f,
                &po,
                &format!("measurement_index,total_steps,{}", obs::entete(o)),
            )?;
            (Some(po), Some(f))
        }
        None => (None, None),
    };
    // séries réservées à leur taille EXACTE (nombre de mesures d'observables) : la mémoire est celle de l'admission
    let nobs = if p.obs.is_some() && p.obs_every > 0 {
        (p.nmeas / p.obs_every) as usize
    } else {
        0
    };
    let mut series_v = Vec::with_capacity(nobs);
    let mut series_w: Vec<Vec<f64>> = Vec::with_capacity(nobs);
    let mut series_frac = Vec::with_capacity(nobs);
    let mut series_pl: Vec<[f64; 6]> = Vec::with_capacity(nobs);
    let mut series_poly: Vec<[[f64; 3]; 4]> = Vec::with_capacity(nobs);
    for mi in 1..=p.nmeas {
        for _ in 0..p.skip {
            pas(&mut c)?;
        }
        let r = c.mesure5(pile);
        verifier(&r, mi)?;
        ecrire(
            &mut fm,
            &pm,
            &format!("{mi},{},sample,{}", p.therm + mi * p.skip, l5(r)),
        )?;
        if let Some(f) = fss.as_mut() {
            let (a, b) = c.plaquettes_ss_st(3);
            ecrire(f, &pss, &format!("{mi},{},{}", g17(a), g17(b)))?;
        }
        if p.dump_every > 0 && mi % p.dump_every == 0 {
            ecrire_npy(&dir.join(format!("cfg_{mi:06}.npy")), spec, &c.u)?;
        }
        if let (Some(o), Some(f), Some(po)) = (&p.obs, fo.as_mut(), po.as_ref()) {
            if p.obs_every > 0 && mi % p.obs_every == 0 {
                let ob = obs::mesurer(&c.g, &c.u, pile, o);
                for x in [ob.v, ob.frac_vortex]
                    .iter()
                    .chain(&ob.plaq_plans)
                    .chain(ob.poly.iter().flatten())
                    .chain(&ob.wilson)
                    .chain(&ob.wilson_brut)
                {
                    fini(*x, &format!("observable du tube (mesure {mi})"))?;
                }
                ecrire(
                    f,
                    po,
                    &format!("{mi},{},{}", p.therm + mi * p.skip, obs::ligne(&ob)),
                )?;
                series_v.push(ob.v);
                series_frac.push(ob.frac_vortex);
                series_pl.push(ob.plaq_plans);
                series_poly.push(ob.poly);
                series_w.push(ob.wilson.clone());
            }
        }
    }
    fm.flush().map_err(|e| io("écriture", &pm, e))?;
    if let Some(f) = fss.as_mut() {
        f.flush().map_err(|e| io("écriture", &pss, e))?;
    }
    if let (Some(f), Some(po)) = (fo.as_mut(), po.as_ref()) {
        f.flush().map_err(|e| io("écriture", po, e))?;
    }
    if c.n_cap != 0 {
        return Err(Erreur::Validation(format!(
            "plafond d'essais du bain thermique atteint {} fois",
            c.n_cap
        )));
    }
    let nb = 20;
    let me = |x: &[f64]| -> Resultat<serde_json::Value> {
        let (m, e) = moyenne_erreur(x, nb);
        Ok(json!([fini(m, "moyenne")?, fini(e, "erreur de blocs")?]))
    };
    let essais = c.n_tries as f64 / (4.0 * c.g.v as f64 * (p.therm + p.nmeas * p.skip) as f64);
    let mut res = json!({
        "beta": beta, "graine": graine, "repertoire": dir.display().to_string(),
        "essais_par_lien_hb": fini(essais, "essais par lien")?,
        "plafond": c.n_cap,
    });
    if !series_v.is_empty() {
        let o = p.obs.as_ref().expect("obs");
        let nw = o.rmax * o.tmax;
        let w: Vec<serde_json::Value> = (0..nw)
            .map(|k| me(&series_w.iter().map(|s| s[k]).collect::<Vec<_>>()))
            .collect::<Resultat<_>>()?;
        let pl: Vec<serde_json::Value> = (0..6)
            .map(|k| me(&series_pl.iter().map(|s| s[k]).collect::<Vec<_>>()))
            .collect::<Resultat<_>>()?;
        let mut poly = Vec::new();
        for d in 0..4 {
            for j in 0..3 {
                poly.push(me(&series_poly
                    .iter()
                    .map(|s| s[d][j])
                    .collect::<Vec<_>>())?);
            }
        }
        res["n_obs"] = json!(series_v.len());
        res["V"] = me(&series_v)?;
        res["plaq_plans"] = json!(pl);
        res["frac_vortex"] = me(&series_frac)?;
        res["wilson_tilde"] = json!(w);
        res["polyakov_P_absP_P2"] = json!(poly);
        res["erreurs"] = json!(format!(
            "blocs : {nb} blocs égaux de la série ; moyenne et erreur sur les mêmes {nb}·⌊n/{nb}⌋ premières mesures, sans temps \
             d'autocorrélation"
        ));
    }
    Ok(res)
}

// ------------------------------------------------------------------------------------------
// su2 tube smin | apparie
// ------------------------------------------------------------------------------------------

pub fn commande() -> Command {
    let a = |n: &'static str, h: &'static str| Arg::new(n).long(n).num_args(1).help(h);
    Command::new("tube")
        .about("[EXPÉRIMENTAL] Tube tordu (tore anisotrope + torsion, CPU F64) : smin | apparie")
        .after_help(format!(
            "{}\n{}\n{}",
            domaine::table_aide("gen"),
            domaine::table_aide("smin"),
            domaine::table_aide("apparie")
        ))
        .subcommand_required(true)
        .subcommand(
            Command::new("smin")
                .about(
                    "[EXPÉRIMENTAL] S/β d'un candidat minimum par refroidissement (départ mince + bruit, abélien + bruit, ou chaud) : \
                     arrêt à max|∇S/β| < gtol, convergence rendue ; la minimalité n'est pas certifiée",
                )
                .after_help(domaine::table_aide("smin"))
                .arg(a("cotes", "l0,l1,l2,l3").required(true))
                .arg(a("torsion", "plans tordus").default_value("01,23"))
                .arg(a("depart", "mince | abelien | chaud").default_value("mince"))
                .arg(
                    a("bruit", "amplitude du bruit initial, dans [0, 2π] ; > 0 pour mince et abelien (départs stationnaires)")
                        .default_value("1e-3"),
                )
                .arg(a("graine", "graine").default_value("1"))
                .arg(a("balayages", "balayages de refroidissement au plus (≤ 2³² − 1)").default_value("200000"))
                .arg(a("gtol", "arrêt si max|∇S/β| < gtol (fini, > 0)").default_value("1e-10"))
                .arg(a("cfg", "écrit la configuration finale (NPY)"))
                .arg(Arg::new("mutant").long("mutant").num_args(1).hide(true).help("contrôle négatif : pile-incomplete|mauvais-plan")),
        )
        .subcommand(
            Command::new("apparie")
                .about("[EXPÉRIMENTAL] Secteurs appariés : torsion fibre seule (0) et fibre + plan long (1) ; ⟨V⟩₁ − ⟨V⟩₀ et erreur")
                .after_help(domaine::table_aide("apparie"))
                .arg(a("cotes", "l0,l1,l2,l3").required(true))
                .arg(a("beta", "liste de β (1 à 256 valeurs)").required(true))
                .arg(a("graine", "graine de base").default_value("1"))
                .arg(a("repliques", "chaînes indépendantes par β et par secteur (1 à 1024)").default_value("1"))
                .arg(a("therm", "pas de thermalisation (therm + nmeas·skip ≤ 2³² − 1 balayages par chaîne)").default_value("2000"))
                .arg(a("nmeas", "mesures (multiple de 20)").default_value("20000"))
                .arg(a("skip", "pas entre mesures").default_value("1"))
                .arg(a("nor", "OR par pas").default_value("1"))
                .arg(a("fibre", "torsion présente dans les deux secteurs (« aucune » possible)").default_value("23"))
                .arg(a("long", "plan long tordu dans le secteur 1").default_value("01"))
                .arg(
                    a(
                        "depart",
                        "minimum (secteur 0 : vide U*(+,+) si la fibre 23 est tordue, identité sinon ; d'action nulle seulement si \
                         --fibre vaut exactement 23 ou aucune, NON nulle sinon (par exemple 01 ou 01,23) ; l'action initiale \
                         réelle, calculée, est consignée dans apparie.json ; secteur 1 : candidat minimum refroidi) | chaud",
                    )
                    .default_value("minimum"),
                )
                .arg(a("series", "écrit les séries V brutes (oui | non)").default_value("oui"))
                .arg(a("outdir", "répertoire NEUF").required(true)),
        )
}

pub fn executer(m: &ArgMatches) -> Resultat<i32> {
    match m.subcommand() {
        Some(("smin", g)) => executer_smin(g),
        Some(("apparie", g)) => executer_apparie(g),
        _ => Err(Erreur::Config(
            "su2 tube : sous-commande attendue (smin | apparie)".into(),
        )),
    }
}

/// Résultat d'un refroidissement.
pub struct Refroid {
    pub s: f64,
    pub gmax: f64,
    pub balayages: u64,
    pub s_depart: f64,
}

/// Refroidit jusqu'à max|∇| < gtol ou `max_b` balayages (contrôle tous les 200 balayages).
pub fn refroidir(c: &mut ChaineTube, pile: &Pile, max_b: u64, gtol: f64) -> Refroid {
    let s_depart = c.action(pile);
    let mut b = 0u64;
    let mut gmax = c.gradient_max(pile);
    while b < max_b && gmax >= gtol {
        // paquets de 200 balayages, le dernier tronqué : jamais plus de max_b (relecture c16b, objection 19)
        let n = (max_b - b).min(200);
        for _ in 0..n {
            c.sweep_refroidir(pile);
        }
        b += n;
        gmax = c.gradient_max(pile);
    }
    Refroid {
        s: c.action(pile),
        gmax,
        balayages: b,
        s_depart,
    }
}

/// Départ « mince » (vide U*(+,+) + bruit) refroidi : CANDIDAT minimum du secteur de la pile (point où max|∇S/β| < gtol si
/// la convergence est atteinte ; la minimalité n'est pas certifiée).
pub fn minimiseur(
    g: &GeoA,
    pile: &Pile,
    depart: &str,
    bruit: f64,
    graine: u64,
    max_b: u64,
    gtol: f64,
) -> Resultat<(ChaineTube, Refroid)> {
    let mut c = ChaineTube::nouvelle(g.clone(), graine);
    match depart {
        "mince" => c.vide(true, true),
        "abelien" => c.abelien(),
        "chaud" => c.init(true),
        a => {
            return Err(Erreur::Config(format!(
                "--depart « {a} » inconnu (mince|abelien|chaud)"
            )))
        }
    }
    if bruit > 0.0 {
        c.bruiter(bruit);
    }
    let r = refroidir(&mut c, pile, max_b, gtol);
    Ok((c, r))
}

/// Résultat de l'analyse de `su2 tube smin`.
struct AnalyseSmin {
    geo: GeoA,
    t: Torsion,
    mutant: Mutant,
    depart: String,
    bruit: f64,
    graine: u64,
    balayages: u64,
    gtol: f64,
    cfg: Option<(PathBuf, GeoSpec, PathBuf)>,
}

/// Analyse de `su2 tube smin` contre le domaine admis : aucune table, aucune écriture (lecture seule d'un geometrie.json
/// voisin si --cfg est donné).
fn analyser_smin(m: &ArgMatches) -> Resultat<AnalyseSmin> {
    let geo = geo_domaine("smin", texte(m, "cotes").unwrap_or(""))?;
    let t = Torsion::analyser(texte(m, "torsion").unwrap_or("01,23"))?;
    let mutant = Mutant::analyser(texte(m, "mutant"))?;
    verifier_mutant(&t, mutant)?;
    let depart = texte(m, "depart").unwrap_or("mince").to_string();
    if !matches!(depart.as_str(), "mince" | "abelien" | "chaud") {
        return Err(Erreur::Config(format!(
            "--depart « {depart} » inconnu (mince|abelien|chaud)"
        )));
    }
    let bruit = domaine::reel("smin", "--bruit", texte(m, "bruit").unwrap_or("1e-3"))?;
    if bruit == 0.0 && depart != "chaud" {
        return Err(Erreur::Config(format!(
            "--bruit 0 avec --depart {depart} : départ stationnaire, le refroidissement n'en bougerait pas (bruit > 0 requis)"
        )));
    }
    let graine = domaine::entier("smin", "--graine", texte(m, "graine").unwrap_or("1"))?;
    let balayages = domaine::entier(
        "smin",
        "--balayages",
        texte(m, "balayages").unwrap_or("200000"),
    )?;
    let gtol = domaine::reel("smin", "--gtol", texte(m, "gtol").unwrap_or("1e-10"))?;
    // --cfg : un geometrie.json décrit TOUTES les configurations de son dossier (notes geom §5-6) : un geometrie.json existant
    // d'une autre géométrie est refusé (code 2), jamais réécrit.
    let cfg = match texte(m, "cfg") {
        Some(f) => {
            let f = PathBuf::from(f);
            let spec = spec_tube(&geo, &pile_torsion(&t, mutant));
            let dir = f
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."))
                .to_path_buf();
            let voisin = dir.join(geom::NOM_FICHIER_GEOMETRIE);
            if voisin.exists() && geom::lire_geometrie(&voisin).ok().as_ref() != Some(&spec) {
                return Err(Erreur::Config(format!(
                    "--cfg : {} décrit une autre géométrie (ou est illisible) ; un geometrie.json décrit toutes les \
                     configurations de son dossier : choisir un autre dossier",
                    voisin.display()
                )));
            }
            Some((f, spec, dir))
        }
        None => None,
    };
    Ok(AnalyseSmin {
        geo,
        t,
        mutant,
        depart,
        bruit,
        graine,
        balayages,
        gtol,
        cfg,
    })
}

fn executer_smin(m: &ArgMatches) -> Resultat<i32> {
    annoncer();
    // 1. domaine admis ; 2. admission mémoire et disque sur des majorants ; 3. seulement ensuite, pile et chaîne
    let AnalyseSmin {
        geo,
        t,
        mutant,
        depart,
        bruit,
        graine,
        balayages,
        gtol,
        cfg,
    } = analyser_smin(m)?;
    admettre_memoire(
        "su2 tube smin",
        somme(&[
            octets_pile(&geo),
            octets_chaine(&geo),
            if cfg.is_some() {
                octets_dump_memoire(&geo)
            } else {
                Some(0)
            },
        ]),
    )?;
    if let Some((f, _, _)) = &cfg {
        let octets = octets_npy(&geo)
            .and_then(|o| o.checked_add(16384))
            .ok_or_else(|| non_representable("su2 tube smin --cfg"))?;
        GardeSsd::machine().verifier(f, octets)?;
    }
    let pile = pile_mutee(&geo, &t, mutant)?;
    let t0 = std::time::Instant::now();
    let (c, r) = minimiseur(&geo, &pile, &depart, bruit, graine, balayages, gtol)?;
    // filet final : aucun nombre non fini ne sort (ni JSON, ni fichier)
    let s = fini(r.s, "S/β")?;
    let s_depart = fini(r.s_depart, "S/β de départ")?;
    let gmax = fini(r.gmax, "gradient")?;
    let pp = obs::polyakov(&geo, &c.u);
    let p2 = fini(pp[2][1], "Polyakov de fibre")?;
    let p3 = fini(pp[3][1], "Polyakov de fibre")?;
    let atteinte = gmax < gtol;
    if !atteinte {
        eprintln!(
            "kvu su2 tube smin : convergence NON atteinte : max|∇S/β| = {gmax:e} ≥ gtol = {gtol:e} après {} balayages ; la \
             configuration n'est pas un point stationnaire à cette tolérance",
            r.balayages
        );
    }
    if let Some((f, spec, dir)) = &cfg {
        // NPY anisotrope + geometrie.json voisin (absent, ou de même géométrie : contrôlé plus haut) + manifeste de fichier
        // (F.npy.manifest.json)
        ecrire_npy(f, spec, &c.u)?;
        geom::ecrire_geometrie(dir, spec)?;
        let argv: Vec<String> = std::env::args().collect();
        ecrire_manifeste_fichier(
            f,
            MOTEUR,
            &argv,
            &[
                ("parametres", parametres()),
                ("geometrie", spec.bloc()),
                ("nature", json!(NATURE_CANDIDAT)),
                ("S_sur_beta", json!(s)),
                ("grad_max", json!(gmax)),
                ("convergence_atteinte", json!(atteinte)),
            ],
        )?;
    }
    let j = json!({
        "cotes": geo.d, "torsion": t.texte(), "mutant": format!("{mutant:?}"), "depart": depart, "nature": NATURE_CANDIDAT,
        "S_sur_beta": s, "S_depart": s_depart, "grad_max": gmax, "balayages": r.balayages,
        "convergence": {"atteinte": atteinte, "grad_max": gmax, "gtol": gtol, "balayages": r.balayages, "balayages_max": balayages},
        "absP_fibre": [p2, p3], "secondes": t0.elapsed().as_secs_f64(),
    });
    println!("{j}");
    Ok(0)
}

/// Résultat de l'analyse de `su2 tube apparie`.
struct AnalyseApparie {
    geo: GeoA,
    betas: Vec<f64>,
    t0: Torsion,
    t1: Torsion,
    depart: String,
    p: ParamApparie,
    outdir: PathBuf,
}

/// Analyse de `su2 tube apparie` contre le domaine admis : aucune table, aucune liste de tâches, aucune écriture.
fn analyser_apparie(m: &ArgMatches) -> Resultat<AnalyseApparie> {
    let geo = geo_domaine("apparie", texte(m, "cotes").unwrap_or(""))?;
    let betas = liste_reels("apparie", "--beta", texte(m, "beta").unwrap_or(""))?;
    domaine::borne("apparie", "nombre de β", betas.len() as f64)?;
    let fibre = texte(m, "fibre").unwrap_or("23");
    let long = texte(m, "long").unwrap_or("01");
    let t0 = Torsion::analyser(fibre)?;
    let t1 = if t0.est_triviale() {
        Torsion::analyser(long)?
    } else {
        Torsion::analyser(&format!("{fibre},{long}"))?
    };
    let depart = texte(m, "depart").unwrap_or("minimum").to_string();
    if depart != "minimum" && depart != "chaud" {
        return Err(Erreur::Config(format!(
            "--depart « {depart} » inconnu (minimum|chaud)"
        )));
    }
    let p = ParamApparie::analyser(m)?;
    let outdir =
        PathBuf::from(texte(m, "outdir").ok_or_else(|| Erreur::Config("--outdir requis".into()))?);
    Ok(AnalyseApparie {
        geo,
        betas,
        t0,
        t1,
        depart,
        p,
        outdir,
    })
}

fn executer_apparie(m: &ArgMatches) -> Resultat<i32> {
    annoncer();
    // 1. domaine admis ; 2. admission mémoire sur des majorants ; 3. seulement ensuite, tâches, piles, chaînes, écritures
    let AnalyseApparie {
        geo,
        betas,
        t0: t0_,
        t1: t1_,
        depart,
        p,
        outdir,
    } = analyser_apparie(m)?;
    let (memoire, projection) = besoins_apparie(&geo, betas.len(), &p, depart == "minimum")?;
    admettre_memoire("su2 tube apparie", memoire)?;
    let taches = taches_apparie(p.graine, betas.len(), p.nrep)?;
    let argv: Vec<String> = std::env::args().collect();
    let man =
        Manifeste::ouvrir_avec_garde(&outdir, MOTEUR, &argv, &GardeSsd::machine(), projection)?;
    sous_manifeste(man, |man| {
        man.champ("mode", "\"su2-tube-apparie\"")?;
        man.champ("parametres", &parametres().to_string())?;
        // deux secteurs : pas de clé « geometrie » au niveau du run (notes geom §6)
        spec_tube(&geo, &t0_).inscrire(man, "geometrie_secteur0")?;
        spec_tube(&geo, &t1_).inscrire(man, "geometrie_secteur1")?;
        executer_apparie_corps(&p, &taches, &geo, &betas, &t0_, &t1_, &depart, &outdir)?;
        consigner_fichiers(man, &outdir)
    })?;
    Ok(0)
}

/// Répliques par β et par secteur de `su2 tube apparie`, au plus.
pub const REPLIQUES_MAX: usize = 1024;

/// Valeurs de β de `su2 tube apparie`, au plus.
pub const BETAS_MAX: usize = 256;

/// Graine de la tâche (β n° `ib`, réplique `rep` < `nrep`, secteur `sect` ∈ {0, 1}) de `su2 tube apparie` :
/// graine + 7919·k, k = ib·max(1000, 2·nrep) + 2·rep + sect, k calculé en arithmétique VÉRIFIÉE (None s'il déborde).
/// Injectivité dans un run (graine fixée) : (1) 2·rep + sect < 2·nrep ≤ pas, donc k est l'écriture en base mixte de
/// (ib, rep, sect) et deux tâches ont deux k distincts < 2⁶⁴ ; (2) 7919 est impair, donc k ↦ 7919·k est une bijection de
/// ℤ/2⁶⁴ ; (3) x ↦ graine + x aussi : deux tâches ont deux graines, donc deux clés Philox (`reseau::cle_philox`), distinctes.
/// Jusqu'à 500 répliques, k = 1000·ib + 2·rep + sect : graines identiques aux versions précédentes. Entre deux runs, deux
/// graines qui diffèrent de 7919·Δk partagent des flux : c'est à l'utilisateur de choisir des graines éloignées.
pub fn graine_tache(graine: u64, ib: usize, rep: usize, sect: usize, nrep: usize) -> Option<u64> {
    let pas = (nrep as u64).checked_mul(2)?.max(1000);
    let k = (ib as u64)
        .checked_mul(pas)?
        .checked_add((rep as u64).checked_mul(2)?)?
        .checked_add(sect as u64)?;
    Some(graine.wrapping_add(k.wrapping_mul(7919)))
}

/// Tâches (ib, réplique, secteur, graine) d'un run `apparie`, celles que le run exécute et dans cet ordre ; refus d'usage
/// (code 2) hors de 1..=BETAS_MAX valeurs de β et 1..=REPLIQUES_MAX répliques.
pub fn taches_apparie(
    graine: u64,
    nbeta: usize,
    nrep: usize,
) -> Resultat<Vec<(usize, usize, usize, u64)>> {
    if !(1..=BETAS_MAX).contains(&nbeta) {
        return Err(Erreur::Config(format!(
            "--beta : 1 à {BETAS_MAX} valeurs (reçu {nbeta})"
        )));
    }
    if !(1..=REPLIQUES_MAX).contains(&nrep) {
        return Err(Erreur::Config(format!(
            "--repliques {nrep} : 1 à {REPLIQUES_MAX} répliques"
        )));
    }
    let mut t = Vec::with_capacity(2 * nbeta * nrep);
    for ib in 0..nbeta {
        for rep in 0..nrep {
            for sect in 0..2 {
                let g = graine_tache(graine, ib, rep, sect, nrep).ok_or_else(|| {
                    Erreur::Config(format!(
                        "--repliques {nrep} : clé de tâche non représentable sur 64 bits"
                    ))
                })?;
                t.push((ib, rep, sect, g));
            }
        }
    }
    Ok(t)
}

/// Chaînes d'un lot `su2 gen --cotes` (listes --beta, --graine, --chaines, --graine-pas, comme `su2 gen`), au plus
/// CHAINES_MAX, de graines DEUX À DEUX DISTINCTES (la clé Philox d'une chaîne est sa graine : deux graines égales, y compris
/// graine + i·pas qui retombe sur elle-même modulo 2⁶⁴, feraient deux flux identiques) ; sinon refus d'usage, code 2.
pub fn chaines_tube(
    betas: &[f64],
    graines: &[u64],
    k: Option<usize>,
    pas: Option<u64>,
) -> Resultat<Vec<crate::su2::ParamChaine>> {
    if k.is_some_and(|k| k > CHAINES_MAX)
        || betas.len() > CHAINES_MAX
        || graines.len() > CHAINES_MAX
    {
        return Err(Erreur::Config(format!(
            "--chaines : au plus {CHAINES_MAX} chaînes par lot (--chaines, listes --beta et --graine)"
        )));
    }
    let chaines = crate::su2::cli::chaines_depuis_listes(
        betas,
        graines,
        k,
        pas,
        crate::commun::Depart::Chaud,
    )?;
    let mut vues = std::collections::HashSet::with_capacity(chaines.len());
    if let Some(c) = chaines.iter().find(|c| !vues.insert(c.graine.0)) {
        return Err(Erreur::Config(format!(
            "--cotes : graines des chaînes non distinctes (la graine {} revient) : deux chaînes auraient la même clé Philox ; \
             donner des graines distinctes (ou --graine-pas qui ne retombe pas sur une graine déjà prise)",
            c.graine.0
        )));
    }
    Ok(chaines)
}

/// Options numériques de `su2 tube apparie`, validées avant toute écriture et tout calcul (objection 7).
#[derive(Clone, Copy)]
struct ParamApparie {
    graine: u64,
    nrep: usize,
    therm: u64,
    nmeas: u64,
    skip: u64,
    nor: usize,
    series: bool,
}

impl ParamApparie {
    fn analyser(m: &ArgMatches) -> Resultat<ParamApparie> {
        // domaine admis (`domaine`), puis contraintes croisées
        let e = |nom: &str, option: &str, defaut: &str| {
            domaine::entier("apparie", option, texte(m, nom).unwrap_or(defaut))
        };
        let p = ParamApparie {
            graine: e("graine", "--graine", "1")?,
            nrep: e("repliques", "--repliques", "1")? as usize,
            therm: e("therm", "--therm", "2000")?,
            nmeas: e("nmeas", "--nmeas", "20000")?,
            skip: e("skip", "--skip", "1")?,
            nor: e("nor", "--nor", "1")? as usize,
            series: texte(m, "series").unwrap_or("oui") != "non",
        };
        if !p.nmeas.is_multiple_of(20) {
            return Err(Erreur::Config(format!(
                "--nmeas {} : multiple de 20 requis (moyenne et erreur sur les mêmes 20 blocs égaux)",
                p.nmeas
            )));
        }
        balayages_par_chaine(p.therm, p.nmeas, p.skip)?;
        Ok(p)
    }
}

/// Un pas d'apparie (bain thermique puis `nor` sur-relaxations, comme `ChaineTube::pas`), ARRÊTÉ dès qu'un lien a épuisé
/// les essais du bain thermique (relecture c16b, objection 11).
fn pas_borne(c: &mut ChaineTube, beta: f64, nor: usize, pile: &Pile) -> Resultat<()> {
    c.sweep_hb(beta, pile);
    if c.n_cap != 0 {
        return Err(Erreur::Validation(format!(
            "plafond d'essais du bain thermique atteint {} fois (balayage {}) : chaîne arrêtée",
            c.n_cap, c.sweep_id
        )));
    }
    for _ in 0..nor {
        c.sweep_or(pile);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn executer_apparie_corps(
    p: &ParamApparie,
    taches: &[(usize, usize, usize, u64)],
    geo: &GeoA,
    betas: &[f64],
    t0_: &Torsion,
    t1_: &Torsion,
    depart: &str,
    outdir: &Path,
) -> Resultat<()> {
    let geo = geo.clone();
    let (t0_, t1_) = (*t0_, *t1_);
    let depart = depart.to_string();
    let ParamApparie {
        graine,
        nrep,
        therm,
        nmeas,
        skip,
        nor,
        series,
    } = *p;
    let p0 = Pile::depuis_torsion(&geo, &t0_);
    let p1 = Pile::depuis_torsion(&geo, &t1_);
    // départ « minimum » : secteur 0 = vide U*(+,+) si la fibre 23 est tordue, identité sinon ; secteur 1 = minimiseur
    // (départ mince + bruit 1e-3, refroidi), calculé une fois.
    let fibre_tordue = t0_.tordu(2, 3);
    // ce que vaut réellement le départ du secteur 0 (relecture c16c, objection 20) : l'action initiale est CALCULÉE sur la
    // configuration de départ avec la pile du secteur 0 (celle des tâches : vide U*(+,+) si la fibre 23 est tordue, identité
    // sinon), jamais supposée ; « S = 0 » n'est écrit que si ce calcul rend 0 (par exemple fibre 23 seule, ou aucune torsion),
    // et pas pour une fibre composite comme 01,23 (S/β = 8 sur 4,4,2,2)
    let action_depart0: Option<f64> = if depart == "minimum" {
        let mut c0 = ChaineTube::nouvelle(geo.clone(), 0);
        if fibre_tordue {
            c0.vide(true, true);
        } else {
            c0.init(false);
        }
        Some(fini(c0.action(&p0), "S/β du départ du secteur 0")?)
    } else {
        None
    };
    let depart_secteur0 = match action_depart0 {
        None => "chaud (Haar)".to_string(),
        Some(s0) => {
            let nature = if fibre_tordue {
                "vide U*(+,+)"
            } else {
                "identité"
            };
            if s0.abs() < 1e-9 {
                format!("{nature}, S = 0 (calculée : S/β = {s0})")
            } else {
                format!("{nature}, action initiale NON nulle : S/β = {s0} (calculée)")
            }
        }
    };
    let (min1, r1) = if depart == "minimum" {
        let (c, r) = minimiseur(&geo, &p1, "mince", 1e-3, 1, 200_000, GTOL_APPARIE)?;
        fini(r.s, "S/β du candidat minimum")?;
        fini(r.gmax, "gradient du candidat minimum")?;
        if r.gmax >= GTOL_APPARIE || r.gmax.is_nan() {
            eprintln!(
                "kvu su2 tube apparie : départ du secteur 1 : convergence du refroidissement NON atteinte (max|∇S/β| = {:e} \
                 après {} balayages) ; ce n'est qu'un point de départ, thermalisé ensuite",
                r.gmax, r.balayages
            );
        }
        (Some(c), Some(r))
    } else {
        (None, None)
    };
    // Chaque tâche rend (ib, rep, secteur, moyenne, moyennes des 20 blocs, série éventuelle, essais par lien) : sans série, la
    // mémoire reste bornée quel que soit nmeas.
    type Sortie = (usize, usize, usize, f64, Vec<f64>, Option<Vec<f64>>, f64);
    let tache = |&(ib, rep, sect, g): &(usize, usize, usize, u64)| -> Resultat<Sortie> {
        let beta = betas[ib];
        let pile = if sect == 0 { &p0 } else { &p1 };
        let mut c = ChaineTube::nouvelle(geo.clone(), g);
        match (&min1, sect) {
            (None, _) => c.init(true),
            (Some(_), 0) if fibre_tordue => c.vide(true, true),
            (Some(_), 0) => c.init(false),
            (Some(m1), _) => c.u = m1.u.clone(),
        }
        for _ in 0..therm {
            pas_borne(&mut c, beta, nor, pile)?;
        }
        let taille = nmeas / 20;
        let mut blocs = vec![0.0; 20];
        let mut somme = 0.0;
        let mut v = if series {
            Some(Vec::with_capacity(nmeas as usize))
        } else {
            None
        };
        for i in 0..nmeas {
            for _ in 0..skip {
                pas_borne(&mut c, beta, nor, pile)?;
            }
            let x = fini(c.action(pile), "V")?;
            somme += x;
            let b = (i / taille) as usize;
            if b < 20 {
                blocs[b] += x;
            }
            if let Some(v) = v.as_mut() {
                v.push(x);
            }
        }
        if c.n_cap != 0 {
            return Err(Erreur::Validation(format!(
                "plafond atteint {} fois",
                c.n_cap
            )));
        }
        for b in blocs.iter_mut() {
            *b /= taille as f64;
        }
        let acc = c.n_tries as f64 / (4.0 * geo.v as f64 * (therm + nmeas * skip) as f64);
        Ok((ib, rep, sect, somme / nmeas as f64, blocs, v, acc))
    };
    let t0 = std::time::Instant::now();
    let res: Vec<Resultat<Sortie>> = taches.par_iter().map(tache).collect();
    let secondes = t0.elapsed().as_secs_f64();
    // (rep, secteur, (moyenne, erreur), essais)
    // (réplique, secteur, (moyenne, erreur), essais par lien)
    type ParReplique = (usize, usize, (f64, f64), f64);
    let mut par_beta: Vec<Vec<ParReplique>> = vec![Vec::new(); betas.len()];
    for r in res {
        let (ib, rep, sect, moy, blocs, v, acc) = r?;
        if let Some(v) = v {
            let p = outdir.join(format!("V_b{ib:02}_r{rep:02}_s{sect}.txt"));
            let mut f = creer(&p)?;
            for x in &v {
                ecrire(&mut f, &p, &g17(*x))?;
            }
            f.flush().map_err(|e| io("écriture", &p, e))?;
        }
        let (_, e) = moyenne_erreur(&blocs, 20);
        let (moy, e) = (fini(moy, "moyenne de V")?, fini(e, "erreur de blocs de V")?);
        // erreur = écart type des 20 moyennes de blocs / √20 (moyenne_erreur sur les blocs, un bloc par valeur)
        par_beta[ib].push((rep, sect, (moy, e), acc));
    }
    let n_sites = geo.v as f64;
    let mut lignes = Vec::new();
    for (ib, &beta) in betas.iter().enumerate() {
        let mut sect_me = [(0.0, 0.0); 2];
        let mut accs = [0.0; 2];
        for (s, sm) in sect_me.iter_mut().enumerate() {
            let rs: Vec<(f64, f64)> = par_beta[ib]
                .iter()
                .filter(|x| x.1 == s)
                .map(|x| x.2)
                .collect();
            let n = rs.len() as f64;
            let m = rs.iter().map(|x| x.0).sum::<f64>() / n;
            let e = rs.iter().map(|x| x.1 * x.1).sum::<f64>().sqrt() / n;
            *sm = (m, e);
            accs[s] = par_beta[ib]
                .iter()
                .filter(|x| x.1 == s)
                .map(|x| x.3)
                .sum::<f64>()
                / n;
        }
        let dv = sect_me[1].0 - sect_me[0].0;
        let edv = (sect_me[0].1.powi(2) + sect_me[1].1.powi(2)).sqrt();
        let equi = beta * sect_me[0].0 / (9.0 * n_sites / 2.0);
        // filet final : tout nombre de la ligne de résultats est fini (sinon code 8, pas de COMPLETE)
        for (x, quoi) in [
            (sect_me[0].0, "V0"),
            (sect_me[0].1, "erreur de V0"),
            (sect_me[1].0, "V1"),
            (sect_me[1].1, "erreur de V1"),
            (dv, "DV"),
            (edv, "erreur de DV"),
            (equi, "équipartition"),
            (accs[0], "essais par lien"),
            (accs[1], "essais par lien"),
        ] {
            fini(x, quoi)?;
        }
        println!(
            "APPARIE cotes={} beta={} V0={:.6}±{:.6} V1={:.6}±{:.6} DV={:.5}±{:.5} equipartition0={:.4} essais_hb={:.3}/{:.3}",
            geo.texte(), beta, sect_me[0].0, sect_me[0].1, sect_me[1].0, sect_me[1].1, dv, edv, equi, accs[0], accs[1]
        );
        let mut reps: Vec<&ParReplique> = par_beta[ib].iter().collect();
        reps.sort_by_key(|x| (x.1, x.0));
        let par_rep = |s: usize| -> Vec<[f64; 2]> {
            reps.iter()
                .filter(|x| x.1 == s)
                .map(|x| [x.2 .0, x.2 .1])
                .collect()
        };
        lignes.push(json!({
            "beta": beta, "V0": [sect_me[0].0, sect_me[0].1], "V1": [sect_me[1].0, sect_me[1].1],
            "DV": [dv, edv], "equipartition0": equi, "essais_par_lien_hb": accs, "repliques": nrep,
            "par_replique": {"V0": par_rep(0), "V1": par_rep(1)},
        }));
    }
    let j = json!({
        "cotes": geo.d, "secteur0": t0_.texte(), "secteur1": t1_.texte(), "depart": depart, "depart_secteur0": depart_secteur0,
        "action_initiale_secteur0": action_depart0,
        "candidat_minimum_secteur1": r1.as_ref().map(|r| json!({"S_sur_beta": r.s, "grad_max": r.gmax, "balayages": r.balayages,
            "gtol": GTOL_APPARIE, "convergence_atteinte": r.gmax < GTOL_APPARIE, "nature": NATURE_CANDIDAT})),
        "therm": therm, "nmeas": nmeas, "skip": skip, "nor": nor, "graine": graine,
        "erreurs": "blocs (20 blocs égaux) par chaîne ; répliques combinées en moyenne simple",
        "equipartition": "beta <V>_0 / (9 V / 2)",
        "resultats": lignes, "secondes": secondes, "fils_rayon": rayon::current_num_threads(),
        "argv": std::env::args().collect::<Vec<_>>(),
    });
    let p = outdir.join("apparie.json");
    fs::write(
        &p,
        serde_json::to_vec_pretty(&j).map_err(|e| Erreur::Format(e.to_string()))?,
    )
    .map_err(|e| io("écriture", &p, e))?;
    Ok(())
}

/// Mesures du chemin carte, sur l'hôte, à partir du champ téléchargé : (plaquette moyenne NON pondérée, P0…P3), mêmes
/// réductions à arbre fixe que `ChaineTube::mesure5` (poids z = 1).
fn mesure_hote(g: &GeoA, u: &[Q]) -> [f64; 5] {
    let f = |s: usize| {
        let x = g.coords(s);
        let mut acc = 0.0;
        for mu in 0..4 {
            for nu in mu + 1..4 {
                acc += super::reseau::plaquette(g, u, s, &x, mu, nu)[0];
            }
        }
        acc
    };
    let site: Vec<f64> = (0..g.v).into_par_iter().map(f).collect();
    let part = crate::heritage::su2_hb::k_reduce(&site);
    let mut out = [0f64; 5];
    out[0] = crate::heritage::su2_hb::k_final(&part, 1.0 / (6.0 * g.v as f64));
    for mu in 0..4 {
        let pol = super::reseau::polyakov_lignes(g, u, mu);
        out[1 + mu] = crate::heritage::su2_hb::k_final(&pol, 1.0 / pol.len() as f64);
    }
    out
}

/// (P_ss, P_st) non pondérées, direction du temps `temps` (même arithmétique que `ChaineTube::plaquettes_ss_st`).
fn ss_st_hote(g: &GeoA, u: &[Q], temps: usize) -> (f64, f64) {
    let f = |s: usize| {
        let x = g.coords(s);
        let (mut a, mut b) = (0.0, 0.0);
        for mu in 0..4 {
            for nu in mu + 1..4 {
                let p = super::reseau::plaquette(g, u, s, &x, mu, nu)[0];
                if mu == temps || nu == temps {
                    b += p;
                } else {
                    a += p;
                }
            }
        }
        (a, b)
    };
    let site: Vec<(f64, f64)> = (0..g.v).into_par_iter().map(f).collect();
    let sa: Vec<f64> = site.iter().map(|t| t.0).collect();
    let sb: Vec<f64> = site.iter().map(|t| t.1).collect();
    let n = 1.0 / (3.0 * g.v as f64);
    let (ka, kb) = (
        crate::heritage::su2_hb::k_reduce(&sa),
        crate::heritage::su2_hb::k_reduce(&sb),
    );
    (
        crate::heritage::su2_hb::k_final(&ka, n),
        crate::heritage::su2_hb::k_final(&kb, n),
    )
}

/// `su2 gen --cotes --backend hip|cuda` : le lot de chaînes tourne sur la carte (moteur `UneBoite`, F64, réseau périodique,
/// côtés quelconques, `--xi0` admis). Écrit, par chaîne, therm.csv (`step,plaquette` : ½ReTr moyen NON pondéré, somme S_tot de la
/// carte), measurements.csv (mêmes colonnes que le chemin processeur ; plaquette et Polyakov calculés sur l'hôte à partir du
/// champ téléchargé), plaq_ss_st.csv (avec `--xi0`) et les dumps ; geometrie.json est écrit par l'appelant.
#[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
fn executer_lot_gpu(
    app: &crate::gpu::Appareil,
    p: &ParamGen,
    spec: &GeoSpec,
    chaines: &[crate::su2::ParamChaine],
    depart: DepartT,
    dirs: &[PathBuf],
) -> Resultat<Vec<serde_json::Value>> {
    use crate::commun::{Depart, Precision};
    use crate::su2::une_boite::{ParamBoite, UneBoite};
    let dep = match depart {
        DepartT::Chaud => Depart::Chaud,
        DepartT::Froid => Depart::Froid,
        DepartT::Vide => return Err(Erreur::Config("--start vide : processeur seulement".into())),
    };
    let nch = chaines.len();
    let g = &p.geo;
    let pb = ParamBoite {
        cotes: g.d,
        torsion: Torsion::triviale(),
        poids: 1.0,
        nor: p.nor,
        chaines: chaines
            .iter()
            .map(|c| crate::su2::ParamChaine {
                beta: c.beta,
                graine: c.graine,
                depart: dep,
            })
            .collect(),
        precision: Precision::F64,
        impairs: true,
        aniso: p.xi0.map(|x| (x, 3)),
        couture: None,
    };
    let mut boite = UneBoite::nouveau(app, &pb)?;
    boite.init()?;
    let v6 = 6.0 * g.v as f64;
    let mut fts = Vec::with_capacity(nch);
    let mut fms = Vec::with_capacity(nch);
    let mut fss: Vec<Option<BufWriter<File>>> = Vec::with_capacity(nch);
    let (nt, nm, ns) = ("therm.csv", "measurements.csv", "plaq_ss_st.csv");
    for d in dirs {
        let mut ft = creer(&d.join(nt))?;
        ecrire(&mut ft, &d.join(nt), "step,plaquette")?;
        let mut fm = creer(&d.join(nm))?;
        ecrire(&mut fm, &d.join(nm), "measurement_index,total_steps,phase,plaquette,P0,P1,P2,P3")?;
        fts.push(ft);
        fms.push(fm);
        fss.push(if p.xi0.is_some() {
            let mut f = creer(&d.join(ns))?;
            ecrire(&mut f, &d.join(ns), "measurement_index,plaq_ss,plaq_st")?;
            Some(f)
        } else {
            None
        });
    }
    // thermalisation : S_tot de chaque chaîne rangée à chaque pas (historique par tranches), sans synchronisation intermédiaire
    let cap = 1024usize;
    boite.allouer_historique(cap)?;
    let mut local = 0usize;
    let mut premier = 0u64;
    let vidanger = |boite: &UneBoite, n: usize, premier: u64, fts: &mut [BufWriter<File>]| -> Resultat<()> {
        let h = boite.historique()?;
        for (c, ft) in fts.iter_mut().enumerate() {
            for i in 0..n {
                let x = h[(c * cap + i) * 2] / v6;
                if !x.is_finite() || x.abs() > 1.0 + 1e-9 {
                    return Err(Erreur::Validation(format!(
                        "contrôle de trajectoire : chaîne {c}, indice {}, plaquette {x}",
                        premier + i as u64
                    )));
                }
                writeln!(ft, "{},{}", premier + i as u64, g17(x)).map_err(|e| io("écriture", Path::new(nt), e))?;
            }
        }
        Ok(())
    };
    boite.mesurer(0)?;
    local += 1;
    for _ in 1..=p.therm {
        boite.pas()?;
        boite.mesurer(local)?;
        local += 1;
        if local == cap {
            vidanger(&boite, local, premier, &mut fts)?;
            premier += local as u64;
            local = 0;
        }
    }
    if local > 0 {
        vidanger(&boite, local, premier, &mut fts)?;
    }
    let hote = |boite: &UneBoite, c: usize| -> Resultat<Vec<Q>> {
        let q = boite.telecharger(c)?;
        Ok(q.chunks_exact(4).map(|a| [a[0], a[1], a[2], a[3]]).collect())
    };
    let ligne5 = |r: [f64; 5]| format!("{},{},{},{},{}", g17(r[0]), g17(r[1]), g17(r[2]), g17(r[3]), g17(r[4]));
    let ecrire_mesure = |boite: &UneBoite, mi: u64, phase: &str, fms: &mut [BufWriter<File>], fss: &mut [Option<BufWriter<File>>]| -> Resultat<()> {
        for c in 0..nch {
            let u = hote(boite, c)?;
            let r = mesure_hote(g, &u);
            if r.iter().any(|x| !x.is_finite() || x.abs() > 1.0 + 1e-9) {
                return Err(Erreur::Validation(format!("contrôle de trajectoire : chaîne {c}, mesure {mi}, {r:?}")));
            }
            let total = p.therm + mi * p.skip;
            let pm = dirs[c].join(nm);
            ecrire(&mut fms[c], &pm, &format!("{mi},{total},{phase},{}", ligne5(r)))?;
            if let Some(f) = fss[c].as_mut() {
                let (a, b) = ss_st_hote(g, &u, 3);
                ecrire(f, &dirs[c].join(ns), &format!("{mi},{},{}", g17(a), g17(b)))?;
            }
            if mi > 0 && p.dump_every > 0 && mi.is_multiple_of(p.dump_every) {
                ecrire_npy(&dirs[c].join(format!("cfg_{mi:06}.npy")), spec, &u)?;
            }
        }
        Ok(())
    };
    if p.dump_initial {
        for c in 0..nch {
            let u = hote(&boite, c)?;
            ecrire_npy(&dirs[c].join("cfg_000000.npy"), spec, &u)?;
        }
    }
    ecrire_mesure(&boite, 0, "initial", &mut fms, &mut fss)?;
    for mi in 1..=p.nmeas {
        for _ in 0..p.skip {
            boite.pas()?;
        }
        ecrire_mesure(&boite, mi, "sample", &mut fms, &mut fss)?;
    }
    for (d, (fm, fs)) in dirs.iter().zip(fms.iter_mut().zip(fss.iter_mut())) {
        fm.flush().map_err(|e| io("écriture", &d.join(nm), e))?;
        if let Some(f) = fs.as_mut() {
            f.flush().map_err(|e| io("écriture", &d.join(ns), e))?;
        }
    }
    for (d, ft) in dirs.iter().zip(fts.iter_mut()) {
        ft.flush().map_err(|e| io("écriture", &d.join(nt), e))?;
    }
    let compteurs = boite.compteurs()?;
    let pas_tot = (p.therm + p.nmeas * p.skip) as f64;
    let mut out = Vec::with_capacity(nch);
    for (c, (cap_atteint, essais)) in compteurs.into_iter().enumerate() {
        if cap_atteint != 0 {
            return Err(Erreur::Validation(format!(
                "plafond d'essais du bain thermique atteint {cap_atteint} fois (chaîne {c}) : chaîne perdue"
            )));
        }
        out.push(json!({
            "beta": chaines[c].beta, "graine": chaines[c].graine.0, "repertoire": dirs[c].display().to_string(),
            "essais_par_lien_hb": fini(essais as f64 / (4.0 * g.v as f64 * pas_tot.max(1.0)), "essais par lien")?,
            "plafond": cap_atteint,
        }));
    }
    Ok(out)
}

fn nor_trop_grand(nor: usize) -> bool {
    nor > 1_000_000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filet_final_refuse_tout_nombre_non_fini() {
        assert_eq!(fini(1.5, "x").unwrap(), 1.5);
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let e = fini(x, "équipartition").unwrap_err();
            assert_eq!(e.code_sortie(), 8, "{e}");
        }
    }

    #[test]
    fn refroidissement_au_plus_le_plafond() {
        let g = GeoA::nouvelle([4, 4, 2, 2]).unwrap();
        let pile = Pile::depuis_torsion(&g, &Torsion::analyser("01,23").unwrap());
        for max_b in [0u64, 1, 199, 201] {
            let mut c = ChaineTube::nouvelle(g.clone(), 5);
            c.init(true);
            let r = refroidir(&mut c, &pile, max_b, 1e-14);
            assert!(r.balayages <= max_b, "{} > {max_b}", r.balayages);
        }
    }
}
