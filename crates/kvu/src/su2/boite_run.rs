// SPDX-License-Identifier: Apache-2.0
//! Boucle de production des boîtes de Dirichlet ℓ⁴ à bord fixé (`kvu su2 boite`). Propriétaire : WP-DIR.
//!
//! Une exécution simule UN LOT de chaînes de même (ℓ, β), une par couple (bord, graine), dans le réseau de calcul périodique
//! de côté L_c = ℓ + 2, avec le masque de liens gelés de [`super::boite::masque_boite`] (liens fixés, inexistants et de
//! remplissage gelés). Déroulement : départ intérieur froid + bord posé (hôte, identique sur les trois backends), `therm` pas
//! sans mesure, puis `nmeas` mesures espacées de `skip` pas (un pas = 1 bain thermique + `nor` surrelaxations) :
//! plaquette centrale `P_c`, boucle 2×2 centrale `W₂` à chaque mesure, profil par couche tous les `profil_tous` mesures.
//!
//! Sorties (répertoire NEUF `outdir`) : `cNNN/mesures.csv` (`pas,pc,w2`), `cNNN/profil.csv` (`pas,d0,…`), `cNNN/config_finale.npy`
//! (option), `manifest.json` (kevinotron-unifie/1), `manifest_regen.json` (kvu-regen/1, rejouable par `kvu regen`), `COMPLETE`
//! en dernier. En fin de run, les liens gelés sont comparés à l'octet à leur état de départ (empreinte sha256) : un écart
//! est une erreur de validation (rc 8), sauf en mode mutant (contrôle négatif, consigné dans le manifeste).
//!
//! Bords lus (`--bord-npy`, lot WP21, docs/BORD_NPY.md) : une chaîne de bord `npy:<h8>` prend les liens fixés du fichier de ce nom
//! ([`super::boite_npy::BordNpy`]) ; les fichiers sont consignés dans `manifest.json` (`bords_npy`) et dans les `entrees` du manifeste
//! de régénération, que `kvu regen` contrôle avant tout rejeu.

use super::boite::{champ_initial, masque_boite, mesurer_hote, BordBoite, GeometrieBoite, NOMS_BORDS};
use super::boite_npy::{self, BordNpy, PREFIXE_NPY};
use super::boite_gpu::{MesureBoiteGpu, SU2_BOITE};
use super::ref64::CHAINES_MAX;
use super::run::{controle_plafond, creer, ecrire_ligne, io, prendre_verrou, sentinelle};
use super::{construire, AppMasque, ConfigLot, NoyauxSu2, ParamChaine};
use crate::commun::{BackendKind, ChampHote, Depart, Erreur, Graine, Precision, Resultat};
use crate::gpu::{Appareil, OptionsCompil, ProvenanceCompil};
use crate::heritage::manifest::{ecrire_atomique, sha256_fichier, sha256_hex};
use crate::heritage::su2_hb::g17;
use crate::machine::admission::{octets_champ, BesoinMemoire};
use crate::regen::classer::{classer, Contexte};
use crate::regen::schema::valider_manifeste;
use crate::stockage::npy::{self, FormatChamp};
use crate::stockage::{GardeSsd, Manifeste};
use serde_json::{json, Value};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Nombre de mesures gardées sur le GPU avant lecture (un vidage tous les CHUNK_MESURES mesures).
pub const CHUNK_MESURES: usize = 256;
/// Constante XOR de la clé de la source thermique (une chaîne `therm` : clé = graine ⊕ constante), pour ne jamais
/// réutiliser les compteurs Philox du bain thermique de la chaîne.
pub const CLE_SOURCE_THERM: u64 = 0x9E37_79B9_7F4A_7C15;
/// Côté minimal du réseau d'une source thermique (QW1 : 10).
pub const L_SOURCE_MIN: usize = 10;

/// Une chaîne du lot : nom du bord et graine 64 bits (clé Philox de la chaîne et de son bord).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ChaineBoite {
    pub bord: String,
    pub graine: u64,
}

/// Paramètres d'un lot de boîtes (CLI `su2 boite`).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConfigBoite {
    pub ell: usize,
    pub beta: f64,
    pub chaines: Vec<ChaineBoite>,
    pub therm: u64,
    pub nmeas: u64,
    pub skip: u64,
    pub nor: usize,
    /// « ref-k1 » ou « compat-prod ».
    pub options: String,
    /// Profil par couche tous les N mesures (0 = jamais).
    pub profil_tous: u64,
    /// Pas de thermalisation des sources thermiques.
    pub therm_source: u64,
    pub dump_final: bool,
    /// Contrôle négatif (jamais en production) ; `Tout` = comportement normal.
    pub mutant: AppMasque,
    pub outdir: PathBuf,
}

/// Rapport de fin de run.
#[derive(Clone, Debug, PartialEq)]
pub struct RapportBoite {
    pub chaines: usize,
    pub backend: BackendKind,
    /// Pas effectués (therm + nmeas·skip).
    pub pas: u64,
    pub mesures: u64,
    /// Temps de la boucle (thermalisation + mesures), hors compilation, chargement et écriture.
    pub s_boucle: f64,
    pub secondes: f64,
    pub ms_par_pas_et_chaine: f64,
    pub liens_libres_par_s: f64,
    pub bord_intact: bool,
}

/// Validation d'une configuration sans bord lu ; rend la géométrie.
pub fn valider_boite(cfg: &ConfigBoite) -> Resultat<GeometrieBoite> {
    valider_boite_bords(cfg, &[])
}

/// Validation d'une configuration et de ses bords lus (`--bord-npy`) ; rend la géométrie. Un nom de bord est l'un de `NOMS_BORDS`
/// ou `npy:<h8>` porté par un fichier de `bords` ; les fichiers ont des noms distincts, le côté du réseau de calcul et sont tous
/// utilisés par au moins une chaîne (docs/BORD_NPY.md §3).
pub fn valider_boite_bords(cfg: &ConfigBoite, bords: &[BordNpy]) -> Resultat<GeometrieBoite> {
    let g = GeometrieBoite::nouvelle(cfg.ell)?;
    if cfg.chaines.is_empty() {
        return Err(Erreur::Config("boîte : au moins une chaîne (--chaines ou --bords) requise".into()));
    }
    if cfg.chaines.len() > CHAINES_MAX {
        return Err(Erreur::Config(format!("{} chaînes : au plus {CHAINES_MAX} par lot", cfg.chaines.len())));
    }
    if !cfg.beta.is_finite() || cfg.beta < 0.0 {
        return Err(Erreur::Config(format!("β = {} invalide (fini, ≥ 0)", cfg.beta)));
    }
    if cfg.nmeas < 1 || cfg.skip < 1 {
        return Err(Erreur::Config("boîte : nmeas ≥ 1 et skip ≥ 1 requis".into()));
    }
    for c in &cfg.chaines {
        let nom = c.bord.as_str();
        if NOMS_BORDS.contains(&nom) {
            continue;
        }
        if nom == PREFIXE_NPY {
            return Err(Erreur::Config(format!(
                "bord « npy » non résolu : il désigne l'unique --bord-npy du lot (fichiers : {})",
                boite_npy::noms(bords)
            )));
        }
        if nom.starts_with("npy:") {
            if !boite_npy::nom_npy_valide(nom) {
                return Err(Erreur::Config(format!(
                    "bord « {nom} » : npy: doit être suivi des 8 premiers hexadécimaux (minuscules) du sha256 du fichier"
                )));
            }
            if !bords.iter().any(|b| b.nom == nom) {
                return Err(Erreur::Config(format!(
                    "bord « {nom} » : aucun --bord-npy de cette empreinte (fichiers : {})",
                    boite_npy::noms(bords)
                )));
            }
            continue;
        }
        return Err(Erreur::Config(format!("bord « {} » inconnu ({})", c.bord, NOMS_BORDS.join("|"))));
    }
    for (i, b) in bords.iter().enumerate() {
        if let Some(a) = bords[..i].iter().find(|a| a.nom == b.nom) {
            return Err(Erreur::Config(format!(
                "--bord-npy : deux fichiers de même empreinte courte {} ({} et {})",
                b.nom,
                a.chemin.display(),
                b.chemin.display()
            )));
        }
        if b.l_calcul != g.lc || b.liens.len() != 4 * g.n_fixes() {
            return Err(Erreur::Config(format!(
                "--bord-npy {} : réseau de côté {} et {} réels ; la boîte ℓ = {} demande le côté {} et 4·{} réels",
                b.chemin.display(),
                b.l_calcul,
                b.liens.len(),
                g.ell,
                g.lc,
                g.n_fixes()
            )));
        }
        if !cfg.chaines.iter().any(|c| c.bord == b.nom) {
            return Err(Erreur::Config(format!(
                "--bord-npy {} ({}) : fichier de bord inutilisé (aucune chaîne {}:GRAINE)",
                b.chemin.display(),
                b.nom,
                b.nom
            )));
        }
    }
    if !matches!(cfg.options.as_str(), "ref-k1" | "compat-prod") {
        return Err(Erreur::Config(format!("--options « {} » inconnu pour la boîte (ref-k1|compat-prod)", cfg.options)));
    }
    Ok(g)
}

/// Ligne de commande canonique (sans le binaire ni `--outdir`) qui rejoue exactement ce lot : utilisée par le manifeste
/// de régénération (`kvu regen` ajoute lui-même `--outdir`). Le backend et la précision sont ceux du run.
pub fn commande_canonique(cfg: &ConfigBoite, backend: BackendKind) -> Vec<String> {
    commande_canonique_bords(cfg, &[], backend)
}

/// Commande canonique d'un lot avec bords lus : les paires `--bord-npy <chemin canonique>`, dans l'ordre de `bords`, juste
/// avant `--chaines` ; sans bord lu, identique à [`commande_canonique`].
pub fn commande_canonique_bords(cfg: &ConfigBoite, bords: &[BordNpy], backend: BackendKind) -> Vec<String> {
    let chaines: Vec<String> = cfg.chaines.iter().map(|c| format!("{}:{}", c.bord, c.graine)).collect();
    let mut v: Vec<String> = vec![
        "su2".into(),
        "boite".into(),
        "--ell".into(),
        cfg.ell.to_string(),
        "--beta".into(),
        cfg.beta.to_string(),
    ];
    for b in bords {
        v.push("--bord-npy".into());
        v.push(b.chemin.to_string_lossy().into_owned());
    }
    v.extend([
        "--chaines".into(),
        chaines.join(","),
        "--therm".into(),
        cfg.therm.to_string(),
        "--nmeas".into(),
        cfg.nmeas.to_string(),
        "--skip".into(),
        cfg.skip.to_string(),
        "--nor".into(),
        cfg.nor.to_string(),
        "--options".into(),
        cfg.options.clone(),
        "--profil-tous".into(),
        cfg.profil_tous.to_string(),
        "--therm-source".into(),
        cfg.therm_source.to_string(),
        "--precision".into(),
        "f64".into(),
        "--backend".into(),
        backend.nom().into(),
    ]);
    if cfg.dump_final {
        v.push("--dump-final".into());
    }
    if cfg.mutant != AppMasque::Tout {
        v.push("--mutant".into());
        v.push(cfg.mutant.nom().into());
    }
    v
}

/// Côté du réseau d'une source thermique pour une boîte de côté `ell`.
pub fn l_source(ell: usize) -> usize {
    L_SOURCE_MIN.max(ell + 2)
}

/// Octets projetés d'un run (séries, profils, configurations finales), pour la garde SSD.
pub fn projection_octets(cfg: &ConfigBoite) -> u64 {
    let k = cfg.chaines.len() as u64;
    let ncl = (cfg.ell / 2) as u64;
    let lignes_mesures = cfg.nmeas;
    let lignes_profil = cfg.nmeas.checked_div(cfg.profil_tous).map_or(0, |n| n + 1);
    let csv = lignes_mesures.saturating_mul(64).saturating_add(lignes_profil.saturating_mul(20 * (ncl + 1)));
    let dump = if cfg.dump_final { npy::octets_champ(cfg.ell + 2, FormatChamp::Q64) } else { 0 };
    k.saturating_mul(csv.saturating_add(dump)).saturating_add(1 << 22)
}

/// Mémoire estimée d'un run : GPU = lots de champs et tampons du moteur ; hôte = champs transitoires (départ, fin, sources).
pub fn besoin_memoire(cfg: &ConfigBoite, gpu: bool) -> BesoinMemoire {
    let lc = cfg.ell + 2;
    let k = cfg.chaines.len() as u64;
    let champ = octets_champ(lc, 8);
    let nth = cfg.chaines.iter().filter(|c| c.bord == "therm").count() as u64;
    let sources = nth.saturating_mul(octets_champ(l_source(cfg.ell), 8));
    let hote_commun = 3u64.saturating_mul(champ).saturating_add(sources);
    if gpu {
        let par_chaine = champ.saturating_add(32u64.saturating_mul((lc as u64).pow(4)));
        BesoinMemoire { hote: hote_commun, gpu: par_chaine.saturating_mul(k.max(nth)) }
    } else {
        BesoinMemoire { hote: hote_commun.saturating_add(champ.saturating_mul(k)), gpu: 0 }
    }
}

/// Un pas : 1 bain thermique + `nor` surrelaxations.
fn pas(n: &mut dyn NoyauxSu2, nor: usize) -> Resultat<()> {
    n.sweep_hb()?;
    for _ in 0..nor {
        n.sweep_or()?;
    }
    Ok(())
}

/// Sources thermiques : configurations d'équilibre périodiques (sans masque) de côté `l_source(ell)` au même β, une par
/// graine de `graines` (clé = graine ⊕ [`CLE_SOURCE_THERM`]), départ chaud, `therm_source` pas.
pub fn sources_thermiques(app: Option<&Appareil>, cfg: &ConfigBoite, options: &OptionsCompil, graines: &[u64]) -> Resultat<Vec<ChampHote>> {
    if graines.is_empty() {
        return Ok(Vec::new());
    }
    let lot = ConfigLot {
        l: l_source(cfg.ell),
        chaines: graines
            .iter()
            .map(|&g| ParamChaine { beta: cfg.beta, graine: Graine(g ^ CLE_SOURCE_THERM), depart: Depart::Chaud })
            .collect(),
        nor: cfg.nor,
        precision: Precision::F64,
        options: options.clone(),
        mutant_metro: false,
    };
    let mut n = construire(app, lot)?;
    n.init()?;
    for _ in 0..cfg.therm_source {
        pas(n.as_mut(), cfg.nor)?;
    }
    n.synchroniser()?;
    (0..graines.len()).map(|c| n.telecharger(c)).collect()
}

/// Octets (little-endian) des liens gelés d'un champ, dans l'ordre des indices de lien.
fn octets_geles(champ: &ChampHote, masque: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    for (li, &m) in masque.iter().enumerate() {
        if m != 0 {
            for k in 0..4 {
                v.extend_from_slice(&champ.q[4 * li + k].to_le_bytes());
            }
        }
    }
    v
}

/// Écart maximal |Δq| entre les liens gelés de deux champs (0,0 = identiques à l'octet près : le test d'égalité est exact).
fn ecart_geles(a: &ChampHote, b: &ChampHote, masque: &[u8]) -> f64 {
    let mut e = 0.0f64;
    for (li, &m) in masque.iter().enumerate() {
        if m != 0 {
            for k in 0..4 {
                e = e.max((a.q[4 * li + k] - b.q[4 * li + k]).abs());
            }
        }
    }
    e
}

/// Une mesure d'une chaîne (profil facultatif).
struct Ligne {
    pc: f64,
    w2: f64,
    profil: Vec<f64>,
}

/// Écrit les lignes d'un tronçon de mesures : `m0` = numéro (à partir de 1) de la première, `lignes[c][r]`.
#[allow(clippy::too_many_arguments)]
fn vider(
    cfg: &ConfigBoite,
    lignes: &[Vec<Ligne>],
    m0: u64,
    csv: &mut [(std::io::BufWriter<fs::File>, PathBuf, std::io::BufWriter<fs::File>, PathBuf)],
) -> Resultat<()> {
    for (c, (wm, pm, wp, pp)) in csv.iter_mut().enumerate() {
        for (r, l) in lignes[c].iter().enumerate() {
            let m = m0 + r as u64;
            let pas_tot = cfg.therm + m * cfg.skip;
            let mut valeurs = vec![l.pc];
            if cfg.ell >= 4 {
                valeurs.push(l.w2);
            }
            sentinelle(&valeurs, c, m as usize, "mesure")?;
            ecrire_ligne(wm, pm, &format!("{},{},{}", pas_tot, g17(l.pc), g17(l.w2)))?;
            if cfg.profil_tous > 0 && m.is_multiple_of(cfg.profil_tous) {
                sentinelle(&l.profil, c, m as usize, "profil")?;
                let mut s = pas_tot.to_string();
                for v in &l.profil {
                    s.push(',');
                    s.push_str(&g17(*v));
                }
                ecrire_ligne(wp, pp, &s)?;
            }
        }
        wm.flush().map_err(|e| io("écriture", pm, e))?;
        wp.flush().map_err(|e| io("écriture", pp, e))?;
    }
    Ok(())
}

/// Exécute le lot décrit par `cfg` dans `cfg.outdir` (répertoire NEUF). `argv` : ligne de commande complète (provenance).
pub fn executer_boite(app: Option<&Appareil>, cfg: &ConfigBoite, argv: &[String]) -> Resultat<RapportBoite> {
    executer_boite_bords(app, cfg, &[], argv)
}

/// Comme [`executer_boite`], avec les bords lus (`--bord-npy`) désignés par les chaînes `npy:<h8>`.
pub fn executer_boite_bords(app: Option<&Appareil>, cfg: &ConfigBoite, bords: &[BordNpy], argv: &[String]) -> Resultat<RapportBoite> {
    let g = valider_boite_bords(cfg, bords)?;
    let backend = app.map_or(BackendKind::Cpu, |a| a.backend());
    let options = super::cli::options_compil(Some(&cfg.options), Precision::F64, backend)?;
    let gpu_unifie = app.is_some_and(|a| crate::machine::detection::arch_memoire_unifiee(&a.arch()));
    let mut besoin = besoin_memoire(cfg, app.is_some());
    // liens fixés des bords lus, gardés sur l'hôte pendant tout le run
    besoin.hote = besoin.hote.saturating_add(bords.iter().map(|b| 8 * b.liens.len() as u64).sum::<u64>());
    crate::machine::admission::admettre(
        &format!("su2 boite ℓ={} ({} chaîne(s), {})", cfg.ell, cfg.chaines.len(), backend.nom()),
        besoin,
        gpu_unifie,
    )?;
    let t0 = Instant::now();
    let verrou = match app {
        Some(_) => Some(prendre_verrou()?),
        None => None,
    };
    let garde = GardeSsd::machine();
    let mut man = Manifeste::ouvrir_avec_garde(&cfg.outdir, "kvu su2 boite", argv, &garde, projection_octets(cfg))?;
    let r = corps(app, cfg, bords, argv, &g, backend, &options, &garde, &mut man, verrou.as_ref().map(|v| v.mode()), t0);
    match r {
        Ok(rapport) => {
            man.terminer(0)?;
            Ok(rapport)
        }
        Err(e) => {
            let _ = man.echouer(e.code_sortie(), &e.to_string());
            Err(e)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn corps(
    app: Option<&Appareil>,
    cfg: &ConfigBoite,
    bords_npy: &[BordNpy],
    argv: &[String],
    g: &GeometrieBoite,
    backend: BackendKind,
    options: &OptionsCompil,
    garde: &GardeSsd,
    man: &mut Manifeste,
    mode_verrou: Option<&'static str>,
    t0: Instant,
) -> Resultat<RapportBoite> {
    let nch = cfg.chaines.len();
    let ncl = cfg.ell / 2;
    man.verrou(match (app, mode_verrou) {
        (Some(_), Some(m)) => m,
        _ => "non_determine",
    })?;
    // --- sources thermiques (un lot, sans masque) puis bords et champs de départ -------------------------------------------
    let idx_therm: Vec<usize> = (0..nch).filter(|&c| cfg.chaines[c].bord == "therm").collect();
    let graines_therm: Vec<u64> = idx_therm.iter().map(|&c| cfg.chaines[c].graine).collect();
    let sources = sources_thermiques(app, cfg, options, &graines_therm)?;
    let mut bords: Vec<BordBoite> = Vec::with_capacity(nch);
    let mut sha_sources: Vec<Value> = Vec::new();
    for (c, ch) in cfg.chaines.iter().enumerate() {
        let graine = Graine(ch.graine);
        bords.push(match ch.bord.as_str() {
            "cold" => BordBoite::Froid,
            "gauge" => BordBoite::Jauge(graine),
            "z2full" => BordBoite::Z2Plein,
            "z2_01" => BordBoite::Z2Torsion01,
            "flux" => BordBoite::Flux,
            "haar" => BordBoite::Haar(graine),
            "therm" => {
                let i = idx_therm.iter().position(|&x| x == c).ok_or_else(|| Erreur::Config("source thermique introuvable".into()))?;
                let octets: Vec<u8> = sources[i].q.iter().flat_map(|x| x.to_le_bytes()).collect();
                sha_sources.push(json!({"chaine": c, "sha256": sha256_hex(&octets)}));
                BordBoite::Therm(sources[i].clone())
            }
            nom if nom.starts_with("npy:") => {
                let b = bords_npy
                    .iter()
                    .find(|b| b.nom == nom)
                    .ok_or_else(|| Erreur::Config(format!("bord « {nom} » : aucun --bord-npy de cette empreinte")))?;
                BordBoite::Npy(b.clone())
            }
            autre => return Err(Erreur::Config(format!("bord « {autre} » inconnu"))),
        });
    }
    drop(sources);
    // --- moteur, champs de départ, masque -----------------------------------------------------------------------------------
    let lot = ConfigLot {
        l: g.lc,
        chaines: cfg
            .chaines
            .iter()
            .map(|c| ParamChaine { beta: cfg.beta, graine: Graine(c.graine), depart: Depart::Froid })
            .collect(),
        nor: cfg.nor,
        precision: Precision::F64,
        options: options.clone(),
        mutant_metro: false,
    };
    let mut n = construire(app, lot)?;
    n.init()?;
    let masque = masque_boite(g);
    let mut empreintes: Vec<String> = Vec::with_capacity(nch);
    for (c, b) in bords.iter().enumerate() {
        let champ = champ_initial(g, b)?;
        empreintes.push(sha256_hex(&octets_geles(&champ, &masque)));
        n.televerser(c, &champ)?;
    }
    n.fixer_masque(Some(&masque), cfg.mutant)?;
    n.synchroniser()?;
    let mut provenances: Vec<ProvenanceCompil> = n.provenance();
    let mesureur = match app {
        Some(a) => {
            let capacite = CHUNK_MESURES.min(cfg.nmeas as usize).max(1);
            let m = MesureBoiteGpu::nouveau(a, g, nch, capacite)?;
            provenances.push(m.provenance());
            Some(m)
        }
        None => None,
    };
    man.ajouter_compil(&provenances)?;
    // --- répertoires et fichiers de séries -------------------------------------------------------------------------------------
    let mut csv = Vec::with_capacity(nch);
    for c in 0..nch {
        let d = cfg.outdir.join(format!("c{c:03}"));
        fs::create_dir_all(&d).map_err(|e| io("création", &d, e))?;
        let (pm, pp) = (d.join("mesures.csv"), d.join("profil.csv"));
        let (mut wm, mut wp) = (creer(&pm)?, creer(&pp)?);
        ecrire_ligne(&mut wm, &pm, "pas,pc,w2")?;
        let mut entete = String::from("pas");
        for d in 0..ncl {
            entete.push_str(&format!(",d{d}"));
        }
        ecrire_ligne(&mut wp, &pp, &entete)?;
        csv.push((wm, pm, wp, pp));
    }
    // --- boucle : thermalisation puis mesures -----------------------------------------------------------------------------------
    let debut_boucle = Instant::now();
    for t in 0..cfg.therm {
        pas(n.as_mut(), cfg.nor)?;
        if t % 200 == 199 {
            n.synchroniser()?;
        }
    }
    let capacite = CHUNK_MESURES.min(cfg.nmeas as usize).max(1);
    let mut tampon: Vec<Vec<Ligne>> = (0..nch).map(|_| Vec::with_capacity(capacite)).collect();
    let mut m0 = 1u64;
    for m in 1..=cfg.nmeas {
        for _ in 0..cfg.skip {
            pas(n.as_mut(), cfg.nor)?;
        }
        let idx = tampon[0].len();
        let avec_profil = cfg.profil_tous > 0 && m.is_multiple_of(cfg.profil_tous);
        match (&mesureur, n.champ_gpu()) {
            (Some(ms), Some(champ)) => ms.mesurer(champ, idx, avec_profil)?,
            _ => {
                for (c, t) in tampon.iter_mut().enumerate() {
                    let h = n.telecharger(c)?;
                    let o = mesurer_hote(g, &h);
                    t.push(Ligne { pc: o.pc, w2: o.w2, profil: o.profil });
                }
            }
        }
        if mesureur.is_some() {
            // marque la case (les valeurs sont lues au vidage)
            for t in tampon.iter_mut() {
                t.push(Ligne { pc: 0.0, w2: 0.0, profil: Vec::new() });
            }
        }
        if tampon[0].len() == capacite || m == cfg.nmeas {
            if let Some(ms) = &mesureur {
                let (centre, profil) = ms.lire()?;
                for (c, t) in tampon.iter_mut().enumerate() {
                    for (r, l) in t.iter_mut().enumerate() {
                        let o = ms.extraire(&centre, &profil, c, r);
                        *l = Ligne { pc: o.pc, w2: o.w2, profil: o.profil };
                    }
                }
            }
            vider(cfg, &tampon, m0, &mut csv)?;
            m0 = m + 1;
            for t in tampon.iter_mut() {
                t.clear();
            }
        }
    }
    n.synchroniser()?;
    let s_boucle = debut_boucle.elapsed().as_secs_f64();
    drop(csv);
    // --- contrôles de fin : plafond, unitarité, liens gelés intacts ----------------------------------------------------------------
    let (mut plafond, mut essais) = (0u64, 0u64);
    for c in n.compteurs()? {
        plafond += c.plafond;
        essais += c.essais;
    }
    controle_plafond(plafond)?;
    let mut ecart_max = 0.0f64;
    let mut defaut_max = 0.0f64;
    let mut fichiers_sortie: Vec<PathBuf> = Vec::new();
    let mut chaines_json: Vec<Value> = Vec::new();
    for c in 0..nch {
        let fin = n.telecharger(c)?;
        defaut_max = defaut_max.max(fin.defaut_unitarite());
        let depart = sha256_hex(&octets_geles(&fin, &masque));
        let intact = depart == empreintes[c];
        if !intact {
            // écart quantitatif : reconstruit le champ de départ (déterministe) pour mesurer max|Δq|
            let init = champ_initial(g, &bords[c])?;
            ecart_max = ecart_max.max(ecart_geles(&init, &fin, &masque));
        }
        if cfg.dump_final {
            let chemin = cfg.outdir.join(format!("c{c:03}")).join("config_finale.npy");
            npy::ecrire_champ_hache(&chemin, &fin, FormatChamp::Q64, garde)?;
            fichiers_sortie.push(chemin);
        }
        chaines_json.push(json!({
            "chaine": c, "bord": cfg.chaines[c].bord, "graine": cfg.chaines[c].graine,
            "empreinte_liens_geles_depart": empreintes[c], "liens_geles_intacts": intact,
        }));
    }
    let bord_intact = ecart_max == 0.0;
    if !bord_intact && cfg.mutant == AppMasque::Tout {
        return Err(Erreur::Validation(format!(
            "liens de bord modifiés pendant le run : max|ΔU_bord| = {ecart_max:e} (attendu 0 exactement)"
        )));
    }
    if defaut_max > 1e-12 {
        return Err(Erreur::Validation(format!("champ final hors SU(2) : max| |q| − 1 | = {defaut_max:e}")));
    }
    for c in 0..nch {
        let d = cfg.outdir.join(format!("c{c:03}"));
        fichiers_sortie.push(d.join("mesures.csv"));
        fichiers_sortie.push(d.join("profil.csv"));
    }
    fichiers_sortie.sort();
    fichiers_sortie.dedup();
    for f in &fichiers_sortie {
        man.fichier(f)?;
    }
    // --- manifeste ---------------------------------------------------------------------------------------------------------
    let pas_total = cfg.therm + cfg.nmeas * cfg.skip;
    let k = nch as f64;
    let ms_par_pas_et_chaine = 1e3 * s_boucle / (pas_total as f64 * k).max(1.0);
    let liens_libres_par_s = (g.n_libres() as f64) * k * (1.0 + cfg.nor as f64) * pas_total as f64 / s_boucle.max(1e-12);
    let secondes = t0.elapsed().as_secs_f64();
    let mut champs: Vec<(&str, Value)> = vec![
        ("backend_effectif", json!(backend.nom())),
        ("precision", json!("f64")),
        ("mode", json!("su2-boite")),
        ("moteur_su2", json!("kvu::su2::boite (boîtes de Dirichlet SU(2), WP-DIR)")),
        ("config_boite", serde_json::to_value(cfg).unwrap_or(Value::Null)),
        ("geometrie", json!({"ell": g.ell, "l_calcul": g.lc, "n_libres": g.n_libres(), "n_fixes": g.n_fixes(), "n_inexistants": g.n_inexistants()})),
        ("chaines", Value::Array(chaines_json.clone())),
        ("sources_thermiques", Value::Array(sha_sources.clone())),
        ("controle_bord", json!({"intact": bord_intact, "max_dq": ecart_max, "mutant": cfg.mutant.nom()})),
        ("defaut_unitarite_max", json!(defaut_max)),
        ("plafond_atteint", json!(plafond)),
        ("essais_bain", json!(essais)),
        ("pas", json!(pas_total)),
        ("s_boucle", json!(s_boucle)),
        ("ms_par_pas_et_chaine", json!(ms_par_pas_et_chaine)),
        ("liens_libres_par_s", json!(liens_libres_par_s)),
        ("wall_s", json!(secondes)),
        ("action", json!("beta sum(1-ReTr(U_p)/2), beta = 4/g^2")),
        ("rng", json!("Philox4x32-10 compteur (lien du réseau de calcul, sweep, tirage, étiquette), clé = graine de la chaîne")),
    ];
    if !bords_npy.is_empty() {
        champs.push(("bords_npy", boite_npy::json_bords(bords_npy)));
    }
    for (k, v) in champs {
        man.champ(k, &v.to_string())?;
    }
    let regen = manifeste_regen(app, cfg, bords_npy, argv, g, backend, &provenances, man, &chaines_json, secondes)?;
    let erreurs = valider_manifeste(&regen);
    if !erreurs.is_empty() {
        return Err(Erreur::Validation(format!(
            "manifeste kvu-regen/1 invalide ({} erreur(s)) : {}",
            erreurs.len(),
            erreurs.iter().take(5).cloned().collect::<Vec<_>>().join(" ; ")
        )));
    }
    let texte = serde_json::to_string_pretty(&regen).map_err(|e| Erreur::Format(e.to_string()))?;
    let chemin = cfg.outdir.join("manifest_regen.json");
    ecrire_atomique(&chemin, (texte + "\n").as_bytes()).map_err(|e| io("écriture", &chemin, e))?;
    println!(
        "PERF boite ell={} beta={} K={} backend={} precision=f64 pas={} s_boucle={:.4} ms_par_pas_et_chaine={:.6} liens_libres_par_s={:.4e}",
        cfg.ell, cfg.beta, nch, backend.nom(), pas_total, s_boucle, ms_par_pas_et_chaine, liens_libres_par_s
    );
    Ok(RapportBoite {
        chaines: nch,
        backend,
        pas: pas_total,
        mesures: cfg.nmeas,
        s_boucle,
        secondes,
        ms_par_pas_et_chaine,
        liens_libres_par_s,
        bord_intact,
    })
}

/// Manifeste de régénération `kvu-regen/1` du run (voir `schemas/kvu-regen-1.schema.json`). `commande` = ligne canonique sans
/// binaire ni `--outdir` ; les fichiers sont les séries, profils et configurations finales (déterministes), jamais les manifestes.
#[allow(clippy::too_many_arguments)]
fn manifeste_regen(
    app: Option<&Appareil>,
    cfg: &ConfigBoite,
    bords_npy: &[BordNpy],
    _argv: &[String],
    g: &GeometrieBoite,
    backend: BackendKind,
    provenances: &[ProvenanceCompil],
    man: &Manifeste,
    chaines: &[Value],
    secondes: f64,
) -> Resultat<Value> {
    let profil = crate::machine::detecter();
    let exe = std::env::current_exe().map_err(|e| Erreur::io("chemin du binaire courant", e))?;
    let binaire = crate::stockage::provenance::sha256_binaire()?;
    let mut noyaux: Vec<Value> =
        crate::noyaux::empreintes().into_iter().map(|(nom, sha)| json!({"nom": nom, "sha256": sha})).collect();
    noyaux.push(json!({"nom": SU2_BOITE.nom, "sha256": sha256_hex(SU2_BOITE.contenu.as_bytes())}));
    let modules: Vec<Value> = provenances
        .iter()
        .map(|p| json!({"source_sha256": p.sha256_source, "options": cfg.options, "arguments": p.arguments, "objet_sha256": p.sha256_code}))
        .collect();
    let versions: serde_json::Map<String, Value> =
        profil.versions.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect();
    let fichiers: Vec<Value> = man
        .fichiers()
        .into_iter()
        .map(|(nom, sha, octets)| json!({"nom": nom, "sha256": sha, "octets": octets}))
        .collect();
    let commande = commande_canonique_bords(cfg, bords_npy, backend);
    let mut commande_origine = vec![exe.to_string_lossy().into_owned()];
    commande_origine.extend(commande.iter().cloned());
    let etiquettes: Vec<String> = cfg.chaines.iter().map(|c| format!("{}:{}", c.bord, c.graine)).collect();
    let recette = json!({"commande": commande, "graines": etiquettes, "ell": cfg.ell, "beta": cfg.beta});
    let source_hash = crate::stockage::provenance::source_hash_unifiee();
    let mut m = json!({
        "schema": "kvu-regen/1",
        "origine": {"schema": "kvu-regen/1", "manifeste": "manifest_regen.json", "sha256_manifeste": sha256_hex(recette.to_string().as_bytes())},
        "binaire_sha256": binaire,
        "binaire_provenance": "manifeste",
        "git_head": crate::stockage::provenance::git_head(),
        "source_hash": source_hash,
        "noyaux_embarques": noyaux,
        "modules_compiles": modules,
        "backend": backend.nom(),
        "arch": app.map(|a| a.arch()),
        "versions": Value::Object(versions),
        "machine": {"nom": profil.machine.nom(), "cpu": profil.cpu_modele, "gpu": app.map(|a| a.nom()), "ram": profil.ram_totale_go},
        "graines": {"cle": cfg.chaines[0].graine, "etiquettes": etiquettes},
        "parametres": {
            "groupe": "SU(2)", "L": cfg.ell, "beta": cfg.beta, "start": "cold", "therm": cfg.therm, "nmeas": cfg.nmeas,
            "skip": cfg.skip, "nor": cfg.nor, "precision": "f64", "options": cfg.options, "dump_every": 0, "bord": true,
            "boite": {
                "ell": g.ell, "l_calcul": g.lc, "n_libres": g.n_libres(), "profil_tous": cfg.profil_tous,
                "therm_source": cfg.therm_source, "masque": "boite : liens fixés, inexistants et de remplissage gelés",
                "dump_final": cfg.dump_final, "mutant": cfg.mutant.nom(), "chaines": chaines
            }
        },
        "commande": commande,
        "commande_origine": commande_origine,
        "entrees": boite_npy::json_entrees(bords_npy),
        "fichiers": fichiers,
        "fichiers_hachage": "manifeste",
        "duree_s": secondes,
        "statut": "COMPLETE",
        "regen_verifiee": {"date": null, "machine": null, "resultat": "non_verifiee"},
    });
    if !bords_npy.is_empty() {
        m["parametres"]["boite"]["bords_npy"] = boite_npy::json_bords(bords_npy);
    }
    let cl = classer(&m, &mut Contexte::nouveau());
    m["classe_regen"] = Value::String(cl.classe.nom().to_string());
    m["raisons_classe"] = json!(cl.raisons);
    Ok(m)
}

/// Observables d'une configuration de réseau de calcul lue dans un fichier NPY (sous-commande `su2 boite-obs`) :
/// `(pc, w2, profil)` sur l'hôte (référence) ou sur le GPU (noyaux), pour les gates croisées avec `box_su2.py`.
pub fn observables_fichier(app: Option<&Appareil>, chemin: &Path, ell: usize) -> Resultat<super::boite::ObsBoite> {
    let g = GeometrieBoite::nouvelle(ell)?;
    let champ = npy::lire_champ(chemin)?;
    if champ.l != g.lc {
        return Err(Erreur::Config(format!(
            "{} : champ de côté {} ; une boîte de côté ℓ = {ell} vit dans un réseau de calcul de côté {}",
            chemin.display(),
            champ.l,
            g.lc
        )));
    }
    match app {
        None => Ok(mesurer_hote(&g, &champ)),
        Some(a) => {
            let mut gpu = super::champ::ChampGpu::allouer(a, g.lc, 1, Precision::F64)?;
            gpu.televerser(a, 0, &champ)?;
            let m = MesureBoiteGpu::nouveau(a, &g, 1, 1)?;
            m.observables(&gpu, 0)
        }
    }
}

/// Empreinte sha256 d'un fichier (utilitaire des tests et du pilote).
pub fn sha_fichier(p: &Path) -> Resultat<String> {
    sha256_fichier(p).map_err(|e| io("sha256", p, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(chaines: &[(&str, u64)]) -> ConfigBoite {
        ConfigBoite {
            ell: 4,
            beta: 2.5,
            chaines: chaines.iter().map(|&(b, g)| ChaineBoite { bord: b.into(), graine: g }).collect(),
            therm: 5,
            nmeas: 10,
            skip: 1,
            nor: 2,
            options: "ref-k1".into(),
            profil_tous: 5,
            therm_source: 20,
            dump_final: false,
            mutant: AppMasque::Tout,
            outdir: PathBuf::from("/tmp/x"),
        }
    }

    #[test]
    fn validation() {
        assert!(valider_boite(&cfg(&[("cold", 1)])).is_ok());
        assert!(valider_boite(&cfg(&[])).is_err());
        assert!(valider_boite(&cfg(&[("tiede", 1)])).is_err());
        let mut c = cfg(&[("cold", 1)]);
        c.ell = 5;
        assert!(valider_boite(&c).is_err());
        c.ell = 4;
        c.nmeas = 0;
        assert!(valider_boite(&c).is_err());
        c.nmeas = 1;
        c.options = "rapide".into();
        assert!(valider_boite(&c).is_err());
        c.options = "compat-prod".into();
        c.beta = f64::NAN;
        assert!(valider_boite(&c).is_err());
    }

    #[test]
    fn commande_canonique_est_rejouable_et_sans_outdir() {
        let c = cfg(&[("cold", 7), ("z2full", 8)]);
        let v = commande_canonique(&c, BackendKind::Hip);
        assert_eq!(&v[..2], ["su2", "boite"]);
        assert!(!v.contains(&"--outdir".to_string()));
        let i = v.iter().position(|a| a == "--chaines").unwrap();
        assert_eq!(v[i + 1], "cold:7,z2full:8");
        assert!(v.windows(2).any(|w| w == ["--backend", "hip"]));
        assert!(!v.contains(&"--mutant".to_string()));
        let mut m = c.clone();
        m.mutant = AppMasque::IgnoreTout;
        m.dump_final = true;
        let v = commande_canonique(&m, BackendKind::Cpu);
        assert!(v.windows(2).any(|w| w == ["--mutant", "masque-ignore"]));
        assert!(v.contains(&"--dump-final".to_string()));
    }

    #[test]
    fn l_source_et_projection() {
        assert_eq!(l_source(2), 10);
        assert_eq!(l_source(8), 10);
        assert_eq!(l_source(12), 14);
        let c = cfg(&[("cold", 1), ("haar", 2)]);
        assert!(projection_octets(&c) > 1 << 22);
        let b = besoin_memoire(&c, true);
        assert!(b.gpu > 0 && b.hote > 0);
        assert_eq!(besoin_memoire(&c, false).gpu, 0);
    }

    #[test]
    fn octets_geles_et_ecart() {
        let g = GeometrieBoite::nouvelle(2).unwrap();
        let masque = masque_boite(&g);
        let a = champ_initial(&g, &BordBoite::Z2Plein).unwrap();
        let mut b = a.clone();
        assert_eq!(octets_geles(&a, &masque), octets_geles(&b, &masque));
        assert_eq!(ecart_geles(&a, &b, &masque), 0.0);
        // un lien libre modifié ne compte pas ; un lien gelé modifié, si
        let libre = masque.iter().position(|&m| m == 0).unwrap();
        b.q[4 * libre] = 0.5;
        assert_eq!(ecart_geles(&a, &b, &masque), 0.0);
        let gele = masque.iter().position(|&m| m != 0).unwrap();
        b.q[4 * gele] += 0.25;
        assert!((ecart_geles(&a, &b, &masque) - 0.25).abs() < 1e-15);
        assert_ne!(octets_geles(&a, &masque), octets_geles(&b, &masque));
    }
}
