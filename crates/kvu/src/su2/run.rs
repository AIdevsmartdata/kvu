//! Boucle de production (thermalisation, mesures, dumps, CSV au format su2_hip, bord, reprise,
//! mesures en ligne). Propriétaire : T3.
//!
//! `executer` reproduit EXACTEMENT la boucle hôte de su2_hip.cpp (et de su2_gen, K1) :
//! - indice 0 = état initial ; thermalisation : une ligne `therm.csv` par pas (indices 0..therm) ;
//! - production : `measurements.csv` indice 0 = fin de thermalisation (phase `initial`), puis indice m après
//!   `skip` pas (phase `sample`), colonne `total_steps` = therm + m·skip ;
//! - dumps `cfg_NNNNNN.npy` au même indice, tous les `dump_every` indices (PAS de dump supplémentaire à la
//!   fin : comme su2_hip) ; `--dump-initial` écrit `cfg_000000.npy` avant la thermalisation ;
//! - CSV en `%.17g` (`heritage::su2_hb::g17`), COMPLETE écrit EN DERNIER.
//!
//! Lots : un sous-répertoire `c000/ … c{K-1}/` par chaîne (K > 1). Mode bord : `bord::executer_bord`.
//!
//! Reprise : point de reprise tous les `reprise_every` pas (aux frontières d'itération) ; `reprendre = true`
//! relit `reprise.json` + `reprise.champs`, tronque les CSV à la position du point et continue : le résultat
//! est identique à l'octet à un run sans interruption (REPRISE-1).

use super::reprise::{self, EtatReprise};
use super::torsion::{self, Torsion};
use super::{bord, ChampGpu, Compteurs, ConfigLot, NoyauxSu2};
use crate::commun::{Depart, Erreur, Precision, Resultat};
use crate::gpu::{Appareil, ProvenanceCompil};
use crate::heritage::manifest::{self, sha256_fichier};
use crate::heritage::su2_hb::g17;
use crate::machine::GardeVerrou;
use crate::stockage::garde::projeter;
use crate::stockage::npy::{self, FormatChamp};
use crate::stockage::{compression, GardeSsd, Manifeste};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Mesure exécutée pendant la génération (implémentée par crate::obs, T7) : évite de stocker
/// les configurations des grands L (désert L = 48-64).
pub trait MesureEnLigne {
    fn nom(&self) -> &str;
    /// Mesurer tous les `periode` pas de production (0 = jamais).
    fn periode(&self) -> usize;
    /// Mesure la chaîne `chaine` du champ résident (ou du champ hôte si CPU) au pas `pas`
    /// et écrit dans `sortie` (fichiers propres à la mesure).
    fn mesurer(&mut self, app: Option<&Appareil>, champ: Option<&ChampGpu>, hote: Option<&crate::commun::ChampHote>,
               chaine: usize, pas: u64, sortie: &std::path::Path) -> Resultat<()>;
}

/// Paramètres d'un run de production (CLI `su2 gen`).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConfigRun {
    pub lot: super::ConfigLot,
    pub therm: u64,
    pub nmeas: u64,
    pub skip: u64,
    /// 0 = jamais ; sinon un dump tous les `dump_every` indices de mesure (comme su2_hip : pas de dump de plus à la fin).
    pub dump_every: u64,
    pub dump_initial: bool,
    /// Format des dumps : « c16 » (NPY complex128, compat su2_hip) ou « q64 » (quaternions f64, 2× plus petit).
    pub format_dump: String,
    pub zstd: bool,
    /// Point de reprise tous les N pas (0 = jamais).
    pub reprise_every: u64,
    pub outdir: PathBuf,
    /// Bord R2-3 : None = périodique.
    pub bord: Option<ConfigBord>,
}

/// Mode bord (arguments de su2_bord --bord).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConfigBord {
    /// « cold » | « hot » | « eq ».
    pub bc: String,
    pub bc_graine: u64,
    pub jauge: Option<u64>,
    pub pre: u64,
    pub bloc: u64,
}

/// Options additionnelles d'un run (M4 : torsion de 't Hooft).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OptionsRun {
    pub torsion: Torsion,
}

/// Rapport de fin de run.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RapportRun {
    pub pas: u64,
    pub mesures: u64,
    pub dumps: u64,
    pub plafond: u64,
    pub secondes: f64,
}

/// Durées imbriquées du même run instrumenté, sans modifier `RapportRun`.
/// Les nanosecondes sont du temps monotone écoulé, préemptions comprises.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProfilRun {
    pub duree_totale_ns: u64,
    pub duree_pas_ns: u64,
    pub pas_executes: u64,
    pub hb_executes: u64,
    pub or_executes: u64,
    pub chaines: usize,
    pub nor: usize,
    pub termine: bool,
}

impl ProfilRun {
    /// Vérifie les mesures brutes sans les borner ni les remplacer.
    pub fn valider(&self, pas_attendus: u64, chaines: usize, nor: usize, termine: bool) -> Resultat<()> {
        let or_attendus = pas_attendus.checked_mul(nor as u64)
            .ok_or_else(|| Erreur::Config("profilage : compteur OR trop grand".into()))?;
        if self.duree_totale_ns == 0 || self.duree_pas_ns == 0 || self.duree_pas_ns > self.duree_totale_ns {
            return Err(Erreur::Config("profilage : durées imbriquées invalides (0 < S ≤ D requis)".into()));
        }
        if pas_attendus == 0 || chaines == 0 || self.pas_executes != pas_attendus
            || self.hb_executes != pas_attendus || self.or_executes != or_attendus
            || self.chaines != chaines || self.nor != nor || self.termine != termine
        {
            return Err(Erreur::Config("profilage : nombre de pas, HB/OR, chaînes ou terminaison incohérent".into()));
        }
        Ok(())
    }

    pub fn secondes_totales(&self) -> f64 { self.duree_totale_ns as f64 * 1e-9 }
    pub fn secondes_pas(&self) -> f64 { self.duree_pas_ns as f64 * 1e-9 }

    fn ajouter_pas(&mut self, duree: std::time::Duration, nor: usize) -> Resultat<()> {
        let trop_grand = || Erreur::Config("profilage : durée ou compteur trop grand".into());
        let ns = u64::try_from(duree.as_nanos()).map_err(|_| trop_grand())?;
        self.duree_pas_ns = self.duree_pas_ns.checked_add(ns).ok_or_else(trop_grand)?;
        self.pas_executes = self.pas_executes.checked_add(1).ok_or_else(trop_grand)?;
        self.hb_executes = self.hb_executes.checked_add(1).ok_or_else(trop_grand)?;
        self.or_executes = self.or_executes.checked_add(nor as u64).ok_or_else(trop_grand)?;
        Ok(())
    }
}

/// Fenêtre de temps (secondes, bits d'un f64 ; 0 = aucune) au-delà de laquelle le run s'arrête proprement à la prochaine frontière
/// d'itération en écrivant un point de reprise (même effet qu'`--arret-apres`, en temps plutôt qu'en pas). Posée par `--fenetre-s`
/// (WP10 : fenêtres GPU ≤ 10 min sous verrou, GATES §0.7). Le temps compte depuis le début de `executer` (attente d'un verrou interne
/// et relecture du point de reprise comprises) ; au moins une itération est faite par fenêtre, donc le run progresse toujours.
static FENETRE_S: AtomicU64 = AtomicU64::new(0);

/// Pose la fenêtre (secondes) ; une valeur non finie ou ≤ 0 la retire.
pub fn fixer_fenetre(secondes: f64) {
    let bits = if secondes.is_finite() && secondes > 0.0 { secondes.to_bits() } else { 0 };
    FENETRE_S.store(bits, Ordering::Relaxed);
}

/// Vrai si la fenêtre `fenetre_s` (> 0) est écoulée (fonction pure, testée).
pub fn fenetre_depassee(ecoule_s: f64, fenetre_s: f64) -> bool {
    fenetre_s > 0.0 && ecoule_s >= fenetre_s
}

/// Période d'écriture des CSV (en lignes), celle de su2_hip.
pub(super) const FL: u64 = 200;
pub const NOM_CONFIG_RUN: &str = "config_run.json";

// ------------------------------------------------------------------------------------------
// Utilitaires partagés avec bord.rs
// ------------------------------------------------------------------------------------------

pub(super) fn io(contexte: &str, chemin: &Path, e: std::io::Error) -> Erreur {
    Erreur::io(format!("{contexte} {}", chemin.display()), e)
}

/// Format de dump demandé (« c16 » | « q64 »).
pub fn format_champ(nom: &str) -> Resultat<FormatChamp> {
    match nom {
        "c16" => Ok(FormatChamp::C16),
        "q64" => Ok(FormatChamp::Q64),
        autre => Err(Erreur::Config(format!("format de dump « {autre} » inconnu (c16|q64)"))),
    }
}

/// Octets d'une configuration de L donné au format de dump (en-tête NPY compris).
pub fn octets_dump(l: usize, format: &str) -> u64 {
    npy::octets_champ(l, format_champ(format).unwrap_or(FormatChamp::C16))
}

/// Projection des octets écrits par un run (dumps + reprises + CSV), refusée AVANT toute écriture
/// (`stockage::garde::projeter` : c16 L = 64 × 20 dumps = 86 Go, refusé par la garde SSD).
pub fn projection_octets(cfg: &ConfigRun) -> u64 {
    let ndumps = cfg.nmeas.checked_div(cfg.dump_every).unwrap_or(0) + u64::from(cfg.dump_initial);
    let nreprises = if cfg.reprise_every > 0 { 2 } else { 0 };
    projeter(cfg.lot.l, cfg.lot.chaines.len(), ndumps, format_champ(&cfg.format_dump).unwrap_or(FormatChamp::C16), nreprises)
}

/// Volume TOTAL écrit par les points de reprise d'un run (octets) : un champ brut f64 (128·L⁴) par chaîne à chaque
/// point (tous les `reprise_every` pas). La projection de la garde SSD compte l'OCCUPATION (deux copies au plus) ;
/// ce volume-ci compte l'USURE et le temps d'écriture (vérificateur du 04/10 : `--reprise-every 1` à L = 48 écrit
/// 680 Mo par pas, ~2 To pour 3000 pas, sans un mot).
pub fn octets_ecrits_reprises(cfg: &ConfigRun) -> u64 {
    if cfg.reprise_every == 0 {
        return 0;
    }
    let pas = cfg.therm.saturating_add(cfg.nmeas.saturating_mul(cfg.skip));
    let points = pas / cfg.reprise_every;
    let champ = 128u64.saturating_mul((cfg.lot.l as u64).saturating_pow(4));
    points.saturating_mul(champ).saturating_mul(cfg.lot.chaines.len() as u64)
}

/// Seuil (Go) du volume écrit par les reprises au-delà duquel le run est signalé.
pub const SEUIL_AVERTISSEMENT_REPRISES_GO: f64 = 100.0;

/// Avertissement si les points de reprise écrivent plus de [`SEUIL_AVERTISSEMENT_REPRISES_GO`] Go, avec la période
/// qui ramènerait ce volume sous le seuil.
pub fn avertissement_reprises(cfg: &ConfigRun) -> Option<String> {
    let go = crate::stockage::garde::OCTETS_PAR_GO;
    let volume = octets_ecrits_reprises(cfg) as f64 / go;
    if volume <= SEUIL_AVERTISSEMENT_REPRISES_GO {
        return None;
    }
    let par_point = 128.0 * (cfg.lot.l as f64).powi(4) * cfg.lot.chaines.len() as f64 / go;
    let pas = cfg.therm + cfg.nmeas * cfg.skip;
    let periode = ((pas as f64 * par_point) / SEUIL_AVERTISSEMENT_REPRISES_GO).ceil() as u64;
    Some(format!(
        "AVERTISSEMENT : --reprise-every {} écrit {par_point:.2} Go par point de reprise, soit {volume:.0} Go au total \
         sur {pas} pas (usure du SSD, temps d'écriture) ; --reprise-every {} ramène ce volume sous {:.0} Go",
        cfg.reprise_every,
        periode.max(cfg.reprise_every + 1),
        SEUIL_AVERTISSEMENT_REPRISES_GO
    ))
}

/// Mémoire estimée d'un run : sur GPU, le champ de chaque chaîne (précision du lot) et les tampons par site
/// (plaquettes, tranches, réductions : 4 réels f64 par site) ; sur l'hôte, le champ f64 rapatrié pour les dumps et
/// les reprises, son encodage, et les mesures en ligne (trois champs f64). En CPU, tout est sur l'hôte.
pub fn besoin_memoire(cfg: &ConfigRun, gpu: bool) -> crate::machine::admission::BesoinMemoire {
    use crate::machine::admission::{octets_champ, BesoinMemoire};
    let nch = cfg.lot.chaines.len() as u64;
    let v = (cfg.lot.l as u64).saturating_pow(4);
    let octets_reel = match cfg.lot.precision {
        Precision::F64 => 8,
        Precision::F32 => 4,
    };
    let par_chaine = octets_champ(cfg.lot.l, octets_reel).saturating_add(32u64.saturating_mul(v));
    let hote = 3u64.saturating_mul(octets_champ(cfg.lot.l, 8));
    if gpu {
        BesoinMemoire { hote, gpu: par_chaine.saturating_mul(nch) }
    } else {
        BesoinMemoire { hote: hote.saturating_add(2u64.saturating_mul(octets_champ(cfg.lot.l, 8)).saturating_mul(nch)), gpu: 0 }
    }
}

/// Répertoire de la chaîne c : `outdir` si le lot n'a qu'une chaîne, sinon `outdir/c000/`, `c001/`…
pub fn repertoire_chaine(outdir: &Path, nchaines: usize, c: usize) -> PathBuf {
    if nchaines == 1 {
        outdir.to_path_buf()
    } else {
        outdir.join(format!("c{c:03}"))
    }
}

/// Tronque un fichier texte à ses `n` premières lignes (reprise : retire les lignes écrites après le point).
pub(super) fn tronquer_lignes(chemin: &Path, n: usize) -> Resultat<()> {
    let txt = fs::read(chemin).map_err(|e| io("lecture", chemin, e))?;
    let mut vus = 0usize;
    let mut fin = txt.len();
    for (i, &b) in txt.iter().enumerate() {
        if b == b'\n' {
            vus += 1;
            if vus == n {
                fin = i + 1;
                break;
            }
        }
    }
    if vus < n {
        return Err(Erreur::Format(format!("{} : {vus} lignes au lieu des {n} attendues au point de reprise", chemin.display())));
    }
    let f = OpenOptions::new().write(true).open(chemin).map_err(|e| io("ouverture", chemin, e))?;
    f.set_len(fin as u64).map_err(|e| io("troncature", chemin, e))
}

pub(super) fn ouvrir_ajout(chemin: &Path) -> Resultat<BufWriter<File>> {
    let f = OpenOptions::new().append(true).open(chemin).map_err(|e| io("ouverture", chemin, e))?;
    Ok(BufWriter::new(f))
}

pub(super) fn creer(chemin: &Path) -> Resultat<BufWriter<File>> {
    Ok(BufWriter::new(File::create(chemin).map_err(|e| io("création", chemin, e))?))
}

pub(super) fn ecrire_ligne(f: &mut BufWriter<File>, chemin: &Path, ligne: &str) -> Resultat<()> {
    writeln!(f, "{ligne}").map_err(|e| io("écriture", chemin, e))
}

// ------------------------------------------------------------------------------------------
// Manifeste du run (stockage::provenance::Manifeste, T8 : répertoire neuf, COMPLETE en dernier)
// ------------------------------------------------------------------------------------------

/// Manifeste d'un run. Sur un répertoire NEUF, c'est `stockage::Manifeste` (garde SSD avec projection avant toute
/// création, provenance complète, COMPLETE en dernier si rc = 0). À la REPRISE, le manifeste existant est complété
/// (statut, rc, fichiers) par une mise à jour atomique du JSON : `Manifeste` ne sait pas rouvrir un répertoire.
pub(super) struct ManifesteRun {
    neuf: Option<Manifeste>,
    dir: PathBuf,
    /// Champs à fusionner à la clôture d'un run repris.
    ajouts: serde_json::Map<String, Value>,
    fichiers: Vec<(String, String, u64)>,
}

impl ManifesteRun {
    pub(super) fn ouvrir(dir: &Path, argv: &[String], projection: u64) -> Resultat<ManifesteRun> {
        let m = Manifeste::ouvrir_avec_garde(dir, "kvu su2", argv, &GardeSsd::machine(), projection)?;
        Ok(ManifesteRun { neuf: Some(m), dir: dir.to_path_buf(), ajouts: serde_json::Map::new(), fichiers: Vec::new() })
    }

    pub(super) fn reprendre(dir: &Path) -> Resultat<ManifesteRun> {
        let p = dir.join(manifest::NOM_MANIFESTE);
        if !p.exists() {
            return Err(Erreur::Config(format!("{} : pas de manifest.json, ce n'est pas un répertoire de run", dir.display())));
        }
        if dir.join(manifest::NOM_COMPLETE).exists() {
            return Err(Erreur::Config(format!("{} porte déjà COMPLETE : rien à reprendre", dir.display())));
        }
        Ok(ManifesteRun { neuf: None, dir: dir.to_path_buf(), ajouts: serde_json::Map::new(), fichiers: Vec::new() })
    }

    /// Ajoute ou remplace un champ (écrit tout de suite pour un run neuf). Les clés d'invariants du manifeste
    /// (`statut`, `rc`, `fichiers`, `modules_compiles`…) ne se posent pas par ici.
    pub(super) fn champ(&mut self, cle: &str, v: Value) -> Resultat<()> {
        match &mut self.neuf {
            Some(m) => m.champ(cle, &v.to_string()),
            None => {
                self.ajouts.insert(cle.to_string(), v);
                Ok(())
            }
        }
    }

    pub(super) fn compil(&mut self, p: &[ProvenanceCompil]) -> Resultat<()> {
        match &mut self.neuf {
            Some(m) => m.ajouter_compil(p),
            None => {
                self.ajouts.insert("modules_compiles".into(), serde_json::to_value(p).map_err(|e| Erreur::Format(e.to_string()))?);
                Ok(())
            }
        }
    }

    pub(super) fn verrou(&mut self, garde: Option<&GardeVerrou>) -> Resultat<()> {
        let etat = match garde {
            Some(g) => g.mode(),
            None if std::env::var("KEVINOTRON_VERROU").is_ok_and(|v| v == "ignorer") => "ignore",
            None => "non_determine",
        };
        match &mut self.neuf {
            Some(m) => m.verrou(etat),
            None => {
                self.ajouts.insert("verrou".into(), json!(etat));
                Ok(())
            }
        }
    }

    pub(super) fn fichier(&mut self, chemin: &Path, sha256: &str, octets: u64) -> Resultat<()> {
        let nom = chemin.strip_prefix(&self.dir).unwrap_or(chemin).to_string_lossy().into_owned();
        self.fichiers.retain(|f| f.0 != nom);
        self.fichiers.push((nom, sha256.to_string(), octets));
        match &mut self.neuf {
            Some(m) => m.fichier_connu(chemin, sha256, octets),
            None => Ok(()),
        }
    }

    /// Clôture : statut, rc, fin ; COMPLETE EN DERNIER, et seulement si rc = 0 (et si les fichiers consignés
    /// existent avec la taille consignée).
    pub(super) fn terminer(&mut self, rc: i32) -> Resultat<()> {
        match self.neuf.take() {
            Some(m) => m.terminer(rc),
            None => self.terminer_repris(rc, None),
        }
    }

    pub(super) fn echec(&mut self, rc: i32, raison: &str) -> Resultat<()> {
        match self.neuf.take() {
            Some(m) => m.echouer(if rc == 0 { 1 } else { rc }, raison),
            None => self.terminer_repris(if rc == 0 { 1 } else { rc }, Some(raison)),
        }
    }

    fn terminer_repris(&mut self, rc: i32, raison: Option<&str>) -> Resultat<()> {
        let p = self.dir.join(manifest::NOM_MANIFESTE);
        let txt = fs::read_to_string(&p).map_err(|e| io("lecture", &p, e))?;
        let mut j: Value = serde_json::from_str(&txt).map_err(|e| Erreur::Format(format!("{} illisible : {e}", p.display())))?;
        let obj = j.as_object_mut().ok_or_else(|| Erreur::Format("manifest.json n'est pas un objet".into()))?;
        for (k, v) in std::mem::take(&mut self.ajouts) {
            obj.insert(k, v);
        }
        let fichiers: Vec<Value> = self.fichiers.iter().map(|(n, s, o)| json!({"nom": n, "sha256": s, "octets": o})).collect();
        obj.insert("fichiers".into(), Value::Array(fichiers));
        let reussi = rc == 0 && raison.is_none();
        if reussi {
            archiver_echec_precedent(obj);
        }
        obj.insert("statut".into(), json!(if reussi { "termine" } else { "echec" }));
        obj.insert("rc".into(), json!(rc));
        if let Some(r) = raison {
            obj.insert("raison_echec".into(), json!(r));
        }
        obj.insert("fin_unix".into(), json!(manifest::maintenant_unix()));
        let out = serde_json::to_vec_pretty(&j).map_err(|e| Erreur::Format(e.to_string()))?;
        manifest::ecrire_atomique(&p, &out).map_err(|e| io("écriture", &p, e))?;
        if reussi {
            manifest::ecrire_atomique(&self.dir.join(manifest::NOM_COMPLETE), b"toutes les mesures et configurations demandees sont ecrites\n")
                .map_err(|e| io("écriture", &self.dir.join(manifest::NOM_COMPLETE), e))?;
        }
        Ok(())
    }
}

/// À la clôture RÉUSSIE d'un run repris, déplace la raison d'un échec antérieur (`raison_echec`, avec le `rc` et le `fin_unix` d'alors)
/// dans la liste `fautes_anterieures` : une fenêtre qui a échoué (faute GPU, par exemple) laisse son `raison_echec` dans le manifeste,
/// et le manifeste du run finalement terminé ne doit pas dire « termine, rc 0 » à côté de la raison d'un échec. L'historique est
/// conservé, il n'est plus pris pour l'état du run. Appelée AVANT l'écriture du nouveau `rc` et du nouveau `fin_unix`.
fn archiver_echec_precedent(obj: &mut serde_json::Map<String, Value>) {
    let Some(raison) = obj.remove("raison_echec") else { return };
    let mut anterieures = match obj.remove("fautes_anterieures") {
        Some(Value::Array(a)) => a,
        _ => Vec::new(),
    };
    anterieures.push(json!({
        "rc": obj.get("rc").cloned().unwrap_or(Value::Null),
        "raison_echec": raison,
        "fin_unix": obj.get("fin_unix").cloned().unwrap_or(Value::Null),
    }));
    obj.insert("fautes_anterieures".into(), Value::Array(anterieures));
}

// ------------------------------------------------------------------------------------------
// Contexte d'un run
// ------------------------------------------------------------------------------------------

/// Fin d'une boucle : terminée ou interrompue sur demande (point de reprise écrit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fin {
    Termine,
    Interrompu,
}

pub(super) struct CtxRun<'a> {
    pub app: Option<&'a Appareil>,
    pub cfg: &'a ConfigRun,
    pub n: Box<dyn NoyauxSu2>,
    pub dirs: Vec<PathBuf>,
    pub reprendre: bool,
    pub arret_apres: Option<u64>,
    pub man: ManifesteRun,
    /// Compteurs déjà accumulés avant la reprise (le GPU repart de zéro).
    pub compteurs_avant: Vec<Compteurs>,
    pub rapport: RapportRun,
    /// sha256 et taille des dumps écrits PAR CE PROCESSUS (calculés pendant l'écriture : pas de relecture).
    pub sorties: BTreeMap<PathBuf, (String, u64)>,
    garde: GardeSsd,
    t_debut: Instant,
    dernier_progres: Instant,
    /// pas déjà faits au démarrage de CE processus (reprise) : le débit ne compte que les pas faits ici
    pub pas_au_depart: u64,
    /// Présent seulement dans l'API de banc ; le chemin de production ne chronomètre pas les pas.
    profilage: Option<&'a mut ProfilRun>,
}

impl CtxRun<'_> {
    pub(super) fn nch(&self) -> usize {
        self.cfg.lot.chaines.len()
    }

    /// Effectue un pas : 1 bain thermique + nor OverRelaxation (step() de su2_hip.cpp).
    pub(super) fn pas(&mut self) -> Resultat<()> {
        let debut = if self.profilage.is_some() {
            // Travail antérieur drainé hors S, mais toujours dans D.
            self.n.synchroniser()?;
            Some(Instant::now())
        } else { None };
        self.n.sweep_hb()?;
        for _ in 0..self.cfg.lot.nor {
            self.n.sweep_or()?;
        }
        if let (Some(debut), Some(profil)) = (debut, self.profilage.as_deref_mut()) {
            // S inclut la fin réelle des HB/OR, même sur un GPU asynchrone.
            self.n.synchroniser()?;
            profil.ajouter_pas(debut.elapsed(), self.cfg.lot.nor)?;
        }
        Ok(())
    }

    /// Point de reprise atomique (garde SSD avant écriture).
    pub(super) fn point_de_reprise(&mut self, pas: u64, mesures: u64) -> Resultat<()> {
        let l = self.cfg.lot.l;
        self.garde.verifier(&self.cfg.outdir, 2 * reprise::octets_champs(l, self.nch()))?;
        reprise::ecrire(&self.cfg.outdir, self.n.as_ref(), pas, mesures)
    }

    /// Ligne de progression sur stderr, au plus une toutes les 30 s (runs longs : L = 32-64, 10⁴ pas).
    pub(super) fn progres(&mut self, pas: u64, total: u64) {
        if self.dernier_progres.elapsed().as_secs() < 30 {
            return;
        }
        self.dernier_progres = Instant::now();
        let fait = pas.saturating_sub(self.pas_au_depart) as f64;
        let debit = fait / self.t_debut.elapsed().as_secs_f64().max(1e-9);
        let reste = if debit > 0.0 { total.saturating_sub(pas) as f64 / debit } else { f64::INFINITY };
        eprintln!(
            "# progression : pas {pas}/{total} ({:.1} %), {debit:.2} pas/s, reste ~{:.0} min",
            100.0 * pas as f64 / total.max(1) as f64,
            reste / 60.0
        );
    }

    pub(super) fn plafond_total(&self) -> Resultat<(u64, u64)> {
        let c = self.n.compteurs()?;
        let mut plafond = 0;
        let mut essais = 0;
        for (i, x) in c.iter().enumerate() {
            let a = self.compteurs_avant.get(i).copied().unwrap_or_default();
            plafond += x.plafond + a.plafond;
            essais += x.essais + a.essais;
        }
        Ok((plafond, essais))
    }

    /// Écrit le dump `idx` de chaque chaîne (atomique, garde SSD avant le premier octet). Un champ F32 est
    /// renormalisé EN F64 pour l'écriture (le format SU(2) l'exige) : l'état de la chaîne n'est pas modifié.
    pub(super) fn dumper(&mut self, idx: u64) -> Resultat<()> {
        self.n.synchroniser()?;
        let format = format_champ(&self.cfg.format_dump)?;
        for c in 0..self.nch() {
            let mut champ = self.n.telecharger(c)?;
            if self.n.precision() == Precision::F32 {
                npy::renormaliser(&mut champ)?;
            }
            let chemin = self.dirs[c].join(format!("cfg_{idx:06}.npy"));
            let r = npy::ecrire_champ_hache(&chemin, &champ, format, &self.garde)?;
            let (chemin, sha, octets) = if self.cfg.zstd {
                let z = compression::compresser(&chemin)?;
                let sha = sha256_fichier(&z).map_err(|e| io("sha256", &z, e))?;
                let octets = fs::metadata(&z).map_err(|e| io("stat", &z, e))?.len();
                (z, sha, octets)
            } else {
                (chemin, r.sha256.unwrap_or_default(), r.octets)
            };
            self.sorties.insert(chemin, (sha, octets));
        }
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------
// Validation de la configuration
// ------------------------------------------------------------------------------------------

pub fn valider_run(cfg: &ConfigRun) -> Resultat<()> {
    super::valider(&cfg.lot)?;
    if cfg.skip < 1 {
        return Err(Erreur::Config("skip ≥ 1 requis".into()));
    }
    format_champ(&cfg.format_dump)?;
    if cfg.zstd && !compression::disponible() {
        return Err(Erreur::Config("--zstd : le binaire système `zstd` est absent".into()));
    }
    if let Some(b) = &cfg.bord {
        if cfg.lot.l < 4 {
            return Err(Erreur::Config("le bord exige L ≥ 4".into()));
        }
        if !matches!(b.bc.as_str(), "cold" | "hot" | "eq") {
            return Err(Erreur::Config(format!("--bc « {} » inconnu (cold|hot|eq)", b.bc)));
        }
        if b.bloc < 1 || !cfg.nmeas.is_multiple_of(b.bloc) {
            return Err(Erreur::Config("nmeas doit être un multiple de bloc (≥ 1)".into()));
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------
// Exécution
// ------------------------------------------------------------------------------------------

/// Exécute un run complet dans `cfg.outdir` (répertoire NEUF, manifeste via crate::stockage,
/// COMPLETE écrit en dernier), ou le reprend si `reprendre` est vrai. Utilise les options par défaut (pas de torsion).
pub fn executer(app: Option<&Appareil>, cfg: &ConfigRun, mesures: &mut [Box<dyn MesureEnLigne>], reprendre: bool)
    -> Resultat<RapportRun> {
    executer_jusqua(app, cfg, mesures, reprendre, None).map(|(r, _)| r)
}

/// Comme `executer`, avec un arrêt demandé : après `arret_apres` pas (à la première frontière d'itération qui
/// convient, alignée sur un bloc en mode bord), un point de reprise est écrit et le run s'arrête SANS COMPLETE
/// (le manifeste reste « en_cours » avec le champ `interruption`). Sert aux tests de reprise (REPRISE-1) et à
/// `--arret-apres`. Rend (rapport, terminé).
pub fn executer_jusqua(app: Option<&Appareil>, cfg: &ConfigRun, mesures: &mut [Box<dyn MesureEnLigne>], reprendre: bool,
                       arret_apres: Option<u64>) -> Resultat<(RapportRun, bool)> {
    executer_avec_verrou(app, cfg, mesures, reprendre, arret_apres, None, None, &OptionsRun::default())
}

/// Nouveau point d'entrée avec options (M4) : permet de passer une torsion.
pub fn executer_options(app: Option<&Appareil>, cfg: &ConfigRun, opts: &OptionsRun, mesures: &mut [Box<dyn MesureEnLigne>],
                        reprendre: bool, arret_apres: Option<u64>) -> Resultat<(RapportRun, bool)> {
    executer_avec_verrou(app, cfg, mesures, reprendre, arret_apres, None, None, opts)
}

/// Run complet sous un verrou GPU DÉJÀ tenu par l'appelant dans ce processus (`kevinotron bench` mesure le temps
/// mur de su2 gen dans sa fenêtre) : un second flock sur un autre descripteur du même processus attendrait le
/// premier jusqu'à expiration.
pub fn executer_sous_verrou(app: Option<&Appareil>, cfg: &ConfigRun, mesures: &mut [Box<dyn MesureEnLigne>],
                            garde: &GardeVerrou) -> Resultat<RapportRun> {
    executer_avec_verrou(app, cfg, mesures, false, None, Some(garde), None, &OptionsRun::default()).map(|(r, _)| r)
}

/// Nouveau run périodique instrumenté. D couvre l'appel interne jusqu'à son retour,
/// clôture et destructeurs compris ; le verrou GPU doit déjà être acquis (attente exclue).
/// Les synchronisations/horloges ajoutées ne sont pas soustraites. Ce débit n'est pas
/// présenté comme celui du chemin de production non instrumenté.
pub fn executer_profile(app: Option<&Appareil>, cfg: &ConfigRun, mesures: &mut [Box<dyn MesureEnLigne>],
                        arret_apres: Option<u64>, externe: Option<&GardeVerrou>)
    -> Resultat<(RapportRun, bool, ProfilRun)> {
    if cfg.bord.is_some() {
        return Err(Erreur::Config("profilage du banc : run périodique requis".into()));
    }
    if app.is_some() && externe.is_none() {
        return Err(Erreur::Config("profilage GPU : verrou déjà acquis requis".into()));
    }
    let prevus = cfg.nmeas.checked_mul(cfg.skip).and_then(|n| n.checked_add(cfg.therm));
    if prevus.is_none_or(|n| n == 0) {
        return Err(Erreur::Config("profilage : nombre positif de pas représentable requis".into()));
    }
    let mut profil = ProfilRun { chaines: cfg.lot.chaines.len(), nor: cfg.lot.nor, ..ProfilRun::default() };
    let debut = Instant::now();
    let (rapport, termine) = executer_avec_verrou(app, cfg, mesures, false, arret_apres, externe, Some(&mut profil), &OptionsRun::default())?;
    profil.duree_totale_ns = u64::try_from(debut.elapsed().as_nanos())
        .map_err(|_| Erreur::Config("profilage : durée totale trop grande".into()))?;
    profil.termine = termine;
    profil.valider(rapport.pas, cfg.lot.chaines.len(), cfg.lot.nor, termine)?;
    Ok((rapport, termine, profil))
}

// M4 (intégrateur) : `opts` s'ajoute aux sept paramètres du point d'entrée commun (privé) ; les points d'entrée publics sont inchangés.
#[allow(clippy::too_many_arguments)]
fn executer_avec_verrou(app: Option<&Appareil>, cfg: &ConfigRun, mesures: &mut [Box<dyn MesureEnLigne>], reprendre: bool,
                        arret_apres: Option<u64>, externe: Option<&GardeVerrou>, profilage: Option<&mut ProfilRun>,
                        opts: &OptionsRun) -> Resultat<(RapportRun, bool)> {
    // M4 : tous les points d'entrée, y compris ceux sans OptionsRun, passent ici.
    // Lire et comparer avant les admissions, verrous, manifestes et troncatures de CSV.
    if reprendre {
        let sauvees = charger_options(&cfg.outdir)?;
        if opts.torsion != sauvees.torsion {
            return Err(Erreur::Config(format!(
                "reprise : torsion demandée ({}) différente de la torsion du run ({}) dans {}",
                opts.torsion.texte(), sauvees.torsion.texte(), cfg.outdir.display()
            )));
        }
    }
    valider_run(cfg)?;
    // Refus AVANT toute écriture pour torsion non triviale
    if !opts.torsion.est_triviale() {
        if cfg.lot.precision != Precision::F64 {
            return Err(Erreur::Config("torsion de 't Hooft incompatible avec la précision f32 (CPU F64 requis)".into()));
        }
        if cfg.lot.chaines.iter().any(|c| c.beta == 0.0) {
            return Err(Erreur::Config("torsion de 't Hooft : β = 0 interdit (bain thermique historique singulier)".into()));
        }
        if cfg.bord.is_some() {
            return Err(Erreur::Config("torsion de 't Hooft incompatible avec le mode bord (--bord)".into()));
        }
        if !mesures.is_empty() {
            return Err(Erreur::Config("torsion de 't Hooft incompatible avec les mesures en ligne (--mesures)".into()));
        }
        if app.is_some() {
            return Err(Erreur::Config("torsion de 't Hooft : les noyaux GPU ne portent pas la torsion (backend CPU requis)".into()));
        }
    }
    // Admission mémoire AVANT le verrou et toute écriture (rc 5) : sur une APU les tampons GPU sont pris sur la RAM.
    let gpu_unifie = app.is_some_and(|a| crate::machine::detection::arch_memoire_unifiee(&a.arch()));
    crate::machine::admission::admettre(
        &format!(
            "su2 gen L={} ({} chaîne(s), {}, {})",
            cfg.lot.l,
            cfg.lot.chaines.len(),
            cfg.lot.precision.nom(),
            app.map_or("cpu".to_string(), |a| a.backend().nom().to_string())
        ),
        besoin_memoire(cfg, app.is_some()),
        gpu_unifie,
    )?;
    if let Some(a) = avertissement_reprises(cfg) {
        eprintln!("su2 gen : {a}");
    }
    let t0 = Instant::now();
    let nch = cfg.lot.chaines.len();
    // verrou GPU (machine::VerrouGpu, T1) : tenu jusqu'à la fin du run, sauf s'il l'est déjà par un ancêtre
    let pris = match (app, externe) {
        (Some(_), None) => Some(prendre_verrou()?),
        _ => None,
    };
    let verrou = externe.or(pris.as_ref());
    let garde = GardeSsd::machine();
    if reprendre {
        if !cfg.outdir.is_dir() {
            return Err(Erreur::Config(format!("--reprendre : {} n'existe pas", cfg.outdir.display())));
        }
    } else {
        // refus AVANT toute écriture (rc 7) : la projection totale du run ne doit pas passer sous le seuil
        garde.verifier(&cfg.outdir, projection_octets(cfg))?;
    }
    let dirs: Vec<PathBuf> = (0..nch).map(|c| repertoire_chaine(&cfg.outdir, nch, c)).collect();
    let mut n = super::construire(app, forcer_depart_bord(cfg))?;
    // Appliquer la torsion (peut échouer si gel/masque actif, mais normalement déjà filtré)
    n.fixer_torsion(&opts.torsion)?;
    let man = if reprendre {
        let mut m = ManifesteRun::reprendre(&cfg.outdir)?;
        m.verrou(verrou)?;
        if !opts.torsion.est_triviale() {
            // Complète aussi les conventions d'un run tordu créé avant r3.
            m.champ("polyakov", conventions_polyakov_torsion())?;
        }
        m
    } else {
        let argv: Vec<String> = std::env::args().collect();
        let mut m = ManifesteRun::ouvrir(&cfg.outdir, &argv, projection_octets(cfg))?;
        let cfg_json = serde_json::to_vec_pretty(cfg).map_err(|e| Erreur::Format(e.to_string()))?;
        manifest::ecrire_atomique(&cfg.outdir.join(NOM_CONFIG_RUN), &cfg_json).map_err(|e| io("écriture", &cfg.outdir, e))?;
        if !opts.torsion.est_triviale() {
            let torsion_json = serde_json::to_vec_pretty(&opts.torsion.json()).map_err(|e| Erreur::Format(e.to_string()))?;
            manifest::ecrire_atomique(&cfg.outdir.join(torsion::NOM_FICHIER_TORSION), &torsion_json).map_err(|e| io("écriture", &cfg.outdir, e))?;
        }
        for d in &dirs {
            if d != &cfg.outdir {
                fs::create_dir_all(d).map_err(|e| io("création", d, e))?;
            }
        }
        m.verrou(verrou)?;
        m.compil(&n.provenance())?;
        for (k, v) in champs_manifeste(app, cfg, n.as_ref(), opts) {
            m.champ(&k, v)?;
        }
        m
    };
    let mut ctx = CtxRun {
        app,
        cfg,
        n,
        dirs,
        reprendre,
        arret_apres,
        man,
        compteurs_avant: vec![Compteurs::default(); nch],
        rapport: RapportRun::default(),
        sorties: BTreeMap::new(),
        garde,
        t_debut: t0,
        dernier_progres: t0,
        pas_au_depart: 0,
        profilage,
    };
    let r = if cfg.bord.is_some() { bord::executer_bord(&mut ctx) } else { boucle_standard(&mut ctx, mesures) };
    ctx.rapport.secondes = t0.elapsed().as_secs_f64();
    let r = r.and_then(|fin| match fin {
        Fin::Termine => controle_plafond(ctx.rapport.plafond).map(|()| fin),
        Fin::Interrompu => Ok(fin),
    });
    // Dernier contrôle d'erreur du GPU AVANT de clore le manifeste : la faute asynchrone des derniers noyaux, que rien n'a relue,
    // fait échouer le run (code 3, statut « echec ») au lieu d'être écrite « termine ».
    let r = r.and_then(|fin| ctx.n.synchroniser().map(|()| fin));
    match r {
        Ok(Fin::Termine) => {
            ctx.man.champ("wall_s", json!(ctx.rapport.secondes))?;
            ctx.man.champ("plafond_atteint", json!(ctx.rapport.plafond))?;
            ctx.man.terminer(0)?;
            Ok((ctx.rapport.clone(), true))
        }
        Ok(Fin::Interrompu) => {
            ctx.man.champ("interruption", json!({"pas": ctx.rapport.pas, "reprise_possible": true}))?;
            Ok((ctx.rapport.clone(), false))
        }
        Err(e) => {
            let _ = ctx.man.echec(e.code_sortie(), &e.to_string());
            Err(e)
        }
    }
}

/// Le mode bord démarre toujours chaud (su2_bord : `ac.hot = true`).
fn forcer_depart_bord(cfg: &ConfigRun) -> ConfigLot {
    let mut lot = cfg.lot.clone();
    if cfg.bord.is_some() {
        for c in &mut lot.chaines {
            c.depart = Depart::Chaud;
        }
    }
    lot
}

/// Prend le verrou GPU de la machine (machine::VerrouGpu, T1) ; tenu jusqu'à la fin du run.
pub(super) fn prendre_verrou() -> Resultat<GardeVerrou> {
    let profil = crate::machine::detection::detecter_rapide();
    let pol = crate::machine::politique(&profil);
    crate::machine::VerrouGpu::nouveau(&pol.verrou_gpu).acquerir(std::time::Duration::from_secs(pol.attente_verrou_s))
}

/// Charge la configuration d'un run existant (`su2 gen --reprendre DIR`).
pub fn charger_config(dir: &Path) -> Resultat<ConfigRun> {
    let p = dir.join(NOM_CONFIG_RUN);
    let txt = fs::read_to_string(&p).map_err(|e| io("lecture", &p, e))?;
    let mut cfg: ConfigRun = serde_json::from_str(&txt).map_err(|e| Erreur::Format(format!("{} illisible : {e}", p.display())))?;
    cfg.outdir = dir.to_path_buf();
    Ok(cfg)
}

/// Charge les options additionnelles d'un run existant (M4 : torsion).
pub fn charger_options(dir: &Path) -> Resultat<OptionsRun> {
    let p = dir.join(torsion::NOM_FICHIER_TORSION);
    if !p.exists() {
        return Ok(OptionsRun::default());
    }
    let txt = fs::read_to_string(&p).map_err(|e| io("lecture", &p, e))?;
    let value: Value = serde_json::from_str(&txt).map_err(|e| Erreur::Format(format!("{} illisible : {e}", p.display())))?;
    let plans_arr = value.get("plans").and_then(|v| v.as_array())
        .ok_or_else(|| Erreur::Format(format!("{} : champ « plans » manquant ou invalide", p.display())))?;
    let plans_str = plans_arr.iter().map(|v| v.as_str().unwrap_or_default()).collect::<Vec<_>>().join(",");
    let torsion = Torsion::analyser(&plans_str)?;
    Ok(OptionsRun { torsion })
}

/// Conventions des observables brutes, communes aux runs tordus neufs et repris.
fn conventions_polyakov_torsion() -> Value {
    json!({
        "valeurs": "brutes : P_rho = moyenne sur les L^3 sites de base de ReTr(produit des liens U_rho)/2",
        "pile": "les valeurs brutes dépendent de la position de la pile ; son déplacement change le signe des boucles de la tranche traversée",
        "symetrie": "pour n_mu_nu = 1 : G = (restauration de la pile) compose (translation selon mu), G^L P_nu(x) = -P_nu(x), et de même en échangeant mu et nu",
        "esperance": "à volume fini, la symétrie de centre impose <P_rho> = 0 dans toute direction ; les moyennes finies fluctuent"
    })
}

/// Champs du manifeste propres au générateur (ceux de la machine, du binaire, des noyaux, de la garde SSD sont
/// posés par `stockage::Manifeste`).
fn champs_manifeste(app: Option<&Appareil>, cfg: &ConfigRun, n: &dyn NoyauxSu2, opts: &OptionsRun) -> Vec<(String, Value)> {
    let mut c: Vec<(String, Value)> = vec![
        // le backend EFFECTIF du run (le profil de la machine peut en choisir un autre : --backend cpu)
        ("backend".into(), json!(n.backend().nom())),
        ("precision".into(), json!(n.precision().nom())),
        ("mode".into(), json!(if cfg.bord.is_some() { "su2-bord" } else { "su2-gen" })),
        ("moteur_su2".into(), json!("kvu::su2 (générateur SU(2) unifié, T3)")),
        ("config_run".into(), serde_json::to_value(cfg).unwrap_or(Value::Null)),
        ("graines".into(), json!(cfg.lot.chaines.iter().map(|c| c.graine.0).collect::<Vec<_>>())),
        ("betas".into(), json!(cfg.lot.chaines.iter().map(|c| c.beta).collect::<Vec<_>>())),
        ("options_numeriques".into(), serde_json::to_value(&cfg.lot.options).unwrap_or(Value::Null)),
        ("l".into(), json!(cfg.lot.l)),
        ("action".into(), json!(if !opts.torsion.est_triviale() {
            "beta sum(1 - z_p ReTr(U_p)/2), z_p = -1 sur la pile de torsion (torsion.json)"
        } else {
            "beta sum(1-ReTr(U_p)/2)"
        })),
        ("rng".into(), json!("Philox4x32-10 compteur (lien local,sweep,tirage,etiquette), cle=graine")),
        ("layout".into(), json!("complex128 C-order (mu,x0,x1,x2,x3,i,j), temporal=3 (c16) ; q64 : float64 (4,L,L,L,L,4)")),
        ("index".into(), json!("0 = etat initial ; 1..N apres skip_steps pas ; dump au meme indice")),
        ("projection_octets".into(), json!(projection_octets(cfg))),
        ("octets_ecrits_reprises".into(), json!(octets_ecrits_reprises(cfg))),
    ];
    if !opts.torsion.est_triviale() {
        c.push(("torsion".into(), opts.torsion.json()));
        c.push(("polyakov".into(), conventions_polyakov_torsion()));
    }
    if let Some(a) = app {
        c.push(("gpu".into(), json!(a.nom())));
        c.push(("arch".into(), json!(a.arch())));
        let versions: serde_json::Map<String, Value> = a.versions().into_iter().map(|(k, v)| (k, Value::String(v))).collect();
        c.push(("versions".into(), Value::Object(versions)));
    }
    c
}

// ------------------------------------------------------------------------------------------
// Boucle périodique standard (su2_hip)
// ------------------------------------------------------------------------------------------

const ENTETE_THERM: &str = "step,plaquette,P0,P1,P2,P3";
const ENTETE_MES: &str = "measurement_index,total_steps,phase,plaquette,P0,P1,P2,P3";

struct Csv {
    therm: BufWriter<File>,
    mes: BufWriter<File>,
    p_therm: PathBuf,
    p_mes: PathBuf,
}

fn ouvrir_csv(dir: &Path, reprise: Option<(usize, usize)>) -> Resultat<Csv> {
    let (p_therm, p_mes) = (dir.join("therm.csv"), dir.join("measurements.csv"));
    match reprise {
        None => {
            let mut c = Csv { therm: creer(&p_therm)?, mes: creer(&p_mes)?, p_therm, p_mes };
            ecrire_ligne(&mut c.therm, &c.p_therm, ENTETE_THERM)?;
            ecrire_ligne(&mut c.mes, &c.p_mes, ENTETE_MES)?;
            Ok(c)
        }
        Some((lignes_therm, lignes_mes)) => {
            tronquer_lignes(&p_therm, 1 + lignes_therm)?;
            tronquer_lignes(&p_mes, 1 + lignes_mes)?;
            Ok(Csv { therm: ouvrir_ajout(&p_therm)?, mes: ouvrir_ajout(&p_mes)?, p_therm, p_mes })
        }
    }
}

fn ligne_therm(i: usize, r: &[f64]) -> String {
    format!("{},{},{},{},{},{}", i, g17(r[0]), g17(r[1]), g17(r[2]), g17(r[3]), g17(r[4]))
}

fn ligne_mes(i: usize, therm: u64, skip: u64, r: &[f64]) -> String {
    format!(
        "{},{},{},{},{},{},{},{}",
        i,
        therm + i as u64 * skip,
        if i == 0 { "initial" } else { "sample" },
        g17(r[0]),
        g17(r[1]),
        g17(r[2]),
        g17(r[3]),
        g17(r[4])
    )
}

/// Fin de run : comme su2_hip (rc 6, pas de COMPLETE), un plafond d'essais du bain thermique atteint au moins une
/// fois signale un champ corrompu (04/10, APU : chaîne FP64 L = 16, β = 2,5, NaN dès le pas 770 avec
/// 18 plafonds). Le run échoue alors sans COMPLETE.
pub fn controle_plafond(plafond: u64) -> Resultat<()> {
    if plafond == 0 {
        return Ok(());
    }
    Err(Erreur::Validation(format!(
        "plafond d'essais du bain thermique atteint {plafond} fois : champ probablement corrompu (faute GPU), \
         aucune COMPLETE ; relancer avec les mêmes graines"
    )))
}

/// Marge du contrôle de trajectoire : le chemin F32 laisse |q| à ~2e-7 de 1, donc ½tr à ~1e-6 hors de [−1, 1].
pub const MARGE_SENTINELLE: f64 = 1e-5;

/// Contrôle de trajectoire d'une ligne d'historique (plaquette, P0..P3) : finie et dans [−1, 1] à
/// `MARGE_SENTINELLE` près (moyennes de ½tr de matrices SU(2)).
///
/// Sur l'iGPU d'une APU, des fautes transitoires ont corrompu des chaînes sans erreur du runtime (04/10 :
/// chaîne FP64 L = 16, β = 16, plaquette NaN à la mesure 2083 puis trajectoire divergente ; la relance aux
/// mêmes graines est saine et identique à l'octet au moteur unifié). Un run qui rencontre une telle ligne
/// s'arrête SANS COMPLETE, avant d'écrire la ligne et avant tout point de reprise postérieur.
pub fn sentinelle(r: &[f64], chaine: usize, indice: usize, phase: &str) -> Resultat<()> {
    match r
        .iter()
        .position(|x| !x.is_finite() || x.abs() > 1.0 + MARGE_SENTINELLE)
    {
        None => Ok(()),
        Some(k) => Err(Erreur::Validation(format!(
            "contrôle de trajectoire : chaîne {chaine}, {phase} indice {indice}, colonne {k} = {} (attendu fini \
             dans [−1, 1]) : faute GPU probable, aucune COMPLETE ; relancer avec les mêmes graines (déterministe) \
             ou --reprendre depuis un point de reprise antérieur",
            r[k]
        ))),
    }
}

/// Écrit les lignes [i0, i1) de l'historique pour chaque chaîne, après le contrôle de trajectoire.
fn vider(ctx: &CtxRun, csv: &mut [Csv], i0: usize, i1: usize, therm: bool) -> Resultat<()> {
    if i1 <= i0 {
        return Ok(());
    }
    let hist = ctx.n.historique()?;
    let hcap = hist.len() / (ctx.nch() * 5);
    for (c, f) in csv.iter_mut().enumerate() {
        for i in i0..i1 {
            let o = (c * hcap + i) * 5;
            let r = &hist[o..o + 5];
            sentinelle(r, c, i, if therm { "thermalisation" } else { "mesure" })?;
            if therm {
                ecrire_ligne(&mut f.therm, &f.p_therm, &ligne_therm(i, r))?;
            } else {
                ecrire_ligne(&mut f.mes, &f.p_mes, &ligne_mes(i, ctx.cfg.therm, ctx.cfg.skip, r))?;
            }
        }
        let (w, p) = if therm { (&mut f.therm, &f.p_therm) } else { (&mut f.mes, &f.p_mes) };
        w.flush().map_err(|e| io("écriture", p, e))?;
    }
    Ok(())
}

fn boucle_standard(ctx: &mut CtxRun, mesures: &mut [Box<dyn MesureEnLigne>]) -> Resultat<Fin> {
    let cfg = ctx.cfg;
    let (therm, nmeas, skip) = (cfg.therm, cfg.nmeas, cfg.skip);
    let hcap = therm.max(nmeas) as usize + 1;
    ctx.n.allouer_historique(hcap)?;
    // --- état de départ ---
    let (mut pas, mut mes_ecrites): (u64, u64) = (0, 0);
    let mut csv: Vec<Csv> = Vec::new();
    if ctx.reprendre {
        let e: EtatReprise = reprise::lire(&cfg.outdir, ctx.n.as_mut())?;
        pas = e.pas_faits;
        ctx.pas_au_depart = pas;
        mes_ecrites = e.mesures_faites;
        ctx.compteurs_avant = e.compteurs;
        let lignes_therm = pas.min(therm) as usize + 1;
        for d in &ctx.dirs {
            csv.push(ouvrir_csv(d, Some((lignes_therm, mes_ecrites as usize)))?);
        }
    } else {
        ctx.n.init()?;
        for d in &ctx.dirs {
            csv.push(ouvrir_csv(d, None)?);
        }
        ctx.n.mesurer(0)?;
        if cfg.dump_initial {
            ctx.dumper(0)?;
        }
    }
    let mut dernier_point = pas;
    let mut ndumps = 0u64;
    // --- thermalisation (indice 0 = état initial) ---
    let mut fait = if ctx.reprendre { pas as usize + 1 } else { 0 };
    let mut t = pas + 1;
    while t <= therm {
        ctx.pas()?;
        ctx.n.mesurer(t as usize)?;
        if t % FL == 0 {
            vider(ctx, &mut csv, fait, t as usize + 1, true)?;
            fait = t as usize + 1;
        }
        pas = t;
        t += 1;
        ctx.progres(pas, therm + nmeas * skip);
        if let Some(fin) = point_si_du(ctx, &mut csv, &mut fait, pas, 0, true, &mut dernier_point)? {
            return Ok(fin);
        }
    }
    vider(ctx, &mut csv, fait, therm as usize + 1, true)?;
    // --- production : indice 0 = fin de thermalisation ---
    if mes_ecrites == 0 {
        ctx.n.mesurer(0)?;
    }
    let mut fait = mes_ecrites as usize;
    let mut m = mes_ecrites.max(1);
    while m <= nmeas {
        for _ in 0..skip {
            ctx.pas()?;
        }
        ctx.n.mesurer(m as usize)?;
        pas = therm + m * skip;
        if cfg.dump_every > 0 && m % cfg.dump_every == 0 {
            ctx.dumper(m)?;
            ndumps += 1;
        }
        for mes in mesures.iter_mut() {
            let p = mes.periode() as u64;
            if p > 0 && m % p == 0 {
                ctx.n.synchroniser()?;
                for c in 0..ctx.nch() {
                    let hote = if ctx.app.is_none() { Some(ctx.n.telecharger(c)?) } else { None };
                    mes.mesurer(ctx.app, ctx.n.champ_gpu(), hote.as_ref(), c, pas, &ctx.dirs[c])?;
                }
            }
        }
        if m % FL == 0 {
            vider(ctx, &mut csv, fait, m as usize + 1, false)?;
            fait = m as usize + 1;
        }
        m += 1;
        ctx.progres(pas, therm + nmeas * skip);
        if let Some(fin) = point_si_du(ctx, &mut csv, &mut fait, pas, m - 1, false, &mut dernier_point)? {
            return Ok(fin);
        }
    }
    vider(ctx, &mut csv, fait, nmeas as usize + 1, false)?;
    // --- fin : compteurs, plafond, sorties ---
    let (plafond, essais) = ctx.plafond_total()?;
    ctx.rapport = RapportRun { pas: therm + nmeas * skip, mesures: nmeas, dumps: ndumps, plafond, secondes: 0.0 };
    let nl = 4.0 * (cfg.lot.l.pow(4)) as f64 * (therm + nmeas * skip) as f64 * ctx.nch() as f64;
    println!(
        "PERF L={} chaines={} backend={} hb_sweeps={} essais_par_lien_hb={:.5} plafond_atteint={} dumps={} pas_par_s={:.2}",
        cfg.lot.l,
        ctx.nch(),
        ctx.n.backend().nom(),
        therm + nmeas * skip,
        essais as f64 / nl.max(1.0),
        plafond,
        ndumps,
        (therm + nmeas * skip - ctx.pas_au_depart) as f64 / ctx.t_debut.elapsed().as_secs_f64().max(1e-9)
    );
    if plafond != 0 {
        return Err(Erreur::Validation(format!("plafond d'essais atteint {plafond} fois")));
    }
    drop(csv);
    enregistrer_sorties(ctx)?;
    Ok(Fin::Termine)
}

/// Point de reprise dû (période) ou arrêt demandé, aux frontières d'itération. Rend Some(fin) pour s'arrêter.
fn point_si_du(ctx: &mut CtxRun, csv: &mut [Csv], fait: &mut usize, pas: u64, m: u64, en_therm: bool,
               dernier: &mut u64) -> Resultat<Option<Fin>> {
    let cfg = ctx.cfg;
    let fenetre = f64::from_bits(FENETRE_S.load(Ordering::Relaxed));
    let arret = ctx.arret_apres.is_some_and(|a| pas >= a) || (fenetre > 0.0 && fenetre_depassee(ctx.t_debut.elapsed().as_secs_f64(), fenetre));
    let du = cfg.reprise_every > 0 && pas - *dernier >= cfg.reprise_every;
    if !(arret || du) {
        return Ok(None);
    }
    // les CSV doivent contenir toutes les lignes jusqu'au point
    if en_therm {
        vider(ctx, csv, *fait, pas as usize + 1, true)?;
        *fait = pas as usize + 1;
    } else {
        vider(ctx, csv, *fait, m as usize + 1, false)?;
        *fait = m as usize + 1;
    }
    let mes = if en_therm { 0 } else { m + 1 };
    ctx.point_de_reprise(pas, mes)?;
    *dernier = pas;
    if arret {
        ctx.rapport.pas = pas;
    }
    Ok(if arret { Some(Fin::Interrompu) } else { None })
}

/// Fichiers produits (nom, sha256, octets) dans le manifeste : les dumps de ce processus ont leur sha256 calculé
/// pendant l'écriture ; les autres fichiers (CSV, dumps d'avant une reprise) sont relus en flux.
pub(super) fn enregistrer_sorties(ctx: &mut CtxRun) -> Resultat<()> {
    let dirs = ctx.dirs.clone();
    for d in &dirs {
        let mut noms: Vec<String> = fs::read_dir(d)
            .map_err(|e| io("lecture du répertoire", d, e))?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".csv") || n.ends_with(".npy") || n.ends_with(".zst"))
            .collect();
        noms.sort();
        for nom in noms {
            let p = d.join(&nom);
            let (sha, octets) = match ctx.sorties.get(&p) {
                Some(x) => x.clone(),
                None => {
                    let sha = sha256_fichier(&p).map_err(|e| io("sha256", &p, e))?;
                    (sha, fs::metadata(&p).map_err(|e| io("stat", &p, e))?.len())
                }
            };
            ctx.man.fichier(&p, &sha, octets)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn plafond_atteint_refuse_la_fin_de_run() {
        assert!(super::controle_plafond(0).is_ok());
        let e = super::controle_plafond(18).unwrap_err();
        assert!(matches!(e, crate::commun::Erreur::Validation(_)));
        assert!(e.to_string().contains("18 fois"), "{e}");
    }

    #[test]
    fn sentinelle_accepte_les_lignes_saines_et_refuse_nan_inf_et_hors_bornes() {
        use super::{sentinelle, MARGE_SENTINELLE};
        assert!(sentinelle(&[0.6, -0.2, 0.0, 1.0, -1.0], 0, 7, "mesure").is_ok());
        // champ froid F32 : ½tr légèrement au-dessus de 1 par arrondi
        assert!(sentinelle(&[1.0 + 1e-6, 1.0, 1.0, 1.0, 1.0], 0, 0, "thermalisation").is_ok());
        for mauvais in [f64::NAN, -f64::NAN, f64::INFINITY, 1.0 + 2.0 * MARGE_SENTINELLE, -1.5] {
            let e = sentinelle(&[0.95, mauvais, 0.5, 0.1, 0.2], 3, 2083, "mesure").unwrap_err();
            assert!(matches!(e, crate::commun::Erreur::Validation(_)), "{e}");
            let m = e.to_string();
            assert!(
                m.contains("chaîne 3") && m.contains("indice 2083") && m.contains("colonne 1"),
                "{m}"
            );
        }
    }

    use super::*;

    #[test]
    fn fenetre_de_temps() {
        assert!(!fenetre_depassee(100.0, 0.0), "fenêtre nulle = aucune");
        assert!(!fenetre_depassee(539.9, 540.0));
        assert!(fenetre_depassee(540.0, 540.0));
        assert!(fenetre_depassee(1e6, 540.0));
        assert!(!fenetre_depassee(f64::NAN, 540.0));
        fixer_fenetre(f64::NAN);
        assert_eq!(FENETRE_S.load(Ordering::Relaxed), 0);
        fixer_fenetre(-3.0);
        assert_eq!(FENETRE_S.load(Ordering::Relaxed), 0);
        fixer_fenetre(540.0);
        assert_eq!(f64::from_bits(FENETRE_S.load(Ordering::Relaxed)), 540.0);
        fixer_fenetre(0.0);
        assert_eq!(FENETRE_S.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn repertoire_de_chaine() {
        let d = Path::new("/x/run");
        assert_eq!(repertoire_chaine(d, 1, 0), PathBuf::from("/x/run"));
        assert_eq!(repertoire_chaine(d, 4, 2), PathBuf::from("/x/run/c002"));
    }

    #[test]
    fn tailles_de_dump_selon_le_format() {
        // c16 = 256·L⁴ octets, q64 = 128·L⁴ (ARCHITECTURE_UNIFIEE.md §5), plus l'en-tête NPY (< 200 octets)
        for (l, c16) in [(32u64, 256 * 32u64.pow(4)), (64, 256 * 64u64.pow(4))] {
            let d = octets_dump(l as usize, "c16");
            assert!(d >= c16 && d < c16 + 200, "c16 L = {l} : {d}");
            let q = octets_dump(l as usize, "q64");
            assert!(q >= c16 / 2 && q < c16 / 2 + 200, "q64 L = {l} : {q}");
        }
    }

    #[test]
    fn troncature_de_lignes() {
        let p = std::env::temp_dir().join(format!("kvu_tronc_{}.csv", std::process::id()));
        fs::write(&p, "a\nb\nc\nd\n").unwrap();
        tronquer_lignes(&p, 2).unwrap();
        assert_eq!(fs::read_to_string(&p).unwrap(), "a\nb\n");
        assert!(tronquer_lignes(&p, 5).is_err());
        fs::remove_file(&p).unwrap();
    }

    #[test]
    fn lignes_csv_au_format_su2_hip() {
        let r = [0.5, 0.1, -0.2, 0.0, 1.0 / 3.0];
        assert_eq!(ligne_therm(3, &r), "3,0.5,0.10000000000000001,-0.20000000000000001,0,0.33333333333333331");
        assert_eq!(ligne_mes(0, 100, 5, &r), "0,100,initial,0.5,0.10000000000000001,-0.20000000000000001,0,0.33333333333333331");
        assert!(ligne_mes(2, 100, 5, &r).starts_with("2,110,sample,"));
    }

    #[test]
    fn projection_du_run() {
        use crate::commun::Graine;
        let cfg = ConfigRun {
            lot: ConfigLot {
                l: 64,
                chaines: vec![super::super::ParamChaine { beta: 2.3, graine: Graine(1), depart: Depart::Chaud }],
                nor: 4,
                precision: Precision::F64,
                options: crate::gpu::OptionsCompil::reference_k1(),
                mutant_metro: false,
            },
            therm: 100,
            nmeas: 20,
            skip: 1,
            dump_every: 1,
            dump_initial: false,
            format_dump: "c16".into(),
            zstd: false,
            reprise_every: 0,
            outdir: "/tmp/x".into(),
            bord: None,
        };
        // 20 dumps c16 à L = 64 : plus de 85 Go (SSD-1 : refusé sur acemagic)
        assert!(projection_octets(&cfg) > 85_000_000_000);
    }

    /// Vérificateur du 04/10 : `--reprise-every 1` à L = 48 sur 3000 pas écrit ~2 To sans avertissement.
    #[test]
    fn volume_des_reprises_signale() {
        use crate::commun::Graine;
        let mut cfg = ConfigRun {
            lot: ConfigLot {
                l: 48,
                chaines: vec![super::super::ParamChaine { beta: 2.3, graine: Graine(1), depart: Depart::Chaud }],
                nor: 4,
                precision: Precision::F64,
                options: crate::gpu::OptionsCompil::reference_k1(),
                mutant_metro: false,
            },
            therm: 1000,
            nmeas: 2000,
            skip: 1,
            dump_every: 0,
            dump_initial: false,
            format_dump: "c16".into(),
            zstd: false,
            reprise_every: 1,
            outdir: "/tmp/x".into(),
            bord: None,
        };
        let par_point = 128 * 48u64.pow(4);
        assert_eq!(octets_ecrits_reprises(&cfg), 3000 * par_point);
        let a = avertissement_reprises(&cfg).expect("~2 To : avertissement attendu");
        assert!(a.contains("--reprise-every 1") && a.contains("Go au total"), "{a}");
        // la période proposée ramène bien le volume sous le seuil
        let propose: u64 = a.rsplit("--reprise-every ").next().unwrap().split_whitespace().next().unwrap().parse().unwrap();
        cfg.reprise_every = propose;
        assert!(avertissement_reprises(&cfg).is_none(), "période proposée {propose}");
        cfg.reprise_every = 0;
        assert_eq!(octets_ecrits_reprises(&cfg), 0);
    }

    #[test]
    fn besoin_memoire_du_run() {
        use crate::commun::Graine;
        let ch = |g| super::super::ParamChaine { beta: 2.3, graine: Graine(g), depart: Depart::Chaud };
        let mut cfg = ConfigRun {
            lot: ConfigLot {
                l: 48,
                chaines: vec![ch(1)],
                nor: 4,
                precision: Precision::F64,
                options: crate::gpu::OptionsCompil::reference_k1(),
                mutant_metro: false,
            },
            therm: 10,
            nmeas: 10,
            skip: 1,
            dump_every: 0,
            dump_initial: false,
            format_dump: "c16".into(),
            zstd: false,
            reprise_every: 0,
            outdir: "/tmp/x".into(),
            bord: None,
        };
        let champ = 128 * 48u64.pow(4);
        let un = besoin_memoire(&cfg, true);
        assert_eq!(un.hote, 3 * champ);
        assert!(un.gpu >= champ);
        cfg.lot.chaines = (0..4).map(ch).collect();
        assert_eq!(besoin_memoire(&cfg, true).gpu, 4 * un.gpu, "le lot multiplie la mémoire GPU");
        cfg.lot.precision = Precision::F32;
        assert!(besoin_memoire(&cfg, true).gpu < 4 * un.gpu);
        assert_eq!(besoin_memoire(&cfg, false).gpu, 0);
    }

    #[test]
    fn validation_du_run() {
        use crate::commun::Graine;
        let mut cfg = ConfigRun {
            lot: ConfigLot {
                l: 4,
                chaines: vec![super::super::ParamChaine { beta: 2.3, graine: Graine(1), depart: Depart::Chaud }],
                nor: 4,
                precision: Precision::F64,
                options: crate::gpu::OptionsCompil::reference_k1(),
                mutant_metro: false,
            },
            therm: 10,
            nmeas: 10,
            skip: 1,
            dump_every: 0,
            dump_initial: false,
            format_dump: "c16".into(),
            zstd: false,
            reprise_every: 0,
            outdir: "/tmp/x".into(),
            bord: None,
        };
        assert!(valider_run(&cfg).is_ok());
        cfg.skip = 0;
        assert!(valider_run(&cfg).is_err());
        cfg.skip = 1;
        cfg.format_dump = "zzz".into();
        assert!(valider_run(&cfg).is_err());
        cfg.format_dump = "c16".into();
        cfg.bord = Some(ConfigBord { bc: "hot".into(), bc_graine: 1, jauge: None, pre: 5, bloc: 3 });
        assert!(valider_run(&cfg).is_err(), "nmeas = 10 n'est pas multiple de bloc = 3");
        cfg.bord = Some(ConfigBord { bc: "froid".into(), bc_graine: 1, jauge: None, pre: 5, bloc: 5 });
        assert!(valider_run(&cfg).is_err());
        cfg.bord = Some(ConfigBord { bc: "cold".into(), bc_graine: 1, jauge: None, pre: 5, bloc: 5 });
        assert!(valider_run(&cfg).is_ok());
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Manifeste d'un run repris après une faute GPU (lot gpu-rc)
    // ---------------------------------------------------------------------------------------------------------------

    const RAISON_700: &str = "GPU : hipStreamSynchronize: an illegal memory access was encountered (700)";

    /// Répertoire d'un run en échec (faute GPU, rc 3), sans COMPLETE, tel que le laisse la fenêtre fautive.
    fn run_en_echec(nom: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kvu_run_repris_{nom}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        let m = json!({
            "statut": "echec", "rc": 3, "raison_echec": RAISON_700, "fin_unix": 1793559097.5, "debut_unix": 1793547524.5, "pid": 994176
        });
        fs::write(d.join(manifest::NOM_MANIFESTE), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
        d
    }

    fn lire_manifeste(d: &Path) -> Value {
        serde_json::from_str(&fs::read_to_string(d.join(manifest::NOM_MANIFESTE)).unwrap()).unwrap()
    }

    #[test]
    fn l_echec_d_une_fenetre_est_archive_a_la_cloture_reussie_d_un_run_repris() {
        let mut obj = json!({"statut": "echec", "rc": 3, "raison_echec": RAISON_700, "fin_unix": 1793559097.5})
            .as_object()
            .unwrap()
            .clone();
        archiver_echec_precedent(&mut obj);
        assert!(!obj.contains_key("raison_echec"), "la raison d'échec ne reste pas à côté d'un statut réussi");
        let a = obj["fautes_anterieures"].as_array().unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(a[0]["rc"], json!(3));
        assert_eq!(a[0]["raison_echec"], json!(RAISON_700));
        assert_eq!(a[0]["fin_unix"], json!(1793559097.5));
        // un second échec antérieur s'ajoute à l'historique au lieu de le remplacer
        obj.insert("raison_echec".into(), json!("GPU : hipMemcpyDtoHAsync : an illegal memory access was encountered (700)"));
        obj.insert("rc".into(), json!(3));
        archiver_echec_precedent(&mut obj);
        assert_eq!(obj["fautes_anterieures"].as_array().unwrap().len(), 2);
        // aucun échec antérieur : le manifeste ne change pas d'un octet
        let mut propre = json!({"statut": "en_cours", "rc": null, "pid": 7}).as_object().unwrap().clone();
        let avant = propre.clone();
        archiver_echec_precedent(&mut propre);
        assert_eq!(propre, avant);
    }

    #[test]
    fn l_archivage_d_un_echec_sans_rc_ni_date_garde_des_nuls() {
        // fenêtre tuée par `timeout` après avoir consigné sa raison mais avant de poser rc et fin_unix : l'historique le dit (null)
        let mut obj = json!({"statut": "en_cours", "raison_echec": RAISON_700}).as_object().unwrap().clone();
        archiver_echec_precedent(&mut obj);
        assert!(!obj.contains_key("raison_echec"));
        let a = obj["fautes_anterieures"].as_array().unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(a[0]["rc"], Value::Null);
        assert_eq!(a[0]["fin_unix"], Value::Null);
        assert_eq!(a[0]["raison_echec"], json!(RAISON_700));
    }

    #[test]
    fn un_run_repris_apres_une_faute_gpu_se_cloture_termine_sans_raison_d_echec() {
        let d = run_en_echec("succes");
        let mut m = ManifesteRun::reprendre(&d).unwrap();
        m.terminer(0).unwrap();
        let j = lire_manifeste(&d);
        assert_eq!(j["statut"], json!("termine"));
        assert_eq!(j["rc"], json!(0));
        assert!(j.get("raison_echec").is_none(), "« termine, rc 0 » à côté de la raison d'un échec : {j}");
        let a = j["fautes_anterieures"].as_array().unwrap();
        assert_eq!(a.len(), 1, "l'historique garde la faute : {j}");
        assert_eq!(a[0]["rc"], json!(3));
        assert!(a[0]["raison_echec"].as_str().unwrap().contains("illegal memory access"));
        assert!(d.join(manifest::NOM_COMPLETE).exists());
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn une_fenetre_qui_echoue_a_son_tour_garde_sa_raison_d_echec_et_n_ecrit_pas_complete() {
        let d = run_en_echec("echec");
        let mut m = ManifesteRun::reprendre(&d).unwrap();
        let raison = "GPU : hipMemcpyDtoHAsync : an illegal memory access was encountered (700)";
        m.echec(3, raison).unwrap();
        let j = lire_manifeste(&d);
        assert_eq!(j["statut"], json!("echec"));
        assert_eq!(j["rc"], json!(3));
        assert_eq!(j["raison_echec"], json!(raison));
        assert!(j.get("fautes_anterieures").is_none(), "l'archivage ne concerne que la clôture réussie : {j}");
        assert!(!d.join(manifest::NOM_COMPLETE).exists());
        fs::remove_dir_all(&d).unwrap();
    }
}
