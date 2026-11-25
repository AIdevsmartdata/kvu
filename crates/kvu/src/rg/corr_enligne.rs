//! Mesure en ligne `corr` de `su2 gen` (B08, 11/11/2026), sans modifier le champ du générateur.
//!
//! Grammaire : `corr:<lissages>:<période>[:mutant=pas-en-moins|direction]`, période ≥ 1 ;
//! Lissages séparés par `+`, chacun `brut` ou `flot/<eps>/<n>`
//! (noms JSON : `brut`, `flot:<eps>:<n>`).
//! Les mutants servent seulement aux portes GPU. Le temps est la direction 0, sauf pour le mutant
//! `direction`.
//!
//! Chaque chaîne reçoit `correlateur_enligne.csv` en ajout : une ligne par lissage, pas total
//! `therm + m·skip`, réels en `%.17g`, tranches O_t et D_t et moyennes de `rg correlateur --rmax 0`
//! (corr vide, c0 = 0). Toutes les valeurs sont contrôlées avant l'ajout.
//! Le GPU travaille sur une copie ; le CPU réutilise `correlateur::mesurer`.
//! À la clôture seulement, `correlateur.json` est reconstruit depuis le CSV,
//! avec contrôle du calendrier, des lissages, de la direction et de la finitude ;
//! son chemin est rendu au générateur pour inscription au manifeste.
//!
//! À la première mesure P de chaque chaîne dans ce processus, la reprise vérifie l'en-tête
//! et conserve le préfixe de lignes complètes de pas < P. Les mesures postérieures et la ligne
//! partielle finale sont retirées ; un retour à un pas < P après la coupure est une corruption.
//! Les noms et directions conservés doivent aussi correspondre.
//! Le mutant, la période et les lissages font partie de la spécification enregistrée dans
//! `mesures_en_ligne.json`, immuable à la reprise (garde de `su2::run`).

use super::correlateur::{
    self, assembler_tranches, lire_lissage, premiere_non_finie, FlotGpu, Lissage, Mutant,
    MutantGpu, DIR_TEMPS_CORR, SCHEMA_CORR,
};
use crate::commun::{ChampHote, Erreur, Precision, Resultat};
use crate::gpu::{Appareil, ProvenanceCompil};
use crate::heritage::su2_hb::g17;
use crate::machine::admission::BesoinMemoire;
use crate::obs::enligne::Ecrivain;
use crate::stockage::GardeSsd;
use crate::su2::run::{ConfigRun, MesureEnLigne};
use crate::su2::ChampGpu;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

pub const NOM_CSV: &str = "correlateur_enligne.csv";
pub const NOM_JSON: &str = "correlateur.json";
/// Marge fixe de l'admission GPU (voir `besoin_memoire`) : code objet du module et granularité des allocations.
pub const MARGE_GPU: u64 = 64 << 20;

/// Mutants des portes, jamais sélectionnés implicitement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutantCorr {
    Aucun,
    PasEnMoins,
    Direction,
}

#[derive(Clone, Debug)]
pub struct SpecCorr {
    pub noms: Vec<String>,
    pub lissages: Vec<Lissage>,
    pub mutant: MutantCorr,
}

/// c08d (V4) : égalité TYPÉE (`flot/0.02/50` = `flot/0.020/050`) : mêmes lissages (genre, pas, nombre de pas) dans le même
/// ordre, même mutant ; les noms (clés du CSV et du JSON) ne comptent pas, la reprise réutilise ceux qui sont enregistrés.
impl PartialEq for SpecCorr {
    fn eq(&self, autre: &Self) -> bool {
        self.mutant == autre.mutant
            && self.lissages.len() == autre.lissages.len()
            && self
                .lissages
                .iter()
                .zip(&autre.lissages)
                .all(|(a, b)| match (a, b) {
                    (Lissage::Brut, Lissage::Brut) => true,
                    (Lissage::Flot { eps: e1, n: n1 }, Lissage::Flot { eps: e2, n: n2 }) => {
                        e1 == e2 && n1 == n2
                    }
                    _ => false,
                })
    }
}

/// Analyse un élément déjà découpé sur `:` par `obs::enligne`.
pub fn analyser_element(champs: &[&str], element: &str) -> Resultat<(SpecCorr, usize)> {
    let erreur = |raison: &str| {
        Erreur::Config(format!(
            "--mesures : élément « {element} » invalide ({raison}). Grammaire : \
             corr:<lissages>:<période>, lissages séparés par + (brut, flot/<eps>/<n>), \
             ex. corr:brut+flot/0.02/50:1"
        ))
    };
    if !(3..=4).contains(&champs.len()) || champs.first() != Some(&"corr") {
        return Err(erreur(
            "trois ou quatre champs attendus, premier champ corr",
        ));
    }
    let mut noms = Vec::new();
    let mut lissages = Vec::new();
    for jeton in champs[1].split('+') {
        if jeton.is_empty() {
            return Err(erreur("lissage vide"));
        }
        let parties: Vec<&str> = jeton.split('/').collect();
        if !matches!(parties.as_slice(), ["brut"] | ["flot", _, _]) {
            return Err(erreur(&format!(
                "lissage « {jeton} » : brut ou flot/<eps>/<n> attendu"
            )));
        }
        let nom = jeton.replace('/', ":");
        let lissage = lire_lissage(&nom).map_err(|raison| erreur(&raison))?;
        if noms.contains(&nom) {
            return Err(erreur(&format!("lissage « {nom} » en double")));
        }
        noms.push(nom);
        lissages.push(lissage);
    }
    let periode = champs[2]
        .parse::<usize>()
        .ok()
        .filter(|&p| p >= 1)
        .ok_or_else(|| erreur(&format!("période « {} » : entier ≥ 1 attendu", champs[2])))?;
    let mutant = match champs.get(3).copied() {
        None => MutantCorr::Aucun,
        Some("mutant=pas-en-moins") => MutantCorr::PasEnMoins,
        Some("mutant=direction") => MutantCorr::Direction,
        Some(autre) => return Err(erreur(&format!("option « {autre} » inconnue"))),
    };
    Ok((
        SpecCorr {
            noms,
            lissages,
            mutant,
        },
        periode,
    ))
}

pub fn entete(l: usize) -> String {
    let mut champs = vec!["pas,lissage,dir_temps,moy_d,moy_g,c0,moy_pp".to_string()];
    champs.extend((0..l).map(|t| format!("O_{t}")));
    champs.extend((0..l).map(|t| format!("D_{t}")));
    champs.join(",")
}

pub fn dir_temps(mutant: MutantCorr) -> usize {
    if mutant == MutantCorr::Direction {
        3
    } else {
        DIR_TEMPS_CORR
    }
}

/// Sérialise une mesure complète, après contrôle de tous ses réels.
fn ligne(pas: u64, nom: &str, dir: usize, v: &Value, l: usize) -> Resultat<String> {
    let invalide = |cle: &str, raison: &str| {
        Erreur::Validation(format!(
            "corr : pas {pas}, lissage « {nom} », clé {cle} : {raison}"
        ))
    };
    if !v["corr"].as_object().is_some_and(|c| c.is_empty()) {
        return Err(invalide("corr", "objet vide attendu à rmax = 0"));
    }
    let reel = |cle: &str, valeur: &Value| -> Resultat<String> {
        match valeur.as_f64() {
            Some(x) if x.is_finite() => Ok(g17(x)),
            _ => Err(invalide(cle, "réel fini attendu")),
        }
    };
    let mut champs = Vec::new();
    for cle in ["moy_d", "moy_g", "c0", "moy_pp"] {
        champs.push(reel(cle, &v[cle])?);
    }
    for cle in ["O_t", "D_t"] {
        let valeurs = v[cle]
            .as_array()
            .filter(|a| a.len() == l)
            .ok_or_else(|| invalide(cle, &format!("tableau de longueur L = {l} attendu")))?;
        for (i, x) in valeurs.iter().enumerate() {
            champs.push(reel(&format!("{cle}[{i}]"), x)?);
        }
    }
    Ok(format!("{pas},{nom},{dir},{}\n", champs.join(",")))
}

fn erreur_reprise(chemin: &Path) -> Erreur {
    Erreur::Config(format!(
        "reprise : l'en-tête ou les lignes de {} ne correspondent pas à la spécification actuelle \
         (lissages, direction ou L changés) : refus ; relancer avec la même --mesures",
        chemin.display()
    ))
}

/// Retire uniquement le suffixe à refaire, sans jamais masquer un retour à un ancien pas.
pub fn preparer_reprise(chemin: &Path, entete: &str, pas: u64) -> Resultat<()> {
    let contenu = match fs::read(chemin) {
        Ok(c) => c,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(Erreur::io(format!("lecture de {}", chemin.display()), e)),
    };
    if contenu.is_empty() {
        return Ok(());
    }
    let fin_entete = contenu
        .iter()
        .position(|&b| b == b'\n')
        .ok_or_else(|| erreur_reprise(chemin))?
        + 1;
    if &contenu[..fin_entete - 1] != entete.as_bytes() {
        return Err(erreur_reprise(chemin));
    }
    let mut position = fin_entete;
    let mut fin = fin_entete;
    let mut couper = false;
    let mut precedent: Option<u64> = None;
    for ligne in contenu[fin_entete..].split_inclusive(|&b| b == b'\n') {
        let complete = ligne.last() == Some(&b'\n');
        let lu = ligne
            .split(|&b| b == b',')
            .next()
            .and_then(|s| std::str::from_utf8(s).ok())
            .and_then(|s| s.parse::<u64>().ok());
        position += ligne.len();
        match lu {
            Some(p) if complete && p < pas => {
                if couper || precedent.is_some_and(|ancien| p < ancien) {
                    return Err(Erreur::Format(format!(
                        "corr : reprise de {}, ordre corrompu au pas {p} avant le point {pas}",
                        chemin.display()
                    )));
                }
                precedent = Some(p);
                fin = position;
            }
            _ => couper = true,
        }
    }
    if fin < contenu.len() {
        let f = OpenOptions::new().write(true).open(chemin).map_err(|e| {
            Erreur::io(
                format!("ouverture de {} pour troncature", chemin.display()),
                e,
            )
        })?;
        f.set_len(fin as u64)
            .map_err(|e| Erreur::io(format!("troncature de {}", chemin.display()), e))?;
        f.sync_all()
            .map_err(|e| Erreur::io(format!("synchronisation de {}", chemin.display()), e))?;
        let n = contenu.len() - fin;
        eprintln!(
            "su2 gen : corr : reprise, {n} octet(s) retiré(s) de {} \
             (mesures postérieures au point de reprise)",
            chemin.display()
        );
    }
    Ok(())
}

/// Les noms et la direction figurent dans les lignes, pas dans l'en-tête commun à un même L.
fn verifier_spec_reprise(chemin: &Path, noms: &[String], dir: usize) -> Resultat<()> {
    if noms.is_empty() {
        return Err(erreur_reprise(chemin));
    }
    let texte = match fs::read_to_string(chemin) {
        Ok(t) => t,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(Erreur::io(format!("lecture de {}", chemin.display()), e)),
    };
    let mut n = 0_usize;
    for (i, ligne) in texte.lines().skip(1).enumerate() {
        let mut champs = ligne.split(',');
        let _ = champs.next();
        if champs.next() != Some(noms[i % noms.len()].as_str())
            || champs.next().and_then(|s| s.parse::<usize>().ok()) != Some(dir)
        {
            return Err(erreur_reprise(chemin));
        }
        n = i + 1;
    }
    if !n.is_multiple_of(noms.len()) {
        return Err(erreur_reprise(chemin));
    }
    Ok(())
}

fn fsync_repertoire(dir: &Path) -> Resultat<()> {
    File::open(dir).and_then(|f| f.sync_all()).map_err(|e| {
        Erreur::io(
            format!("synchronisation du répertoire {}", dir.display()),
            e,
        )
    })
}

/// Calendrier sans collection intermédiaire, avec contrôle du dernier pas avant l'itération.
fn calendrier(
    therm: u64,
    skip: u64,
    nmeas: u64,
    periode: u64,
) -> Resultat<impl Iterator<Item = u64> + Clone> {
    if periode == 0 || skip == 0 {
        return Err(Erreur::Config("corr : période et skip ≥ 1 requis".into()));
    }
    let n = nmeas / periode;
    if n > 0 {
        (n * periode)
            .checked_mul(skip)
            .and_then(|p| therm.checked_add(p))
            .ok_or_else(|| Erreur::Config("corr : pas total trop grand".into()))?;
    }
    Ok((1..=n).map(move |m| therm + m * periode * skip))
}

/// Vérifie le préfixe conservé, sans construire les configurations JSON de toute la chaîne.
#[allow(clippy::too_many_arguments)]
fn verifier_calendrier_partiel(
    csv: &str,
    l: usize,
    noms: &[String],
    dir: usize,
    therm: u64,
    skip: u64,
    nmeas: u64,
    periode: u64,
    pas_courant: u64,
) -> Resultat<()> {
    let pas = calendrier(therm, skip, nmeas, periode)?.take_while(|&p| p < pas_courant);
    // Un CSV absent ou vide est permis seulement si aucun pas antérieur n'est attendu.
    let vide = format!("{}\n", entete(l));
    parcourir_csv(
        if csv.is_empty() { &vide } else { csv },
        l,
        noms,
        dir,
        pas,
        |_, _| {},
    )
    .map_err(|e| match e {
        Erreur::Format(message) => Erreur::Format(format!(
            "corr : reprise : calendrier incohérent : {message}"
        )),
        autre => autre,
    })
}

pub struct MesureCorr {
    spec: SpecCorr,
    lis: Vec<(String, Lissage)>,
    periode: usize,
    sortie: Ecrivain,
    flot: Option<FlotGpu>,
    vues: BTreeSet<usize>,
    calendrier: Option<(u64, u64, u64)>,
}

impl MesureCorr {
    pub fn nouvelle(spec: SpecCorr, periode: usize, garde: Option<GardeSsd>) -> MesureCorr {
        let lis = spec
            .noms
            .iter()
            .cloned()
            .zip(spec.lissages.iter().cloned())
            .collect();
        MesureCorr {
            spec,
            lis,
            periode,
            sortie: Ecrivain::nouveau(garde),
            flot: None,
            vues: BTreeSet::new(),
            calendrier: None,
        }
    }
}

impl MesureEnLigne for MesureCorr {
    fn nom(&self) -> &str {
        "corr"
    }

    fn periode(&self) -> usize {
        self.periode
    }

    fn demarrer(&mut self, cfg: &ConfigRun, dirs: &[PathBuf], reprise: bool) -> Resultat<()> {
        for dir in dirs {
            let chemin = dir.join(NOM_JSON);
            if reprise {
                match fs::remove_file(&chemin) {
                    Ok(()) => fsync_repertoire(dir)?,
                    Err(e) if e.kind() == ErrorKind::NotFound => {}
                    Err(e) => {
                        return Err(Erreur::io(
                            format!("suppression de {}", chemin.display()),
                            e,
                        ));
                    }
                }
            } else if dir.join(NOM_CSV).exists() || chemin.exists() {
                return Err(Erreur::Config(format!(
                    "corr : run neuf dans un répertoire qui contient déjà des sorties corr : {}",
                    dir.display()
                )));
            }
        }
        self.calendrier = Some((cfg.therm, cfg.skip, cfg.nmeas));
        self.vues.clear();
        Ok(())
    }

    /// Pic de travail en octets, V = L⁴, nb = ⌈V/256⌉, F = au moins un lissage flot.
    /// GPU F64 : gpu = 8·(8V + 2L) + (F ? 8·(52V + 3nb + 3) : 8·16V), hôte = 8·2L.
    /// GPU F32 : gpu = 8·(8V + 2L) + 8·16V [champ F64 monté] + (F ? 8·(52V + 3nb + 3) : 8·16V [copie]),
    /// hôte = 8·16V (rapatriement). Les tranches gardent 6V plaquettes, d, pp et 2L sorties ;
    /// le flot garde une copie 16V, trois forces 12V, 3nb partiels et trois sorties.
    /// CPU : hôte = F ? 8·76V : 8·24V, gpu = 0. Le majorant avec flot comprend
    /// 16V quaternions + 36V forces + 16V champ final + 8V plaquettes et densités
    /// (`ref_cpu::flot_n_pas`, `pas_rk3` et `correlateur::densites`). Calcul saturé.
    /// GPU : + `MARGE_GPU` (64 Mio, fixe) pour ce que le runtime ajoute aux tampons comptés : code objet du module `obs`,
    /// mémoire privée (scratch) allouée au lancement des noyaux, granularité des ≤ 10 allocations. Mesuré sur gfx1151
    /// (porte O7, `memoire_libre_octets` avant/après la première mesure, reproductible) : + 0,17 Mo à L = 16, + 10,3 Mio à
    /// L = 24, + 0,08 Mo à L = 32.
    fn besoin_memoire(&self, cfg: &ConfigRun, gpu: bool) -> BesoinMemoire {
        let l = cfg.lot.l as u64;
        let v = l.saturating_pow(4);
        let avec_flot = self
            .spec
            .lissages
            .iter()
            .any(|lis| matches!(lis, Lissage::Flot { .. }));
        if !gpu {
            return BesoinMemoire {
                hote: v
                    .saturating_mul(if avec_flot { 76 } else { 24 })
                    .saturating_mul(8),
                gpu: 0,
            };
        }
        let tranches = v.saturating_mul(8).saturating_add(l.saturating_mul(2));
        let copie = v.saturating_mul(16);
        let flot = if avec_flot {
            v.saturating_mul(52)
                .saturating_add(v.div_ceil(256).saturating_mul(3))
                .saturating_add(3)
        } else {
            0
        };
        if cfg.lot.precision == Precision::F32 {
            BesoinMemoire {
                hote: copie.saturating_mul(8),
                gpu: tranches
                    .saturating_add(copie)
                    .saturating_add(if avec_flot { flot } else { copie })
                    .saturating_mul(8)
                    .saturating_add(MARGE_GPU),
            }
        } else {
            BesoinMemoire {
                hote: l.saturating_mul(2).saturating_mul(8),
                gpu: tranches
                    .saturating_add(if avec_flot { flot } else { copie })
                    .saturating_mul(8)
                    .saturating_add(MARGE_GPU),
            }
        }
    }

    fn provenance(&self) -> Vec<ProvenanceCompil> {
        match &self.flot {
            Some(flot) => vec![flot.provenance()],
            None => Vec::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mesurer(
        &mut self,
        app: Option<&Appareil>,
        champ: Option<&ChampGpu>,
        hote: Option<&ChampHote>,
        chaine: usize,
        pas: u64,
        sortie: &Path,
    ) -> Resultat<()> {
        if self.lis.is_empty() || self.spec.noms.len() != self.spec.lissages.len() {
            return Err(Erreur::Config(
                "corr : spécification de lissages incohérente".into(),
            ));
        }
        let dir = dir_temps(self.spec.mutant);
        let l = champ
            .map(|c| c.l)
            .or(hote.map(|u| u.l))
            .ok_or_else(|| Erreur::Config("corr : ni champ GPU ni champ hôte".into()))?;
        let chemin = sortie.join(NOM_CSV);
        let entete = entete(l);
        if !self.vues.contains(&chaine) {
            let (therm, skip, nmeas) = self.calendrier.ok_or_else(|| {
                Erreur::Config("corr : calendrier du run absent (demarrer requis)".into())
            })?;
            preparer_reprise(&chemin, &entete, pas)?;
            verifier_spec_reprise(&chemin, &self.spec.noms, dir).map_err(|e| match e {
                Erreur::Config(message) => Erreur::Format(format!(
                    "corr : reprise : calendrier des lissages incohérent : {message}"
                )),
                autre => autre,
            })?;
            let texte = match fs::read_to_string(&chemin) {
                Ok(t) => t,
                Err(e) if e.kind() == ErrorKind::NotFound => String::new(),
                Err(e) => return Err(Erreur::io(format!("lecture de {}", chemin.display()), e)),
            };
            verifier_calendrier_partiel(
                &texte,
                l,
                &self.spec.noms,
                dir,
                therm,
                skip,
                nmeas,
                self.periode as u64,
                pas,
            )?;
            self.vues.insert(chaine);
        }
        let (l, valeurs) = if let Some(champ) = champ {
            let app =
                app.ok_or_else(|| Erreur::Config("corr : champ GPU fourni sans appareil".into()))?;
            let l = champ.l;
            if l < 2 || !l.is_multiple_of(2) {
                return Err(Erreur::Config(format!(
                    "corr : L = {l} : L pair ≥ 2 requis sur GPU"
                )));
            }
            if self.flot.is_none() {
                let mutant = if self.spec.mutant == MutantCorr::PasEnMoins {
                    MutantGpu::PasEnMoins
                } else {
                    MutantGpu::Aucun
                };
                self.flot = Some(FlotGpu::nouveau(app, mutant)?);
            }
            let flot = self
                .flot
                .as_mut()
                .ok_or_else(|| Erreur::Config("corr : module GPU absent".into()))?;
            let tranches = flot.tranches_resident(champ, chaine, &self.lis, dir)?;
            let valeurs: Vec<Value> = tranches
                .into_iter()
                .map(|(d_t, o_t)| assembler_tranches(l, d_t, o_t))
                .collect();
            (l, valeurs)
        } else if let Some(u) = hote {
            if self.spec.mutant != MutantCorr::Aucun {
                return Err(Erreur::Config("mutants de corr : GPU seulement".into()));
            }
            let valeurs = self
                .lis
                .iter()
                .map(|(_, lissage)| correlateur::mesurer(u, None, lissage, 0, Mutant::Aucun, 1))
                .collect::<Resultat<Vec<_>>>()?;
            (u.l, valeurs)
        } else {
            return Err(Erreur::Config("corr : ni champ GPU ni champ hôte".into()));
        };
        if valeurs.len() != self.lis.len() {
            return Err(Erreur::Validation(
                "corr : nombre de mesures différent du nombre de lissages".into(),
            ));
        }
        // Construire TOUTES les lignes avant le moindre ajout.
        let lignes = self
            .lis
            .iter()
            .zip(&valeurs)
            .map(|((nom, _), valeur)| ligne(pas, nom, dir, valeur, l))
            .collect::<Resultat<Vec<_>>>()?
            .concat();
        let existait = chemin.exists();
        self.sortie.ajouter(&chemin, &entete, &lignes)?;
        File::open(&chemin)
            .and_then(|f| f.sync_all())
            .map_err(|e| Erreur::io(format!("synchronisation de {}", chemin.display()), e))?;
        if !existait {
            fsync_repertoire(sortie)?;
        }
        Ok(())
    }

    /// Clôture (c08c, S3 et O5) : PREMIÈRE PASSE, chaque CSV est synchronisé (toujours, même sans nouvel ajout dans ce
    /// processus) et validé EN FLUX, sans construire de document ; aucune publication si une chaîne échoue. SECONDE PASSE, un
    /// document à la fois : assemblé, publié, synchronisé, libéré (mémoire bornée par une chaîne).
    fn terminer(&mut self, cfg: &ConfigRun, dirs: &[PathBuf]) -> Resultat<Vec<PathBuf>> {
        let pas = calendrier(cfg.therm, cfg.skip, cfg.nmeas, self.periode as u64)?;
        let aucune = pas.clone().next().is_none();
        let dir = dir_temps(self.spec.mutant);
        let lire = |chemin: &Path| -> Resultat<String> {
            match fs::read_to_string(chemin) {
                Ok(t) => Ok(t),
                Err(e) if e.kind() == ErrorKind::NotFound && aucune => Ok(String::new()),
                Err(e) if e.kind() == ErrorKind::NotFound => Err(Erreur::Format(format!(
                    "corr : clôture : CSV absent : {}",
                    chemin.display()
                ))),
                Err(e) => Err(Erreur::io(format!("lecture de {}", chemin.display()), e)),
            }
        };
        for d in dirs {
            let chemin = d.join(NOM_CSV);
            let texte = lire(&chemin)?;
            if texte.is_empty() && aucune {
                continue;
            }
            File::open(&chemin)
                .and_then(|f| f.sync_all())
                .map_err(|e| Erreur::io(format!("synchronisation de {}", chemin.display()), e))?;
            parcourir_csv(
                &texte,
                cfg.lot.l,
                &self.spec.noms,
                dir,
                pas.clone(),
                |_, _| {},
            )
            .map_err(|e| match e {
                Erreur::Format(m) => {
                    Erreur::Format(format!("corr : clôture de {} : {m}", chemin.display()))
                }
                autre => autre,
            })?;
        }
        if aucune {
            eprintln!("su2 gen : corr : avertissement, aucune mesure attendue : aucun JSON écrit");
            return Ok(Vec::new());
        }
        let mut chemins = Vec::new();
        for (c, d) in dirs.iter().enumerate() {
            let beta = cfg
                .lot
                .chaines
                .get(c)
                .ok_or_else(|| {
                    Erreur::Config(format!(
                        "corr : chaîne {c} absente de la configuration du run"
                    ))
                })?
                .beta;
            let chaine = d
                .file_name()
                .ok_or_else(|| {
                    Erreur::Config(format!("corr : nom de chaîne absent pour {}", d.display()))
                })?
                .to_string_lossy()
                .into_owned();
            let doc = assembler_json(
                &lire(&d.join(NOM_CSV))?,
                cfg.lot.l,
                &self.spec.noms,
                dir,
                pas.clone(),
                &chaine,
                beta,
                self.spec.mutant,
            )?;
            if let Some(cle) = premiere_non_finie(&doc) {
                return Err(Erreur::Validation(format!(
                    "corr : chaîne « {chaine} », valeur non finie en {cle}"
                )));
            }
            let chemin = d.join(NOM_JSON);
            super::domaine::ecrire_json(&chemin, &doc)?;
            drop(doc);
            File::open(&chemin)
                .and_then(|f| f.sync_all())
                .map_err(|e| Erreur::io(format!("synchronisation de {}", chemin.display()), e))?;
            fsync_repertoire(d)?;
            chemins.push(chemin);
        }
        Ok(chemins)
    }

    fn fichiers_en_ajout(&self, cfg: &ConfigRun) -> Vec<crate::su2::run::FichierEnAjout> {
        // c08d : en-tête de L, une ligne par lissage à chaque pas, clés « lissage,dir_temps »
        let dir = dir_temps(self.spec.mutant);
        vec![crate::su2::run::FichierEnAjout {
            nom: NOM_CSV.into(),
            entete: entete(cfg.lot.l),
            n_cles: 2,
            cles: self
                .spec
                .noms
                .iter()
                .map(|n| format!("{n},{dir}"))
                .collect(),
        }]
    }

    /// c08d (V2) : restauration générique (format, clés, calendrier, troncature), puis validation COMPLÈTE du préfixe conservé
    /// (réels finis, nombre de colonnes) avant tout balayage ; préfixe invalide ⇒ refus code 2.
    fn restaurer(&mut self, cfg: &ConfigRun, dirs: &[PathBuf], pas_point: u64) -> Resultat<()> {
        crate::su2::run::restaurer_fichiers_en_ajout(
            dirs,
            &self.fichiers_en_ajout(cfg),
            cfg,
            self.periode as u64,
            pas_point,
        )?;
        for d in dirs {
            let chemin = d.join(NOM_CSV);
            let texte = match fs::read_to_string(&chemin) {
                Ok(t) => t,
                Err(e) if e.kind() == ErrorKind::NotFound => String::new(),
                Err(e) => return Err(Erreur::io(format!("lecture de {}", chemin.display()), e)),
            };
            verifier_calendrier_partiel(
                &texte,
                cfg.lot.l,
                &self.spec.noms,
                dir_temps(self.spec.mutant),
                cfg.therm,
                cfg.skip,
                cfg.nmeas,
                self.periode as u64,
                pas_point.saturating_add(1),
            )
            .map_err(|e| match e {
                Erreur::Format(m) => {
                    Erreur::Config(format!("reprise : {} : {m} ; refus", chemin.display()))
                }
                autre => autre,
            })?;
        }
        Ok(())
    }
}

/// Reconstruit exactement le schéma hors ligne à rmax = 0, avec des noms `pas_<pas total>`.
#[allow(clippy::too_many_arguments)]
pub fn assembler_json(
    csv: &str,
    l: usize,
    noms: &[String],
    dir: usize,
    pas_attendus: impl Iterator<Item = u64> + Clone,
    chaine: &str,
    beta: f64,
    mutant: MutantCorr,
) -> Resultat<Value> {
    let mut cfgs = Vec::new();
    parcourir_csv(csv, l, noms, dir, pas_attendus, |pas, mesures| {
        cfgs.push(json!({"cfg": format!("pas_{pas:06}"), "mesures": mesures}));
    })?;
    let chaine = json!({"chaine": chaine, "beta": beta, "L": l, "configs": cfgs});
    let texte = if mutant == MutantCorr::Aucun {
        format!("{:?}", Mutant::Aucun)
    } else {
        format!("{mutant:?}")
    };
    Ok(json!({
        "schema": SCHEMA_CORR, "mutant": texte, "rmax": 0usize,
        "lissages": noms, "chaines": [chaine],
    }))
}

fn parcourir_csv(
    csv: &str,
    l: usize,
    noms: &[String],
    dir: usize,
    pas_attendus: impl Iterator<Item = u64> + Clone,
    mut ajouter: impl FnMut(u64, serde_json::Map<String, Value>),
) -> Resultat<()> {
    let mut lignes = csv.split_inclusive('\n');
    let entete = entete(l);
    if lignes.next().and_then(|s| s.strip_suffix('\n')) != Some(entete.as_str()) {
        return Err(Erreur::Format(
            "corr : en-tête du CSV différent de celui attendu".into(),
        ));
    }
    if noms.is_empty() {
        return Err(Erreur::Format("corr : ordre des lissages vide".into()));
    }
    let n_champs = l
        .checked_mul(2)
        .and_then(|n| n.checked_add(7))
        .ok_or_else(|| Erreur::Format("corr : nombre de colonnes trop grand".into()))?;
    let calendrier = pas_attendus.clone();
    let mut precedent = None;
    for pas in pas_attendus {
        if precedent.is_some_and(|p| p >= pas) {
            return Err(Erreur::Format(format!(
                "corr : ordre des pas attendus invalide, pas attendu {pas}"
            )));
        }
        precedent = Some(pas);
        let mut mesures = serde_json::Map::new();
        for nom in noms {
            let erreur = |raison: &str| {
                Erreur::Format(format!(
                    "corr : {raison} ; pas attendu {pas}, lissage attendu « {nom} »"
                ))
            };
            let brute = lignes
                .next()
                .ok_or_else(|| erreur("trou : pas trouvé absent (fin du CSV)"))?;
            let ligne = brute
                .strip_suffix('\n')
                .ok_or_else(|| erreur("trou : ligne trouvée incomplète"))?;
            let champs: Vec<&str> = ligne.split(',').collect();
            if champs.len() != n_champs {
                return Err(erreur(&format!(
                    "ordre : {} champs trouvés au lieu de {n_champs}, pas trouvé « {} »",
                    champs.len(),
                    champs.first().copied().unwrap_or("")
                )));
            }
            let trouve = champs[0]
                .parse::<u64>()
                .map_err(|_| erreur(&format!("ordre : pas trouvé « {} » illisible", champs[0])))?;
            let deja_vu = if trouve == pas {
                mesures.contains_key(champs[1])
            } else {
                // Une recherche dans le calendrier n'est nécessaire qu'en cas de retour en arrière.
                trouve < pas
                    && noms.iter().any(|n| n == champs[1])
                    && calendrier
                        .clone()
                        .take_while(|&p| p < pas)
                        .any(|p| p == trouve)
            };
            if deja_vu {
                return Err(erreur(&format!(
                    "doublon : pas trouvé {trouve}, lissage trouvé « {} »",
                    champs[1]
                )));
            }
            if trouve != pas {
                let raison = if trouve > pas { "trou" } else { "ordre" };
                return Err(erreur(&format!("{raison} : pas trouvé {trouve}")));
            }
            if champs[1] != nom.as_str() || champs[2].parse::<usize>().ok() != Some(dir) {
                return Err(erreur(&format!(
                    "ordre : pas trouvé {trouve}, lissage trouvé « {} », \
                     dir_temps trouvée « {} » au lieu de {dir}",
                    champs[1], champs[2]
                )));
            }
            let reel = |i: usize, cle: &str| -> Resultat<f64> {
                match champs[i].parse::<f64>() {
                    Ok(x) if x.is_finite() => Ok(x),
                    _ => Err(erreur(&format!(
                        "ordre : pas trouvé {trouve}, clé {cle} : réel fini attendu"
                    ))),
                }
            };
            let moy_d = reel(3, "moy_d")?;
            let moy_g = reel(4, "moy_g")?;
            let c0 = reel(5, "c0")?;
            let moy_pp = reel(6, "moy_pp")?;
            let o_t = (0..l)
                .map(|t| reel(7 + t, &format!("O_t[{t}]")))
                .collect::<Resultat<Vec<_>>>()?;
            let d_t = (0..l)
                .map(|t| reel(7 + l + t, &format!("D_t[{t}]")))
                .collect::<Resultat<Vec<_>>>()?;
            mesures.insert(
                nom.clone(),
                json!({
                    "moy_d": moy_d, "moy_g": moy_g, "c0": c0, "moy_pp": moy_pp,
                    "O_t": o_t, "D_t": d_t, "corr": serde_json::Map::new(),
                }),
            );
        }
        ajouter(pas, mesures);
    }
    if let Some(ligne) = lignes.next() {
        let trouve = ligne.split(',').next().unwrap_or("");
        let Some(dernier) = precedent else {
            return Err(Erreur::Format(format!(
                "corr : calendrier incohérent : aucun pas attendu, pas trouvé « {trouve} »"
            )));
        };
        return Err(Erreur::Format(format!(
            "corr : doublon, ligne en trop : dernier pas attendu {dernier}, pas trouvé « {trouve} »"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obs::enligne::{decrire, ElementMesure};

    fn specification(texte: &str) -> (SpecCorr, usize) {
        match decrire(texte).unwrap().remove(0) {
            ElementMesure::Corr { spec, periode } => (spec, periode),
            autre => panic!("élément corr attendu, trouvé {autre:?}"),
        }
    }

    #[test]
    fn grammaire_acceptee() {
        for (texte, noms, periode, mutant) in [
            ("corr:brut:1", vec!["brut"], 1, MutantCorr::Aucun),
            (
                "corr:brut+flot/0.02/50:1",
                vec!["brut", "flot:0.02:50"],
                1,
                MutantCorr::Aucun,
            ),
            (
                "corr:flot/0.02/3+brut:5:mutant=direction",
                vec!["flot:0.02:3", "brut"],
                5,
                MutantCorr::Direction,
            ),
            (
                "corr:brut:2:mutant=pas-en-moins",
                vec!["brut"],
                2,
                MutantCorr::PasEnMoins,
            ),
        ] {
            let (spec, p) = specification(texte);
            assert_eq!(spec.noms, noms);
            assert_eq!(spec.lissages.len(), noms.len());
            assert_eq!(p, periode);
            assert_eq!(spec.mutant, mutant);
            assert_eq!(spec, spec.clone());
        }
        assert_eq!(dir_temps(MutantCorr::Aucun), DIR_TEMPS_CORR);
        assert_eq!(dir_temps(MutantCorr::Direction), 3);
    }

    #[test]
    fn grammaire_refusee() {
        for texte in [
            "corr",
            "corr:brut",
            "corr::1",
            "corr:brut:0",
            "corr:brut:x",
            "corr:lisse:1",
            "corr:flot/0.02:1",
            "corr:flot/abc/5:1",
            "corr:flot/-1/5:1",
            "corr:flot/0.02/5/1:1",
            "corr:brut+brut:1",
            "corr:brut+:1",
            "corr:brut:1:mutant=autre",
            "corr:brut:1:2",
            "corr:brut:1,corr:brut:2",
            "corr:flot/NaN/5:1",
            "corr:flot/inf/5:1",
        ] {
            assert!(
                matches!(decrire(texte), Err(Erreur::Config(_))),
                "{texte} accepté"
            );
        }
    }

    #[test]
    fn entete_exact() {
        assert_eq!(
            entete(2),
            "pas,lissage,dir_temps,moy_d,moy_g,c0,moy_pp,O_0,O_1,D_0,D_1"
        );
    }

    fn valeur() -> Value {
        json!({
            "moy_d": 0.1, "moy_g": -0.0, "c0": 0.0, "moy_pp": 1e-300,
            "O_t": [2.0 / 3.0, f64::MAX], "D_t": [-0.0, 0.1], "corr": {},
        })
    }

    fn reconstruire(csv: &str, noms: &[&str], pas: &[u64]) -> Resultat<Value> {
        let noms: Vec<String> = noms.iter().map(|s| (*s).to_string()).collect();
        assembler_json(
            csv,
            2,
            &noms,
            0,
            pas.iter().copied(),
            "chaine",
            2.3,
            MutantCorr::Aucun,
        )
    }

    fn verifier_bits(a: &Value, b: &Value) {
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => {
                assert_eq!(x.as_f64().unwrap().to_bits(), y.as_f64().unwrap().to_bits());
            }
            (Value::Array(x), Value::Array(y)) => {
                assert_eq!(x.len(), y.len());
                for (a, b) in x.iter().zip(y) {
                    verifier_bits(a, b);
                }
            }
            (Value::Object(x), Value::Object(y)) => {
                assert_eq!(x.keys().collect::<Vec<_>>(), y.keys().collect::<Vec<_>>());
                for (k, v) in x {
                    verifier_bits(v, &y[k]);
                }
            }
            _ => assert_eq!(a, b),
        }
    }

    #[test]
    fn aller_retour_exact_au_bit() {
        for x in [
            0.1,
            f64::from_bits(1),
            5e-324,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE / 4.0,
        ] {
            let mut v = valeur();
            v["moy_d"] = json!(x);
            v["O_t"][0] = json!(x);
            v["D_t"][1] = json!(x);
            let csv = format!("{}\n{}", entete(2), ligne(8, "brut", 0, &v, 2).unwrap());
            let doc = reconstruire(&csv, &["brut"], &[8]).unwrap();
            verifier_bits(&v, &doc["chaines"][0]["configs"][0]["mesures"]["brut"]);
            assert_eq!(doc["chaines"][0]["configs"][0]["cfg"], "pas_000008");
            assert_eq!(doc["schema"], SCHEMA_CORR);
            assert_eq!(doc["mutant"], "Aucun");
        }
    }

    #[test]
    fn ligne_refuse_valeurs_non_finies_et_formes_invalides() {
        for non_fini in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for cle in ["moy_d", "moy_g", "c0", "moy_pp", "O_t", "D_t"] {
                let mut v = valeur();
                if matches!(cle, "O_t" | "D_t") {
                    v[cle][1] = json!(non_fini);
                } else {
                    v[cle] = json!(non_fini);
                }
                let e = ligne(8, "brut", 0, &v, 2).unwrap_err();
                assert!(matches!(e, Erreur::Validation(_)));
                let message = e.to_string();
                assert!(
                    message.contains("pas 8") && message.contains("brut") && message.contains(cle),
                    "{message}"
                );
            }
        }
        for cle in ["corr", "O_t", "D_t"] {
            let mut v = valeur();
            v[cle] = json!([0.0]);
            assert!(matches!(
                ligne(8, "brut", 0, &v, 2),
                Err(Erreur::Validation(_))
            ));
        }
        let mut v = valeur();
        v["corr"] = json!({"0,0,0,0": 0.0});
        assert!(matches!(
            ligne(8, "brut", 0, &v, 2),
            Err(Erreur::Validation(_))
        ));
    }

    #[test]
    fn assemblage_refuse_trous_doublons_ordre_et_non_finis() {
        let h = format!("{}\n", entete(2));
        let a = ligne(8, "brut", 0, &valeur(), 2).unwrap();
        let b = ligne(10, "brut", 0, &valeur(), 2).unwrap();
        let f = ligne(8, "flot:0.02:3", 0, &valeur(), 2).unwrap();
        let d = ligne(8, "brut", 3, &valeur(), 2).unwrap();
        for (csv, noms, pas, mot) in [
            (format!("{h}{b}"), vec!["brut"], vec![8, 10], "trou"),
            (format!("{h}{a}"), vec!["brut"], vec![8, 10], "trou"),
            (
                format!("{h}{a}{a}{b}"),
                vec!["brut"],
                vec![8, 10],
                "doublon",
            ),
            (
                format!("{h}{f}{a}"),
                vec!["brut", "flot:0.02:3"],
                vec![8],
                "ordre",
            ),
            (format!("{h}{d}"), vec!["brut"], vec![8], "ordre"),
            (format!("{h}{a}{b}"), vec!["brut"], vec![8], "doublon"),
            (format!("autre\n{a}"), vec!["brut"], vec![8], "en-tête"),
            (
                format!("{h}{}", a.replacen(",0.10000000000000001,", ",NaN,", 1)),
                vec!["brut"],
                vec![8],
                "fini",
            ),
            (
                format!("{h}{}", a.replacen(",0.10000000000000001,", ",inf,", 1)),
                vec!["brut"],
                vec![8],
                "fini",
            ),
            (
                format!("{h}{}", a.trim_end()),
                vec!["brut"],
                vec![8],
                "incomplète",
            ),
        ] {
            let e = reconstruire(&csv, &noms, &pas).unwrap_err();
            assert!(matches!(e, Erreur::Format(_)), "{e}");
            assert!(e.to_string().contains(mot), "{mot} absent de {e}");
        }
    }

    fn config(l: usize) -> ConfigRun {
        ConfigRun {
            lot: crate::su2::ConfigLot {
                l,
                chaines: vec![crate::su2::ParamChaine {
                    beta: 2.3,
                    graine: crate::commun::Graine(26111108),
                    depart: crate::commun::Depart::Chaud,
                }],
                nor: 4,
                precision: Precision::F64,
                options: crate::gpu::OptionsCompil::reference_k1(),
                mutant_metro: false,
            },
            therm: 6,
            nmeas: 6,
            skip: 2,
            dump_every: 0,
            dump_initial: false,
            format_dump: "q64".into(),
            zstd: false,
            reprise_every: 2,
            outdir: PathBuf::new(),
            bord: None,
        }
    }

    #[test]
    fn besoin_memoire_pic_exact_et_sature() {
        let mut cfg = config(4);
        for (texte, gpu64, gpu32, cpu) in [
            (
                "corr:brut:1",
                49_216 + MARGE_GPU,
                81_984 + MARGE_GPU,
                49_152,
            ),
            (
                "corr:brut+flot/0.02/3:1",
                122_992 + MARGE_GPU,
                155_760 + MARGE_GPU,
                155_648,
            ),
        ] {
            let (spec, periode) = specification(texte);
            let mesure = MesureCorr::nouvelle(spec, periode, None);
            assert_eq!(
                mesure.besoin_memoire(&cfg, true),
                BesoinMemoire {
                    hote: 64,
                    gpu: gpu64,
                }
            );
            assert_eq!(
                mesure.besoin_memoire(&cfg, false),
                BesoinMemoire { hote: cpu, gpu: 0 }
            );
            cfg.lot.precision = Precision::F32;
            assert_eq!(
                mesure.besoin_memoire(&cfg, true),
                BesoinMemoire {
                    hote: 32_768,
                    gpu: gpu32,
                }
            );
            assert_eq!(
                mesure.besoin_memoire(&cfg, false),
                BesoinMemoire { hote: cpu, gpu: 0 }
            );
            cfg.lot.precision = Precision::F64;
        }
        let (spec, periode) = specification("corr:brut+flot/0.02/3:1");
        let mesure = MesureCorr::nouvelle(spec, periode, None);
        cfg.lot.l = usize::MAX;
        assert_eq!(mesure.besoin_memoire(&cfg, true).gpu, u64::MAX);
        assert_eq!(mesure.besoin_memoire(&cfg, false).hote, u64::MAX);
        assert!(mesure.provenance().is_empty());
    }

    fn csv_calendrier(pas: &[u64], noms: &[String]) -> String {
        let mut csv = format!("{}\n", entete(2));
        for &p in pas {
            for nom in noms {
                csv.push_str(&ligne(p, nom, 0, &valeur(), 2).unwrap());
            }
        }
        csv
    }

    #[test]
    fn calendrier_en_flux_periode_deux() {
        let noms = vec!["brut".to_string()];
        let pas = calendrier(6, 2, 6, 2).unwrap();
        assert_eq!(pas.clone().collect::<Vec<_>>(), vec![10, 14, 18]);
        let doc = assembler_json(
            &csv_calendrier(&[10, 14, 18], &noms),
            2,
            &noms,
            0,
            pas.clone(),
            "chaine",
            2.3,
            MutantCorr::Aucun,
        )
        .unwrap();
        assert_eq!(doc["chaines"][0]["configs"].as_array().unwrap().len(), 3);
        for (trouves, mot) in [
            (vec![14, 18], "trou"),
            (vec![10, 10, 14, 18], "doublon"),
            (vec![8, 10, 14, 18], "ordre"),
        ] {
            let e = assembler_json(
                &csv_calendrier(&trouves, &noms),
                2,
                &noms,
                0,
                pas.clone(),
                "chaine",
                2.3,
                MutantCorr::Aucun,
            )
            .unwrap_err();
            assert!(
                matches!(e, Erreur::Format(_)) && e.to_string().contains(mot),
                "{e}"
            );
        }
        let e = assembler_json(
            &csv_calendrier(&[10], &noms),
            2,
            &noms,
            0,
            calendrier(6, 2, 1, 2).unwrap(),
            "chaine",
            2.3,
            MutantCorr::Aucun,
        )
        .unwrap_err();
        assert!(e.to_string().contains("calendrier"), "{e}");
        assert!(calendrier(6, 2, 6, 0).is_err());
        assert!(calendrier(u64::MAX, 2, 6, 2).is_err());
        assert!(calendrier(6, 2, u64::MAX, 1).is_err());
        let mut grand = calendrier(0, 1, u64::MAX, 1).unwrap();
        assert_eq!(grand.next(), Some(1));
        assert_eq!(grand.next(), Some(2));
    }

    #[test]
    fn calendrier_partiel_reprise_exact() {
        let noms = vec!["brut".to_string(), "flot:0.02:3".to_string()];
        let csv = csv_calendrier(&[10, 14], &noms);
        verifier_calendrier_partiel(&csv, 2, &noms, 0, 6, 2, 6, 2, 18).unwrap();
        verifier_calendrier_partiel("", 2, &noms, 0, 6, 2, 6, 2, 10).unwrap();
        for invalide in [
            String::new(),
            csv_calendrier(&[14], &noms),
            csv_calendrier(&[10, 10], &noms),
            csv_calendrier(&[10, 12], &noms),
            csv_calendrier(&[10, 14, 18], &noms),
            csv_calendrier(&[10], &noms[..1]),
            csv_calendrier(&[10, 14], &[noms[1].clone(), noms[0].clone()]),
            csv.replacen(",brut,0,", ",brut,3,", 1),
        ] {
            let e =
                verifier_calendrier_partiel(&invalide, 2, &noms, 0, 6, 2, 6, 2, 18).unwrap_err();
            assert!(matches!(e, Erreur::Format(_)), "{e}");
            assert!(e.to_string().contains("reprise : calendrier"), "{e}");
        }
    }

    fn dossier(nom: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kvu_corr_{nom}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn demarrage_et_cloture_valident_toutes_les_chaines() {
        let dir = dossier("cloture");
        let a = dir.join("c000");
        let b = dir.join("c001");
        fs::create_dir_all(&a).unwrap();
        fs::create_dir_all(&b).unwrap();
        let dirs = vec![a.clone(), b.clone()];
        let mut cfg = config(2);
        cfg.nmeas = 1;
        cfg.lot.chaines.push(cfg.lot.chaines[0].clone());
        let (spec, periode) = specification("corr:brut:1");
        let noms = spec.noms.clone();
        let mut mesure = MesureCorr::nouvelle(spec, periode, None);
        mesure.demarrer(&cfg, &dirs, false).unwrap();
        fs::write(a.join(NOM_CSV), csv_calendrier(&[8], &noms)).unwrap();
        fs::write(b.join(NOM_CSV), csv_calendrier(&[], &noms)).unwrap();
        assert!(matches!(
            mesure.terminer(&cfg, &dirs),
            Err(Erreur::Format(_))
        ));
        assert!(!a.join(NOM_JSON).exists());
        assert!(!b.join(NOM_JSON).exists());
        let e = mesure.demarrer(&cfg, &dirs, false).unwrap_err();
        assert!(matches!(e, Erreur::Config(_)));
        assert!(e.to_string().contains("run neuf"));
        fs::write(a.join(NOM_JSON), "périmé").unwrap();
        fs::write(b.join(NOM_JSON), "périmé").unwrap();
        mesure.demarrer(&cfg, &dirs, true).unwrap();
        assert!(!a.join(NOM_JSON).exists());
        assert!(!b.join(NOM_JSON).exists());
        fs::write(b.join(NOM_CSV), csv_calendrier(&[8], &noms)).unwrap();
        assert_eq!(mesure.terminer(&cfg, &dirs).unwrap().len(), 2);
        mesure.demarrer(&cfg, &dirs, true).unwrap();
        cfg.nmeas = 0;
        let e = mesure.terminer(&cfg, &dirs).unwrap_err();
        assert!(matches!(e, Erreur::Format(_)) && e.to_string().contains("calendrier"));
        assert!(!a.join(NOM_JSON).exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reprise_conserve_le_prefixe_exact_et_refuse_la_corruption() {
        let dir = dossier("reprise");
        let chemin = dir.join(NOM_CSV);
        let h = entete(2);
        preparer_reprise(&chemin, &h, 12).unwrap();
        assert!(!chemin.exists());
        fs::write(&chemin, "").unwrap();
        preparer_reprise(&chemin, &h, 12).unwrap();
        let a = ligne(8, "brut", 0, &valeur(), 2).unwrap();
        let b = ligne(10, "brut", 0, &valeur(), 2).unwrap();
        let c = ligne(12, "brut", 0, &valeur(), 2).unwrap();
        let prefixe = format!("{h}\n{a}{b}");
        fs::write(&chemin, format!("{prefixe}{c}12,brut,0,0.5")).unwrap();
        preparer_reprise(&chemin, &h, 12).unwrap();
        assert_eq!(fs::read(&chemin).unwrap(), prefixe.as_bytes());
        let noms = vec!["brut".to_string()];
        verifier_spec_reprise(&chemin, &noms, 0).unwrap();
        assert!(matches!(
            verifier_spec_reprise(&chemin, &noms, 3),
            Err(Erreur::Config(_))
        ));
        let autres = vec!["brut".to_string(), "flot:0.02:3".to_string()];
        assert!(matches!(
            verifier_spec_reprise(&chemin, &autres, 0),
            Err(Erreur::Config(_))
        ));
        fs::write(&chemin, format!("{h}\n{a}")).unwrap();
        assert!(matches!(
            verifier_spec_reprise(&chemin, &autres, 0),
            Err(Erreur::Config(_))
        ));
        fs::write(&chemin, format!("{prefixe}illisible,brut\n12,brut")).unwrap();
        preparer_reprise(&chemin, &h, 12).unwrap();
        assert_eq!(fs::read(&chemin).unwrap(), prefixe.as_bytes());
        fs::write(&chemin, "autre\n").unwrap();
        assert!(matches!(
            preparer_reprise(&chemin, &h, 12),
            Err(Erreur::Config(_))
        ));
        let corrompu = format!("{h}\n{a}{c}{b}");
        fs::write(&chemin, &corrompu).unwrap();
        assert!(matches!(
            preparer_reprise(&chemin, &h, 12),
            Err(Erreur::Format(_))
        ));
        assert_eq!(fs::read(&chemin).unwrap(), corrompu.as_bytes());
        fs::write(&chemin, format!("{h}\n{b}{a}")).unwrap();
        assert!(matches!(
            preparer_reprise(&chemin, &h, 12),
            Err(Erreur::Format(_))
        ));
        fs::remove_dir_all(&dir).unwrap();
    }
}
