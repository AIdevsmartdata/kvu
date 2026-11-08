//! Générateur SU(2) Wilson 4D unifié. Propriétaire : T3 (sauf `rapide32`, T4).
//!
//! Un `Moteur` pilote un LOT de chaînes indépendantes de même L et même précision, sur un
//! backend (HIP, CUDA ou CPU) et avec une implémentation de [`NoyauxSu2`] :
//! - [`ref64::Ref64Gpu`]   FP64 de référence, bit-identique à su2_hip / su2_cuda / su2_gen (T3) ;
//! - [`rapide32::Rapide32Gpu`] FP32 rapide, même loi (T4) ;
//! - [`cpu::Cpu64`]        référence CPU FP64 (crate::heritage::su2_hb), avec gel (T3).
//!
//! INVARIANT DES LOTS : la chaîne c d'un lot produit EXACTEMENT les mêmes octets que la même
//! chaîne lancée seule (même graine, mêmes compteurs Philox (lien local, sweep, tirage, étiquette)).
//!
//! Signatures du trait et des types GELÉES (squelette unifie/base). Ajout T3 : la méthode FOURNIE
//! `poser_bord_device` du trait (corps par défaut = None, donc sans effet sur T4) et `valider`.
//! Ajout M4 : `torsion` et la méthode fournie `fixer_torsion`.

pub mod bord;
pub mod boite;
pub mod boite_cli;
pub mod boite_gpu;
pub mod boite_npy;
pub mod archive_cli;
pub mod boite_run;
pub mod champ;
pub mod cli;
pub mod cpu;
pub mod rapide32;
pub mod ref64;
pub mod reprise;
pub mod run;
pub mod torsion;

pub use champ::{ChampGpu, TamponChamp};

use crate::commun::{BackendKind, ChampHote, Depart, Erreur, Graine, Precision, Resultat};
use crate::gpu::{Appareil, ProvenanceCompil};

/// Paramètres d'une chaîne du lot.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParamChaine {
    pub beta: f64,
    pub graine: Graine,
    pub depart: Depart,
}

/// Liens gelés (ni bain thermique ni OR).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gel {
    Aucun,
    /// R2-3 : liens SPATIAUX U_mu(x), mu ∈ {1,2,3}, de l'hyperplan x0 = 0 (su2_bord.cu, Geo.freeze).
    BordX0Spatial,
}

/// Application du masque de liens gelés (WP-DIR). `Tout` = production (le bain thermique ET l'overrelaxation respectent le masque).
/// Les autres variantes sont des CONTRÔLES NÉGATIFS (mutants des gates GATES-DIR : M-DIR-1 = `IgnoreTout`,
/// M-DIR-3 = `IgnoreBain`, M-DIR-2 = `IgnoreSurrelax`), jamais utilisés en production.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppMasque {
    #[default]
    Tout,
    IgnoreTout,
    IgnoreBain,
    IgnoreSurrelax,
}

impl AppMasque {
    /// Le bain thermique respecte-t-il le masque ?
    pub fn bain(self) -> bool {
        matches!(self, AppMasque::Tout | AppMasque::IgnoreSurrelax)
    }
    /// L'overrelaxation respecte-t-elle le masque ?
    pub fn surrelax(self) -> bool {
        matches!(self, AppMasque::Tout | AppMasque::IgnoreBain)
    }
    pub fn nom(self) -> &'static str {
        match self {
            AppMasque::Tout => "tout",
            AppMasque::IgnoreTout => "masque-ignore",
            AppMasque::IgnoreBain => "masque-ignore-bain",
            AppMasque::IgnoreSurrelax => "masque-ignore-surrelax",
        }
    }
}

/// Configuration d'un lot.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConfigLot {
    /// L pair ≥ 2 (≥ 4 pour le bord et le conditionnel).
    pub l: usize,
    pub chaines: Vec<ParamChaine>,
    /// Sweeps d'OR par pas (pas = 1 bain thermique + nor OR).
    pub nor: usize,
    pub precision: Precision,
    /// Options numériques de compilation (FP64 : référence K1 ou compat production).
    pub options: crate::gpu::OptionsCompil,
    /// Mutant de contrôle négatif (Metropolis naïf sur x0, K1_MUTANT_METRO). JAMAIS en production.
    pub mutant_metro: bool,
}

/// Compteurs du bain thermique d'une chaîne.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Compteurs {
    /// Tirages ayant atteint le plafond de 10⁶ essais (attendu : 0).
    pub plafond: u64,
    pub essais: u64,
}

/// Noyaux d'un générateur SU(2) (implémenté par Ref64Gpu, Rapide32Gpu, Cpu64).
/// Toutes les méthodes agissent sur TOUTES les chaînes du lot.
pub trait NoyauxSu2 {
    fn precision(&self) -> Precision;
    fn backend(&self) -> BackendKind;
    fn config(&self) -> &ConfigLot;
    /// Départ chaud (Haar, compteur (lien, 0xFFFFFFFF, 0|1, ETIQUETTE_INIT)) ou froid.
    fn init(&mut self) -> Resultat<()>;
    /// Change les liens gelés (pré-thermalisation périodique, puis gel du bord).
    fn fixer_gel(&mut self, gel: Gel) -> Resultat<()>;
    /// Un sweep de bain thermique (4 mu × 2 parités) ; incrémente sweep_id.
    fn sweep_hb(&mut self) -> Resultat<()>;
    /// Un sweep d'overrelaxation U → V⁺U⁺V⁺ (aucun compteur Philox consommé).
    fn sweep_or(&mut self) -> Resultat<()>;
    /// Alloue l'historique de mesures (capacité en nombre de mesures).
    fn allouer_historique(&mut self, capacite: usize) -> Resultat<()>;
    /// Plaquette + Polyakov P0..P3 de chaque chaîne dans l'historique, case `idx` (FP64, arbre fixe
    /// k_reduce/k_final : égal à l'octet à su2_hip à configuration égale), sans synchronisation.
    fn mesurer(&mut self, idx: usize) -> Resultat<()>;
    /// Historique [chaîne][idx][5] (synchronise).
    fn historique(&self) -> Resultat<Vec<f64>>;
    /// Tranches R2-3 : S_t et T_t (t = 0..L-1) de chaque chaîne, case `idx` d'un historique de
    /// tranches de capacité `capacite` (alloué au premier appel), FP64, sommes à arbre fixe (k_tranche).
    fn mesurer_tranches(&mut self, idx: usize, capacite: usize) -> Resultat<()>;
    /// Historique des tranches [chaîne][idx][2L] (synchronise).
    fn historique_tranches(&self) -> Resultat<Vec<f64>>;
    fn telecharger(&self, chaine: usize) -> Resultat<ChampHote>;
    fn televerser(&mut self, chaine: usize, c: &ChampHote) -> Resultat<()>;
    /// Champ résident (None pour le CPU) : lu par les observables et la mesure de fibre.
    fn champ_gpu(&self) -> Option<&ChampGpu>;
    fn compteurs(&self) -> Resultat<Vec<Compteurs>>;
    fn sweep_id(&self) -> u32;
    /// Reprise : repositionne le compteur de sweeps (tout l'état RNG d'une chaîne Philox).
    fn fixer_sweep_id(&mut self, s: u32);
    fn synchroniser(&self) -> Resultat<()>;
    /// Provenance des modules compilés (vide pour le CPU).
    fn provenance(&self) -> Vec<ProvenanceCompil>;

    /// AJOUT T3 (méthode fournie, corps par défaut `None` : aucune implémentation existante à modifier).
    /// Pose du bord R2-3 et jauge de l'hyperplan par les noyaux DEVICE de su2_bord.cu (bit-identité avec
    /// su2_bord : les `log/cos/sin` du device ne sont pas ceux de la libm). Rend `None` si le générateur
    /// n'a pas de noyaux de bord (CPU, FP32) : `bord::poser` applique alors les mêmes formules sur l'hôte.
    fn poser_bord_device(&mut self, chaine: usize, bord: &bord::Bord, jauge: Option<Graine>)
        -> Option<Resultat<(u64, f64)>> {
        let _ = (chaine, bord, jauge);
        None
    }

    /// AJOUT T3 (méthode fournie) : empreinte FNV-1a du bord lue sur le device, sans télécharger le champ.
    fn empreinte_bord_device(&self, chaine: usize) -> Option<Resultat<u64>> {
        let _ = chaine;
        None
    }

    /// AJOUT WP-DIR (méthode fournie) : masque de liens gelés, un octet par lien (4·L⁴ octets, indexé comme le champ :
    /// `li = mu·V + s`), partagé par toutes les chaînes du lot ; octet non nul = lien gelé (ni bain thermique ni
    /// overrelaxation). `None` retire le masque. `application` ne vaut autre chose que `Tout` que pour les mutants.
    /// Le générateur F32 refuse un masque (erreur : un masque ne doit jamais être ignoré en silence).
    fn fixer_masque(&mut self, masque: Option<&[u8]>, application: AppMasque) -> Resultat<()> {
        let _ = application;
        match masque {
            None => Ok(()),
            Some(_) => Err(Erreur::Config("ce générateur ne gère pas le masque de liens gelés (F64 requis : CPU, HIP ou CUDA)".into())),
        }
    }

    /// AJOUT M4 : torsion de 't Hooft. Corps par défaut : Ok si triviale, sinon Erreur::Config.
    fn fixer_torsion(&mut self, t: &crate::su2::torsion::Torsion) -> Resultat<()> {
        if t.est_triviale() {
            Ok(())
        } else {
            Err(Erreur::Config("torsion de 't Hooft : générateur CPU F64 requis".into()))
        }
    }
}

/// Vérifie la cohérence d'une configuration de lot (L pair ≥ 2, au moins une chaîne, β fini ≥ 0).
pub fn valider(cfg: &ConfigLot) -> Resultat<()> {
    if cfg.l < 2 || !cfg.l.is_multiple_of(2) {
        return Err(Erreur::Config(format!("L = {} : L pair ≥ 2 requis", cfg.l)));
    }
    if cfg.chaines.is_empty() {
        return Err(Erreur::Config("lot vide : au moins une chaîne".into()));
    }
    // les indices de lien (4V) et de site tiennent dans un u32 (compteur Philox) : L ≤ 180
    if cfg.l > 180 {
        return Err(Erreur::Config(format!("L = {} : trop grand pour un compteur Philox sur 32 bits", cfg.l)));
    }
    for (c, p) in cfg.chaines.iter().enumerate() {
        if p.beta.is_nan() || p.beta < 0.0 || !p.beta.is_finite() {
            return Err(Erreur::Config(format!("chaîne {c} : β = {} invalide (fini, ≥ 0)", p.beta)));
        }
    }
    Ok(())
}

/// Construit l'implémentation adaptée : `app` None ⇒ CPU (F64 seulement) ; F64 ⇒ Ref64Gpu ;
/// F32 ⇒ Rapide32Gpu.
pub fn construire(app: Option<&Appareil>, cfg: ConfigLot) -> Resultat<Box<dyn NoyauxSu2>> {
    valider(&cfg)?;
    match (app, cfg.precision) {
        (None, Precision::F64) => Ok(Box::new(cpu::Cpu64::nouveau(cfg)?)),
        (None, Precision::F32) => {
            Err(Erreur::Config("précision f32 : un GPU est requis (le CPU ne calcule qu'en f64)".into()))
        }
        (Some(a), Precision::F64) => Ok(Box::new(ref64::Ref64Gpu::nouveau(a, cfg)?)),
        (Some(a), Precision::F32) => Ok(Box::new(rapide32::Rapide32Gpu::nouveau(a, cfg)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commun::Depart;

    fn cfg(l: usize, beta: f64) -> ConfigLot {
        ConfigLot {
            l,
            chaines: vec![ParamChaine { beta, graine: Graine(1), depart: Depart::Chaud }],
            nor: 4,
            precision: Precision::F64,
            options: crate::gpu::OptionsCompil::reference_k1(),
            mutant_metro: false,
        }
    }

    #[test]
    fn validation_des_configurations() {
        assert!(valider(&cfg(4, 2.3)).is_ok());
        assert!(valider(&cfg(3, 2.3)).is_err());
        assert!(valider(&cfg(0, 2.3)).is_err());
        assert!(valider(&cfg(4, -1.0)).is_err());
        assert!(valider(&cfg(4, f64::NAN)).is_err());
        assert!(valider(&cfg(4, f64::INFINITY)).is_err());
        assert!(valider(&cfg(400, 2.3)).is_err());
        let mut vide = cfg(4, 2.3);
        vide.chaines.clear();
        assert!(valider(&vide).is_err());
    }

    #[test]
    fn construire_sans_gpu_donne_le_cpu() {
        let n = construire(None, cfg(4, 2.3)).unwrap();
        assert_eq!(n.backend(), BackendKind::Cpu);
        assert_eq!(n.precision(), Precision::F64);
        assert!(n.champ_gpu().is_none());
        let mut f32 = cfg(4, 2.3);
        f32.precision = Precision::F32;
        assert!(matches!(construire(None, f32), Err(Erreur::Config(_))));
    }

    #[test]
    fn serde_de_la_configuration() {
        let c = cfg(6, 2.5);
        let j = serde_json::to_string(&c).unwrap();
        let r: ConfigLot = serde_json::from_str(&j).unwrap();
        assert_eq!(c, r);
    }
}
