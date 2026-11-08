//! Garde SSD et verrou GPU de `cond chaine`. Propriétaire : T5.
//!
//! - garde SSD : `stockage::GardeSsd` (T8), seuil de la machine (clé `ssd_min_go` du profil ; surcharge
//!   `KEVINOTRON_SSD_MIN_GO`) ; refus AVANT tout octet écrit (code 7) ;
//! - verrou GPU : `machine::VerrouGpu` (T1) ; un `flock` externe tenu par un ancêtre est reconnu (pas d'interblocage) ;
//! - admission mémoire : `machine::admission` (RAM disponible − besoin estimé ≥ plancher de la politique, rc 5).

use crate::commun::Resultat;
use crate::gpu::Appareil;
use crate::machine::admission::{octets_champ, BesoinMemoire};
use crate::stockage::GardeSsd;
use std::path::Path;

/// Ce que la garde SSD a constaté (consigné dans le manifeste).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct InfoSsd {
    pub seuil_go: f64,
    pub libre_go: f64,
    pub projection_go: f64,
}

/// Refuse (Erreur::SsdPlein, code 7) si l'écriture de `octets` laisserait moins que le seuil de Go libres.
pub fn verifier_ssd(chemin: &Path, octets: u64) -> Resultat<InfoSsd> {
    let garde = GardeSsd::machine();
    garde.verifier(chemin, octets)?;
    Ok(InfoSsd { seuil_go: garde.seuil_go, libre_go: GardeSsd::libre_go(chemin)?, projection_go: octets as f64 / crate::stockage::garde::OCTETS_PAR_GO })
}

/// Mémoire estimée d'une chaîne conditionnelle à côté L (`nmes` mesures F de `nf` composantes, `nd` mesures D) :
/// - hôte : les mesures accumulées (F, D, blk) et leur copie sérialisée au point de reprise, trois champs f64
///   (départ, rapatriement, point de reprise) ; en CPU en plus les plaquettes et gradients de la mesure de fibre
///   (78 réels par site) et les tables de géométrie (8 entiers par site) ;
/// - GPU : trois champs, plaquettes et gradients, tables.
pub fn besoin_memoire(l: usize, nmes: usize, nd: usize, nf: usize, gpu: bool) -> BesoinMemoire {
    let v = (l as u64).pow(4);
    let champ = octets_champ(l, 8);
    let lc4 = (l as u64 / 2).pow(4);
    let mesures = 8 * (nmes * nf) as u64 + 8 * (nd * nf * nf) as u64 + 4 * nd as u64 * 6 * lc4;
    let travail = 78 * 8 * v + 64 * v;
    BesoinMemoire {
        hote: 2 * mesures + 3 * champ + if gpu { 0 } else { travail },
        gpu: if gpu { 3 * champ + travail } else { 0 },
    }
}

/// Admission mémoire d'une commande `cond` (rc 5 si la RAM disponible passerait sous le plancher).
pub fn admettre(quoi: &str, besoin: BesoinMemoire, app: Option<&Appareil>) -> Resultat<()> {
    let unifie = app.is_some_and(|a| crate::machine::detection::arch_memoire_unifiee(&a.arch()));
    crate::machine::admission::admettre(quoi, besoin, unifie)
}

/// Verrou GPU : garde RAII et mode (« pris_ici », « ancetre », « partage », « ignore », ou « sans_gpu »).
pub struct VerrouCond {
    _garde: Option<crate::machine::GardeVerrou>,
    pub etat: &'static str,
}

/// Prend le verrou GPU de la machine si on travaille sur un GPU ; sans GPU, aucun verrou.
pub fn prendre_verrou(app: Option<&Appareil>) -> Resultat<VerrouCond> {
    if app.is_none() {
        return Ok(VerrouCond { _garde: None, etat: "sans_gpu" });
    }
    let politique = crate::machine::politique(&crate::machine::detection::detecter_rapide());
    let verrou = crate::machine::VerrouGpu::nouveau(&politique.verrou_gpu);
    let garde = verrou.acquerir(std::time::Duration::from_secs(politique.attente_verrou_s))?;
    let etat = garde.mode();
    Ok(VerrouCond { _garde: Some(garde), etat })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commun::Erreur;

    #[test]
    fn projection_et_libre_renseignes() {
        std::env::set_var("KEVINOTRON_SSD_MIN_GO", "0");
        let i = verifier_ssd(&std::env::temp_dir(), 1_000_000);
        std::env::remove_var("KEVINOTRON_SSD_MIN_GO");
        let i = i.unwrap();
        assert!(i.libre_go > 0.0 && (i.projection_go - 1e6 / crate::stockage::garde::OCTETS_PAR_GO).abs() < 1e-12);
    }

    #[test]
    fn seuil_trop_haut_refuse_avant_toute_ecriture() {
        // « seuil forcé au-dessus du libre ⇒ refus » (SSD-1)
        std::env::set_var("KEVINOTRON_SSD_MIN_GO", "1e9");
        let r = verifier_ssd(&std::env::temp_dir(), 1);
        std::env::remove_var("KEVINOTRON_SSD_MIN_GO");
        match r {
            Err(e @ Erreur::SsdPlein { .. }) => assert_eq!(e.code_sortie(), 7),
            autre => panic!("refus attendu, obtenu {autre:?}"),
        }
    }

    #[test]
    fn sans_gpu_pas_de_verrou() {
        assert_eq!(prendre_verrou(None).unwrap().etat, "sans_gpu");
    }
}
