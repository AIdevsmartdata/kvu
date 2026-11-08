//! Admission mémoire des runs DIRECTS (su2 gen, cond, obs, fibre), comme la file le fait pour ses jobs.
//! Propriétaire : T1 (correctif du vérificateur, 04/10).
//!
//! Avant ce correctif, seul `doctor` CONSEILLAIT un L maximal : un appel direct à grand L ou à gros lot
//! pouvait faire tomber MemAvailable sous 15 Go (zone des fautes GPU silencieuses d'une APU), voire sous le
//! seuil de ram-guard, et frapper la production. Règle : un run démarre seulement si
//!
//! ```text
//! RAM disponible − besoin estimé ≥ plancher de la politique (clé `ram_min_go` du profil ; KEVINOTRON_RAM_MIN_GO)
//! ```
//!
//! sinon `Erreur::Ressources` (rc 5) AVANT toute écriture et avant d'ouvrir le GPU. Sur une APU la mémoire
//! est UNIFIÉE : les tampons GPU comptent dans la RAM ; avec une VRAM dédiée seul l'hôte compte.

use super::detection::{detecter_rapide, ram_disponible_go};
use super::politique::politique;
use crate::commun::{Erreur, Resultat};

const GIB: f64 = (1u64 << 30) as f64;
/// Réserve fixe par processus (binaire, runtime GPU, tampons de fichiers) : 256 Mio.
pub const RESERVE_PROCESSUS: u64 = 256 << 20;

/// Mémoire estimée d'un run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BesoinMemoire {
    /// Mémoire de l'hôte (octets).
    pub hote: u64,
    /// Mémoire du GPU (octets) ; comptée dans la RAM si le GPU partage la mémoire de l'hôte.
    pub gpu: u64,
}

impl BesoinMemoire {
    /// Octets pris sur la RAM de l'hôte (réserve du processus comprise).
    pub fn ram_octets(&self, gpu_unifie: bool) -> u64 {
        self.hote
            .saturating_add(if gpu_unifie { self.gpu } else { 0 })
            .saturating_add(RESERVE_PROCESSUS)
    }
}

/// Décision PURE : `quoi` demande `besoin_go` Go ; refus si `dispo_go − besoin_go < plancher_go`.
pub fn decider(quoi: &str, besoin_go: f64, dispo_go: f64, plancher_go: f64) -> Resultat<()> {
    let reste = dispo_go - besoin_go;
    if reste < plancher_go {
        return Err(Erreur::Ressources(format!(
            "{quoi} demande ~{besoin_go:.1} Go de RAM ; disponible {dispo_go:.1} Go : il resterait {reste:.1} Go \
             < plancher {plancher_go:.0} Go de la politique (KEVINOTRON_RAM_MIN_GO) — réduire L ou le lot, ou attendre"
        )));
    }
    Ok(())
}

/// Admission du run `quoi` sur la machine courante (RAM disponible lue dans /proc/meminfo).
pub fn admettre(quoi: &str, besoin: BesoinMemoire, gpu_unifie: bool) -> Resultat<()> {
    let plancher = politique(&detecter_rapide()).ram_min_go;
    decider(quoi, besoin.ram_octets(gpu_unifie) as f64 / GIB, ram_disponible_go(), plancher)
}

/// Octets d'un champ SU(2) L⁴ (4 directions × 4 réels par lien) en `octets_reel` octets par réel.
pub fn octets_champ(l: usize, octets_reel: u64) -> u64 {
    16u64.saturating_mul((l as u64).saturating_pow(4)).saturating_mul(octets_reel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_aux_bornes() {
        assert!(decider("x", 10.0, 35.0, 25.0).is_ok(), "pile au plancher");
        let e = decider("su2 gen L=64", 10.1, 35.0, 25.0).unwrap_err();
        assert_eq!(e.code_sortie(), 5);
        assert!(e.to_string().contains("su2 gen L=64"), "{e}");
        // déjà sous le plancher : même un besoin nul est refusé
        assert!(decider("x", 0.0, 20.0, 25.0).is_err());
    }

    #[test]
    fn memoire_unifiee_compte_le_gpu() {
        let b = BesoinMemoire { hote: 1 << 30, gpu: 4 << 30 };
        assert_eq!(b.ram_octets(true), (5 << 30) + RESERVE_PROCESSUS);
        assert_eq!(b.ram_octets(false), (1 << 30) + RESERVE_PROCESSUS);
        assert_eq!(octets_champ(48, 8), 16 * 48u64.pow(4) * 8);
    }
}
