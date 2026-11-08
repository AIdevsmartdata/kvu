//! Préparation du processus avant toute sous-commande : affinité CPU de la politique et contrôle de la
//! surcharge de machine. Propriétaire : T1 (correctif du vérificateur, 04/10).
//!
//! Avant ce correctif, l'affinité « obligatoire » du profil (ex. 12-19) (cœurs E ; les cœurs P sont instables et
//! réservés à maison3d) n'était appliquée que par la file de jobs (`taskset`). Un appel direct
//! (`kevinotron cond chaine --backend cpu`, `su2 gen`…) tournait sur les 20 CPU. Désormais le binaire pose
//! lui-même son affinité (`sched_setaffinity`) dès que la politique a `cpus`, AVANT de créer le moindre fil :
//! tous les fils (rayon, runtime GPU) et tous les processus enfants en héritent.
//!
//! L'affinité posée est l'INTERSECTION de l'affinité courante et de la politique : un `taskset -c 14-15`
//! externe plus étroit est respecté ; une affinité disjointe de la politique est refusée (rc 5).

use super::detection::{detecter_rapide, ProfilMachine};
use super::politique::{politique, Politique};
use crate::commun::{Erreur, Resultat};
use std::ffi::c_int;

/// Taille du masque `cpu_set_t` de la glibc (1024 CPU).
const OCTETS_MASQUE: usize = 128;

extern "C" {
    fn sched_getaffinity(pid: c_int, taille: usize, masque: *mut u8) -> c_int;
    fn sched_setaffinity(pid: c_int, taille: usize, masque: *const u8) -> c_int;
}

/// Analyse une liste de CPU au format `taskset -c` / `Cpus_allowed_list` : « 12-19 », « 0,2,4-5 ».
/// Rend la liste triée sans doublon ; vide ou mal formée ⇒ erreur.
pub fn analyser_liste_cpus(s: &str) -> Resultat<Vec<usize>> {
    let invalide = || Erreur::Config(format!("liste de CPU « {s} » invalide (ex. 12-19 ou 0,2,4-5)"));
    let mut v = Vec::new();
    for morceau in s.trim().split(',') {
        let morceau = morceau.trim();
        let (a, b) = match morceau.split_once('-') {
            Some((a, b)) => (a.trim(), b.trim()),
            None => (morceau, morceau),
        };
        let (a, b): (usize, usize) = (a.parse().map_err(|_| invalide())?, b.parse().map_err(|_| invalide())?);
        if a > b || b >= 8 * OCTETS_MASQUE {
            return Err(invalide());
        }
        v.extend(a..=b);
    }
    v.sort_unstable();
    v.dedup();
    if v.is_empty() {
        return Err(invalide());
    }
    Ok(v)
}

/// Écrit une liste triée de CPU au format compact « 0,2,4-5 ».
pub fn formater_liste_cpus(cpus: &[usize]) -> String {
    let mut morceaux = Vec::new();
    let mut i = 0;
    while i < cpus.len() {
        let mut j = i;
        while j + 1 < cpus.len() && cpus[j + 1] == cpus[j] + 1 {
            j += 1;
        }
        morceaux.push(if i == j { cpus[i].to_string() } else { format!("{}-{}", cpus[i], cpus[j]) });
        i = j + 1;
    }
    morceaux.join(",")
}

/// Intersection triée de deux listes triées.
pub fn intersection(a: &[usize], b: &[usize]) -> Vec<usize> {
    a.iter().copied().filter(|x| b.binary_search(x).is_ok()).collect()
}

/// Affinité du fil `tid` (0 = fil appelant), `sched_getaffinity`.
fn affinite_de(tid: c_int) -> Resultat<Vec<usize>> {
    let mut masque = [0u8; OCTETS_MASQUE];
    // SAFETY : tampon de 128 octets = sizeof(cpu_set_t).
    if unsafe { sched_getaffinity(tid, OCTETS_MASQUE, masque.as_mut_ptr()) } != 0 {
        return Err(Erreur::io("sched_getaffinity", std::io::Error::last_os_error()));
    }
    Ok((0..8 * OCTETS_MASQUE).filter(|c| masque[c / 8] & (1 << (c % 8)) != 0).collect())
}

/// Fixe l'affinité du fil `tid` (0 = fil appelant).
fn fixer_affinite_de(tid: c_int, cpus: &[usize]) -> Resultat<()> {
    let mut masque = [0u8; OCTETS_MASQUE];
    for &c in cpus {
        if c >= 8 * OCTETS_MASQUE {
            return Err(Erreur::Config(format!("CPU {c} hors du masque")));
        }
        masque[c / 8] |= 1 << (c % 8);
    }
    // SAFETY : masque de 128 octets.
    if unsafe { sched_setaffinity(tid, OCTETS_MASQUE, masque.as_ptr()) } != 0 {
        return Err(Erreur::io("sched_setaffinity", std::io::Error::last_os_error()));
    }
    Ok(())
}

/// Affinité du fil courant (`sched_getaffinity`).
pub fn affinite_courante() -> Resultat<Vec<usize>> {
    affinite_de(0)
}

/// Fixe l'affinité du fil courant ; les fils et processus créés ENSUITE en héritent.
pub fn fixer_affinite(cpus: &[usize]) -> Resultat<()> {
    fixer_affinite_de(0, cpus)
}

/// Étend l'affinité du fil PRINCIPAL à tous les fils du processus. Les runtimes GPU (ROCr/HSA, pilote CUDA) créent
/// des fils internes avec leur propre affinité (tous les CPU) : on la leur retire après chaque initialisation
/// (`machine::detecter`, `gpu::ouvrir`). Au mieux : un fil disparu entre-temps est ignoré.
pub fn reappliquer_affinite() {
    let pid = std::process::id() as c_int;
    let Ok(principal) = affinite_de(pid) else { return };
    let Ok(taches) = std::fs::read_dir("/proc/self/task") else { return };
    for t in taches.flatten() {
        let Some(tid) = t.file_name().to_str().and_then(|s| s.parse::<c_int>().ok()) else { continue };
        if tid != pid && affinite_de(tid).is_ok_and(|a| a != principal) {
            let _ = fixer_affinite_de(tid, &principal);
        }
    }
}

/// Décision PURE : affinité à poser (None = rien à changer) d'après l'affinité courante et la politique.
pub fn affinite_voulue(courante: &[usize], politique_cpus: Option<&str>) -> Resultat<Option<Vec<usize>>> {
    let Some(liste) = politique_cpus else { return Ok(None) };
    let voulus = analyser_liste_cpus(liste)?;
    let inter = intersection(courante, &voulus);
    if inter.is_empty() {
        return Err(Erreur::Ressources(format!(
            "affinité courante {} disjointe de l'affinité obligatoire {liste} de la politique",
            formater_liste_cpus(courante)
        )));
    }
    Ok((inter != courante).then_some(inter))
}

/// Applique l'affinité de la politique au processus courant (à appeler AVANT de créer des fils).
/// Rend l'affinité effective.
pub fn appliquer_affinite(pol: &Politique) -> Resultat<Vec<usize>> {
    let courante = affinite_courante()?;
    match affinite_voulue(&courante, pol.cpus.as_deref())? {
        Some(v) => {
            fixer_affinite(&v)?;
            Ok(v)
        }
        None => Ok(courante),
    }
}

/// `Cpus_allowed_list` du processus (lu dans /proc/self/status), pour `doctor`.
pub fn cpus_autorises() -> Option<String> {
    let t = std::fs::read_to_string("/proc/self/status").ok()?;
    t.lines()
        .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
        .map(|v| v.trim().to_string())
}

/// Refus d'une surcharge de machine incohérente (rc 2), lu dans le profil.
pub fn controler_profil(p: &ProfilMachine) -> Resultat<()> {
    match &p.surcharge_refusee {
        Some(m) => Err(Erreur::Config(m.clone())),
        None => Ok(()),
    }
}

/// Préparation commune à TOUTES les sous-commandes (appelée par `cli::executer_argv` avant le module) :
/// refus d'une surcharge KEVINOTRON_MACHINE incohérente, puis affinité de la politique.
pub fn preparer() -> Resultat<()> {
    let p = detecter_rapide();
    controler_profil(&p)?;
    appliquer_affinite(&politique(&p))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listes_de_cpu() {
        assert_eq!(analyser_liste_cpus("12-19").unwrap(), (12..=19).collect::<Vec<_>>());
        assert_eq!(analyser_liste_cpus(" 0,2,4-5 ,4").unwrap(), vec![0, 2, 4, 5]);
        for mauvais in ["", "a", "5-3", "1-", "-", "2000"] {
            assert!(analyser_liste_cpus(mauvais).is_err(), "{mauvais:?}");
        }
        assert_eq!(formater_liste_cpus(&[0, 2, 4, 5, 12, 13, 14]), "0,2,4-5,12-14");
        assert_eq!(formater_liste_cpus(&(12..=19).collect::<Vec<_>>()), "12-19");
    }

    /// Le cas du vérificateur : processus sur 0-19, politique du gamer 12-19 ⇒ on restreint à 12-19.
    #[test]
    fn decision_d_affinite() {
        let tous: Vec<usize> = (0..20).collect();
        let e: Vec<usize> = (12..20).collect();
        assert_eq!(affinite_voulue(&tous, Some("12-19")).unwrap(), Some(e.clone()));
        // déjà dans la politique : rien à changer
        assert_eq!(affinite_voulue(&e, Some("12-19")).unwrap(), None);
        // taskset externe plus étroit : respecté
        assert_eq!(affinite_voulue(&[14, 15], Some("12-19")).unwrap(), None);
        // chevauchement partiel : intersection
        assert_eq!(affinite_voulue(&[10, 11, 12, 13], Some("12-19")).unwrap(), Some(vec![12, 13]));
        // disjoint : refus (rc 5)
        assert_eq!(affinite_voulue(&[0, 1], Some("12-19")).unwrap_err().code_sortie(), 5);
        // pas d'affinité dans la politique (acemagic) : rien
        assert_eq!(affinite_voulue(&tous, None).unwrap(), None);
    }

    #[test]
    fn affinite_lue_non_vide() {
        let c = affinite_courante().unwrap();
        assert!(!c.is_empty());
        assert!(cpus_autorises().is_some());
    }
}
