//! Points de reprise atomiques d'une chaîne conditionnelle. Propriétaire : T5.
//!
//! Un point de reprise = le champ en bits bruts (f64 little-endian), les mesures accumulées (F, D, blk),
//! la source d'aléa (état PCG64 ou compteur Philox) et la position. REPRISE À L'OCTET : lire un point de
//! reprise puis finir la chaîne donne exactement les mêmes sorties qu'une chaîne ininterrompue.
//!
//! Atomicité : les fichiers compagnons portent un numéro de génération (`reprise.<gen>.<nom>`), écrits et
//! synchronisés d'abord ; `reprise.json` (qui les nomme et donne leur sha256) est renommé EN DERNIER. Un arrêt
//! brutal laisse donc soit l'ancienne génération complète, soit la nouvelle. Les générations périmées sont
//! supprimées après coup.

use super::moteur::SourceAlea;
use super::npy_min::ecrire_atomique;
use crate::commun::{Erreur, Resultat};
use crate::heritage::manifest::{hex, Sha256};
use std::path::{Path, PathBuf};

pub const NOM_REPRISE: &str = "reprise.json";
const SCHEMA: &str = "kvu-cond-reprise/1";

/// Empreinte d'un fichier compagnon.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Compagnon {
    pub nom: String,
    pub sha256: String,
    pub octets: u64,
}

/// Contenu de `reprise.json`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EtatReprise {
    pub schema: String,
    pub generation: u64,
    pub sha_cfg: String,
    pub l: usize,
    pub beta: f64,
    pub graine: u64,
    pub mode: super::ModeCond,
    pub ntherm: u64,
    pub nsweep: u64,
    pub every: u64,
    pub deach: u64,
    pub nf: usize,
    /// Balayages de mesure déjà faits (thermalisation exclue).
    pub t: u64,
    pub n_f: u64,
    pub n_d: u64,
    pub alea: SourceAlea,
    pub champ: Compagnon,
    pub f: Compagnon,
    pub d: Compagnon,
    pub blk: Compagnon,
    /// Lecture tordue (`--tordu`) : Ft et Dtw accumulés (absents des points de reprise sans torsion).
    #[serde(default)]
    pub tordu: bool,
    #[serde(default)]
    pub ft: Option<Compagnon>,
    #[serde(default)]
    pub dtw: Option<Compagnon>,
}

/// Données lues d'un point de reprise.
pub struct Reprise {
    pub etat: EtatReprise,
    pub champ: Vec<f64>,
    pub f: Vec<f64>,
    pub d: Vec<f64>,
    pub blk: Vec<f32>,
    /// Ft et Dtw accumulés (vides sans lecture tordue).
    pub ft: Vec<f64>,
    pub dtw: Vec<f64>,
}

fn sha_octets(b: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b);
    hex(&h.finalize())
}

fn octets_f64(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn octets_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ecrire_compagnon(dir: &Path, generation: u64, suffixe: &str, octets: &[u8]) -> Resultat<Compagnon> {
    let nom = format!("reprise.{generation}.{suffixe}");
    ecrire_atomique(&dir.join(&nom), &[octets])?;
    Ok(Compagnon { nom, sha256: sha_octets(octets), octets: octets.len() as u64 })
}

fn lire_compagnon(dir: &Path, c: &Compagnon) -> Resultat<Vec<u8>> {
    let p = dir.join(&c.nom);
    let b = std::fs::read(&p).map_err(|e| Erreur::io(format!("lecture {}", p.display()), e))?;
    if b.len() as u64 != c.octets || sha_octets(&b) != c.sha256 {
        return Err(Erreur::Format(format!(
            "point de reprise corrompu : {} (taille ou sha256 différent)",
            p.display()
        )));
    }
    Ok(b)
}

fn f64_de(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn f32_de(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Octets qu'occupera une écriture de point de reprise (une génération), pour la garde SSD.
pub fn octets_projetes(champ: usize, f: usize, d: usize, blk: usize) -> u64 {
    (8 * champ + 8 * f + 8 * d + 4 * blk) as u64
}

/// Écrit un point de reprise de façon atomique et supprime les générations périmées.
/// `modele` fournit les champs fixes ; les compagnons et la génération sont posés ici.
pub fn ecrire(dir: &Path, etat: EtatReprise, champ: &[f64], f: &[f64], d: &[f64], blk: &[f32]) -> Resultat<()> {
    ecrire_avec(dir, etat, champ, f, d, blk, None)
}

/// Comme [`ecrire`], avec les accumulateurs de la lecture tordue `(Ft, Dtw)` s'il y en a.
pub fn ecrire_avec(
    dir: &Path,
    mut etat: EtatReprise,
    champ: &[f64],
    f: &[f64],
    d: &[f64],
    blk: &[f32],
    tordu: Option<(&[f64], &[f64])>,
) -> Resultat<()> {
    let generation = etat.generation;
    etat.schema = SCHEMA.to_string();
    etat.champ = ecrire_compagnon(dir, generation, "champ", &octets_f64(champ))?;
    etat.f = ecrire_compagnon(dir, generation, "F", &octets_f64(f))?;
    etat.d = ecrire_compagnon(dir, generation, "D", &octets_f64(d))?;
    etat.blk = ecrire_compagnon(dir, generation, "blk", &octets_f32(blk))?;
    etat.tordu = tordu.is_some();
    (etat.ft, etat.dtw) = match tordu {
        Some((ft, dtw)) => (
            Some(ecrire_compagnon(dir, generation, "Ft", &octets_f64(ft))?),
            Some(ecrire_compagnon(dir, generation, "Dtw", &octets_f64(dtw))?),
        ),
        None => (None, None),
    };
    let json = serde_json::to_vec_pretty(&etat).map_err(|e| Erreur::Format(e.to_string()))?;
    ecrire_atomique(&dir.join(NOM_REPRISE), &[&json])?;
    nettoyer(dir, Some(generation));
    Ok(())
}

/// Supprime les compagnons de toute génération autre que `garder` (None : toutes) ainsi que `reprise.json`
/// si `garder` est None.
pub fn nettoyer(dir: &Path, garder: Option<u64>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let nom = e.file_name().to_string_lossy().into_owned();
        let Some(reste) = nom.strip_prefix("reprise.") else { continue };
        if nom == NOM_REPRISE {
            if garder.is_none() {
                let _ = std::fs::remove_file(e.path());
            }
            continue;
        }
        let gen_du_nom = reste.split('.').next().and_then(|g| g.parse::<u64>().ok());
        if gen_du_nom != garder || reste.ends_with(".part") {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

/// Lit et vérifie (sha256) le point de reprise de `dir`.
pub fn lire(dir: &Path) -> Resultat<Reprise> {
    let p: PathBuf = dir.join(NOM_REPRISE);
    let t = std::fs::read_to_string(&p).map_err(|e| Erreur::io(format!("lecture {}", p.display()), e))?;
    let etat: EtatReprise = serde_json::from_str(&t).map_err(|e| Erreur::Format(format!("{} : {e}", p.display())))?;
    if etat.schema != SCHEMA {
        return Err(Erreur::Format(format!("{} : schéma « {} » inattendu", p.display(), etat.schema)));
    }
    let champ = f64_de(&lire_compagnon(dir, &etat.champ)?);
    let f = f64_de(&lire_compagnon(dir, &etat.f)?);
    let d = f64_de(&lire_compagnon(dir, &etat.d)?);
    let blk = f32_de(&lire_compagnon(dir, &etat.blk)?);
    let lire_option = |c: &Option<Compagnon>| -> Resultat<Vec<f64>> {
        match c {
            Some(c) => Ok(f64_de(&lire_compagnon(dir, c)?)),
            None => Ok(Vec::new()),
        }
    };
    let (ft, dtw) = (lire_option(&etat.ft)?, lire_option(&etat.dtw)?);
    Ok(Reprise { etat, champ, f, d, blk, ft, dtw })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cond::ModeCond;

    fn etat(generation: u64) -> EtatReprise {
        let vide = Compagnon { nom: String::new(), sha256: String::new(), octets: 0 };
        EtatReprise {
            schema: String::new(),
            generation,
            sha_cfg: "abc".into(),
            l: 4,
            beta: 2.3,
            graine: 7,
            mode: ModeCond::RefRng,
            ntherm: 1,
            nsweep: 10,
            every: 2,
            deach: 5,
            nf: 2,
            t: 4,
            n_f: 2,
            n_d: 1,
            alea: SourceAlea::depuis_graine(ModeCond::RefRng, 7),
            champ: vide.clone(),
            f: vide.clone(),
            d: vide.clone(),
            blk: vide,
            tordu: false,
            ft: None,
            dtw: None,
        }
    }

    #[test]
    fn aller_retour_avec_lecture_tordue() {
        let d = dossier("t");
        ecrire_avec(&d, etat(1), &[1.0], &[1.0], &[], &[], Some((&[0.5, 0.25], &[2.0; 3]))).unwrap();
        let r = lire(&d).unwrap();
        assert!(r.etat.tordu);
        assert_eq!((r.ft, r.dtw), (vec![0.5, 0.25], vec![2.0; 3]));
        // sans torsion : aucun compagnon Ft/Dtw, lecture vide
        ecrire(&d, etat(2), &[1.0], &[1.0], &[], &[]).unwrap();
        let r = lire(&d).unwrap();
        assert!(!r.etat.tordu && r.ft.is_empty() && r.dtw.is_empty());
    }

    fn dossier(nom: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kvu_cond_reprise_{nom}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn aller_retour_a_l_octet() {
        let d = dossier("a");
        let champ = vec![1.0, -0.0, f64::MIN_POSITIVE, 0.1 + 0.2];
        ecrire(&d, etat(1), &champ, &[1.5, 2.5], &[0.25; 4], &[1.0f32, 2.0]).unwrap();
        let r = lire(&d).unwrap();
        assert_eq!(
            r.champ.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            champ.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(r.f, vec![1.5, 2.5]);
        assert_eq!(r.blk, vec![1.0f32, 2.0]);
        assert_eq!(r.etat.alea, SourceAlea::depuis_graine(ModeCond::RefRng, 7));
        assert_eq!(r.etat.t, 4);
    }

    #[test]
    fn generations_perimees_supprimees() {
        let d = dossier("b");
        ecrire(&d, etat(1), &[1.0], &[1.0], &[], &[]).unwrap();
        ecrire(&d, etat(2), &[2.0], &[2.0], &[], &[]).unwrap();
        let noms: Vec<String> =
            std::fs::read_dir(&d).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        assert!(noms.iter().all(|n| !n.starts_with("reprise.1.")), "{noms:?}");
        assert!(noms.iter().any(|n| n.starts_with("reprise.2.")));
        assert_eq!(lire(&d).unwrap().champ, vec![2.0]);
        nettoyer(&d, None);
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0);
    }

    #[test]
    fn corruption_detectee() {
        let d = dossier("c");
        ecrire(&d, etat(1), &[1.0, 2.0], &[1.0], &[], &[]).unwrap();
        let p = d.join("reprise.1.champ");
        let mut b = std::fs::read(&p).unwrap();
        b[3] ^= 0x01;
        std::fs::write(&p, b).unwrap();
        assert!(matches!(lire(&d), Err(Erreur::Format(_))));
    }

    #[test]
    fn arret_entre_compagnons_et_json_laisse_l_ancienne_generation() {
        // simule un arrêt brutal : compagnons de la génération 2 écrits, reprise.json non renommé
        let d = dossier("d");
        ecrire(&d, etat(1), &[1.0], &[1.0], &[], &[]).unwrap();
        ecrire_compagnon(&d, 2, "champ", &octets_f64(&[9.0])).unwrap();
        assert_eq!(lire(&d).unwrap().champ, vec![1.0]);
    }
}
