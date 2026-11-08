//! Points de reprise (checkpoint / restart) à l'octet. Propriétaire : T3.
//!
//! Un point de reprise = `reprise.json` (EtatReprise, serde) + `reprise.champs` (octets bruts des champs de
//! TOUTES les chaînes, chaîne après chaîne : f64 little-endian de `telecharger`). Pour le F32, f32 → f64 → f32
//! est exact : le fichier de reprise restitue donc le champ F32 au bit près (voir `champ::f64_vers_f32`).
//!
//! État d'une chaîne Philox = (champ, sweep_id, compteurs, position) : le compteur de sweeps est le SEUL état
//! du générateur, car les tirages sont des fonctions pures de (clé, lien, sweep, tirage, étiquette).
//!
//! Écriture ATOMIQUE et tolérante au crash (aucun instant où reprise.json désigne un champ introuvable) :
//! 1. `reprise.champs.new` écrit et fsyncé ;
//! 2. l'ancien `reprise.champs` devient `reprise.champs.prev` ;
//! 3. `.new` devient `reprise.champs` ;
//! 4. `reprise.json` est remplacé (rename atomique) ;
//! 5. `.prev` est supprimé.
//!
//! À la lecture, si le sha256 de `reprise.champs` ne correspond pas au JSON, on essaie `.prev` (crash entre 3 et 4).

use super::NoyauxSu2;
use crate::commun::{Erreur, Resultat};
use crate::heritage::manifest::{hex, ecrire_atomique, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Version du format de reprise.
pub const SCHEMA: u32 = 1;
pub const NOM_JSON: &str = "reprise.json";
pub const NOM_CHAMPS: &str = "reprise.champs";

/// État complet d'un run interrompu : champs bruts (bits exacts, F64 ou F32) de chaque chaîne,
/// sweep_id, compteurs, pas effectués, index de mesure, position dans les CSV, config du lot.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EtatReprise {
    pub schema: u32,
    pub config: super::ConfigLot,
    pub sweep_id: u32,
    pub pas_faits: u64,
    pub mesures_faites: u64,
    pub compteurs: Vec<super::Compteurs>,
    /// sha256 du fichier binaire des champs (vérifié à la relecture).
    pub sha256_champs: String,
}

fn io(contexte: &str, chemin: &Path, e: std::io::Error) -> Erreur {
    Erreur::io(format!("{contexte} {}", chemin.display()), e)
}

fn sha256_fichier(p: &Path) -> Resultat<String> {
    let mut f = File::open(p).map_err(|e| io("ouverture", p, e))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| io("lecture", p, e))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// Taille en octets du fichier de champs d'un lot (chaînes × 16·L⁴ × 8).
pub fn octets_champs(l: usize, nchaines: usize) -> u64 {
    (nchaines * 16 * l * l * l * l * 8) as u64
}

/// Écrit atomiquement <dir>/reprise.json + reprise.champs (garde SSD appliquée par l'appelant).
pub fn ecrire(dir: &Path, n: &dyn NoyauxSu2, pas_faits: u64, mesures_faites: u64) -> Resultat<()> {
    let cfg = n.config();
    let nch = cfg.chaines.len();
    let champs = dir.join(NOM_CHAMPS);
    let nouveau = dir.join(format!("{NOM_CHAMPS}.new"));
    let prev = dir.join(format!("{NOM_CHAMPS}.prev"));
    // (1) fichier des champs, calcul du sha256 au fil de l'écriture
    let mut h = Sha256::new();
    {
        let mut f = std::io::BufWriter::with_capacity(1 << 22, File::create(&nouveau).map_err(|e| io("création", &nouveau, e))?);
        for c in 0..nch {
            let champ = n.telecharger(c)?;
            let mut octets = Vec::with_capacity(champ.q.len() * 8);
            for x in &champ.q {
                octets.extend_from_slice(&x.to_le_bytes());
            }
            h.update(&octets);
            f.write_all(&octets).map_err(|e| io("écriture", &nouveau, e))?;
        }
        f.flush().map_err(|e| io("écriture", &nouveau, e))?;
        f.into_inner().map_err(|e| io("écriture", &nouveau, e.into_error()))?.sync_all().map_err(|e| io("fsync", &nouveau, e))?;
    }
    let etat = EtatReprise {
        schema: SCHEMA,
        config: cfg.clone(),
        sweep_id: n.sweep_id(),
        pas_faits,
        mesures_faites,
        compteurs: n.compteurs()?,
        sha256_champs: hex(&h.finalize()),
    };
    let json = serde_json::to_vec_pretty(&etat).map_err(|e| Erreur::Format(format!("reprise.json : {e}")))?;
    // (2)-(5) bascule
    if champs.exists() {
        fs::rename(&champs, &prev).map_err(|e| io("renommage", &champs, e))?;
    }
    fs::rename(&nouveau, &champs).map_err(|e| io("renommage", &nouveau, e))?;
    ecrire_atomique(&dir.join(NOM_JSON), &json).map_err(|e| io("écriture", &dir.join(NOM_JSON), e))?;
    let _ = fs::remove_file(&prev);
    Ok(())
}

/// Lit reprise.json et choisit le fichier de champs cohérent (principal, sinon `.prev`).
fn lire_etat(dir: &Path) -> Resultat<(EtatReprise, PathBuf)> {
    let pj = dir.join(NOM_JSON);
    let txt = fs::read_to_string(&pj).map_err(|e| io("lecture", &pj, e))?;
    let etat: EtatReprise = serde_json::from_str(&txt).map_err(|e| Erreur::Format(format!("{} illisible : {e}", pj.display())))?;
    if etat.schema != SCHEMA {
        return Err(Erreur::Format(format!("reprise.json de schéma {} (attendu {SCHEMA})", etat.schema)));
    }
    let attendu = octets_champs(etat.config.l, etat.config.chaines.len());
    for nom in [NOM_CHAMPS.to_string(), format!("{NOM_CHAMPS}.prev")] {
        let p = dir.join(&nom);
        let Ok(m) = fs::metadata(&p) else { continue };
        if m.len() != attendu {
            continue;
        }
        if sha256_fichier(&p)? == etat.sha256_champs {
            return Ok((etat, p));
        }
    }
    Err(Erreur::Format(format!(
        "reprise corrompue dans {} : aucun fichier de champs ne correspond au sha256 de reprise.json",
        dir.display()
    )))
}

/// Relit un point de reprise et restaure les champs et le compteur dans `n`.
/// `n` doit avoir été construit avec la MÊME configuration de lot (sinon erreur de configuration).
pub fn lire(dir: &Path, n: &mut dyn NoyauxSu2) -> Resultat<EtatReprise> {
    let (etat, chemin) = lire_etat(dir)?;
    if &etat.config != n.config() {
        return Err(Erreur::Config(format!(
            "reprise : la configuration du lot diffère de celle du point de reprise ({})",
            dir.join(NOM_JSON).display()
        )));
    }
    let l = etat.config.l;
    let nvals = 16 * l * l * l * l;
    let mut f = File::open(&chemin).map_err(|e| io("ouverture", &chemin, e))?;
    let mut octets = vec![0u8; nvals * 8];
    for c in 0..etat.config.chaines.len() {
        f.read_exact(&mut octets).map_err(|e| io("lecture", &chemin, e))?;
        let q: Vec<f64> = octets.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap_or([0; 8]))).collect();
        let champ = crate::commun::ChampHote { l, q };
        n.televerser(c, &champ)?;
        // vérification : le générateur doit restituer le champ AU BIT (un générateur qui renormalise au
        // téléversement ferait diverger silencieusement la chaîne reprise)
        let relu = n.telecharger(c)?;
        if relu.q.iter().zip(&champ.q).any(|(a, b)| a.to_bits() != b.to_bits()) {
            return Err(Erreur::Validation(format!(
                "reprise : la chaîne {c} n'est pas restituée au bit par le générateur ({}) : reprise à l'octet impossible",
                n.backend().nom()
            )));
        }
    }
    n.fixer_sweep_id(etat.sweep_id);
    n.synchroniser()?;
    Ok(etat)
}

/// Lit seulement l'état (sans restaurer) : le répertoire de run existe-t-il avec une reprise valide ?
pub fn lire_etat_seul(dir: &Path) -> Resultat<EtatReprise> {
    lire_etat(dir).map(|x| x.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commun::{Depart, Graine, Precision};
    use crate::gpu::OptionsCompil;
    use crate::su2::cpu::Cpu64;
    use crate::su2::{ConfigLot, ParamChaine};

    fn cfg(nch: usize) -> ConfigLot {
        ConfigLot {
            l: 4,
            chaines: (0..nch)
                .map(|i| ParamChaine { beta: 2.0 + 0.3 * i as f64, graine: Graine(100 + i as u64), depart: Depart::Chaud })
                .collect(),
            nor: 2,
            precision: Precision::F64,
            options: OptionsCompil::reference_k1(),
            mutant_metro: false,
        }
    }

    fn dossier(nom: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kvu_reprise_{nom}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn avancer(n: &mut Cpu64, k: usize) {
        use crate::su2::NoyauxSu2;
        for _ in 0..k {
            n.sweep_hb().unwrap();
            n.sweep_or().unwrap();
            n.sweep_or().unwrap();
        }
    }

    #[test]
    fn deux_n_pas_egalent_n_reprise_n() {
        use crate::su2::NoyauxSu2;
        let d = dossier("a");
        let mut a = Cpu64::nouveau(cfg(2)).unwrap();
        a.init().unwrap();
        avancer(&mut a, 6);
        // N = 3 puis point de reprise puis NOUVEL objet
        let mut b = Cpu64::nouveau(cfg(2)).unwrap();
        b.init().unwrap();
        avancer(&mut b, 3);
        ecrire(&d, &b, 3, 3).unwrap();
        let mut c = Cpu64::nouveau(cfg(2)).unwrap();
        let e = lire(&d, &mut c).unwrap();
        assert_eq!((e.pas_faits, e.mesures_faites, e.sweep_id), (3, 3, 3));
        assert_eq!(c.sweep_id(), 3);
        avancer(&mut c, 3);
        for ch in 0..2 {
            assert_eq!(a.telecharger(ch).unwrap().q, c.telecharger(ch).unwrap().q, "chaîne {ch}");
        }
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn corruption_detectee_et_prev_de_secours() {
        use crate::su2::NoyauxSu2;
        let d = dossier("b");
        let mut a = Cpu64::nouveau(cfg(1)).unwrap();
        a.init().unwrap();
        avancer(&mut a, 2);
        ecrire(&d, &a, 2, 0).unwrap();
        // corruption d'un octet : refus net
        let p = d.join(NOM_CHAMPS);
        let mut b = fs::read(&p).unwrap();
        b[100] ^= 1;
        fs::write(&p, &b).unwrap();
        let mut c = Cpu64::nouveau(cfg(1)).unwrap();
        assert!(matches!(lire(&d, &mut c), Err(Erreur::Format(_))));
        // simulation d'un crash entre les étapes 3 et 4 : champs neufs, JSON ancien, ancien champ en .prev
        let bon = {
            b[100] ^= 1;
            b.clone()
        };
        fs::write(d.join(format!("{NOM_CHAMPS}.prev")), &bon).unwrap();
        fs::write(&p, vec![7u8; bon.len()]).unwrap();
        let e = lire(&d, &mut c).unwrap();
        assert_eq!(e.pas_faits, 2);
        assert_eq!(c.telecharger(0).unwrap().q, a.telecharger(0).unwrap().q);
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn configuration_differente_refusee() {
        use crate::su2::NoyauxSu2;
        let d = dossier("c");
        let mut a = Cpu64::nouveau(cfg(1)).unwrap();
        a.init().unwrap();
        ecrire(&d, &a, 0, 0).unwrap();
        let mut autre_cfg = cfg(1);
        autre_cfg.chaines[0].beta = 9.0;
        let mut c = Cpu64::nouveau(autre_cfg).unwrap();
        assert!(matches!(lire(&d, &mut c), Err(Erreur::Config(_))));
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn dossier_sans_reprise_et_taille() {
        assert!(lire_etat_seul(Path::new("/nonexistent_kvu")).is_err());
        assert_eq!(octets_champs(4, 3), 3 * 16 * 256 * 8);
    }
}
