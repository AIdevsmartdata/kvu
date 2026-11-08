//! Banc de la mesure de fibre (consommé par `kevinotron bench fibre`, T9) : durées de `mesurer` et de
//! `mesurer_f` sur une configuration SU(2) pseudo-aléatoire ou fournie, CPU ou GPU.
//!
//! Référence (README de noyau2, machine principale, `cb.mesure`, numba) : 124 ms à L = 16, 780 ms à L = 24, 2,04 s à L = 32.

use super::{poids_fourier, Mesureur};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::cond::GeometrieCond;
use crate::gpu::Appareil;
use crate::su2::{ChampGpu, TamponChamp};
use std::sync::Arc;
use std::time::Instant;

/// Durées mesurées (millisecondes, médiane et minimum de `repetitions` mesures après un échauffement).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ResultatBanc {
    pub l: usize,
    /// « hip », « cuda » ou « cpu ».
    pub backend: String,
    /// Champ envoyé depuis l'hôte à chaque appel (GPU : téléversement compris).
    pub mesurer_ms: f64,
    pub mesurer_min_ms: f64,
    /// Champ résident sur le GPU (cas de la chaîne) ; None en CPU.
    pub mesurer_residente_ms: Option<f64>,
    pub mesurer_f_ms: f64,
    /// Construction du mesureur (poids, tables, téléversement, compilation), en secondes.
    pub init_s: f64,
    /// Mémoire GPU estimée (octets) ; 0 en CPU.
    pub octets_gpu: u64,
}

/// Configuration SU(2) pseudo-aléatoire déterministe (xorshift64*, somme de 4 uniformes centrées puis
/// normalisation : ni Haar exacte ni utile comme loi, seulement un champ générique pour les bancs).
pub fn champ_pseudo_aleatoire(l: usize, graine: u64) -> ChampHote {
    let mut s = graine.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut u = move || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        ((s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut c = ChampHote::froid(l).expect("L pair requis");
    for lien in c.q.chunks_exact_mut(4) {
        let mut n = 0.0;
        for x in lien.iter_mut() {
            *x = (0..4).map(|_| u() - 0.5).sum::<f64>();
            n += *x * *x;
        }
        let n = n.sqrt();
        for x in lien.iter_mut() {
            *x /= n;
        }
    }
    c
}

fn mediane(v: &[f64]) -> f64 {
    let mut t = v.to_vec();
    t.sort_by(|a, b| a.partial_cmp(b).expect("durées finies"));
    t[t.len() / 2]
}

/// Banc à L donné. `champ` None ⇒ champ pseudo-aléatoire (graine 40 + L). `app` None ⇒ référence CPU.
pub fn banc(app: Option<&Appareil>, l: usize, champ: Option<&ChampHote>, repetitions: usize) -> Resultat<ResultatBanc> {
    if repetitions == 0 {
        return Err(Erreur::Config("banc fibre : au moins une répétition".into()));
    }
    let propre;
    let champ = match champ {
        Some(c) => c,
        None => {
            propre = champ_pseudo_aleatoire(l, 40 + l as u64);
            &propre
        }
    };
    let t0 = Instant::now();
    let w = poids_fourier(l, 1);
    let mut mes = Mesureur::nouveau(app, Arc::new(GeometrieCond::nouvelle(l)?), &w)?;
    mes.mesurer(None, 0, Some(champ))?; // échauffement : compilation, cache, première allocation
    let init_s = t0.elapsed().as_secs_f64();
    let mut mesurer = Vec::with_capacity(repetitions);
    let mut mesurer_f = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let t = Instant::now();
        mes.mesurer(None, 0, Some(champ))?;
        mesurer.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        mes.mesurer_f(None, 0, Some(champ))?;
        mesurer_f.push(t.elapsed().as_secs_f64() * 1e3);
    }
    // Champ résident : un lot d'une chaîne F64 déposé une fois sur le GPU.
    let residente_ms = match app {
        Some(a) => {
            let mut t = a.alloc::<f64>(champ.q.len())?;
            a.ecrire(&mut t, 0, &champ.q)?;
            let resident = ChampGpu {
                l,
                nchaines: 1,
                tampon: TamponChamp::F64(t),
            };
            mes.mesurer(Some(&resident), 0, None)?;
            let mut v = Vec::with_capacity(repetitions);
            for _ in 0..repetitions {
                let t = Instant::now();
                mes.mesurer(Some(&resident), 0, None)?;
                v.push(t.elapsed().as_secs_f64() * 1e3);
            }
            Some(mediane(&v))
        }
        None => None,
    };
    Ok(ResultatBanc {
        l,
        backend: app.map_or("cpu", |a| a.backend().nom()).to_string(),
        mesurer_ms: mediane(&mesurer),
        mesurer_min_ms: mesurer.iter().cloned().fold(f64::MAX, f64::min),
        mesurer_residente_ms: residente_ms,
        mesurer_f_ms: mediane(&mesurer_f),
        init_s,
        octets_gpu: if app.is_some() { mes.octets_gpu_estimes() } else { 0 },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn champ_pseudo_aleatoire_deterministe_et_unitaire() {
        let (a, b) = (champ_pseudo_aleatoire(4, 7), champ_pseudo_aleatoire(4, 7));
        assert_eq!(a, b);
        assert_ne!(a, champ_pseudo_aleatoire(4, 8));
        assert!(a.defaut_unitarite() < 1e-14);
        assert!(a.q.iter().any(|&x| x != 1.0 && x != 0.0));
    }

    #[test]
    fn banc_cpu_petit() {
        let r = banc(None, 4, None, 2).unwrap();
        assert_eq!((r.l, r.backend.as_str(), r.octets_gpu), (4, "cpu", 0));
        assert!(r.mesurer_residente_ms.is_none());
        assert!(r.mesurer_ms > 0.0 && r.mesurer_min_ms <= r.mesurer_ms && r.mesurer_f_ms > 0.0);
        assert!(banc(None, 4, None, 0).is_err());
    }
}
