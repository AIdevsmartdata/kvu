//! Mesure d'une configuration pendant une chaîne conditionnelle. Propriétaire : T5.
//!
//! Le trait [`Mesure`] isole la chaîne de la mesure de fibre : la production passe par
//! [`MesureFibreReelle`] (= `fibre::Mesureur`, T6, GPU ou CPU) ; les tests de logique de chaîne (cadence,
//! reprise à l'octet) utilisent [`MesureProvisoire`], un DOUBLE DE TEST honnête mais volontairement réduit
//! (F = 6 plaquettes moyennes de plan, D = F Fᵀ) qui n'a PAS le sens physique de la fibre à 54 composantes.

use super::{GeometrieCond, MoteurCond};
use crate::commun::{Erreur, Resultat};
use crate::fibre::{poids_fourier, MesureFibre, Mesureur};
use crate::gpu::Appareil;
use crate::heritage::su2_hb::{qdag, qmul};
use std::sync::Arc;

/// Mesures d'une chaîne : F seul (entre deux mesures D) ou (F, D, blk).
pub trait Mesure {
    /// Dimension de F (54 pour la fibre de Fourier à pmax² = 1).
    fn nf(&self) -> usize;
    /// F seulement.
    fn mesurer_f(&mut self, m: &MoteurCond) -> Resultat<Vec<f64>>;
    /// (F, D, blk) de `cb.mesure`.
    fn mesurer(&mut self, m: &MoteurCond) -> Resultat<MesureFibre>;
    /// Lecture tordue (`--tordu`) : plaquettes pv (6V) et, si `gradients`, leurs gradients gp (6V × 4 × 3) du
    /// champ courant. Défaut : champ rapatrié et référence CPU.
    fn plaquettes_tordues(&mut self, m: &MoteurCond, gradients: bool) -> Resultat<(Vec<f64>, Vec<f64>)> {
        let (pv, gp) = crate::fibre::plaquettes_gradients_cpu(&m.telecharger()?);
        Ok((pv, if gradients { gp } else { Vec::new() }))
    }
}

/// Mesure de production : `fibre::Mesureur` (T6) lu directement sur le champ résident (sans copie) ou, en CPU,
/// sur le champ rapatrié.
pub struct MesureFibreReelle {
    inner: Mesureur,
    nf: usize,
}

impl MesureFibreReelle {
    pub fn nouvelle(app: Option<&Appareil>, geo: Arc<GeometrieCond>) -> Resultat<Self> {
        let w = poids_fourier(geo.l, 1);
        let nf = w.nf;
        if nf == 0 {
            return Err(Erreur::Config(
                "fibre::poids_fourier a rendu 0 fonctionnelle (module fibre non livré ?)".into(),
            ));
        }
        Ok(MesureFibreReelle { inner: Mesureur::nouveau(app, geo, &w)?, nf })
    }
}

impl Mesure for MesureFibreReelle {
    fn nf(&self) -> usize {
        self.nf
    }

    fn mesurer_f(&mut self, m: &MoteurCond) -> Resultat<Vec<f64>> {
        match m.champ_gpu() {
            Some(c) => self.inner.mesurer_f(Some(c), 0, None),
            None => self.inner.mesurer_f(None, 0, Some(&m.telecharger()?)),
        }
    }

    fn mesurer(&mut self, m: &MoteurCond) -> Resultat<MesureFibre> {
        match m.champ_gpu() {
            Some(c) => self.inner.mesurer(Some(c), 0, None),
            None => self.inner.mesurer(None, 0, Some(&m.telecharger()?)),
        }
    }

    /// Sur le GPU : `k_plaq_grad` du champ résident (sans gradients quand ils ne servent pas : 6V réels rapatriés).
    fn plaquettes_tordues(&mut self, m: &MoteurCond, gradients: bool) -> Resultat<(Vec<f64>, Vec<f64>)> {
        match (m.champ_gpu(), gradients) {
            (Some(c), true) => self.inner.plaquettes(Some(c), 0, None),
            (Some(c), false) => Ok((self.inner.valeurs_plaquettes(Some(c), 0, None)?, Vec::new())),
            (None, _) => {
                let (pv, gp) = self.inner.plaquettes(None, 0, Some(&m.telecharger()?))?;
                Ok((pv, if gradients { gp } else { Vec::new() }))
            }
        }
    }
}

/// DOUBLE DE TEST (pas la fibre) : F_p = moyenne de ½ReTr U_p sur le plan p (6 plans), D = F Fᵀ,
/// blk = sommes de plaquettes par bloc 2⁴ (float32), dans l'ordre des plans (μ<ν) puis des sites.
pub struct MesureProvisoire {
    l: usize,
}

impl MesureProvisoire {
    pub fn nouvelle(l: usize) -> Self {
        MesureProvisoire { l }
    }

    /// Plaquettes ½ReTr U_p, 6V valeurs : plan (μ<ν) majeur puis site (s = ((x0 L + x1) L + x2) L + x3).
    fn plaquettes(&self, m: &MoteurCond) -> Resultat<Vec<f64>> {
        let c = m.telecharger()?;
        let (l, v) = (self.l, self.l.pow(4));
        let st = [l * l * l, l * l, l, 1];
        let haut = |s: usize, d: usize| {
            let x = (s / st[d]) % l;
            if x == l - 1 {
                s - (l - 1) * st[d]
            } else {
                s + st[d]
            }
        };
        let mut pv = Vec::with_capacity(6 * v);
        for mu in 0..4 {
            for nu in mu + 1..4 {
                for s in 0..v {
                    let (smu, snu) = (haut(s, mu), haut(s, nu));
                    let a = c.lien(mu * v + s);
                    let b = c.lien(nu * v + smu);
                    let cc = c.lien(mu * v + snu);
                    let d = c.lien(nu * v + s);
                    pv.push(qmul(qmul(a, b), qmul(qdag(cc), qdag(d)))[0]);
                }
            }
        }
        Ok(pv)
    }

    fn f_de(&self, pv: &[f64]) -> Vec<f64> {
        let v = self.l.pow(4);
        pv.chunks_exact(v).map(|p| p.iter().sum::<f64>() / v as f64).collect()
    }
}

impl Mesure for MesureProvisoire {
    fn nf(&self) -> usize {
        6
    }

    fn mesurer_f(&mut self, m: &MoteurCond) -> Resultat<Vec<f64>> {
        Ok(self.f_de(&self.plaquettes(m)?))
    }

    fn mesurer(&mut self, m: &MoteurCond) -> Resultat<MesureFibre> {
        let pv = self.plaquettes(m)?;
        let f = self.f_de(&pv);
        let mut d = vec![0.0; 36];
        for i in 0..6 {
            for j in 0..6 {
                d[6 * i + j] = f[i] * f[j];
            }
        }
        // sommes de blocs 2^4, comme block_sums : (6, Lc, 2, Lc, 2, Lc, 2, Lc, 2) → (6, Lc⁴)
        let (l, lc) = (self.l, self.l / 2);
        let mut blk = vec![0.0f32; 6 * lc.pow(4)];
        for p in 0..6 {
            let mut somme = vec![0.0f64; lc.pow(4)];
            for s in 0..l.pow(4) {
                let (x0, x1, x2, x3) = (s / (l * l * l), (s / (l * l)) % l, (s / l) % l, s % l);
                let b = ((x0 / 2 * lc + x1 / 2) * lc + x2 / 2) * lc + x3 / 2;
                somme[b] += pv[p * l.pow(4) + s];
            }
            for (b, x) in somme.iter().enumerate() {
                blk[p * lc.pow(4) + b] = *x as f32;
            }
        }
        Ok(MesureFibre { f, d, blk })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commun::ChampHote;
    use crate::cond::geometrie::v_paires;
    use crate::cond::moteur::{ModeCond, SourceAlea};

    fn moteur(champ: &ChampHote, beta: f64, graine: u64) -> MoteurCond {
        let geo = Arc::new(GeometrieCond::nouvelle(champ.l).unwrap());
        let vfix = v_paires(champ, &geo);
        MoteurCond::nouveau(None, geo, champ, vfix, beta, SourceAlea::depuis_graine(ModeCond::Rapide, graine)).unwrap()
    }

    #[test]
    fn champ_froid_plaquettes_toutes_egales_a_un() {
        let m = moteur(&ChampHote::froid(4).unwrap(), 2.3, 0);
        let mut mesure = MesureProvisoire::nouvelle(4);
        let r = mesure.mesurer(&m).unwrap();
        assert_eq!(r.f, vec![1.0; 6]);
        assert!(r.d.iter().all(|&x| x == 1.0));
        // 16 plaquettes par bloc 2⁴, chacune valant 1 : chaque somme de bloc = 16
        assert_eq!(r.blk.len(), 6 * 16);
        assert!(r.blk.iter().all(|&x| x == 16.0));
        assert_eq!(mesure.mesurer_f(&m).unwrap(), r.f);
        assert_eq!(mesure.nf(), 6);
    }

    #[test]
    fn plaquette_moyenne_diminue_apres_thermalisation() {
        let mut m = moteur(&ChampHote::froid(4).unwrap(), 2.3, 3);
        let mut mesure = MesureProvisoire::nouvelle(4);
        for _ in 0..20 {
            m.sweep().unwrap();
        }
        let f = mesure.mesurer_f(&m).unwrap();
        assert!(f.iter().all(|&p| p < 1.0 && p > 0.0), "{f:?}");
    }

    #[test]
    fn la_somme_des_blocs_redonne_la_somme_des_plaquettes() {
        let mut m = moteur(&ChampHote::froid(4).unwrap(), 2.3, 5);
        for _ in 0..5 {
            m.sweep().unwrap();
        }
        let mut mesure = MesureProvisoire::nouvelle(4);
        let r = mesure.mesurer(&m).unwrap();
        for p in 0..6 {
            let somme: f64 = r.blk[p * 16..(p + 1) * 16].iter().map(|&x| x as f64).sum();
            assert!((somme - r.f[p] * 256.0).abs() < 1e-3, "plan {p} : {somme} contre {}", r.f[p] * 256.0);
        }
    }
}
