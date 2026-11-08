//! Champ résident sur le GPU. Propriétaire : T3.
//!
//! Disposition (F64 comme F32) : chaîne c, lien li = mu·V + s, composante k (x, y, z, w) à l'indice
//! (c·4V + li)·4 + k, soit 16·V valeurs par chaîne, exactement la disposition de `ChampHote.q`.

use crate::commun::{ChampHote, Erreur, Precision, Resultat};
use crate::gpu::{Appareil, Tampon};

/// Stockage device : chaîne c, lien li, composante k à l'indice (c·4V + li)·4 + k.
pub enum TamponChamp {
    F64(Tampon<f64>),
    F32(Tampon<f32>),
}

/// Champ d'un lot résident sur le GPU (lu par cond, fibre, obs).
pub struct ChampGpu {
    pub l: usize,
    pub nchaines: usize,
    pub tampon: TamponChamp,
}

/// Nombre de valeurs (quaternions × 4) d'UNE chaîne : 16·L⁴.
pub fn valeurs_par_chaine(l: usize) -> usize {
    16 * l * l * l * l
}

/// F32 → F64 : exact (toute valeur f32 est un f64).
pub fn f32_vers_f64(src: &[f32]) -> Vec<f64> {
    src.iter().map(|&x| x as f64).collect()
}

/// F64 → F32 pour le téléversement d'un champ hôte.
///
/// - Si TOUTES les valeurs sont déjà des f32 exacts (cas d'un champ issu d'un `telecharger` F32, d'un point de
///   reprise ou d'une copie), la conversion est EXACTE et ne renormalise rien : f32 → f64 → f32 est l'identité,
///   ce qui rend la reprise F32 exacte à l'octet.
/// - Sinon (champ F64 quelconque) : chaque quaternion est renormalisé EN F64, puis arrondi au plus proche en f32.
pub fn f64_vers_f32(src: &[f64]) -> Vec<f32> {
    let exact = src.iter().all(|&x| (x as f32) as f64 == x);
    if exact {
        return src.iter().map(|&x| x as f32).collect();
    }
    let mut out = Vec::with_capacity(src.len());
    for q in src.chunks_exact(4) {
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        let s = if n > 0.0 { 1.0 / n } else { 1.0 };
        out.extend(q.iter().map(|&x| (x * s) as f32));
    }
    out
}

impl ChampGpu {
    pub fn v(&self) -> usize {
        self.l * self.l * self.l * self.l
    }
    pub fn precision(&self) -> Precision {
        match self.tampon {
            TamponChamp::F64(_) => Precision::F64,
            TamponChamp::F32(_) => Precision::F32,
        }
    }
    /// Alloue un champ (zéros) de `nchaines` chaînes.
    pub fn allouer(app: &Appareil, l: usize, nchaines: usize, p: Precision) -> Resultat<ChampGpu> {
        if l < 2 || !l.is_multiple_of(2) || nchaines == 0 {
            return Err(Erreur::Config(format!("champ GPU : L = {l} (pair ≥ 2) et {nchaines} chaîne(s) (≥ 1) requis")));
        }
        let n = nchaines * valeurs_par_chaine(l);
        let tampon = match p {
            Precision::F64 => TamponChamp::F64(app.alloc::<f64>(n)?),
            Precision::F32 => TamponChamp::F32(app.alloc::<f32>(n)?),
        };
        Ok(ChampGpu { l, nchaines, tampon })
    }

    fn verifier_chaine(&self, c: usize) -> Resultat<()> {
        if c >= self.nchaines {
            return Err(Erreur::Config(format!("chaîne {c} hors du lot ({} chaînes)", self.nchaines)));
        }
        Ok(())
    }

    /// Copie la chaîne `c` vers l'hôte (F32 promu en f64, sans renormaliser).
    pub fn telecharger(&self, app: &Appareil, c: usize) -> Resultat<ChampHote> {
        self.verifier_chaine(c)?;
        let n = valeurs_par_chaine(self.l);
        let q = match &self.tampon {
            TamponChamp::F64(t) => {
                let mut h = vec![0f64; n];
                app.lire(t, c * n, &mut h)?;
                h
            }
            TamponChamp::F32(t) => {
                let mut h = vec![0f32; n];
                app.lire(t, c * n, &mut h)?;
                f32_vers_f64(&h)
            }
        };
        Ok(ChampHote { l: self.l, q })
    }

    /// Copie un champ hôte dans la chaîne `c` (F32 : voir `f64_vers_f32` : exact pour une donnée déjà
    /// représentable en f32, sinon renormalisation en f64 puis arrondi au plus proche).
    pub fn televerser(&mut self, app: &Appareil, c: usize, h: &ChampHote) -> Resultat<()> {
        self.verifier_chaine(c)?;
        let n = valeurs_par_chaine(self.l);
        if h.l != self.l || h.q.len() != n {
            return Err(Erreur::Config(format!(
                "champ hôte (L = {}, {} valeurs) incompatible avec le lot (L = {}, {n} valeurs)",
                h.l,
                h.q.len(),
                self.l
            )));
        }
        match &mut self.tampon {
            TamponChamp::F64(t) => app.ecrire(t, c * n, &h.q),
            TamponChamp::F32(t) => app.ecrire(t, c * n, &f64_vers_f32(&h.q)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valeurs_par_chaine_est_16_l4() {
        assert_eq!(valeurs_par_chaine(4), 16 * 256);
        assert_eq!(valeurs_par_chaine(6), 16 * 1296);
    }

    #[test]
    fn f32_vers_f64_est_exact_et_aller_retour_identite() {
        let src: Vec<f32> = vec![0.5, -0.25, 0.1, 0.70710677, 1.0, 0.0, 0.0, 0.0];
        let d = f32_vers_f64(&src);
        let r = f64_vers_f32(&d);
        assert_eq!(src.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), r.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }

    #[test]
    fn f64_quelconque_est_renormalise_puis_arrondi() {
        // quaternion non unitaire et non représentable en f32
        let q = [0.3_f64, 0.4, 0.5, 0.6];
        let r = f64_vers_f32(&q);
        let n: f64 = r.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-7, "norme {n}");
        // arrondi au plus proche de la valeur renormalisée en f64
        let nn = (0.3_f64 * 0.3 + 0.4 * 0.4 + 0.5 * 0.5 + 0.6 * 0.6).sqrt();
        assert_eq!(r[0], (0.3 / nn) as f32);
    }

    #[test]
    fn la_conversion_exacte_ne_renormalise_pas() {
        // un lien f32 de norme 1 + 5e-8 (issu d'un noyau F32) doit revenir BIT À BIT
        let x: f32 = 1.0000001;
        let src = [x as f64, 0.0, 0.0, 0.0];
        assert_eq!(f64_vers_f32(&src)[0].to_bits(), x.to_bits());
    }

    #[test]
    fn quaternion_nul_ne_produit_pas_de_nan() {
        let r = f64_vers_f32(&[0.0; 4]);
        assert!(r.iter().all(|x| x.is_finite()));
    }
}
