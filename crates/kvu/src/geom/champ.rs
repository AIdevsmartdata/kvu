//! Champ SU(2) hôte associé à une spécification géométrique explicite.

use super::{Chemin, GeoSpec};
use crate::commun::{ChampHote, Erreur, Resultat};

#[derive(Clone, Debug, PartialEq)]
pub struct ChampGeo {
    pub geo: GeoSpec,
    /// 16·V réels : li = μ·V + s, quaternion (a0, a1, a2, a3).
    pub q: Vec<f64>,
}

impl ChampGeo {
    pub fn froid(geo: GeoSpec) -> Resultat<ChampGeo> {
        geo.valider()?;
        let taille = geo
            .v()
            .checked_mul(16)
            .ok_or_else(|| Erreur::Config("champ géométrique : 16·V hors limites".into()))?;
        let mut q = vec![0.0; taille];
        for lien in q.chunks_exact_mut(4) {
            lien[0] = 1.0;
        }
        Ok(ChampGeo { geo, q })
    }

    pub fn depuis_hote(h: ChampHote, dir_temps: u8) -> Resultat<ChampGeo> {
        let champ = ChampGeo {
            geo: GeoSpec::cubique(h.l, dir_temps),
            q: h.q,
        };
        champ.valider()?;
        Ok(champ)
    }

    pub fn v(&self) -> usize {
        self.geo.v()
    }

    pub fn valider(&self) -> Resultat<()> {
        self.geo.valider()?;
        let taille = self
            .v()
            .checked_mul(16)
            .ok_or_else(|| Erreur::Config("champ géométrique : 16·V hors limites".into()))?;
        if self.q.len() != taille {
            return Err(Erreur::Config(format!(
                "champ géométrique incohérent : {taille} réels attendus, reçu {}",
                self.q.len()
            )));
        }
        Ok(())
    }
}

impl TryFrom<ChampGeo> for ChampHote {
    type Error = Erreur;

    fn try_from(champ: ChampGeo) -> Resultat<ChampHote> {
        champ.valider()?;
        if champ.geo.chemin() != Chemin::C0 {
            return Err(Erreur::Config(
                "configuration non C0 (chemin C1/C2) : utiliser ChampGeo".into(),
            ));
        }
        Ok(ChampHote {
            l: champ.geo.cotes[0],
            q: champ.q,
        })
    }
}
