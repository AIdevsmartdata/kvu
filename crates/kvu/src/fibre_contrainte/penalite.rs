//! Route A : Metropolis local pour Wilson et la pénalité géodésique, κ = β ε².

use super::action::{action_wilson, delta_action_lien_q, plaquette_moyenne};
use super::alea::{Flux, TAG_ACCEPTATION, TAG_PROPOSITION};
use super::jacobien::{Reseau, Q};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul, Geo};
use crate::rg::bloc::{exp_q, log_q, ubar_lien};
use std::sync::Arc;

/// Paramètres de la loi et de la proposition locale symétrique.
#[derive(Clone, Copy, Debug)]
pub struct ParamsPenalite {
    pub beta: f64,
    /// κ = β ε² ; `None` retire entièrement la pénalité.
    pub kappa: Option<f64>,
    /// Écart-type par composante de la proposition dans l'algèbre.
    pub delta: f64,
}

/// Une chaîne séquentielle ; le parallélisme éventuel appartient à l'appelant.
pub struct ChainePenalite {
    champ: ChampHote,
    u: Vec<Q>,
    ubar: Vec<Q>,
    v: Vec<Q>,
    pen_total: f64,
    facteur_pen: f64,
    geo: Geo,
    reseau: Option<Arc<Reseau>>,
    voisins: Vec<Vec<usize>>,
    params: ParamsPenalite,
    flux: Flux,
    chaine: u32,
}

/// Renormalisation d'un quaternion non nul : q / |q|.
pub fn renormaliser(q: Q) -> Q {
    let norme = q[0].hypot(q[1]).hypot(q[2]).hypot(q[3]);
    q.map(|x| x / norme)
}

fn residu_c(cible: Q, moyenne: Q) -> f64 {
    let w = log_q(qmul(qdag(cible), moyenne)).0;
    w[0].hypot(w[1]).hypot(w[2])
}

fn unitaire(q: Q) -> bool {
    q.iter().all(|x| x.is_finite())
        && (q[0].hypot(q[1]).hypot(q[2]).hypot(q[3]) - 1.0).abs() <= 1e-10
}

impl ChainePenalite {
    /// Construit les moyennes et inverse les dépendances du réseau une seule fois.
    /// `v` est dans l'ordre c ; `v` et `reseau` sont ignorés si κ est absent.
    pub fn nouvelle(
        reseau: Option<Arc<Reseau>>,
        v: Vec<Q>,
        champ: ChampHote,
        params: ParamsPenalite,
        flux: Flux,
        chaine: u32,
    ) -> Resultat<ChainePenalite> {
        let taille = champ.l.checked_pow(4).and_then(|v| v.checked_mul(16));
        if champ.l < 2 || !champ.l.is_multiple_of(2) || taille != Some(champ.q.len()) {
            return Err(Erreur::Config(
                "champ : L pair ≥ 2 et 16 L⁴ réels requis".into(),
            ));
        }
        // Chaque composante doit avoir un compteur Philox distinct, sans troncature.
        if (champ.q.len() / 4)
            .checked_mul(3)
            .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(Erreur::Config(
                "trop de liens pour les compteurs de proposition".into(),
            ));
        }
        if !params.beta.is_finite()
            || params.beta < 0.0
            || !params.delta.is_finite()
            || params.delta < 0.0
        {
            return Err(Erreur::Config(
                "β et delta doivent être finis et positifs ou nuls".into(),
            ));
        }
        let u: Vec<Q> = champ
            .q
            .chunks_exact(4)
            .map(|q| [q[0], q[1], q[2], q[3]])
            .collect();
        if !u.iter().copied().all(unitaire) {
            return Err(Erreur::Config(
                "les liens du champ doivent appartenir à SU(2)".into(),
            ));
        }
        let mut resultat = Self {
            geo: Geo::new(champ.l),
            champ,
            u,
            ubar: Vec::new(),
            v: Vec::new(),
            pen_total: 0.0,
            facteur_pen: 0.0,
            reseau: None,
            voisins: Vec::new(),
            params,
            flux,
            chaine,
        };
        if let Some(kappa) = params.kappa {
            let facteur = 0.5 * (params.beta / kappa);
            if !kappa.is_finite()
                || kappa <= 0.0
                || params.beta <= 0.0
                || !facteur.is_finite()
                || facteur <= 0.0
            {
                return Err(Erreur::Config(
                    "κ et β doivent donner une raideur finie strictement positive".into(),
                ));
            }
            let reseau = reseau
                .ok_or_else(|| Erreur::Config("réseau requis avec une pénalité finie".into()))?;
            if reseau.l != resultat.champ.l
                || reseau.n_liens != resultat.u.len()
                || v.len() != reseau.n_c
                || !v.iter().copied().all(unitaire)
            {
                return Err(Erreur::Config(
                    "réseau, champ ou cible SU(2) incompatibles".into(),
                ));
            }
            let mut voisins = vec![Vec::new(); resultat.u.len()];
            for c in 0..reseau.n_c {
                for li in reseau.liens_de(c) {
                    let liste = voisins
                        .get_mut(li as usize)
                        .ok_or_else(|| Erreur::Config("dépendance hors du champ".into()))?;
                    // c croît ; une éventuelle répétition dans les slots reste unique.
                    if liste.last() != Some(&c) {
                        liste.push(c);
                    }
                }
                let (z, mu) = reseau.coords_c(c);
                resultat
                    .ubar
                    .push(ubar_lien(&resultat.champ, &reseau.table, z, mu).0);
            }
            if voisins.iter().any(|liste| liste.len() > 8) {
                return Err(Erreur::Config(
                    "plus de huit voisins grossiers pour un lien fin".into(),
                ));
            }
            resultat.voisins = voisins;
            resultat.reseau = Some(reseau);
            resultat.v = v;
            resultat.facteur_pen = facteur;
            resultat.pen_total = resultat.penalite_recalculee();
        }
        Ok(resultat)
    }

    // Point d'instrumentation des tests : même proposition que le balayage, avec
    // sa variation locale d'énergie avant la décision de Metropolis.
    fn proposer(&mut self, numero: u32, li: usize) -> (bool, f64) {
        let epsilon = std::array::from_fn(|a| {
            self.params.delta
                * self
                    .flux
                    .gauss(self.chaine, numero, (3 * li + a) as u32, TAG_PROPOSITION)
        });
        let ancien = self.u[li];
        let nouveau = renormaliser(qmul(exp_q(epsilon), ancien));
        let delta_action = delta_action_lien_q(&self.u, &self.geo, li, nouveau, self.params.beta);
        self.champ.q[4 * li..4 * li + 4].copy_from_slice(&nouveau);
        let mut delta_pen = 0.0;
        // Aucun tampon alloué ici. Les allocations internes de ubar_lien sont
        // imposées par son API ; son noyau reste identique à celui de bloquer.
        let mut moyennes = [[0.0; 4]; 8];
        if let Some(reseau) = &self.reseau {
            for (i, &c) in self.voisins[li].iter().enumerate() {
                let (z, mu) = reseau.coords_c(c);
                let moyenne = ubar_lien(&self.champ, &reseau.table, z, mu).0;
                moyennes[i] = moyenne;
                delta_pen += self.facteur_pen * residu_c(self.v[c], moyenne).powi(2)
                    - self.facteur_pen * residu_c(self.v[c], self.ubar[c]).powi(2);
            }
        }
        let variation = delta_action + delta_pen;
        let accepte = self
            .flux
            .uniforme(self.chaine, numero, li as u32, TAG_ACCEPTATION)
            < (-variation).exp();
        if accepte {
            self.u[li] = nouveau;
            if self.reseau.is_some() {
                for (i, &c) in self.voisins[li].iter().enumerate() {
                    self.ubar[c] = moyennes[i];
                }
            }
            self.pen_total += delta_pen;
        } else {
            self.champ.q[4 * li..4 * li + 4].copy_from_slice(&ancien);
        }
        (accepte, variation)
    }

    /// Une seule proposition locale (même noyau que le balayage), pour les gates d'instrument ; rend l'acceptation.
    #[cfg(test)]
    pub(crate) fn proposer_un(&mut self, numero: u32, li: usize) -> bool {
        self.proposer(numero, li).0
    }

    /// Propose tous les liens, dans l'ordre li = μV + site, puis resynchronise la somme.
    pub fn balayage(&mut self, numero: u32) -> (u64, u64) {
        let mut acceptees = 0;
        for li in 0..self.u.len() {
            acceptees += u64::from(self.proposer(numero, li).0);
        }
        self.pen_total = self.penalite_recalculee();
        (self.u.len() as u64, acceptees)
    }

    pub fn champ(&self) -> &ChampHote {
        &self.champ
    }
    pub fn ubar(&self) -> &[Q] {
        &self.ubar
    }
    pub fn penalite_totale(&self) -> f64 {
        self.pen_total
    }

    /// Somme complète des pénalités courantes, dans l'ordre c croissant.
    pub fn penalite_recalculee(&self) -> f64 {
        self.ubar
            .iter()
            .zip(&self.v)
            .map(|(&moyenne, &cible)| self.facteur_pen * residu_c(cible, moyenne).powi(2))
            .sum()
    }

    pub fn residu_max(&self) -> f64 {
        self.ubar
            .iter()
            .zip(&self.v)
            .map(|(&moyenne, &cible)| residu_c(cible, moyenne))
            .fold(0.0, f64::max)
    }

    pub fn action(&self) -> f64 {
        action_wilson(&self.champ, self.params.beta)
    }
    pub fn plaquette(&self) -> f64 {
        plaquette_moyenne(&self.champ)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fibre_contrainte::jacobien::tests::champ_amp;
    use crate::heritage::su2_hb::CpuSu2;
    use crate::rg::bloc::{bloquer, OptionsBloc};
    use crate::rg::geometrie::Geometrie;

    fn options() -> OptionsBloc {
        OptionsBloc {
            geometrie: Geometrie::TroisCentre,
            fils: 1,
            angle_so3: false,
        }
    }

    fn penalisee(chaine: u32) -> ChainePenalite {
        depuis_champ(champ_amp(6, 0.6, 713), chaine)
    }

    fn depuis_champ(champ: ChampHote, chaine: u32) -> ChainePenalite {
        let cible = bloquer(&champ, &options())
            .unwrap()
            .grossier
            .q
            .chunks_exact(4)
            .map(|q| renormaliser(qmul(exp_q([0.02, -0.01, 0.03]), [q[0], q[1], q[2], q[3]])))
            .collect();
        ChainePenalite::nouvelle(
            Some(Arc::new(Reseau::nouveau(6).unwrap())),
            cible,
            champ,
            ParamsPenalite {
                beta: 2.8,
                kappa: Some(2e-3),
                delta: 0.05,
            },
            Flux::nouveau(12345),
            chaine,
        )
        .unwrap()
    }

    fn verifier_miroir(chaine: &ChainePenalite) {
        for (li, &q) in chaine.u.iter().enumerate() {
            assert_eq!(
                q.map(f64::to_bits),
                chaine.champ().lien(li).map(f64::to_bits)
            );
        }
        assert!(chaine.champ().defaut_unitarite() < 1e-12);
    }

    #[test]
    fn fib_ubar_local_mille_propositions() {
        // champ_amp emploie des angles de Haar réduits ; l'instrument demandé
        // utilise ici trois vraies gaussiennes indépendantes par lien.
        let flux = Flux::nouveau(713);
        let champ = ChampHote {
            l: 6,
            q: (0..4 * 6_usize.pow(4))
                .flat_map(|li| {
                    exp_q(std::array::from_fn(|a| {
                        0.6 * flux.gauss(0, 0, (3 * li + a) as u32, TAG_PROPOSITION)
                    }))
                })
                .collect(),
        };
        let mut chaine = depuis_champ(champ, 0);
        let mut acceptees = 0;
        for i in 0..1000 {
            let li = (i * 7919) % chaine.u.len();
            acceptees += usize::from(chaine.proposer(i as u32, li).0);
            verifier_miroir(&chaine);
        }
        assert!(acceptees > 0 && acceptees < 1000, "acceptées : {acceptees}");
        let reference = bloquer(chaine.champ(), &options()).unwrap().grossier;
        for (c, &q) in chaine.ubar().iter().enumerate() {
            assert_eq!(
                q.map(f64::to_bits),
                reference.lien(c).map(f64::to_bits),
                "c={c}"
            );
        }
        let recalculee = chaine.penalite_recalculee();
        assert!((chaine.penalite_totale() - recalculee).abs() <= 1e-12 * recalculee.abs());
        let residu = (0..chaine.v.len())
            .map(|c| residu_c(chaine.v[c], reference.lien(c)))
            .fold(0.0, f64::max);
        assert_eq!(chaine.residu_max().to_bits(), residu.to_bits());
        eprintln!("FIB-UBAR-LOCAL : {acceptees}/1000 acceptées, résidu maximal {residu:e}");
    }

    #[test]
    fn accord_variation_locale_et_difference_directe() {
        let mut chaine = penalisee(0);
        let mut acceptees = 0;
        for i in 0..200 {
            let avant = chaine.action() + chaine.penalite_totale();
            let (accepte, variation) = chaine.proposer(i as u32, (i * 7919) % chaine.u.len());
            let apres = chaine.action() + chaine.penalite_recalculee();
            let attendue = if accepte { variation } else { 0.0 };
            assert!(
                (apres - avant - attendue).abs() <= 1e-9,
                "proposition {i} : différence={}, variation={attendue}",
                apres - avant
            );
            acceptees += usize::from(accepte);
        }
        assert!(acceptees > 0 && acceptees < 200);
        verifier_miroir(&chaine);
    }

    #[test]
    fn determinisme_et_numeros_de_chaine() {
        // La voie pénalisée exerce aussi le cache et la resynchronisation.
        let mut a = penalisee(0);
        let mut b = penalisee(0);
        let mut c = penalisee(1);
        for numero in 0..3 {
            let compte = a.balayage(numero);
            assert_eq!(compte.0, 4 * 6_u64.pow(4));
            assert_eq!(compte, b.balayage(numero));
            c.balayage(numero);
            assert_eq!(
                a.penalite_totale().to_bits(),
                a.penalite_recalculee().to_bits()
            );
        }
        assert_eq!(
            a.champ.q.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.champ.q.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert!(a
            .champ
            .q
            .iter()
            .zip(&c.champ.q)
            .any(|(x, y)| x.to_bits() != y.to_bits()));
        verifier_miroir(&a);
        verifier_miroir(&b);
        verifier_miroir(&c);
    }

    #[test]
    fn loi_libre_comparee_au_bain_thermique() {
        let beta = 2.3;
        let mut chaine = ChainePenalite::nouvelle(
            None,
            Vec::new(),
            ChampHote::froid(4).unwrap(),
            ParamsPenalite {
                beta,
                kappa: None,
                delta: 0.6,
            },
            Flux::nouveau(98765),
            0,
        )
        .unwrap();
        // Un pool local borne aussi les opérations Rayon du générateur historique.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let mut reference = CpuSu2::new(4, 192837).unwrap();
        let mut somme = 0.0;
        let mut somme_reference = 0.0;
        pool.install(|| {
            reference.init(true);
            for numero in 0..20_000 {
                chaine.balayage(numero);
                reference.step(beta, 0);
                if numero >= 10_000 {
                    somme += chaine.plaquette();
                    somme_reference += reference.measure()[0];
                }
            }
        });
        let moyenne = somme / 10_000.0;
        let moyenne_reference = somme_reference / 10_000.0;
        let relatif = (moyenne / moyenne_reference - 1.0).abs();
        eprintln!("loi libre : Metropolis={moyenne}, bain thermique={moyenne_reference}, écart relatif={relatif:e}");
        assert!(relatif <= 0.05);
        assert_eq!(reference.n_cap, 0);
        assert!(chaine.ubar().is_empty());
        verifier_miroir(&chaine);
    }

    #[test]
    fn absence_de_penalite_et_reseau_requis() {
        for l in [2, 4] {
            let mut chaine = ChainePenalite::nouvelle(
                None,
                vec![[f64::NAN; 4]],
                ChampHote::froid(l).unwrap(),
                ParamsPenalite {
                    beta: 2.3,
                    kappa: None,
                    delta: 0.6,
                },
                Flux::nouveau(7),
                0,
            )
            .unwrap();
            chaine.balayage(0);
            assert!(chaine.ubar().is_empty());
            assert!(chaine.reseau.is_none() && chaine.voisins.is_empty());
            assert_eq!(chaine.penalite_totale(), 0.0);
            assert_eq!(chaine.penalite_recalculee(), 0.0);
            assert_eq!(chaine.residu_max(), 0.0);
        }
        let resultat = ChainePenalite::nouvelle(
            None,
            Vec::new(),
            ChampHote::froid(6).unwrap(),
            ParamsPenalite {
                beta: 2.8,
                kappa: Some(2e-3),
                delta: 0.05,
            },
            Flux::nouveau(7),
            0,
        );
        assert!(matches!(resultat, Err(Erreur::Config(_))));
    }

    #[test]
    fn configurations_invalides_et_renormalisation() {
        for (beta, kappa, delta) in [
            (f64::NAN, None, 0.1),
            (-1.0, None, 0.1),
            (2.8, Some(0.0), 0.1),
            (2.8, Some(-1.0), 0.1),
            (2.8, Some(f64::INFINITY), 0.1),
            (2.8, Some(f64::NAN), 0.1),
            (0.0, Some(0.1), 0.1),
            (2.8, None, f64::NAN),
            (2.8, None, -0.1),
        ] {
            assert!(matches!(
                ChainePenalite::nouvelle(
                    None,
                    Vec::new(),
                    ChampHote::froid(4).unwrap(),
                    ParamsPenalite { beta, kappa, delta },
                    Flux::nouveau(0),
                    0
                ),
                Err(Erreur::Config(_))
            ));
        }
        let q = renormaliser([2.0, -3.0, 4.0, -5.0]);
        assert!(unitaire(q));
        assert!(q[0] > 0.0 && q[1] < 0.0 && q[2] > 0.0 && q[3] < 0.0);
    }
}
