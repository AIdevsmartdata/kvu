//! Topologie GPU F64 : trèfles améliorés, flot RK3 et cooling en place.
//! Source et options identiques au module historique des observables (cache T2).

use super::{Charges, FlotT0, ParamsTopo, RecordTopo, SerieCooling, IDX_T3, MULT_T0, TEMPS_A2};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::gpu::{
    assembler, Appareil, Arg, Fonction, Lancement, Module, OptionsCompil, ProvenanceCompil, Tampon,
};
use crate::heritage::su2_hb::Q;
use crate::noyaux::{OBS_FLOT, OBS_WILSON, SU2_COMMUN};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

const BLOC: u32 = 256;

/// Champ résident à observer ; `charger` initialise les trois copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cible {
    Flot,
    Cooling,
    Source,
}

/// Module compilé gardé en vie et fonctions de topologie et de flot.
pub struct ModuleTopo {
    app: Appareil,
    module: Module,
    identite: Rc<()>,
    f_dens: Fonction,
    f_plaq: Fonction,
    f_cool: Fonction,
    f_cool_id: Fonction,
    f_final: Fonction,
    f_copie: Fonction,
    f_z: Fonction,
    f_maj: Fonction,
}

/// Tampons alloués une fois par module et L, réutilisés entre configurations.
/// Les opérations sont ordonnées sur le flux de l'appareil ; pas d'usage concurrent.
pub struct EspaceTopo {
    l: usize,
    v: usize,
    identite: Rc<()>,
    charge: Cell<bool>,
    source: RefCell<Tampon<f64>>,
    flot: Tampon<f64>,
    cooling: Tampon<f64>,
    z0: Tampon<f64>,
    z1: Tampon<f64>,
    z2: Tampon<f64>,
    part: Tampon<f64>,
    sortie: Tampon<f64>,
}

impl ModuleTopo {
    pub fn nouveau(app: &Appareil) -> Resultat<ModuleTopo> {
        let src = assembler(
            app.backend(),
            "obs",
            &[&SU2_COMMUN, &OBS_WILSON, &OBS_FLOT],
            &[],
        );
        let module = app.compiler(&src, &OptionsCompil::reference_k1())?;
        let f = |nom: &str| app.fonction(&module, nom);
        Ok(Self {
            app: app.clone(),
            identite: Rc::new(()),
            f_dens: f("k_topo_densites")?,
            f_plaq: f("k_topo_plaq")?,
            f_cool: f("k_topo_cool")?,
            f_cool_id: f("k_topo_cool_id")?,
            f_final: f("k_obs_final")?,
            f_copie: f("k_obs_copie")?,
            f_z: f("k_obs_z")?,
            f_maj: f("k_obs_maj")?,
            module,
        })
    }

    pub fn provenance(&self) -> ProvenanceCompil {
        self.module.provenance().clone()
    }

    pub fn espace(&self, l: usize) -> Resultat<EspaceTopo> {
        let v = l
            .checked_pow(4)
            .filter(|&v| {
                l >= 2
                    && l.is_multiple_of(2)
                    && l <= i32::MAX as usize
                    && v <= i64::MAX as usize / 16
                    && v.div_ceil(BLOC as usize) <= u32::MAX as usize / 4
            })
            .ok_or_else(|| {
                Erreur::Config("topologie GPU : L pair ≥ 2 et volume représentable requis".into())
            })?;
        Ok(EspaceTopo {
            l,
            v,
            identite: self.identite.clone(),
            charge: Cell::new(false),
            source: RefCell::new(self.app.alloc::<f64>(16 * v)?),
            flot: self.app.alloc::<f64>(16 * v)?,
            cooling: self.app.alloc::<f64>(16 * v)?,
            z0: self.app.alloc::<f64>(12 * v)?,
            z1: self.app.alloc::<f64>(12 * v)?,
            z2: self.app.alloc::<f64>(12 * v)?,
            part: self.app.alloc::<f64>(3 * v.div_ceil(BLOC as usize))?,
            sortie: self.app.alloc::<f64>(3)?,
        })
    }

    fn verifier(&self, e: &EspaceTopo, charge: bool) -> Resultat<()> {
        if !Rc::ptr_eq(&self.identite, &e.identite) {
            return Err(Erreur::Config(
                "topologie GPU : espace d'un autre module".into(),
            ));
        }
        if charge && !e.charge.get() {
            return Err(Erreur::Config(
                "topologie GPU : charger un champ avant utilisation".into(),
            ));
        }
        Ok(())
    }

    fn copie(&self, e: &EspaceTopo, destination: &Tampon<f64>) -> Resultat<()> {
        self.app.lancer(
            &self.f_copie,
            &Lancement::lineaire(4 * e.v, BLOC),
            &[
                e.source.borrow().arg(),
                destination.arg(),
                Arg::I64((4 * e.v) as i64),
            ],
        )
    }

    /// Téléverse le champ brut et réinitialise flot et cooling sans réallocation.
    pub fn charger(&self, e: &EspaceTopo, champ: &ChampHote) -> Resultat<()> {
        self.verifier(e, false)?;
        if champ.l != e.l || champ.q.len() != 16 * e.v {
            return Err(Erreur::Config(
                "topologie GPU : champ incompatible avec l'espace".into(),
            ));
        }
        e.charge.set(false);
        self.app.ecrire(&mut e.source.borrow_mut(), 0, &champ.q)?;
        self.copie(e, &e.flot)?;
        self.copie(e, &e.cooling)?;
        e.charge.set(true);
        Ok(())
    }

    fn tampon(&self, e: &EspaceTopo, cible: Cible) -> Arg {
        match cible {
            Cible::Source => e.source.borrow().arg(),
            Cible::Flot => e.flot.arg(),
            Cible::Cooling => e.cooling.arg(),
        }
    }

    fn finale(&self, e: &EspaceTopo, stride: usize) -> Resultat<()> {
        self.app.lancer(
            &self.f_final,
            &Lancement {
                grille: [stride as u32, 1, 1],
                bloc: [BLOC, 1, 1],
                shmem: 0,
            },
            &[
                e.part.arg(),
                Arg::I64(e.v.div_ceil(BLOC as usize) as i64),
                Arg::I64(stride as i64),
                Arg::F64(1.0),
                e.sortie.arg(),
            ],
        )
    }

    /// Trois sommes brutes ; une synchronisation et un transfert de trois doubles.
    fn densites(&self, e: &EspaceTopo, cible: Cible) -> Resultat<[f64; 3]> {
        self.verifier(e, true)?;
        self.app.lancer(
            &self.f_dens,
            &Lancement::lineaire(e.v, BLOC),
            &[self.tampon(e, cible), Arg::I32(e.l as i32), e.part.arg()],
        )?;
        self.finale(e, 3)?;
        self.app.synchroniser()?;
        let mut sommes = [0.0; 3];
        self.app.lire(&e.sortie, 0, &mut sommes)?;
        Ok(sommes)
    }

    pub fn charges(&self, e: &EspaceTopo, cible: Cible) -> Resultat<Charges> {
        let s = self.densites(e, cible)?;
        Ok(Charges {
            plaq: s[0] / (6.0 * e.v as f64),
            q_tre: s[1],
            q_imp: s[2],
        })
    }

    /// Avance RK3 : exactement les six lancements de ModuleObs::flot_pas par pas.
    /// Aucun transfert ni synchronisation, y compris entre plusieurs pas.
    pub fn flot_vers(&self, e: &EspaceTopo, dt: f64, pas: usize) -> Resultat<()> {
        self.verifier(e, true)?;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(Erreur::Config("topologie GPU : dt fini > 0 requis".into()));
        }
        let n = 4 * e.v;
        let lanc = Lancement::lineaire(n, BLOC);
        for _ in 0..pas {
            for (etape, zi) in [&e.z0, &e.z1, &e.z2].into_iter().enumerate() {
                self.app.lancer(
                    &self.f_z,
                    &lanc,
                    &[e.flot.arg(), Arg::I32(e.l as i32), zi.arg()],
                )?;
                self.app.lancer(
                    &self.f_maj,
                    &lanc,
                    &[
                        e.flot.arg(),
                        Arg::I64(n as i64),
                        e.z0.arg(),
                        e.z1.arg(),
                        e.z2.arg(),
                        Arg::I32(etape as i32),
                        Arg::F64(dt),
                    ],
                )?;
            }
        }
        Ok(())
    }

    /// Plaquette seule : deux réductions, une synchronisation, un double rapatrié.
    fn somme_plaquettes(&self, e: &EspaceTopo, cible: Cible) -> Resultat<f64> {
        self.verifier(e, true)?;
        self.app.lancer(
            &self.f_plaq,
            &Lancement::lineaire(e.v, BLOC),
            &[self.tampon(e, cible), Arg::I32(e.l as i32), e.part.arg()],
        )?;
        self.finale(e, 1)?;
        self.app.synchroniser()?;
        let mut s = [0.0];
        self.app.lire(&e.sortie, 0, &mut s)?;
        Ok(s[0])
    }

    pub fn e_plaq(&self, e: &EspaceTopo, cible: Cible) -> Resultat<f64> {
        Ok(24.0 * (1.0 - self.somme_plaquettes(e, cible)? / (6.0 * e.v as f64)))
    }

    pub fn action(&self, e: &EspaceTopo, cible: Cible) -> Resultat<f64> {
        Ok(6.0 * e.v as f64 - self.somme_plaquettes(e, cible)?)
    }

    pub fn balayage_cooling(&self, e: &EspaceTopo, dagger: bool) -> Resultat<()> {
        self.verifier(e, true)?;
        for mu in 0..4 {
            for parite in 0..2 {
                self.app.lancer(
                    &self.f_cool,
                    &Lancement::lineaire(e.v, BLOC),
                    &[
                        e.cooling.arg(),
                        Arg::I32(e.l as i32),
                        Arg::I32(mu),
                        Arg::I32(parite),
                        Arg::I32(i32::from(dagger)),
                    ],
                )?;
            }
        }
        Ok(())
    }

    pub fn champ_hote(&self, e: &EspaceTopo, cible: Cible) -> Resultat<ChampHote> {
        self.verifier(e, true)?;
        self.app.synchroniser()?;
        let mut q = vec![0.0; 16 * e.v];
        match cible {
            Cible::Source => self.app.lire(&e.source.borrow(), 0, &mut q)?,
            Cible::Flot => self.app.lire(&e.flot, 0, &mut q)?,
            Cible::Cooling => self.app.lire(&e.cooling, 0, &mut q)?,
        }
        Ok(ChampHote { l: e.l, q })
    }

    /// Témoin instrumental sur des W non unitaires ; W nul donne l'identité comme sur CPU.
    pub fn cool_id_maj(&self, w: &[Q], dagger: bool) -> Resultat<Vec<Q>> {
        if w.is_empty() {
            return Ok(Vec::new());
        }
        let mut entree = self.app.alloc::<Q>(w.len())?;
        let sortie = self.app.alloc::<Q>(w.len())?;
        self.app.ecrire(&mut entree, 0, w)?;
        self.app.lancer(
            &self.f_cool_id,
            &Lancement::lineaire(w.len(), BLOC),
            &[
                entree.arg(),
                sortie.arg(),
                Arg::I64(w.len() as i64),
                Arg::I32(i32::from(dagger)),
            ],
        )?;
        self.app.synchroniser()?;
        let mut u = vec![[0.0; 4]; w.len()];
        self.app.lire(&sortie, 0, &mut u)?;
        Ok(u)
    }

    /// Même grille, interpolation, prolongation et mode partiel que mesurer_cpu.
    pub fn mesurer(
        &self,
        e: &EspaceTopo,
        champ: &ChampHote,
        p: &ParamsTopo,
    ) -> Resultat<RecordTopo> {
        self.verifier(e, false)?;
        if !p.dt.is_finite()
            || p.dt <= 0.0
            || !p.cible_t2e.is_finite()
            || p.cible_t2e <= 0.0
            || p.pas_max < 500
        {
            return Err(Erreur::Config(
                "topologie : dt et cible finis > 0, pas_max ≥ 500 requis".into(),
            ));
        }
        let pas_a2 = TEMPS_A2.map(|t| (t / p.dt).round());
        if pas_a2.iter().any(|&n| n < 1.0 || n > p.pas_max as f64) {
            return Err(Erreur::Config(
                "topologie : les temps TEMPS_A2 doivent être accessibles avant pas_max".into(),
            ));
        }
        let pas_a2 = pas_a2.map(|n| n as usize);
        // Flot partiel : seulement t/a² = 1, 2, 3 (pas 100, 200, 300) ; t₀ non mesuré (avenant v4.3.1 §B).
        let pas_mesures: Vec<usize> = if p.flot_complet {
            pas_a2.to_vec()
        } else {
            pas_a2[..=IDX_T3].to_vec()
        };
        self.charger(e, champ)?;
        let mut record = RecordTopo {
            backend_mesure: self.app.backend().nom().into(),
            l: champ.l,
            plaq0: self.charges(e, Cible::Source)?.plaq,
            ..RecordTopo::default()
        };
        if !p.sans_flot {
            let mut precedent = (0.0, 0.0);
            let mut pas_t0 = None;
            let mut arret = if p.flot_complet {
                p.pas_max
            } else {
                pas_a2[IDX_T3]
            };
            let mut n = 0;
            loop {
                n += 1;
                self.flot_vers(e, p.dt, 1)?;
                let e_plaq = self.e_plaq(e, Cible::Flot)?;
                let t = n as f64 * p.dt;
                let t2e = t * t * e_plaq;
                if p.flot_complet
                    && record.t0.is_none()
                    && precedent.1 < p.cible_t2e
                    && t2e >= p.cible_t2e
                {
                    // Même interpolation et même garde sur le dénominateur que flot::echelles.
                    let denom = t2e - precedent.1;
                    let t0 = if denom.abs() < 1e-300 {
                        precedent.0
                    } else {
                        precedent.0 + (p.cible_t2e - precedent.1) / denom * (t - precedent.0)
                    };
                    let grille = MULT_T0.map(|k| (k * t0 / p.dt - 1e-9).ceil().max(n as f64));
                    if grille
                        .iter()
                        .any(|x| !x.is_finite() || *x >= usize::MAX as f64)
                    {
                        return Err(Erreur::Config(
                            "topologie : temps k·t₀ hors de la grille représentable".into(),
                        ));
                    }
                    let grille = grille.map(|x| x as usize);
                    arret = 500.max(pas_a2[3]).max(grille[3]);
                    pas_t0 = Some(grille);
                    record.t0 = Some(t0);
                    record.flot_t0 = Some(FlotT0::default());
                }
                let mesure_t0 = pas_t0.is_some_and(|grille| grille.contains(&n));
                if pas_mesures.contains(&n) || mesure_t0 {
                    let c = self.charges(e, Cible::Flot)?;
                    for &pas in &pas_mesures {
                        if pas == n {
                            record.flot_a2.e_plaq.push(e_plaq);
                            record.flot_a2.q_tre.push(c.q_tre);
                            record.flot_a2.q_imp.push(c.q_imp);
                        }
                    }
                    if let (Some(grille), Some(flot)) = (pas_t0, record.flot_t0.as_mut()) {
                        for pas in grille {
                            if pas == n {
                                flot.t_eff.push(t);
                                flot.q_tre.push(c.q_tre);
                                flot.q_imp.push(c.q_imp);
                            }
                        }
                    }
                }
                precedent = (t, t2e);
                if n >= arret {
                    break;
                }
            }
        }
        // Repartir explicitement de la copie brute ; aucune allocation ni téléversement.
        self.copie(e, &e.cooling)?;
        record.cooling = SerieCooling::default();
        for k in 0..=p.balayages {
            if k > 0 {
                self.balayage_cooling(e, true)?;
            }
            let s = self.densites(e, Cible::Cooling)?;
            record.cooling.q_imp.push(s[2]);
            record.cooling.q_tre.push(s[1]);
            record.cooling.action.push(6.0 * e.v as f64 - s[0]);
        }
        Ok(record)
    }
}
