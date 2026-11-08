//! Pilote GPU des observables : compile `su2_commun.h` + `obs_wilson.cu` + `obs_flot.cu` en un seul
//! module (« obs ») et lance les noyaux sur un champ résident (F64, ou F32 promu en FP64 à la lecture).
//! Propriétaire : T7.
//!
//! Toutes les sommes sont FP64 à arbre fixe : un même champ donne les mêmes octets à chaque lancement.
//! Les noyaux sont compilés SANS contraction multiplication-addition (`OptionsCompil::reference_k1`),
//! pour retrouver la référence CPU quaternionique à l'arrondi près.

use super::flot::ParamsFlot;
use super::ref_cpu::{
    assembler_courbe, nombre_de_pas, orientations, pas_mesures, verifier_params_wilson, Densites, COEF_E_PLAQ,
};
use super::wilson::{ParamsWilson, TableWilson};
use crate::commun::{ChampHote, Erreur, Precision, Resultat};
use crate::gpu::{assembler, Appareil, Arg, Fonction, Lancement, Module, OptionsCompil, Tampon};
use crate::noyaux::{OBS_FLOT, OBS_WILSON, SU2_COMMUN};
use crate::su2::{ChampGpu, TamponChamp};

/// Taille de bloc des noyaux (doit valoir OBS_BLOC dans obs_wilson.cu).
const BLOC: u32 = 256;
/// Longueur R maximale (OBS_MAXR dans obs_wilson.cu).
pub const R_MAX_GPU: usize = 32;

/// Module compilé et fonctions des observables.
pub struct ModuleObs {
    app: Appareil,
    /// Gardé en vie : les fonctions en dépendent.
    #[allow(dead_code)]
    module: Module,
    f_wilson: Fonction,
    f_wilson_f32: Fonction,
    f_final: Fonction,
    f_poly: Fonction,
    f_poly_f32: Fonction,
    f_pcorr: Fonction,
    f_copie: Fonction,
    f_copie_f32: Fonction,
    f_z: Fonction,
    f_maj: Fonction,
    f_dens: Fonction,
}

/// Pointeur de départ de la chaîne `c` dans le champ résident, et précision.
fn base_chaine(champ: &ChampGpu, c: usize) -> Resultat<(Arg, Precision)> {
    if c >= champ.nchaines {
        return Err(Erreur::Config(format!(
            "chaîne {c} hors du lot ({} chaînes)",
            champ.nchaines
        )));
    }
    let decalage = c * 16 * champ.v();
    Ok(match &champ.tampon {
        TamponChamp::F64(t) => (t.arg_decale(decalage), Precision::F64),
        TamponChamp::F32(t) => (t.arg_decale(decalage), Precision::F32),
    })
}

impl ModuleObs {
    /// Provenance du module compilé (source assemblée, options, code objet), pour le manifeste des sorties.
    pub fn provenance(&self) -> crate::gpu::ProvenanceCompil {
        self.module.provenance().clone()
    }

    /// Assemble, compile (cache disque de T2) et résout les fonctions.
    pub fn nouveau(app: &Appareil) -> Resultat<ModuleObs> {
        let src = assembler(app.backend(), "obs", &[&SU2_COMMUN, &OBS_WILSON, &OBS_FLOT], &[]);
        let module = app.compiler(&src, &OptionsCompil::reference_k1())?;
        let f = |nom: &str| app.fonction(&module, nom);
        Ok(ModuleObs {
            app: app.clone(),
            f_wilson: f("k_obs_wilson")?,
            f_wilson_f32: f("k_obs_wilson_f32")?,
            f_final: f("k_obs_final")?,
            f_poly: f("k_obs_polyakov")?,
            f_poly_f32: f("k_obs_polyakov_f32")?,
            f_pcorr: f("k_obs_pcorr")?,
            f_copie: f("k_obs_copie")?,
            f_copie_f32: f("k_obs_copie_f32")?,
            f_z: f("k_obs_z")?,
            f_maj: f("k_obs_maj")?,
            f_dens: f("k_obs_densites")?,
            module,
        })
    }

    /// Somme finale : `out[idx] = scale · Σ_i in[i·stride + idx]`, idx = 0..nsorties.
    fn finale(&self, entree: Arg, n: usize, stride: usize, scale: f64, sortie: Arg, nsorties: usize) -> Resultat<()> {
        let cfg = Lancement {
            grille: [nsorties as u32, 1, 1],
            bloc: [BLOC, 1, 1],
            shmem: 0,
        };
        self.app.lancer(
            &self.f_final,
            &cfg,
            &[
                entree,
                Arg::I64(n as i64),
                Arg::I64(stride as i64),
                Arg::F64(scale),
                sortie,
            ],
        )
    }

    /// Table des boucles de Wilson de la chaîne `c`.
    pub fn boucles_wilson(&self, champ: &ChampGpu, c: usize, p: &ParamsWilson) -> Resultat<TableWilson> {
        verifier_params_wilson(p)?;
        if p.r_max > R_MAX_GPU {
            return Err(Erreur::Config(format!(
                "GPU : r_max = {} > {R_MAX_GPU} (limite du noyau)",
                p.r_max
            )));
        }
        let (base, prec) = base_chaine(champ, c)?;
        let (l, v) = (champ.l, champ.v());
        let nb = v.div_ceil(BLOC as usize);
        let rt = p.r_max * p.t_max;
        let part = self.app.alloc::<f64>(nb * rt)?;
        let sortie = self.app.alloc::<f64>(rt)?;
        let f = if prec == Precision::F64 {
            &self.f_wilson
        } else {
            &self.f_wilson_f32
        };
        let ors = orientations(p);
        let mut w = vec![0.0; rt];
        let mut tmp = vec![0.0; rt];
        for &(mu, nu) in &ors {
            let cfg = Lancement {
                grille: [nb as u32, 1, 1],
                bloc: [BLOC, 1, 1],
                shmem: 0,
            };
            self.app.lancer(
                f,
                &cfg,
                &[
                    base,
                    Arg::I32(l as i32),
                    Arg::I32(mu as i32),
                    Arg::I32(nu as i32),
                    Arg::I32(p.r_max as i32),
                    Arg::I32(p.t_max as i32),
                    part.arg(),
                ],
            )?;
            self.finale(part.arg(), nb, rt, 1.0 / v as f64, sortie.arg(), rt)?;
            self.app.synchroniser()?;
            self.app.lire(&sortie, 0, &mut tmp)?;
            for (a, b) in w.iter_mut().zip(&tmp) {
                *a += *b;
            }
        }
        let n = ors.len() as f64;
        for x in &mut w {
            *x /= n;
        }
        Ok(TableWilson {
            r_max: p.r_max,
            t_max: p.t_max,
            w,
        })
    }

    /// ℓ(x⃗) de la direction `dir` : tampon de L³ doubles (non synchronisé).
    fn polyakov_sites(&self, champ: &ChampGpu, c: usize, dir: usize) -> Resultat<Tampon<f64>> {
        if dir > 3 {
            return Err(Erreur::Config(format!("direction de Polyakov {dir} hors de 0..3")));
        }
        let (base, prec) = base_chaine(champ, c)?;
        let l3 = champ.v() / champ.l;
        let ell = self.app.alloc::<f64>(l3)?;
        let f = if prec == Precision::F64 {
            &self.f_poly
        } else {
            &self.f_poly_f32
        };
        self.app.lancer(
            f,
            &Lancement::lineaire(l3, BLOC),
            &[base, Arg::I32(champ.l as i32), Arg::I32(dir as i32), ell.arg()],
        )?;
        Ok(ell)
    }

    /// ⟨ℓ⟩ dans les 4 directions (P0..P3 de su2_hip : même arbre fixe que `k_final`).
    pub fn polyakov_moyen(&self, champ: &ChampGpu, c: usize) -> Resultat<[f64; 4]> {
        let l3 = champ.v() / champ.l;
        let sortie = self.app.alloc::<f64>(4)?;
        // Les tampons ℓ restent vivants jusqu'à la synchronisation : les noyaux sont asynchrones.
        let mut sites = Vec::with_capacity(4);
        for d in 0..4 {
            let ell = self.polyakov_sites(champ, c, d)?;
            self.finale(ell.arg(), l3, 1, 1.0 / l3 as f64, sortie.arg_decale(d), 1)?;
            sites.push(ell);
        }
        self.app.synchroniser()?;
        let mut out = [0.0; 4];
        self.app.lire(&sortie, 0, &mut out)?;
        drop(sites);
        Ok(out)
    }

    /// Corrélateur de Polyakov r = 0..L/2.
    pub fn correlateur_polyakov(&self, champ: &ChampGpu, c: usize, dir_temps: usize) -> Resultat<Vec<f64>> {
        let l = champ.l;
        let l3 = champ.v() / l;
        let rmax = l / 2;
        let ell = self.polyakov_sites(champ, c, dir_temps)?;
        let nb = l3.div_ceil(BLOC as usize);
        let part = self.app.alloc::<f64>(nb * (rmax + 1))?;
        let sortie = self.app.alloc::<f64>(rmax + 1)?;
        let cfg = Lancement {
            grille: [nb as u32, 1, 1],
            bloc: [BLOC, 1, 1],
            shmem: 0,
        };
        self.app.lancer(
            &self.f_pcorr,
            &cfg,
            &[ell.arg(), Arg::I32(l as i32), Arg::I32(rmax as i32), part.arg()],
        )?;
        self.finale(
            part.arg(),
            nb,
            rmax + 1,
            1.0 / (3.0 * l3 as f64),
            sortie.arg(),
            rmax + 1,
        )?;
        self.app.synchroniser()?;
        let mut out = vec![0.0; rmax + 1];
        self.app.lire(&sortie, 0, &mut out)?;
        Ok(out)
    }

    /// Espace de travail d'un flot : copie FP64 du champ de la chaîne `c` et forces.
    pub fn flot_prepare(&self, champ: &ChampGpu, c: usize) -> Resultat<EspaceFlot> {
        let (base, prec) = base_chaine(champ, c)?;
        let v = champ.v();
        let n = 4 * v;
        let u = self.app.alloc::<f64>(16 * v)?;
        let f = if prec == Precision::F64 {
            &self.f_copie
        } else {
            &self.f_copie_f32
        };
        self.app
            .lancer(f, &Lancement::lineaire(n, BLOC), &[base, u.arg(), Arg::I64(n as i64)])?;
        let nb = v.div_ceil(BLOC as usize);
        Ok(EspaceFlot {
            l: champ.l,
            v,
            u,
            z0: self.app.alloc::<f64>(12 * v)?,
            z1: self.app.alloc::<f64>(12 * v)?,
            z2: self.app.alloc::<f64>(12 * v)?,
            part: self.app.alloc::<f64>(3 * nb)?,
            sortie: self.app.alloc::<f64>(3)?,
        })
    }

    /// Un pas RK3 sur l'espace de travail (6 lancements, sans synchronisation).
    pub fn flot_pas(&self, e: &EspaceFlot, dt: f64) -> Resultat<()> {
        let n = 4 * e.v;
        let lanc = Lancement::lineaire(n, BLOC);
        let z = |zi: &Tampon<f64>| {
            self.app
                .lancer(&self.f_z, &lanc, &[e.u.arg(), Arg::I32(e.l as i32), zi.arg()])
        };
        let maj = |etape: i32| {
            self.app.lancer(
                &self.f_maj,
                &lanc,
                &[
                    e.u.arg(),
                    Arg::I64(n as i64),
                    e.z0.arg(),
                    e.z1.arg(),
                    e.z2.arg(),
                    Arg::I32(etape),
                    Arg::F64(dt),
                ],
            )
        };
        z(&e.z0)?;
        maj(0)?;
        z(&e.z1)?;
        maj(1)?;
        z(&e.z2)?;
        maj(2)
    }

    /// Densités du champ de travail (synchronise).
    pub fn flot_densites(&self, e: &EspaceFlot) -> Resultat<Densites> {
        let nb = e.v.div_ceil(BLOC as usize);
        let cfg = Lancement {
            grille: [nb as u32, 1, 1],
            bloc: [BLOC, 1, 1],
            shmem: 0,
        };
        self.app
            .lancer(&self.f_dens, &cfg, &[e.u.arg(), Arg::I32(e.l as i32), e.part.arg()])?;
        self.finale(e.part.arg(), nb, 3, 1.0, e.sortie.arg(), 3)?;
        self.app.synchroniser()?;
        let mut s = [0.0; 3];
        self.app.lire(&e.sortie, 0, &mut s)?;
        let v = e.v as f64;
        Ok(Densites {
            e_plaq: COEF_E_PLAQ * (1.0 - s[0] / (6.0 * v)),
            e_trefle: s[1] / v,
            q_top: s[2],
        })
    }

    /// Champ de travail rapatrié sur l'hôte (synchronise).
    pub fn flot_champ(&self, e: &EspaceFlot) -> Resultat<ChampHote> {
        self.app.synchroniser()?;
        let mut q = vec![0.0; 16 * e.v];
        self.app.lire(&e.u, 0, &mut q)?;
        Ok(ChampHote { l: e.l, q })
    }

    /// Flot complet de la chaîne `c` : mesures aux pas voulus, sur une copie du champ.
    pub fn flot_wilson(&self, champ: &ChampGpu, c: usize, p: &ParamsFlot) -> Resultat<super::flot::CourbeFlot> {
        let n_pas = nombre_de_pas(p)?;
        let e = self.flot_prepare(champ, c)?;
        let mesures = pas_mesures(n_pas, p.mesure_chaque);
        let mut dens = Vec::with_capacity(mesures.len());
        let mut fait = 0usize;
        for &m in &mesures {
            while fait < m {
                self.flot_pas(&e, p.dt)?;
                fait += 1;
            }
            dens.push(self.flot_densites(&e)?);
        }
        Ok(assembler_courbe(p, &mesures, &dens))
    }

    /// `n_pas` pas de flot sur une copie, champ final rapatrié.
    pub fn flot_champ_final(&self, champ: &ChampGpu, c: usize, dt: f64, n_pas: usize) -> Resultat<ChampHote> {
        let e = self.flot_prepare(champ, c)?;
        for _ in 0..n_pas {
            self.flot_pas(&e, dt)?;
        }
        self.flot_champ(&e)
    }

    /// Densités du champ de la chaîne `c` sans flot.
    pub fn densites(&self, champ: &ChampGpu, c: usize) -> Resultat<Densites> {
        let e = self.flot_prepare(champ, c)?;
        self.flot_densites(&e)
    }
}

/// Tampons d'un flot : copie FP64 du champ (4V quaternions), forces Z0..Z2 (3 doubles par lien),
/// partiels et sortie des densités.
pub struct EspaceFlot {
    l: usize,
    v: usize,
    u: Tampon<f64>,
    z0: Tampon<f64>,
    z1: Tampon<f64>,
    z2: Tampon<f64>,
    part: Tampon<f64>,
    sortie: Tampon<f64>,
}
