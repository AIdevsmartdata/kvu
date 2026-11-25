//! [EXPÉRIMENTAL] Torelons (lignes de Polyakov enroulées lissées APE, impulsion transverse) SUR CARTE : l'équivalent de
//! `kvu su2 spectro run` (CPU). Même opérateur P_w(x⊥,t) = ½Tr ∏ Ũ_w, φ_s(t;p), corrélateur symétrisé (la fonction hôte
//! `spectro::correlateur` est réutilisée telle quelle). Génération par `UneBoite` (noyaux de production). Sorties au format de `run`.

use super::bloc::ParamBloc;
use super::flux::{SecteurCentre, SecteurFlux};
use super::{correlateur, Axes, Mutant};
use crate::commun::{Depart, Erreur, Graine, Precision, Resultat};
use crate::gpu::{assembler, Appareil, Arg, Fonction, Lancement, Module, OptionsCompil, Tampon};
use crate::su2::torsion::Torsion;
use crate::su2::une_boite::{ParamBoite, UneBoite};
use crate::su2::ParamChaine;

#[derive(Clone, Debug)]
pub struct ParamTorelon {
    pub cotes: [usize; 4],
    pub w: usize,
    pub temps: usize,
    pub niveaux: Vec<usize>,
    pub alpha: f64,
    pub moments: Vec<[i32; 2]>,
    pub tau_max: usize,
    pub precision: Precision,
    pub nor: usize,
    pub betas: Vec<f64>,
    pub graines: Vec<u64>,
    /// blocage de Teper (niveaux mesurés APRÈS les niveaux APE ; 2^b doit diviser L_w ; vide = comportement d'origine)
    pub bloc: ParamBloc,
    /// `--xi0` (voir `ParamGpu::xi0`)
    pub xi0: Option<f64>,
    /// `--ops-bruts` : remonte aussi les sommes brutes Phi_a(p,t) dans `MesTor::ops` (false = aucune conversion, sorties inchangees au bit)
    pub ops_bruts: bool,
    /// `--profil` : séparations τ du profil du tube (vide = éteint, aucune allocation ni lancement supplémentaire)
    pub profil: Vec<usize>,
    /// `--profil-niveau` : indice de l'opérateur (APE puis blocs) dont la carte Φ est prise ; None = le dernier (le plus lissé)
    pub profil_niveau: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct MesTor {
    /// [impulsion][paire][τ]
    pub corr: Vec<Vec<Vec<f64>>>,
    /// [impulsion][niveau] = ⟨φ⟩_t (Re, Im)
    pub phi_moyen: Vec<Vec<[f64; 2]>>,
    pub plaq: f64,
    pub plaq_st: Option<(f64, f64)>,
    /// Polyakov brut moyen dans la direction w
    pub poly_w: f64,
    /// `--ops-bruts` : Phi_a(p,t) de cette configuration, [t][p][a][re, im] aplati (vide si eteinte)
    pub ops: Vec<f64>,
    /// `--profil` : cartes Φ(x⊥,t) (niveau choisi) et ρ(x⊥,t) (électrique, magnétique) de cette configuration
    pub cartes: Option<super::profil::Cartes>,
}

enum Buf {
    F32(Tampon<f32>),
    F64(Tampon<f64>),
}
impl Buf {
    fn arg(&self) -> Arg {
        match self {
            Buf::F32(t) => t.arg(),
            Buf::F64(t) => t.arg(),
        }
    }
}

pub struct TorelonGpu {
    pub boite: UneBoite,
    app: Appareil,
    p: ParamTorelon,
    ax: Axes,
    _module: Module,
    f_ape: Fonction,
    f_lignes: Fonction,
    f_phi: Fonction,
    f_plaq: Fonction,
    plaq_an: Option<super::gpu::PlaqAn>,
    f_bloc: Fonction,
    f_ape_l: Fonction,
    f_lignes_l: Fonction,
    f_rho: Option<Fonction>,
    rho: Option<Tampon<f64>>,
    buf_a: Buf,
    buf_b: Buf,
    lig: Tampon<f64>,
    phi: Tampon<f64>,
    phi_raw: Tampon<f64>,
    ph: Tampon<f64>,
    ph0: Tampon<f64>,
    out_plaq: Tampon<f64>,
    nch: usize,
    v: usize,
    nblk: usize,
    /// `--secteur-flux` : noyau et tampon créés à la première mesure du secteur.
    flux_110: Option<(Fonction, Tampon<f64>)>,
    flux_111: Option<(Fonction, Tampon<f64>)>,
}

impl TorelonGpu {
    pub fn nouveau(app: &Appareil, p: &ParamTorelon) -> Resultat<TorelonGpu> {
        let nch = p.betas.len();
        if nch == 0 || p.graines.len() != nch {
            return Err(Erreur::Config("spectro gpu torelon : une graine et un β par chaîne".into()));
        }
        let ax = Axes::nouveaux(p.w, p.temps)?;
        if p.tau_max >= p.cotes[p.temps] || p.niveaux.is_empty() || p.niveaux.windows(2).any(|w| w[0] >= w[1]) {
            return Err(Erreur::Config("tau_max < ℓ_temps et niveaux croissants non vides requis".into()));
        }
        p.bloc.controler(&p.cotes, p.temps)?;
        if p.profil.iter().any(|&t| t == 0 || t > p.tau_max) {
            return Err(Erreur::Config("--profil : séparations 1 ≤ τ ≤ tau_max attendues".into()));
        }
        if p.profil_niveau.is_some_and(|i| i >= p.niveaux.len() + p.bloc.niveaux.len()) {
            return Err(Erreur::Config("--profil-niveau : indice d'opérateur hors liste".into()));
        }
        if let Some(&b) = p.bloc.niveaux.iter().find(|&&b| b >= usize::BITS as usize || !p.cotes[p.w].is_multiple_of(1usize << b)) {
            return Err(Erreur::Config(format!("--blocage : 2^{b} ne divise pas la longueur d'enroulement {}", p.cotes[p.w])));
        }
        let v: usize = p.cotes.iter().product();
        let (ta, tb, tt) = (p.cotes[ax.tr[0]], p.cotes[ax.tr[1]], p.cotes[p.temps]);
        let nperp = ta * tb;
        let nmom = p.moments.len();
        let mut ph = vec![0f64; nmom * nperp * 2];
        for (m, &[na, nb]) in p.moments.iter().enumerate() {
            for a in 0..ta {
                for b in 0..tb {
                    let t = 2.0 * std::f64::consts::PI * (na as f64 * a as f64 / ta as f64 + nb as f64 * b as f64 / tb as f64);
                    ph[(m * nperp + a * tb + b) * 2] = t.cos();
                    ph[(m * nperp + a * tb + b) * 2 + 1] = t.sin();
                }
            }
        }
        let mut ph_t = app.alloc::<f64>(ph.len())?;
        app.ecrire(&mut ph_t, 0, &ph)?;
        let mut ph0 = app.alloc::<f64>(nperp * 2)?;
        app.ecrire(&mut ph0, 0, &(0..nperp * 2).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect::<Vec<f64>>())?;
        let f32 = p.precision == Precision::F32;
        let boite = UneBoite::nouveau(
            app,
            &ParamBoite {
                cotes: p.cotes,
                torsion: Torsion::analyser("aucune")?,
                poids: 1.0,
                nor: p.nor,
                chaines: (0..nch)
                    .map(|c| ParamChaine { beta: p.betas[c], graine: Graine(p.graines[c]), depart: Depart::Chaud })
                    .collect(),
                precision: p.precision,
                impairs: p.cotes.iter().any(|c| c % 2 == 1),
                aniso: p.xi0.map(|x| (x, p.temps)),
                couture: None,
            },
        )?;
        let src = assembler(
            app.backend(),
            "su2_spectro",
            &[&super::gpu::SU2_SPECTRO],
            &[("KV_T", if f32 { "float" } else { "double" }), ("MUT", "0")],
        );
        let module = app.compiler(&src, &if f32 { OptionsCompil::rapide() } else { OptionsCompil::reference_k1() })?;
        let f = |n: &str| app.fonction(&module, n);
        let alloc_champ = |n: usize| -> Resultat<Buf> {
            if f32 {
                Ok(Buf::F32(app.alloc::<f32>(n)?))
            } else {
                Ok(Buf::F64(app.alloc::<f64>(n)?))
            }
        };
        let nblk = v.div_ceil(256);
        Ok(TorelonGpu {
            f_ape: f("k_ape")?,
            f_lignes: f("k_lignes")?,
            f_phi: f("k_phi")?,
            f_plaq: f("k_plaq")?,
            plaq_an: if p.xi0.is_some_and(|x| x != 1.0) { Some(super::gpu::PlaqAn::nouveau(app, &module, nch, v)?) } else { None },
            f_bloc: f("k_bloc")?,
            f_ape_l: f("k_ape_l")?,
            f_lignes_l: f("k_lignes_l")?,
            f_rho: if p.profil.is_empty() { None } else { Some(f("k_rho")?) },
            rho: if p.profil.is_empty() { None } else { Some(app.alloc::<f64>(nch * 2 * tt * nperp)?) },
            buf_a: alloc_champ(nch * 16 * v)?,
            buf_b: alloc_champ(nch * 16 * v)?,
            lig: app.alloc::<f64>(nch * tt * nperp)?,
            phi: app.alloc::<f64>(nch * (p.niveaux.len() + p.bloc.niveaux.len()) * nmom * tt * 2)?,
            phi_raw: app.alloc::<f64>(nch * tt * 2)?,
            ph: ph_t,
            ph0,
            out_plaq: app.alloc::<f64>(nch * nblk)?,
            nch,
            v,
            nblk,
            flux_110: None,
            flux_111: None,
            ax,
            app: app.clone(),
            p: p.clone(),
            _module: module,
            boite,
        })
    }

    /// Impulsions de `ops.bin` : les `moments` demandes (ordre inchange), puis leurs opposes absents.
    pub fn ops_impulsions(&self) -> Vec<[i32; 2]> {
        let mut v: Vec<[i32; 2]> = self.p.moments.clone();
        for &m in &self.p.moments {
            let opp = [-m[0], -m[1]];
            if !v.contains(&opp) {
                v.push(opp);
            }
        }
        v
    }

    pub fn init(&mut self) -> Resultat<()> {
        self.boite.init()
    }
    pub fn avancer(&mut self, n: usize) -> Resultat<()> {
        for _ in 0..n {
            self.boite.pas()?;
        }
        self.app.synchroniser()
    }

    /// `l` = 0 : lignes de liens ordinaires (noyau d'origine `k_lignes`) ; `l` ≥ 1 : superliens de longueur l (`k_lignes_l`).
    #[allow(clippy::too_many_arguments)]
    fn lignes_phi(&self, champ: Arg, geo: Arg, ph: &Tampon<f64>, nmom: usize, nlev: usize, lev: usize, out: &Tampon<f64>, l: usize) -> Resultat<()> {
        let p = &self.p;
        let (ta, tb, tt) = (p.cotes[self.ax.tr[0]], p.cotes[self.ax.tr[1]], p.cotes[p.temps]);
        if l == 0 {
            self.app.lancer(
                &self.f_lignes,
                &Lancement::lineaire_lot(tt * ta * tb, self.nch, 128),
                &[champ, geo, Arg::I32(p.w as i32), Arg::I32(p.temps as i32), Arg::I32(self.ax.tr[0] as i32), Arg::I32(self.ax.tr[1] as i32), self.lig.arg()],
            )?;
        } else {
            self.app.lancer(
                &self.f_lignes_l,
                &Lancement::lineaire_lot(tt * ta * tb, self.nch, 128),
                &[
                    champ,
                    geo,
                    Arg::I32(p.w as i32),
                    Arg::I32(p.temps as i32),
                    Arg::I32(self.ax.tr[0] as i32),
                    Arg::I32(self.ax.tr[1] as i32),
                    Arg::I32(l as i32),
                    self.lig.arg(),
                ],
            )?;
        }
        self.app.lancer(
            &self.f_phi,
            &Lancement { grille: [(nmom * tt) as u32, self.nch as u32, 1], bloc: [256, 1, 1], shmem: 0 },
            &[
                self.lig.arg(),
                ph.arg(),
                Arg::I32(nmom as i32),
                Arg::I32(tt as i32),
                Arg::I32((ta * tb) as i32),
                Arg::I32(nlev as i32),
                Arg::I32(lev as i32),
                out.arg(),
            ],
        )
    }

    /// carte Φ(x⊥,t) du dernier `lignes_phi` (tampon `lig`), toutes répliques : [c][t][a][b]
    fn lire_lig(&self) -> Resultat<Vec<f64>> {
        self.app.synchroniser()?;
        let p = &self.p;
        let mut h = vec![0f64; self.nch * p.cotes[p.temps] * p.cotes[self.ax.tr[0]] * p.cotes[self.ax.tr[1]]];
        self.app.lire(&self.lig, 0, &mut h)?;
        Ok(h)
    }

    /// Opérateurs (Polyakov brut, φ_s(t; p) de tous les niveaux) de toutes les chaînes sur leur champ courant, sans plaquette.
    fn calc_phi(&self) -> Resultat<Option<Vec<f64>>> {
        let p = &self.p;
        let (nch, v) = (self.nch, self.v);
        let nlev = p.niveaux.len() + p.bloc.niveaux.len();
        let nmom = p.moments.len();
        let geo = self.boite.arg_geo();
        let champ0 = self.boite.arg_champ_complet();
        // Polyakov brut (p = 0)
        self.lignes_phi(champ0, geo, &self.ph0, 1, 1, 0, &self.phi_raw, 0)?;
        let ilev_prof = p.profil_niveau.unwrap_or(nlev - 1);
        let mut phi_map: Option<Vec<f64>> = None;
        let mut cur = champ0;
        let mut qui = 0usize;
        let mut fait = 0usize;
        for (lev, &n) in p.niveaux.iter().enumerate() {
            for _ in fait..n {
                let dst = if qui == 0 { &self.buf_a } else { &self.buf_b };
                self.app.lancer(
                    &self.f_ape,
                    &Lancement::lineaire_lot(4 * v, nch, 256),
                    &[cur, dst.arg(), geo, Arg::I32(p.temps as i32), Arg::F64(p.alpha)],
                )?;
                cur = dst.arg();
                qui ^= 1;
            }
            fait = n;
            self.lignes_phi(cur, geo, &self.ph, nmom, nlev, lev, &self.phi, 0)?;
            if !p.profil.is_empty() && lev == ilev_prof {
                phi_map = Some(self.lire_lig()?);
            }
        }
        // niveaux de blocage (même chaîne que `spectro gpu`, référence processeur `bloc::torelons_avec_blocs`)
        if p.bloc.actif() {
            let pb = p.bloc.clone();
            let mut cur = champ0;
            let mut qui = 0usize;
            let lot = Lancement::lineaire_lot(4 * v, nch, 256);
            for _ in 0..pb.ape_bloc {
                let dst = if qui == 0 { self.buf_a.arg() } else { self.buf_b.arg() };
                self.app.lancer(&self.f_ape, &lot, &[cur, dst, geo, Arg::I32(p.temps as i32), Arg::F64(p.alpha)])?;
                cur = dst;
                qui ^= 1;
            }
            for b in 1..=*pb.niveaux.last().unwrap() {
                let dst = if qui == 0 { self.buf_a.arg() } else { self.buf_b.arg() };
                self.app.lancer(&self.f_bloc, &lot, &[cur, dst, geo, Arg::I32(p.temps as i32), Arg::I32(1 << (b - 1)), Arg::F64(pb.p_bloc)])?;
                cur = dst;
                qui ^= 1;
                for _ in 0..pb.lissage_bloc {
                    let dst = if qui == 0 { self.buf_a.arg() } else { self.buf_b.arg() };
                    self.app.lancer(&self.f_ape_l, &lot, &[cur, dst, geo, Arg::I32(p.temps as i32), Arg::F64(p.alpha), Arg::I32(1 << b)])?;
                    cur = dst;
                    qui ^= 1;
                }
                if let Some(ib) = pb.niveaux.iter().position(|&x| x == b) {
                    self.lignes_phi(cur, geo, &self.ph, nmom, nlev, p.niveaux.len() + ib, &self.phi, 1 << b)?;
                    if !p.profil.is_empty() && p.niveaux.len() + ib == ilev_prof {
                        phi_map = Some(self.lire_lig()?);
                    }
                }
            }
        }
        Ok(phi_map)
    }

    /// Algorithme à deux niveaux : masque de liens gelés (voir `UneBoite::fixer_masque`).
    pub fn fixer_masque(&mut self, m: Option<&[u8]>) -> Resultat<()> {
        self.boite.fixer_masque(m)
    }

    /// Algorithme à deux niveaux : φ_s(t; p) de toutes les chaînes, remonté sur l'hôte, indice ((((c·nlev + lev)·nmom + m)·tt + t)·2) (Re, Im).
    pub fn phi_hote(&mut self) -> Resultat<Vec<f64>> {
        let _ = self.calc_phi()?;
        self.app.synchroniser()?;
        let p = &self.p;
        let tt = p.cotes[p.temps];
        let nlev = p.niveaux.len() + p.bloc.niveaux.len();
        let mut phi = vec![0f64; self.nch * nlev * p.moments.len() * tt * 2];
        self.app.lire(&self.phi, 0, &mut phi)?;
        Ok(phi)
    }

    /// Flux à deux enroulements, opérateurs bruts [t][niveau][7] pour chaque chaîne.
    pub fn mesurer_flux(&mut self, f: &SecteurFlux) -> Resultat<Vec<Vec<f64>>> {
        self.mesurer_flux_centre(&SecteurCentre::Deux(*f))
    }

    /// `--secteur-flux` : opérateurs de toutes les chaînes sur leur champ courant, mêmes niveaux (APE puis blocs)
    /// que `mesurer`. Retourne, par chaîne, les opérateurs bruts [t][niveau][k].
    pub fn mesurer_flux_centre(&mut self, f: &SecteurCentre) -> Resultat<Vec<Vec<f64>>> {
        let p = &self.p;
        let (nch, v) = (self.nch, self.v);
        let tt = p.cotes[p.temps];
        let nlev = p.niveaux.len() + p.bloc.niveaux.len();
        f.controler_niveaux(&p.cotes, p.temps, &p.bloc)?;
        let taille_chaine = nlev * tt * f.n_sommes();
        let (nom, cache) = match f {
            SecteurCentre::Deux(_) => ("k_flux", &mut self.flux_110),
            SecteurCentre::Trois(_) => ("k_flux_111", &mut self.flux_111),
        };
        if cache.is_none() {
            *cache = Some((self.app.fonction(&self._module, nom)?, self.app.alloc::<f64>(nch * taille_chaine)?));
        }
        let (fk, out) = cache.as_ref().unwrap();
        let geo = self.boite.arg_geo();
        let champ0 = self.boite.arg_champ_complet();
        let lan = Lancement { grille: [tt as u32, nch as u32, 1], bloc: [256, 1, 1], shmem: 0 };
        let lancer = |champ: Arg, lev: usize, l: usize| -> Resultat<()> {
            let temps = Arg::I32(p.temps as i32);
            let (l, nlev, lev) = (Arg::I32(l as i32), Arg::I32(nlev as i32), Arg::I32(lev as i32));
            match *f {
                SecteurCentre::Deux(SecteurFlux { d: [d1, d2] }) => self.app.lancer(fk, &lan, &[champ, geo, Arg::I32(d1 as i32), Arg::I32(d2 as i32), temps, l, nlev, lev, out.arg()]),
                SecteurCentre::Trois(_) => self.app.lancer(fk, &lan, &[champ, geo, temps, l, nlev, lev, out.arg()]),
            }
        };
        let mut cur = champ0;
        let mut qui = 0usize;
        let mut fait = 0usize;
        for (lev, &n) in p.niveaux.iter().enumerate() {
            for _ in fait..n {
                let dst = if qui == 0 { &self.buf_a } else { &self.buf_b };
                self.app.lancer(&self.f_ape, &Lancement::lineaire_lot(4 * v, nch, 256), &[cur, dst.arg(), geo, Arg::I32(p.temps as i32), Arg::F64(p.alpha)])?;
                cur = dst.arg();
                qui ^= 1;
            }
            fait = n;
            lancer(cur, lev, 1)?;
        }
        if p.bloc.actif() {
            let pb = &p.bloc;
            let mut cur = champ0;
            let mut qui = 0usize;
            let lot = Lancement::lineaire_lot(4 * v, nch, 256);
            for _ in 0..pb.ape_bloc {
                let dst = if qui == 0 { self.buf_a.arg() } else { self.buf_b.arg() };
                self.app.lancer(&self.f_ape, &lot, &[cur, dst, geo, Arg::I32(p.temps as i32), Arg::F64(p.alpha)])?;
                cur = dst;
                qui ^= 1;
            }
            for b in 1..=*pb.niveaux.last().unwrap() {
                let dst = if qui == 0 { self.buf_a.arg() } else { self.buf_b.arg() };
                self.app.lancer(&self.f_bloc, &lot, &[cur, dst, geo, Arg::I32(p.temps as i32), Arg::I32(1 << (b - 1)), Arg::F64(pb.p_bloc)])?;
                cur = dst;
                qui ^= 1;
                for _ in 0..pb.lissage_bloc {
                    let dst = if qui == 0 { self.buf_a.arg() } else { self.buf_b.arg() };
                    self.app.lancer(&self.f_ape_l, &lot, &[cur, dst, geo, Arg::I32(p.temps as i32), Arg::F64(p.alpha), Arg::I32(1 << b)])?;
                    cur = dst;
                    qui ^= 1;
                }
                if let Some(ib) = pb.niveaux.iter().position(|&x| x == b) {
                    lancer(cur, p.niveaux.len() + ib, 1 << b)?;
                }
            }
        }
        self.app.synchroniser()?;
        let mut h = vec![0f64; nch * taille_chaine];
        self.app.lire(out, 0, &mut h)?;
        Ok(h.chunks_exact(taille_chaine).map(|c| super::flux::assembler_centre(f, c, nlev, tt)).collect())
    }

    #[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
    pub fn mesurer(&mut self) -> Resultat<Vec<MesTor>> {
        let phi_map = self.calc_phi()?;
        let p = &self.p;
        let (ta_, tb_) = (p.cotes[self.ax.tr[0]], p.cotes[self.ax.tr[1]]);
        let (nch, v) = (self.nch, self.v);
        let tt = p.cotes[p.temps];
        let nlev = p.niveaux.len() + p.bloc.niveaux.len();
        let nmom = p.moments.len();
        let geo = self.boite.arg_geo();
        let champ0 = self.boite.arg_champ_complet();
        self.app.lancer(&self.f_plaq, &Lancement::lineaire_lot(v, nch, 256), &[champ0, geo, self.out_plaq.arg()])?;
        let mut rho_map: Option<Vec<f64>> = None;
        if let (Some(fr), Some(rb)) = (&self.f_rho, &self.rho) {
            self.app.lancer(
                fr,
                &Lancement::lineaire_lot(tt * ta_ * tb_, nch, 128),
                &[champ0, geo, Arg::I32(p.w as i32), Arg::I32(p.temps as i32), Arg::I32(self.ax.tr[0] as i32), Arg::I32(self.ax.tr[1] as i32), rb.arg()],
            )?;
            self.app.synchroniser()?;
            let mut h = vec![0f64; nch * 2 * tt * ta_ * tb_];
            self.app.lire(rb, 0, &mut h)?;
            rho_map = Some(h);
        }
        if let Some(pa) = &self.plaq_an {
            pa.lancer(&self.app, champ0, geo, p.temps, v, nch)?;
        }
        self.app.synchroniser()?;
        let plaq_st = match &self.plaq_an {
            Some(pa) => Some(pa.lire(&self.app, nch, v)?),
            None => None,
        };
        let mut phi = vec![0f64; nch * nlev * nmom * tt * 2];
        self.app.lire(&self.phi, 0, &mut phi)?;
        let mut raw = vec![0f64; nch * tt * 2];
        self.app.lire(&self.phi_raw, 0, &mut raw)?;
        let mut pl = vec![0f64; nch * self.nblk];
        self.app.lire(&self.out_plaq, 0, &mut pl)?;
        // `--ops-bruts` : phi[(((c*nlev+lev)*nmom+m)*tt+t)*2] -> [t][p][a][re,im] ; O(p demande)=conj(stocke), O(-p ajoute)=stocke, p=0 : im = 0
        let imps = self.ops_impulsions();
        let nimp = imps.len();
        let mut ops_par_chaine: Vec<Vec<f64>> = vec![Vec::new(); nch];
        if p.ops_bruts {
            for c in 0..nch {
                let mut o = vec![0f64; tt * nimp * nlev * 2];
                for (ip, &m) in imps.iter().enumerate() {
                    let (src, conj) = match p.moments.iter().position(|&x| x == m) {
                        Some(j) => (j, true),
                        None => (p.moments.iter().position(|&x| x == [-m[0], -m[1]]).unwrap(), false),
                    };
                    let zero = m == [0, 0];
                    for lev in 0..nlev {
                        for t in 0..tt {
                            let s = (((c * nlev + lev) * nmom + src) * tt + t) * 2;
                            let dst = ((t * nimp + ip) * nlev + lev) * 2;
                            o[dst] = phi[s];
                            o[dst + 1] = if zero { 0.0 } else if conj { -phi[s + 1] } else { phi[s + 1] };
                        }
                    }
                }
                ops_par_chaine[c] = o;
            }
        }
        let mut out = Vec::new();
        for c in 0..nch {
            // phis[impulsion][niveau][t] = (Re, Im)
            let mut corr = Vec::new();
            let mut pm = Vec::new();
            for m in 0..nmom {
                let phis: Vec<Vec<[f64; 2]>> = (0..nlev)
                    .map(|lev| {
                        (0..tt)
                            .map(|t| {
                                let o = ((((c * nlev + lev) * nmom + m) * tt) + t) * 2;
                                [phi[o], phi[o + 1]]
                            })
                            .collect()
                    })
                    .collect();
                corr.push(correlateur(&phis, p.tau_max, Mutant::Aucun));
                pm.push(super::phi_moyen(&phis));
            }
            let poly = (0..tt).map(|t| raw[(c * tt + t) * 2]).sum::<f64>() / tt as f64;
            out.push(MesTor {
                corr,
                phi_moyen: pm,
                plaq: pl[c * self.nblk..(c + 1) * self.nblk].iter().sum::<f64>() / (6.0 * v as f64),
                plaq_st: plaq_st.as_ref().map(|x| x[c]),
                poly_w: poly,
                ops: std::mem::take(&mut ops_par_chaine[c]),
                cartes: match (&phi_map, &rho_map) {
                    (Some(ph), Some(rh)) => {
                        let n3 = tt * ta_ * tb_;
                        Some(super::profil::Cartes { phi: ph[c * n3..(c + 1) * n3].to_vec(), rho: rh[c * 2 * n3..(c + 1) * 2 * n3].to_vec() })
                    }
                    _ => None,
                },
            });
        }
        Ok(out)
    }
}
