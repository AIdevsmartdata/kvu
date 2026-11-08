//! Chemin FP64 de référence (noyaux su2_commun.h + su2_ref64.cu + su2_bord.cu). Propriétaire : T3.
//!
//! Une seule compilation par générateur : `assembler(backend, "su2_ref64", [SU2_COMMUN, SU2_REF64, SU2_BORD])`.
//! Le pas, la mesure et les compteurs reproduisent la procédure hôte de su2_hip.cpp / su2_bord.cu :
//! - sweep = 4 directions × 2 parités, un lancement par (mu, parité), blocs de 256, grille (V/2, chaînes) ;
//! - `mesurer` = `measure_async` de K1 (k_plaq_site, k_reduce, k_final, k_polyakov, 4 k_final), SANS
//!   synchronisation, par chaîne (blockIdx.y), arbre de réduction fixe ;
//! - tranches R2-3 = k_plaq_st + k_tranche.

use super::bord::{Bord, CAPACITE_TRANCHES};
use super::champ::{valeurs_par_chaine, ChampGpu};
use super::{AppMasque, Compteurs, ConfigLot, Gel, NoyauxSu2};
use crate::commun::{BackendKind, ChampHote, Depart, Erreur, Graine, Precision, Resultat};
use crate::gpu::{assembler, Appareil, Arg, Fonction, Lancement, Module, ProvenanceCompil, Tampon};
use crate::noyaux::{SU2_BORD, SU2_COMMUN, SU2_REF64};

/// Taille de bloc des noyaux (celle de K1 : l'arbre de réduction en dépend).
const BS: u32 = 256;

/// Historique de tranches (S_t, T_t) : plaquettes par site, historique, capacité.
struct Tranches {
    site_s: Tampon<f64>,
    site_t: Tampon<f64>,
    hist: Tampon<f64>,
    capacite: usize,
}

/// Générateur FP64 sur GPU (HIP ou CUDA), implémente `NoyauxSu2`.
pub struct Ref64Gpu {
    pub cfg: ConfigLot,
    app: Appareil,
    champ: ChampGpu,
    prov: ProvenanceCompil,
    _module: Module,
    f_init: Fonction,
    f_hb: Fonction,
    f_or: Fonction,
    f_plaq_site: Fonction,
    f_reduce: Fonction,
    f_final: Fonction,
    f_pol: Fonction,
    f_test: Fonction,
    f_set_bord: Fonction,
    f_gen_jauge: Fonction,
    f_app_jauge: Fonction,
    f_plaq_st: Fonction,
    f_tranche: Fonction,
    betas: Tampon<f64>,
    k0s: Tampon<u32>,
    k1s: Tampon<u32>,
    hot: Tampon<i32>,
    cap: Tampon<u64>,
    tries: Tampon<u64>,
    site: Tampon<f64>,
    part: Tampon<f64>,
    pol: Tampon<f64>,
    hist: Option<(Tampon<f64>, usize)>,
    tranches: Option<Tranches>,
    sweep_id: u32,
    gel: i32,
    /// Masque de liens gelés (WP-DIR) : un octet par lien, 4V octets, partagé par les chaînes du lot.
    masque: Option<Tampon<u8>>,
    app_masque: AppMasque,
}

/// Nombre maximal de chaînes d'un lot (limite de la dimension y de la grille).
pub const CHAINES_MAX: usize = 65_535;

/// Mémoire device nécessaire à un lot (octets) : champ (16·L⁴ f64 par chaîne), tampons par site de la plaquette
/// et des tranches (3·L⁴ f64), réduction et Polyakov, historiques (non comptés : choisis par l'appelant).
pub fn memoire_necessaire(l: usize, nchaines: usize) -> u64 {
    let v = (l as u64).pow(4);
    let par_chaine = 8 * (16 * v + 3 * v + v / 256 + 4 * (v / l as u64) + 64);
    par_chaine * nchaines as u64
}

/// Source assemblée du chemin FP64 (testable sans GPU) : socle commun, noyaux K1 étendus, bord.
pub fn source_ref64(backend: BackendKind, mutant_metro: bool) -> crate::gpu::SourceAssemblee {
    let defines: &[(&str, &str)] = if mutant_metro { &[("K1_MUTANT_METRO", "1")] } else { &[] };
    assembler(backend, "su2_ref64", &[&SU2_COMMUN, &SU2_REF64, &SU2_BORD], defines)
}

impl Ref64Gpu {
    pub fn nouveau(app: &Appareil, cfg: ConfigLot) -> Resultat<Ref64Gpu> {
        super::valider(&cfg)?;
        if cfg.precision != Precision::F64 {
            return Err(Erreur::Config("Ref64Gpu ne calcule qu'en F64".into()));
        }
        if cfg.chaines.len() > CHAINES_MAX {
            return Err(Erreur::Config(format!("{} chaînes : au plus {CHAINES_MAX} par lot", cfg.chaines.len())));
        }
        let besoin = memoire_necessaire(cfg.l, cfg.chaines.len());
        if let Ok(libre) = app.memoire_libre_octets() {
            if besoin > libre / 10 * 9 {
                return Err(Erreur::Ressources(format!(
                    "lot de {} chaînes à L = {} : {:.1} Go de mémoire GPU nécessaires, {:.1} Go libres",
                    cfg.chaines.len(),
                    cfg.l,
                    besoin as f64 / crate::stockage::garde::OCTETS_PAR_GO,
                    libre as f64 / crate::stockage::garde::OCTETS_PAR_GO
                )));
            }
        }
        let backend = app.backend();
        let src = source_ref64(backend, cfg.mutant_metro);
        let module = app.compiler(&src, &cfg.options)?;
        let arch = app.arch();
        let prov = ProvenanceCompil {
            nom: src.nom.clone(),
            backend,
            arch: arch.clone(),
            sha256_source: src.sha256.clone(),
            parties: src.parties.clone(),
            options: cfg.options.clone(),
            arguments: cfg.options.arguments(backend, &arch),
            sha256_code: module.sha256_code.clone(),
        };
        let f = |nom: &str| app.fonction(&module, nom);
        let (l, nch) = (cfg.l, cfg.chaines.len());
        let v = l * l * l * l;
        let nb = v.div_ceil(BS as usize);
        let l3 = v / l;

        let mut betas = app.alloc::<f64>(nch)?;
        let mut k0s = app.alloc::<u32>(nch)?;
        let mut k1s = app.alloc::<u32>(nch)?;
        let mut hot = app.alloc::<i32>(nch)?;
        let beta_h: Vec<f64> = cfg.chaines.iter().map(|c| c.beta).collect();
        let cles: Vec<(u32, u32)> = cfg.chaines.iter().map(|c| c.graine.cle()).collect();
        let hot_h: Vec<i32> = cfg.chaines.iter().map(|c| i32::from(c.depart == Depart::Chaud)).collect();
        app.ecrire(&mut betas, 0, &beta_h)?;
        app.ecrire(&mut k0s, 0, &cles.iter().map(|k| k.0).collect::<Vec<_>>())?;
        app.ecrire(&mut k1s, 0, &cles.iter().map(|k| k.1).collect::<Vec<_>>())?;
        app.ecrire(&mut hot, 0, &hot_h)?;

        Ok(Ref64Gpu {
            champ: ChampGpu::allouer(app, l, nch, Precision::F64)?,
            f_init: f("k_init")?,
            f_hb: f("k_heatbath")?,
            f_or: f("k_overrelax")?,
            f_plaq_site: f("k_plaq_site")?,
            f_reduce: f("k_reduce")?,
            f_final: f("k_final")?,
            f_pol: f("k_polyakov")?,
            f_test: f("k_test_x0")?,
            f_set_bord: f("k_set_bord")?,
            f_gen_jauge: f("k_gen_jauge")?,
            f_app_jauge: f("k_app_jauge")?,
            f_plaq_st: f("k_plaq_st")?,
            f_tranche: f("k_tranche")?,
            betas,
            k0s,
            k1s,
            hot,
            cap: app.alloc::<u64>(nch)?,
            tries: app.alloc::<u64>(nch)?,
            site: app.alloc::<f64>(nch * v)?,
            part: app.alloc::<f64>(nch * nb)?,
            pol: app.alloc::<f64>(nch * 4 * l3)?,
            hist: None,
            tranches: None,
            sweep_id: 0,
            gel: 0,
            masque: None,
            app_masque: AppMasque::Tout,
            app: app.clone(),
            prov,
            _module: module,
            cfg,
        })
    }

    fn l(&self) -> usize {
        self.cfg.l
    }
    fn nch(&self) -> usize {
        self.cfg.chaines.len()
    }
    fn v(&self) -> usize {
        self.cfg.l.pow(4)
    }

    fn tampon_champ(&self) -> Resultat<&Tampon<f64>> {
        match &self.champ.tampon {
            super::TamponChamp::F64(t) => Ok(t),
            super::TamponChamp::F32(_) => Err(Erreur::Config("champ F32 dans un générateur F64".into())),
        }
    }

    /// Argument « début du champ de la chaîne `c` » (un lancement sur UNE chaîne).
    fn arg_champ_chaine(&self, c: usize) -> Resultat<Arg> {
        Ok(self.tampon_champ()?.arg_decale(c * valeurs_par_chaine(self.l())))
    }

    fn lancement_chaines(&self, n: usize) -> Lancement {
        Lancement::lineaire_lot(n, self.nch(), BS)
    }

    /// Un sweep de bain thermique avec diagnostic `diag_ntry` (4V entrées par chaîne : essais du tirage de
    /// chaque lien, 0 pour un lien gelé ou un tirage au plafond, 0xFFFFFFFF pour une agrafe nulle).
    pub fn sweep_hb_diag(&mut self) -> Resultat<Vec<u32>> {
        let diag = self.app.alloc::<u32>(self.nch() * 4 * self.v())?;
        self.hb(diag.arg())?;
        self.app.synchroniser()?;
        let mut h = vec![0u32; diag.len()];
        self.app.lire(&diag, 0, &mut h)?;
        Ok(h)
    }

    /// Argument « masque » d'un noyau : le tampon si un masque est posé ET que cette étape le respecte, sinon un pointeur nul.
    fn arg_masque(&self, respecte: bool) -> Arg {
        match (&self.masque, respecte) {
            (Some(m), true) => m.arg(),
            _ => Arg::Nul,
        }
    }

    fn hb(&mut self, diag: Arg) -> Resultat<()> {
        let cfg = self.lancement_chaines(self.v() / 2);
        let u = self.tampon_champ()?.arg();
        let masque = self.arg_masque(self.app_masque.bain());
        for mu in 0..4i32 {
            for p in 0..2i32 {
                self.app.lancer(
                    &self.f_hb,
                    &cfg,
                    &[
                        u,
                        Arg::I32(self.l() as i32),
                        Arg::I32(mu),
                        Arg::I32(p),
                        self.betas.arg(),
                        self.k0s.arg(),
                        self.k1s.arg(),
                        Arg::U32(self.sweep_id),
                        self.cap.arg(),
                        self.tries.arg(),
                        diag,
                        Arg::I32(self.gel),
                        masque,
                    ],
                )?;
            }
        }
        self.sweep_id = self.sweep_id.wrapping_add(1);
        Ok(())
    }

    /// n tirages indépendants de x0 à α fixé (`--test-hb` de su2_hip) : (tirages, nombre de plafonds).
    pub fn test_x0(&self, alpha: f64, n: usize, graine: Graine) -> Resultat<(Vec<f64>, u64)> {
        let out = self.app.alloc::<f64>(n.max(1))?;
        let cap = self.app.alloc::<u64>(1)?;
        let (k0, k1) = graine.cle();
        self.app.lancer(
            &self.f_test,
            &Lancement::lineaire(n, BS),
            &[Arg::F64(alpha), Arg::I64(n as i64), Arg::U32(k0), Arg::U32(k1), out.arg(), cap.arg()],
        )?;
        self.app.synchroniser()?;
        let mut h = vec![0f64; n.max(1)];
        self.app.lire(&out, 0, &mut h)?;
        h.truncate(n);
        let mut c = [0u64; 1];
        self.app.lire(&cap, 0, &mut c)?;
        Ok((h, c[0]))
    }

    /// Empreinte FNV-1a 64 des octets f64 des liens mu = 1..3, s ∈ [0, L³) de la chaîne `c`
    /// (fnv_bord de su2_bord), lue directement sur le device (sans télécharger tout le champ).
    pub fn fnv_bord_device(&self, c: usize) -> Resultat<u64> {
        let (v, l3) = (self.v(), self.v() / self.l());
        let base = c * valeurs_par_chaine(self.l());
        let mut h = super::bord::FNV_DEPART;
        let mut buf = vec![0f64; 4 * l3];
        for mu in 1..4 {
            self.app.lire(self.tampon_champ()?, base + mu * 4 * v, &mut buf)?;
            h = super::bord::fnv1a_f64(h, &buf);
        }
        Ok(h)
    }

    /// Alloue (ou agrandit, en conservant les cases déjà écrites) l'historique de tranches.
    fn preparer_tranches(app: &Appareil, slot: &mut Option<Tranches>, nch: usize, v: usize, l: usize, capacite: usize)
        -> Resultat<()> {
        let l2 = 2 * l;
        if slot.as_ref().is_some_and(|t| t.capacite >= capacite) {
            return Ok(());
        }
        let mut hist = app.alloc::<f64>(nch * capacite * l2)?;
        let (site_s, site_t) = match slot.take() {
            Some(old) => {
                let n = old.capacite * l2;
                let mut h = vec![0f64; nch * n];
                app.lire(&old.hist, 0, &mut h)?;
                for c in 0..nch {
                    app.ecrire(&mut hist, c * capacite * l2, &h[c * n..(c + 1) * n])?;
                }
                (old.site_s, old.site_t)
            }
            None => (app.alloc::<f64>(nch * v)?, app.alloc::<f64>(nch * v)?),
        };
        *slot = Some(Tranches { site_s, site_t, hist, capacite });
        Ok(())
    }

    /// Pose + jauge sur le device (corps de `poser_bord_device`).
    fn poser_device(&mut self, c: usize, bord: &Bord, jauge: Option<Graine>) -> Resultat<(u64, f64)> {
        if c >= self.nch() {
            return Err(Erreur::Config(format!("chaîne {c} hors du lot")));
        }
        let (l, v) = (self.l(), self.v());
        let l3 = v / l;
        let u = self.arg_champ_chaine(c)?;
        let cfg_bord = Lancement::lineaire(3 * l3, BS);
        // (a) pose du bord
        match bord {
            Bord::Froid => self.app.lancer(
                &self.f_set_bord,
                &cfg_bord,
                &[u, Arg::I32(l as i32), Arg::I32(0), Arg::U32(0), Arg::U32(0), Arg::Nul],
            )?,
            Bord::Chaud(g) => {
                let (k0, k1) = g.cle();
                self.app.lancer(
                    &self.f_set_bord,
                    &cfg_bord,
                    &[u, Arg::I32(l as i32), Arg::I32(1), Arg::U32(k0), Arg::U32(k1), Arg::Nul],
                )?
            }
            Bord::Copie(h) => {
                if h.l != l {
                    return Err(Erreur::Config(format!("bord copié de L = {} dans un lot de L = {l}", h.l)));
                }
                let mut src = self.app.alloc::<f64>(valeurs_par_chaine(l))?;
                self.app.ecrire(&mut src, 0, &h.q)?;
                self.app.lancer(
                    &self.f_set_bord,
                    &cfg_bord,
                    &[u, Arg::I32(l as i32), Arg::I32(2), Arg::U32(0), Arg::U32(0), src.arg()],
                )?;
                self.app.synchroniser()?;
            }
        }
        self.app.synchroniser()?;
        let fnv = self.fnv_bord_device(c)?;
        // (b) jauge aléatoire de l'hyperplan x0 = 0 (contrôle nul C2a : invariance exacte des S_t, T_t)
        let mut ecart = -1.0;
        if let Some(g) = jauge {
            let l2 = 2 * l;
            self.mesurer_tranches(0, CAPACITE_TRANCHES)?;
            let m0 = self.historique_tranches()?;
            let gtab = self.app.alloc::<f64>(4 * l3)?;
            let (k0, k1) = g.cle();
            self.app.lancer(
                &self.f_gen_jauge,
                &Lancement::lineaire(l3, BS),
                &[gtab.arg(), Arg::I32(l as i32), Arg::U32(k0), Arg::U32(k1)],
            )?;
            self.app.lancer(&self.f_app_jauge, &Lancement::lineaire(4 * v, BS), &[u, Arg::I32(l as i32), gtab.arg()])?;
            self.app.synchroniser()?;
            self.mesurer_tranches(1, CAPACITE_TRANCHES)?;
            let m1 = self.historique_tranches()?;
            let cap = self.tranches.as_ref().map_or(CAPACITE_TRANCHES, |t| t.capacite);
            let o = c * cap * l2;
            ecart = 0.0;
            for q in 0..l2 {
                ecart = f64::max(ecart, (m1[o + l2 + q] - m0[o + q]).abs());
            }
        }
        Ok((fnv, ecart))
    }
}

impl NoyauxSu2 for Ref64Gpu {
    fn precision(&self) -> Precision {
        Precision::F64
    }
    fn backend(&self) -> BackendKind {
        self.app.backend()
    }
    fn config(&self) -> &ConfigLot {
        &self.cfg
    }
    fn init(&mut self) -> Resultat<()> {
        let n = 4 * self.v();
        self.app.lancer(
            &self.f_init,
            &self.lancement_chaines(n),
            &[self.tampon_champ()?.arg(), Arg::I32(self.l() as i32), self.hot.arg(), self.k0s.arg(), self.k1s.arg()],
        )?;
        self.app.synchroniser()?;
        let zeros = vec![0u64; self.nch()];
        self.app.ecrire(&mut self.cap, 0, &zeros)?;
        self.app.ecrire(&mut self.tries, 0, &zeros)?;
        self.sweep_id = 0;
        Ok(())
    }
    fn fixer_gel(&mut self, gel: Gel) -> Resultat<()> {
        self.gel = i32::from(gel == Gel::BordX0Spatial);
        Ok(())
    }
    fn fixer_masque(&mut self, masque: Option<&[u8]>, application: AppMasque) -> Resultat<()> {
        self.app_masque = application;
        match masque {
            None => {
                self.masque = None;
                Ok(())
            }
            Some(m) => {
                let attendu = 4 * self.v();
                if m.len() != attendu {
                    return Err(Erreur::Config(format!("masque de {} octets : {attendu} attendus (4·L⁴, L = {})", m.len(), self.l())));
                }
                let mut t = self.app.alloc::<u8>(attendu)?;
                self.app.ecrire(&mut t, 0, m)?;
                self.app.synchroniser()?;
                self.masque = Some(t);
                Ok(())
            }
        }
    }
    fn sweep_hb(&mut self) -> Resultat<()> {
        self.hb(Arg::Nul)
    }
    fn sweep_or(&mut self) -> Resultat<()> {
        let cfg = self.lancement_chaines(self.v() / 2);
        let u = self.tampon_champ()?.arg();
        let masque = self.arg_masque(self.app_masque.surrelax());
        for mu in 0..4i32 {
            for p in 0..2i32 {
                self.app.lancer(
                    &self.f_or,
                    &cfg,
                    &[u, Arg::I32(self.l() as i32), Arg::I32(mu), Arg::I32(p), Arg::I32(self.gel), masque],
                )?;
            }
        }
        Ok(())
    }
    fn allouer_historique(&mut self, capacite: usize) -> Resultat<()> {
        let t = self.app.alloc::<f64>(self.nch() * capacite * 5)?;
        self.hist = Some((t, capacite));
        self.app.synchroniser()
    }
    fn mesurer(&mut self, idx: usize) -> Resultat<()> {
        let (hist, hcap) = self.hist.as_ref().ok_or_else(|| Erreur::Config("historique non alloué".into()))?;
        if idx >= *hcap {
            return Err(Erreur::Config(format!("historique plein : idx {idx} ≥ capacité {hcap}")));
        }
        let (l, v, nch) = (self.l(), self.v(), self.nch());
        let nb = v.div_ceil(BS as usize);
        let l3 = v / l;
        let u = self.tampon_champ()?.arg();
        let un_bloc = Lancement { grille: [1, nch as u32, 1], bloc: [BS, 1, 1], shmem: 0 };
        let pas_hist = Arg::I64((5 * hcap) as i64);
        let app = &self.app;
        app.lancer(&self.f_plaq_site, &Lancement::lineaire_lot(v, nch, BS), &[u, Arg::I32(l as i32), self.site.arg()])?;
        app.lancer(
            &self.f_reduce,
            &Lancement::lineaire_lot(v, nch, BS),
            &[self.site.arg(), Arg::I64(v as i64), self.part.arg()],
        )?;
        app.lancer(
            &self.f_final,
            &un_bloc,
            &[
                self.part.arg(),
                Arg::I64(nb as i64),
                Arg::I64(nb as i64),
                Arg::F64(1.0 / (6.0 * (v as i64) as f64)),
                hist.arg_decale(5 * idx),
                pas_hist,
            ],
        )?;
        app.lancer(&self.f_pol, &Lancement::lineaire_lot(4 * l3, nch, BS), &[u, Arg::I32(l as i32), self.pol.arg()])?;
        for mu in 0..4usize {
            app.lancer(
                &self.f_final,
                &un_bloc,
                &[
                    self.pol.arg_decale(mu * l3),
                    Arg::I64(l3 as i64),
                    Arg::I64((4 * l3) as i64),
                    Arg::F64(1.0 / (l3 as i64) as f64),
                    hist.arg_decale(5 * idx + 1 + mu),
                    pas_hist,
                ],
            )?;
        }
        Ok(())
    }
    fn historique(&self) -> Resultat<Vec<f64>> {
        let (hist, hcap) = self.hist.as_ref().ok_or_else(|| Erreur::Config("historique non alloué".into()))?;
        self.app.synchroniser()?;
        let mut h = vec![0f64; self.nch() * hcap * 5];
        self.app.lire(hist, 0, &mut h)?;
        Ok(h)
    }
    fn mesurer_tranches(&mut self, idx: usize, capacite: usize) -> Resultat<()> {
        let (l, v, nch) = (self.l(), self.v(), self.nch());
        let u = self.tampon_champ()?.arg();
        Self::preparer_tranches(&self.app, &mut self.tranches, nch, v, l, capacite)?;
        let t = self.tranches.as_ref().ok_or_else(|| Erreur::Gpu("historique de tranches absent".into()))?;
        if idx >= t.capacite {
            return Err(Erreur::Config(format!("historique de tranches plein : idx {idx} ≥ capacité {}", t.capacite)));
        }
        let l2 = 2 * l;
        let l3 = v / l;
        let sc = 1.0 / (3.0 * (l3 as i64) as f64);
        let pas = Arg::I64((t.capacite * l2) as i64);
        let app = &self.app;
        app.lancer(
            &self.f_plaq_st,
            &Lancement::lineaire_lot(v, nch, BS),
            &[u, Arg::I32(l as i32), t.site_s.arg(), t.site_t.arg()],
        )?;
        let grille = Lancement { grille: [l as u32, nch as u32, 1], bloc: [BS, 1, 1], shmem: 0 };
        let f = &self.f_tranche;
        app.lancer(f, &grille, &[t.site_s.arg(), Arg::I32(l as i32), Arg::F64(sc), t.hist.arg_decale(idx * l2), pas])?;
        app.lancer(f, &grille, &[t.site_t.arg(), Arg::I32(l as i32), Arg::F64(sc), t.hist.arg_decale(idx * l2 + l), pas])?;
        Ok(())
    }
    fn historique_tranches(&self) -> Resultat<Vec<f64>> {
        let t = self.tranches.as_ref().ok_or_else(|| Erreur::Config("historique de tranches non alloué".into()))?;
        self.app.synchroniser()?;
        let mut h = vec![0f64; self.nch() * t.capacite * 2 * self.l()];
        self.app.lire(&t.hist, 0, &mut h)?;
        Ok(h)
    }
    fn telecharger(&self, chaine: usize) -> Resultat<ChampHote> {
        self.app.synchroniser()?;
        self.champ.telecharger(&self.app, chaine)
    }
    fn televerser(&mut self, chaine: usize, c: &ChampHote) -> Resultat<()> {
        self.app.synchroniser()?;
        let app = self.app.clone();
        self.champ.televerser(&app, chaine, c)
    }
    fn champ_gpu(&self) -> Option<&ChampGpu> {
        Some(&self.champ)
    }
    fn compteurs(&self) -> Resultat<Vec<Compteurs>> {
        self.app.synchroniser()?;
        let n = self.nch();
        let (mut cap, mut tries) = (vec![0u64; n], vec![0u64; n]);
        self.app.lire(&self.cap, 0, &mut cap)?;
        self.app.lire(&self.tries, 0, &mut tries)?;
        Ok(cap.into_iter().zip(tries).map(|(plafond, essais)| Compteurs { plafond, essais }).collect())
    }
    fn sweep_id(&self) -> u32 {
        self.sweep_id
    }
    fn fixer_sweep_id(&mut self, s: u32) {
        self.sweep_id = s;
    }
    fn synchroniser(&self) -> Resultat<()> {
        self.app.synchroniser()
    }
    fn provenance(&self) -> Vec<ProvenanceCompil> {
        vec![self.prov.clone()]
    }
    fn poser_bord_device(&mut self, chaine: usize, bord: &Bord, jauge: Option<Graine>) -> Option<Resultat<(u64, f64)>> {
        Some(self.poser_device(chaine, bord, jauge))
    }
    fn empreinte_bord_device(&self, chaine: usize) -> Option<Resultat<u64>> {
        Some(self.fnv_bord_device(chaine))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memoire_d_un_lot() {
        // L = 32, une chaîne : champ = 128·L⁴ octets = 134 Mo, plus les tampons de mesure
        let m = memoire_necessaire(32, 1);
        assert!(m > 134_217_728 && m < 200_000_000, "{m}");
        assert_eq!(memoire_necessaire(32, 16), 16 * m);
        // L = 64 : le champ F64 d'une chaîne pèse 2,1 Go
        assert!(memoire_necessaire(64, 1) > 2_147_483_648);
    }
}
