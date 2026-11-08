//! Chemin FP32 rapide (noyaux su2_commun.h + su2_f32.cu). Propriétaire : T4.
//!
//! Stockage float4, arithmétique FP32 (contraction FMA autorisée), réunitarisation de chaque
//! lien mis à jour, clé et compteurs Philox comme le FP64 (compteur (lien local, sweep, tirage,
//! étiquette), un mot de 32 bits par uniforme float), mesures (plaquette, Polyakov, tranches)
//! accumulées en FP64 à arbre fixe.
//! Doit prouver la MÊME LOI que le FP64 (gates LOI-0 à LOI-4 et LOI-τ, ARCHITECTURE_UNIFIEE.md §6.2) :
//! les bits ne sont PAS ceux du FP64, la distribution l'est.
//!
//! Les fonctions pures de ce fichier (promotion, arrondi, miroir hôte du tirage de x0) sont testées
//! sans GPU ; les noyaux eux-mêmes le sont par `tests/su2_f32_*.rs` (tests `#[ignore = "GPU"]`).

use super::{ChampGpu, Compteurs, ConfigLot, Gel, NoyauxSu2, TamponChamp};
use crate::commun::{BackendKind, ChampHote, Depart, Erreur, Graine, Precision, Resultat};
use crate::gpu::{
    assembler, Appareil, Arg, Fonction, Lancement, Module, ProvenanceCompil, Tampon,
};
use crate::heritage::su2_hb::philox4x32_10;
use crate::noyaux::{SU2_COMMUN, SU2_F32};

/// Fils par bloc de tous les noyaux de mise à jour et de mesure (les réductions en exigent 256).
const BLOC: u32 = 256;
/// Nombre maximal de chaînes d'un lot (limite de la grille y).
pub const CHAINES_MAX: usize = 65_535;

// ------------------------------------------------------------------------------------------
// Fonctions pures : conversion hôte ↔ device et miroir du tirage de x0
// ------------------------------------------------------------------------------------------

/// Promotion float → double, exacte (aucun arrondi).
pub fn promouvoir(src: &[f32]) -> Vec<f64> {
    src.iter().map(|&v| f64::from(v)).collect()
}

/// Tolérance sur |q|² − 1 sous laquelle un quaternion arrondi en float est considéré déjà unitaire : 4e-6, soit le
/// double de la borne de LOI-0 sur | |q| − 1 | (2e-6). La réunitarisation float laisse |q|² − 1 de l'ordre de
/// quelques 1e-7 ; une vraie dérive ou un champ hors sphère (≥ 1e-4 en pratique) est très au-dessus.
pub const TOLERANCE_NORME_F32: f64 = 4e-6;

/// Arrondi au plus proche en float, puis normalisation de chaque quaternion (groupes de 4).
///
/// La normalisation n'agit que si |q|² s'écarte de 1 de plus de [`TOLERANCE_NORME_F32`] : un champ issu
/// du device (déjà unitaire à la précision du float) revient donc à l'identique, bit pour bit
/// (`televerser(telecharger(x)) == x`), ce que la reprise à l'octet exige. Un champ FP64 de norme 1 à
/// 1e-12 passe de même sans changement après arrondi.
pub fn arrondir_normaliser(src: &[f64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(src.len());
    for q in src.chunks_exact(4) {
        let mut f = [q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32];
        let n2: f64 = f.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        if (n2 - 1.0).abs() > TOLERANCE_NORME_F32 {
            let n = n2.sqrt();
            for v in &mut f {
                *v = (f64::from(*v) / n) as f32;
            }
        }
        out.extend_from_slice(&f);
    }
    out
}

/// Uniforme float dans ]0,1[ d'un mot de 32 bits : miroir exact de `u01f` du noyau
/// (23 bits, milieu de cellule ; jamais 0 ni 1).
pub fn u01f_hote(a: u32) -> f32 {
    ((a >> 9) as f32 + 0.5) * (1.0 / 8_388_608.0)
}

/// Forme à 24 bits de la spécification d'origine, ((a >> 8) + 0,5)·2⁻²⁴, conservée UNIQUEMENT pour
/// démontrer par un test qu'elle sort de ]0,1[ (le plus grand mot donne exactement 1,0f).
pub fn u01f_24bits_refusee(a: u32) -> f32 {
    ((a >> 8) as f32 + 0.5) * (1.0 / 16_777_216.0)
}

const DEUX_PI: f32 = 6.283_185_5;
const ETIQUETTE_HB: u32 = 0x4842_4154;
const PLAFOND_ESSAIS: u32 = 1_000_000;

/// Miroir hôte de `sample_x0f` : tirage de x0 ∝ sqrt(1−x²) exp(α x) sur [−1,1] en arithmétique float,
/// mêmes uniformes (mêmes mots Philox, même ordre) que le noyau. Renvoie (x0, essais, ok).
/// Sert au test d'accord noyau/hôte et aux tests de loi sans GPU.
pub fn tirage_x0_hote(alpha: f32, li: u32, sweep_id: u32, draw: &mut u32, k0: u32, k1: u32) -> (f32, u32, bool) {
    let mut ntry = 0u32;
    let bloc = |draw: &mut u32| {
        let r = philox4x32_10([li, sweep_id, *draw, ETIQUETTE_HB], k0, k1);
        *draw += 1;
        r
    };
    if alpha >= 2.0 {
        let deux_alpha = 2.0 * alpha;
        while ntry < PLAFOND_ESSAIS {
            let r = bloc(draw);
            let (a1, a2, a3, a4) = (u01f_hote(r[0]), u01f_hote(r[1]), u01f_hote(r[2]), u01f_hote(r[3]));
            let c = (DEUX_PI * a2).cos();
            let lam2 = -(a1.ln() + c * c * a3.ln()) / deux_alpha;
            ntry += 1;
            if a4 * a4 <= 1.0 - lam2 {
                return (1.0 - 2.0 * lam2, ntry, true);
            }
        }
    } else if alpha < 1e-6 {
        while ntry < PLAFOND_ESSAIS {
            let r = bloc(draw);
            let (y, a2) = (2.0 * u01f_hote(r[0]) - 1.0, u01f_hote(r[1]));
            ntry += 1;
            if a2 * a2 <= 1.0 - y * y {
                return (y, ntry, true);
            }
        }
    } else {
        let um = -(-2.0 * alpha).exp_m1();
        while ntry < PLAFOND_ESSAIS {
            let r = bloc(draw);
            let (a1, a2) = (u01f_hote(r[0]), u01f_hote(r[1]));
            let y = 1.0 + (-a1 * um).ln_1p() / alpha;
            ntry += 1;
            if a2 * a2 <= 1.0 - y * y {
                return (y, ntry, true);
            }
        }
    }
    (1.0, ntry, false)
}

/// Valide une configuration de lot pour le chemin FP32 (pur, testé sans GPU).
pub fn valider(cfg: &ConfigLot) -> Resultat<()> {
    if cfg.precision != Precision::F32 {
        return Err(Erreur::Config("Rapide32Gpu exige precision = f32".into()));
    }
    if cfg.l < 2 || !cfg.l.is_multiple_of(2) {
        return Err(Erreur::Config(format!("L = {} : L pair ≥ 2 requis", cfg.l)));
    }
    if cfg.chaines.is_empty() || cfg.chaines.len() > CHAINES_MAX {
        return Err(Erreur::Config(format!("{} chaînes : 1 à {CHAINES_MAX} requises", cfg.chaines.len())));
    }
    let v = cfg.l.pow(4);
    if 4 * v >= (1usize << 32) {
        return Err(Erreur::Config(format!("L = {} : l'indice de lien dépasse 32 bits (compteur Philox)", cfg.l)));
    }
    if let Some(c) = cfg.chaines.iter().find(|c| !c.beta.is_finite() || c.beta < 0.0) {
        return Err(Erreur::Config(format!("β = {} invalide", c.beta)));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------
// Générateur sur GPU
// ------------------------------------------------------------------------------------------

/// Fonctions device du module `su2_f32`.
struct Noyaux {
    init: Fonction,
    heatbath: Fonction,
    overrelax: Fonction,
    plaq_site: Fonction,
    reduce: Fonction,
    finale: Fonction,
    polyakov: Fonction,
    plaq_st: Fonction,
    tranche: Fonction,
    test_x0: Fonction,
}

/// Tampons de travail des mesures (alloués à la première mesure).
struct TravailMesure {
    site: Tampon<f64>,
    part: Tampon<f64>,
    pol: Tampon<f64>,
}

/// Tampons des tranches (alloués au premier appel de `mesurer_tranches`).
struct TravailTranches {
    s: Tampon<f64>,
    t: Tampon<f64>,
    hist: Tampon<f64>,
    capacite: usize,
}

/// Générateur FP32 sur GPU, implémente `NoyauxSu2`.
pub struct Rapide32Gpu {
    pub cfg: ConfigLot,
    app: Appareil,
    champ: ChampGpu,
    noyaux: Noyaux,
    _module: Module,
    prov: ProvenanceCompil,
    betas: Tampon<f32>,
    cles: Tampon<u32>,
    departs: Tampon<u32>,
    n_cap: Tampon<u64>,
    n_essais: Tampon<u64>,
    travail: Option<TravailMesure>,
    hist: Option<(Tampon<f64>, usize)>,
    tranches: Option<TravailTranches>,
    sweep_id: u32,
    gel: Gel,
}

impl Rapide32Gpu {
    /// Configuration type du chemin rapide : options de compilation `rapide()` (contraction FMA permise).
    pub fn config(l: usize, chaines: Vec<super::ParamChaine>, nor: usize) -> ConfigLot {
        ConfigLot {
            l,
            chaines,
            nor,
            precision: Precision::F32,
            options: crate::gpu::OptionsCompil::rapide(),
            mutant_metro: false,
        }
    }

    /// Compile le module, alloue le champ (zéros) et les tampons par chaîne. Aucun départ n'est fait :
    /// appeler `init()`.
    pub fn nouveau(app: &Appareil, cfg: ConfigLot) -> Resultat<Rapide32Gpu> {
        valider(&cfg)?;
        let backend = app.backend();
        let nch = cfg.chaines.len();
        let v = cfg.l.pow(4);
        // Le mutant de contrôle négatif est un `#define` assemblé : sa trace entre dans l'empreinte de la source.
        let defines: &[(&str, &str)] = if cfg.mutant_metro { &[("K1_MUTANT_METRO", "1")] } else { &[] };
        let src = assembler(backend, "su2_f32", &[&SU2_COMMUN, &SU2_F32], defines);
        let module = app.compiler(&src, &cfg.options)?;
        let f = |nom: &str| app.fonction(&module, nom);
        let noyaux = Noyaux {
            init: f("k_init_f32")?,
            heatbath: f("k_heatbath_f32")?,
            overrelax: f("k_overrelax_f32")?,
            plaq_site: f("k_plaq_site_f32")?,
            reduce: f("k_reduce_f32")?,
            finale: f("k_final_f32")?,
            polyakov: f("k_polyakov_f32")?,
            plaq_st: f("k_plaq_st_f32")?,
            tranche: f("k_tranche_f32")?,
            test_x0: f("k_test_x0_f32")?,
        };
        let arch = app.arch();
        let prov = ProvenanceCompil {
            nom: src.nom.clone(),
            backend,
            arguments: cfg.options.arguments(backend, &arch),
            arch,
            sha256_source: src.sha256.clone(),
            parties: src.parties.clone(),
            options: cfg.options.clone(),
            sha256_code: module.sha256_code.clone(),
        };
        let champ = ChampGpu { l: cfg.l, nchaines: nch, tampon: TamponChamp::F32(app.alloc::<f32>(16 * v * nch)?) };
        let betas_h: Vec<f32> = cfg.chaines.iter().map(|c| c.beta as f32).collect();
        let cles_h: Vec<u32> = cfg.chaines.iter().flat_map(|c| [c.graine.cle().0, c.graine.cle().1]).collect();
        let departs_h: Vec<u32> = cfg.chaines.iter().map(|c| u32::from(c.depart == Depart::Chaud)).collect();
        let mut betas = app.alloc::<f32>(nch)?;
        let mut cles = app.alloc::<u32>(2 * nch)?;
        let mut departs = app.alloc::<u32>(nch)?;
        app.ecrire(&mut betas, 0, &betas_h)?;
        app.ecrire(&mut cles, 0, &cles_h)?;
        app.ecrire(&mut departs, 0, &departs_h)?;
        Ok(Rapide32Gpu {
            n_cap: app.alloc::<u64>(nch)?,
            n_essais: app.alloc::<u64>(nch)?,
            app: app.clone(),
            champ,
            noyaux,
            _module: module,
            prov,
            betas,
            cles,
            departs,
            travail: None,
            hist: None,
            tranches: None,
            sweep_id: 0,
            gel: Gel::Aucun,
            cfg,
        })
    }

    fn v(&self) -> usize {
        self.cfg.l.pow(4)
    }

    fn nch(&self) -> usize {
        self.cfg.chaines.len()
    }

    fn champ_f32(&self) -> Resultat<&Tampon<f32>> {
        match &self.champ.tampon {
            TamponChamp::F32(t) => Ok(t),
            TamponChamp::F64(_) => Err(Erreur::Gpu("champ FP64 dans un générateur FP32".into())),
        }
    }

    /// Gel passé aux noyaux : 1 = liens spatiaux de l'hyperplan x0 = 0 gelés.
    fn gel_i32(&self) -> i32 {
        i32::from(self.gel == Gel::BordX0Spatial)
    }

    /// Un balayage des 8 sous-réseaux (4 directions × 2 parités) avec le noyau `f`.
    fn balayer(&self, f: &Fonction, args_apres_parite: &dyn Fn(i32, i32) -> Vec<Arg>) -> Resultat<()> {
        let lan = Lancement::lineaire_lot(self.v() / 2, self.nch(), BLOC);
        for mu in 0..4 {
            for parite in 0..2 {
                self.app.lancer(f, &lan, &args_apres_parite(mu, parite))?;
            }
        }
        Ok(())
    }

    /// Champ brut (bits exacts float, disposition (c·4V + li)·4 + k) de TOUTES les chaînes.
    pub fn etat_brut(&self) -> Resultat<Vec<f32>> {
        let t = self.champ_f32()?;
        let mut h = vec![0f32; t.len()];
        self.app.lire(t, 0, &mut h)?;
        Ok(h)
    }

    /// Restaure un champ brut (bits exacts) et le compteur de sweeps : la reprise à l'octet.
    pub fn restaurer_brut(&mut self, champ: &[f32], sweep_id: u32) -> Resultat<()> {
        let attendu = 16 * self.v() * self.nch();
        if champ.len() != attendu {
            return Err(Erreur::Config(format!("état brut de {} flottants, {attendu} attendus", champ.len())));
        }
        let app = self.app.clone();
        match &mut self.champ.tampon {
            TamponChamp::F32(t) => app.ecrire(t, 0, champ)?,
            TamponChamp::F64(_) => return Err(Erreur::Gpu("champ FP64 dans un générateur FP32".into())),
        }
        self.sweep_id = sweep_id;
        Ok(())
    }

    /// Test unitaire du tirage de x0 : `n` tirages indépendants à `alpha` (arrondi au float, comme dans
    /// le noyau), avec la fonction même du bain thermique. Renvoie les tirages promus en double et le nombre
    /// de tirages ayant atteint le plafond de 10⁶ essais (attendu : 0). Avec `cfg.mutant_metro`, le tirage
    /// est le Metropolis naïf du contrôle négatif.
    pub fn test_x0(&self, alpha: f64, n: usize, graine: u64) -> Resultat<(Vec<f64>, u64)> {
        if n == 0 {
            return Ok((Vec::new(), 0));
        }
        let sortie = self.app.alloc::<f32>(n)?;
        let mut cap = self.app.alloc::<u64>(1)?;
        self.app.zero(&mut cap)?;
        let (k0, k1) = Graine(graine).cle();
        let lan = Lancement::lineaire(n, BLOC);
        self.app.lancer(
            &self.noyaux.test_x0,
            &lan,
            &[Arg::F32(alpha as f32), Arg::I64(n as i64), Arg::U32(k0), Arg::U32(k1), sortie.arg(), cap.arg()],
        )?;
        self.app.synchroniser()?;
        let mut h = vec![0f32; n];
        self.app.lire(&sortie, 0, &mut h)?;
        let mut c = [0u64; 1];
        self.app.lire(&cap, 0, &mut c)?;
        Ok((promouvoir(&h), c[0]))
    }

    fn travail(&mut self) -> Resultat<()> {
        if self.travail.is_none() {
            let (v, nch) = (self.v(), self.nch());
            let nb = v.div_ceil(BLOC as usize);
            let l3 = v / self.cfg.l;
            self.travail = Some(TravailMesure {
                site: self.app.alloc::<f64>(nch * v)?,
                part: self.app.alloc::<f64>(nch * nb)?,
                pol: self.app.alloc::<f64>(nch * 4 * l3)?,
            });
        }
        Ok(())
    }
}

impl NoyauxSu2 for Rapide32Gpu {
    fn precision(&self) -> Precision {
        Precision::F32
    }

    fn backend(&self) -> BackendKind {
        self.app.backend()
    }

    fn config(&self) -> &ConfigLot {
        &self.cfg
    }

    fn init(&mut self) -> Resultat<()> {
        let (l, v, nch) = (self.cfg.l, self.v(), self.nch());
        self.app.zero(&mut self.n_cap)?;
        self.app.zero(&mut self.n_essais)?;
        self.sweep_id = 0;
        let lan = Lancement::lineaire_lot(4 * v, nch, BLOC);
        let t = self.champ_f32()?;
        self.app.lancer(&self.noyaux.init, &lan, &[t.arg(), Arg::I32(l as i32), self.departs.arg(), self.cles.arg()])?;
        Ok(())
    }

    fn fixer_gel(&mut self, gel: Gel) -> Resultat<()> {
        self.gel = gel;
        Ok(())
    }

    fn sweep_hb(&mut self) -> Resultat<()> {
        let l = self.cfg.l as i32;
        let (gel, sweep) = (self.gel_i32(), self.sweep_id);
        let t = self.champ_f32()?;
        self.balayer(&self.noyaux.heatbath, &|mu, parite| {
            vec![
                t.arg(),
                Arg::I32(l),
                Arg::I32(mu),
                Arg::I32(parite),
                self.betas.arg(),
                self.cles.arg(),
                Arg::U32(sweep),
                Arg::I32(gel),
                self.n_cap.arg(),
                self.n_essais.arg(),
            ]
        })?;
        self.sweep_id = self.sweep_id.wrapping_add(1);
        Ok(())
    }

    fn sweep_or(&mut self) -> Resultat<()> {
        let l = self.cfg.l as i32;
        let gel = self.gel_i32();
        let t = self.champ_f32()?;
        self.balayer(&self.noyaux.overrelax, &|mu, parite| {
            vec![t.arg(), Arg::I32(l), Arg::I32(mu), Arg::I32(parite), Arg::I32(gel)]
        })
    }

    fn allouer_historique(&mut self, capacite: usize) -> Resultat<()> {
        if capacite == 0 {
            return Err(Erreur::Config("capacité d'historique nulle".into()));
        }
        self.hist = Some((self.app.alloc::<f64>(self.nch() * capacite * 5)?, capacite));
        Ok(())
    }

    fn mesurer(&mut self, idx: usize) -> Resultat<()> {
        self.travail()?;
        let (l, v, nch) = (self.cfg.l, self.v(), self.nch());
        let (nb, l3) = (v.div_ceil(BLOC as usize), v / l);
        let (hist, cap) = self.hist.as_ref().ok_or_else(|| Erreur::Config("historique non alloué".into()))?;
        if idx >= *cap {
            return Err(Erreur::Config(format!("historique plein : case {idx} ≥ capacité {cap}")));
        }
        let tr = self.travail.as_ref().ok_or_else(|| Erreur::Gpu("tampons de mesure absents".into()))?;
        let champ = self.champ_f32()?;
        let (ostride, i5) = ((*cap * 5) as i64, 5 * idx);
        let un_bloc = |n: usize| Lancement { grille: [1, nch as u32, 1], bloc: [BLOC, 1, 1], shmem: n as u32 };
        // plaquette : somme par site, réduction par bloc, somme finale
        self.app.lancer(
            &self.noyaux.plaq_site,
            &Lancement::lineaire_lot(v, nch, BLOC),
            &[champ.arg(), Arg::I32(l as i32), tr.site.arg()],
        )?;
        self.app.lancer(
            &self.noyaux.reduce,
            &Lancement::lineaire_lot(v, nch, BLOC),
            &[tr.site.arg(), Arg::I64(v as i64), Arg::I64(v as i64), tr.part.arg(), Arg::I64(nb as i64)],
        )?;
        self.app.lancer(
            &self.noyaux.finale,
            &un_bloc(0),
            &[
                tr.part.arg(),
                Arg::I64(nb as i64),
                Arg::I64(nb as i64),
                Arg::F64(1.0 / (6.0 * v as f64)),
                hist.arg_decale(i5),
                Arg::I64(ostride),
            ],
        )?;
        // Polyakov : 4 directions
        self.app.lancer(
            &self.noyaux.polyakov,
            &Lancement::lineaire_lot(4 * l3, nch, BLOC),
            &[champ.arg(), Arg::I32(l as i32), tr.pol.arg()],
        )?;
        for mu in 0..4 {
            self.app.lancer(
                &self.noyaux.finale,
                &un_bloc(0),
                &[
                    tr.pol.arg_decale(mu * l3),
                    Arg::I64(l3 as i64),
                    Arg::I64((4 * l3) as i64),
                    Arg::F64(1.0 / l3 as f64),
                    hist.arg_decale(i5 + 1 + mu),
                    Arg::I64(ostride),
                ],
            )?;
        }
        Ok(())
    }

    fn historique(&self) -> Resultat<Vec<f64>> {
        let (hist, cap) = self.hist.as_ref().ok_or_else(|| Erreur::Config("historique non alloué".into()))?;
        self.app.synchroniser()?;
        let mut h = vec![0f64; self.nch() * cap * 5];
        self.app.lire(hist, 0, &mut h)?;
        Ok(h)
    }

    fn mesurer_tranches(&mut self, idx: usize, capacite: usize) -> Resultat<()> {
        let (l, v, nch) = (self.cfg.l, self.v(), self.nch());
        if self.tranches.is_none() {
            if capacite == 0 {
                return Err(Erreur::Config("capacité d'historique de tranches nulle".into()));
            }
            self.tranches = Some(TravailTranches {
                s: self.app.alloc::<f64>(nch * v)?,
                t: self.app.alloc::<f64>(nch * v)?,
                hist: self.app.alloc::<f64>(nch * capacite * 2 * l)?,
                capacite,
            });
        }
        let tr = self.tranches.as_ref().ok_or_else(|| Erreur::Gpu("tampons de tranches absents".into()))?;
        if idx >= tr.capacite {
            return Err(Erreur::Config(format!("historique de tranches plein : case {idx} ≥ capacité {}", tr.capacite)));
        }
        let champ = self.champ_f32()?;
        let l3 = v / l;
        let scale = 1.0 / (3.0 * l3 as f64);
        let ostride = (tr.capacite * 2 * l) as i64;
        self.app.lancer(
            &self.noyaux.plaq_st,
            &Lancement::lineaire_lot(v, nch, BLOC),
            &[champ.arg(), Arg::I32(l as i32), tr.s.arg(), tr.t.arg()],
        )?;
        let lan = Lancement { grille: [l as u32, nch as u32, 1], bloc: [BLOC, 1, 1], shmem: 0 };
        for (k, src) in [&tr.s, &tr.t].into_iter().enumerate() {
            self.app.lancer(
                &self.noyaux.tranche,
                &lan,
                &[
                    src.arg(),
                    Arg::I32(l as i32),
                    Arg::I64(v as i64),
                    Arg::F64(scale),
                    tr.hist.arg_decale(2 * l * idx + k * l),
                    Arg::I64(ostride),
                ],
            )?;
        }
        Ok(())
    }

    fn historique_tranches(&self) -> Resultat<Vec<f64>> {
        let tr = self.tranches.as_ref().ok_or_else(|| Erreur::Config("tranches non mesurées".into()))?;
        self.app.synchroniser()?;
        let mut h = vec![0f64; self.nch() * tr.capacite * 2 * self.cfg.l];
        self.app.lire(&tr.hist, 0, &mut h)?;
        Ok(h)
    }

    fn telecharger(&self, chaine: usize) -> Resultat<ChampHote> {
        if chaine >= self.nch() {
            return Err(Erreur::Config(format!("chaîne {chaine} hors du lot ({} chaînes)", self.nch())));
        }
        self.app.synchroniser()?;
        let n = 16 * self.v();
        let mut h = vec![0f32; n];
        self.app.lire(self.champ_f32()?, chaine * n, &mut h)?;
        Ok(ChampHote { l: self.cfg.l, q: promouvoir(&h) })
    }

    fn televerser(&mut self, chaine: usize, c: &ChampHote) -> Resultat<()> {
        if chaine >= self.nch() {
            return Err(Erreur::Config(format!("chaîne {chaine} hors du lot ({} chaînes)", self.nch())));
        }
        if c.l != self.cfg.l || c.q.len() != 16 * self.v() {
            return Err(Erreur::Config(format!("champ hôte L = {} ({} réels) incompatible avec L = {}", c.l, c.q.len(), self.cfg.l)));
        }
        self.app.synchroniser()?;
        let h = arrondir_normaliser(&c.q);
        let n = h.len();
        let app = self.app.clone();
        match &mut self.champ.tampon {
            TamponChamp::F32(t) => app.ecrire(t, chaine * n, &h),
            TamponChamp::F64(_) => Err(Erreur::Gpu("champ FP64 dans un générateur FP32".into())),
        }
    }

    fn champ_gpu(&self) -> Option<&ChampGpu> {
        Some(&self.champ)
    }

    fn compteurs(&self) -> Resultat<Vec<Compteurs>> {
        self.app.synchroniser()?;
        let nch = self.nch();
        let (mut cap, mut essais) = (vec![0u64; nch], vec![0u64; nch]);
        self.app.lire(&self.n_cap, 0, &mut cap)?;
        self.app.lire(&self.n_essais, 0, &mut essais)?;
        Ok(cap.into_iter().zip(essais).map(|(plafond, essais)| Compteurs { plafond, essais }).collect())
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::su2::ParamChaine;

    fn cfg(l: usize, n: usize) -> ConfigLot {
        let chaines = (0..n)
            .map(|i| ParamChaine { beta: 2.3, graine: Graine(100 + i as u64), depart: Depart::Chaud })
            .collect();
        Rapide32Gpu::config(l, chaines, 4)
    }

    #[test]
    fn config_type_est_f32_rapide() {
        let c = cfg(8, 2);
        assert_eq!(c.precision, Precision::F32);
        assert_eq!(c.options, crate::gpu::OptionsCompil::rapide());
        assert!(!c.mutant_metro);
        assert!(valider(&c).is_ok());
    }

    #[test]
    fn validation_refuse_les_configurations_invalides() {
        let mut c = cfg(8, 2);
        c.precision = Precision::F64;
        assert!(valider(&c).is_err(), "FP64 refusé");
        assert!(valider(&cfg(7, 1)).is_err(), "L impair refusé");
        assert!(valider(&cfg(0, 1)).is_err(), "L nul refusé");
        assert!(valider(&cfg(8, 0)).is_err(), "lot vide refusé");
        assert!(valider(&cfg(8, CHAINES_MAX + 1)).is_err(), "lot trop grand refusé");
        assert!(valider(&cfg(256, 1)).is_err(), "L = 256 : l'indice de lien dépasse 32 bits");
        let mut c = cfg(4, 1);
        c.chaines[0].beta = f64::NAN;
        assert!(valider(&c).is_err(), "β NaN refusé");
        c.chaines[0].beta = -1.0;
        assert!(valider(&c).is_err(), "β < 0 refusé");
        let mut c = cfg(4, 1);
        c.chaines[0].beta = 0.0;
        assert!(valider(&c).is_ok(), "β = 0 admis (mesure de Haar)");
    }

    #[test]
    fn promotion_exacte() {
        let f = [0.5f32, -1.25, 3.0e-8, f32::MIN_POSITIVE];
        let d = promouvoir(&f);
        for (a, b) in f.iter().zip(&d) {
            assert_eq!(f64::from(*a).to_bits(), b.to_bits());
            assert_eq!(*b as f32, *a, "aller-retour sans perte");
        }
    }

    #[test]
    fn arrondi_au_plus_proche_et_normalisation() {
        // quaternion unitaire FP64 : arrondi seul (norme à 1e-8 près ⇒ pas de renormalisation)
        let q = [0.5f64, 0.5, 0.5, 0.5];
        assert_eq!(arrondir_normaliser(&q), vec![0.5f32; 4]);
        // quaternion hors sphère : normalisé
        let h = arrondir_normaliser(&[2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0]);
        assert_eq!(h[..4], [1.0, 0.0, 0.0, 0.0]);
        assert!((f64::from(h[5]) - 0.6).abs() < 1e-7 && (f64::from(h[6]) - 0.8).abs() < 1e-7);
        // arrondi au plus proche (pas de troncature) : 1/3 en double → float le plus proche
        let t = 1.0f64 / 3.0;
        let n = (1.0 - t * t).sqrt();
        let h = arrondir_normaliser(&[t, n, 0.0, 0.0]);
        assert_eq!(h[0], t as f32);
    }

    #[test]
    fn arrondi_idempotent_sur_un_champ_issu_du_device() {
        // un champ « device » : quaternions unitaires à la précision du float. Le téléversement de leur
        // promotion doit rendre les mêmes bits (condition de la reprise à l'octet).
        let mut x = 12345u32;
        let mut f = Vec::new();
        for _ in 0..2000 {
            let mut q = [0f32; 4];
            for v in &mut q {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = u01f_hote(x) * 2.0 - 1.0;
            }
            let n = q.iter().map(|v| v * v).sum::<f32>().sqrt();
            f.extend(q.iter().map(|v| v / n));
        }
        let aller_retour = arrondir_normaliser(&promouvoir(&f));
        assert_eq!(f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), aller_retour.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }

    #[test]
    fn u01f_dans_ouvert_01_pour_tous_les_mots_extremes() {
        for a in [0u32, 1, 0x1FF, 0x200, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FDFF, 0xFFFF_FE00, 0xFFFF_FFFF] {
            let u = u01f_hote(a);
            assert!(u > 0.0 && u < 1.0, "u01f({a:#x}) = {u}");
        }
        assert_eq!(u01f_hote(0), 2f32.powi(-24));
        assert_eq!(u01f_hote(u32::MAX), 1.0 - 2f32.powi(-24));
    }

    #[test]
    fn la_forme_24_bits_d_origine_atteint_1() {
        // c'est la raison de l'écart : ((2²⁴−1) + 0,5) n'est pas représentable en float, il est arrondi à 2²⁴.
        assert_eq!(u01f_24bits_refusee(u32::MAX), 1.0);
        assert_eq!(u01f_24bits_refusee(0), 2f32.powi(-25));
    }

    #[test]
    fn u01f_uniforme_sur_cellules_regulieres() {
        // 2²³ cellules égales : le milieu des cellules est exact, donc la moyenne sur un échantillon dense est 1/2
        let n = 1 << 16;
        let m: f64 = (0..n).map(|i| f64::from(u01f_hote((i as u32) << 9))).sum::<f64>() / n as f64;
        // valeurs échantillonnées tous les 2⁹ mots de 32 bits ⇒ cellules 0, 1, 2, … 65535 sur 2²³
        assert!((m - (65_535.0 / 2.0 + 0.5) / 8_388_608.0).abs() < 1e-9);
    }

    #[test]
    fn miroir_hote_tirage_dans_les_bornes_et_deterministe() {
        for &alpha in &[0.0f32, 1e-7, 0.3, 1.0, 1.999, 2.0, 5.0, 100.0, 400.0] {
            for li in 0..200u32 {
                let (mut d1, mut d2) = (0u32, 0u32);
                let (x, ntry, ok) = tirage_x0_hote(alpha, li, 7, &mut d1, 11, 22);
                let (y, ntry2, ok2) = tirage_x0_hote(alpha, li, 7, &mut d2, 11, 22);
                assert!(ok && ok2);
                assert!((-1.0..=1.0).contains(&x), "x0 = {x} hors [-1,1] à α = {alpha}");
                assert_eq!((x.to_bits(), ntry, d1), (y.to_bits(), ntry2, d2));
                assert_eq!(ntry, d1, "un appel Philox par essai");
            }
        }
    }

    #[test]
    fn source_sans_intrinseques_rapides_ni_double_dans_les_chemins_chauds() {
        let sans_commentaires: String = SU2_F32
            .contenu
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        for interdit in ["__logf", "__expf", "__sinf", "__cosf", "__tanf", "__powf", "__fdividef", "__fsqrt", "use_fast_math"] {
            assert!(!sans_commentaires.contains(interdit), "intrinsèque interdit : {interdit}");
        }
        // chemins chauds : du tirage de x0 à l'overrelaxation (avant la section des mesures)
        let debut = sans_commentaires.find("struct __attribute__((aligned(16))) QF").expect("QF");
        let fin = sans_commentaires.find("k_plaq_site_f32").expect("mesures");
        let chaud = &sans_commentaires[debut..fin];
        // les mesures exceptées, aucune fonction FP64 : ni `double`, ni log/cos/sin/exp/sqrt sans suffixe f
        for interdit in ["double", " log(", "(log(", " cos(", " sin(", " exp(", " sqrt(", " fabs(", " fmax(", " fmin(", "M_PI"] {
            // `qdbl` (promotion pour les mesures) est la seule fonction autorisée à nommer Q/double
            let sans_qdbl = chaud.replace("Q qdbl(QF a) { Q c = {(double)a.x, (double)a.y, (double)a.z, (double)a.w}; return c; }", "");
            assert!(!sans_qdbl.contains(interdit), "FP64 ou fonction non-f dans un chemin chaud : {interdit}");
        }
    }

    #[test]
    fn source_declare_tous_les_noyaux_attendus() {
        for nom in [
            "k_init_f32",
            "k_heatbath_f32",
            "k_overrelax_f32",
            "k_plaq_site_f32",
            "k_polyakov_f32",
            "k_plaq_st_f32",
            "k_tranche_f32",
            "k_reduce_f32",
            "k_final_f32",
            "k_test_x0_f32",
        ] {
            assert!(SU2_F32.contenu.contains(&format!("void {nom}(")), "noyau absent : {nom}");
            assert!(SU2_F32.contenu.contains(&format!("extern \"C\" __global__ void {nom}(")), "{nom} doit être extern \"C\"");
        }
        assert!(SU2_F32.contenu.contains("#ifdef K1_MUTANT_METRO"), "mutant de contrôle négatif");
    }
}
