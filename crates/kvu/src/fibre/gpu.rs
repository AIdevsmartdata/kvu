//! Chemin GPU (HIP ou CUDA) de la mesure de fibre : noyaux de `noyaux/fibre.cu`.
//!
//! Tampons alloués UNE fois par L (tables de Wf, de plaquettes, supports, pv, gradients, partiels de D).
//! Le champ F64 ou F32 résident d'un `ChampGpu` est lu sans copie (F32 promu en double dans le noyau) ;
//! à défaut, un champ hôte est téléversé dans un tampon interne.
//!
//! Compilation : `OptionsCompil::reference_k1()` (aucune contraction FMA), de sorte que le résultat est
//! identique à l'octet à la référence CPU (`cpu.rs`) : gate FIB-3.

use super::cpu::{pointeurs_segments, SEGMENT_F, VOIES_F};
use super::plan::{ModeFibre, PlanFibre, TRANCHE};
use super::poids::PoidsFourier;
use crate::commun::{ChampHote, Erreur, Precision, Resultat};
use crate::gpu::{assembler, Appareil, Arg, Fonction, Lancement, Module, OptionsCompil, Pod, ProvenanceCompil, Tampon};
use crate::noyaux::{FIBRE, SU2_COMMUN};
use crate::su2::{ChampGpu, TamponChamp};

/// Largeur de la matrice D que le noyau sait traiter (FB_NF) et colonnes maximales d'une ligne (FB_NSMAX).
pub const NF_GPU: usize = 54;
pub const NSMAX_GPU: usize = 27;
/// Fils par bloc de `k_D_partiel` (FB_BLOC).
const BLOC_D: u32 = 256;
/// Partiels de D résidents au maximum (≈ 65 Mo) : au-delà, la somme se fait en plusieurs lancements, dans le
/// même ordre croissant des tranches.
const LOT_DEFAUT: usize = 2800;

/// Fonctions d'un module compilé pour un type de champ.
struct Noyaux {
    _module: Module,
    plaq: Fonction,
    f_seg: Fonction,
    f_fin: Fonction,
    d_partiel: Fonction,
    d_final: Fonction,
    blk: Fonction,
}

/// Chronomètre par étape (actif avec `KVU_FIBRE_PROFIL=1`) : chaque marque synchronise le GPU puis note la durée
/// écoulée depuis la marque précédente. Inactif, il ne coûte rien (aucune synchronisation ajoutée).
struct Chrono {
    actif: bool,
    t: std::time::Instant,
    etapes: Vec<(String, f64)>,
}

impl Chrono {
    fn nouveau(actif: bool) -> Chrono {
        Chrono {
            actif,
            t: std::time::Instant::now(),
            etapes: Vec::new(),
        }
    }
    fn marque(&mut self, app: &Appareil, nom: &str) -> Resultat<()> {
        if self.actif {
            app.synchroniser()?;
            self.etapes
                .push((nom.to_string(), self.t.elapsed().as_secs_f64() * 1e3));
            self.t = std::time::Instant::now();
        }
        Ok(())
    }
}

/// Sortie brute d'une mesure GPU (mêmes champs que la CPU).
pub(crate) struct SortieGpu {
    pub f: Vec<f64>,
    pub d: Option<Vec<f64>>,
    pub blk: Option<Vec<f32>>,
}

pub(crate) struct MoteurGpu {
    app: Appareil,
    noyaux_f64: Option<Noyaux>,
    noyaux_f32: Option<Noyaux>,
    // tables (téléversées une fois)
    lp: Tampon<u32>,
    wt_ptr: Tampon<u32>,
    wt_idx: Tampon<u8>,
    wt_val: Tampon<f64>,
    f_ptr: Tampon<u64>,
    f_idx: Tampon<u32>,
    f_val: Tampon<f64>,
    seg_ptr: Tampon<u32>,
    seg_s: Tampon<f64>,
    seg_c: Tampon<f64>,
    n_seg: usize,
    lfree: Tampon<u32>,
    e1: Tampon<u32>,
    e2: Tampon<u32>,
    sup: Tampon<u8>,
    cmap: Tampon<u8>,
    // tampons de travail
    pv: Tampon<f64>,
    gp: Tampon<f64>,
    f_sortie: Tampon<f64>,
    d_sortie: Tampon<f64>,
    blk_sortie: Tampon<f32>,
    partiels: Tampon<f64>,
    champ_hote: Option<Tampon<f64>>,
    lot: usize,
    profil_actif: bool,
    profil: Vec<(String, f64)>,
    nf: usize,
    l: usize,
    nfree_mu: usize,
    npair_mu: usize,
    /// Provenance des modules compilés (F64, F32), pour le manifeste des sorties.
    provenances: Vec<ProvenanceCompil>,
}

/// Supports et cartes de colonnes au format du noyau : `sup` = (4, 1 + NSMAX) octets, [mu][0] = nombre de
/// colonnes puis les lignes de Wf du support ; `cmap` = (4, 256) octets, position dans le support ou 255.
pub(crate) fn empaqueter_supports(plan: &PlanFibre) -> (Vec<u8>, Vec<u8>) {
    let mut sup = vec![0u8; 4 * (1 + NSMAX_GPU)];
    let mut cmap = vec![255u8; 4 * 256];
    for mu in 0..4 {
        sup[mu * (1 + NSMAX_GPU)] = plan.sup[mu].len() as u8;
        for (i, &r) in plan.sup[mu].iter().enumerate() {
            sup[mu * (1 + NSMAX_GPU) + 1 + i] = r;
        }
        for (r, &c) in plan.cmap[mu].iter().enumerate() {
            cmap[mu * 256 + r] = c;
        }
    }
    (sup, cmap)
}

/// Découpe des `n_tr` tranches en lancements de `lot` tranches au plus : (première tranche, nombre), dans
/// l'ordre croissant. Le résultat de D ne dépend pas de ce découpage (somme séquentielle dans l'ordre).
pub(crate) fn lancements(n_tr: usize, lot: usize) -> Vec<(usize, usize)> {
    let lot = lot.max(1);
    (0..n_tr).step_by(lot).map(|t0| (t0, lot.min(n_tr - t0))).collect()
}

/// Alloue un tampon et y copie `d`.
fn televerser<T: Pod>(app: &Appareil, d: &[T]) -> Resultat<Tampon<T>> {
    let mut t = app.alloc::<T>(d.len())?;
    app.ecrire(&mut t, 0, d)?;
    Ok(t)
}

/// Mémoire GPU approximative d'un mesureur (octets) : information pour l'utilisateur et les bancs. Aucun refus
/// anticipé (sur une APU à mémoire unifiée, `memoire_libre_octets` ne voit pas la GTT) : une allocation impossible
/// remonte en `Erreur::Gpu`.
pub fn octets_estimes(v: usize, nnz: usize) -> u64 {
    let v = v as u64;
    let nnz = nnz as u64;
    96 * v                      // lp
        + 24 * v                // wt_ptr
        + nnz * (1 + 8 + 4 + 8) // wt_idx, wt_val, f_idx, f_val
        + 48 * v                // pv
        + 576 * v               // gp
        + 128 * v               // champ hôte téléversé
        + (LOT_DEFAUT * NF_GPU * NF_GPU * 8) as u64
}

impl MoteurGpu {
    pub fn nouveau(app: &Appareil, plan: &PlanFibre, w: &PoidsFourier) -> Resultat<MoteurGpu> {
        if plan.nf != NF_GPU {
            return Err(Erreur::Config(format!(
                "fibre GPU : {} lignes de Wf, le noyau traite {NF_GPU} (pmax² = 1)",
                plan.nf
            )));
        }
        if plan.support_max() > NSMAX_GPU {
            return Err(Erreur::Config(format!(
                "fibre GPU : support de {} colonnes, le noyau en admet {NSMAX_GPU}",
                plan.support_max()
            )));
        }
        let v = plan.v;
        let (sup, cmap) = empaqueter_supports(plan);
        let lp = televerser(app, &plan.lp)?;
        let wt_ptr = televerser(app, &plan.wt_ptr)?;
        let wt_idx = televerser(app, &plan.wt_idx)?;
        let wt_val = televerser(app, &plan.wt_val)?;
        let f_ptr = televerser(app, &w.ptr)?;
        let f_idx = televerser(app, &w.idx)?;
        let f_val = televerser(app, &w.val)?;
        let sp = pointeurs_segments(&w.ptr, SEGMENT_F);
        let n_seg = sp[w.nf] as usize;
        let seg_ptr = televerser(app, &sp)?;
        let lfree = televerser(app, &plan.lfree)?;
        let e1 = televerser(app, &plan.e1)?;
        let e2 = televerser(app, &plan.e2)?;
        let sup_t = televerser(app, &sup)?;
        let cmap_t = televerser(app, &cmap)?;
        Ok(MoteurGpu {
            app: app.clone(),
            noyaux_f64: None,
            noyaux_f32: None,
            lp,
            wt_ptr,
            wt_idx,
            wt_val,
            f_ptr,
            f_idx,
            f_val,
            seg_ptr,
            seg_s: app.alloc::<f64>(n_seg)?,
            seg_c: app.alloc::<f64>(n_seg)?,
            n_seg,
            lfree,
            e1,
            e2,
            sup: sup_t,
            cmap: cmap_t,
            pv: app.alloc::<f64>(6 * v)?,
            gp: app.alloc::<f64>(6 * v * 12)?,
            f_sortie: app.alloc::<f64>(NF_GPU)?,
            d_sortie: app.alloc::<f64>(NF_GPU * NF_GPU)?,
            blk_sortie: app.alloc::<f32>(6 * (plan.l / 2).pow(4))?,
            partiels: app.alloc::<f64>(LOT_DEFAUT * NF_GPU * NF_GPU)?,
            champ_hote: None,
            lot: LOT_DEFAUT,
            profil_actif: std::env::var("KVU_FIBRE_PROFIL").is_ok_and(|v| v == "1"),
            profil: Vec::new(),
            nf: plan.nf,
            l: plan.l,
            nfree_mu: plan.nfree_mu,
            npair_mu: plan.npair_mu,
            provenances: Vec::new(),
        })
    }

    /// Provenance des modules compilés jusqu'ici.
    pub fn provenances(&self) -> &[ProvenanceCompil] {
        &self.provenances
    }

    /// Durées (ms) des étapes de la dernière mesure, si `KVU_FIBRE_PROFIL=1` (sinon vide).
    pub fn dernier_profil(&self) -> Vec<(String, f64)> {
        self.profil.clone()
    }

    /// Tranches par lancement de `k_D_partiel` (≥ 1). Réalloue le tampon des partiels si besoin.
    pub fn fixer_lot(&mut self, n: usize) -> Resultat<()> {
        if n == 0 {
            return Err(Erreur::Config(
                "fibre : un lot d'au moins une tranche est requis".into(),
            ));
        }
        if n != self.lot {
            self.partiels = self.app.alloc::<f64>(n * NF_GPU * NF_GPU)?;
            self.lot = n;
        }
        Ok(())
    }

    fn compiler(&mut self, p: Precision) -> Resultat<()> {
        let slot = match p {
            Precision::F64 => &mut self.noyaux_f64,
            Precision::F32 => &mut self.noyaux_f32,
        };
        if slot.is_some() {
            return Ok(());
        }
        let defines: &[(&str, &str)] = match p {
            Precision::F64 => &[],
            Precision::F32 => &[("FB_CHAMP_F32", "1")],
        };
        let nom = match p {
            Precision::F64 => "fibre_f64",
            Precision::F32 => "fibre_f32",
        };
        let src = assembler(self.app.backend(), nom, &[&SU2_COMMUN, &FIBRE], defines);
        let module = self.app.compiler(&src, &OptionsCompil::reference_k1())?;
        self.provenances.push(module.provenance().clone());
        let slot = match p {
            Precision::F64 => &mut self.noyaux_f64,
            Precision::F32 => &mut self.noyaux_f32,
        };
        *slot = Some(Noyaux {
            plaq: self.app.fonction(&module, "k_plaq_grad")?,
            f_seg: self.app.fonction(&module, "k_F_seg")?,
            f_fin: self.app.fonction(&module, "k_F_fin")?,
            d_partiel: self.app.fonction(&module, "k_D_partiel")?,
            d_final: self.app.fonction(&module, "k_D_final")?,
            blk: self.app.fonction(&module, "k_blk")?,
            _module: module,
        });
        Ok(())
    }

    /// Champ d'entrée : résident (chaîne c, précision du champ) ou hôte téléversé (F64).
    fn entree(
        &mut self,
        plan: &PlanFibre,
        champ: Option<&ChampGpu>,
        c: usize,
        hote: Option<&ChampHote>,
    ) -> Resultat<(Precision, Arg)> {
        let (l, v) = (self.l, plan.v);
        match (champ, hote) {
            (Some(ch), _) => {
                if ch.l != l {
                    return Err(Erreur::Config(format!(
                        "fibre : champ GPU L = {} pour un mesureur L = {l}",
                        ch.l
                    )));
                }
                if c >= ch.nchaines {
                    return Err(Erreur::Config(format!(
                        "fibre : chaîne {c} ≥ {} chaînes du champ",
                        ch.nchaines
                    )));
                }
                Ok(match &ch.tampon {
                    TamponChamp::F64(t) => (Precision::F64, t.arg_decale(c * 16 * v)),
                    TamponChamp::F32(t) => (Precision::F32, t.arg_decale(c * 16 * v)),
                })
            }
            (None, Some(h)) => {
                if h.l != l || h.q.len() != 16 * v {
                    return Err(Erreur::Config(format!(
                        "fibre : champ hôte L = {} pour un mesureur L = {l}",
                        h.l
                    )));
                }
                if self.champ_hote.is_none() {
                    self.champ_hote = Some(self.app.alloc::<f64>(16 * v)?);
                }
                let t = self.champ_hote.as_mut().expect("tampon créé ci-dessus");
                self.app.ecrire(t, 0, &h.q)?;
                Ok((Precision::F64, t.arg()))
            }
            (None, None) => Err(Erreur::Config("fibre : ni champ GPU ni champ hôte fourni".into())),
        }
    }

    /// Plaquettes seules (6V), sans les gradients (lecture tordue de F à chaque mesure : 6V réels rapatriés au lieu
    /// de 78V).
    pub fn valeurs_plaquettes(
        &mut self,
        plan: &PlanFibre,
        champ: Option<&ChampGpu>,
        c: usize,
        hote: Option<&ChampHote>,
    ) -> Resultat<Vec<f64>> {
        let (prec, arg_champ) = self.entree(plan, champ, c, hote)?;
        self.compiler(prec)?;
        let k = match prec {
            Precision::F64 => self.noyaux_f64.as_ref(),
            Precision::F32 => self.noyaux_f32.as_ref(),
        }
        .expect("compilé ci-dessus");
        let v = plan.v;
        self.app.lancer(
            &k.plaq,
            &Lancement::lineaire(6 * v, 128),
            &[arg_champ, Arg::I32(self.l as i32), self.pv.arg(), Arg::Nul],
        )?;
        self.app.synchroniser()?;
        let mut pv = vec![0.0; 6 * v];
        self.app.lire(&self.pv, 0, &mut pv)?;
        Ok(pv)
    }

    /// Plaquettes et gradients tels que le noyau les calcule (lecture complète : usage de validation).
    pub fn plaquettes(
        &mut self,
        plan: &PlanFibre,
        champ: Option<&ChampGpu>,
        c: usize,
        hote: Option<&ChampHote>,
    ) -> Resultat<(Vec<f64>, Vec<f64>)> {
        let (prec, arg_champ) = self.entree(plan, champ, c, hote)?;
        self.compiler(prec)?;
        let k = match prec {
            Precision::F64 => self.noyaux_f64.as_ref(),
            Precision::F32 => self.noyaux_f32.as_ref(),
        }
        .expect("compilé ci-dessus");
        let v = plan.v;
        self.app.lancer(
            &k.plaq,
            &Lancement::lineaire(6 * v, 128),
            &[arg_champ, Arg::I32(self.l as i32), self.pv.arg(), self.gp.arg()],
        )?;
        self.app.synchroniser()?;
        let mut pv = vec![0.0; 6 * v];
        let mut gp = vec![0.0; 72 * v];
        self.app.lire(&self.pv, 0, &mut pv)?;
        self.app.lire(&self.gp, 0, &mut gp)?;
        Ok((pv, gp))
    }

    /// Une mesure : plaquettes et gradients, F, puis (selon `mode`) D, et blk si demandé.
    pub fn mesurer(
        &mut self,
        plan: &PlanFibre,
        champ: Option<&ChampGpu>,
        c: usize,
        hote: Option<&ChampHote>,
        mode: Option<ModeFibre>,
        avec_blk: bool,
    ) -> Resultat<SortieGpu> {
        let l = self.l;
        let v = plan.v;
        let mut chrono = Chrono::nouveau(self.profil_actif);
        let (prec, arg_champ) = self.entree(plan, champ, c, hote)?;
        chrono.marque(&self.app, "champ (téléversement)")?;
        self.compiler(prec)?;
        let k = match prec {
            Precision::F64 => self.noyaux_f64.as_ref(),
            Precision::F32 => self.noyaux_f32.as_ref(),
        }
        .expect("compilé ci-dessus");
        let app = &self.app;
        chrono.marque(app, "compilation")?;

        // Gradients seulement si D est demandé : F seul se contente des valeurs (gp nul).
        let arg_gp = if mode.is_some() { self.gp.arg() } else { Arg::Nul };
        app.lancer(
            &k.plaq,
            &Lancement::lineaire(6 * v, 128),
            &[arg_champ, Arg::I32(l as i32), self.pv.arg(), arg_gp],
        )?;
        chrono.marque(app, "k_plaq_grad")?;
        app.lancer(
            &k.f_seg,
            &Lancement {
                grille: [self.n_seg as u32, 1, 1],
                bloc: [VOIES_F as u32, 1, 1],
                shmem: 0,
            },
            &[
                self.f_ptr.arg(),
                self.f_idx.arg(),
                self.f_val.arg(),
                self.pv.arg(),
                self.seg_ptr.arg(),
                Arg::I32(self.nf as i32),
                Arg::I32(SEGMENT_F as i32),
                self.seg_s.arg(),
                self.seg_c.arg(),
            ],
        )?;
        chrono.marque(app, "k_F_seg")?;
        app.lancer(
            &k.f_fin,
            &Lancement::lineaire(self.nf, 64),
            &[
                self.seg_ptr.arg(),
                self.seg_s.arg(),
                self.seg_c.arg(),
                Arg::I32(self.nf as i32),
                self.f_sortie.arg(),
            ],
        )?;
        chrono.marque(app, "k_F_fin")?;
        if avec_blk {
            let n = 6 * (l / 2).pow(4);
            app.lancer(
                &k.blk,
                &Lancement::lineaire(n, 128),
                &[self.pv.arg(), Arg::I32(l as i32), self.blk_sortie.arg()],
            )?;
            chrono.marque(app, "k_blk")?;
        }
        if let Some(m) = mode {
            app.zero(&mut self.d_sortie)?;
            let (libre, n_tr) = (i32::from(m == ModeFibre::Libre), plan.tranches(m));
            let tr_par_mu = plan.tranches_mu(m);
            for (tr0, n) in lancements(n_tr, self.lot) {
                let cfg = Lancement {
                    grille: [n as u32, 1, 1],
                    bloc: [BLOC_D, 1, 1],
                    shmem: 0,
                };
                app.lancer(
                    &k.d_partiel,
                    &cfg,
                    &[
                        arg_champ,
                        self.gp.arg(),
                        self.lp.arg(),
                        self.wt_ptr.arg(),
                        self.wt_idx.arg(),
                        self.wt_val.arg(),
                        self.lfree.arg(),
                        self.e1.arg(),
                        self.e2.arg(),
                        self.sup.arg(),
                        self.cmap.arg(),
                        Arg::I32(l as i32),
                        Arg::I32(libre),
                        Arg::I32(self.nfree_mu as i32),
                        Arg::I32(self.npair_mu as i32),
                        Arg::I32(TRANCHE as i32),
                        Arg::I32(tr_par_mu as i32),
                        Arg::I32(tr0 as i32),
                        self.partiels.arg(),
                    ],
                )?;
                app.lancer(
                    &k.d_final,
                    &Lancement::lineaire(NF_GPU * NF_GPU, 128),
                    &[self.partiels.arg(), Arg::I32(n as i32), self.d_sortie.arg()],
                )?;
            }
            chrono.marque(app, "k_D_partiel + k_D_final")?;
        }
        app.synchroniser()?;
        let mut f = vec![0.0; self.nf];
        app.lire(&self.f_sortie, 0, &mut f)?;
        let d = if mode.is_some() {
            let mut d = vec![0.0; NF_GPU * NF_GPU];
            app.lire(&self.d_sortie, 0, &mut d)?;
            Some(d)
        } else {
            None
        };
        let blk = if avec_blk {
            let mut b = vec![0f32; 6 * (l / 2).pow(4)];
            app.lire(&self.blk_sortie, 0, &mut b)?;
            Some(b)
        } else {
            None
        };
        chrono.marque(app, "lectures")?;
        self.profil = chrono.etapes;
        Ok(SortieGpu { f, d, blk })
    }
}

#[cfg(test)]
mod tests {
    use super::super::poids::poids_fourier;
    use super::*;
    use crate::cond::GeometrieCond;

    #[test]
    fn supports_empaquetes_comme_le_noyau_les_lit() {
        let plan = PlanFibre::nouveau(&GeometrieCond::nouvelle(4).unwrap(), &poids_fourier(4, 1)).unwrap();
        let (sup, cmap) = empaqueter_supports(&plan);
        assert_eq!(sup.len(), 4 * 28);
        assert_eq!(cmap.len(), 4 * 256);
        for mu in 0..4 {
            let ns = sup[mu * 28] as usize;
            assert_eq!(ns, 27);
            for i in 0..ns {
                let r = sup[mu * 28 + 1 + i] as usize;
                assert_eq!(cmap[mu * 256 + r] as usize, i, "cmap ∘ sup = identité");
            }
            // lignes hors support : 255 ; au-delà de nf : 255
            assert_eq!(cmap[mu * 256 + 255], 255);
            assert_eq!(cmap[mu * 256..(mu + 1) * 256].iter().filter(|&&c| c != 255).count(), 27);
        }
    }

    #[test]
    fn lancements_couvrent_les_tranches_dans_l_ordre() {
        for (n, lot) in [(10, 3), (10, 10), (10, 1), (7, 2800), (1, 1), (4, 3)] {
            let l = lancements(n, lot);
            let mut suivant = 0;
            for &(t0, k) in &l {
                assert_eq!(t0, suivant);
                assert!(k >= 1 && k <= lot);
                suivant += k;
            }
            assert_eq!(suivant, n);
        }
        assert_eq!(lancements(10, 3), vec![(0, 3), (3, 3), (6, 3), (9, 1)]);
        assert_eq!(lancements(5, 0), lancements(5, 1), "lot nul traité comme 1");
    }

    #[test]
    fn memoire_estimee_croit_comme_v() {
        let (a, b) = (
            octets_estimes(65536, 54 * 65536),
            octets_estimes(1_048_576, 54 * 1_048_576),
        );
        assert!(b > 15 * a / 2, "≈ proportionnel à V (hors partiels fixes)");
        // L = 32 : de l'ordre de 2 Go
        assert!(b > 1_500_000_000 && b < 3_500_000_000, "{b}");
    }
}
