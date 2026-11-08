//! Moteur GPU de la fibre dure de DEC de facteur 3 (SU(2), d = 4, HIP ou CUDA, source unique `noyaux/cond3.cu`) : balayages (bain thermique et
//! surrelaxation à agrafes effectives), observables des faces, scores des sondes et accumulation par lots des sommes Σ O σ sur le périphérique.
//!
//! Le CPU (`cond3::su2`) reste la référence : un miroir `FibreSu2` sert aux contrôles (contrainte, unitarité, Ward) sur le champ téléchargé tous
//! les `controle_chaque` balayages, et à l'empreinte finale. Les noyaux évitent la contraction FMA (`OptionsCompil::reference_k1`) : leur ordre
//! d'évaluation est celui du CPU, à l'arrondi des agrafes près (le sens des produits d'agrafes diffère : concordance par balayage ⩽ 1e-12, pas au bit
//! entre CPU et GPU ; rejeu au bit GPU contre GPU).

use super::geom::Geom;
use super::mesure::{Controle, ConfigMesure, SortieReplique};
use super::modele::{Canal, Modele, StatsEchantillonneur};
use super::paires::Paires;
use super::su2::FibreSu2;
use crate::commun::{Erreur, Graine, Resultat};
use crate::gpu::{assembler, Appareil, Arg, Fonction, Lancement, OptionsCompil, Tampon};
use crate::heritage::su2_hb::Q;
use crate::noyaux::{Fichier, COND, SU2_COMMUN};
use std::sync::Arc;
use std::time::Instant;

static COND3: Fichier = Fichier { nom: "cond3.cu", contenu: include_str!("../../noyaux/cond3.cu") };

const BLOC: u32 = 256;

struct Fonctions {
    libres_hb: Fonction,
    libres_or: Fonction,
    lignes: Fonction,
    obs: Fonction,
    scores: Fonction,
    acc: Fonction,
    sommes_o: Fonction,
    sommes_s: Fonction,
    serie_o: Fonction,
}

struct LignesDir {
    s1: Tampon<u32>,
    s2: Tampon<u32>,
    s3: Tampon<u32>,
    v: Tampon<[f64; 4]>,
    n: usize,
}

pub struct FibreSu2Gpu {
    app: Appareil,
    f: Fonctions,
    g: Arc<Geom>,
    beta: f64,
    k0: u32,
    k1: u32,
    u: Tampon<[f64; 4]>,
    diag: Tampon<u32>,
    ncap: Tampon<u64>,
    libres: Vec<Tampon<u32>>,
    nlibres: Vec<usize>,
    lignes: Vec<LignesDir>,
    /// Miroir CPU (champ téléchargé, V figé) : contrôles et empreinte.
    pub miroir: FibreSu2,
    pub provenance_sha256_code: String,
    pub arch: String,
    pub backend: crate::commun::BackendKind,
}

fn tv_u32(app: &Appareil, t: &[u32]) -> Resultat<Tampon<u32>> {
    let mut d = app.alloc::<u32>(t.len().max(1))?;
    if !t.is_empty() {
        app.ecrire(&mut d, 0, t)?;
    }
    Ok(d)
}

impl FibreSu2Gpu {
    /// Construit le moteur à partir d'un modèle CPU déjà au départ exact (champ et V figé) ; la graine Philox est reprise de `graine`.
    pub fn depuis_cpu(app: &Appareil, cpu: FibreSu2, graine: u64) -> Resultat<FibreSu2Gpu> {
        let g = cpu.geom().clone();
        if g.d != 4 {
            return Err(Erreur::Config("cond3 GPU : d = 4 seulement".into()));
        }
        let backend = app.backend();
        let src = assembler(backend, "cond3", &[&SU2_COMMUN, &COND, &COND3], &[]);
        let module = app.compiler(&src, &OptionsCompil::reference_k1())?;
        let f = |n: &str| app.fonction(&module, n);
        let fonctions = Fonctions {
            libres_hb: f("k_t3_libres_hb")?,
            libres_or: f("k_t3_libres_or")?,
            lignes: f("k_t3_lignes")?,
            obs: f("k_t3_obs")?,
            scores: f("k_t3_scores")?,
            acc: f("k_t3_acc")?,
            sommes_o: f("k_t3_sommes_o")?,
            sommes_s: f("k_t3_sommes_s")?,
            serie_o: f("k_t3_serie_o")?,
        };
        let mut u = app.alloc::<[f64; 4]>(cpu.u.len())?;
        app.ecrire(&mut u, 0, &cpu.u)?;
        let mut libres = Vec::new();
        let mut nlibres = Vec::new();
        for mu in 0..4 {
            for p in 0..2 {
                libres.push(tv_u32(app, &g.libres[mu][p])?);
                nlibres.push(g.libres[mu][p].len());
            }
        }
        let mut lignes = Vec::new();
        for mu in 0..4 {
            let cis = &g.lignes_dir[mu];
            let s1: Vec<u32> = cis.iter().map(|&ci| g.lignes[ci as usize][0] % g.v as u32).collect();
            let s2: Vec<u32> = cis.iter().map(|&ci| g.lignes[ci as usize][1] % g.v as u32).collect();
            let s3: Vec<u32> = cis.iter().map(|&ci| g.lignes[ci as usize][2] % g.v as u32).collect();
            let vl: Vec<[f64; 4]> = cis.iter().map(|&ci| cpu.vfix[ci as usize]).collect();
            let mut dv = app.alloc::<[f64; 4]>(vl.len().max(1))?;
            app.ecrire(&mut dv, 0, &vl)?;
            lignes.push(LignesDir { s1: tv_u32(app, &s1)?, s2: tv_u32(app, &s2)?, s3: tv_u32(app, &s3)?, v: dv, n: cis.len() });
        }
        let (k0, k1) = Graine(graine).cle();
        let mut miroir = cpu;
        miroir.set_graine(graine);
        let mut diag = app.alloc::<u32>(g.nliens())?;
        app.zero(&mut diag)?;
        let mut ncap = app.alloc::<u64>(1)?;
        app.zero(&mut ncap)?;
        Ok(FibreSu2Gpu {
            app: app.clone(),
            f: fonctions,
            beta: miroir.beta(),
            g: g.clone(),
            k0,
            k1,
            u,
            diag,
            ncap,
            libres,
            nlibres,
            lignes,
            provenance_sha256_code: module.sha256_code.clone(),
            arch: app.arch(),
            backend,
            miroir,
        })
    }

    fn lancer_libres(&self, hb: bool, mu: usize, par: usize, sweep: u32) -> Resultat<()> {
        let k = 2 * mu + par;
        let n = self.nlibres[k];
        let cfg = Lancement::lineaire(n, BLOC);
        if hb {
            self.app.lancer(
                &self.f.libres_hb,
                &cfg,
                &[
                    self.u.arg(),
                    Arg::I32(self.g.l as i32),
                    Arg::I32(mu as i32),
                    self.libres[k].arg(),
                    Arg::F64(self.beta),
                    Arg::U32(self.k0),
                    Arg::U32(self.k1),
                    Arg::U32(sweep),
                    self.diag.arg(),
                    self.ncap.arg(),
                    Arg::I64(n as i64),
                ],
            )
        } else {
            self.app.lancer(
                &self.f.libres_or,
                &cfg,
                &[self.u.arg(), Arg::I32(self.g.l as i32), Arg::I32(mu as i32), self.libres[k].arg(), Arg::I64(n as i64)],
            )
        }
    }

    fn lancer_lignes(&self, hb: bool, mu: usize, quelle: i32, sweep: u32) -> Resultat<()> {
        let ld = &self.lignes[mu];
        let cfg = Lancement::lineaire(ld.n, BLOC);
        self.app.lancer(
            &self.f.lignes,
            &cfg,
            &[
                self.u.arg(),
                Arg::I32(self.g.l as i32),
                Arg::I32(mu as i32),
                ld.s1.arg(),
                ld.s2.arg(),
                ld.s3.arg(),
                ld.v.arg(),
                Arg::I32(quelle),
                Arg::I32(hb as i32),
                Arg::F64(self.beta),
                Arg::U32(self.k0),
                Arg::U32(self.k1),
                Arg::U32(sweep),
                self.diag.arg(),
                self.ncap.arg(),
                Arg::I64(ld.n as i64),
            ],
        )
    }

    /// Un pas contraint (HB + n_or OR) : mêmes sous-étapes et même ordre que le CPU (SPEC §3.1).
    pub fn pas(&mut self, sweep: u32, n_or: usize) -> Resultat<()> {
        for mu in 0..4 {
            self.lancer_libres(true, mu, 0, sweep)?;
            self.lancer_libres(true, mu, 1, sweep)?;
            self.lancer_lignes(true, mu, 1, sweep)?;
            self.lancer_lignes(true, mu, 2, sweep)?;
        }
        for _ in 0..n_or {
            for mu in 0..4 {
                self.lancer_libres(false, mu, 0, sweep)?;
                self.lancer_libres(false, mu, 1, sweep)?;
                self.lancer_lignes(false, mu, 1, sweep)?;
                self.lancer_lignes(false, mu, 2, sweep)?;
            }
        }
        Ok(())
    }

    /// Télécharge le champ dans le miroir CPU.
    pub fn telecharger(&mut self) -> Resultat<()> {
        self.app.synchroniser()?;
        let mut v = vec![[0.0f64; 4]; self.u.len()];
        self.app.lire(&self.u, 0, &mut v)?;
        self.miroir.u = v;
        Ok(())
    }

    /// Nombre de tirages ayant atteint le plafond d'essais.
    pub fn plafond(&self) -> Resultat<u64> {
        self.app.synchroniser()?;
        let mut v = [0u64; 1];
        self.app.lire(&self.ncap, 0, &mut v)?;
        Ok(v[0])
    }

    /// (mises à jour, essais) de la dernière HB d'après le tampon de diagnostic.
    pub fn essais_dernier_balayage(&self) -> Resultat<(u64, u64)> {
        self.app.synchroniser()?;
        let mut d = vec![0u32; self.u.len()];
        self.app.lire(&self.diag, 0, &mut d)?;
        let majs = d.iter().filter(|&&x| x > 0).count() as u64;
        let essais: u64 = d.iter().map(|&x| x as u64).sum();
        Ok((majs, essais))
    }
}

/// Réplique sur le GPU : même contrat et mêmes sorties que `mesure::executer_replique`.
pub fn executer_replique_gpu(m: &mut FibreSu2Gpu, p: &Paires, cfg: &ConfigMesure) -> Resultat<SortieReplique> {
    if cfg.n_mes == 0 || cfg.nblk == 0 || !cfg.n_mes.is_multiple_of(cfg.nblk) || cfg.lot == 0 {
        return Err(Erreur::Config("n_mes doit être un multiple non nul de nblk ; lot ⩾ 1".into()));
    }
    super::mesure::verifier_compteur_philox(cfg)?;
    if cfg.separes || !cfg.canaux.is_empty() || cfg.ord != super::modele::Ordonnancement::Simultane {
        return Err(Erreur::Config("GPU : scores moyens, ordonnancement simultané, aucun canal d'oracle".into()));
    }
    if !p.px.is_empty() {
        return Err(Erreur::Config("GPU : fenêtre b ⩾ 2 seulement (pas de terme E[X(O)])".into()));
    }
    let t0 = Instant::now();
    let app = m.app.clone();
    let g = m.g.clone();
    let (nf, ns, np) = (p.faces.len(), p.sondes.len(), p.npaires());
    let nc = 3usize;
    let nblk = cfg.nblk;
    let lon_blk = cfg.n_mes / nblk;
    let lot = cfg.lot;
    // tables de mesure
    let fmu: Vec<u32> = p.faces.iter().map(|&pi| g.planes[pi as usize / g.v].0 as u32).collect();
    let fnu: Vec<u32> = p.faces.iter().map(|&pi| g.planes[pi as usize / g.v].1 as u32).collect();
    let fsite: Vec<u32> = p.faces.iter().map(|&pi| (pi as usize % g.v) as u32).collect();
    let pmu: Vec<u32> = p.sondes.iter().map(|&ci| (ci as usize / g.vc) as u32).collect();
    let ps = |j: usize| -> Vec<u32> { p.sondes.iter().map(|&ci| g.lignes[ci as usize][j] % g.v as u32).collect() };
    let (d_fmu, d_fnu, d_fsite) = (tv_u32(&app, &fmu)?, tv_u32(&app, &fnu)?, tv_u32(&app, &fsite)?);
    let (d_pmu, d_ps1, d_ps2, d_ps3) = (tv_u32(&app, &pmu)?, tv_u32(&app, &ps(0))?, tv_u32(&app, &ps(1))?, tv_u32(&app, &ps(2))?);
    let d_pface = tv_u32(&app, &p.face_loc)?;
    let mut pprobe = vec![0u32; np];
    for s in 0..ns {
        for q in p.debut[s] as usize..p.debut[s + 1] as usize {
            pprobe[q] = s as u32;
        }
    }
    let d_pprobe = tv_u32(&app, &pprobe)?;
    drop(pprobe);
    let obuf = app.alloc::<f64>(nf * lot)?;
    let sbuf = app.alloc::<f64>(ns * nc * lot)?;
    let mut so_d = app.alloc::<f64>(nf)?;
    let mut ss_d = app.alloc::<f64>(ns * nc)?;
    let mut sos_d = app.alloc::<f64>(np * nc)?;
    app.zero(&mut so_d)?;
    app.zero(&mut ss_d)?;
    app.zero(&mut sos_d)?;
    let mut so = vec![0.0f64; nblk * nf];
    let mut ss = vec![0.0f64; nblk * ns * nc];
    let mut sos = vec![0.0f64; nblk * np * nc];
    let mut n_blk = vec![0u64; nblk];
    let n_so = cfg.series_faces.len();
    let n_ss = cfg.series_sondes.len() * nc;
    let mut serie_o = vec![0.0f64; n_so * cfg.n_mes];
    let mut serie_s = vec![0.0f32; n_ss * cfg.n_mes];
    let mut controles = Vec::new();
    let mut jalons: Vec<(usize, String)> = Vec::new();
    let mut moy_classes_t: Vec<Vec<f64>> = Vec::new();
    let mut essais_tot = 0u64;
    let mut majs_tot = 0u64;
    let mut nb = 0usize;
    let mut t_lot0 = 0usize;
    let mut sbuf_h = vec![0.0f64; if n_ss > 0 { ns * nc * lot } else { 0 }];
    let sidx: Vec<u32> = cfg.series_faces.iter().map(|&f| f as u32).collect();
    let d_sidx = tv_u32(&app, &sidx)?;
    let oser = app.alloc::<f64>((n_so * lot).max(1))?;
    let mut oser_h = vec![0.0f64; n_so * lot];
    let total = cfg.n_dec + cfg.n_mes;
    for t in 0..total {
        let sweep = cfg.sweep0.wrapping_add(t as u32);
        m.pas(sweep, cfg.n_or)?;
        if (t + 1).is_multiple_of(super::mesure::JALON) {
            m.telecharger()?;
            jalons.push((t + 1, m.miroir.empreinte()));
        }
        if cfg.controle_chaque > 0 && (t + 1).is_multiple_of(cfg.controle_chaque) {
            m.telecharger()?;
            controles.push(Controle {
                iteration: t + 1,
                contrainte: m.miroir.residu_contrainte(),
                unitarite: m.miroir.defaut_unitarite(),
                ward: m.miroir.residu_ward(),
            });
            let (mj, es) = m.essais_dernier_balayage()?;
            majs_tot += mj;
            essais_tot += es;
            if cfg.classes_t {
                let mut out = vec![0.0f64; g.d - 1 + 3];
                m.miroir.canal(&Canal::GroupesPlaquettes, &mut out);
                moy_classes_t.push(out[..g.d - 1].to_vec());
            }
        }
        if t < cfg.n_dec {
            continue;
        }
        let tm = t - cfg.n_dec;
        let blk = tm / lon_blk;
        if nb == 0 {
            t_lot0 = tm;
        }
        app.lancer(
            &m.f.obs,
            &Lancement::lineaire(nf, BLOC),
            &[m.u.arg(), Arg::I32(g.l as i32), d_fmu.arg(), d_fnu.arg(), d_fsite.arg(), obuf.arg(), Arg::I32(nb as i32), Arg::I64(nf as i64)],
        )?;
        app.lancer(
            &m.f.scores,
            &Lancement::lineaire(ns, BLOC),
            &[
                m.u.arg(),
                Arg::I32(g.l as i32),
                d_pmu.arg(),
                d_ps1.arg(),
                d_ps2.arg(),
                d_ps3.arg(),
                Arg::F64(m.beta),
                sbuf.arg(),
                Arg::I32(lot as i32),
                Arg::I32(nb as i32),
                Arg::I64(ns as i64),
            ],
        )?;
        nb += 1;
        n_blk[blk] += 1;
        let fin_de_bloc = (tm + 1).is_multiple_of(lon_blk);
        if nb == lot || fin_de_bloc {
            // sommes Σ O, Σ σ, Σ O σ du lot
            app.lancer(&m.f.sommes_o, &Lancement::lineaire(nf, BLOC), &[so_d.arg(), obuf.arg(), Arg::I32(nb as i32), Arg::I64(nf as i64)])?;
            app.lancer(
                &m.f.sommes_s,
                &Lancement::lineaire(ns * nc, BLOC),
                &[ss_d.arg(), sbuf.arg(), Arg::I32(lot as i32), Arg::I32(nb as i32), Arg::I64((ns * nc) as i64)],
            )?;
            app.lancer(
                &m.f.acc,
                &Lancement::lineaire(np, BLOC),
                &[sos_d.arg(), obuf.arg(), sbuf.arg(), d_pface.arg(), d_pprobe.arg(), Arg::I32(lot as i32), Arg::I32(nb as i32), Arg::I64(nf as i64), Arg::I64(np as i64)],
            )?;
            // séries temporelles du lot
            if n_so > 0 {
                app.lancer(
                    &m.f.serie_o,
                    &Lancement::lineaire(n_so * nb, BLOC),
                    &[oser.arg(), obuf.arg(), d_sidx.arg(), Arg::I32(lot as i32), Arg::I32(nb as i32), Arg::I64(nf as i64), Arg::I64(n_so as i64)],
                )?;
                app.synchroniser()?;
                app.lire(&oser, 0, &mut oser_h)?;
                for k in 0..n_so {
                    serie_o[k * cfg.n_mes + t_lot0..k * cfg.n_mes + t_lot0 + nb].copy_from_slice(&oser_h[k * lot..k * lot + nb]);
                }
            }
            if n_ss > 0 {
                app.synchroniser()?;
                app.lire(&sbuf, 0, &mut sbuf_h)?;
                for (k, &sd) in cfg.series_sondes.iter().enumerate() {
                    for c in 0..nc {
                        let base = (sd * nc + c) * lot;
                        for j in 0..nb {
                            serie_s[(k * nc + c) * cfg.n_mes + t_lot0 + j] = sbuf_h[base + j] as f32;
                        }
                    }
                }
            }
            nb = 0;
            if fin_de_bloc {
                app.synchroniser()?;
                app.lire(&so_d, 0, &mut so[blk * nf..(blk + 1) * nf])?;
                app.lire(&ss_d, 0, &mut ss[blk * ns * nc..(blk + 1) * ns * nc])?;
                app.lire(&sos_d, 0, &mut sos[blk * np * nc..(blk + 1) * np * nc])?;
                app.zero(&mut so_d)?;
                app.zero(&mut ss_d)?;
                app.zero(&mut sos_d)?;
            }
        }
    }
    app.synchroniser()?;
    m.telecharger()?;
    controles.push(Controle {
        iteration: total,
        contrainte: m.miroir.residu_contrainte(),
        unitarite: m.miroir.defaut_unitarite(),
        ward: m.miroir.residu_ward(),
    });
    let plafond = m.plafond()?;
    Ok(SortieReplique {
        nc,
        nblk,
        nfaces: nf,
        nsondes: ns,
        npaires: np,
        n_blk,
        so,
        ss,
        sos,
        px: Vec::new(),
        sx: Vec::new(),
        serie_o,
        serie_s,
        n_serie_o: n_so,
        n_serie_s: n_ss,
        serie_canaux: Vec::new(),
        n_canal: 0,
        moy_classes_t,
        controles,
        jalons,
        stats: StatsEchantillonneur { majs: majs_tot, essais: essais_tot, plafond },
        empreinte_finale: m.miroir.empreinte(),
        secondes: t0.elapsed().as_secs_f64(),
        n_mes: cfg.n_mes,
    })
}

/// Champ de départ → `Vec<Q>` (utilitaire de test).
pub fn champ_vers_q(v: &[[f64; 4]]) -> Vec<Q> {
    v.to_vec()
}
