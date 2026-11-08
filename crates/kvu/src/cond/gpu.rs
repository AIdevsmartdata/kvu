//! Moteur GPU du bain conditionnel (HIP ou CUDA, même source `noyaux/cond.cu`). Propriétaire : T5.
//!
//! Une chaîne, champ FP64 résident (`su2::ChampGpu`, chaîne 0). Les tables d'indices de la géométrie
//! (paires, libres par parité) sont téléversées une fois ; V figé aussi.
//! - `ref_rng` : noyau de préparation → lecture de α → tirages numpy de l'hôte → noyau de fin ;
//! - `rapide` : un noyau par étape, Philox à compteur.

use super::cpu::DiagRapide;
use super::geometrie::GeometrieCond;
use super::numpy_rng::GenerateurNumpy;
use crate::commun::{ChampHote, Graine, Resultat};
use crate::gpu::{assembler, Appareil, Arg, Fonction, Lancement, Module, OptionsCompil, ProvenanceCompil, Tampon};
use crate::noyaux::{COND, SU2_COMMUN};
use crate::su2::{ChampGpu, TamponChamp};

const BLOC: u32 = 256;

/// Les fonctions du module `cond`.
struct Fonctions {
    reunit: Fonction,
    paires_prep: Fonction,
    paires_fin: Fonction,
    libres_prep: Fonction,
    libres_fin: Fonction,
    paires_rapide: Fonction,
    libres_rapide: Fonction,
    vdev: Fonction,
    unite: Fonction,
}

/// État GPU d'une chaîne conditionnelle.
pub struct EtatGpu {
    app: Appareil,
    champ: ChampGpu,
    l: usize,
    v: usize,
    npair: usize,
    fonctions: Fonctions,
    provenance: ProvenanceCompil,
    p1: Vec<Tampon<u32>>,
    p1pm: Vec<Tampon<u32>>,
    /// Indexé par mu·2 + parité.
    libres: Vec<Tampon<u32>>,
    taille_libres: Vec<usize>,
    vfix: Tampon<[f64; 4]>,
    vn: Tampon<[f64; 4]>,
    alpha: Tampon<f64>,
    tir: Tampon<f64>,
    plafond: Tampon<u64>,
    diag: Option<Tampon<u32>>,
    alpha_h: Vec<f64>,
    tir_h: Vec<f64>,
}

fn televerser_u32(app: &Appareil, t: &[u32]) -> Resultat<Tampon<u32>> {
    let mut d = app.alloc::<u32>(t.len().max(1))?;
    if !t.is_empty() {
        app.ecrire(&mut d, 0, t)?;
    }
    Ok(d)
}

impl EtatGpu {
    pub fn nouveau(app: &Appareil, geo: &GeometrieCond, c: &ChampHote, vfix: &[[f64; 4]]) -> Resultat<EtatGpu> {
        let backend = app.backend();
        let src = assembler(backend, "cond", &[&SU2_COMMUN, &COND], &[]);
        // Référence : pas de contraction FMA (cb.py : numba fastmath=False), pour rester à 1e-12 de cb.py
        // et pour que CPU et GPU prennent les mêmes décisions (EQC-3).
        let opts = OptionsCompil::reference_k1();
        let module: Module = app.compiler(&src, &opts)?;
        let provenance = ProvenanceCompil {
            nom: src.nom.clone(),
            backend,
            arch: app.arch(),
            sha256_source: src.sha256.clone(),
            parties: src.parties.clone(),
            options: opts.clone(),
            arguments: opts.arguments(backend, &app.arch()),
            sha256_code: module.sha256_code.clone(),
        };
        let f = |n: &str| app.fonction(&module, n);
        let fonctions = Fonctions {
            reunit: f("k_cond_reunit")?,
            paires_prep: f("k_cond_paires_prep")?,
            paires_fin: f("k_cond_paires_fin")?,
            libres_prep: f("k_cond_libres_prep")?,
            libres_fin: f("k_cond_libres_fin")?,
            paires_rapide: f("k_cond_paires_rapide")?,
            libres_rapide: f("k_cond_libres_rapide")?,
            vdev: f("k_cond_vdev")?,
            unite: f("k_cond_unite")?,
        };
        // Champ résident : chaîne 0, FP64, stockage (c·4V + li)·4 + k = disposition de `ChampHote::q` (c = 0).
        let mut tampon_champ = app.alloc::<f64>(c.q.len())?;
        app.ecrire(&mut tampon_champ, 0, &c.q)?;
        let champ_gpu = ChampGpu { l: geo.l, nchaines: 1, tampon: TamponChamp::F64(tampon_champ) };
        let np = geo.npair();
        let mut p1 = Vec::new();
        let mut p1pm = Vec::new();
        let mut libres = Vec::new();
        let mut taille_libres = Vec::new();
        for mu in 0..4 {
            p1.push(televerser_u32(app, &geo.p1[mu])?);
            p1pm.push(televerser_u32(app, &geo.p1pm[mu])?);
            for p in 0..2 {
                libres.push(televerser_u32(app, &geo.fr[mu][p])?);
                taille_libres.push(geo.fr[mu][p].len());
            }
        }
        let nmax = taille_libres.iter().copied().max().unwrap_or(0).max(np);
        let mut vfix_d = app.alloc::<[f64; 4]>(vfix.len())?;
        app.ecrire(&mut vfix_d, 0, vfix)?;
        Ok(EtatGpu {
            app: app.clone(),
            champ: champ_gpu,
            l: geo.l,
            v: geo.v,
            npair: np,
            fonctions,
            provenance,
            p1,
            p1pm,
            libres,
            taille_libres,
            vfix: vfix_d,
            vn: app.alloc::<[f64; 4]>(nmax)?,
            alpha: app.alloc::<f64>(nmax)?,
            tir: app.alloc::<f64>(5 * nmax)?,
            plafond: app.alloc::<u64>(1)?,
            diag: None,
            alpha_h: vec![0.0; nmax],
            tir_h: vec![0.0; 5 * nmax],
        })
    }

    pub fn backend(&self) -> crate::commun::BackendKind {
        self.app.backend()
    }

    pub fn champ(&self) -> &ChampGpu {
        &self.champ
    }

    pub fn provenance(&self) -> &ProvenanceCompil {
        &self.provenance
    }

    fn arg_champ(&self) -> Arg {
        match &self.champ.tampon {
            TamponChamp::F64(t) => t.arg(),
            TamponChamp::F32(_) => unreachable!("le champ conditionnel est toujours FP64"),
        }
    }

    pub fn telecharger(&self) -> Resultat<ChampHote> {
        let mut q = vec![0.0f64; 16 * self.v];
        match &self.champ.tampon {
            TamponChamp::F64(t) => {
                self.app.synchroniser()?;
                self.app.lire(t, 0, &mut q)?;
            }
            TamponChamp::F32(_) => unreachable!("le champ conditionnel est toujours FP64"),
        }
        Ok(ChampHote { l: self.l, q })
    }

    /// Remplace le champ résident (reprise) sans réallouer.
    pub fn televerser(&mut self, c: &ChampHote) -> Resultat<()> {
        match &mut self.champ.tampon {
            TamponChamp::F64(t) => self.app.ecrire(t, 0, &c.q),
            TamponChamp::F32(_) => unreachable!("le champ conditionnel est toujours FP64"),
        }
    }

    fn reunitariser(&self) -> Resultat<()> {
        let n = 4 * self.v;
        self.app.lancer(&self.fonctions.reunit, &Lancement::lineaire(n, BLOC), &[self.arg_champ(), Arg::I64(n as i64)])
    }

    /// Un balayage `ref_rng` : GPU pour les agrafes et les liens, hôte pour les tirages numpy.
    pub fn balayage_ref_rng(&mut self, beta: f64, rng: &mut GenerateurNumpy) -> Resultat<()> {
        self.reunitariser()?;
        let np = self.npair;
        let (l, u) = (self.l as i32, self.arg_champ());
        for mu in 0..4usize {
            let cfg = Lancement::lineaire(np, BLOC);
            self.app.lancer(
                &self.fonctions.paires_prep,
                &cfg,
                &[
                    u,
                    Arg::I32(l),
                    Arg::I32(mu as i32),
                    self.p1[mu].arg(),
                    self.p1pm[mu].arg(),
                    self.vfix.arg_decale(mu * np),
                    Arg::F64(beta),
                    self.vn.arg(),
                    self.alpha.arg(),
                    Arg::I64(np as i64),
                ],
            )?;
            self.tirer(np, rng)?;
            self.app.lancer(
                &self.fonctions.paires_fin,
                &cfg,
                &[
                    u,
                    Arg::I32(l),
                    Arg::I32(mu as i32),
                    self.p1[mu].arg(),
                    self.p1pm[mu].arg(),
                    self.vfix.arg_decale(mu * np),
                    self.tir.arg(),
                    self.vn.arg(),
                    Arg::I64(np as i64),
                ],
            )?;
        }
        for mu in 0..4usize {
            for p in 0..2usize {
                let k = 2 * mu + p;
                let n = self.taille_libres[k];
                let cfg = Lancement::lineaire(n, BLOC);
                self.app.lancer(
                    &self.fonctions.libres_prep,
                    &cfg,
                    &[
                        u,
                        Arg::I32(l),
                        Arg::I32(mu as i32),
                        self.libres[k].arg(),
                        Arg::F64(beta),
                        self.vn.arg(),
                        self.alpha.arg(),
                        Arg::I64(n as i64),
                    ],
                )?;
                self.tirer(n, rng)?;
                self.app.lancer(
                    &self.fonctions.libres_fin,
                    &cfg,
                    &[
                        u,
                        Arg::I32(l),
                        Arg::I32(mu as i32),
                        self.libres[k].arg(),
                        self.tir.arg(),
                        self.vn.arg(),
                        Arg::I64(n as i64),
                    ],
                )?;
            }
        }
        self.app.synchroniser()
    }

    /// Lit les α (`n` premiers), tire (x0, r, v) avec numpy, écrit (x0, r, v0, v1, v2) ligne par ligne.
    fn tirer(&mut self, n: usize, rng: &mut GenerateurNumpy) -> Resultat<()> {
        self.app.synchroniser()?;
        self.app.lire(&self.alpha, 0, &mut self.alpha_h[..n])?;
        let (x0, r, v) = rng.hb_numpy(&self.alpha_h[..n])?;
        for i in 0..n {
            let t = &mut self.tir_h[5 * i..5 * i + 5];
            t[0] = x0[i];
            t[1] = r[i];
            t[2..5].copy_from_slice(&v[i]);
        }
        self.app.ecrire(&mut self.tir, 0, &self.tir_h[..5 * n])
    }

    /// Active le diagnostic du nombre d'essais par lien (EQC-3) ; une seule allocation de 4V entiers.
    pub fn activer_diag(&mut self) -> Resultat<()> {
        if self.diag.is_none() {
            self.diag = Some(self.app.alloc::<u32>(4 * self.v)?);
        }
        Ok(())
    }

    /// Un balayage rapide : Philox, tout sur le GPU. Rend le nombre de tirages au plafond depuis la dernière lecture.
    pub fn balayage_rapide(&mut self, beta: f64, graine: Graine, sweep_id: u32) -> Resultat<()> {
        self.reunitariser()?;
        let np = self.npair;
        let (l, u) = (self.l as i32, self.arg_champ());
        let (k0, k1) = graine.cle();
        let diag = self.diag.as_ref().map(|d| d.arg()).unwrap_or(Arg::Nul);
        for mu in 0..4usize {
            self.app.lancer(
                &self.fonctions.paires_rapide,
                &Lancement::lineaire(np, BLOC),
                &[
                    u,
                    Arg::I32(l),
                    Arg::I32(mu as i32),
                    self.p1[mu].arg(),
                    self.p1pm[mu].arg(),
                    self.vfix.arg_decale(mu * np),
                    Arg::F64(beta),
                    Arg::U32(k0),
                    Arg::U32(k1),
                    Arg::U32(sweep_id),
                    diag,
                    self.plafond.arg(),
                    Arg::I64(np as i64),
                ],
            )?;
        }
        for mu in 0..4usize {
            for p in 0..2usize {
                let k = 2 * mu + p;
                let n = self.taille_libres[k];
                self.app.lancer(
                    &self.fonctions.libres_rapide,
                    &Lancement::lineaire(n, BLOC),
                    &[
                        u,
                        Arg::I32(l),
                        Arg::I32(mu as i32),
                        self.libres[k].arg(),
                        Arg::F64(beta),
                        Arg::U32(k0),
                        Arg::U32(k1),
                        Arg::U32(sweep_id),
                        diag,
                        self.plafond.arg(),
                        Arg::I64(n as i64),
                    ],
                )?;
            }
        }
        self.app.synchroniser()
    }

    /// Diagnostic du dernier balayage rapide (plafond cumulé ; essais par lien si activé).
    pub fn diag_rapide(&self) -> Resultat<DiagRapide> {
        let mut cap = [0u64];
        self.app.lire(&self.plafond, 0, &mut cap)?;
        let mut essais = Vec::new();
        if let Some(d) = &self.diag {
            essais = vec![0u32; 4 * self.v];
            self.app.lire(d, 0, &mut essais)?;
        }
        Ok(DiagRapide { plafond: cap[0], essais })
    }

    /// (max |V(C) − Vfix|, max | |q|² − 1 |) ; les maxima sont pris à l'hôte (déterministe).
    pub fn controles(&self) -> Resultat<(f64, f64)> {
        let np = self.npair;
        let u = self.arg_champ();
        let mut vdev = 0.0f64;
        let out = self.app.alloc::<f64>(np.max(1))?;
        let mut h = vec![0.0f64; np];
        for mu in 0..4usize {
            self.app.lancer(
                &self.fonctions.vdev,
                &Lancement::lineaire(np, BLOC),
                &[
                    u,
                    Arg::I32(self.l as i32),
                    Arg::I32(mu as i32),
                    self.p1[mu].arg(),
                    self.p1pm[mu].arg(),
                    self.vfix.arg_decale(mu * np),
                    out.arg(),
                    Arg::I64(np as i64),
                ],
            )?;
            self.app.synchroniser()?;
            self.app.lire(&out, 0, &mut h)?;
            vdev = h.iter().copied().fold(vdev, f64::max);
        }
        let n = 4 * self.v;
        let out_u = self.app.alloc::<f64>(n)?;
        self.app.lancer(&self.fonctions.unite, &Lancement::lineaire(n, BLOC), &[u, Arg::I64(n as i64), out_u.arg()])?;
        self.app.synchroniser()?;
        let mut hu = vec![0.0f64; n];
        self.app.lire(&out_u, 0, &mut hu)?;
        Ok((vdev, hu.iter().copied().fold(0.0, f64::max)))
    }
}

/// Loi du tirage rapide sur le GPU : `n` tirages de x₀ pour UN α (compteur (i, 0x7E57, tirage, ETIQUETTE_COND)).
/// Rend (x₀, tirages au plafond). Sert à la gate de loi d'EQC-2.
pub fn tirer_x0(app: &Appareil, alpha: f64, n: usize, graine: Graine) -> Resultat<(Vec<f64>, u64)> {
    let backend = app.backend();
    let src = assembler(backend, "cond", &[&SU2_COMMUN, &COND], &[]);
    let module = app.compiler(&src, &OptionsCompil::reference_k1())?;
    let f = app.fonction(&module, "k_cond_test_x0")?;
    let sortie = app.alloc::<f64>(n.max(1))?;
    let plafond = app.alloc::<u64>(1)?;
    let (k0, k1) = graine.cle();
    app.lancer(
        &f,
        &Lancement::lineaire(n, BLOC),
        &[Arg::F64(alpha), Arg::I64(n as i64), Arg::U32(k0), Arg::U32(k1), sortie.arg(), plafond.arg()],
    )?;
    app.synchroniser()?;
    let mut x = vec![0.0f64; n];
    app.lire(&sortie, 0, &mut x)?;
    let mut cap = [0u64];
    app.lire(&plafond, 0, &mut cap)?;
    Ok((x, cap[0]))
}
