//! [EXPÉRIMENTAL] `kvu su2 spectro gpu` : génération (noyaux de production de `UneBoite`, F32 ou F64) ET mesure sur la carte des
//! opérateurs scalaires A₁⁺⁺ lissés APE de `spectro::neutre`, à impulsion nulle et aux crans `--impulsions`.
//!
//! Même définition que `neutre` (CPU) : S_f(t) = (1/(N_im V₃)) Σ_x Σ_im ½Tr W_im(x,t) sur le champ lissé, op = i_niveau·N_formes + i_forme ;
//! corrélateur symétrisé sur t (temps périodique) ; Fourier à la phase du CENTRE de chaque image ; moyenne des 3 axes. Les produits de
//! liens sont faits dans la précision du champ (F32 : float), toutes les sommes en double à arbre fixe (déterministe).
//! Sorties : un dossier `r<i>` par chaîne (réplique) avec `neutre.bin`, `neutre_p<k>.bin`, `plaq.csv`, `manifeste.json`, au MÊME
//! format que `spectro neutre` (les lecteurs existants les lisent sans changement).

use super::bloc::ParamBloc;
use super::cubique::{analyser_rep, poids_images, Irrep};
use super::neutre::{centre_chemin, images, Orbite, FORMES};
use crate::commun::{Depart, Erreur, Graine, Precision, Resultat};
use crate::gpu::{assembler, Appareil, Arg, Fonction, Lancement, Module, OptionsCompil, Tampon};
use crate::noyaux::Fichier;
use crate::su2::une_boite::{ParamBoite, UneBoite};
use crate::su2::torsion::Torsion;
use crate::su2::ParamChaine;

pub static SU2_SPECTRO: Fichier = Fichier { nom: "su2_spectro.cu", contenu: include_str!("../../../noyaux/su2_spectro.cu") };

/// Toutes les formes connues (nom, chemin) : les cinq de la base ordinaire, puis les six de la base riche (`FORMES[5..]`).
pub fn toutes_formes() -> Vec<(&'static str, &'static [i8])> {
    FORMES.to_vec()
}

#[derive(Clone, Debug)]
pub struct ParamGpu {
    pub cotes: [usize; 4],
    pub temps: usize,
    pub niveaux: Vec<usize>,
    pub alpha: f64,
    /// noms de formes (dans `toutes_formes`)
    pub formes: Vec<String>,
    pub tau_max: usize,
    pub ks: Vec<i64>,
    pub precision: Precision,
    /// contrôle négatif de noyau (0 = production ; 1 à 4 : voir su2_spectro.cu)
    pub mutant: u8,
    pub nor: usize,
    /// représentations cubiques supplémentaires à impulsion nulle (« A2+ », « E+ », « T1- », … ; A1+ est la mesure principale)
    pub reps: Vec<String>,
    /// une entrée par chaîne
    pub betas: Vec<f64>,
    pub graines: Vec<u64>,
    /// blocage de Teper (niveaux mesurés APRÈS les niveaux APE ; vide = comportement d'origine, au bit près)
    pub bloc: ParamBloc,
    /// `--ops-bruts` : remonte aussi les sommes brutes O_a(p,t) (tampon `sg`) dans `MesCh::ops`.
    /// false (défaut) = aucune lecture de plus, sorties inchangées au bit près.
    pub ops_bruts: bool,
    /// `--xi0` : réseau anisotrope (poids ξ₀ des plaquettes temporelles, 1/ξ₀ des spatiales). None ou Some(1.0) : inchangé au bit près.
    pub xi0: Option<f64>,
    /// `--impulsions-vecteurs` : orbites cubiques d'impulsion hors axe (voir `neutre::Orbite`). Vide (défaut) = comportement
    /// d'origine, au bit près. Incompatible avec `ks`, le blocage, `--ops-bruts` (refus à la construction).
    pub orbites: Vec<Orbite>,
}

/// Mesure séparée des plaquettes spatiales et temporelles (réseau anisotrope seulement ; jamais construite si ξ₀ = 1).
pub struct PlaqAn {
    f: Fonction,
    s: Tampon<f64>,
    t: Tampon<f64>,
    nblk: usize,
}

impl PlaqAn {
    pub fn nouveau(app: &Appareil, module: &Module, nch: usize, v: usize) -> Resultat<PlaqAn> {
        let nblk = v.div_ceil(256);
        Ok(PlaqAn { f: app.fonction(module, "k_plaq_st_an")?, s: app.alloc::<f64>(nch * nblk)?, t: app.alloc::<f64>(nch * nblk)?, nblk })
    }
    /// Lance le noyau (à appeler avant `app.synchroniser()`).
    pub fn lancer(&self, app: &Appareil, champ: Arg, geo: Arg, temps: usize, v: usize, nch: usize) -> Resultat<()> {
        app.lancer(&self.f, &Lancement::lineaire_lot(v, nch, 256), &[champ, geo, Arg::I32(temps as i32), self.s.arg(), self.t.arg()])
    }
    /// (P_ss, P_st) par chaîne : chaque moyenne sur 3V plaquettes.
    pub fn lire(&self, app: &Appareil, nch: usize, v: usize) -> Resultat<Vec<(f64, f64)>> {
        let mut s = vec![0f64; nch * self.nblk];
        let mut t = vec![0f64; nch * self.nblk];
        app.lire(&self.s, 0, &mut s)?;
        app.lire(&self.t, 0, &mut t)?;
        let n = 3.0 * v as f64;
        Ok((0..nch)
            .map(|c| {
                let r = c * self.nblk..(c + 1) * self.nblk;
                (s[r.clone()].iter().sum::<f64>() / n, t[r].iter().sum::<f64>() / n)
            })
            .collect())
    }
}

/// Résultat sur une chaîne et une configuration.
#[derive(Clone, Debug)]
pub struct MesCh {
    /// [paire][τ] aplati (ordre de `neutre.bin`)
    pub corr: Vec<f64>,
    pub moy: Vec<f64>,
    /// par cran, [paire][τ] aplati
    pub corr_ps: Vec<Vec<f64>>,
    /// par orbite demandée (`orbites`), [paire][τ] aplati, moyenné sur l'orbite (vide si aucune)
    pub corr_orb: Vec<Vec<f64>>,
    pub plaq: f64,
    /// (P_ss, P_st) si ξ₀ ≠ 1 (sinon None)
    pub plaq_st: Option<(f64, f64)>,
    /// par représentation demandée : (corrélateurs [paire][τ] aplatis, moyennes), moyennés sur les lignes de la représentation
    pub reps: Vec<(Vec<f64>, Vec<f64>)>,
    /// `--ops-bruts` : O_a(p,t) de cette configuration, [t][p][a][re, im] aplati (vide si l'option est éteinte)
    pub ops: Vec<f64>,
}

impl RepGpu {
    pub fn nops(&self) -> usize {
        self.nops
    }
}

/// Une représentation cubique mesurée : formes non nulles, une ligne de poids par (ligne de la rep, forme).
pub struct RepGpu {
    pub nom: String,
    pub dim: usize,
    /// noms des formes retenues (opérateurs non nuls), dans l'ordre des opérateurs d'un niveau
    pub formes: Vec<String>,
    opidx: Vec<Tampon<i32>>,
    out_corr: Vec<Tampon<f64>>,
    out_moy: Vec<Tampon<f64>>,
    nops: usize,
}

/// Tampons du cumul sur la carte (algorithme à deux niveaux) : sommes des opérateurs et des produits « non valides ».
struct CumulCarte {
    d: usize,
    n: usize,
    /// Σ_j sg
    s: Tampon<f64>,
    /// [c][slot+1][paire][τ]
    o: Tampon<f64>,
    s2: Option<Tampon<f64>>,
    /// par représentation, par ligne : [c][paire][τ]
    o2: Vec<Vec<Tampon<f64>>>,
}

/// Sommes remontées (une fois par configuration externe).
pub struct SommesCarte {
    pub n: usize,
    /// Σ_j sg : [c][lev][f][t][q]
    pub s: Vec<f64>,
    /// [c][slot+1][paire][τ]
    pub o: Vec<f64>,
    /// par représentation, par ligne : (somme [(c·nops_r + a)·tt + t], ordinaire [c][paire][τ])
    pub reps: Vec<Vec<(Vec<f64>, Vec<f64>)>>,
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

pub struct SpectroGpu {
    pub boite: UneBoite,
    app: Appareil,
    p: ParamGpu,
    _module: Module,
    f_ape: Fonction,
    f_boucles: Fonction,
    f_boucles_v: Fonction,
    f_boucles_l_v: Fonction,
    /// phases de centre ×ℓ des vecteurs d'orbite, une table par niveau de blocage
    cen_bl_v: Vec<Tampon<f64>>,
    f_corr_orb: Fonction,
    /// (tables d'axe, phases de centre) des vecteurs d'orbite, à plat ; None si aucune orbite
    tab_v: Option<(Tampon<f64>, Tampon<f64>)>,
    nv: usize,
    f_oper: Fonction,
    f_corr: Fonction,
    f_moy: Fonction,
    f_cum_s: Fonction,
    f_cum_o: Fonction,
    cum: Option<CumulCarte>,
    f_plaq: Fonction,
    plaq_an: Option<PlaqAn>,
    f_bloc: Fonction,
    f_ape_l: Fonction,
    f_boucles_l: Fonction,
    /// tables de phase des niveaux de blocage (centre ×ℓ), une par niveau de `p.bloc.niveaux`
    ph_bl: Vec<Tampon<f64>>,
    buf_a: Buf,
    buf_b: Buf,
    part: Tampon<f64>,
    sg: Tampon<f64>,
    out_corr: Tampon<f64>,
    out_moy: Tampon<f64>,
    out_plaq: Tampon<f64>,
    pas: Tampon<i32>,
    off: Tampon<i32>,
    ph: Tampon<f64>,
    wop: Tampon<f64>,
    opidx: Tampon<i32>,
    nch: usize,
    v: usize,
    nim: usize,
    nf: usize,
    nq: usize,
    chunks: usize,
    lmax: usize,
    nblk_plaq: usize,
    pub noms_formes: Vec<String>,
    pub nombre_images: Vec<usize>,
    pub reps: Vec<RepGpu>,
    sg2: Option<Tampon<f64>>,
    wop2: Option<Tampon<f64>>,
    nf2: usize,
}

pub fn axes_spatiaux(temps: usize) -> [usize; 3] {
    let v: Vec<usize> = (0..4).filter(|&d| d != temps).collect();
    [v[0], v[1], v[2]]
}

impl SpectroGpu {
    #[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
    pub fn nouveau(app: &Appareil, p: &ParamGpu) -> Resultat<SpectroGpu> {
        let nch = p.betas.len();
        if nch == 0 || p.graines.len() != nch {
            return Err(Erreur::Config("spectro gpu : une graine et un β par chaîne".into()));
        }
        if p.temps > 3 || p.tau_max >= p.cotes[p.temps] || p.niveaux.is_empty() || p.niveaux.windows(2).any(|w| w[0] >= w[1]) {
            return Err(Erreur::Config("spectro gpu : temps 0..3, tau_max < ℓ_temps, niveaux croissants non vides".into()));
        }
        if p.mutant > 8 {
            return Err(Erreur::Config("spectro gpu : mutant 0 à 8".into()));
        }
        p.bloc.controler(&p.cotes, p.temps)?;
        let ax = axes_spatiaux(p.temps);
        let tt = p.cotes[p.temps];
        let v: usize = p.cotes.iter().product();
        let v3 = v / tt;
        // formes → images
        let connues = toutes_formes();
        let mut pas: Vec<i32> = Vec::new();
        let mut off: Vec<i32> = vec![0];
        let mut im_forme: Vec<usize> = Vec::new();
        let mut centres: Vec<[f64; 3]> = Vec::new();
        let mut nombre_images = Vec::new();
        let mut vus: Vec<Vec<Vec<i8>>> = Vec::new();
        for (f, nom) in p.formes.iter().enumerate() {
            let (_, chemin) = connues
                .iter()
                .find(|c| c.0 == nom)
                .ok_or_else(|| Erreur::Config(format!("--formes : « {nom} » inconnue")))?;
            let mut s = [0i32; 3];
            for &st in chemin.iter() {
                s[st.unsigned_abs() as usize - 1] += st.signum() as i32;
            }
            if s != [0, 0, 0] {
                return Err(Erreur::Config(format!("forme {nom} : chemin non fermé")));
            }
            let mut imgs = images(chemin);
            if p.mutant == 5 && imgs.len() > 1 {
                imgs.pop();
            }
            if vus.contains(&imgs) {
                return Err(Erreur::Config(format!("forme {nom} : mêmes images qu'une forme déjà demandée")));
            }
            nombre_images.push(imgs.len());
            for im in &imgs {
                pas.extend(im.iter().map(|&x| x as i32));
                off.push(pas.len() as i32);
                im_forme.push(f);
                centres.push(centre_chemin(im));
            }
            vus.push(imgs);
        }
        let nim = im_forme.len();
        let nf = p.formes.len();
        let nk = p.ks.len();
        let vs: Vec<[i64; 3]> = p.orbites.iter().flat_map(|o| o.vecteurs.iter().copied()).collect();
        let nv = vs.len();
        if nv > 0 {
            if nk > 0 || p.ops_bruts || p.mutant != 0 {
                return Err(Erreur::Config("spectro gpu : --impulsions-vecteurs est incompatible avec --impulsions, --ops-bruts et --mutant (le blocage est accepté)".into()));
            }
            for o in &p.orbites {
                let ax0 = axes_spatiaux(p.temps);
                if ax0.iter().any(|&d| p.cotes[d] != p.cotes[ax0[0]]) || o.representant == [0, 0, 0]
                    || o.representant.iter().any(|&k| 2 * k.abs() >= p.cotes[ax0[0]] as i64)
                {
                    return Err(Erreur::Config("spectro gpu : impulsions-vecteurs : vecteur nul, hors zone de Brillouin ou côtés spatiaux inégaux".into()));
                }
            }
        }
        let nq = if nv > 0 { 1 + 2 * nv } else { 1 + 6 * nk };
        let lmax = ax.iter().map(|&d| p.cotes[d]).max().unwrap();
        let chunks = v3.div_ceil(256);
        let nlev = p.niveaux.len() + p.bloc.niveaux.len();
        // phases : cos, sin de 2π k (x + c_d)/ℓ_d
        let pi2 = 2.0 * std::f64::consts::PI;
        let mut ph = vec![0f64; nim * nk.max(1) * 3 * lmax * 2];
        for im in 0..nim {
            for (ik, &k) in p.ks.iter().enumerate() {
                for d in 0..3 {
                    let l = p.cotes[ax[d]];
                    for x in 0..l {
                        let th = pi2 * ((x as f64) + centres[im][d]) * (k.unsigned_abs() as f64) / l as f64;
                        let o = ((((im * nk + ik) * 3 + d) * lmax) + x) * 2;
                        ph[o] = th.cos();
                        ph[o + 1] = th.sin();
                    }
                }
            }
        }
        // impulsions hors axe : tables d'axe (cos, sin)(k_d 2π x/ℓ) et phases de centre (cos, sin)(k·c) par (image, vecteur)
        let mut tv_h = vec![0f64; nv * 3 * lmax * 2];
        let mut cen_h = vec![0f64; nim * nv * 2];
        for (iv, k) in vs.iter().enumerate() {
            for d in 0..3 {
                let l = p.cotes[ax[d]];
                let sgn = if k[d] < 0 { -1.0 } else { 1.0 };
                for x in 0..l {
                    let th = pi2 * x as f64 * (k[d].unsigned_abs() as f64) / l as f64;
                    let o = ((iv * 3 + d) * lmax + x) * 2;
                    tv_h[o] = th.cos();
                    tv_h[o + 1] = sgn * th.sin();
                }
            }
            for im in 0..nim {
                let th: f64 = (0..3).map(|d| pi2 * centres[im][d] * (k[d] as f64) / p.cotes[ax[d]] as f64).sum();
                cen_h[(im * nv + iv) * 2] = th.cos();
                cen_h[(im * nv + iv) * 2 + 1] = th.sin();
            }
        }
        // idem pour les niveaux de blocage : centre de l'image agrandie ℓ = 2^b fois
        let cen_bl_h: Vec<Vec<f64>> = p
            .bloc
            .niveaux
            .iter()
            .map(|&b| {
                let lb = (1usize << b) as f64;
                let mut t = vec![0f64; nim * nv * 2];
                for (iv, k) in vs.iter().enumerate() {
                    for im in 0..nim {
                        let th: f64 = (0..3).map(|d| pi2 * lb * centres[im][d] * (k[d] as f64) / p.cotes[ax[d]] as f64).sum();
                        t[(im * nv + iv) * 2] = th.cos();
                        t[(im * nv + iv) * 2 + 1] = th.sin();
                    }
                }
                t
            })
            .collect();
        // phases des niveaux de blocage : cos, sin de 2π k (x + ℓ c_d)/ℓ_d (centre de l'image agrandie ℓ fois)
        let mut ph_bl_h: Vec<Vec<f64>> = Vec::new();
        // crans k ≥ 1 garantis par la CLI (`--impulsions` refuse k < 1) : la table est bâtie sur |k| comme celle des niveaux APE
        debug_assert!(p.ks.iter().all(|&k| k >= 1));
        for &b in &p.bloc.niveaux {
            let lb = (1usize << b) as f64;
            let mut t = vec![0f64; nim * nk.max(1) * 3 * lmax * 2];
            for im in 0..nim {
                for (ik, &k) in p.ks.iter().enumerate() {
                    for d in 0..3 {
                        let l = p.cotes[ax[d]];
                        for x in 0..l {
                            let th = pi2 * ((x as f64) + lb * centres[im][d]) * (k.unsigned_abs() as f64) / l as f64;
                            let o = ((((im * nk + ik) * 3 + d) * lmax) + x) * 2;
                            t[o] = th.cos();
                            t[o + 1] = th.sin();
                        }
                    }
                }
            }
            ph_bl_h.push(t);
        }
        // poids : 1/(N_im V3) sur les images de la forme
        let mut w = vec![0f64; nf * nim];
        for im in 0..nim {
            let f = im_forme[im];
            w[f * nim + im] = 1.0 / (nombre_images[f] * v3) as f64;
        }
        // représentations cubiques (impulsion nulle) : lignes de poids sur les mêmes images
        if !p.reps.is_empty() && p.mutant == 5 {
            return Err(Erreur::Config("spectro gpu : --reps incompatible avec le mutant image-oubliee".into()));
        }
        let mut deb_forme = vec![0usize; nf + 1];
        for f in 0..nf {
            deb_forme[f + 1] = deb_forme[f] + nombre_images[f];
        }
        let mut w2: Vec<Vec<f64>> = Vec::new(); // lignes de W2 (nim chacune)
        let mut plan: Vec<(String, usize, Vec<usize>, Vec<usize>)> = Vec::new(); // (nom, dim, formes retenues, base de ligne par rang)
        for nom in &p.reps {
            let (ir, par) = analyser_rep(nom).ok_or_else(|| Erreur::Config(format!("--reps : « {nom} » inconnue (A1,A2,E,T1,T2 suivis de + ou -)")))?;
            if ir == Irrep::A1 && par == 1 {
                return Err(Erreur::Config("--reps : A1+ est la mesure principale".into()));
            }
            let mut poids: Vec<Vec<Vec<f64>>> = Vec::new(); // [forme][rang][image de la forme]
            for nm in p.formes.iter() {
                let (_, ch) = connues.iter().find(|c| c.0 == nm).unwrap();
                poids.push((0..ir.dim() * ir.dim()).map(|rr| poids_images(ch, ir, par, rr / ir.dim(), rr % ir.dim())).collect());
            }
            let retenues: Vec<usize> = (0..nf)
                .filter(|&f| poids[f].iter().flatten().any(|x| x.abs() > 1e-10))
                .collect();
            let mut bases = Vec::new();
            for r in 0..ir.dim() * ir.dim() {
                bases.push(w2.len());
                for &f in &retenues {
                    let mut lig = vec![0f64; nim];
                    for (k, x) in poids[f][r].iter().enumerate() {
                        lig[deb_forme[f] + k] = x / v3 as f64;
                    }
                    w2.push(lig);
                }
            }
            plan.push((nom.clone(), ir.dim() * ir.dim(), retenues, bases));
        }
        let nf2 = w2.len();
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
        let kvt = if f32 { "float" } else { "double" };
        let mut_s = (if p.mutant == 5 { 0 } else { p.mutant }).to_string(); // 6 à 8 : noyaux de blocage seulement
        let src = assembler(app.backend(), "su2_spectro", &[&SU2_SPECTRO], &[("KV_T", kvt), ("MUT", mut_s.as_str())]);
        let module = app.compiler(&src, &if f32 { OptionsCompil::rapide() } else { OptionsCompil::reference_k1() })?;
        let f = |n: &str| app.fonction(&module, n);
        let alloc_champ = |n: usize| -> Resultat<Buf> {
            if f32 {
                Ok(Buf::F32(app.alloc::<f32>(n)?))
            } else {
                Ok(Buf::F64(app.alloc::<f64>(n)?))
            }
        };
        let nblk_plaq = v.div_ceil(256);
        let npair = nf * nlev * (nf * nlev + 1) / 2;
        let mut pas_t = app.alloc::<i32>(pas.len())?;
        app.ecrire(&mut pas_t, 0, &pas)?;
        let mut off_t = app.alloc::<i32>(off.len())?;
        app.ecrire(&mut off_t, 0, &off)?;
        let mut ph_t = app.alloc::<f64>(ph.len())?;
        app.ecrire(&mut ph_t, 0, &ph)?;
        let tab_v = if nv > 0 {
            let mut a = app.alloc::<f64>(tv_h.len())?;
            app.ecrire(&mut a, 0, &tv_h)?;
            let mut b = app.alloc::<f64>(cen_h.len())?;
            app.ecrire(&mut b, 0, &cen_h)?;
            Some((a, b))
        } else {
            None
        };
        let nsl = 1 + nk + p.orbites.len();
        let mut cen_bl_v = Vec::new();
        if nv > 0 {
            for t in &cen_bl_h {
                let mut tb = app.alloc::<f64>(t.len())?;
                app.ecrire(&mut tb, 0, t)?;
                cen_bl_v.push(tb);
            }
        }
        let mut w_t = app.alloc::<f64>(w.len())?;
        app.ecrire(&mut w_t, 0, &w)?;
        let mut ph_bl = Vec::new();
        for t in &ph_bl_h {
            let mut tb = app.alloc::<f64>(t.len())?;
            app.ecrire(&mut tb, 0, t)?;
            ph_bl.push(tb);
        }
        let mut reps = Vec::new();
        for (nom, dim, retenues, bases) in &plan {
            let nops2 = retenues.len() * nlev;
            let npair2 = nops2 * (nops2 + 1) / 2;
            let (mut oi, mut oc, mut om) = (Vec::new(), Vec::new(), Vec::new());
            for &b in bases {
                let mut t = app.alloc::<i32>(nops2)?;
                let v: Vec<i32> = (0..nlev).flat_map(|lev| (0..retenues.len()).map(move |j| (lev * nf2 + b + j) as i32)).collect();
                app.ecrire(&mut t, 0, &v)?;
                oi.push(t);
                oc.push(app.alloc::<f64>(nch * npair2 * (p.tau_max + 1))?);
                om.push(app.alloc::<f64>(nch * nops2)?);
            }
            reps.push(RepGpu { nom: nom.clone(), dim: *dim, formes: retenues.iter().map(|&f| p.formes[f].clone()).collect(), opidx: oi, out_corr: oc, out_moy: om, nops: nops2 });
        }
        let (sg2, wop2) = if nf2 > 0 {
            let mut w = app.alloc::<f64>(nf2 * nim)?;
            app.ecrire(&mut w, 0, &w2.iter().flatten().copied().collect::<Vec<f64>>())?;
            (Some(app.alloc::<f64>(nch * nlev * nf2 * tt)?), Some(w))
        } else {
            (None, None)
        };
        let nops = nf * nlev;
        let mut opidx = app.alloc::<i32>(nops)?;
        app.ecrire(&mut opidx, 0, &(0..nops as i32).collect::<Vec<_>>())?;
        Ok(SpectroGpu {
            f_ape: f("k_ape")?,
            f_boucles: f("k_boucles")?,
            f_boucles_v: f("k_boucles_v")?,
            f_boucles_l_v: f("k_boucles_l_v")?,
            cen_bl_v,
            f_corr_orb: f("k_corr_orb")?,
            tab_v,
            nv,
            f_oper: f("k_operateurs")?,
            f_corr: f("k_corr")?,
            f_moy: f("k_moy")?,
            f_cum_s: f("k_cumul_somme")?,
            f_cum_o: f("k_cumul_ord")?,
            cum: None,
            f_plaq: f("k_plaq")?,
            plaq_an: if p.xi0.is_some_and(|x| x != 1.0) { Some(PlaqAn::nouveau(app, &module, nch, v)?) } else { None },
            f_bloc: f("k_bloc")?,
            f_ape_l: f("k_ape_l")?,
            f_boucles_l: f("k_boucles_l")?,
            ph_bl,
            buf_a: alloc_champ(nch * 16 * v)?,
            buf_b: alloc_champ(nch * 16 * v)?,
            part: app.alloc::<f64>(nch * tt * nim * chunks * nq)?,
            sg: app.alloc::<f64>(nch * nlev * nf * tt * nq)?,
            out_corr: app.alloc::<f64>(nch * nsl * npair * (p.tau_max + 1))?,
            out_moy: app.alloc::<f64>(nch * nops)?,
            out_plaq: app.alloc::<f64>(nch * nblk_plaq)?,
            pas: pas_t,
            off: off_t,
            ph: ph_t,
            wop: w_t,
            opidx,
            nch,
            v,
            nim,
            nf,
            nq,
            chunks,
            lmax,
            nblk_plaq,
            reps,
            sg2,
            wop2,
            nf2,
            noms_formes: p.formes.clone(),
            nombre_images,
            app: app.clone(),
            p: p.clone(),
            _module: module,
            boite,
        })
    }

    pub fn nops(&self) -> usize {
        self.nf * self.nlev_tot()
    }
    /// niveaux APE + niveaux de blocage
    pub fn nlev_tot(&self) -> usize {
        self.p.niveaux.len() + self.p.bloc.niveaux.len()
    }
    pub fn npaires(&self) -> usize {
        self.nops() * (self.nops() + 1) / 2
    }

    /// Impulsions de `ops.bin`, dans l'ordre des indices p : `[0]`, puis par cran k de `ks`, par axe spatial croissant,
    /// `+k·e_ax` puis `−k·e_ax` (composantes en unités de 2π/ℓ_μ, sur les 4 directions). 1 + 6·nk entrées.
    pub fn ops_impulsions(&self) -> Vec<[i64; 4]> {
        let mut v = vec![[0i64; 4]];
        for &k in &self.p.ks {
            for &d in axes_spatiaux(self.p.temps).iter() {
                let mut n = [0i64; 4];
                n[d] = k;
                v.push(n);
                n[d] = -k;
                v.push(n);
            }
        }
        v
    }

    pub fn init(&mut self) -> Resultat<()> {
        self.boite.init()
    }
    /// `n` pas (1 bain thermique + nor surrelaxations).
    pub fn avancer(&mut self, n: usize) -> Resultat<()> {
        for _ in 0..n {
            self.boite.pas()?;
        }
        self.app.synchroniser()
    }

    /// Opérateurs du niveau `lev` (sur `nlev` niveaux au total) à partir des sommes partielles `part` du dernier noyau de boucles.
    fn operateurs_niveau(&self, lev: usize, nlev: usize) -> Resultat<()> {
        let (nch, nf, nq) = (self.nch, self.nf, self.nq);
        let tt = self.p.cotes[self.p.temps];
        self.app.lancer(
            &self.f_oper,
            &Lancement::lineaire_lot(nf * tt * nq, nch, 128),
            &[
                self.part.arg(),
                self.wop.arg(),
                Arg::I32(self.nim as i32),
                Arg::I32(nf as i32),
                Arg::I32(tt as i32),
                Arg::I32(nq as i32),
                Arg::I32(nq as i32),
                Arg::I32(self.chunks as i32),
                Arg::I32(nlev as i32),
                Arg::I32(lev as i32),
                self.sg.arg(),
            ],
        )?;
        if let (Some(sg2), Some(w2)) = (&self.sg2, &self.wop2) {
            self.app.lancer(
                &self.f_oper,
                &Lancement::lineaire_lot(self.nf2 * tt, nch, 128),
                &[
                    self.part.arg(),
                    w2.arg(),
                    Arg::I32(self.nim as i32),
                    Arg::I32(self.nf2 as i32),
                    Arg::I32(tt as i32),
                    Arg::I32(nq as i32),
                    Arg::I32(1),
                    Arg::I32(self.chunks as i32),
                    Arg::I32(nlev as i32),
                    Arg::I32(lev as i32),
                    sg2.arg(),
                ],
            )?;
        }
        Ok(())
    }

    /// Opérateurs de toutes les chaînes sur leur champ courant (APE, boucles, blocage) : remplit `sg` (et `sg2`), sans corrélateurs.
    fn calc_operateurs(&self) -> Resultat<()> {
        let p = &self.p;
        let (nch, v, _nf, nq) = (self.nch, self.v, self.nf, self.nq);
        let tt = p.cotes[p.temps];
        let nlev = p.niveaux.len() + p.bloc.niveaux.len();
        let nk = p.ks.len();
        let geo = self.boite.arg_geo();
        let mut cur = self.boite.arg_champ_complet();
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
            if let Some((tv, cen)) = &self.tab_v {
                self.app.lancer(
                    &self.f_boucles_v,
                    &Lancement { grille: [self.chunks as u32, (tt * self.nim) as u32, nch as u32], bloc: [256, 1, 1], shmem: 0 },
                    &[
                        cur,
                        geo,
                        Arg::I32(p.temps as i32),
                        Arg::I32(self.nim as i32),
                        self.pas.arg(),
                        self.off.arg(),
                        tv.arg(),
                        cen.arg(),
                        Arg::I32(self.nv as i32),
                        Arg::I32(self.lmax as i32),
                        self.part.arg(),
                        Arg::I32(nq as i32),
                    ],
                )?;
            } else {
                self.app.lancer(
                    &self.f_boucles,
                    &Lancement { grille: [self.chunks as u32, (tt * self.nim) as u32, nch as u32], bloc: [256, 1, 1], shmem: 0 },
                    &[
                        cur,
                        geo,
                        Arg::I32(p.temps as i32),
                        Arg::I32(self.nim as i32),
                        self.pas.arg(),
                        self.off.arg(),
                        self.ph.arg(),
                        Arg::I32(nk as i32),
                        Arg::I32(self.lmax as i32),
                        self.part.arg(),
                        Arg::I32(nq as i32),
                    ],
                )?;
            }
            self.operateurs_niveau(lev, nlev)?;
        }
        // ---- niveaux de blocage : chaîne APE^{n0} → (blocage ℓ→2ℓ, lissage des superliens)^b, ping-pong dans buf_a/buf_b
        // (libres : les niveaux APE sont déjà mesurés, la file de la carte est ordonnée). Champ d'origine en lecture seule.
        if p.bloc.actif() {
            let mut cur = self.boite.arg_champ_complet();
            let mut qui = 0usize;
            let pb = p.bloc.clone();
            let lancer_champ = |f: &Fonction, src: Arg, dst: Arg, extra: Vec<Arg>| -> Resultat<()> {
                let mut args = vec![src, dst, geo, Arg::I32(p.temps as i32)];
                args.extend(extra);
                self.app.lancer(f, &Lancement::lineaire_lot(4 * v, nch, 256), &args)
            };
            for _ in 0..pb.ape_bloc {
                let dst = if qui == 0 { self.buf_a.arg() } else { self.buf_b.arg() };
                lancer_champ(&self.f_ape, cur, dst, vec![Arg::F64(p.alpha)])?;
                cur = dst;
                qui ^= 1;
            }
            let bmax = *pb.niveaux.last().unwrap();
            for b in 1..=bmax {
                let dst = if qui == 0 { self.buf_a.arg() } else { self.buf_b.arg() };
                lancer_champ(&self.f_bloc, cur, dst, vec![Arg::I32(1 << (b - 1)), Arg::F64(pb.p_bloc)])?;
                cur = dst;
                qui ^= 1;
                for _ in 0..pb.lissage_bloc {
                    let dst = if qui == 0 { self.buf_a.arg() } else { self.buf_b.arg() };
                    lancer_champ(&self.f_ape_l, cur, dst, vec![Arg::F64(p.alpha), Arg::I32(1 << b)])?;
                    cur = dst;
                    qui ^= 1;
                }
                if let (Some(ib), Some((tv, _))) = (pb.niveaux.iter().position(|&x| x == b), &self.tab_v) {
                    self.app.lancer(
                        &self.f_boucles_l_v,
                        &Lancement { grille: [self.chunks as u32, (tt * self.nim) as u32, nch as u32], bloc: [256, 1, 1], shmem: 0 },
                        &[
                            cur,
                            geo,
                            Arg::I32(p.temps as i32),
                            Arg::I32(self.nim as i32),
                            self.pas.arg(),
                            self.off.arg(),
                            tv.arg(),
                            self.cen_bl_v[ib].arg(),
                            Arg::I32(self.nv as i32),
                            Arg::I32(self.lmax as i32),
                            self.part.arg(),
                            Arg::I32(nq as i32),
                            Arg::I32(1 << b),
                        ],
                    )?;
                    self.operateurs_niveau(p.niveaux.len() + ib, nlev)?;
                } else if let Some(ib) = pb.niveaux.iter().position(|&x| x == b) {
                    self.app.lancer(
                        &self.f_boucles_l,
                        &Lancement { grille: [self.chunks as u32, (tt * self.nim) as u32, nch as u32], bloc: [256, 1, 1], shmem: 0 },
                        &[
                            cur,
                            geo,
                            Arg::I32(p.temps as i32),
                            Arg::I32(self.nim as i32),
                            self.pas.arg(),
                            self.off.arg(),
                            self.ph_bl[ib].arg(),
                            Arg::I32(nk as i32),
                            Arg::I32(self.lmax as i32),
                            self.part.arg(),
                            Arg::I32(nq as i32),
                            Arg::I32(1 << b),
                        ],
                    )?;
                    self.operateurs_niveau(p.niveaux.len() + ib, nlev)?;
                }
            }
        }
        Ok(())
    }

    /// Algorithme à deux niveaux : pose (ou retire) le masque de liens gelés (4·V octets, li = μ·V + s).
    pub fn fixer_masque(&mut self, m: Option<&[u8]>) -> Resultat<()> {
        self.boite.fixer_masque(m)
    }

    /// Algorithme à deux niveaux : opérateurs d'impulsion nulle O_a(t) de toutes les chaînes sur leur champ courant, remontés
    /// sur l'hôte. Rend (principal, reps) : principal[(c·nops + a)·tt + t] (a = niveau·nf + forme), reps[i][(c·nops_i + a)·tt + t]
    /// moyennés sur les lignes de la représentation i n'est PAS fait ici : un tableau par ligne, reps[i][row][...].
    #[allow(clippy::type_complexity)] // style seulement (indices explicites conservés : aucun changement numérique)
    pub fn operateurs_hote(&mut self) -> Resultat<(Vec<f64>, Vec<Vec<Vec<f64>>>)> {
        self.calc_operateurs()?;
        self.app.synchroniser()?;
        let p = &self.p;
        let (nch, nf, nq) = (self.nch, self.nf, self.nq);
        let tt = p.cotes[p.temps];
        let nlev = p.niveaux.len() + p.bloc.niveaux.len();
        let mut sgv = vec![0f64; nch * nlev * nf * tt * nq];
        self.app.lire(&self.sg, 0, &mut sgv)?;
        let nops = self.nops();
        let mut main = vec![0f64; nch * nops * tt];
        for c in 0..nch {
            for a in 0..nops {
                for t in 0..tt {
                    main[(c * nops + a) * tt + t] = sgv[(((c * nlev) * nf) + a) * tt * nq + t * nq];
                }
            }
        }
        let mut reps = Vec::new();
        if let Some(sg2) = &self.sg2 {
            let mut s2 = vec![0f64; nch * nlev * self.nf2 * tt];
            self.app.lire(sg2, 0, &mut s2)?;
            for r in &self.reps {
                let mut rows = Vec::new();
                for row in 0..r.dim {
                    let mut oi = vec![0i32; r.nops];
                    self.app.lire(&r.opidx[row], 0, &mut oi)?;
                    let mut v = vec![0f64; nch * r.nops * tt];
                    for c in 0..nch {
                        for (a, &o) in oi.iter().enumerate() {
                            for t in 0..tt {
                                v[(c * r.nops + a) * tt + t] = s2[((c * nlev * self.nf2) + o as usize) * tt + t];
                            }
                        }
                    }
                    rows.push(v);
                }
                reps.push(rows);
            }
        }
        Ok((main, reps))
    }

    /// Cumul sur la carte : remet à zéro les sommes (D = `d`).
    pub fn cumul_init(&mut self, d: usize) -> Resultat<()> {
        let (nch, nf, nq) = (self.nch, self.nf, self.nq);
        let tt = self.p.cotes[self.p.temps];
        let nlev = self.nlev_tot();
        let nk = self.p.ks.len();
        let ntau = self.p.tau_max + 1;
        if self.cum.is_none() {
            let s = self.app.alloc::<f64>(nch * nlev * nf * tt * nq)?;
            let o = self.app.alloc::<f64>(nch * (1 + nk) * self.npaires() * ntau)?;
            let s2 = match &self.sg2 {
                Some(_) => Some(self.app.alloc::<f64>(nch * nlev * self.nf2 * tt)?),
                None => None,
            };
            let mut o2 = Vec::new();
            for r in &self.reps {
                let np2 = r.nops * (r.nops + 1) / 2;
                let mut rows = Vec::new();
                for _ in 0..r.dim {
                    rows.push(self.app.alloc::<f64>(nch * np2 * ntau)?);
                }
                o2.push(rows);
            }
            self.cum = Some(CumulCarte { d, n: 0, s, o, s2, o2 });
        }
        let app = self.app.clone();
        let c = self.cum.as_mut().unwrap();
        c.d = d;
        c.n = 0;
        let z = |t: &mut Tampon<f64>, n: usize| app.ecrire(t, 0, &vec![0f64; n]);
        z(&mut c.s, nch * nlev * nf * tt * nq)?;
        z(&mut c.o, nch * (1 + nk) * (self.nf * nlev) * (self.nf * nlev + 1) / 2 * ntau)?;
        if let Some(s2) = c.s2.as_mut() {
            z(s2, nch * nlev * self.nf2 * tt)?;
        }
        for (r, rows) in self.reps.iter().zip(c.o2.iter_mut()) {
            for t in rows.iter_mut() {
                z(t, nch * (r.nops * (r.nops + 1) / 2) * ntau)?;
            }
        }
        self.app.synchroniser()
    }

    /// Cumul sur la carte : calcule les opérateurs du champ courant et les ajoute aux sommes (aucune remontée).
    pub fn cumul_ajouter(&mut self) -> Resultat<()> {
        self.calc_operateurs()?;
        let (nch, nf, nq) = (self.nch, self.nf, self.nq);
        let tt = self.p.cotes[self.p.temps];
        let nlev = self.nlev_tot();
        let nk = self.p.ks.len();
        let nops = self.nops();
        let npair = self.npaires();
        let ntau = self.p.tau_max + 1;
        let c = self.cum.as_ref().expect("cumul_init");
        let ns = nch * nlev * nf * tt * nq;
        self.app.lancer(&self.f_cum_s, &Lancement::lineaire(ns, 256), &[self.sg.arg(), c.s.arg(), Arg::I64(ns as i64)])?;
        for slot in -1..nk as i32 {
            self.app.lancer(
                &self.f_cum_o,
                &Lancement::lineaire_lot(npair * ntau, nch, 128),
                &[
                    self.sg.arg(),
                    self.opidx.arg(),
                    Arg::I32(nops as i32),
                    Arg::I32(nf as i32),
                    Arg::I32(tt as i32),
                    Arg::I32(nq as i32),
                    Arg::I32(self.p.tau_max as i32),
                    Arg::I32(slot),
                    Arg::I32(1 + nk as i32),
                    Arg::I32(nlev as i32),
                    Arg::I32(c.d as i32),
                    c.o.arg(),
                ],
            )?;
        }
        if let (Some(sg2), Some(s2)) = (&self.sg2, &c.s2) {
            let n2 = nch * nlev * self.nf2 * tt;
            self.app.lancer(&self.f_cum_s, &Lancement::lineaire(n2, 256), &[sg2.arg(), s2.arg(), Arg::I64(n2 as i64)])?;
            for (r, rows) in self.reps.iter().zip(&c.o2) {
                let np2 = r.nops * (r.nops + 1) / 2;
                for (row, t) in rows.iter().enumerate() {
                    self.app.lancer(
                        &self.f_cum_o,
                        &Lancement::lineaire_lot(np2 * ntau, nch, 128),
                        &[
                            sg2.arg(),
                            r.opidx[row].arg(),
                            Arg::I32(r.nops as i32),
                            Arg::I32(self.nf2 as i32),
                            Arg::I32(tt as i32),
                            Arg::I32(1),
                            Arg::I32(self.p.tau_max as i32),
                            Arg::I32(-1),
                            Arg::I32(1),
                            Arg::I32(nlev as i32),
                            Arg::I32(c.d as i32),
                            t.arg(),
                        ],
                    )?;
                }
            }
        }
        self.cum.as_mut().unwrap().n += 1;
        Ok(())
    }

    /// Cumul sur la carte : remonte les sommes (une fois par configuration externe).
    pub fn cumul_lire(&mut self) -> Resultat<SommesCarte> {
        self.app.synchroniser()?;
        let (nch, nf, nq) = (self.nch, self.nf, self.nq);
        let tt = self.p.cotes[self.p.temps];
        let nlev = self.nlev_tot();
        let nk = self.p.ks.len();
        let npair = self.npaires();
        let ntau = self.p.tau_max + 1;
        let c = self.cum.as_ref().expect("cumul_init");
        let mut s = vec![0f64; nch * nlev * nf * tt * nq];
        self.app.lire(&c.s, 0, &mut s)?;
        let mut o = vec![0f64; nch * (1 + nk) * npair * ntau];
        self.app.lire(&c.o, 0, &mut o)?;
        let mut reps = Vec::new();
        if let Some(s2t) = &c.s2 {
            let mut s2 = vec![0f64; nch * nlev * self.nf2 * tt];
            self.app.lire(s2t, 0, &mut s2)?;
            for (r, rows) in self.reps.iter().zip(&c.o2) {
                let np2 = r.nops * (r.nops + 1) / 2;
                let mut v = Vec::new();
                for (row, t) in rows.iter().enumerate() {
                    let mut oi = vec![0i32; r.nops];
                    self.app.lire(&r.opidx[row], 0, &mut oi)?;
                    let mut sm = vec![0f64; nch * r.nops * tt];
                    for ch in 0..nch {
                        for (a, &oo) in oi.iter().enumerate() {
                            for tm in 0..tt {
                                sm[(ch * r.nops + a) * tt + tm] = s2[((ch * nlev * self.nf2) + oo as usize) * tt + tm];
                            }
                        }
                    }
                    let mut ord = vec![0f64; nch * np2 * ntau];
                    self.app.lire(t, 0, &mut ord)?;
                    v.push((sm, ord));
                }
                reps.push(v);
            }
        }
        Ok(SommesCarte { n: c.n, s, o, reps })
    }

    /// Somme brute `sg` du dernier calcul d'opérateurs (ancienne voie hôte avec impulsions).
    pub fn sg_hote(&mut self) -> Resultat<Vec<f64>> {
        self.app.synchroniser()?;
        let mut sgv = vec![0f64; self.nch * self.nlev_tot() * self.nf * self.p.cotes[self.p.temps] * self.nq];
        self.app.lire(&self.sg, 0, &mut sgv)?;
        Ok(sgv)
    }

    /// Mesure TOUTES les chaînes sur leur champ courant.
    #[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
    pub fn mesurer(&mut self) -> Resultat<Vec<MesCh>> {
        self.calc_operateurs()?;
        let p = &self.p;
        let (nch, v, nf, nq) = (self.nch, self.v, self.nf, self.nq);
        let tt = p.cotes[p.temps];
        let nlev = p.niveaux.len() + p.bloc.niveaux.len();
        let nk = p.ks.len();
        let nsl = 1 + nk + p.orbites.len();
        let geo = self.boite.arg_geo();
        let nops = self.nops();
        let npair = self.npaires();
        let ntau = p.tau_max + 1;
        for slot in -1..nk as i32 {
            self.app.lancer(
                &self.f_corr,
                &Lancement::lineaire_lot(npair * ntau, nch, 128),
                &[
                    self.sg.arg(),
                    self.opidx.arg(),
                    Arg::I32(nops as i32),
                    Arg::I32(nf as i32),
                    Arg::I32(tt as i32),
                    Arg::I32(nq as i32),
                    Arg::I32(p.tau_max as i32),
                    Arg::I32(slot),
                    Arg::I32(nsl as i32),
                    Arg::I32(nlev as i32),
                    self.out_corr.arg(),
                ],
            )?;
        }
        let mut v0 = 0usize;
        for (io, o) in p.orbites.iter().enumerate() {
            self.app.lancer(
                &self.f_corr_orb,
                &Lancement::lineaire_lot(npair * ntau, nch, 128),
                &[
                    self.sg.arg(),
                    self.opidx.arg(),
                    Arg::I32(nops as i32),
                    Arg::I32(nf as i32),
                    Arg::I32(tt as i32),
                    Arg::I32(nq as i32),
                    Arg::I32(p.tau_max as i32),
                    Arg::I32((1 + nk + io) as i32),
                    Arg::I32(nsl as i32),
                    Arg::I32(nlev as i32),
                    Arg::I32(v0 as i32),
                    Arg::I32(o.vecteurs.len() as i32),
                    self.out_corr.arg(),
                ],
            )?;
            v0 += o.vecteurs.len();
        }
        self.app.lancer(
            &self.f_moy,
            &Lancement::lineaire_lot(nops, nch, 64),
            &[
                self.sg.arg(),
                self.opidx.arg(),
                Arg::I32(nops as i32),
                Arg::I32(nf as i32),
                Arg::I32(tt as i32),
                Arg::I32(nq as i32),
                Arg::I32(nlev as i32),
                self.out_moy.arg(),
            ],
        )?;
        if let Some(sg2) = &self.sg2 {
            for r in &self.reps {
                let npair2 = r.nops * (r.nops + 1) / 2;
                for row in 0..r.dim {
                    self.app.lancer(
                        &self.f_corr,
                        &Lancement::lineaire_lot(npair2 * ntau, nch, 128),
                        &[
                            sg2.arg(),
                            r.opidx[row].arg(),
                            Arg::I32(r.nops as i32),
                            Arg::I32(self.nf2 as i32),
                            Arg::I32(tt as i32),
                            Arg::I32(1),
                            Arg::I32(p.tau_max as i32),
                            Arg::I32(-1),
                            Arg::I32(1),
                            Arg::I32(nlev as i32),
                            r.out_corr[row].arg(),
                        ],
                    )?;
                    self.app.lancer(
                        &self.f_moy,
                        &Lancement::lineaire_lot(r.nops, nch, 64),
                        &[
                            sg2.arg(),
                            r.opidx[row].arg(),
                            Arg::I32(r.nops as i32),
                            Arg::I32(self.nf2 as i32),
                            Arg::I32(tt as i32),
                            Arg::I32(1),
                            Arg::I32(nlev as i32),
                            r.out_moy[row].arg(),
                        ],
                    )?;
                }
            }
        }
        self.app.lancer(
            &self.f_plaq,
            &Lancement::lineaire_lot(v, nch, 256),
            &[self.boite.arg_champ_complet(), geo, self.out_plaq.arg()],
        )?;
        if let Some(pa) = &self.plaq_an {
            pa.lancer(&self.app, self.boite.arg_champ_complet(), geo, p.temps, v, nch)?;
        }
        self.app.synchroniser()?;
        let plaq_st = match &self.plaq_an {
            Some(pa) => Some(pa.lire(&self.app, nch, v)?),
            None => None,
        };
        let mut corr = vec![0f64; nch * nsl * npair * ntau];
        self.app.lire(&self.out_corr, 0, &mut corr)?;
        let mut moy = vec![0f64; nch * nops];
        self.app.lire(&self.out_moy, 0, &mut moy)?;
        let mut pl = vec![0f64; nch * self.nblk_plaq];
        self.app.lire(&self.out_plaq, 0, &mut pl)?;
        let bloc = npair * ntau;
        // `--ops-bruts` : remontée des sommes brutes O_a(p,t) (`sg`, intact : seul `k_operateurs` l'écrit, `k_corr` le lit).
        // Source : sg[((((c*nlev + lev)*nf + f)*tt + t)*nq + q] ; cible par chaîne : [t][p][a][re,im], a = lev*nf + f.
        // Le noyau stocke Σ·e^{+iθ} (q = 1 + ik*6 + 2*axe + {0:cos, 1:sin}) ; on écrit e^{-ip·x} : p = +k·e_ax ⇒ conjugué.
        let mut ops_par_chaine: Vec<Vec<f64>> = vec![Vec::new(); nch];
        if p.ops_bruts {
            let mut sgv = vec![0f64; nch * nlev * nf * tt * nq];
            self.app.lire(&self.sg, 0, &mut sgv)?;
            let nimp = 1 + 6 * nk;
            for c in 0..nch {
                let mut o = vec![0f64; tt * nimp * nops * 2];
                for lev in 0..nlev {
                    for f in 0..nf {
                        let a = lev * nf + f;
                        for t in 0..tt {
                            let s = (((c * nlev + lev) * nf + f) * tt + t) * nq;
                            let dst = |ip: usize| ((t * nimp + ip) * nops + a) * 2;
                            o[dst(0)] = sgv[s];
                            for ik in 0..nk {
                                for axe in 0..3 {
                                    let q = 1 + ik * 6 + axe * 2;
                                    let (re, im) = (sgv[s + q], sgv[s + q + 1]);
                                    let ip = 1 + ik * 6 + axe * 2;
                                    o[dst(ip)] = re;
                                    o[dst(ip) + 1] = -im;
                                    o[dst(ip + 1)] = re;
                                    o[dst(ip + 1) + 1] = im;
                                }
                            }
                        }
                    }
                }
                ops_par_chaine[c] = o;
            }
        }
        // représentations : moyenne sur les lignes
        let mut rep_c: Vec<Vec<f64>> = Vec::new();
        let mut rep_m: Vec<Vec<f64>> = Vec::new();
        for r in &self.reps {
            let npair2 = r.nops * (r.nops + 1) / 2;
            let mut c = vec![0f64; nch * npair2 * ntau];
            let mut mo = vec![0f64; nch * r.nops];
            for row in 0..r.dim {
                let mut tc = vec![0f64; c.len()];
                self.app.lire(&r.out_corr[row], 0, &mut tc)?;
                let mut tm = vec![0f64; mo.len()];
                self.app.lire(&r.out_moy[row], 0, &mut tm)?;
                for (x, y) in c.iter_mut().zip(&tc) {
                    *x += y / r.dim as f64;
                }
                for (x, y) in mo.iter_mut().zip(&tm) {
                    *x += y / r.dim as f64;
                }
            }
            rep_c.push(c);
            rep_m.push(mo);
        }
        Ok((0..nch)
            .map(|c| {
                let base = c * nsl * bloc;
                MesCh {
                    corr: corr[base..base + bloc].to_vec(),
                    moy: moy[c * nops..(c + 1) * nops].to_vec(),
                    corr_ps: (0..nk).map(|ik| corr[base + (1 + ik) * bloc..base + (2 + ik) * bloc].to_vec()).collect(),
                    corr_orb: (0..p.orbites.len()).map(|io| corr[base + (1 + nk + io) * bloc..base + (2 + nk + io) * bloc].to_vec()).collect(),
                    reps: self
                        .reps
                        .iter()
                        .enumerate()
                        .map(|(i, r)| {
                            let npair2 = r.nops * (r.nops + 1) / 2;
                            (rep_c[i][c * npair2 * ntau..(c + 1) * npair2 * ntau].to_vec(), rep_m[i][c * r.nops..(c + 1) * r.nops].to_vec())
                        })
                        .collect(),
                    ops: std::mem::take(&mut ops_par_chaine[c]),
                    plaq: pl[c * self.nblk_plaq..(c + 1) * self.nblk_plaq].iter().sum::<f64>() / (6.0 * v as f64),
                    plaq_st: plaq_st.as_ref().map(|x| x[c]),
                }
            })
            .collect())
    }
}
