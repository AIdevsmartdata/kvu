//! Exécution d'une cellule de mesure : configuration de départ (fichier ou génération de Wilson par le MÊME échantillonneur sans
//! contrainte), figeage de V, deux répliques indépendantes, sorties (NPY + JSON). Voir `cond3::cli` pour la ligne de commande.

use super::geom::Geom;
use super::mesure::{executer_replique, sommes_croisees, ConfigMesure, SortieReplique};
use super::modele::{avancer, Modele, Ordonnancement};
use super::paires::{Paires, B_COMPTE};
use super::su2::{FibreSu2, MutantSu2};
use super::u1::FibreU1;
use crate::commun::{Erreur, Resultat};
use crate::heritage::su2_hb::Q;
use crate::stockage::{npy, GardeSsd};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Groupe {
    Su2,
    U1,
}

#[derive(Clone, Debug)]
pub enum Depart {
    /// Configuration SU(2) d = 4 lue dans un NPY (c16 ou q64).
    Fichier(PathBuf),
    /// Génération par l'échantillonneur SANS contrainte : départ froid, `therm` pas, puis `index` × `espacement` pas.
    Wilson { therm: usize, espacement: usize, index: usize, graine: u64 },
}

#[derive(Clone, Debug)]
pub struct ParamsCellule {
    pub groupe: Groupe,
    pub d: usize,
    pub l: usize,
    pub beta: f64,
    pub depart: Depart,
    /// Graines des répliques A et B.
    pub graines: [u64; 2],
    /// Nombre de répliques exécutées (1 ou 2).
    pub nrep: usize,
    pub faces: FacesChoix,
    pub sondes: SondesChoix,
    pub bmin: u32,
    pub bmax: u32,
    pub mesure: ConfigMesure,
    /// Nombre de plaquettes (de P_obs) dont on garde la série, et de sondes (toutes si `usize::MAX`).
    pub n_series_faces: usize,
    pub n_series_sondes: usize,
    pub nchunk_croise: usize,
    pub mutant: Option<String>,
    pub sans_renorm: bool,
    /// Jauge fine de Haar appliquée à U⁰ (graine) : U(x, μ) → g(x) U(x, μ) g(x + μ)† (oracle O-10a).
    pub jauge: Option<u64>,
    /// Réduction de production (sommes par tranche et histogrammes de charge de tube) au lieu des accumulateurs complets.
    pub reduire: bool,
    /// Bords des classes de charge de tube (NPY f64 [tranches, nbin − 1]).
    pub tube_edges: Option<PathBuf>,
    /// Backend de calcul : `cpu` (référence rayon), `hip`, `cuda` ou `auto` (SU(2), d = 4, hors mutants et canaux).
    pub backend: String,
    /// Écrit le champ final de chaque réplique (`A_champ.npy`, `B_champ.npy`, f64 [4V, 4]).
    pub champ_final: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FacesChoix {
    /// Plaquettes de classe t = d − 2.
    Faces,
    /// Toutes les plaquettes.
    Toutes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SondesChoix {
    Paires,
    Toutes,
}

fn ecrire_json(chemin: &Path, v: &Value) -> Resultat<()> {
    std::fs::write(chemin, serde_json::to_string_pretty(v).unwrap() + "\n").map_err(|e| Erreur::io(format!("écriture {}", chemin.display()), e))
}

/// Configuration SU(2) de départ (fichier ou Wilson, jauge éventuelle) et informations d'entrée.
pub fn depart_su2(prm: &ParamsCellule, g: &Arc<Geom>) -> Resultat<(Vec<Q>, Value)> {
    let mut info = json!({});
    let u0: Vec<Q> = match &prm.depart {
        Depart::Fichier(p) => {
            if prm.d != 4 {
                return Err(Erreur::Config("un fichier de configuration SU(2) n'existe qu'en d = 4".into()));
            }
            let c = npy::lire_champ(p)?;
            if c.l != prm.l {
                return Err(Erreur::Config(format!("{} : L = {} au lieu de {}", p.display(), c.l, prm.l)));
            }
            info["sha256_cfg"] = json!(crate::heritage::manifest::sha256_fichier(p).map_err(|e| Erreur::Config(format!("sha256 : {e}")))?);
            c.q.chunks_exact(4).map(|q| [q[0], q[1], q[2], q[3]]).collect()
        }
        Depart::Wilson { therm, espacement, index, graine } => {
            let mut f = FibreSu2::nouveau(g.clone(), super::su2::froid(g.nliens()), prm.beta, *graine);
            let pas = therm + index * espacement;
            for t in 0..pas {
                avancer(&mut f, Ordonnancement::Libre, t as u32, prm.mesure.n_or);
            }
            f.u.clone()
        }
    };
    let u0 = match prm.jauge {
        Some(sg) => appliquer_jauge(g, &u0, sg),
        None => u0,
    };
    Ok((u0, info))
}

/// Configuration de départ + modèles de répliques (même U⁰, V figé, flux distincts).
pub fn preparer_modeles(prm: &ParamsCellule, g: &Arc<Geom>) -> Resultat<(Vec<Box<dyn Modele>>, Value)> {
    let nrep = prm.nrep;
    let mut modeles: Vec<Box<dyn Modele>> = Vec::new();
    let mut info = json!({});
    match prm.groupe {
        Groupe::Su2 => {
            let (u0, inf) = depart_su2(prm, g)?;
            info = inf;
            for r in 0..nrep {
                let mut f = FibreSu2::nouveau(g.clone(), u0.clone(), prm.beta, prm.graines[r]);
                match prm.mutant.as_deref() {
                    None | Some("aucun") => {}
                    Some("agrafe") => f.mutant = MutantSu2::Agrafe,
                    Some("u3") => f.mutant = MutantSu2::U3NonRecalcule,
                    Some("releve-sans-rotation") => f.mutant = MutantSu2::ReleveSansRotation,
                    Some("derive-f32") => {}
                    Some(autre) => return Err(Erreur::Config(format!("mutant inconnu : {autre}"))),
                }
                if prm.sans_renorm {
                    f.renormaliser = false;
                }
                if prm.mutant.as_deref() == Some("derive-f32") {
                    f.renormaliser = false;
                    f.arrondi_f32 = true;
                }
                modeles.push(Box::new(f));
            }
        }
        Groupe::U1 => {
            let th0: Vec<f64> = match &prm.depart {
                Depart::Fichier(_) => return Err(Erreur::Config("U(1) : pas de fichier de configuration, utiliser --wilson".into())),
                Depart::Wilson { therm, espacement, index, graine } => {
                    let mut f = FibreU1::nouveau(g.clone(), vec![0.0; g.nliens()], prm.beta, *graine);
                    let pas = therm + index * espacement;
                    for t in 0..pas {
                        avancer(&mut f, Ordonnancement::Libre, t as u32, prm.mesure.n_or);
                    }
                    f.th.clone()
                }
            };
            for r in 0..nrep {
                modeles.push(Box::new(FibreU1::nouveau(g.clone(), th0.clone(), prm.beta, prm.graines[r])));
            }
        }
    }
    Ok((modeles, info))
}

/// Jauge fine de Haar g(x) (graine) : U'(x, μ) = g(x) U(x, μ) g(x + μ)†.
pub fn appliquer_jauge(g: &Geom, u: &[Q], graine: u64) -> Vec<Q> {
    let gj = super::su2::champ_haar_test(g.v, graine);
    let mut out = u.to_vec();
    for mu in 0..g.d {
        for s in 0..g.v {
            let sn = g.nbr[mu][s] as usize;
            out[mu * g.v + s] = crate::heritage::su2_hb::qmul(crate::heritage::su2_hb::qmul(gj[s], u[mu * g.v + s]), crate::heritage::su2_hb::qdag(gj[sn]));
        }
    }
    out
}

pub fn selection_paires(prm: &ParamsCellule, g: &Geom) -> Paires {
    let faces = match prm.faces {
        FacesChoix::Faces => g.faces(),
        FacesChoix::Toutes => (0..g.nplaq() as u32).collect(),
    };
    let sondes = match prm.sondes {
        SondesChoix::Paires => g.sondes_paires(),
        SondesChoix::Toutes => g.tous_liens_gros(),
    };
    Paires::nouvelles(g, faces, sondes, prm.bmin, prm.bmax)
}

/// Écrit les sorties d'une réplique (NPY) sous le préfixe `nom`.
fn ecrire_replique(dir: &Path, nom: &str, s: &SortieReplique, garde: &GardeSsd) -> Resultat<()> {
    let nb: Vec<u32> = s.n_blk.iter().map(|&x| x as u32).collect();
    npy::ecrire_u32(&dir.join(format!("{nom}_nblk.npy")), &[s.nblk], &nb, garde)?;
    npy::ecrire_f64(&dir.join(format!("{nom}_so.npy")), &[s.nblk, s.nfaces], &s.so, garde)?;
    npy::ecrire_f64(&dir.join(format!("{nom}_ss.npy")), &[s.nblk, s.nsondes, s.nc], &s.ss, garde)?;
    npy::ecrire_f64(&dir.join(format!("{nom}_sos.npy")), &[s.nblk, s.npaires, s.nc], &s.sos, garde)?;
    if !s.px.is_empty() {
        npy::ecrire_f64(&dir.join(format!("{nom}_sx.npy")), &[s.nblk, s.px.len(), s.nc], &s.sx, garde)?;
    }
    if s.n_serie_o > 0 {
        npy::ecrire_f64(&dir.join(format!("{nom}_serie_o.npy")), &[s.n_serie_o, s.n_mes], &s.serie_o, garde)?;
    }
    if s.n_serie_s > 0 {
        npy::ecrire_f32(&dir.join(format!("{nom}_serie_s.npy")), &[s.n_serie_s, s.n_mes], &s.serie_s, garde)?;
    }
    Ok(())
}

fn resume_replique(s: &SortieReplique) -> Value {
    let (to, ts) = s.taus();
    let maxf = |v: &[f64]| v.iter().cloned().fold(f64::NAN, f64::max);
    let ctrl_max = |f: &dyn Fn(&super::mesure::Controle) -> f64| s.controles.iter().map(f).fold(0.0f64, |a, b| if b.is_nan() { a } else { a.max(b) });
    json!({
        "empreinte_finale": s.empreinte_finale,
        "jalons": s.jalons.iter().map(|(t, h)| json!([t, h])).collect::<Vec<_>>(),
        "secondes": s.secondes,
        "majs": s.stats.majs, "essais": s.stats.essais, "plafond": s.stats.plafond,
        "essais_par_maj": if s.stats.majs > 0 { s.stats.essais as f64 / s.stats.majs as f64 } else { 0.0 },
        "n_mes": s.n_mes,
        "controle_contrainte_max": ctrl_max(&|c| c.contrainte),
        "controle_unitarite_max": ctrl_max(&|c| c.unitarite),
        "controle_ward_max": ctrl_max(&|c| c.ward),
        "n_controles": s.controles.len(),
        "tau_o_max": maxf(&to), "tau_o_med": mediane(&to),
        "tau_s_max": maxf(&ts), "tau_s_med": mediane(&ts),
        "moy_classes_t": s.moy_classes_t,
    })
}

fn mediane(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut w = v.to_vec();
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    w[w.len() / 2]
}

/// Contrôles de cohérence des paramètres, AVANT tout calcul et toute écriture (relecture indépendante : graines A = B, `--reduire` mal paramétré,
/// options SU(2) demandées en U(1), séquentiel avec mutants non pris en charge, tranche maximale hors du tableau d'effectifs).
pub fn valider_parametres(prm: &ParamsCellule) -> Resultat<()> {
    let cfg = |msg: String| Err(Erreur::Config(msg));
    if prm.nrep == 0 || prm.nrep > 2 {
        return cfg(format!("nombre de répliques {} : 1 ou 2 attendu", prm.nrep));
    }
    if prm.nrep == 2 && prm.graines[0] == prm.graines[1] {
        return cfg(format!("graines A et B identiques ({}) : les deux répliques seraient identiques et Ŝ = R^A·R^B serait biaisé (SPEC §2.6)", prm.graines[0]));
    }
    if prm.reduire {
        if prm.nrep != 2 {
            return cfg("--reduire exige deux répliques (--graine-b)".into());
        }
        if prm.mesure.nblk != 2 {
            return cfg(format!("--reduire exige nblk = 2 (reçu {})", prm.mesure.nblk));
        }
        if prm.bmin < 2 {
            return cfg(format!("--reduire exige bmin ⩾ 2 (reçu {} : le terme E[X(O)] n'est pas nul pour b < 2 et la réduction l'ignore)", prm.bmin));
        }
    }
    if prm.nchunk_croise > 0 && !prm.mesure.n_mes.is_multiple_of(prm.nchunk_croise) {
        return cfg(format!("--chunks {} ne divise pas n_mes = {} (corrélation croisée A/B par chunks égaux)", prm.nchunk_croise, prm.mesure.n_mes));
    }
    if prm.groupe == Groupe::U1 && (prm.jauge.is_some() || prm.mutant.is_some() || prm.sans_renorm) {
        return cfg("U(1) : --jauge, --mutant et --sans-renorm ne sont pas pris en charge (SU(2) seulement)".into());
    }
    if prm.bmin > prm.bmax || prm.bmax as usize >= B_COMPTE {
        return cfg(format!("tranches bmin = {}, bmax = {} : 0 ⩽ bmin ⩽ bmax ⩽ {} attendu", prm.bmin, prm.bmax, B_COMPTE - 1));
    }
    // le balayage séquentiel (oracle O-2b) ignore les mutants U3NonRecalcule et de dérive : on le refuse plutôt que de tester autre chose que demandé
    if prm.mesure.ord == Ordonnancement::Sequentiel && (prm.sans_renorm || matches!(prm.mutant.as_deref(), Some("u3") | Some("derive-f32"))) {
        return cfg("ordonnancement séquentiel : les mutants u3 et de dérive (--sans-renorm, derive-f32) ne sont pas pris en charge".into());
    }
    Ok(())
}

/// Exécute la cellule et écrit les sorties dans `dir` (créé, refusé s'il contient déjà `meta.json`).
pub fn executer_cellule(prm: &ParamsCellule, g: &Arc<Geom>, p: &Paires, dir: &Path, garde: &GardeSsd, ecrire_acc: bool) -> Resultat<Value> {
    valider_parametres(prm)?;
    if dir.join("meta.json").exists() {
        return Err(Erreur::Config(format!("{} contient déjà meta.json (jamais d'écrasement)", dir.display())));
    }
    std::fs::create_dir_all(dir).map_err(|e| Erreur::io(format!("création {}", dir.display()), e))?;
    let mut cfg = prm.mesure.clone();
    // séries : plaquettes régulièrement espacées dans la liste de P_obs, sondes : les premières
    let nsf = prm.n_series_faces.min(p.faces.len());
    cfg.series_faces = (0..nsf).map(|k| k * p.faces.len() / nsf.max(1)).collect();
    cfg.series_sondes = (0..prm.n_series_sondes.min(p.sondes.len())).collect();
    let (mut modeles, info) = preparer_modeles(prm, g)?;
    let mut sorties: Vec<SortieReplique> = Vec::new();
    // backend : GPU seulement pour SU(2) d = 4 sans mutant ni canal d'oracle ; `auto` retombe sur le CPU, un backend demandé explicitement refuse
    let gpu_app = if prm.backend == "cpu" || prm.groupe != Groupe::Su2 || prm.d != 4 {
        if prm.backend != "cpu" && prm.backend != "auto" {
            return Err(Erreur::Config(format!("backend {} : SU(2) d = 4 seulement", prm.backend)));
        }
        None
    } else {
        crate::cond::cli::ouvrir_backend(&prm.backend)?
    };
    let compatible = prm.mutant.is_none()
        && !prm.sans_renorm
        && !cfg.separes
        && cfg.canaux.is_empty()
        && p.px.is_empty()
        && cfg.ord == Ordonnancement::Simultane;
    if gpu_app.is_some() && !compatible && prm.backend != "auto" {
        return Err(Erreur::Config("GPU : scores moyens, ordonnancement simultané, fenêtre b ⩾ 2, ni mutant ni canal".into()));
    }
    let gpu_app = if compatible { gpu_app } else { None };
    let _verrou = crate::cond::ressources::prendre_verrou(gpu_app.as_ref())?;
    let mut backend_effectif = "cpu".to_string();
    let mut champs_finaux: Vec<Vec<f64>> = Vec::new();
    let mut sha_code_gpu = Value::Null;
    if let (Some(app), Groupe::Su2) = (gpu_app.as_ref(), &prm.groupe) {
        let (u0, _) = depart_su2(prm, g)?;
        for r in 0..prm.nrep {
            let cpu = FibreSu2::nouveau(g.clone(), u0.clone(), prm.beta, prm.graines[r]);
            let mut mg = super::gpu::FibreSu2Gpu::depuis_cpu(app, cpu, prm.graines[r])?;
            sorties.push(super::gpu::executer_replique_gpu(&mut mg, p, &cfg)?);
            champs_finaux.push(mg.miroir.champ_plat());
            backend_effectif = format!("{} ({})", mg.backend.nom(), mg.arch);
            sha_code_gpu = json!(mg.provenance_sha256_code);
        }
    } else {
        for m in modeles.iter_mut() {
            sorties.push(executer_replique(&mut **m, p, &cfg)?);
            champs_finaux.push(m.champ_plat());
        }
    }
    // sorties
    let mut meta = json!({
        "schema": "kvu-cond3-cellule/1",
        "groupe": match prm.groupe { Groupe::Su2 => "su2", Groupe::U1 => "u1" },
        "d": prm.d, "l": prm.l, "beta": prm.beta,
        "graines": prm.graines, "nrep": prm.nrep,
        "n_dec": cfg.n_dec, "n_mes": cfg.n_mes, "n_or": cfg.n_or, "nblk": cfg.nblk, "lot": cfg.lot, "separes": cfg.separes,
        "ordonnancement": format!("{:?}", cfg.ord),
        "bmin": prm.bmin, "bmax": prm.bmax,
        "nfaces": p.faces.len(), "nsondes": p.sondes.len(), "npaires": p.npaires(),
        "effectifs_paires": p.effectifs[..=(prm.bmax as usize + 2).min(p.effectifs.len() - 1)].to_vec(),
        "backend": backend_effectif,
        "sha256_code_gpu": sha_code_gpu,
        "mutant": prm.mutant,
        "jauge": prm.jauge,
        "canaux": cfg.canaux.iter().map(|c| format!("{c:?}")).collect::<Vec<_>>(),
        "depart": format!("{:?}", prm.depart),
        "entrees": info,
        "repliques": sorties.iter().map(resume_replique).collect::<Vec<_>>(),
    });
    npy::ecrire_u32(&dir.join("p_faces.npy"), &[p.faces.len()], &p.faces, garde)?;
    npy::ecrire_u32(&dir.join("p_sondes.npy"), &[p.sondes.len()], &p.sondes, garde)?;
    let debut: Vec<u32> = p.debut.iter().map(|&x| u32::try_from(x).map_err(|_| Erreur::Config("p_debut.npy : plus de 2^32 paires (L trop grand pour ce format)".into()))).collect::<Resultat<_>>()?;
    npy::ecrire_u32(&dir.join("p_debut.npy"), &[debut.len()], &debut, garde)?;
    npy::ecrire_u32(&dir.join("p_face_loc.npy"), &[p.face_loc.len()], &p.face_loc, garde)?;
    let tr: Vec<u32> = p.tranche.iter().map(|&x| x as u32).collect();
    npy::ecrire_u32(&dir.join("p_tranche.npy"), &[tr.len()], &tr, garde)?;
    if !p.px.is_empty() {
        npy::ecrire_u32(&dir.join("p_px.npy"), &[p.px.len()], &p.px, garde)?;
    }
    if prm.champ_final {
        for (i, c) in champs_finaux.iter().enumerate() {
            let n = c.len() / if prm.groupe == Groupe::Su2 { 4 } else { 1 };
            let forme: Vec<usize> = if prm.groupe == Groupe::Su2 { vec![n, 4] } else { vec![n] };
            npy::ecrire_f64(&dir.join(format!("{}_champ.npy", if i == 0 { "A" } else { "B" })), &forme, c, garde)?;
        }
    }
    {
        let vf = modeles[0].v_fige_plat();
        let nv = vf.len() / g.ngros();
        npy::ecrire_f64(&dir.join("v_fige.npy"), &[g.ngros(), nv], &vf, garde)?;
        let c0 = modeles[0].champ_plat();
        let _ = c0;
    }
    if prm.reduire {
        if sorties.len() != 2 {
            return Err(Erreur::Config("--reduire exige deux répliques".into()));
        }
        let (bins, fval, nbin, tube_info) = match &prm.tube_edges {
            Some(chemin) => {
                let (forme, edges) = npy::lire_f64(chemin)?;
                if forme.len() != 2 || forme[0] < p.bmax as usize + 1 {
                    return Err(Erreur::Config(format!("{} : forme {:?} inattendue", chemin.display(), forme)));
                }
                let nbin = forme[1] + 1;
                let vf = modeles[0].v_fige_plat();
                let v: Vec<Q> = vf.chunks_exact(4).map(|q| [q[0], q[1], q[2], q[3]]).collect();
                let tt = super::tube::TableTubes::nouvelle(g);
                let cp = super::tube::charges_grossieres(g, &v);
                let f = super::tube::charges_paires(g, p, &tt, &cp);
                let mut bins = vec![0u16; f.len()];
                for q in 0..f.len() {
                    let b = p.tranche[q] as usize;
                    let e = &edges[b * forme[1]..(b + 1) * forme[1]];
                    bins[q] = e.partition_point(|&x| x <= f[q] as f64) as u16;
                }
                let fmin = f.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
                let fmax = f.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                (Some(bins), Some(f), nbin, json!({"f_min": fmin, "f_max": fmax}))
            }
            None => (None, None, 1, json!(null)),
        };
        let red = super::reduction::reduire(&sorties[0], &sorties[1], p, nbin, bins.as_deref(), fval.as_deref());
        let rj = json!({
            "schema": "kvu-cond3-reduction/1",
            "bmin": p.bmin, "bmax": p.bmax, "nbin": nbin,
            "n_paires": red.n_paires, "s_full": red.s_full, "s_early": red.s_early, "s_late": red.s_late, "a2": red.a2, "b2": red.b2,
            "tube": tube_info,
            "moyenne_o_faces_A": sorties[0].f_chapeau(&[0, 1]).iter().sum::<f64>() / sorties[0].nfaces as f64,
        });
        ecrire_json(&dir.join("reduction.json"), &rj)?;
        if !red.hist_n.is_empty() {
            let nb = p.bmax as usize + 1;
            let mut h = Vec::with_capacity(3 * nb * nbin);
            h.extend(red.hist_n.iter().map(|&x| x as f64));
            h.extend_from_slice(&red.hist_s);
            h.extend_from_slice(&red.hist_f);
            npy::ecrire_f64(&dir.join("reduction_hist.npy"), &[3, nb, nbin], &h, garde)?;
        }
    }
    if ecrire_acc {
        for (i, s) in sorties.iter().enumerate() {
            ecrire_replique(dir, if i == 0 { "A" } else { "B" }, s, garde)?;
        }
    }
    for (i, s) in sorties.iter().enumerate() {
        if s.n_canal > 0 {
            let nom = if i == 0 { "A" } else { "B" };
            npy::ecrire_f64(&dir.join(format!("{nom}_canaux.npy")), &[s.n_canal, s.n_mes], &s.serie_canaux, garde)?;
        }
    }
    if sorties.len() == 2 && sorties[0].n_serie_o + sorties[0].n_serie_s > 0 && prm.nchunk_croise > 0 {
        let c = sommes_croisees(&sorties[0], &sorties[1], prm.nchunk_croise);
        let nser = sorties[0].n_serie_o + sorties[0].n_serie_s;
        npy::ecrire_f64(&dir.join("croisees.npy"), &[prm.nchunk_croise, nser, 5], &c, garde)?;
        if !ecrire_acc {
            // les séries sont nécessaires à l'analyse de G-DECORR (c) : tau par série, déjà dans meta ; on les garde en NPY légers
            for (i, s) in sorties.iter().enumerate() {
                let nom = if i == 0 { "A" } else { "B" };
                if s.n_serie_o > 0 {
                    npy::ecrire_f64(&dir.join(format!("{nom}_serie_o.npy")), &[s.n_serie_o, s.n_mes], &s.serie_o, garde)?;
                }
            }
        }
    }
    ecrire_json(&dir.join("meta.json"), &meta)?;
    meta["dir"] = json!(dir.display().to_string());
    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> ParamsCellule {
        ParamsCellule {
            groupe: Groupe::Su2,
            d: 4,
            l: 6,
            beta: 2.5,
            depart: Depart::Wilson { therm: 2, espacement: 1, index: 0, graine: 1 },
            graines: [11, 12],
            nrep: 2,
            faces: FacesChoix::Faces,
            sondes: SondesChoix::Paires,
            bmin: 2,
            bmax: 4,
            mesure: ConfigMesure { nblk: 2, ..ConfigMesure::default() },
            n_series_faces: 0,
            n_series_sondes: 0,
            nchunk_croise: 0,
            mutant: None,
            sans_renorm: false,
            jauge: None,
            reduire: true,
            tube_edges: None,
            backend: "cpu".into(),
            champ_final: false,
        }
    }

    fn refuse(prm: &ParamsCellule, morceau: &str) {
        match valider_parametres(prm) {
            Err(Erreur::Config(m)) => assert!(m.contains(morceau), "message inattendu : {m}"),
            autre => panic!("refus attendu ({morceau}), obtenu {autre:?}"),
        }
    }

    #[test]
    fn parametres_valides_acceptes() {
        assert!(valider_parametres(&base()).is_ok());
    }

    #[test]
    fn graines_a_et_b_identiques_refusees() {
        let mut p = base();
        p.graines = [7, 7];
        refuse(&p, "identiques");
        p.nrep = 1; // une seule réplique : l'égalité des graines est sans objet
        p.reduire = false;
        assert!(valider_parametres(&p).is_ok());
    }

    #[test]
    fn reduire_exige_deux_repliques_nblk_2_et_bmin_2() {
        let mut p = base();
        p.nrep = 1;
        refuse(&p, "deux répliques");
        let mut p = base();
        p.mesure.nblk = 1;
        refuse(&p, "nblk = 2");
        let mut p = base();
        p.bmin = 1;
        refuse(&p, "bmin ⩾ 2");
    }

    #[test]
    fn options_su2_refusees_en_u1_et_tranches_hors_tableau_refusees() {
        let mut p = base();
        p.groupe = Groupe::U1;
        p.reduire = false;
        p.jauge = Some(3);
        refuse(&p, "U(1)");
        let mut p = base();
        p.bmax = B_COMPTE as u32;
        refuse(&p, "bmax");
        let mut p = base();
        p.bmin = 5;
        refuse(&p, "bmax");
    }

    #[test]
    fn chunks_doit_diviser_n_mes() {
        let mut p = base();
        p.mesure.n_mes = 1000;
        p.nchunk_croise = 40;
        assert!(valider_parametres(&p).is_ok());
        p.nchunk_croise = 37;
        refuse(&p, "--chunks");
        p.nchunk_croise = 0;
        assert!(valider_parametres(&p).is_ok(), "0 = aucune corrélation croisée");
    }

    #[test]
    fn sequentiel_refuse_les_mutants_non_pris_en_charge() {
        let mut p = base();
        p.reduire = false;
        p.mesure.ord = Ordonnancement::Sequentiel;
        p.mutant = Some("u3".into());
        refuse(&p, "séquentiel");
        p.mutant = None;
        p.sans_renorm = true;
        refuse(&p, "séquentiel");
        p.sans_renorm = false;
        p.mutant = Some("agrafe".into());
        assert!(valider_parametres(&p).is_ok(), "le mutant agrafe est pris en charge en séquentiel");
    }
}
