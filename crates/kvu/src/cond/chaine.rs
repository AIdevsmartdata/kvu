//! Chaîne conditionnelle complète (= cb.chaine et chemin principal de chaine_v3.py) : thermalisation,
//! balayages, F tous les `every`, D toutes les `deach` mesures, contrôles Vdev/unitarité, points de reprise
//! atomiques (champ, état de l'aléa, accumulateurs). Propriétaire : T5.
//!
//! Cadence de chaine_v3 (inchangée) : `ntherm` balayages de thermalisation conditionnelle depuis V, puis `nsweep`
//! balayages ; une mesure F tous les `every` balayages ; à la mesure d'indice k (0, 1, …), D si `k % deach == 0`
//! (alors `blk` aussi). Avec les valeurs de la saturation v3 (nsweep = 6000, every = 2, deach = 5) : 3000 F et
//! 600 D.
//!
//! Sorties (répertoire NEUF, COMPLETE en dernier, seulement si tout est vert) :
//! - `F.npy` (nmes, nF) f64 ; `D.npy` (nD, nF, nF) f64 ; `blk.npy` (nD, 6, (L/2)⁴) f32 aux mesures D ;
//! - avec `--tordu` (lecture secondaire (ii) de chaine_v3) : `Ft.npy` (nmes, 18) et `Dtw.npy` (nD, 3, 18, 18) ;
//! - `controles.json` (Vdev, unitarité, finitude, plafond : déterministe, sans horloge) ;
//! - `alea.json` (état final de la source d'aléa) ;
//! - `manifest.json` (provenance : hôte, binaire, backend, modules GPU compilés, garde SSD, verrou, empreintes).
//!
//! Reprise : `reprise.json` + `reprise.<gen>.*` (supprimés à la fin) ; voir [`super::reprise`].

use super::mesure::{Mesure, MesureFibreReelle};
use super::moteur::{ModeCond, MoteurCond, SourceAlea};
use super::npy_min::ecrire_atomique;
use super::reprise::{self, Compagnon, EtatReprise};
use super::ressources::{prendre_verrou, verifier_ssd};
use super::{geometrie::v_paires, GeometrieCond};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::fibre::torsion::{MesureurTordu, C_TW, NF_TW};
use crate::gpu::Appareil;
use crate::heritage::manifest::{champs_de_base, creer_repertoire_neuf, sha256_fichier, ErreurRepertoire};
use crate::stockage::{npy, GardeSsd};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConfigChaine {
    /// Configuration fine de départ (NPY c16 ou q64) ; V = dec(cfg) est figé par la chaîne.
    pub cfg: PathBuf,
    pub beta: f64,
    pub graine: u64,
    pub mode: ModeCond,
    pub ntherm: u64,
    pub nsweep: u64,
    pub every: u64,
    pub deach: u64,
    pub ckpt: u64,
    /// Répertoire de sortie (NPY : F (nmes, nF), D (nD, nF, nF), blk, controles.json, manifeste).
    pub out: PathBuf,
    /// Lecture tordue de chaine_v3 (`fibre::torsion`) : Ft (nmes, 18) à chaque mesure, Dtw (nD, 3, 18, 18) aux
    /// mesures D, écrits dans Ft.npy et Dtw.npy (lisibles tels quels par analyse_v3.py / torsion.lam_tw).
    #[serde(default)]
    pub tordu: bool,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RapportChaine {
    pub mesures_f: u64,
    pub mesures_d: u64,
    pub vdev: f64,
    pub unitarite: f64,
    pub secondes: f64,
}

/// Options d'exécution hors de la configuration scientifique (tests, reprise).
#[derive(Default)]
pub struct OptionsChaine {
    /// Arrêt volontaire juste après l'écriture du N-ième point de reprise (équivaut à un arrêt propre ;
    /// sert aux tests de reprise à l'octet). `None` : on va jusqu'au bout.
    pub arret_apres_ckpt: Option<u64>,
    /// Mesure à employer ; `None` : la mesure de fibre de T6 (`fibre::Mesureur`).
    pub mesure: Option<Box<dyn Mesure>>,
    /// Écrit la progression sur stderr (aucune valeur de C_P, comme la v3).
    pub journal: bool,
}

/// Issue d'une exécution.
#[derive(Clone, Debug, PartialEq)]
pub enum Issue {
    Terminee(RapportChaine),
    /// Arrêt volontaire demandé (`arret_apres_ckpt`) : le point de reprise de ce numéro est écrit.
    Interrompue {
        ckpt: u64,
    },
}

pub fn executer(app: Option<&Appareil>, cfg: &ConfigChaine, reprendre: bool) -> Resultat<RapportChaine> {
    let opts = OptionsChaine { journal: true, ..Default::default() };
    match executer_avec(app, cfg, reprendre, opts)? {
        Issue::Terminee(r) => Ok(r),
        Issue::Interrompue { .. } => unreachable!("aucun arrêt volontaire demandé"),
    }
}

fn valider(cfg: &ConfigChaine) -> Resultat<()> {
    let refus = |m: String| Err(Erreur::Config(m));
    if cfg.every == 0 || cfg.deach == 0 {
        return refus("every et deach doivent être ≥ 1".into());
    }
    if cfg.nsweep < cfg.every {
        return refus(format!("nsweep = {} < every = {} : aucune mesure", cfg.nsweep, cfg.every));
    }
    if !cfg.beta.is_finite() || cfg.beta < 0.0 {
        return refus(format!("β = {} (β ≥ 0 fini requis)", cfg.beta));
    }
    if cfg.mode == ModeCond::RefRng && cfg.beta == 0.0 {
        return refus("β = 0 refusé en mode ref_rng (la référence ne termine pas) ; utiliser le mode rapide".into());
    }
    Ok(())
}

fn err_rep(e: ErreurRepertoire) -> Erreur {
    match e {
        ErreurRepertoire::Existe(p) => Erreur::Config(format!(
            "{} existe déjà : un répertoire de run doit être NEUF (utiliser --reprendre pour continuer une chaîne)",
            p.display()
        )),
        ErreurRepertoire::Vide => Erreur::Config("répertoire de sortie vide".into()),
        ErreurRepertoire::Io(p, e) => Erreur::io(format!("répertoire {}", p.display()), e),
    }
}

/// Même exécution que [`executer`], avec options (mesure injectée, arrêt volontaire pour les tests de reprise).
/// En cas d'échec APRÈS l'ouverture du répertoire de sortie, le manifeste est marqué `echec` (jamais de COMPLETE).
pub fn executer_avec(
    app: Option<&Appareil>,
    cfg: &ConfigChaine,
    reprendre: bool,
    opts: OptionsChaine,
) -> Resultat<Issue> {
    let mut ouvert = false;
    let r = executer_interne(app, cfg, reprendre, opts, &mut ouvert);
    if let (Err(e), true) = (&r, ouvert) {
        marquer_echec(&cfg.out, &e.to_string());
    }
    r
}

/// Marque `manifest.json` en échec (meilleur effort ; ne touche à rien si le manifeste est illisible).
fn marquer_echec(out: &Path, raison: &str) {
    let p = out.join("manifest.json");
    let Ok(t) = std::fs::read_to_string(&p) else { return };
    let Ok(mut v) = serde_json::from_str::<serde_json::Value>(&t) else { return };
    if let Some(o) = v.as_object_mut() {
        o.insert("statut".into(), "echec".into());
        o.insert("raison_echec".into(), raison.into());
        if let Ok(txt) = serde_json::to_string_pretty(&v) {
            let _ = ecrire_atomique(&p, &[txt.as_bytes()]);
        }
    }
}

fn executer_interne(
    app: Option<&Appareil>,
    cfg: &ConfigChaine,
    reprendre: bool,
    mut opts: OptionsChaine,
    ouvert: &mut bool,
) -> Resultat<Issue> {
    valider(cfg)?;
    if !reprendre && std::fs::symlink_metadata(&cfg.out).is_ok() {
        return Err(err_rep(ErreurRepertoire::Existe(cfg.out.clone())));
    }
    let debut = Instant::now();
    // Admission mémoire AVANT de lire le champ et d'allouer quoi que ce soit (rc 5).
    let (_, l_entete) = npy::inspecter_champ(&cfg.cfg)?;
    let nmes_prevu = (cfg.nsweep / cfg.every) as usize;
    super::ressources::admettre(
        &format!("cond chaine L={l_entete} ({})", app.map_or("cpu".to_string(), |a| a.backend().nom().to_string())),
        super::ressources::besoin_memoire(l_entete, nmes_prevu, nmes_prevu.div_ceil(cfg.deach as usize), 54, app.is_some()),
        app,
    )?;
    let sha_cfg = sha256_fichier(&cfg.cfg).map_err(|e| Erreur::io(format!("empreinte {}", cfg.cfg.display()), e))?;
    let c0 = npy::lire_champ(&cfg.cfg)?;
    let l = c0.l;
    let geo = Arc::new(GeometrieCond::nouvelle(l)?);
    let vfix = v_paires(&c0, &geo);
    let nmes_tot = (cfg.nsweep / cfg.every) as usize;
    let nd_tot = nmes_tot.div_ceil(cfg.deach as usize);

    // --- ressources : verrou GPU, répertoire, garde SSD (projection totale AVANT tout octet écrit)
    let verrou = prendre_verrou(app)?;
    let mut mesure: Box<dyn Mesure> = match opts.mesure.take() {
        Some(m) => m,
        None => Box::new(MesureFibreReelle::nouvelle(app, geo.clone())?),
    };
    let nf = mesure.nf();
    let lc4 = (l / 2).pow(4);
    let tordu = if cfg.tordu { Some(MesureurTordu::nouveau(&geo)?) } else { None };
    let (nft, ndtw) = if cfg.tordu { (NF_TW, 3 * NF_TW * NF_TW) } else { (0, 0) };
    let sortie = (8 * nmes_tot * (nf + nft) + 8 * nd_tot * (nf * nf + ndtw) + 4 * nd_tot * 6 * lc4) as u64;
    let projection = sortie
        + 2 * reprise::octets_projetes(
            16 * geo.v,
            nmes_tot * (nf + nft),
            nd_tot * (nf * nf + ndtw),
            nd_tot * 6 * lc4,
        );
    let ssd = verifier_ssd(&cfg.out, projection)?;

    // --- départ neuf ou reprise
    let mut f: Vec<f64> = Vec::new();
    let mut d: Vec<f64> = Vec::new();
    let mut blk: Vec<f32> = Vec::new();
    let mut ft: Vec<f64> = Vec::new();
    let mut dtw: Vec<f64> = Vec::new();
    let (mut n_f, mut n_d, mut t_debut, mut generation) = (0u64, 0u64, 0u64, 0u64);
    let mut t_therm = 0.0f64;
    let mut moteur = if reprendre {
        if cfg.out.join("COMPLETE").exists() {
            return Err(Erreur::Config(format!("{} : chaîne déjà terminée (COMPLETE présent)", cfg.out.display())));
        }
        let r = reprise::lire(&cfg.out)?;
        let e = &r.etat;
        let identique = e.sha_cfg == sha_cfg
            && e.l == l
            && e.beta == cfg.beta
            && e.graine == cfg.graine
            && e.mode == cfg.mode
            && (e.ntherm, e.nsweep, e.every, e.deach) == (cfg.ntherm, cfg.nsweep, cfg.every, cfg.deach)
            && e.nf == nf
            && e.tordu == cfg.tordu;
        if !identique {
            return Err(Erreur::Config(format!(
                "{} : le point de reprise ne correspond pas à la configuration demandée (cfg, β, graine, mode, cadence)",
                cfg.out.display()
            )));
        }
        (n_f, n_d, t_debut, generation) = (e.n_f, e.n_d, e.t, e.generation);
        let champ_depart = ChampHote { l, q: r.champ.clone() };
        let m = MoteurCond::nouveau(app, geo.clone(), &champ_depart, vfix.clone(), cfg.beta, e.alea.clone())?;
        (f, d, blk, ft, dtw) = (r.f, r.d, r.blk, r.ft, r.dtw);
        *ouvert = true;
        if opts.journal {
            eprintln!("{} REPRISE t={t_debut} nF={n_f} nD={n_d}", horodatage());
        }
        m
    } else {
        creer_repertoire_neuf(&cfg.out).map_err(err_rep)?;
        *ouvert = true;
        let mut m = MoteurCond::nouveau(
            app,
            geo.clone(),
            &c0,
            vfix.clone(),
            cfg.beta,
            SourceAlea::depuis_graine(cfg.mode, cfg.graine),
        )?;
        ecrire_manifeste(&Manifeste {
            cfg,
            sha_cfg: &sha_cfg,
            moteur: &m,
            ssd: &ssd,
            verrou: verrou.etat,
            nf,
            statut: "en_cours",
            rapport: None,
            raison: None,
            temps: None,
        })?;
        let t_th = Instant::now();
        for _ in 0..cfg.ntherm {
            m.sweep()?;
        }
        t_therm = t_th.elapsed().as_secs_f64();
        if opts.journal {
            eprintln!(
                "{} THERM ok ({} balayages, {:.0} s) L={l} beta={}",
                horodatage(),
                cfg.ntherm,
                debut.elapsed().as_secs_f64(),
                cfg.beta
            );
        }
        m
    };

    // --- boucle de mesure
    let t1 = Instant::now();
    let mut ncheckpoints = 0u64;
    let (mut t_sweep, mut t_mesure) = (0.0f64, 0.0f64);
    for t in t_debut..cfg.nsweep {
        let ts = Instant::now();
        moteur.sweep()?;
        t_sweep += ts.elapsed().as_secs_f64();
        if (t + 1) % cfg.every != 0 {
            continue;
        }
        let tm = Instant::now();
        if let Some(tw) = &tordu {
            // Lecture tordue : Ft à chaque mesure, Dtw aux mesures D (même configuration que la mesure principale).
            let avec_d = n_f % cfg.deach == 0;
            let (pv, gp) = mesure.plaquettes_tordues(&moteur, avec_d)?;
            ft.extend(tw.f_tw(&pv)?);
            if avec_d {
                dtw.extend(tw.d_tw(&moteur.telecharger()?.q, &gp)?);
            }
        }
        if n_f % cfg.deach == 0 {
            let m = mesure.mesurer(&moteur)?;
            check_dims(&m.f, nf, &m.d, &m.blk, nf, 6 * lc4)?;
            f.extend_from_slice(&m.f);
            d.extend_from_slice(&m.d);
            blk.extend_from_slice(&m.blk);
            n_d += 1;
        } else {
            let m = mesure.mesurer_f(&moteur)?;
            if m.len() != nf {
                return Err(Erreur::Validation(format!("la mesure F a {} composantes au lieu de {nf}", m.len())));
            }
            f.extend_from_slice(&m);
        }
        n_f += 1;
        t_mesure += tm.elapsed().as_secs_f64();
        if cfg.ckpt > 0 && n_f % cfg.ckpt == 0 && (n_f as usize) < nmes_tot {
            generation += 1;
            let champ = moteur.telecharger()?;
            verifier_ssd(
                &cfg.out,
                reprise::octets_projetes(champ.q.len(), f.len() + ft.len(), d.len() + dtw.len(), blk.len()),
            )?;
            let vide = Compagnon { nom: String::new(), sha256: String::new(), octets: 0 };
            let etat = EtatReprise {
                schema: String::new(),
                generation,
                sha_cfg: sha_cfg.clone(),
                l,
                beta: cfg.beta,
                graine: cfg.graine,
                mode: cfg.mode,
                ntherm: cfg.ntherm,
                nsweep: cfg.nsweep,
                every: cfg.every,
                deach: cfg.deach,
                nf,
                t: t + 1,
                n_f,
                n_d,
                alea: moteur.alea(),
                champ: vide.clone(),
                f: vide.clone(),
                d: vide.clone(),
                blk: vide,
                tordu: cfg.tordu,
                ft: None,
                dtw: None,
            };
            let accus_tordus = cfg.tordu.then_some((ft.as_slice(), dtw.as_slice()));
            reprise::ecrire_avec(&cfg.out, etat, &champ.q, &f, &d, &blk, accus_tordus)?;
            ncheckpoints += 1;
            if opts.journal {
                let (vdev, _) = moteur.controles()?;
                eprintln!(
                    "{} PROGRES balayage={}/{} nF={n_f} nD={n_d} Vdev={vdev:.1e} {:.3} s/balayage",
                    horodatage(),
                    t + 1,
                    cfg.nsweep,
                    t1.elapsed().as_secs_f64() / (t + 1 - t_debut) as f64
                );
            }
            if opts.arret_apres_ckpt.is_some_and(|n| ncheckpoints >= n) {
                return Ok(Issue::Interrompue { ckpt: ncheckpoints });
            }
        }
    }

    // --- contrôles et sorties
    let (vdev, unitarite) = moteur.controles()?;
    let fini = f.iter().chain(&d).chain(&ft).chain(&dtw).all(|x| x.is_finite()) && blk.iter().all(|x| x.is_finite());
    let plafond = moteur.diag_rapide().map(|x| x.plafond).unwrap_or(0);
    if !(vdev < 1e-10 && unitarite < 1e-10 && fini && plafond == 0) {
        return Err(Erreur::Validation(format!(
            "contraintes violées : Vdev = {vdev:e}, unitarité = {unitarite:e}, finitude = {fini}, tirages au plafond = {plafond}"
        )));
    }
    debug_assert_eq!(n_f as usize, nmes_tot);
    let garde = GardeSsd::machine();
    npy::ecrire_f64(&cfg.out.join("F.npy"), &[n_f as usize, nf], &f, &garde)?;
    npy::ecrire_f64(&cfg.out.join("D.npy"), &[n_d as usize, nf, nf], &d, &garde)?;
    npy::ecrire_f32(&cfg.out.join("blk.npy"), &[n_d as usize, 6, lc4], &blk, &garde)?;
    if cfg.tordu {
        npy::ecrire_f64(&cfg.out.join("Ft.npy"), &[n_f as usize, NF_TW], &ft, &garde)?;
        npy::ecrire_f64(&cfg.out.join("Dtw.npy"), &[n_d as usize, 3, NF_TW, NF_TW], &dtw, &garde)?;
    }
    let controles = serde_json::json!({
        "schema": "kvu-cond-controles/1", "L": l, "beta": cfg.beta, "mode": cfg.mode, "graine": cfg.graine,
        "ntherm": cfg.ntherm, "nsweep": cfg.nsweep, "every": cfg.every, "deach": cfg.deach, "nF": nf,
        "nmes": n_f, "nD": n_d, "Vdev": vdev, "unitarite": unitarite, "finite": fini, "plafond": plafond,
        "tordu": cfg.tordu, "c_tw": cfg.tordu.then_some(C_TW),
    });
    ecrire_atomique(&cfg.out.join("controles.json"), &[serde_json::to_string_pretty(&controles).unwrap().as_bytes()])?;
    let alea = serde_json::to_string_pretty(&moteur.alea()).map_err(|e| Erreur::Format(e.to_string()))?;
    ecrire_atomique(&cfg.out.join("alea.json"), &[alea.as_bytes()])?;

    let secondes = debut.elapsed().as_secs_f64();
    let rapport = RapportChaine { mesures_f: n_f, mesures_d: n_d, vdev, unitarite, secondes };
    ecrire_manifeste(&Manifeste {
        cfg,
        sha_cfg: &sha_cfg,
        moteur: &moteur,
        ssd: &ssd,
        verrou: verrou.etat,
        nf,
        statut: "termine",
        rapport: Some(&rapport),
        raison: None,
        temps: Some([t_therm, t_sweep, t_mesure]),
    })?;
    reprise::nettoyer(&cfg.out, None);
    ecrire_complete(&cfg.out)?;
    Ok(Issue::Terminee(rapport))
}

fn check_dims(f: &[f64], nf: usize, d: &[f64], blk: &[f32], nd: usize, nblk: usize) -> Resultat<()> {
    if f.len() != nf || d.len() != nd * nd || blk.len() != nblk {
        return Err(Erreur::Validation(format!(
            "dimensions de la mesure : F {} (attendu {nf}), D {} (attendu {}), blk {} (attendu {nblk})",
            f.len(),
            d.len(),
            nd * nd,
            blk.len()
        )));
    }
    Ok(())
}

fn horodatage() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{:02}:{:02}:{:02}", (s / 3600) % 24, (s / 60) % 60, s % 60)
}

/// Ce qu'écrit `manifest.json` (provenance) : au départ (`en_cours`), à la fin (`termine`) ou en échec.
struct Manifeste<'a> {
    cfg: &'a ConfigChaine,
    sha_cfg: &'a str,
    moteur: &'a MoteurCond,
    ssd: &'a super::ressources::InfoSsd,
    verrou: &'a str,
    nf: usize,
    statut: &'a str,
    rapport: Option<&'a RapportChaine>,
    raison: Option<&'a str>,
    /// Répartition du temps de CE processus (thermalisation, balayages de mesure, mesures), en secondes.
    temps: Option<[f64; 3]>,
}

fn ecrire_manifeste(m: &Manifeste) -> Resultat<()> {
    let cfg = m.cfg;
    let termine = m.statut == "termine";
    let mut produits = Vec::new();
    if termine {
        let tordus: &[&str] = if cfg.tordu { &["Ft.npy", "Dtw.npy"] } else { &[] };
        for nom in ["F.npy", "D.npy", "blk.npy", "controles.json", "alea.json"].iter().chain(tordus) {
            let p = cfg.out.join(nom);
            let sha = sha256_fichier(&p).map_err(|e| Erreur::io(format!("empreinte {}", p.display()), e))?;
            let octets = std::fs::metadata(&p).map(|x| x.len()).unwrap_or(0);
            produits.push(serde_json::json!({"nom": nom, "sha256": sha, "octets": octets}));
        }
    }
    let argv: Vec<String> = std::env::args().collect();
    let base: serde_json::Value =
        serde_json::from_str(&champs_de_base("kvu-cond-chaine", env!("CARGO_PKG_VERSION"), &argv).json())
            .map_err(|e| Erreur::Format(e.to_string()))?;
    let mut obj = base.as_object().cloned().unwrap_or_default();
    let ajouts = serde_json::json!({
        "schema": "kevinotron-unifie/1",
        "commande": "cond chaine",
        "statut": m.statut,
        "rc": if termine { serde_json::json!(0) } else { serde_json::Value::Null },
        "raison_echec": m.raison,
        "backend": m.moteur.backend().nom(),
        "mode": cfg.mode, "precision": "f64",
        "config": cfg,
        "L": m.moteur.geo.l, "nF": m.nf, "sha256_cfg": m.sha_cfg,
        "modules_compiles": m.moteur.provenance().map(|p| vec![p.clone()]).unwrap_or_default(),
        "verrou": m.verrou,
        "garde_ssd": m.ssd,
        "rapport": m.rapport,
        "temps_s": m.temps.map(|t| serde_json::json!({"thermalisation": t[0], "balayages": t[1], "mesures": t[2]})),
        "fichiers": produits,
    });
    for (k, v) in ajouts.as_object().unwrap() {
        obj.insert(k.clone(), v.clone());
    }
    let txt =
        serde_json::to_string_pretty(&serde_json::Value::Object(obj)).map_err(|e| Erreur::Format(e.to_string()))?;
    ecrire_atomique(&cfg.out.join("manifest.json"), &[txt.as_bytes()])
}

fn ecrire_complete(out: &Path) -> Resultat<()> {
    let complete =
        format!("rc=0\nmanifest_sha256={}\n", sha256_fichier(&out.join("manifest.json")).unwrap_or_default());
    ecrire_atomique(&out.join("COMPLETE"), &[complete.as_bytes()])
}

/// Racine d'un nom de fichier de sortie, pour les tests.
pub fn fichiers_de_sortie(dir: &Path) -> Vec<PathBuf> {
    ["F.npy", "D.npy", "blk.npy", "controles.json", "alea.json"].iter().map(|n| dir.join(n)).collect()
}
