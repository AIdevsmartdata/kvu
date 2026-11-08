//! `kvu fibre echantillonner` : routes pénalité et RATTLE, recettes rejouables.

use super::action::{action_wilson, delta_action_lien_q, plaquette_moyenne};
use super::alea::{Flux, TAG_ACCEPTATION, TAG_PROPOSITION};
use super::jacobien::{Reseau, Q};
use super::jouets::{Courbe, DeuxLiens};
use super::modele_su2::{projeter_continuation, FibreSu2};
use super::penalite::{renormaliser, ChainePenalite, ParamsPenalite};
use super::rattle::{self, Echec, ParamsRattle, SortieProposition, Variete};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul, Geo};
use crate::rg::bloc::{exp_q, log_q, ubar_lien};
use crate::stockage::npy::{self, FormatChamp};
use crate::stockage::{GardeSsd, Manifeste};
use clap::{Arg, ArgMatches, Command};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub fn commande() -> Command {
    let mut c = Command::new("echantillonner")
        .about("Échantillonne une fibre SU(2) par pénalité ou RATTLE (CPU)")
        .arg(
            Arg::new("methode")
                .long("methode")
                .required(true)
                .value_parser(["penalite", "rattle"])
                .help("route d'échantillonnage"),
        )
        .arg(
            Arg::new("jouet")
                .long("jouet")
                .value_parser(["deux-liens", "courbe"])
                .help("modèle jouet"),
        )
        .arg(
            Arg::new("sortie")
                .long("sortie")
                .visible_alias("outdir")
                .required(true)
                .help("répertoire NEUF"),
        );
    for (nom, defaut, aide) in [
        ("entree-v", None, "champ grossier NPY c16 ou q64"),
        ("entree-u0", None, "champ fin de départ (défaut : froid)"),
        ("L", None, "côté fin, multiple de 3 ≥ 6 ; κ inf : pair ≥ 2"),
        (
            "beta",
            None,
            "couplage β (requis sur réseau ; défaut jouet : 2,3)",
        ),
        (
            "kappa",
            Some("2e-3"),
            "raideur κ > 0 ou inf (pénalité retirée)",
        ),
        (
            "delta",
            None,
            "amplitude locale (défaut calculé depuis κ et β)",
        ),
        ("dt", Some("0.02"), "pas de RATTLE"),
        ("duree", Some("0.2"), "durée de trajectoire RATTLE"),
        (
            "therm",
            Some("0"),
            "balayages ou propositions de thermalisation",
        ),
        ("nmeas", Some("1"), "nombre de mesures par chaîne"),
        ("skip", Some("1"), "balayages ou propositions entre mesures"),
        ("chaines", Some("1"), "chaînes indépendantes"),
        ("graine", Some("26102612001"), "clé Philox u64"),
        ("fils", None, "fils CPU (RAYON_NUM_THREADS, sinon 4)"),
        (
            "dump-tous",
            Some("0"),
            "écrit les champs tous les K relevés (0 : aucun)",
        ),
        ("tol-newton", Some("1e-13"), "tolérance Newton"),
        ("max-newton", Some("50"), "corrections Newton maximales"),
        ("tol-rev", Some("1e-9"), "tolérance de réversibilité"),
        ("t", None, "paramètre du jouet deux liens, dans ]0,1["),
        ("v", Some("1,0,0,0"), "cible unitaire du jouet deux liens"),
        (
            "masse",
            Some("1,1"),
            "masses positives m1,m2 du jouet courbe",
        ),
    ] {
        let mut a = Arg::new(nom).long(nom).help(aide).allow_hyphen_values(true);
        if let Some(d) = defaut {
            a = a.default_value(d);
        }
        c = c.arg(a);
    }
    c
}

fn config(message: impl Into<String>) -> Erreur {
    Erreur::Config(message.into())
}
fn lire<T: std::str::FromStr>(m: &ArgMatches, nom: &str) -> Resultat<T> {
    m.get_one::<String>(nom)
        .ok_or_else(|| config(format!("--{nom} requis")))?
        .parse()
        .map_err(|_| config(format!("--{nom} : valeur invalide")))
}
fn liste<const N: usize>(m: &ArgMatches, nom: &str) -> Resultat<[f64; N]> {
    let s = m.get_one::<String>(nom).expect("défaut");
    let v: Vec<f64> = s
        .split(',')
        .map(|x| {
            x.parse::<f64>().map_err(|_| {
                config(format!(
                    "--{nom} : nombres séparés par des virgules attendus"
                ))
            })
        })
        .collect::<Resultat<_>>()?;
    if v.len() != N || !v.iter().all(|x| x.is_finite()) {
        return Err(config(format!("--{nom} : {N} réels finis attendus")));
    }
    v.try_into().map_err(|_| config("taille de liste invalide"))
}

struct Config {
    methode: String,
    jouet: Option<String>,
    l: usize,
    beta: f64,
    kappa: Option<f64>,
    delta: f64,
    duree: f64,
    rattle: ParamsRattle,
    therm: usize,
    nmeas: usize,
    skip: usize,
    chaines: usize,
    graine: u64,
    fils: usize,
    dump: usize,
    t: f64,
    v: Q,
    masse: [f64; 2],
    entree_v: Option<PathBuf>,
    entree_u0: Option<PathBuf>,
    sortie: PathBuf,
}
impl Config {
    fn lire(m: &ArgMatches) -> Resultat<Self> {
        let jouet = m.get_one::<String>("jouet").cloned();
        let methode = m.get_one::<String>("methode").expect("requis").clone();
        let beta: f64 = if jouet.is_some() && !m.contains_id("beta") {
            2.3
        } else {
            lire(m, "beta")?
        };
        let l: usize = if jouet.is_some() && !m.contains_id("L") {
            2
        } else {
            lire(m, "L")?
        };
        let kappa = match m.get_one::<String>("kappa").expect("défaut").as_str() {
            "inf" => None,
            _ => Some(lire::<f64>(m, "kappa")?),
        };
        if !beta.is_finite() || beta <= 0.0 || kappa.is_some_and(|x| !x.is_finite() || x <= 0.0) {
            return Err(config(
                "β et κ doivent être finis et strictement positifs (ou κ = inf)",
            ));
        }
        let delta = if m.contains_id("delta") {
            lire(m, "delta")?
        } else {
            kappa.map_or(0.5, |k| (2.0 * (k / beta).sqrt()).min(0.5))
        };
        let dt: f64 = lire(m, "dt")?;
        let duree: f64 = lire(m, "duree")?;
        let tol_newton: f64 = lire(m, "tol-newton")?;
        let tol_rev: f64 = lire(m, "tol-rev")?;
        let max_newton = lire(m, "max-newton")?;
        if !delta.is_finite()
            || delta < 0.0
            || [dt, duree, tol_newton, tol_rev]
                .iter()
                .any(|x| !x.is_finite() || *x <= 0.0)
            || max_newton == 0
            || !(duree / dt).is_finite()
            || (duree / dt).round() >= usize::MAX as f64
        {
            return Err(config("delta ≥ 0 ; dt, durée, tolérances et max-newton strictement positifs et finis requis"));
        }
        let therm: usize = lire(m, "therm")?;
        let nmeas: usize = lire(m, "nmeas")?;
        let skip: usize = lire(m, "skip")?;
        let chaines: usize = lire(m, "chaines")?;
        let fils = m
            .get_one::<String>("fils")
            .cloned()
            .or_else(|| std::env::var("RAYON_NUM_THREADS").ok())
            .unwrap_or_else(|| "4".into())
            .parse::<usize>()
            .map_err(|_| config("--fils : entier attendu"))?;
        if skip == 0
            || chaines == 0
            || chaines > u32::MAX as usize
            || fils == 0
            || nmeas
                .checked_mul(skip)
                .and_then(|n| n.checked_add(therm))
                .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(config(
                "skip, chaînes et fils > 0 ; compteurs Philox limités à u32",
            ));
        }
        let t: f64 = if m.contains_id("t") {
            lire(m, "t")?
        } else {
            0.5
        };
        let v = liste(m, "v")?;
        let masse: [f64; 2] = liste(m, "masse")?;
        if !t.is_finite()
            || t <= 0.0
            || t >= 1.0
            || (v.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() >= 1e-10
            || masse.iter().any(|x| *x <= 0.0 || !(1.0 / x).is_finite())
        {
            return Err(config(
                "t dans ]0,1[, v unitaire et masses strictement positives requis",
            ));
        }
        if jouet.as_deref() == Some("deux-liens") && !m.contains_id("t") {
            return Err(config("--t requis pour deux-liens"));
        }
        if jouet.as_deref() == Some("courbe") && methode != "rattle" {
            return Err(config("courbe requiert --methode rattle"));
        }
        let entree_v = m.get_one::<String>("entree-v").map(PathBuf::from);
        let entree_u0 = m.get_one::<String>("entree-u0").map(PathBuf::from);
        if jouet.is_some() && (entree_v.is_some() || entree_u0.is_some()) {
            return Err(config("les jouets ne lisent pas de champs NPY"));
        }
        if jouet.is_none() {
            if l < 2 || l.checked_pow(4).and_then(|v| v.checked_mul(16)).is_none() {
                return Err(config("L ≥ 2 requis, taille représentable"));
            }
            if methode == "penalite" && kappa.is_none() && !l.is_multiple_of(2) {
                return Err(config("κ inf : L pair ≥ 2 requis"));
            }
            if l.checked_pow(4)
                .and_then(|v| v.checked_mul(12))
                .is_none_or(|n| n > u32::MAX as usize)
            {
                return Err(config("trop de liens pour les compteurs Philox"));
            }
            if methode == "penalite"
                && kappa.is_some_and(|k| !(0.5 * (beta / k)).is_finite() || 0.5 * (beta / k) <= 0.0)
            {
                return Err(config(
                    "κ et β doivent donner une raideur finie strictement positive",
                ));
            }
            if (methode == "rattle" || kappa.is_some())
                && (l < 6 || !l.is_multiple_of(3) || entree_v.is_none())
            {
                return Err(config("fibre : L multiple de 3, ≥ 6, et --entree-v requis"));
            }
        } else if l < 2 {
            return Err(config("L ≥ 2 requis"));
        }
        Ok(Self {
            methode,
            jouet,
            l,
            beta,
            kappa,
            delta,
            duree,
            rattle: ParamsRattle {
                dt,
                nb_pas: (duree / dt).round().max(1.0) as usize,
                tol_newton,
                max_newton,
                tol_rev,
                ..Default::default()
            },
            therm,
            nmeas,
            skip,
            chaines,
            graine: lire(m, "graine")?,
            fils,
            dump: lire(m, "dump-tous")?,
            t,
            v,
            masse,
            entree_v,
            entree_u0,
            sortie: PathBuf::from(m.get_one::<String>("sortie").expect("requis")),
        })
    }

    fn parametres(&self) -> Value {
        json!({"groupe": "SU(2)", "L": self.l, "beta": self.beta, "start": "cold", "therm": self.therm,
            "nmeas": self.nmeas, "skip": self.skip, "nor": 0, "precision": "f64", "options": "ref-k1",
            "methode": self.methode, "kappa": self.kappa.map_or(json!("inf"), |k| json!(k)), "delta": self.delta,
            "dt": self.rattle.dt, "duree": self.duree, "nb_pas": self.rattle.nb_pas, "chaines": self.chaines,
            "graine": self.graine, "dump-tous": self.dump, "tol-newton": self.rattle.tol_newton,
            "max-newton": self.rattle.max_newton, "tol-rev": self.rattle.tol_rev, "jouet": self.jouet,
            "t": self.t, "v": self.v, "masse": self.masse})
    }

    fn commande(&self) -> Vec<String> {
        let p = self.parametres();
        let mut c = vec!["fibre".into(), "echantillonner".into()];
        for n in [
            "methode",
            "L",
            "beta",
            "kappa",
            "delta",
            "dt",
            "duree",
            "therm",
            "nmeas",
            "skip",
            "chaines",
            "graine",
            "dump-tous",
            "tol-newton",
            "max-newton",
            "tol-rev",
            "t",
        ] {
            c.push(format!("--{n}"));
            c.push(
                p[n].as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| p[n].to_string()),
            );
        }
        // Le nombre de fils est un réglage d'exécution, sans effet sur la recette.
        // Le rejeu séquentiel rend la recette et ses octets indépendants de --fils.
        c.extend(["--fils".into(), "1".into()]);
        for (n, v) in [("v", self.v.as_slice()), ("masse", self.masse.as_slice())] {
            c.extend([
                format!("--{n}"),
                v.iter().map(f64::to_string).collect::<Vec<_>>().join(","),
            ]);
        }
        if let Some(j) = &self.jouet {
            c.extend(["--jouet".into(), j.clone()]);
        }
        for (n, p) in [("entree-v", &self.entree_v), ("entree-u0", &self.entree_u0)] {
            if let Some(p) = p {
                c.extend([format!("--{n}"), p.to_string_lossy().into_owned()]);
            }
        }
        c
    }
    fn total(&self) -> usize {
        self.therm + self.nmeas * self.skip
    }
    fn mesure(&self, i: usize) -> Option<usize> {
        (i >= self.therm && (i - self.therm + 1).is_multiple_of(self.skip))
            .then(|| (i - self.therm) / self.skip)
    }
}

#[derive(Default)]
struct Resume {
    propositions: u64,
    acceptees: u64,
    newton: u64,
    reversibilite: u64,
    evaluation: u64,
    somme_dh: f64,
    nb_dh: u64,
    residu: f64,
}
impl Resume {
    fn ajouter(&mut self, s: &SortieProposition, residu: f64, proposees: u64, acceptees: u64) {
        self.propositions += proposees;
        self.acceptees += acceptees;
        match s.echec {
            Some(Echec::Newton) => self.newton += 1,
            Some(Echec::Reversibilite) => self.reversibilite += 1,
            Some(Echec::Evaluation) => self.evaluation += 1,
            None => (),
        }
        if s.delta_h.is_finite() {
            self.somme_dh += s.delta_h;
            self.nb_dh += 1;
        }
        self.residu = self.residu.max(residu);
    }
    fn json(&self, chaine: usize) -> Value {
        json!({"chaine": chaine, "propositions": self.propositions, "acceptees": self.acceptees,
            "taux_acceptation": if self.propositions == 0 { 0.0 } else { self.acceptees as f64 / self.propositions as f64 },
            "rejets": {"newton": self.newton, "reversibilite": self.reversibilite, "evaluation": self.evaluation},
            "delta_h_moyen": if self.nb_dh == 0 { None } else { Some(self.somme_dh / self.nb_dh as f64) },
            "residu_max": self.residu})
    }
}
struct SortieChaine {
    secondes: f64,
    csv: String,
    resume: Value,
    fichiers: Vec<String>,
}
fn statut(accepte: bool) -> SortieProposition {
    SortieProposition {
        accepte,
        echec: None,
        delta_h: 0.0,
        nb_newton: 0,
        residu_max: 0.0,
    }
}
fn diagnostic(csv: &mut String, s: &SortieProposition, residu: f64) {
    let echec = match s.echec {
        None => "aucun",
        Some(Echec::Newton) => "newton",
        Some(Echec::Reversibilite) => "reversibilite",
        Some(Echec::Evaluation) => "evaluation",
    };
    writeln!(
        csv,
        ",{},{:.17e},{:.17e},{},{}",
        u8::from(s.accepte),
        s.delta_h,
        residu,
        echec,
        s.nb_newton
    )
    .expect("String");
}
fn sauver(
    c: &Config,
    chaine: usize,
    mesure: Option<usize>,
    q: &ChampHote,
    fichiers: &mut Vec<String>,
) -> Resultat<()> {
    let nom = match mesure {
        Some(m) if c.dump > 0 && m.is_multiple_of(c.dump) => {
            format!("config_c{chaine:03}_m{m:06}.npy")
        }
        Some(_) => return Ok(()),
        None => format!("etat_final_c{chaine:03}.npy"),
    };
    npy::ecrire_champ_hache(
        &c.sortie.join(&nom),
        q,
        FormatChamp::Q64,
        &GardeSsd::machine(),
    )?;
    fichiers.push(nom);
    Ok(())
}

fn route_rattle<M: Variete>(
    c: &Config,
    k: usize,
    modele: &M,
    mut q: M::Etat,
    mut observer: impl FnMut(&M::Etat, Option<usize>, &mut SortieChaine) -> Resultat<()>,
) -> Resultat<SortieChaine> {
    let debut = Instant::now();
    let mut sortie = SortieChaine {
        secondes: 0.0,
        csv: String::new(),
        resume: Value::Null,
        fichiers: Vec::new(),
    };
    let mut resume = Resume::default();
    let mut ev = modele.evaluer(&q)?;
    let flux = Flux::nouveau(c.graine);
    for i in 0..c.total() {
        let (propose, s) =
            rattle::proposition(modele, &q, &ev, &c.rattle, &flux, k as u32, i as u32);
        if s.accepte {
            if let Some((q1, ev1)) = propose {
                (q, ev) = (q1, ev1);
            }
        }
        let residu = modele
            .residu(&q)?
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        if i >= c.therm {
            resume.ajouter(&s, residu, 1, u64::from(s.accepte));
        }
        if let Some(m) = c.mesure(i) {
            write!(&mut sortie.csv, "{k},{m}").expect("String");
            observer(&q, Some(m), &mut sortie)?;
            diagnostic(&mut sortie.csv, &s, residu);
        }
    }
    observer(&q, None, &mut sortie)?;
    sortie.resume = resume.json(k);
    sortie.secondes = debut.elapsed().as_secs_f64();
    Ok(sortie)
}

// ChainePenalite impose historiquement L pair. Le balayage séquentiel local
// ne dépend pas du coloriage pair/impair ; ce petit adaptateur accepte aussi
// les multiples impairs de 3 exigés par S4, sans modifier le moteur partagé.
trait Balayeur {
    fn balayage(&mut self, numero: u32) -> (u64, u64);
    fn champ(&self) -> &ChampHote;
    fn penalite_totale(&self) -> f64;
    fn residu_max(&self) -> f64;
}
impl Balayeur for ChainePenalite {
    fn balayage(&mut self, numero: u32) -> (u64, u64) {
        self.balayage(numero)
    }
    fn champ(&self) -> &ChampHote {
        self.champ()
    }
    fn penalite_totale(&self) -> f64 {
        self.penalite_totale()
    }
    fn residu_max(&self) -> f64 {
        self.residu_max()
    }
}
struct PenaliteImpaire {
    reseau: Arc<Reseau>,
    v: Vec<Q>,
    champ: ChampHote,
    u: Vec<Q>,
    ubar: Vec<Q>,
    voisins: Vec<Vec<usize>>,
    params: ParamsPenalite,
    flux: Flux,
    chaine: u32,
}
impl PenaliteImpaire {
    fn nouvelle(
        reseau: Arc<Reseau>,
        v: Vec<Q>,
        champ: ChampHote,
        params: ParamsPenalite,
        flux: Flux,
        chaine: u32,
    ) -> Self {
        let u = champ
            .q
            .chunks_exact(4)
            .map(|q| [q[0], q[1], q[2], q[3]])
            .collect();
        let mut voisins = vec![Vec::new(); reseau.n_liens];
        let mut ubar = Vec::with_capacity(reseau.n_c);
        for c in 0..reseau.n_c {
            for li in reseau.liens_de(c) {
                let liste = &mut voisins[li as usize];
                if liste.last() != Some(&c) {
                    liste.push(c);
                }
            }
            let (z, mu) = reseau.coords_c(c);
            ubar.push(ubar_lien(&champ, &reseau.table, z, mu).0);
        }
        Self {
            reseau,
            v,
            champ,
            u,
            ubar,
            voisins,
            params,
            flux,
            chaine,
        }
    }
    fn residu(&self, c: usize, moyenne: Q) -> f64 {
        let w = log_q(qmul(qdag(self.v[c]), moyenne)).0;
        w[0].hypot(w[1]).hypot(w[2])
    }
    fn energie(&self, c: usize, moyenne: Q) -> f64 {
        0.5 * (self.params.beta / self.params.kappa.expect("κ fini"))
            * self.residu(c, moyenne).powi(2)
    }
}
impl Balayeur for PenaliteImpaire {
    fn balayage(&mut self, numero: u32) -> (u64, u64) {
        let geo = Geo::new(self.champ.l);
        let mut acceptes = 0;
        for li in 0..self.u.len() {
            let xi = std::array::from_fn(|a| {
                self.params.delta
                    * self
                        .flux
                        .gauss(self.chaine, numero, (3 * li + a) as u32, TAG_PROPOSITION)
            });
            let ancien = self.u[li];
            let nouveau = renormaliser(qmul(exp_q(xi), ancien));
            let mut variation = delta_action_lien_q(&self.u, &geo, li, nouveau, self.params.beta);
            self.champ.q[4 * li..4 * li + 4].copy_from_slice(&nouveau);
            let moyennes: Vec<Q> = self.voisins[li]
                .iter()
                .map(|&c| {
                    let (z, mu) = self.reseau.coords_c(c);
                    ubar_lien(&self.champ, &self.reseau.table, z, mu).0
                })
                .collect();
            for (&c, &m) in self.voisins[li].iter().zip(&moyennes) {
                variation += self.energie(c, m) - self.energie(c, self.ubar[c]);
            }
            if self
                .flux
                .uniforme(self.chaine, numero, li as u32, TAG_ACCEPTATION)
                < (-variation).exp()
            {
                acceptes += 1;
                self.u[li] = nouveau;
                for (&c, m) in self.voisins[li].iter().zip(moyennes) {
                    self.ubar[c] = m;
                }
            } else {
                self.champ.q[4 * li..4 * li + 4].copy_from_slice(&ancien);
            }
        }
        (self.u.len() as u64, acceptes)
    }
    fn champ(&self) -> &ChampHote {
        &self.champ
    }
    fn penalite_totale(&self) -> f64 {
        self.ubar
            .iter()
            .enumerate()
            .map(|(c, &m)| self.energie(c, m))
            .sum()
    }
    fn residu_max(&self) -> f64 {
        self.ubar
            .iter()
            .enumerate()
            .map(|(c, &m)| self.residu(c, m))
            .fold(0.0, f64::max)
    }
}

fn route_penalite(
    c: &Config,
    k: usize,
    reseau: Option<Arc<Reseau>>,
    v: &[Q],
    q: &ChampHote,
) -> Resultat<SortieChaine> {
    let debut = Instant::now();
    let params = ParamsPenalite {
        beta: c.beta,
        kappa: c.kappa,
        delta: c.delta,
    };
    let flux = Flux::nouveau(c.graine);
    let mut a: Box<dyn Balayeur> = if c.l.is_multiple_of(2) {
        Box::new(ChainePenalite::nouvelle(
            reseau,
            v.to_vec(),
            q.clone(),
            params,
            flux,
            k as u32,
        )?)
    } else {
        Box::new(PenaliteImpaire::nouvelle(
            reseau.expect("fibre impaire"),
            v.to_vec(),
            q.clone(),
            params,
            flux,
            k as u32,
        ))
    };
    let mut sortie = SortieChaine {
        secondes: 0.0,
        csv: String::new(),
        resume: Value::Null,
        fichiers: Vec::new(),
    };
    let mut resume = Resume::default();
    for i in 0..c.total() {
        let (n, acceptes) = a.balayage(i as u32);
        let s = statut(acceptes > 0);
        let residu = a.residu_max();
        if i >= c.therm {
            resume.ajouter(&s, residu, n, acceptes);
        }
        if let Some(m) = c.mesure(i) {
            write!(
                &mut sortie.csv,
                "{k},{m},{:.17e},{:.17e},{:.17e}",
                plaquette_moyenne(a.champ()),
                action_wilson(a.champ(), c.beta),
                a.penalite_totale()
            )
            .expect("String");
            diagnostic(&mut sortie.csv, &s, residu);
            sauver(c, k, Some(m), a.champ(), &mut sortie.fichiers)?;
        }
    }
    sauver(c, k, None, a.champ(), &mut sortie.fichiers)?;
    sortie.resume = resume.json(k);
    sortie.secondes = debut.elapsed().as_secs_f64();
    Ok(sortie)
}

fn jouet_penalite(c: &Config, k: usize) -> Resultat<SortieChaine> {
    // Même noyau local que chaine_penalite_deux_liens, avec diagnostics et κ inf.
    let debut = Instant::now();
    let modele = DeuxLiens::nouveau(c.beta, c.t, c.v);
    let mut q = modele.depart(c.v, [1.0, 0.0, 0.0, 0.0]);
    let flux = Flux::nouveau(c.graine);
    let energie = |q: &[Q; 2]| -> Resultat<f64> {
        let residu = modele.residu(q)?;
        Ok(-c.beta * DeuxLiens::observable(q)
            + c.kappa.map_or(0.0, |k| {
                c.beta * residu.iter().map(|x| x * x).sum::<f64>() / (2.0 * k)
            }))
    };
    let mut e = energie(&q)?;
    let mut sortie = SortieChaine {
        secondes: 0.0,
        csv: String::new(),
        resume: Value::Null,
        fichiers: Vec::new(),
    };
    let mut resume = Resume::default();
    for i in 0..c.total() {
        let mut acceptes = 0;
        for lien in 0..2 {
            let xi = std::array::from_fn(|a| {
                c.delta * flux.gauss(k as u32, i as u32, (3 * lien + a) as u32, TAG_PROPOSITION)
            });
            let mut essai = q;
            essai[lien] = renormaliser(qmul(exp_q(xi), q[lien]));
            let suivant = energie(&essai)?;
            if flux
                .uniforme(k as u32, i as u32, lien as u32, TAG_ACCEPTATION)
                .ln()
                < e - suivant
            {
                q = essai;
                e = suivant;
                acceptes += 1;
            }
        }
        let s = statut(acceptes > 0);
        let residu = modele
            .residu(&q)?
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        if i >= c.therm {
            resume.ajouter(&s, residu, 2, acceptes);
        }
        if let Some(m) = c.mesure(i) {
            let x = DeuxLiens::observable(&q);
            write!(&mut sortie.csv, "{k},{m},{:.17e},{:.17e}", x, x * x).expect("String");
            diagnostic(&mut sortie.csv, &s, residu);
        }
    }
    sortie.resume = resume.json(k);
    sortie.secondes = debut.elapsed().as_secs_f64();
    Ok(sortie)
}

struct Depart {
    reseau: Option<Arc<Reseau>>,
    v: Vec<Q>,
    q: ChampHote,
}
fn preparer(c: &Config) -> Resultat<Option<Depart>> {
    if c.jouet.is_some() {
        return Ok(None);
    }
    let mut q = match &c.entree_u0 {
        Some(p) => npy::lire_champ(p)?,
        None => {
            let mut q = vec![0.0; 16 * c.l.pow(4)];
            for lien in q.chunks_exact_mut(4) {
                lien[0] = 1.0;
            }
            ChampHote { l: c.l, q }
        }
    };
    if q.l != c.l || q.defaut_unitarite() > 1e-10 || !q.q.iter().all(|x| x.is_finite()) {
        return Err(config("U0 incompatible avec L ou non unitaire"));
    }
    if c.methode == "penalite" && c.kappa.is_none() {
        if let Some(p) = &c.entree_v {
            let (_, l) = npy::inspecter_champ(p)?;
            if l * 3 != c.l {
                return Err(config("V doit avoir le côté L/3"));
            }
        }
        return Ok(Some(Depart {
            reseau: None,
            v: Vec::new(),
            q,
        }));
    }
    let champ_v = npy::lire_champ(c.entree_v.as_ref().expect("validé"))?;
    if champ_v.l != c.l / 3 {
        return Err(config("V doit avoir le côté L/3"));
    }
    let reseau = Arc::new(Reseau::nouveau(c.l)?);
    let v: Vec<Q> = (0..reseau.n_c).map(|i| champ_v.lien(i)).collect();
    let modele = FibreSu2::nouveau(reseau.clone(), v.clone(), c.beta, c.fils)?;
    let initial = modele.residu_max(&q)?;
    q = projeter_continuation(
        reseau.clone(),
        &q,
        &v,
        c.beta,
        c.fils,
        if initial > 1e-3 { 8 } else { 1 },
    )
    .map_err(|e| Erreur::Validation(format!("projection initiale : {e}")))?;
    let final_ = modele.residu_max(&q)?;
    if !final_.is_finite() || final_ > 1e-12 {
        return Err(Erreur::Validation(format!(
            "projection initiale : résidu {final_:e} > 1e-12"
        )));
    }
    Ok(Some(Depart {
        reseau: Some(reseau),
        v,
        q,
    }))
}

fn lancer_chaine(c: &Config, k: usize, depart: Option<&Depart>) -> Resultat<SortieChaine> {
    match c.jouet.as_deref() {
        Some("courbe") => {
            let m = Courbe::nouveau(c.masse[0], c.masse[1]);
            route_rattle(c, k, &m, [0.0, 0.0], |q, mesure, s| {
                if mesure.is_some() {
                    write!(
                        &mut s.csv,
                        ",{:.17e},{:.17e}",
                        q[0].cos(),
                        (2.0 * q[0]).cos()
                    )
                    .expect("String");
                }
                Ok(())
            })
        }
        Some("deux-liens") if c.methode == "penalite" => jouet_penalite(c, k),
        Some("deux-liens") => {
            let m = DeuxLiens::nouveau(c.beta, c.t, c.v);
            let q = m.depart(c.v, [1.0, 0.0, 0.0, 0.0]);
            route_rattle(c, k, &m, q, |q, mesure, s| {
                if mesure.is_some() {
                    let x = DeuxLiens::observable(q);
                    write!(&mut s.csv, ",{:.17e},{:.17e}", x, x * x).expect("String");
                }
                Ok(())
            })
        }
        _ => {
            let d = depart.expect("réseau préparé");
            if c.methode == "penalite" {
                return route_penalite(c, k, d.reseau.clone(), &d.v, &d.q);
            }
            let modele = FibreSu2::nouveau(
                d.reseau.clone().expect("fibre"),
                d.v.clone(),
                c.beta,
                c.fils,
            )?;
            route_rattle(c, k, &modele, d.q.clone(), |q, mesure, s| {
                if mesure.is_some() {
                    write!(
                        &mut s.csv,
                        ",{:.17e},{:.17e},{:.17e}",
                        plaquette_moyenne(q),
                        action_wilson(q, c.beta),
                        0.0
                    )
                    .expect("String");
                }
                sauver(c, k, mesure, q, &mut s.fichiers)
            })
        }
    }
}

fn ecrire(chemin: &Path, texte: &str) -> Resultat<()> {
    GardeSsd::machine().verifier(chemin, texte.len() as u64)?;
    std::fs::write(chemin, texte)
        .map_err(|e| Erreur::io(format!("écriture de {}", chemin.display()), e))
}

pub fn executer(m: &ArgMatches) -> Resultat<i32> {
    let mut c = Config::lire(m)?;
    // Chemins d'entrée absolus : la recette reste rejouable depuis un autre cwd.
    let mut entrees = Vec::new();
    for p in [&mut c.entree_v, &mut c.entree_u0].into_iter().flatten() {
        *p = std::fs::canonicalize(&*p).map_err(|e| Erreur::io("chemin d'entrée", e))?;
        entrees.push((
            p.to_string_lossy().into_owned(),
            crate::regen::manifeste::sha_fichier(p)?,
        ));
    }
    let depart = preparer(&c)?;
    let nombre_dumps = if c.dump == 0 || c.nmeas == 0 {
        0
    } else {
        1 + (c.nmeas - 1) / c.dump
    };
    let octets = (c.chaines as u64)
        .checked_mul(
            (c.nmeas as u64)
                .saturating_mul(512)
                .saturating_add(if c.jouet.is_none() {
                    (nombre_dumps as u64 + 1)
                        .saturating_mul(npy::octets_champ(c.l, FormatChamp::Q64))
                } else {
                    0
                })
                .saturating_add(4096),
        )
        .ok_or_else(|| config("sortie trop volumineuse"))?;
    let mut man = Manifeste::ouvrir_avec_garde(
        &c.sortie,
        "kvu fibre echantillonner",
        &std::env::args().collect::<Vec<_>>(),
        &GardeSsd::machine(),
        octets,
    )?;
    let travail = || -> Resultat<Vec<SortieChaine>> {
        if c.methode == "penalite" {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(c.fils.min(c.chaines))
                .build()
                .map_err(|e| config(format!("pool des chaînes : {e}")))?;
            pool.install(|| {
                (0..c.chaines)
                    .into_par_iter()
                    .map(|k| lancer_chaine(&c, k, depart.as_ref()))
                    .collect()
            })
        } else {
            (0..c.chaines)
                .map(|k| lancer_chaine(&c, k, depart.as_ref()))
                .collect()
        }
    };
    let resultat = (|| -> Resultat<()> {
        let sorties = travail()?;
        let mut csv = if c.jouet.is_some() {
            "chaine,mesure,obs1,obs2,accepte,delta_h,residu_max,echec,nb_newton\n"
        } else {
            "chaine,mesure,P,S,penalite,accepte,delta_h,residu_max,echec,nb_newton\n"
        }
        .to_string();
        let mut resumes = Vec::new();
        // La CLI racine tient StderrLock sur le fil principal : aucun eprintln dans les travailleurs.
        for (k, s) in sorties.into_iter().enumerate() {
            eprintln!("fibre echantillonner : chaîne {k}, {:.3} s", s.secondes);
            csv.push_str(&s.csv);
            resumes.push(s.resume);
            for nom in s.fichiers {
                man.fichier(&c.sortie.join(nom))?;
            }
        }
        ecrire(&c.sortie.join("echantillons.csv"), &csv)?;
        man.fichier(&c.sortie.join("echantillons.csv"))?;
        let resume = serde_json::to_string_pretty(&json!({"chaines": resumes}))
            .map_err(|e| Erreur::Format(e.to_string()))?
            + "\n";
        ecrire(&c.sortie.join("resume.json"), &resume)?;
        man.fichier(&c.sortie.join("resume.json"))?;
        let p = c.parametres();
        man.champ("parametres", &p.to_string())?;
        man.champ("entrees", &json!(entrees).to_string())?;
        let mut fichiers = man.fichiers();
        fichiers.sort_by(|a, b| a.0.cmp(&b.0));
        let regen = super::manifeste::ecrire_manifeste_regen(
            &c.sortie,
            &c.commande(),
            p,
            &entrees,
            &fichiers,
        )?;
        man.fichier(&regen)?;
        Ok(())
    })();
    match resultat {
        Ok(()) => {
            man.terminer(0)?;
            Ok(0)
        }
        Err(e) => {
            let _ = man.echouer(e.code_sortie(), &e.to_string());
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fibre_contrainte_adaptateur_local_accord_avec_moteur() {
        let q = ChampHote::froid(6).unwrap();
        let reseau = Arc::new(Reseau::nouveau(6).unwrap());
        let v = super::super::modele_su2::cible_depuis(&q, &reseau);
        let params = ParamsPenalite {
            beta: 2.3,
            kappa: Some(2e-3),
            delta: 0.02,
        };
        let flux = Flux::nouveau(26102612077);
        let mut a =
            ChainePenalite::nouvelle(Some(reseau.clone()), v.clone(), q.clone(), params, flux, 1)
                .unwrap();
        let mut b = PenaliteImpaire::nouvelle(reseau, v, q, params, flux, 1);
        assert_eq!(a.balayage(0), b.balayage(0));
        assert_eq!(a.champ().q, b.champ().q);
        assert_eq!(a.penalite_totale(), b.penalite_totale());
        assert_eq!(a.residu_max(), b.residu_max());
    }

    #[test]
    fn fibre_contrainte_options_et_compteurs() {
        commande().debug_assert();
        let m = commande()
            .try_get_matches_from([
                "echantillonner",
                "--methode",
                "rattle",
                "--jouet",
                "courbe",
                "--therm",
                "2",
                "--nmeas",
                "3",
                "--skip",
                "2",
                "--outdir",
                "x",
            ])
            .unwrap();
        let c = Config::lire(&m).unwrap();
        assert_eq!(c.total(), 8);
        assert_eq!(
            (0..8).filter_map(|i| c.mesure(i)).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(c.rattle.nb_pas, 10);
        assert!(!c
            .commande()
            .iter()
            .any(|s| s == "--sortie" || s == "--outdir"));
        let m = commande()
            .try_get_matches_from([
                "echantillonner",
                "--methode",
                "penalite",
                "--L",
                "2",
                "--beta",
                "2",
                "--kappa",
                "inf",
                "--sortie",
                "x",
            ])
            .unwrap();
        let c = Config::lire(&m).unwrap();
        assert_eq!(c.delta, 0.5);
    }
}
