//! Gates WP22-A2 : protocoles et seuils gelés par l'avenant v5.WP22, section A.
//! Les mutants sont verts uniquement lorsque le défaut attendu est détecté.
use super::bloc::{coordonnees, deplacer, site, ubar_lien, Q};
use super::gates::{champ_haar, ecart_champs, jauge_haar, transformer_jauge, Max2};
use super::geometrie::{Geometrie, Table};
use super::lissage::{self, bloquer_operateur, OptionsOperateur, Variante};
use super::operateur::Operateur;
use super::support;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::groupes::gates::{certifiable, tout_vert, LigneGate, Verdict};
use crate::heritage::manifest::sha256_fichier;
use crate::heritage::su2_hb::{qdag, qmul};
use crate::obs::ref_cpu::{boucle, Reseau};
use crate::stockage::npy::lire_champ;
use rayon::prelude::*;
use serde_json::Value;
use std::path::{Path, PathBuf};

// A.0 / A.1 / A.3 : toute modification de ces constantes exige un nouvel avenant.
pub const SCHEMA_CERT: &str = "kvu-rg-ql-cert/1";
pub const CERT_DEFAUT: &str = "validation/rg_ql/rg_ql.json";
pub const SEUIL_REFERENCE: f64 = 1e-12;
pub const SEUIL_COVARIANCE: f64 = 1e-12;
pub const SEUIL_COVARIANCE_FLOT: f64 = 1e-11;
pub const SEUIL_MUTANT: f64 = 1e-3;
pub const SEUIL_BOUCLES: f64 = 1e-13;
pub const SEUIL_NORM: f64 = 1e-6;
pub const SEUIL_UNIT: f64 = 1e-14;
pub const GRAINE_JAUGE: u64 = 26_102_822_001;
pub const GRAINE_HAAR: u64 = 26_102_822_002;
pub const GRAINES_FIXTURES: [u64; 3] = [7, 11, 13];
pub const OPERATEURS: [(&str, &str); 5] = [
    ("DEC", "dec"),
    ("APE6", "ape:0.5:6"),
    ("APE13", "ape:0.5:13"),
    ("FLOT1", "flot:1:0.01"),
    ("AGR1", "agr:1"),
];
pub const REFERENCES: [(&str, &str); 5] = [
    ("DEC", "dec"),
    ("APE6", "ape:0.5:6"),
    ("APE13", "ape:0.5:13"),
    ("AGR1", "agr:1"),
    ("APE40", "ape:0.5:40"),
];
pub const SUPPORTS: [(&str, &str); 6] = [
    ("DEC", "dec"),
    ("APE6", "ape:0.5:6"),
    ("APE13", "ape:0.5:13"),
    ("FLOT002", "flot:0.02:0.01"),
    ("AGR1", "agr:1"),
    ("3CENTRE", "3centre"),
];
pub const SUPPORT_TAILLES: [(&str, usize); 4] = [
    ("dec", 3),
    ("ape:0.5:6", 7377),
    ("ape:0.5:13", 62129),
    ("agr:1", 33),
];
pub const LIENS_SUPPORT: [([usize; 4], usize); 2] = [([0; 4], 0), ([1, 2, 0, 1], 3)];
pub const L_HAAR: usize = 12;
pub const ECHANTILLON_SUPPORT: usize = 32;
pub const ANGLE_PERTURBATION: f64 = 1.0;
pub const AXE_PERTURBATION: [f64; 3] = [1.0, 2.0, 3.0];
pub const GRANDL: &str = "L12_b2.3_c1/cfg_002000.npy";
pub const ENTREE_L12: &str = "L12_b2.3_c1_cfg_002000";
pub const ZERO: [&str; 3] = ["ape:0.5:0", "flot:0", "flot:0:0.01"];
pub const RECTANGLES: [(usize, usize); 4] = [(1, 1), (1, 2), (2, 1), (2, 2)];
pub const FLOTS: [(&str, f64, usize); 3] = [
    ("flot:1:0.01", 0.01, 100),
    ("flot:0.5:0.02", 0.02, 25),
    ("flot:0.02:0.01", 0.01, 2),
];
pub const FLOT_MUTANT: &str = "flot:1:0.005";
pub const L_ONDE: usize = 12;
pub const AMPLITUDE_ONDE: f64 = 1e-4;
pub const EPS_ONDE: f64 = 0.01;
pub const PAS_ONDE: usize = 50;
pub const ALPHA_ONDE: f64 = 0.5;
pub const N_APE_ONDE: usize = 6;
pub const RAYON_REFUSES: [&str; 5] = [
    "ape:0.5:40",
    "ape:0.5:14",
    "ape:0.8:1",
    "flot:1.13",
    "flot:1:0.2",
];
pub const RAYON_ACCEPTES: [&str; 4] =
    ["ape:0.5:13", "ape:0.75:9", "flot:1.125:0.005", "flot:1:0.1"];
pub const INVALIDES: [&str; 6] = [
    "ape:1:3",
    "ape:0.5:-1",
    "flot:-1",
    "flot:0.25:0.1",
    "agr:-1",
    "inconnu",
];
pub const FILS_TESTES: [usize; 3] = [1, 4, 16];
pub const OPERATEURS_FILS: [&str; 4] = ["dec", "ape:0.5:13", "flot:1:0.01", "agr:1"];
pub const IDS_OBLIGATOIRES: [&str; 51] = [
    "RG-QL-FICHIERS",
    "RG-QL-REF-DEC",
    "RG-QL-REF-APE6",
    "RG-QL-REF-APE13",
    "RG-QL-REF-AGR1",
    "RG-QL-REF-APE40",
    "RG-QL-REF-L12",
    "RG-QL-REF-MUT",
    "RG-QL-JAUGE-DEC",
    "RG-QL-JAUGE-APE6",
    "RG-QL-JAUGE-APE13",
    "RG-QL-JAUGE-FLOT1",
    "RG-QL-JAUGE-AGR1",
    "RG-QL-JAUGE-L12",
    "RG-QL-JAUGE-MUT",
    "RG-QL-HYPER-DEC",
    "RG-QL-HYPER-APE6",
    "RG-QL-HYPER-APE13",
    "RG-QL-HYPER-FLOT1",
    "RG-QL-HYPER-AGR1",
    "RG-QL-HYPER-MUT",
    "RG-QL-SUPPORT-TAILLES",
    "RG-QL-SUPPORT-DEC",
    "RG-QL-SUPPORT-APE6",
    "RG-QL-SUPPORT-APE13",
    "RG-QL-SUPPORT-FLOT002",
    "RG-QL-SUPPORT-AGR1",
    "RG-QL-SUPPORT-3CENTRE",
    "RG-QL-SUPPORT-MUT",
    "RG-QL-SUPPORT-MUT2",
    "RG-QL-DEC-ZERO",
    "RG-QL-DEC-ZERO-L12",
    "RG-QL-DEC-BOUCLES",
    "RG-QL-DEC-MUT",
    "RG-QL-FLOT",
    "RG-QL-FLOT-L12",
    "RG-QL-FLOT-MUT",
    "RG-QL-NORM-FLOT",
    "RG-QL-NORM-APE",
    "RG-QL-NORM-MUT",
    "RG-QL-UNIT-DEC",
    "RG-QL-UNIT-APE6",
    "RG-QL-UNIT-APE13",
    "RG-QL-UNIT-FLOT1",
    "RG-QL-UNIT-AGR1",
    "RG-QL-UNIT-L12",
    "RG-QL-UNIT-MUT",
    "RG-QL-RAYON",
    "RG-QL-FILS",
    "RG-QL-3CENTRE",
    "RG-QL-FLOT-RG",
];

/// A.3 énumère cinq identifiants L12 (et 51 au total) ; la spécification de code A2 disait « six » par erreur.
/// La liste fermée de l'avenant gelé fait autorité : aucun sixième identifiant n'est inventé.
pub const IDS_L12: [&str; 5] = [
    "RG-QL-REF-L12",
    "RG-QL-JAUGE-L12",
    "RG-QL-DEC-ZERO-L12",
    "RG-QL-FLOT-L12",
    "RG-QL-UNIT-L12",
];

pub struct Contexte {
    pub fixtures: PathBuf,
    pub references: PathBuf,
    pub grandl: Option<PathBuf>,
    pub fils: usize,
    pub seul: Vec<String>,
    pub mutants: bool,
}

fn validation(message: impl Into<String>) -> Erreur {
    Erreur::Validation(message.into())
}
fn verifier(ok: bool, message: &str) -> Resultat<()> {
    if ok {
        Ok(())
    } else {
        Err(validation(message))
    }
}
fn lire_json(p: &Path) -> Resultat<Value> {
    let t = std::fs::read(p).map_err(|e| Erreur::io("lecture JSON de gate", e))?;
    serde_json::from_slice(&t).map_err(|e| Erreur::Format(e.to_string()))
}
fn hacher(p: &Path) -> Resultat<String> {
    sha256_fichier(p).map_err(|e| Erreur::io("hachage de gate", e))
}
fn calcul(c: &ChampHote, spec: &str, variante: Variante, fils: usize) -> Resultat<ChampHote> {
    let grossier = bloquer_operateur(
        c,
        &Operateur::analyser(spec)?,
        &OptionsOperateur { fils, variante },
    )?
    .grossier;
    verifier(grossier.q.iter().all(|x| x.is_finite()), "sortie non finie")?;
    Ok(grossier)
}
fn bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn ecart(a: &ChampHote, b: &ChampHote) -> Resultat<f64> {
    verifier(
        a.l == b.l && a.q.len() == b.q.len(),
        "dimensions différentes",
    )?;
    verifier(
        a.q.iter().chain(&b.q).all(|x| x.is_finite()),
        "composante non finie",
    )?;
    Ok(ecart_champs(a, b).comp)
}
fn limite(m: f64, seuil: f64) -> Resultat<String> {
    verifier(
        m.is_finite() && m <= seuil,
        &format!("max |Δq| = {m:.6e}, seuil = {seuil:.6e}"),
    )?;
    Ok(format!("max |Δq| = {m:.6e}, seuil = {seuil:.6e}"))
}
fn mutant(m: f64) -> Resultat<String> {
    verifier(
        m.is_finite() && m >= SEUIL_MUTANT,
        &format!("mutant non détecté : {m:.6e}"),
    )?;
    Ok(format!("mutant détecté, max |Δq| = {m:.6e}"))
}
fn unitarite(m: f64, mutante: bool) -> Resultat<String> {
    let seuil = if mutante { SEUIL_MUTANT } else { SEUIL_UNIT };
    let detail = format!("max |‖q‖ − 1| = {m:.6e}, seuil = {seuil:.6e}");
    verifier(
        m.is_finite() && if mutante { m >= seuil } else { m <= seuil },
        &detail,
    )?;
    Ok(detail)
}
fn seuil_cov(spec: &str) -> f64 {
    if spec.starts_with("flot:") {
        SEUIL_COVARIANCE_FLOT
    } else {
        SEUIL_COVARIANCE
    }
}

/// Échantillon canonique A.0 : indices triés, rangs floor(j |S| / k).
pub fn echantillon(s: &[usize], k: usize) -> Vec<usize> {
    let mut s = s.to_vec();
    s.sort_unstable();
    s.dedup();
    if s.len() <= k {
        s
    } else {
        (0..k).map(|j| s[j * s.len() / k]).collect()
    }
}

/// Vérification fermée, indépendante du filtre CLI : aucun manque ni doublon ni SKIP admis.
pub fn verifier_certificat(lignes: &[LigneGate]) -> Resultat<()> {
    verifier(
        tout_vert(lignes) && certifiable(lignes, "cpu").is_some(),
        "certificat refusé : ligne FAIL ou SKIP",
    )?;
    for id in IDS_OBLIGATOIRES {
        let occurrences: Vec<_> = lignes.iter().filter(|l| l.id == id).collect();
        verifier(
            occurrences.len() == 1 && occurrences[0].verdict == Verdict::Pass,
            &format!("certificat refusé : {id} doit apparaître exactement une fois avec PASS"),
        )?;
    }
    Ok(())
}

fn nettoyer(message: String, ctx: &Contexte) -> String {
    let mut t = message;
    let mut chemins = vec![
        std::env::temp_dir().join(format!("kvu_ql_gates_{}", std::process::id())),
        ctx.fixtures.clone(),
        ctx.references.clone(),
        std::env::temp_dir(),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")),
    ];
    if let Some(p) = &ctx.grandl {
        chemins.push(p.clone());
    }
    // Supprimer les chemins les plus spécifiques d'abord.
    chemins.sort_by_key(|p| std::cmp::Reverse(p.as_os_str().len()));
    for p in chemins {
        let s = p.to_string_lossy();
        if !s.is_empty() {
            t = t.replace(s.as_ref(), "<dossier>");
        }
    }
    t.lines().collect::<Vec<_>>().join(" ")
}

/// Ordonnanceur commun, également testé avec des mesures factices sans lancer la batterie de verdict.
fn emettre(ctx: &Contexte, mut mesurer: impl FnMut(&str) -> Resultat<String>) -> Vec<LigneGate> {
    IDS_OBLIGATOIRES
        .iter()
        .filter(|id| {
            (ctx.mutants || !id.contains("-MUT"))
                && (ctx.seul.is_empty() || ctx.seul.iter().any(|p| id.starts_with(p)))
        })
        .map(|&id| {
            if id.ends_with("-L12") && ctx.grandl.is_none() {
                return LigneGate::nouvelle(Verdict::Skip, id, "--grandl absent");
            }
            match mesurer(id) {
                Ok(t) => LigneGate::nouvelle(Verdict::Pass, id, t),
                Err(e) => LigneGate::nouvelle(Verdict::Fail, id, nettoyer(e.to_string(), ctx)),
            }
        })
        .collect()
}

pub fn batterie(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    if ctx.fils == 0 {
        return Err(Erreur::Config("--fils doit être ≥ 1".into()));
    }
    // Les E/S des fixtures sont des erreurs de commande, pas des mesures manquantes silencieuses.
    let fixtures: Vec<ChampHote> = GRAINES_FIXTURES
        .iter()
        .map(|g| lire_champ(&ctx.fixtures.join(format!("cfg_s{g}_q64.npy"))))
        .collect::<Resultat<_>>()?;
    let mut mesures = Mesures {
        ctx,
        fixtures,
        norm: None,
    };
    Ok(emettre(ctx, |id| mesures.executer(id)))
}

struct Mesures<'a> {
    ctx: &'a Contexte,
    fixtures: Vec<ChampHote>,
    norm: Option<(f64, f64)>,
}
impl Mesures<'_> {
    fn grand(&self) -> Resultat<ChampHote> {
        lire_champ(
            &self
                .ctx
                .grandl
                .as_ref()
                .ok_or_else(|| validation("--grandl absent"))?
                .join(GRANDL),
        )
    }
    fn reference(&self, nom: &str, suffixe: &str) -> Resultat<ChampHote> {
        lire_champ(
            &self
                .ctx
                .references
                .join(format!("v_{nom}_{suffixe}_q64.npy")),
        )
    }
    fn fichiers(&self) -> Resultat<String> {
        let refs = lire_json(&self.ctx.references.join("manifeste.json"))?;
        let fs = refs["fichiers"]
            .as_object()
            .ok_or_else(|| validation("manifeste des références sans fichiers"))?;
        verifier(
            fs.len() == REFERENCES.len() * (GRAINES_FIXTURES.len() + 1),
            "20 références attendues",
        )?;
        for (nom, _) in REFERENCES {
            for suffixe in GRAINES_FIXTURES
                .iter()
                .map(|g| format!("s{g}"))
                .chain(std::iter::once(ENTREE_L12.to_string()))
            {
                let nom = format!("v_{nom}_{suffixe}_q64.npy");
                verifier(
                    fs.get(&nom).and_then(Value::as_str)
                        == Some(hacher(&self.ctx.references.join(&nom))?.as_str()),
                    &format!("sha256 incorrect : {nom}"),
                )?;
            }
        }
        let man = lire_json(&self.ctx.fixtures.join("manifeste.json"))?;
        for graine in GRAINES_FIXTURES {
            let nom = format!("cfg_s{graine}_q64.npy");
            verifier(
                man["fichiers"][&nom].as_str()
                    == Some(hacher(&self.ctx.fixtures.join(&nom))?.as_str()),
                &format!("sha256 incorrect : {nom}"),
            )?;
        }
        Ok("sha256 conformes : 20 références et 3 fixtures".into())
    }
    fn refs(&self, nom: &str, spec: &str, grand: bool, mutante: bool) -> Resultat<String> {
        let mut m: f64 = if mutante { f64::INFINITY } else { 0.0 };
        let champs = if grand {
            vec![self.grand()?]
        } else {
            self.fixtures.clone()
        };
        if grand {
            let refs = lire_json(&self.ctx.references.join("manifeste.json"))?;
            let p = self.ctx.grandl.as_ref().unwrap().join(GRANDL);
            verifier(
                refs["entrees"][ENTREE_L12]["sha256"].as_str() == Some(hacher(&p)?.as_str()),
                "sha256 de G incorrect",
            )?;
        }
        for (i, c) in champs.iter().enumerate() {
            let suffixe = if grand {
                ENTREE_L12.into()
            } else {
                format!("s{}", GRAINES_FIXTURES[i])
            };
            let obtenu = calcul(c, spec, Variante::Normale, self.ctx.fils)?;
            let d = ecart(&obtenu, &self.reference(nom, &suffixe)?)?;
            m = if mutante { m.min(d) } else { m.max(d) };
        }
        if mutante {
            mutant(m)
        } else {
            limite(m, SEUIL_REFERENCE)
        }
    }
    fn jauge(&self, champs: &[ChampHote], spec: &str, variante: Variante) -> Resultat<f64> {
        let mut m = Max2::default();
        for c in champs {
            let g = jauge_haar(c.l, GRAINE_JAUGE);
            let a = calcul(c, spec, variante, self.ctx.fils)?;
            let b = calcul(&transformer_jauge(c, &g), spec, variante, self.ctx.fils)?;
            for mu in 0..4 {
                for s in 0..a.v() {
                    let x = coordonnees(s, a.l).map(|t| 3 * t);
                    let y = deplacer(x, mu, 3, c.l);
                    let attendu = qmul(
                        qmul(g[site(x, c.l)], a.lien(mu * a.v() + s)),
                        qdag(g[site(y, c.l)]),
                    );
                    m.ajouter(attendu, b.lien(mu * b.v() + s));
                }
            }
        }
        Ok(m.comp)
    }
    fn zero(&self, champs: &[ChampHote]) -> Resultat<String> {
        for c in champs {
            let dec = calcul(c, "dec", Variante::Normale, self.ctx.fils)?;
            for spec in ZERO {
                verifier(
                    bits(
                        &dec.q,
                        &calcul(c, spec, Variante::Normale, self.ctx.fils)?.q,
                    ),
                    "n ou t nul : différence au bit",
                )?;
            }
        }
        Ok("n = 0 et t = 0 : égalité au bit".into())
    }
    fn boucles(&self, variante: Variante) -> Resultat<f64> {
        let mut m = Max2::default();
        for c in &self.fixtures {
            let v = calcul(c, "dec", variante, self.ctx.fils)?;
            let fine = Reseau::nouveau(c)?;
            let grosse = Reseau::nouveau(&v)?;
            for s in 0..v.v() {
                for mu in 0..4 {
                    for nu in 0..4 {
                        if mu == nu {
                            continue;
                        }
                        let sf = site(coordonnees(s, v.l).map(|t| 3 * t), c.l);
                        for (r, t) in RECTANGLES {
                            m.ajouter(
                                boucle(&grosse, s, mu, nu, r, t),
                                boucle(&fine, sf, mu, nu, 3 * r, 3 * t),
                            );
                        }
                    }
                }
            }
        }
        Ok(m.comp)
    }
    fn flot(&self, champs: &[ChampHote], grand: bool) -> Resultat<String> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.ctx.fils)
            .build()
            .map_err(|e| Erreur::Config(e.to_string()))?;
        for c in champs {
            for &(spec, eps, n) in if grand { &FLOTS[..1] } else { &FLOTS[..] } {
                let lisse =
                    pool.install(|| crate::obs::flot::flot_champ(None, None, 0, Some(c), eps, n))?;
                let attendu = calcul(&lisse, "dec", Variante::Normale, self.ctx.fils)?;
                verifier(
                    bits(
                        &attendu.q,
                        &calcul(c, spec, Variante::Normale, self.ctx.fils)?.q,
                    ),
                    "flot OBS puis dec : différence au bit",
                )?;
            }
        }
        Ok("flot OBS puis dec : égalité au bit".into())
    }
    fn norme(&mut self, id: &str) -> Resultat<String> {
        let (rf, ra) = match self.norm {
            Some(r) => r,
            None => {
                let p = onde();
                let f = lissage::lisser_flot(&p, EPS_ONDE, PAS_ONDE, self.ctx.fils)?;
                let (a, _) = lissage::lisser_ape(
                    &p,
                    ALPHA_ONDE,
                    N_APE_ONDE,
                    &OptionsOperateur::normale(self.ctx.fils),
                )?;
                let r = (
                    amplitude(&f) / AMPLITUDE_ONDE,
                    amplitude(&a) / AMPLITUDE_ONDE,
                );
                self.norm = Some(r);
                r
            }
        };
        let p2 = 4.0 * (std::f64::consts::PI / L_ONDE as f64).sin().powi(2);
        let t = EPS_ONDE * PAS_ONDE as f64;
        if id.ends_with("MUT") {
            let ef = (rf / (-2.0 * t * p2).exp() - 1.0).abs();
            let ea = (ra / (1.0 - ALPHA_ONDE * p2 / 3.0).powi(N_APE_ONDE as i32) - 1.0).abs();
            verifier(
                ef > SEUIL_NORM && ea > SEUIL_NORM,
                "normalisations fausses non rejetées",
            )?;
            return Ok(format!(
                "normalisations fausses rejetées : écarts flot = {ef:.6e}, APE = {ea:.6e}"
            ));
        }
        let (rho, theorique) = if id.ends_with("FLOT") {
            (rf, (-t * p2).exp())
        } else {
            (ra, (1.0 - ALPHA_ONDE * p2 / 6.0).powi(N_APE_ONDE as i32))
        };
        let d = (rho / theorique - 1.0).abs();
        let detail = format!("ρ = {rho:.12e}, ρ_th = {theorique:.12e}, écart relatif = {d:.6e}");
        verifier(d.is_finite() && d <= SEUIL_NORM, &detail)?;
        Ok(detail)
    }
    fn executer(&mut self, id: &str) -> Resultat<String> {
        let normal = Variante::Normale;
        let fils = self.ctx.fils;
        match id {
            "RG-QL-FICHIERS" => self.fichiers(),
            "RG-QL-REF-L12" => {
                let mut detail = Vec::new();
                for (nom, spec) in REFERENCES {
                    detail.push(format!("{nom} : {}", self.refs(nom, spec, true, false)?));
                }
                Ok(detail.join(" ; "))
            }
            "RG-QL-REF-MUT" => self.refs("APE13", "ape:0.5:6", false, true),
            "RG-QL-JAUGE-L12" => {
                let c = self.grand()?;
                let mut detail = Vec::new();
                for (nom, spec) in OPERATEURS {
                    detail.push(format!(
                        "{nom} : {}",
                        limite(
                            self.jauge(std::slice::from_ref(&c), spec, normal)?,
                            seuil_cov(spec)
                        )?
                    ));
                }
                Ok(detail.join(" ; "))
            }
            "RG-QL-JAUGE-MUT" => mutant(self.jauge(
                &self.fixtures[..1],
                "ape:0.5:6",
                Variante::AgrafeMalOrientee,
            )?),
            "RG-QL-HYPER-MUT" => hyper(
                &self.fixtures[0],
                "ape:0.5:6",
                Variante::AgrafesPlusSeulement,
                fils,
            ),
            "RG-QL-SUPPORT-TAILLES" => {
                for (spec, taille) in SUPPORT_TAILLES {
                    verifier(
                        support::support(&Operateur::analyser(spec)?, L_HAAR, [0; 4], 0).len()
                            == taille,
                        &format!("taille du support incorrecte : {spec}"),
                    )?;
                }
                Ok("12⁴ : dec = 3, APE6 = 7377, APE13 = 62129, agr = 33".into())
            }
            "RG-QL-SUPPORT-MUT" => support_gate(
                "ape:0.5:6",
                "ape:0.5:6",
                Variante::PasSupplementaire,
                fils,
                true,
                false,
            ),
            "RG-QL-SUPPORT-MUT2" => {
                support_gate("ape:0.5:6", "ape:0.5:7", normal, fils, false, true)
            }
            "RG-QL-DEC-ZERO" => self.zero(&self.fixtures),
            "RG-QL-DEC-ZERO-L12" => self.zero(&[self.grand()?]),
            "RG-QL-DEC-BOUCLES" => limite(self.boucles(normal)?, SEUIL_BOUCLES),
            "RG-QL-DEC-MUT" => mutant(self.boucles(Variante::DecOrdreInverse)?),
            "RG-QL-FLOT" => self.flot(&self.fixtures[..1], false),
            "RG-QL-FLOT-L12" => self.flot(&[self.grand()?], true),
            "RG-QL-FLOT-MUT" => {
                let a = calcul(&self.fixtures[0], FLOTS[0].0, normal, fils)?;
                let b = calcul(&self.fixtures[0], FLOT_MUTANT, normal, fils)?;
                verifier(
                    !bits(&a.q, &b.q),
                    "pas d'intégration différents : mutant non détecté",
                )?;
                Ok(format!("mutant détecté, max |Δq| = {:.6e}", ecart(&a, &b)?))
            }
            "RG-QL-NORM-FLOT" | "RG-QL-NORM-APE" | "RG-QL-NORM-MUT" => self.norme(id),
            "RG-QL-UNIT-L12" => {
                let c = self.grand()?;
                let mut m: f64 = 0.0;
                for (_, spec) in OPERATEURS {
                    m = m.max(calcul(&c, spec, normal, fils)?.defaut_unitarite());
                }
                unitarite(m, false)
            }
            "RG-QL-UNIT-MUT" => {
                let mut m: f64 = 0.0;
                for c in &self.fixtures {
                    m = m.max(
                        calcul(c, "ape:0.5:6", Variante::SansProjection, fils)?.defaut_unitarite(),
                    );
                }
                unitarite(m, true)
            }
            "RG-QL-RAYON" | "RG-QL-FILS" | "RG-QL-3CENTRE" | "RG-QL-FLOT-RG" => {
                processus_gate(id, self.ctx, &self.fixtures)
            }
            _ => {
                for (nom, spec) in REFERENCES {
                    if id == format!("RG-QL-REF-{nom}") {
                        return self.refs(nom, spec, false, false);
                    }
                }
                for (nom, spec) in OPERATEURS {
                    if id == format!("RG-QL-JAUGE-{nom}") {
                        return limite(self.jauge(&self.fixtures, spec, normal)?, seuil_cov(spec));
                    }
                    if id == format!("RG-QL-HYPER-{nom}") {
                        return hyper(&self.fixtures[0], spec, normal, fils);
                    }
                    if id == format!("RG-QL-UNIT-{nom}") {
                        let mut m: f64 = 0.0;
                        for c in &self.fixtures {
                            m = m.max(calcul(c, spec, normal, fils)?.defaut_unitarite());
                        }
                        return unitarite(m, false);
                    }
                }
                for (nom, spec) in SUPPORTS {
                    if id == format!("RG-QL-SUPPORT-{nom}") {
                        return support_gate(spec, spec, normal, fils, false, false);
                    }
                }
                Err(validation(format!("gate sans protocole : {id}")))
            }
        }
    }
}

fn onde() -> ChampHote {
    let v = L_ONDE.pow(4);
    let mut p = ChampHote {
        l: L_ONDE,
        q: vec![0.0; 16 * v],
    };
    for li in 0..4 * v {
        p.q[4 * li] = 1.0;
    }
    for s in 0..v {
        let x = coordonnees(s, L_ONDE);
        let theta = AMPLITUDE_ONDE * (std::f64::consts::TAU * x[1] as f64 / L_ONDE as f64).cos();
        p.q[4 * s] = theta.cos();
        p.q[4 * s + 3] = theta.sin();
    }
    p
}
fn amplitude(c: &ChampHote) -> f64 {
    let mut somme = 0.0;
    // Moyenne sur les trois axes transverses, Fourier (2/L) sur x1.
    for s in 0..c.v() {
        let q = c.lien(s);
        let x = coordonnees(s, c.l);
        somme += q[3].atan2(q[0]) * (std::f64::consts::TAU * x[1] as f64 / c.l as f64).cos();
    }
    2.0 * somme / c.v() as f64
}

// Même action que gates::{g_site,g_lien}, autour de ctr = 0 sur les deux réseaux.
fn g_site(x: [usize; 4], perm: &[usize; 4], eps: &[i8; 4], modulo: usize) -> [usize; 4] {
    let mut y = [0; 4];
    for m in 0..4 {
        y[perm[m]] = (eps[m] as isize * x[m] as isize).rem_euclid(modulo as isize) as usize;
    }
    y
}
fn g_lien(
    x: [usize; 4],
    nu: usize,
    perm: &[usize; 4],
    eps: &[i8; 4],
    modulo: usize,
) -> ([usize; 4], usize, i8) {
    let y = g_site(x, perm, eps, modulo);
    let d = perm[nu];
    (
        if eps[nu] == 1 {
            y
        } else {
            deplacer(y, d, -1, modulo)
        },
        d,
        eps[nu],
    )
}
fn transformer_hyper(c: &ChampHote, perm: &[usize; 4], eps: &[i8; 4]) -> ChampHote {
    let mut q = vec![0.0; c.q.len()];
    for mu in 0..4 {
        for s in 0..c.v() {
            let (y, d, sg) = g_lien(coordonnees(s, c.l), mu, perm, eps, c.l);
            let u = c.lien(mu * c.v() + s);
            let u = if sg == 1 { u } else { qdag(u) };
            let i = 4 * (d * c.v() + site(y, c.l));
            q[i..i + 4].copy_from_slice(&u);
        }
    }
    ChampHote { l: c.l, q }
}
fn elements_hyper() -> Vec<([usize; 4], [i8; 4])> {
    let mut elements = Vec::new();
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let p = [a, b, c, d];
                    if (0..4).any(|i| (i + 1..4).any(|j| p[i] == p[j])) {
                        continue;
                    }
                    for k in 0..16 {
                        elements.push((
                            p,
                            std::array::from_fn(|m| if (k >> (3 - m)) & 1 == 0 { 1 } else { -1 }),
                        ));
                    }
                }
            }
        }
    }
    elements
}
fn hyper(c: &ChampHote, spec: &str, variante: Variante, fils: usize) -> Resultat<String> {
    let reference = calcul(c, spec, variante, 1)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(fils)
        .build()
        .map_err(|e| Erreur::Config(e.to_string()))?;
    let elements = elements_hyper();
    let mesures: Vec<Resultat<f64>> = pool.install(|| {
        elements
            .par_iter()
            .map(|(p, e)| {
                let v = calcul(&transformer_hyper(c, p, e), spec, variante, 1)?;
                ecart(&v, &transformer_hyper(&reference, p, e))
            })
            .collect()
    });
    let (mut permutations, mut reflexions) = (0.0f64, 0.0f64);
    for ((_, e), m) in elements.iter().zip(mesures) {
        let m = m?;
        if e.contains(&-1) {
            reflexions = reflexions.max(m);
        } else {
            permutations = permutations.max(m);
        }
    }
    let detail = format!(
        "24 permutations : {permutations:.6e}, 360 éléments avec réflexion : {reflexions:.6e}"
    );
    let ok = if variante == Variante::Normale {
        permutations <= seuil_cov(spec) && reflexions <= seuil_cov(spec)
    } else {
        reflexions >= SEUIL_MUTANT
    };
    verifier(ok, &detail)?;
    Ok(detail)
}

fn support_gate(
    spec: &str,
    support_spec: &str,
    variante: Variante,
    fils: usize,
    sous_estime: bool,
    sur_estime: bool,
) -> Resultat<String> {
    let c = champ_haar(L_HAAR, GRAINE_HAAR);
    let op = Operateur::analyser(spec)?;
    let declare = Operateur::analyser(support_spec)?;
    let table = if op.est_trois_centre() {
        Some(Table::nouvelle(Geometrie::TroisCentre))
    } else {
        None
    };
    let lien = |u: &ChampHote, z, mu| -> Resultat<Q> {
        if let Some(t) = &table {
            Ok(ubar_lien(u, t, z, mu).0)
        } else {
            let v = bloquer_operateur(u, &op, &OptionsOperateur { fils, variante })?.grossier;
            Ok(v.lien(mu * v.v() + site(z, v.l)))
        }
    };
    let axe_norme = AXE_PERTURBATION.iter().map(|x| x * x).sum::<f64>().sqrt();
    let h = [
        ANGLE_PERTURBATION.cos(),
        ANGLE_PERTURBATION.sin() * AXE_PERTURBATION[0] / axe_norme,
        ANGLE_PERTURBATION.sin() * AXE_PERTURBATION[1] / axe_norme,
        ANGLE_PERTURBATION.sin() * AXE_PERTURBATION[2] / axe_norme,
    ];
    let mut detail = Vec::new();
    let (mut fuite, mut toutes_frontieres_actives) = (false, true);
    for (z, mu) in LIENS_SUPPORT {
        let s = support::support(&declare, c.l, z, mu);
        let boule = declare.rayon_support();
        let rayon = boule.unwrap_or(0);
        let depart = if boule.is_some() {
            support::chemin_droit(c.l, z, mu).to_vec()
        } else {
            s.clone()
        };
        let distances = support::distances(c.l, &depart, rayon + 1);
        let couche = |r: usize| -> Vec<usize> {
            distances
                .iter()
                .enumerate()
                .filter(|(_, d)| **d as usize == r)
                .map(|(i, _)| i)
                .collect()
        };
        let front = echantillon(
            &if boule.is_some() {
                couche(rayon)
            } else {
                s.clone()
            },
            ECHANTILLON_SUPPORT,
        );
        let complement: Vec<_> = (0..4 * c.v())
            .filter(|li| s.binary_search(li).is_err())
            .collect();
        let mut dehors = echantillon(&couche(rayon + 1), ECHANTILLON_SUPPORT);
        dehors.extend(echantillon(&complement, ECHANTILLON_SUPPORT));
        dehors.sort_unstable();
        dehors.dedup();
        let base = lien(&c, z, mu)?;
        let perturber = |li: usize| -> Resultat<(bool, f64)> {
            let mut u = c.clone();
            let q = qmul(h, c.lien(li));
            u.q[4 * li..4 * li + 4].copy_from_slice(&q);
            let obtenu = lien(&u, z, mu)?;
            let d = base
                .iter()
                .zip(obtenu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            Ok((!bits(&base, &obtenu), d))
        };
        let mut deltas = Vec::new();
        let mut n_change = 0;
        for li in front {
            let (change, d) = perturber(li)?;
            n_change += usize::from(change);
            deltas.push(d);
        }
        for li in dehors {
            fuite |= perturber(li)?.0;
        }
        let active = n_change > 0;
        toutes_frontieres_actives &= active;
        deltas.sort_by(f64::total_cmp);
        let n = deltas.len();
        verifier(n > 0, "frontière vide")?;
        detail.push(format!(
            "{z:?}/{mu} : frontière {n_change}/{n}, min = {:.6e}, médiane = {:.6e}",
            deltas[0],
            deltas[(n - 1) / 2]
        ));
    }
    let ok = if sous_estime {
        fuite
    } else if sur_estime {
        !toutes_frontieres_actives
    } else {
        !fuite && toutes_frontieres_actives
    };
    detail.push(format!("changement extérieur = {fuite}"));
    let detail = detail.join(" ; ");
    verifier(ok, &detail)?;
    Ok(detail)
}

/// Un répertoire propre par processus ; Drop assure le nettoyage même si une gate échoue.
struct Travail(PathBuf);
impl Travail {
    fn nouveau() -> Resultat<Self> {
        let p = std::env::temp_dir().join(format!("kvu_ql_gates_{}", std::process::id()));
        std::fs::create_dir(&p)
            .map_err(|e| Erreur::io("création du dossier temporaire des gates", e))?;
        Ok(Self(p))
    }
    fn ensemble(&self, fixtures: &Path) -> Resultat<PathBuf> {
        let ens = self.0.join("ensemble");
        let chaine = ens.join("c1");
        std::fs::create_dir_all(&chaine)
            .map_err(|e| Erreur::io("création de l'ensemble temporaire", e))?;
        for (i, g) in GRAINES_FIXTURES.iter().enumerate() {
            std::fs::copy(
                fixtures.join(format!("cfg_s{g}_q64.npy")),
                chaine.join(format!("cfg_{:06}.npy", (i + 1) * 100)),
            )
            .map_err(|e| Erreur::io("copie de fixture", e))?;
        }
        std::fs::write(chaine.join("manifest.json"), r#"{"beta": 2.3, "l": 9}"#)
            .map_err(|e| Erreur::io("écriture du manifeste temporaire", e))?;
        Ok(ens)
    }
}
impl Drop for Travail {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn lire_octets(p: &Path) -> Resultat<Vec<u8>> {
    std::fs::read(p).map_err(|e| Erreur::io("lecture de sortie temporaire", e))
}
fn lancer(args: &[String]) -> Resultat<i32> {
    let exe = std::env::current_exe().map_err(|e| Erreur::io("binaire courant", e))?;
    let sortie = std::process::Command::new(exe)
        .args(args)
        .output()
        .map_err(|e| Erreur::io("lancement du binaire courant", e))?;
    sortie
        .status
        .code()
        .ok_or_else(|| validation("sous-processus terminé par signal"))
}
fn args_bloc(
    entree: &Path,
    sortie: &Path,
    spec: Option<&str>,
    fils: usize,
    hors: bool,
) -> Vec<String> {
    let mut a = vec![
        "rg".into(),
        "bloc".into(),
        "--entree".into(),
        entree.to_string_lossy().into_owned(),
        "--sortie".into(),
        sortie.to_string_lossy().into_owned(),
        "--fils".into(),
        fils.to_string(),
    ];
    if let Some(s) = spec {
        a.extend(["--operateur".into(), s.into()]);
    }
    if hors {
        a.push("--hors-famille".into());
    }
    a
}
fn fichiers_bloc(p: &Path) -> Resultat<(Vec<u8>, Vec<u8>)> {
    Ok((
        lire_octets(p)?,
        lire_octets(&super::cli::chemin_manifeste(p))?,
    ))
}
fn processus_gate(id: &str, ctx: &Contexte, fixtures: &[ChampHote]) -> Resultat<String> {
    let t = Travail::nouveau()?;
    let entree = ctx
        .fixtures
        .join(format!("cfg_s{}_q64.npy", GRAINES_FIXTURES[0]));
    match id {
        "RG-QL-RAYON" => {
            let mut cas: Vec<(&str, bool, i32)> =
                RAYON_REFUSES.iter().map(|&s| (s, false, 2)).collect();
            cas.extend(RAYON_ACCEPTES.iter().map(|&s| (s, false, 0)));
            cas.push(("ape:0.5:40", true, 0));
            cas.extend(
                INVALIDES
                    .iter()
                    .flat_map(|&s| [(s, false, 2), (s, true, 2)]),
            );
            for (i, (spec, hors, attendu)) in cas.into_iter().enumerate() {
                let p = t.0.join(format!("rayon_{i}.npy"));
                let rc = lancer(&args_bloc(&entree, &p, Some(spec), ctx.fils, hors))?;
                verifier(
                    rc == attendu,
                    &format!("{spec}, hors famille = {hors} : rc {rc}, attendu {attendu}"),
                )?;
                if attendu == 2 {
                    verifier(
                        !p.exists() && !super::cli::chemin_manifeste(&p).exists(),
                        "refus après écriture d'un fichier",
                    )?;
                } else {
                    let man = lire_json(&super::cli::chemin_manifeste(&p))?;
                    verifier(
                        man["hors_famille"] == hors
                            && man["operateur"]["famille"]["dans_famille"] == !hors,
                        "drapeaux de famille incorrects",
                    )?;
                }
            }
            Ok("codes de sortie, refus sans fichier et drapeaux de famille conformes".into())
        }
        "RG-QL-FILS" => {
            for (i, spec) in OPERATEURS_FILS.iter().enumerate() {
                let mut reference = None;
                for fils in FILS_TESTES {
                    let p = t.0.join(format!("fils_{i}_{fils}.npy"));
                    verifier(
                        lancer(&args_bloc(&entree, &p, Some(spec), fils, false))? == 0,
                        "échec du sous-processus rg bloc",
                    )?;
                    let octets = fichiers_bloc(&p)?;
                    if let Some(r) = &reference {
                        verifier(r == &octets, "NPY ou manifeste dépend du nombre de fils")?;
                    } else {
                        reference = Some(octets);
                    }
                }
            }
            Ok("fils 1, 4, 16 : NPY et manifestes identiques à l'octet".into())
        }
        "RG-QL-3CENTRE" => {
            let mut reference = None;
            for (i, spec) in [None, Some("3centre")].into_iter().enumerate() {
                let p = t.0.join(format!("centre_{i}.npy"));
                verifier(
                    lancer(&args_bloc(&entree, &p, spec, ctx.fils, false))? == 0,
                    "échec du sous-processus rg bloc 3centre",
                )?;
                let octets = fichiers_bloc(&p)?;
                if let Some(r) = &reference {
                    verifier(r == &octets, "3centre explicite : bloc différent")?;
                } else {
                    reference = Some(octets);
                }
            }
            let ens = t.ensemble(&ctx.fixtures)?;
            let mut reference = None;
            for (i, spec) in [None, Some("3centre")].into_iter().enumerate() {
                let p = t.0.join(format!("flot_{i}"));
                let mut a = vec![
                    "rg".into(),
                    "flot".into(),
                    "--ensemble".into(),
                    ens.to_string_lossy().into_owned(),
                    "--sortie".into(),
                    p.to_string_lossy().into_owned(),
                    "--estimateurs".into(),
                    "sd".into(),
                    "--fils".into(),
                    ctx.fils.to_string(),
                ];
                if let Some(s) = spec {
                    a.extend(["--operateur".into(), s.into()]);
                }
                verifier(lancer(&a)? == 0, "échec du sous-processus rg flot 3centre")?;
                let octets = lire_octets(&p.join("flot.json"))?;
                if let Some(r) = &reference {
                    verifier(r == &octets, "3centre explicite : flot différent")?;
                } else {
                    reference = Some(octets);
                }
            }
            Ok("3centre explicite : NPY, manifeste et flot.json identiques à l'octet".into())
        }
        "RG-QL-FLOT-RG" => {
            use super::flot::cellule::{flot_options, Estimateurs, ParamFlot, Schema};
            use super::flot::extensions::OptionsFlot;
            use super::flot::sd::{stats_sd, RapportBessel};
            let ens = t.ensemble(&ctx.fixtures)?;
            let par = ParamFlot {
                schema: Schema::Balaban,
                pas: 1,
                estimateurs: Estimateurs {
                    sd: true,
                    plaq: false,
                    mcrg2: false,
                },
                rapport: RapportBessel::I2SurI1,
                courbes: vec![],
                fils: ctx.fils,
            };
            let op = Operateur::analyser("ape:0.5:6")?;
            let options = OptionsFlot {
                operateur: op,
                detail_configs: true,
                ..Default::default()
            };
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(ctx.fils)
                .build()
                .map_err(|e| Erreur::Config(e.to_string()))?;
            let doc = pool.install(|| flot_options(&ens, None, &par, &[], &options))?;
            verifier(
                doc.get("operateur").is_some() && sans_cles_anciennes(&doc),
                "clés opérateur/domaine/a_j incorrectes",
            )?;
            let configs = doc["cellules"][0]["configs"]
                .as_array()
                .ok_or_else(|| validation("configs absent"))?;
            verifier(
                configs.len() == fixtures.len(),
                "nombre de configurations différent",
            )?;
            for (i, (cfg, c)) in configs.iter().zip(fixtures).enumerate() {
                verifier(
                    cfg["fichier"] == format!("cfg_{:06}.npy", (i + 1) * 100),
                    "ordre des configurations différent",
                )?;
                let v = bloquer_operateur(c, &op, &OptionsOperateur::normale(ctx.fils))?.grossier;
                let stats = pool.install(|| stats_sd(&v))?;
                let w = cfg["sd_somme_w"]
                    .as_f64()
                    .ok_or_else(|| validation("sd_somme_w absent"))?;
                let k = cfg["sd_somme_k"]
                    .as_f64()
                    .ok_or_else(|| validation("sd_somme_k absent"))?;
                verifier(
                    w.to_bits() == stats.somme_w.to_bits()
                        && k.to_bits() == stats.ks.iter().sum::<f64>().to_bits(),
                    "sommes SD différentes au bit",
                )?;
            }
            Ok("3 configurations : sommes SD identiques au bit, clés conformes".into())
        }
        _ => Err(validation("gate de processus inconnue")),
    }
}
fn sans_cles_anciennes(v: &Value) -> bool {
    match v {
        Value::Object(o) => {
            !o.contains_key("domaine")
                && !o.contains_key("a_j")
                && o.values().all(sans_cles_anciennes)
        }
        Value::Array(a) => a.iter().all(sans_cles_anciennes),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn contexte() -> Contexte {
        Contexte {
            fixtures: "fixtures".into(),
            references: "references".into(),
            grandl: None,
            fils: 1,
            seul: vec![],
            mutants: true,
        }
    }
    #[test]
    fn liste_fermee_et_identifiants_emis() {
        let ids: std::collections::BTreeSet<_> = IDS_OBLIGATOIRES.into_iter().collect();
        assert_eq!(ids.len(), 51);
        // Mesures substituées : vérifie l'ordonnanceur de la batterie sans effectuer le run de verdict.
        let lignes = emettre(&contexte(), |_| Ok("mesure substituée".into()));
        assert_eq!(
            lignes.iter().map(|l| l.id.as_str()).collect::<Vec<_>>(),
            IDS_OBLIGATOIRES
        );
        assert_eq!(
            lignes
                .iter()
                .filter(|l| l.verdict == Verdict::Skip && l.id.ends_with("-L12"))
                .count(),
            IDS_L12.len()
        );
        assert!(verifier_certificat(&lignes).is_err());
    }
    #[test]
    fn certificat_refuse_manques_doublons_et_verdicts_non_verts() {
        let mut ctx = contexte();
        ctx.grandl = Some("grandl".into());
        let lignes = emettre(&ctx, |_| Ok("mesure substituée".into()));
        verifier_certificat(&lignes).unwrap();
        assert!(verifier_certificat(&lignes[1..]).is_err());
        let mut double = lignes.clone();
        double.push(lignes[0].clone());
        assert!(verifier_certificat(&double).is_err());
        for verdict in [Verdict::Fail, Verdict::Skip, Verdict::Info] {
            let mut ls = lignes.clone();
            ls[0].verdict = verdict;
            assert!(verifier_certificat(&ls).is_err());
        }
        let mut ls = lignes;
        ls.push(LigneGate::nouvelle(Verdict::Skip, "AUTRE", "absent"));
        assert!(verifier_certificat(&ls).is_err());
    }
    #[test]
    fn filtres_erreurs_et_echantillon() {
        assert_eq!(
            echantillon(&[9, 2, 7, 1, 6, 3, 4, 5, 8, 0], 4),
            [0, 2, 5, 7]
        );
        assert_eq!(echantillon(&[2, 1, 2], 3), [1, 2]);
        let mut ctx = contexte();
        ctx.seul = vec!["RG-QL-NORM".into()];
        ctx.mutants = false;
        let lignes = emettre(&ctx, |_| Err(validation("erreur interne")));
        assert_eq!(lignes.len(), 2);
        assert!(lignes.iter().all(|l| l.verdict == Verdict::Fail));
    }
    #[test]
    fn action_hyper_sur_tous_les_liens_et_groupe_complet() {
        let e = elements_hyper();
        assert_eq!(e.len(), 384);
        assert_eq!(e.iter().filter(|(_, s)| !s.contains(&-1)).count(), 24);
        let c = onde();
        assert_eq!(transformer_hyper(&c, &[0, 1, 2, 3], &[1; 4]).q, c.q);
        let p = [0, 1, 2, 3];
        let s = [-1, 1, 1, 1];
        assert_eq!(
            transformer_hyper(&transformer_hyper(&c, &p, &s), &p, &s).q,
            c.q
        );
    }
}
