//! `kvu rg correlateur` : corrélateur de la densité d'action (plaquettes) et tranches de temps du 0++ (chantier correlateur_action_2026-11-11).
//!
//! Définitions (PREREG du chantier) :
//! - p_{μν}(x) = 1 − ½Tr[U_μ(x) U_ν(x+μ̂) U_μ(x+ν̂)† U_ν(x)†] ;
//! - densité d'action en trèfle (réflexion-symétrique autour de x) : d(x) = Σ_{μ<ν} ¼[p(x) + p(x−μ̂) + p(x−ν̂) + p(x−μ̂−ν̂)]_{μν} ;
//! - tranche 0++ : O_t = Σ_{x : x₀ = t} Σ_{1≤i<j≤3} p_{ij}(x) (plaquettes spatiales), temps = direction 0 ;
//! - corrélateur brut par configuration : m(r) = (1/V) Σ_x d(x) d(x+r), calculé par DFT séparable (tout côté L, pas seulement 2^k),
//!   moyenné sur l'orbite hypercubique (permutations et signes) de chaque vecteur trié a ≥ b ≥ c ≥ d ≥ 0.
//!
//! La soustraction du vide (⟨d⟩²) se fait à l'analyse, sur l'ensemble (jackknife), pas ici.
use super::action::cli::{decouvrir_chaines, ChaineTrouvee};
use super::bloc::{coordonnees, deplacer, site};
use super::domaine::ecrire_json;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul};
use crate::stockage::npy::lire_champ;
use clap::{Arg, ArgMatches, Command};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::Path;

pub const SCHEMA_CORR: &str = "kvu-rg-correlateur/1";
const PLANS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// p_{μν}(x) pour les six plans, `p[plan][site]`.
pub fn plaquettes(u: &ChampHote) -> Vec<Vec<f64>> {
    let l = u.l;
    let v = u.v();
    let lien = |mu: usize, x: [usize; 4]| u.lien(mu * v + site(x, l));
    PLANS
        .iter()
        .map(|&(mu, nu)| {
            (0..v)
                .into_par_iter()
                .map(|s| {
                    let x = coordonnees(s, l);
                    let xm = deplacer(x, mu, 1, l);
                    let xn = deplacer(x, nu, 1, l);
                    let a = qmul(lien(mu, x), lien(nu, xm));
                    let b = qmul(lien(nu, x), lien(mu, xn));
                    // U_μ(x) U_ν(x+μ) · (U_ν(x) U_μ(x+ν))† ; partie réelle = ½Tr
                    let w = qmul(a, qdag(b));
                    1.0 - w[0]
                })
                .collect()
        })
        .collect()
}

/// Densité en trèfle d(x) et plaquettes spatiales sommées pp(x) = Σ_{i<j≥1} p_ij(x).
pub fn densites(u: &ChampHote) -> (Vec<f64>, Vec<f64>) {
    let l = u.l;
    let v = u.v();
    let p = plaquettes(u);
    let d: Vec<f64> = (0..v)
        .into_par_iter()
        .map(|s| {
            let x = coordonnees(s, l);
            let mut tot = 0.0;
            for (k, &(mu, nu)) in PLANS.iter().enumerate() {
                let xm = deplacer(x, mu, -1, l);
                let xn = deplacer(x, nu, -1, l);
                let xmn = deplacer(xm, nu, -1, l);
                tot += 0.25 * (p[k][s] + p[k][site(xm, l)] + p[k][site(xn, l)] + p[k][site(xmn, l)]);
            }
            tot
        })
        .collect();
    let pp: Vec<f64> = (0..v).into_par_iter().map(|s| p[3][s] + p[4][s] + p[5][s]).collect();
    (d, pp)
}

/// Sommes par tranche x₀ = t.
pub fn tranches(f: &[f64], l: usize) -> Vec<f64> {
    let t = l * l * l;
    (0..l).map(|x0| f[x0 * t..(x0 + 1) * t].iter().sum()).collect()
}

// ---------- DFT séparable 4D ----------
fn dft_axe(re: &mut [f64], im: &mut [f64], l: usize, axe: usize, signe: f64) {
    let pas = l.pow((3 - axe) as u32);
    let v = l * l * l * l;
    // O1 (revue c02) : les accès bruts ci-dessous supposent exactement L⁴ valeurs ; les appelants le vérifient, ceci est le dernier rempart.
    assert!(axe < 4 && re.len() == v && im.len() == v, "dft_axe : {} et {} valeurs pour L⁴ = {v}", re.len(), im.len());
    let tw: Vec<(f64, f64)> = (0..l)
        .map(|k| {
            let a = signe * 2.0 * std::f64::consts::PI * k as f64 / l as f64;
            (a.cos(), a.sin())
        })
        .collect();
    let mut debuts = Vec::new();
    for s in 0..v {
        if (s / pas).is_multiple_of(l) {
            debuts.push(s);
        }
    }
    let (rp, ip) = (re.as_mut_ptr() as usize, im.as_mut_ptr() as usize);
    debuts.par_iter().for_each(|&s0| {
        // SAFETY : les lignes (s0 + n·pas) sont disjointes d'un début à l'autre.
        let re = rp as *mut f64;
        let im = ip as *mut f64;
        let mut xr = vec![0.0; l];
        let mut xi = vec![0.0; l];
        for n in 0..l {
            unsafe {
                xr[n] = *re.add(s0 + n * pas);
                xi[n] = *im.add(s0 + n * pas);
            }
        }
        for k in 0..l {
            let (mut sr, mut si) = (0.0, 0.0);
            for n in 0..l {
                let (c, s) = tw[(n * k) % l];
                sr += xr[n] * c - xi[n] * s;
                si += xr[n] * s + xi[n] * c;
            }
            unsafe {
                *re.add(s0 + k * pas) = sr;
                *im.add(s0 + k * pas) = si;
            }
        }
    });
}

fn dft4(re: &mut [f64], im: &mut [f64], l: usize, signe: f64) {
    for axe in 0..4 {
        dft_axe(re, im, l, axe, signe);
    }
}

/// c(r) = (1/V) Σ_x f(x) g(x+r), pour tout r (DFT).
pub fn intercorrelation(f: &[f64], g: &[f64], l: usize) -> Vec<f64> {
    let v = l * l * l * l;
    assert!(f.len() == v && g.len() == v, "intercorrelation : {} et {} valeurs pour L⁴ = {v}", f.len(), g.len());
    let (mut fr, mut fi) = (f.to_vec(), vec![0.0; v]);
    let (mut gr, mut gi) = (g.to_vec(), vec![0.0; v]);
    dft4(&mut fr, &mut fi, l, -1.0);
    dft4(&mut gr, &mut gi, l, -1.0);
    // conj(F)·G
    let mut hr = vec![0.0; v];
    let mut hi = vec![0.0; v];
    for k in 0..v {
        hr[k] = fr[k] * gr[k] + fi[k] * gi[k];
        hi[k] = fr[k] * gi[k] - fi[k] * gr[k];
    }
    dft4(&mut hr, &mut hi, l, 1.0);
    let n = (v * v) as f64;
    hr.iter().map(|x| x / n).collect()
}

/// Vecteurs triés a ≥ b ≥ c ≥ d ≥ 0, 1 ≤ |r|² ≤ rmax², a ≤ L/2.
pub fn orbites(l: usize, rmax: usize) -> Vec<[usize; 4]> {
    let mut out = Vec::new();
    let amax = rmax.min(l / 2);
    // O10 (revue c02) : |r|² ≤ 4·(L/2)² ≤ L² pour toute composante ≤ L/2, donc borner rmax par L ne change aucune orbite
    // et évite le débordement de rmax² (ex. --rmax 4294967296).
    let rmax = rmax.min(l);
    for a in 0..=amax {
        for b in 0..=a {
            for c in 0..=b {
                for d in 0..=c {
                    let n2 = a * a + b * b + c * c + d * d;
                    if n2 >= 1 && n2 <= rmax * rmax {
                        out.push([a, b, c, d]);
                    }
                }
            }
        }
    }
    out
}

fn permutations() -> Vec<[usize; 4]> {
    let mut out = Vec::new();
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let mut s = [a, b, c, d];
                    s.sort();
                    if s == [0, 1, 2, 3] {
                        out.push([a, b, c, d]);
                    }
                }
            }
        }
    }
    out
}

/// Moyenne de c sur l'orbite hypercubique de r (permutations et signes).
pub fn moyenne_orbite(c: &[f64], l: usize, r: [usize; 4]) -> f64 {
    let mut vus: BTreeSet<[usize; 4]> = BTreeSet::new();
    let mut somme = 0.0;
    for p in permutations() {
        let rp = [r[p[0]], r[p[1]], r[p[2]], r[p[3]]];
        for signes in 0..16u32 {
            let mut x = [0usize; 4];
            for m in 0..4 {
                let comp = if (signes >> m) & 1 == 1 { (l - rp[m] % l) % l } else { rp[m] % l };
                x[m] = comp;
            }
            if vus.insert(x) {
                somme += c[site(x, l)];
            }
        }
    }
    somme / vus.len() as f64
}

#[derive(Clone, Debug)]
pub enum Lissage {
    Brut,
    Flot { eps: f64, n: usize },
}

pub fn lire_lissage(t: &str) -> Result<Lissage, String> {
    let p: Vec<&str> = t.split(':').collect();
    match p.as_slice() {
        ["brut"] => Ok(Lissage::Brut),
        ["flot", e, n] => {
            let eps: f64 = e.parse().map_err(|_| format!("lissage « {t} » : eps invalide"))?;
            // O4 (revue c02) : un pas non fini ou ≤ 0 (ex. NaN) propagerait des NaN jusqu'aux mesures
            if !(eps.is_finite() && eps > 0.0) {
                return Err(format!("lissage « {t} » : le pas eps doit être fini et > 0"));
            }
            Ok(Lissage::Flot { eps, n: n.parse().map_err(|_| format!("lissage « {t} » : n invalide"))? })
        }
        _ => Err(format!("lissage « {t} » inconnu (brut | flot:eps:n)")),
    }
}

pub fn appliquer(u: &ChampHote, lis: &Lissage, fils: usize) -> Resultat<ChampHote> {
    match lis {
        Lissage::Brut => Ok(u.clone()),
        Lissage::Flot { eps, n } => {
            let _ = fils;
            // pas de lisser_flot : sa garde « L multiple de 3 » est propre au blocage ; le flot n'exige qu'un L pair
            crate::obs::ref_cpu::flot_n_pas(u, *eps, *n)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutant {
    Aucun,
    /// Corrèle d(config i) avec d(config i + n/2) : le connexe doit s'annuler.
    Melange,
    /// Observable faussée : d(x) remplacée par (1 − Re U_0(x)_0), non invariante de jauge (somme sur les liens).
    Lien,
}

/// Mesure complète d'une configuration (ou d'un couple pour le mutant `Melange`).
pub fn mesurer(u: &ChampHote, autre: Option<&ChampHote>, lis: &Lissage, rmax: usize, mutant: Mutant, fils: usize) -> Resultat<Value> {
    // le champ lissé est un temporaire : libéré dès ses densités calculées (même pic mémoire que la version d'origine)
    let (d, pp) = densite_et_plaquettes(&appliquer(u, lis, fils)?, mutant);
    let g = match autre {
        Some(a) => Some(densite_et_plaquettes(&appliquer(a, lis, fils)?, mutant).0),
        None => None,
    };
    assembler_mesure(u.l, d, g, pp, rmax)
}

/// Observables d'une configuration DÉJÀ lissée `w` (et du partenaire `autre` du mutant `Melange`, lissé de même) : chemin GPU,
/// mêmes fonctions que [`mesurer`] (donc mêmes octets pour un même champ lissé).
pub fn mesurer_lisses(w: &ChampHote, autre: Option<&ChampHote>, rmax: usize, mutant: Mutant) -> Resultat<Value> {
    let (d, pp) = densite_et_plaquettes(w, mutant);
    let g = autre.map(|a| densite_et_plaquettes(a, mutant).0);
    assembler_mesure(w.l, d, g, pp, rmax)
}

/// d(x) (ou l'observable faussée du mutant `Lien`) et pp(x) d'un champ lissé.
fn densite_et_plaquettes(w: &ChampHote, mutant: Mutant) -> (Vec<f64>, Vec<f64>) {
    let (mut d, pp) = densites(w);
    if mutant == Mutant::Lien {
        let v = w.v();
        d = (0..v).map(|s| (0..4).map(|mu| 1.0 - w.lien(mu * v + s)[0]).sum()).collect();
    }
    (d, pp)
}

/// Moyennes, tranches et corrélateur (orbites) à partir de d, g (d si `None`) et pp.
fn assembler_mesure(l: usize, d: Vec<f64>, g: Option<Vec<f64>>, pp: Vec<f64>, rmax: usize) -> Resultat<Value> {
    let g = match g {
        Some(g) => g,
        None => d.clone(),
    };
    // O1 (revue c02) : mêmes longueurs L⁴ pour d, g (partenaire du mutant `Melange`) et pp, avant la DFT
    let v = l.checked_pow(4).ok_or_else(|| Erreur::Config(format!("L = {l} trop grand")))?;
    if d.len() != v || g.len() != v || pp.len() != v {
        return Err(Erreur::Format(format!("champs de tailles {}, {}, {} pour L⁴ = {v} (configurations de côtés différents ?)", d.len(), g.len(), pp.len())));
    }
    let moy_d = d.iter().sum::<f64>() / d.len() as f64;
    let moy_g = g.iter().sum::<f64>() / g.len() as f64;
    let mut corr = serde_json::Map::new();
    let mut c0 = 0.0;
    if rmax > 0 {
        let c = intercorrelation(&d, &g, l);
        c0 = c[0];
        for r in orbites(l, rmax) {
            corr.insert(format!("{},{},{},{}", r[0], r[1], r[2], r[3]), json!(moyenne_orbite(&c, l, r)));
        }
    }
    Ok(json!({
        "moy_d": moy_d, "moy_g": moy_g, "c0": c0,
        "moy_pp": pp.iter().sum::<f64>() / pp.len() as f64,
        "O_t": tranches(&pp, l), "D_t": tranches(&d, l),
        "corr": corr,
    }))
}

/// O5/O16 (revue c02) : provenance (binaire, source, machine, argv, noyaux, modules compilés) et conventions de mesure dans
/// `<sortie>.manifest.json`, écrit APRÈS la sortie par `stockage::provenance::ecrire_manifeste_fichier` ; le fichier `--sortie`
/// lui-même n'est pas modifié (chemin CPU identique à l'octet à e281693).
fn ecrire_provenance(
    sortie: &Path,
    chaines: &[ChaineTrouvee],
    lis: &[(String, Lissage)],
    rmax: usize,
    mutant: Mutant,
    mesure: &str,
    gpu: Option<Value>,
) -> Resultat<std::path::PathBuf> {
    let argv: Vec<String> = std::env::args().collect();
    let l = chaines.first().map(|c| c.l);
    let ajouts = [
        (
            "parametres",
            json!({
                "lissages": lis.iter().map(|x| x.0.clone()).collect::<Vec<_>>(), "rmax": rmax, "mutant": format!("{mutant:?}"),
                "mesure": mesure,
                "chaines": chaines.iter().map(|c| json!({"nom": c.nom, "beta": c.beta, "L": c.l, "n_configs": c.configs.len(),
                    "premiere": c.configs.first().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string()),
                    "derniere": c.configs.last().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string())})).collect::<Vec<_>>(),
            }),
        ),
        ("geometrie", json!({"cotes": l.map(|l| [l, l, l, l]), "dir_temps": DIR_TEMPS_CORR, "torsion": Value::Null, "bords": "periodiques"})),
        (
            "conventions",
            json!({
                "plaquette": "p_munu(x) = 1 - 1/2 Tr[U_mu(x) U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+]",
                "d": "densite en trefle d(x) = sum_{mu<nu} 1/4 [p(x) + p(x-mu) + p(x-nu) + p(x-mu-nu)]_munu",
                "O_t": "somme sur la tranche x[dir_temps] = t (L^3 sites, NON divisee par L^3) des plaquettes spatiales (plans sans dir_temps)",
                "D_t": "somme sur la tranche x[dir_temps] = t de d(x) (NON divisee par L^3)",
                "moy_d|moy_g|moy_pp": "moyennes par site (divisees par V = L^4)",
                "corr": "c(r) = (1/V) sum_x d(x) g(x+r) (g = d sauf mutant melange), NON connexe, moyenne sur l'orbite hypercubique de r",
                "c0": "c(0) ; 0 si rmax = 0",
                "connexe": "soustraction du vide a l'analyse (jackknife), pas ici",
            }),
        ),
        ("mesure_gpu", gpu.unwrap_or(Value::Null)),
    ];
    crate::stockage::provenance::ecrire_manifeste_fichier(sortie, "kvu rg correlateur", &argv, &ajouts)
}

/// O4 (revue c02) : chemin de la première valeur non finie d'une sortie (serde_json écrit NaN et ±∞ en `null`), ou `None`.
pub fn premiere_non_finie(v: &Value) -> Option<String> {
    match v {
        Value::Null => Some(String::new()),
        Value::Number(n) => (!n.as_f64().is_some_and(f64::is_finite)).then(String::new),
        Value::Array(a) => a.iter().enumerate().find_map(|(i, x)| premiere_non_finie(x).map(|p| format!("[{i}]{p}"))),
        Value::Object(o) => o.iter().find_map(|(k, x)| premiere_non_finie(x).map(|p| format!("/{k}{p}"))),
        _ => None,
    }
}

/// O4 (revue c02) : toutes les mesures doivent être finies AVANT l'écriture (parade, entre autres, à une faute GPU silencieuse) :
/// sinon `Erreur::Validation` (rc 8) et aucun JSON n'est écrit.
fn verifier_finitude(chaines: &[Value]) -> Resultat<()> {
    match chaines.iter().enumerate().find_map(|(i, c)| premiere_non_finie(c).map(|p| format!("chaines[{i}]{p}"))) {
        Some(p) => Err(Erreur::Validation(format!("mesure non finie en {p} : aucun JSON écrit"))),
        None => Ok(()),
    }
}

/// O1 (revue c02) : une configuration lue doit avoir le côté L du manifeste de sa chaîne et exactement 16·L⁴ réels.
pub fn verifier_champ(u: &ChampHote, l: usize, chemin: &Path) -> Resultat<()> {
    let attendu = l.checked_pow(4).and_then(|v| v.checked_mul(16));
    if u.l != l || Some(u.q.len()) != attendu {
        return Err(Erreur::Format(format!(
            "{} : côté {} et {} réels, alors que le manifeste de la chaîne annonce L = {l} (16·L⁴ réels) : refus",
            chemin.display(),
            u.l,
            u.q.len()
        )));
    }
    Ok(())
}

/// `lire_champ` puis [`verifier_champ`] contre le L du manifeste.
fn lire_verifie(chemin: &Path, l: usize) -> Resultat<ChampHote> {
    let u = lire_champ(chemin)?;
    verifier_champ(&u, l, chemin)?;
    Ok(u)
}

fn mesurer_chaine(ch: &ChaineTrouvee, lis: &[(String, Lissage)], rmax: usize, mutant: Mutant, fils: usize) -> Resultat<Value> {
    let n = ch.configs.len();
    let cfgs: Vec<Value> = (0..n)
        .into_par_iter()
        .map(|i| -> Resultat<Value> {
            let u = lire_verifie(&ch.configs[i], ch.l)?;
            let autre = if mutant == Mutant::Melange { Some(lire_verifie(&ch.configs[(i + n / 2) % n], ch.l)?) } else { None };
            let mut m = serde_json::Map::new();
            for (nom, lissage) in lis {
                m.insert(nom.clone(), mesurer(&u, autre.as_ref(), lissage, rmax, mutant, fils)?);
            }
            Ok(json!({"cfg": ch.configs[i].file_name().map(|s| s.to_string_lossy().to_string()), "mesures": m}))
        })
        .collect::<Resultat<Vec<_>>>()?;
    Ok(json!({"chaine": ch.nom, "beta": ch.beta, "L": ch.l, "configs": cfgs}))
}

// ---------- Mesure GPU (module mesure-gpu du 11/11) ----------
//
// `--mesure gpu` : le flot de Wilson (seul calcul lourd : 50 pas RK3 × 3 étapes sur 4V liens) passe sur le GPU, en F64, par
// les noyaux validés d'OBS-3 (`obs_flot.cu` : MÊME intégrateur de Lüscher, mêmes coefficients, même exponentielle exacte,
// réunitarisation après chaque étape, compilés sans contraction) ; tout le reste (plaquettes, trèfle, tranches, DFT, orbites)
// est le MÊME code hôte que le chemin CPU (`mesurer_lisses`). Le chemin CPU reste le défaut et n'est pas modifié.
// Pipeline à trois étages : un fil lit/décompresse la configuration suivante, le fil principal (seul à parler au GPU) fait le
// flot, un fil mesure sur l'hôte (pool de `--fils` fils) ; files bornées (2) pour plafonner la mémoire, ordre des
// configurations conservé.

/// Où se fait la mesure de `rg correlateur`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OuMesure {
    /// Défaut : tout sur l'hôte (chemin d'origine, inchangé).
    Cpu,
    /// Flot de Wilson sur le GPU (F64) ; si `--rmax 0` et sans mutant de l'hôte, plaquettes, trèfle et sommes par tranche
    /// aussi sur le GPU (chemin « tout GPU » : le champ n'est pas rapatrié) ; sinon reste sur l'hôte.
    Gpu,
    /// Flot sur le GPU seulement, toujours rapatrié et mesuré sur l'hôte (diagnostic, porte croisée).
    GpuFlot,
}

/// Mutants de la porte GPU contre CPU (jamais utilisés en production).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutantGpu {
    Aucun,
    /// Le flot GPU fait un pas RK3 de moins.
    PasEnMoins,
    /// Chemin tout GPU : plaquette mal orientée dans `k_corr_plaq`.
    Plaquette,
}

/// Direction du temps des tranches de `rg correlateur` (x₀ : indice le plus lent de `ChampHote`).
pub const DIR_TEMPS_CORR: usize = 0;

/// Tampons du chemin tout GPU pour un côté L : plaquettes par plan (6V), d(x) et pp(x) (V chacun), sommes par tranche (2L).
struct TamponsTranches {
    l: usize,
    p: crate::gpu::Tampon<f64>,
    d: crate::gpu::Tampon<f64>,
    pp: crate::gpu::Tampon<f64>,
    sortie: crate::gpu::Tampon<f64>,
}

/// Flot sur le GPU : module des observables compilé une fois, champ résident F64 d'une chaîne réutilisé.
pub struct FlotGpu {
    /// Clone de l'appareil (Arc) : la mesure en ligne de `su2 gen` (B08) garde son `FlotGpu` d'un pas à l'autre.
    app: crate::gpu::Appareil,
    module: crate::obs::gpu::ModuleObs,
    f_plaq: crate::gpu::Fonction,
    f_dens: crate::gpu::Fonction,
    f_tranches: crate::gpu::Fonction,
    champ: Option<crate::su2::ChampGpu>,
    tampons: Option<TamponsTranches>,
    /// c08b (O7) : noyau de copie et espaces de travail persistants de la mesure en ligne (L, espace).
    f_copie: crate::gpu::Fonction,
    espace: Option<(usize, crate::obs::gpu::EspaceFlot)>,
    copie: Option<(usize, crate::gpu::Tampon<f64>)>,
    mutant: MutantGpu,
    /// Temps passé sur le GPU (téléversement, pas, noyaux de mesure, rapatriement), cumulé.
    pub duree: std::time::Duration,
    /// Nombre de flots faits.
    pub n_flots: usize,
}

impl FlotGpu {
    pub fn nouveau(app: &crate::gpu::Appareil, mutant: MutantGpu) -> Resultat<FlotGpu> {
        let module = crate::obs::gpu::ModuleObs::nouveau(app)?;
        Ok(FlotGpu {
            app: app.clone(),
            f_plaq: module.fonction("k_corr_plaq")?,
            f_dens: module.fonction("k_corr_dens")?,
            f_tranches: module.fonction("k_corr_tranches")?,
            f_copie: module.fonction("k_obs_copie")?,
            module,
            champ: None,
            tampons: None,
            espace: None,
            copie: None,
            mutant,
            duree: Default::default(),
            n_flots: 0,
        })
    }

    /// Champ résident F64 (une chaîne) de côté `l`, alloué au premier usage ou au changement de L, rempli avec `u`.
    fn monter(&mut self, u: &ChampHote) -> Resultat<()> {
        if self.champ.as_ref().is_none_or(|c| c.l != u.l) {
            self.champ = None; // libère l'ancien champ avant d'allouer le nouveau
            self.champ = Some(crate::su2::ChampGpu::allouer(&self.app, u.l, 1, crate::commun::Precision::F64)?);
        }
        self.champ.as_mut().expect("champ GPU alloué").televerser(&self.app, 0, u)
    }

    fn n_pas(&self, n: usize) -> usize {
        match self.mutant {
            MutantGpu::PasEnMoins => n.saturating_sub(1),
            _ => n,
        }
    }

    /// Plaquettes, d(x), pp(x) et sommes par tranche (direction `dir_temps`) du champ `v` (argument noyau, côté `l`) :
    /// rend (D_t, O_t) ; synchronise.
    fn sommes_tranches(&mut self, v: crate::gpu::Arg, l: usize, dir_temps: usize) -> Resultat<(Vec<f64>, Vec<f64>)> {
        use crate::gpu::{Arg, Lancement};
        let vol = l.pow(4);
        if self.tampons.as_ref().is_none_or(|t| t.l != l) {
            self.tampons = None;
            self.tampons = Some(TamponsTranches {
                l,
                p: self.app.alloc::<f64>(6 * vol)?,
                d: self.app.alloc::<f64>(vol)?,
                pp: self.app.alloc::<f64>(vol)?,
                sortie: self.app.alloc::<f64>(2 * l)?,
            });
        }
        let t = self.tampons.as_ref().expect("tampons alloués");
        let lin = Lancement::lineaire(vol, 256);
        let mutant = i32::from(self.mutant == MutantGpu::Plaquette);
        self.app.lancer(&self.f_plaq, &lin, &[v, Arg::I32(l as i32), Arg::I32(mutant), t.p.arg()])?;
        self.app.lancer(&self.f_dens, &lin, &[t.p.arg(), Arg::I32(l as i32), Arg::I32(dir_temps as i32), t.d.arg(), t.pp.arg()])?;
        let par_tranche = Lancement { grille: [l as u32, 1, 1], bloc: [256, 1, 1], shmem: 0 };
        self.app.lancer(&self.f_tranches, &par_tranche, &[t.d.arg(), Arg::I32(l as i32), Arg::I32(dir_temps as i32), t.sortie.arg()])?;
        self.app.lancer(&self.f_tranches, &par_tranche, &[t.pp.arg(), Arg::I32(l as i32), Arg::I32(dir_temps as i32), t.sortie.arg_decale(l)])?;
        self.app.synchroniser()?;
        let mut out = vec![0.0; 2 * l];
        self.app.lire(&t.sortie, 0, &mut out)?;
        let pp_t = out.split_off(l);
        Ok((out, pp_t))
    }

    /// Chemin tout GPU d'une configuration : un téléversement, puis pour chaque lissage (D_t, O_t) calculés sur le GPU
    /// (brut : sur le champ résident ; flot : sur la copie de travail du flot, jamais rapatriée).
    pub fn tranches(&mut self, u: &ChampHote, lis: &[(String, Lissage)], dir_temps: usize) -> Resultat<Vec<(Vec<f64>, Vec<f64>)>> {
        if dir_temps > 3 {
            return Err(Erreur::Config(format!("dir_temps = {dir_temps} hors de 0..3")));
        }
        let t0 = std::time::Instant::now();
        self.monter(u)?;
        let mut out = Vec::with_capacity(lis.len());
        for (_, l) in lis {
            let r = match l {
                Lissage::Brut => {
                    let v = match &self.champ.as_ref().expect("champ GPU monté").tampon {
                        crate::su2::TamponChamp::F64(t) => t.arg(),
                        crate::su2::TamponChamp::F32(_) => return Err(Erreur::Config("champ GPU F32 inattendu".into())),
                    };
                    self.sommes_tranches(v, u.l, dir_temps)?
                }
                Lissage::Flot { eps, n } => {
                    let n_pas = self.n_pas(*n);
                    let e = self.module.flot_prepare(self.champ.as_ref().expect("champ GPU monté"), 0)?;
                    for _ in 0..n_pas {
                        self.module.flot_pas(&e, *eps)?;
                    }
                    self.n_flots += 1;
                    let r = self.sommes_tranches(e.champ_arg(), u.l, dir_temps)?;
                    drop(e);
                    r
                }
            };
            out.push(r);
        }
        self.duree += t0.elapsed();
        Ok(out)
    }

    /// Mesure EN LIGNE de `su2 gen --mesures corr:…` (B08) : comme [`FlotGpu::tranches`], mais sur la chaîne `c` d'un champ
    /// RÉSIDENT du générateur, sans passer par le disque. Champ F64 : pour chaque lissage, la chaîne est COPIÉE (noyau
    /// `k_obs_copie` : le champ résident n'est que LU) dans un espace de travail PERSISTANT (c08b, O7 : alloué une fois par L ;
    /// copie F64 seule, 16·V réels, si aucun lissage n'est un flot ; sinon l'espace du flot, 52·V réels, dont la copie est la
    /// première partie), flottée s'il le faut, puis [`FlotGpu::sommes_tranches`] ; mêmes octets d'entrée que le dump
    /// `cfg_*.npy`, mêmes noyaux dans le même ordre, donc mêmes sorties que `--mesure gpu`. Les forces de l'espace du flot sont
    /// réécrites entières avant chaque usage (`k_obs_z`) : leur contenu antérieur n'intervient pas. Champ F32 : rapatrié et
    /// renormalisé en F64 exactement comme le dump (`npy::renormaliser`), puis [`FlotGpu::tranches`].
    pub fn tranches_resident(
        &mut self,
        champ: &crate::su2::ChampGpu,
        c: usize,
        lis: &[(String, Lissage)],
        dir_temps: usize,
    ) -> Resultat<Vec<(Vec<f64>, Vec<f64>)>> {
        use crate::gpu::{Arg, Lancement};
        if dir_temps > 3 {
            return Err(Erreur::Config(format!("dir_temps = {dir_temps} hors de 0..3")));
        }
        // c08c (O7) : champ F32 rapatrié, renormalisé comme le dump, puis MONTÉ dans le champ F64 persistant du FlotGpu ; la
        // suite est le chemin F64 (même espace de travail persistant), avec ce champ monté pour source.
        let f32 = matches!(champ.tampon, crate::su2::TamponChamp::F32(_));
        if f32 {
            let mut u = champ.telecharger(&self.app, c)?;
            crate::stockage::npy::renormaliser(&mut u)?;
            self.monter(&u)?;
        } else if c >= champ.nchaines {
            return Err(Erreur::Config(format!("chaîne {c} hors du lot ({} chaînes)", champ.nchaines)));
        }
        let (src, idx) = if f32 {
            (self.champ.as_ref().ok_or_else(|| Erreur::Config("champ F64 monté absent".into()))?, 0)
        } else {
            (champ, c)
        };
        let base = match &src.tampon {
            crate::su2::TamponChamp::F64(t) => t.arg_decale(idx * 16 * src.v()),
            crate::su2::TamponChamp::F32(_) => return Err(Erreur::Config("champ F64 attendu".into())),
        };
        let t0 = std::time::Instant::now();
        let (l, v) = (src.l, src.v());
        let avec_flot = lis.iter().any(|(_, x)| matches!(x, Lissage::Flot { .. }));
        if avec_flot && self.espace.as_ref().is_none_or(|(le, _)| *le != l) {
            let e = self.module.flot_prepare(src, idx)?;
            self.espace = None; // libère l'ancien espace avant de garder le nouveau
            self.espace = Some((l, e));
        }
        if !avec_flot && self.copie.as_ref().is_none_or(|(le, _)| *le != l) {
            self.copie = None;
            self.copie = Some((l, self.app.alloc::<f64>(16 * v)?));
        }
        let cible = match (&self.espace, &self.copie) {
            (Some((_, e)), _) if avec_flot => e.champ_arg(),
            (_, Some((_, t))) if !avec_flot => t.arg(),
            _ => return Err(Erreur::Config("espace de travail de la mesure en ligne absent".into())),
        };
        let mut out = Vec::with_capacity(lis.len());
        for (_, x) in lis {
            self.app.lancer(&self.f_copie, &Lancement::lineaire(4 * v, 256), &[base, cible, Arg::I64((4 * v) as i64)])?;
            if let Lissage::Flot { eps, n } = x {
                let n_pas = self.n_pas(*n);
                let (_, e) = self.espace.as_ref().ok_or_else(|| Erreur::Config("espace du flot absent".into()))?;
                for _ in 0..n_pas {
                    self.module.flot_pas(e, *eps)?;
                }
                self.n_flots += 1;
            }
            // sommes_tranches synchronise avant de rendre : la copie suivante ne commence qu'après
            out.push(self.sommes_tranches(cible, l, dir_temps)?);
        }
        self.duree += t0.elapsed();
        Ok(out)
    }

    /// sha256 du code objet du module (manifeste de la sortie).
    pub fn sha256_code(&self) -> String {
        self.module.provenance().sha256_code.clone()
    }

    /// Provenance complète du module compilé (source assemblée, options, code objet).
    pub fn provenance(&self) -> crate::gpu::ProvenanceCompil {
        self.module.provenance()
    }

    /// Équivalent GPU de [`appliquer`] : `Brut` = copie ; `Flot` = n pas RK3 de taille eps sur le GPU (F64), champ rapatrié.
    pub fn appliquer(&mut self, u: &ChampHote, lis: &Lissage) -> Resultat<ChampHote> {
        match lis {
            Lissage::Brut => Ok(u.clone()),
            Lissage::Flot { eps, n } => {
                let t0 = std::time::Instant::now();
                let n_pas = self.n_pas(*n);
                self.monter(u)?;
                let w = self.module.flot_champ_final(self.champ.as_ref().expect("champ GPU monté"), 0, *eps, n_pas)?;
                self.duree += t0.elapsed();
                self.n_flots += 1;
                Ok(w)
            }
        }
    }
}

type Lu = (usize, ChampHote, Option<ChampHote>);

/// Configurations lues (décompressées et décodées) en parallèle par l'étage de lecture du chemin GPU.
const LECTEURS_GPU: usize = 3;

/// Ce que l'étage GPU passe à l'étage hôte pour une configuration.
enum Etape {
    /// Champs lissés rapatriés (et partenaires du mutant `Melange`) : mesure complète sur l'hôte.
    Champs(Vec<(ChampHote, Option<ChampHote>)>),
    /// Chemin tout GPU : (D_t, O_t) par lissage.
    Tranches(Vec<(Vec<f64>, Vec<f64>)>),
}
type Lisse = (usize, Etape);

/// Mesure à `--rmax 0` sans mutant de l'hôte, à partir des seules sommes par tranche (chemin tout GPU) : mêmes clés que
/// [`assembler_mesure`] ; moy_d = Σ_t D_t / V (= Σ_x d(x) / V à l'ordre de sommation près), moy_g = moy_d (g = d sans
/// mutant), c0 = 0 et corr vide (rmax = 0), moy_pp = Σ_t O_t / V.
pub fn assembler_tranches(l: usize, d_t: Vec<f64>, pp_t: Vec<f64>) -> Value {
    let v = l.pow(4) as f64;
    let moy_d = d_t.iter().sum::<f64>() / v;
    json!({
        "moy_d": moy_d, "moy_g": moy_d, "c0": 0.0,
        "moy_pp": pp_t.iter().sum::<f64>() / v,
        "O_t": pp_t, "D_t": d_t,
        "corr": serde_json::Map::new(),
    })
}

/// Chemin GPU d'une chaîne : mêmes sorties que [`mesurer_chaine`] (même ordre, mêmes clés), flot sur le GPU.
fn mesurer_chaine_gpu(
    ch: &ChaineTrouvee,
    lis: &[(String, Lissage)],
    rmax: usize,
    mutant: Mutant,
    pool: &rayon::ThreadPool,
    flot: &mut FlotGpu,
    tout_gpu: bool,
) -> Resultat<Value> {
    use std::sync::mpsc::sync_channel;
    let n = ch.configs.len();
    let l_chaine = ch.l;
    let t0 = std::time::Instant::now();
    let duree_gpu0 = flot.duree;
    let cfgs = std::thread::scope(|sc| -> Resultat<(Vec<Value>, std::time::Duration, std::time::Duration)> {
        let (tx_lu, rx_lu) = sync_channel::<Resultat<Lu>>(LECTEURS_GPU);
        let (tx_lisse, rx_lisse) = sync_channel::<Lisse>(2);
        let configs = &ch.configs;
        // Lecture par lots de LECTEURS_GPU configurations décompressées en parallèle (la lecture zstd + décodage limitait le
        // chemin tout GPU à L = 32 : 2 à 3 s par configuration sous charge, contre 0,75 s de GPU) ; envoi dans l'ordre.
        let lecteur = sc.spawn(move || -> std::time::Duration {
            let lire_un = |i: usize| -> Resultat<Lu> {
                let u = lire_verifie(&configs[i], l_chaine)?;
                let autre = if mutant == Mutant::Melange { Some(lire_verifie(&configs[(i + n / 2) % n], l_chaine)?) } else { None };
                Ok((i, u, autre))
            };
            let mut duree = std::time::Duration::ZERO;
            let mut debut = 0;
            'lots: while debut < n {
                let fin = (debut + LECTEURS_GPU).min(n);
                let t = std::time::Instant::now();
                let lot: Vec<Resultat<Lu>> = std::thread::scope(|s2| {
                    let fils: Vec<_> = (debut..fin).map(|i| s2.spawn(move || lire_un(i))).collect();
                    fils.into_iter().map(|f| f.join().unwrap_or_else(|_| Err(Erreur::Config("fil de lecture : panique".into())))).collect()
                });
                duree += t.elapsed();
                for r in lot {
                    let echec = r.is_err();
                    if tx_lu.send(r).is_err() || echec {
                        break 'lots;
                    }
                }
                debut = fin;
            }
            duree
        });
        let mesureur = sc.spawn(move || -> Resultat<(Vec<Value>, std::time::Duration)> {
            let mut out = Vec::with_capacity(n);
            let mut duree = std::time::Duration::ZERO;
            for (i, etape) in rx_lisse {
                let t = std::time::Instant::now();
                let mut m = serde_json::Map::new();
                match etape {
                    Etape::Champs(lisses) => {
                        for ((nom, _), (w, wa)) in lis.iter().zip(&lisses) {
                            m.insert(nom.clone(), pool.install(|| mesurer_lisses(w, wa.as_ref(), rmax, mutant))?);
                        }
                    }
                    Etape::Tranches(sommes) => {
                        for ((nom, _), (d_t, pp_t)) in lis.iter().zip(sommes) {
                            m.insert(nom.clone(), assembler_tranches(l_chaine, d_t, pp_t));
                        }
                    }
                }
                out.push(json!({"cfg": configs[i].file_name().map(|s| s.to_string_lossy().to_string()), "mesures": m}));
                duree += t.elapsed();
            }
            Ok((out, duree))
        });
        let mut erreur = None;
        for r in rx_lu {
            let etape = r.and_then(|(i, u, autre)| -> Resultat<Lisse> {
                if tout_gpu {
                    if u.l != l_chaine {
                        return Err(Erreur::Config(format!("{} : configuration de côté {} dans une chaîne L = {l_chaine}", ch.nom, u.l)));
                    }
                    return Ok((i, Etape::Tranches(flot.tranches(&u, lis, DIR_TEMPS_CORR)?)));
                }
                let mut v = Vec::with_capacity(lis.len());
                for (_, l) in lis {
                    let w = flot.appliquer(&u, l)?;
                    let wa = match &autre {
                        Some(a) => Some(flot.appliquer(a, l)?),
                        None => None,
                    };
                    v.push((w, wa));
                }
                Ok((i, Etape::Champs(v)))
            });
            match etape {
                Ok(x) => {
                    if tx_lisse.send(x).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    erreur = Some(e);
                    break;
                }
            }
        }
        drop(tx_lisse);
        let mesure = mesureur.join().map_err(|_| Erreur::Config("fil de mesure hôte : panique".into()))?;
        let d_lecture = lecteur.join().map_err(|_| Erreur::Config("fil de lecture : panique".into()))?;
        // une erreur de l'étage GPU (ou de lecture) prime ; sinon celle de la mesure hôte
        if let Some(e) = erreur {
            return Err(e);
        }
        let (out, d_mesure) = mesure?;
        if out.len() != n {
            return Err(Erreur::Config(format!("{} : {} configurations mesurées sur {n}", ch.nom, out.len())));
        }
        Ok((out, d_lecture, d_mesure))
    })?;
    let (cfgs, d_lecture, d_mesure) = cfgs;
    let total = t0.elapsed().as_secs_f64();
    let gpu = (flot.duree - duree_gpu0).as_secs_f64();
    let par = |x: f64| x / n.max(1) as f64;
    eprintln!(
        "kvu rg correlateur [gpu{}] : {} : {n} configuration(s) L={} en {total:.2} s ({:.3} s/config) ; par config : lecture {:.3} s, GPU {:.3} s, mesure hôte {:.3} s",
        if tout_gpu { " tout" } else { "" },
        ch.nom,
        ch.l,
        par(total),
        par(d_lecture.as_secs_f64()),
        par(gpu),
        par(d_mesure.as_secs_f64())
    );
    Ok(json!({"chaine": ch.nom, "beta": ch.beta, "L": ch.l, "configs": cfgs}))
}

// ---------- Juge des portes GPU contre CPU ----------

/// Écart relatif maximal entre deux arbres JSON de MÊME structure (mêmes clés, mêmes longueurs, mêmes chaînes, mêmes booléens) :
/// max sur les nombres de |a − b| / max(|a|, |b|) (0 si a = b, y compris 0 = 0). Toute différence de structure est une erreur.
pub fn ecart_relatif_max(a: &Value, b: &Value, chemin: &str) -> Result<(f64, String), String> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap_or(f64::NAN), y.as_f64().unwrap_or(f64::NAN));
            let e = if x == y {
                0.0
            } else if !(x.is_finite() && y.is_finite()) {
                f64::INFINITY
            } else {
                (x - y).abs() / x.abs().max(y.abs())
            };
            Ok((e, chemin.to_string()))
        }
        (Value::Object(x), Value::Object(y)) => {
            if x.len() != y.len() || x.keys().any(|k| !y.contains_key(k)) {
                return Err(format!("{chemin} : clés différentes"));
            }
            let mut pire = (0.0, chemin.to_string());
            for (k, va) in x {
                let r = ecart_relatif_max(va, &y[k], &format!("{chemin}/{k}"))?;
                if r.0 > pire.0 || r.0.is_nan() {
                    pire = r;
                }
            }
            Ok(pire)
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Err(format!("{chemin} : longueurs {} et {}", x.len(), y.len()));
            }
            let mut pire = (0.0, chemin.to_string());
            for (i, (va, vb)) in x.iter().zip(y).enumerate() {
                let r = ecart_relatif_max(va, vb, &format!("{chemin}[{i}]"))?;
                if r.0 > pire.0 || r.0.is_nan() {
                    pire = r;
                }
            }
            Ok(pire)
        }
        (x, y) if x == y => Ok((0.0, chemin.to_string())),
        _ => Err(format!("{chemin} : valeurs non numériques différentes ({a} contre {b})")),
    }
}

/// Nombre de valeurs numériques d'un arbre JSON.
pub fn nombre_de_valeurs(a: &Value) -> usize {
    match a {
        Value::Number(_) => 1,
        Value::Object(x) => x.values().map(nombre_de_valeurs).sum(),
        Value::Array(x) => x.iter().map(nombre_de_valeurs).sum(),
        _ => 0,
    }
}

/// Clés d'une mesure (une configuration, un lissage) dans la sortie de `rg correlateur`.
const CLES_MESURE: [&str; 7] = ["D_t", "O_t", "c0", "corr", "moy_d", "moy_g", "moy_pp"];

/// O2 (revue c07) : validation d'une sortie de `rg correlateur` avant comparaison : schéma, lissages non vides, au moins une
/// chaîne et une configuration, L ≥ 2, chaque configuration a exactement les lissages déclarés, chaque mesure exactement les
/// clés attendues, O_t et D_t de longueur L, nombres finis partout (aucun `null`). Rend le nombre de valeurs numériques.
pub fn valider_sortie(doc: &Value) -> Result<usize, String> {
    if doc["schema"] != json!(SCHEMA_CORR) {
        return Err(format!("schéma {} au lieu de {SCHEMA_CORR}", doc["schema"]));
    }
    let lissages: Vec<&str> = doc["lissages"].as_array().map(|v| v.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    if lissages.is_empty() || doc["lissages"].as_array().is_some_and(|v| v.len() != lissages.len()) {
        return Err("lissages absents ou invalides".into());
    }
    if !doc["rmax"].is_u64() || !doc["mutant"].is_string() {
        return Err("rmax ou mutant absent".into());
    }
    let rmax = doc["rmax"].as_u64().unwrap_or(0) as usize;
    // c07-O2 (seconde relecture) : un nombre fini exigé (pas une chaîne « NaN », pas un booléen, pas null)
    let fini = |v: &Value| v.as_f64().is_some_and(f64::is_finite) && v.is_number();
    let chaines = doc["chaines"].as_array().filter(|v| !v.is_empty()).ok_or("aucune chaîne")?;
    for (i, ch) in chaines.iter().enumerate() {
        let l = ch["L"].as_u64().filter(|&l| l >= 2).ok_or(format!("chaines[{i}] : L absent ou < 2"))? as usize;
        if !ch["beta"].as_f64().is_some_and(f64::is_finite) || !ch["chaine"].is_string() {
            return Err(format!("chaines[{i}] : beta ou nom absent"));
        }
        let cfgs = ch["configs"].as_array().filter(|v| !v.is_empty()).ok_or(format!("chaines[{i}] : aucune configuration"))?;
        for (k, c) in cfgs.iter().enumerate() {
            let ou = format!("chaines[{i}]/configs[{k}]");
            if !c["cfg"].is_string() {
                return Err(format!("{ou} : nom de configuration absent"));
            }
            let m = c["mesures"].as_object().ok_or(format!("{ou} : mesures absentes"))?;
            if m.len() != lissages.len() || lissages.iter().any(|x| !m.contains_key(*x)) {
                return Err(format!("{ou} : lissages {:?} au lieu de {lissages:?}", m.keys().collect::<Vec<_>>()));
            }
            for (nom, me) in m {
                let o = me.as_object().ok_or(format!("{ou}/{nom} : pas un objet"))?;
                if o.len() != CLES_MESURE.len() || CLES_MESURE.iter().any(|x| !o.contains_key(*x)) {
                    return Err(format!("{ou}/{nom} : clés {:?} au lieu de {CLES_MESURE:?}", o.keys().collect::<Vec<_>>()));
                }
                for t in ["O_t", "D_t"] {
                    let a = o[t].as_array().filter(|a| a.len() == l).ok_or(format!("{ou}/{nom}/{t} : pas un tableau de longueur L = {l}"))?;
                    if let Some(j) = a.iter().position(|x| !fini(x)) {
                        return Err(format!("{ou}/{nom}/{t}[{j}] : {} n'est pas un nombre fini", a[j]));
                    }
                }
                for x in ["moy_d", "moy_g", "moy_pp", "c0"] {
                    if !fini(&o[x]) {
                        return Err(format!("{ou}/{nom}/{x} : {} n'est pas un nombre fini", o[x]));
                    }
                }
                // clés de corr : exactement les orbites de (L, rmax) (aucune si rmax = 0), valeurs finies
                let corr = o["corr"].as_object().ok_or(format!("{ou}/{nom}/corr : pas un objet"))?;
                let attendues: Vec<String> = if rmax > 0 {
                    orbites(l, rmax).iter().map(|r| format!("{},{},{},{}", r[0], r[1], r[2], r[3])).collect()
                } else {
                    Vec::new()
                };
                if corr.len() != attendues.len() || attendues.iter().any(|k| !corr.contains_key(k)) {
                    return Err(format!("{ou}/{nom}/corr : {} clés au lieu des {} orbites de L = {l}, rmax = {rmax}", corr.len(), attendues.len()));
                }
                if let Some((k, v)) = corr.iter().find(|(_, v)| !fini(v)) {
                    return Err(format!("{ou}/{nom}/corr/{k} : {v} n'est pas un nombre fini"));
                }
            }
        }
    }
    if let Some(p) = premiere_non_finie(&doc["chaines"]) {
        return Err(format!("valeur non finie ou nulle en chaines{p}"));
    }
    Ok(nombre_de_valeurs(&doc["chaines"]))
}

/// Juge de la porte CORR-GPU-CPU : deux sorties VALIDES (voir [`valider_sortie`]), mêmes schéma, mutant, rmax et lissages, même
/// structure ; rend (écart relatif max, chemin, nombre de valeurs). Tolérance finie et ≥ 0 exigée.
pub fn juger(a: &Value, b: &Value, tol: f64) -> Result<(f64, String, usize), String> {
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(format!("tolérance {tol} : finie et ≥ 0 exigée"));
    }
    let na = valider_sortie(a).map_err(|e| format!("A invalide : {e}"))?;
    let nb = valider_sortie(b).map_err(|e| format!("B invalide : {e}"))?;
    for cle in ["schema", "mutant", "rmax", "lissages"] {
        if a[cle] != b[cle] {
            return Err(format!("champ « {cle} » différent ({} contre {})", a[cle], b[cle]));
        }
    }
    if na != nb {
        return Err(format!("{na} valeurs contre {nb}"));
    }
    let (e, ou) = ecart_relatif_max(&a["chaines"], &b["chaines"], "chaines").map_err(|s| format!("structure différente : {s}"))?;
    Ok((e, ou, na))
}

fn comparer_cmd(m: &ArgMatches) -> Resultat<i32> {
    let lire = |n: &str| -> Resultat<Value> {
        let p = m.get_one::<String>(n).ok_or_else(|| Erreur::Config(format!("--{n} requis")))?;
        let t = std::fs::read_to_string(p).map_err(|e| Erreur::io(format!("lecture de {p}"), e))?;
        serde_json::from_str(&t).map_err(|e| Erreur::Format(format!("{p} : {e}")))
    };
    let (a, b) = (lire("a")?, lire("b")?);
    let tol: f64 = m.get_one::<String>("tol").unwrap().parse().map_err(|_| Erreur::Config("--tol invalide".into()))?;
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(Erreur::Config(format!("--tol {tol} : finie et ≥ 0 exigée")));
    }
    match juger(&a, &b, tol) {
        Ok((e, ou, n)) => {
            let ok = e <= tol;
            println!(
                "{} CORR-GPU-CPU : {n} valeurs, écart relatif max {e:.3e} (seuil {tol:e}) en {ou}",
                if ok { "PASS" } else { "FAIL" }
            );
            Ok(if ok { crate::commun::code::OK } else { crate::commun::code::VALIDATION })
        }
        Err(s) => {
            println!("FAIL CORR-GPU-CPU : {s}");
            Ok(crate::commun::code::VALIDATION)
        }
    }
}

pub fn sous_commandes() -> Vec<Command> {
    vec![
        Command::new("correlateur")
            .about("Corrélateur de la densité d'action (orbites hypercubiques) et tranches du 0++ par configuration (données brutes JSON)")
            .arg(Arg::new("ens").long("ens").required(true).num_args(1).help("dossier d'une chaîne ou parent de chaînes (un seul L)"))
            .arg(Arg::new("lissage").long("lissage").num_args(1).default_value("brut").help("liste (virgules) : brut | flot:eps:n"))
            .arg(Arg::new("rmax").long("rmax").num_args(1).default_value("8").help("0 = pas de corrélateur spatial (tranches seulement)"))
            .arg(Arg::new("max-configs").long("max-configs").num_args(1).help("n'utilise que les N premières configurations de chaque chaîne"))
            .arg(Arg::new("mutant").long("mutant").num_args(1).default_value("aucun").help("aucun | melange | lien (gate : le mutant doit échouer)"))
            .arg(Arg::new("sortie").long("sortie").required(true).num_args(1).help("fichier JSON (refus d'écraser)"))
            .arg(Arg::new("fils").long("fils").num_args(1))
            .arg(Arg::new("mesure").long("mesure").num_args(1).default_value("cpu").help(
                "cpu (défaut, inchangé) | gpu : flot de Wilson sur le GPU de la machine (F64, noyaux d'OBS-3, L pair) ; avec --rmax 0 et \
                 sans --mutant, plaquettes, trèfle et tranches aussi sur le GPU, sinon mesure sur l'hôte | gpu-flot : flot seul sur le GPU ; \
                 prend le verrou GPU de la politique kvu de la machine (fichier `verrou` du profil) ; un verrou propre au site \
                 (ex. flock /work/gpu.lock sur la machine louée) reste à prendre par l'appelant",
            ))
            .arg(Arg::new("mutant-gpu").long("mutant-gpu").num_args(1).default_value("aucun").hide(true).help("aucun | pas-en-moins | plaquette (porte CORR-GPU-CPU : doit être ROUGE)")),
        Command::new("correlateur-comparer")
            .about("Porte CORR-GPU-CPU : compare deux sorties de `rg correlateur` (même structure, écart relatif max ≤ --tol) ; rc 0 ssi PASS")
            .arg(Arg::new("a").long("a").required(true).num_args(1))
            .arg(Arg::new("b").long("b").required(true).num_args(1))
            .arg(Arg::new("tol").long("tol").num_args(1).default_value("1e-10")),
    ]
}

pub fn executer_comparer(m: &ArgMatches) -> Resultat<i32> {
    comparer_cmd(m)
}

pub fn executer(m: &ArgMatches) -> Resultat<i32> {
    let get = |n: &str| m.get_one::<String>(n).cloned();
    let ens = get("ens").ok_or_else(|| Erreur::Config("--ens requis".into()))?;
    let rmax: usize = get("rmax").unwrap().parse().map_err(|_| Erreur::Config("--rmax invalide".into()))?;
    let fils: usize = match get("fils") {
        Some(t) => t.parse().map_err(|_| Erreur::Config("--fils invalide".into()))?,
        None => std::env::var("RAYON_NUM_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(4),
    };
    // O10 (revue c02) : 0 fil voudrait dire « tous les cœurs » pour rayon ; refus, comme les autres commandes
    if fils == 0 {
        return Err(Erreur::Config("--fils doit être ≥ 1".into()));
    }
    let mutant = match get("mutant").unwrap().as_str() {
        "aucun" => Mutant::Aucun,
        "melange" => Mutant::Melange,
        "lien" => Mutant::Lien,
        x => return Err(Erreur::Config(format!("--mutant « {x} » inconnu"))),
    };
    let mut lis = Vec::new();
    for t in get("lissage").unwrap().split(',') {
        lis.push((t.to_string(), lire_lissage(t).map_err(Erreur::Config)?));
    }
    let sortie = std::path::PathBuf::from(get("sortie").unwrap());
    if sortie.exists() {
        return Err(Erreur::Config(format!("{} existe déjà (refus d'écraser)", sortie.display())));
    }
    let mut chaines = decouvrir_chaines(Path::new(&ens), None)?;
    // O1 (revue c02) : un seul L par appel (aide de --ens) ; chaque configuration est ensuite vérifiée contre ce L
    if let Some(c0) = chaines.first() {
        if let Some(c) = chaines.iter().find(|c| c.l != c0.l) {
            return Err(Erreur::Format(format!("ensemble hétérogène : {} (L = {}) et {} (L = {}) ; un seul L par appel", c0.nom, c0.l, c.nom, c.l)));
        }
    }
    if let Some(t) = get("max-configs") {
        let k: usize = t.parse().map_err(|_| Erreur::Config("--max-configs invalide".into()))?;
        for ch in chaines.iter_mut() {
            ch.configs.truncate(k);
        }
    }
    // O12 (revue c02) : avec une seule configuration, le décalage n/2 = 0 corrélerait la configuration avec elle-même
    if mutant == Mutant::Melange {
        if let Some(ch) = chaines.iter().find(|c| c.configs.len() < 2) {
            return Err(Erreur::Config(format!("--mutant melange exige au moins 2 configurations par chaîne ({} en a {})", ch.nom, ch.configs.len())));
        }
    }
    let ou = match get("mesure").unwrap().as_str() {
        "cpu" => OuMesure::Cpu,
        "gpu" => OuMesure::Gpu,
        "gpu-flot" => OuMesure::GpuFlot,
        x => return Err(Erreur::Config(format!("--mesure « {x} » inconnue (cpu | gpu | gpu-flot)"))),
    };
    let mutant_gpu = match get("mutant-gpu").unwrap().as_str() {
        "aucun" => MutantGpu::Aucun,
        "pas-en-moins" => MutantGpu::PasEnMoins,
        "plaquette" => MutantGpu::Plaquette,
        x => return Err(Erreur::Config(format!("--mutant-gpu « {x} » inconnu"))),
    };
    if ou == OuMesure::Cpu && mutant_gpu != MutantGpu::Aucun {
        return Err(Erreur::Config("--mutant-gpu exige --mesure gpu".into()));
    }
    // chemin tout GPU : rien d'autre à mesurer que les tranches et les moyennes (pas de corrélateur spatial, pas de mutant de l'hôte)
    let tout_gpu = ou == OuMesure::Gpu && rmax == 0 && mutant == Mutant::Aucun;
    if mutant_gpu == MutantGpu::Plaquette && !tout_gpu {
        return Err(Erreur::Config("--mutant-gpu plaquette exige le chemin tout GPU (--mesure gpu --rmax 0, sans --mutant)".into()));
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(fils).build().map_err(|e| Erreur::Config(e.to_string()))?;
    if ou != OuMesure::Cpu {
        return executer_gpu(&chaines, &lis, rmax, mutant, mutant_gpu, &pool, &sortie, tout_gpu);
    }
    let res: Vec<Value> = pool.install(|| chaines.iter().map(|ch| mesurer_chaine(ch, &lis, rmax, mutant, fils)).collect::<Resultat<Vec<_>>>())?;
    verifier_finitude(&res)?;
    ecrire_json(&sortie, &json!({"schema": SCHEMA_CORR, "mutant": format!("{mutant:?}"), "rmax": rmax, "lissages": lis.iter().map(|x| x.0.clone()).collect::<Vec<_>>(), "chaines": res}))?;
    ecrire_provenance(&sortie, &chaines, &lis, rmax, mutant, "cpu", None)?;
    eprintln!("kvu rg correlateur : {} chaîne(s) -> {}", chaines.len(), sortie.display());
    Ok(0)
}

/// `--mesure gpu` : verrou GPU de kvu pris AVANT d'ouvrir le GPU (règle 0.7), module compilé une fois, chaînes en séquence.
/// La sortie a les mêmes champs que le chemin CPU, plus `mesure_gpu` (appareil, backend, empreinte du code objet, mutant).
#[allow(clippy::too_many_arguments)]
fn executer_gpu(
    chaines: &[ChaineTrouvee],
    lis: &[(String, Lissage)],
    rmax: usize,
    mutant: Mutant,
    mutant_gpu: MutantGpu,
    pool: &rayon::ThreadPool,
    sortie: &Path,
    tout_gpu: bool,
) -> Resultat<i32> {
    if let Some(ch) = chaines.iter().find(|c| c.l < 2 || c.l % 2 != 0) {
        return Err(Erreur::Config(format!("--mesure gpu : L = {} ({}) : L pair ≥ 2 requis", ch.l, ch.nom)));
    }
    let calcul = super::cli::ouvrir_calcul_nom("auto")?;
    let app = calcul
        .appareil()
        .ok_or_else(|| Erreur::Config("--mesure gpu : aucun GPU sur cette machine (le backend choisi est le CPU)".into()))?;
    let mut flot = FlotGpu::nouveau(app, mutant_gpu)?;
    let t0 = std::time::Instant::now();
    let mut res = Vec::with_capacity(chaines.len());
    for ch in chaines {
        res.push(mesurer_chaine_gpu(ch, lis, rmax, mutant, pool, &mut flot, tout_gpu)?);
    }
    verifier_finitude(&res)?;
    let info = json!({
        "appareil": app.nom(), "backend": app.backend().nom(), "arch": app.arch(),
        "sha256_code": flot.sha256_code(), "mutant_gpu": format!("{mutant_gpu:?}"),
        "chemin": if tout_gpu { "tout_gpu" } else { "flot_gpu" }, "dir_temps": DIR_TEMPS_CORR,
        "duree_s": t0.elapsed().as_secs_f64(), "duree_flot_gpu_s": flot.duree.as_secs_f64(), "n_flots": flot.n_flots,
    });
    // O4 (revue c07) : sortie déterministe et de mêmes clés que le chemin CPU ; appareil, durées et code objet vont au manifeste
    ecrire_json(sortie, &json!({"schema": SCHEMA_CORR, "mutant": format!("{mutant:?}"), "rmax": rmax, "lissages": lis.iter().map(|x| x.0.clone()).collect::<Vec<_>>(), "chaines": res}))?;
    let mut info_prov = info;
    info_prov["module_compile"] = serde_json::to_value(flot.provenance()).unwrap_or(Value::Null);
    ecrire_provenance(sortie, chaines, lis, rmax, mutant, if tout_gpu { "gpu (tout GPU)" } else { "gpu (flot GPU, mesure hôte)" }, Some(info_prov))?;
    eprintln!("kvu rg correlateur [gpu] : {} chaîne(s) -> {}", chaines.len(), sortie.display());
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rg::bloc::tests::{aleatoire, froid, jauges, transformer};

    fn proche(a: f64, b: f64, e: f64) {
        assert!((a - b).abs() < e, "{a} vs {b}");
    }

    #[test]
    fn champ_froid_tout_nul() {
        let u = froid(4);
        let (d, pp) = densites(&u);
        assert!(d.iter().all(|x| x.abs() < 1e-14) && pp.iter().all(|x| x.abs() < 1e-14));
        let m = mesurer(&u, None, &Lissage::Brut, 2, Mutant::Aucun, 2).unwrap();
        assert!(m["c0"].as_f64().unwrap().abs() < 1e-14);
    }

    #[test]
    fn dft_contre_force_brute() {
        let l = 6; // pas une puissance de 2 : 2·3
        let v = l * l * l * l;
        let f: Vec<f64> = (0..v).map(|i| ((i * 7919) % 101) as f64 / 101.0).collect();
        let g: Vec<f64> = (0..v).map(|i| ((i * 104729) % 97) as f64 / 97.0).collect();
        let c = intercorrelation(&f, &g, l);
        for &r in &[[0, 0, 0, 0], [1, 0, 0, 0], [2, 5, 1, 3], [5, 5, 5, 5]] {
            let mut s = 0.0;
            for (x, fx) in f.iter().enumerate() {
                let xc = coordonnees(x, l);
                let y = [(xc[0] + r[0]) % l, (xc[1] + r[1]) % l, (xc[2] + r[2]) % l, (xc[3] + r[3]) % l];
                s += fx * g[site(y, l)];
            }
            proche(c[site(r, l)], s / v as f64, 1e-12);
        }
    }

    #[test]
    fn invariance_de_jauge_et_orbites() {
        let u = aleatoire(4);
        let t = transformer(&u, &jauges(4));
        let a = mesurer(&u, None, &Lissage::Brut, 2, Mutant::Aucun, 2).unwrap();
        let b = mesurer(&t, None, &Lissage::Brut, 2, Mutant::Aucun, 2).unwrap();
        for k in a["corr"].as_object().unwrap().keys() {
            proche(a["corr"][k].as_f64().unwrap(), b["corr"][k].as_f64().unwrap(), 1e-11);
        }
        for t in 0..4 {
            proche(a["O_t"][t].as_f64().unwrap(), b["O_t"][t].as_f64().unwrap(), 1e-11);
        }
        // somme des D_t = V·⟨d⟩
        let s: f64 = (0..4).map(|t| a["D_t"][t].as_f64().unwrap()).sum();
        proche(s, 256.0 * a["moy_d"].as_f64().unwrap(), 1e-9);
    }

    #[test]
    fn mutant_lien_n_est_pas_invariant_de_jauge() {
        let u = aleatoire(4);
        let t = transformer(&u, &jauges(4));
        let a = mesurer(&u, None, &Lissage::Brut, 2, Mutant::Lien, 2).unwrap();
        let b = mesurer(&t, None, &Lissage::Brut, 2, Mutant::Lien, 2).unwrap();
        let ecart = a["corr"]["1,0,0,0"].as_f64().unwrap() - b["corr"]["1,0,0,0"].as_f64().unwrap();
        assert!(ecart.abs() > 1e-6, "le mutant devrait casser l'invariance de jauge ({ecart})");
    }

    #[test]
    fn flot_sur_l_pair_non_multiple_de_3_et_invariance() {
        let u = aleatoire(4);
        let t = transformer(&u, &jauges(4));
        let lis = Lissage::Flot { eps: 0.02, n: 3 };
        let a = mesurer(&u, None, &lis, 2, Mutant::Aucun, 2).unwrap();
        let b = mesurer(&t, None, &lis, 2, Mutant::Aucun, 2).unwrap();
        for t in 0..4 {
            proche(a["O_t"][t].as_f64().unwrap(), b["O_t"][t].as_f64().unwrap(), 1e-9);
        }
        // le flot lisse : la plaquette moyenne monte
        let brut = mesurer(&u, None, &Lissage::Brut, 2, Mutant::Aucun, 2).unwrap();
        assert!(a["moy_pp"].as_f64().unwrap() < brut["moy_pp"].as_f64().unwrap());
    }

    #[test]
    fn orbite_compte() {
        // (1,0,0,0) : 8 vecteurs ; (1,1,0,0) : 24 ; (1,1,1,1) : 16
        let l = 8;
        let c: Vec<f64> = (0..l * l * l * l).map(|_| 1.0).collect();
        proche(moyenne_orbite(&c, l, [1, 1, 1, 1]), 1.0, 1e-14);
        assert_eq!(orbites(8, 2).len(), 5);
    }

    // ---------- module mesure-gpu (11/11) ----------

    fn bits(v: &Value) -> String {
        serde_json::to_string(v).unwrap()
    }

    /// Le découpage « lissage puis mesure du champ lissé » rend les mêmes octets que l'ancien `mesurer` (chemin CPU inchangé).
    #[test]
    fn mesurer_egale_lisser_puis_mesurer_lisses_au_bit() {
        let u = aleatoire(4);
        let t = transformer(&u, &jauges(4));
        for lis in [Lissage::Brut, Lissage::Flot { eps: 0.02, n: 3 }] {
            for (mutant, autre) in [(Mutant::Aucun, None), (Mutant::Lien, None), (Mutant::Melange, Some(&t))] {
                let a = mesurer(&u, autre, &lis, 2, mutant, 2).unwrap();
                let w = appliquer(&u, &lis, 2).unwrap();
                let wa = autre.map(|x| appliquer(x, &lis, 2).unwrap());
                assert_eq!(bits(&a), bits(&mesurer_lisses(&w, wa.as_ref(), 2, mutant).unwrap()), "{lis:?} {mutant:?}");
            }
        }
    }

    /// Juge CORR-GPU-CPU : 0 sur deux arbres égaux ; une perturbation relative de 1e-9 sur UNE valeur le rend rouge à 1e-10 ;
    /// toute différence de structure est refusée.
    #[test]
    fn juge_ecart_relatif_et_mutants() {
        let u = aleatoire(4);
        let a = json!([mesurer(&u, None, &Lissage::Brut, 2, Mutant::Aucun, 2).unwrap()]);
        assert_eq!(ecart_relatif_max(&a, &a, "x").unwrap().0, 0.0);
        assert!(nombre_de_valeurs(&a) > 10);
        let mut b = a.clone();
        let x = b[0]["O_t"][2].as_f64().unwrap();
        b[0]["O_t"][2] = json!(x * (1.0 + 1e-9));
        let (e, ou) = ecart_relatif_max(&a, &b, "x").unwrap();
        assert!(e > 1e-10 && e < 2e-9, "{e}");
        assert!(ou.contains("O_t[2]"), "{ou}");
        let mut c = a.clone();
        c[0]["O_t"].as_array_mut().unwrap().pop();
        assert!(ecart_relatif_max(&a, &c, "x").is_err());
        let mut d = a.clone();
        d[0].as_object_mut().unwrap().remove("c0");
        assert!(ecart_relatif_max(&a, &d, "x").is_err());
        // zéros des deux côtés : écart nul ; zéro contre non-zéro : écart 1
        assert_eq!(ecart_relatif_max(&json!(0.0), &json!(0.0), "z").unwrap().0, 0.0);
        assert_eq!(ecart_relatif_max(&json!(0.0), &json!(1e-300), "z").unwrap().0, 1.0);
    }

    /// Dossier de chaîne temporaire (manifeste L = `l_manifeste`, β = 2,5) avec les configurations données.
    fn chaine_temp(nom: &str, l_manifeste: usize, champs: &[ChampHote]) -> std::path::PathBuf {
        use crate::stockage::npy::{ecrire_champ, FormatChamp};
        let d = std::env::temp_dir().join(format!("kvu_corr_{nom}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("manifest.json"), format!("{{\"beta\": 2.5, \"l\": {l_manifeste}}}")).unwrap();
        for (i, c) in champs.iter().enumerate() {
            ecrire_champ(&d.join(format!("cfg_{:06}.npy", i + 1)), c, FormatChamp::Q64, &crate::stockage::GardeSsd::nouvelle(0.0)).unwrap();
        }
        d
    }

    fn lancer(args: &[&str]) -> Resultat<i32> {
        let m = sous_commandes().remove(0).try_get_matches_from(std::iter::once("correlateur").chain(args.iter().copied())).unwrap();
        executer(&m)
    }

    /// O1 (revue c02) : une configuration dont le côté diffère du manifeste est refusée AVANT toute DFT (le mutant `melange`
    /// apparierait L = 4 et L = 2), avec ou sans mutant ; aucun JSON n'est écrit. Une chaîne valide passe.
    #[test]
    fn o1_cote_different_du_manifeste_refuse() {
        let d = chaine_temp("o1", 4, &[aleatoire(4), froid(2)]);
        for mutant in ["melange", "aucun"] {
            let sortie = d.join(format!("sortie_{mutant}.json"));
            let r = lancer(&["--ens", d.to_str().unwrap(), "--mutant", mutant, "--rmax", "2", "--fils", "2", "--sortie", sortie.to_str().unwrap()]);
            assert!(matches!(r, Err(Erreur::Format(_))), "{mutant} : {r:?}");
            assert!(!sortie.exists(), "{mutant} : un JSON a été écrit");
        }
        let ok = chaine_temp("o1ok", 4, &[aleatoire(4), froid(4)]);
        let sortie = ok.join("sortie.json");
        assert_eq!(lancer(&["--ens", ok.to_str().unwrap(), "--mutant", "melange", "--rmax", "2", "--fils", "2", "--sortie", sortie.to_str().unwrap()]).unwrap(), 0);
        assert!(sortie.exists());
        // garde de dernier recours de la DFT
        assert!(std::panic::catch_unwind(|| intercorrelation(&[0.0; 256], &[0.0; 16], 4)).is_err());
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&ok);
    }

    /// O4 (revue c02) : pas de flot non fini ou ≤ 0 refusé à l'analyse de la ligne de commande ; une mesure non finie (null
    /// dans le JSON) est détectée avec son chemin ; une sortie valide passe ; la commande refuse `flot:NaN:1` sans rien écrire.
    #[test]
    fn o4_pas_fini_et_finitude_des_mesures() {
        for t in ["flot:NaN:1", "flot:inf:2", "flot:-0.02:50", "flot:0:5", "flot:-inf:1"] {
            assert!(lire_lissage(t).is_err(), "{t} accepté");
        }
        assert!(matches!(lire_lissage("flot:0.02:50"), Ok(Lissage::Flot { n: 50, .. })));
        let u = aleatoire(4);
        let bonne = mesurer(&u, None, &Lissage::Flot { eps: 0.02, n: 2 }, 2, Mutant::Aucun, 2).unwrap();
        assert_eq!(premiere_non_finie(&bonne), None);
        assert!(verifier_finitude(&[json!({"configs": [{"mesures": {"brut": bonne.clone()}}]})]).is_ok());
        let mut mauvaise = bonne.clone();
        mauvaise["O_t"][1] = json!(f64::NAN); // serde_json : NaN devient null
        assert_eq!(premiere_non_finie(&mauvaise).as_deref(), Some("/O_t[1]"));
        let r = verifier_finitude(&[json!({"configs": [{"mesures": {"flot": mauvaise}}]})]);
        assert!(matches!(&r, Err(Erreur::Validation(m)) if m.contains("chaines[0]/configs[0]/mesures/flot/O_t[1]")), "{r:?}");
        let d = chaine_temp("o4", 4, &[aleatoire(4)]);
        let sortie = d.join("sortie.json");
        assert!(matches!(lancer(&["--ens", d.to_str().unwrap(), "--lissage", "brut,flot:NaN:1", "--sortie", sortie.to_str().unwrap()]), Err(Erreur::Config(_))));
        assert!(!sortie.exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// O13 (revue c02) : cardinalités des orbites par des tableaux delta (8, 24, 16, 48 pour (2,1,0,0), 192 pour (3,2,1,0)) et
    /// dégénérescences à L/2 (L = 4 : ±2 confondus, (2,0,0,0) a 4 éléments, (2,1,0,0) 24, (2,2,2,2) un seul).
    #[test]
    fn o13_cardinalites_des_orbites() {
        let delta = |l: usize, x: [usize; 4]| {
            let mut c = vec![0.0; l.pow(4)];
            c[site(x, l)] = 1.0;
            c
        };
        for (l, r, card) in [
            (8, [1, 0, 0, 0], 8.0),
            (8, [1, 1, 0, 0], 24.0),
            (8, [1, 1, 1, 1], 16.0),
            (8, [2, 1, 0, 0], 48.0),
            (8, [3, 2, 1, 0], 192.0),
            (4, [2, 0, 0, 0], 4.0),
            (4, [2, 1, 0, 0], 24.0),
            (4, [2, 2, 2, 2], 1.0),
        ] {
            proche(moyenne_orbite(&delta(l, r), l, r), 1.0 / card, 1e-15);
            // un élément de l'orbite ailleurs que r (permutation + signe) compte aussi
            let x = [(l - r[1]) % l, r[0], r[3], r[2]];
            proche(moyenne_orbite(&delta(l, x), l, r), 1.0 / card, 1e-15);
            // un site hors de l'orbite ne compte pas
            proche(moyenne_orbite(&delta(l, [0, 0, 0, 0]), l, r), 0.0, 1e-15);
        }
    }

    /// O10 (revue c02) : --rmax énorme = mêmes orbites que rmax = L (pas de débordement de rmax²) ; --fils 0 refusé.
    /// O12 : --mutant melange refusé avec une seule configuration par chaîne ; avec deux, les partenaires diffèrent.
    #[test]
    fn o10_o12_bornes_et_melange() {
        assert_eq!(orbites(8, 4294967296), orbites(8, 8));
        assert_eq!(orbites(8, usize::MAX), orbites(8, 8));
        assert_eq!(orbites(8, 3), orbites(8, 3).into_iter().filter(|r| r.iter().map(|a| a * a).sum::<usize>() <= 9).collect::<Vec<_>>());
        let d = chaine_temp("o12", 4, &[aleatoire(4), froid(4)]);
        let sortie = d.join("s.json");
        let s = sortie.to_str().unwrap();
        let e = d.to_str().unwrap();
        assert!(matches!(lancer(&["--ens", e, "--fils", "0", "--rmax", "1", "--sortie", s]), Err(Erreur::Config(_))));
        assert!(matches!(lancer(&["--ens", e, "--mutant", "melange", "--max-configs", "1", "--rmax", "1", "--sortie", s]), Err(Erreur::Config(_))));
        assert!(!sortie.exists());
        assert_eq!(lancer(&["--ens", e, "--mutant", "melange", "--rmax", "1", "--fils", "2", "--sortie", s]).unwrap(), 0);
        let j: Value = serde_json::from_str(&std::fs::read_to_string(&sortie).unwrap()).unwrap();
        for c in j["chaines"][0]["configs"].as_array().unwrap() {
            let m = &c["mesures"]["brut"];
            assert_ne!(m["moy_d"], m["moy_g"], "le partenaire du mélange doit être une autre configuration");
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    /// O5/O16 (revue c02) : `<sortie>.manifest.json` porte provenance, géométrie (dir_temps 0) et conventions ; il référence la
    /// sortie par son sha256 ; la sortie elle-même ne contient que les clés d'origine.
    #[test]
    fn o5_o16_manifeste_a_cote_de_la_sortie() {
        let d = chaine_temp("o5", 4, &[aleatoire(4), froid(4)]);
        let sortie = d.join("s.json");
        assert_eq!(lancer(&["--ens", d.to_str().unwrap(), "--rmax", "1", "--fils", "2", "--sortie", sortie.to_str().unwrap()]).unwrap(), 0);
        let j: Value = serde_json::from_str(&std::fs::read_to_string(&sortie).unwrap()).unwrap();
        let mut cles: Vec<&String> = j.as_object().unwrap().keys().collect();
        cles.sort();
        assert_eq!(cles, ["chaines", "lissages", "mutant", "rmax", "schema"]);
        let man: Value = serde_json::from_str(&std::fs::read_to_string(d.join("s.json.manifest.json")).unwrap()).unwrap();
        assert_eq!(man["geometrie"]["dir_temps"], json!(0));
        assert_eq!(man["geometrie"]["cotes"], json!([4, 4, 4, 4]));
        assert_eq!(man["parametres"]["chaines"][0]["n_configs"], json!(2));
        assert!(man["conventions"]["O_t"].as_str().unwrap().contains("NON divisee"));
        assert_eq!(man["fichiers"][0]["sha256"].as_str().unwrap(), crate::heritage::manifest::sha256_fichier(&sortie).unwrap());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// O2 (revue c07) : le juge refuse `{}` contre `{}`, des `null` aux mêmes endroits, des clés supprimées des deux côtés, une
    /// longueur de tranche fausse, une tolérance infinie ou négative ; il accepte deux sorties valides identiques (écart 0).
    #[test]
    fn o2_juge_refuse_les_documents_invalides() {
        let d = chaine_temp("juge", 4, &[aleatoire(4), froid(4)]);
        let sortie = d.join("s.json");
        assert_eq!(lancer(&["--ens", d.to_str().unwrap(), "--lissage", "brut,flot:0.02:2", "--rmax", "1", "--fils", "2", "--sortie", sortie.to_str().unwrap()]).unwrap(), 0);
        let bon: Value = serde_json::from_str(&std::fs::read_to_string(&sortie).unwrap()).unwrap();
        let (e, _, n) = juger(&bon, &bon, 1e-10).unwrap();
        assert_eq!(e, 0.0);
        assert!(n > 20);
        assert!(juger(&json!({}), &json!({}), 1e-10).is_err());
        let mut nul = bon.clone();
        nul["chaines"][0]["configs"][0]["mesures"]["flot:0.02:2"]["O_t"][0] = Value::Null;
        assert!(juger(&nul, &nul, 1e-10).is_err());
        let mut amputee = bon.clone();
        amputee["chaines"][0]["configs"][1]["mesures"]["brut"].as_object_mut().unwrap().remove("moy_d");
        assert!(juger(&amputee, &amputee, 1e-10).is_err());
        let mut courte = bon.clone();
        courte["chaines"][0]["configs"][0]["mesures"]["brut"]["D_t"].as_array_mut().unwrap().pop();
        assert!(juger(&courte, &courte, 1e-10).is_err());
        let mut vide = bon.clone();
        vide["chaines"] = json!([]);
        assert!(juger(&vide, &vide, 1e-10).is_err());
        assert!(juger(&bon, &bon, f64::INFINITY).is_err());
        assert!(juger(&bon, &bon, -1.0).is_err());
        // seconde relecture (c07b) : « NaN » en chaîne dans une tranche, booléen à la place d'un scalaire, des deux côtés
        let mut chaine_nan = bon.clone();
        chaine_nan["chaines"][0]["configs"][0]["mesures"]["brut"]["O_t"][0] = json!("NaN");
        assert!(juger(&chaine_nan, &chaine_nan, 1e-10).is_err());
        let mut booleen = bon.clone();
        booleen["chaines"][0]["configs"][1]["mesures"]["flot:0.02:2"]["moy_d"] = json!(true);
        assert!(juger(&booleen, &booleen, 1e-10).is_err());
        // clés de corr : une orbite en moins, une orbite étrangère, une valeur non numérique (des deux côtés)
        let mut orbite = bon.clone();
        orbite["chaines"][0]["configs"][0]["mesures"]["brut"]["corr"].as_object_mut().unwrap().remove("1,0,0,0");
        assert!(juger(&orbite, &orbite, 1e-10).is_err());
        let mut etrangere = bon.clone();
        etrangere["chaines"][0]["configs"][0]["mesures"]["brut"]["corr"]["9,9,9,9"] = json!(0.5);
        assert!(juger(&etrangere, &etrangere, 1e-10).is_err());
        let mut corr_texte = bon.clone();
        corr_texte["chaines"][0]["configs"][0]["mesures"]["brut"]["corr"]["1,0,0,0"] = json!("0.1");
        assert!(juger(&corr_texte, &corr_texte, 1e-10).is_err());
        // rmax = 0 : corr doit être vide
        let mut r0 = bon.clone();
        r0["rmax"] = json!(0);
        assert!(juger(&r0, &r0, 1e-10).is_err());
        assert_eq!(bon["rmax"], json!(1));
        assert_eq!(bon["chaines"][0]["configs"][0]["mesures"]["brut"]["corr"].as_object().unwrap().len(), orbites(4, 1).len());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// GPU de la machine, ou `None` (SKIP, qui ne vaut pas vert). À lancer sous le verrou GPU de la machine.
    fn appareil_test() -> Option<crate::gpu::Appareil> {
        // O6 (revue c07) : un test GPU explicitement demandé (--ignored) ÉCHOUE si le GPU manque ou ne s'ouvre pas
        match crate::gpu::ouvrir_auto() {
            Ok(Some(a)) => Some(a),
            Ok(None) => panic!("FAIL CORR-GPU : test GPU demandé, mais le backend choisi est le CPU (aucun GPU)"),
            Err(e) => panic!("FAIL CORR-GPU : ouverture du GPU impossible : {e}"),
        }
    }

    /// Porte CORR-GPU-CPU en petit (L = 6, champ de Haar, jauge transformée, mutants) : le chemin GPU égale le CPU à 1e-10
    /// relatif sur toutes les sorties ; le mutant « un pas de flot en moins » est rouge.
    #[test]
    #[ignore = "GPU"]
    fn corr_gpu_egal_cpu_et_mutant_rouge() {
        let Some(app) = appareil_test() else { return };
        let u = aleatoire(6);
        let t = transformer(&u, &jauges(6));
        let lis = [Lissage::Brut, Lissage::Flot { eps: 0.02, n: 10 }];
        let mut gpu = FlotGpu::nouveau(&app, MutantGpu::Aucun).unwrap();
        let mut gpu_m = FlotGpu::nouveau(&app, MutantGpu::PasEnMoins).unwrap();
        for (mutant, autre) in [(Mutant::Aucun, None), (Mutant::Melange, Some(&t))] {
            for l in &lis {
                let cpu = mesurer(&u, autre, l, 3, mutant, 4).unwrap();
                let lisser = |g: &mut FlotGpu| {
                    let w = g.appliquer(&u, l).unwrap();
                    let wa = autre.map(|x| g.appliquer(x, l).unwrap());
                    mesurer_lisses(&w, wa.as_ref(), 3, mutant).unwrap()
                };
                let (e, ou) = ecart_relatif_max(&cpu, &lisser(&mut gpu), "m").unwrap();
                println!("CORR-GPU {l:?} {mutant:?} : écart relatif max {e:e} ({ou})");
                assert!(e <= 1e-10, "FAIL CORR-GPU-CPU {l:?} {mutant:?} : {e:e} en {ou}");
                let (em, _) = ecart_relatif_max(&cpu, &lisser(&mut gpu_m), "m").unwrap();
                match l {
                    Lissage::Brut => assert_eq!(em, 0.0, "le brut ne passe pas par le GPU"),
                    Lissage::Flot { .. } => assert!(em > 1e-6, "FAIL CORR-GPU-MUTANT : pas en moins non détecté ({em:e})"),
                }
            }
        }
        // invariance de jauge du chemin GPU
        let a = mesurer_lisses(&gpu.appliquer(&u, &lis[1]).unwrap(), None, 3, Mutant::Aucun).unwrap();
        let b = mesurer_lisses(&gpu.appliquer(&t, &lis[1]).unwrap(), None, 3, Mutant::Aucun).unwrap();
        let (e, _) = ecart_relatif_max(&a, &b, "j").unwrap();
        assert!(e <= 1e-10, "jauge GPU : {e:e}");
        println!("PASS CORR-GPU-CPU (L = 6) ; PASS CORR-GPU-MUTANT");
    }

    /// Chemin tout GPU (plaquettes, trèfle, tranches sur le GPU) contre le CPU : à rmax = 0, mêmes sorties à 1e-10 relatif ;
    /// sommes par tranche justes pour les quatre directions du temps (référence : d et p du CPU, sommées ici) ;
    /// mutants « plaquette mal orientée » et « un pas en moins » rouges.
    #[test]
    #[ignore = "GPU"]
    fn corr_tout_gpu_egal_cpu_et_mutants_rouges() {
        let Some(app) = appareil_test() else { return };
        // O7 (revue c07) : L³ = 216 < 256 n'exerce pas la boucle de k_corr_tranches ; L = 8 (512) et 10 (1000, reste non nul)
        for l in [6, 8, 10] {
            tout_gpu_un_cote(&app, l);
        }
        println!("PASS CORR-TOUT-GPU (L = 6, 8, 10)");
    }

    fn tout_gpu_un_cote(app: &crate::gpu::Appareil, l: usize) {
        let u = aleatoire(l);
        let lis = vec![("brut".to_string(), Lissage::Brut), ("flot".to_string(), Lissage::Flot { eps: 0.02, n: 10 })];
        let mut gpu = FlotGpu::nouveau(app, MutantGpu::Aucun).unwrap();
        let sommes = gpu.tranches(&u, &lis, DIR_TEMPS_CORR).unwrap();
        for ((nom, li), (d_t, pp_t)) in lis.iter().zip(sommes) {
            let cpu = mesurer(&u, None, li, 0, Mutant::Aucun, 4).unwrap();
            let (e, ou) = ecart_relatif_max(&cpu, &assembler_tranches(l, d_t, pp_t), "m").unwrap();
            println!("CORR-TOUT-GPU {nom} : écart relatif max {e:e} ({ou})");
            assert!(e <= 1e-10, "FAIL CORR-TOUT-GPU {nom} : {e:e} en {ou}");
        }
        // quatre directions du temps, sur le champ flotté par le CPU (référence) et par le GPU
        let w = appliquer(&u, &lis[1].1, 4).unwrap();
        let p = plaquettes(&w);
        let (d, _) = densites(&w);
        for dir in 0..4 {
            let mut d_ref = vec![0.0; l];
            let mut pp_ref = vec![0.0; l];
            for s in 0..l.pow(4) {
                let x = coordonnees(s, l);
                d_ref[x[dir]] += d[s];
                for (k, &(mu, nu)) in PLANS.iter().enumerate() {
                    if mu != dir && nu != dir {
                        pp_ref[x[dir]] += p[k][s];
                    }
                }
            }
            let (d_t, pp_t) = gpu.tranches(&u, &lis[1..], dir).unwrap().pop().unwrap();
            let (e, ou) = ecart_relatif_max(&json!([d_ref, pp_ref]), &json!([d_t, pp_t]), &format!("dir{dir}")).unwrap();
            println!("CORR-TOUT-GPU dir_temps = {dir} : écart relatif max {e:e} ({ou})");
            assert!(e <= 1e-10, "FAIL CORR-TOUT-GPU dir_temps = {dir} : {e:e}");
        }
        for m in [MutantGpu::Plaquette, MutantGpu::PasEnMoins] {
            let mut g = FlotGpu::nouveau(app, m).unwrap();
            let (d_t, pp_t) = g.tranches(&u, &lis[1..], DIR_TEMPS_CORR).unwrap().pop().unwrap();
            let cpu = mesurer(&u, None, &lis[1].1, 0, Mutant::Aucun, 4).unwrap();
            let (e, _) = ecart_relatif_max(&cpu, &assembler_tranches(l, d_t, pp_t), "m").unwrap();
            assert!(e > 1e-6, "FAIL CORR-TOUT-GPU-MUTANT {m:?} non détecté ({e:e})");
            println!("PASS CORR-TOUT-GPU-MUTANT {m:?} L = {l} : rouge ({e:e})");
        }
    }

    /// O7 (revue c07) : pipeline complet de la commande sur GPU (lecture, GPU, mesure hôte, publication), chemins tout GPU
    /// (rmax 0) et flot GPU + hôte (rmax 2), 1 et 3 fils : sorties jugées égales au CPU à 1e-10 par `juger` ; refus O1 sur le
    /// chemin GPU (configuration de côté différent) sans sortie.
    #[test]
    #[ignore = "GPU"]
    fn corr_gpu_pipeline_de_bout_en_bout() {
        let Some(_app) = appareil_test() else { return };
        drop(_app);
        let d = chaine_temp("pipe", 8, &[aleatoire(8), transformer(&aleatoire(8), &jauges(8)), froid(8)]);
        let e = d.to_str().unwrap().to_string();
        let lire = |p: &std::path::Path| -> Value { serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap() };
        for rmax in ["0", "2"] {
            let cpu = d.join(format!("cpu_{rmax}.json"));
            assert_eq!(lancer(&["--ens", &e, "--lissage", "brut,flot:0.02:5", "--rmax", rmax, "--fils", "2", "--sortie", cpu.to_str().unwrap()]).unwrap(), 0);
            for fils in ["1", "3"] {
                let g = d.join(format!("gpu_{rmax}_{fils}.json"));
                assert_eq!(lancer(&["--ens", &e, "--lissage", "brut,flot:0.02:5", "--rmax", rmax, "--fils", fils, "--mesure", "gpu", "--sortie", g.to_str().unwrap()]).unwrap(), 0);
                let (ec, ou, n) = juger(&lire(&cpu), &lire(&g), 1e-10).unwrap();
                println!("CORR-GPU-PIPELINE rmax {rmax} fils {fils} : {n} valeurs, écart {ec:e} ({ou})");
                assert!(ec <= 1e-10, "FAIL CORR-GPU-PIPELINE rmax {rmax} fils {fils} : {ec:e} en {ou}");
                let man = lire(&d.join(format!("gpu_{rmax}_{fils}.json.manifest.json")));
                assert!(man["mesure_gpu"]["sha256_code"].is_string());
            }
        }
        let mauvaise = chaine_temp("pipe_o1", 8, &[aleatoire(8), froid(4)]);
        let g = mauvaise.join("gpu.json");
        let r = lancer(&["--ens", mauvaise.to_str().unwrap(), "--rmax", "0", "--mesure", "gpu", "--sortie", g.to_str().unwrap()]);
        assert!(matches!(r, Err(Erreur::Format(_))), "{r:?}");
        assert!(!g.exists());
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&mauvaise);
        println!("PASS CORR-GPU-PIPELINE");
    }
}
