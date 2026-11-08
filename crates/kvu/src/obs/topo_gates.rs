//! Gates de données topologiques WP19 (GATES v4.3), sans mesure de champ ni accès GPU.
//! Les décisions utilisent les valeurs brutes ; l'arrondi décimal concerne seulement le rapport.

use super::topo::{RecordTopo, IDX_T3, MULT_T0, TEMPS_A2};
use crate::commun::{Erreur, Resultat};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Verdict {
    Pass,
    Fail,
    Skip,
    Info,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::Skip => "SKIP",
            Self::Info => "INFO",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LigneGate {
    pub verdict: Verdict,
    pub id: String,
    pub detail: String,
}

impl fmt::Display for LigneGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.verdict, self.id, self.detail)
    }
}

pub const QUANTILE_T31: f64 = 3.26094191889439;

fn ligne(verdict: Verdict, id: &str, detail: impl Into<String>) -> LigneGate {
    LigneGate {
        verdict,
        id: id.into(),
        detail: detail.into(),
    }
}

/// Une erreur indique le chemin et le numéro de ligne physique, lignes blanches comprises.
pub fn lire_jsonl(chemin: &Path) -> Resultat<Vec<RecordTopo>> {
    let fichier = File::open(chemin).map_err(|e| Erreur::io(chemin.display().to_string(), e))?;
    let mut records = Vec::new();
    for (i, texte) in BufReader::new(fichier).lines().enumerate() {
        let contexte = format!("{}: ligne {}", chemin.display(), i + 1);
        let texte = texte.map_err(|e| Erreur::io(&contexte, e))?;
        if !texte.trim().is_empty() {
            records.push(
                serde_json::from_str(&texte)
                    .map_err(|e| Erreur::Format(format!("{contexte}: {e}")))?,
            );
        }
    }
    Ok(records)
}

// Somme directe des probabilités binomiales, évaluées en logarithmes pour éviter les
// débordements des coefficients binomiaux. On somme directement la queue demandée.
fn queue_binomiale(n: u64, k: u64, p: f64, superieure: bool) -> f64 {
    if p == 0.0 {
        return f64::from(!superieure || k == 0);
    }
    if p == 1.0 {
        return f64::from(superieure || k == n);
    }
    let mut log_coef = 0.0;
    let mut somme = 0.0;
    for j in 0..=n {
        if (superieure && j >= k) || (!superieure && j <= k) {
            somme += (log_coef + j as f64 * p.ln() + (n - j) as f64 * (-p).ln_1p()).exp();
        }
        if j < n {
            log_coef += ((n - j) as f64).ln() - ((j + 1) as f64).ln();
        }
    }
    somme.min(1.0)
}

/// Intervalle bilatéral exact ; paramètres invalides (dont n = 0) : `(NaN, NaN)`.
pub fn clopper_pearson(k: u64, n: u64, niveau: f64) -> (f64, f64) {
    if n == 0 || k > n || !niveau.is_finite() || !(0.0..1.0).contains(&niveau) || niveau == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let alpha = (1.0 - niveau) / 2.0;
    let borne = |superieure: bool| {
        let (mut bas, mut haut) = (0.0, 1.0);
        for _ in 0..80 {
            let milieu = (bas + haut) / 2.0;
            let proba = queue_binomiale(n, k, milieu, superieure);
            if (superieure && proba < alpha) || (!superieure && proba > alpha) {
                bas = milieu;
            } else {
                haut = milieu;
            }
        }
        (bas + haut) / 2.0
    };
    (
        if k == 0 { 0.0 } else { borne(true) },
        if k == n { 1.0 } else { borne(false) },
    )
}

/// Autocovariance centrée, normalisation commune 1/N à tous les retards (estimateur biaisé).
/// Plus petite fenêtre positive W < N/2 vérifiant W >= c*tau(W).
/// Série de moins de quatre points, constante, non finie ou fenêtre absente : `None`.
/// Une estimation négative est conservée si elle satisfait le critère littéral de fenêtre.
pub fn tau_int_sokal(x: &[f64], c: f64) -> Option<f64> {
    if x.len() < 4
        || !c.is_finite()
        || c <= 0.0
        || x.iter().any(|v| !v.is_finite())
        || x.iter().all(|v| *v == x[0])
    {
        return None;
    }
    let centre = moyenne(x);
    let y: Vec<_> = x.iter().map(|v| v - centre).collect();
    let variance: f64 = y.iter().map(|v| v * v).sum();
    if !variance.is_finite() || variance <= 0.0 {
        return None;
    }
    let mut tau = 0.5;
    for w in 1..x.len().div_ceil(2) {
        tau += y[..x.len() - w]
            .iter()
            .zip(&y[w..])
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / variance;
        if w as f64 >= c * tau {
            return Some(tau);
        }
    }
    None
}

/// Taille cible arrondie à la dizaine supérieure. Paramètres invalides : 0 ;
/// dépassement de usize : saturation au plus grand multiple de dix représentable.
pub fn extension_configs(n_par_chaine: usize, r: f64, r_max: f64) -> usize {
    if !r.is_finite() || r < 0.0 || !r_max.is_finite() || r_max <= 0.0 {
        return 0;
    }
    let cible = n_par_chaine as f64 * (r / (0.8 * r_max)).powi(2);
    let dizaines = (cible.ceil() / 10.0).ceil() as usize;
    dizaines.min(usize::MAX / 10) * 10
}

fn moyenne(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Chaînes de poids égaux. Aucune chaîne : `(NaN, NaN)` ; une seule : `(moyenne, NaN)`.
pub fn jackknife_chaines(valeurs_par_chaine: &[f64]) -> (f64, f64) {
    let n = valeurs_par_chaine.len();
    let m = moyenne(valeurs_par_chaine);
    if n < 2 || valeurs_par_chaine.iter().any(|v| !v.is_finite()) {
        return (m, f64::NAN);
    }
    // Identité du jackknife d'une moyenne : s(moyennes de chaîne)/sqrt(n).
    // Évite la soustraction de deux sommes presque égales dans les réplicats.
    let variance = valeurs_par_chaine
        .iter()
        .map(|v| (v - m).powi(2))
        .sum::<f64>()
        / (n * (n - 1)) as f64;
    (m, variance.sqrt())
}

fn fini(x: &[f64], i: usize) -> Option<f64> {
    x.get(i).copied().filter(|v| v.is_finite())
}
fn quantifie(q: f64, tolerance: f64) -> bool {
    (q - q.round()).abs() < tolerance
}
fn compte(valeurs: impl Iterator<Item = Option<f64>>) -> (usize, usize) {
    valeurs.flatten().fold((0, 0), |(k, n), q| {
        (k + usize::from(quantifie(q, 0.1)), n + 1)
    })
}

fn histogramme(x: &[f64], bas: f64, haut: f64, largeur: f64) -> String {
    let mut bacs = vec![0usize; ((haut - bas) / largeur).round() as usize];
    let (mut dessous, mut dessus) = (0, 0);
    for &q in x {
        if q < bas {
            dessous += 1;
        } else if q > haut {
            dessus += 1;
        } else {
            // Échelle d'abord : évite la perte par soustraction, par exemple
            // (-0.4 + 0.5)/0.05 légèrement inférieur à 2 en binaire.
            let echelle = largeur.recip();
            let i = ((q * echelle - bas * echelle).floor() as usize).min(bacs.len() - 1);
            bacs[i] += 1;
        }
    }
    let texte = bacs
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let a = bas + i as f64 * largeur;
            let b = a + largeur;
            let fin = if i + 1 == bacs.len() { ']' } else { ')' };
            format!("[{a:.6},{b:.6}{fin}:{n}")
        })
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "{texte} sous_borne={dessous} hors_borne_haute={dessus} n={}",
        x.len()
    )
}

pub fn topo_q(recs: &[RecordTopo]) -> Vec<LigneGate> {
    let charges: Vec<_> = recs
        .iter()
        .filter_map(|r| fini(&r.flot_a2.q_imp, IDX_T3))
        .collect();
    let n = charges.iter().filter(|&&q| quantifie(q, 0.1)).count();
    let n_mut = charges.iter().filter(|&&q| quantifie(q + 0.5, 0.1)).count();
    let mut lignes = Vec::new();
    let valide = recs.len() == 100 && charges.len() == recs.len();
    for (id, ok, detail) in [
        ("TOPO-Q", n >= 90, format!("N={n}/{} ; seuil 90/100 ; |Q-round(Q)| < 0.100000", recs.len())),
        ("TOPO-Q-MUT", n_mut < 90, format!("N′={n_mut}/{} ; Q′=Q+0.500000 ; PASS = mutant rouge ; si TOPO-Q passe, N′ ≤ 10 par construction", recs.len())),
    ] {
        if !valide {
            lignes.push(ligne(Verdict::Skip, id, format!("100 configurations requises, {} reçues, {} charges finies disponibles", recs.len(), charges.len())));
        } else {
            let prefixe = if id == "TOPO-Q-MUT" && !ok { "ECHEC_OUTIL mutant non rouge ; " } else { "" };
            lignes.push(ligne(if ok { Verdict::Pass } else { Verdict::Fail }, id, format!("{prefixe}{detail}")));
        }
    }
    let intervalle = if charges.is_empty() {
        "non estimable (aucune charge finie)".into()
    } else {
        let (bas, haut) = clopper_pearson(n as u64, charges.len() as u64, 0.95);
        format!("[{bas:.6}, {haut:.6}] sur {} charges finies", charges.len())
    };
    lignes.push(ligne(Verdict::Info, "TOPO-Q-INFO", format!("Clopper–Pearson 95 % {intervalle} ; sous hypothèse iid, couverture non garantie (chaînes corrélées)")));
    let mut chaines: BTreeMap<usize, Vec<&RecordTopo>> = BTreeMap::new();
    for r in recs {
        chaines.entry(r.chaine).or_default().push(r);
    }
    for (id, chaine) in chaines {
        let (k, total) = compte(chaine.iter().map(|r| fini(&r.flot_a2.q_imp, IDX_T3)));
        lignes.push(ligne(
            Verdict::Info,
            "TOPO-Q-INFO",
            format!(
                "chaîne={id} quantifiées={k}/{total} disponibles sur {}",
                chaine.len()
            ),
        ));
    }
    let (tre, total_tre) = compte(recs.iter().map(|r| fini(&r.flot_a2.q_tre, IDX_T3)));
    let accords: Vec<_> = recs
        .iter()
        .filter_map(|r| {
            Some(fini(&r.cooling.q_imp, 20)?.round() == fini(&r.flot_a2.q_imp, IDX_T3)?.round())
        })
        .collect();
    lignes.push(ligne(
        Verdict::Info,
        "TOPO-Q-INFO",
        format!(
            "trèfle non amélioré={tre}/{total_tre} ; accord cooling/flot={}/{} disponibles sur {}",
            accords.iter().filter(|&&v| v).count(),
            accords.len(),
            recs.len()
        ),
    ));
    lignes.push(ligne(
        Verdict::Info,
        "TOPO-Q-INFO",
        format!("histogramme Q ; {}", histogramme(&charges, -8.0, 8.0, 0.25)),
    ));
    let residus: Vec<_> = charges.iter().map(|q| q - q.round()).collect();
    lignes.push(ligne(
        Verdict::Info,
        "TOPO-Q-INFO",
        format!(
            "histogramme Q-round(Q) ; {}",
            histogramme(&residus, -0.5, 0.5, 0.05)
        ),
    ));
    for (i, t) in TEMPS_A2.iter().enumerate() {
        let (imp, ni) = compte(recs.iter().map(|r| fini(&r.flot_a2.q_imp, i)));
        let (tre, nt) = compte(recs.iter().map(|r| fini(&r.flot_a2.q_tre, i)));
        lignes.push(ligne(
            Verdict::Info,
            "TOPO-Q-T",
            format!(
                "t/a²={t:.6} amélioré={imp}/{ni} trèfle={tre}/{nt} disponibles sur {}",
                recs.len()
            ),
        ));
    }
    for (i, k) in MULT_T0.iter().enumerate() {
        let (imp, ni) = compte(
            recs.iter()
                .map(|r| r.flot_t0.as_ref().and_then(|f| fini(&f.q_imp, i))),
        );
        let (tre, nt) = compte(
            recs.iter()
                .map(|r| r.flot_t0.as_ref().and_then(|f| fini(&f.q_tre, i))),
        );
        lignes.push(ligne(Verdict::Info, "TOPO-Q-T", format!("t={k:.6}·t₀ amélioré={imp}/{ni} trèfle={tre}/{nt} disponibles sur {} ; absences de t₀ exclues des comptes", recs.len())));
    }
    let mut tries: Vec<_> = recs.iter().collect();
    tries.sort_by_key(|r| (r.chaine, r.cfg));
    let premiers = &tries[..tries.len().min(10)];
    let complet = premiers.iter().all(|r| {
        r.cooling.action.len() >= 21 && r.cooling.action[..21].iter().all(|s| s.is_finite())
    });
    let monotones = premiers
        .iter()
        .filter(|r| {
            r.cooling.action.len() >= 21
                && r.cooling.action[..21]
                    .windows(2)
                    .all(|s| s[1] - s[0] <= 1e-12 * s[0].abs())
        })
        .count();
    lignes.push(if premiers.len() < 10 || !complet {
        ligne(Verdict::Skip, "TOPO-COOL", "10 configurations et 21 actions finies par configuration requises")
    } else {
        ligne(if monotones == 10 { Verdict::Pass } else { Verdict::Fail }, "TOPO-COOL", format!("{monotones}/10 actions non croissantes (tri chaîne,cfg), tolérance 1e-12·|action[k]|"))
    });
    lignes
}

pub fn obs4_rejeu(recs: &[RecordTopo]) -> Vec<LigneGate> {
    let (n, total) = compte(recs.iter().map(|r| fini(&r.flot_a2.q_imp, IDX_T3)));
    let historique = if recs.len() != 20 || total != 20 {
        "non applicable (20 configurations et charges finies requises)"
    } else if n >= 18 {
        "PASS historique à titre indicatif"
    } else {
        "FAIL historique à titre indicatif"
    };
    vec![ligne(Verdict::Info, "OBS-4-L16-REJEU", format!("amélioré t/a²=3.000000 : {n}/{total}, {} configurations ; seuil historique 18/20 ; {historique}", recs.len()))]
}

/// Minimum global sur les 401 valeurs de grille, puis départage parmi les valeurs
/// **exactement** égales (f64) au minimum (avenant v4.3.1 : aucune tolérance ajoutée).
fn attribution(trefle: &[f64]) -> f64 {
    let couts: Vec<_> = (800..=1200)
        .map(|m| {
            let s = m as f64 / 1000.0;
            trefle
                .iter()
                .map(|q| (s * q - (s * q).round()).powi(2))
                .sum::<f64>()
        })
        .collect();
    let minimum = couts.iter().copied().fold(f64::INFINITY, f64::min);
    let m = (800i32..=1200)
        .filter(|&m| couts[(m - 800) as usize] == minimum)
        .min_by_key(|m| ((m - 1000).abs(), *m))
        .unwrap_or(1000);
    m as f64 / 1000.0
}

fn verdict_lit(mutant: bool, motif: Option<String>, z: f64, detail: String) -> LigneGate {
    let id = if mutant { "TOPO-LIT-MUT" } else { "TOPO-LIT" };
    if let Some(motif) = motif {
        return ligne(
            if mutant { Verdict::Fail } else { Verdict::Skip },
            id,
            format!(
                "{}{motif} ; {detail}",
                if mutant {
                    "ECHEC_OUTIL non concluant ; "
                } else {
                    ""
                }
            ),
        );
    }
    let ok = if mutant {
        z.abs() > QUANTILE_T31
    } else {
        z.abs() < QUANTILE_T31
    };
    let prefixe = if mutant {
        if ok {
            "mutant rouge ; "
        } else {
            "ECHEC_OUTIL mutant non rouge ; "
        }
    } else {
        ""
    };
    ligne(
        if ok { Verdict::Pass } else { Verdict::Fail },
        id,
        format!("{prefixe}{detail} ; z={z:.6} ; quantile Student ν=31 : {QUANTILE_T31:.6}"),
    )
}

fn lit_invalide(mutant: bool, motif: String) -> Vec<LigneGate> {
    vec![
        verdict_lit(mutant, Some(motif), f64::NAN, "statistiques non estimables".into()),
        ligne(Verdict::Info, "TOPO-LIT-INFO", "non estimable : données ou structure invalides"),
        ligne(Verdict::Info, "TOPO-LIT-PLAQ", "non estimable ; référence AT21 0.6364293(15), réseau 20³×16 ≠ 16⁴ ; sans verdict, réseau différent"),
    ]
}

pub fn topo_lit(recs: &[RecordTopo], mutant: bool) -> Vec<LigneGate> {
    let mut chaines: BTreeMap<usize, Vec<&RecordTopo>> = BTreeMap::new();
    for r in recs {
        chaines.entry(r.chaine).or_default().push(r);
    }
    let attendu = if mutant { 25 } else { 100 };
    if chaines.len() != 32 || chaines.values().any(|ch| ch.len() != attendu) {
        return lit_invalide(mutant, format!("32 chaînes × {attendu} configurations requises (longueurs constantes), {} chaînes et {} configurations reçues", chaines.len(), recs.len()));
    }
    for ch in chaines.values_mut() {
        ch.sort_by_key(|r| r.cfg);
        if ch.windows(2).any(|p| p[0].cfg == p[1].cfg) {
            return lit_invalide(mutant, "configuration dupliquée (chaîne,cfg)".into());
        }
    }
    if recs.iter().any(|r| {
        [fini(&r.cooling.q_imp, 20), fini(&r.cooling.q_tre, 20)]
            .iter()
            .any(|q| q.is_none_or(|v| !(1.2 * v).powi(2).is_finite()))
    }) {
        return lit_invalide(
            mutant,
            "charges cooling[20] absentes, non finies ou carrés non représentables".into(),
        );
    }
    let qi = |r: &RecordTopo| r.cooling.q_imp[20].round();
    let moyennes: Vec<_> = chaines
        .values()
        .map(|ch| ch.iter().map(|r| qi(r).powi(2)).sum::<f64>() / ch.len() as f64)
        .collect();
    let (x, sigma) = jackknife_chaines(&moyennes);
    let n = recs.len();
    let ambigus = recs
        .iter()
        .filter(|r| !quantifie(r.cooling.q_imp[20], 0.25))
        .count();
    let figees: Vec<_> = chaines
        .iter()
        .filter(|(_, ch)| ch.iter().all(|r| qi(r) == qi(ch[0])))
        .map(|(&id, _)| id)
        .collect();
    let trefle: Vec<_> = recs.iter().map(|r| r.cooling.q_tre[20]).collect();
    let s = attribution(&trefle);
    let x_prime = trefle.iter().map(|q| (s * q).round().powi(2)).sum::<f64>() / n as f64;
    let desaccords = recs
        .iter()
        .filter(|r| (s * r.cooling.q_tre[20]).round() != qi(r))
        .count();
    let ecart = (x_prime - x).abs();
    // L'ordre a, b, c, puissance est celui du texte gelé, même en cas d'échecs multiples.
    let motif = if ambigus * 100 > 2 * n {
        Some(format!(
            "(a) attribution ambiguë : {ambigus}/{n} hors |Q-round(Q)| < 0.250000"
        ))
    } else if !sigma.is_finite() || sigma <= 0.0 || 32 - figees.len() < 8 {
        Some(format!("(b) erreur dégénérée ou moins de 8 chaînes non constantes ; σ={sigma:.6}, non constantes={}", 32 - figees.len()))
    } else if desaccords * 100 > 2 * n || !ecart.is_finite() || ecart > sigma {
        Some(format!("(c) attribution dépendante de l'opérateur ; désaccords={desaccords}/{n}, |X′-X|={ecart:.6}, σ={sigma:.6}"))
    } else if sigma > 0.35 {
        Some(format!(
            "puissance insuffisante ; σ={sigma:.6} > 0.350000 ; extension proposée={}/chaîne",
            extension_configs(attendu, sigma, 0.35)
        ))
    } else {
        None
    };
    let z = (x - 8.066) / sigma.hypot(0.064);
    let mut lignes = vec![verdict_lit(mutant, motif, z, format!("X̂={x:.6}, σ={sigma:.6}, référence AT21 β=2.427000 L=16 : 8.066000 ± 0.064000 ; {n} configurations"))];
    lignes.push(ligne(
        Verdict::Info,
        "TOPO-LIT-INFO",
        format!(
            "X̂={x:.6}, V=65536, χ_lat={:.6e}, ⟨Q_I⟩={:.6}",
            x / 65536.0,
            recs.iter().map(qi).sum::<f64>() / n as f64
        ),
    ));
    lignes.push(ligne(Verdict::Info, "TOPO-LIT-INFO", format!("s={s:.6}, X̂′={x_prime:.6}, |X̂′-X̂|={ecart:.6}, désaccords={desaccords}/{n} ; chaînes figées={figees:?}")));
    let ql2 = moyenne(&trefle.iter().map(|q| q * q).collect::<Vec<_>>());
    lignes.push(ligne(Verdict::Info, "TOPO-LIT-INFO", format!("⟨Q_L²⟩={ql2:.6} (trèfle non amélioré, non arrondi, cooling 20) ; référence AT21 6.635000 ± 0.056000")));
    let flot: Vec<_> = recs
        .iter()
        .filter_map(|r| fini(&r.flot_a2.q_imp, IDX_T3))
        .map(|q| q.round().powi(2))
        .collect();
    let x_flot = if flot.len() == n && flot.iter().all(|q| q.is_finite()) {
        format!("{:.6}", moyenne(&flot))
    } else {
        "non estimable".into()
    };
    lignes.push(ligne(Verdict::Info, "TOPO-LIT-INFO", format!("X̂ flot={x_flot} ; procédure de lissage différente (flot t/a² = 3), sans verdict ; disponibles={}/{}", flot.len(), n)));
    let mut taus = Vec::new();
    let mut non_estimables = Vec::new();
    let mut backends: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut mixtes = Vec::new();
    for ((&id, ch), &m) in chaines.iter().zip(&moyennes) {
        let serie: Vec<_> = ch.iter().map(|r| qi(r).powi(2)).collect();
        if let Some(tau) = tau_int_sokal(&serie, 6.0) {
            taus.push(tau);
        } else {
            non_estimables.push(id);
        }
        if ch.iter().all(|r| r.backend_gen == ch[0].backend_gen) {
            backends.entry(&ch[0].backend_gen).or_default().push(m);
        } else {
            mixtes.push(id);
        }
    }
    taus.sort_by(f64::total_cmp);
    let resume_tau = if taus.is_empty() {
        "médiane=non estimable, max=non estimable".into()
    } else {
        let mediane = (taus[(taus.len() - 1) / 2] + taus[taus.len() / 2]) / 2.0;
        format!(
            "médiane={mediane:.6}, max={:.6} (chaînes estimables seulement)",
            taus[taus.len() - 1]
        )
    };
    lignes.push(ligne(Verdict::Info, "TOPO-LIT-INFO", format!("τ_int(Q_I²), Sokal c=6.000000 : {resume_tau} ; estimables={}/32 ; non estimable pour chaînes={non_estimables:?}", taus.len())));
    for (backend, valeurs) in backends {
        let (m, erreur) = jackknife_chaines(&valeurs);
        let erreur = if erreur.is_finite() {
            format!("{erreur:.6}")
        } else {
            "non estimable".into()
        };
        lignes.push(ligne(Verdict::Info, "TOPO-LIT-INFO", format!("backend_gen={backend} X̂={m:.6}, erreur standard={erreur}, chaînes={}, configurations={}", valeurs.len(), valeurs.len() * attendu)));
    }
    if !mixtes.is_empty() {
        lignes.push(ligne(
            Verdict::Info,
            "TOPO-LIT-INFO",
            format!("backend_gen non estimable pour chaînes à backends mixtes={mixtes:?}"),
        ));
    }
    if recs.iter().all(|r| r.plaq0.is_finite()) {
        let plaquettes: Vec<_> = chaines
            .values()
            .map(|ch| ch.iter().map(|r| r.plaq0).sum::<f64>() / ch.len() as f64)
            .collect();
        let (p, erreur) = jackknife_chaines(&plaquettes);
        let delta = p - 0.6364293;
        let z = delta / erreur.hypot(0.0000015);
        lignes.push(ligne(Verdict::Info, "TOPO-LIT-PLAQ", format!("⟨½Re Tr U_p⟩={p:.6}, σ_jackknife={erreur:.6e} ; référence AT21 0.6364293(15), réseau 20³×16 ≠ 16⁴ ; écart={delta:.6e}, écart en σ combinés={z:.6} ; sans verdict, réseau différent")));
    } else {
        lignes.push(ligne(Verdict::Info, "TOPO-LIT-PLAQ", "non estimable : plaquette non finie ; référence AT21 0.6364293(15), réseau 20³×16 ≠ 16⁴ ; sans verdict, réseau différent"));
    }
    lignes
}

pub fn analyser(ens: &str, recs: &[RecordTopo]) -> Vec<LigneGate> {
    match ens {
        "q" => topo_q(recs),
        "obs4" => obs4_rejeu(recs),
        "lit" => topo_lit(recs, false),
        "lit_mut" => topo_lit(recs, true),
        _ => vec![ligne(
            Verdict::Skip,
            "TOPO-ANALYSE",
            format!("ensemble inconnu « {ens} »"),
        )],
    }
}

// ---------------------------------------------------------------------------------------------
// Avenant v5.WP19 (tranche 4, 28/10/2026) : TOPO-Q-V5 au temps 5·t₀, TOPO-LIT-V5 sur ⟨Q_L²⟩ d'AT21,
// mutants TOPO-Q-MUT-V5, TOPO-LIT-MUT-V5 (β × 1,05) et TOPO-LIT-MUT-OP (charge améliorée), intégrité
// de mesure (règle 15). Les fonctions v4.3 ci-dessus ne sont pas modifiées.

/// Indice de 5·t₀ dans `MULT_T0` (v5.WP19 : temps de TOPO-Q-V5).
pub const IDX_5T0: usize = 3;
/// AT21, table_QQ_SU2 (TeX l. 5469), β = 2,427, réseau 16⁴ : ⟨Q_L²⟩ = 6,635(56).
pub const AT21_QL2: (f64, f64) = (6.635, 0.056);
/// AT21, même ligne : ⟨Q_I²⟩ = 8,066(64) (INFO seulement en v5).
pub const AT21_QI2: (f64, f64) = (8.066, 0.064);
/// Plafond de puissance de TOPO-LIT-V5 et de ses mutants.
pub const SIGMA_MAX_V5: f64 = 0.28;
/// Écart maximal de la plaquette d'une configuration à la médiane de son ensemble (intégrité, règle 15).
pub const ECART_PLAQ_V5: f64 = 0.005;

/// Rôle d'une analyse TOPO-LIT-V5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoleLitV5 {
    /// `lit5` : Q_L = trèfle après 20 balayages.
    Nominal,
    /// `lit5_mut` (β × 1,05, 25 configurations par chaîne) : même statistique.
    MutantBeta,
    /// `lit5` : charge améliorée non arrondie à la place du trèfle.
    MutantOp,
}

impl RoleLitV5 {
    fn id(self) -> &'static str {
        match self {
            Self::Nominal => "TOPO-LIT-V5",
            Self::MutantBeta => "TOPO-LIT-MUT-V5",
            Self::MutantOp => "TOPO-LIT-MUT-OP",
        }
    }
    fn par_chaine(self) -> usize {
        match self {
            Self::MutantBeta => 25,
            _ => 100,
        }
    }
}

fn finies(x: &[f64], n: usize) -> bool {
    x.len() == n && x.iter().all(|v| v.is_finite())
}

/// Regroupe par chaîne, trie par cfg ; Err si les chaînes ne sont pas exactement 1..=n_chaines avec
/// les configurations 1..=par_chaine, chacune une fois.
fn chaines_exactes(
    recs: &[RecordTopo],
    n_chaines: usize,
    par_chaine: usize,
) -> Result<BTreeMap<usize, Vec<&RecordTopo>>, String> {
    let mut chaines: BTreeMap<usize, Vec<&RecordTopo>> = BTreeMap::new();
    for r in recs {
        chaines.entry(r.chaine).or_default().push(r);
    }
    if chaines.len() != n_chaines || chaines.keys().copied().ne(1..=n_chaines) {
        return Err(format!(
            "{n_chaines} chaînes numérotées 1…{n_chaines} requises, {} reçues ({} configurations)",
            chaines.len(),
            recs.len()
        ));
    }
    for (id, ch) in chaines.iter_mut() {
        ch.sort_by_key(|r| r.cfg);
        if ch.len() != par_chaine || ch.iter().map(|r| r.cfg).ne(1..=par_chaine) {
            return Err(format!(
                "chaîne {id} : configurations 1…{par_chaine} requises une fois chacune, {} reçues",
                ch.len()
            ));
        }
    }
    Ok(chaines)
}

/// Provenance statique : le mutant d'opérateur conserve les records de `lit5`.
fn provenance_v5(r: &RecordTopo, ens: &str) -> Result<(), String> {
    let backend = if r.chaine <= if ens == "q5" { 2 } else { 16 } {
        "hip"
    } else {
        "cuda"
    };
    let motif = if r.ens != ens {
        Some(format!("ens={} au lieu de {ens}", r.ens))
    } else if r.l != 16 {
        Some(format!("L={} au lieu de 16", r.l))
    } else if r.backend_gen != backend {
        Some(format!(
            "backend_gen={} au lieu de {backend}",
            r.backend_gen
        ))
    } else if r.backend_mesure != r.backend_gen {
        Some(format!(
            "backend_mesure={} différent de backend_gen={}",
            r.backend_mesure, r.backend_gen
        ))
    } else {
        None
    };
    match motif {
        Some(m) => Err(format!(
            "chaîne {} cfg {} : provenance : {m}",
            r.chaine, r.cfg
        )),
        None => Ok(()),
    }
}

/// Finitude commune, y compris les champs optionnels ; le flot vide de lit5 est permis.
fn valeurs_finies_v5(r: &RecordTopo) -> bool {
    r.plaq0.is_finite()
        && r.t0.is_none_or(|t| t.is_finite() && t > 0.0)
        && r.flot_a2
            .e_plaq
            .iter()
            .chain(&r.flot_a2.q_imp)
            .chain(&r.flot_a2.q_tre)
            .all(|v| v.is_finite())
        && r.flot_t0.as_ref().is_none_or(|f| {
            f.t_eff
                .iter()
                .chain(&f.q_imp)
                .chain(&f.q_tre)
                .all(|v| v.is_finite())
        })
        && [&r.cooling.action, &r.cooling.q_imp, &r.cooling.q_tre]
            .iter()
            .all(|v| v.len() >= 21 && v.iter().all(|x| x.is_finite()))
}

/// Validité d'un record de `q5` : grille v4.3.1 §B, dt = 0,01 et tolérance 10⁻⁹.
fn record_q5_valide(r: &RecordTopo) -> Result<(), String> {
    let lieu = format!("chaîne {} cfg {}", r.chaine, r.cfg);
    if !valeurs_finies_v5(r)
        || !finies(&r.flot_a2.q_imp, 4)
        || !finies(&r.flot_a2.q_tre, 4)
        || !finies(&r.flot_a2.e_plaq, 4)
    {
        return Err(format!(
            "{lieu} : valeurs absentes ou non finies (t₀ > 0, flot t/a² à 4 valeurs, cooling ≥ 21)"
        ));
    }
    match (r.t0, r.flot_t0.as_ref()) {
        (None, None) => Ok(()),
        (Some(t0), Some(f)) if finies(&f.q_imp, 4) && finies(&f.q_tre, 4) && finies(&f.t_eff, 4)
            && f.t_eff.iter().zip(MULT_T0).all(|(&t, k)| {
                let attendu = 0.01 * (k * t0 / 0.01 - 1e-9).ceil();
                attendu.is_finite() && (t - attendu).abs() <= 1e-9
            })
            && f.t_eff.windows(2).all(|w| matches!(w, [a, b] if a < b)) => Ok(()),
        _ => Err(format!("{lieu} : t₀ et flot_t0 incohérents (présence, 4 valeurs, grille dt=0.01 à 1e-9 ou temps strictement croissants)")),
    }
}

fn hors_plafond_v5(r: &RecordTopo) -> bool {
    r.t0.is_some()
        && r.flot_t0
            .as_ref()
            .and_then(|f| fini(&f.t_eff, IDX_5T0))
            .is_some_and(|t| t > 50.0 + 1e-9)
}

/// Charge améliorée à 5·t₀, seulement avec t₀ et dans le plafond de 5 000 pas.
fn q_5t0(r: &RecordTopo) -> Option<f64> {
    if r.t0.is_none() || hors_plafond_v5(r) {
        return None;
    }
    r.flot_t0.as_ref().and_then(|f| fini(&f.q_imp, IDX_5T0))
}

/// Recertification sur les moyennes de chaîne de génération, indépendamment de l'opérateur mutant.
/// q5 reste informatif ; lit5 et lit5_mut ont une gate à seuil strict 3,5.
fn recert_gen_v5(ens: &str, chaines: Option<&BTreeMap<usize, Vec<&RecordTopo>>>) -> LigneGate {
    let id = if ens == "q5" {
        "TOPO-Q-V5-RECERT-GEN-INFO".into()
    } else {
        format!("RECERT-GEN-{ens}")
    };
    let indisponible = |motif: &str| {
        ligne(
            if ens == "q5" {
                Verdict::Info
            } else {
                Verdict::Skip
            },
            &id,
            motif,
        )
    };
    let Some(chaines) = chaines else {
        return indisponible("non disponible ; ensemble invalide");
    };
    let mut donnees: BTreeMap<&str, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for ch in chaines.values() {
        let Some(premier) = ch.first() else {
            return indisponible("non disponible ; chaîne vide");
        };
        let p = ch.iter().map(|r| r.plaq0).sum::<f64>() / ch.len() as f64;
        let q = ch
            .iter()
            .map(|r| fini(&r.cooling.q_tre, 20).unwrap_or(f64::NAN).powi(2))
            .sum::<f64>()
            / ch.len() as f64;
        let (ps, qs) = donnees.entry(&premier.backend_gen).or_default();
        ps.push(p);
        qs.push(q);
    }
    let (Some((ph, qh)), Some((pc, qc))) = (donnees.get("hip"), donnees.get("cuda")) else {
        return indisponible("non disponible ; backend absent, moins de 2 chaînes");
    };
    if ph.len() < 2 || pc.len() < 2 {
        return indisponible("non disponible ; backend à moins de 2 chaînes");
    }
    let mut details = Vec::new();
    let mut passe = true;
    for (nom, hip, cuda) in [("plaq0", ph, pc), ("Q_L²", qh, qc)] {
        let (mh, sh) = jackknife_chaines(hip);
        let (mc, sc) = jackknife_chaines(cuda);
        let z = (mc - mh) / sh.hypot(sc);
        passe &= z.abs() < 3.5;
        details.push(format!("{nom} HIP={mh:.6} ± {sh:.6e} ({} chaînes), CUDA={mc:.6} ± {sc:.6e} ({} chaînes), z={z:.6}", hip.len(), cuda.len()));
    }
    ligne(
        if ens == "q5" {
            Verdict::Info
        } else if passe {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        &id,
        format!(
            "{} ; |z| < 3.5 pour les deux observables",
            details.join(" ; ")
        ),
    )
}

fn cool_v4_3(recs: &[RecordTopo], id: &str) -> LigneGate {
    let mut tries: Vec<_> = recs.iter().collect();
    tries.sort_by_key(|r| (r.chaine, r.cfg));
    let premiers: Vec<_> = tries.iter().take(10).collect();
    let complet = premiers.iter().all(|r| {
        r.cooling.action.len() >= 21
            && r.cooling
                .action
                .get(..21)
                .unwrap_or(&[])
                .iter()
                .all(|s| s.is_finite())
    });
    let monotones = premiers
        .iter()
        .filter(|r| {
            r.cooling.action.len() >= 21
                && r.cooling
                    .action
                    .get(..21)
                    .unwrap_or(&[])
                    .windows(2)
                    .all(|s| matches!(s, [a, b] if b - a <= 1e-12 * a.abs()))
        })
        .count();
    if premiers.len() < 10 || !complet {
        ligne(
            Verdict::Skip,
            id,
            "10 configurations et 21 actions finies par configuration requises",
        )
    } else {
        ligne(
            if monotones == 10 { Verdict::Pass } else { Verdict::Fail },
            id,
            format!("{monotones}/10 actions non croissantes (tri chaîne,cfg), tolérance 1e-12·|action[k]|"),
        )
    }
}

fn info_t0_v5(recs: &[RecordTopo]) -> LigneGate {
    let sans_t0: Vec<_> = recs
        .iter()
        .filter(|r| r.t0.is_none())
        .map(|r| (r.chaine, r.cfg))
        .collect();
    let t0s: Vec<f64> = recs
        .iter()
        .filter_map(|r| r.t0)
        .filter(|t| t.is_finite())
        .collect();
    let resume_t0 = if t0s.is_empty() {
        "aucun t₀".to_string()
    } else {
        format!(
            "moyenne={:.6}, min={:.6}, max={:.6}",
            moyenne(&t0s),
            t0s.iter().copied().fold(f64::INFINITY, f64::min),
            t0s.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        )
    };
    ligne(
        Verdict::Info,
        "TOPO-Q-V5-INFO",
        format!(
            "t₀ trouvés={}/{} ; {resume_t0} ; sans t₀ (chaîne, cfg)={sans_t0:?}",
            t0s.len(),
            recs.len()
        ),
    )
}

/// TOPO-Q-V5, TOPO-Q-MUT-V5, TOPO-COOL-V5 et lignes INFO (ensemble `q5`, avenant v5.WP19 §v5.3).
pub fn topo_q_v5(recs: &[RecordTopo]) -> Vec<LigneGate> {
    let mut lignes = Vec::new();
    let provenance = recs.iter().try_for_each(|r| provenance_v5(r, "q5"));
    let provenance_invalide = provenance.is_err();
    let validite = provenance.and_then(|()| {
        chaines_exactes(recs, 4, 25).and_then(|_| recs.iter().try_for_each(record_q5_valide))
    });
    if let Err(motif) = &validite {
        return vec![
            ligne(
                Verdict::Skip,
                "TOPO-Q-V5",
                format!("donnée invalide : {motif}"),
            ),
            ligne(
                if provenance_invalide {
                    Verdict::Fail
                } else {
                    Verdict::Skip
                },
                "TOPO-Q-MUT-V5",
                format!(
                    "{}donnée invalide : {motif}",
                    if provenance_invalide {
                        "ECHEC_OUTIL non concluant ; "
                    } else {
                        ""
                    }
                ),
            ),
            cool_v4_3(recs, "TOPO-COOL-V5"),
            recert_gen_v5("q5", None),
            info_t0_v5(recs),
        ];
    }
    let n = recs
        .iter()
        .filter(|r| q_5t0(r).is_some_and(|q| quantifie(q, 0.1)))
        .count();
    let n_mut = recs
        .iter()
        .filter(|r| q_5t0(r).is_some_and(|q| quantifie(q + 0.5, 0.1)))
        .count();
    let hors_plafond: Vec<_> = recs
        .iter()
        .filter(|r| hors_plafond_v5(r))
        .map(|r| (r.chaine, r.cfg))
        .collect();
    let sans_t0: Vec<_> = recs
        .iter()
        .filter(|r| r.t0.is_none())
        .map(|r| (r.chaine, r.cfg))
        .collect();
    lignes.push(ligne(
        if n >= 90 { Verdict::Pass } else { Verdict::Fail },
        "TOPO-Q-V5",
        format!(
            "N={n}/100 à t=5·t₀ (t₀ par configuration) ; seuil 90/100 ; |Q-round(Q)| < 0.100000 ; configurations sans t₀ comptées non quantifiées : {} ; hors plafond comptées non quantifiées : {}",
            sans_t0.len(), hors_plafond.len()
        ),
    ));
    let rouge = n_mut < 90;
    lignes.push(ligne(
        if rouge { Verdict::Pass } else { Verdict::Fail },
        "TOPO-Q-MUT-V5",
        format!(
            "{}N′={n_mut}/100 ; Q′=Q(5·t₀)+0.500000 ; PASS = mutant rouge ; si TOPO-Q-V5 passe, N′ ≤ 10 par construction",
            if rouge { "" } else { "ECHEC_OUTIL mutant non rouge ; " }
        ),
    ));
    lignes.push(cool_v4_3(recs, "TOPO-COOL-V5"));
    let (bas, haut) = clopper_pearson(n as u64, recs.len() as u64, 0.95);
    lignes.push(ligne(
        Verdict::Info,
        "TOPO-Q-V5-INFO",
        format!(
            "Clopper–Pearson 95 % [{bas:.6}, {haut:.6}] sur {} configurations ; sous hypothèse iid, couverture non garantie (chaînes corrélées)",
            recs.len()
        ),
    ));
    let mut chaines: BTreeMap<usize, Vec<&RecordTopo>> = BTreeMap::new();
    for r in recs {
        chaines.entry(r.chaine).or_default().push(r);
    }
    for ch in chaines.values_mut() {
        ch.sort_by_key(|r| r.cfg);
    }
    lignes.push(recert_gen_v5(
        "q5",
        (integrite_v5("q5", recs).verdict == Verdict::Pass).then_some(&chaines),
    ));
    lignes.push(ligne(
        Verdict::Info,
        "TOPO-Q-V5-HORS-PLAFOND",
        format!(
            "hors plafond (> 50 + 1e-9) : {} ; (chaîne, cfg)={hors_plafond:?}",
            hors_plafond.len()
        ),
    ));
    let admissibles: Vec<_> = recs.iter().filter(|r| q_5t0(r).is_some()).collect();
    let changements = admissibles
        .iter()
        .filter(|r| {
            r.flot_t0.as_ref().is_some_and(|f| {
                fini(&f.q_imp, 2).map(f64::round) != fini(&f.q_imp, 3).map(f64::round)
            })
        })
        .count();
    lignes.push(ligne(
        Verdict::Info,
        "TOPO-Q-V5-STABILITE",
        format!(
            "round(Q(3·t₀)) ≠ round(Q(5·t₀)) : {changements}/{} (t₀ présent, dans le plafond)",
            admissibles.len()
        ),
    ));
    for (i, k) in MULT_T0.iter().enumerate() {
        let valeurs: Vec<_> = admissibles
            .iter()
            .filter_map(|r| r.flot_t0.as_ref().and_then(|f| fini(&f.q_imp, i)))
            .map(|q| q.round().powi(2))
            .collect();
        lignes.push(ligne(
            Verdict::Info,
            "TOPO-Q-V5-Q2",
            format!(
                "k={k:.0} ; ⟨round(Q(k·t₀))²⟩={:.6} ; N={} (t₀ présent, dans le plafond)",
                moyenne(&valeurs),
                valeurs.len()
            ),
        ));
    }
    for (id, ch) in &chaines {
        let k = ch
            .iter()
            .filter(|r| q_5t0(r).is_some_and(|q| quantifie(q, 0.1)))
            .count();
        let absents = ch.iter().filter(|r| r.t0.is_none()).count();
        let hors = ch.iter().filter(|r| hors_plafond_v5(r)).count();
        lignes.push(ligne(
            Verdict::Info,
            "TOPO-Q-V5-INFO",
            format!(
                "chaîne={id} quantifiées à 5·t₀={k}/{} ; sans t₀={absents} ; hors plafond={hors}",
                ch.len()
            ),
        ));
    }
    lignes.push(info_t0_v5(recs));
    for (i, t) in TEMPS_A2.iter().enumerate() {
        let (imp, ni) = compte(recs.iter().map(|r| fini(&r.flot_a2.q_imp, i)));
        let (tre, nt) = compte(recs.iter().map(|r| fini(&r.flot_a2.q_tre, i)));
        lignes.push(ligne(
            Verdict::Info,
            "TOPO-Q-V5-T",
            format!("t/a²={t:.6} amélioré={imp}/{ni} trèfle={tre}/{nt}"),
        ));
    }
    for (i, k) in MULT_T0.iter().enumerate() {
        let (imp, ni) = compte(
            admissibles
                .iter()
                .map(|r| r.flot_t0.as_ref().and_then(|f| fini(&f.q_imp, i))),
        );
        let (tre, nt) = compte(
            admissibles
                .iter()
                .map(|r| r.flot_t0.as_ref().and_then(|f| fini(&f.q_tre, i))),
        );
        lignes.push(ligne(
            Verdict::Info,
            "TOPO-Q-V5-T",
            format!(
                "t={k:.6}·t₀ amélioré={imp}/{ni} trèfle={tre}/{nt} (configurations à t₀ trouvé dans le plafond)"
            ),
        ));
    }
    let (cool, nc) = compte(recs.iter().map(|r| fini(&r.cooling.q_imp, 20)));
    lignes.push(ligne(
        Verdict::Info,
        "TOPO-Q-V5-INFO",
        format!("cooling 20 balayages : amélioré quantifié (< 0,1)={cool}/{nc}"),
    ));
    let residus: Vec<f64> = recs
        .iter()
        .filter_map(q_5t0)
        .filter(|q| q.is_finite())
        .map(|q| q - q.round())
        .collect();
    lignes.push(ligne(
        Verdict::Info,
        "TOPO-Q-V5-INFO",
        format!(
            "histogramme Q(5·t₀)-round(Q) ; {}",
            histogramme(&residus, -0.5, 0.5, 0.05)
        ),
    ));
    lignes
}

/// Moyennes de chaîne (jackknife v4.3.1 §C) d'une grandeur par configuration.
fn moyennes_chaines(
    chaines: &BTreeMap<usize, Vec<&RecordTopo>>,
    f: impl Fn(&RecordTopo) -> f64,
) -> Vec<f64> {
    chaines
        .values()
        .map(|ch| ch.iter().map(|r| f(r)).sum::<f64>() / ch.len() as f64)
        .collect()
}

/// Les diagnostics à 10 et 15 balayages ne disparaissent jamais silencieusement.
fn info_balayages_v5(
    chaines: Option<&BTreeMap<usize, Vec<&RecordTopo>>>,
    info: &str,
) -> Vec<LigneGate> {
    [10, 15]
        .into_iter()
        .map(|k| {
            let valeur = chaines.and_then(|ch| {
                if !ch
                    .values()
                    .flatten()
                    .all(|r| fini(&r.cooling.q_tre, k).is_some_and(|q| q.powi(2).is_finite()))
                {
                    return None;
                }
                let (x, s) = jackknife_chaines(&moyennes_chaines(ch, |r| {
                    fini(&r.cooling.q_tre, k).unwrap_or(f64::NAN).powi(2)
                }));
                (x.is_finite() && s.is_finite()).then_some((x, s))
            });
            ligne(
                Verdict::Info,
                info,
                match valeur {
                    Some((x, s)) => format!("⟨Q_L²⟩ après {k} balayages={x:.6} ± {s:.6}"),
                    None => format!("⟨Q_L²⟩ après {k} balayages : non disponible"),
                },
            )
        })
        .collect()
}

/// TOPO-LIT-V5 (`lit5`), TOPO-LIT-MUT-V5 (`lit5_mut`) ou TOPO-LIT-MUT-OP (`lit5`), avenant v5.WP19 §v5.3.
pub fn topo_lit_v5(recs: &[RecordTopo], role: RoleLitV5) -> Vec<LigneGate> {
    let id = role.id();
    let mutant = role != RoleLitV5::Nominal;
    let invalide = |motif: String| {
        if mutant {
            ligne(
                Verdict::Fail,
                id,
                format!("ECHEC_OUTIL non concluant ; donnée invalide : {motif}"),
            )
        } else {
            ligne(Verdict::Skip, id, format!("donnée invalide : {motif}"))
        }
    };
    let ens = if role == RoleLitV5::MutantBeta {
        "lit5_mut"
    } else {
        "lit5"
    };
    let info = if mutant {
        "TOPO-LIT-MUT-V5-INFO"
    } else {
        "TOPO-LIT-V5-INFO"
    };
    let validite = chaines_exactes(recs, 32, role.par_chaine()).and_then(|ch| {
        for r in recs {
            provenance_v5(r, ens)?;
            if !valeurs_finies_v5(r)
                || [&r.cooling.q_tre, &r.cooling.q_imp]
                    .iter()
                    .any(|v| !fini(v, 20).is_some_and(|q| (1.2 * q).powi(2).is_finite()))
            {
                return Err(format!(
                    "chaîne {} cfg {} : valeurs absentes, non finies ou carrés non représentables",
                    r.chaine, r.cfg
                ));
            }
        }
        Ok(ch)
    });
    let chaines = match validite {
        Ok(c) => c,
        Err(motif) => {
            let mut lignes = vec![invalide(motif)];
            if role != RoleLitV5::MutantOp {
                lignes.push(recert_gen_v5(ens, None));
                let ch = chaines_exactes(recs, 32, role.par_chaine()).ok();
                lignes.extend(info_balayages_v5(ch.as_ref(), info));
            }
            return lignes;
        }
    };
    let recert = recert_gen_v5(
        ens,
        (integrite_v5(ens, recs).verdict == Verdict::Pass).then_some(&chaines),
    );
    // Tout calcul sensible à l'ordre, y compris attribution(s), reçoit (chaîne,cfg).
    let tries: Vec<_> = chaines.values().flatten().copied().collect();
    let recs = &tries;
    let charge = |r: &RecordTopo| -> f64 {
        fini(
            if role == RoleLitV5::MutantOp {
                &r.cooling.q_imp
            } else {
                &r.cooling.q_tre
            },
            20,
        )
        .unwrap_or(f64::NAN)
    };
    let m = moyennes_chaines(&chaines, |r| charge(r).powi(2));
    let (x, sigma) = jackknife_chaines(&m);
    let non_constantes = chaines
        .values()
        .filter(|ch| {
            ch.iter().any(|r| {
                charge(r).powi(2) != ch.first().map(|r| charge(r).powi(2)).unwrap_or(f64::NAN)
            })
        })
        .count();
    let (reference, sref) = AT21_QL2;
    let z = (x - reference) / sigma.hypot(sref);
    let motif = if !sigma.is_finite() || sigma <= 0.0 || non_constantes < 8 {
        Some(format!("(b) erreur dégénérée ou moins de 8 chaînes non constantes ; σ={sigma:.6}, non constantes={non_constantes}"))
    } else if sigma > SIGMA_MAX_V5 {
        Some(format!(
            "puissance insuffisante (définitif, aucune extension en v5) ; σ={sigma:.6} > {SIGMA_MAX_V5:.6}"
        ))
    } else {
        None
    };
    let quantite = if role == RoleLitV5::MutantOp {
        "Q = charge améliorée non arrondie (mutant d'opérateur)"
    } else {
        "Q_L = trèfle non arrondi"
    };
    let detail = format!(
        "X̂={x:.6}, σ={sigma:.6} ; {quantite}, cooling 20 ; référence AT21 β=2.427000 L=16 ⟨Q_L²⟩ : {reference:.6} ± {sref:.6} ; {} configurations ; z={z:.6} ; quantile Student ν=31 : {QUANTILE_T31:.6}",
        recs.len()
    );
    // Frontières strictes : égalité au quantile ⇒ FAIL pour le nominal ET les mutants.
    let mut lignes = vec![match (motif, mutant) {
        (Some(motif), false) => ligne(Verdict::Skip, id, format!("{motif} ; {detail}")),
        (Some(motif), true) => ligne(
            Verdict::Fail,
            id,
            format!("ECHEC_OUTIL non concluant ; {motif} ; {detail}"),
        ),
        (None, false) => ligne(
            if z.abs() < QUANTILE_T31 {
                Verdict::Pass
            } else {
                Verdict::Fail
            },
            id,
            detail,
        ),
        (None, true) => {
            let rouge = z.abs() > QUANTILE_T31;
            ligne(
                if rouge { Verdict::Pass } else { Verdict::Fail },
                id,
                format!(
                    "{}{detail}",
                    if rouge {
                        "mutant rouge ; "
                    } else {
                        "ECHEC_OUTIL mutant non rouge ; "
                    }
                ),
            )
        }
    }];
    if role == RoleLitV5::MutantOp {
        return lignes;
    }
    lignes.push(recert);
    let info = if mutant {
        "TOPO-LIT-MUT-V5-INFO"
    } else {
        "TOPO-LIT-V5-INFO"
    };
    let n = recs.len() as f64;
    lignes.push(ligne(
        Verdict::Info,
        info,
        format!(
            "X̂_L={x:.6}, V=65536, χ_L={:.6e}, ⟨Q_L⟩={:.6}",
            x / 65536.0,
            recs.iter()
                .map(|r| fini(&r.cooling.q_tre, 20).unwrap_or(f64::NAN))
                .sum::<f64>()
                / n
        ),
    ));
    lignes.extend(info_balayages_v5(Some(&chaines), info));
    if mutant {
        return lignes;
    }
    let mut taus = Vec::new();
    let mut non_estimables = Vec::new();
    for (&cid, ch) in &chaines {
        let serie: Vec<f64> = ch
            .iter()
            .map(|r| fini(&r.cooling.q_tre, 20).unwrap_or(f64::NAN).powi(2))
            .collect();
        match tau_int_sokal(&serie, 6.0) {
            Some(t) => taus.push(t),
            None => non_estimables.push(cid),
        }
    }
    taus.sort_by(f64::total_cmp);
    let resume_tau = if taus.is_empty() {
        "médiane=non estimable, max=non estimable".into()
    } else {
        format!(
            "médiane={:.6}, max={:.6}",
            (taus.get((taus.len() - 1) / 2).copied().unwrap_or(f64::NAN)
                + taus.get(taus.len() / 2).copied().unwrap_or(f64::NAN))
                / 2.0,
            taus.last().copied().unwrap_or(f64::NAN)
        )
    };
    lignes.push(ligne(
        Verdict::Info,
        info,
        format!(
            "τ_int(Q_L²), Sokal c=6.000000 : {resume_tau} ; estimables={}/32 ; non estimable pour chaînes={non_estimables:?}",
            taus.len()
        ),
    ));
    if recs.iter().all(|r| r.plaq0.is_finite()) {
        let (p, e) = jackknife_chaines(&moyennes_chaines(&chaines, |r| r.plaq0));
        let delta = p - 0.6364293;
        lignes.push(ligne(Verdict::Info, "TOPO-LIT-V5-PLAQ", format!("⟨½Re Tr U_p⟩={p:.6}, σ_jackknife={e:.6e} ; référence AT21 0.6364293(15), réseau 20³×16 ≠ 16⁴ ; écart={delta:.6e}, écart en σ combinés={:.6} ; sans verdict, réseau différent", delta / e.hypot(0.0000015))));
    }
    // Attributions entières (INFO, aucune n'est un verdict ; l'algorithme d'AT21 n'est pas publié).
    let (ri, si) = AT21_QI2;
    let (xi, sxi) = jackknife_chaines(&moyennes_chaines(&chaines, |r| {
        fini(&r.cooling.q_imp, 20)
            .unwrap_or(f64::NAN)
            .round()
            .powi(2)
    }));
    let ambigus = recs
        .iter()
        .filter(|r| !quantifie(fini(&r.cooling.q_imp, 20).unwrap_or(f64::NAN), 0.25))
        .count();
    lignes.push(ligne(Verdict::Info, "TOPO-LIT-V5-ATTRIB", format!("attribution v4.3 round(Q amélioré) : X̂_I={xi:.6} ± {sxi:.6} ; hors |Q-round(Q)| < 0.25 : {ambigus}/{} ; z contre AT21 ⟨Q_I²⟩ {ri:.6} ± {si:.6} : {:.6} (sans verdict)", recs.len(), (xi - ri) / sxi.hypot(si))));
    let (xr, sxr) = jackknife_chaines(&moyennes_chaines(&chaines, |r| {
        fini(&r.cooling.q_tre, 20)
            .unwrap_or(f64::NAN)
            .round()
            .powi(2)
    }));
    lignes.push(ligne(Verdict::Info, "TOPO-LIT-V5-ATTRIB", format!("round(Q_L) (Smith–Teper 1998, première définition) : {xr:.6} ± {sxr:.6} ; z contre {ri:.6} ± {si:.6} : {:.6} (sans verdict)", (xr - ri) / sxr.hypot(si))));
    let trefle: Vec<f64> = recs
        .iter()
        .map(|r| fini(&r.cooling.q_tre, 20).unwrap_or(f64::NAN))
        .collect();
    let s = attribution(&trefle);
    let (xs, sxs) = jackknife_chaines(&moyennes_chaines(&chaines, |r| {
        (s * fini(&r.cooling.q_tre, 20).unwrap_or(f64::NAN))
            .round()
            .powi(2)
    }));
    lignes.push(ligne(Verdict::Info, "TOPO-LIT-V5-ATTRIB", format!("round(s·Q_L), s={s:.6} (grille de 401 valeurs de v4.3) : {xs:.6} ± {sxs:.6} ; z contre {ri:.6} ± {si:.6} : {:.6} (sans verdict)", (xs - ri) / sxs.hypot(si))));
    lignes
}

/// Intégrité de mesure (règle 15, avenant v5.WP19 §v5.2) : charges et actions finies, action de cooling non croissante
/// pour CHAQUE configuration, plaquette à moins de `ECART_PLAQ_V5` de la médiane de l'ensemble.
pub fn integrite_v5(ens: &str, recs: &[RecordTopo]) -> LigneGate {
    let id = format!("TOPO-V5-INTEGRITE-{ens}");
    if recs.is_empty() {
        return ligne(Verdict::Fail, &id, "aucune configuration");
    }
    let mut defauts = Vec::new();
    let mut plaq: Vec<f64> = recs.iter().map(|r| r.plaq0).collect();
    if plaq.iter().any(|p| !p.is_finite()) {
        defauts.push("plaquette non finie".to_string());
    }
    plaq.sort_by(f64::total_cmp);
    let mediane = plaq.get((plaq.len() - 1) / 2).copied().unwrap_or(f64::NAN) / 2.0
        + plaq.get(plaq.len() / 2).copied().unwrap_or(f64::NAN) / 2.0;
    for r in recs {
        let lieu = format!("({},{})", r.chaine, r.cfg);
        let c = &r.cooling;
        if !valeurs_finies_v5(r) {
            defauts.push(format!(
                "{lieu} valeurs absentes ou non finies, ou t₀ non positif"
            ));
            continue;
        }
        if !c
            .action
            .windows(2)
            .all(|s| matches!(s, [a, b] if b - a <= 1e-12 * a.abs()))
        {
            defauts.push(format!("{lieu} action de cooling croissante"));
        }
        if (r.plaq0 - mediane).abs() >= ECART_PLAQ_V5 {
            defauts.push(format!(
                "{lieu} plaquette {:.6} à ≥ {ECART_PLAQ_V5} de la médiane {mediane:.6}",
                r.plaq0
            ));
        }
    }
    if defauts.is_empty() {
        ligne(Verdict::Pass, &id, format!("{} configurations : valeurs finies, cooling non croissant (1e-12·|S|), plaquettes à < {ECART_PLAQ_V5} de la médiane {mediane:.6}", recs.len()))
    } else {
        let n = defauts.len();
        defauts.truncate(20);
        ligne(
            Verdict::Fail,
            &id,
            format!("{n} défaut(s) : {}", defauts.join(" ; ")),
        )
    }
}

/// Analyse v5.WP19 d'un ensemble : `q5`, `lit5` (TOPO-LIT-V5 puis TOPO-LIT-MUT-OP) ou `lit5_mut`.
pub fn analyser_v5(ens: &str, recs: &[RecordTopo]) -> Vec<LigneGate> {
    let integrite = integrite_v5(ens, recs);
    let echec_integrite = integrite.verdict == Verdict::Fail;
    let mut gates = match ens {
        "q5" => topo_q_v5(recs),
        "lit5" => {
            let mut l = topo_lit_v5(recs, RoleLitV5::Nominal);
            l.extend(topo_lit_v5(recs, RoleLitV5::MutantOp));
            l
        }
        "lit5_mut" => topo_lit_v5(recs, RoleLitV5::MutantBeta),
        _ => {
            return vec![ligne(
                Verdict::Skip,
                "TOPO-ANALYSE-V5",
                format!("ensemble inconnu « {ens} »"),
            )]
        }
    };
    if ens != "q5" {
        gates.push(cool_v4_3(recs, "TOPO-COOL-V5"));
    }
    if echec_integrite {
        for l in &mut gates {
            match l.id.as_str() {
                "TOPO-Q-V5" | "TOPO-Q-MUT-V5" | "TOPO-LIT-V5" => {
                    l.verdict = Verdict::Skip;
                    l.detail = "intégrité en échec".into();
                }
                "TOPO-LIT-MUT-OP" | "TOPO-LIT-MUT-V5" => {
                    l.verdict = Verdict::Fail;
                    l.detail = "ECHEC_OUTIL non concluant ; intégrité en échec".into();
                }
                _ if l.id.starts_with("RECERT-GEN-") => {
                    l.verdict = Verdict::Skip;
                    l.detail = "ensemble invalide ; intégrité en échec".into();
                }
                _ => {}
            }
        }
    }
    let mut lignes = vec![integrite];
    lignes.extend(gates);
    lignes
}
