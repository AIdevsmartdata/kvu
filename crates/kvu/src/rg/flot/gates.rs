//! Gates RG-* des estimateurs de couplage effectif (GATES v1 §4, v3 §v3.3 ; journal « WP8 GEL des protocoles »). Propriétaire : WP8.
//!
//! Une ligne `PASS|FAIL|SKIP|INFO <ID> <détail>` par gate ; les mutants rendent leur propre ligne (PASS = le mutant est bien ROUGE ; un mutant
//! vert = FAIL `ECHEC_OUTIL`). Seuils, graines et protocoles = ceux de GATES.md et du journal ; un changement ici sans nouvelle version est un défaut.

pub use super::etalonnage_gates::survie_chi2;

use super::cellule::{
    beta_sd_cellule, decouvrir, flot, lire_courbes, traiter_chaine, DonneesChaine, Estimateurs, ParamFlot, Schema, C_SWENDSEN,
};
use super::etalonner::{courbe_depuis, depuis_json, ChaineMesuree, ParamEtalonnage, GRILLE_GELEE};
use super::jackknife::jackknife_groupes;
use super::mcrg2::plaquette_et_rectangle;
use super::plaq::{Courbe, Inversion, Observable};
use super::sd::{beta_sd, stats_sd, ModeRapport, RapportBessel};
use super::tau::{stat_serie, tau_int};
use crate::commun::{Erreur, Resultat};
use crate::groupes::gates::{LigneGate, Verdict};
use super::ape::{bloquer_script, SchemaScript};
use crate::rg::swendsen::{bloquer_b2, plaquette_moyenne};
use crate::stockage::npy::lire_champ;
use serde_json::Value;
use std::path::{Path, PathBuf};

// ------------------------------------------------------------------------------------------
// Constantes gelées
// ------------------------------------------------------------------------------------------

pub const SEUIL_SD_BIAIS: f64 = 0.005;
pub const SEUIL_SD_SIGMA: f64 = 0.0025;
pub const BETAS_SD: [f64; 2] = [2.3, 8.0];
pub const SEUIL_Z_PLAQ: f64 = 3.0;
pub const SEUIL_P_CHI2: f64 = 0.01;
pub const SEUIL_N3: f64 = 1e-10;
pub const SEUIL_TAU: f64 = 1e-10;
pub const SEUIL_BESSEL: f64 = 1e-14;
pub const FACTEUR_MUTANT_PLAQ: f64 = 1.05;
const CHAINES_SD: usize = 3;
const CONFIGS_PAR_CHAINE: usize = 20;

pub struct Contexte {
    /// Données de test versionnées (`tests/donnees/rg_flot`).
    pub donnees: PathBuf,
    /// Racine des ensembles grandL (`…/ensembles/production`) ; `None` ⇒ SKIP des gates qui en dépendent.
    pub grandl: Option<PathBuf>,
    /// Dossier des étalonnages JSON : `etal_L4.json`, `indep_L4.json`, `indep_mut_L4.json`, `etal_L6.json` (supplément).
    pub etalonnages: Option<PathBuf>,
    pub fils: usize,
    pub mutants: bool,
    pub seul: Vec<String>,
}

impl Contexte {
    fn veut(&self, id: &str) -> bool {
        self.seul.is_empty() || self.seul.iter().any(|p| id.starts_with(p.as_str()) || p.starts_with(id))
    }
    fn garde_ligne(&self, id: &str) -> bool {
        self.seul.is_empty() || self.seul.iter().any(|p| id.starts_with(p.as_str()))
    }
}

fn ligne(ok: bool, id: &str, detail: impl Into<String>) -> LigneGate {
    LigneGate::nouvelle(if ok { Verdict::Pass } else { Verdict::Fail }, id, detail)
}
fn skip(id: &str, detail: impl Into<String>) -> LigneGate {
    LigneGate::nouvelle(Verdict::Skip, id, detail)
}
fn info(id: &str, detail: impl Into<String>) -> LigneGate {
    LigneGate::nouvelle(Verdict::Info, id, detail)
}

fn json_de(chemin: &Path) -> Resultat<Value> {
    let t = std::fs::read_to_string(chemin).map_err(|e| Erreur::io(format!("lecture {}", chemin.display()), e))?;
    serde_json::from_str(&t).map_err(|e| Erreur::Format(format!("{} : {e}", chemin.display())))
}

// ------------------------------------------------------------------------------------------
// RG-BESSEL (instrument)
// ------------------------------------------------------------------------------------------

pub fn gate_bessel(ctx: &Contexte) -> Resultat<LigneGate> {
    let v = json_de(&ctx.donnees.join("bessel_ratios.json"))?;
    let pts = v["points"].as_array().ok_or_else(|| Erreur::Format("bessel_ratios.json".into()))?;
    let (mut m21, mut m10) = (0.0_f64, 0.0_f64);
    for e in pts {
        let (x, a, b) = (e["x"].as_f64().unwrap_or(f64::NAN), e["r21"].as_f64().unwrap_or(f64::NAN), e["r10"].as_f64().unwrap_or(f64::NAN));
        m21 = m21.max(((super::bessel::i2_sur_i1(x) - a) / a).abs());
        m10 = m10.max(((super::bessel::i1_sur_i0(x) - b) / b).abs());
    }
    Ok(ligne(
        m21 <= SEUIL_BESSEL && m10 <= SEUIL_BESSEL,
        "RG-BESSEL",
        format!("{} points x ∈ [1e-8, 1e3] contre mpmath (60 chiffres) : max relatif I2/I1 = {m21:.2e}, I1/I0 = {m10:.2e} (seuil {SEUIL_BESSEL:.0e})", pts.len()),
    ))
}

// ------------------------------------------------------------------------------------------
// RG-TAU
// ------------------------------------------------------------------------------------------

fn lire_serie(chemin: &Path) -> Resultat<Vec<f64>> {
    let t = std::fs::read_to_string(chemin).map_err(|e| Erreur::io(format!("lecture {}", chemin.display()), e))?;
    t.lines().filter(|l| !l.trim().is_empty()).map(|l| l.trim().parse::<f64>().map_err(|e| Erreur::Format(format!("{} : {e}", chemin.display())))).collect()
}

pub fn gate_tau(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let serie = lire_serie(&ctx.donnees.join("serie_plaquette_L12_b2.3_c1.txt"))?;
    let r = json_de(&ctx.donnees.join("tau_ref.json"))?;
    let (tau_ref, w_ref, dtau_ref) = (r["tau"].as_f64().unwrap_or(f64::NAN), r["W"].as_u64().unwrap_or(0) as usize, r["dtau"].as_f64().unwrap_or(f64::NAN));
    let s = stat_serie(&serie);
    let (e_tau, e_dtau) = ((s.tau - tau_ref).abs(), (s.dtau - dtau_ref).abs());
    let e_err = (s.err - r["stat"]["err"].as_f64().unwrap_or(f64::NAN)).abs();
    let mut v = vec![ligne(
        serie.len() == 2000 && e_tau <= SEUIL_TAU && e_dtau <= SEUIL_TAU && s.fenetre == w_ref && e_err <= SEUIL_TAU,
        "RG-TAU",
        format!(
            "L12_b2.3_c1, {} valeurs de plaquette : τ = {:.16} (analyse_plaquette.py {tau_ref:.16}, écart {e_tau:.2e}), W = {} ({w_ref}), dτ écart {e_dtau:.2e}, err écart {e_err:.2e} (seuil {SEUIL_TAU:.0e})",
            serie.len(), s.tau, s.fenetre
        ),
    )];
    if let Some(g) = &ctx.grandl {
        let direct = super::cellule::serie_plaquette(&g.join("L12_b2.3_c1"));
        match direct {
            Some(d) => v.push(ligne(d == serie, "RG-TAU-SOURCE", format!("la série de la fixture = la colonne `plaquette` (phase sample) de measurements.csv de grandL ({} valeurs) : {}", d.len(), d == serie))),
            None => v.push(skip("RG-TAU-SOURCE", "measurements.csv de grandL illisible")),
        }
    }
    if ctx.mutants {
        // mutant : fenêtre c = 3 au lieu de 6 ⇒ la valeur doit s'écarter du script
        let m = tau_int(&serie, 3.0);
        v.push(ligne((m.tau - tau_ref).abs() > SEUIL_TAU || m.fenetre != w_ref, "RG-TAU-MUT", format!("mutant c = 3 : τ = {:.6} (écart {:.2e}), W = {} : doit s'écarter", m.tau, (m.tau - tau_ref).abs(), m.fenetre)));
    }
    Ok(v)
}

// ------------------------------------------------------------------------------------------
// RG-N3
// ------------------------------------------------------------------------------------------

pub fn gate_n3(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let r = json_de(&ctx.donnees.join("n3_ref_ape_L12_b2.3_c1_cfg002000.json"))?;
    let (b0, p0) = (r["niveaux"]["0"]["beta_eff"].as_f64().unwrap_or(f64::NAN), r["niveaux"]["0"]["plaq"].as_f64().unwrap_or(f64::NAN));
    let (b1, p1) = (r["niveaux"]["1"]["beta_eff"].as_f64().unwrap_or(f64::NAN), r["niveaux"]["1"]["plaq"].as_f64().unwrap_or(f64::NAN));
    let ape = lire_champ(&ctx.donnees.join("n3_ape_niv1_L6.npy"))?;
    let sa = stats_sd(&ape)?;
    let ba = beta_sd(&sa, RapportBessel::I2SurI1, ModeRapport::TableauScript).unwrap_or(f64::NAN);
    let pa = plaquette_moyenne(&ape);
    let mut v = vec![ligne(
        (ba - b1).abs() <= SEUIL_N3 && (pa - p1).abs() <= SEUIL_N3,
        "RG-N3-FIXTURE",
        format!("sd (mode tableau-script) sur la configuration bloquée par le script (ape, c = 0,5, L = 6) : β_eff = {ba:.15} (script {b1:.15}, écart {:.2e}), P écart {:.2e} (seuil {SEUIL_N3:.0e})", (ba - b1).abs(), (pa - p1).abs()),
    )];
    let exact_a = beta_sd(&sa, RapportBessel::I2SurI1, ModeRapport::Exact).unwrap_or(f64::NAN);
    v.push(info("RG-N3-EXACT", format!("INFO mode exact (fraction continue) contre le script sur la même configuration : écart {:.3e} = erreur d'interpolation de la table du script (4000 nœuds, np.interp)", (exact_a - b1).abs())));
    if ctx.mutants {
        v.push(ligne((exact_a - b1).abs() > SEUIL_N3, "RG-N3-MUT", format!("mutant « exact au lieu de tableau-script » : écart {:.2e} doit dépasser {SEUIL_N3:.0e}", (exact_a - b1).abs())));
    }
    let Some(g) = &ctx.grandl else {
        v.push(skip("RG-N3", "ensembles grandL non fournis (--grandl) : chaîne complète Rust non rejouée"));
        return Ok(v);
    };
    let chemin = g.join("L12_b2.3_c1").join("cfg_002000.npy");
    if !chemin.is_file() {
        v.push(skip("RG-N3", format!("configuration absente : {}", chemin.display())));
        return Ok(v);
    }
    let cfg = lire_champ(&chemin)?;
    // niveau 0 : la configuration brute
    let s0 = stats_sd(&cfg)?;
    let c0 = beta_sd(&s0, RapportBessel::I2SurI1, ModeRapport::TableauScript).unwrap_or(f64::NAN);
    let q0 = plaquette_moyenne(&cfg);
    let q0b = plaquette_et_rectangle(&cfg)?.plaquette;
    // niveau 1 : blocage `ape` (c = 0,5) du script, porté en Rust ; le témoin de Swendsen de WP6 est un AUTRE schéma (écart rapporté)
    let blq = bloquer_script(&cfg, SchemaScript::Ape, C_SWENDSEN)?;
    let sw = bloquer_b2(&cfg, C_SWENDSEN)?;
    let ecart_swendsen = sw.q.iter().zip(&ape.q).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let b_sw = beta_sd(&stats_sd(&sw)?, RapportBessel::I2SurI1, ModeRapport::TableauScript).unwrap_or(f64::NAN);
    let s1 = stats_sd(&blq)?;
    let c1 = beta_sd(&s1, RapportBessel::I2SurI1, ModeRapport::TableauScript).unwrap_or(f64::NAN);
    let q1 = plaquette_moyenne(&blq);
    let ecart_liens = blq.q.iter().zip(&ape.q).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let ecarts = [(c0 - b0).abs(), (q0 - p0).abs(), (q0b - p0).abs(), (c1 - b1).abs(), (q1 - p1).abs()];
    let pire = ecarts.iter().cloned().fold(0.0, f64::max);
    v.push(ligne(
        pire <= SEUIL_N3 && blq.l == 6,
        "RG-N3",
        format!(
            "grandL L12_b2.3_c1/cfg_002000, sd en mode tableau-script contre n3_blocage_beta_eff.py (ape, c = 0,5) : niveau 0 β_eff {c0:.15} (script {b0:.15}, écart {:.2e}), P {:.2e} / {:.2e} ; niveau 1 (blocage ape porté en Rust) β_eff {c1:.15} (script {b1:.15}, écart {:.2e}), P écart {:.2e} ; pire écart {pire:.2e} (seuil {SEUIL_N3:.0e}) ; max|Δq| des liens ape Rust contre blocage du script {ecart_liens:.2e} (INFO) ; témoin de Swendsen de WP6 (autre schéma) : max|Δq| {ecart_swendsen:.2e}, β_eff {b_sw:.6} (INFO, RG-SCHEMA)",
            ecarts[0], ecarts[1], ecarts[2], ecarts[3], ecarts[4]
        ),
    ));
    let (e0, _) = (beta_sd(&s0, RapportBessel::I2SurI1, ModeRapport::Exact).unwrap_or(f64::NAN), 0);
    v.push(info("RG-N3-EXACT-L12", format!("INFO mode exact contre script : niveau 0 écart {:.3e}", (e0 - b0).abs())));
    Ok(v)
}

// ------------------------------------------------------------------------------------------
// RG-SD
// ------------------------------------------------------------------------------------------

fn charger_cellule(g: &Path, beta: f64, fils: usize) -> Resultat<Vec<DonneesChaine>> {
    let motif = format!("L12_b{}_c*", if beta == beta.trunc() { format!("{}", beta as i64) } else { format!("{beta}") });
    let chaines = decouvrir(g, Some(&motif))?;
    chaines.iter().map(|c| traiter_chaine(c, Schema::Balaban, 0, fils)).collect()
}

fn verdict_sd(rapport: RapportBessel, beta: f64, donnees: &[DonneesChaine]) -> (bool, f64, f64, String) {
    let b = beta_sd_cellule(&donnees.iter().collect::<Vec<_>>(), rapport, ModeRapport::Exact);
    let jk = jackknife_groupes(donnees, |g| beta_sd_cellule(g, rapport, ModeRapport::Exact));
    match (b, jk) {
        (Some(b), Some(jk)) => {
            let (biais, rel) = ((b / beta - 1.0).abs(), jk.erreur / beta);
            let par_chaine: Vec<String> = donnees
                .iter()
                .map(|d| format!("{}: {:.5}", d.nom, beta_sd_cellule(&[d], rapport, ModeRapport::Exact).unwrap_or(f64::NAN)))
                .collect();
            let ok = biais <= SEUIL_SD_BIAIS && rel <= SEUIL_SD_SIGMA;
            (ok, biais, rel, format!("b = {b:.6} ± {:.6} (jackknife par chaîne, 3 réplicats), |b/β − 1| = {biais:.3e} (seuil {SEUIL_SD_BIAIS}), σ(b)/β = {rel:.3e} (seuil {SEUIL_SD_SIGMA}) ; par chaîne : {}", jk.erreur, par_chaine.join(", ")))
        }
        _ => (false, f64::NAN, f64::NAN, "estimateur indéfini (rhs(50) < Σw ou jackknife impossible)".into()),
    }
}

pub fn gates_sd(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let Some(g) = &ctx.grandl else {
        return Ok(vec![skip("RG-SD", "ensembles grandL non fournis (--grandl)")]);
    };
    let mut v = Vec::new();
    let mut rouges_mutant = 0;
    let mut details_mutant = Vec::new();
    for beta in BETAS_SD {
        let donnees = charger_cellule(g, beta, ctx.fils)?;
        let n: Vec<usize> = donnees.iter().map(|d| d.p.len()).collect();
        let id = format!("RG-SD-b{beta}");
        if donnees.len() != CHAINES_SD || n.iter().any(|&k| k != CONFIGS_PAR_CHAINE) {
            v.push(LigneGate::nouvelle(Verdict::Fail, &id, format!("protocole non tenu : {} chaînes, configurations {n:?} au lieu de 3 × 20", donnees.len())));
            continue;
        }
        let (ok, _, _, detail) = verdict_sd(RapportBessel::I2SurI1, beta, &donnees);
        v.push(ligne(ok, &id, format!("grandL L = 12, β = {beta}, 3 chaînes × 20 configurations : {detail}")));
        if ctx.mutants {
            let (ok_m, biais, rel, detail_m) = verdict_sd(RapportBessel::I1SurI0, beta, &donnees);
            if !ok_m {
                rouges_mutant += 1;
            }
            details_mutant.push(format!("β = {beta} : {} ({detail_m}) ; |b/β − 1| = {biais:.3e}, σ/β = {rel:.3e}", if ok_m { "VERT" } else { "ROUGE" }));
        }
    }
    if ctx.mutants {
        v.push(ligne(rouges_mutant >= 1, "RG-SD-MUT", format!("mutant I2/I1 → I1/I0 (forme U(1)) : {rouges_mutant} β rouge(s) sur {} (≥ 1 exigé, sinon ECHEC_OUTIL) ; {}", BETAS_SD.len(), details_mutant.join(" | "))));
    }
    Ok(v)
}

// ------------------------------------------------------------------------------------------
// RG-PLAQ
// ------------------------------------------------------------------------------------------

/// Survie du χ² à `ddl` PAIR degrés de liberté : exp(−x/2) Σ_{k < ddl/2} (x/2)^k / k!.
pub fn survie_chi2_pair(chi2: f64, ddl: usize) -> f64 {
    let x = chi2 / 2.0;
    let (mut terme, mut somme) = (1.0, 1.0);
    for k in 1..ddl / 2 {
        terme *= x / k as f64;
        somme += terme;
    }
    (-x).exp() * somme
}

pub struct PointPlaq {
    pub beta: f64,
    pub beta_eff: Option<f64>,
    pub sigma_beta: Option<f64>,
    pub z: f64,
    pub hors: bool,
}

/// z de chaque point : chaîne indépendante contre la courbe (protocole gelé : espace β dans la plage, espace P hors plage).
pub fn z_plaq(courbe: &Courbe, indep: &[ChaineMesuree]) -> Resultat<Vec<PointPlaq>> {
    z_observable(courbe, indep, Observable::Plaquette)
}

/// Même inversion et même propagation que WP8, pour P ou R.
pub fn z_observable(courbe: &Courbe, indep: &[ChaineMesuree], obs: Observable) -> Resultat<Vec<PointPlaq>> {
    let mut v = Vec::new();
    for c in indep {
        let mesure = match obs { Observable::Plaquette => &c.p, Observable::Rectangle => &c.r };
        let i = c.indice_beta;
        let nœud = &courbe.points[i];
        if (nœud.beta - c.beta).abs() > 1e-12 {
            return Err(Erreur::Validation("indices de β incohérents entre courbe et chaîne indépendante".into()));
        }
        let inv = courbe.inverser(obs, mesure.moy).map_err(Erreur::Validation)?;
        v.push(match inv {
            Inversion::Dans { beta, pente, sigma_cal, .. } => {
                let sigma = mesure.err.hypot(sigma_cal) / pente.abs();
                PointPlaq { beta: c.beta, beta_eff: Some(beta), sigma_beta: Some(sigma), z: (beta - c.beta) / sigma, hors: false }
            }
            Inversion::HorsEtalonnage { sigma_extreme, valeur_extreme, .. } => {
                let z = (mesure.moy - valeur_extreme) / mesure.err.hypot(sigma_extreme);
                PointPlaq { beta: c.beta, beta_eff: None, sigma_beta: None, z, hors: true }
            }
        });
    }
    Ok(v)
}

fn protocole_gele_ok(par: &ParamEtalonnage, independante: bool, facteur: f64, l: usize) -> bool {
    let g = ParamEtalonnage::gele(l);
    match g {
        Ok(g) => {
            par.l == g.l
                && par.betas == GRILLE_GELEE.to_vec()
                && par.therm == g.therm
                && par.mesures == g.mesures
                && par.graine_base == g.graine_base
                && par.independante == independante
                && (independante || par.n_chaines == g.n_chaines)
                && par.facteur_beta == facteur
        }
        Err(_) => false,
    }
}

fn evaluer_plaq(courbe: &Courbe, indep: &[ChaineMesuree]) -> Resultat<(bool, f64, f64, f64, usize, String)> {
    let pts = z_plaq(courbe, indep)?;
    let zmax = pts.iter().map(|p| p.z.abs()).fold(0.0, f64::max);
    let chi2: f64 = pts.iter().map(|p| p.z * p.z).sum();
    let p = survie_chi2_pair(chi2, pts.len() + pts.len() % 2);
    let n_hors = pts.iter().filter(|p| p.hors).count();
    let ok = pts.iter().all(|p| p.z.abs() < SEUIL_Z_PLAQ) && p > SEUIL_P_CHI2 && pts.len() == GRILLE_GELEE.len();
    let detail = pts
        .iter()
        .map(|p| match p.beta_eff {
            Some(b) => format!("β={}: β_eff {:.4}±{:.4} z {:+.2}", p.beta, b, p.sigma_beta.unwrap_or(f64::NAN), p.z),
            None => format!("β={}: HORS_ETALONNAGE z(P) {:+.2}", p.beta, p.z),
        })
        .collect::<Vec<_>>()
        .join(" ; ");
    Ok((ok, zmax, chi2, p, n_hors, detail))
}

fn lire_etal(chemin: &Path) -> Resultat<(ParamEtalonnage, Vec<ChaineMesuree>)> {
    depuis_json(&json_de(chemin)?)
}

pub fn gates_plaq(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let Some(d) = &ctx.etalonnages else {
        return Ok(vec![skip("RG-PLAQ", "dossier des étalonnages non fourni (--etalonnages)")]);
    };
    let (f_etal, f_ind) = (d.join("etal_L4.json"), d.join("indep_L4.json"));
    if !f_etal.is_file() || !f_ind.is_file() {
        return Ok(vec![skip("RG-PLAQ", format!("{} ou {} absent : lancer `kvu rg etalonner` (voir validation/rg_flot/README.md)", f_etal.display(), f_ind.display()))]);
    }
    let (par_e, ch_e) = lire_etal(&f_etal)?;
    let (par_i, ch_i) = lire_etal(&f_ind)?;
    let mut v = Vec::new();
    let proto = protocole_gele_ok(&par_e, false, 1.0, 4) && protocole_gele_ok(&par_i, true, 1.0, 4);
    v.push(ligne(proto, "RG-PLAQ-PROTOCOLE", format!("protocole du journal respecté (grille de 14 β, L = 4, 3 chaînes par point + 1 indépendante, therm {}, mesures {}, graines {} + 100 i + c) : {proto}", par_e.therm, par_e.mesures, par_e.graine_base)));
    let courbe = courbe_depuis(4, &ch_e)?;
    for obs in [Observable::Plaquette, Observable::Rectangle] {
        courbe.verifier_monotone(obs).map_err(Erreur::Validation)?;
    }
    let (ok, zmax, chi2, p, n_hors, detail) = evaluer_plaq(&courbe, &ch_i)?;
    v.push(ligne(ok, "RG-PLAQ", format!("14 points, chaîne indépendante (c = 9) contre la courbe L = 4 : max|z| = {zmax:.2} (seuil < {SEUIL_Z_PLAQ}), χ² = {chi2:.2} pour 14 degrés, p = {p:.4} (seuil > {SEUIL_P_CHI2}), {n_hors} point(s) hors plage rapportés « hors étalonnage » ; {detail}")));
    // INFO : systématique d'interpolation (PCHIP contre linéaire) au milieu de chaque intervalle
    let mut ecart_interp = 0.0_f64;
    for w in courbe.points.windows(2) {
        let b = 0.5 * (w[0].beta + w[1].beta);
        if let (Some((a, _)), Some(l)) = (courbe.evaluer(Observable::Plaquette, b), courbe.evaluer_lineaire(Observable::Plaquette, b)) {
            ecart_interp = ecart_interp.max((a - l).abs());
        }
    }
    v.push(info("RG-PLAQ-INTERP", format!("INFO systématique d'interpolation : max |P_PCHIP − P_linéaire| au milieu des intervalles = {ecart_interp:.3e}")));
    if ctx.mutants {
        let f_mut = d.join("indep_mut_L4.json");
        if !f_mut.is_file() {
            v.push(skip("RG-PLAQ-MUT", format!("{} absent", f_mut.display())));
        } else {
            let (par_m, ch_m) = lire_etal(&f_mut)?;
            let proto_m = protocole_gele_ok(&par_m, true, FACTEUR_MUTANT_PLAQ, 4);
            let (ok_m, zmax_m, chi2_m, p_m, _, _) = evaluer_plaq(&courbe, &ch_m)?;
            v.push(ligne(proto_m && !ok_m, "RG-PLAQ-MUT", format!("mutant : chaîne indépendante simulée à {FACTEUR_MUTANT_PLAQ} β : max|z| = {zmax_m:.1}, χ² = {chi2_m:.1}, p = {p_m:.2e} ⇒ {} (ROUGE exigé, sinon ECHEC_OUTIL) ; protocole du mutant respecté : {proto_m}", if ok_m { "VERT" } else { "ROUGE" })));
        }
    }
    Ok(v)
}

// ------------------------------------------------------------------------------------------
// RG-SCHEMA (rapporté) et RG-FILS-FLOT
// ------------------------------------------------------------------------------------------

fn valeur_est(c: &Value, nom: &str) -> String {
    let e = &c["estimateurs"][nom];
    match e["etat"].as_str() {
        Some("ok") => format!("{:.4} ± {:.4}", e["valeur"].as_f64().unwrap_or(f64::NAN), e["erreur_totale"].as_f64().unwrap_or(f64::NAN)),
        Some(s) => s.to_string(),
        None => "—".into(),
    }
}

pub fn gate_schema(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let Some(g) = &ctx.grandl else {
        return Ok(vec![skip("RG-SCHEMA", "ensembles grandL non fournis (--grandl)")]);
    };
    let mut courbes_f = Vec::new();
    if let Some(d) = &ctx.etalonnages {
        for n in ["etal_L4.json", "etal_L6.json"] {
            if d.join(n).is_file() {
                courbes_f.push(d.join(n));
            }
        }
    }
    let (courbes, fichiers) = lire_courbes(&courbes_f)?;
    let mut v = Vec::new();
    for beta in BETAS_SD {
        let motif = format!("L12_b{}_c*", if beta == beta.trunc() { format!("{}", beta as i64) } else { format!("{beta}") });
        let mut textes = Vec::new();
        for schema in [Schema::Balaban, Schema::Swendsen] {
            let par = ParamFlot {
                schema,
                pas: 1,
                estimateurs: Estimateurs { sd: true, plaq: true, mcrg2: true },
                rapport: RapportBessel::I2SurI1,
                courbes: courbes.clone(),
                fils: ctx.fils,
            };
            let doc = flot(g, Some(&motif), &par, &fichiers)?;
            let c = &doc["cellules"][0];
            let dom = c["domaine"][0]["etiquette"].as_str().map(|e| format!(", domaine {e} f = {:.3e}", c["domaine"][0]["fraction_hors"].as_f64().unwrap_or(f64::NAN))).unwrap_or_default();
            textes.push(format!("{} (L {} → {}) : sd {} ; plaq {} ; mcrg2 {}{dom}", schema.nom(), c["l_entree"], c["l_fin"], valeur_est(c, "sd"), valeur_est(c, "plaq"), valeur_est(c, "mcrg2")));
        }
        v.push(info(&format!("RG-SCHEMA-b{beta}"), format!("INFO β = {beta}, un pas, différence de schéma RAPPORTÉE (pas un échec ; facteurs de bloc 3 et 2 différents, aucune lecture physique ici) : {}", textes.join(" || "))));
    }
    Ok(v)
}

pub fn gate_fils_flot(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let Some(g) = &ctx.grandl else {
        return Ok(vec![skip("RG-FILS-FLOT", "ensembles grandL non fournis (--grandl)")]);
    };
    let mut courbes_f = Vec::new();
    if let Some(d) = &ctx.etalonnages {
        if d.join("etal_L4.json").is_file() {
            courbes_f.push(d.join("etal_L4.json"));
        }
    }
    let (courbes, fichiers) = lire_courbes(&courbes_f)?;
    let mut sorties = Vec::new();
    for fils in [1usize, 4, 16] {
        let par = ParamFlot { schema: Schema::Balaban, pas: 1, estimateurs: Estimateurs { sd: true, plaq: true, mcrg2: true }, rapport: RapportBessel::I2SurI1, courbes: courbes.clone(), fils };
        sorties.push(serde_json::to_string_pretty(&flot(g, Some("L12_b8_c*"), &par, &fichiers)?).map_err(|e| Erreur::Format(e.to_string()))?);
    }
    let ok = sorties.iter().all(|s| *s == sorties[0]);
    Ok(vec![ligne(ok, "RG-FILS-FLOT", format!("flot.json de L12_b8_c* (Bałaban, 1 pas, sd+plaq+mcrg2) identique à l'octet pour 1, 4 et 16 fils ({} octets) : {ok}", sorties[0].len()))])
}

// ------------------------------------------------------------------------------------------
// Batterie
// ------------------------------------------------------------------------------------------

pub fn batterie(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let mut v = Vec::new();
    if ctx.veut("RG-BESSEL") {
        v.push(gate_bessel(ctx)?);
    }
    if ctx.veut("RG-TAU") {
        v.extend(gate_tau(ctx)?);
    }
    if ctx.veut("RG-N3") {
        v.extend(gate_n3(ctx)?);
    }
    if ctx.veut("RG-SD") {
        v.extend(gates_sd(ctx)?);
    }
    if ctx.veut("RG-PLAQ") {
        v.extend(gates_plaq(ctx)?);
    }
    if ctx.veut("RG-SCHEMA") {
        v.extend(gate_schema(ctx)?);
    }
    if ctx.veut("RG-FILS-FLOT") {
        v.extend(gate_fils_flot(ctx)?);
    }
    v.retain(|l| ctx.garde_ligne(&l.id));
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi2_pair_valeurs_connues() {
        // χ² à 14 ddl : valeur critique 29,141 ⇒ p = 0,01 ; χ² = 14 ⇒ p ≈ 0,4497 ; ddl 2 : exp(−x/2)
        assert!((survie_chi2_pair(29.1412, 14) - 0.01).abs() < 1e-4);
        assert!((survie_chi2_pair(14.0, 14) - 0.4497).abs() < 1e-3);
        assert!((survie_chi2_pair(4.0, 2) - (-2.0_f64).exp()).abs() < 1e-15);
        assert!(survie_chi2_pair(0.0, 14) == 1.0);
    }
}
