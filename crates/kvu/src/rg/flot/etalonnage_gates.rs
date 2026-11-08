//! Gates du protocole WP9, indépendantes des textes et du calcul gelés de WP8.
use super::etalonner::{
    courbe_depuis, depuis_json, ChaineMesuree, ParamEtalonnage, GRILLE_WP9, NOR_GELE,
};
use super::gates::{z_observable, FACTEUR_MUTANT_PLAQ, SEUIL_P_CHI2};
use super::plaq::{Courbe, Observable};
use crate::commun::{Erreur, Resultat};
use crate::groupes::gates::{LigneGate, Verdict};
use serde_json::Value;
use std::path::Path;

/// Bonferroni α = 0,01 sur les 31 points de WP9 (PREREG §6).
pub const SEUIL_Z_WP9: f64 = 3.59;

/// Q(ddl/2, χ²/2), gamma incomplète régularisée : série de P ou fraction continue de Q.
/// La normalisation utilise Γ(n) et Γ(n+1/2) sans approximation de Stirling.
pub fn survie_chi2(chi2: f64, ddl: usize) -> f64 {
    if ddl == 0 || chi2.is_nan() || chi2 < 0.0 {
        return f64::NAN;
    }
    if chi2 == 0.0 {
        return 1.0;
    }
    if chi2.is_infinite() {
        return 0.0;
    }
    let a = ddl as f64 / 2.0;
    let x = chi2 / 2.0;
    let (mut z, mut log_gamma) = if ddl.is_multiple_of(2) {
        (1.0_f64, 0.0)
    } else {
        (0.5_f64, 0.5 * std::f64::consts::PI.ln())
    };
    while z < a {
        log_gamma += z.ln();
        z += 1.0;
    }
    let facteur = (a * x.ln() - x - log_gamma).exp();
    if x < a + 1.0 {
        let mut terme = 1.0 / a;
        let mut somme = terme;
        for n in 1..100_000 {
            terme *= x / (a + n as f64);
            somme += terme;
            if terme.abs() <= somme.abs() * 2e-16 {
                break;
            }
        }
        (1.0 - facteur * somme).clamp(0.0, 1.0)
    } else {
        const MIN: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / MIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..100_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < MIN {
                d = MIN;
            }
            c = b + an / c;
            if c.abs() < MIN {
                c = MIN;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 4e-16 {
                break;
            }
        }
        (facteur * h).clamp(0.0, 1.0)
    }
}

struct Etalonnage {
    doc: Value,
    par: ParamEtalonnage,
    chaines: Vec<ChaineMesuree>,
}

impl Etalonnage {
    fn lire(f: &Path) -> Resultat<Self> {
        let texte = std::fs::read_to_string(f)
            .map_err(|e| Erreur::io(format!("lecture {}", f.display()), e))?;
        let doc = serde_json::from_str(&texte)
            .map_err(|e| Erreur::Format(format!("{} : {e}", f.display())))?;
        let (par, chaines) = depuis_json(&doc)?;
        Ok(Self { doc, par, chaines })
    }

    fn protocole_ok(&self, l: usize, independante: bool, facteur: f64) -> bool {
        let Ok(g) = ParamEtalonnage::wp9(l) else {
            return false;
        };
        let p = &self.par;
        if self.doc["protocole"] != "wp9"
            || self.doc["nor"].as_u64() != Some(NOR_GELE as u64)
            || p.l != l
            || p.betas != g.betas
            || p.therm != g.therm
            || p.mesures != g.mesures
            || p.graine_base != g.graine_base
            || p.independante != independante
            || p.est_milieux()
            || p.facteur_beta != facteur
            || p.n_chaines != if independante { 1 } else { g.n_chaines }
        {
            return false;
        }
        let cs: &[u64] = if independante { &[9] } else { &[1, 2, 3] };
        self.chaines.len() == GRILLE_WP9.len() * cs.len()
            && g.betas.iter().enumerate().all(|(i, &beta)| {
                cs.iter().all(|&c| {
                    let mut ch = self
                        .chaines
                        .iter()
                        .filter(|ch| ch.indice_beta == i && ch.c == c);
                    let Some(champ) = ch.next() else { return false };
                    ch.next().is_none()
                        && champ.beta == beta
                        && (champ.beta_simule - beta * facteur).abs() <= 1e-12
                        && champ.graine == g.graine(i, c)
                        && champ.p.n == g.mesures
                        && champ.r.n == g.mesures
                })
            })
    }

    fn protocole_milieux_ok(&self, c: &Etalonnage) -> bool {
        let p = &self.par;
        let g = &c.par;
        if self.doc["protocole"] != "wp9"
            || self.doc["nor"].as_u64() != Some(NOR_GELE as u64)
            || !p.est_milieux()
            || self.doc["milieux"] != true
            || p.independante
            || p.facteur_beta != 1.0
            || p.l != g.l
            || p.n_chaines != 1
            || p.graine_base != g.graine_base
            || p.therm != g.therm
            || p.mesures != g.mesures
            || p.betas.len() != 30
            || self.chaines.len() != 30
        {
            return false;
        }
        g.betas.windows(2).enumerate().all(|(i, w)| {
            let beta = (w[0] + w[1]) / 2.0;
            let mut chaines = self.chaines.iter().filter(|ch| ch.indice_beta == i);
            let Some(ch) = chaines.next() else {
                return false;
            };
            chaines.next().is_none()
                && (p.betas[i] - beta).abs() <= 1e-12
                && (ch.beta - beta).abs() <= 1e-12
                && (ch.beta_simule - beta).abs() <= 1e-12
                && ch.c == 8
                && ch.graine == g.graine(i, 8)
                && ch.p.n == g.mesures
                && ch.r.n == g.mesures
        })
    }
}

fn ligne(ok: bool, id: String, detail: String) -> LigneGate {
    LigneGate::nouvelle(if ok { Verdict::Pass } else { Verdict::Fail }, &id, detail)
}

fn evaluer(
    courbe: &Courbe,
    chaines: &[ChaineMesuree],
    obs: Observable,
) -> Resultat<(bool, String)> {
    // Les indices ont été vérifiés par protocole_ok avant cet appel.
    let points = z_observable(courbe, chaines, obs)?;
    let zmax = points.iter().map(|p| p.z.abs()).fold(0.0, f64::max);
    let chi2: f64 = points.iter().map(|p| p.z * p.z).sum();
    let ddl = points.len();
    let p = survie_chi2(chi2, ddl);
    let hors = points.iter().filter(|p| p.hors).count();
    let ok = ddl == 31
        && points
            .iter()
            .all(|p| p.z.is_finite() && p.z.abs() < SEUIL_Z_WP9)
        && p > SEUIL_P_CHI2;
    let detail = points
        .iter()
        .map(|p| match p.beta_eff {
            Some(b) => format!(
                "β={}: β_eff {:.4}±{:.4} z {:+.2}",
                p.beta,
                b,
                p.sigma_beta.unwrap_or(f64::NAN),
                p.z
            ),
            None => format!(
                "β={}: HORS_ETALONNAGE z({}) {:+.2}",
                p.beta,
                nom_observable(obs),
                p.z
            ),
        })
        .collect::<Vec<_>>()
        .join(" ; ");
    Ok((ok, format!("{ddl} points : max|z| = {zmax:.6}, χ² = {chi2:.6}, ddl = {ddl}, p = {p:.8e} ; seuils max|z| < {SEUIL_Z_WP9}, p > {SEUIL_P_CHI2} ; {hors} point(s) hors étalonnage ; {detail}")))
}

fn nom_observable(obs: Observable) -> &'static str {
    match obs {
        Observable::Plaquette => "P",
        Observable::Rectangle => "R",
    }
}

fn evaluer_milieux(courbe: &Courbe, m: &Etalonnage, obs: Observable) -> Resultat<(bool, String)> {
    let mut zs = Vec::new();
    let mut details = Vec::new();
    for (i, w) in courbe.points.windows(2).enumerate() {
        let beta = (w[0].beta + w[1].beta) / 2.0;
        let ch = m
            .chaines
            .iter()
            .find(|ch| ch.indice_beta == i)
            .ok_or_else(|| Erreur::Validation("milieu manquant".into()))?;
        let (mesure, sigma_courbe) = match obs {
            Observable::Plaquette => (&ch.p, w[0].sigma_p.max(w[1].sigma_p)),
            Observable::Rectangle => (&ch.r, w[0].sigma_r.max(w[1].sigma_r)),
        };
        let (valeur, _) = courbe
            .evaluer(obs, beta)
            .ok_or_else(|| Erreur::Validation("PCHIP indéfini au milieu".into()))?;
        let sigma = mesure.err.hypot(sigma_courbe);
        let z = (mesure.moy - valeur) / sigma;
        zs.push(z);
        details.push(format!(
            "β={beta}: {}_mes {:.8e}, PCHIP {:.8e}, σ {:.8e}, z {:+.6}",
            nom_observable(obs),
            mesure.moy,
            valeur,
            sigma,
            z
        ));
    }
    let zmax = zs.iter().map(|z| z.abs()).fold(0.0, f64::max);
    let chi2 = zs.iter().map(|z| z * z).sum();
    let ddl = zs.len();
    let p = survie_chi2(chi2, ddl);
    let ok =
        ddl == 30 && zs.iter().all(|z| z.is_finite() && z.abs() < SEUIL_Z_WP9) && p > SEUIL_P_CHI2;
    Ok((ok, format!("protocole milieux tenu ; {ddl} milieux : max|z| = {zmax:.6}, χ² = {chi2:.6}, ddl = {ddl}, p = {p:.8e} ; seuils max|z| < {SEUIL_Z_WP9}, p > {SEUIL_P_CHI2} ; {}", details.join(" ; "))))
}

fn interpolation(c: &Courbe) -> Resultat<String> {
    let obs = Observable::Plaquette;
    let mut milieu = 0.0_f64;
    for w in c.points.windows(2) {
        let beta = 0.5 * (w[0].beta + w[1].beta);
        let (a, _) = c
            .evaluer(obs, beta)
            .ok_or_else(|| Erreur::Validation("PCHIP indéfini".into()))?;
        let b = c
            .evaluer_lineaire(obs, beta)
            .ok_or_else(|| Erreur::Validation("interpolation linéaire indéfinie".into()))?;
        milieu = milieu.max((a - b).abs());
    }
    let (mut loo, mut beta_loo, mut loo_beta, mut beta_max) =
        (0.0_f64, c.points[1].beta, 0.0_f64, c.points[1].beta);
    for i in 1..c.points.len() - 1 {
        let mut points = c.points.clone();
        let point = points.remove(i);
        let reduite = Courbe::nouvelle(c.l, points).map_err(Erreur::Validation)?;
        let (p, _) = reduite
            .evaluer(obs, point.beta)
            .ok_or_else(|| Erreur::Validation("PCHIP laisser-un-de-côté indéfini".into()))?;
        let (_, pente) = c
            .evaluer(obs, point.beta)
            .ok_or_else(|| Erreur::Validation("pente locale indéfinie".into()))?;
        let ecart = (p - point.p).abs();
        if ecart > loo {
            loo = ecart;
            beta_loo = point.beta;
        }
        let ecart_beta = ecart / pente.abs();
        if ecart_beta > loo_beta {
            loo_beta = ecart_beta;
            beta_max = point.beta;
        }
    }
    let pente = c.evaluer(obs, beta_loo).unwrap().1.abs();
    Ok(format!("max |P_PCHIP − P_linéaire| aux milieux = {milieu:.8e} ; laisser-un-de-côté : max |P_LOO − P| = {loo:.8e} à β = {beta_loo}, écart en β = {:.8e} (pente locale) ; max écart en β = {loo_beta:.8e} à β = {beta_max}", loo / pente))
}

fn recouvrement(c: &Courbe, w: &Courbe) -> Resultat<String> {
    if c.l != w.l {
        return Err(Erreur::Config(
            "--comparer exige le même côté L que --courbe".into(),
        ));
    }
    let mut details = Vec::new();
    for obs in [Observable::Plaquette, Observable::Rectangle] {
        let (mut zmax, mut chi2, mut ddl) = (0.0_f64, 0.0, 0usize);
        for pc in &c.points {
            if let Some(pw) = w
                .points
                .iter()
                .find(|pw| (pc.beta - pw.beta).abs() <= 1e-12)
            {
                let (a, b, sa, sb) = match obs {
                    Observable::Plaquette => (pc.p, pw.p, pc.sigma_p, pw.sigma_p),
                    Observable::Rectangle => (pc.r, pw.r, pc.sigma_r, pw.sigma_r),
                };
                let z = (a - b) / sa.hypot(sb);
                zmax = zmax.max(z.abs());
                chi2 += z * z;
                ddl += 1;
            }
        }
        let nom = if obs == Observable::Plaquette {
            "P"
        } else {
            "R"
        };
        details.push(format!(
            "{nom} : max|z| = {zmax:.6}, χ² = {chi2:.6}, ddl = {ddl}, p = {:.8e}",
            survie_chi2(chi2, ddl)
        ));
    }
    Ok(details.join(" ; "))
}

/// Une batterie WP9 ; aucune dépendance aux jeux de données des gates WP8.
pub fn batterie(
    courbe: &Path,
    independante: &Path,
    mutant: Option<&Path>,
    comparer: Option<&Path>,
) -> Resultat<Vec<LigneGate>> {
    batterie_milieux(courbe, independante, mutant, comparer, None)
}

/// Variante avec contrôle facultatif des chaînes c = 8 entre les nœuds.
pub fn batterie_milieux(
    courbe: &Path,
    independante: &Path,
    mutant: Option<&Path>,
    comparer: Option<&Path>,
    milieux: Option<&Path>,
) -> Resultat<Vec<LigneGate>> {
    let c = Etalonnage::lire(courbe)?;
    let i = Etalonnage::lire(independante)?;
    let m = mutant.map(Etalonnage::lire).transpose()?;
    let l = c.par.l;
    let proto = c.protocole_ok(l, false, 1.0)
        && i.protocole_ok(l, true, 1.0)
        && m.as_ref()
            .is_none_or(|m| m.protocole_ok(l, true, FACTEUR_MUTANT_PLAQ));
    let mut lignes = vec![ligne(proto, format!("RG-PLAQ-WP9-PROTOCOLE-L{l}"),
        format!("protocole wp9 : grille 31 β, L = {l}, 3 chaînes + indépendante c = 9, paramètres et graines vérifiés : {proto}"))];
    if !proto {
        return Ok(lignes);
    }
    let courbe = courbe_depuis(l, &c.chaines)?;
    for obs in [Observable::Plaquette, Observable::Rectangle] {
        courbe.verifier_monotone(obs).map_err(Erreur::Validation)?;
    }
    let (ok, detail) = evaluer(&courbe, &i.chaines, Observable::Plaquette)?;
    lignes.push(ligne(ok, format!("RG-PLAQ-WP9-L{l}"), detail));
    let (ok, detail) = evaluer(&courbe, &i.chaines, Observable::Rectangle)?;
    lignes.push(ligne(ok, format!("RG-PLAQ-WP9-R-L{l}"), detail));
    lignes.push(LigneGate::nouvelle(
        Verdict::Info,
        &format!("RG-PLAQ-WP9-INTERP-L{l}"),
        interpolation(&courbe)?,
    ));
    if let Some(m) = m {
        let (ok, detail) = evaluer(&courbe, &m.chaines, Observable::Plaquette)?;
        lignes.push(ligne(
            !ok,
            format!("RG-PLAQ-WP9-MUT-L{l}"),
            format!(
                "mutant facteur {FACTEUR_MUTANT_PLAQ} : {} (ROUGE exigé) ; {detail}",
                if ok { "VERT" } else { "ROUGE" }
            ),
        ));
    }
    if let Some(f) = milieux {
        let m = Etalonnage::lire(f)?;
        for obs in [Observable::Plaquette, Observable::Rectangle] {
            let (ok, detail) = if m.protocole_milieux_ok(&c) {
                evaluer_milieux(&courbe, &m, obs)?
            } else {
                (false, "protocole milieux invalide : wp9, même L/base/therm/mesures, facteur 1, 30 milieux exacts et chaînes c = 8 exigés".into())
            };
            lignes.push(ligne(
                ok,
                format!("RG-PLAQ-WP9-MILIEU-{}-L{l}", nom_observable(obs)),
                detail,
            ));
        }
    }
    if let Some(f) = comparer {
        let w = Etalonnage::lire(f)?;
        if w.doc["protocole"] == "wp9" {
            return Err(Erreur::Config(
                "--comparer exige un étalonnage d'un autre protocole".into(),
            ));
        }
        let w = courbe_depuis(w.par.l, &w.chaines)?;
        lignes.push(LigneGate::nouvelle(
            Verdict::Info,
            &format!("RG-ETAL-RECOUVREMENT-L{l}"),
            recouvrement(&courbe, &w)?,
        ));
    }
    Ok(lignes)
}
