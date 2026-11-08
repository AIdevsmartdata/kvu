//! Batterie de gates à réponse connue d'un groupe de jauge (GATES v1 §3, v3 §v3.2, v3.1). Propriétaire : WP5.
//!
//! Chaque gate rend UNE ligne `PASS|FAIL|SKIP <ID> <détail>` (règle 1 de GATES.md) ; les mutants rendent leur propre ligne
//! (`G-MUT-*`, PASS = le mutant est bien ROUGE) ; les mesures d'information rendent une ligne `INFO`.
//! Tous les seuils, graines et protocoles de ce fichier sont ceux de GATES.md (v1, v3, v3.1) : un changement ici sans nouvelle
//! version de GATES.md est un défaut. Les coefficients `A_ATTENDU` et `C1_ATTENDU` sont ceux de `docs/CONVENTIONS.md` (gelés en
//! v3.1 AVANT le premier run de G-SC/G-WC) ; le test `a_attendu_recalcule_egale_la_constante_gelee` les recalcule indépendamment.

use super::su2::{MutantSu2, Su2};
use super::{Cmat, Flux, Groupe, Realite, ETIQUETTE_GROUPE};
use crate::commun::{Depart, Erreur, Graine, Precision, Resultat};
use crate::gpu::{Appareil, OptionsCompil};
use crate::su2::{self, ConfigLot, NoyauxSu2, ParamChaine};
use serde::Serialize;
use std::fmt;

// ------------------------------------------------------------------------------------------
// Constantes gelées (GATES v3 / v3.1)
// ------------------------------------------------------------------------------------------

pub const GRAINE_HAAR_1: u64 = 26_102_605_001;
pub const GRAINE_HAAR_2: u64 = 26_102_605_002;
pub const GRAINE_HAAR_3: u64 = 26_102_605_003;
pub const GRAINES_SC: [u64; 2] = [26_102_605_101, 26_102_605_102];
pub const BETAS_SC: [f64; 2] = [0.1, 0.2];
pub const GRAINES_WC: [u64; 4] = [26_102_605_201, 26_102_605_202, 26_102_605_203, 26_102_605_204];
pub const BETAS_WC: [f64; 4] = [50.0, 100.0, 200.0, 400.0];
pub const GRAINE_UNIT: u64 = 26_102_605_301;
/// Graine de l'intégration de Haar numérique de la dérivation de v3.1 (canal « Haar numérique »).
pub const GRAINE_DERIVATION_HAAR: u64 = 26_102_605_401;
/// G-REG : (β, graine, valeur archivée, erreur archivée) — `A/tests/CRITERIA.md` T3.
pub const ANCRES_REG: [(f64, u64, f64, f64); 2] = [(2.3, 2_510_023, 0.602_963_7, 5.3e-6), (2.5, 2_510_025, 0.652_901_0, 3.1e-6)];

/// v3.1 : coefficient a de l'ajustement exact aβ + bβ² aux points β = 0,1 et 0,2 de u + 4u⁵ (u = I₂/I₁).
/// SymPy/mpmath, 60 chiffres : 0.25020200596430649696 (`scratch/wp5/sympy/wp5_sympy.out`).
pub const A_ATTENDU: f64 = 0.250_202_005_964_306_5;
/// v3.1 : c₁ = lim β⟨1 − P⟩ = d_adj/4 = 3/4 (SU(2), V → ∞) ; la correction gaussienne de volume fini est −1/V (INFO).
pub const C1_ATTENDU: f64 = 0.75;

/// Seuils (GATES v1 §3).
pub const SEUIL_ALG: f64 = 1e-12;
pub const SEUIL_HAAR_Z: f64 = 4.0;
pub const SEUIL_APPARTENANCE: f64 = 1e-13;
pub const SEUIL_UNIT_F64: f64 = 1e-13;
pub const SEUIL_UNIT_F32: f64 = 2e-6;

// ------------------------------------------------------------------------------------------
// Verdicts et lignes
// ------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
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
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Skip => "SKIP",
            Verdict::Info => "INFO",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LigneGate {
    pub verdict: Verdict,
    pub id: String,
    pub detail: String,
}

impl LigneGate {
    pub fn nouvelle(verdict: Verdict, id: &str, detail: impl Into<String>) -> Self {
        LigneGate { verdict, id: id.to_string(), detail: detail.into() }
    }
    fn selon(ok: bool, id: &str, detail: impl Into<String>) -> Self {
        Self::nouvelle(if ok { Verdict::Pass } else { Verdict::Fail }, id, detail)
    }
}

impl fmt::Display for LigneGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.verdict, self.id, self.detail)
    }
}

/// Verdict d'ensemble : `true` ssi aucune ligne FAIL ni SKIP (les INFO ne comptent pas).
pub fn tout_vert(lignes: &[LigneGate]) -> bool {
    lignes.iter().all(|l| matches!(l.verdict, Verdict::Pass | Verdict::Info))
}

/// SKIP PRESCRITS par la règle 0.2 de GATES.md (gate propre à un autre backend) : admis dans l'entrée `cpu` d'un certificat,
/// jamais dans l'entrée d'un backend GPU. Ils restent des SKIP (non verts) dans la sortie de `gates`.
pub const SKIPS_PRESCRITS_CPU: &[&str] = &["G-UNIT-F32"];

/// Une batterie est certifiable sur `backend` ssi chaque ligne est PASS/INFO, ou SKIP prescrit sur le CPU. Rend les SKIP admis.
pub fn certifiable(lignes: &[LigneGate], backend: &str) -> Option<Vec<String>> {
    let mut admis = Vec::new();
    for l in lignes {
        match l.verdict {
            Verdict::Pass | Verdict::Info => {}
            Verdict::Skip if backend == "cpu" && SKIPS_PRESCRITS_CPU.contains(&l.id.as_str()) => admis.push(l.id.clone()),
            _ => return None,
        }
    }
    Some(admis)
}

// ------------------------------------------------------------------------------------------
// Statistique : fenêtre de Sokal (c = 6), comme `tests/stats.py` de (A)
// ------------------------------------------------------------------------------------------

/// Temps d'autocorrélation intégré à fenêtre automatique de Sokal : τ = ½ + Σ_{t ≤ W} ρ(t), W = premier t ≥ c·τ(t).
pub fn tau_int(x: &[f64], c: f64) -> (f64, usize) {
    let n = x.len();
    if n < 8 {
        return (0.5, 0);
    }
    let m = x.iter().sum::<f64>() / n as f64;
    let xc: Vec<f64> = x.iter().map(|v| v - m).collect();
    let v = xc.iter().map(|a| a * a).sum::<f64>() / n as f64;
    if v == 0.0 {
        return (0.5, 0);
    }
    let (mut t, mut fenetre) = (0.5, 0usize);
    for w in 1..n / 4 {
        let s: f64 = (0..n - w).map(|i| xc[i] * xc[i + w]).sum();
        t += s / (n - w) as f64 / v;
        fenetre = w;
        if w as f64 >= c * t {
            break;
        }
    }
    (t.max(0.5), fenetre)
}

/// Moyenne, erreur √(2 τ var / n) (var à n − 1) et τ.
#[derive(Clone, Copy, Debug)]
pub struct Estimation {
    pub moyenne: f64,
    pub erreur: f64,
    pub tau: f64,
    pub n: usize,
}

pub fn estimer(x: &[f64]) -> Estimation {
    let n = x.len();
    let m = x.iter().sum::<f64>() / n.max(1) as f64;
    let var = if n > 1 { x.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / (n - 1) as f64 } else { 0.0 };
    let (tau, _) = tau_int(x, 6.0);
    Estimation { moyenne: m, erreur: (2.0 * tau * var / n.max(1) as f64).sqrt(), tau, n }
}

// ------------------------------------------------------------------------------------------
// Bessel (série entière, x ≤ ~10) : u = I₂/I₁ et la série de couplage fort u + 4u⁵
// ------------------------------------------------------------------------------------------

/// I_ν(x), ν entier ≥ 0, par sa série entière (80 termes : exacte à la précision machine pour x ≤ 10).
pub fn bessel_i(nu: u32, x: f64) -> f64 {
    let (mut terme, mut somme) = ((x / 2.0).powi(nu as i32) / (1..=nu).map(f64::from).product::<f64>(), 0.0);
    for k in 0..80u32 {
        somme += terme;
        terme *= (x / 2.0) * (x / 2.0) / (f64::from(k + 1) * f64::from(k + 1 + nu));
    }
    somme
}

pub fn u_bessel(beta: f64) -> f64 {
    bessel_i(2, beta) / bessel_i(1, beta)
}

/// ⟨P⟩ attendu par la série de couplage fort de (A) : u + 4u⁵.
pub fn p_couplage_fort(beta: f64) -> f64 {
    let u = u_bessel(beta);
    u + 4.0 * u.powi(5)
}

/// Coefficient a de l'ajustement EXACT aβ + bβ² par les points (β₁, f₁), (β₂ = 2β₁, f₂) : a = (4 f₁ − f₂)/(2 β₁).
pub fn a_ajuste(beta1: f64, f1: f64, f2: f64) -> f64 {
    (4.0 * f1 - f2) / (2.0 * beta1)
}

/// `a_attendu` recalculé : même ajustement appliqué à la série exacte u + 4u⁵.
pub fn a_attendu_recalcule() -> f64 {
    a_ajuste(BETAS_SC[0], p_couplage_fort(BETAS_SC[0]), p_couplage_fort(BETAS_SC[1]))
}

// ------------------------------------------------------------------------------------------
// Protocole (valeurs gelées par défaut ; les tests de plomberie en prennent de réduites)
// ------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Protocole {
    pub n_haar: u64,
    pub l_sc: usize,
    pub therm_sc: u64,
    pub pas_sc: u64,
    pub l_wc: usize,
    pub therm_wc: u64,
    pub pas_wc: u64,
    pub l_unit: usize,
    pub beta_unit: f64,
    pub balayages_unit: u64,
    pub l_reg: usize,
    pub therm_reg: u64,
    pub pas_reg: u64,
    pub nor: usize,
    /// Inclure les mutants (G-MUT) et les lignes INFO du mutant β × 1,002.
    pub mutants: bool,
}

impl Protocole {
    /// Protocole gelé de GATES v1/v3/v3.1 (WP5).
    pub fn gele() -> Self {
        Protocole {
            n_haar: 1_000_000,
            l_sc: 6,
            therm_sc: 1_000,
            pas_sc: 20_000,
            l_wc: 8,
            therm_wc: 2_000,
            pas_wc: 20_000,
            l_unit: 6,
            beta_unit: 2.3,
            balayages_unit: 10_000,
            l_reg: 6,
            therm_reg: 2_000,
            pas_reg: 10_000,
            nor: 4,
            mutants: true,
        }
    }
}

/// Contexte d'exécution : backend (None = CPU), protocole, filtre d'identifiants (`--seul`).
pub struct Contexte<'a> {
    pub app: Option<&'a Appareil>,
    pub protocole: Protocole,
    pub seul: Vec<String>,
}

impl Contexte<'_> {
    fn veut(&self, id: &str) -> bool {
        self.seul.is_empty() || self.seul.iter().any(|p| id.starts_with(p.as_str()))
    }
    pub fn backend(&self) -> &'static str {
        self.app.map_or("cpu", |a| a.backend().nom())
    }
}

// ------------------------------------------------------------------------------------------
// G-ALG : algèbre (fermeture, Casimir, indice)
// ------------------------------------------------------------------------------------------

/// T_F attendu (gelé par groupe, GATES v3.0 : SymPy) ; None si la fiche n'est pas au registre.
pub fn t_f_attendu(nom: &str) -> Option<f64> {
    match nom {
        "su2" => Some(0.5),
        _ => None,
    }
}

/// Résidu de la projection de `m` sur le sous-espace RÉEL engendré par les `t` (hermitiennes) : `anti = false` ⇒ m = Σ x_c T^c
/// (x_c réels, cas de U T^a U†), `anti = true` ⇒ m = i Σ x_c T^c (cas de [T^a, T^b] = i f^{abc} T^c). Les coefficients sont
/// x_c = Tr(m T^c)/T_F (ou Im de cela) : la partie « mauvaise » (imaginaire pour l'un, réelle pour l'autre) reste dans le résidu.
fn residu_dans_span(m: &Cmat, t: &[Cmat], t_f: f64, anti: bool) -> f64 {
    let mut r = m.clone();
    for tc in t {
        let (cr, ci) = m.mul(tc).trace();
        let terme = if anti { tc.echelle_c((0.0, ci / t_f)) } else { tc.echelle(cr / t_f) };
        r = r.sub(&terme);
    }
    r.norme_frobenius()
}

pub fn g_alg(g: &dyn Groupe) -> Vec<LigneGate> {
    let t = g.generateurs();
    let n = t.len();
    let t_f = g.t_f();
    let mut flux = Flux::nouveau(Graine(GRAINE_HAAR_1), ETIQUETTE_GROUPE);
    // G-ALG-1 : [T^a, T^b] ∈ span(T) et ad-invariance (U T^a U† ∈ span(T), U de Haar).
    let mut ferm = 0.0f64;
    for a in 0..n {
        for b in 0..n {
            ferm = ferm.max(residu_dans_span(&t[a].commutateur(&t[b]), &t, t_f, true));
        }
    }
    let mut adinv = 0.0f64;
    for _ in 0..50 {
        let u = g.haar(&mut flux);
        for ta in &t {
            adinv = adinv.max(residu_dans_span(&u.mul(ta).mul(&u.dag()), &t, t_f, false));
        }
    }
    let herm = t.iter().map(|ta| ta.max_abs_diff(&ta.dag()).max(ta.trace().0.hypot(ta.trace().1))).fold(0.0f64, f64::max);
    let l1 = LigneGate::selon(
        ferm <= SEUIL_ALG && adinv <= SEUIL_ALG && herm <= SEUIL_ALG,
        "G-ALG-1",
        format!("{} : fermeture [T^a,T^b] = i f T {ferm:.3e}, ad-invariance {adinv:.3e}, hermiticité et trace nulle {herm:.3e} (seuil {SEUIL_ALG:.0e}), {n} générateurs", g.nom()),
    );
    // G-ALG-2 : f^{acd} f^{bcd} = h∨ δ^{ab}, f CALCULÉS depuis les T.
    let f = g.f_abc();
    let l2 = if f.len() != n * n * n {
        LigneGate::nouvelle(Verdict::Fail, "G-ALG-2", format!("f_abc de longueur {} ≠ {n}³", f.len()))
    } else {
        let mut dev = 0.0f64;
        for a in 0..n {
            for b in 0..n {
                let s: f64 = (0..n).flat_map(|c| (0..n).map(move |d| (c, d))).map(|(c, d)| f[(a * n + c) * n + d] * f[(b * n + c) * n + d]).sum();
                let att = if a == b { g.h_vee() as f64 } else { 0.0 };
                dev = dev.max((s - att).abs());
            }
        }
        LigneGate::selon(dev <= SEUIL_ALG, "G-ALG-2", format!("max|f^acd f^bcd − h∨ δ| = {dev:.3e} (h∨ = {}, seuil {SEUIL_ALG:.0e})", g.h_vee()))
    };
    // G-ALG-3 : Tr(T^a T^b) = T_F δ^{ab}, T_F gelé par groupe ; T_F = d_F C₂(F)/d_adj avec ΣT^aT^a = C₂(F)·1 mesuré.
    let l3 = match t_f_attendu(g.nom()) {
        None => LigneGate::nouvelle(Verdict::Skip, "G-ALG-3", format!("T_F de « {} » non gelé (à geler par SymPy avant le lot du groupe)", g.nom())),
        Some(att) => {
            let mut dev = 0.0f64;
            for a in 0..n {
                for b in 0..n {
                    let (re, im) = t[a].mul(&t[b]).trace();
                    dev = dev.max((re - if a == b { att } else { 0.0 }).abs()).max(im.abs());
                }
            }
            let mut cas = Cmat::zeros(g.d_f());
            for ta in &t {
                cas = cas.add(&ta.mul(ta));
            }
            let c2 = cas.get(0, 0).0;
            let dev_cas = cas.sub(&Cmat::identite(g.d_f()).echelle(c2)).norme_frobenius();
            let t_f_casimir = g.d_f() as f64 * c2 / g.d_adj() as f64;
            let ok = dev <= SEUIL_ALG && dev_cas <= SEUIL_ALG && (t_f_casimir - att).abs() <= SEUIL_ALG;
            LigneGate::selon(
                ok,
                "G-ALG-3",
                format!("max|Tr(TT) − {att}δ| = {dev:.3e} ; T_F mesuré {t_f:.15} ; C₂(F) = {c2:.15} (ΣTT ∝ 1 à {dev_cas:.1e}) ; d_F C₂/d_adj = {t_f_casimir:.15} (seuil {SEUIL_ALG:.0e})"),
            )
        }
    };
    vec![l1, l2, l3]
}

// ------------------------------------------------------------------------------------------
// G-HAAR : mesure de Haar
// ------------------------------------------------------------------------------------------

fn moments(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let v = x.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / (n - 1.0);
    (m, (v / n).sqrt())
}

fn fs_attendu(r: Realite) -> f64 {
    match r {
        Realite::Reel => 1.0,
        Realite::Pseudoreel => -1.0,
        Realite::Complexe => 0.0,
    }
}

pub fn g_haar(g: &dyn Groupe, n: u64) -> Vec<LigneGate> {
    // G-HAAR-1 : ⟨χ⟩ = 0, ⟨|χ|²⟩ = 1.
    let mut f1 = Flux::nouveau(Graine(GRAINE_HAAR_1), ETIQUETTE_GROUPE);
    let (mut chi, mut chi2) = (Vec::with_capacity(n as usize), Vec::with_capacity(n as usize));
    for _ in 0..n {
        let (re, im) = g.haar(&mut f1).trace();
        chi.push(re);
        chi2.push(re * re + im * im);
    }
    let (m1, e1) = moments(&chi);
    let (m2, e2) = moments(&chi2);
    let (z1, z2) = (m1 / e1, (m2 - 1.0) / e2);
    let l1 = LigneGate::selon(
        z1.abs() < SEUIL_HAAR_Z && z2.abs() < SEUIL_HAAR_Z,
        "G-HAAR-1",
        format!("⟨χ⟩ = {m1:+.3e} ± {e1:.1e} (z = {z1:+.2}) ; ⟨|χ|²⟩ = {m2:.6} ± {e2:.1e} (z = {z2:+.2}) ; n = {n}, graine {GRAINE_HAAR_1}, seuil |z| < {SEUIL_HAAR_Z}"),
    );
    // G-HAAR-2 : indicateur de Frobenius–Schur ⟨χ(U²)⟩ à ± 4σ.
    let mut f2 = Flux::nouveau(Graine(GRAINE_HAAR_2), ETIQUETTE_GROUPE);
    let fs: Vec<f64> = (0..n).map(|_| { let u = g.haar(&mut f2); u.mul(&u).trace().0 }).collect();
    let (mf, ef) = moments(&fs);
    let att = fs_attendu(g.realite());
    let zf = (mf - att) / ef;
    let l2 = LigneGate::selon(
        zf.abs() <= 4.0,
        "G-HAAR-2",
        format!("⟨χ(U²)⟩ = {mf:+.5} ± {ef:.1e} (attendu {att:+}, z = {zf:+.2}) ; n = {n}, graine {GRAINE_HAAR_2}, seuil |z| ≤ 4"),
    );
    // G-HAAR-3 : appartenance au groupe.
    let mut f3 = Flux::nouveau(Graine(GRAINE_HAAR_3), ETIQUETTE_GROUPE);
    let (mut mu, mut md, mut ms) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..n {
        let a = g.appartient(&g.haar(&mut f3));
        mu = mu.max(a.unitarite);
        md = md.max(a.det);
        ms = ms.max(a.structure);
    }
    let l3 = LigneGate::selon(
        mu <= SEUIL_APPARTENANCE && md <= SEUIL_APPARTENANCE && ms <= SEUIL_APPARTENANCE,
        "G-HAAR-3",
        format!("max|U†U−1| = {mu:.2e} ; max|det U−1| = {md:.2e} ; structure {ms:.2e} ; n = {n}, graine {GRAINE_HAAR_3}, seuil {SEUIL_APPARTENANCE:.0e}"),
    );
    vec![l1, l2, l3]
}

// ------------------------------------------------------------------------------------------
// Simulation (backend CPU ou GPU) : plaquette par pas
// ------------------------------------------------------------------------------------------

fn un_pas(n: &mut dyn NoyauxSu2, nor: usize) -> Resultat<()> {
    n.sweep_hb()?;
    for _ in 0..nor {
        n.sweep_or()?;
    }
    Ok(())
}

fn lot(l: usize, chaines: &[(f64, u64)], depart: Depart, nor: usize, precision: Precision) -> ConfigLot {
    ConfigLot {
        l,
        chaines: chaines.iter().map(|&(beta, g)| ParamChaine { beta, graine: Graine(g), depart }).collect(),
        nor,
        precision,
        options: match precision {
            Precision::F64 => OptionsCompil::reference_k1(),
            Precision::F32 => OptionsCompil::rapide(),
        },
        mutant_metro: false,
    }
}

/// Simule un lot (β simulé, graine) : `therm` pas, puis `nmeas` pas mesurés (indices 1..=nmeas, comme `su2 gen` avec skip = 1) ;
/// rend la plaquette de chaque pas, par chaîne. Refuse un tirage au plafond (invariant de l'échantillonneur).
pub fn simuler_plaquettes(
    app: Option<&Appareil>, l: usize, chaines: &[(f64, u64)], depart: Depart, therm: u64, nmeas: u64, nor: usize,
) -> Resultat<Vec<Vec<f64>>> {
    let mut n = su2::construire(app, lot(l, chaines, depart, nor, Precision::F64))?;
    n.init()?;
    let cap = nmeas as usize + 1;
    n.allouer_historique(cap)?;
    for _ in 0..therm {
        un_pas(n.as_mut(), nor)?;
    }
    for m in 0..=nmeas as usize {
        if m > 0 {
            un_pas(n.as_mut(), nor)?;
        }
        n.mesurer(m)?;
    }
    let h = n.historique()?;
    if n.compteurs()?.iter().any(|c| c.plafond != 0) {
        return Err(Erreur::Validation("tirage au plafond d'essais : simulation invalide".into()));
    }
    Ok((0..chaines.len()).map(|c| (1..=nmeas as usize).map(|m| h[(c * cap + m) * 5]).collect()).collect())
}

// ------------------------------------------------------------------------------------------
// G-SC : couplage fort
// ------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ResultatSc {
    pub ligne: LigneGate,
    pub a: f64,
    pub sigma_a: f64,
    pub z_a: f64,
    pub z_p: [f64; 2],
}

/// G-SC (v3, v3.1) : simule à `facteur` × β, analyse aux β nominaux.
pub fn g_sc(ctx: &Contexte, id: &str, facteur: f64) -> Resultat<ResultatSc> {
    let p = &ctx.protocole;
    let chaines: Vec<(f64, u64)> = BETAS_SC.iter().zip(GRAINES_SC).map(|(&b, g)| (b * facteur, g)).collect();
    let xs = simuler_plaquettes(ctx.app, p.l_sc, &chaines, Depart::Chaud, p.therm_sc, p.pas_sc, p.nor)?;
    let est: Vec<Estimation> = xs.iter().map(|x| estimer(x)).collect();
    let a = a_ajuste(BETAS_SC[0], est[0].moyenne, est[1].moyenne);
    let sigma_a = (16.0 * est[0].erreur.powi(2) + est[1].erreur.powi(2)).sqrt() / (2.0 * BETAS_SC[0]);
    let z_a = (a - A_ATTENDU) / sigma_a;
    let mut z_p = [0.0; 2];
    let mut morceaux = Vec::new();
    for i in 0..2 {
        let att = p_couplage_fort(BETAS_SC[i]);
        z_p[i] = (est[i].moyenne - att) / est[i].erreur;
        morceaux.push(format!("β={}: ⟨P⟩ = {:.7} ± {:.1e} (τ = {:.2}) attendu {:.7} z = {:+.2}", BETAS_SC[i], est[i].moyenne, est[i].erreur, est[i].tau, att, z_p[i]));
    }
    let ok = z_a.abs() < 3.0 && z_p.iter().all(|z| z.abs() < 3.0);
    let detail = format!(
        "a = {a:.6} ± {sigma_a:.1e} (attendu {A_ATTENDU:.9}, z = {z_a:+.2}) ; {} ; L = {}, therm {}, {} pas, simulé à {facteur}×β, seuils |z| < 3",
        morceaux.join(" ; "),
        p.l_sc,
        p.therm_sc,
        p.pas_sc
    );
    Ok(ResultatSc { ligne: LigneGate::selon(ok, id, detail), a, sigma_a, z_a, z_p })
}

// ------------------------------------------------------------------------------------------
// G-WC : couplage faible
// ------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ResultatWc {
    pub ligne: LigneGate,
    pub c1: f64,
    pub sigma_c1: f64,
    pub c2: f64,
    pub ecart_relatif: f64,
}

/// Moindres carrés pondérés y = c₁ + c₂ x ; rend (c₁, c₂, σ(c₁), σ(c₂)).
pub fn ajuster_c1_c2(x: &[f64], y: &[f64], sy: &[f64]) -> (f64, f64, f64, f64) {
    let (mut s, mut sx, mut sxx, mut sy_, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let w = 1.0 / (sy[i] * sy[i]);
        s += w;
        sx += w * x[i];
        sxx += w * x[i] * x[i];
        sy_ += w * y[i];
        sxy += w * x[i] * y[i];
    }
    let det = s * sxx - sx * sx;
    ((sxx * sy_ - sx * sxy) / det, (s * sxy - sx * sy_) / det, (sxx / det).sqrt(), (s / det).sqrt())
}

/// G-WC (v3, v3.1) : y(β) = β⟨1 − P⟩ ajusté par c₁ + c₂/β ; PASS ssi |c₁/c₁_att − 1| ≤ 0,02 ET σ(c₁)/c₁ ≤ 0,005 ;
/// la condition de puissance non tenue donne SKIP (non vert), jamais PASS.
pub fn g_wc(ctx: &Contexte, id: &str, facteur: f64) -> Resultat<ResultatWc> {
    let p = &ctx.protocole;
    let chaines: Vec<(f64, u64)> = BETAS_WC.iter().zip(GRAINES_WC).map(|(&b, g)| (b * facteur, g)).collect();
    let xs = simuler_plaquettes(ctx.app, p.l_wc, &chaines, Depart::Froid, p.therm_wc, p.pas_wc, p.nor)?;
    let est: Vec<Estimation> = xs.iter().map(|x| estimer(x)).collect();
    let xi: Vec<f64> = BETAS_WC.iter().map(|b| 1.0 / b).collect();
    let y: Vec<f64> = BETAS_WC.iter().zip(&est).map(|(b, e)| b * (1.0 - e.moyenne)).collect();
    let sy: Vec<f64> = BETAS_WC.iter().zip(&est).map(|(b, e)| b * e.erreur).collect();
    let (c1, c2, s1, _s2) = ajuster_c1_c2(&xi, &y, &sy);
    let ecart = c1 / C1_ATTENDU - 1.0;
    let puissance = s1 / c1;
    let mut morceaux = Vec::new();
    for i in 0..4 {
        morceaux.push(format!("β={}: y = {:.5} ± {:.1e} (τ = {:.1})", BETAS_WC[i], y[i], sy[i], est[i].tau));
    }
    let detail = format!(
        "c₁ = {c1:.5} ± {s1:.1e} (attendu {C1_ATTENDU}, écart {:+.3}%, seuil 2%) ; σ(c₁)/c₁ = {:.2e} (seuil 5e-3) ; c₂ = {c2:.3} ; {} ; L = {}, therm {}, {} pas, simulé à {facteur}×β",
        100.0 * ecart,
        puissance,
        morceaux.join(" ; "),
        p.l_wc,
        p.therm_wc,
        p.pas_wc
    );
    let ligne = if puissance > 0.005 {
        LigneGate::nouvelle(Verdict::Skip, id, format!("PUISSANCE INSUFFISANTE : {detail}"))
    } else {
        LigneGate::selon(ecart.abs() <= 0.02, id, detail)
    };
    Ok(ResultatWc { ligne, c1, sigma_c1: s1, c2, ecart_relatif: ecart })
}

// ------------------------------------------------------------------------------------------
// G-UNIT, G-REG, G-HAAR-3 sur le champ du backend
// ------------------------------------------------------------------------------------------

/// G-UNIT : défaut d'unitarité max | |q| − 1 | après `balayages_unit` pas (F64 sur tout backend ; F32 sur GPU seulement).
pub fn g_unit(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let p = &ctx.protocole;
    let mut out = Vec::new();
    for (prec, id, seuil) in [(Precision::F64, "G-UNIT-F64", SEUIL_UNIT_F64), (Precision::F32, "G-UNIT-F32", SEUIL_UNIT_F32)] {
        if prec == Precision::F32 && ctx.app.is_none() {
            out.push(LigneGate::nouvelle(
                Verdict::Skip,
                id,
                "propre au backend GPU (le CPU ne calcule qu'en F64) : doit être PASS sur HIP/CUDA (GATES 0.2)",
            ));
            continue;
        }
        let mut n = su2::construire(ctx.app, lot(p.l_unit, &[(p.beta_unit, GRAINE_UNIT)], Depart::Chaud, p.nor, prec))?;
        n.init()?;
        for _ in 0..p.balayages_unit {
            un_pas(n.as_mut(), p.nor)?;
        }
        let d = n.telecharger(0)?.defaut_unitarite();
        out.push(LigneGate::selon(
            d <= seuil,
            id,
            format!("max||q|−1| = {d:.3e} après {} pas (1 HB + {} OR) ; L = {}, β = {}, graine {GRAINE_UNIT}, backend {}, seuil {seuil:.0e}", p.balayages_unit, p.nor, p.l_unit, p.beta_unit, ctx.backend()),
        ));
    }
    Ok(out)
}

/// G-REG : régression contre les valeurs archivées de `A/tests/CRITERIA.md` T3 (|z| < 3, erreur combinée).
pub fn g_reg(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let p = &ctx.protocole;
    let mut out = Vec::new();
    for (beta, graine, val, err) in ANCRES_REG {
        let id = format!("G-REG-{beta}");
        let xs = simuler_plaquettes(ctx.app, p.l_reg, &[(beta, graine)], Depart::Chaud, p.therm_reg, p.pas_reg, p.nor)?;
        let e = estimer(&xs[0]);
        let z = (e.moyenne - val) / e.erreur.hypot(err);
        out.push(LigneGate::selon(
            z.abs() < 3.0,
            &id,
            format!("β = {beta} : ⟨P⟩ = {:.7} ± {:.1e} (τ = {:.2}, n = {}) ; archivé {val:.7}({:.0}e-7) ; z = {z:+.2} (seuil 3) ; L = {}, graine {graine}, {}+{} pas, backend {}", e.moyenne, e.erreur, e.tau, e.n, err * 1e7, p.l_reg, p.therm_reg, p.pas_reg, ctx.backend()),
        ));
    }
    Ok(out)
}

/// G-HAAR-3 sur le champ du backend : le départ chaud (échantillonneur Haar du noyau, CPU ou device) donne des liens de SU(2)
/// à 1e-13 (unitarité et déterminant, par `Su2::appartient` sur la matrice du lien).
pub fn g_haar3_champ(ctx: &Contexte) -> Resultat<LigneGate> {
    let su2 = Su2::nouveau();
    let l = 8;
    let mut n = su2::construire(ctx.app, lot(l, &[(2.3, GRAINE_HAAR_3)], Depart::Chaud, 4, Precision::F64))?;
    n.init()?;
    let champ = n.telecharger(0)?;
    let (mut mu, mut md) = (0.0f64, 0.0f64);
    for li in 0..4 * champ.v() {
        let a = su2.appartient(&Su2::vers_mat(champ.lien(li)));
        mu = mu.max(a.unitarite);
        md = md.max(a.det);
    }
    Ok(LigneGate::selon(
        mu <= SEUIL_APPARTENANCE && md <= SEUIL_APPARTENANCE,
        "G-HAAR-3-CHAMP",
        format!("départ chaud L = {l} ({} liens), backend {} : max|U†U−1| = {mu:.2e} ; max|det U−1| = {md:.2e} (seuil {SEUIL_APPARTENANCE:.0e})", 4 * champ.v(), ctx.backend()),
    ))
}

// ------------------------------------------------------------------------------------------
// G-MUT : mutants (un mutant VERT = ECHEC_OUTIL)
// ------------------------------------------------------------------------------------------

fn ligne_mutant(id: &str, rouge: bool, quoi: &str) -> LigneGate {
    if rouge {
        LigneGate::nouvelle(Verdict::Pass, id, format!("mutant ROUGE comme attendu : {quoi}"))
    } else {
        LigneGate::nouvelle(Verdict::Fail, id, format!("ECHEC_OUTIL : le mutant est VERT ({quoi}) ; la gate est à réécrire avant de lire d'autres données"))
    }
}

pub fn g_mut_algebre_haar(n_haar: u64) -> Vec<LigneGate> {
    let mut out = Vec::new();
    let sans_gen = g_alg(&Su2::avec_mutant(MutantSu2::GenerateurRetire));
    let l1 = &sans_gen[0];
    out.push(ligne_mutant("G-MUT-ALG", l1.verdict == Verdict::Fail, &format!("générateur retiré → {}", l1)));
    let qr = g_haar(&Su2::avec_mutant(MutantSu2::QrNaif), n_haar);
    out.push(ligne_mutant("G-MUT-HAAR2", qr[1].verdict == Verdict::Fail, &format!("QR naïf (U(2)) → {}", qr[1])));
    out.push(ligne_mutant("G-MUT-HAAR3", qr[2].verdict == Verdict::Fail, &format!("QR naïf (U(2)) → {}", qr[2])));
    out
}

// ------------------------------------------------------------------------------------------
// Batterie complète
// ------------------------------------------------------------------------------------------

/// Batterie SU(2) complète sur le backend du contexte. L'ordre des lignes est celui de GATES v3.
pub fn batterie_su2(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let g = Su2::nouveau();
    let p = &ctx.protocole;
    let mut out: Vec<LigneGate> = Vec::new();
    out.extend(g_alg(&g).into_iter().filter(|l| ctx.veut(&l.id)));
    out.extend(g_haar(&g, p.n_haar).into_iter().filter(|l| ctx.veut(&l.id)));
    if ctx.veut("G-HAAR-3-CHAMP") {
        out.push(g_haar3_champ(ctx)?);
    }
    if ctx.veut("G-SC") {
        out.push(g_sc(ctx, "G-SC", 1.0)?.ligne);
    }
    if ctx.veut("G-WC") {
        out.push(g_wc(ctx, "G-WC", 1.0)?.ligne);
    }
    if ctx.veut("G-UNIT") {
        out.extend(g_unit(ctx)?);
    }
    if ctx.veut("G-REG") {
        out.extend(g_reg(ctx)?);
    }
    if p.mutants && ctx.veut("G-MUT") {
        out.extend(g_mut_algebre_haar(p.n_haar));
        let sc = g_sc(ctx, "G-SC-MUT", 1.05)?;
        out.push(ligne_mutant("G-MUT-SC", sc.ligne.verdict == Verdict::Fail, &format!("β × 1,05 → {}", sc.ligne)));
        let wc = g_wc(ctx, "G-WC-MUT", 1.05)?;
        out.push(ligne_mutant("G-MUT-WC", wc.ligne.verdict == Verdict::Fail, &format!("β × 1,05 → {}", wc.ligne)));
        let sc2 = g_sc(ctx, "G-SC-MUT002", 1.002)?;
        out.push(LigneGate::nouvelle(Verdict::Info, "G-MUT-SC-1.002", format!("(information, non verdict) {}", sc2.ligne)));
        let wc2 = g_wc(ctx, "G-WC-MUT002", 1.002)?;
        out.push(LigneGate::nouvelle(Verdict::Info, "G-MUT-WC-1.002", format!("(information, non verdict) {}", wc2.ligne)));
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------
// Dérivation numérique de Haar (canal « Haar numérique » de v3.1)
// ------------------------------------------------------------------------------------------

/// Intégration de Haar numérique : ⟨x²⟩ (x = Re Tr U / d_F) et ⟨x⟩_β = Σ x e^{βx} / Σ e^{βx} (repondération), avec erreur par
/// la méthode delta. Rend (⟨x²⟩, σ, [(β, ⟨x⟩_β, σ)]).
pub fn haar_pente(g: &dyn Groupe, n: u64, graine: u64, betas: &[f64]) -> (f64, f64, Vec<(f64, f64, f64)>) {
    let mut flux = Flux::nouveau(Graine(graine), ETIQUETTE_GROUPE);
    let d = g.d_f() as f64;
    let xs: Vec<f64> = (0..n).map(|_| g.haar(&mut flux).trace().0 / d).collect();
    let (m2, e2) = moments(&xs.iter().map(|x| x * x).collect::<Vec<_>>());
    let mut v = Vec::new();
    for &b in betas {
        let w: Vec<f64> = xs.iter().map(|x| (b * x).exp()).collect();
        let sw: f64 = w.iter().sum();
        let u: f64 = xs.iter().zip(&w).map(|(x, w)| x * w).sum::<f64>() / sw;
        let var = xs.iter().zip(&w).map(|(x, w)| ((x - u) * w).powi(2)).sum::<f64>() / (sw * sw);
        v.push((b, u, var.sqrt()));
    }
    (m2, e2, v)
}

// ------------------------------------------------------------------------------------------
// Tests (plomberie et propriétés ; les verdicts de la batterie gelée sont ceux de `kvu groupe gates`)
// ------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_attendu_recalcule_egale_la_constante_gelee() {
        let a = a_attendu_recalcule();
        assert!((a - A_ATTENDU).abs() < 1e-15, "a recalculé {a:.18} ≠ constante gelée {A_ATTENDU:.18}");
    }

    #[test]
    fn bessel_serie_contre_valeurs_de_reference() {
        // valeurs mpmath (60 chiffres) du script SymPy : u(0,1), u(0,2)
        assert!((u_bessel(0.1) - 0.024_989_589_839_412_66).abs() < 1e-16);
        assert!((u_bessel(0.2) - 0.049_916_874_445_944_99).abs() < 1e-16);
        assert!((bessel_i(0, 0.0) - 1.0).abs() < 1e-16);
    }

    #[test]
    fn tau_int_bruit_blanc_et_ar1() {
        let mut flux = Flux::nouveau(Graine(7), ETIQUETTE_GROUPE);
        let u = |f: &mut Flux| {
            let (i, s, k0, k1, t) = f.suivant();
            crate::heritage::su2_hb::u01(
                crate::heritage::su2_hb::philox4x32_10([i, s, 0, t], k0, k1)[0],
                crate::heritage::su2_hb::philox4x32_10([i, s, 0, t], k0, k1)[1],
            ) - 0.5
        };
        let blanc: Vec<f64> = (0..20_000).map(|_| u(&mut flux)).collect();
        let (t, _) = tau_int(&blanc, 6.0);
        assert!((t - 0.5).abs() < 0.1, "τ du bruit blanc = {t}");
        // AR(1) de coefficient 0,8 : τ = ½ (1 + 0,8)/(1 − 0,8) = 4,5
        let mut x = 0.0;
        let ar: Vec<f64> = (0..200_000).map(|_| { x = 0.8 * x + u(&mut flux); x }).collect();
        let (t, _) = tau_int(&ar, 6.0);
        assert!((t - 4.5).abs() < 0.6, "τ de l'AR(1) = {t}");
    }

    #[test]
    fn ajustement_exact_sur_un_polynome() {
        // y = 0,5 + 0,25 x : c₁ = 0,5, c₂ = 0,25 retrouvés exactement, quelle que soit la pondération
        let x = [0.02, 0.01, 0.005, 0.0025];
        let y: Vec<f64> = x.iter().map(|v| 0.5 + 0.25 * v).collect();
        let (c1, c2, s1, s2) = ajuster_c1_c2(&x, &y, &[1e-3, 2e-3, 1e-3, 3e-3]);
        assert!((c1 - 0.5).abs() < 1e-12 && (c2 - 0.25).abs() < 1e-10, "{c1} {c2}");
        assert!(s1 > 0.0 && s2 > 0.0);
    }

    #[test]
    fn le_residu_distingue_reel_et_imaginaire() {
        let g = Su2::nouveau();
        let t = g.generateurs();
        // T¹ est dans le span réel (résidu 0) mais PAS dans i·span réel ; i T¹ l'inverse : le test du span complexe, trop faible, les confondrait
        assert!(residu_dans_span(&t[0], &t, 0.5, false) < 1e-15);
        assert!((residu_dans_span(&t[0], &t, 0.5, true) - t[0].norme_frobenius()).abs() < 1e-15);
        let it = t[0].echelle_c((0.0, 1.0));
        assert!(residu_dans_span(&it, &t, 0.5, true) < 1e-15);
        assert!((residu_dans_span(&it, &t, 0.5, false) - t[0].norme_frobenius()).abs() < 1e-15);
        // une matrice non nulle de trace non nulle laisse un résidu dans les deux cas
        let id = Cmat::identite(2);
        assert!(residu_dans_span(&id, &t, 0.5, false) > 1.0 && residu_dans_span(&id, &t, 0.5, true) > 1.0);
    }

    #[test]
    fn g_alg_su2_vert_et_mutant_rouge() {
        let l = g_alg(&Su2::nouveau());
        assert!(tout_vert(&l), "{l:?}");
        let m = g_alg(&Su2::avec_mutant(MutantSu2::GenerateurRetire));
        assert_eq!(m[0].verdict, Verdict::Fail, "{m:?}");
    }

    #[test]
    fn g_haar_su2_vert_et_qr_naif_rouge() {
        let l = g_haar(&Su2::nouveau(), 200_000);
        assert!(tout_vert(&l), "{l:?}");
        let m = g_haar(&Su2::avec_mutant(MutantSu2::QrNaif), 200_000);
        assert_eq!(m[1].verdict, Verdict::Fail, "{m:?}");
        assert_eq!(m[2].verdict, Verdict::Fail, "{m:?}");
    }

    #[test]
    fn plomberie_mc_reduite_cpu() {
        // protocole réduit : ne juge PAS les verdicts physiques, vérifie que les lignes sont émises et finies
        let mut p = Protocole::gele();
        p.therm_sc = 20;
        p.pas_sc = 200;
        p.l_sc = 4;
        p.therm_wc = 20;
        p.pas_wc = 200;
        p.l_wc = 4;
        p.balayages_unit = 50;
        p.l_unit = 4;
        p.therm_reg = 20;
        p.pas_reg = 200;
        p.l_reg = 4;
        let ctx = Contexte { app: None, protocole: p, seul: vec![] };
        let sc = g_sc(&ctx, "G-SC", 1.0).unwrap();
        assert!(sc.a.is_finite() && sc.sigma_a > 0.0);
        let wc = g_wc(&ctx, "G-WC", 1.0).unwrap();
        assert!(wc.c1.is_finite() && wc.sigma_c1 > 0.0);
        let u = g_unit(&ctx).unwrap();
        assert_eq!(u[0].verdict, Verdict::Pass, "{u:?}");
        assert_eq!(u[1].verdict, Verdict::Skip, "{u:?}");
        assert_eq!(g_reg(&ctx).unwrap().len(), 2);
        assert_eq!(g_haar3_champ(&ctx).unwrap().verdict, Verdict::Pass);
    }

    #[test]
    fn haar_pente_su2_contre_bessel() {
        let (m2, e2, v) = haar_pente(&Su2::nouveau(), 200_000, GRAINE_DERIVATION_HAAR, &[0.1, 0.2]);
        assert!((m2 - 0.25).abs() < 5.0 * e2, "⟨x²⟩ = {m2} ± {e2}");
        for (b, u, s) in v {
            assert!((u - u_bessel(b)).abs() < 5.0 * s, "β = {b} : u = {u} ± {s} contre {}", u_bessel(b));
        }
    }
}
