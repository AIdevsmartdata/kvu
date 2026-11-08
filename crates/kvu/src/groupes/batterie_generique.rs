//! Batterie de gates à réponse connue des groupes génériques (Sp(4) ici ; G₂ au lot WP18). Propriétaire : WP16.
//!
//! Protocoles, seuils et graines : `GATES.md` v4.0, v4.1 et **v4.4** (gelé et haché avant tout run). Chaque gate rend une ligne
//! `PASS|FAIL|SKIP|INFO <ID> <détail>` ; les mutants rendent leur propre ligne (PASS = le mutant est ROUGE pour le critère visé,
//! puissance exigée, v4.4.2-4). Les chaînes Monte Carlo (G-SC, G-WC, G-REG et leurs mutants) sont des SÉRIES de plaquettes lues dans
//! un dossier (produites en parallèle par `kvu groupe chaine`, ou calculées ici si `calculer`) ; une ligne SKIP n'affiche aucune valeur
//! physique (règle « la puissance est jugée avant le critère »).
//! La batterie SU(2) (`gates.rs`) n'est pas touchée : seules ses fonctions publiques d'estimation sont appelées.

use super::generique::{chemin_serie, executer_chaine, lire_serie, simuler_reseau, Reseau, Serie, SpecChaine};
use super::gates::{ajuster_c1_c2, estimer, Estimation, LigneGate, Verdict};
use super::sp4::{MutantSp4, Sp4};
use super::{Cmat, Flux, Groupe, Realite, ETIQUETTE_GROUPE};
use crate::commun::{Depart, Erreur, Graine, Resultat};
use std::path::{Path, PathBuf};

// ------------------------------------------------------------------------------------------
// Constantes gelées (GATES v4.1, v4.4)
// ------------------------------------------------------------------------------------------

pub const GRAINE_HAAR_1: u64 = 26_102_616_001;
pub const GRAINE_HAAR_2: u64 = 26_102_616_002;
pub const GRAINE_HAAR_3: u64 = 26_102_616_003;
pub const GRAINE_CHAMP: u64 = 26_102_616_004;
pub const GRAINES_SC: [u64; 2] = [26_102_616_101, 26_102_616_102];
pub const BETAS_SC: [f64; 2] = [0.1, 0.2];
pub const GRAINES_WC: [u64; 4] = [26_102_616_201, 26_102_616_202, 26_102_616_203, 26_102_616_204];
pub const BETAS_WC: [f64; 4] = [50.0, 100.0, 200.0, 400.0];
pub const GRAINE_UNIT: u64 = 26_102_616_301;
/// Graine de la chaîne k (k = 1…16) de G-REG et de G-MUT-REG : 26102616400 + k.
pub const GRAINE_REG_BASE: u64 = 26_102_616_400;
pub const BETA_UNIT: f64 = 8.0;
pub const BETA_REG: f64 = 12.0;
pub const FACTEUR_MUTANT: f64 = 1.05;
pub const FACTEUR_MUTANT_INFO: f64 = 1.002;

/// v4.1 : a_attendu de G-SC, Sp(4) (ajustement exact aux points 0,1 et 0,2 de u + 4u⁵, 20 chiffres).
#[allow(clippy::excessive_precision)]
pub const A_ATTENDU: f64 = 0.062_499_997_506_418_985_25;
/// v4.1 : c₁ = d_adj/4 = 5/2.
pub const C1_ATTENDU: f64 = 2.5;
/// v4.1 : G-REG, valeur de référence de `sp4_plat_v3` et son erreur (10 blocs de 10 configurations).
pub const P_REF: f64 = 0.773_831_73;
pub const SIGMA_REF: f64 = 6.99e-6;
pub const SIGMA_REG_MAX: f64 = 1.5e-5;
pub const R_MAX_SC: f64 = 0.01;
pub const R_MAX_WC: f64 = 0.005;
/// p bilatéral à 3σ : 2 (1 − Φ(3)) (v4_student.py).
pub const P_3_SIGMA: f64 = 2.699_796_063_260_207e-3;
pub const QUANTILE_HSU_GELE: f64 = 4.094_255_3;
pub const SEUIL_ALG: f64 = 1e-12;
pub const SEUIL_APPARTENANCE: f64 = 1e-13;
pub const SEUIL_HAAR_Z: f64 = 4.0;

/// Journaux consignés de G-REG (INFO).
pub const JOURNAUX_L16: [&str; 3] = [
    "campaigns/l16_saturate_20260712/logs/sp4_plat_s91001.log",
    "campaigns/l16_saturate_20260712/logs/sp4_plat_s91002.log",
    "campaigns/l16_saturate_20260712/logs/sp4_plat_s91003.log",
];

/// Protocole numérique (valeurs gelées par défaut ; les tests de plomberie en prennent de réduites).
#[derive(Clone, Debug)]
pub struct ProtoGen {
    pub n_haar: u64,
    pub l_sc: usize,
    pub therm_sc: u64,
    pub pas_sc: u64,
    pub l_wc: usize,
    pub therm_wc: u64,
    pub pas_wc: u64,
    pub l_champ: usize,
    pub l_unit: usize,
    pub pas_unit_total: u64,
    pub therm_unit: u64,
    pub l_reg: usize,
    pub therm_reg: u64,
    pub pas_reg: u64,
    pub therm_reg_mut: u64,
    pub pas_reg_mut: u64,
    pub n_chaines_reg: usize,
    pub mutants: bool,
}

impl ProtoGen {
    pub fn gele() -> Self {
        ProtoGen {
            n_haar: 1_000_000,
            l_sc: 6,
            therm_sc: 1_000,
            pas_sc: 50_000,
            l_wc: 8,
            therm_wc: 2_000,
            pas_wc: 20_000,
            l_champ: 8,
            l_unit: 6,
            pas_unit_total: 10_000,
            therm_unit: 1_000,
            l_reg: 16,
            therm_reg: 2_000,
            pas_reg: 1_000,
            therm_reg_mut: 1_000,
            pas_reg_mut: 500,
            n_chaines_reg: 16,
            mutants: true,
        }
    }
}

/// Où se trouvent (ou se calculent) les chaînes.
#[derive(Clone, Debug)]
pub struct Calcul {
    pub dossier: PathBuf,
    /// Vrai : une série absente ou trop courte est calculée (ou prolongée) ici, en série ; faux : SKIP.
    pub calculer: bool,
    pub pas_reprise: u64,
}

pub struct ContexteGen {
    pub protocole: ProtoGen,
    pub seul: Vec<String>,
    pub calcul: Calcul,
}

impl ContexteGen {
    fn veut(&self, id: &str) -> bool {
        self.seul.is_empty() || self.seul.iter().any(|p| id.starts_with(p.as_str()))
    }
}

fn selon(ok: bool, id: &str, detail: impl Into<String>) -> LigneGate {
    LigneGate::nouvelle(if ok { Verdict::Pass } else { Verdict::Fail }, id, detail)
}

fn echec_outil(id: &str, detail: impl Into<String>) -> LigneGate {
    LigneGate::nouvelle(Verdict::Fail, id, format!("ECHEC_OUTIL : {}", detail.into()))
}

fn ligne_mutant(id: &str, rouge: bool, quoi: &str) -> LigneGate {
    if rouge {
        LigneGate::nouvelle(Verdict::Pass, id, format!("mutant ROUGE comme attendu : {quoi}"))
    } else {
        echec_outil(id, format!("le mutant n'est pas rouge pour le critère visé ({quoi}) ; la gate est à réécrire avant de lire d'autres données"))
    }
}

// ------------------------------------------------------------------------------------------
// Moments de Weyl de Sp(4), série de couplage fort exacte (≥ 30 termes), contrôle de Taylor
// ------------------------------------------------------------------------------------------

/// Moments m_k = ⟨χ^k⟩ (χ = Re Tr U) de la mesure de Haar de USp(4), k = 0…kmax, par la formule d'intégration de Weyl
/// (densité (cosθ₁ − cosθ₂)² sin²θ₁ sin²θ₂ sur le tore) ; la règle des trapèzes à `n` points par angle est EXACTE pour un
/// polynôme trigonométrique de degré < n (ici degré ≤ kmax + 4 ; n = 128 jusqu'à kmax = 100). Termes tous positifs (k pair).
pub fn moments_sp4(kmax: usize, n: usize) -> Vec<f64> {
    assert!(kmax + 4 < n, "quadrature trop courte pour le degré demandé");
    let (cs, sn): (Vec<f64>, Vec<f64>) = (0..n)
        .map(|j| {
            let t = 2.0 * std::f64::consts::PI * j as f64 / n as f64;
            (t.cos(), t.sin())
        })
        .unzip();
    let mut m = vec![0.0; kmax + 1];
    for j1 in 0..n {
        for j2 in 0..n {
            let w = (cs[j1] - cs[j2]).powi(2) * sn[j1].powi(2) * sn[j2].powi(2);
            let chi2 = (2.0 * (cs[j1] + cs[j2])).powi(2);
            let mut p = 1.0;
            for k in (0..=kmax).step_by(2) {
                m[k] += w * p;
                p *= chi2;
            }
        }
    }
    let m0 = m[0];
    m.iter().map(|x| x / m0).collect()
}

/// Coefficients de Taylor de u(β) = ⟨(χ/d) e^{βχ/d}⟩/⟨e^{βχ/d}⟩ jusqu'à l'ordre `ordre` (série en β, d = 4), à partir des moments.
pub fn taylor_u(m: &[f64], d: f64, ordre: usize) -> Vec<f64> {
    assert!(m.len() > ordre + 1);
    let fact = |n: usize| (1..=n).map(|x| x as f64).product::<f64>().max(1.0);
    let z: Vec<f64> = (0..=ordre).map(|n| m[n] / (d.powi(n as i32) * fact(n))).collect();
    let nu: Vec<f64> = (0..=ordre).map(|n| m[n + 1] / (d.powi(n as i32 + 1) * fact(n))).collect();
    let mut u = vec![0.0; ordre + 1];
    for j in 0..=ordre {
        let s: f64 = (0..j).map(|i| u[i] * z[j - i]).sum();
        u[j] = (nu[j] - s) / z[0];
    }
    u
}

/// Nombre de termes de la série en β (≥ 30 exigés par le brief ; 40).
pub const TERMES_SERIE: usize = 40;

/// u(β) par la série exacte (40 termes) ; les moments sont recalculés à chaque appel (coût négligeable).
pub fn u_sp4(beta: f64) -> f64 {
    let m = moments_sp4(TERMES_SERIE + 1, 128);
    taylor_u(&m, 4.0, TERMES_SERIE).iter().enumerate().map(|(j, c)| c * beta.powi(j as i32)).sum()
}

/// f(β) = u + 4u⁵ (v4.1).
pub fn f_sp4(beta: f64) -> f64 {
    let u = u_sp4(beta);
    u + 4.0 * u.powi(5)
}

pub fn a_ajuste(beta1: f64, f1: f64, f2: f64) -> f64 {
    (4.0 * f1 - f2) / (2.0 * beta1)
}

/// G-SC-SERIE (instrument) : coefficients de Taylor (1/16, 0, 0) à 1e-12 et a_attendu recalculé = constante gelée.
pub fn g_sc_serie() -> LigneGate {
    let m = moments_sp4(TERMES_SERIE + 1, 128);
    let c = taylor_u(&m, 4.0, TERMES_SERIE);
    let dev = (c[1] - 1.0 / 16.0).abs().max(c[2].abs()).max(c[3].abs());
    let attendus_m = [1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 14.0, 0.0, 84.0];
    let dev_m = attendus_m.iter().enumerate().map(|(k, a)| (m[k] - a).abs() / a.max(1.0)).fold(0.0f64, f64::max);
    let (f1, f2) = (f_sp4(BETAS_SC[0]), f_sp4(BETAS_SC[1]));
    let a = a_ajuste(BETAS_SC[0], f1, f2);
    let dev_a = (a - A_ATTENDU).abs();
    selon(
        dev <= 1e-12 && dev_m <= 1e-12 && dev_a <= 1e-13,
        "G-SC-SERIE",
        format!(
            "série exacte à {TERMES_SERIE} termes (moments de Weyl, quadrature 128²) : c₁ = {:.15} (1/16), |c₂| = {:.1e}, |c₃| = {:.1e} (écart max {dev:.1e}, seuil 1e-12) ; moments m₀…m₈ à {dev_m:.1e} de (1,0,1,0,3,0,14,0,84) ; a recalculé {a:.20} contre a_attendu gelé {A_ATTENDU:.20} (écart {dev_a:.1e}, seuil 1e-13)",
            c[1],
            c[2].abs(),
            c[3].abs()
        ),
    )
}

// ------------------------------------------------------------------------------------------
// Student : quantile bilatéral (bêta incomplète régularisée + bissection)
// ------------------------------------------------------------------------------------------

fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..500 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Bêta incomplète régularisée I_x(a, b).
pub fn beta_incomplete(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// P(|T| > t) pour une loi de Student à `nu` degrés de liberté.
pub fn queue_bilaterale_student(t: f64, nu: f64) -> f64 {
    beta_incomplete(nu / 2.0, 0.5, nu / (nu + t * t))
}

/// Quantile t tel que P(|T| > t) = p (bissection).
pub fn quantile_student_bilateral(p: f64, nu: f64) -> f64 {
    let (mut lo, mut hi) = (0.0f64, 200.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if queue_bilaterale_student(mid, nu) > p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

// ------------------------------------------------------------------------------------------
// G-ALG, G-ALG-DIM
// ------------------------------------------------------------------------------------------

/// Résidu de la projection de `m` sur le sous-espace RÉEL engendré par les `t` (copie de `gates::residu_dans_span`) :
/// `anti = false` ⇒ m = Σ x_c T^c ; `anti = true` ⇒ m = i Σ x_c T^c.
fn residu_dans_span(m: &Cmat, t: &[Cmat], t_f: f64, anti: bool) -> f64 {
    let mut r = m.clone();
    for tc in t {
        let (cr, ci) = m.mul(tc).trace();
        let terme = if anti { tc.echelle_c((0.0, ci / t_f)) } else { tc.echelle(cr / t_f) };
        r = r.sub(&terme);
    }
    r.norme_frobenius()
}

/// (fermeture, ad-invariance, hermiticité+trace nulle) de G-ALG-1.
pub fn alg1_mesures(g: &dyn Groupe, graine: u64) -> (f64, f64, f64) {
    let t = g.generateurs();
    let n = t.len();
    let t_f = g.t_f();
    let mut flux = Flux::nouveau(Graine(graine), ETIQUETTE_GROUPE);
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
    (ferm, adinv, herm)
}

pub fn g_alg(g: &dyn Groupe, t_f_attendu: f64) -> Vec<LigneGate> {
    let t = g.generateurs();
    let n = t.len();
    let (ferm, adinv, herm) = alg1_mesures(g, GRAINE_HAAR_1);
    let l1 = selon(
        ferm <= SEUIL_ALG && adinv <= SEUIL_ALG && herm <= SEUIL_ALG,
        "G-ALG-1",
        format!("{} : fermeture [T^a,T^b] = i f T {ferm:.3e}, ad-invariance {adinv:.3e} (50 tirages de Haar, graine {GRAINE_HAAR_1}), hermiticité et trace nulle {herm:.3e} (seuil {SEUIL_ALG:.0e}), {n} générateurs", g.nom()),
    );
    let f = g.f_abc();
    let l2 = if f.len() != n * n * n {
        echec_outil("G-ALG-2", format!("f_abc de longueur {} ≠ {n}³", f.len()))
    } else {
        let mut dev = 0.0f64;
        for a in 0..n {
            for b in 0..n {
                let s: f64 = (0..n).flat_map(|c| (0..n).map(move |d| (c, d))).map(|(c, d)| f[(a * n + c) * n + d] * f[(b * n + c) * n + d]).sum();
                let att = if a == b { g.h_vee() as f64 } else { 0.0 };
                dev = dev.max((s - att).abs());
            }
        }
        selon(dev <= SEUIL_ALG, "G-ALG-2", format!("max|f^acd f^bcd − h∨ δ| = {dev:.3e} (h∨ = {}, f calculés depuis les T de la fiche sans renormalisation, seuil {SEUIL_ALG:.0e})", g.h_vee()))
    };
    let mut dev = 0.0f64;
    for a in 0..n {
        for b in 0..n {
            let (re, im) = t[a].mul(&t[b]).trace();
            dev = dev.max((re - if a == b { t_f_attendu } else { 0.0 }).abs()).max(im.abs());
        }
    }
    let mut cas = Cmat::zeros(g.d_f());
    for ta in &t {
        cas = cas.add(&ta.mul(ta));
    }
    let c2 = cas.get(0, 0).0;
    let dev_cas = cas.sub(&Cmat::identite(g.d_f()).echelle(c2)).norme_frobenius();
    let t_f_casimir = g.d_f() as f64 * c2 / g.d_adj() as f64;
    let ok = dev <= SEUIL_ALG && dev_cas <= SEUIL_ALG && (t_f_casimir - t_f_attendu).abs() <= SEUIL_ALG;
    let l3 = selon(
        ok,
        "G-ALG-3",
        format!("max|Tr(TT) − {t_f_attendu}δ| = {dev:.3e} ; T_F mesuré {:.15} ; C₂(F) = {c2:.15} (ΣTT ∝ 1 à {dev_cas:.1e}) ; d_F C₂/d_adj = {t_f_casimir:.15} (seuil {SEUIL_ALG:.0e})", g.t_f()),
    );
    vec![l1, l2, l3]
}

fn matrice_j() -> Cmat {
    let mut j = Cmat::zeros(4);
    for (a, b, v) in [(0, 2, 1.0), (1, 3, 1.0), (2, 0, -1.0), (3, 1, -1.0)] {
        j.set(a, b, (v, 0.0));
    }
    j
}

fn transposee(m: &Cmat) -> Cmat {
    let mut t = Cmat::zeros(m.n);
    for i in 0..m.n {
        for j in 0..m.n {
            t.set(j, i, m.get(i, j));
        }
    }
    t
}

/// Valeurs singulières d'une matrice réelle (lignes × colonnes) par SVD de Jacobi unilatérale (Hestenes), sans former MᵀM.
pub fn valeurs_singulieres(mut cols: Vec<Vec<f64>>) -> Vec<f64> {
    let n = cols.len();
    for _ in 0..100 {
        let mut max_off = 0.0f64;
        for p in 0..n {
            for q in p + 1..n {
                let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
                for (x, y) in cols[p].iter().zip(&cols[q]) {
                    a += x * x;
                    b += y * y;
                    c += x * y;
                }
                if c.abs() <= 1e-300 || c.abs() <= 1e-17 * (a * b).sqrt() {
                    continue;
                }
                max_off = max_off.max(c.abs() / (a * b).sqrt());
                let zeta = (b - a) / (2.0 * c);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let (cs, sn) = (1.0 / (1.0 + t * t).sqrt(), t / (1.0 + t * t).sqrt());
                let (gauche, droite) = cols.split_at_mut(q);
                for (x, y) in gauche[p].iter_mut().zip(droite[0].iter_mut()) {
                    let (u, v) = (*x, *y);
                    *x = cs * u - sn * v;
                    *y = sn * u + cs * v;
                }
            }
        }
        if max_off < 1e-16 {
            break;
        }
    }
    let mut s: Vec<f64> = cols.iter().map(|c| c.iter().map(|x| x * x).sum::<f64>().sqrt()).collect();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    s
}

fn rang(s: &[f64], seuil_relatif: f64) -> usize {
    let max = s.first().copied().unwrap_or(0.0);
    s.iter().filter(|&&x| x > seuil_relatif * max).count()
}

/// G-ALG-DIM (v4.4.2-6).
pub fn g_alg_dim(g: &dyn Groupe) -> LigneGate {
    let j = matrice_j();
    let contrainte = |h: &Cmat| transposee(h).mul(&j).add(&j.mul(h));
    let aplatir = |m: &Cmat| -> Vec<f64> { m.vers_entrelace() };
    // base réelle de 16 hermitiennes de u(4) : E_ii ; E_ij + E_ji ; i(E_ij − E_ji)
    let mut base = Vec::new();
    for i in 0..4 {
        let mut m = Cmat::zeros(4);
        m.set(i, i, (1.0, 0.0));
        base.push(m);
    }
    for i in 0..4 {
        for k in i + 1..4 {
            let mut m = Cmat::zeros(4);
            m.set(i, k, (1.0, 0.0));
            m.set(k, i, (1.0, 0.0));
            base.push(m);
            let mut m = Cmat::zeros(4);
            m.set(i, k, (0.0, 1.0));
            m.set(k, i, (0.0, -1.0));
            base.push(m);
        }
    }
    let cols: Vec<Vec<f64>> = base.iter().map(|b| aplatir(&contrainte(b))).collect();
    let s = valeurs_singulieres(cols);
    let r = rang(&s, 1e-10);
    let noyau = base.len() - r;
    let t = g.generateurs();
    let viol = t.iter().map(|ta| contrainte(ta).norme_frobenius()).fold(0.0f64, f64::max);
    let sg = valeurs_singulieres(t.iter().map(aplatir).collect());
    let rg = rang(&sg, 1e-10);
    selon(
        noyau == 10 && viol <= 1e-12 && rg == t.len() && t.len() == 10,
        "G-ALG-DIM",
        format!("dim ker(H ↦ HᵀJ + JH) dans u(4) = {noyau} (rang {r}/16, σ_min/σ_max du rang plein {:.2e}) ; contrainte sur les {} générateurs de la fiche {viol:.2e} ; rang des générateurs {rg}", s[r - 1] / s[0], t.len()),
    )
}

// ------------------------------------------------------------------------------------------
// G-HAAR
// ------------------------------------------------------------------------------------------

fn moments(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let v = x.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / (n - 1.0);
    (m, (v / n).sqrt())
}

pub struct MesuresHaar {
    pub z1: f64,
    pub z2: f64,
    pub zf: f64,
    pub max_unite: f64,
    pub max_det: f64,
    pub max_struct: f64,
    pub moyenne_fs: f64,
    pub erreur_fs: f64,
}

pub fn haar_mesures(g: &dyn Groupe, n: u64) -> MesuresHaar {
    let (mut chi, mut chi2) = (Vec::with_capacity(n as usize), Vec::with_capacity(n as usize));
    let mut f1 = Flux::nouveau(Graine(GRAINE_HAAR_1), ETIQUETTE_GROUPE);
    for _ in 0..n {
        let (re, im) = g.haar(&mut f1).trace();
        chi.push(re);
        chi2.push(re * re + im * im);
    }
    let (m1, e1) = moments(&chi);
    let (m2, e2) = moments(&chi2);
    let mut f2 = Flux::nouveau(Graine(GRAINE_HAAR_2), ETIQUETTE_GROUPE);
    let fs: Vec<f64> = (0..n).map(|_| {
        let u = g.haar(&mut f2);
        u.mul(&u).trace().0
    }).collect();
    let (mf, ef) = moments(&fs);
    let att = match g.realite() {
        Realite::Reel => 1.0,
        Realite::Pseudoreel => -1.0,
        Realite::Complexe => 0.0,
    };
    let mut f3 = Flux::nouveau(Graine(GRAINE_HAAR_3), ETIQUETTE_GROUPE);
    let (mut mu, mut md, mut ms) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..n {
        let a = g.appartient(&g.haar(&mut f3));
        mu = mu.max(a.unitarite);
        md = md.max(a.det);
        ms = ms.max(a.structure);
    }
    MesuresHaar { z1: m1 / e1, z2: (m2 - 1.0) / e2, zf: (mf - att) / ef, max_unite: mu, max_det: md, max_struct: ms, moyenne_fs: mf, erreur_fs: ef }
}

pub fn g_haar(g: &dyn Groupe, n: u64) -> (Vec<LigneGate>, MesuresHaar) {
    let h = haar_mesures(g, n);
    let l1 = selon(
        h.z1.abs() < SEUIL_HAAR_Z && h.z2.abs() < SEUIL_HAAR_Z,
        "G-HAAR-1",
        format!("z(⟨χ⟩ = 0) = {:+.2} ; z(⟨|χ|²⟩ = 1) = {:+.2} ; n = {n}, graine {GRAINE_HAAR_1}, seuil |z| < {SEUIL_HAAR_Z}", h.z1, h.z2),
    );
    let l2 = selon(
        h.zf.abs() <= 4.0,
        "G-HAAR-2",
        format!("⟨χ(U²)⟩ = {:+.5} ± {:.1e} (attendu −1, z = {:+.2}) ; n = {n}, graine {GRAINE_HAAR_2}, seuil |z| ≤ 4", h.moyenne_fs, h.erreur_fs, h.zf),
    );
    let l3 = selon(
        h.max_unite <= SEUIL_APPARTENANCE && h.max_det <= SEUIL_APPARTENANCE && h.max_struct <= SEUIL_APPARTENANCE,
        "G-HAAR-3",
        format!("max|U†U−1| = {:.2e} ; max|det U−1| = {:.2e} ; max|UᵀJU−J| = {:.2e} ; n = {n}, graine {GRAINE_HAAR_3}, seuil {SEUIL_APPARTENANCE:.0e}", h.max_unite, h.max_det, h.max_struct),
    );
    (vec![l1, l2, l3], h)
}

pub fn g_haar3_champ(g: &Sp4, l: usize) -> LigneGate {
    let r = Reseau::<4>::nouveau(g, l, 8.0, GRAINE_CHAMP, Depart::Chaud);
    let (mut mu, mut md, mut ms) = (0.0f64, 0.0f64, 0.0f64);
    for lien in &r.liens {
        let a = g.appartient(&lien.vers_cmat());
        mu = mu.max(a.unitarite);
        md = md.max(a.det);
        ms = ms.max(a.structure);
    }
    selon(
        mu <= SEUIL_APPARTENANCE && md <= SEUIL_APPARTENANCE && ms <= SEUIL_APPARTENANCE,
        "G-HAAR-3-CHAMP",
        format!("départ chaud L = {l} ({} liens), graine {GRAINE_CHAMP} : max|U†U−1| = {mu:.2e} ; max|det U−1| = {md:.2e} ; max|UᵀJU−J| = {ms:.2e} (seuil {SEUIL_APPARTENANCE:.0e})", r.liens.len()),
    )
}

// ------------------------------------------------------------------------------------------
// Chaînes : identifiants, catalogue, obtention des séries
// ------------------------------------------------------------------------------------------

fn fmt_b(b: f64) -> String {
    format!("{b}")
}

pub fn id_sc(prefixe: &str, i: usize) -> String {
    format!("{prefixe}-b{}", fmt_b(BETAS_SC[i]))
}
pub fn id_wc(prefixe: &str, i: usize) -> String {
    format!("{prefixe}-b{}", fmt_b(BETAS_WC[i]))
}
pub fn id_reg(prefixe: &str, k: usize) -> String {
    format!("{prefixe}-{k:02}")
}

fn spec_sc(p: &ProtoGen, prefixe: &str, facteur: f64, mesures: u64) -> Vec<SpecChaine> {
    (0..2).map(|i| SpecChaine { id: id_sc(prefixe, i), l: p.l_sc, beta: BETAS_SC[i] * facteur, graine: GRAINES_SC[i], depart: Depart::Chaud, therm: p.therm_sc, mesures }).collect()
}
fn spec_wc(p: &ProtoGen, prefixe: &str, facteur: f64, mesures: u64) -> Vec<SpecChaine> {
    (0..4).map(|i| SpecChaine { id: id_wc(prefixe, i), l: p.l_wc, beta: BETAS_WC[i] * facteur, graine: GRAINES_WC[i], depart: Depart::Froid, therm: p.therm_wc, mesures }).collect()
}
fn spec_reg(p: &ProtoGen, mutant: bool, mesures: u64) -> Vec<SpecChaine> {
    (1..=p.n_chaines_reg)
        .map(|k| SpecChaine {
            id: id_reg(if mutant { "REGM" } else { "REG" }, k),
            l: p.l_reg,
            beta: if mutant { BETA_REG * FACTEUR_MUTANT } else { BETA_REG },
            graine: GRAINE_REG_BASE + k as u64,
            depart: Depart::Froid,
            therm: if mutant { p.therm_reg_mut } else { p.therm_reg },
            mesures,
        })
        .collect()
}

/// Toutes les chaînes nominales de la batterie (mutants compris si demandé), pour `kvu groupe plan` : (spec, essentiel).
/// `essentiel` = faux pour les chaînes des lignes INFO (β × 1,002).
pub fn catalogue(p: &ProtoGen) -> Vec<(SpecChaine, bool)> {
    let mut v: Vec<(SpecChaine, bool)> = Vec::new();
    v.extend(spec_reg(p, false, p.pas_reg).into_iter().map(|s| (s, true)));
    if p.mutants {
        v.extend(spec_reg(p, true, p.pas_reg_mut).into_iter().map(|s| (s, true)));
    }
    v.extend(spec_wc(p, "WC", 1.0, p.pas_wc).into_iter().map(|s| (s, true)));
    v.extend(spec_sc(p, "SC", 1.0, p.pas_sc).into_iter().map(|s| (s, true)));
    if p.mutants {
        v.extend(spec_wc(p, "WCM105", FACTEUR_MUTANT, p.pas_wc).into_iter().map(|s| (s, true)));
        v.extend(spec_sc(p, "SCM105", FACTEUR_MUTANT, p.pas_sc).into_iter().map(|s| (s, true)));
        v.extend(spec_wc(p, "WCM102", FACTEUR_MUTANT_INFO, p.pas_wc).into_iter().map(|s| (s, false)));
        v.extend(spec_sc(p, "SCM102", FACTEUR_MUTANT_INFO, p.pas_sc).into_iter().map(|s| (s, false)));
    }
    v
}

/// Retrouve la spécification d'une chaîne par son identifiant (pour `kvu groupe chaine`), avec `mesures` éventuellement forcé.
pub fn spec_par_id(p: &ProtoGen, id: &str, mesures: Option<u64>) -> Option<SpecChaine> {
    catalogue(p).into_iter().map(|(s, _)| s).find(|s| s.id == id).map(|mut s| {
        if let Some(m) = mesures {
            s.mesures = m;
        }
        s
    })
}

enum Obtenue {
    Serie(Serie),
    Manque(String),
}

fn obtenir(noyau: &Sp4, c: &Calcul, spec: &SpecChaine) -> Resultat<Obtenue> {
    if c.calculer {
        return Ok(Obtenue::Serie(executer_chaine(noyau, spec, &c.dossier, c.pas_reprise)?));
    }
    let chemin = chemin_serie(&c.dossier, &spec.id);
    if !chemin.is_file() {
        return Ok(Obtenue::Manque(format!("série {} absente de {}", spec.id, c.dossier.display())));
    }
    let s = lire_serie(&chemin)?;
    let m = &s.spec;
    if m.l != spec.l || m.beta != spec.beta || m.graine != spec.graine || m.depart != spec.depart || m.therm != spec.therm {
        return Err(Erreur::Config(format!("série {} : paramètres (L, β, graine, départ, thermalisation) différents de la gate", spec.id)));
    }
    if (s.valeurs.len() as u64) < spec.mesures {
        return Ok(Obtenue::Manque(format!("série {} incomplète : {} valeurs sur {}", spec.id, s.valeurs.len(), spec.mesures)));
    }
    Ok(Obtenue::Serie(s))
}

fn obtenir_toutes(noyau: &Sp4, c: &Calcul, specs: &[SpecChaine]) -> Resultat<Result<Vec<Serie>, String>> {
    let mut out = Vec::new();
    for s in specs {
        match obtenir(noyau, c, s)? {
            Obtenue::Serie(x) => out.push(x),
            Obtenue::Manque(m) => return Ok(Err(m)),
        }
    }
    Ok(Ok(out))
}

fn n_prime(n: u64, r: f64, r_max: f64) -> u64 {
    let brut = (n as f64 * (r / (0.8 * r_max)).powi(2)).ceil() as u64;
    brut.div_ceil(1000) * 1000
}

// ------------------------------------------------------------------------------------------
// Lignes à chaînes : résultat générique avec extension unique
// ------------------------------------------------------------------------------------------

/// Analyse d'une ligne : séries tronquées → (valeur, puissance r) ou raison d'une erreur dégénérée.
type Analyse<'a, T> = dyn Fn(&[&[f64]]) -> Result<(T, f64), String> + 'a;

/// Issue de la lecture d'une ligne à chaînes.
pub enum Issue<T> {
    /// Séries absentes, incomplètes ou puissance non établie : SKIP (sans valeur physique).
    Skip(String),
    /// Erreur dégénérée ou paramètres invalides : FAIL ECHEC_OUTIL.
    Invalide(String),
    /// Puissance satisfaite ; le critère physique est lu.
    Lue { t: T, n_lu: u64, etendue: bool },
}

/// Évalue une ligne : `specs(n)` donne les chaînes pour n pas mesurés ; `analyse(séries tronquées à n)` rend (valeur, puissance r) ou
/// Err(raison) pour une erreur dégénérée. `r_max` : plafond ; `n_nominal` : N de la gate.
fn evaluer_ligne<T>(
    noyau: &Sp4, c: &Calcul, n_nominal: u64, r_max: f64, specs: &dyn Fn(u64) -> Vec<SpecChaine>,
    analyse: &Analyse<T>,
) -> Resultat<Issue<T>> {
    let series = match obtenir_toutes(noyau, c, &specs(n_nominal))? {
        Ok(s) => s,
        Err(m) => return Ok(Issue::Skip(m)),
    };
    let tronque = |n: u64| -> Vec<&[f64]> { series.iter().map(|s| &s.valeurs[..n as usize]).collect() };
    let (t, r) = match analyse(&tronque(n_nominal)) {
        Ok(x) => x,
        Err(m) => return Ok(Issue::Invalide(m)),
    };
    if r <= r_max {
        return Ok(Issue::Lue { t, n_lu: n_nominal, etendue: false });
    }
    let np = n_prime(n_nominal, r, r_max);
    let series2 = match obtenir_toutes(noyau, c, &specs(np))? {
        Ok(s) => s,
        Err(m) => {
            return Ok(Issue::Skip(format!("PUISSANCE INSUFFISANTE sur {n_nominal} pas (r = {r:.3e} > {r_max:.3e}) ; extension unique à N′ = {np} pas par chaîne : {m}")));
        }
    };
    let tr2: Vec<&[f64]> = series2.iter().map(|s| &s.valeurs[..np as usize]).collect();
    let (t2, r2) = match analyse(&tr2) {
        Ok(x) => x,
        Err(m) => return Ok(Issue::Invalide(m)),
    };
    if r2 > r_max {
        return Ok(Issue::Skip(format!("PUISSANCE INSUFFISANTE après l'extension unique (N′ = {np}) : r = {r2:.3e} > {r_max:.3e} ; la ligne reste SKIP définitivement (v4.0)")));
    }
    Ok(Issue::Lue { t: t2, n_lu: np, etendue: true })
}

// ------------------------------------------------------------------------------------------
// G-SC
// ------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ResSc {
    pub a: f64,
    pub sigma_a: f64,
    pub z_a: f64,
    pub z_p: [f64; 2],
    pub puissance: f64,
    pub detail: String,
}

/// Erreur « nulle ou non finie » de v4.4.2-5bis, lue comme nulle à la précision machine (σ ≤ 1e-14 |⟨P⟩|) : une série constante
/// donne un σ de l'ordre de 1e-17 par arrondi, pas exactement 0.
fn erreur_degeneree(e: &Estimation) -> bool {
    !(e.erreur.is_finite() && e.moyenne.is_finite()) || e.erreur <= 1e-14 * e.moyenne.abs()
}

fn analyse_sc(series: &[&[f64]], l: usize, therm: u64) -> Result<(ResSc, f64), String> {
    let est: Vec<Estimation> = series.iter().map(|x| estimer(x)).collect();
    if est.iter().any(erreur_degeneree) {
        return Err(format!("erreur de série dégénérée (σ = {:?})", est.iter().map(|e| e.erreur).collect::<Vec<_>>()));
    }
    let a = a_ajuste(BETAS_SC[0], est[0].moyenne, est[1].moyenne);
    let sigma_a = (400.0 * est[0].erreur.powi(2) + 25.0 * est[1].erreur.powi(2)).sqrt();
    let z_a = (a - A_ATTENDU) / sigma_a;
    let mut z_p = [0.0; 2];
    let mut morceaux = Vec::new();
    for i in 0..2 {
        let att = f_sp4(BETAS_SC[i]);
        z_p[i] = (est[i].moyenne - att) / est[i].erreur;
        morceaux.push(format!("β={}: ⟨P⟩ = {:.8} ± {:.1e} (τ = {:.2}) attendu {:.8} z = {:+.2}", BETAS_SC[i], est[i].moyenne, est[i].erreur, est[i].tau, att, z_p[i]));
    }
    let puissance = sigma_a / A_ATTENDU;
    let detail = format!(
        "a = {a:.7} ± {sigma_a:.1e} (attendu {A_ATTENDU:.10}, z = {z_a:+.2}) ; {} ; σ_a/a = {puissance:.2e} (plafond {R_MAX_SC}) ; L = {l}, therm {therm}, {} pas",
        morceaux.join(" ; "),
        series[0].len()
    );
    Ok((ResSc { a, sigma_a, z_a, z_p, puissance, detail }, puissance))
}

pub fn g_sc(noyau: &Sp4, c: &Calcul, p: &ProtoGen, prefixe: &str, facteur: f64) -> Resultat<Issue<ResSc>> {
    evaluer_ligne(noyau, c, p.pas_sc, R_MAX_SC, &|n| spec_sc(p, prefixe, facteur, n), &|s| analyse_sc(s, p.l_sc, p.therm_sc))
}

fn ligne_sc(id: &str, i: &Issue<ResSc>) -> LigneGate {
    match i {
        Issue::Skip(m) => LigneGate::nouvelle(Verdict::Skip, id, m.clone()),
        Issue::Invalide(m) => echec_outil(id, m.clone()),
        Issue::Lue { t, n_lu, etendue } => {
            let ok = t.z_a.abs() < 3.0 && t.z_p.iter().all(|z| z.abs() < 3.0);
            selon(ok, id, format!("{} ; N lus = {n_lu}{} ; seuils |z| < 3", t.detail, if *etendue { " (extension unique v4.0)" } else { "" }))
        }
    }
}

// ------------------------------------------------------------------------------------------
// G-WC
// ------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ResWc {
    pub c1: f64,
    pub sigma_c1: f64,
    pub ecart: f64,
    pub detail: String,
    pub serre: String,
}

fn analyse_wc(series: &[&[f64]], l: usize, therm: u64) -> Result<(ResWc, f64), String> {
    let est: Vec<Estimation> = series.iter().map(|x| estimer(x)).collect();
    if est.iter().any(erreur_degeneree) {
        return Err("erreur de série dégénérée".into());
    }
    let xi: Vec<f64> = BETAS_WC.iter().map(|b| 1.0 / b).collect();
    let y: Vec<f64> = BETAS_WC.iter().zip(&est).map(|(b, e)| b * (1.0 - e.moyenne)).collect();
    let sy: Vec<f64> = BETAS_WC.iter().zip(&est).map(|(b, e)| b * e.erreur).collect();
    let (c1, c2, s1, _) = ajuster_c1_c2(&xi, &y, &sy);
    if !(c1.is_finite() && c1 > 0.0 && s1.is_finite() && s1 > 1e-14 * c1) {
        return Err(format!("c₁ = {c1} ou σ(c₁) = {s1} invalide (c₁ > 0 et σ > 0 exigés)"));
    }
    let ecart = c1 / C1_ATTENDU - 1.0;
    let puissance = s1 / c1;
    let v = (l as f64).powi(4);
    let serre_val = (c1 / (C1_ATTENDU * (1.0 - 1.0 / v)) - 1.0).abs();
    let serre_seuil = (4.0 * puissance).max(0.002);
    let morceaux: Vec<String> = (0..4).map(|i| format!("β={}: y = {:.5} ± {:.1e} (τ = {:.1})", BETAS_WC[i], y[i], sy[i], est[i].tau)).collect();
    let detail = format!(
        "c₁ = {c1:.5} ± {s1:.1e} (attendu {C1_ATTENDU}, écart {:+.3}%, seuil 2%) ; σ(c₁)/c₁ = {puissance:.2e} (plafond {R_MAX_WC}) ; c₂ = {c2:.3} ; {} ; L = {l}, therm {therm}, {} pas",
        100.0 * ecart,
        morceaux.join(" ; "),
        series[0].len()
    );
    let serre = format!("|c₁/(c₁_attendu (1 − 1/V)) − 1| = {serre_val:.3e} contre max(4σ(c₁)/c₁, 0,002) = {serre_seuil:.3e} (V = {l}⁴ ; non un verdict)");
    Ok((ResWc { c1, sigma_c1: s1, ecart, detail, serre }, puissance))
}

pub fn g_wc(noyau: &Sp4, c: &Calcul, p: &ProtoGen, prefixe: &str, facteur: f64) -> Resultat<Issue<ResWc>> {
    evaluer_ligne(noyau, c, p.pas_wc, R_MAX_WC, &|n| spec_wc(p, prefixe, facteur, n), &|s| analyse_wc(s, p.l_wc, p.therm_wc))
}

fn ligne_wc(id: &str, i: &Issue<ResWc>) -> LigneGate {
    match i {
        Issue::Skip(m) => LigneGate::nouvelle(Verdict::Skip, id, m.clone()),
        Issue::Invalide(m) => echec_outil(id, m.clone()),
        Issue::Lue { t, n_lu, etendue } => selon(t.ecart.abs() <= 0.02, id, format!("{} ; N lus = {n_lu}{}", t.detail, if *etendue { " (extension unique v4.0)" } else { "" })),
    }
}

// ------------------------------------------------------------------------------------------
// G-REG
// ------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ResReg {
    pub p: f64,
    pub sigma: f64,
    pub t: f64,
    pub nu_ws: f64,
    pub detail: String,
}

/// Seuil de Hsu : quantile de Student bilatéral à 3σ pour ν = min(15, 9) = 9, calculé dans le code (non arrondi).
pub fn seuil_hsu() -> f64 {
    quantile_student_bilateral(P_3_SIGMA, 9.0)
}

fn analyse_reg(series: &[&[f64]], l: usize, beta: f64) -> Result<(ResReg, f64), String> {
    let n = series.len() as f64;
    let m: Vec<f64> = series.iter().map(|s| s.iter().sum::<f64>() / s.len() as f64).collect();
    let p = m.iter().sum::<f64>() / n;
    let sigma = (m.iter().map(|x| (x - p) * (x - p)).sum::<f64>() / (n * (n - 1.0))).sqrt();
    if !(sigma.is_finite() && sigma > 1e-14 * p.abs() && p.is_finite()) {
        return Err(format!("erreur de G-REG dégénérée (σ = {sigma})"));
    }
    let t = (p - P_REF) / (sigma * sigma + SIGMA_REF * SIGMA_REF).sqrt();
    let nu_ws = (sigma * sigma + SIGMA_REF * SIGMA_REF).powi(2) / (sigma.powi(4) / (n - 1.0) + SIGMA_REF.powi(4) / 9.0);
    let q = seuil_hsu();
    let detail = format!(
        "⟨P⟩ = {p:.8} ± {sigma:.2e} (jackknife {} chaînes, N = {} pas par chaîne, L = {l}, β = {beta}) ; référence sp4_plat_v3 {P_REF} ± {SIGMA_REF:.2e} ; t = {t:+.3} (seuil de Hsu {q:.7}, ν = 9) ; σ ≤ {SIGMA_REG_MAX:.1e} exigé ; ν_WS = {nu_ws:.1} (INFO) ; la calibration ne vaut que sous le modèle déclaré (moyennes gaussiennes, variances estimées indépendantes) ; volume 16 contre 32 non corrigé",
        series.len(),
        series[0].len()
    );
    Ok((ResReg { p, sigma, t, nu_ws, detail }, sigma))
}

pub fn g_reg(noyau: &Sp4, c: &Calcul, p: &ProtoGen, mutant: bool) -> Resultat<Issue<ResReg>> {
    let n = if mutant { p.pas_reg_mut } else { p.pas_reg };
    let beta = if mutant { BETA_REG * FACTEUR_MUTANT } else { BETA_REG };
    evaluer_ligne(noyau, c, n, SIGMA_REG_MAX, &|m| spec_reg(p, mutant, m), &|s| analyse_reg(s, p.l_reg, beta))
}

/// INFO : valeurs `P=` consignées dans les journaux L = 16 (v4.4.2-3).
pub fn info_anciens_l16() -> LigneGate {
    let mut vals = Vec::new();
    let mut absents = 0;
    for f in JOURNAUX_L16 {
        match std::fs::read_to_string(f) {
            Ok(t) => {
                for l in t.lines() {
                    if let Some(i) = l.find("P=") {
                        let s: String = l[i + 2..].chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
                        if let Ok(v) = s.parse::<f64>() {
                            vals.push(v);
                        }
                    }
                }
            }
            Err(_) => absents += 1,
        }
    }
    if vals.is_empty() {
        return LigneGate::nouvelle(Verdict::Info, "G-REG-L16", format!("aucune valeur P= trouvée ({absents} journaux absents) ; non un verdict"));
    }
    let m = vals.iter().sum::<f64>() / vals.len() as f64;
    LigneGate::nouvelle(
        Verdict::Info,
        "G-REG-L16",
        format!("{} plaquettes consignées à L = 16, β = 12 (sp4_plat_s9100{{1,2,3}}.log) : {} ; moyenne {m:.6} ; non un verdict", vals.len(), vals.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(", ")),
    )
}

fn ligne_reg(id: &str, i: &Issue<ResReg>) -> LigneGate {
    match i {
        Issue::Skip(m) => LigneGate::nouvelle(Verdict::Skip, id, m.clone()),
        Issue::Invalide(m) => echec_outil(id, m.clone()),
        Issue::Lue { t, n_lu, etendue } => {
            let ok = t.t.abs() < seuil_hsu();
            let suffixe = if *etendue { " (extension unique v4.0)" } else { "" };
            if ok {
                LigneGate::nouvelle(Verdict::Pass, id, format!("{} ; N lus = {n_lu}{suffixe}", t.detail))
            } else {
                LigneGate::nouvelle(Verdict::Fail, id, format!("FAIL-REG, référence (C) : {} ; N lus = {n_lu}{suffixe} ; ouvre un diagnostic, ne dit pas seul lequel des deux codes se trompe", t.detail))
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// G-UNIT
// ------------------------------------------------------------------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResUnit {
    pub defaut: f64,
    pub avant_reproj: f64,
    pub l: usize,
    pub beta: f64,
    pub graine: u64,
    pub pas: u64,
    pub eps: f64,
}

pub fn chemin_unit(dossier: &Path) -> PathBuf {
    dossier.join("UNIT.res.json")
}

/// Calcule (ou relit) G-UNIT : 10⁴ pas (1 000 de thermalisation), mesure sur le champ final.
pub fn calculer_unit(noyau: &Sp4, c: &Calcul, p: &ProtoGen) -> Resultat<Option<ResUnit>> {
    let chemin = chemin_unit(&c.dossier);
    if let Ok(t) = std::fs::read_to_string(&chemin) {
        let r: ResUnit = serde_json::from_str(&t).map_err(|e| Erreur::Format(format!("{} : {e}", chemin.display())))?;
        if r.l == p.l_unit && r.beta == BETA_UNIT && r.graine == GRAINE_UNIT && r.pas == p.pas_unit_total {
            return Ok(Some(r));
        }
        return Err(Erreur::Config(format!("{} : paramètres différents de la gate", chemin.display())));
    }
    if !c.calculer {
        return Ok(None);
    }
    let r = simuler_reseau(noyau, p.l_unit, BETA_UNIT, GRAINE_UNIT, Depart::Chaud, p.therm_unit, p.pas_unit_total - p.therm_unit, true);
    let res = ResUnit { defaut: r.defaut_max(noyau), avant_reproj: r.defaut_avant_reproj, l: p.l_unit, beta: BETA_UNIT, graine: GRAINE_UNIT, pas: p.pas_unit_total, eps: r.eps };
    std::fs::create_dir_all(&c.dossier).map_err(|e| Erreur::io("création du dossier des chaînes", e))?;
    std::fs::write(&chemin, serde_json::to_string(&res).map_err(|e| Erreur::Format(e.to_string()))?).map_err(|e| Erreur::io("écriture UNIT", e))?;
    Ok(Some(res))
}

// ------------------------------------------------------------------------------------------
// Batterie complète
// ------------------------------------------------------------------------------------------

fn mutant_stat(id: &str, rouge: bool, quoi: String) -> LigneGate {
    ligne_mutant(id, rouge, &quoi)
}

fn memo<T>(slot: &mut Option<T>, f: impl FnOnce() -> Resultat<T>) -> Resultat<&T> {
    if slot.is_none() {
        *slot = Some(f()?);
    }
    slot.as_ref().ok_or_else(|| Erreur::Config("état interne : mémo vide".into()))
}

/// Batterie Sp(4) complète sur CPU. L'ordre des lignes est celui de GATES v4.1. `--seul` filtre les lignes émises par préfixe ;
/// les calculs dont dépend une ligne demandée (G-SC pour G-MUT-SC, etc.) sont faits même s'ils ne sont pas affichés.
pub fn batterie_sp4(ctx: &ContexteGen) -> Resultat<Vec<LigneGate>> {
    let g = Sp4::nouveau();
    let p = &ctx.protocole;
    let c = &ctx.calcul;
    let mut out: Vec<LigneGate> = Vec::new();
    let emet = |l: LigneGate, out: &mut Vec<LigneGate>| {
        if ctx.veut(&l.id) {
            out.push(l);
        }
    };
    // vrai si une ligne dont l'identifiant commence par `prefixe` (ou en est un préfixe) est demandée
    let touche = |prefixe: &str| ctx.seul.is_empty() || ctx.seul.iter().any(|q| prefixe.starts_with(q.as_str()) || q.starts_with(prefixe));
    let mutants = p.mutants && touche("G-MUT");

    let fiche_ok = g.d_f() == 4 && g.d_adj() == 10 && g.h_vee() == 3 && g.realite() == Realite::Pseudoreel && g.generateurs().len() == 10;
    emet(selon(fiche_ok, "Fiche", format!("sp4 : d_F = {}, d_adj = {}, h∨ = {}, pseudoréel, {} générateurs normalisés une fois (T_F gelé ½, aucune gate ne les renormalise)", g.d_f(), g.d_adj(), g.h_vee(), g.generateurs().len())), &mut out);
    for l in g_alg(&g, 0.5) {
        emet(l, &mut out);
    }
    emet(g_alg_dim(&g), &mut out);
    if touche("G-HAAR-1") || touche("G-HAAR-2") || touche("G-HAAR-3") {
        let (lignes, _) = g_haar(&g, p.n_haar);
        for l in lignes {
            emet(l, &mut out);
        }
    }
    if touche("G-HAAR-3-CHAMP") {
        emet(g_haar3_champ(&g, p.l_champ), &mut out);
    }
    if touche("G-SC-SERIE") {
        emet(g_sc_serie(), &mut out);
    }

    let mut sc: Option<Issue<ResSc>> = None;
    let mut wc: Option<Issue<ResWc>> = None;
    let mut reg: Option<Issue<ResReg>> = None;
    if ctx.veut("G-SC") {
        emet(ligne_sc("G-SC", memo(&mut sc, || g_sc(&g, c, p, "SC", 1.0))?), &mut out);
    }
    if ctx.veut("G-WC") {
        let w = memo(&mut wc, || g_wc(&g, c, p, "WC", 1.0))?;
        emet(ligne_wc("G-WC", w), &mut out);
        if let Issue::Lue { t, .. } = w {
            emet(LigneGate::nouvelle(Verdict::Info, "G-WC-SERRE", t.serre.clone()), &mut out);
        }
    }
    if ctx.veut("G-UNIT") {
        match calculer_unit(&g, c, p)? {
            None => emet(LigneGate::nouvelle(Verdict::Skip, "G-UNIT", format!("résultat absent de {} et calcul non demandé", c.dossier.display())), &mut out),
            Some(r) => {
                emet(selon(r.defaut <= SEUIL_APPARTENANCE, "G-UNIT", format!("max(|U†U−1|, |UᵀJU−J|) = {:.3e} après {} pas (dont {} de thermalisation) ; L = {}, β = {}, graine {GRAINE_UNIT}, ε final {:.5}, seuil {SEUIL_APPARTENANCE:.0e}", r.defaut, r.pas, p.therm_unit, r.l, r.beta, r.eps)), &mut out);
                emet(LigneGate::nouvelle(Verdict::Info, "G-UNIT-AVANT-REPROJ", format!("défaut maximal des liens du dernier pas AVANT ré-unitarisation : {:.3e} (non un verdict)", r.avant_reproj)), &mut out);
            }
        }
    }
    if ctx.veut("G-REG") {
        emet(ligne_reg("G-REG", memo(&mut reg, || g_reg(&g, c, p, false))?), &mut out);
        emet(info_anciens_l16(), &mut out);
    }

    if mutants {
        if touche("G-MUT-ALG") {
            let sans = Sp4::avec_mutant(MutantSp4::GenerateurRetire);
            let (ferm, _adinv, herm) = alg1_mesures(&sans, GRAINE_HAAR_1);
            emet(ligne_mutant("G-MUT-ALG", ferm > SEUIL_ALG && herm <= SEUIL_ALG, &format!("générateur d'indice 9 retiré → fermeture {ferm:.3e} (seuil {SEUIL_ALG:.0e}), hermiticité {herm:.1e}")), &mut out);
        }
        if touche("G-MUT-HAAR") {
            let qr = Sp4::avec_mutant(MutantSp4::QrNaif);
            let h = haar_mesures(&qr, p.n_haar);
            emet(ligne_mutant("G-MUT-HAAR2", h.zf.is_finite() && h.zf.abs() > 4.0, &format!("QR naïf (U(4)) → ⟨χ(U²)⟩ = {:+.5} ± {:.1e}, z(FS = −1) = {:+.1}", h.moyenne_fs, h.erreur_fs, h.zf)), &mut out);
            emet(ligne_mutant("G-MUT-HAAR3", h.max_struct > SEUIL_APPARTENANCE, &format!("QR naïf (U(4)) → max|UᵀJU−J| = {:.2e}, max|det−1| = {:.2e} (seuil {SEUIL_APPARTENANCE:.0e})", h.max_struct, h.max_det)), &mut out);
        }
        if touche("G-MUT-SC") {
            let nom = memo(&mut sc, || g_sc(&g, c, p, "SC", 1.0))?;
            let mu = g_sc(&g, c, p, "SCM105", FACTEUR_MUTANT)?;
            emet(mutant_sc("G-MUT-SC", nom, &mu), &mut out);
        }
        if touche("G-MUT-WC") {
            let nom = memo(&mut wc, || g_wc(&g, c, p, "WC", 1.0))?;
            let mu = g_wc(&g, c, p, "WCM105", FACTEUR_MUTANT)?;
            emet(mutant_wc("G-MUT-WC", nom, &mu), &mut out);
        }
        if touche("G-MUT-REG") {
            let nom = memo(&mut reg, || g_reg(&g, c, p, false))?;
            let mu = g_reg(&g, c, p, true)?;
            emet(mutant_reg("G-MUT-REG", nom, &mu), &mut out);
        }
        if touche("G-MUT-SC-1.002") {
            let i = g_sc(&g, c, p, "SCM102", FACTEUR_MUTANT_INFO)?;
            emet(LigneGate::nouvelle(Verdict::Info, "G-MUT-SC-1.002", format!("(information, non verdict) {}", ligne_sc("G-MUT-SC-1.002", &i))), &mut out);
        }
        if touche("G-MUT-WC-1.002") {
            let i = g_wc(&g, c, p, "WCM102", FACTEUR_MUTANT_INFO)?;
            emet(LigneGate::nouvelle(Verdict::Info, "G-MUT-WC-1.002", format!("(information, non verdict) {}", ligne_wc("G-MUT-WC-1.002", &i))), &mut out);
        }
    }
    Ok(out)
}

fn conclu<T>(i: &Issue<T>) -> Option<&T> {
    match i {
        Issue::Lue { t, .. } => Some(t),
        _ => None,
    }
}

fn mutant_sc(id: &str, nominal: &Issue<ResSc>, mutant: &Issue<ResSc>) -> LigneGate {
    match (conclu(nominal), mutant) {
        (None, _) => LigneGate::nouvelle(Verdict::Skip, id, "non concluant : la puissance du run nominal de G-SC n'est pas établie (v4.4.2-4)"),
        (Some(_), Issue::Skip(m)) => LigneGate::nouvelle(Verdict::Skip, id, format!("non concluant : {m}")),
        (Some(_), Issue::Invalide(m)) => echec_outil(id, format!("mutant : {m}")),
        (Some(_), Issue::Lue { t, .. }) => mutant_stat(id, t.z_a.is_finite() && t.z_a.abs() >= 3.0, format!("β × 1,05 → a = {:.7} ± {:.1e}, z_a = {:+.1} (critère |z_a| ≥ 3) ; σ_a/a = {:.2e}", t.a, t.sigma_a, t.z_a, t.puissance)),
    }
}

fn mutant_wc(id: &str, nominal: &Issue<ResWc>, mutant: &Issue<ResWc>) -> LigneGate {
    match (conclu(nominal), mutant) {
        (None, _) => LigneGate::nouvelle(Verdict::Skip, id, "non concluant : la puissance du run nominal de G-WC n'est pas établie (v4.4.2-4)"),
        (Some(_), Issue::Skip(m)) => LigneGate::nouvelle(Verdict::Skip, id, format!("non concluant : {m}")),
        (Some(_), Issue::Invalide(m)) => echec_outil(id, format!("mutant : {m}")),
        (Some(_), Issue::Lue { t, .. }) => mutant_stat(id, t.ecart.abs() > 0.02, format!("β × 1,05 → c₁ = {:.5} ± {:.1e}, écart {:+.2}% (critère > 2 %) ; σ(c₁)/c₁ = {:.2e}", t.c1, t.sigma_c1, 100.0 * t.ecart, t.sigma_c1 / t.c1)),
    }
}

fn mutant_reg(id: &str, nominal: &Issue<ResReg>, mutant: &Issue<ResReg>) -> LigneGate {
    // G-MUT-REG exige la puissance du mutant (σ ≤ 1,5e-5) ; la puissance du nominal est celle de G-REG, requise aussi.
    match (conclu(nominal), mutant) {
        (None, _) => LigneGate::nouvelle(Verdict::Skip, id, "non concluant : la puissance du run nominal de G-REG n'est pas établie (v4.4.2-4)"),
        (Some(_), Issue::Skip(m)) => LigneGate::nouvelle(Verdict::Skip, id, format!("non concluant : {m}")),
        (Some(_), Issue::Invalide(m)) => echec_outil(id, format!("mutant : {m}")),
        (Some(_), Issue::Lue { t, .. }) => mutant_stat(id, t.t.is_finite() && t.t.abs() >= seuil_hsu(), format!("β × 1,05 = 12,6 → ⟨P⟩ = {:.8} ± {:.2e}, t = {:+.1} (critère |t| ≥ {:.7})", t.p, t.sigma, t.t, seuil_hsu())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments_de_weyl_sp4() {
        let m = moments_sp4(20, 128);
        let att = [1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 14.0, 0.0, 84.0, 0.0, 594.0, 0.0, 4719.0, 0.0, 40898.0, 0.0, 379236.0, 0.0, 3711916.0, 0.0, 37975756.0];
        for k in 0..=20 {
            assert!((m[k] - att[k]).abs() <= 1e-12 * att[k].max(1.0), "m_{k} = {} contre {}", m[k], att[k]);
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn serie_exacte_contre_valeurs_gelees() {
        let l = g_sc_serie();
        assert_eq!(l.verdict, Verdict::Pass, "{l}");
        // v4_sympy.out : u(0,1) et u(0,2) (25 chiffres, mpmath)
        let d1 = (u_sp4(0.1) - 0.006_249_999_979_657_067_098).abs();
        assert!(d1 < 1e-15, "écart u(0,1) {d1:e}");
        let d2 = (u_sp4(0.2) - 0.012_499_999_349_229_552_245).abs();
        assert!(d2 < 1e-15, "écart u(0,2) {d2:e}");
    }

    #[test]
    fn student_quantile_hsu() {
        let q = seuil_hsu();
        assert!((q - QUANTILE_HSU_GELE).abs() < 1e-6, "quantile {q}");
        assert!((quantile_student_bilateral(P_3_SIGMA, 15.0) - 3.5864232).abs() < 1e-6);
        assert!((quantile_student_bilateral(P_3_SIGMA, 31.0) - 3.2609419).abs() < 1e-6);
        assert!((quantile_student_bilateral(P_3_SIGMA, 4.0) - 6.620206).abs() < 1e-5);
    }

    #[test]
    fn extension_n_prime() {
        // r = 2 r_max, N = 10000 : N' = ceil(10000 · (2/0,8)²) = 62500 → 63000
        assert_eq!(n_prime(10_000, 2.0, 1.0), 63_000);
        assert_eq!(n_prime(1_000, 1.6, 1.0), 4_000);
    }

    #[test]
    fn svd_de_jacobi_rang() {
        let cols = vec![vec![1.0, 0.0, 0.0], vec![0.0, 2.0, 0.0], vec![1.0, 2.0, 0.0]];
        let s = valeurs_singulieres(cols);
        assert_eq!(rang(&s, 1e-10), 2);
    }

    // ---- logique des lignes à chaînes, sur des séries FABRIQUÉES à la main (aucune simulation) ----

    fn dossier_test(nom: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kvu-wp16-bat-{nom}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn ecrire_fausse(d: &Path, spec: &SpecChaine, valeurs: &[f64]) {
        let mut t = format!(
            "# id = {}\n# groupe = sp4\n# L = {}\n# beta = {:.17e}\n# graine = {}\n# depart = {}\n# therm = {}\n# mesures = {}\n# eps = {:.17e}\n",
            spec.id, spec.l, spec.beta, spec.graine, if spec.depart == Depart::Chaud { "chaud" } else { "froid" }, spec.therm, valeurs.len(), 0.3
        );
        for v in valeurs {
            t.push_str(&format!("{v:.17e}\n"));
        }
        t.push_str(&format!("# fin n = {}\n", valeurs.len()));
        std::fs::write(chemin_serie(d, &spec.id), t).unwrap();
    }

    /// Bruit déterministe de moyenne nulle et d'écart ~ amp (somme de sinus incommensurables).
    fn bruit(n: usize, amp: f64, phase: f64) -> Vec<f64> {
        (0..n).map(|i| amp * ((i as f64 * 2.399 + phase).sin() + (i as f64 * 0.7311 + 2.0 * phase).cos())).collect()
    }

    fn proto_test() -> ProtoGen {
        let mut p = ProtoGen::gele();
        p.pas_sc = 4000;
        p.pas_wc = 4000;
        p.pas_reg = 400;
        p.pas_reg_mut = 200;
        p.n_chaines_reg = 4;
        p.l_reg = 4;
        p
    }

    fn calcul_test(d: &Path) -> Calcul {
        Calcul { dossier: d.to_path_buf(), calculer: false, pas_reprise: 0 }
    }

    #[test]
    fn lignes_sc_wc_reg_sur_series_fabriquees() {
        let d = dossier_test("lignes");
        let p = proto_test();
        let g = Sp4::nouveau();
        let c = calcul_test(&d);
        // moyennes exactes : SC (f(β)), WC (c₁ = 5/2 exactement, c₂ = 0), REG (P_REF) ; mutants : β × 1,05 pour SC et WC, P(12,6) pour REG
        for (facteur, prefixe, n) in [(1.0, "SC", p.pas_sc), (FACTEUR_MUTANT, "SCM105", p.pas_sc)] {
            for (i, spec) in spec_sc(&p, prefixe, facteur, n).iter().enumerate() {
                let m = f_sp4(spec.beta);
                let v: Vec<f64> = bruit(n as usize, 1e-7, i as f64).iter().map(|x| m + x).collect();
                ecrire_fausse(&d, spec, &v);
            }
        }
        for (facteur, prefixe, n) in [(1.0, "WC", p.pas_wc), (FACTEUR_MUTANT, "WCM105", p.pas_wc)] {
            for (i, spec) in spec_wc(&p, prefixe, facteur, n).iter().enumerate() {
                let beta = spec.beta;
                let m = 1.0 - 2.5 / beta;
                let v: Vec<f64> = bruit(n as usize, 1e-7, i as f64).iter().map(|x| m + x).collect();
                ecrire_fausse(&d, spec, &v);
            }
        }
        for (mutant, n, centre) in [(false, p.pas_reg, P_REF), (true, p.pas_reg_mut, 0.7846)] {
            for (k, spec) in spec_reg(&p, mutant, n).iter().enumerate() {
                let v: Vec<f64> = bruit(n as usize, 1e-6, k as f64).iter().map(|x| centre + 3e-6 * k as f64 + x).collect();
                ecrire_fausse(&d, spec, &v);
            }
        }
        let sc = g_sc(&g, &c, &p, "SC", 1.0).unwrap();
        let l = ligne_sc("G-SC", &sc);
        assert_eq!(l.verdict, Verdict::Pass, "{l}");
        let sc_m = g_sc(&g, &c, &p, "SCM105", FACTEUR_MUTANT).unwrap();
        let l = mutant_sc("G-MUT-SC", &sc, &sc_m);
        assert_eq!(l.verdict, Verdict::Pass, "{l}");
        let wc = g_wc(&g, &c, &p, "WC", 1.0).unwrap();
        let l = ligne_wc("G-WC", &wc);
        assert_eq!(l.verdict, Verdict::Pass, "{l}");
        let wc_m = g_wc(&g, &c, &p, "WCM105", FACTEUR_MUTANT).unwrap();
        let l = mutant_wc("G-MUT-WC", &wc, &wc_m);
        assert_eq!(l.verdict, Verdict::Pass, "{l}");
        let reg = g_reg(&g, &c, &p, false).unwrap();
        let l = ligne_reg("G-REG", &reg);
        assert_eq!(l.verdict, Verdict::Pass, "{l}");
        let reg_m = g_reg(&g, &c, &p, true).unwrap();
        let l = mutant_reg("G-MUT-REG", &reg, &reg_m);
        assert_eq!(l.verdict, Verdict::Pass, "{l}");
        // un mutant dont le nominal n'est pas concluant est SKIP
        let skip: Issue<ResSc> = Issue::Skip("x".into());
        assert_eq!(mutant_sc("G-MUT-SC", &skip, &sc_m).verdict, Verdict::Skip);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn puissance_insuffisante_donne_skip_sans_valeur_physique_et_extension_unique() {
        let d = dossier_test("puissance");
        let p = proto_test();
        let g = Sp4::nouveau();
        let c = calcul_test(&d);
        // séries bruitées (σ_a/a ≫ 1 %) : SKIP, avec N′ annoncé, aucune valeur z affichée
        for (i, spec) in spec_sc(&p, "SC", 1.0, p.pas_sc).iter().enumerate() {
            let m = f_sp4(spec.beta);
            let v: Vec<f64> = bruit(p.pas_sc as usize, 4e-3, i as f64).iter().map(|x| m + x).collect();
            ecrire_fausse(&d, spec, &v);
        }
        let sc = g_sc(&g, &c, &p, "SC", 1.0).unwrap();
        let l = ligne_sc("G-SC", &sc);
        assert_eq!(l.verdict, Verdict::Skip, "{l}");
        assert!(l.detail.contains("N′ = ") && !l.detail.contains("z ="), "{l}");
        // extension : les séries sont prolongées à N′ valeurs (fabriquées) : la ligne est alors lue une seule fois
        let np: u64 = l.detail.split("N′ = ").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap();
        assert!(np > p.pas_sc && np.is_multiple_of(1000));
        for (i, spec) in spec_sc(&p, "SC", 1.0, np).iter().enumerate() {
            let m = f_sp4(spec.beta);
            let v: Vec<f64> = bruit(np as usize, 4e-3, i as f64).iter().map(|x| m + x).collect();
            ecrire_fausse(&d, spec, &v);
        }
        let l = ligne_sc("G-SC", &g_sc(&g, &c, &p, "SC", 1.0).unwrap());
        assert!(l.detail.contains("extension unique v4.0") || l.verdict == Verdict::Skip, "{l}");
        // erreur dégénérée : série constante ⇒ FAIL ECHEC_OUTIL, jamais PASS
        for spec in spec_sc(&p, "SC", 1.0, p.pas_sc).iter() {
            let v = vec![f_sp4(spec.beta); p.pas_sc as usize];
            ecrire_fausse(&d, spec, &v);
        }
        let l = ligne_sc("G-SC", &g_sc(&g, &c, &p, "SC", 1.0).unwrap_or(Issue::Skip("x".into())));
        assert!(l.verdict != Verdict::Pass, "{l}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn serie_absente_ou_incoherente() {
        let d = dossier_test("absente");
        let p = proto_test();
        let g = Sp4::nouveau();
        let c = calcul_test(&d);
        assert!(matches!(g_sc(&g, &c, &p, "SC", 1.0).unwrap(), Issue::Skip(_)));
        // série de paramètres différents : erreur de configuration, pas de lecture silencieuse
        let mut spec = spec_sc(&p, "SC", 1.0, p.pas_sc)[0].clone();
        spec.graine += 1;
        ecrire_fausse(&d, &spec, &bruit(p.pas_sc as usize, 1e-6, 0.0));
        assert!(g_sc(&g, &c, &p, "SC", 1.0).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
