//! Gates RG-* du bloc de Bałaban L_b = 3 et du témoin de Swendsen (GATES v1 §4, v3 §v3.3). Propriétaire : WP6.
//!
//! Chaque gate rend UNE ligne `PASS|FAIL|SKIP <ID> <détail>` (règle 1 de GATES.md) ; les mutants rendent leur propre ligne (`*-MUT`,
//! PASS = le mutant est bien ROUGE). Tous les seuils, graines et protocoles ci-dessous sont ceux de GATES.md (v1 + v3.3) et du journal
//! (ligne « WP6 GEL des protocoles ») ; un changement ici sans nouvelle version de GATES.md est un défaut.
//!
//! Métrique : max sur les liens et les 4 composantes de |Δq| (celle de RG-GPU, v3.3). Le module des entrées de la matrice 2×2
//! (métrique de `g_nl.py`, ≤ √2 fois la précédente) est rapporté dans le détail des lignes concernées.

use super::bloc::{bloquer, coordonnees, deplacer, site, ubar_lien, OptionsBloc, StatsTheta, Q};
use super::domaine;
use super::geometrie::{Geometrie, Table};
use super::swendsen::{bloquer_b2, plaquette_moyenne};
use crate::commun::{ChampHote, Erreur, Graine, Resultat};
use crate::groupes::gates::{LigneGate, Verdict};
use crate::groupes::su2::Su2;
use crate::groupes::{Flux, Groupe, ETIQUETTE_GROUPE};
use crate::heritage::su2_hb::{qdag, qmul};
use crate::stockage::npy::lire_champ;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

// ------------------------------------------------------------------------------------------
// Constantes gelées
// ------------------------------------------------------------------------------------------

pub const SEUIL_REG_PY: f64 = 1e-12;
pub const SEUIL_JAUGE: f64 = 1e-14;
pub const SEUIL_HYPER: f64 = 1e-14;
pub const SEUIL_MUTANT_COIN: f64 = 1e-3;
pub const SEUIL_PUR: f64 = 1e-15;
pub const SEUIL_ABEL: f64 = 1e-15;
pub const SEUIL_THETA_TRIVIAL: f64 = 1e-12;
pub const SEUIL_SWENDSEN: f64 = 1e-10;
pub const GRAINE_HAAR_L12: u64 = 26_102_606_001;
pub const GRAINE_JAUGE: u64 = 26_102_606_002;
pub const GRAINE_PUR: u64 = 26_102_606_003;
pub const AMPLITUDE_ABEL: f64 = 0.3;
pub const GRAINES_FIXTURES: [u64; 3] = [7, 11, 13];
pub const C_SWENDSEN: f64 = 0.5;
/// Liens grossiers testés par RG-HYPER (comme `g_nl.py`) : z ∈ {(0,0,0,0), (1,2,0,1)} × μ.
pub const Z_HYPER: [[usize; 4]; 2] = [[0, 0, 0, 0], [1, 2, 0, 1]];
/// Configurations grandL rapportées (lecture seule) : (nom, sous-dossier, fichier).
pub const GRANDL: [(&str, &str, &str); 2] = [
    ("L12_b2.3_c1", "L12_b2.3_c1", "cfg_002000.npy"),
    ("L12_b16_c1", "L12_b16_c1", "cfg_002000.npy"),
];
/// Entrée grandL du témoin Swendsen (le brief la donne par erreur sous forme de fichier W(R,T) : voir le journal).
pub const GRANDL_SWENDSEN: (&str, &str) = ("L12_b2.3_c1", "cfg_002000.npy");

// ------------------------------------------------------------------------------------------
// Contexte
// ------------------------------------------------------------------------------------------

pub struct Contexte {
    /// Fixtures RG-REG-PY (`validation/rg/fixtures`).
    pub fixtures: PathBuf,
    /// Données de test versionnées (`tests/donnees/rg`).
    pub donnees: PathBuf,
    /// Racine des ensembles grandL (`…/ensembles/production`) ; `None` ⇒ SKIP des gates qui en dépendent.
    pub grandl: Option<PathBuf>,
    /// Dossier des rapports JSON (RG-DOMAINE) ; `None` ⇒ rapports non écrits.
    pub rapports: Option<PathBuf>,
    pub fils: usize,
    pub mutants: bool,
    pub seul: Vec<String>,
}

impl Contexte {
    /// Un groupe de gates d'identifiant `id` doit tourner si un préfixe demandé le recouvre (dans un sens ou dans l'autre).
    fn veut(&self, id: &str) -> bool {
        self.seul.is_empty() || self.seul.iter().any(|p| id.starts_with(p.as_str()) || p.starts_with(id))
    }

    /// Une ligne de verdict est gardée si son identifiant commence par l'un des préfixes demandés.
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

// ------------------------------------------------------------------------------------------
// Outils : écarts, champs de test
// ------------------------------------------------------------------------------------------

/// (max des |Δq| par composante, max du module des entrées de la matrice 2×2) entre deux quaternions.
pub fn ecarts(a: Q, b: Q) -> (f64, f64) {
    let d: [f64; 4] = std::array::from_fn(|i| (a[i] - b[i]).abs());
    let comp = d.iter().cloned().fold(0.0, f64::max);
    // m00 = a0 + i a3, m01 = a2 + i a1 (m11, m10 : mêmes modules)
    let mat = d[0].hypot(d[3]).max(d[2].hypot(d[1]));
    (comp, mat)
}

/// Maximum courant des deux métriques.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Max2 {
    pub comp: f64,
    pub mat: f64,
}

impl Max2 {
    pub fn ajouter(&mut self, a: Q, b: Q) {
        let (c, m) = ecarts(a, b);
        self.comp = self.comp.max(c);
        self.mat = self.mat.max(m);
    }
    pub fn fusionner(self, o: Max2) -> Max2 {
        Max2 { comp: self.comp.max(o.comp), mat: self.mat.max(o.mat) }
    }
}

/// Écart maximal entre deux champs de même taille.
pub fn ecart_champs(a: &ChampHote, b: &ChampHote) -> Max2 {
    assert_eq!(a.l, b.l, "champs de côtés différents");
    let mut m = Max2::default();
    for (x, y) in a.q.chunks_exact(4).zip(b.q.chunks_exact(4)) {
        m.ajouter([x[0], x[1], x[2], x[3]], [y[0], y[1], y[2], y[3]]);
    }
    m
}

fn quat_haar(g: &Su2, flux: &mut Flux) -> Q {
    Su2::vers_quat(&g.haar(flux))
}

/// Champ de Haar : un tirage de la fiche `su2` par lien, dans l'ordre de fichier (direction la plus lente, site C-order).
pub fn champ_haar(l: usize, graine: u64) -> ChampHote {
    let g = Su2::nouveau();
    let mut flux = Flux::nouveau(Graine(graine), ETIQUETTE_GROUPE);
    let n = 4 * l.pow(4);
    let mut q = Vec::with_capacity(4 * n);
    for _ in 0..n {
        q.extend_from_slice(&quat_haar(&g, &mut flux));
    }
    ChampHote { l, q }
}

/// Jauge de Haar : un élément par site (C-order).
pub fn jauge_haar(l: usize, graine: u64) -> Vec<Q> {
    let g = Su2::nouveau();
    let mut flux = Flux::nouveau(Graine(graine), ETIQUETTE_GROUPE);
    (0..l.pow(4)).map(|_| quat_haar(&g, &mut flux)).collect()
}

/// U'(x, μ) = g(x) U(x, μ) g(x + μ)†.
pub fn transformer_jauge(champ: &ChampHote, g: &[Q]) -> ChampHote {
    let (l, v) = (champ.l, champ.v());
    let mut q = champ.q.clone();
    for mu in 0..4 {
        for s in 0..v {
            let t = site(deplacer(coordonnees(s, l), mu, 1, l), l);
            let u = qmul(qmul(g[s], champ.lien(mu * v + s)), qdag(g[t]));
            q[4 * (mu * v + s)..4 * (mu * v + s) + 4].copy_from_slice(&u);
        }
    }
    ChampHote { l, q }
}

/// Jauge pure : U(x, μ) = g(x) g(x + μ)†.
pub fn champ_jauge_pure(l: usize, g: &[Q]) -> ChampHote {
    let v = l.pow(4);
    let mut q = Vec::with_capacity(16 * v);
    for mu in 0..4 {
        for s in 0..v {
            let t = site(deplacer(coordonnees(s, l), mu, 1, l), l);
            q.extend_from_slice(&qmul(g[s], qdag(g[t])));
        }
    }
    ChampHote { l, q }
}

/// Champ abélien constant : liens de direction `mu0` = exp(i a σ₃) = (cos a, 0, 0, sin a), identité ailleurs.
pub fn champ_abelien(l: usize, mu0: usize, a: f64) -> ChampHote {
    let v = l.pow(4);
    let mut q = Vec::with_capacity(16 * v);
    for mu in 0..4 {
        for _ in 0..v {
            q.extend_from_slice(&if mu == mu0 { [a.cos(), 0.0, 0.0, a.sin()] } else { [1.0, 0.0, 0.0, 0.0] });
        }
    }
    ChampHote { l, q }
}

/// Site de référence c(z) = 3z + REF (modulo L) et c′ = c + 3e_μ, pour un lien grossier (z, μ).
fn centres(table: &Table, l: usize, z: [usize; 4], mu: usize) -> ([usize; 4], [usize; 4]) {
    let c: [usize; 4] = std::array::from_fn(|m| (3 * z[m] + table.ref_site[m]) % l);
    (c, deplacer(c, mu, 3, l))
}

/// Écart entre `Ū` calculé et un attendu par lien : `attendu(z, μ)`.
fn ecart_attendu(grossier: &ChampHote, attendu: impl Fn([usize; 4], usize) -> Q) -> Max2 {
    let (lc, vc) = (grossier.l, grossier.v());
    let mut m = Max2::default();
    for mu in 0..4 {
        for s in 0..vc {
            let z = coordonnees(s, lc);
            m.ajouter(grossier.lien(mu * vc + s), attendu(z, mu));
        }
    }
    m
}

fn options(g: Geometrie, fils: usize) -> OptionsBloc {
    OptionsBloc { geometrie: g, fils, angle_so3: false }
}

fn lire_fixture(dossier: &Path, nom: &str) -> Resultat<ChampHote> {
    lire_champ(&dossier.join(nom))
}

// ------------------------------------------------------------------------------------------
// RG-FIXTURES, RG-REG-PY
// ------------------------------------------------------------------------------------------

/// Les fichiers de fixtures sont ceux du manifeste (sha256).
pub fn gate_fixtures(ctx: &Contexte) -> LigneGate {
    let id = "RG-FIXTURES";
    let chemin = ctx.fixtures.join("manifeste.json");
    let Ok(txt) = std::fs::read_to_string(&chemin) else {
        return LigneGate::nouvelle(Verdict::Fail, id, format!("manifeste illisible : {}", chemin.display()));
    };
    let Ok(m) = serde_json::from_str::<serde_json::Value>(&txt) else {
        return LigneGate::nouvelle(Verdict::Fail, id, "manifeste JSON invalide");
    };
    let Some(fichiers) = m["fichiers"].as_object() else {
        return LigneGate::nouvelle(Verdict::Fail, id, "manifeste sans « fichiers »");
    };
    let mut mauvais = Vec::new();
    for (nom, attendu) in fichiers {
        let reel = crate::heritage::manifest::sha256_fichier(&ctx.fixtures.join(nom)).unwrap_or_else(|_| "illisible".into());
        if Some(reel.as_str()) != attendu.as_str() {
            mauvais.push(nom.clone());
        }
    }
    let attendus = GRAINES_FIXTURES.len() * 2;
    ligne(
        mauvais.is_empty() && fichiers.len() == attendus,
        id,
        format!("{} fichiers hachés conformes au manifeste (g_nl.py {})", fichiers.len() - mauvais.len(), &m["g_nl_sha256"].as_str().unwrap_or("?")[..16.min(m["g_nl_sha256"].as_str().unwrap_or("?").len())]),
    )
}

pub fn gate_reg_py(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let mut pire = Max2::default();
    let mut details = Vec::new();
    let mut pire_coin = f64::INFINITY;
    for graine in GRAINES_FIXTURES {
        let cfg = lire_fixture(&ctx.fixtures, &format!("cfg_s{graine}_q64.npy"))?;
        let reference = lire_fixture(&ctx.fixtures, &format!("ubar_s{graine}_q64.npy"))?;
        let sortie = bloquer(&cfg, &options(Geometrie::TroisCentre, ctx.fils))?;
        let e = ecart_champs(&sortie.grossier, &reference);
        details.push(format!("graine {graine} : {:.2e}", e.comp));
        pire = pire.fusionner(e);
        if ctx.mutants {
            let m = bloquer(&cfg, &options(Geometrie::TroisCoin, ctx.fils))?;
            pire_coin = pire_coin.min(ecart_champs(&m.grossier, &reference).comp);
        }
    }
    let mut v = vec![ligne(
        pire.comp <= SEUIL_REG_PY,
        "RG-REG-PY",
        format!(
            "max|Δq| = {:.3e} (seuil {:.0e}) sur 3 fixtures × 324 liens ; {} ; module des entrées de matrice (g_nl) {:.3e}",
            pire.comp,
            SEUIL_REG_PY,
            details.join(", "),
            pire.mat
        ),
    )];
    if ctx.mutants {
        v.push(ligne(
            pire_coin >= SEUIL_MUTANT_COIN,
            "RG-REG-PY-MUT",
            format!("mutant 3coin contre les références 3centre : écart minimal sur les 3 fixtures {pire_coin:.3e} (doit être ≥ {SEUIL_MUTANT_COIN:.0e})"),
        ));
    }
    Ok(v)
}

// ------------------------------------------------------------------------------------------
// RG-JAUGE
// ------------------------------------------------------------------------------------------

pub fn gate_jauge(ctx: &Contexte) -> Resultat<LigneGate> {
    let cfg = lire_fixture(&ctx.fixtures, "cfg_s7_q64.npy")?;
    let g = jauge_haar(cfg.l, GRAINE_JAUGE);
    let transforme = transformer_jauge(&cfg, &g);
    let table = Table::nouvelle(Geometrie::TroisCentre);
    let opt = options(Geometrie::TroisCentre, ctx.fils);
    let a = bloquer(&transforme, &opt)?.grossier;
    let b = bloquer(&cfg, &opt)?.grossier;
    let l = cfg.l;
    let e = ecart_attendu(&a, |z, mu| {
        let (c, cp) = centres(&table, l, z, mu);
        qmul(qmul(g[site(c, l)], b.lien(mu * b.v() + site(z, b.l))), qdag(g[site(cp, l)]))
    });
    Ok(ligne(
        e.comp <= SEUIL_JAUGE,
        "RG-JAUGE",
        format!("max|Δq| = {:.3e} (seuil {:.0e}) : Ū(g·U) = g(c) Ū(U) g(c′)† sur 324 liens, graine {GRAINE_JAUGE} ; module matrice {:.3e}", e.comp, SEUIL_JAUGE, e.mat),
    ))
}

// ------------------------------------------------------------------------------------------
// RG-HYPER
// ------------------------------------------------------------------------------------------

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn rec(n: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == n {
            out.push(cur.clone());
            return;
        }
        for i in 0..n {
            if !cur.contains(&i) {
                cur.push(i);
                rec(n, cur, out);
                cur.pop();
            }
        }
    }
    let mut out = Vec::new();
    rec(n, &mut Vec::new(), &mut out);
    out
}

/// Image d'un site par l'élément hyperoctaédral (perm, eps) centré en `ctr`, modulo `modulo`.
fn g_site(x: [usize; 4], perm: &[usize], eps: &[i8; 4], ctr: [usize; 4], modulo: usize) -> [usize; 4] {
    let mut y = [0isize; 4];
    for m in 0..4 {
        y[perm[m]] = eps[m] as isize * (x[m] as isize - ctr[m] as isize);
    }
    std::array::from_fn(|m| (ctr[m] as isize + y[m]).rem_euclid(modulo as isize) as usize)
}

/// Image d'un lien : (site de base, direction, signe).
fn g_lien(x: [usize; 4], nu: usize, perm: &[usize], eps: &[i8; 4], ctr: [usize; 4], modulo: usize) -> ([usize; 4], usize, i8) {
    let gx = g_site(x, perm, eps, ctr, modulo);
    let nn = perm[nu];
    if eps[nu] == 1 {
        (gx, nn, 1)
    } else {
        (deplacer(gx, nn, -1, modulo), nn, -1)
    }
}

/// Résultat de RG-HYPER : (max sur les 24 permutations, max sur les 360 éléments avec réflexion, effectifs).
#[derive(Clone, Copy, Debug)]
pub struct ResHyper {
    pub perm: Max2,
    pub refl: Max2,
    pub n_perm: usize,
    pub n_refl: usize,
}

/// Covariance du bloc sous les 384 éléments hyperoctaédraux centrés sur le site de référence (port de `g_nl.py` (b)).
pub fn hyper(champ: &ChampHote, geometrie: Geometrie, fils: usize) -> Resultat<ResHyper> {
    let table = Table::nouvelle(geometrie);
    let l = champ.l;
    let lc = l / 3;
    let v = champ.v();
    let mut elements: Vec<(Vec<usize>, [i8; 4])> = Vec::new();
    for perm in permutations(4) {
        for k in 0..16u32 {
            let eps: [i8; 4] = std::array::from_fn(|m| if (k >> (3 - m)) & 1 == 0 { 1 } else { -1 });
            elements.push((perm.clone(), eps));
        }
    }
    // Les liens de référence (non transformés) sont calculés une fois.
    let reference: Vec<([usize; 4], usize, Q)> = Z_HYPER
        .iter()
        .flat_map(|&z| (0..4).map(move |mu| (z, mu)))
        .map(|(z, mu)| (z, mu, ubar_lien(champ, &table, z, mu).0))
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(fils.max(1))
        .build()
        .map_err(|e| Erreur::Config(format!("pool de fils : {e}")))?;
    let un_element = |(perm, eps): &(Vec<usize>, [i8; 4])| -> Max2 {
        let mut q = vec![0.0; champ.q.len()];
        for nu in 0..4 {
            for s in 0..v {
                let x = coordonnees(s, l);
                let (y, nn, sg) = g_lien(x, nu, perm, eps, table.ref_site, l);
                let u = champ.lien(nu * v + s);
                let u = if sg == 1 { u } else { qdag(u) };
                let o = 4 * (nn * v + site(y, l));
                q[o..o + 4].copy_from_slice(&u);
            }
        }
        let transforme = ChampHote { l, q };
        let mut m = Max2::default();
        for (z, mu, b) in &reference {
            let (gz, gmu, sg) = g_lien(*z, *mu, perm, eps, [0; 4], lc);
            let a = ubar_lien(&transforme, &table, gz, gmu).0;
            m.ajouter(a, if sg == 1 { *b } else { qdag(*b) });
        }
        m
    };
    let resultats: Vec<Max2> = pool.install(|| elements.par_iter().map(un_element).collect());
    let mut r = ResHyper { perm: Max2::default(), refl: Max2::default(), n_perm: 0, n_refl: 0 };
    for ((_, eps), m) in elements.iter().zip(resultats) {
        if eps.contains(&-1) {
            r.refl = r.refl.fusionner(m);
            r.n_refl += 1;
        } else {
            r.perm = r.perm.fusionner(m);
            r.n_perm += 1;
        }
    }
    Ok(r)
}

pub fn gate_hyper(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let cfg = lire_fixture(&ctx.fixtures, "cfg_s7_q64.npy")?;
    let r = hyper(&cfg, Geometrie::TroisCentre, ctx.fils)?;
    let pire = r.perm.comp.max(r.refl.comp);
    let mut v = vec![ligne(
        r.n_perm == 24 && r.n_refl == 360 && pire <= SEUIL_HYPER,
        "RG-HYPER",
        format!(
            "{} éléments : permutations ({}) max|Δq| = {:.3e}, avec réflexion ({}) {:.3e} (seuil {:.0e}) ; module matrice {:.3e}",
            r.n_perm + r.n_refl,
            r.n_perm,
            r.perm.comp,
            r.n_refl,
            r.refl.comp,
            SEUIL_HYPER,
            r.perm.mat.max(r.refl.mat)
        ),
    )];
    if ctx.mutants {
        let m = hyper(&cfg, Geometrie::TroisCoin, ctx.fils)?;
        v.push(ligne(
            m.refl.comp >= SEUIL_MUTANT_COIN,
            "RG-HYPER-MUT",
            format!(
                "mutant 3coin : réflexions {:.3e} (doit être ≥ {SEUIL_MUTANT_COIN:.0e}), permutations {:.3e}",
                m.refl.comp, m.perm.comp
            ),
        ));
    }
    Ok(v)
}

// ------------------------------------------------------------------------------------------
// RG-PUR, RG-ABEL, RG-DOMAINE (instrument)
// ------------------------------------------------------------------------------------------

/// Résultat de RG-PUR : écart et statistiques de θ.
pub fn mesure_pur(fils: usize) -> Resultat<(Max2, StatsTheta)> {
    let table = Table::nouvelle(Geometrie::TroisCentre);
    let l = 9;
    let g = jauge_haar(l, GRAINE_PUR);
    let champ = champ_jauge_pure(l, &g);
    let s = bloquer(&champ, &options(Geometrie::TroisCentre, fils))?;
    let e = ecart_attendu(&s.grossier, |z, mu| {
        let (c, cp) = centres(&table, l, z, mu);
        qmul(g[site(c, l)], qdag(g[site(cp, l)]))
    });
    Ok((e, s.stats))
}

/// Résultat de RG-ABEL : maximum sur les 4 directions μ₀ du champ abélien.
pub fn mesure_abel(fils: usize) -> Resultat<(Max2, StatsTheta)> {
    let a = AMPLITUDE_ABEL;
    let attendu_mu0 = [(3.0 * a).cos(), 0.0, 0.0, (3.0 * a).sin()];
    let mut pire = Max2::default();
    let mut stats = StatsTheta::vide();
    for mu0 in 0..4 {
        let champ = champ_abelien(9, mu0, a);
        let s = bloquer(&champ, &options(Geometrie::TroisCentre, fils))?;
        let e = ecart_attendu(&s.grossier, |_, mu| if mu == mu0 { attendu_mu0 } else { [1.0, 0.0, 0.0, 0.0] });
        pire = pire.fusionner(e);
        stats.fusionner(&s.stats);
    }
    Ok((pire, stats))
}

pub fn gates_pur_abel(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let mut v = Vec::new();
    let (ep, sp) = mesure_pur(ctx.fils)?;
    let (ea, sa) = mesure_abel(ctx.fils)?;
    if ctx.veut("RG-PUR") {
        v.push(ligne(
            ep.comp <= SEUIL_PUR,
            "RG-PUR",
            format!("max|Δq| = {:.3e} (seuil {:.0e}) : jauge pure, Ū = g(c) g(c′)†, graine {GRAINE_PUR}", ep.comp, SEUIL_PUR),
        ));
    }
    if ctx.veut("RG-ABEL") {
        v.push(ligne(
            ea.comp <= SEUIL_ABEL,
            "RG-ABEL",
            format!("max|Δq| = {:.3e} (seuil {:.0e}) : exp(i a σ₃) sur les liens μ₀ ∈ 0..3, a = {AMPLITUDE_ABEL}, attendu exp(3 i a σ₃)", ea.comp, SEUIL_ABEL),
        ));
    }
    if ctx.veut("RG-DOMAINE") {
        let ok = sp.n_hors == 0 && sa.n_hors == 0 && sp.theta_max <= SEUIL_THETA_TRIVIAL && sa.theta_max <= SEUIL_THETA_TRIVIAL;
        v.push(ligne(
            ok,
            "RG-DOMAINE-PUR-ABEL",
            format!(
                "PUR : f = {} θ max = {:.2e} ; ABEL : f = {} θ max = {:.2e} (exigé : f = 0 et θ max ≤ {SEUIL_THETA_TRIVIAL:.0e})",
                sp.fraction_hors(),
                sp.theta_max,
                sa.fraction_hors(),
                sa.theta_max
            ),
        ));
    }
    Ok(v)
}

/// Statistiques de θ d'un champ de Haar L = 12 (graine gelée), avec ou sans le mutant 2θ.
pub fn stats_haar(fils: usize, angle_so3: bool) -> Resultat<StatsTheta> {
    let champ = champ_haar(12, GRAINE_HAAR_L12);
    let opt = OptionsBloc { geometrie: Geometrie::TroisCentre, fils, angle_so3 };
    Ok(bloquer(&champ, &opt)?.stats)
}

pub fn gates_domaine_haar(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let s = stats_haar(ctx.fils, false)?;
    let f = s.fraction_hors();
    let mut v = vec![ligne(
        (f - domaine::F_HAAR_ATTENDU).abs() <= domaine::TOLERANCE_HAAR,
        "RG-DOMAINE-HAAR",
        format!(
            "Haar L = 12, graine {GRAINE_HAAR_L12} : f = {f:.5} (attendu {} ± {}), θ max = {:.6}, {} boucles, {} ambiguës",
            domaine::F_HAAR_ATTENDU,
            domaine::TOLERANCE_HAAR,
            s.theta_max,
            s.n,
            s.n_ambigues
        ),
    )];
    if ctx.mutants {
        let m = stats_haar(ctx.fils, true)?;
        let fm = m.fraction_hors();
        v.push(ligne(
            (fm - domaine::F_HAAR_ATTENDU).abs() > domaine::TOLERANCE_HAAR,
            "RG-DOMAINE-MUT",
            format!("mutant angle SO(3) (2θ) : f = {fm:.5} (attendu ≈ 0,909 ; doit sortir de {} ± {})", domaine::F_HAAR_ATTENDU, domaine::TOLERANCE_HAAR),
        ));
    }
    Ok(v)
}

/// Rapports grandL (iii) : émis ; les étiquettes sont rapportées telles qu'elles sortent.
pub fn gates_domaine_grandl(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let mut v = Vec::new();
    for (nom, dossier, fichier) in GRANDL {
        let id = format!("RG-DOMAINE-{nom}");
        let Some(racine) = &ctx.grandl else {
            v.push(skip(&id, "ensembles grandL non fournis (--grandl) : rapport non émis"));
            continue;
        };
        let chemin = racine.join(dossier).join(fichier);
        if !chemin.is_file() {
            v.push(skip(&id, format!("configuration absente : {}", chemin.display())));
            continue;
        }
        let champ = lire_champ(&chemin)?;
        let sha = crate::heritage::manifest::sha256_fichier(&chemin).map_err(|e| Erreur::io("hachage", e))?;
        let s = bloquer(&champ, &options(Geometrie::TroisCentre, ctx.fils))?.stats;
        let rap = domaine::rapport(nom, &sha, champ.l, Geometrie::TroisCentre, &s);
        let mut detail = format!(
            "{} : f = {:.6e}, θ max = {:.4}, {} ambiguës, étiquette {}",
            nom,
            s.fraction_hors(),
            s.theta_max,
            s.n_ambigues,
            domaine::etiquette(&s)
        );
        let emis = match &ctx.rapports {
            Some(d) => {
                std::fs::create_dir_all(d).map_err(|e| Erreur::io("création du dossier des rapports", e))?;
                let f = d.join(format!("domaine_{nom}.json"));
                domaine::ecrire_json(&f, &rap)?;
                detail.push_str(&format!(" ; rapport {}", f.display()));
                true
            }
            None => {
                detail.push_str(" ; rapport non écrit (--rapports absent)");
                true
            }
        };
        v.push(ligne(emis, &id, detail));
    }
    Ok(v)
}

// ------------------------------------------------------------------------------------------
// RG-SWENDSEN
// ------------------------------------------------------------------------------------------

fn swendsen_un(id: &str, nom: &str, entree: &Path, reference: &Path) -> Resultat<LigneGate> {
    let cfg = lire_champ(entree)?;
    let c_ref = lire_champ(reference)?;
    let sortie = bloquer_b2(&cfg, C_SWENDSEN)?;
    let (pr, pc) = (plaquette_moyenne(&sortie), plaquette_moyenne(&c_ref));
    let rel = (pr - pc).abs() / pc.abs();
    let liens = ecart_champs(&sortie, &c_ref);
    Ok(ligne(
        sortie.l == c_ref.l && rel <= SEUIL_SWENDSEN,
        id,
        format!(
            "{nom} : plaq_rel = {rel:.3e} (seuil {SEUIL_SWENDSEN:.0e}), P_rust = {pr:.12}, P_C = {pc:.12}, plaquette de l'entrée {:.6} ; max|Δq| des liens {:.3e} (INFO), c = {C_SWENDSEN}, 1 pas",
            plaquette_moyenne(&cfg),
            liens.comp
        ),
    ))
}

pub fn gates_swendsen(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let mut v = vec![swendsen_un(
        "RG-SWENDSEN-L8",
        "SU(2) 8⁴ β = 2,40 graine 24001",
        &ctx.donnees.join("cfg_su2_L8_b2.40_s24001.npy"),
        &ctx.donnees.join("ref_C_L8_to_L4_c0.5.npy"),
    )?];
    let id = "RG-SWENDSEN-L12";
    match &ctx.grandl {
        None => v.push(skip(id, "ensembles grandL non fournis (--grandl)")),
        Some(racine) => {
            let entree = racine.join(GRANDL_SWENDSEN.0).join(GRANDL_SWENDSEN.1);
            let attendu = std::fs::read_to_string(ctx.donnees.join("grandL_L12_b2.3_c1_cfg002000.sha256")).unwrap_or_default();
            if !entree.is_file() {
                v.push(skip(id, format!("configuration absente : {}", entree.display())));
            } else if crate::heritage::manifest::sha256_fichier(&entree).ok().as_deref() != attendu.split_whitespace().next() {
                v.push(LigneGate::nouvelle(Verdict::Fail, id, "le sha256 de la configuration grandL ne correspond pas à celui de la référence de block_config"));
            } else {
                v.push(swendsen_un(id, "grandL L12 β = 2,3 cfg_002000", &entree, &ctx.donnees.join("ref_C_grandL_L12_b2.3_c1_cfg002000_c0.5.npy"))?);
            }
        }
    }
    Ok(v)
}

// ------------------------------------------------------------------------------------------
// RG-FILS (par le chemin de fichiers du CLI)
// ------------------------------------------------------------------------------------------

fn octets(chemin: &Path) -> Resultat<Vec<u8>> {
    std::fs::read(chemin).map_err(|e| Erreur::io(format!("lecture {}", chemin.display()), e))
}

fn fils_un(id: &str, nom: &str, entree: &Path, dossier: &Path) -> Resultat<LigneGate> {
    let mut sorties = Vec::new();
    for fils in [1usize, 4, 16] {
        let s = dossier.join(format!("{id}_{fils}.npy"));
        super::cli::bloquer_fichier(entree, &s, Geometrie::TroisCentre, fils)?;
        sorties.push((fils, octets(&s)?, octets(&super::cli::chemin_manifeste(&s))?));
    }
    let ok = sorties.iter().all(|(_, n, m)| *n == sorties[0].1 && *m == sorties[0].2);
    Ok(ligne(
        ok,
        id,
        format!(
            "{nom} : sorties NPY ({} octets) et manifestes ({} octets) identiques à l'octet pour 1, 4 et 16 fils (cmp)",
            sorties[0].1.len(),
            sorties[0].2.len()
        ),
    ))
}

pub fn gates_fils(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let tmp = std::env::temp_dir().join(format!("kvu_rg_fils_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).map_err(|e| Erreur::io("création du dossier temporaire", e))?;
    let mut v = vec![fils_un("RG-FILS-9", "fixture 9⁴ graine 7", &ctx.fixtures.join("cfg_s7_q64.npy"), &tmp)?];
    // Champ de Haar L = 12 écrit au format q64 (même format que la fixture).
    let haar = tmp.join("haar_L12.npy");
    crate::stockage::npy::ecrire_champ(
        &haar,
        &champ_haar(12, GRAINE_HAAR_L12),
        crate::stockage::npy::FormatChamp::Q64,
        &crate::stockage::garde::GardeSsd::machine(),
    )?;
    v.push(fils_un("RG-FILS-12", "Haar L = 12", &haar, &tmp)?);
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(v)
}

// ------------------------------------------------------------------------------------------
// Batterie
// ------------------------------------------------------------------------------------------

pub fn batterie(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let mut v = Vec::new();
    if ctx.veut("RG-FIXTURES") {
        v.push(gate_fixtures(ctx));
    }
    if ctx.veut("RG-REG-PY") {
        v.extend(gate_reg_py(ctx)?);
    }
    if ctx.veut("RG-JAUGE") {
        v.push(gate_jauge(ctx)?);
    }
    if ctx.veut("RG-HYPER") {
        v.extend(gate_hyper(ctx)?);
    }
    if ctx.veut("RG-PUR") || ctx.veut("RG-ABEL") || ctx.veut("RG-DOMAINE") {
        v.extend(gates_pur_abel(ctx)?);
    }
    if ctx.veut("RG-DOMAINE") {
        v.extend(gates_domaine_haar(ctx)?);
        v.extend(gates_domaine_grandl(ctx)?);
    }
    if ctx.veut("RG-SWENDSEN") {
        v.extend(gates_swendsen(ctx)?);
    }
    if ctx.veut("RG-FILS") {
        v.extend(gates_fils(ctx)?);
    }
    v.retain(|l| ctx.garde_ligne(&l.id));
    Ok(v)
}
